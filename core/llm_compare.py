from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os


class ExtFeatures(BaseModel):
    """B超报告特征提取结果"""
    OA: int = Field(description="卵巢是否萎缩 (0-正常，1-萎缩/偏小/回声偏实/显示不清/未显示，2-未提取到数据)")
    EA: int = Field(description="子宫内膜是否萎缩 (0-正常，1-变薄/菲薄/双层<5mm 或单层<2.5mm，2-未提取到数据)")
    UA: int = Field(description="子宫是否萎缩 (0-正常，1-变小/萎缩/绝经后子宫，2-未提取到数据)")


def _build_chain(api_key: str, model_name: str):
    model = Tongyi(model=model_name, api_key=api_key)
    parser = JsonOutputParser(pydantic_object=ExtFeatures)
    prompt = PromptTemplate(
        template="""
你是一位资深B超医生，请根据以下B超报告提取三个形态学特征。

【判断规则】

OA - 卵巢是否萎缩：
- 0（正常）：双侧卵巢大小形态正常，描述中未见萎缩、偏小、回声偏实、显示不清等异常
- 1（萎缩）：卵巢偏小、萎缩、回声偏实、显示不清、未显示、隐约可见
- 2（未提取）：报告中完全未提及卵巢相关信息

EA - 子宫内膜是否萎缩：
- 优先基于内膜厚度数值判断：
  - 双层内膜厚度 ≥ 5mm（即≥0.5cm） → 0（正常）
  - 双层内膜厚度 < 5mm（即<0.5cm），或单层 < 2.5mm，或描述为变薄/菲薄 → 1（萎缩）
- 如果报告中完全没有提到内膜厚度或内膜状态 → 2（未提取）
- 重要：不要因为报告中存在肌瘤、囊肿等其他异常就输出2。只要有内膜厚度数据，优先根据厚度判断。

UA - 子宫是否萎缩：
- 0（正常）：子宫大小正常或增大（如有肌瘤会导致增大）
- 1（萎缩）：子宫变小、萎缩、描述为"绝经后子宫"
- 2（未提取）：完全未提及子宫大小
- 重要：有肌瘤、结节、增大的子宫不属于萎缩。

【示例】

示例1（正常，有肌瘤但内膜和卵巢正常）：
报告：子宫前位，增大，双层内膜厚约0.8cm，前壁肌层可见一大小约5.5*4.7cm混合回声团块，边界清，内见多个片状无回声区，透声欠佳，彩色血流信号不丰富。双侧卵巢大小形态正常，彩色血流信号未见异常。
输出：{{"OA": 0, "EA": 0, "UA": 0}}

示例2（卵巢和内膜萎缩，子宫偏小）：
报告：子宫前位，偏小，双层内膜厚约0.3cm，肌层回声均匀。双侧卵巢大小形态正常，回声偏实，彩色血流信号未见异常。
输出：{{"OA": 1, "EA": 1, "UA": 1}}

示例3（未提取到数据）：
报告：无
输出：{{"OA": 2, "EA": 2, "UA": 2}}

报告内容：{report_text}
请提取以下特征并以JSON格式返回：
{format_instructions}
""",
        input_variables=["report_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | model | parser


def extract_with_model(
    df: pd.DataFrame,
    model_name: str,
    api_key: str = "sk-d7f9d706ffac4128aa5ee97e0a6064e1",
    max_workers: int = 20,
) -> pd.DataFrame:
    """使用指定模型从超声报告中提取形态学特征。

    Parameters
    ----------
    df : pd.DataFrame
        包含 'Report' 列和标注列 OA/EA/UA 的 DataFrame。
    model_name : str
        Tongyi 模型名称，如 qwen-plus、qwen-flash、qwen-max。
    api_key : str
        Tongyi API Key。
    max_workers : int
        并发线程数。

    Returns
    -------
    pd.DataFrame
        包含原始报告、模型预测和人工标注的 DataFrame。
    """
    reports = df["超声报告"].tolist()
    reports = [{"report_text": text} for text in reports]

    chain = _build_chain(api_key, model_name)

    def _task(i, report):
        return i, chain.invoke(report)

    total = len(reports)
    responses = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_task, i, rep) for i, rep in enumerate(reports)]
        for future in tqdm(as_completed(futures), total=total, desc=f"[{model_name}] Extracting"):
            idx, res = future.result()
            responses[idx] = res

    pred_df = pd.DataFrame(responses)
    pred_df.columns = ["pred_OA", "pred_EA", "pred_UA"]

    result = pd.concat([
        df[["Report"]].reset_index(drop=True),
        pred_df.reset_index(drop=True),
        df[["OA", "EA", "UA"]].reset_index(drop=True),
    ], axis=1)

    return result


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "dataset", "label.xlsx")
    output_dir = os.path.join(project_root, "dataset")

    df = pd.read_excel(input_path)

    models = ["qwen-turbo", "qwen-flash", "qwen-plus"]

    for model_name in models:
        print(f"\n>>> Start extracting: {model_name}")
        result = extract_with_model(df, model_name)

        safe_name = model_name.replace("-", "_")
        output_path = os.path.join(output_dir, f"{safe_name}_extraction.xlsx")
        result.to_excel(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
