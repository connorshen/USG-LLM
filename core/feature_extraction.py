from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class ExtFeatures(BaseModel):
    """B超报告特征提取结果"""
    OA: int = Field(description="卵巢是否萎缩 (0-正常，1-萎缩/偏小/回声偏实/显示不清/未显示，2-未提取到数据)")
    EA: int = Field(description="子宫内膜是否萎缩 (0-正常，1-变薄/菲薄/双层<5mm 或单层<2.5mm，2-未提取到数据)")
    UA: int = Field(description="子宫是否萎缩 (0-正常，1-变小/萎缩/绝经后子宫，2-未提取到数据)")


def _build_chain(api_key: str):
    model = Tongyi(model="qwen-plus", api_key=api_key)
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


def extract_features(
    input_path: str,
    output_path: str | None = None,
    report_col: str = "阴道超声",
    api_key: str = "sk-d7f9d706ffac4128aa5ee97e0a6064e1",
    max_workers: int = 20,
) -> pd.DataFrame:
    """从超声报告中提取形态学特征。

    Args:
        input_path: 输入 Excel 文件路径。
        output_path: 输出 Excel 文件路径，为 None 则不保存文件。
        report_col: 超声报告所在的列名。
        api_key: Tongyi API Key。
        max_workers: 并发线程数。

    Returns:
        包含提取特征的 DataFrame。
    """
    df = pd.read_excel(input_path)
    reports = df[report_col].tolist()
    reports = [{"report_text": report_text} for report_text in reports]
    df["BMI"] = df["Weight"] / (df["Height"]) ** 2

    chain = _build_chain(api_key)

    def _task(i, report):
        return i, chain.invoke(report)

    total = len(reports)
    responses = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_task, i, rep) for i, rep in enumerate(reports)]
        for future in tqdm(as_completed(futures), total=total, desc="提取进度"):
            idx, res = future.result()
            responses[idx] = res

    new_df = df.join(pd.DataFrame(responses))
    new_df["Menopause"] = new_df.pop("Menopause")
    for col in ["FSH", "LH", "E2"]:
        new_df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

    if output_path:
        new_df.to_excel(output_path, index=False)

    return new_df


if __name__ == "__main__":
    extract_features("../dataset/data1.xlsx", "../dataset/data1_extraction.xlsx")
    extract_features("../dataset/data2.xlsx", "../dataset/data2_extraction.xlsx")
