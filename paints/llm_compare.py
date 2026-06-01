import os
import pandas as pd
from sklearn.metrics import f1_score


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute evaluation metrics for a single model extraction result."""
    metrics = {}

    for feat in ["OA", "EA", "UA"]:
        y_true = df[feat].values
        y_pred = df[f"pred_{feat}"].values
        metrics[f"{feat} F1 Score"] = f1_score(y_true, y_pred, average="macro")

    # Macro average F1
    metrics["Macro F1 Score"] = (
        metrics["OA F1 Score"] + metrics["EA F1 Score"] + metrics["UA F1 Score"]
    ) / 3

    return metrics


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "dataset")
    tables_dir = os.path.join(project_root, "tables")

    models = ["qwen-turbo", "qwen-flash", "qwen-plus"]
    rows = []

    for model_name in models:
        safe_name = model_name.replace("-", "_")
        file_path = os.path.join(dataset_dir, f"{safe_name}_extraction.xlsx")

        if not os.path.exists(file_path):
            print(f"Warning: file not found, skip {file_path}")
            continue

        df = pd.read_excel(file_path)
        metrics = compute_metrics(df)
        metrics["Model"] = model_name
        rows.append(metrics)

    result_df = pd.DataFrame(rows)
    result_df = result_df[[
        "Model","OA F1 Score", "EA F1 Score", "UA F1 Score", "Macro F1 Score"
    ]]

    # Format output
    for col in ["OA F1 Score", "EA F1 Score", "UA F1 Score", "Macro F1 Score"]:
        result_df[col] = result_df[col].apply(lambda x: f"{x:.4f}")

    output_path = os.path.join(tables_dir, "llm_compare.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(result_df.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
