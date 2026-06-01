import pandas as pd
from scipy import stats
import argparse


def distribution(input_path, output_path):
    df = pd.read_excel(input_path)

    if "Age" not in df.columns and "Name" in df.columns:
        df = df.rename(columns={"Name": "Age"})

    feature_mapping = {
        "Age": ["Age (years)", "Age"],
        "Height": ["Height (m)", "Height"],
        "Weight": ["Weight (kg)", "Weight"],
        "FSH": ["FSH (mIU/mL)", "FSH"],
        "LH": ["LH (mIU/mL)", "LH"],
        "P": ["P (ng/mL)", "P"],
        "E2": ["E2 (pg/mL)", "E2"],
        "Prolactin": ["Prolactin (ng/mL)", "Prolactin"],
        "Testosterone": ["Testosterone (ng/mL)", "Testosterone"],
        "AMH": ["AMH (ng/mL)", "AMH"],
        "BMI": ["BMI (kg/m²)", "BMI"],
    }

    results = []
    for name, possible_features in feature_mapping.items():
        actual_feature = None
        for feat in possible_features:
            if feat in df.columns:
                actual_feature = feat
                break
        if actual_feature is None:
            continue

        display_name = possible_features[0]
        prem = pd.to_numeric(df.loc[df["Menopause"] == 0, actual_feature], errors='coerce').dropna()
        post = pd.to_numeric(df.loc[df["Menopause"] == 1, actual_feature], errors='coerce').dropna()
        _, p = stats.ttest_ind(prem, post)
        results.append({
            "Feature": display_name,
            "Missing Rate": f"{df[actual_feature].isna().mean():.2%}",
            "PreM": f"{prem.mean():.2f} ± {prem.std():.2f}",
            "PostM": f"{post.mean():.2f} ± {post.std():.2f}",
            "P Value": f"{p:.4f}" if p >= 0.01 else "<0.01",
        })

    for feature in ["OA", "EA", "UA"]:
        if feature not in df.columns:
            continue
        valid = df[df[feature].isin([0, 1])]
        _, p, _, _ = stats.chi2_contingency(pd.crosstab(valid["Menopause"], valid[feature]))
        missing = ((df[feature] == 2) | df[feature].isna()).mean()
        results.append({
            "Feature": feature,
            "Missing Rate": f"{missing:.2%}",
            "PreM": "-",
            "PostM": "-",
            "P Value": f"{p:.4f}" if p >= 0.01 else "<0.01",
        })

    pd.DataFrame(results).to_csv(output_path, index=False)


if __name__ == "__main__":
    distribution("dataset/data1_extraction.xlsx", "tables/distribution1.csv")
    distribution("dataset/data2_extraction.xlsx", "tables/distribution2.csv")
