import pandas as pd
import torch, random, numpy as np
from os import path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

FEATURE_GROUPS = {
    'anthropometric': ['Age', 'Height', 'Weight', 'BMI'],
    'morphology': ['OA', 'EA', 'UA'],
    'hormone': ['AMH', 'FSH', 'LH', 'P', 'E2', 'Prolactin', 'Testosterone'],
}


def build_dataset(data_file='data1_extraction.xlsx', features=None):
    """
    Load dataset with optional feature category selection.

    Parameters
    ----------
    features : None or list of str
        None -> all features (default)
        ['anthropometric'] -> anthropometric only
        ['morphology'] -> morphology only
        ['hormone'] -> hormone only
        ['anthropometric', 'morphology'] -> combination
    """
    set_seed()
    current_dir = path.dirname(path.abspath(__file__))
    df = pd.read_excel(path.abspath(path.join(current_dir, "../dataset", data_file)))

    all_cols = ["Age", "Height", "Weight", "FSH", 'LH', 'P', 'E2', 'Prolactin', 'Testosterone', 'AMH', "OA", "EA", "UA", "BMI", 'Menopause']
    df = df[all_cols]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())

    # Filter to 45-55 age group
    df = df[(df['Age'] >= 45) & (df['Age'] <= 55)]

    if features is None:
        return df

    # Ensure consistent column order: Anthropometric -> Morphology -> Hormone -> Menopause
    ordered_cols = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        if group_name in features:
            ordered_cols.extend(group_cols)
    ordered_cols.append('Menopause')
    return df[ordered_cols]
