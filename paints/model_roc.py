import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc
from catboost import CatBoostClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPS_DIR = os.path.join(PROJECT_ROOT, "exps")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

sys.path.insert(0, PROJECT_ROOT)
from core.build_dataset import build_dataset

font_size = 18
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['legend.title_fontsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size

FEATURE_COMBINATIONS = {
    'Anthropometric': ['anthropometric'],
    'Hormone': ['hormone'],
    'Morphology': ['morphology'],
    'Anthropometric + Morph': ['anthropometric', 'morphology'],
    'Anthropometric + Hormone': ['anthropometric', 'hormone'],
    'All':['anthropometric','hormone', 'morphology']
}

MODELS = {
    'XGBoost': ('xgboost_roc.csv', 0),
    'Naive Bayes': ('nb_roc.csv', 1),
    'CatBoost': ('catboost_roc.csv', 2),
    'SVM': ('svm_roc.csv', 3),
    'KNN': ('knn_roc.csv', 4),
    'ANN': ('ann_roc.csv', 5),
    'LightGBM': ('lightgbm_roc.csv', 6),
    'RF': ('rf_roc.csv', 7),
}

custom_colors = [
    '#2E86AB',
    '#F6B555',
    '#D72638',
    '#0A9263',
    '#7D5BA6',
    '#E87C3F',
    '#17BEBB',
    '#A33B20',
]


def train_model(X_train, y_train, X_val, y_val):
    """在训练集上训练模型并在验证集上返回预测分数"""
    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Logloss',
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_val)[:, 1]
    return y_scores


def evaluate_model(y_true, y_scores):
    """计算模型评估指标"""
    auc_score = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return {
        'auc': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'ap': ap
    }


def paint1():
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # 左侧：特征组合 ROC（morphology 内容）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for idx, (name, feature_groups) in enumerate(FEATURE_COMBINATIONS.items()):
        df_train = build_dataset('data1_extraction.xlsx', features=feature_groups)
        df_val = build_dataset('data2_extraction.xlsx', features=feature_groups)

        X_train = df_train.drop('Menopause', axis=1).values
        y_train = df_train['Menopause'].values
        X_val = df_val.drop('Menopause', axis=1).values
        y_val = df_val['Menopause'].values

        y_scores = train_model(X_train, y_train, X_val, y_val)
        metrics = evaluate_model(y_val, y_scores)
        axes[0].plot(metrics['fpr'], metrics['tpr'],
                color=colors[idx], lw=2.5,
                label=f'{name} (AUC={metrics["auc"]:.3f})')

    axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
    axes[0].set_title('(a) ROC Curves of Feature Combinations')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')

    # 右侧：8 模型 ROC 对比
    for name, (file, idx) in MODELS.items():
        df = pd.read_csv(os.path.join(EXPS_DIR, file))
        roc_auc = auc(df['fpr'], df['tpr'])
        axes[1].plot(df['fpr'], df['tpr'], color=custom_colors[idx], lw=2.5,
                 label=f'{name} (AUC = {roc_auc:.3f})')

    axes[1].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.5)
    axes[1].set_title('(b) ROC Curves of Eight Models')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "model_roc.jpeg"), dpi=300)
    plt.close()


if __name__ == '__main__':
    paint1()
