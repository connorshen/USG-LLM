import io
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import numpy as np
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.build_dataset import build_dataset


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = plt.imread(buf)
    buf.close()
    return img


font_size = 18
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['legend.title_fontsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size
sns.set_style("darkgrid")


def set_style():
    ax = plt.gca()
    fig = plt.gcf()
    ax.xaxis.label.set_fontsize(font_size)
    [text.set_fontsize(font_size) for text in ax.get_xticklabels()]
    ax.yaxis.label.set_fontsize(font_size)
    [text.set_fontsize(font_size) for text in ax.get_yticklabels()]
    color_bar = fig.axes[1]
    color_bar.yaxis.label.set_fontsize(font_size)
    [text.set_fontsize(font_size) for text in color_bar.get_yticklabels()]


def _draw_importance(fig, shap_values, feature_names):
    shap_importance = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(shap_importance)[::-1]
    shap_importance = shap_importance[idx]
    feature_names = [feature_names[i] for i in idx]

    ax = fig.add_subplot(111)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_names)))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, shap_importance, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Mean |SHAP value|')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n_pie = min(5, len(feature_names))
    pie_importance = shap_importance[:n_pie].tolist()
    pie_labels = feature_names[:n_pie]

    if len(feature_names) > n_pie:
        pie_importance.append(shap_importance[n_pie:].sum())
        pie_labels.append('Other')

    pie_colors = ['#E69F00', '#56B4E9', '#009E73', '#CC9900', '#0072B2', '#D55E00']

    ax_pie = fig.add_axes([0.3, 0.1, 0.7, 0.7])
    wedges, texts, autotexts = ax_pie.pie(
        pie_importance,
        labels=pie_labels,
        autopct='%1.1f%%',
        colors=pie_colors[:len(pie_importance)],
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2.5),
        textprops={'fontsize': 16, 'weight': 'bold'},
        pctdistance=0.75
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    plt.tight_layout()
    img = fig_to_image(fig)
    plt.close(fig)
    return img


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def paint1():
    df = build_dataset('data1_extraction.xlsx')
    X = df.drop('Menopause', axis=1)
    y = df['Menopause'].values

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.1,
        n_estimators=100,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_data = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X.values,
        feature_names=X.columns.tolist()
    )

    fig_bee = plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_data, show=False, color_bar=True, max_display=14)
    set_style()
    img_bee = fig_to_image(fig_bee)
    plt.close(fig_bee)

    fig_imp = plt.figure(figsize=(10, 8))
    img_imp = _draw_importance(fig_imp, shap_values, X.columns.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img_bee)
    axes[0].axis('off')
    axes[0].set_title('(a) SHAP Beeswarm Plot')
    axes[1].imshow(img_imp)
    axes[1].axis('off')
    axes[1].set_title('(b) SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(
        os.path.join(PROJECT_ROOT, "images", "shap_combined.jpeg"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def paint2():
    df = build_dataset('data1_extraction.xlsx', features=['anthropometric', 'morphology'])
    X = df.drop('Menopause', axis=1)
    y = df['Menopause'].values

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.1,
        n_estimators=100,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_data = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X.values,
        feature_names=X.columns.tolist()
    )

    fig_bee = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_data, show=False, color_bar=True, max_display=14)
    set_style()
    img_bee = fig_to_image(fig_bee)
    plt.close(fig_bee)

    fig_imp = plt.figure(figsize=(10, 6))
    img_imp = _draw_importance(fig_imp, shap_values, X.columns.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img_bee)
    axes[0].axis('off')
    axes[0].set_title('(a) SHAP Beeswarm Plot')
    axes[1].imshow(img_imp)
    axes[1].axis('off')
    axes[1].set_title('(b) SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(
        os.path.join(PROJECT_ROOT, "images", "shap_combined_1.jpeg"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


if __name__ == '__main__':
    paint1()
    paint2()
