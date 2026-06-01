import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from core.build_dataset import build_dataset

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
IMAGES_DIR = path.join(PROJECT_ROOT, "images")

font_size = 18
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['legend.title_fontsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size


def paint2():
    current_dir = path.dirname(path.abspath(__file__))
    df = pd.read_excel(path.abspath(path.join(current_dir, "../dataset/data1_extraction.xlsx")))
    cols = ["Age", "Height", "Weight", "FSH", 'LH', 'P', 'E2', 'Prolactin', 'Testosterone', 'AMH', "OA", "EA", "UA", "BMI", 'Menopause']
    df = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['AMH'])
    features = ["Age", "Height", "Weight", "FSH", 'LH', 'P', 'E2', 'Prolactin', 'Testosterone', 'AMH']
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.boxplot(x='Menopause', y=feature, data=df, ax=axes[i], hue='Menopause', palette='Set1', legend=False)
        axes[i].set_xlabel('')
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['PreM', 'PostM'])
        axes[i].set_ylabel(feature)
    plt.tight_layout()
    plt.savefig(path.join(IMAGES_DIR, 'plot_box.jpeg'), dpi=300, bbox_inches='tight')
    plt.close()


def paint3():
    dataset = build_dataset()
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    sns.kdeplot(data=dataset, x='Age', hue='Menopause', fill=True, palette="Set1", ax=axes[0])
    axes[0].set_title('(a)Age Kernel Density Distribution')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Density')
    axes[0].legend(labels=['PostM', 'PreM'], title='Menopause', loc='best')

    sns.countplot(x='Menopause', data=dataset, palette='Set1', hue='Menopause', ax=axes[1])
    axes[1].set_title('(b)Menopause Status Distribution(Training Set)')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['PreM', 'PostM'])
    axes[1].set_xlabel('Menopause')
    axes[1].set_ylabel('Count')
    axes[1].legend(labels=['PreM', 'PostM'], title='Menopause', loc='best')
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%d', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path.join(IMAGES_DIR, 'kde_age_countplot.jpeg'), dpi=300, bbox_inches='tight')
    plt.close()


def paint4():
    current_dir = path.dirname(path.abspath(__file__))
    df_raw = pd.read_excel(path.abspath(path.join(current_dir, "../dataset/data1_extraction.xlsx")))
    df_raw = df_raw[["Age", "AMH", "FSH", "E2", "Menopause"]].apply(pd.to_numeric, errors='coerce')
    df = df_raw.dropna(subset=['AMH'])

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    sns.scatterplot(data=df, x='Age', y='AMH', hue='Menopause', palette="Set1", hue_order=[1, 0], ax=axes[0, 0])
    axes[0, 0].set_title('(a) Age vs AMH')
    axes[0, 0].legend(labels=['PostM', 'PreM'], title='Menopause', loc='best')

    sns.scatterplot(data=df_raw, x='Age', y='FSH', hue='Menopause', palette="Set1", hue_order=[1, 0], ax=axes[0, 1])
    axes[0, 1].set_title('(b) Age vs FSH')
    axes[0, 1].legend(labels=['PostM', 'PreM'], title='Menopause', loc='best')

    sns.scatterplot(data=df, x='FSH', y='AMH', hue='Menopause', palette="Set1", hue_order=[1, 0], ax=axes[1, 0])
    axes[1, 0].set_title('(c) FSH vs AMH')
    axes[1, 0].legend(labels=['PostM', 'PreM'], title='Menopause', loc='best')

    sns.scatterplot(data=df_raw, x='E2', y='FSH', hue='Menopause', palette="Set1", hue_order=[1, 0], ax=axes[1, 1])
    axes[1, 1].set_title('(d) E2 vs FSH')
    axes[1, 1].legend(labels=['PostM', 'PreM'], title='Menopause', loc='best')

    plt.tight_layout()
    plt.savefig(path.join(IMAGES_DIR, 'scatter_combined.jpeg'), dpi=300, bbox_inches='tight')
    plt.close()


def paint8():
    dataset = build_dataset()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    palette = sns.color_palette('Set1', 2)[::-1]

    for ax, col, title in zip(
        axes,
        ['OA', 'EA', 'UA'],
        ['(a) Ovarian Atrophy', '(b) Endometrial Atrophy', '(c) Uterine Atrophy']
    ):
        df = dataset[dataset[col].isin([0, 1])]
        cross_tab = pd.crosstab(df['Menopause'], df[col], normalize='index')
        cross_tab.plot(kind='bar', stacked=True, color=palette, width=0.5, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Menopause Status')
        ax.set_ylabel('Proportion')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['PreM', 'PostM'], rotation=0)
        ax.legend(['Normal', 'Atrophic'], loc='upper right')
        ax.set_ylim(0, 1.05)
        ax.set_yticks([i * 0.2 for i in range(6)], [f'{i * 20}%' for i in range(6)])

    plt.tight_layout()
    plt.savefig(path.join(IMAGES_DIR, 'atrophy_stacked.jpeg'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    paint2()
    paint3()
    paint4()
    paint8()
