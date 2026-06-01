import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core.build_dataset import build_dataset
from core.metrics_saver import save_metrics

FONT_SIZE = 18
plt.rcParams['axes.labelsize'] = FONT_SIZE
plt.rcParams['xtick.labelsize'] = FONT_SIZE
plt.rcParams['ytick.labelsize'] = FONT_SIZE
plt.rcParams['legend.fontsize'] = FONT_SIZE
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERFORMANCE_CSV = os.path.join(PROJECT_ROOT, "tables", "performance.csv")

# 使用 data1 作为训练集，data2 作为外部验证集
df_train = build_dataset('data1_extraction.xlsx', features=['anthropometric', 'morphology'])
X_train = df_train.drop('Menopause', axis=1)
y_train = df_train['Menopause'].values

df_val = build_dataset('data2_extraction.xlsx', features=['anthropometric', 'morphology'])
X_val = df_val.drop('Menopause', axis=1)
y_val = df_val['Menopause'].values

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_pred))

save_metrics(y_val, y_pred, y_pred_proba, 'RF', PERFORMANCE_CSV)

fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
roc_df.to_csv(os.path.join(PROJECT_ROOT, "exps", "rf_roc.csv"), index=False)

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['PreM', 'PostM'],
            yticklabels=['PreM', 'PostM'],
            annot_kws={'size': 18, 'weight': 'bold'})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "images", "rf_confusion.jpeg"), dpi=300)
plt.close()
