import os
import xgboost as xgb
from sklearn.metrics import classification_report, roc_curve, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.build_dataset import build_dataset
from core.metrics_saver import save_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERFORMANCE_CSV = os.path.join(PROJECT_ROOT, "tables", "performance.csv")

font_size = 18
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['legend.title_fontsize'] = font_size

df_train = build_dataset('data1_extraction.xlsx', features=['anthropometric', 'morphology'])
X_train = df_train.drop('Menopause', axis=1)
y_train = df_train['Menopause'].values

df_val = build_dataset('data2_extraction.xlsx', features=['anthropometric', 'morphology'])
X_val = df_val.drop('Menopause', axis=1)
y_val = df_val['Menopause'].values

model = xgb.XGBClassifier(
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_pred))

save_metrics(y_val, y_pred, y_pred_proba, 'XGBoost', PERFORMANCE_CSV)

y_pred_proba = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
roc_df = pd.DataFrame({
    'fpr': fpr,
    'tpr': tpr,
    'thresholds': thresholds
})
roc_df.to_csv(os.path.join(PROJECT_ROOT, "exps", "xgboost_roc.csv"), index=False)
