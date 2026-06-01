"""Utility module to save model performance metrics to CSV."""

import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def save_metrics(y_true, y_pred, y_pred_proba, model_name, output_path):
    """
    Save model performance metrics to a shared CSV file.
    Appends metrics for each model to the same file.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str
        Name of the model
    output_path : str
        Path to the shared performance.csv file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate metrics (rounded to 4 decimal places)
    row = {
        'Model': model_name,
        'AUC': round(roc_auc_score(y_true, y_pred_proba), 4),
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Presion': round(precision_score(y_true, y_pred), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
        'F1 Score': round(f1_score(y_true, y_pred), 4)
    }

    # Check if file exists and model already has metrics
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        # Check if model exists and update that row
        if model_name in df['Model'].values:
            idx = df[df['Model'] == model_name].index[0]
            for key, value in row.items():
                df.loc[idx, key] = value
            print(f"Updated metrics for {model_name}")
        else:
            # Append new row
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to: {output_path}")

    return df
