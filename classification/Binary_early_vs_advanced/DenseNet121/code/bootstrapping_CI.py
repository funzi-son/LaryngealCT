# =========================================
# Bootsrapping for 95% Confidence Interval
# ==========================================
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)

# Load test predictions CSV to get true labels and calibrated probs
output_root = OUTPUT_ROOT="path/to/your/output/dir"
df_preds = pd.read_csv(f"{output_root}/test_predictions_densenet121.csv")
y_te = df_preds['Label'].values
p_te_cal = df_preds['Calibrated_Probability'].values

# Specificity function 
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# Bootstrap function for general metrics
def bootstrap_ci(y_true, y_pred_prob, metric_func, n_bootstrap=1000, alpha=0.05, threshold=0.5):
    stats = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = resample(np.arange(n), replace=True, n_samples=n)
        y_true_bs = y_true[idx]
        y_pred_bs = y_pred_prob[idx]
        if metric_func.__name__ in ['roc_auc_score', 'average_precision_score']:
            stat = metric_func(y_true_bs, y_pred_bs)
        else:
            y_pred_label_bs = (y_pred_bs >= threshold).astype(int)
            stat = metric_func(y_true_bs, y_pred_label_bs)
        stats.append(stat)
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper

# Bootstrap function for per-class metrics
def per_class_bootstrap_ci(y_true, y_pred_prob, metric_func, class_label,
                           n_bootstrap=1000, alpha=0.05, threshold=0.5):
    stats = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = resample(np.arange(n), replace=True, n_samples=n)
        y_true_bs = y_true[idx]
        y_pred_bs = y_pred_prob[idx]
        y_pred_label_bs = (y_pred_bs >= threshold).astype(int)
        stat = metric_func(y_true_bs, y_pred_label_bs,
                           labels=[0,1], pos_label=class_label, zero_division=0)
        stats.append(stat)
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper

# Overall metrics to evaluate
overall_metrics = {
    'Accuracy': accuracy_score,
    'Balanced Accuracy': balanced_accuracy_score,
    'F1 Score': f1_score,
    'Specificity': specificity_score,
    'Sensitivity': recall_score,
    'AUC': roc_auc_score,
}

# Per-class metrics for both classes (e.g., 0=majority, 1=minority)
per_class_metrics = {
    'Precision': precision_score,
    'Recall': recall_score,
    'F1': f1_score
}
classes = [0, 1]

print("95% Confidence Intervals for Overall Metrics:")
for name, func in overall_metrics.items():
    thr = 0.5 if name != 'AUC' else None
    if thr is not None:
        low, high = bootstrap_ci(y_te, p_te_cal, func, n_bootstrap=1000, threshold=thr)
    else:
        low, high = bootstrap_ci(y_te, p_te_cal, func, n_bootstrap=1000)
    print(f"{name}: {low:.3f} – {high:.3f}")

print("\n95% Confidence Intervals for Per-Class Metrics:")
for cls in classes:
    print(f"\nClass {cls} metrics:")
    for name, func in per_class_metrics.items():
        low, high = per_class_bootstrap_ci(y_te, p_te_cal, func, cls, n_bootstrap=1000, threshold=0.5)
        print(f"  {name}: {low:.3f} – {high:.3f}")