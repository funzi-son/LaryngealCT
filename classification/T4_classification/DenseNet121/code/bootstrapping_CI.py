# =========================================
# Bootsrapping for 95% Confidence Interval
# ==========================================
import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ========= Paths and data loading =========
output_root = r"path\to\Ensemble\T4_Classification\DenseNet121"

pred_path   = os.path.join(output_root, "T4_vs_Non_T4_DenseNet121_ensemble_predictions.csv")
metrics_js  = os.path.join(output_root, "T4_vs_Non_T4_DenseNet121_ensemble_test_metrics.json")
out_csv     = os.path.join(output_root, "T4_vs_Non_T4_DenseNet121_raw_and_calibrated_bootstrap_CIs.csv")
out_xlsx    = os.path.join(output_root, "T4_vs_Non_T4_DenseNet121_raw_and_calibrated_bootstrap_CIs.xlsx")

df_preds = pd.read_csv(pred_path)
y_te      = df_preds["y"].values
p_te_raw  = df_preds["prob_raw"].values
p_te_cal  = df_preds["prob_cal"].values   # calibrated probs

with open(metrics_js, "r") as f:
    js = json.load(f)
t_opt = js["t_opt_ens"]
print("Using t_opt_ens =", t_opt)

# ========= Helpers =========
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def f1_macro_score(y_true, y_pred):
    # Macro-averaged F1 over classes 0 and 1 (matches JSON f1_macro)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def bootstrap_ci(y_true, y_pred_prob, metric_func,
                 n_bootstrap=1000, alpha=0.05, threshold=0.5):
    stats = []
    n = len(y_true)
    idx_all = np.arange(n)

    for _ in range(n_bootstrap):
        idx = resample(idx_all, replace=True, n_samples=n)
        y_true_bs = y_true[idx]
        y_pred_bs = y_pred_prob[idx]

        # Metrics that use probabilities directly
        if metric_func in [roc_auc_score, average_precision_score]:
            stat = metric_func(y_true_bs, y_pred_bs)
        else:
            y_pred_label_bs = (y_pred_bs >= threshold).astype(int)
            if metric_func in [precision_score, recall_score]:
                stat = metric_func(y_true_bs, y_pred_label_bs,
                                   pos_label=1, zero_division=0)
            elif metric_func in [f1_macro_score, specificity_score,
                                 accuracy_score, balanced_accuracy_score]:
                stat = metric_func(y_true_bs, y_pred_label_bs)
            else:
                stat = metric_func(y_true_bs, y_pred_label_bs)
        stats.append(stat)

    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return lower, upper

def per_class_bootstrap_ci(y_true, y_pred_prob, metric_func, class_label,
                           n_bootstrap=1000, alpha=0.05, threshold=0.5):
    stats = []
    n = len(y_true)
    idx_all = np.arange(n)

    for _ in range(n_bootstrap):
        idx = resample(idx_all, replace=True, n_samples=n)
        y_true_bs = y_true[idx]
        y_pred_bs = y_pred_prob[idx]
        y_pred_label_bs = (y_pred_bs >= threshold).astype(int)

        stat = metric_func(
            y_true_bs,
            y_pred_label_bs,
            labels=[0, 1],
            pos_label=class_label,
            zero_division=0,
        )
        stats.append(stat)

    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return lower, upper

# ========= Metrics =========
overall_metrics = {
    "Accuracy": accuracy_score,
    "Balanced Accuracy": balanced_accuracy_score,
    "F1-macro": f1_macro_score,               # fixed macro F1
    "Precision": precision_score,             # positive-class precision
    "Recall (Sensitivity)": recall_score,     # positive-class recall
    "Specificity": specificity_score,
    "ROC AUC": roc_auc_score,
    "PR AUC": average_precision_score,
}

per_class_metrics = {
    "Precision": precision_score,
    "Recall": recall_score,
    "F1": f1_score,
}
classes = [0, 1]

# ========= Compute CIs for RAW@0.5 and CALIBRATED@t_opt =========
rows = []

configs = [
    ("Raw ensemble",        p_te_raw,  0.5),
    ("Calibrated ensemble", p_te_cal,  t_opt),
]

for model_name, probs, thr in configs:
    # Overall
    for name, func in overall_metrics.items():
        thr_used = None if func in [roc_auc_score, average_precision_score] else thr
        low, high = bootstrap_ci(
            y_te, probs, func,
            n_bootstrap=1000,
            threshold=(thr if thr_used is not None else 0.5),
        )
        rows.append({
            "Model": model_name,
            "Type": "Overall",
            "Class": "NA",
            "Metric": name,
            "Threshold": thr if thr_used is not None else "probability",
            "CI_lower": low,
            "CI_upper": high,
        })

    # Per-class
    for cls in classes:
        for name, func in per_class_metrics.items():
            low, high = per_class_bootstrap_ci(
                y_te, probs, func, cls,
                n_bootstrap=1000,
                threshold=thr,
            )
            rows.append({
                "Model": model_name,
                "Type": "Per-class",
                "Class": cls,
                "Metric": name,
                "Threshold": thr,
                "CI_lower": low,
                "CI_upper": high,
            })

df_out = pd.DataFrame(rows)
df_out.to_csv(out_csv, index=False)
df_out.to_excel(out_xlsx, index=False)

print("Saved CIs to:")
print(out_csv)
print(out_xlsx)
