# =======================================
# Calculate and save per-class metrics
# =======================================
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# ======= Paths and config (3D-CNN, T4) =======
output_root = Path(r"path\to\Ensemble\Binary\ResNet50")
pred_path   = output_root / "Early_vs_Advanced_ResNet50_ensemble_predictions.csv"
metrics_path = output_root / "Early_vs_Advanced_ResNet50_ensemble_test_metrics.json"

# For reference only; we use prob_cal and thresholds from metrics JSON
df = pd.read_csv(pred_path)
with open(metrics_path, "r") as f:
    mjson = json.load(f)

y_true   = df["y"].astype(int).values
prob_raw = df["prob_raw"].values
prob_cal = df["prob_cal"].values

topt_ens = float(mjson["t_opt_ens"])   # ensemble t_opt
print("Using ensemble t_opt:", topt_ens)

def compute_row(y, prob, thr, row_label):
    """
    Compute a full row for the per-class table:
    - confusion matrix TN,FP,FN,TP
    - per-class precision/recall/F1 for class 0 and 1
    - ROC AUC, PR AUC for reference
    """
    y_pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()

    # Class 0 metrics
    prec0 = precision_score(y, y_pred, labels=[0], average=None, zero_division=0)[0]
    rec0  = recall_score(y, y_pred,   labels=[0], average=None, zero_division=0)[0]
    f10   = f1_score(y, y_pred,       labels=[0], average=None, zero_division=0)[0]

    # Class 1 metrics
    prec1 = precision_score(y, y_pred, labels=[1], average=None, zero_division=0)[0]
    rec1  = recall_score(y, y_pred,   labels=[1], average=None, zero_division=0)[0]
    f11   = f1_score(y, y_pred,       labels=[1], average=None, zero_division=0)[0]

    rocauc = roc_auc_score(y, prob)
    prauc  = average_precision_score(y, prob)

    return {
        "row": row_label,
        "threshold": thr,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "precision_0": prec0, "recall_0": rec0, "f1_0": f10,
        "precision_1": prec1, "recall_1": rec1, "f1_1": f11,
        "roc_auc": rocauc, "pr_auc": prauc,
    }

rows = []

# 1) Raw @ 0.5
rows.append(compute_row(y_true, prob_raw, 0.5, row_label="Raw@0.5"))

# 2) Calibrated @ 0.5
rows.append(compute_row(y_true, prob_cal, 0.5, row_label="Cal@0.5"))

# 3) Calibrated @ t_opt_ens (this is what you’ll usually use for the “Cal.” row)
rows.append(compute_row(y_true, prob_cal, topt_ens, row_label="Cal@topt_ens"))

df_rows = pd.DataFrame(rows)
out_csv = output_root / "Early_vs_Advanced_ResNet50_per_class_from_ensemble.csv"
df_rows.to_csv(out_csv, index=False)
print("Saved per-class table rows to:", out_csv)
print(df_rows)
