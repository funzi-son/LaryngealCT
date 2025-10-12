# ===================================================
# Code to calculate and save per-class metrics
# ===================================================
import json, csv
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

OUTPUT_ROOT="path/to/your/output/dir"

# Load predictions if not in memory
preds_file = Path(OUTPUT_ROOT) / "test_predictions_custom3dcnn.csv"  
df_preds = pd.read_csv(preds_file)

test_labels = df_preds["Label"].values
test_probs = df_preds["Probability"].values
test_probs_cal = df_preds["Calibrated_Probability"].values

# Load threshold + temperature info from your saved fold summary
fold_summary = json.load(open(Path(OUTPUT_ROOT) / "cv_summary_custom3dcnn.json"))
best_fold = max(fold_summary, key=lambda x: x["val_cal"]["f1_macro"])
optimal_threshold = best_fold["t_opt"]
temperature = best_fold["T"]

# ---- Per-class metrics function ----
def per_class_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Class 1 (positive, e.g., T4 / Advanced)
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    # Class 0 (negative, e.g., Non-T4 / Early)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

    # Clinical metrics
    TPR = recall_1  # sensitivity
    TNR = recall_0  # specificity
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
    FNR = fn / (fn + tp) if (fn + tp) > 0 else 0
    PPV = precision_1
    NPV = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Curves
    roc_auc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        "precision_0": precision_0, "recall_0": recall_0, "f1_0": f1_0,
        "precision_1": precision_1, "recall_1": recall_1, "f1_1": f1_1,
        "TPR": TPR, "TNR": TNR, "FPR": FPR, "FNR": FNR, "PPV": PPV, "NPV": NPV,
        "roc_auc": roc_auc, "pr_auc": pr_auc
    }

# ---- Compute metrics ----
metrics_perclass = {
    "raw@0.5": per_class_metrics(test_labels, test_probs, threshold=0.5),
    "uncalibrated@t_opt": per_class_metrics(test_labels, test_probs, threshold=optimal_threshold),
    "calibrated@t_opt": per_class_metrics(test_labels, test_probs_cal, threshold=optimal_threshold),
}

# ---- Save JSON ----
with open(Path(OUTPUT_ROOT) / "per_class_metrics_custom3dcnn.json", "w") as f:
    json.dump(metrics_perclass, f, indent=2)

# ---- Save CSV ----
csv_save_path = Path(OUTPUT_ROOT) / "per_class_metrics_custom3dcnn.csv"
with open(csv_save_path, mode="w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["mode", "metric", "value"])
    for mode, dct in metrics_perclass.items():
        for k, v in dct.items():
            writer.writerow([mode, k, v])

print(f"✅ Saved per-class metrics:\n- {csv_save_path}\n- {OUTPUT_ROOT}/per_class_metrics_custom3dcnn.json")