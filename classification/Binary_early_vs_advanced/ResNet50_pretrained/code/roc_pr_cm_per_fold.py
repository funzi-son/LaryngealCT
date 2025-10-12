# =============================================================
# Calculate weighted ROC,P-R curve, Confusion matrices per fold
# =============================================================

import torch

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay

import numpy as np

OUTPUT_ROOT="path/to/your/output/dir"

def softmax_np(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Class 0 (early)", "Class 1 (advanced)"], 
        cmap=plt.cm.Blues, values_format="d"
    )
    disp.ax_.set_title(title)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

def plot_roc_curve(y_true, y_prob, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

def plot_pr_curve(y_true, y_prob, title, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color="green", lw=2, label=f"AUC = {pr_auc:.3f}")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

# ============================
# Confusion Matrices + Curves
# ============================
out_dir = Path(OUTPUT_ROOT)

# Test set (Raw)
y_pred_raw = (test_probs >= 0.5).astype(int)
plot_confusion_matrix(test_labels, y_pred_raw, "Test Confusion Matrix (Raw@0.5)", out_dir / "confusion_test_raw.png")
plot_roc_curve(test_labels, test_probs, "ROC Curve (Test Raw@0.5)", out_dir / "roc_test_raw.png")
plot_pr_curve(test_labels, test_probs, "PR Curve (Test Raw@0.5)", out_dir / "pr_test_raw.png")

# Test set (Calibrated)
y_pred_cal = (test_probs_cal >= optimal_threshold).astype(int)
plot_confusion_matrix(test_labels, y_pred_cal, "Test Confusion Matrix (Calibrated@t_opt)", out_dir / "confusion_test_cal.png")
plot_roc_curve(test_labels, test_probs_cal, "ROC Curve (Test Calibrated@t_opt)", out_dir / "roc_test_cal.png")
plot_pr_curve(test_labels, test_probs_cal, "PR Curve (Test Calibrated@t_opt)", out_dir / "pr_test_cal.png")

print("✅ Saved test confusion matrices, ROC and PR curves.")

# ============================
# Validation (per-fold)
# ============================
# Load the saved CV summary file

cv_summary_path = Path(OUTPUT_ROOT) / "cv_summary_resnet50_TL.json"   # adjust filename
with open(cv_summary_path, "r") as f:
    fold_summ = json.load(f)

print(f"Loaded {len(fold_summ)} folds from CV summary.")

for fold in fold_summ:
    fold_dir = out_dir / f"fold_{fold['fold']}"
    fold_dir.mkdir(exist_ok=True)

    # Raw predictions
    val_logits = np.load(fold_dir / "val_logits.npy")
    val_labels = np.load(fold_dir / "val_y.npy")
    val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    plot_confusion_matrix(val_labels, val_preds,
                          f"Fold {fold['fold']} Confusion Matrix (Raw@0.5)",
                          fold_dir / "confusion_val_raw.png")
    plot_roc_curve(val_labels, val_probs,
                   f"Fold {fold['fold']} ROC (Raw@0.5)",
                   fold_dir / "roc_val_raw.png")
    plot_pr_curve(val_labels, val_probs,
                  f"Fold {fold['fold']} PR (Raw@0.5)",
                  fold_dir / "pr_val_raw.png")

    # Calibrated predictions
    T = fold["T"]
    t_opt = fold["t_opt"]
    val_probs_cal = torch.softmax(torch.from_numpy(val_logits) / T, dim=1).numpy()[:, 1]
    val_preds_cal = (val_probs_cal >= t_opt).astype(int)

    plot_confusion_matrix(val_labels, val_preds_cal,
                          f"Fold {fold['fold']} Confusion Matrix (Calibrated@t_opt)",
                          fold_dir / "confusion_val_cal.png")
    plot_roc_curve(val_labels, val_probs_cal,
                   f"Fold {fold['fold']} ROC (Calibrated@t_opt)",
                   fold_dir / "roc_val_cal.png")
    plot_pr_curve(val_labels, val_probs_cal,
                  f"Fold {fold['fold']} PR (Calibrated@t_opt)",
                  fold_dir / "pr_val_cal.png")

print("✅ Saved validation confusion matrices, ROC and PR curves per fold.")