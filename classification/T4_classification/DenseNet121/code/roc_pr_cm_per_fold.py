# =============================================================
# Calculate weighted ROC,P-R curve, Confusion matrices per fold
# =============================================================
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import json

# Load ensemble predictions
df = pd.read_csv(r"path\to\Ensemble\T4_Classification\DenseNet121\T4_vs_Non_T4_DenseNet121_ensemble_predictions.csv")  # y_true, prob_raw, prob_cal
y_true   = df.iloc[:, 4].values          # adjust if your column order differs
prob_raw = df.iloc[:, 5].values
prob_cal = df.iloc[:, 6].values          # calibrated probs

# Load t_opt_ens from json
with open(r"path\to\Ensemble\T4_Classification\DenseNet121\T4_vs_Non_T4_DenseNet121_ensemble_test_metrics.json", "r") as f:
    metrics = json.load(f)
t_opt = metrics["t_opt_ens"]             
print("t_opt_ens:", t_opt)

def plot_and_save_roc_pr_confusion(y_true, prob_pred, threshold, save_dir, prefix, figsize=(15,5)):
    fpr, tpr, _ = roc_curve(y_true, prob_pred)
    roc_auc_val = roc_auc_score(y_true, prob_pred)

    precision, recall, _ = precision_recall_curve(y_true, prob_pred)
    pr_auc_val = auc(recall, precision)

    y_pred = (prob_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # ROC
    axs[0].plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
    axs[0].plot([0,1],[0,1],'--', color='gray')
    axs[0].set_title("ROC Curve")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend()

    # PR
    axs[1].step(recall, precision, where='post', label=f"PR AUC = {pr_auc_val:.3f}")
    axs[1].set_title("Precision-Recall Curve")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].legend()

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(ax=axs[2], cmap=plt.cm.Blues, colorbar=False)
    axs[2].set_title(f"Confusion Matrix (threshold={threshold:.3f})")

    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save combined figure
    fig.savefig(save_dir / f"{prefix}_all_plots.png", dpi=300)

    # Optional: save each separately
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(save_dir / f"{prefix}_roc_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.step(recall, precision, where='post', label=f"PR AUC = {pr_auc_val:.3f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(save_dir / f"{prefix}_pr_curve.png", dpi=300)
    plt.close()

    plt.figure()
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (threshold={threshold:.3f})")
    plt.savefig(save_dir / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close()

    print(prefix, "confusion matrix:\n", cm)

save_folder = r"path\to\Ensemble\T4_Classification\DenseNet121"

# Raw ensemble at 0.5
plot_and_save_roc_pr_confusion(
    y_true, prob_raw, threshold=0.5,
    save_dir=save_folder, prefix="raw_t0p5"
)

# Calibrated ensemble at optimal threshold t_opt_ens
plot_and_save_roc_pr_confusion(
    y_true, prob_cal, threshold=t_opt,
    save_dir=save_folder, prefix="calib_topt"
)
