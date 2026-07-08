#=============================================================================
#Code to generate combined ROC overlays for all six models (T4 vs non-T4 task)
#=============================================================================
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.switch_backend("Agg")

EXPERIMENTS = [
    # (panel title, experiment_root_dir, test_predictions_filename)
    ("3DCNN",              r"path\to\Ensemble\T4_Classification\3DCNN",          "T4_vs_Non_T4_Custom3DCNN_ensemble_predictions.csv"),
    ("ResNet18",           r"path\to\Ensemble\T4_Classification\ResNet18",       "T4_vs_Non_T4_ResNet18_ensemble_predictions.csv"),
    ("ResNet50",           r"path\to\Ensemble\T4_Classification\ResNet50",       "T4_vs_Non_T4_ResNet50_ensemble_predictions.csv"),
    ("ResNet101",          r"path\to\Ensemble\T4_Classification\ResNet101",      "T4_vs_Non_T4_ResNet101_ensemble_predictions.csv"),
    ("DenseNet121",        r"path\to\Ensemble\T4_Classification\DenseNet121",    "T4_vs_Non_T4_DenseNet121_ensemble_predictions.csv"),
    ("ResNet50 pretrained",r"path\to\Ensemble\T4_Classification\ResNet50_pretrained",    "T4_vs_Non_T4_ResNet50_pretrained_ensemble_predictions.csv"),
]

N_FOLDS = 5
OUT_FIG = r"path\to\Ensemble\T4_Classification\roc_overlays_T4_vs_nonT4.png"  

# -------- helper: mean ROC ±1 SD over folds --------
def mean_roc_with_ci(fpr_list, tpr_list, num_points=100):
    grid_fpr = np.linspace(0.0, 1.0, num_points)
    interp_tprs = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        interp = np.interp(grid_fpr, fpr, tpr)
        interp[0] = 0.0
        interp_tprs.append(interp)
    interp_tprs = np.asarray(interp_tprs)
    mean_tpr = interp_tprs.mean(axis=0)
    std_tpr = interp_tprs.std(axis=0)
    mean_tpr[-1] = 1.0
    return grid_fpr, mean_tpr, std_tpr

# -------- main figure --------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

fig.suptitle(
    r"$\mathbf{Validation\ (5\text{-}fold\ CV)\ vs\ Test\ ROC\ curves\ for\ all\ DL\ models\ for\ T4\ Classification\ task}$",
    fontsize=16,
    y=0.98
)

for ax, (title, root, test_csv) in zip(axes, EXPERIMENTS):
    ...

for ax, (title, root, test_csv) in zip(axes, EXPERIMENTS):
    root = Path(root)

    # ---- collect per‑fold validation ROC ----
    fprs, tprs, aucs = [], [], []
    for fold in range(1, N_FOLDS + 1):
        fold_dir = root / f"fold_{fold}"
        val_logits = np.load(fold_dir / "val_logits.npy")
        val_labels = np.load(fold_dir / "val_y.npy")
        # softmax to get probabilities for class 1
        exp_logits = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
        val_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        val_probs = val_probs[:, 1]

        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc(fpr, tpr))

    mean_fpr, mean_tpr, std_tpr = mean_roc_with_ci(fprs, tprs)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color="b",
            label=f"Validation mean ROC (AUC={mean_auc:.2f}±{std_auc:.2f})")
    ax.fill_between(mean_fpr,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    color="b", alpha=0.2, label="±1 std. dev.")

    # ---- test ROC from calibrated probabilities ----
    test_df = pd.read_csv(root / test_csv)
    y_test = test_df["y"].values
    if "Calibrated_Probability" in test_df.columns:
        p_test = test_df["Calibrated_Probability"].values
    else:
        p_test = test_df["prob_raw"].values

    fpr_test, tpr_test, _ = roc_curve(y_test, p_test)
    auc_test = auc(fpr_test, tpr_test)
    ax.plot(fpr_test, tpr_test, color="r", lw=2,
            label=f"Test ROC (AUC={auc_test:.2f})")

    # diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)

    ax.set_title(f"Validation (5-fold CV) vs Test ROC - {title}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=600)
print(f"Saved figure to {OUT_FIG}")
