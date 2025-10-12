# =========================================
# ROC overlays for Cross Validation and Test
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
from pathlib import Path
import torch
import pandas as pd

OUTPUT_ROOT="path/to/your/output/dir"

# ========================
# Load validation folds
# ========================
out_dir = Path(OUTPUT_ROOT)
cv_summary = json.load(open(out_dir/"cv_summary_resnet101.json"))  

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for fold in cv_summary:
    fold_dir = out_dir / f"fold_{fold['fold']}"
    val_logits = np.load(fold_dir/"val_logits.npy")
    val_labels = np.load(fold_dir/"val_y.npy")

    # ✅ Stable softmax with PyTorch
    val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()[:, 1]

    fpr, tpr, _ = roc_curve(val_labels, val_probs)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
std_tpr  = np.std(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

# ========================
# Load test predictions
# ========================
test_preds = pd.read_csv(out_dir/"test_predictions_resnet101.csv")  
y_test = test_preds["Label"].values
p_test = test_preds["Probability"].values

fpr_test, tpr_test, _ = roc_curve(y_test, p_test)
auc_test = auc(fpr_test, tpr_test)

# ========================
# Plot overlay
# ========================
plt.figure(figsize=(6,6))

# Validation mean ROC
plt.plot(mean_fpr, mean_tpr, color="b",
         label=f"Validation mean ROC (AUC={mean_auc:.2f})", lw=2)
plt.fill_between(mean_fpr,
                 np.maximum(mean_tpr - std_tpr, 0),
                 np.minimum(mean_tpr + std_tpr, 1),
                 color="b", alpha=0.2, label="±1 std. dev.")

# Test ROC
plt.plot(fpr_test, tpr_test, color="r", lw=2,
         label=f"Test ROC (AUC={auc_test:.2f})")

plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation vs Test ROC")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir/"roc_overlay_val_vs_test.png", dpi=600)
plt.close()

print("✅ Saved Validation vs Test ROC overlay:", out_dir/"roc_overlay_val_vs_test.png")