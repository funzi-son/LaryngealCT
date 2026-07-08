#========================================
#Wilcoxon test for early vs advanced task
#========================================
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(r"path\to\Ensemble\Binary")

cv_summaries = {
    "3DCNN":       ROOT / "3DCNN"       / "cv_summary_custom3dcnn.json",
    "ResNet18":    ROOT / "ResNet18"    / "cv_summary_resnet18.json",
    "ResNet50":    ROOT / "ResNet50"    / "cv_summary_resnet50.json",
    "ResNet101":   ROOT / "ResNet101"   / "cv_summary_resnet101.json",
    "DenseNet121": ROOT / "DenseNet121" / "cv_summary_densenet121.json",
    "ResNet50_pretrained": ROOT / "ResNet50_pretrained" / "cv_summary_resnet50_pretrained.json",
}

# metrics inside each fold entry to compare
metrics = ["auc", "f1_macro"]   # from fold["cal_metrics"]["auc"], ["f1_macro"]

# -------- 1. Load per-fold calibrated metrics --------
cv_metrics = {}   # {model: {metric: np.array of length N_folds}}

for model, path in cv_summaries.items():
    with open(path, "r") as f:
        folds = json.load(f)   # list of folds
    cv_metrics[model] = {}
    for met in metrics:
        cv_metrics[model][met] = np.array([fold["val_cal"][met] for fold in folds])

# Quick sanity check
for model in cv_metrics:
    print(model, "AUC per fold:", cv_metrics[model]["auc"])

models = list(cv_summaries.keys())
n = len(models)

# -------- 2. Pairwise Wilcoxon p-values --------
out_dir = ROOT / "stats_T4"
out_dir.mkdir(exist_ok=True)

for met in metrics:
    p_mat = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            x = cv_metrics[models[i]][met]
            y = cv_metrics[models[j]][met]
            # paired signed-rank test (two-sided)
            stat, p = wilcoxon(x, y)
            p_mat[i, j] = p_mat[j, i] = p
    df_p = pd.DataFrame(p_mat, index=models, columns=models)
    out_path = out_dir / f"wilcoxon_{met}_cv_calibrated.csv"
    df_p.to_csv(out_path)
    print(f"Saved Wilcoxon p-values for {met} to {out_path}")
