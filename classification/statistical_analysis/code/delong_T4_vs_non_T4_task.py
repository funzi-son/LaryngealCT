#=====================================
#Delong test for T4 vs non T4 task
#=====================================
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import json

# ---------- DeLong helpers ----------

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

from scipy.stats import norm

def calc_pvalue(aucs, covar):
    l = np.array([[1, -1]])
    z = np.abs(l.dot(aucs)) / np.sqrt(l.dot(covar).dot(l.T))
    return 2 * (1 - norm.cdf(z))

def delong_roc_test(y_true, y_pred1, y_pred2):
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    # sort by first model’s scores (descending)
    order = np.argsort(-y_pred1)
    y_true_sorted = y_true[order]
    y_pred1_sorted = y_pred1[order]
    y_pred2_sorted = y_pred2[order]

    preds = np.vstack((y_pred1_sorted, y_pred2_sorted))
    aucs, cov = fastDeLong(preds, int(y_true_sorted.sum()))
    p = calc_pvalue(aucs, cov)[0][0]
    return p

# ---------- Paths for T4 task ----------


ROOT = Path(r"path\to\Ensemble\T4_Classification")  # common root for this task
MODELS = [
    ("3DCNN",           ROOT / "3DCNN" / "T4_vs_Non_T4_Custom3DCNN_ensemble_predictions.csv"),
    ("ResNet18",        ROOT / "ResNet18" / "T4_vs_Non_T4_ResNet18_ensemble_predictions.csv"),
    ("ResNet50",        ROOT / "ResNet50" / "T4_vs_Non_T4_ResNet50_ensemble_predictions.csv"),
    ("ResNet101",       ROOT / "ResNet101" / "T4_vs_Non_T4_ResNet101_ensemble_predictions.csv"),
    ("DenseNet121",     ROOT / "DenseNet121" / "T4_vs_Non_T4_DenseNet121_ensemble_predictions.csv"),
    ("ResNet50_pretrained",     ROOT / "ResNet50_pretrained" / "T4_vs_Non_T4_ResNet50_pretrained_ensemble_predictions.csv"),
]

# ---------- Load test probabilities (use calibrated if available) ----------

y_true = None
probs = {}
for model, path in MODELS:
    df = pd.read_csv(path)
    if y_true is None:
        y_true = df["y"].values
    else:
        assert np.array_equal(y_true, df["y"].values)
    col = "Calibrated_Probability" if "Calibrated_Probability" in df.columns else "prob_raw"
    probs[model] = df[col].values

# Optional sanity: AUROCs
for m, p in probs.items():
    print(m, "AUROC", roc_auc_score(y_true, p))

# ---------- DeLong pairwise p-values ----------

names = [name for name, _ in MODELS]
delong_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

for m1 in names:
    for m2 in names:
        if m1 == m2:
            delong_matrix.loc[m1, m2] = np.nan
        else:
            p_val = delong_roc_test(y_true, probs[m1], probs[m2])
            delong_matrix.loc[m1, m2] = p_val

out_dir = ROOT / "stats_T4"
out_dir.mkdir(exist_ok=True)
delong_path = out_dir / "delong_pvalues_T4.csv"
delong_matrix.to_csv(delong_path)
print("DeLong p-values saved to", delong_path)

# ---------- Optional: heatmap ----------

plt.figure(figsize=(8, 6))
sns.heatmap(delong_matrix.astype(float), annot=True, fmt=".3f",
            cmap="viridis", cbar_kws={"label": "p-value"})
plt.title("DeLong test pairwise p-values (T4 vs non-T4)")
plt.tight_layout()
plt.savefig(out_dir / "delong_heatmap_T4.png", dpi=600)
plt.close()
