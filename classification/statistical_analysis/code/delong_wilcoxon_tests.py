import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# DeLong Implementation
# ------------------------
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
        T[i:j] = 0.5*(i+j-1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m+n], dtype=float)

    for r in range(k):
        tx[r] = compute_midrank(positive_examples[r])
        ty[r] = compute_midrank(negative_examples[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m*n) - (m+1.0)/(2*n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx/m + sy/n
    return aucs, delongcov

def calc_pvalue(aucs, covar):
    l = np.array([[1, -1]])
    z = np.abs(l.dot(aucs)) / np.sqrt(l.dot(covar).dot(l.T))
    from scipy.stats import norm
    return 2*(1-norm.cdf(z))

def delong_roc_test(y_true, y_pred1, y_pred2):
    order = np.argsort(-y_pred1)
    y_true = np.array(y_true)[order]
    y_pred1 = np.array(y_pred1)[order]
    y_pred2 = np.array(y_pred2)[order]

    predictions_sorted_transposed = np.vstack((y_pred1, y_pred2))
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, int(sum(y_true)))
    return calc_pvalue(aucs, delongcov)[0][0]

# ------------------------
# Paths (example)
# ------------------------
models = {
    "3DCNN": "path/to/test_predictions_3dcnn.csv",
    "ResNet18": "path/to/test_predictions_resnet18.csv",
    "ResNet50": "path/to/test_predictions_resnet50.csv",
    "ResNet101": "path/to/test_predictions_resnet101.csv",
    "DenseNet121": "path/to/test_predictions_densenet121.csv",
    "ResNet50_TL": "path/to/test_predictions_resnet50_TL.csv",
}


cv_summaries = {
    "3DCNN": "path/to/\cv_summary_3dcnn.json",
    "ResNet18": "path/to/cv_summary_resnet18.json",
    "ResNet50": "path/to/cv_summary_resnet50.json",
    "ResNet101": "path/to/cv_summary_resnet101_2.json",
    "DenseNet121": "path/to/cv_summary_densenet121.json",
    "ResNet50_TL": "path/to/cv_summary_resnet50_TL.json",
}

# ------------------------
# Load test probabilities
# ------------------------
y_true = None
probs = {}
for model, path in models.items():
    df = pd.read_csv(path)
    if y_true is None:
        y_true = df["Label"].values
    probs[model] = df["Probability"].values

# ------------------------
# DeLong Pairwise P-values
# ------------------------
delong_matrix = pd.DataFrame(index=models.keys(), columns=models.keys(), dtype=float)

for m1 in models.keys():
    for m2 in models.keys():
        if m1 == m2:
            delong_matrix.loc[m1, m2] = np.nan
        else:
            try:
                p_val = delong_roc_test(y_true, probs[m1], probs[m2])
                delong_matrix.loc[m1, m2] = p_val
            except Exception as e:
                delong_matrix.loc[m1, m2] = np.nan

# Save CSV
delong_csv = Path(r"D:\RQ1\MICCAI\stats\delong_matrix.csv")
delong_csv.parent.mkdir(parents=True, exist_ok=True)
delong_matrix.to_csv(delong_csv)
print(f"✅ DeLong p-values saved to {delong_csv}")

# Plot Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(delong_matrix.astype(float), annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'p-value'})
plt.title("DeLong Test (AUROC) Pairwise p-values T4 classification")
plt.tight_layout()
plt.savefig(delong_csv.parent / "delong_heatmap.png", dpi=300)
plt.close()

# ------------------------
# Wilcoxon Pairwise (per-fold metrics)
# ------------------------
metrics = ["f1_macro", "balanced_accuracy", "auc"]
wilcoxon_results = {m: pd.DataFrame(index=models.keys(), columns=models.keys(), dtype=float) for m in metrics}

cv_metrics = {}
for model, path in cv_summaries.items():
    with open(path, "r") as f:
        folds = json.load(f)
    cv_metrics[model] = {met: [fold["val_cal"][met] for fold in folds] for met in metrics}

for met in metrics:
    for m1 in models.keys():
        for m2 in models.keys():
            if m1 == m2:
                wilcoxon_results[met].loc[m1, m2] = np.nan
            else:
                try:
                    stat, p = wilcoxon(cv_metrics[m1][met], cv_metrics[m2][met])
                    wilcoxon_results[met].loc[m1, m2] = p
                except ValueError:
                    wilcoxon_results[met].loc[m1, m2] = np.nan

    # Save CSV per metric
    csv_path = delong_csv.parent / f"wilcoxon_{met}.csv"
    wilcoxon_results[met].to_csv(csv_path)
    print(f"✅ Wilcoxon {met} p-values saved to {csv_path}")

    # Heatmap per metric
    plt.figure(figsize=(8,6))
    sns.heatmap(wilcoxon_results[met].astype(float), annot=True, fmt=".3f", cmap="magma", cbar_kws={'label': 'p-value'})
    plt.title(f"Wilcoxon Test (per-fold {met}) Pairwise p-values")
    plt.tight_layout()
    plt.savefig(delong_csv.parent / f"wilcoxon_{met}_heatmap.png", dpi=300)
    plt.close()