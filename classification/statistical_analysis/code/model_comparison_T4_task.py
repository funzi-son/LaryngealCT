#====================================================================
#code to compare performances for all six models (run before Delong)
#====================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# MODELS and ROOT as before
ROOT = Path(r"path\to\Ensemble\T4_Classification")  # common root for this task
MODELS = [
    ("3DCNN",           ROOT / "3DCNN" / "T4_vs_Non_T4_Custom3DCNN_ensemble_predictions.csv"),
    ("ResNet18",        ROOT / "ResNet18" / "T4_vs_Non_T4_ResNet18_ensemble_predictions.csv"),
    ("ResNet50",        ROOT / "ResNet50" / "T4_vs_Non_T4_ResNet50_ensemble_predictions.csv"),
    ("ResNet101",       ROOT / "ResNet101" / "T4_vs_Non_T4_ResNet101_ensemble_predictions.csv"),
    ("DenseNet121",     ROOT / "DenseNet121" / "T4_vs_Non_T4_DenseNet121_ensemble_predictions.csv"),
    ("ResNet50_pretrained",     ROOT / "ResNet50_pretrained" / "T4_vs_Non_T4_ResNet50_pretrained_ensemble_predictions.csv"),
]

names = []
scores = []
labels = None

for name, path in MODELS:
    df = pd.read_csv(path)
    if labels is None:
        labels = df["y"].values
    else:
        print(name, "labels equal to first model?",
              np.array_equal(labels, df["y"].values))
    if "Calibrated_Probability" in df.columns:
        s = df["Calibrated_Probability"].values
    else:
        s = df["prob_raw"].values
    names.append(name)
    scores.append(s)

scores = np.vstack(scores)

# 1) Basic label sanity
print("Unique labels and counts:", np.unique(labels, return_counts=True))

# 2) AUROC per model
for i, name in enumerate(names):
    auc_i = roc_auc_score(labels, scores[i])
    print(f"{name}: AUROC = {auc_i:.3f}")

# 3) Check if any pair of score vectors is (almost) identical
n = len(names)
for i in range(n):
    for j in range(i + 1, n):
        same = np.allclose(scores[i], scores[j])
        print(f"{names[i]} vs {names[j]} identical scores? {same}")
