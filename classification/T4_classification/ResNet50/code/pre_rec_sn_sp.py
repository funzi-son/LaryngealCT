# ==============================================================
# Code to calculate Precision, Reacll, Sensitivity, Specificity
# ==============================================================
import pandas as pd

df_preds = pd.read_csv("path/to/test_predictions_resnet50.csv")
y_true = df_preds["Label"].values
y_scores = df_preds["Probability"].values  # Raw predicted probabilities

#=====================

import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

threshold = 0.5
y_pred = (y_scores >= threshold).astype(int)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")