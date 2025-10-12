# =======================================
# Calculate weighted F1 score
# =======================================

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Path to your predictions file
csv_path = "path/to/test_predictions_resnet50.csv"

# Load predictions
df = pd.read_csv(csv_path)

# True labels and calibrated probabilities
y_true = df['Label'].values
p_cal = df['Calibrated_Probability'].values

# Convert probabilities to predicted labels (threshold 0.5)
y_pred = (p_cal >= 0.5).astype(int)

# Calculate weighted F1 score
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Weighted F1 score: {weighted_f1:.4f}")