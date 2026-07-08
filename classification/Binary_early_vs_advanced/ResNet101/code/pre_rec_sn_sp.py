# ==============================================================
# Code to calculate Precision, Recall, Sensitivity, Specificity
# ==============================================================
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Paths
base = r"path\to\\Ensemble\Binary\ResNet101"
pred_path  = base + r"\Early_vs_Advanced_ResNet101_ensemble_predictions.csv"
metrics_js = base + r"\Early_vs_Advanced_ResNet101_ensemble_test_metrics.json"
out_csv    = base + r"\Early_vs_Advanced_ResNet101_raw_vs_calibrated_metrics.csv"
out_xlsx   = base + r"\Early_vs_Advanced_ResNet101_raw_vs_calibrated_metrics.xlsx"

# Load predictions
df_preds = pd.read_csv(pred_path)
y_true      = df_preds["y"].values
y_scores_raw = df_preds["prob_raw"].values
y_scores_cal = df_preds["prob_cal"].values      # calibrated
# [file:53]

# Load optimal threshold
with open(metrics_js, "r") as f:
    js = json.load(f)
t_opt = js["t_opt_ens"]                          # ≈ 0.476...
# [file:56]

def metrics_for(y_true, y_scores, thr):
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    y_pred = (y_scores >= thr).astype(int)
    prec   = precision_score(y_true, y_pred)
    rec    = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens   = tp / (tp + fn)
    spec   = tn / (tn + fp)
    return prec, rec, sens, spec, tn, fp, fn, tp

# RAW @ 0.5
prec_r, rec_r, sens_r, spec_r, tn_r, fp_r, fn_r, tp_r = metrics_for(y_true, y_scores_raw, 0.5)

# CALIBRATED @ t_opt
prec_c, rec_c, sens_c, spec_c, tn_c, fp_c, fn_c, tp_c = metrics_for(y_true, y_scores_cal, t_opt)

# Build table
df_out = pd.DataFrame([
    {
        "Model": "Raw ensemble",
        "Threshold": 0.5,
        "Precision": prec_r,
        "Recall": rec_r,
        "Sensitivity": sens_r,
        "Specificity": spec_r,
        "TN": tn_r, "FP": fp_r, "FN": fn_r, "TP": tp_r,
    },
    {
        "Model": "Calibrated ensemble",
        "Threshold": t_opt,
        "Precision": prec_c,
        "Recall": rec_c,
        "Sensitivity": sens_c,
        "Specificity": spec_c,
        "TN": tn_c, "FP": fp_c, "FN": fn_c, "TP": tp_c,
    },
])

# Save to CSV and Excel
df_out.to_csv(out_csv, index=False)
df_out.to_excel(out_xlsx, index=False)

print("Saved to:")
print(out_csv)
print(out_xlsx)
