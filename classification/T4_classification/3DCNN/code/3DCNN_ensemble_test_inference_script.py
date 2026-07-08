#=========================================
#3DCNN ensemble test inference script
#=========================================
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from scipy.ndimage import zoom
import torchio as tio

from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    accuracy_score, classification_report, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score, brier_score_loss
)

# =========================
# Configurations (T4 vs non-T4, 3D-CNN)
# =========================

IMAGE_DIR   = r"path\to\Cropped_Volumes"
META_FILE   = r"path\to\LaryngealCT_metadata.xlsx"
TRAIN_SPLIT = r"path\to\train_split_T4.xlsx"
TEST_SPLIT  = r"path\to\test_split_T4.xlsx"

OUTPUT_ROOT = r"path\to\Ensemble\T4_Classification\3DCNN"

LABEL_COL = "Label_T4"

TARGET_SHAPE = (32, 96, 96)
BATCH_SIZE = 2
N_FOLDS = 5
SEED = 42

NUM_WORKERS = 0  # Windows compatibility

# Task/model names for saving
TASK_NAME  = "T4_vs_nonT4"
MODEL_NAME = "Custom3DCNN"

# =========================
# Reproducibility and device
# =========================

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================
# Helper functions
# =========================

def hu_clip_norm(arr, lo=-300, hi=300):
    arr = np.clip(arr, lo, hi)
    m, s = arr.mean(), arr.std()
    return (arr - m) / (s + 1e-6)

def to_filename(tcia_id: str):
    return tcia_id.replace("-", "_") + "_Cropped_Volume.nrrd"

# No augmentation at test time
val_transforms = tio.Compose([])

class OnTheFlyDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        f = self.image_dir / row["Filename"]
        img = sitk.ReadImage(str(f))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        vol = hu_clip_norm(vol)
        zf = [t/s for t, s in zip(TARGET_SHAPE, vol.shape)]
        vol = zoom(vol, zf, order=1)
        x = torch.from_numpy(vol[None, ...]).float()
        y = int(row[LABEL_COL])
        subj = tio.Subject(img=tio.ScalarImage(tensor=x))
        subj = val_transforms(subj)
        return subj.img.data.float(), torch.tensor(y).long()

class Custom3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def summarize(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    rep = classification_report(y, yhat, output_dict=True, zero_division=0)
    return dict(
        acc=accuracy_score(y, yhat),
        bal_acc=balanced_accuracy_score(y, yhat),
        f1_macro=rep["macro avg"]["f1-score"],
        auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        pr=average_precision_score(y, p),
        mcc=matthews_corrcoef(y, yhat) if len(np.unique(y)) > 1 else float("nan"),
        brier=brier_score_loss(y, p),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        sensitivity=tp/(tp+fn) if (tp+fn) > 0 else float("nan"),
        specificity=tn/(tn+fp) if (tn+fp) > 0 else float("nan")
    )

def pretty_print(tag, d):
    print(
        f"{tag}: acc={d['acc']:.3f} | bal_acc={d['bal_acc']:.3f} "
        f"| sens={d['sensitivity']:.3f} | spec={d['specificity']:.3f} "
        f"| f1_macro={d['f1_macro']:.3f} | auc={d['auc']:.3f} | prAUC={d['pr']:.3f} "
        f"| mcc={d['mcc']:.3f} | brier={d['brier']:.3f} "
        f"| TN={d['TN']} FP={d['FP']} FN={d['FN']} TP={d['TP']}"
    )

# =========================
# Load metadata and prepare test loader
# =========================

tr = pd.read_excel(TRAIN_SPLIT)
te = pd.read_excel(TEST_SPLIT)

for df in [tr, te]:
    df["Filename"] = df["TCIA_ID"].apply(to_filename)

tem = te.dropna(subset=[LABEL_COL]).reset_index(drop=True)

print(f"[INFO] Test rows (non-NaN {LABEL_COL}): {len(tem)}")
print("[INFO] Test label counts:", tem[LABEL_COL].value_counts(dropna=False).to_dict())

te_loader = DataLoader(
    OnTheFlyDataset(tem, IMAGE_DIR),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =========================
# Load CV summary (T, t_opt per fold)
# =========================

cv_summary_path = Path(OUTPUT_ROOT) / "cv_summary_custom3dcnn.json"
assert cv_summary_path.exists(), f"Missing {cv_summary_path} – ensure training has been run."

fold_summ = json.load(open(cv_summary_path, "r"))

# =========================
# 5-fold ensemble inference on test set
# =========================

all_fold_logits = []   # [n_folds, n_test, 2]
all_fold_probs  = []   # [n_folds, n_test]
all_fold_ids    = tem["TCIA_ID"].values

for fold in range(1, N_FOLDS + 1):
    print(f"\n=== Test inference for fold {fold} ===")
    fold_dir = Path(OUTPUT_ROOT) / f"fold_{fold}"
    state_path = fold_dir / "best.pt"
    assert state_path.exists(), f"Missing checkpoint: {state_path}"

    state = torch.load(state_path, map_location=device)

    model = Custom3DCNN().to(device)
    model.load_state_dict(state["model"])
    model.eval()

    logits_list = []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(device)
            logits_list.append(model(xb).detach().cpu().numpy())

    logits_fold = np.concatenate(logits_list, axis=0)  # [n_test, 2]
    probs_fold  = torch.softmax(torch.from_numpy(logits_fold), dim=1).numpy()[:, 1]

    all_fold_logits.append(logits_fold)
    all_fold_probs.append(probs_fold)

all_fold_logits = np.stack(all_fold_logits, axis=0)  # [n_folds, n_test, 2]
all_fold_probs  = np.stack(all_fold_probs, axis=0)   # [n_folds, n_test]

# Ensemble (average) over folds
logits_ens = all_fold_logits.mean(axis=0)   # [n_test, 2]
probs_ens  = all_fold_probs.mean(axis=0)    # [n_test]
y_te = tem[LABEL_COL].astype(int).values

# =========================
# Ensemble calibration parameters (T_ens, t_opt_ens)
# =========================

T_vals    = np.array([fs["T"] for fs in fold_summ])
topt_vals = np.array([fs["t_opt"] for fs in fold_summ])

T_ens    = float(T_vals.mean())
topt_ens = float(topt_vals.mean())

print(f"\nEnsemble calibration parameters: T_ens={T_ens:.3f}, t_opt_ens={topt_ens:.3f}")

# Apply temperature scaling to ensemble logits
p_te_ens_raw = probs_ens

logits_ens_t = torch.from_numpy(logits_ens) / T_ens
p_te_ens_cal = torch.softmax(logits_ens_t, dim=1).numpy()[:, 1]

# =========================
# Compute and print overall metrics (raw + calibrated)
# =========================

print("\n=== TEST RESULTS (5-fold ensemble) ===")

# Raw ensemble probs at 0.5
res_raw_0p5 = summarize(y_te, p_te_ens_raw, 0.5)
pretty_print("RawEns@0.5    ", res_raw_0p5)

# Calibrated ensemble at 0.5
res_cal_0p5 = summarize(y_te, p_te_ens_cal, 0.5)
pretty_print("CalibEns@0.5  ", res_cal_0p5)

# Calibrated ensemble at t_opt_ens
res_cal_topt = summarize(y_te, p_te_ens_cal, topt_ens)
pretty_print("CalibEns@topt ", res_cal_topt)

# =========================
# Save per-case and per-fold predictions
# =========================

out_root = Path(OUTPUT_ROOT)

df_ens = pd.DataFrame(
    {
        "Task": TASK_NAME,
        "Model": MODEL_NAME,
        "TCIA_ID": tem["TCIA_ID"],
        "Filename": tem["Filename"],
        "y": y_te,
        "prob_raw": p_te_ens_raw,
        "prob_cal": p_te_ens_cal,
    }
)
ens_path = out_root / f"{TASK_NAME}_{MODEL_NAME}_ensemble_predictions.csv"
df_ens.to_csv(ens_path, index=False)
print("Saved ensemble per-case predictions to:", ens_path)

rows = []
n_folds, n_test = all_fold_probs.shape
for f in range(n_folds):
    for i in range(n_test):
        rows.append(
            {
                "Task": TASK_NAME,
                "Model": MODEL_NAME,
                "fold": f + 1,
                "TCIA_ID": all_fold_ids[i],
                "y": int(y_te[i]),
                "prob_fold": float(all_fold_probs[f, i]),
            }
        )
df_by_fold = pd.DataFrame(rows)
byfold_path = out_root / f"{TASK_NAME}_{MODEL_NAME}_ensemble_predictions_by_fold.csv"
df_by_fold.to_csv(byfold_path, index=False)
print("Saved per-fold predictions to:", byfold_path)

# =========================
# Save metrics summary JSON (now like ResNet18)
# =========================

metrics_json_path = out_root / f"{TASK_NAME}_{MODEL_NAME}_ensemble_test_metrics.json"
with open(metrics_json_path, "w") as f:
    json.dump(
        {
            "RawEns@0.5":    res_raw_0p5,
            "CalibEns@0.5":  res_cal_0p5,
            "CalibEns@topt": res_cal_topt,
            "t_opt_ens": topt_ens,
            "T_ens": T_ens,
        },
        f,
        indent=2,
    )
print("Saved ensemble test metrics to:", metrics_json_path)
