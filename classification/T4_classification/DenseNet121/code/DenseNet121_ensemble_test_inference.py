#===========================================
#DenseNet121 ensemble test inference
#===========================================
import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    brier_score_loss,
)
import SimpleITK as sitk
from scipy.ndimage import zoom
import torchio as tio
from monai.networks.nets import densenet121  # NOTE: lowercase name

# ------------------
# CONFIG (DenseNet121 T4)
# ------------------
IMAGE_DIR   = r"path\to\Cropped_Volumes"
META_FILE   = r"path\to\LaryngealCT_metadata.xlsx"
TRAIN_SPLIT = r"path\to\train_split_T4.xlsx"
TEST_SPLIT  = r"path\to\test_split_T4.xlsx"

OUTPUT_ROOT = r"path\to\Ensemble\T4_Classification\DenseNet121"

LABEL_COL    = "Label_T4"
TARGET_SHAPE = (32, 96, 96)
BATCH_SIZE   = 2
NUM_WORKERS  = 0
N_FOLDS      = 5

TASK_NAME  = "T4_vs_Non_T4"
MODEL_NAME = "DenseNet121"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------
# Helpers (same as training)
# ------------------
def hu_clip_norm(arr, lo=-300, hi=300):
    arr = np.clip(arr, lo, hi)
    m, s = arr.mean(), arr.std()
    return (arr - m) / (s + 1e-6)

def to_filename(tcia_id: str) -> str:
    return tcia_id.replace("-", "_") + "_Cropped_Volume.nrrd"

val_tf = tio.Compose([])

class OnTheFlyDataset(Dataset):
    def __init__(self, df, image_dir, augment=False):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.augment = augment

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
        subj = val_tf(subj)
        return subj.img.data.float(), torch.tensor(y).long()

# ------------------
# Model: replicate Colab DenseNet121Custom
# ------------------
class DenseNet121Custom(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        # this matches Colab: self.backbone with densenet121(...)
        self.backbone = densenet121(spatial_dims=3, in_channels=1, out_channels=2)
        self.dropout = nn.Dropout(dropout_p)

        # get feature_size dynamically exactly as in Colab
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 1, *TARGET_SHAPE)
            feats = self.backbone.features(dummy)
            self.feature_size = feats.numel() // feats.size(0)

        self.classifier = nn.Linear(self.feature_size, 2)

    def forward(self, x):
        feats = self.backbone.features(x)
        feats = feats.view(x.size(0), -1)
        x = self.dropout(feats)
        x = self.classifier(x)
        return x

def build_model(dropout_p=0.3):
    return DenseNet121Custom(dropout_p)

# ------------------
# Metrics utils
# ------------------
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
        sensitivity=tp/(tp+fn) if (tp+fn)>0 else float("nan"),
        specificity=tn/(tn+fp) if (tn+fp)>0 else float("nan"),
    )

def pretty_print(tag, d):
    print(
        f"{tag}: acc={d['acc']:.3f} | bal_acc={d['bal_acc']:.3f} "
        f"| sens={d['sensitivity']:.3f} | spec={d['specificity']:.3f} "
        f"| f1_macro={d['f1_macro']:.3f} | auc={d['auc']:.3f} | prAUC={d['pr']:.3f} "
        f"| mcc={d['mcc']:.3f} | brier={d['brier']:.3f} "
        f"| TN={d['TN']} FP={d['FP']} FN={d['FN']} TP={d['TP']}"
    )

# ------------------
# Load CV summary
# ------------------
cv_summary_path = Path(OUTPUT_ROOT) / "cv_summary_densenet121.json"
with open(cv_summary_path, "r") as f:
    cv_summary = json.load(f)
print("Loaded CV summary from:", cv_summary_path)

# ------------------
# Prepare test set
# ------------------
te = pd.read_excel(TEST_SPLIT)
te["Filename"] = te["TCIA_ID"].apply(to_filename)
te = te.dropna(subset=[LABEL_COL]).reset_index(drop=True)

test_ds = OnTheFlyDataset(te, IMAGE_DIR, augment=False)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

y_list = []
fold_raw_probs = []
fold_cal_probs = []

# ------------------
# Test inference per fold
# ------------------
for fold_entry in cv_summary:
    fold = fold_entry["fold"]
    t_opt_fold = float(fold_entry["t_opt"])
    T_fold     = float(fold_entry["T"])

    fold_dir = Path(OUTPUT_ROOT) / f"fold_{fold}"
    ckpt_path = fold_dir / "best.pt"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    print(f"\n[Fold {fold}] loading {ckpt_path}")
    model = build_model().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    probs_raw, probs_cal, ys = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pr = F.softmax(logits, dim=1)[:, 1]
            pc = F.softmax(logits / T_fold, dim=1)[:, 1]
            probs_raw.append(pr.cpu().numpy())
            probs_cal.append(pc.cpu().numpy())
            ys.append(yb.numpy())

    probs_raw = np.concatenate(probs_raw)
    probs_cal = np.concatenate(probs_cal)
    ys = np.concatenate(ys)

    if not y_list:
        y_list.append(ys)
    else:
        assert np.array_equal(y_list[0], ys), "Test label order mismatch across folds"

    fold_raw_probs.append(probs_raw)
    fold_cal_probs.append(probs_cal)

y_te = y_list[0]
fold_raw_probs = np.stack(fold_raw_probs, axis=0)
fold_cal_probs = np.stack(fold_cal_probs, axis=0)

# ------------------
# Ensemble averaging
# ------------------
p_te_raw_ens = fold_raw_probs.mean(axis=0)
p_te_cal_ens = fold_cal_probs.mean(axis=0)

t_opt_ens = float(np.mean([fe["t_opt"] for fe in cv_summary]))
T_ens     = float(np.mean([fe["T"]     for fe in cv_summary]))
print("\nEnsemble t_opt_ens =", t_opt_ens)
print("Ensemble T_ens      =", T_ens)

# ------------------
# Metrics
# ------------------
res_raw_0p5   = summarize(y_te, p_te_raw_ens, 0.5)
res_cal_0p5   = summarize(y_te, p_te_cal_ens, 0.5)
res_cal_topt  = summarize(y_te, p_te_cal_ens, t_opt_ens)

print("\n=== ENSEMBLE TEST RESULTS (DenseNet121 T4) ===")
pretty_print("RawEns@0.5    ", res_raw_0p5)
pretty_print("CalibEns@0.5  ", res_cal_0p5)
pretty_print("CalibEns@topt ", res_cal_topt)

# ------------------
# Save predictions and metrics
# ------------------
out_dir = Path(OUTPUT_ROOT)
pred_df = pd.DataFrame({
    "task": TASK_NAME,
    "model": MODEL_NAME,
    "TCIA_ID": te["TCIA_ID"],
    "Filename": te["Filename"],
    "y": y_te,
    "prob_raw": p_te_raw_ens,
    "prob_cal": p_te_cal_ens,
})
pred_path = out_dir / "T4_vs_Non_T4_DenseNet121_ensemble_predictions.csv"
pred_df.to_csv(pred_path, index=False)

metrics = {
    "RawEns@0.5":    res_raw_0p5,
    "CalibEns@0.5":  res_cal_0p5,
    "CalibEns@topt": res_cal_topt,
    "t_opt_ens": t_opt_ens,
    "T_ens": T_ens,
}
metrics_path = out_dir / "T4_vs_Non_T4_DenseNet121_ensemble_test_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved ensemble predictions to:", pred_path)
print("Saved ensemble metrics to:", metrics_path)
