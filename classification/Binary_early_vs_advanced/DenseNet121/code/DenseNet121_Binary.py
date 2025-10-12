# =======================================================
# DenseNet121 model for early vs advanced classification
# =======================================================

# =========================
# Setup (installs + imports)
# =========================
%pip -q install pandas openpyxl torchio SimpleITK monai[all] matplotlib

import os
import json
import warnings
import numpy as np
import pandas as pd
import time
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             accuracy_score, classification_report, confusion_matrix,
                             matthews_corrcoef, brier_score_loss, balanced_accuracy_score)

import torchio as tio
import SimpleITK as sitk
from scipy.ndimage import zoom
from monai.networks.nets import DenseNet121

# =========================
# CONFIG
# =========================
IMAGE_DIR   = "path/to/Cropped_Volumes/dir"
META_FILE   = "path/to/LaryngealCT_metadata.xlsx"
TRAIN_SPLIT = "path/to/train_split_binary.xlsx"
TEST_SPLIT  = "path/to/test_split_binary.xlsx"
OUTPUT_ROOT = "path/to/your/output/dir"

LABEL_COL = "Binary_TisT1T2_vs_T3T4"
TARGET_SHAPE = (32, 96, 96)
BATCH_SIZE   = 2
ACCUM_STEPS  = 4
MAX_EPOCHS   = 500
PATIENCE     = 20
LR           = 1e-4
N_FOLDS      = 5
SEED         = 42

LOSS_TYPE   = "focal"
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

NUM_WORKERS = 0

# =========================
# Helper functions
# =========================
def hu_clip_norm(arr, lo=-300, hi=300):
    arr = np.clip(arr, lo, hi)
    m, s = arr.mean(), arr.std()
    return (arr - m) / (s + 1e-6)

def to_filename(tcia_id: str) -> str:
    return tcia_id.replace("-", "_") + "_Cropped_Volume.nrrd"

def create_weighted_sampler(df, label_col):
    class_counts = df[label_col].value_counts().to_dict()
    w_per_class = {c: 1.0 / max(1, n) for c, n in class_counts.items()}
    weights = df[label_col].map(w_per_class).values
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)

# =========================
# Dataset
# =========================
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
        zf = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
        vol = zoom(vol, zf, order=1)
        x = torch.from_numpy(vol[None, ...]).float()
        y = int(row[LABEL_COL])
        subj = tio.Subject(img=tio.ScalarImage(tensor=x))
        subj = (train_tf if self.augment else val_tf)(subj)
        return subj.img.data.float(), torch.tensor(y).long()

# =========================
# Augmentations
# =========================
train_tf = tio.Compose([
    tio.RandomFlip(axes=('LR',), p=0.5),
    tio.RandomAffine(scales=(0.95,1.05), degrees=8, translation=2, p=0.30),
    tio.RandomGamma(log_gamma=(-0.2,0.2), p=0.20),
    tio.RandomNoise(std=(0,0.02), p=0.15),
])
val_tf = tio.Compose([])

# =========================
# Model
# =========================
def build_model(dropout_p=0.30):
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2, pretrained=False)
    # DenseNet121 internally uses dropout, so no extra dropout layer needed.
    return model

# =========================
# Loss functions
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = torch.exp(-ce)
        return ((self.alpha * (1-pt)**self.gamma) * ce).mean()

class WBCE(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        w = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
        self.register_buffer("w", w)
    def forward(self, logits, y):
        return F.cross_entropy(logits, y, weight=self.w)

# =========================
# Train / Eval utils with Timing and Memory Tracking
# =========================
def epoch_pass(model, loader, optimizer=None, loss_obj=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    losses, probs, ys = [], [], []

    if train:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()
        for i, (xb, yb) in enumerate(loader):
            if i % 50 == 0:
                print(f"[DEBUG] Training batch {i+1}/{len(loader)}")
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = loss_obj(logits, yb) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss.detach().item() * ACCUM_STEPS)
            probs.append(F.softmax(logits, dim=1)[:,1].detach().cpu().numpy())
            ys.append(yb.detach().cpu().numpy())
        end_time = time.time()
        duration = end_time - start_time
        max_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"Training epoch duration: {duration:.1f}s | Peak GPU mem: {max_mem:.2f} GB")
    else:
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats(device)
            start_time = time.time()
            for i, (xb, yb) in enumerate(loader):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                if loss_obj is not None:
                    loss = loss_obj(logits, yb)
                    losses.append(loss.detach().item())
                probs.append(F.softmax(logits, dim=1)[:,1].detach().cpu().numpy())
                ys.append(yb.detach().cpu().numpy())
            end_time = time.time()
            duration = end_time - start_time
            max_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"Eval epoch duration: {duration:.1f}s | Peak GPU mem: {max_mem:.2f} GB")

    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    preds = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0,1]).ravel()
    rep = classification_report(ys, preds, output_dict=True, zero_division=0)

    return dict(
        loss=float(np.mean(losses)),
        acc=accuracy_score(ys, preds),
        bal_acc=balanced_accuracy_score(ys, preds),
        mcc=matthews_corrcoef(ys, preds),
        auc=roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan"),
        f1_macro=rep["macro avg"]["f1-score"],
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        sensitivity=tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        specificity=tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        y=ys,
        p=probs
    )

def optimize_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmax(tpr - fpr)]

def temperature_scale(logits_val, y_val, max_iter=100):
    T = torch.tensor([1.0], device=device, requires_grad=True)
    yv = torch.tensor(y_val, device=device)
    lv = torch.tensor(logits_val, device=device)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lv / T, yv)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().cpu().item())

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
        sensitivity=tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        specificity=tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    )

# =========================
# MAIN TRAINING LOOP
# =========================
torch.backends.cudnn.benchmark = True
os.makedirs(OUTPUT_ROOT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

tr = pd.read_excel(TRAIN_SPLIT)
te = pd.read_excel(TEST_SPLIT)

for df in [tr, te]:
    df["Filename"] = df["TCIA_ID"].apply(to_filename)

trm = tr.copy()
tem = te.copy()

print(f"[INFO] Train rows: {len(trm)} | Test rows: {len(tem)}")
print("[INFO] Train label counts:", trm[LABEL_COL].value_counts(dropna=False).to_dict())
print("[INFO] Test label counts:", tem[LABEL_COL].value_counts(dropna=False).to_dict())

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_summ = []

for fold, (idx_tr, idx_va) in enumerate(skf.split(trm, trm[LABEL_COL]), 1):
    print(f"\n===== Fold {fold}/{N_FOLDS} =====")
    fold_dir = Path(OUTPUT_ROOT) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    df_tr = trm.iloc[idx_tr].reset_index(drop=True)
    df_va = trm.iloc[idx_va].reset_index(drop=True)

    pos_count = int((df_tr[LABEL_COL] == 1).sum())
    neg_count = int((df_tr[LABEL_COL] == 0).sum())
    print(f"[INFO] Fold {fold} balance: pos={pos_count} neg={neg_count} ({pos_count/(pos_count+neg_count):.2%} pos)")

    sampler_tr = create_weighted_sampler(df_tr, LABEL_COL)

    tr_loader = DataLoader(
        OnTheFlyDataset(df_tr, IMAGE_DIR, augment=True),
        batch_size=BATCH_SIZE, shuffle=False, sampler=sampler_tr,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    va_loader = DataLoader(
        OnTheFlyDataset(df_va, IMAGE_DIR, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    if LOSS_TYPE == "wbce":
        pos_weight_fold = max(1.0, neg_count / max(1, pos_count))
        loss_obj = WBCE(pos_weight=pos_weight_fold)
        print(f"[INFO] Using WBCE with pos_weight={pos_weight_fold:.3f}")
    else:
        loss_obj = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        print(f"[INFO] Using FocalLoss alpha={FOCAL_ALPHA} gamma={FOCAL_GAMMA}")

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.3, patience=3, min_lr=1e-6, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()

    best_f1, bad, logs = -1, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        trmtr = epoch_pass(model, tr_loader, optimizer=opt, loss_obj=loss_obj, scaler=scaler)
        vamtr = epoch_pass(model, va_loader, optimizer=None, loss_obj=loss_obj)
        logs.append(
            {"epoch": epoch, **{f"tr_{k}": v for k, v in trmtr.items() if k not in ["y", "p"]},
             **{f"va_{k}": v for k, v in vamtr.items() if k not in ["y", "p"]}}
        )

        print(
            f"Epoch {epoch:03d} | tr_loss={trmtr['loss']:.4f} "
            f"| va_loss={vamtr['loss']:.4f} | va_acc={vamtr['acc']:.3f} "
            f"| va_bal_acc={vamtr['bal_acc']:.3f} | sens={vamtr['sensitivity']:.3f} "
            f"| spec={vamtr['specificity']:.3f} | va_f1_macro={vamtr['f1_macro']:.3f} "
            f"| va_auc={vamtr['auc']:.3f} | va_mcc={vamtr['mcc']:.3f}"
        )

        scheduler.step(vamtr["f1_macro"])

        if vamtr["f1_macro"] > best_f1:
            best_f1, bad = vamtr["f1_macro"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, fold_dir / "best.pt")
            model.eval()
            with torch.no_grad():
                logits_val, ys_val_all = [], []
                for xb, yb in va_loader:
                    xb = xb.to(device)
                    logits_val.append(model(xb).detach().cpu().numpy())
                    ys_val_all.append(yb.numpy())
                logits_val = np.concatenate(logits_val, axis=0)
                y_val = np.concatenate(ys_val_all, axis=0)
                p_val = torch.softmax(torch.from_numpy(logits_val), dim=1).numpy()[:, 1]
                np.save(fold_dir / "val_logits.npy", logits_val)
                np.save(fold_dir / "val_y.npy", y_val)
                np.save(fold_dir / "val_p.npy", p_val)
        else:
            bad += 1
        if bad >= PATIENCE:
            print("⏹️ Early stopping.")
            break

    pd.DataFrame(logs).to_csv(fold_dir / "logs.csv", index=False)

    p_val = np.load(fold_dir / "val_p.npy")
    y_val = np.load(fold_dir / "val_y.npy")
    t_opt = optimize_threshold(y_val, p_val)
    logits_val = np.load(fold_dir / "val_logits.npy")
    T = temperature_scale(logits_val, y_val)

    cal_p_val = torch.softmax(torch.from_numpy(logits_val) / T, dim=1).numpy()[:, 1]
    fold_summ.append({
        "fold": fold,
        "val_default": summarize(y_val, p_val, 0.5),
        "val_opt": summarize(y_val, p_val, t_opt),
        "val_cal": summarize(y_val, cal_p_val, t_opt),
        "t_opt": float(t_opt),
        "T": float(T)
    })
    print(f"[Fold {fold}] t_opt={t_opt:.3f} | Temp={T:.3f}")
    m = fold_summ[-1]["val_cal"]
    print(
        f"  Calibrated@topt | acc={m['acc']:.3f} | bal_acc={m['bal_acc']:.3f} "
        f"| sens={m['sensitivity']:.3f} | spec={m['specificity']:.3f} "
        f"| f1_macro={m['f1_macro']:.3f} | auc={m['auc']:.3f} | mcc={m['mcc']:.3f}"
    )

with open(Path(OUTPUT_ROOT) / "cv_summary_densenet121.json", "w") as f:
    json.dump(fold_summ, f, indent=2)
print("Saved CV summary:", Path(OUTPUT_ROOT) / "cv_summary_densenet121.json")

best_idx = int(np.argmax([fs["val_cal"]["f1_macro"] for fs in fold_summ]))
best_fold = fold_summ[best_idx]["fold"]
print(f"\n>>> Best fold by calibrated macro-F1: {best_fold}")

best_dir = Path(OUTPUT_ROOT) / f"fold_{best_fold}"
state = torch.load(best_dir / "best.pt", map_location=device)
t_opt = float(fold_summ[best_idx]["t_opt"])
T = float(fold_summ[best_idx]["T"])

tem = tem.dropna(subset=[LABEL_COL]).reset_index(drop=True)
te_loader = DataLoader(
    OnTheFlyDataset(tem, IMAGE_DIR, augment=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

model = build_model().to(device)
model.load_state_dict(state["model"])
model.eval()

with torch.no_grad():
    logits_te, y_te = [], []
    for xb, yb in te_loader:
        xb = xb.to(device)
        logits_te.append(model(xb).detach().cpu().numpy())
        y_te.append(yb.numpy())
    logits_te = np.concatenate(logits_te, axis=0)
    y_te = np.concatenate(y_te, axis=0)
    p_te = torch.softmax(torch.from_numpy(logits_te), dim=1).numpy()[:, 1]
    p_te_cal = torch.softmax(torch.from_numpy(logits_te) / T, dim=1).numpy()[:, 1]

def pretty_print(tag, d):
    print(
        f"{tag}: acc={d['acc']:.3f} | bal_acc={d['bal_acc']:.3f} "
        f"| sens={d['sensitivity']:.3f} | spec={d['specificity']:.3f} "
        f"| f1_macro={d['f1_macro']:.3f} | auc={d['auc']:.3f} | prAUC={d['pr']:.3f} "
        f"| mcc={d['mcc']:.3f} | brier={d['brier']:.3f} "
        f"| TN={d['TN']} FP={d['FP']} FN={d['FN']} TP={d['TP']}"
    )

print("\n=== TEST RESULTS ===")

res_default = summarize(y_te, p_te, 0.5)
pretty_print("Default@0.5     ", res_default)
res_topt = summarize(y_te, p_te, t_opt)
pretty_print("Uncalib@t_opt   ", res_topt)
res_cal = summarize(y_te, p_te_cal, t_opt)
pretty_print("Calibrated@t_opt", res_cal)

pd.DataFrame(
    {
        "TCIA_ID": tem["TCIA_ID"],
        "Filename": tem["Filename"],
        "Label": y_te,
        "Probability": p_te,
        "Calibrated_Probability": p_te_cal,
    }
).to_csv(Path(OUTPUT_ROOT) / "test_predictions_densenet121.csv", index=False)

with open(Path(OUTPUT_ROOT) / "test_metrics_densenet121.json", "w") as f:
    json.dump(
        {"default": res_default, "topt": res_topt, "calibrated": res_cal, "t_opt": t_opt, "T": T},
        f,
        indent=2,
    )

print("\nSaved test predictions and metrics in:", OUTPUT_ROOT)
