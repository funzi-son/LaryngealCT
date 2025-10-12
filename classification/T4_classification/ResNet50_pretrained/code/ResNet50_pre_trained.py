# =====================================================================================
# ResNet50 model pretrained on MedicalNet weights for early vs advanced classification
# =====================================================================================

# =========================
# Setup (installs + imports)
# =========================
%pip -q install pandas openpyxl torchio SimpleITK monai[all] matplotlib

import os, json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    accuracy_score, classification_report, confusion_matrix,
    matthews_corrcoef, brier_score_loss, balanced_accuracy_score
)

import torchio as tio
import SimpleITK as sitk
from scipy.ndimage import zoom
from monai.networks.nets import resnet50  # MONAI 3D ResNet50

# =========================
# CONFIG
# =========================
IMAGE_DIR   = "path/to/Cropped_Volumes/dir"
META_FILE   = "path/to/LaryngealCT_metadata.xlsx"
TRAIN_SPLIT = "path/to/train_split_T4.xlsx"
TEST_SPLIT  = "path/to/test_split_T4.xlsx"
OUTPUT_ROOT = "path/to/your/output/dir"


LABEL_COL = "Label_T4"
TARGET_SHAPE = (32, 96, 96)
BATCH_SIZE = 2
ACCUM_STEPS = 4
MAX_EPOCHS = 500
PATIENCE = 20
LR = 1e-4
N_FOLDS = 5
SEED = 42

LOSS_TYPE   = "focal"
FOCAL_ALPHA = 8.0
FOCAL_GAMMA = 2.0
NUM_WORKERS = 0

# ---- Transfer Learning switches ----
USE_MEDICALNET     = True
TL_MODE            = "full"   # "head" or "full"

# Discriminative LR (Approach 2)
BASE_LR   = 3e-5
HEAD_LR   = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 0

# =========================
# Repro / helpers
# =========================
def set_seeds(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)

def hu_clip_norm(arr, lo=-300, hi=300):
    arr = np.clip(arr, lo, hi)
    m, s = arr.mean(), arr.std()
    return (arr - m) / (s + 1e-6)

def to_filename(tcia_id: str):
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

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = self.image_dir / row["Filename"]
        img = sitk.ReadImage(str(filepath))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        vol = hu_clip_norm(vol)
        zf = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
        vol = zoom(vol, zf, order=1)
        x = torch.from_numpy(vol[None, ...]).float()
        y = int(row[LABEL_COL])
        subj = tio.Subject(img=tio.ScalarImage(tensor=x))
        subj = train_tf(subj) if self.augment else val_tf(subj)
        return subj.img.data.float(), torch.tensor(y).long()

# =========================
# Augmentations
# =========================
train_tf = tio.Compose([
    tio.RandomFlip(axes=('LR',), p=0.5),
    tio.RandomAffine(scales=(0.95, 1.05), degrees=8, translation=2, p=0.30),
    tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.20),
    tio.RandomNoise(std=(0, 0.02), p=0.15),
])
val_tf = tio.Compose([])

# =========================
# Model builders (with Dropout head)
# =========================
class MedicalNetResNet50WithHead(nn.Module):
    def __init__(self, dropout_p=0.3, num_classes=2, use_medicalnet=True):
        super().__init__()
        if use_medicalnet:
            print("[INFO] Loading MedicalNet pretrained ResNet50...")
            backbone = torch.hub.load("Warvito/MedicalNet-models", "medicalnet_resnet50")
            self.backbone = backbone
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_p),
                nn.Linear(2048, num_classes)
            )
        else:
            from monai.networks.nets import resnet50
            backbone = resnet50(spatial_dims=3, n_input_channels=1, num_classes=num_classes, pretrained=False)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop fc
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_p),
                nn.Linear(backbone.fc.in_features, num_classes)
            )

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.avgpool(feats)
        out = self.head(pooled)
        return out


def build_resnet50_with_dropout(dropout_p=0.3, num_classes=2, use_medicalnet=True):
    return MedicalNetResNet50WithHead(dropout_p=dropout_p, num_classes=num_classes, use_medicalnet=use_medicalnet)


# =========================
# Losses
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((self.alpha * (1-pt)**self.gamma) * ce).mean()

class WBCE(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__(); w = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
        self.register_buffer("weights", w)
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weights)


# =========================
# Train / Eval
# =========================
def epoch_pass(model, dataloader, optimizer=None, loss_fn=None, scaler=None, accum_steps=1, device="cuda"):
    training = optimizer is not None
    model.train(training)
    losses, probs, ys = [], [], []

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_time = time.time()

    if training:
        optimizer.zero_grad(set_to_none=True)
        for i, (xb, yb) in enumerate(dataloader):
            if i % 50 == 0:
                print(f"[DEBUG] Training batch {i+1}/{len(dataloader)}")
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = loss_fn(logits, yb) / accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.detach().item() * accum_steps)
            probs.append(F.softmax(logits, dim=1)[:,1].detach().cpu().numpy())
            ys.append(yb.detach().cpu().numpy())
    else:
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                if loss_fn is not None:
                    loss = loss_fn(logits, yb)
                    losses.append(loss.detach().item())
                probs.append(F.softmax(logits, dim=1)[:,1].detach().cpu().numpy())
                ys.append(yb.detach().cpu().numpy())

    duration = time.time() - start_time
    peak_gpu_mem = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

    probs = np.concatenate(probs); ys = np.concatenate(ys)
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0,1]).ravel()
    rep = classification_report(ys, preds, output_dict=True, zero_division=0)

    return dict(
        loss=float(np.mean(losses)) if len(losses) else float('nan'),
        accuracy=accuracy_score(ys, preds),
        balanced_accuracy=balanced_accuracy_score(ys, preds),
        mcc=matthews_corrcoef(ys, preds),
        auc=roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan"),
        f1_macro=rep["macro avg"]["f1-score"],
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        sensitivity=tp/(tp+fn) if (tp+fn) > 0 else float("nan"),
        specificity=tn/(tn+fp) if (tn+fp) > 0 else float("nan"),
        elapsed_time=duration,
        peak_gpu_mem_gb=peak_gpu_mem,
        y=ys, p=probs
    )

# =========================
# Threshold / Calibration
# =========================
def optimize_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmax(tpr - fpr)]

def temperature_scale(logits_val, y_val, max_iter=100, device="cuda"):
    T = torch.tensor([1.0], device=device, requires_grad=True)
    yv = torch.tensor(y_val, device=device)
    lv = torch.tensor(logits_val, device=device)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lv / T, yv); loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().cpu().item())

def summarize(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    rep = classification_report(y, yhat, output_dict=True, zero_division=0)
    return dict(
        accuracy=accuracy_score(y, yhat),
        balanced_accuracy=balanced_accuracy_score(y, yhat),
        f1_macro=rep["macro avg"]["f1-score"],
        auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        precision_recall_auc=average_precision_score(y, p),
        mcc=matthews_corrcoef(y, yhat) if len(np.unique(y)) > 1 else float("nan"),
        brier=brier_score_loss(y, p),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        sensitivity=tp/(tp+fn) if (tp+fn) > 0 else float("nan"),
        specificity=tn/(tn+fp) if (tn+fp) > 0 else float("nan"),
    )

# =========================
# Optimizers for TL
# =========================
def freeze_backbone_head_only(model):
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True  # classifier head

def make_discriminative_optimizer(model, base_lr=BASE_LR, head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY):
    head_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head_params if n.startswith("fc") else base_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": base_params, "lr": base_lr},
         {"params": head_params, "lr": head_lr}],
        weight_decay=weight_decay
    )
    return opt

# =========================
# Main training loop
# =========================
torch.backends.cudnn.benchmark = True
os.makedirs(OUTPUT_ROOT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seeds(SEED)

train_df = pd.read_excel(TRAIN_SPLIT)
test_df  = pd.read_excel(TEST_SPLIT)

for df in [train_df, test_df]:
    df["Filename"] = df["TCIA_ID"].apply(to_filename)

train_df = train_df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
test_df  = test_df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

print(f"[INFO] Train rows: {len(train_df)} | Test rows: {len(test_df)}")
print("[INFO] Train label counts:", train_df[LABEL_COL].value_counts().to_dict())
print("[INFO] Test  label counts:",  test_df[LABEL_COL].value_counts().to_dict())

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_summ = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[LABEL_COL]), 1):
    print(f"\n===== Fold {fold}/{N_FOLDS} =====")
    fold_dir = Path(OUTPUT_ROOT) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    df_tr = train_df.iloc[train_idx].reset_index(drop=True)
    df_va = train_df.iloc[val_idx].reset_index(drop=True)

    sampler = create_weighted_sampler(df_tr, LABEL_COL) if LOSS_TYPE == "wbce" else None

    tr_loader = DataLoader(OnTheFlyDataset(df_tr, IMAGE_DIR, augment=True),
                           batch_size=BATCH_SIZE,
                           sampler=sampler,
                           shuffle=(sampler is None),
                           num_workers=NUM_WORKERS, pin_memory=True)

    va_loader = DataLoader(OnTheFlyDataset(df_va, IMAGE_DIR, augment=False),
                           batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    # ----- Build model -----
    model = build_resnet50_with_dropout(dropout_p=0.30, num_classes=2, use_medicalnet=USE_MEDICALNET)
    model = model.to(device)

    # ----- Loss -----
    if LOSS_TYPE == "wbce":
        pos = (df_tr[LABEL_COL] == 1).sum(); neg = len(df_tr) - pos
        loss_fn = WBCE(pos_weight=neg / max(1, pos))
        print(f"[INFO] Using WBCE (pos_weight={neg/max(1,pos):.3f})")
    else:
        loss_fn = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        print(f"[INFO] Using FocalLoss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ----- Choose TL mode -----
    if USE_MEDICALNET and TL_MODE == "head":
        # Approach 1: train head only (all backbone frozen)
        freeze_backbone_head_only(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=3, min_lr=1e-6, verbose=True)
        print("[INFO] TL mode: head-only fine-tune (classifier only).")
    elif USE_MEDICALNET and TL_MODE == "full":
        # Approach 2: full fine-tune with discriminative LRs
        for p in model.parameters(): p.requires_grad = True
        optimizer = make_discriminative_optimizer(model, base_lr=BASE_LR, head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6, verbose=True)
        print(f"[INFO] TL mode: full fine-tune (base_lr={BASE_LR}, head_lr={HEAD_LR}).")
    else:
        # From-scratch (your original)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=3, min_lr=1e-6, verbose=True)
        print("[INFO] Training from scratch.")

    # (Optional) short warm-up for full TL
    if USE_MEDICALNET and TL_MODE == "full" and WARMUP_EPOCHS > 0:
        # temporarily freeze backbone and train head a few epochs
        freeze_backbone_head_only(model)
        warm_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        print(f"[INFO] Warm-up head-only for {WARMUP_EPOCHS} epochs...")
        for ep in range(1, WARMUP_EPOCHS + 1):
            trm = epoch_pass(model, tr_loader, warm_opt, loss_fn, scaler, accum_steps=ACCUM_STEPS, device=device)
            vam = epoch_pass(model, va_loader, None, loss_fn, scaler, device=device)
            print(f"[WARMUP {ep:02d}] tr_loss={trm['loss']:.4f} | va_f1={vam['f1_macro']:.3f} | va_auc={vam['auc']:.3f}")
        # unfreeze everything and recreate discriminative optimizer
        for p in model.parameters(): p.requires_grad = True
        optimizer = make_discriminative_optimizer(model, base_lr=BASE_LR, head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY)

    # ----- Main train loop -----
    best_f1, no_improve = -np.inf, 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        trm = epoch_pass(model, tr_loader, optimizer, loss_fn, scaler, accum_steps=ACCUM_STEPS, device=device)
        vam = epoch_pass(model, va_loader, None, loss_fn, scaler, device=device)

        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in trm.items()},
                        **{f"val_{k}": v for k, v in vam.items()}})

        print(f"Epoch {epoch:03d} | tr_loss={trm['loss']:.4f} | va_loss={vam['loss']:.4f} "
              f"| va_acc={vam['accuracy']:.3f} | va_f1={vam['f1_macro']:.3f} | va_auc={vam['auc']:.3f}")

        scheduler.step(vam["f1_macro"])

        if vam["f1_macro"] > best_f1:
            best_f1, no_improve = vam["f1_macro"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, fold_dir / "best.pt")
            # store val logits for calibration
            model.eval()
            val_logits, val_labels = [], []
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(device)
                    val_logits.append(model(xb).detach().cpu().numpy())
                    val_labels.append(yb.cpu().numpy())
            np.save(fold_dir / "val_logits.npy", np.concatenate(val_logits))
            np.save(fold_dir / "val_labels.npy", np.concatenate(val_labels))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"⏹️  Early stopping at epoch {epoch}.")
                break

    pd.DataFrame(history).to_csv(fold_dir / "training_log.csv", index=False)

    # ----- Calibration & fold summary -----
    val_logits = np.load(fold_dir / "val_logits.npy")
    val_labels = np.load(fold_dir / "val_labels.npy")
    val_probs  = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()[:, 1]

    t_opt = optimize_threshold(val_labels, val_probs)
    T     = temperature_scale(val_logits, val_labels, device=device)

    val_probs_cal = torch.softmax(torch.from_numpy(val_logits) / T, dim=1).numpy()[:, 1]

    metrics_raw = summarize(val_labels, val_probs, 0.5)
    metrics_opt = summarize(val_labels, val_probs, t_opt)
    metrics_cal = summarize(val_labels, val_probs_cal, t_opt)

    print(f"[Fold {fold}] t_opt={t_opt:.3f} | Temp={T:.3f}")
    print(f"  Calibrated@topt | acc={metrics_cal['accuracy']:.3f} | bal_acc={metrics_cal['balanced_accuracy']:.3f} "
          f"| sens={metrics_cal['sensitivity']:.3f} | spec={metrics_cal['specificity']:.3f} "
          f"| f1={metrics_cal['f1_macro']:.3f} | auc={metrics_cal['auc']:.3f} | mcc={metrics_cal['mcc']:.3f}")

    fold_summ.append({
        "fold": fold,
        "raw_metrics": metrics_raw,
        "opt_metrics": metrics_opt,
        "val_cal": metrics_cal,
        "optimal_threshold": float(t_opt),
        "temperature": float(T),
        "tl_mode": TL_MODE,
        "used_medicalnet": bool(USE_MEDICALNET),
    })

# Save CV summary
with open(Path(OUTPUT_ROOT) / "cv_summary_resnet18.json", "w") as f:
    json.dump(fold_summ, f, indent=2)
print(f"Saved CV summary to {OUTPUT_ROOT}")

# =========================
# Test evaluation with best fold
# =========================
best_fold = max(fold_summ, key=lambda x: x["cal_metrics"]["f1_macro"])["fold"]
best_info = next(fs for fs in fold_summ if fs["fold"] == best_fold)
print(f"\n>>> Evaluating best fold {best_fold} on test set (mode={best_info['tl_mode']}, MedicalNet={best_info['used_medicalnet']}).")

best_dir = Path(OUTPUT_ROOT) / f"fold_{best_fold}"
checkpoint = torch.load(best_dir / "best.pt", map_location=device)

# Rebuild the same model architecture for inference
model = build_resnet50_with_dropout(dropout_p=0.30, num_classes=2, use_medicalnet=USE_MEDICALNET)
model = model.to(device)

model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

test_loader = DataLoader(OnTheFlyDataset(test_df, IMAGE_DIR, augment=False),
                         batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

test_logits, test_labels = [], []
t0 = time.time()
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        test_logits.append(model(xb).detach().cpu().numpy())
        test_labels.append(yb.cpu().numpy())
t1 = time.time()

test_logits = np.concatenate(test_logits)
test_labels = np.concatenate(test_labels)
test_probs  = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()[:, 1]

# Use the best fold's calibrated threshold and temperature
temperature      = best_info["temperature"]
optimal_threshold= best_info["optimal_threshold"]
test_probs_cal   = torch.softmax(torch.from_numpy(test_logits) / temperature, dim=1).numpy()[:, 1]

test_metrics_raw = summarize(test_labels, test_probs, 0.5)
test_metrics_opt = summarize(test_labels, test_probs, optimal_threshold)
test_metrics_cal = summarize(test_labels, test_probs_cal, optimal_threshold)

print(f"\nTest inference time for {len(test_loader.dataset)} samples: {t1 - t0:.2f}s")
print("\n=== TEST RESULTS ===")
print("Raw@0.5          :", test_metrics_raw)
print("Uncalib@t_opt    :", test_metrics_opt)
print("Calibrated@t_opt :", test_metrics_cal)

pd.DataFrame({
    "TCIA_ID": test_df["TCIA_ID"].values,
    "Filename": test_df["Filename"].values,
    "Label": test_labels,
    "Probability": test_probs,
    "Calibrated_Probability": test_probs_cal,
}).to_csv(Path(OUTPUT_ROOT) / "test_predictions_resnet18.csv", index=False)

with open(Path(OUTPUT_ROOT) / "test_metrics_resnet18.json", "w") as f:
    json.dump({
        "raw_threshold": test_metrics_raw,
        "optimized_threshold": test_metrics_opt,
        "calibrated": test_metrics_cal,
        "temperature": float(temperature),
        "optimal_threshold": float(optimal_threshold),
        "tl_mode": TL_MODE,
        "used_medicalnet": bool(USE_MEDICALNET)
    }, f, indent=2)

print(f"\nSaved test predictions and metrics into: {OUTPUT_ROOT}")