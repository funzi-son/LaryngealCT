# =======================================================
# DenseNet121 model for T4 classification
# =======================================================

# =========================
# Setup (installs + imports)
# =========================

%pip install pandas openpyxl torchio SimpleITK monai[all] matplotlib scikit-learn scipy

import os
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path
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
from monai.networks.nets import densenet121
import torch.nn as nn
# =========================
# Configurations
# =========================
IMAGE_DIR = "path/to/Cropped_Volumes"
META_FILE = "path/to/LaryngealCT_metadata.xlsx"
TRAIN_SPLIT = "path/to/train_split.xlsx"
TEST_SPLIT = "path/to/test_split.xlsx"
OUTPUT_ROOT = "path/to/your/output/folder"


LABEL_COL = "LabelT4"  # 1 = T4, 0 = NonT4
TARGET_SHAPE = (32, 96, 96)
BATCH_SIZE = 2
ACCUM_STEPS = 4
MAX_EPOCHS = 500
PATIENCE = 20
LR = 1e-4
N_FOLDS = 5
SEED = 42
LOSS_TYPE = "focal"  # Options: "focal" or "wbce"
FOCAL_ALPHA = 8.0
FOCAL_GAMMA = 2.0
NUM_WORKERS = 0


# =========================
# Helper functions
# =========================
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


def convert_to_builtin_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_types(v) for v in obj]
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj


# =========================
# Model definition
# =========================


from monai.networks.nets import densenet121
import torch.nn as nn
import torch


class DenseNet121Custom(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.backbone = densenet121(spatial_dims=3, in_channels=1, out_channels=2)
        self.dropout = nn.Dropout(dropout_p)
        # Get feature size dynamically
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 1, *TARGET_SHAPE)
            feats = self.backbone.features(dummy)
            self.feature_size = feats.numel() // feats.size(0)  # Flatten excluding batch dim
        self.classifier = nn.Linear(self.feature_size, 2)


    def forward(self, x):
        feats = self.backbone.features(x)
        feats = feats.view(x.size(0), -1)  # flatten for classifier
        x = self.dropout(feats)
        x = self.classifier(x)
        return x



def build_model(dropout_p=0.3):
    return DenseNet121Custom(dropout_p)



# =========================
# Dataset class
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
# Augmentation transforms
# =========================
train_tf = tio.Compose([
    tio.RandomFlip(axes=('LR',), p=0.5),
    tio.RandomAffine(scales=(0.95, 1.05), degrees=8, translation=2, p=0.3),
    tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.2),
    tio.RandomNoise(std=(0, 0.02), p=0.15),
])
val_tf = tio.Compose([])


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
        return ((self.alpha * (1 - pt) ** self.gamma) * ce).mean()


class WBCE(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        w = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
        self.register_buffer("w", w)
    def forward(self, logits, y):
        return F.cross_entropy(logits, y, weight=self.w)


# =========================
# Training and evaluation epoch
# =========================
def epoch_pass(model, dataloader, optimizer=None, loss_fn=None, scaler=None, return_logits=False):
    is_train = optimizer is not None
    model.train(is_train)
    losses, preds, trues = [], [], []
    logits_list = []
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, labels) / ACCUM_STEPS
        if is_train:
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss = loss.detach()
        losses.append(loss.item() * ACCUM_STEPS)
        probabilities = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds.extend(probabilities)
        trues.extend(labels.cpu().numpy())
        if return_logits:
            logits_list.append(outputs.detach().cpu().numpy())
    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    preds = np.array(preds)
    trues = np.array(trues)
    binary_preds = (preds >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(trues, binary_preds, labels=[0,1]).ravel()
    stats = classification_report(trues, binary_preds, output_dict=True, zero_division=0)
    auc = roc_auc_score(trues, preds) if len(np.unique(trues)) > 1 else float('nan')
    pr_auc = average_precision_score(trues, preds) if len(np.unique(trues)) > 1 else float('nan')
    mcc = matthews_corrcoef(trues, binary_preds) if len(np.unique(trues)) > 1 else float('nan')
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    precision = stats['1']['precision'] if '1' in stats else float('nan')
    recall = stats['1']['recall'] if '1' in stats else float('nan')
    f1 = stats['1']['f1-score'] if '1' in stats else float('nan')
    metrics = {
        'loss': np.mean(losses),
        'accuracy': accuracy_score(trues, binary_preds),
        'balanced_accuracy': balanced_accuracy_score(trues, binary_preds),
        'mcc': mcc,
        'auc': auc,
        'f1_macro': stats['macro avg']['f1-score'],
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'pr_auc': pr_auc,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'elapsed_time': elapsed,
        'peak_memory_gb': peak_mem
    }
    if return_logits:
        return metrics, trues, preds, np.concatenate(logits_list, axis=0)
    else:
        return metrics, trues, preds


# =========================
# Threshold optimization
# =========================
def optimize_threshold(y_true, pred_probs):
    fpr, tpr, thresholds = roc_curve(y_true, pred_probs)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


# =========================
# Temperature scaling calibration
# =========================
def temperature_scale(logits_val, y_val, max_iter=100):
    T = torch.tensor([1.0], device=device, requires_grad=True)
    y_val_t = torch.tensor(y_val, device=device)
    logits_val_t = torch.tensor(logits_val, device=device)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_val_t / T, y_val_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(T.detach().cpu().item())


# =========================
# Results summarization
# =========================
def summarize(y_true, pred_probs, threshold=0.5):
    pred_labels = (pred_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_labels, labels=[0,1]).ravel()
    rep = classification_report(y_true, pred_labels, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, pred_labels),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred_labels),
        "f1_macro": rep["macro avg"]["f1-score"],
        "auc": roc_auc_score(y_true, pred_probs) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, pred_probs),
        "mcc": matthews_corrcoef(y_true, pred_labels) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": brier_score_loss(y_true, pred_probs),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    }


# =========================
# Main
# =========================
torch.backends.cudnn.benchmark = True
os.makedirs(OUTPUT_ROOT, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# Load splits and filenames
train_df = pd.read_excel(TRAIN_SPLIT)
test_df = pd.read_excel(TEST_SPLIT)
for df in [train_df, test_df]:
    df["Filename"] = df["TCIA_ID"].apply(to_filename)


train_df_filtered = train_df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
test_df_filtered = test_df.dropna(subset=[LABEL_COL]).reset_index(drop=True)


print(f"[INFO] Train rows: {len(train_df_filtered)} | Test rows: {len(test_df_filtered)}")
print("[INFO] Train label counts:", train_df_filtered[LABEL_COL].value_counts(dropna=False).to_dict())
print("[INFO] Test label counts:", test_df_filtered[LABEL_COL].value_counts(dropna=False).to_dict())


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_summ = []


for fold, (idx_tr, idx_va) in enumerate(skf.split(train_df_filtered, train_df_filtered[LABEL_COL]), 1):
    print(f"\n===== Fold {fold}/{N_FOLDS} =====")
    fold_dir = Path(OUTPUT_ROOT) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    df_tr = train_df_filtered.iloc[idx_tr].reset_index(drop=True)
    df_va = train_df_filtered.iloc[idx_va].reset_index(drop=True)


    pos_count = int((df_tr[LABEL_COL] == 1).sum())
    neg_count = int((df_tr[LABEL_COL] == 0).sum())
    print(f"[INFO] Fold {fold} balance: pos={pos_count} neg={neg_count} ({pos_count / (pos_count + neg_count):.2%} pos)")


    sampler_tr = create_weighted_sampler(df_tr, LABEL_COL)


    tr_loader = DataLoader(
        OnTheFlyDataset(df_tr, IMAGE_DIR, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=sampler_tr,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    va_loader = DataLoader(
        OnTheFlyDataset(df_va, IMAGE_DIR, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
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
        opt, mode="max", factor=0.3, patience=3, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()


    best_f1, bad, logs = -1, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        trmtr, _, _ = epoch_pass(model, tr_loader, optimizer=opt, loss_fn=loss_obj, scaler=scaler)
        vamtr, _, _ = epoch_pass(model, va_loader, optimizer=None, loss_fn=loss_obj)
        logs.append(
            {
                "epoch": epoch,
                **{f"tr_{k}": v for k, v in trmtr.items() if k not in ["y", "p"]},
                **{f"va_{k}": v for k, v in vamtr.items() if k not in ["y", "p"]},
            }
        )



        print(
            f"Epoch {epoch:03d} | tr_loss={trmtr['loss']:.4f} "
            f"| va_loss={vamtr['loss']:.4f} | va_acc={vamtr['accuracy']:.3f} "
            f"| va_bal_acc={vamtr['balanced_accuracy']:.3f} | sens={vamtr['sensitivity']:.3f} "
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
    fold_summ.append(
        {
            "fold": fold,
            "val_default": summarize(y_val, p_val, 0.5),
            "val_opt": summarize(y_val, p_val, t_opt),
            "val_cal": summarize(y_val, cal_p_val, t_opt),
            "t_opt": float(t_opt),
            "T": float(T),
        }
    )
    print(
    f"[Fold {fold}] t_opt={t_opt:.3f} | Temp={T:.3f} | Calibrated metrics: "
    f"accuracy={fold_summ[-1]['val_cal']['accuracy']:.3f} "
    f"balanced_accuracy={fold_summ[-1]['val_cal']['balanced_accuracy']:.3f} "
    f"sensitivity={fold_summ[-1]['val_cal']['sensitivity']:.3f} "
    f"specificity={fold_summ[-1]['val_cal']['specificity']:.3f} "
    f"f1_macro={fold_summ[-1]['val_cal']['f1_macro']:.3f} "
    f"auc={fold_summ[-1]['val_cal']['auc']:.3f} "
    f"mcc={fold_summ[-1]['val_cal']['mcc']:.3f}"
    )



with open(Path(OUTPUT_ROOT) / "cv_summary_densenet121.json", "w") as f:
    json.dump(convert_to_builtin_types(fold_summ), f, indent=2)
print("Saved CV summary:", Path(OUTPUT_ROOT) / "cv_summary_densenet121.json")



# === Test evaluation with best fold ===
best_idx = int(np.argmax([fs["val_cal"]["f1_macro"] for fs in fold_summ]))
best_fold = fold_summ[best_idx]["fold"]
print(f"\n>>> Best fold by calibrated macro-F1: {best_fold}")


best_dir = Path(OUTPUT_ROOT) / f"fold_{best_fold}"
state = torch.load(best_dir / "best.pt", map_location=device)
t_opt = float(fold_summ[best_idx]["t_opt"])
T = float(fold_summ[best_idx]["T"])


tem = test_df_filtered.dropna(subset=[LABEL_COL]).reset_index(drop=True)
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


print("\n=== TEST RESULTS ===")
def pretty_print(tag, d):
    print(
        f"{tag}: acc={d.get('accuracy', d.get('acc', float('nan'))):.3f} "
        f"| bal_acc={d.get('balanced_accuracy', d.get('bal_acc', float('nan'))):.3f} "
        f"| sens={d['sensitivity']:.3f} | spec={d['specificity']:.3f} "
        f"| f1_macro={d['f1_macro']:.3f} | auc={d['auc']:.3f} "
        f"| prAUC={d.get('pr_auc', d.get('pr', float('nan'))):.3f} "
        f"| mcc={d['mcc']:.3f} | brier={d.get('brier', float('nan')):.3f} "
        f"| TN={d['TN']} FP={d['FP']} FN={d['FN']} TP={d['TP']}"
    )



res_default = summarize(y_te, p_te, 0.5)
pretty_print("Default@0.5     ", res_default)
res_topt = summarize(y_te, p_te, t_opt)
pretty_print("Uncalib@t_opt   ", res_topt)
res_cal = summarize(y_te, p_te_cal, t_opt)
pretty_print("Calibrated@t_opt", res_cal)


df_pred = pd.DataFrame(
    {"TCIA_ID": tem["TCIA_ID"], "Filename": tem["Filename"], "y": y_te, "prob": p_te, "prob_cal": p_te_cal}
)
df_pred.to_csv(Path(OUTPUT_ROOT) / "test_predictions_densenet121.csv", index=False)


with open(Path(OUTPUT_ROOT) / "test_metrics_densenet121.json", "w") as f:
    json.dump(
        {
            "default": res_default,
            "topt": res_topt,
            "calibrated": res_cal,
            "t_opt": t_opt,
            "T": T,
        },
        f,
        indent=2,
    )


print("\nSaved test predictions and metrics in:", OUTPUT_ROOT)
