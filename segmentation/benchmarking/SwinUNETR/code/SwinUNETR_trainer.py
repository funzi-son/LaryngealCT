# ============================================================
# SwinUNETR baseline (MONAI) - nnUNet-split-matched training
# Uses nnUNet v2 splits_final.json for fair comparison
# 
# Orientationd(RAS) + Spacingd(1mm) + HU clip + ZNorm
# SpatialPadd + RandCropByPosNegLabeld
# RandFlip + RandRotate90
# Early stopping + best checkpoint saving
# ============================================================

import os, json, time, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import monai
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd,
    ScaleIntensityRanged, NormalizeIntensityd,
    SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d,
    AsDiscrete
)
from monai.data import CacheDataset, list_data_collate
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.data.utils import decollate_batch
from monai.networks.nets import SwinUNETR

print("MONAI:", monai.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ===========================
# CONFIG (edit if needed)
# ===========================
DATASET_ID = "001"

NNUNET_RAW = r"C:\Users\your_folder_name\LarynxCTSeg\nnUNet_raw"
NNUNET_PREPROCESSED = r"C:\Users\your_folder_name\LarynxCTSeg\nnUNet_preprocessed"

OUT_DIR = r"C:\Users\your_folder_name\LarynxCTSeg\swinunetr_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

# Fold control:
# - set FOLD=None to train all 5 folds
# - set FOLD=0..4 to train one fold
FOLD = None  # e.g., 0 or 4

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preproc params (match SegResNet/DynUNet)
HU_MIN, HU_MAX = -300, 300
SPACING = (1.0, 1.0, 1.0)
AXCODES = "RAS"

# Patch training
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 1
NUM_SAMPLES_PER_CASE = 2

# Training
MAX_EPOCHS = 500
VAL_EVERY = 1
LR = 2e-4
WEIGHT_DECAY = 1e-5

# Early stopping
PATIENCE = 50
MIN_DELTA = 1e-4

# System
NUM_WORKERS = 0        
CACHE_RATE = 0.2       # 0.0 if RAM issues
AMP = True

# SwinUNETR config 
SWIN_FEATURE_SIZE = 24
WINDOW_SIZE = 7
USE_CHECKPOINT = True

# ===========================
# HELPERS
# ===========================
def find_dataset_folder(base_dir, dataset_id):
    for d in os.listdir(base_dir):
        if d.startswith(f"Dataset{dataset_id}_"):
            return os.path.join(base_dir, d)
    return None

def get_nnunet_split_items(dataset_id="001", fold=0):
    """
    Uses nnUNet v2 split file to build train/val lists.
    Reads:
      nnUNet_preprocessed/DatasetXXX_*/splits_final.json
    Maps to:
      nnUNet_raw/DatasetXXX_*/imagesTr + labelsTr
    """
    pref_folder = find_dataset_folder(NNUNET_PREPROCESSED, dataset_id)
    if pref_folder is None:
        raise FileNotFoundError(f"Could not find Dataset{dataset_id}_* in {NNUNET_PREPROCESSED}")

    splits_path = os.path.join(pref_folder, "splits_final.json")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Missing splits_final.json: {splits_path}")

    with open(splits_path, "r") as f:
        splits = json.load(f)

    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range. splits has {len(splits)} folds")

    train_ids = splits[fold]["train"]
    val_ids   = splits[fold]["val"]

    raw_folder = find_dataset_folder(NNUNET_RAW, dataset_id)
    if raw_folder is None:
        raise FileNotFoundError(f"Could not find Dataset{dataset_id}_* in {NNUNET_RAW}")

    imagesTr = os.path.join(raw_folder, "imagesTr")
    labelsTr = os.path.join(raw_folder, "labelsTr")

    def build_items(case_ids):
        items = []
        for cid in case_ids:
            img = os.path.join(imagesTr, f"{cid}_0000.nii.gz")
            lab = os.path.join(labelsTr, f"{cid}.nii.gz")
            if not os.path.exists(img):
                raise FileNotFoundError(f"Missing image: {img}")
            if not os.path.exists(lab):
                raise FileNotFoundError(f"Missing label: {lab}")
            items.append({"image": img, "label": lab, "case_id": cid})
        return items

    return build_items(train_ids), build_items(val_ids), raw_folder, pref_folder

def build_model():
    model = SwinUNETR(
        in_channels=1,
        out_channels=2,
        patch_size=2,
        feature_size=SWIN_FEATURE_SIZE,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=WINDOW_SIZE,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        norm_name="instance",
        use_checkpoint=USE_CHECKPOINT,
        spatial_dims=3,
        use_v2=False
    ).to(DEVICE)
    return model

# ===========================
# DETERMINISM
# ===========================
set_determinism(SEED)

# ===========================
# TRANSFORMS (MATCH SegResNet)
# ===========================
train_tfms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes=AXCODES),
    Spacingd(keys=["image", "label"], pixdim=SPACING, mode=("bilinear", "nearest")),

    ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

    SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE, method="end"),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,
        pos=1, neg=1,
        num_samples=NUM_SAMPLES_PER_CASE,
        image_key="image",
        image_threshold=0,
        allow_smaller=True
    ),

    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),

    EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.long)),
])

val_tfms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes=AXCODES),
    Spacingd(keys=["image", "label"], pixdim=SPACING, mode=("bilinear", "nearest")),

    ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

    SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE, method="end"),

    EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.long)),
])

# ===========================
# TRAIN ONE FOLD
# ===========================
def train_one_fold(fold_idx: int):
    fold_dir = os.path.join(OUT_DIR, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    log_csv  = os.path.join(fold_dir, f"swinunetr_fold{fold_idx}_log.csv")
    best_ckpt = os.path.join(fold_dir, f"swinunetr_fold{fold_idx}_best.pth")

    train_items, val_items, raw_folder, pref_folder = get_nnunet_split_items(DATASET_ID, fold_idx)
    print("\n==============================")
    print(f"Fold {fold_idx}")
    print("Train:", len(train_items), "Val:", len(val_items))
    print("Raw:", raw_folder)
    print("Preprocessed:", pref_folder)
    print("Log:", log_csv)
    print("Best:", best_ckpt)

    train_ds = CacheDataset(train_items, transform=train_tfms, cache_rate=CACHE_RATE, num_workers=NUM_WORKERS)
    val_ds   = CacheDataset(val_items,   transform=val_tfms,   cache_rate=1.0,      num_workers=NUM_WORKERS)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=list_data_collate,
    )

    model = build_model()
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=(AMP and DEVICE.type == "cuda"))

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred  = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.5, mode="gaussian")

    best_dice = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # -------- train --------
        model.train()
        train_loss = 0.0
        n_steps = 0

        for batch in tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch}/{MAX_EPOCHS} [train]"):
            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(AMP and DEVICE.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += float(loss.item())
            n_steps += 1

        train_loss /= max(n_steps, 1)

        # -------- val --------
        val_dice = None
        val_loss = None

        if epoch % VAL_EVERY == 0:
            model.eval()
            dice_metric.reset()

            vloss_sum = 0.0
            vn = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Fold {fold_idx} Epoch {epoch}/{MAX_EPOCHS} [val]"):
                    x = batch["image"].to(DEVICE)
                    y = batch["label"].to(DEVICE)

                    with torch.cuda.amp.autocast(enabled=(AMP and DEVICE.type == "cuda")):
                        logits = inferer(x, model)
                        vloss = loss_fn(logits, y)

                    vloss_sum += float(vloss.item())
                    vn += 1

                    preds_list  = [post_pred(p) for p in decollate_batch(logits)]
                    labels_list = [post_label(l) for l in decollate_batch(y)]
                    dice_metric(y_pred=preds_list, y=labels_list)

            val_loss = vloss_sum / max(vn, 1)
            val_dice = float(dice_metric.aggregate().item())

            # early stopping
            if val_dice > best_dice + MIN_DELTA:
                best_dice = val_dice
                best_epoch = epoch
                epochs_no_improve = 0

                torch.save({
                    "model": model.state_dict(),
                    "best_dice": best_dice,
                    "epoch": epoch,
                    "config": {
                        "PATCH_SIZE": PATCH_SIZE,
                        "HU_MIN": HU_MIN, "HU_MAX": HU_MAX,
                        "SPACING": SPACING,
                        "AXCODES": AXCODES,
                        "LR": LR,
                        "WEIGHT_DECAY": WEIGHT_DECAY,
                        "SWIN_FEATURE_SIZE": SWIN_FEATURE_SIZE,
                        "WINDOW_SIZE": WINDOW_SIZE
                    }
                }, best_ckpt)

                print(f" New best Dice={best_dice:.4f} @ epoch {epoch} -> saved")
            else:
                epochs_no_improve += 1

        dt = time.time() - t0

        row = {
            "fold": fold_idx,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "best_dice_so_far": best_dice,
            "best_epoch_so_far": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "epoch_time_sec": dt
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_csv, index=False)

        print(f"[Fold {fold_idx}][Epoch {epoch}] "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss if val_loss is not None else 'NA'} | "
              f"val_dice={val_dice if val_dice is not None else 'NA'} | "
              f"best={best_dice:.4f} (epoch {best_epoch}) | "
              f"no_improve={epochs_no_improve}/{PATIENCE} | "
              f"time={dt:.1f}s")

        if val_dice is not None and epochs_no_improve >= PATIENCE:
            print(f"\n Early stopping at epoch {epoch}. Best epoch={best_epoch}, Best Dice={best_dice:.4f}")
            break

    print(f"\n Fold {fold_idx} finished. Best Dice={best_dice:.4f} at epoch {best_epoch}")
    print("Best checkpoint:", best_ckpt)
    print("Log CSV:", log_csv)

    return best_ckpt, log_csv

# ===========================
# RUN
# ===========================
folds_to_run = [FOLD] if FOLD is not None else [0, 1, 2, 3, 4]
print("Folds to run:", folds_to_run)

ckpts = []
logs = []
for f in folds_to_run:
    c, l = train_one_fold(f)
    ckpts.append((f, c))
    logs.append(l)

print("\nDONE.")
print("Checkpoints:", ckpts)