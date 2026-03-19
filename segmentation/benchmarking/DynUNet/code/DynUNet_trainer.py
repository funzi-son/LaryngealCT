# ============================================================
# DynUNet baseline (MONAI) - nnUNet-split-matched training
# - Runs inside Jupyter Notebook (Windows-friendly)
# - Uses nnUNet v2 splits_final.json for fair comparison
# - HU clip + normalize, patch-based training, SW inference
# - Early Stopping + Best checkpoint saving
# ============================================================

import os, json, time, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd, ScaleIntensityRanged, NormalizeIntensityd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    SpatialPadd, AsDiscrete
)
from monai.data import CacheDataset, list_data_collate
from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.data.utils import decollate_batch

# ===========================
# CONFIG (edit if needed)
# ===========================
DATASET_ID = "001"

NNUNET_RAW = r"C:\Users\your_folder_name\LarynxCTSeg\nnUNet_raw"
NNUNET_PREPROCESSED = r"C:\Users\your_folder_name\LarynxCTSeg\nnUNet_preprocessed"
OUT_DIR = r"C:\Users\your_folder_name\LarynxCTSeg\dynunet_baseline"

FOLD = 0    			# change to 1,2,3,4 for each fold
SEED = 42

HU_MIN, HU_MAX = -300, 300
SPACING = (1.0, 1.0, 1.0)

PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 1
NUM_SAMPLES_PER_CASE = 2

MAX_EPOCHS = 500
VAL_EVERY = 1
LR = 1e-4

NUM_WORKERS = 2         # if hangs -> 0
CACHE_RATE = 0.2
AMP = True

PATIENCE = 50
MIN_DELTA = 1e-4

IN_CHANNELS = 1
OUT_CHANNELS = 2

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# HELPERS
# ===========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_dataset_folder(base_dir, dataset_id):
    for d in os.listdir(base_dir):
        if d.startswith(f"Dataset{dataset_id}_"):
            return os.path.join(base_dir, d)
    return None

def get_nnunet_split_items(dataset_id="001", fold=0):
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
    val_ids = splits[fold]["val"]

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

# ===========================
# DynUNet builder
# ===========================
def build_dynunet():
    """
    A solid default DynUNet config for 96^3 patches on 1mm CT.
    """
    # 5-level encoder/decoder (works well for 96^3)
    # Strides reduce spatial dims: /2 /2 /2 /2 (4 downsamples)
    strides = (1, 2, 2, 2, 2)
    kernel_size = (3, 3, 3, 3, 3)

    model = DynUNet(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=strides[1:],   # (2,2,2,2)
        deep_supervision=False,
        res_block=True,
    )
    return model

# ===========================
# MAIN
# ===========================
seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Device:", device)

train_items, val_items, raw_folder, pref_folder = get_nnunet_split_items(DATASET_ID, FOLD)
print("Raw folder:", raw_folder)
print("Preprocessed folder:", pref_folder)
print(f"Train cases: {len(train_items)} | Val cases: {len(val_items)}")

# -------------------------
# Transforms (match SegResNet training)
# -------------------------
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
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
        allow_smaller=True,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=SPACING, mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

    SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE, method="end"),
    EnsureTyped(keys=["image", "label"]),
])

# -------------------------
# Data
# -------------------------
train_ds = CacheDataset(train_items, transform=train_transforms, cache_rate=CACHE_RATE, num_workers=NUM_WORKERS)
val_ds   = CacheDataset(val_items, transform=val_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
    collate_fn=list_data_collate,
)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
    collate_fn=list_data_collate,
)

batch = next(iter(train_loader))
print("Train batch image/label:", batch["image"].shape, batch["label"].shape)
print("Train batches:", len(train_loader), "Val cases:", len(val_loader))

# -------------------------
# Model / Loss / Optim
# -------------------------
model = build_dynunet().to(device)

loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler(enabled=(AMP and device.type == "cuda"))

dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = AsDiscrete(argmax=True, to_onehot=OUT_CHANNELS)
post_label = AsDiscrete(to_onehot=OUT_CHANNELS)

inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.5)

log_csv = os.path.join(OUT_DIR, f"dynunet_fold{FOLD}_log.csv")
best_ckpt = os.path.join(OUT_DIR, f"dynunet_fold{FOLD}_best.pth")
print("CSV log:", log_csv)
print("Best ckpt:", best_ckpt)

# -------------------------
# Training loop (with Early stopping)
# -------------------------
history = []
best_dice = -1.0
best_epoch = -1
epochs_no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_loss = 0.0
    n_steps = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [train]"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(AMP and device.type == "cuda")):
            logits = model(images)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += float(loss.item())
        n_steps += 1

    train_loss /= max(n_steps, 1)

    # ----- validation -----
    val_dice = None
    if epoch % VAL_EVERY == 0:
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [val]"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                with torch.cuda.amp.autocast(enabled=(AMP and device.type == "cuda")):
                    logits = inferer(images, model)

                preds_list = [post_pred(p) for p in decollate_batch(logits)]
                labels_list = [post_label(l) for l in decollate_batch(labels)]

                dice_metric(y_pred=preds_list, y=labels_list)

        val_dice = float(dice_metric.aggregate().item())

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
                    "LR": LR,
                }
            }, best_ckpt)
        else:
            epochs_no_improve += 1

    dt = time.time() - t0
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_dice": val_dice,
        "best_dice_so_far": best_dice,
        "best_epoch_so_far": best_epoch,
        "epochs_no_improve": epochs_no_improve,
        "epoch_time_sec": dt
    })
    pd.DataFrame(history).to_csv(log_csv, index=False)

    print(f"[Epoch {epoch}] train_loss={train_loss:.4f} "
          f"val_dice={val_dice if val_dice is not None else 'NA'} "
          f"best={best_dice:.4f} (epoch {best_epoch}) "
          f"no_improve={epochs_no_improve}/{PATIENCE} "
          f"time={dt:.1f}s")

    if val_dice is not None and epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}, Best Dice: {best_dice:.4f}")
        break

print("\nTraining complete.")
print("Best Dice:", best_dice)
print("Best checkpoint:", best_ckpt)
print("Log CSV:", log_csv)

print("\nIf OOM, try:")
print("  PATCH_SIZE=(80,80,80), NUM_SAMPLES_PER_CASE=1")