# ======================================================
# SwinUNETR test evaluation script
# ======================================================

import os, glob
import numpy as np
from tqdm import tqdm
import torch
import SimpleITK as sitk

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ScaleIntensityRanged, NormalizeIntensityd
)
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import SwinUNETR

# ==============================
# 1) PATHS
# ==============================
RAW_DATASET = r"C:\Users\your_folder_name\LarynxCTSeg\nnUNet_raw\Dataset001_LarynxCT"

SEEDB_DIR  = os.path.join(RAW_DATASET, "imagesTs")
HANSEG_DIR = os.path.join(RAW_DATASET, "HanSeg")

CKPT_ROOT = r"C:\Users\your_folder_name\LarynxCTSeg\swinunetr_baseline"
CKPT_PATHS = [os.path.join(CKPT_ROOT, f"fold{i}", f"swinunetr_fold{i}_best.pth") for i in range(5)]

OUT_ROOT = r"C:\Users\your_folder_name\LarynxCTSeg\swinunetr_ensemble_preds_fullres"
OUT_SEEDB = os.path.join(OUT_ROOT, "SeedB_labels")
OUT_HANSEG = os.path.join(OUT_ROOT, "HanSeg_labels")
OUT_SEEDB_PROB = os.path.join(OUT_ROOT, "SeedB_prob_cartilage")
OUT_HANSEG_PROB = os.path.join(OUT_ROOT, "HanSeg_prob_cartilage")

for d in [OUT_SEEDB, OUT_HANSEG, OUT_SEEDB_PROB, OUT_HANSEG_PROB]:
    os.makedirs(d, exist_ok=True)

# ==============================
# 2) CONFIG (MATCH TRAINING)
# ==============================
HU_MIN, HU_MAX = -300, 300
PATCH_SIZE = (96, 96, 96)  # sliding window ROI size
IN_CHANNELS = 1
OUT_CHANNELS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Allow running with whatever folds exist right now (e.g., after fold0 only)
existing_ckpts = [p for p in CKPT_PATHS if os.path.exists(p)]
missing_ckpts = [p for p in CKPT_PATHS if not os.path.exists(p)]
print("Found checkpoints:", len(existing_ckpts))
if missing_ckpts:
    print("Missing checkpoints (OK for now):")
    for p in missing_ckpts:
        print(" -", p)

if len(existing_ckpts) == 0:
    raise FileNotFoundError("No SwinUNETR checkpoints found yet. Train at least one fold first.")

# ==============================
# 3) PREPROCESS (NO pad, NO resample, NO reorient)
# Keeps shapes identical to disk -> saving + metrics match GT
# ==============================
preproc = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image"], dtype=torch.float32),
])

# ==============================
# 4) BUILD + LOAD MODELS
# ==============================
def build_swinunetr():
    # Must match your training config
    model = SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        patch_size=2,
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        norm_name="instance",
        use_checkpoint=True,   # OK for inference too (slower but fine)
        spatial_dims=3,
        use_v2=False
    ).to(DEVICE)
    return model

def load_model(ckpt_path: str):
    model = build_swinunetr()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

models = [load_model(p) for p in existing_ckpts]
print("Loaded models:", len(models))

# IMPORTANT: SlidingWindowInferer returns output cropped back to original input size.
inferer = SlidingWindowInferer(
    roi_size=PATCH_SIZE,
    sw_batch_size=1,
    overlap=0.5,
    mode="gaussian"
)

# ==============================
# 5) UTILS
# ==============================
def collect_niftis(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.nii")) + glob.glob(os.path.join(folder, "*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in: {folder}")
    return files

def case_id_from_path(p):
    base = os.path.basename(p)
    if base.endswith(".nii.gz"): base = base[:-7]
    elif base.endswith(".nii"): base = base[:-4]
    if base.endswith("_0000"): base = base[:-5]
    return base

@torch.no_grad()
def ensemble_prob_mean(x):
    prob_sum = None
    for m in models:
        logits = inferer(x, m)                 # (1,C,Z,Y,X) SAME SIZE AS x
        probs = torch.softmax(logits, dim=1)
        prob_sum = probs if prob_sum is None else (prob_sum + probs)
    return prob_sum / len(models)

def save_like_ref(ref_path, arr, out_path, is_prob=False):
    """
    Save prediction/prob map with reference image geometry.
    `arr` can be (Z,Y,X) (preferred) or (X,Y,Z). This will correct safely.
    """
    ref = sitk.ReadImage(ref_path)
    sx, sy, sz = ref.GetSize()  # (x,y,z)

    a = np.asarray(arr)

    if a.shape == (sz, sy, sx):        # (z,y,x) OK
        a_zyx = a
    elif a.shape == (sx, sy, sz):      # (x,y,z) -> (z,y,x)
        a_zyx = np.transpose(a, (2, 1, 0))
    else:
        raise RuntimeError(
            f"Prediction shape {a.shape} does not match ref size "
            f"(x,y,z)=({sx},{sy},{sz}) or (z,y,x)=({sz},{sy},{sx}). Ref: {ref_path}"
        )

    out = sitk.GetImageFromArray(a_zyx.astype(np.float32 if is_prob else np.uint8))
    out.CopyInformation(ref)
    sitk.WriteImage(out, out_path)

# ==============================
# 6) MAIN PREDICTOR
# ==============================
def run_folder_fullres(input_dir, out_lbl_dir, out_prob_dir, save_probmap=True):
    files = collect_niftis(input_dir)
    ds = Dataset(data=[{"image": f} for f in files], transform=preproc)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for batch in tqdm(loader, desc=f"SwinUNETR ensemble predicting: {os.path.basename(input_dir)}"):
        img_mt = batch["image"]

        fn = img_mt.meta.get("filename_or_obj", None)
        if isinstance(fn, (list, tuple)): fn = fn[0]
        fn = str(fn)

        cid = case_id_from_path(fn)

        x = img_mt.to(DEVICE)
        prob_mean = ensemble_prob_mean(x).detach().cpu().numpy()[0]  # (C,Z,Y,X)

        pred = np.argmax(prob_mean, axis=0).astype(np.uint8)         # (Z,Y,X)
        prob1 = prob_mean[1].astype(np.float32)                      # (Z,Y,X)

        out_lbl_path = os.path.join(out_lbl_dir, f"{cid}.nii.gz")
        save_like_ref(fn, pred, out_lbl_path, is_prob=False)

        if save_probmap:
            out_prob_path = os.path.join(out_prob_dir, f"{cid}_prob1.nii.gz")
            save_like_ref(fn, prob1, out_prob_path, is_prob=True)

    print("\nDone:", input_dir)
    print("Labels ->", out_lbl_dir)
    if save_probmap:
        print("Prob ->", out_prob_dir)

run_folder_fullres(SEEDB_DIR, OUT_SEEDB, OUT_SEEDB_PROB, save_probmap=True)
run_folder_fullres(HANSEG_DIR, OUT_HANSEG, OUT_HANSEG_PROB, save_probmap=True)

print("\n SwinUNETR ensemble inference done.")
print("Using checkpoints:")
for p in existing_ckpts:
    print(" -", p)