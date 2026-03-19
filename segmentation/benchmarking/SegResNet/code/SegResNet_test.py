# =====================================
# Test evaluation script for SegResNet
# =====================================

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
from monai.networks.nets import SegResNet

# ==============================
# 1) PATHS
# ==============================
SEEDB_DIR  = r"C:\Users\your_folder_name\LarynxCTSeg\nnUNet_raw\Dataset001_LarynxCT\imagesTs"
HANSEG_DIR = r"C:\Users\your_folder_name\LarynxCTSeg\nnUNet_raw\Dataset001_LarynxCT\HanSeg"

CKPT_DIR = r"C:\Users\your_folder_name\LarynxCTSeg\segresnet_baseline"
CKPT_PATHS = [os.path.join(CKPT_DIR, f"segresnet_fold{i}_best.pth") for i in range(5)]

OUT_ROOT = r"C:\Users\your_folder_name\LarynxCTSeg\segresnet_ensemble_preds_fullres"
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
PATCH_SIZE = (96, 96, 96)
INIT_FILTERS = 32
IN_CHANNELS = 1
OUT_CHANNELS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

missing = [p for p in CKPT_PATHS if not os.path.exists(p)]
if missing:
    raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
print("All checkpoints found.")

# ==============================
# 3) PREPROCESS (NO spacing/orientation/pad)
# Keeps shapes identical to disk -> metrics will match GT
# ==============================
preproc = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image"]),
])

# ==============================
# 4) LOAD MODELS
# ==============================
def load_model(ckpt_path: str):
    model = SegResNet(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        init_filters=INIT_FILTERS,
        dropout_prob=0.0
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

models = [load_model(p) for p in CKPT_PATHS]
print("Loaded models:", len(models))

inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.5, mode="gaussian")

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
        logits = inferer(x, m)                 # (1,C,Z,Y,X) same size as x
        probs = torch.softmax(logits, dim=1)
        prob_sum = probs if prob_sum is None else (prob_sum + probs)
    return prob_sum / len(models)

def save_like_ref(ref_path, arr, out_path, is_prob=False):
    """
    Save prediction/prob map with reference image geometry.
    Works whether `arr` is (Z,Y,X) or (X,Y,Z), and fixes it safely.
    """
    ref = sitk.ReadImage(ref_path)
    sx, sy, sz = ref.GetSize()  # (x,y,z)

    a = np.asarray(arr)

    # Case A: arr is (z,y,x) -> OK for GetImageFromArray
    if a.shape == (sz, sy, sx):
        a_zyx = a

    # Case B: arr is (x,y,z) -> transpose to (z,y,x)
    elif a.shape == (sx, sy, sz):
        a_zyx = np.transpose(a, (2, 1, 0))

    else:
        raise RuntimeError(
            f"Prediction shape {a.shape} does not match ref size "
            f"(x,y,z)=({sx},{sy},{sz}) or (z,y,x)=({sz},{sy},{sx}). "
            f"Ref: {ref_path}"
        )

    out = sitk.GetImageFromArray(a_zyx.astype(np.float32 if is_prob else np.uint8))  # expects (z,y,x)
    out.CopyInformation(ref)
    sitk.WriteImage(out, out_path)


# ==============================
# 6) MAIN
# ==============================
def run_folder_fullres(input_dir, out_lbl_dir, out_prob_dir, save_probmap=True):
    files = collect_niftis(input_dir)
    ds = Dataset(data=[{"image": f} for f in files], transform=preproc)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for batch in tqdm(loader, desc=f"Ensemble predicting (fullres): {os.path.basename(input_dir)}"):
        img_mt = batch["image"]

        fn = img_mt.meta.get("filename_or_obj", None)
        if isinstance(fn, (list, tuple)):
            fn = fn[0]
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