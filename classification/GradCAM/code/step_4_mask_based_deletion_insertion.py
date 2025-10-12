# ============================================================
# Mask-based Deletion / Insertion Curves for DL Classifier
# ============================================================
import os, numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import SimpleITK as sitk
from tqdm import tqdm
from monai.networks.nets import resnet18
from sklearn.metrics import auc

# ---------------------------
# CONFIGURATION
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "path/to/resnet18_T4_dir/fold_2/best_model.pth"
MODEL_NAME = "resnet18"
CLASS_INDEX = 1  # T4 class
INPUT_SHAPE = (32, 96, 96)
SAVE_DIR = "./mask_deletion_insertion_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# UTILITIES
# ---------------------------
def load_nrrd(path):
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    return vol

def resize3d(vol, out_shape):
    factors = [out_shape[0]/vol.shape[0],
               out_shape[1]/vol.shape[1],
               out_shape[2]/vol.shape[2]]
    return zoom(vol, factors, order=1)

def to_tensor(vol):
    return torch.from_numpy(vol[None, None]).float().to(DEVICE)

def zscore_norm(vol):
    return (vol - vol.mean()) / (vol.std() + 1e-8)

def model_infer(model, x):
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    return probs[CLASS_INDEX]

# ---------------------------
# MODEL LOADING
# ---------------------------
model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint, strict=False)
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# ---------------------------
# DELETION / INSERTION LOGIC
# ---------------------------
def compute_mask_curves(model, vol, mask, n_steps=10, replace_mode="mean"):
    """Run deletion and insertion using a binary mask region."""
    vol = zscore_norm(vol)
    baseline = model_infer(model, to_tensor(vol))
    mean_val = vol.mean()

    # --- flatten mask and sort region voxels ---
    mask_idx = np.where(mask > 0)
    n_vox = len(mask_idx[0])
    step = max(1, n_vox // n_steps)

    deletion_probs, insertion_probs = [], []

    for i in range(0, n_vox, step):
        frac = (i + step) / n_vox

        # ----- Deletion -----
        vol_del = vol.copy()
        # progressively zero out (replace with mean) the first i voxels in mask
        if replace_mode == "mean":
            vol_del[mask_idx[0][:i], mask_idx[1][:i], mask_idx[2][:i]] = mean_val
        prob_del = model_infer(model, to_tensor(vol_del))
        deletion_probs.append(prob_del)

        # ----- Insertion -----
        vol_ins = np.ones_like(vol) * mean_val
        vol_ins[mask_idx[0][:i], mask_idx[1][:i], mask_idx[2][:i]] = vol[mask_idx[0][:i], mask_idx[1][:i], mask_idx[2][:i]]
        prob_ins = model_infer(model, to_tensor(vol_ins))
        insertion_probs.append(prob_ins)

    return baseline, deletion_probs, insertion_probs

# ---------------------------
# EXPERIMENT DRIVER
# ---------------------------
cases = [
    ("Case001_T4", "path/to/HN_CHUS_034_Cropped_Volume.nrrd",
     "path/to/HN_CHUS_034.seg.nrrd"),
    ("Case002_T4", "path/to/QIN_HEADNECK_02_2502_Cropped_Volume.nrrd",
     "path/to/QIN_HEADNECK_02_2502.seg.nrrd"),
    ("Case003_T4", "path/to/RADCURE_1726_Cropped_Volume.nrrd",
     "path/to/RADCURE_1726.seg.nrrd"),
    ("Case004_non_T4", "path/to/RADCURE_2362_Cropped_Volume.nrrd",
     "path/to/RADCURE_2362.seg.nrrd"),
    ("Case005_non_T4", "path/to/QIN_HEADNECK_01_0048_Cropped_Volume.nrrd",
     "path/to/QIN_HEADNECK_01_0048.seg.nrrd"),
    ("Case006_non_T4", "path/to/RADCURE_0039_Cropped_Volume.nrrd",
     "path/to/RADCURE_0039.seg.nrrd"),
]

results = []
for cid, img_p, mask_p in tqdm(cases):
    vol = load_nrrd(img_p)
    mask = load_nrrd(mask_p)
    if vol.shape != INPUT_SHAPE:
        vol = resize3d(vol, INPUT_SHAPE)
        mask = resize3d(mask, INPUT_SHAPE)
    mask = (mask > 0).astype(np.uint8)

    baseline, del_probs, ins_probs = compute_mask_curves(model, vol, mask)

    x = np.linspace(0, 100, len(del_probs))
    delta_del = baseline - np.array(del_probs)
    delta_ins = np.array(ins_probs) - ins_probs[0]

    auc_del = auc(x/100, delta_del)
    auc_ins = auc(x/100, delta_ins)
    results.append((cid, auc_del, auc_ins))

    # ----- Plot per-case -----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(x, delta_del, 'r-o', label=f"{cid} (Δ)")
    axs[0].set_title("Deletion curve (Δprob)")
    axs[0].set_xlabel("Fraction of mask deleted (%)")
    axs[0].set_ylabel("ΔProb (baseline - current)")
    axs[1].plot(x, delta_ins, 'b-o', label=f"{cid} (Δ)")
    axs[1].set_title("Insertion curve (Δprob)")
    axs[1].set_xlabel("Fraction of mask inserted (%)")
    axs[1].set_ylabel("ΔProb (current - base)")
    for ax in axs: ax.legend(); ax.grid(True)
    plt.suptitle(cid, fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{cid}_Mask_DelIns.png"), dpi=300)
    plt.close()

# ---------------------------
# SAVE AGGREGATE RESULTS
# ---------------------------
import pandas as pd
df = pd.DataFrame(results, columns=["CaseID", "AUC_Deletion", "AUC_Insertion"])
df.to_csv(os.path.join(SAVE_DIR, "Mask_DelIns_AUCs.csv"), index=False)
print(df)
print("✅ Mask-based deletion/insertion curves completed.")
