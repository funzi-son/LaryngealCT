# ============================================================
# UNIFIED EXPLAINABILITY PIPELINE
# For BOTH:
#   1) ResNet101 ensemble
#   2) Custom 5-layer 3D CNN ensemble
#Features:
#   1. HU clipping is used consistently in perturbation analysis
#   2. Explainability is computed for the PREDICTED CLASS by default
#   3. TP/TN/FP/FN status is stored
#   4. Curves are saved grouped by:
#        - true class (T4 / non-T4)
#        - outcome type (TP / TN / FP / FN)
#   5. Same 6 cases, same preprocessing, same input shape, same ensemble logic
# ============================================================

!pip install monai SimpleITK scipy scikit-image pandas matplotlib tqdm pillow --quiet

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import SimpleITK as sitk
from scipy.ndimage import zoom, gaussian_filter
from skimage import measure
from tqdm import tqdm

from monai.networks.nets import resnet101

# ============================================================
# GLOBAL CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SHAPE = (32, 96, 96)
SEED = 42

# IMPORTANT:
# explainability target mode:
#   "predicted" -> CAM/perturbation for predicted class (recommended)
#   "positive"  -> always class 1 (T4)
EXPLAIN_CLASS_MODE = "predicted"

# ---------------------------
# Preprocessing
# ---------------------------
HU_CLIP_LOW = -300
HU_CLIP_HIGH = 300

# ---------------------------
# Explainability settings
# ---------------------------
STEPS = 11  # 0..100%
CAM_THRESHOLD_PERCENTILE = 30
CAM_CLIP_LOW = 1
CAM_CLIP_HIGH = 99
CAM_GAMMA = 0.95
SMOOTH_SIGMA = 0.75
UNSHARP_SIGMA = 0.9
UNSHARP_AMOUNT = 0.9
STRETCH_LOW = 8
STRETCH_HIGH = 97

# ---------------------------
# Visualization settings
# ---------------------------
CMAP_NAME = "turbo"
OVERLAY_ALPHA_MIN = 0.06
OVERLAY_ALPHA_MAX = 0.62
SAVE_DPI = 400

# ---------------------------
# Reproducibility
# ---------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

# ============================================================
# PATHS
# ============================================================

BASE_SAVE_DIR = r"path\to\Ensemble\Explainability_Comparison"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

RESNET101_CKPTS = [
    r"path\to\Ensemble\T4_Classification\ResNet101\fold_1\best.pt",
    r"path\to\Ensemble\T4_Classification\ResNet101\fold_2\best.pt",
    r"path\to\Ensemble\T4_Classification\ResNet101\fold_3\best.pt",
    r"path\to\Ensemble\T4_Classification\ResNet101\fold_4\best.pt",
    r"path\to\Ensemble\T4_Classification\ResNet101\fold_5\best.pt",
]

CNN3D_CKPTS = [
    r"path\to\Ensemble\T4_Classification\3DCNN\fold_1\best.pt",
    r"path\to\Ensemble\T4_Classification\3DCNN\fold_2\best.pt",
    r"path\to\Ensemble\T4_Classification\3DCNN\fold_3\best.pt",
    r"path\to\Ensemble\T4_Classification\3DCNN\fold_4\best.pt",
    r"path\to\Ensemble\T4_Classification\3DCNN\fold_5\best.pt",
]

# Same 6 cases
CASES = [
    ("HN-CHUS-034",            "T4",
     r"path\to\Ensemble\GradCAM\Sample_images\HN_CHUS_034_Cropped_Volume.nrrd",
     r"path\to\Ensemble\GradCAM\Thyroid_cartilage_masks\nrrd_converted\HN_CHUS_034.seg.nrrd"),

    ("QIN-HEADNECK-02-2502",   "non-T4",
     r"path\to\Ensemble\GradCAM\Sample_images\QIN_HEADNECK_02_2502_Cropped_Volume.nrrd",
     r"path\to\Ensemble\GradCAM\Thyroid_cartilage_masks\nrrd_converted\QIN_HEADNECK_02_2502.seg.nrrd"),

    ("RADCURE-2362",           "T4",
     r"path\to\Ensemble\GradCAM\Sample_images\RADCURE_2362_Cropped_Volume.nrrd",
     r"path\to\Ensemble\GradCAM\Thyroid_cartilage_masks\nrrd_converted\RADCURE_2362.seg.nrrd"),

    ("QIN-HEADNECK-01-0107",   "non-T4",
     r"path\to\Ensemble\GradCAM\Sample_images\QIN_HEADNECK_01_0107_Cropped_Volume.nrrd",
     r"path\to\Ensemble\GradCAM\Thyroid_cartilage_masks\nrrd_converted\QIN_HEADNECK_01_0107.seg.nrrd"),

    ("RADCURE-0016",           "non-T4",
     r"path\to\Ensemble\GradCAM\Sample_images\RADCURE_0016_Cropped_Volume.nrrd",
     r"path\to\Ensemble\GradCAM\Thyroid_cartilage_masks\nrrd_converted\RADCURE_0016.seg.nrrd"),

    ("RADCURE-2832",           "T4",
     r"path\to\Ensemble\GradCAM\Sample_images\RADCURE_2832_Cropped_Volume.nrrd",
     r"path\to\nsemble\GradCAM\Thyroid_cartilage_masks\nrrd_converted\RADCURE_2832.seg.nrrd"),
]

# ============================================================
# MODEL DEFINITIONS
# ============================================================

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
            nn.AdaptiveAvgPool3d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================
# IO + PREPROCESSING
# ============================================================

def load_nrrd(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)

def resize3d(vol, out_shape, order=1):
    factors = [
        out_shape[0] / vol.shape[0],
        out_shape[1] / vol.shape[1],
        out_shape[2] / vol.shape[2]
    ]
    return zoom(vol, factors, order=order)

def hu_clip_only(vol, lo=HU_CLIP_LOW, hi=HU_CLIP_HIGH):
    return np.clip(vol, lo, hi).astype(np.float32)

def hu_clip_norm(vol, lo=HU_CLIP_LOW, hi=HU_CLIP_HIGH):
    vol = hu_clip_only(vol, lo, hi)
    m, s = vol.mean(), vol.std()
    return (vol - m) / (s + 1e-6)

def zscore_with_fixed_stats(vol, mu, sd):
    return (vol - mu) / (sd + 1e-8)

def to_tensor3d(vol):
    return torch.from_numpy(vol[None, None]).float()

def norm01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def robust_norm_slice(x, p_low=1, p_high=99):
    lo, hi = np.percentile(x, [p_low, p_high])
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return x.astype(np.float32)

# ============================================================
# CAM POST-PROCESSING
# ============================================================

def robust_norm_cam(cam, p_low=CAM_CLIP_LOW, p_high=CAM_CLIP_HIGH):
    cam = np.maximum(cam.astype(np.float32), 0)
    if cam.max() <= 0:
        return cam
    nz = cam[cam > 0]
    if len(nz) < 10:
        return cam / (cam.max() + 1e-8)
    lo = np.percentile(nz, p_low)
    hi = np.percentile(nz, p_high)
    cam = np.clip(cam, lo, hi)
    cam = (cam - lo) / (hi - lo + 1e-8)
    return np.clip(cam, 0, 1)

def enhance_cam(cam, gamma=CAM_GAMMA):
    return np.clip(cam, 0, 1) ** gamma

def smooth_cam(cam, sigma=SMOOTH_SIGMA):
    return gaussian_filter(cam, sigma=sigma)

def threshold_cam(cam, percentile=CAM_THRESHOLD_PERCENTILE):
    cam = cam.copy()
    nz = cam[cam > 0]
    if len(nz) > 10:
        thr = np.percentile(nz, percentile)
        cam[cam < thr] = 0
    return cam

def stretch_nonzero_contrast(cam, low=STRETCH_LOW, high=STRETCH_HIGH):
    cam = cam.copy().astype(np.float32)
    nz = cam[cam > 0]
    if len(nz) < 10:
        return cam
    lo = np.percentile(nz, low)
    hi = np.percentile(nz, high)
    cam[cam > 0] = np.clip((cam[cam > 0] - lo) / (hi - lo + 1e-8), 0, 1)
    return np.clip(cam, 0, 1)

def unsharp_cam(cam, sigma=UNSHARP_SIGMA, amount=UNSHARP_AMOUNT):
    blur = gaussian_filter(cam, sigma=sigma)
    sharp = cam + amount * (cam - blur)
    return np.clip(sharp, 0, 1)

def process_cam_3d(cam):
    cam = robust_norm_cam(cam)
    cam = enhance_cam(cam)
    cam = smooth_cam(cam)
    cam = threshold_cam(cam)
    cam = stretch_nonzero_contrast(cam)
    cam = unsharp_cam(cam)
    return np.clip(cam, 0, 1)

# ============================================================
# GRAD-CAM++
# ============================================================

class SingleModelGradCAMpp3D:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=1):
        logits = self.model(x)
        score = logits[0, class_idx]

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        A = self.activations
        dY = self.gradients
        eps = 1e-8

        dY2 = dY ** 2
        dY3 = F.relu(dY) + eps

        alpha_num = dY2
        alpha_denom = 2 * dY2 + (A * dY3).sum(dim=(2, 3, 4), keepdim=True)
        alpha = alpha_num / (alpha_denom + eps)

        weights = (alpha * F.relu(dY)).sum(dim=(2, 3, 4), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-8)

        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))
        return cam, probs, pred

# ============================================================
# MODEL BUILD / LOAD
# ============================================================

def load_checkpoint_flexible(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    return model

def build_resnet101():
    return resnet101(spatial_dims=3, n_input_channels=1, num_classes=2)

def get_resnet101_target_layer(model):
    return model.layer4[-1].conv3

def build_custom3dcnn():
    return Custom3DCNN()

def get_custom3dcnn_target_layer(model):
    return model.features[16]

def load_fold_models(model_name):
    if model_name == "ResNet101":
        ckpts = RESNET101_CKPTS
        builder = build_resnet101
        target_getter = get_resnet101_target_layer
    elif model_name == "3DCNN":
        ckpts = CNN3D_CKPTS
        builder = build_custom3dcnn
        target_getter = get_custom3dcnn_target_layer
    else:
        raise ValueError("Unknown model_name")

    models = []
    explainers = []
    for ck in ckpts:
        m = builder().to(DEVICE)
        m = load_checkpoint_flexible(m, ck)
        m.eval()
        models.append(m)
        explainers.append(SingleModelGradCAMpp3D(m, target_getter(m)))
    return models, explainers

# ============================================================
# ENSEMBLE INFERENCE
# ============================================================

def ensemble_predict(fold_models, x):
    probs_all = []
    preds_all = []
    for model in fold_models:
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = int(np.argmax(probs))
        probs_all.append(probs)
        preds_all.append(pred)

    probs_mean = np.mean(np.stack(probs_all, axis=0), axis=0)
    pred_mean = int(np.argmax(probs_mean))
    return probs_mean, pred_mean, probs_all, preds_all

def foldwise_cam_average(explainers, x, class_idx):
    cams = []
    probs_all = []
    preds_all = []

    for explainer in explainers:
        cam_i, probs_i, pred_i = explainer.generate(x, class_idx)
        cam_i = robust_norm_cam(cam_i)
        cams.append(cam_i)
        probs_all.append(probs_i)
        preds_all.append(pred_i)

    cam_mean = np.mean(np.stack(cams, axis=0), axis=0)
    cam_mean = process_cam_3d(cam_mean)

    probs_mean = np.mean(np.stack(probs_all, axis=0), axis=0)
    pred_mean = int(np.argmax(probs_mean))

    return cam_mean, probs_mean, pred_mean, probs_all, preds_all

# ============================================================
# HELPERS
# ============================================================

def get_true_class(group_label):
    return 1 if group_label == "T4" else 0

def get_outcome_type(true_class, pred_class):
    if true_class == 1 and pred_class == 1:
        return "TP"
    elif true_class == 0 and pred_class == 0:
        return "TN"
    elif true_class == 0 and pred_class == 1:
        return "FP"
    elif true_class == 1 and pred_class == 0:
        return "FN"
    return "UNK"

def get_explain_class(pred_class):
    if EXPLAIN_CLASS_MODE == "predicted":
        return pred_class
    elif EXPLAIN_CLASS_MODE == "positive":
        return 1
    else:
        raise ValueError("EXPLAIN_CLASS_MODE must be 'predicted' or 'positive'")

def get_mask_center(mask):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    return np.round(coords.mean(axis=0)).astype(int)

# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def overlay_heatmap(bg, cam, cmap_name=CMAP_NAME):
    bg = robust_norm_slice(bg, 1, 99)
    bg_rgb = np.repeat(bg[..., None], 3, axis=-1)

    cam_vis = np.clip(cam, 0, 1)
    cam_display = np.clip(cam_vis ** 0.90, 0, 1)
    cam_rgb = plt.get_cmap(cmap_name)(cam_display)[..., :3]

    alpha_map = OVERLAY_ALPHA_MIN + (OVERLAY_ALPHA_MAX - OVERLAY_ALPHA_MIN) * cam_display
    alpha_map = np.clip(alpha_map, 0, 1)
    alpha_map[cam_display <= 0] = 0

    overlay = bg_rgb.copy()
    for c in range(3):
        overlay[..., c] = (1 - alpha_map) * bg_rgb[..., c] + alpha_map * cam_rgb[..., c]

    return np.clip(overlay, 0, 1)

def save_cam_overlay_3views(vol_norm, cam, roi, out_png, explain_class_idx):
    center = get_mask_center(roi)
    if center is not None:
        dz, dy, dx = center
    else:
        dz = int(cam.sum(axis=(1, 2)).argmax())
        dy = int(cam.sum(axis=(0, 2)).argmax())
        dx = int(cam.sum(axis=(0, 1)).argmax())

    class_name = "T4" if explain_class_idx == 1 else "non-T4"
    views = [
        (vol_norm[dz],       cam[dz],       roi[dz],       f"Axial + Mask Outline\nCAM for {class_name}"),
        (vol_norm[:, dy, :], cam[:, dy, :], roi[:, dy, :], f"Coronal + Mask Outline\nCAM for {class_name}"),
        (vol_norm[:, :, dx], cam[:, :, dx], roi[:, :, dx], f"Sagittal + Mask Outline\nCAM for {class_name}"),
    ]

    fig, axs = plt.subplots(3, 1, figsize=(6, 10))
    for ax, (bg, heat, mask, title) in zip(axs, views):
        overlay = overlay_heatmap(bg, heat)
        ax.imshow(overlay, interpolation="bilinear")

        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='white', linewidth=2.0)
            ax.plot(contour[:, 1], contour[:, 0], color='lime', linewidth=1.2)

        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

# ============================================================
# DELETION / INSERTION USING CAM
# ============================================================

def compute_deletion_insertion_curves(vol_raw, cam, fold_models, class_idx, steps=11):
    vol_proc = hu_clip_only(vol_raw)
    mu0 = float(vol_proc.mean())
    sd0 = float(vol_proc.std() + 1e-8)

    flat_order = np.argsort(cam.flatten())[::-1]
    N = cam.size
    ks = (np.linspace(0, 1, steps) * N).astype(int)

    x_full = to_tensor3d(zscore_with_fixed_stats(vol_proc, mu0, sd0)).to(DEVICE)
    probs0_mean, _, _, _ = ensemble_predict(fold_models, x_full)
    p0 = float(probs0_mean[class_idx])

    canvas = np.full_like(vol_proc, mu0, dtype=np.float32)
    x_canvas = to_tensor3d(zscore_with_fixed_stats(canvas, mu0, sd0)).to(DEVICE)
    probs_canvas_mean, _, _, _ = ensemble_predict(fold_models, x_canvas)
    p_base = float(probs_canvas_mean[class_idx])

    del_probs = []
    ins_probs = []

    flat_orig = vol_proc.reshape(-1)

    for k in ks:
        # deletion
        vol_del = vol_proc.copy().reshape(-1)
        if k > 0:
            vol_del[flat_order[:k]] = mu0
        vol_del = vol_del.reshape(vol_proc.shape)
        x_del = to_tensor3d(zscore_with_fixed_stats(vol_del, mu0, sd0)).to(DEVICE)
        p_del = float(ensemble_predict(fold_models, x_del)[0][class_idx])
        del_probs.append(p_del)

        # insertion
        vol_ins = canvas.copy().reshape(-1)
        if k > 0:
            vol_ins[flat_order[:k]] = flat_orig[flat_order[:k]]
        vol_ins = vol_ins.reshape(vol_proc.shape)
        x_ins = to_tensor3d(zscore_with_fixed_stats(vol_ins, mu0, sd0)).to(DEVICE)
        p_ins = float(ensemble_predict(fold_models, x_ins)[0][class_idx])
        ins_probs.append(p_ins)

    pct = np.linspace(0, 100, steps)
    del_probs = np.array(del_probs)
    ins_probs = np.array(ins_probs)

    del_delta = p0 - del_probs
    ins_delta = ins_probs - p_base

    return pct, del_probs, ins_probs, del_delta, ins_delta, p0, p_base

# ============================================================
# MASK-BASED DELETION / INSERTION
# ============================================================

def compute_mask_based_curves(vol_raw, mask, fold_models, class_idx, steps=11):
    vol_proc = hu_clip_only(vol_raw)
    mu0 = float(vol_proc.mean())
    sd0 = float(vol_proc.std() + 1e-8)

    x_full = to_tensor3d(zscore_with_fixed_stats(vol_proc, mu0, sd0)).to(DEVICE)
    probs0_mean, _, _, _ = ensemble_predict(fold_models, x_full)
    p0 = float(probs0_mean[class_idx])

    canvas = np.full_like(vol_proc, mu0, dtype=np.float32)
    x_canvas = to_tensor3d(zscore_with_fixed_stats(canvas, mu0, sd0)).to(DEVICE)
    probs_canvas_mean, _, _, _ = ensemble_predict(fold_models, x_canvas)
    p_base = float(probs_canvas_mean[class_idx])

    coords = np.argwhere(mask > 0)
    n_vox = len(coords)

    if n_vox == 0:
        pct = np.linspace(0, 100, steps)
        zeros = np.zeros_like(pct, dtype=float)
        return pct, zeros, zeros, zeros, zeros, p0, p_base

    coords = coords[np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))]
    ks = (np.linspace(0, 1, steps) * n_vox).astype(int)

    del_probs = []
    ins_probs = []

    for k in ks:
        # deletion
        vol_del = vol_proc.copy()
        if k > 0:
            c = coords[:k]
            vol_del[c[:, 0], c[:, 1], c[:, 2]] = mu0
        x_del = to_tensor3d(zscore_with_fixed_stats(vol_del, mu0, sd0)).to(DEVICE)
        p_del = float(ensemble_predict(fold_models, x_del)[0][class_idx])
        del_probs.append(p_del)

        # insertion
        vol_ins = canvas.copy()
        if k > 0:
            c = coords[:k]
            vol_ins[c[:, 0], c[:, 1], c[:, 2]] = vol_proc[c[:, 0], c[:, 1], c[:, 2]]
        x_ins = to_tensor3d(zscore_with_fixed_stats(vol_ins, mu0, sd0)).to(DEVICE)
        p_ins = float(ensemble_predict(fold_models, x_ins)[0][class_idx])
        ins_probs.append(p_ins)

    pct = np.linspace(0, 100, steps)
    del_probs = np.array(del_probs)
    ins_probs = np.array(ins_probs)

    del_delta = p0 - del_probs
    ins_delta = ins_probs - p_base

    return pct, del_probs, ins_probs, del_delta, ins_delta, p0, p_base

# ============================================================
# OVERLAP METRICS
# ============================================================

def compute_overlap_metrics(cam, mask, thr):
    cam_bin = (cam > thr).astype(np.float32)
    intersection = np.sum(cam_bin * mask)

    overlap_fraction = intersection / (np.sum(cam_bin) + 1e-8)
    coverage_ratio = intersection / (np.sum(mask) + 1e-8)
    dice = (2 * intersection) / (np.sum(cam_bin) + np.sum(mask) + 1e-8)

    inside_mean = np.mean(cam[mask == 1]) if np.sum(mask) > 0 else 0.0
    outside_mean = np.mean(cam[mask == 0])
    enrich = inside_mean / (outside_mean + 1e-8)

    return overlap_fraction, coverage_ratio, dice, enrich

# ============================================================
# AUC
# ============================================================

def auc_trapz(x_pct, y):
    x = x_pct / 100.0
    return float(np.trapz(y, x))

# ============================================================
# PLOTTING CURVES
# ============================================================

def save_individual_curve_plot(pct, del_delta, ins_delta, case_id, out_png, title_prefix="CAM", class_name="pred"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(pct, del_delta, 'r-o', linewidth=2)
    axs[0].set_title(f"{title_prefix} Deletion ({class_name})")
    axs[0].set_xlabel("Top voxels removed (%)")
    axs[0].set_ylabel("Δ Prob (p0 − p)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(pct, ins_delta, 'b-o', linewidth=2)
    axs[1].set_title(f"{title_prefix} Insertion ({class_name})")
    axs[1].set_xlabel("Top voxels inserted (%)")
    axs[1].set_ylabel("Δ Prob (p − p_base)")
    axs[1].grid(True, alpha=0.3)

    plt.suptitle(case_id, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def save_group_curve_plot(pct, curves_dict, title, ylabel, out_png):
    plt.figure(figsize=(7.5, 5.5))
    color_map = {
        "T4": "red",
        "non-T4": "blue",
        "TP": "darkgreen",
        "TN": "navy",
        "FP": "orange",
        "FN": "purple"
    }

    for grp, arrs in curves_dict.items():
        if len(arrs) == 0:
            continue
        color = color_map.get(grp, "gray")
        for arr in arrs:
            plt.plot(pct, arr, alpha=0.20, lw=1, color=color)

        A = np.vstack(arrs)
        mean = A.mean(axis=0)
        sd = A.std(axis=0)
        plt.plot(pct, mean, color=color, lw=3, label=f"{grp} mean±SD")
        plt.fill_between(pct, mean - sd, mean + sd, color=color, alpha=0.12)

    plt.xlabel("Top voxels (%)")
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=400)
    plt.close()

# ============================================================
# MAIN DRIVER
# ============================================================

def run_model_explainability(model_name):
    print(f"\n{'='*72}")
    print(f"Running explainability for: {model_name}")
    print(f"{'='*72}")

    model_dir = os.path.join(BASE_SAVE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    cam_dir = os.path.join(model_dir, "cams")
    curve_dir = os.path.join(model_dir, "curves")
    overlay_dir = os.path.join(model_dir, "overlays")
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    fold_models, explainers = load_fold_models(model_name)

    records_cam_curves = []
    records_mask_curves = []
    records_overlap = []

    # grouped by TRUE CLASS
    group_cam_del_true = {"T4": [], "non-T4": []}
    group_cam_ins_true = {"T4": [], "non-T4": []}
    group_mask_del_true = {"T4": [], "non-T4": []}
    group_mask_ins_true = {"T4": [], "non-T4": []}

    # grouped by OUTCOME TYPE
    group_cam_del_outcome = {"TP": [], "TN": [], "FP": [], "FN": []}
    group_cam_ins_outcome = {"TP": [], "TN": [], "FP": [], "FN": []}
    group_mask_del_outcome = {"TP": [], "TN": [], "FP": [], "FN": []}
    group_mask_ins_outcome = {"TP": [], "TN": [], "FP": [], "FN": []}

    pct_axis = None

    for case_id, group_label, img_path, mask_path in tqdm(CASES, desc=f"{model_name} cases"):
        true_class = get_true_class(group_label)

        # ---------------------------
        # load image and mask
        # ---------------------------
        vol_raw = load_nrrd(img_path)
        mask = load_nrrd(mask_path)

        if vol_raw.shape != INPUT_SHAPE:
            vol_raw = resize3d(vol_raw, INPUT_SHAPE, order=1)
        if mask.shape != INPUT_SHAPE:
            mask = resize3d(mask, INPUT_SHAPE, order=0)

        mask = (mask > 0).astype(np.float32)

        # ---------------------------
        # model input preprocessing
        # ---------------------------
        vol_norm = hu_clip_norm(vol_raw)
        x = to_tensor3d(vol_norm).to(DEVICE)

        # first get prediction
        probs_pred, pred_mean, _, _ = ensemble_predict(fold_models, x)
        explain_class_idx = get_explain_class(pred_mean)
        explain_class_name = "T4" if explain_class_idx == 1 else "non-T4"

        is_correct = int(pred_mean == true_class)
        outcome_type = get_outcome_type(true_class, pred_mean)

        # ---------------------------
        # fold-wise averaged CAM FOR EXPLAIN CLASS
        # ---------------------------
        cam, probs_mean, pred_mean2, probs_all, preds_all = foldwise_cam_average(
            explainers, x, explain_class_idx
        )

        np.save(os.path.join(cam_dir, f"{case_id}_cam.npy"), cam)
        np.save(os.path.join(cam_dir, f"{case_id}_mask.npy"), mask)

        save_cam_overlay_3views(
            vol_norm=vol_norm,
            cam=cam,
            roi=mask,
            out_png=os.path.join(overlay_dir, f"{case_id}_overlay.png"),
            explain_class_idx=explain_class_idx
        )

        # ---------------------------
        # CAM-guided deletion/insertion
        # ---------------------------
        pct, del_probs, ins_probs, del_delta, ins_delta, p0, p_base = compute_deletion_insertion_curves(
            vol_raw=vol_raw,
            cam=cam,
            fold_models=fold_models,
            class_idx=explain_class_idx,
            steps=STEPS
        )
        pct_axis = pct

        group_cam_del_true[group_label].append(del_delta)
        group_cam_ins_true[group_label].append(ins_delta)
        group_cam_del_outcome[outcome_type].append(del_delta)
        group_cam_ins_outcome[outcome_type].append(ins_delta)

        save_individual_curve_plot(
            pct, del_delta, ins_delta, case_id,
            os.path.join(curve_dir, f"{case_id}_CAM_DelIns.png"),
            title_prefix="CAM",
            class_name=explain_class_name
        )

        records_cam_curves.append({
            "CaseID": case_id,
            "true_group": group_label,
            "true_class": true_class,
            "pred_class": int(pred_mean),
            "is_correct": is_correct,
            "outcome_type": outcome_type,
            "explain_class": explain_class_idx,
            "explain_class_name": explain_class_name,
            "p0": p0,
            "p_base": p_base,
            "del_auc_raw": auc_trapz(pct, del_probs),
            "del_auc_delta": auc_trapz(pct, del_delta),
            "ins_auc_raw": auc_trapz(pct, ins_probs),
            "ins_auc_delta": auc_trapz(pct, ins_delta),
            "prob_class0": float(probs_pred[0]),
            "prob_class1": float(probs_pred[1]),
        })

        # ---------------------------
        # MASK-guided deletion/insertion
        # ---------------------------
        pct_m, del_probs_m, ins_probs_m, del_delta_m, ins_delta_m, p0_m, p_base_m = compute_mask_based_curves(
            vol_raw=vol_raw,
            mask=mask,
            fold_models=fold_models,
            class_idx=explain_class_idx,
            steps=STEPS
        )

        group_mask_del_true[group_label].append(del_delta_m)
        group_mask_ins_true[group_label].append(ins_delta_m)
        group_mask_del_outcome[outcome_type].append(del_delta_m)
        group_mask_ins_outcome[outcome_type].append(ins_delta_m)

        save_individual_curve_plot(
            pct_m, del_delta_m, ins_delta_m, case_id,
            os.path.join(curve_dir, f"{case_id}_MASK_DelIns.png"),
            title_prefix="Mask",
            class_name=explain_class_name
        )

        records_mask_curves.append({
            "CaseID": case_id,
            "true_group": group_label,
            "true_class": true_class,
            "pred_class": int(pred_mean),
            "is_correct": is_correct,
            "outcome_type": outcome_type,
            "explain_class": explain_class_idx,
            "explain_class_name": explain_class_name,
            "p0": p0_m,
            "p_base": p_base_m,
            "AUC_Deletion_Raw": auc_trapz(pct_m, del_probs_m),
            "AUC_Deletion_Delta": auc_trapz(pct_m, del_delta_m),
            "AUC_Insertion_Raw": auc_trapz(pct_m, ins_probs_m),
            "AUC_Insertion_Delta": auc_trapz(pct_m, ins_delta_m),
        })

        # ---------------------------
        # CAM-mask overlap metrics
        # ---------------------------
        cam_norm = norm01(cam)
        for thr in [0.3, 0.5, 0.7]:
            overlap_fraction, coverage_ratio, dice, enrich = compute_overlap_metrics(cam_norm, mask, thr)
            records_overlap.append({
                "CaseID": case_id,
                "true_group": group_label,
                "true_class": true_class,
                "pred_class": int(pred_mean),
                "is_correct": is_correct,
                "outcome_type": outcome_type,
                "explain_class": explain_class_idx,
                "explain_class_name": explain_class_name,
                "Threshold": thr,
                "CAM_Mask_OverlapFraction": overlap_fraction,
                "Mask_CoverageRatio": coverage_ratio,
                "Dice_CAM_Mask": dice,
                "Enrichment": enrich
            })

    # ========================================================
    # SAVE CSVs
    # ========================================================
    cam_curve_csv = os.path.join(model_dir, f"{model_name}_CAM_DelIns_AUCs.csv")
    mask_curve_csv = os.path.join(model_dir, f"{model_name}_Mask_DelIns_AUCs.csv")
    overlap_csv = os.path.join(model_dir, f"{model_name}_CAM_Mask_Overlap_Metrics.csv")

    cam_df = pd.DataFrame(records_cam_curves)
    mask_df = pd.DataFrame(records_mask_curves)
    overlap_df = pd.DataFrame(records_overlap)

    cam_df.to_csv(cam_curve_csv, index=False)
    mask_df.to_csv(mask_curve_csv, index=False)
    overlap_df.to_csv(overlap_csv, index=False)

    # ========================================================
    # GROUP PLOTS - TRUE CLASS
    # ========================================================
    save_group_curve_plot(
        pct_axis, group_cam_del_true,
        title=f"{model_name}: CAM-guided Deletion Curves by TRUE CLASS",
        ylabel="Δ Prob (p0 − p)",
        out_png=os.path.join(model_dir, f"{model_name}_CAM_Deletion_Group_TRUECLASS.png")
    )

    save_group_curve_plot(
        pct_axis, group_cam_ins_true,
        title=f"{model_name}: CAM-guided Insertion Curves by TRUE CLASS",
        ylabel="Δ Prob (p − p_base)",
        out_png=os.path.join(model_dir, f"{model_name}_CAM_Insertion_Group_TRUECLASS.png")
    )

    save_group_curve_plot(
        pct_axis, group_mask_del_true,
        title=f"{model_name}: Mask-guided Deletion Curves by TRUE CLASS",
        ylabel="Δ Prob (p0 − p)",
        out_png=os.path.join(model_dir, f"{model_name}_MASK_Deletion_Group_TRUECLASS.png")
    )

    save_group_curve_plot(
        pct_axis, group_mask_ins_true,
        title=f"{model_name}: Mask-guided Insertion Curves by TRUE CLASS",
        ylabel="Δ Prob (p − p_base)",
        out_png=os.path.join(model_dir, f"{model_name}_MASK_Insertion_Group_TRUECLASS.png")
    )

    # ========================================================
    # GROUP PLOTS - OUTCOME TYPE
    # ========================================================
    save_group_curve_plot(
        pct_axis, group_cam_del_outcome,
        title=f"{model_name}: CAM-guided Deletion Curves by OUTCOME TYPE",
        ylabel="Δ Prob (p0 − p)",
        out_png=os.path.join(model_dir, f"{model_name}_CAM_Deletion_Group_OUTCOME.png")
    )

    save_group_curve_plot(
        pct_axis, group_cam_ins_outcome,
        title=f"{model_name}: CAM-guided Insertion Curves by OUTCOME TYPE",
        ylabel="Δ Prob (p − p_base)",
        out_png=os.path.join(model_dir, f"{model_name}_CAM_Insertion_Group_OUTCOME.png")
    )

    save_group_curve_plot(
        pct_axis, group_mask_del_outcome,
        title=f"{model_name}: Mask-guided Deletion Curves by OUTCOME TYPE",
        ylabel="Δ Prob (p0 − p)",
        out_png=os.path.join(model_dir, f"{model_name}_MASK_Deletion_Group_OUTCOME.png")
    )

    save_group_curve_plot(
        pct_axis, group_mask_ins_outcome,
        title=f"{model_name}: Mask-guided Insertion Curves by OUTCOME TYPE",
        ylabel="Δ Prob (p − p_base)",
        out_png=os.path.join(model_dir, f"{model_name}_MASK_Insertion_Group_OUTCOME.png")
    )

    # ========================================================
    # SUMMARIES
    # ========================================================
    cam_summary_true = cam_df.groupby("true_group")[["del_auc_raw", "del_auc_delta", "ins_auc_raw", "ins_auc_delta"]].agg(["mean", "std"])
    cam_summary_outcome = cam_df.groupby("outcome_type")[["del_auc_raw", "del_auc_delta", "ins_auc_raw", "ins_auc_delta"]].agg(["mean", "std"])

    mask_summary_true = mask_df.groupby("true_group")[["AUC_Deletion_Raw", "AUC_Deletion_Delta", "AUC_Insertion_Raw", "AUC_Insertion_Delta"]].agg(["mean", "std"])
    mask_summary_outcome = mask_df.groupby("outcome_type")[["AUC_Deletion_Raw", "AUC_Deletion_Delta", "AUC_Insertion_Raw", "AUC_Insertion_Delta"]].agg(["mean", "std"])

    overlap_case_means = overlap_df.groupby(["CaseID", "true_group", "outcome_type"]).mean(numeric_only=True).reset_index()
    overlap_summary_true = overlap_case_means.groupby("true_group")[["CAM_Mask_OverlapFraction", "Mask_CoverageRatio", "Dice_CAM_Mask", "Enrichment"]].agg(["mean", "std"])
    overlap_summary_outcome = overlap_case_means.groupby("outcome_type")[["CAM_Mask_OverlapFraction", "Mask_CoverageRatio", "Dice_CAM_Mask", "Enrichment"]].agg(["mean", "std"])

    cam_summary_true.to_csv(os.path.join(model_dir, f"{model_name}_CAM_DelIns_Summary_TRUECLASS.csv"))
    cam_summary_outcome.to_csv(os.path.join(model_dir, f"{model_name}_CAM_DelIns_Summary_OUTCOME.csv"))

    mask_summary_true.to_csv(os.path.join(model_dir, f"{model_name}_Mask_DelIns_Summary_TRUECLASS.csv"))
    mask_summary_outcome.to_csv(os.path.join(model_dir, f"{model_name}_Mask_DelIns_Summary_OUTCOME.csv"))

    overlap_summary_true.to_csv(os.path.join(model_dir, f"{model_name}_Overlap_Summary_TRUECLASS.csv"))
    overlap_summary_outcome.to_csv(os.path.join(model_dir, f"{model_name}_Overlap_Summary_OUTCOME.csv"))

    print(f"\n Finished {model_name}")
    print("Saved to:", model_dir)

    return {
        "cam_df": cam_df,
        "mask_df": mask_df,
        "overlap_df": overlap_df,
        "model_dir": model_dir,
    }

# ============================================================
# RUN BOTH MODELS
# ============================================================

results_resnet101 = run_model_explainability("ResNet101")
results_3dcnn = run_model_explainability("3DCNN")

print("\n All explainability analyses completed for both models.")
print("Base output folder:", BASE_SAVE_DIR)
