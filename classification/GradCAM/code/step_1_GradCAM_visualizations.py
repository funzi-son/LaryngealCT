# install dependencies
!pip install torch torchvision monai SimpleITK scipy matplotlib numpy tqdm


# ============================================================
# Grad-CAM++ for 3D CT Classifiers (ResNet18 / CNN5 baseline)
# Deterministic + Mask Outline + Composite Figure Edition
# ============================================================

import os, math, numpy as np, random
import torch, torch.nn as nn, torch.nn.functional as F
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.cm as cm
from skimage import measure
import string

# ============================================================
# 0. CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "resnet18"         # "resnet18" or "cnn5"
CLASS_INDEX = 1                 # Target class: T4
INPUT_SHAPE = (32, 96, 96)
SAVE_DIR = "./gradcampp_outputs_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 1. DETERMINISTIC MODE (WARN ONLY)
# ============================================================
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
print("✅ Deterministic mode (warn-only) enabled — Grad-CAMs reproducible with stable fallback.")

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def load_nrrd(path):
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    return vol

def to_tensor3d(vol):
    return torch.from_numpy(vol[None, None]).float()

def resize3d(vol, out_shape):
    factors = [out_shape[0]/vol.shape[0],
               out_shape[1]/vol.shape[1],
               out_shape[2]/vol.shape[2]]
    return zoom(vol, factors, order=1)

def norm01(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min + 1e-6)

# ============================================================
# 3. VISUALIZATION UTILITIES
# ============================================================
def overlay_heatmap(bg, cam, alpha=0.4):
    bg = norm01(bg)
    cam[cam < cam.max() * 0.2] = 0
    vmax = np.percentile(cam, 99)
    cam = np.clip(cam / (vmax + 1e-8), 0, 1)
    cmap = cm.get_cmap('jet')
    cam_rgb = cmap(cam)[..., :3]
    overlay = (1 - alpha) * np.repeat(bg[..., None], 3, axis=-1) + alpha * cam_rgb
    return np.clip(overlay, 0, 1)

def save_slices_with_mask(vol, cam, roi, prefix):
    D, H, W = vol.shape
    dz, dy, dx = D//2, H//2, W//2
    views = [
        (vol[dz], cam[dz], roi[dz], "Axial"),
        (vol[:, dy, :], cam[:, dy, :], roi[:, dy, :], "Coronal"),
        (vol[:, :, dx], cam[:, :, dx], roi[:, :, dx], "Sagittal")
    ]
    fig, axs = plt.subplots(3, 1, figsize=(6, 9))
    for i, (bg, heat, mask, title) in enumerate(views):
        overlay = overlay_heatmap(bg, heat)
        axs[i].imshow(overlay)
        # Mask outline
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            axs[i].plot(contour[:, 1], contour[:, 0], color='lime', linewidth=1.0)
        axs[i].set_title(f"{title} + Mask Outline", fontsize=10)
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(f"{prefix}_slices_masked.png", dpi=300, bbox_inches="tight")
    plt.close()

def save_mip(cam, prefix):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    vmax = np.percentile(cam, 99)
    axs[0].imshow(cam.max(0), cmap='hot', vmin=0, vmax=vmax); axs[0].set_title('Axial MIP')
    axs[1].imshow(cam.max(1), cmap='hot', vmin=0, vmax=vmax); axs[1].set_title('Coronal MIP')
    axs[2].imshow(cam.max(2), cmap='hot', vmin=0, vmax=vmax); axs[2].set_title('Sagittal MIP')
    for a in axs: a.axis('off')
    plt.tight_layout(); plt.savefig(f"{prefix}_mip.png", dpi=200, bbox_inches="tight"); plt.close()

def roi_enrichment(cam, roi):
    if isinstance(cam, np.ndarray): cam = torch.tensor(cam)
    if isinstance(roi, np.ndarray): roi = torch.tensor(roi)
    cam = cam / (cam.sum() + 1e-8)
    roi = roi.float()
    mass_in = (cam * roi).sum()
    frac_roi = roi.mean()
    enrich = (mass_in / (frac_roi + 1e-8)).item()
    return float(mass_in), float(frac_roi), float(enrich)

# ============================================================
# 4. MODEL LOADING
# ============================================================
def build_model(name):
    if name == "resnet18":
        from monai.networks.nets import resnet18
        model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
    elif name == "cnn5":
        model = nn.Sequential(
            nn.Conv3d(1,16,3,1,1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,1,1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,1,1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64,128,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
            nn.Flatten(), nn.Linear(128,2)
        )
    else:
        raise ValueError("Unknown model type.")
    return model

def get_target_layer(model, name):
    if "resnet" in name:
        return model.layer4[-1].conv2
    elif "cnn5" in name:
        return list(model.children())[6]
    else:
        raise ValueError("Target layer undefined")

# ============================================================
# 5. GRAD-CAM++ ENGINE
# ============================================================
class GradCAMpp3D:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx):
        logits = self.model(x)
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        A = self.activations; dY = self.gradients
        eps = 1e-8
        dY2 = (dY ** 2); dY3 = F.relu(dY) + eps
        alpha_num = dY2
        alpha_denom = 2 * dY2 + (A * dY3).sum(dim=(2,3,4), keepdim=True)
        alpha = alpha_num / (alpha_denom + eps)
        weights = (alpha * F.relu(dY)).sum(dim=(2,3,4), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam[cam < cam.max() * 0.2] = 0
        vmax = np.percentile(cam, 99)
        cam = np.clip(cam / (vmax + 1e-8), 0, 1)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return cam, probs

# ============================================================
# 6. RUN CASES
# ============================================================
def run_case(case_id, img_path, mask_path, model, name):
    vol = load_nrrd(img_path)
    if vol.shape != INPUT_SHAPE:
        vol = resize3d(vol, INPUT_SHAPE)
    x = to_tensor3d(vol).to(DEVICE)
    roi = None
    if mask_path and os.path.exists(mask_path):
        roi = load_nrrd(mask_path)
        if roi.shape != INPUT_SHAPE:
            roi = resize3d(roi, INPUT_SHAPE)
        roi = (roi > 0).astype(np.float32)
    target = get_target_layer(model, name)
    campp = GradCAMpp3D(model, target)
    cam, probs = campp.generate(x, CLASS_INDEX)
    prefix = os.path.join(SAVE_DIR, case_id)
    if roi is not None:
        save_slices_with_mask(norm01(vol), cam, roi, prefix)
    else:
        print(f"⚠️ No mask found for {case_id}")
    save_mip(cam, prefix)
    if roi is not None:
        m, frac, e = roi_enrichment(torch.tensor(cam), torch.tensor(roi))
        with open(f"{prefix}_report.txt", "w") as f:
            f.write(f"Probs={probs}\nMassIn={m:.4f} ROIfrac={frac:.4f} Enrich={e:.2f}x\n")
    print(f"✅ Saved Grad-CAM++ overlays for {case_id}")

# ============================================================
# 7. MAIN DRIVER
# ============================================================
model = build_model(MODEL_NAME).to(DEVICE)
ckpt_path = "path/to/resnet18_T4_dir/fold_2/best_model.pth"
checkpoint = torch.load(ckpt_path, map_location=DEVICE)
if "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ Model weights loaded successfully from:", ckpt_path)

cases = [
    ("Case001_T4", "path/to/HN_CHUS_034_Cropped_Volume.nrrd", "path/to/HN_CHUS_034.seg.nrrd"),
    ("Case002_T4", "path/to/QIN_HEADNECK_02_2502_Cropped_Volume.nrrd", "path/to/QIN_HEADNECK_02_2502.seg.nrrd"),
    ("Case003_T4", "path/to/RADCURE_1726_Cropped_Volume.nrrd", "path/to/RADCURE_1726.seg.nrrd"),
    ("Case004_non_T4", "path/to/RADCURE_2362_Cropped_Volume.nrrd", "path/to/RADCURE_2362.seg.nrrd"),
    ("Case005_non_T4", "path/to/QIN_HEADNECK_01_0048_Cropped_Volume.nrrd", "path/to/QIN_HEADNECK_01_0048.seg.nrrd"),
    ("Case006_non_T4", "paht/to/RADCURE_0039_Cropped_Volume.nrrd", "path/to/RADCURE_0039.seg.nrrd"),
]

for cid, img, mask in tqdm(cases):
    run_case(cid, img, mask, model, MODEL_NAME)
print("🎯 Grad-CAM++ completed with deterministic reproducibility and mask overlays.")

# ============================================================
# 8. COMPOSITE FIGURE (2×3, COLORBAR + ANNOTATIONS)
# ============================================================
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

def create_composite_no_titles(image_dir, output_path):
    # ----------------------------------------------------------
    # File order and labels
    # ----------------------------------------------------------
    case_order = [
        "Case001_T4_slices_masked.png", 
        "Case002_T4_slices_masked.png", 
        "Case003_T4_slices_masked.png",
        "Case004_non_T4_slices_masked.png", 
        "Case005_non_T4_slices_masked.png", 
        "Case006_non_T4_slices_masked.png"
    ]
    labels = list(string.ascii_lowercase[:6])
    imgs = [mpimg.imread(os.path.join(image_dir, f)) for f in case_order]

    # ----------------------------------------------------------
    # Layout setup
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(left=0.03, right=0.88, top=0.95, bottom=0.05, wspace=0.02, hspace=0.05)

    for ax, img, label in zip(axes.flat, imgs, labels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"({label})", fontsize=13, fontweight="bold", loc="left", pad=3)

    # ----------------------------------------------------------
    # Colorbar
    # ----------------------------------------------------------
    cax = fig.add_axes([0.9, 0.25, 0.02, 0.5])
    cmap = get_cmap('jet')
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Grad-CAM++ intensity", fontsize=12)

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✅ Composite figure saved to: {output_path}")

# Example usage
create_composite_no_titles(
    image_dir=r"./gradcampp_outputs_final",
    output_path=r"./gradcampp_outputs_final/GradCAMpp_Composite.png"
)

