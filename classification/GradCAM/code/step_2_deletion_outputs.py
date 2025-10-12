# ============================================================
# Grad-CAM++ for 3D CT classifiers (ResNet18 / CNN5 baseline)
# + Deletion Curve Analysis (All Six Cases)
# ============================================================
import os, math, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.cm as cm

# ---------------------------
# CONFIGURATION
# ---------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "resnet18"         # "resnet18" or "cnn5"
CLASS_INDEX  = 1                  # positive class index (T4)
INPUT_SHAPE  = (32, 96, 96)
SAVE_DIR     = "./gradcampp_outputs_with_deletion"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
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
# Grad-CAM++ CLASS
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
        A, dY = self.activations, self.gradients
        eps = 1e-8
        dY2, dY3 = (dY ** 2), (F.relu(dY) + eps)
        alpha_num = dY2
        alpha_denom = 2 * dY2 + (A * dY3).sum(dim=(2,3,4), keepdim=True)
        alpha = alpha_num / (alpha_denom + eps)
        weights = (alpha * F.relu(dY)).sum(dim=(2,3,4), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0,0].detach().cpu().numpy()
        cam[cam < cam.max() * 0.2] = 0
        vmax = np.percentile(cam, 99)
        cam = np.clip(cam / (vmax + 1e-8), 0, 1)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return cam, probs

# ---------------------------
# MODEL LOADING
# ---------------------------
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
        raise ValueError("Unknown model")
    return model

def get_target_layer(model, name):
    if "resnet" in name:
        return model.layer4[-1].conv2
    elif "cnn5" in name:
        return list(model.children())[6]
    else:
        raise ValueError("Target layer undefined")

# ============================================================
# DELETION CURVE ANALYSIS
# ============================================================
def deletion_curve(vol, cam, model, class_idx=1, steps=10):
    """Iteratively zero out top CAM voxels and record drop in T4 probability."""
    x0 = torch.from_numpy(vol[None,None]).float().to(DEVICE)
    flat = cam.flatten()
    order = np.argsort(flat)[::-1]  # descending importance
    N = len(flat)
    chunk = N // steps
    probs = []

    with torch.no_grad():
        base_prob = torch.softmax(model(x0), dim=1)[0,class_idx].item()
    probs.append(base_prob)

    masked = x0.clone()
    for k in range(1, steps+1):
        idxs = order[:k*chunk]
        d, h, w = np.unravel_index(idxs, cam.shape)
        masked[0,0,d,h,w] = 0.0
        with torch.no_grad():
            pk = torch.softmax(model(masked), dim=1)[0,class_idx].item()
        probs.append(pk)

    return np.linspace(0,100,steps+1), probs

# ============================================================
# MAIN DRIVER
# ============================================================
model = build_model(MODEL_NAME).to(DEVICE)
ckpt_path = "path/to/resnet18_T4_dir/fold_2/best_model.pth"
checkpoint = torch.load(ckpt_path, map_location=DEVICE)
state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ Model weights loaded successfully from:", ckpt_path)

# ---------------------------
# CASES
# ---------------------------
cases = [
    ("Case001_T4", "path/to/HN_CHUS_034_Cropped_Volume.nrrd"),
    ("Case002_T4", "path/to/QIN_HEADNECK_02_2502_Cropped_Volume.nrrd"),
    ("Case003_T4", "path/to/RADCURE_0049_Cropped_Volume.nrrd"),
    ("Case004_non_T4", "path/to/RADCURE_2362_Cropped_Volume.nrrd"),
    ("Case005_non_T4", "path/to/QIN_HEADNECK_01_0048_Cropped_Volume.nrrd"),
    ("Case006_non_T4", "path/to/RADCURE_0039_Cropped_Volume.nrrd"),
]

# ============================================================
# LOOP THROUGH CASES – GENERATE Grad-CAM++ + Deletion Curve
# ============================================================
target = get_target_layer(model, MODEL_NAME)
gradcampp = GradCAMpp3D(model, target)

all_curves = []
plt.figure(figsize=(6,5))

for cid, img_path in tqdm(cases):
    vol = load_nrrd(img_path)
    if vol.shape != INPUT_SHAPE:
        vol = resize3d(vol, INPUT_SHAPE)
    x = to_tensor3d(vol).to(DEVICE)
    cam, probs = gradcampp.generate(x, CLASS_INDEX)

    # --- save individual CAM ---
    np.save(os.path.join(SAVE_DIR, f"{cid}_cam.npy"), cam)

    # --- run deletion curve ---
    xvals, yvals = deletion_curve(vol, cam, model, CLASS_INDEX, steps=10)
    all_curves.append(yvals)
    plt.plot(xvals, yvals, marker='o', alpha=0.6, label=cid)

# ============================================================
# AGGREGATED CURVE (Mean ± SD)
# ============================================================
all_curves = np.array(all_curves)
mean_curve = all_curves.mean(axis=0)
std_curve = all_curves.std(axis=0)

plt.plot(xvals, mean_curve, color='black', linewidth=2.5, label='Mean ± SD')
plt.fill_between(xvals, mean_curve-std_curve, mean_curve+std_curve, color='gray', alpha=0.3)
plt.xlabel("Top-CAM voxels removed (%)")
plt.ylabel("Predicted T4 probability")
plt.title("Deletion-Curve Analysis (All Six Cases)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "Deletion_Curves_AllCases.png"), dpi=600)
plt.show()

print("✅ Grad-CAM++ and Deletion Curve analysis completed for all cases.")