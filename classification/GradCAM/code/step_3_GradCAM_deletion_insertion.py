# ============================================================
# 3D Grad-CAM++ + Deletion & Insertion Curves (T4 vs Non-T4)
# ============================================================
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
import csv

# ---------------------------
# CONFIG
# ---------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "resnet18"          # "resnet18" / "cnn5"
CLASS_INDEX  = 1                    # positive class (T4)
INPUT_SHAPE  = (32, 96, 96)
STEPS        = 11                   # 0..100% (inclusive)
SAVE_DIR     = "./explainability_curves"
SHOW_INDIVIDUAL = True              # per-case curve PNGs
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# IO / UTILS
# ---------------------------
def load_nrrd(path):
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    return vol

def resize3d(vol, out_shape):
    f = (out_shape[0]/vol.shape[0], out_shape[1]/vol.shape[1], out_shape[2]/vol.shape[2])
    return zoom(vol, f, order=1)

def to_tensor3d(vol):
    return torch.from_numpy(vol[None, None]).float()

# ---------------------------
# Model
# ---------------------------
def build_model(name):
    if name == "resnet18":
        from monai.networks.nets import resnet18
        return resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
    elif name == "cnn5":
        return nn.Sequential(
            nn.Conv3d(1,16,3,1,1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,1,1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,1,1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64,128,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
            nn.Flatten(), nn.Linear(128,2)
        )
    else:
        raise ValueError("Unknown model")

def get_target_layer(model, name):
    if "resnet" in name: return model.layer4[-1].conv2
    if "cnn5"   in name: return list(model.children())[6]
    raise ValueError("Target layer undefined")

# ---------------------------
# Grad-CAM++ (3D)
# ---------------------------
class GradCAMpp3D:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations, self.gradients = None, None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, i, o): self.activations = o.detach()
    def _bwd_hook(self, m, gi, go): self.gradients = go[0].detach()

    def generate(self, x, class_idx=1):
        logits = self.model(x)
        score  = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        A, dY = self.activations, self.gradients
        eps   = 1e-8
        dY2   = dY**2
        dY3   = F.relu(dY) + eps
        alpha = dY2 / (2*dY2 + (A*dY3).sum(dim=(2,3,4), keepdim=True) + eps)
        weights = (alpha * F.relu(dY)).sum(dim=(2,3,4), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0,0].detach().cpu().numpy()

        # stabilize visualization; does not affect masking indices (we use ranks)
        if cam.max() > 0:
            cam[cam < cam.max()*0.2] = 0
            vmax = np.percentile(cam, 99)
            cam  = np.clip(cam / (vmax + 1e-8), 0, 1)

        probs = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
        return cam, probs

# ---------------------------
# Deletion / Insertion
# ---------------------------
def zscore(x, mu, sd):
    return (x - mu) / (sd + 1e-8)

def compute_deletion_insertion(vol_raw, cam, model, class_idx=1, steps=11):
    """
    vol_raw  : resized raw volume (not normalized)
    cam      : CAM in [0,1]
    model    : torch model
    returns  : pct, del_probs, ins_probs, p0, pins_base
    """
    # FIX: use ORIGINAL baseline mean/std for ALL masked variants
    mu0 = float(vol_raw.mean())
    sd0 = float(vol_raw.std() + 1e-8)

    # sort indices by importance (descending)
    order = np.argsort(cam.flatten())[::-1]
    N = cam.size

    # baseline full-input prob
    x0 = to_tensor3d(zscore(vol_raw, mu0, sd0)).to(DEVICE)
    with torch.no_grad():
        p0 = torch.softmax(model(x0), dim=1)[0, class_idx].item()

    # insertion baseline: canvas filled with mu0 (mean-intensity image)
    canvas = np.full_like(vol_raw, mu0, dtype=np.float32)
    xc = to_tensor3d(zscore(canvas, mu0, sd0)).to(DEVICE)
    with torch.no_grad():
        pins_base = torch.softmax(model(xc), dim=1)[0, class_idx].item()

    del_probs, ins_probs = [], []
    # Percentages to evaluate
    ks = (np.linspace(0, 1, steps) * N).astype(int)

    for k in ks:
        # --- Deletion: replace top-k voxels with mean intensity (mu0), keep others unchanged
        vol_del = vol_raw.copy().reshape(-1)
        if k > 0:
            vol_del[order[:k]] = mu0
        vol_del = vol_del.reshape(vol_raw.shape)
        x_del = to_tensor3d(zscore(vol_del, mu0, sd0)).to(DEVICE)
        with torch.no_grad():
            pd = torch.softmax(model(x_del), dim=1)[0, class_idx].item()
        del_probs.append(pd)

        # --- Insertion: start with mean canvas, insert top-k voxels from original
        vol_ins = canvas.copy().reshape(-1)
        if k > 0:
            # insert original voxel values at top-k important indices
            vol_ins[order[:k]] = vol_raw.reshape(-1)[order[:k]]
        vol_ins = vol_ins.reshape(vol_raw.shape)
        x_ins = to_tensor3d(zscore(vol_ins, mu0, sd0)).to(DEVICE)
        with torch.no_grad():
            pi = torch.softmax(model(x_ins), dim=1)[0, class_idx].item()
        ins_probs.append(pi)

    pct = np.linspace(0, 100, steps)
    return pct, np.array(del_probs), np.array(ins_probs), p0, pins_base

def auc_trapz(x_pct, y):
    """Area under curve on [0,1] scale (useful for comparison)."""
    x = x_pct / 100.0
    return float(np.trapz(y, x))

# ---------------------------
# Load model & cases
# ---------------------------
model = build_model(MODEL_NAME).to(DEVICE)
ckpt_path = "path/to/resnet18_T4_dir/fold_2/best_model.pth"
state = torch.load(ckpt_path, map_location=DEVICE)
state = state.get("model_state", state.get("state_dict", state))
model.load_state_dict(state, strict=False)
model.eval()
print("✅ Model loaded:", ckpt_path)

cases = [
    ("Case001_T4",     "path/to/HN_CHUS_034_Cropped_Volume.nrrd", "T4"),
    ("Case002_T4",     "path/to/QIN_HEADNECK_02_2502_Cropped_Volume.nrrd", "T4"),
    ("Case003_T4",     "path/to/RADCURE_1726_Cropped_Volume.nrrd", "T4"),
    ("Case004_non_T4", "path/to/RADCURE_2362_Cropped_Volume.nrrd", "non-T4"),
    ("Case005_non_T4", "path/to/QIN_HEADNECK_01_0048_Cropped_Volume.nrrd", "non-T4"),
    ("Case006_non_T4", "path/to/RADCURE_0039_Cropped_Volume.nrrd", "non-T4"),
]

# ---------------------------
# Grad-CAM++ engine (single instance)
# ---------------------------
target = get_target_layer(model, MODEL_NAME)
campp  = GradCAMpp3D(model, target)

# ---------------------------
# Run over cases
# ---------------------------
records = []
group_curves_del = {"T4": [], "non-T4": []}
group_curves_ins = {"T4": [], "non-T4": []}
pct_axis = None

for cid, img_path, grp in tqdm(cases):
    vol = load_nrrd(img_path)
    if vol.shape != INPUT_SHAPE:
        vol = resize3d(vol, INPUT_SHAPE)

    # CAM from original z-scored input (needed for gradient flow)
    x_for_cam = to_tensor3d((vol - vol.mean()) / (vol.std() + 1e-8)).to(DEVICE)
    cam, _ = campp.generate(x_for_cam, CLASS_INDEX)

    # Compute deletion/insertion using baseline μ0/σ0 constants (no renorm drift)
    pct, del_probs, ins_probs, p0, pins_base = compute_deletion_insertion(
        vol_raw=vol, cam=cam, model=model, class_idx=CLASS_INDEX, steps=STEPS
    )
    if pct_axis is None: pct_axis = pct

    # Deltas for plotting/report
    del_delta = p0 - del_probs         # larger = stronger causal effect
    ins_delta = ins_probs - pins_base  # larger = stronger causal effect

    # AUCs
    del_auc_raw   = auc_trapz(pct, del_probs)      # lower better (fast drop)
    del_auc_delta = auc_trapz(pct, del_delta)      # higher better (fast drop)
    ins_auc_raw   = auc_trapz(pct, ins_probs)      # higher better (fast rise)
    ins_auc_delta = auc_trapz(pct, ins_delta)      # higher better

    records.append({
        "case": cid, "group": grp, "p0": p0, "p_insertion_base": pins_base,
        "del_auc_raw": del_auc_raw, "del_auc_delta": del_auc_delta,
        "ins_auc_raw": ins_auc_raw, "ins_auc_delta": ins_auc_delta
    })

    group_curves_del[grp].append(del_delta)   # store deltas for group mean/SD
    group_curves_ins[grp].append(ins_delta)

    # Optional per-case plots
    if SHOW_INDIVIDUAL:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(pct, del_delta, 'o-', label=f"{cid} (Δ)")
        plt.xlabel("Top-CAM voxels removed (%)"); plt.ylabel("Δ Prob (p0 − p)")
        plt.title("Deletion curve (delta)"); plt.grid(True, alpha=.3); plt.legend(fontsize=8)

        plt.subplot(1,2,2)
        plt.plot(pct, ins_delta, 'o-', label=f"{cid} (Δ)")
        plt.xlabel("Top-CAM voxels inserted (%)"); plt.ylabel("Δ Prob (p − p_base)")
        plt.title("Insertion curve (delta)"); plt.grid(True, alpha=.3); plt.legend(fontsize=8)

        plt.suptitle(cid, fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(os.path.join(SAVE_DIR, f"{cid}_DelIns_Delta.png"), dpi=300)
        plt.close()

# ---------------------------
# Group plots (mean ± SD of DELTA)
# ---------------------------
def plot_group_delta(pct, curves_dict, title, ylabel, fname):
    plt.figure(figsize=(7.5,5.5))

    # per-case faint lines
    for grp, arrs in curves_dict.items():
        for arr in arrs:
            plt.plot(pct, arr, alpha=0.3, lw=1)

    # group mean ± SD
    colors = {"T4":"red", "non-T4":"blue"}
    for grp, arrs in curves_dict.items():
        A = np.vstack(arrs) if len(arrs) else np.zeros((1,len(pct)))
        mean, sd = A.mean(0), A.std(0)
        plt.plot(pct, mean, color=colors[grp], lw=3, label=f"{grp} mean±SD")
        plt.fill_between(pct, mean-sd, mean+sd, color=colors[grp], alpha=0.15)

    plt.xlabel("Top-CAM voxels (%)")
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=400)
    plt.close()

plot_group_delta(
    pct_axis, group_curves_del,
    title="Deletion Curves (Δ probability) — T4 vs Non-T4",
    ylabel="Δ Prob (p0 − p)",
    fname="Deletion_T4_vs_NonT4_Delta.png"
)

plot_group_delta(
    pct_axis, group_curves_ins,
    title="Insertion Curves (Δ probability) — T4 vs Non-T4",
    ylabel="Δ Prob (p − p_base)",
    fname="Insertion_T4_vs_NonT4_Delta.png"
)

# ---------------------------
# Save metrics CSV
# ---------------------------
csv_path = os.path.join(SAVE_DIR, "DeletionInsertion_AUCs.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader()
    for r in records: w.writerow(r)

print("✅ Saved:")
print(" - Group plots:", 
      "Deletion_T4_vs_NonT4_Delta.png,", 
      "Insertion_T4_vs_NonT4_Delta.png in", SAVE_DIR)
print(" - Per-case curves:", "one PNG per case" if SHOW_INDIVIDUAL else "skipped")
print(" - Metrics CSV:", csv_path)