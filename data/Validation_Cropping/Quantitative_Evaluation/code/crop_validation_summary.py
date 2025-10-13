# ==========================================================
# Quantitative validation of parameter-search crops vs manual crops
# Using bbox3d.txt 
# ==========================================================

#Install dependencies

!pip install SimpleITK pandas scikit-image matplotlib

try:
    import SimpleITK as sitk
    import pandas as pd
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "SimpleITK", "pandas", "matplotlib"])
    import SimpleITK as sitk
    import pandas as pd

import os, re, json, math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
BBOX_FILE  = "path/to/bbox3d.txt"  
ORIG_DIR   = "path/to/Images_NRRD"     #path to your raw nrrd images
MANUAL_DIR = "path/to/Cropped_Volumes"
OUTPUT_DIR = "path/to/your/output/dir"

os.makedirs(OUTPUT_DIR, exist_ok=True)
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays")
os.makedirs(OVERLAY_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def read_bbox_file(path):
    """Read bbox3d.txt formatted: case x y z dx dy dz"""
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 7:
                continue
            case = parts[0]
            try:
                x,y,z,dx,dy,dz = map(int, parts[1:7])
                d[case] = (x,y,z,dx,dy,dz)
            except Exception as e:
                print(f"⚠️ Skipping line: {line} ({e})")
    return d

def resample_isotropic(img, spacing=(1.0,1.0,1.0), interp=sitk.sitkLinear):
    """Resample to 1mm iso spacing."""
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [int(round(osz*ospc/nspc))
                for osz, ospc, nspc in zip(original_size, original_spacing, spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(img)

def crop(img, x,y,z,dx,dy,dz):
    return sitk.RegionOfInterest(img, [dx,dy,dz], [x,y,z])

def sitk_to_np(img):
    return sitk.GetArrayFromImage(img).astype(np.float32)

def normalize_z(a):
    return (a - a.mean())/(a.std() + 1e-8)

def dice_iou(a,b):
    ma, mb = (a!=0), (b!=0)
    inter = (ma & mb).sum()
    va, vb = ma.sum(), mb.sum()
    dice = (2*inter)/(va+vb) if (va+vb)>0 else np.nan
    iou = inter/(va+vb-inter) if (va+vb-inter)>0 else np.nan
    return float(dice), float(iou), int(va), int(vb), int(inter)

def save_overlay(case, A, M, outdir):
    k = A.shape[0]//2
    def norm(x):
        lo,hi = np.percentile(x,1), np.percentile(x,99)
        if hi<=lo: hi=lo+1
        return np.clip((x-lo)/(hi-lo),0,1)
    a,m = norm(A[k]), norm(M[k])
    overlay = 0.6*a + 0.4*m
    fig,axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(a,cmap="gray"); axes[0].set_title("AUTO"); axes[0].axis("off")
    axes[1].imshow(m,cmap="gray"); axes[1].set_title("MANUAL"); axes[1].axis("off")
    axes[2].imshow(overlay,cmap="gray"); axes[2].set_title("Overlay"); axes[2].axis("off")
    fig.suptitle(case); fig.tight_layout()
    path = os.path.join(outdir, f"{case}_overlay.png")
    plt.savefig(path,dpi=200); plt.close(fig)

# -----------------------
# Main loop
# -----------------------
bbox = read_bbox_file(BBOX_FILE)
records, fails = [], []

for case,(x,y,z,dx,dy,dz) in bbox.items():
    orig_path  = os.path.join(ORIG_DIR, f"{case}.nrrd")
    man_path   = os.path.join(MANUAL_DIR, f"{case}_Cropped_Volume.nrrd")

    if not os.path.exists(orig_path) or not os.path.exists(man_path):
        fails.append((case,"missing_file",{"orig":os.path.exists(orig_path),"manual":os.path.exists(man_path)}))
        continue

    try:
        orig = sitk.ReadImage(orig_path)
        man  = sitk.ReadImage(man_path)
        orig_iso = resample_isotropic(orig)

        # --- Clamping fix ---
        sizeX, sizeY, sizeZ = orig_iso.GetSize()
        if x+dx > sizeX or y+dy > sizeY or z+dz > sizeZ:
            fails.append((case,"clamped_bbox",{"orig_size":(sizeX,sizeY,sizeZ),
                                               "requested":(x,y,z,dx,dy,dz)}))
        dx = min(dx, sizeX - x)
        dy = min(dy, sizeY - y)
        dz = min(dz, sizeZ - z)
        x = max(0, min(x, sizeX-1))
        y = max(0, min(y, sizeY-1))
        z = max(0, min(z, sizeZ-1))
        # --------------------

        auto = crop(orig_iso,x,y,z,dx,dy,dz)

        A,M = sitk_to_np(auto), sitk_to_np(man)
        if A.shape != M.shape:
            fails.append((case,"shape_mismatch",(A.shape,M.shape)))
            continue

        dice,iou,va,vb,inter = dice_iou(A,M)
        A_z,M_z = normalize_z(A), normalize_z(M)
        mse = float(np.mean((A_z-M_z)**2))
        mae = float(np.mean(np.abs(A_z-M_z)))

        records.append(dict(case=case,dice=dice,iou=iou,mse=mse,mae=mae,
                            voxels_manual=vb,voxels_auto=va,voxels_inter=inter))
        save_overlay(case,A,M,OVERLAY_DIR)

    except Exception as e:
        fails.append((case,"exception",str(e)))

# -----------------------
# Save results
# -----------------------
df = pd.DataFrame(records).sort_values("case")
csv_path = os.path.join(OUTPUT_DIR,"crop_validation_metrics.csv")
df.to_csv(csv_path,index=False)

# summary stats
summary={}
for col in ["dice","iou","mse","mae"]:
    vals=df[col].dropna().values
    if len(vals)==0: continue
    summary[col]={ "n":int(len(vals)),
                   "mean":float(np.mean(vals)),
                   "sd":float(np.std(vals,ddof=1)) if len(vals)>1 else 0,
                   "median":float(np.median(vals)),
                   "q1":float(np.percentile(vals,25)),
                   "q3":float(np.percentile(vals,75)),
                   "min":float(np.min(vals)),
                   "max":float(np.max(vals)) }
with open(os.path.join(OUTPUT_DIR,"crop_validation_summary.json"),"w") as f:
    json.dump(summary,f,indent=2)

# plots
for col in ["dice","iou","mse","mae"]:
    vals=df[col].dropna().values
    if len(vals)==0: continue
    plt.figure(figsize=(6,4)); plt.hist(vals,bins=30); plt.title(col); plt.savefig(os.path.join(OUTPUT_DIR,f"hist_{col}.png")); plt.close()
    plt.figure(figsize=(4,5)); plt.boxplot(vals,vert=True); plt.title(col); plt.savefig(os.path.join(OUTPUT_DIR,f"box_{col}.png")); plt.close()

if fails:
    with open(os.path.join(OUTPUT_DIR,"failures.log"),"w") as f:
        for item in fails: f.write(repr(item)+"\n")

print("===============================================")
print(f"Processed {len(records)} successful, {len(fails)} logged (see failures.log)")
print(f"CSV: {csv_path}")
print(f"Summary JSON + plots in {OUTPUT_DIR}")
print(f"Overlay images in {OVERLAY_DIR}")
print("===============================================")