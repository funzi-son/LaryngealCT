# ============================================================
# CAM + Mask Overlap Metrics
# ============================================================
import os, numpy as np, torch
import SimpleITK as sitk
import pandas as pd
from scipy.ndimage import zoom

# ---- helper functions ----
def load_nrrd(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)

def resize3d(vol, out_shape):
    factors = [out_shape[0]/vol.shape[0],
               out_shape[1]/vol.shape[1],
               out_shape[2]/vol.shape[2]]
    return zoom(vol, factors, order=1)

def norm01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def compute_overlap_metrics(cam, mask, thr):
    cam_bin = (cam > thr).astype(np.float32)
    intersection = np.sum(cam_bin * mask)
    cmof = intersection / (np.sum(cam_bin) + 1e-8)
    mcr  = intersection / (np.sum(mask) + 1e-8)
    dice = (2*intersection) / (np.sum(cam_bin) + np.sum(mask) + 1e-8)
    inside_mean = np.mean(cam[mask==1]) if np.sum(mask)>0 else 0
    outside_mean = np.mean(cam[mask==0])
    enrich = inside_mean / (outside_mean + 1e-8)
    return cmof, mcr, dice, enrich

# ---- paths ----
cases = [
    ("Case001_T4", "path/to/gradcampp_outputs_with_deletion\Case001_T4_cam.npy", "path/to/HN_CHUS_034.seg.nrrd"),
    ("Case002_T4", "path/to/gradcampp_outputs_with_deletion\Case002_T4_cam.npy", "path/to/QIN_HEADNECK_02_2502.seg.nrrd"),
    ("Case003_T4", "path/to/gradcampp_outputs_with_deletion\Case003_T4_cam.npy", "path/to/RADCURE_1726.seg.nrrd"),
    ("Case004_non_T4", "path/to/gradcampp_outputs_with_deletion\Case004_non_T4_cam.npy", "path/to/RADCURE_2362.seg.nrrd"),
    ("Case005_non_T4", "path/to/gradcampp_outputs_with_deletion\Case005_non_T4_cam.npy",  "path/to/QIN_HEADNECK_01_0048.seg.nrrd"),
    ("Case006_non_T4", "path/to/gradcampp_outputs_with_deletion\Case006_non_T4_cam.npy",  "path/to/RADCURE_0039.seg.nrrd"),
]


# ---- evaluation ----
results = []
for cid, cam_path, mask_path in cases:
    cam = np.load(cam_path)
    mask = (load_nrrd(mask_path) > 0).astype(np.float32)
    if cam.shape != mask.shape:
        mask = resize3d(mask, cam.shape)
    cam = norm01(cam)

    for thr in [0.3, 0.5, 0.7]:
        cmof, mcr, dice, enrich = compute_overlap_metrics(cam, mask, thr)
        results.append({
            "CaseID": cid, "Threshold": thr,
            "CAM_Mask_OverlapFraction": cmof,
            "Mask_CoverageRatio": mcr,
            "Dice_CAM_Mask": dice,
            "Enrichment": enrich
        })

df = pd.DataFrame(results)
out_path = "./CAM_Mask_Overlap_Metrics.csv"
df.to_csv(out_path, index=False)
print(f"✅ Results saved to: {out_path}")
df


##########
import pandas as pd

# === CONFIG ===
csv_path = "path/to/explainability_curves/DeletionInsertion_AUCs.csv"

# === LOAD FILE ===
df = pd.read_csv(csv_path)
print("✅ File loaded. Columns:", list(df.columns))
print(df.head())

# === BASIC SUMMARY ===
mean_auc_del = df["del_auc_raw"].mean()
std_auc_del = df["del_auc_raw"].std()
mean_auc_ins = df["ins_auc_raw"].mean()
std_auc_ins = df["ins_auc_raw"].std()

print("\n🔹 Overall Results:")
print(f"Deletion AUC (mean ± SD): {mean_auc_del:.3f} ± {std_auc_del:.3f}")
print(f"Insertion AUC (mean ± SD): {mean_auc_ins:.3f} ± {std_auc_ins:.3f}")

# === GROUP-LEVEL SUMMARY (T4 vs non-T4) ===
if "group" in df.columns:
    group_summary = (
        df.groupby("group")[["del_auc_raw", "ins_auc_raw"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\n🔹 Group-level mean ± SD:")
    print(group_summary)
else:
    print("\n⚠️ Column 'group' not found — skipping group analysis.")
	
	
#====================================================================
#Calculate group-level summary metrics for AUC deletion and insertion
#====================================================================

import pandas as pd

# === CONFIG ===
csv_path = "path/to/mask_deletion_insertion_outputs/Mask_DelIns_AUCs.csv"

# === LOAD FILE ===
df = pd.read_csv(csv_path)
print("✅ File loaded. Columns:", list(df.columns))
print(df.head())

# === BASIC SUMMARY ===
mean_auc_del = df["AUC_Deletion"].mean()
std_auc_del = df["AUC_Deletion"].std()
mean_auc_ins = df["AUC_Insertion"].mean()
std_auc_ins = df["AUC_Insertion"].std()

print("\n🔹 Overall Results:")
print(f"Deletion AUC (mean ± SD): {mean_auc_del:.3f} ± {std_auc_del:.3f}")
print(f"Insertion AUC (mean ± SD): {mean_auc_ins:.3f} ± {std_auc_ins:.3f}")

# === GROUP-LEVEL SUMMARY (T4 vs non-T4) ===
if "group" in df.columns:
    group_summary = (
        df.groupby("group")[["AUC_Deletion", "AUC_Insertion"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\n🔹 Group-level mean ± SD:")
    print(group_summary)
else:
    print("\n⚠️ Column 'group' not found — skipping group analysis.")


#==================================================================================
#Calculate group-level summary metrics for CAM-Mask overlap fraction and enrichment
#===================================================================================

import pandas as pd

# Load your file
df = pd.read_csv(r"C:\Users\clarynse\CAM_Mask_Overlap_Metrics.csv")

# --- FIX: ensure CaseID is treated as string ---
df["CaseID"] = df["CaseID"].astype(str)

# Derive group labels safely
df["group"] = df["CaseID"].apply(lambda x: "T4" if "T4" in x else "non-T4")

# Compute per-case means across thresholds (0.3, 0.5, 0.7)
case_means = df.groupby(["CaseID", "group"]).mean(numeric_only=True).reset_index()

# Compute group-level statistics
summary = case_means.groupby("group")[["CAM_Mask_OverlapFraction", "Enrichment"]].agg(['mean','std'])
print(summary.round(4))