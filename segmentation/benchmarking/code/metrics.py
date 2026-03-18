#Metrics calculation for all models

!pip install -q surface-distance SimpleITK openpyxl

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

from surface_distance.metrics import (
    compute_surface_distances,
    compute_average_surface_distance,
    compute_robust_hausdorff,
    compute_surface_dice_at_tolerance,
)

# ---------------------------
# PATHS
# ---------------------------
pred_dir_ModelName_SeedB = r"path\to\SeedB_predicted_labels"
pred_dir_ModelName_han = r"path\to\HanSeg_predicted_labels"

gt_dir_SeedB = r"path\to\SeedB_gt"
gt_dir_han   = r"path\to\HanSeg_gt"

out_root =  r"path\to\metrics_folder"
os.makedirs(out_root, exist_ok=True)

out_csv_SeedB  = os.path.join(out_root, "metrics_ModelName_SeedB.csv")
out_xlsx_SeedB = os.path.join(out_root, "metrics_ModelName_SeedB.xlsx")

out_csv_han  = os.path.join(out_root, "metrics_ModelName_HanSeg.csv")
out_xlsx_han = os.path.join(out_root, "metrics_ModelName_HanSeg.xlsx")

# ---------------------------
# Helpers
# ---------------------------
def load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing_xyz = img.GetSpacing()     # (x, y, z)
    spacing_zyx = spacing_xyz[::-1]    # (z, y, x) for array order
    return arr, np.array(spacing_zyx, dtype=np.float32)

def dice_coefficient(gt, pred):
    gt = gt.astype(bool); pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    return 1.0 if denom == 0 else (2.0 * inter / denom)

def iou_score(gt, pred):
    gt = gt.astype(bool); pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return 1.0 if union == 0 else (inter / union)

def sensitivity_precision(gt, pred):
    gt = gt.astype(bool); pred = pred.astype(bool)
    tp = np.logical_and(gt, pred).sum()
    fn = np.logical_and(gt, np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(gt), pred).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return float(sens), float(prec)

def compute_volumes(gt, pred, spacing_zyx):
    voxel_vol_mm3 = float(np.prod(spacing_zyx))
    gt_vol_mm3   = gt.astype(bool).sum()   * voxel_vol_mm3
    pred_vol_mm3 = pred.astype(bool).sum() * voxel_vol_mm3
    diff_mm3     = pred_vol_mm3 - gt_vol_mm3
    abs_diff_mm3 = abs(diff_mm3)
    return (
        gt_vol_mm3, pred_vol_mm3, diff_mm3, abs_diff_mm3,
        gt_vol_mm3/1000.0, pred_vol_mm3/1000.0, diff_mm3/1000.0, abs_diff_mm3/1000.0
    )

def compute_hd95_assd_surfdice(gt, pred, spacing_zyx, tol_mm=(1.0, 2.0)):
    gt = gt.astype(bool); pred = pred.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 0.0, 0.0, 1.0, 1.0
    if gt.sum() == 0 or pred.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    sd = compute_surface_distances(gt, pred, spacing_zyx)
    assd = float(np.mean(compute_average_surface_distance(sd)))
    hd95 = float(compute_robust_hausdorff(sd, 95))
    sd1 = float(compute_surface_dice_at_tolerance(sd, tol_mm[0]))
    sd2 = float(compute_surface_dice_at_tolerance(sd, tol_mm[1]))
    return hd95, assd, sd1, sd2

def list_pred_files(pred_dir):
    return sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

def strip_nii_ext(fname):
    return os.path.splitext(os.path.splitext(fname)[0])[0]

def find_gt_path(gt_dir, case_id):
    p1 = os.path.join(gt_dir, case_id + ".nii.gz")
    if os.path.exists(p1): return p1
    p2 = os.path.join(gt_dir, case_id + ".nii")
    if os.path.exists(p2): return p2
    return None

def evaluate_folder(pred_dir, gt_dir, out_csv, out_xlsx, tag=""):
    results = []
    pred_files = list_pred_files(pred_dir)
    print(f"\n[{tag}] Found {len(pred_files)} prediction files in: {pred_dir}")

    for f in pred_files:
        case_id = strip_nii_ext(f)
        pred_path = os.path.join(pred_dir, f)

        gt_path = find_gt_path(gt_dir, case_id)
        if gt_path is None:
            print(f"[WARN] No GT found for {case_id}, skipping")
            continue

        pred_arr, _ = load_nifti(pred_path)
        gt_arr, spacing_gt = load_nifti(gt_path)

        pred_bin = pred_arr > 0
        gt_bin   = gt_arr > 0

        if pred_bin.shape != gt_bin.shape:
            print(f"[WARN] Shape mismatch for {case_id}: pred {pred_bin.shape}, gt {gt_bin.shape} -> skipping")
            continue

        spacing = spacing_gt  # use GT spacing

        dice = dice_coefficient(gt_bin, pred_bin)
        iou  = iou_score(gt_bin, pred_bin)
        sens, prec = sensitivity_precision(gt_bin, pred_bin)
        hd95, assd, sd1, sd2 = compute_hd95_assd_surfdice(gt_bin, pred_bin, spacing, tol_mm=(1.0, 2.0))

        (gt_vol_mm3, pred_vol_mm3, diff_mm3, abs_diff_mm3,
         gt_vol_ml, pred_vol_ml, diff_ml, abs_diff_ml) = compute_volumes(gt_bin, pred_bin, spacing)

        results.append({
            "case_id": case_id,
            "Dice": dice,
            "IoU": iou,
            "HD95_mm": hd95,
            "ASSD_mm": assd,
            "SurfDice_1mm": sd1,
            "SurfDice_2mm": sd2,
            "Sensitivity": sens,
            "Precision": prec,
            "GT_vol_mL": gt_vol_ml,
            "Pred_vol_mL": pred_vol_ml,
            "VolDiff_mL": diff_ml,
            "AbsVolDiff_mL": abs_diff_ml,
        })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)

    print(f"\n[{tag}] Saved:")
    print(" -", out_csv)
    print(" -", out_xlsx)

    if len(df) == 0:
        print(f"[{tag}] No cases evaluated (check filenames & GT paths).")
        return df

    cols = ["Dice", "IoU", "HD95_mm", "ASSD_mm", "SurfDice_1mm", "SurfDice_2mm",
            "Sensitivity", "Precision", "VolDiff_mL", "AbsVolDiff_mL"]
    print(f"\n[{tag}] Mean:\n", df[cols].mean(numeric_only=True))
    print(f"\n[{tag}] Std:\n", df[cols].std(numeric_only=True))
    return df

df_SeedB = evaluate_folder(pred_dir_ModelName_SeedB, gt_dir_SeedB, out_csv_SeedB, out_xlsx_SeedB, tag="SeedB")
df_han   = evaluate_folder(pred_dir_ModelName_han,   gt_dir_han,   out_csv_han,   out_xlsx_han,   tag="HanSeg")