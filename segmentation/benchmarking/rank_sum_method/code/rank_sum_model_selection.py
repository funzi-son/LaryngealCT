# =======================================
# Model Selection using Rank-sum method
# =======================================

# Calculate ranks based on overlap, boundary and outliers

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# EDIT THESE PATHS (local Windows paths)
# ============================================================
FILES = {
    "SeedB": {
        "Baseline":   r"path\to\metrics_baseline_nnUNetEnsemble_SeedB.xlsx",
        "Boundary":   r"path\to\metrics_BoundarynnUNetEnsemble_SeedB.xlsx",
        "BC_nnUNet":  r"path\to\metrics_BCnnUNetEnsemble_SeedB.xlsx",
    },
    "HanSeg": {
        "Baseline":   r"path\to\metrics_baseline_nnUNetEnsemble_HanSeg.xlsx",
        "Boundary":   r"path\to\metrics_BoundarynnUNetEnsemble_HanSeg.xlsx",
        "BC_nnUNet":  r"path\to\metrics_BCnnUNetEnsemble_HanSeg.xlsx",
    }
}

# ============================================================
# Metrics to summarize
#   - Higher is better for: Dice/IoU/SurfDice/Precision/Sensitivity
#   - Lower is better for: HD95/ASSD/AbsVolDiff
# ============================================================
HIGHER_BETTER = ["Dice", "IoU", "SurfDice_1mm", "SurfDice_2mm", "Precision", "Sensitivity"]
LOWER_BETTER  = ["HD95_mm", "ASSD_mm", "AbsVolDiff_mL", "AbsVolDiff_mm3"]


BASE_METRICS = ["Dice", "IoU", "HD95_mm", "ASSD_mm", "SurfDice_1mm", "SurfDice_2mm",
                "Precision", "Sensitivity", "AbsVolDiff_mL", "AbsVolDiff_mm3"]

# outlier rules (edit thresholds if needed)
OUTLIER_RULES = {
    "HD95_mm": 10.0,    # count cases with HD95 > 10 mm
    "ASSD_mm": 1.0,     # count cases with ASSD > 1 mm
}

# ============================================================
# Helpers
# ============================================================
def describe_metric(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {"mean": np.nan, "sd": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    return {
        "mean":   float(x.mean()),
        "sd":     float(x.std(ddof=1)),
        "median": float(x.median()),
        "p05":    float(np.percentile(x, 5)),
        "p95":    float(np.percentile(x, 95)),
        "min":    float(x.min()),
        "max":    float(x.max()),
        "n":      int(x.shape[0]),
    }

def fmt_mean_sd(m, s):
    if np.isnan(m) or np.isnan(s):
        return "NA"
    return f"{m:.4f} ± {s:.4f}"

def fmt_range(a, b):
    if np.isnan(a) or np.isnan(b):
        return "NA"
    return f"{a:.4f}–{b:.4f}"

def load_model_df(path):
    df = pd.read_excel(path)
    # ensure case_id exists
    if "case_id" not in df.columns:
        raise ValueError(f"'case_id' column not found in {path}")
    return df

def get_existing_metrics(df):
    return [m for m in BASE_METRICS if m in df.columns]

# ============================================================
# Core analysis per dataset
# ============================================================
def analyze_dataset(dataset_name, model_paths):
    # Load all models
    dfs = {model: load_model_df(path) for model, path in model_paths.items()}

    # Use intersection of case_ids across models (safest)
    common_cases = set(dfs[list(dfs.keys())[0]]["case_id"].astype(str))
    for model in dfs:
        common_cases &= set(dfs[model]["case_id"].astype(str))
    common_cases = sorted(list(common_cases))

    if len(common_cases) == 0:
        raise ValueError(f"No common case_id across models for {dataset_name}")

    # Filter and sort identically
    for model in dfs:
        dfs[model]["case_id"] = dfs[model]["case_id"].astype(str)
        dfs[model] = dfs[model].loc[dfs[model]["case_id"].isin(common_cases)].copy()
        dfs[model] = dfs[model].sort_values("case_id").reset_index(drop=True)

    # Decide metrics present in all models
    metrics_all = set(get_existing_metrics(dfs[list(dfs.keys())[0]]))
    for model in dfs:
        metrics_all &= set(get_existing_metrics(dfs[model]))
    metrics_all = [m for m in BASE_METRICS if m in metrics_all]  # keep nice order

    # ------------------------------
    # (A) Summary table: mean±sd, median, p05/p95, min/max
    # ------------------------------
    summary_rows = []
    for model, df in dfs.items():
        for m in metrics_all:
            stats = describe_metric(df[m])
            summary_rows.append({
                "Dataset": dataset_name,
                "Model": model,
                "Metric": m,
                "N": stats["n"],
                "Mean±SD": fmt_mean_sd(stats["mean"], stats["sd"]),
                "Median": stats["median"],
                "P05": stats["p05"],
                "P95": stats["p95"],
                "Min": stats["min"],
                "Max": stats["max"],
            })
    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------
    # (B) Outlier counts
    # ------------------------------
    outlier_rows = []
    for model, df in dfs.items():
        for m, thr in OUTLIER_RULES.items():
            if m not in metrics_all:
                continue
            x = pd.to_numeric(df[m], errors="coerce")
            outlier_rows.append({
                "Dataset": dataset_name,
                "Model": model,
                "Metric": m,
                "Threshold": thr,
                "OutlierCount": int((x > thr).sum())
            })
    outlier_df = pd.DataFrame(outlier_rows)

    # ------------------------------
    # (C) Wins per case (metric-wise)
    # ------------------------------
    # Create a combined case table
    combined = pd.DataFrame({"case_id": dfs[list(dfs.keys())[0]]["case_id"].astype(str)})
    for model, df in dfs.items():
        for m in metrics_all:
            combined[f"{m}__{model}"] = pd.to_numeric(df[m], errors="coerce")

    win_rows = []
    for m in metrics_all:
        cols = [f"{m}__{model}" for model in dfs.keys()]
        mat = combined[cols]

        if m in HIGHER_BETTER:
            winners = mat.idxmax(axis=1, skipna=True)
        elif m in LOWER_BETTER:
            winners = mat.idxmin(axis=1, skipna=True)
        else:
            # default: higher is better
            winners = mat.idxmax(axis=1, skipna=True)

        # map "metric__Model" -> Model
        winners_model = winners.str.split("__").str[-1]
        counts = winners_model.value_counts(dropna=True).to_dict()

        for model in dfs.keys():
            win_rows.append({
                "Dataset": dataset_name,
                "Metric": m,
                "Model": model,
                "WinCount": int(counts.get(model, 0))
            })
    wins_metric_df = pd.DataFrame(win_rows)

    # ------------------------------
    # (D) Overall wins per case (rank-sum composite)
    # ------------------------------
    # For each case, rank models per metric (best rank=1), then sum ranks across metrics.
    # This avoids arbitrary weighting and is very reviewer-defensible.
    rank_sum = pd.DataFrame({"case_id": combined["case_id"].copy()})
    for model in dfs.keys():
        rank_sum[model] = 0.0

    for m in metrics_all:
        cols = [f"{m}__{model}" for model in dfs.keys()]
        mat = combined[cols].copy()

        # rank across models for each case
        if m in HIGHER_BETTER:
            ranks = mat.rank(axis=1, method="average", ascending=False)  # higher=better -> descending
        else:
            ranks = mat.rank(axis=1, method="average", ascending=True)   # lower=better -> ascending

        # add ranks to model totals
        for model in dfs.keys():
            rank_sum[model] += ranks[f"{m}__{model}"]

    # winner is model with smallest rank-sum
    rank_sum["OverallWinner"] = rank_sum[list(dfs.keys())].idxmin(axis=1)
    overall_counts = rank_sum["OverallWinner"].value_counts().to_dict()

    overall_wins_df = pd.DataFrame([
        {"Dataset": dataset_name, "Model": model, "OverallWinCount": int(overall_counts.get(model, 0))}
        for model in dfs.keys()
    ])

    return summary_df, outlier_df, wins_metric_df, overall_wins_df, rank_sum


# ============================================================
# Run all
# ============================================================
all_summary = []
all_outliers = []
all_wins_metric = []
all_overall_wins = []
rank_sum_tables = {}

for ds, paths in FILES.items():
    summary_df, outlier_df, wins_metric_df, overall_wins_df, rank_sum = analyze_dataset(ds, paths)
    all_summary.append(summary_df)
    all_outliers.append(outlier_df)
    all_wins_metric.append(wins_metric_df)
    all_overall_wins.append(overall_wins_df)
    rank_sum_tables[ds] = rank_sum

summary_all = pd.concat(all_summary, ignore_index=True)
outliers_all = pd.concat(all_outliers, ignore_index=True)
wins_metric_all = pd.concat(all_wins_metric, ignore_index=True)
overall_wins_all = pd.concat(all_overall_wins, ignore_index=True)

# Save outputs
out_dir = Path(r"path\to\Rank_Sum_Model_Comparison")
out_dir.mkdir(parents=True, exist_ok=True)

summary_all.to_csv(out_dir / "seg_metrics_summary_mean_sd_median_p05_p95_min_max.csv", index=False)
outliers_all.to_csv(out_dir / "seg_outlier_counts.csv", index=False)
wins_metric_all.to_csv(out_dir / "seg_metric_wins_per_case.csv", index=False)
overall_wins_all.to_csv(out_dir / "seg_overall_wins_rank_sum.csv", index=False)

for ds, df_rank in rank_sum_tables.items():
    df_rank.to_csv(out_dir / f"{ds}_rank_sum_per_case.csv", index=False)

print("Saved to:", out_dir)
print("\nOverall wins:\n", overall_wins_all)
print("\nOutliers:\n", outliers_all)


