# ==================================================
# Code to plot results of model selection
# ==================================================

####################################################
# Version-1
####################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODELS = ["Baseline","Boundary","BC_nnUNet"]

def bootstrap_winrate_from_winner_series(winners, B=5000, seed=0):
    winners = np.array(winners)
    n = len(winners)
    rng = np.random.default_rng(seed)
    stats = {m: [] for m in MODELS}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sample = winners[idx]
        for m in MODELS:
            stats[m].append((sample == m).mean())
    out = {}
    for m in MODELS:
        arr = np.array(stats[m])
        out[m] = (arr.mean(), np.percentile(arr,2.5), np.percentile(arr,97.5))
    return out

def get_overall_winner_from_ranksum(csv_path):
    df = pd.read_csv(csv_path)
    winners = df[MODELS].idxmin(axis=1)
    return winners

# --- Load your rank-sum per case
seedb_rs  = r"path\to\SeedB_rank_sum_per_case.csv"
hanseg_rs = r"path\to\HanSeg_rank_sum_per_case.csv"

seedb_overall = get_overall_winner_from_ranksum(seedb_rs)
hanseg_overall = get_overall_winner_from_ranksum(hanseg_rs)

seedb_ci = bootstrap_winrate_from_winner_series(seedb_overall)
hanseg_ci = bootstrap_winrate_from_winner_series(hanseg_overall)

# OPTIONAL: if you generate grouped winners using grouped_rank_sum_from_files(),
# you can compute CI for overlap_winner and boundary_winner similarly.

# --- Load outliers
outliers_csv = r"path\to\seg_outlier_counts.csv"
out_df = pd.read_csv(outliers_csv)

def plot_winrate_panel(ax, title, ci_seedb, ci_hanseg):
    x = np.arange(len(MODELS))
    # SeedB
    y1 = [ci_seedb[m][0] for m in MODELS]
    e1 = [[ci_seedb[m][0]-ci_seedb[m][1] for m in MODELS],
          [ci_seedb[m][2]-ci_seedb[m][0] for m in MODELS]]
    # HanSeg
    y2 = [ci_hanseg[m][0] for m in MODELS]
    e2 = [[ci_hanseg[m][0]-ci_hanseg[m][1] for m in MODELS],
          [ci_hanseg[m][2]-ci_hanseg[m][0] for m in MODELS]]

    width = 0.35
    ax.bar(x - width/2, y1, width, yerr=e1, capsize=3, label="SeedB")
    ax.bar(x + width/2, y2, width, yerr=e2, capsize=3, label="HanSeg")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win-rate")
    ax.set_title(title)
    ax.legend()

def plot_outliers_panel(ax, out_df):
    # expects columns: Dataset, Model, Metric, Threshold, OutlierCount
    # plot HD95 and ASSD outliers for HanSeg (external) primarily
    sub = out_df[out_df["Dataset"].isin(["SeedB","HanSeg"])].copy()
    # pivot for easy plotting
    piv = sub.pivot_table(index=["Dataset","Model"], columns="Metric", values="OutlierCount", aggfunc="sum").reset_index()
    # keep only HD95_mm, ASSD_mm
    metrics = [m for m in ["HD95_mm","ASSD_mm"] if m in piv.columns]

    # create grouped bars: one group per dataset-model
    labels = [f"{r.Dataset}-{r.Model}" for r in piv.itertuples(index=False)]
    x = np.arange(len(labels))
    width = 0.35

    if len(metrics) == 2:
        ax.bar(x - width/2, piv[metrics[0]].values, width, label=metrics[0])
        ax.bar(x + width/2, piv[metrics[1]].values, width, label=metrics[1])
    else:
        ax.bar(x, piv[metrics[0]].values, width, label=metrics[0])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Outlier count")
    ax.set_title("Failure-mode outliers")
    ax.legend()

# ---- Build 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

plot_winrate_panel(axes[0,0], "Overall (rank-sum) win-rate with 95% CI", seedb_ci, hanseg_ci)

# Placeholders: if you compute grouped winners, replace these with overlap/boundary CI
# For now, duplicate overall; you will replace once you generate group winners
plot_winrate_panel(axes[0,1], "Boundary-focused win-rate", seedb_ci, hanseg_ci)
plot_winrate_panel(axes[1,0], "Overlap-focused win-rate", seedb_ci, hanseg_ci)

plot_outliers_panel(axes[1,1], out_df)

plt.tight_layout()
plt.savefig(r"path\to\model_selection_figure.png", dpi=600)
plt.savefig(r"path\to\model_selection_figure.svg")
plt.show()


###############################################################################
# Version 2
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODELS = ["Baseline", "Boundary", "BC_nnUNet"]

def bootstrap_winrate_from_winner_series(winners, B=5000, seed=0):
    winners = np.array(winners, dtype=str)
    n = len(winners)
    rng = np.random.default_rng(seed)
    stats = {m: [] for m in MODELS}

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sample = winners[idx]
        for m in MODELS:
            stats[m].append((sample == m).mean())

    out = {}
    for m in MODELS:
        arr = np.array(stats[m])
        out[m] = (arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5))
    return out

def get_overall_winner_from_ranksum(csv_path):
    df = pd.read_csv(csv_path)
    return df[MODELS].idxmin(axis=1)

def get_group_winner(grouped_csv, col):
    df = pd.read_csv(grouped_csv)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {grouped_csv}. Available: {list(df.columns)}")
    return df[col].astype(str)

def plot_winrate_panel(ax, title, ci_seedb, ci_hanseg):
    x = np.arange(len(MODELS))

    y1 = [ci_seedb[m][0] for m in MODELS]
    e1 = [[ci_seedb[m][0] - ci_seedb[m][1] for m in MODELS],
          [ci_seedb[m][2] - ci_seedb[m][0] for m in MODELS]]

    y2 = [ci_hanseg[m][0] for m in MODELS]
    e2 = [[ci_hanseg[m][0] - ci_hanseg[m][1] for m in MODELS],
          [ci_hanseg[m][2] - ci_hanseg[m][0] for m in MODELS]]

    width = 0.35
    ax.bar(x - width/2, y1, width, yerr=e1, capsize=3, label="SeedB")
    ax.bar(x + width/2, y2, width, yerr=e2, capsize=3, label="HanSeg")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win-rate")
    ax.set_title(title)
    ax.legend()

def plot_outliers_panel_hanseg(ax, out_df, hd95_thr=10.0, assd_thr=1.0):
    df = out_df.copy()

    df = df[df["Dataset"].astype(str).str.lower() == "hanseg"].copy()
    df = df[df["Metric"].isin(["HD95_mm", "ASSD_mm"])].copy()

    if "Threshold" in df.columns:
        df_hd = df[(df["Metric"] == "HD95_mm") & (df["Threshold"] == hd95_thr)]
        df_as = df[(df["Metric"] == "ASSD_mm") & (df["Threshold"] == assd_thr)]
        df = pd.concat([df_hd, df_as], ignore_index=True)

    piv = df.pivot_table(index="Model", columns="Metric", values="OutlierCount", aggfunc="sum")
    piv = piv.reindex(MODELS)

    for met in ["HD95_mm", "ASSD_mm"]:
        if met not in piv.columns:
            piv[met] = 0

    x = np.arange(len(MODELS))
    width = 0.35

    ax.bar(x - width/2, piv["HD95_mm"].values, width, label="HD95 > 10 mm")
    ax.bar(x + width/2, piv["ASSD_mm"].values, width, label="ASSD > 1 mm")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Outlier count (HanSeg)")
    ax.set_title("External failure-mode outliers (HanSeg)")
    ax.legend()

# ---------------------------
# Paths (EDIT as required)
# ---------------------------
seedb_rs  = r"path\to\SeedB_rank_sum_per_case.csv"
hanseg_rs = r"path\to\HanSeg_rank_sum_per_case.csv"

seedb_grouped_csv  = r"path\to\SeedB_rank_sum_per_case_grouped.csv"
hanseg_grouped_csv = r"path\to\HanSeg_rank_sum_per_case_grouped.csv"

outliers_csv = r"path\to\seg_outlier_counts.csv"
out_df = pd.read_csv(outliers_csv)

# ---------------------------
# Compute CIs
# ---------------------------
# Overall (rank-sum based)
seedb_overall = get_overall_winner_from_ranksum(seedb_rs)
hanseg_overall = get_overall_winner_from_ranksum(hanseg_rs)
seedb_ci_overall = bootstrap_winrate_from_winner_series(seedb_overall, B=5000, seed=0)
hanseg_ci_overall = bootstrap_winrate_from_winner_series(hanseg_overall, B=5000, seed=0)

# Grouped overlap and boundary winners
seedb_overlap = get_group_winner(seedb_grouped_csv, "overlap_winner")
hanseg_overlap = get_group_winner(hanseg_grouped_csv, "overlap_winner")
seedb_ci_overlap = bootstrap_winrate_from_winner_series(seedb_overlap, B=5000, seed=1)
hanseg_ci_overlap = bootstrap_winrate_from_winner_series(hanseg_overlap, B=5000, seed=1)

seedb_boundary = get_group_winner(seedb_grouped_csv, "boundary_winner")
hanseg_boundary = get_group_winner(hanseg_grouped_csv, "boundary_winner")
seedb_ci_boundary = bootstrap_winrate_from_winner_series(seedb_boundary, B=5000, seed=2)
hanseg_ci_boundary = bootstrap_winrate_from_winner_series(hanseg_boundary, B=5000, seed=2)

# ---------------------------
# Plot 2x2
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

plot_winrate_panel(axes[0,0], "Overall (rank-sum) win-rate with 95% CI", seedb_ci_overall, hanseg_ci_overall)
plot_winrate_panel(axes[0,1], "Boundary-focused win-rate with 95% CI", seedb_ci_boundary, hanseg_ci_boundary)
plot_winrate_panel(axes[1,0], "Overlap-focused win-rate with 95% CI", seedb_ci_overlap, hanseg_ci_overlap)

plot_outliers_panel_hanseg(axes[1,1], out_df)

plt.tight_layout()
plt.savefig(r"path\to\model_selection_figure_grouped.png", dpi=600)
plt.savefig(r"path\to\model_selection_figure_grouped.svg")
plt.show()