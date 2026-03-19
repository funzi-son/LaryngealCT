# ====================================================
# compute confidence interval using Bootstrapping
# ====================================================
import pandas as pd
import numpy as np

def bootstrap_rank_sum_ci(rank_sum_csv, models=("Baseline","Boundary","BC_nnUNet"), B=5000, seed=0):
    df = pd.read_csv(rank_sum_csv)
    # df columns: CaseID, Baseline, Boundary, TTA, OverallWinner
    n = len(df)
    rng = np.random.default_rng(seed)

    # store bootstrap stats
    win_counts = {m: [] for m in models}
    mean_rank  = {m: [] for m in models}

    for _ in range(B):
        idx = rng.integers(0, n, size=n)  # resample cases with replacement
        boot = df.iloc[idx].copy()

        # recompute winner from rank sums (don’t trust stored OverallWinner)
        boot_winner = boot[list(models)].idxmin(axis=1)

        for m in models:
            win_counts[m].append((boot_winner == m).sum())
            mean_rank[m].append(boot[m].mean())

    # summarize
    summary_rows = []
    for m in models:
        wc = np.array(win_counts[m])
        mr = np.array(mean_rank[m])

        summary_rows.append({
            "Model": m,
            "WinCount_mean": wc.mean(),
            "WinCount_CI95_low": np.percentile(wc, 2.5),
            "WinCount_CI95_high": np.percentile(wc, 97.5),
            "WinRate_mean": wc.mean()/n,
            "WinRate_CI95_low": np.percentile(wc, 2.5)/n,
            "WinRate_CI95_high": np.percentile(wc, 97.5)/n,
            "MeanRankSum_mean": mr.mean(),
            "MeanRankSum_CI95_low": np.percentile(mr, 2.5),
            "MeanRankSum_CI95_high": np.percentile(mr, 97.5),
        })

    return pd.DataFrame(summary_rows).sort_values("MeanRankSum_mean")

# Example:
seedb_ci  = bootstrap_rank_sum_ci(r"path\to\SeedB_rank_sum_per_case.csv")
hanseg_ci = bootstrap_rank_sum_ci(r"path\to\HanSeg_rank_sum_per_case.csv")

seedb_ci.to_csv(r"path\to\SeedB_rank_sum_overall_CI.csv")
hanseg_ci.to_csv(r"path\to\HanSeg_rank_sum_overall_CI.csv")

print("SeedB:\n", seedb_ci, "\n")
print("HanSeg:\n", hanseg_ci)

# print summary

for name, df in [("SeedB", seedb_grouped), ("HanSeg", hanseg_grouped)]:
    print("\n", name)
    for g in ["overlap","boundary","all"]:
        print(g, df[f"{g}_winner"].value_counts())

