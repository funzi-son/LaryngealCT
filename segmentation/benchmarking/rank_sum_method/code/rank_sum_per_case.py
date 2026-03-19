# ==================================
# Calculate per-case rank-sum scores
# ==================================

import pandas as pd
import numpy as np

HIGHER_BETTER = {"Dice","IoU","SurfDice_1mm","SurfDice_2mm","Precision","Sensitivity"}
LOWER_BETTER  = {"HD95_mm","ASSD_mm","AbsVolDiff_mL","AbsVolDiff_mm3"}

GROUPS = {
    "overlap": ["Dice","IoU","Precision","Sensitivity"],
    "boundary": ["HD95_mm","ASSD_mm","SurfDice_1mm","SurfDice_2mm","AbsVolDiff_mL"],
    "all": ["Dice","IoU","HD95_mm","ASSD_mm","SurfDice_1mm","SurfDice_2mm",
            "Precision","Sensitivity","AbsVolDiff_mL"]
}

def grouped_rank_sum_from_files(model_paths: dict, groups=GROUPS):
    dfs = {m: pd.read_excel(p) for m,p in model_paths.items()}
    models = list(dfs.keys())

    # ---- common cases
    common = set(dfs[models[0]]["case_id"].astype(str))
    for m in models[1:]:
        common &= set(dfs[m]["case_id"].astype(str))
    common = sorted(common)
    if len(common) == 0:
        raise ValueError("No common case_id across models")

    for m in models:
        dfs[m]["case_id"] = dfs[m]["case_id"].astype(str)
        dfs[m] = (dfs[m]
                  .loc[dfs[m]["case_id"].isin(common)]
                  .sort_values("case_id")
                  .reset_index(drop=True))

    # ---- combined table
    combined = pd.DataFrame({"case_id": dfs[models[0]]["case_id"]})
    for m in models:
        for col in dfs[m].columns:
            if col == "case_id":
                continue
            combined[f"{col}__{m}"] = pd.to_numeric(dfs[m][col], errors="coerce")

    out = pd.DataFrame({"case_id": combined["case_id"]})

    # ---- rank-sum within each group
    n_cases = len(out)

    for gname, metrics in groups.items():
        metrics_present = []
        for met in metrics:
            cols = [f"{met}__{m}" for m in models]
            if all(c in combined.columns for c in cols):
                metrics_present.append(met)

        if len(metrics_present) == 0:
            print(f"[WARN] No metrics present for group '{gname}'. Skipping.")
            continue

        #rs must be per-case (n_cases rows)
        rs = pd.DataFrame(0.0, index=np.arange(n_cases), columns=models)

        for met in metrics_present:
            cols = [f"{met}__{m}" for m in models]
            mat = combined[cols].copy()

            if met in HIGHER_BETTER:
                ranks = mat.rank(axis=1, method="average", ascending=False)
            else:
                ranks = mat.rank(axis=1, method="average", ascending=True)

            for m in models:
                rs[m] += ranks[f"{met}__{m}"].to_numpy()

        out[f"{gname}_winner"] = rs.idxmin(axis=1)
        for m in models:
            out[f"{gname}_ranksum__{m}"] = rs[m].values

    return out

# Example usage:
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

seedb_grouped = grouped_rank_sum_from_files(FILES["SeedB"])
print(seedb_grouped.filter(["case_id","overlap_winner","boundary_winner","all_winner"]).head())
hanseg_grouped = grouped_rank_sum_from_files(FILES["HanSeg"])
print(hanseg_grouped.filter(["case_id","overlap_winner","boundary_winner","all_winner"]).head())

seedb_grouped.to_csv(r"path\to\SeedB_rank_sum_per_case_grouped.csv", index=False)
hanseg_grouped.to_csv(r"path\to\HanSeg_rank_sum_per_case_grouped.csv", index=False)
