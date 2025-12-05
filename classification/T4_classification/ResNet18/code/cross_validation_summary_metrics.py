# ========================================================
# Aggregate Cross-Validation Metrics (from cv_summary JSON)
# ========================================================

import json
import pandas as pd
import numpy as np
from pathlib import Path

# ================
# CONFIG
# ================
cv_summary_path = Path("path/to/cv_summary_resnet18.json")

# ================
# Load JSON
# ================
with open(cv_summary_path, "r") as f:
    folds = json.load(f)

print(f"✅ Loaded {len(folds)} folds from {cv_summary_path}")

# =========================================================
# Collect per-fold metrics (use calibrated metrics at t_opt)
# =========================================================
records = []
for fold in folds:
    rec = fold.get("val_cal", fold.get("cal_metrics", {}))  # adapt to your key naming
    rec["fold"] = fold["fold"]
    records.append(rec)

df = pd.DataFrame(records)
print("\nPer-fold metrics:")
display(df)

# ===================
# Compute mean ± SD
# ===================
summary_stats = {}
for col in df.columns:
    if col == "fold":
        continue
    try:
        vals = df[col].astype(float).values
        mean, std = np.mean(vals), np.std(vals)
        summary_stats[col] = f"{mean:.3f} ± {std:.3f}"
    except:
        continue

summary_df = pd.DataFrame(summary_stats, index=["Mean ± SD"]).T

print("\n✅ Cross-validation summary (Mean ± SD):")
display(summary_df)

# ===================
# Save outputs
# ===================
out_dir = cv_summary_path.parent
summary_df.to_csv(out_dir / "cv_summary_table.csv")

with open(out_dir / "cv_summary_table.tex", "w") as f:
    f.write(summary_df.to_latex())

print(f"\n📂 Saved summary to:\n- {out_dir/'cv_summary_table.csv'}\n- {out_dir/'cv_summary_table.tex'}")