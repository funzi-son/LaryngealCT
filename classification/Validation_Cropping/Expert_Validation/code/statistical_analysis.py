#=======================================
#Expert Evalation Statistical Analyis
#=======================================

#install dependencies

!pip install pandas openpyxl matplotlib seaborn pingouin scikit-learn

#=========================
#Descriptive Statistics
#=========================

import pandas as pd
from scipy import stats

# --- Load Excel files ---
file1 = "path/to/Expert_Validation_scores_Dr_Devaraja.xlsx"
file2 = "path/to/Expert_Validation_scores_Dr_Prakashini.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# --- Sanity check ---
assert list(df1.columns) == list(df2.columns), "Column names do not match"
columns = df1.columns[1:]  # exclude case ID

# --- Descriptive stats ---
def descriptive_summary(df, expert_name):
    print(f"\n📊 Descriptive Statistics for {expert_name}")
    for col in columns:
        print(f"\n--- {col} ---")
        data = df[col].dropna()
        value_counts = data.value_counts().sort_index()
        print(f"Value counts:\n{value_counts}")
        print(f"Mean: {data.mean():.2f}")
        print(f"Median: {data.median():.2f}")
        print(f"Std Dev: {data.std():.2f}")
        mode_val = stats.mode(data, keepdims=True)[0][0]
        mode_freq = value_counts.get(mode_val, 0)
        print(f"Mode: {mode_val} (frequency: {mode_freq})")

# --- Run summaries for both experts ---
descriptive_summary(df1, "Dr. Devaraja")
descriptive_summary(df2, "Dr. Prakashini")

#=========================================================
#Calculate Weighted Kappa and Raw Agreement% with 95% CI
#=========================================================

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import norm

# Load Excel files
df_d = pd.read_excel("path/to/Expert_Validation_scores_Dr_Devaraja.xlsx")
df_p = pd.read_excel("path/to/Expert_Validation_scores_Dr_Prakashini.xlsx")

# Strip whitespace from column names
df_d.columns = df_d.columns.str.strip()
df_p.columns = df_p.columns.str.strip()

# Rename columns to avoid collision
df_d = df_d.rename(columns={
    'Anatomical_Coverage': 'Anatomical_Coverage_DrD',
    'Image_Quality': 'Image_Quality_DrD',
    'Segmentation_Feasibility': 'Segmentation_Feasibility_DrD',
    'Classification_Feasibility': 'Classification_Feasibility_DrD'
})
df_p = df_p.rename(columns={
    'Anatomical_Coverage': 'Anatomical_Coverage_DrP',
    'Image_Quality': 'Image_Quality_DrP',
    'Segmentation_Feasibility': 'Segmentation_Feasibility_DrP',
    'Classification_Feasibility': 'Classification_Feasibility_DrP'
})

# Merge
df = pd.merge(df_d, df_p, on='TCIA_ID')

def bootstrap_metric(x, y, metric_func, n_bootstraps=1000, alpha=0.05):
    """Bootstrap CI for metric_func applied to paired x,y"""
    n = len(x)
    stats = []
    rng = np.random.default_rng()
    for _ in range(n_bootstraps):
        indices = rng.choice(n, n, replace=True)
        stats.append(metric_func(x[indices], y[indices]))
    stats = np.array(stats)
    lower = np.percentile(stats, 100*alpha/2)
    upper = np.percentile(stats, 100*(1-alpha/2))
    return lower, upper

def weighted_kappa(x, y):
    return cohen_kappa_score(x, y, weights='quadratic')

def raw_agreement(x, y):
    return np.mean(x == y) * 100

results = []
criteria = ['Anatomical_Coverage', 'Image_Quality', 'Segmentation_Feasibility', 'Classification_Feasibility']

for crit in criteria:
    col_d = df[f'{crit}_DrD'].values
    col_p = df[f'{crit}_DrP'].values

    # Weighted Kappa and CI
    w_kappa_val = weighted_kappa(col_d, col_p)
    w_kappa_low, w_kappa_high = bootstrap_metric(col_d, col_p, weighted_kappa)

    # Raw agreement and CI
    raw_agree_val = raw_agreement(col_d, col_p)
    raw_agree_low, raw_agree_high = bootstrap_metric(col_d, col_p, raw_agreement)

    results.append({
        'Criterion': crit,
        'Weighted Kappa': round(w_kappa_val, 3),
        'Weighted Kappa 95% CI': f"({w_kappa_low:.3f}, {w_kappa_high:.3f})",
        'Raw Agreement (%)': round(raw_agree_val, 2),
        'Raw Agreement 95% CI': f"({raw_agree_low:.2f}, {raw_agree_high:.2f})"
    })

df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv("Expert_Interrater_Results_WeightedKappa_RawAgreement.csv", index=False)


