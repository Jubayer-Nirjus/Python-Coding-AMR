"""
04_amr_risk_index.py
====================
Step 4: AMR Risk Index — distribution, Kruskal-Wallis, farm-type breakdown.
Ground truth: Low=114(53.8%), Moderate=71(33.5%), High=27(12.7%).
"""
import pandas as pd
import numpy as np
from scipy import stats
import importlib.util
import sys

# Import 00_load_data.py dynamically since it starts with digits
spec = importlib.util.spec_from_file_location("load_data", "00_load_data.py")
load_data_module = importlib.util.module_from_spec(spec)
sys.modules["load_data"] = load_data_module
spec.loader.exec_module(load_data_module)

from load_data import load_all, FT_MAP
data = load_all()
df = data["combined"]

print("="*65); print("STEP 4 — AMR RISK INDEX ANALYSIS"); print("="*65)

# ── Distribution ──────────────────────────────────────────────────────────
print(f"\nAMR Risk Index (0–8):")
print(f"  Mean ± SD : {df['AMR_Risk_Index'].mean():.2f} ± {df['AMR_Risk_Index'].std():.2f}")
print(f"  Median(IQR): {df['AMR_Risk_Index'].median():.1f} "
      f"({df['AMR_Risk_Index'].quantile(.25):.1f}–{df['AMR_Risk_Index'].quantile(.75):.1f})")
print(f"  Min–Max   : {df['AMR_Risk_Index'].min():.0f}–{df['AMR_Risk_Index'].max():.0f}")

# ── Category counts ───────────────────────────────────────────────────────
n = len(df)
for cat, lo, hi in [("Low (0–2)",114,53.8),("Moderate (3–4)",71,33.5),("High ≥5",27,12.7)]:
    col_mask = df["AMR_Risk_Cat"] == cat.split(" ")[0]
    actual_n = col_mask.sum()
    print(f"  {cat}: n={actual_n} ({actual_n/n*100:.1f}%) — GT: {lo}({hi}%) "
          f"{'✅' if actual_n==lo else '❌'}")

# ── By farm type ──────────────────────────────────────────────────────────
print("\nBy Farm Type:")
for ft, lbl in FT_MAP.items():
    sub = df[df.Farm_Type==ft]
    n_ft = len(sub)
    n_h = (sub["AMR_Risk_High"]==1).sum()
    print(f"  {lbl} (n={n_ft}): Mean={sub['AMR_Risk_Index'].mean():.2f}±{sub['AMR_Risk_Index'].std():.2f} "
          f"| High n={n_h} ({n_h/n_ft*100:.1f}%)")

groups = [df[df.Farm_Type==ft]["AMR_Risk_Index"].dropna().values for ft in [0,1,2]]
h, p = stats.kruskal(*groups)
print(f"  Kruskal-Wallis: H={h:.2f}, p={p:.4f} {'**' if p<0.01 else ('*' if p<0.05 else 'ns')}")

# ── Component-level analysis ──────────────────────────────────────────────
print("\nAMR Risk Component Prevalence (Table 4):")
comp_cols = {
    "Non-Rx AM purchase": ("AM_Without_Rx", 0, df),
    "Watch/Reserve category": ("AWaRE_Category", [1,2], df),
    "Polytherapy (≥2 AMs)": ("Number_of_AM", [2,3], df),
    "Poor withdrawal adherence": ("Withdrawal_Practice", [1,2], df),
    "Growth promoter use": ("AM_Growth_Promoter", 0, df),
    "Reuse leftover ABs": ("Reuse_Leftover", 0, df),
    "Non-vet prescriber": ("Prescriber_of_AM", [1,2], df),
    "Unlabelled storage": ("Leftover_Storage", 1, df),
}
for label, (col, val, dff) in comp_cols.items():
    if col not in dff.columns: continue
    nn = dff[col].notna().sum()
    if isinstance(val, list):
        n_risk = dff[col].isin(val).sum()
    else:
        n_risk = (dff[col] == val).sum()
    print(f"  {label:<35} {n_risk}/{nn} ({n_risk/nn*100:.1f}%)")

# ── High-risk farms: component profiles ───────────────────────────────────
print("\nHigh-Risk farms (n=27) — component means (higher=riskier):")
high_df = df[df["AMR_Risk_High"]==1]
for col, lbl in [("AM_Without_Rx","Non-Rx"),("AM_Growth_Promoter","GrowthPromoter"),
                  ("AMR_Risk_Index","Mean Index")]:
    if col in high_df.columns:
        print(f"  {lbl}: {high_df[col].mean():.2f}")

print("\n✅ Step 4 complete.")
