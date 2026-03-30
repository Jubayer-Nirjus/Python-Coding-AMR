"""
01_data_cleaning.py
===================
Step 1: Data cleaning, missingness report, Supplementary Table S1.

Outputs:
  - Missing data summary (console)
  - Supp Table S1: Variable-level missing data report
  - Basic data quality flags
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import importlib.util
import sys

# Import 00_load_data.py dynamically since it starts with digits
spec = importlib.util.spec_from_file_location("load_data", "00_load_data.py")
load_data_module = importlib.util.module_from_spec(spec)
sys.modules["load_data"] = load_data_module
spec.loader.exec_module(load_data_module)

from load_data import load_all, FT_MAP, SCORE_COLS, RISK_COLS

# ── Load ───────────────────────────────────────────────────────────────────
data = load_all()
df = data["combined"]
df_am = df[df["AM_use_binary"] == 0].copy()   # AM users (n=165)
df_non = df[df["AM_use_binary"] == 1].copy()  # Non-users (n=47)
df_am = df[df["AM_use_binary"] == 0].copy()   # AM users (n=165)
df_non = df[df["AM_use_binary"] == 1].copy()  # Non-users (n=47)

print("=" * 65)
print("STEP 1 — DATA CLEANING & MISSING DATA REPORT")
print("=" * 65)
print(f"Total enrolled:  n={len(df)}")
print(f"AM users:        n={len(df_am)} ({len(df_am)/len(df)*100:.1f}%)")
print(f"Non-AM users:    n={len(df_non)} ({len(df_non)/len(df)*100:.1f}%)")
print(f"Farm type:       Broiler={len(df[df.Farm_Type==0])}, "
      f"Layer={len(df[df.Farm_Type==1])}, Sonali={len(df[df.Farm_Type==2])}")

# ── Sentinel (99) → NaN already handled in load_all ──────────────────────
print("\n── Sentinel value (99=structural N/A) already replaced with NaN ──")

# ── 1. Variable-level missingness ──────────────────────────────────────────
print("\n── Supplementary Table S1: Missing Data Report ──")

STRUCTURAL_COLS = [  # Expected N/A for non-AM users
    "Prescriber_of_AM", "AM_Purchase_Source", "Withdrawal_Practice",
    "AM_Growth_Promoter", "Number_of_AM", "AWaRE_Category",
    "Reuse_Leftover", "AM_Without_Rx", "Leftover_AMs",
    "Leftover_Storage", "AM_Name_Code", "AM_Name_Decoded",
    "AWaRE_Label", "WHO_CIA_Category", "Purpose_of_AM_Use",
]

key_vars = [
    "Gender", "Age_Group", "Education", "Farm_Type", "Flock_Size",
    "Farm_Duration", "Training", "Knowledge_Score", "Attitude_Score",
    "Practice_Score_Adjusted", "Performance_Score", "AMR_Risk_Index",
    "AMR_Risk_High", "AMR_Risk_Cat", "Heard_of_AMR",
    "AM_use_binary", "Prescriber_of_AM", "AM_Without_Rx",
    "AM_Growth_Promoter", "AWaRE_Label", "AWaRE_Category",
    "Latitude", "Longitude",
]

rows = []
for col in key_vars:
    if col not in df.columns:
        continue
    n_missing = df[col].isna().sum()
    pct_miss = n_missing / len(df) * 100
    structural = col in STRUCTURAL_COLS
    note = "Structural N/A for non-AM users (n=47)" if structural else ""
    rows.append({
        "Variable": col,
        "n_missing": n_missing,
        "pct_missing": round(pct_miss, 1),
        "Structural_NA": structural,
        "Note": note,
    })

supp_s1 = pd.DataFrame(rows)
print(supp_s1.to_string(index=False))

# ── 2. GPS missingness ─────────────────────────────────────────────────────
print("\n── GPS Data Availability ──")
n_gps = df[["Latitude", "Longitude"]].dropna().shape[0]
n_miss_gps = len(df) - n_gps
print(f"GPS available: n={n_gps} ({n_gps/len(df)*100:.1f}%)")
print(f"GPS missing:   n={n_miss_gps} ({n_miss_gps/len(df)*100:.1f}%)")

# Test if GPS-missing farms differ systematically (MAR check)
gps_miss = df["Latitude"].isna().astype(int)
for col in ["AMR_Risk_High", "AMR_Risk_Index"]:
    if col in df.columns:
        g0 = df.loc[gps_miss == 0, col].dropna()
        g1 = df.loc[gps_miss == 1, col].dropna()
        if len(g1) > 0:
            stat, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            print(f"  GPS-missing vs GPS-available [{col}]: U={stat:.0f}, p={p:.3f}"
                  f" {'(ns)' if p > 0.05 else '(*)'}")

# ── 3. Duplicate check ────────────────────────────────────────────────────
dups = df["Unique_ID"].duplicated().sum()
print(f"\nDuplicate IDs: {dups} {'✅' if dups == 0 else '❌ ISSUE'}")

# ── 4. Score range checks ─────────────────────────────────────────────────
print("\n── Score range checks ──")
ranges = {
    "Knowledge_Score": (0, 6),
    "Attitude_Score": (0, 4),
    "Practice_Score_Adjusted": (0, 30),
    "Performance_Score": (0, 3),
    "AMR_Risk_Index": (0, 8),
}
for col, (lo, hi) in ranges.items():
    if col not in df.columns:
        continue
    out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"  {col} [{lo}–{hi}]: {out} out-of-range {'✅' if out == 0 else '❌'}")

# ── 5. AM_use_binary consistency ──────────────────────────────────────────
print("\n── AM use consistency ──")
# Farms coded AM_use_binary=1 (non-user) should have AWaRE_Label=N/A
non_user_aware = df[df["AM_use_binary"] == 1]["AWaRE_Label"].value_counts()
print("Non-user AWaRE_Label distribution (all should be N/A):")
print(non_user_aware)

print("\n✅ Data cleaning complete. Proceed to Step 2.")
