"""
05_bivariate.py
===============
Step 5: Bivariate analyses — AWaRe × stratifiers, AMU risk behaviors.
Chi-square tests, Fisher's exact for small cells, Mann-Whitney U.
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
df_am = df[df["AM_use_binary"]==0].copy()

print("="*65); print("STEP 5 — BIVARIATE ANALYSES"); print("="*65)

def chi2_or_fisher(col1, col2, dff):
    ct = pd.crosstab(dff[col1], dff[col2])
    if ct.min().min() < 5:
        _, p = stats.fisher_exact(ct) if ct.shape == (2,2) else (np.nan, np.nan)
        return "Fisher", p
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return f"χ²({dof})={chi2:.2f}", p

def sig(p): return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))

print("\n── AWaRe Category × Farm Type ──")
stat, p = chi2_or_fisher("Farm_Type", "AWaRE_Label", df_am)
print(f"  {stat}, p={p:.3f} {sig(p)}")
ct = pd.crosstab(df_am["Farm_Type"], df_am["AWaRE_Label"])
ct.index = [FT_MAP[i] for i in ct.index]
print(ct.to_string())

print("\n── AMR Risk High × Farm Type ──")
stat, p = chi2_or_fisher("Farm_Type", "AMR_Risk_High", df)
print(f"  {stat}, p={p:.3f} {sig(p)}")

print("\n── AMR Risk High × Knowledge Category ──")
if "Knowledge_Cat" in df.columns:
    stat, p = chi2_or_fisher("Knowledge_Cat", "AMR_Risk_High", df)
    print(f"  {stat}, p={p:.3f} {sig(p)}")
    ct = pd.crosstab(df["Knowledge_Cat"], df["AMR_Risk_High"])
    print(ct.to_string())

print("\n── AMR Risk Index × Knowledge Score (Mann-Whitney) ──")
hi = df[df["AMR_Risk_High"]==1]["Knowledge_Score"].dropna()
lo = df[df["AMR_Risk_High"]==0]["Knowledge_Score"].dropna()
u, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
print(f"  High-risk Knowledge: {hi.median():.1f} ({hi.quantile(.25):.1f}–{hi.quantile(.75):.1f})")
print(f"  Low/Mod-risk Knowledge: {lo.median():.1f} ({lo.quantile(.25):.1f}–{lo.quantile(.75):.1f})")
print(f"  U={u:.0f}, p={p:.4f} {sig(p)}")

print("\n── Non-Rx Purchase × Farm Type (AM users) ──")
stat, p = chi2_or_fisher("Farm_Type", "AM_Without_Rx", df_am)
print(f"  {stat}, p={p:.3f} {sig(p)}")
for ft, lbl in FT_MAP.items():
    sub = df_am[df_am.Farm_Type==ft]
    n_rx = (sub["AM_Without_Rx"]==0).sum()
    print(f"  {lbl}: {n_rx}/{len(sub)} ({n_rx/len(sub)*100:.1f}%)")

print("\n── Heard of AMR × AMR Risk High ──")
stat, p = chi2_or_fisher("Heard_of_AMR", "AMR_Risk_High", df)
print(f"  {stat}, p={p:.3f} {sig(p)}")
ct = pd.crosstab(df["Heard_of_AMR"].map({0:"Yes",1:"No"}), df["AMR_Risk_High"])
print(ct.to_string())

print("\n── Automation × AI Adoption Willingness ──")
stat, p = chi2_or_fisher("Use_of_Automation", "AI_Adoption_Willingness", df)
print(f"  {stat}, p={p:.3f} {sig(p)}")

print("\n✅ Step 5 complete.")
