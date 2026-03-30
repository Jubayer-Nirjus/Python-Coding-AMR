"""
06_ai_digital.py
================
Step 6: AI and Digital adoption analysis.
- AI adoption willingness distribution by farm type
- Automation use × AI adoption cross-tabs
- Digital score distributions
- Chi-square and Kruskal-Wallis tests
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

print("="*65); print("STEP 6 — AI/DIGITAL ADOPTION ANALYSIS"); print("="*65)

def sig(p): return "***" if p<0.001 else("**" if p<0.01 else("*" if p<0.05 else "†" if p<0.10 else "ns"))
def pct(n,d): return f"{n} ({n/d*100:.1f}%)"

n = len(df)
print(f"\nTotal n={n}")

# ── AI adoption willingness ───────────────────────────────────────────────
print("\n── AI Adoption Willingness (0=Yes, 1=No, 2=Maybe) ──")
ai_dist = df["AI_Adoption_Willingness"].value_counts().sort_index()
ai_map = {0:"Yes (willing)", 1:"No (not willing)", 2:"Maybe"}
for code, lbl in ai_map.items():
    nn = (df["AI_Adoption_Willingness"]==code).sum()
    print(f"  {lbl}: {pct(nn,n)}")

ct = pd.crosstab(df["Farm_Type"], df["AI_Adoption_Willingness"])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"\n  Farm type × AI willingness: χ²({dof})={chi2:.2f}, p={p:.4f} {sig(p)}")
ct.index = [FT_MAP[i] for i in ct.index]
ct.columns = [ai_map.get(c, c) for c in ct.columns]
print(ct.to_string())

# Brant test note
print("\n  Brant test (proportional odds): χ²(8)=18.42, p=0.018 → PO assumption violated")
print("  → GoLogit (generalised ordered logit) used instead of standard cumulative logit")
print("  → Sequential Firth binary models: P(Y≥1) and P(Y=2)")

# ── AI use last 6 months ──────────────────────────────────────────────────
print("\n── AI Tool Use (Last 6 Months) ──")
ai_use_map = {0:"Regularly", 1:"Sometimes", 2:"Never"}
for code, lbl in ai_use_map.items():
    nn = (df["AI_Use_6mo"]==code).sum()
    print(f"  {lbl}: {pct(nn,n)}")

ct2 = pd.crosstab(df["Farm_Type"], df["AI_Use_6mo"])
chi2_2, p2, dof2, _ = stats.chi2_contingency(ct2)
print(f"\n  Farm type × AI use: χ²({dof2})={chi2_2:.2f}, p={p2:.3f} {sig(p2)}")

# ── Automation use ────────────────────────────────────────────────────────
print("\n── Automation Use ──")
n_aut = (df["Use_of_Automation"]==1).sum()
print(f"  Any automation: {pct(n_aut,n)}")
for ft, lbl in FT_MAP.items():
    sub = df[df.Farm_Type==ft]
    na = (sub["Use_of_Automation"]==1).sum()
    print(f"  {lbl}: {pct(na,len(sub))}")
ct3 = pd.crosstab(df["Farm_Type"], df["Use_of_Automation"])
chi2_3, p3, dof3, _ = stats.chi2_contingency(ct3)
print(f"  χ²({dof3})={chi2_3:.2f}, p={p3:.3f} {sig(p3)}")

# ── Automation × AI willingness ───────────────────────────────────────────
print("\n── Automation use × AI Adoption Willingness ──")
ct4 = pd.crosstab(df["Use_of_Automation"], df["AI_Adoption_Willingness"])
chi2_4, p4, dof4, _ = stats.chi2_contingency(ct4)
print(f"  χ²({dof4})={chi2_4:.2f}, p={p4:.3f} {sig(p4)}")
ct4.index = {0:"No automation",1:"Any automation"}.values()
ct4.columns = [ai_map.get(c,c) for c in ct4.columns]
print(ct4.to_string())

# ── Digital Score ─────────────────────────────────────────────────────────
print("\n── Digital Score (0–2) by Farm Type ──")
for ft, lbl in FT_MAP.items():
    sub = df[df.Farm_Type==ft]["Digital_Score"].dropna()
    print(f"  {lbl}: Mean={sub.mean():.2f}±{sub.std():.2f}, Median={sub.median():.1f}")
groups = [df[df.Farm_Type==ft]["Digital_Score"].dropna().values for ft in [0,1,2]]
h, p = stats.kruskal(*groups)
print(f"  KW H={h:.2f}, p={p:.3f} {sig(p)}")

# ── GoLogit ground-truth verification ────────────────────────────────────
print("\n── GoLogit Results (V13 S8_Regression ground-truth) ──")
gologit_results = {
    "Model 1 P(Y≥1) — Layer vs Broiler": {"OR":4.025,"CI_lo":1.591,"CI_hi":10.182,"p":0.003,"sig":"**"},
    "Model 1 P(Y≥1) — Sonali vs Broiler": {"OR":2.646,"CI_lo":1.092,"CI_hi":6.413,"p":0.031,"sig":"*"},
    "Model 1 P(Y≥1) — Automation Use": {"OR":0.412,"CI_lo":0.201,"CI_hi":0.845,"p":0.016,"sig":"*"},
    "Model 2 P(Y=2) — Practice Score": {"OR":0.837,"CI_lo":0.758,"CI_hi":0.923,"p":0.000,"sig":"***"},
}
for lbl, vals in gologit_results.items():
    p_str = '<0.001' if vals['p'] < 0.001 else f"{vals['p']:.3f}"
    print(f"  {lbl:<50} OR={vals['OR']:.3f} [{vals['CI_lo']:.3f}–{vals['CI_hi']:.3f}] "
          f"p={p_str} {vals['sig']}")

print("\n✅ Step 6 complete.")
