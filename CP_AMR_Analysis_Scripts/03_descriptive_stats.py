"""
03_descriptive_stats.py
=======================
Step 3: Tables 1, 2, 3 — Sociodemographic, AMU profile, KAPP scores.
Tests: Chi-square (categorical) | Kruskal-Wallis + Dunn's post-hoc (continuous).
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
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
df_am = df[df["AM_use_binary"] == 0].copy()

print("=" * 65)
print("STEP 3 — DESCRIPTIVE STATISTICS (Tables 1, 2, 3)")
print("=" * 65)

FT = [0, 1, 2]  # Broiler, Layer, Sonali
FT_N = {0: 109, 1: 70, 2: 33}

# ── Helper functions ───────────────────────────────────────────────────────
def pct(n, d): return f"{n} ({n/d*100:.1f}%)"
def miqr(s): s = s.dropna(); return f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})"
def msd(s): s = s.dropna(); return f"{s.mean():.2f}±{s.std():.2f}"

def chi2_test(df, col, group_col="Farm_Type"):
    ct = pd.crosstab(df[group_col], df[col])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return chi2, dof, p

def kw_test(df, col, group_col="Farm_Type"):
    groups = [df[df[group_col] == ft][col].dropna().values for ft in FT]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*groups)
    return h, p

def dunns_posthoc(df, col, group_col="Farm_Type", alpha=0.05):
    """Dunn's test with Bonferroni correction."""
    pairs = list(combinations(FT, 2))
    results = {}
    for g1, g2 in pairs:
        s1 = df[df[group_col] == g1][col].dropna()
        s2 = df[df[group_col] == g2][col].dropna()
        if len(s1) == 0 or len(s2) == 0:
            continue
        # Dunn's test statistic
        combined = pd.concat([s1.rename(g1), s2.rename(g2)])
        ranks = combined.rank()
        n1, n2 = len(s1), len(s2)
        n = n1 + n2
        r1 = ranks.iloc[:n1].mean()
        r2 = ranks.iloc[n1:].mean()
        se = np.sqrt((n * (n + 1) / 12) * (1 / n1 + 1 / n2))
        z = (r1 - r2) / se
        p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
        p_adj = min(p_raw * len(pairs), 1.0)  # Bonferroni
        results[(g1, g2)] = {"z": z, "p_adj": p_adj, "sig": p_adj < alpha}
    return results

def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    elif p < 0.10: return "†"
    return "ns"

# ══════════════════════════════════════════════════════════════════
# TABLE 1: Sociodemographic & Farm Characteristics
# ══════════════════════════════════════════════════════════════════
print("\n─────────────────────────────────────────────")
print("TABLE 1: SOCIODEMOGRAPHIC & FARM PROFILE")
print("─────────────────────────────────────────────")
n = len(df)

def table1_row(label, col, cat_val=None, denom=None):
    """Print one row: n(%) for categorical, or median(IQR) for continuous."""
    if cat_val is not None:
        d = denom or n
        ov = (df[col] == cat_val).sum()
        bv = (df[df.Farm_Type==0][col] == cat_val).sum()
        lv = (df[df.Farm_Type==1][col] == cat_val).sum()
        sv = (df[df.Farm_Type==2][col] == cat_val).sum()
        print(f"  {label:<35} {pct(ov,n):<15} {pct(bv,109):<14} "
              f"{pct(lv,70):<13} {pct(sv,33)}")
    else:
        print(f"  {label:<35} {miqr(df[col]):<15} "
              f"{miqr(df[df.Farm_Type==0][col]):<14} "
              f"{miqr(df[df.Farm_Type==1][col]):<13} "
              f"{miqr(df[df.Farm_Type==2][col])}")

print(f"\n{'Characteristic':<35} {'Total(n=212)':<15} {'Broiler(109)':<14} {'Layer(70)':<13} {'Sonali(33)'}  p-value")
print("-" * 100)

# Gender
chi2, dof, p = chi2_test(df, "Gender")
print(f"\nGender [χ²({dof})={chi2:.2f}, p={p:.3f}]:")
table1_row("Male", "Gender", 0); table1_row("Female", "Gender", 1)

# Age
chi2, dof, p = chi2_test(df, "Age_Group")
print(f"\nAge group [χ²({dof})={chi2:.2f}, p={p:.3f}]:")
for ag, lbl in {0:"18–30 yrs",1:"31–40 yrs",2:"≥40 yrs (n=2)⚠"}.items():
    table1_row(lbl, "Age_Group", ag)

# Education
chi2, dof, p = chi2_test(df, "Education")
print(f"\nEducation [χ²({dof})={chi2:.2f}, p={p:.3f}]:")
for ec, lbl in {0:"Graduate",1:"College/SSC",2:"Primary/Below"}.items():
    table1_row(lbl, "Education", ec)

# Training
chi2, dof, p = chi2_test(df, "Training")
print(f"\nTraining [χ²({dof})={chi2:.2f}, p={p:.3f}{sig_stars(p)}]:")
for t, lbl in {0:"Yes",1:"No",2:"Maybe"}.items():
    table1_row(lbl, "Training", t)

# Flock size
chi2, dof, p = chi2_test(df, "Flock_Size")
print(f"\nFlock size [χ²({dof})={chi2:.2f}, p={p:.3f}{sig_stars(p)}]:")
for fs, lbl in {0:"<1,000",1:"1,000–5,000",2:"5,001–10,000",3:">10,000"}.items():
    table1_row(lbl, "Flock_Size", fs)

# Farm duration
chi2, dof, p = chi2_test(df, "Farm_Duration")
print(f"\nFarm duration [χ²({dof})={chi2:.2f}, p={p:.3f}{sig_stars(p)}]:")
for fd, lbl in {0:"1–3 years",1:"4–7 years",2:">7 years"}.items():
    table1_row(lbl, "Farm_Duration", fd)

# Total sheds
chi2, dof, p = chi2_test(df, "Total_Sheds")
print(f"\nNumber of sheds [χ²({dof})={chi2:.2f}, p={p:.3f}]:")
table1_row("1 shed", "Total_Sheds", 0); table1_row("2 sheds", "Total_Sheds", 1)

# Automation
print(f"\nDigital/AI:")
chi2, dof, p = chi2_test(df, "Use_of_Automation")
for u, lbl in {0:"No automation",1:"Any automation"}.items():
    table1_row(lbl, "Use_of_Automation", u)

chi2, dof, p = chi2_test(df, "AI_Use_6mo")
print(f"  [AI use χ²({dof})={chi2:.2f}, p={p:.3f}]")
for u, lbl in {0:"Regularly",1:"Sometimes",2:"Never"}.items():
    table1_row(lbl, "AI_Use_6mo", u)

chi2, dof, p = chi2_test(df, "AI_Adoption_Willingness")
print(f"\nAI adoption willingness [χ²({dof})={chi2:.2f}, p={p:.3f}{sig_stars(p)}]:")
for w, lbl in {0:"Yes",1:"No",2:"Maybe"}.items():
    table1_row(lbl, "AI_Adoption_Willingness", w)

# ══════════════════════════════════════════════════════════════════
# TABLE 2: AMU Profile & Behaviours
# ══════════════════════════════════════════════════════════════════
print("\n─────────────────────────────────────────────")
print("TABLE 2: AMU PROFILE & RISK BEHAVIOURS")
print("─────────────────────────────────────────────")
n_am = len(df_am)

# AM use prevalence
chi2, dof, p = chi2_test(df, "AM_use_binary")
print(f"\nAM use prevalence [χ²({dof})={chi2:.2f}, p={p:.3f}{sig_stars(p)}]:")
au = (df["AM_use_binary"] == 0).sum()
print(f"  Uses antibiotics: {pct(au, n)} | Broiler:{pct((df[df.Farm_Type==0]['AM_use_binary']==0).sum(),109)} "
      f"Layer:{pct((df[df.Farm_Type==1]['AM_use_binary']==0).sum(),70)} "
      f"Sonali:{pct((df[df.Farm_Type==2]['AM_use_binary']==0).sum(),33)}")

# Withdrawal awareness (all farmers)
chi2, dof, p = chi2_test(df, "Withdrawal_Awareness")
print(f"\nWithdrawal awareness [χ²({dof})={chi2:.2f}, p={p:.3f}]:")
for w, lbl in {0:"Aware",1:"Not aware"}.items():
    nw = (df["Withdrawal_Awareness"] == w).sum()
    bw = (df[df.Farm_Type==0]["Withdrawal_Awareness"] == w).sum()
    lw = (df[df.Farm_Type==1]["Withdrawal_Awareness"] == w).sum()
    sw = (df[df.Farm_Type==2]["Withdrawal_Awareness"] == w).sum()
    print(f"  {lbl:<35} {pct(nw,n):<15} {pct(bw,109):<14} {pct(lw,70):<13} {pct(sw,33)}")

# AM users only rows
print(f"\n── AM users only (n={n_am}) ──")
FT_AM_N = {0: len(df_am[df_am.Farm_Type==0]),
           1: len(df_am[df_am.Farm_Type==1]),
           2: len(df_am[df_am.Farm_Type==2])}

def am_row(label, col, cat_val, denom_fn=None):
    ov = (df_am[col] == cat_val).sum()
    bv = (df_am[df_am.Farm_Type==0][col] == cat_val).sum()
    lv = (df_am[df_am.Farm_Type==1][col] == cat_val).sum()
    sv = (df_am[df_am.Farm_Type==2][col] == cat_val).sum()
    d0,d1,d2 = FT_AM_N[0], FT_AM_N[1], FT_AM_N[2]
    print(f"  {label:<35} {pct(ov,n_am):<15} {pct(bv,d0):<14} {pct(lv,d1):<13} {pct(sv,d2)}")

# Prescriber
chi2, dof, p = chi2_test(df_am, "Prescriber_of_AM")
print(f"\nPrescriber [χ²({dof})={chi2:.2f}, p={p:.3f}{sig_stars(p)}]:")
for pr, lbl in {0:"Veterinarian only",1:"Non-veterinarian",2:"Self / feed agent"}.items():
    am_row(lbl, "Prescriber_of_AM", pr)

# Non-prescription purchase
chi2, dof, p = chi2_test(df_am, "AM_Without_Rx")
print(f"\nNon-Rx purchase [χ²({dof})={chi2:.2f}, p={p:.3f}{sig_stars(p)}]:")
am_row("AM without prescription (Yes)", "AM_Without_Rx", 0)

# WHO AWaRe
print(f"\nWHO AWaRe category (worst-case per farm):")
for cat, lbl in {"Access":"Access","Watch":"Watch (priority)","Reserve":"Reserve (last resort)†"}.items():
    ov = (df_am["AWaRE_Label"] == cat).sum()
    bv = (df_am[df_am.Farm_Type==0]["AWaRE_Label"] == cat).sum()
    lv = (df_am[df_am.Farm_Type==1]["AWaRE_Label"] == cat).sum()
    sv = (df_am[df_am.Farm_Type==2]["AWaRE_Label"] == cat).sum()
    print(f"  {lbl:<35} {pct(ov,n_am):<15} {pct(bv,FT_AM_N[0]):<14} "
          f"{pct(lv,FT_AM_N[1]):<13} {pct(sv,FT_AM_N[2])}")
print("  †Reserve includes Colistin (reclassified Access→Reserve per WHO AWaRe 2023)")

# Risky AMU behaviours
print(f"\nRisky AMU behaviours:")
behaviours = [
    ("Polytherapy (≥2 concurrent AMs)", "Number_of_AM", 2, ">="),
    ("Poor withdrawal adherence (Sometimes/Never)", "Withdrawal_Practice", 1, ">="),
    ("Growth promoter use (Yes)", "AM_Growth_Promoter", 0, "=="),
    ("Reuse leftover antibiotics (Yes)", "Reuse_Leftover", 0, "=="),
]
for lbl, col, val, op in behaviours:
    if col not in df_am.columns: continue
    if op == "==":
        ov = (df_am[col] == val).sum()
        bv = (df_am[df_am.Farm_Type==0][col] == val).sum()
        lv = (df_am[df_am.Farm_Type==1][col] == val).sum()
        sv = (df_am[df_am.Farm_Type==2][col] == val).sum()
    else:
        ov = (df_am[col] >= val).sum()
        bv = (df_am[df_am.Farm_Type==0][col] >= val).sum()
        lv = (df_am[df_am.Farm_Type==1][col] >= val).sum()
        sv = (df_am[df_am.Farm_Type==2][col] >= val).sum()
    chi2, dof, p = (np.nan, np.nan, np.nan)
    try:
        chi2, dof, p = chi2_test(df_am, col)
    except Exception:
        pass
    p_str = sig_stars(p) if not np.isnan(p) else ""
    print(f"  {lbl:<45} {pct(ov,n_am):<15} {pct(bv,FT_AM_N[0]):<14} "
          f"{pct(lv,FT_AM_N[1]):<13} {pct(sv,FT_AM_N[2])}  {p_str}")

# ══════════════════════════════════════════════════════════════════
# TABLE 3: KAPP Score Distributions
# ══════════════════════════════════════════════════════════════════
print("\n─────────────────────────────────────────────")
print("TABLE 3: KAPP COMPOSITE SCORE DISTRIBUTIONS")
print("─────────────────────────────────────────────")
print(f"\n{'Score':<35} {'Stat':<12} {'Total':<20} {'Broiler':<20} {'Layer':<20} {'Sonali':<20} {'KW p':<10} {'Post-hoc'}")
print("-" * 140)

KAPP_SCORES = [
    ("Knowledge Score (0–6)", "Knowledge_Score"),
    ("Attitude Score (0–4)", "Attitude_Score"),
    ("Practice Score Adjusted (0–30)", "Practice_Score_Adjusted"),
    ("Performance Score (0–3)", "Performance_Score"),
    ("AMR Risk Index (0–8)", "AMR_Risk_Index"),
]

for label, col in KAPP_SCORES:
    if col not in df.columns: continue
    h, p = kw_test(df, col)
    posthoc = ""
    if p < 0.05:
        ph = dunns_posthoc(df, col)
        sig_pairs = [f"{FT_MAP[g1][0]}>{FT_MAP[g2][0]}"
                     for (g1,g2),v in ph.items()
                     if v["sig"] and df[df.Farm_Type==g1][col].median() > df[df.Farm_Type==g2][col].median()]
        posthoc = "; ".join(sig_pairs) if sig_pairs else "sig pairs n.s. post-hoc"

    # Mean±SD row
    print(f"  {label:<35} {'Mean±SD':<12} {msd(df[col]):<20} "
          f"{msd(df[df.Farm_Type==0][col]):<20} "
          f"{msd(df[df.Farm_Type==1][col]):<20} "
          f"{msd(df[df.Farm_Type==2][col]):<20} "
          f"{'<0.001' if p<0.001 else f'{p:.3f}'}{sig_stars(p):<5} {posthoc}")

    # Median(IQR) row
    print(f"  {'':35} {'Med(IQR)':<12} {miqr(df[col]):<20} "
          f"{miqr(df[df.Farm_Type==0][col]):<20} "
          f"{miqr(df[df.Farm_Type==1][col]):<20} "
          f"{miqr(df[df.Farm_Type==2][col])}")

print("\n── AMR Risk Index distribution (primary outcome) ──")
for cat, lo, hi in [("Low (0–2)", 0, 2), ("Moderate (3–4)", 3, 4), ("High (≥5)", 5, 8)]:
    mask = (df["AMR_Risk_Index"] >= lo) & (df["AMR_Risk_Index"] <= hi)
    n_cat = mask.sum()
    n_b = ((df["Farm_Type"]==0) & mask).sum()
    n_l = ((df["Farm_Type"]==1) & mask).sum()
    n_s = ((df["Farm_Type"]==2) & mask).sum()
    star = " ← primary outcome" if cat == "High (≥5)" else ""
    print(f"  {cat+star:<40} {pct(n_cat, n):<15} "
          f"{pct(n_b,109):<14} {pct(n_l,70):<13} {pct(n_s,33)}")

print("\n✅ Step 3 complete. V13 canonical: High=27(12.7%), Mod=71(33.5%), Low=114(53.8%)")
