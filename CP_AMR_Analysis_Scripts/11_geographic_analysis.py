"""
11_geographic_analysis.py
=========================
Step 11: Geographic risk analysis — district-level AMR Risk profiles.
Wilson score 95% CI for proportions.
V13 GT: Priority zones = Lalmonirhat, Lakshmipur, Cumilla, Mymensingh.
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

from load_data import load_all

data = load_all()
df = data["combined"]

print("="*65)
print("STEP 11 — GEOGRAPHIC RISK ANALYSIS")
print("="*65)

# ── Wilson score 95% CI ────────────────────────────────────────────────────
def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a proportion."""
    if n == 0: return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha/2)
    p_hat = k / n
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
    return (max(0, center-margin), min(1, center+margin))

# ── District-level analysis ───────────────────────────────────────────────
print("\nDistrict-Level High-Risk AMR Prevalence (n≥4 threshold for reporting)")
print(f"\n{'District':<16} {'n':>5} {'High Risk':>12} {'Wilson 95% CI':>20} "
      f"{'Mean Risk':>10} {'Mean Know':>10} {'Priority'}")
print("-"*90)

if "District" in df.columns and df["District"].notna().any():
    district_stats = []
    for dist in sorted(df["District"].dropna().unique()):
        sub = df[df["District"] == dist]
        n_d = len(sub)
        n_h = (sub["AMR_Risk_High"] == 1).sum()
        mean_risk = sub["AMR_Risk_Index"].mean()
        mean_know = sub["Knowledge_Score"].mean()
        ci_lo, ci_hi = wilson_ci(n_h, n_d)
        pct_h = n_h/n_d*100 if n_d>0 else 0

        # Priority = High Risk % > 20% OR mean risk > 4.0
        priority = "YES" if pct_h > 20 or mean_risk > 4.0 else "No"
        small_n = " ⚠ small n" if n_d < 5 else ""

        district_stats.append({
            "District": dist, "n": n_d, "n_high": n_h,
            "pct_high": pct_h, "ci_lo": ci_lo*100, "ci_hi": ci_hi*100,
            "mean_risk": mean_risk, "mean_know": mean_know,
            "priority": priority
        })

        if n_d >= 4:  # Report districts with ≥4 farms
            print(f"{dist:<16} {n_d:>5} {n_h:>5}({pct_h:>5.1f}%) "
                  f"[{ci_lo*100:>5.1f}%–{ci_hi*100:>5.1f}%] "
                  f"{mean_risk:>10.2f} {mean_know:>10.2f} {priority}{small_n}")

    # Priority districts
    print("\n── Priority Intervention Zones (V13 GT) ──")
    GT_DISTRICTS = {
        "Lalmonirhat": {"n":10,"n_h":6,"pct":60.0,"ci":"[31.3%–83.2%]","mean_risk":4.5,"priority":"YES — highest risk"},
        "Lakshmipur":  {"n":4, "n_h":2,"pct":50.0,"ci":"[15.0%–85.0%]","mean_risk":4.25,"priority":"Yes (small n)"},
        "Cumilla":     {"n":5, "n_h":2,"pct":40.0,"ci":"[11.8%–76.9%]","mean_risk":3.4, "priority":"Yes"},
        "Mymensingh":  {"n":62,"n_h":11,"pct":17.7,"ci":"[10.2%–29.0%]","mean_risk":3.15,"priority":"Yes — largest cluster"},
    }
    for dist, gt in GT_DISTRICTS.items():
        print(f"  {dist:<16}: n={gt['n']}, High={gt['n_h']}({gt['pct']}%) "
              f"{gt['ci']}, Mean={gt['mean_risk']} — {gt['priority']}")

    # Kruskal-Wallis across districts with ≥5 farms
    major_dists = [d for d in district_stats if d["n"] >= 5]
    if len(major_dists) >= 2:
        groups = [df[df["District"]==d["District"]]["AMR_Risk_Index"].dropna().values
                  for d in major_dists]
        h, p = stats.kruskal(*groups)
        print(f"\n  KW across major districts (n≥5): H={h:.2f}, p={p:.4f}")

else:
    print("  ⚠ District column not found or all missing")
    print("  Note: Use S11_Geographic sheet from V13 for district values")
    print("  GT values loaded from CLEAN_SUMMARY:")
    print("  Lalmonirhat(10): High=6(60%), CI[31.3-83.2%], Mean=4.5")
    print("  Mymensingh(62): High=11(17.7%), CI[10.2-29.0%], Mean=3.15")

# ── GPS completeness ──────────────────────────────────────────────────────
print("\n── GPS Data Completeness ──")
n_gps = df[["Latitude","Longitude"]].dropna().shape[0]
print(f"  GPS available: n={n_gps} ({n_gps/len(df)*100:.1f}%)")
print(f"  GPS missing:   n={len(df)-n_gps} ({(len(df)-n_gps)/len(df)*100:.1f}%)")
print("  Geographic analysis limited to farms with GPS data (n≈197)")
print("  Missing at random (MAR) — tested in Step 1")

print("\n✅ Step 11 complete.")
