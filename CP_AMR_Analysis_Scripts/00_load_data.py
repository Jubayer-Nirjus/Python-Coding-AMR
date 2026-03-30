"""
00_load_data.py
===============
Data loading utility — imported by every analysis script.
Study: AMU/AMR/KAP in Commercial Poultry Farms, Bangladesh (n=212)
Journal target: Preventive Veterinary Medicine (Elsevier Q1)

Usage:
    from load_data import load_all, PATHS, FT_MAP, SCORE_COLS, RISK_COLS
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── File path ──────────────────────────────────────────────────────────────
MASTER_FILE = "CP_AMR_Master_File_V13.xlsx"

# ── Canonical ground-truth values (V13 CLEAN_SUMMARY) ─────────────────────
GT = {
    "n_total": 212, "n_broiler": 109, "n_layer": 70, "n_sonali": 33,
    "n_am_users": 165, "n_non_users": 47,
    "n_high": 27, "n_moderate": 71, "n_low": 114,
    "pct_high": 12.7, "pct_moderate": 33.5, "pct_low": 53.8,
    "mean_amr": 2.34, "sd_amr": 1.82,
    "access": 49, "watch": 96, "reserve": 20,
    "parm_n_pos": 27, "parm_epv": 2.7,
    "parm_lr_roc": 0.576, "parm_rf_roc": 0.640, "parm_gb_roc": 0.586,
    "parm_baseline_pr": 0.127,
    "firth_layer_A_aOR": 4.749,
    "firth_knowl_A_aOR": 1.597,
    "firth_knowl_B_aOR": 0.607,
    "mediation_indirect": 0.141,
    "mediation_pct": 115.4,
}

# ── Farm type mapping ──────────────────────────────────────────────────────
FT_MAP = {0: "Broiler", 1: "Layer", 2: "Sonali"}

# ── Score columns ──────────────────────────────────────────────────────────
SCORE_COLS = [
    "Knowledge_Score", "Attitude_Score", "Practice_Score_Adjusted",
    "Performance_Score", "AMR_Risk_Index",
]

# ── AMR Risk Index components (8 binary) ──────────────────────────────────
RISK_COLS = [
    "AM_Without_Rx",   # 0=risky, recode to 1=risky
    "AWaRE_Category",  # 0=Access(low), 1=Watch(medium), 2=Reserve(high) → binary ≥1
    "Number_of_AM",    # 0=None,1=Mono,2=Dual,3=Poly → binary ≥2
    "Withdrawal_Practice",    # 0=Always(safe), 1/2=risk → binary
    "AM_Growth_Promoter",     # 0=Yes(risky) → binary
    "Reuse_Leftover",         # 0=Yes(risky) → binary
    "Prescriber_of_AM",       # 0=Vet(safe),1/2=risk → binary
    "Leftover_Storage",       # 0=OK,1=risk → binary
]

# ── Regression predictors ─────────────────────────────────────────────────
PREDICTORS = [
    "Gender", "Age_Group", "Education", "Farm_Type",
    "Flock_Size", "Farm_Duration", "Training",
    "Knowledge_Score", "Heard_of_AMR",
]
PREDICTORS_OLS_EXTRA = ["Attitude_Score"]


def load_all(filepath: str = MASTER_FILE, sentinel: int = 99) -> dict:
    """
    Load and return all analysis-ready DataFrames.
    Replaces sentinel value (99=structural N/A) with NaN.

    Returns dict with keys:
        'main'   : 1_Master_Data (n=212)
        'scores' : 3_Scores_Summary (n=212) — merged scores
        'amr'    : 5_AMR_Risk_Index (component-level)
        'reg'    : 6_Regression_Ready (all outcomes + predictors)
        'practice': 4_Practice_Detail (item-level)
        'rel'    : S2a_Reliability
        'efa'    : S2b_EFA
        'geo'    : S11_Geographic
        'cluster': S9_Cluster
        'parm'   : S10_PARM
        'sens'   : S12_Sensitivity
    """
    def _load(sheet, header=1):
        df = pd.read_excel(filepath, sheet_name=sheet, header=header)
        for c in df.select_dtypes(include=[np.number]).columns:
            df[c] = df[c].replace(sentinel, np.nan)
        return df

    main = _load("1_Master_Data")
    scores = _load("3_Scores_Summary")
    amr = _load("5_AMR_Risk_Index")
    reg = _load("6_Regression_Ready")
    practice = _load("4_Practice_Detail")
    rel = _load("S2a_Reliability")
    efa = _load("S2b_EFA")
    geo = _load("S11_Geographic")
    cluster = _load("S9_Cluster")
    parm = _load("S10_PARM")
    sens = _load("S12_Sensitivity")

    # Merge scores into main for convenience
    merge_cols = [c for c in scores.columns if c not in main.columns or c == "Unique_ID"]
    combined = main.merge(scores[["Unique_ID"] + [c for c in merge_cols if c != "Unique_ID"]],
                          on="Unique_ID", how="left")

    # Derived convenience columns
    combined["FT_Label"] = combined["Farm_Type"].map(FT_MAP)
    combined["AM_user"] = (combined["AM_use_binary"] == 0).astype(int)

    return {
        "main": main,
        "scores": scores,
        "combined": combined,
        "amr": amr,
        "reg": reg,
        "practice": practice,
        "rel": rel,
        "efa": efa,
        "geo": geo,
        "cluster": cluster,
        "parm": parm,
        "sens": sens,
    }


def verify_ground_truth(df: pd.DataFrame, verbose: bool = True) -> bool:
    """Quick sanity check that the loaded data matches V13 canonical values."""
    checks = [
        ("n_total", len(df) == GT["n_total"]),
        ("n_broiler", (df["Farm_Type"] == 0).sum() == GT["n_broiler"]),
        ("n_layer",   (df["Farm_Type"] == 1).sum() == GT["n_layer"]),
        ("n_sonali",  (df["Farm_Type"] == 2).sum() == GT["n_sonali"]),
        ("n_am_users",(df["AM_use_binary"] == 0).sum() == GT["n_am_users"]),
        ("n_high",    (df["AMR_Risk_High"] == 1).sum() == GT["n_high"]),
        ("n_low",     (df["AMR_Risk_Cat"] == "Low").sum() == GT["n_low"]),
        ("n_moderate",(df["AMR_Risk_Cat"] == "Moderate").sum() == GT["n_moderate"]),
        ("access",    (df["AWaRE_Label"] == "Access").sum() == GT["access"]),
        ("watch",     (df["AWaRE_Label"] == "Watch").sum() == GT["watch"]),
        ("reserve",   (df["AWaRE_Label"] == "Reserve").sum() == GT["reserve"]),
    ]
    all_ok = True
    for label, result in checks:
        sym = "✅" if result else "❌"
        if verbose:
            print(f"  {sym} {label}")
        if not result:
            all_ok = False
    return all_ok


if __name__ == "__main__":
    print("Loading V13 data...")
    data = load_all()
    df = data["main"]
    print(f"n={len(df)} | columns={len(df.columns)}")
    print("\nGround-truth verification:")
    ok = verify_ground_truth(df)
    print(f"\n{'✅ ALL PASS' if ok else '❌ ISSUES FOUND'}")
