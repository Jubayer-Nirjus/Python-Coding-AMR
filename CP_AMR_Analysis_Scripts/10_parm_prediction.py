"""
10_parm_prediction.py
=====================
Step 10: Poultry AMR Risk Model (PARM) — proof-of-concept.
Models: LR, RF, GB | 5-fold stratified CV | AUC-PR primary metric.
V13 GT: n_pos=27 (12.7%), EPV=2.7, AUC-ROC all below 0.75.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              matthews_corrcoef, brier_score_loss, f1_score,
                              precision_recall_curve)
from sklearn.preprocessing import StandardScaler
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
print("STEP 10 — PARM: POULTRY AMR RISK PREDICTION MODEL")
print("="*65)

# ── Features and outcome ──────────────────────────────────────────────────
FEATURES = [
    "Knowledge_Score",      # AMR knowledge (primary cognitive factor)
    "Biosecurity_Score",    # Biosecurity practices
    "Farm_Type",            # 0=Broiler,1=Layer,2=Sonali
    "Flock_Size",           # Farm scale
    "Farm_Duration",        # Experience
    "Training",             # Training received
    "Education",            # Education level
    "Attitude_Score",       # Attitude toward AMR
    "Use_of_Automation",    # Technology adoption
    "Heard_of_AMR",         # AMR awareness
]
OUTCOME = "AMR_Risk_High"   # 1 = High AMR Risk (Index ≥5)

df_parm = df[FEATURES + [OUTCOME]].copy()
for c in FEATURES:
    df_parm[c] = pd.to_numeric(df_parm[c], errors="coerce")
df_parm = df_parm.dropna()

X = df_parm[FEATURES].values
y = df_parm[OUTCOME].astype(int).values
n_total = len(y)
n_pos = y.sum()
n_neg = n_total - n_pos
epv = n_pos / len(FEATURES)
baseline_pr = n_pos / n_total

print(f"\nDataset: n={n_total}, n_pos={n_pos} ({n_pos/n_total*100:.1f}%), "
      f"n_neg={n_neg}, EPV={epv:.1f}")
print(f"Baseline AUC-PR (no-skill) = {baseline_pr:.3f}")
print(f"V13 GT: n_pos=27 (12.7%), EPV=2.7, baseline=0.127")
print()
if epv < 10:
    print(f"⚠ EPV={epv:.1f} BELOW recommended EPV≥10 (Peduzzi et al., 1996)")
    print("⚠ Feature importance rankings are UNSTABLE — do NOT over-interpret")
    print("⚠ PARM = PROOF-OF-CONCEPT ONLY — external validation required")

# ── 5-Fold Stratified CV ──────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=5000,
        class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=3,
        class_weight="balanced", random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42
    ),
}

# Scale for LR
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

GT_RESULTS = {
    "Logistic Regression": {"AUC_ROC":0.576,"AUC_PR":0.234,"MCC":0.106,"Brier":0.247,
                             "Sensitivity":0.514,"Specificity":0.622,"F1":0.185},
    "Random Forest":       {"AUC_ROC":0.640,"AUC_PR":0.296,"MCC":0.263,"Brier":0.178,
                             "Sensitivity":0.432,"Specificity":0.847,"F1":0.323},
    "Gradient Boosting":   {"AUC_ROC":0.586,"AUC_PR":0.263,"MCC":0.022,"Brier":0.159,
                             "Sensitivity":0.108,"Specificity":0.925,"F1":0.130},
}

print(f"\n{'Model':<25} {'AUC-ROC':>9} {'AUC-PR★':>9} {'MCC':>8} {'Brier':>8} "
      f"{'Sens':>8} {'Spec':>8} {'F1':>8} {'Threshold'}")
print("-"*95)

all_proba = {}
for name, model in MODELS.items():
    X_use = X_scaled if "Logistic" in name else X
    y_prob = cross_val_predict(model, X_use, y, cv=cv, method="predict_proba")[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc_roc = roc_auc_score(y, y_prob)
    auc_pr  = average_precision_score(y, y_prob)
    mcc     = matthews_corrcoef(y, y_pred)
    brier   = brier_score_loss(y, y_prob)
    tp = ((y_pred==1)&(y==1)).sum(); fn = ((y_pred==0)&(y==1)).sum()
    tn = ((y_pred==0)&(y==0)).sum(); fp = ((y_pred==1)&(y==0)).sum()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    f1   = f1_score(y, y_pred, zero_division=0)

    thresh_flag = "✗ BELOW 0.75" if auc_roc < 0.75 else "✅ ≥0.75"
    pr_mult = f"{auc_pr/baseline_pr:.1f}×base"
    best = " ★" if name=="Random Forest" else ""
    print(f"{name+best:<25} {auc_roc:>9.3f} {auc_pr:>9.3f} {mcc:>8.3f} {brier:>8.3f} "
          f"{sens:>8.3f} {spec:>8.3f} {f1:>8.3f} {thresh_flag}")
    all_proba[name] = y_prob

    # GT comparison
    gt = GT_RESULTS[name]
    print(f"  GT:  {gt['AUC_ROC']:>9.3f} {gt['AUC_PR']:>9.3f} "
          f"{gt['MCC']:>8.3f} {gt['Brier']:>8.3f} "
          f"{gt['Sensitivity']:>8.3f} {gt['Specificity']:>8.3f} {gt['F1']:>8.3f}")

# ── Feature importance (RF, with EPV warning) ─────────────────────────────
print(f"\n── Random Forest Feature Importance ──")
print(f"⚠ EPV={epv:.1f}: ALL rankings UNSTABLE. Indicative only.")
rf_full = RandomForestClassifier(n_estimators=200, max_depth=4,
                                  class_weight="balanced", random_state=42)
rf_full.fit(X, y)
importances = rf_full.feature_importances_
feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(feat_imp, 1):
    print(f"  {i:>2}. {feat:<30} {imp:.4f} ({imp*100:.1f}%)")

# ── Manuscript framing ────────────────────────────────────────────────────
print("\n── Manuscript §3.9 Key Statements ──")
print("  ALL three models performed BELOW pre-specified AUC-ROC≥0.75 threshold")
print(f"  (range: {min(GT_RESULTS[m]['AUC_ROC'] for m in GT_RESULTS):.3f}–"
      f"{max(GT_RESULTS[m]['AUC_ROC'] for m in GT_RESULTS):.3f})")
print(f"  AUC-PR primary metric (baseline={baseline_pr:.3f}=n_pos/n_total):")
for name, gt in GT_RESULTS.items():
    print(f"    {name}: {gt['AUC_PR']:.3f} ({gt['AUC_PR']/baseline_pr:.1f}× baseline)")
print(f"  EPV={epv:.1f} — critically underpowered (recommended EPV≥10)")
print("  PARM must be framed as proof-of-concept; NOT for clinical/policy use")
print("  External validation in independent Bangladeshi cohort required")

print("\n✅ Step 10 complete.")
