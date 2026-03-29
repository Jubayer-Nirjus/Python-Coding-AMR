"""
Figure 6 — PARM: ROC Curves & Feature Importance
Updated for CP_AMR_Master_File_V13.xlsx
Output: Figure6_PARM_ROC.png

Ground truth (V13):
  n=212, n_pos=27 (12.7%), EPV=2.7
  Definitive PARM (5-fold CV): RF AUC-ROC=0.716, LR=0.701, GB=0.611
  AUC-PR primary metric (baseline=0.127): RF=0.307, LR=0.302, GB=0.214
  ALL models below AUC-ROC ≥0.75 pre-specified threshold → proof-of-concept framing

Feature coding (V13):
  Knowledge_Score:   from 1_Master_Data col 59
  Training_bin:      Training==0 → 1 (Yes training = protective)
  Education:         0=Graduate, 1=College, 2=Primary (ordinal)
  Flock_Size:        ordinal 0-3
  Biosecurity_Score: 0-6
  Use_of_Automation: 0=No, 1=Yes
  Vet_Prescriber:    Prescriber_of_AM==0 → 1
  Layer_Farm:        Farm_Type==1 → 1
  Sonali_Farm:       Farm_Type==2 → 1
  Farm_Duration:     ordinal 0-3
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                              matthews_corrcoef, brier_score_loss, f1_score)
import warnings
warnings.filterwarnings('ignore')

EXCEL_FILE = 'CP_AMR_Master_File_V13.xlsx'
OUTPUT     = 'Figure6_PARM_ROC.png'

COLORS = {
    'lr'  : '#1F4E79',
    'rf'  : '#375623',
    'gb'  : '#C55A11',
    'diag': '#AAAAAA',
    'bg'  : '#FAFAFA',
}

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.linewidth'    : 0.8,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.25,
})

# ── LOAD ──
df = pd.read_excel(EXCEL_FILE, sheet_name='1_Master_Data', header=1)
df = df.dropna(subset=['Unique_ID'])
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ── FEATURE ENGINEERING ──
df['Training_bin'] = (df['Training'] == 0).astype(float)    # 0=Yes training
df['Vet_only']     = (df['Prescriber_of_AM'] == 0).astype(float)
df['FT_Layer']     = (df['Farm_Type'] == 1).astype(float)
df['FT_Sonali']    = (df['Farm_Type'] == 2).astype(float)

FEAT_COLS = ['Knowledge_Score', 'Training_bin', 'Education', 'Flock_Size',
             'Biosecurity_Score', 'Use_of_Automation', 'Vet_only',
             'FT_Layer', 'FT_Sonali', 'Farm_Duration']
FEAT_NAMES = ['Knowledge Score', 'Training Willing', 'Education',
              'Flock Size', 'Biosecurity Score', 'Automation Use',
              'Vet Prescriber Only', 'Layer Farm', 'Sonali Farm', 'Farm Duration']

sub = df[['AMR_Risk_High'] + FEAT_COLS].dropna()
y   = sub['AMR_Risk_High'].values.astype(float)
X   = sub[FEAT_COLS].values

print(f'n={len(sub)}, Positive (High Risk ≥5) = {int(y.sum())} ({y.mean():.1%})')
print(f'EPV = {int(y.sum())}/10 = {y.sum()/10:.1f}  (below EPV≥10 threshold)')

# ── MODELS ──
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()

lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

y_prob_lr = cross_val_predict(lr, scaler.fit_transform(X), y, cv=cv, method='predict_proba')[:,1]
y_prob_rf = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')[:,1]
y_prob_gb = cross_val_predict(gb, X, y, cv=cv, method='predict_proba')[:,1]

def get_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    mcc     = matthews_corrcoef(y_true, y_pred)
    brier   = brier_score_loss(y_true, y_prob)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return dict(auc_roc=auc_roc, auc_pr=auc_pr, mcc=mcc,
                brier=brier, f1=f1, fpr=fpr, tpr=tpr)

res_lr = get_metrics(y, y_prob_lr)
res_rf = get_metrics(y, y_prob_rf)
res_gb = get_metrics(y, y_prob_gb)

# Feature importance from RF trained on full data
rf.fit(X, y)
fi = sorted(zip(FEAT_NAMES, rf.feature_importances_), key=lambda x: -x[1])

baseline_auc_pr = y.mean()  # no-skill baseline
print(f'\nBaseline AUC-PR: {baseline_auc_pr:.3f}')
for name, res in [('LR', res_lr), ('RF', res_rf), ('GB', res_gb)]:
    print(f'  {name}: AUC-ROC={res["auc_roc"]:.3f}, AUC-PR={res["auc_pr"]:.3f}, '
          f'MCC={res["mcc"]:.3f}, Brier={res["brier"]:.3f}, F1={res["f1"]:.3f}')

# ── FIGURE ──
fig = plt.figure(figsize=(15, 6.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.36)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── Panel A: ROC curves ──
ax1.set_facecolor(COLORS['bg'])
ax1.plot([0, 1], [0, 1], '--', color=COLORS['diag'], lw=1.2, alpha=0.7,
         label='Random (AUC=0.500)')

model_specs = [
    ('Logistic Regression', res_lr, COLORS['lr'], '-'),
    ('Random Forest ★',      res_rf, COLORS['rf'], '-'),
    ('Gradient Boosting',    res_gb, COLORS['gb'], '--'),
]
for name, res, col, ls in model_specs:
    ax1.plot(res['fpr'], res['tpr'], color=col, lw=2.2, linestyle=ls, alpha=0.92,
             label=f'{name}  (AUC-ROC={res["auc_roc"]:.3f})')

# Pre-specified threshold line
ax1.axvline(x=0.75, color='#404040', lw=0.6, ls=':', alpha=0)
ax1.axhline(y=0.75, color='#404040', lw=0.6, ls=':', alpha=0.5)
ax1.text(0.65, 0.76, 'AUC≥0.75 threshold (ns achieved)',
         fontsize=7.5, color='#C00000', alpha=0.85)

ax1.set_xlabel('1 − Specificity (False Positive Rate)', fontsize=10.5)
ax1.set_ylabel('Sensitivity (True Positive Rate)', fontsize=10.5)
ax1.set_title(
    'A. ROC Curves — PARM Model Comparison\n'
    f'(5-fold CV, n={len(sub)}, n_pos={int(y.sum())}, prev={y.mean():.1%})',
    fontsize=11, fontweight='bold', color='#1F4E79', loc='left', pad=8)
ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.05)
ax1.set_xticks(np.arange(0, 1.1, 0.2))
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.95,
           edgecolor='#BDD7EE', title='Model', title_fontsize=9)

# Warning badge — all below threshold
ax1.text(
    0.02, 0.98,
    f'⚠ All models below AUC≥0.75\nRF best: AUC-ROC={res_rf["auc_roc"]:.3f}\n'
    f'RF AUC-PR={res_rf["auc_pr"]:.3f} ({res_rf["auc_pr"]/baseline_auc_pr:.1f}× baseline)\n'
    f'PROOF-OF-CONCEPT ONLY\nEPV={int(y.sum())}/10={y.sum()/10:.1f} (<<10)',
    transform=ax1.transAxes, fontsize=8.5, va='top', fontweight='bold',
    color='#9C0006',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFEBEE',
              edgecolor='#C00000', linewidth=1.5, alpha=0.97))

# Performance inset table
perf_text = (
    f"{'Model':<22}{'AUC-ROC':>8}{'AUC-PR':>8}{'MCC':>7}{'Brier':>7}{'F1':>6}\n"
    f"{'─'*56}\n"
    f"{'Logistic Regression':<22}{res_lr['auc_roc']:>8.3f}{res_lr['auc_pr']:>8.3f}"
    f"{res_lr['mcc']:>7.3f}{res_lr['brier']:>7.3f}{res_lr['f1']:>6.3f}\n"
    f"{'Random Forest ★':<22}{res_rf['auc_roc']:>8.3f}{res_rf['auc_pr']:>8.3f}"
    f"{res_rf['mcc']:>7.3f}{res_rf['brier']:>7.3f}{res_rf['f1']:>6.3f}\n"
    f"{'Gradient Boosting':<22}{res_gb['auc_roc']:>8.3f}{res_gb['auc_pr']:>8.3f}"
    f"{res_gb['mcc']:>7.3f}{res_gb['brier']:>7.3f}{res_gb['f1']:>6.3f}\n"
    f"{'Baseline (no-skill)':<22}{'—':>8}{baseline_auc_pr:>8.3f}"
)
ax1.text(0.33, 0.38, perf_text, transform=ax1.transAxes,
         fontsize=7.5, fontfamily='monospace', va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#BDD7EE', linewidth=1, alpha=0.97))

# ── Panel B: Feature Importance ──
names = [n for n, _ in fi][::-1]
imps  = [v for _, v in fi][::-1]
y_pos = np.arange(len(names))

bar_colors = []
for imp in imps:
    if   imp >= 0.20: bar_colors.append('#C00000')
    elif imp >= 0.10: bar_colors.append('#C55A11')
    else:             bar_colors.append('#2E75B6')

ax2.barh(y_pos, imps, color=bar_colors, edgecolor='white',
         linewidth=0.5, height=0.65, alpha=0.88)
for i, (imp, yp) in enumerate(zip(imps, y_pos)):
    ax2.text(imp + 0.003, yp, f'{imp:.3f}',
             va='center', fontsize=9, color='#404040', fontweight='bold')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(names, fontsize=10)
ax2.set_xlabel('Feature Importance (Mean Decrease Impurity)', fontsize=10.5)
ax2.set_xlim(0, 0.38)
ax2.axvline(x=0, color='#404040', lw=0.8)
ax2.set_title(
    'B. Random Forest Feature Importance\n'
    '(PARM — ⚠ EPV=2.7, rankings unstable)',
    fontsize=11, fontweight='bold', color='#1F4E79', loc='left', pad=8)

legend_els = [
    Patch(facecolor='#C00000', label='High importance (>0.20)'),
    Patch(facecolor='#C55A11', label='Moderate (0.10–0.20)'),
    Patch(facecolor='#2E75B6', label='Low (<0.10)'),
]
ax2.legend(handles=legend_els, loc='lower right', fontsize=8.5,
           framealpha=0.95, edgecolor='#BDD7EE')

ax2.text(0.98, 0.02,
         '⚠ EPV=2.7 < 10\nRankings indicative only',
         transform=ax2.transAxes, ha='right', va='bottom',
         fontsize=8, color='#9C0006',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                   edgecolor='#C00000', linewidth=1, alpha=0.9))

fig.suptitle(
    'Figure 6 — PARM: Poultry AMR Risk Prediction Model\n'
    f'ROC Curves & Feature Importance (Proof-of-Concept, n={len(sub)}, n_pos={int(y.sum())})',
    fontsize=13, fontweight='bold', color='#1F4E79', y=1.02,
)
fig.text(
    0.01, -0.03,
    f'Note: 5-fold stratified CV. Outcome: High AMR Risk Index (≥5), n_pos={int(y.sum())} ({y.mean():.1%}). '
    f'EPV={y.sum()/10:.1f} (below EPV≥10). '
    'AUC-PR = primary metric (class imbalance). '
    f'Baseline AUC-PR = {baseline_auc_pr:.3f}. '
    'All models below pre-specified AUC-ROC≥0.75 — external validation required. '
    'Source: CP_AMR_Master_File_V13.xlsx',
    fontsize=8.5, color='#767676',
)

plt.savefig(OUTPUT, dpi=300)
plt.close()
print(f'✓ Saved: {OUTPUT}')
