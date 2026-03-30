"""
Figure 6 — PARM: ROC Curves & Feature Importance
Preventive Veterinary Medicine (Elsevier) — journal-compliant output

Journal requirements applied:
  - No figure title on figure (caption only)
  - PNG, 300 dpi, 15×6.5 in = 4500×1950 px ✓
  - Wong (2011) colorblind-safe palette
  - Output: Figure_6.png

CAPTION (use in manuscript):
  Fig. 6. Poultry AMR Risk Model (PARM) performance and predictor importance.
  (A) Receiver operating characteristic (ROC) curves for three machine learning classifiers
  predicting high AMR Risk Index (≥5) under 5-fold stratified cross-validation (n = 212,
  n positive = 27, prevalence = 12.7%). AUC-PR = area under the precision–recall curve;
  baseline AUC-PR = 0.127. All models fell below the pre-specified AUC-ROC ≥ 0.75
  threshold; PARM is therefore exploratory (proof-of-concept) only. EPV = events per
  variable = 2.7 (well below EPV ≥ 10). External validation is required before any
  operational application. (B) Random Forest feature importance scores (mean decrease
  impurity); rankings are unstable at EPV = 2.7 and should be interpreted with caution.
  LR = logistic regression; RF = random forest; GB = gradient boosting; AUC-ROC = area
  under the receiver operating characteristic curve; MCC = Matthews correlation coefficient.
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
OUTPUT     = 'Figure_6.png'

# Wong (2011) colorblind-safe palette
COLORS = {
    'lr'        : '#0072B2',   # blue
    'rf'        : '#009E73',   # green
    'gb'        : '#E69F00',   # orange
    'diag'      : '#AAAAAA',
    'bg'        : '#FAFAFA',
    'blue'      : '#0072B2',
    'green'     : '#009E73',
    'orange'    : '#E69F00',
    'vermillion': '#D55E00',
    'gray_dark' : '#404040',
    'gray_mid'  : '#767676',
}

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.linewidth'    : 0.8,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.15,
})

# ── LOAD ──
df = pd.read_excel(EXCEL_FILE, sheet_name='1_Master_Data', header=1)
df = df.dropna(subset=['Unique_ID'])
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Training_bin'] = (df['Training'] == 0).astype(float)
df['Vet_only']     = (df['Prescriber_of_AM'] == 0).astype(float)
df['FT_Layer']     = (df['Farm_Type'] == 1).astype(float)
df['FT_Sonali']    = (df['Farm_Type'] == 2).astype(float)

FEAT_COLS  = ['Knowledge_Score', 'Training_bin', 'Education', 'Flock_Size',
              'Biosecurity_Score', 'Use_of_Automation', 'Vet_only',
              'FT_Layer', 'FT_Sonali', 'Farm_Duration']
FEAT_NAMES = ['Knowledge Score', 'Training Willing', 'Education',
              'Flock Size', 'Biosecurity Score', 'Automation Use',
              'Vet Prescriber Only', 'Layer Farm', 'Sonali Farm', 'Farm Duration']

sub = df[['AMR_Risk_High'] + FEAT_COLS].dropna()
y   = sub['AMR_Risk_High'].values.astype(float)
X   = sub[FEAT_COLS].values

print(f'n = {len(sub)}, n_pos = {int(y.sum())} ({y.mean():.1%}), EPV = {y.sum()/10:.1f}')

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()

lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

y_prob_lr = cross_val_predict(lr, scaler.fit_transform(X), y, cv=cv, method='predict_proba')[:,1]
y_prob_rf = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')[:,1]
y_prob_gb = cross_val_predict(gb, X, y, cv=cv, method='predict_proba')[:,1]

def get_metrics(y_true, y_prob):
    y_pred  = (y_prob >= 0.5).astype(int)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    mcc     = matthews_corrcoef(y_true, y_pred)
    brier   = brier_score_loss(y_true, y_prob)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return dict(auc_roc=auc_roc, auc_pr=auc_pr, mcc=mcc, brier=brier, f1=f1, fpr=fpr, tpr=tpr)

res_lr = get_metrics(y, y_prob_lr)
res_rf = get_metrics(y, y_prob_rf)
res_gb = get_metrics(y, y_prob_gb)

rf.fit(X, y)
fi = sorted(zip(FEAT_NAMES, rf.feature_importances_), key=lambda x: -x[1])
baseline_auc_pr = y.mean()

for name, res in [('LR', res_lr), ('RF', res_rf), ('GB', res_gb)]:
    print(f'  {name}: AUC-ROC = {res["auc_roc"]:.3f}, AUC-PR = {res["auc_pr"]:.3f}, '
          f'MCC = {res["mcc"]:.3f}, Brier = {res["brier"]:.3f}')

# ── FIGURE ──
fig = plt.figure(figsize=(15, 6.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.36)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── Panel A: ROC curves ──
ax1.set_facecolor(COLORS['bg'])
ax1.plot([0, 1], [0, 1], '--', color=COLORS['diag'], lw=1.2, alpha=0.7,
         label='No-skill (AUC-ROC = 0.500)')

for name, res, col, ls in [
    ('LR', res_lr, COLORS['lr'], '-'),
    ('RF', res_rf, COLORS['rf'], '-'),
    ('GB',  res_gb, COLORS['gb'], '--'),
]:
    ax1.plot(res['fpr'], res['tpr'], color=col, lw=2.2, linestyle=ls, alpha=0.92,
             label=f'{name}  (AUC-ROC = {res["auc_roc"]:.3f})')

# Pre-specified threshold
ax1.axhline(y=0.75, color=COLORS['gray_dark'], lw=0.6, ls=':', alpha=0.4)
ax1.text(0.98, 0.73, 'AUC-ROC = 0.75 threshold\n(not achieved)',
         fontsize=7.5, color=COLORS['vermillion'], alpha=0.9, va='top', ha='right')

ax1.set_xlabel('1 − Specificity (false positive rate)', fontsize=10.5)
ax1.set_ylabel('Sensitivity (true positive rate)', fontsize=10.5)
ax1.set_title(
    f'A. ROC Curves — PARM (5-fold CV, n = {len(sub)}, n$_{{pos}}$ = {int(y.sum())}, prev = {y.mean():.1%})',
    fontsize=10.5, fontweight='bold', color=COLORS['gray_dark'], loc='left', pad=8)
ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.05)
ax1.set_xticks(np.arange(0, 1.1, 0.2)); ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.95, edgecolor='#BDD7EE',
           title='Model', title_fontsize=9)

# Warning badge
ax1.text(
    0.02, 0.98,
    f'All models below AUC-ROC ≥ 0.75\n'
    f'Best: RF AUC-ROC = {res_rf["auc_roc"]:.3f}\n'
    f'RF AUC-PR = {res_rf["auc_pr"]:.3f} ({res_rf["auc_pr"]/baseline_auc_pr:.1f}× baseline)\n'
    f'EPV = {int(y.sum())}/10 = {y.sum()/10:.1f}  (recommended: ≥10)\n'
    'PROOF-OF-CONCEPT ONLY',
    transform=ax1.transAxes, fontsize=8.5, va='top', fontweight='bold',
    color='#9C0006',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFEBEE',
              edgecolor=COLORS['vermillion'], linewidth=1.5, alpha=0.97))

# Performance table inset
perf_text = (
    f"{'Model':<20}{'AUC-ROC':>8}{'AUC-PR':>8}{'MCC':>7}{'Brier':>7}\n"
    f"{'─'*50}\n"
    f"{'LR':<20}{res_lr['auc_roc']:>8.3f}{res_lr['auc_pr']:>8.3f}{res_lr['mcc']:>7.3f}{res_lr['brier']:>7.3f}\n"
    f"{'RF':<20}{res_rf['auc_roc']:>8.3f}{res_rf['auc_pr']:>8.3f}{res_rf['mcc']:>7.3f}{res_rf['brier']:>7.3f}\n"
    f"{'GB':<20}{res_gb['auc_roc']:>8.3f}{res_gb['auc_pr']:>8.3f}{res_gb['mcc']:>7.3f}{res_gb['brier']:>7.3f}\n"
    f"{'Baseline':<20}{'—':>8}{baseline_auc_pr:>8.3f}"
)
ax1.text(0.32, 0.38, perf_text, transform=ax1.transAxes,
         fontsize=7.5, fontfamily='monospace', va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#CCCCCC', linewidth=1, alpha=0.97))

# ── Panel B: Feature Importance ──
names = [n for n, _ in fi][::-1]
imps  = [v for _, v in fi][::-1]
y_pos = np.arange(len(names))

bar_colors = [COLORS['vermillion'] if v >= 0.20 else
              COLORS['orange']     if v >= 0.10 else
              COLORS['blue']       for v in imps]

ax2.barh(y_pos, imps, color=bar_colors, edgecolor='white',
         linewidth=0.5, height=0.65, alpha=0.88)
for i, (imp, yp) in enumerate(zip(imps, y_pos)):
    ax2.text(imp + 0.003, yp, f'{imp:.3f}',
             va='center', fontsize=9, color=COLORS['gray_dark'], fontweight='bold')

ax2.set_yticks(y_pos); ax2.set_yticklabels(names, fontsize=10)
ax2.set_xlabel('Feature importance (mean decrease impurity)', fontsize=10.5)
ax2.set_xlim(0, 0.38)
ax2.axvline(x=0, color=COLORS['gray_dark'], lw=0.8)
ax2.set_title('B. RF Feature Importance\n(EPV = 2.7; rankings indicative only)',
              fontsize=10.5, fontweight='bold', color=COLORS['gray_dark'], loc='left', pad=8)

legend_els = [
    Patch(facecolor=COLORS['vermillion'], label='High (> 0.20)'),
    Patch(facecolor=COLORS['orange'],     label='Moderate (0.10–0.20)'),
    Patch(facecolor=COLORS['blue'],       label='Low (< 0.10)'),
]
ax2.legend(handles=legend_els, loc='lower right', fontsize=8.5,
           framealpha=0.95, edgecolor='#CCCCCC', title='Importance level', title_fontsize=8)

fig.text(
    0.01, -0.03,
    f'LR = logistic regression; RF = random forest; GB = gradient boosting. '
    f'5-fold stratified CV. n = {len(sub)}, n$_{{pos}}$ = {int(y.sum())} ({y.mean():.1%}), EPV = {y.sum()/10:.1f}. '
    f'AUC-PR baseline (no-skill) = {baseline_auc_pr:.3f}. External validation required.',
    fontsize=8, color=COLORS['gray_mid'],
)

plt.savefig(OUTPUT, dpi=300, format='png')
plt.close()
print(f'✓ Saved: {OUTPUT}')
