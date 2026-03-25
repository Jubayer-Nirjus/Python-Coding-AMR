"""
Figure 6 — PARM: ROC Curves & Feature Importance
Run: python figure6_parm_roc.py
Output: Figure6_PARM_ROC.png

Requirements:
    pip install pandas numpy matplotlib scikit-learn openpyxl
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
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──
EXCEL_FILE = 'Master_Analysis_Workbook.xlsx'
OUTPUT     = 'Figure6_PARM_ROC.png'

COLORS = {
    'lr'   : '#1F4E79',
    'rf'   : '#375623',
    'gb'   : '#C55A11',
    'diag' : '#AAAAAA',
    'bg'   : '#FAFAFA',
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

# ── LOAD DATA ──
df    = pd.read_excel(EXCEL_FILE, sheet_name='1_Master_Data', header=1)
df_sc = pd.read_excel(EXCEL_FILE, sheet_name='3_Scores_Summary', header=1)
df    = df.replace(99, np.nan)

df['Knowledge_Score']  = pd.to_numeric(df_sc['Knowledge_Score'],  errors='coerce')
df['Biosecurity_Score']= pd.to_numeric(df['Biosecurity_Score'],    errors='coerce')

# ── FEATURE ENGINEERING ──
df['Training_bin'] = (df['Training'] == 0).astype(float)
df['Vet_only']     = (df['Prescriber_of_AM'] == 0).astype(float)
df['FT_Layer']     = (df['Farm_Type'] == 1).astype(float)
df['FT_Sonali']    = (df['Farm_Type'] == 2).astype(float)
df['Edu_ord']      = df['Education'].astype(float)
df['Flock_ord']    = df['Flock_Size'].astype(float)
df['Duration_ord'] = df['Farm_Duration'].astype(float)
df['Auto_bin']     = df['Use_of_Automation'].astype(float)

FEAT_COLS  = ['Knowledge_Score', 'Training_bin', 'Edu_ord', 'Flock_ord',
              'Biosecurity_Score', 'Auto_bin', 'Vet_only',
              'FT_Layer', 'FT_Sonali', 'Duration_ord']
FEAT_NAMES = ['Knowledge Score', 'Training Willing', 'Education',
              'Flock Size', 'Biosecurity Score', 'Automation Use',
              'Vet Prescriber Only', 'Layer Farm', 'Sonali Farm', 'Farm Duration']

sub = df[['AMR_Risk_High'] + FEAT_COLS].dropna()
y   = sub['AMR_Risk_High'].values.astype(float)
X   = sub[FEAT_COLS].values
print(f'n={len(sub)}, Positive (High Risk)={int(y.sum())} ({y.mean():.1%})')

# ── MODELS ──
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler  = StandardScaler()

lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5,
                             random_state=42, class_weight='balanced')
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=42)

y_prob_lr = cross_val_predict(lr, scaler.fit_transform(X), y, cv=cv, method='predict_proba')[:,1]
y_prob_rf = cross_val_predict(rf, X,                       y, cv=cv, method='predict_proba')[:,1]
y_prob_gb = cross_val_predict(gb, X,                       y, cv=cv, method='predict_proba')[:,1]

def get_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return dict(auc=auc, sens=sens, spec=spec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn, fpr=fpr, tpr=tpr)

res_lr = get_metrics(y, y_prob_lr)
res_rf = get_metrics(y, y_prob_rf)
res_gb = get_metrics(y, y_prob_gb)

# Feature importance from RF trained on full data
rf.fit(X, y)
fi = sorted(zip(FEAT_NAMES, rf.feature_importances_), key=lambda x: -x[1])

print(f"\nModel performance:")
for name, res in [('LR', res_lr), ('RF', res_rf), ('GB', res_gb)]:
    print(f"  {name}: AUC={res['auc']:.3f}, Sens={res['sens']:.3f}, "
          f"Spec={res['spec']:.3f}, F1={res['f1']:.3f}")

# ── FIGURE ──
fig = plt.figure(figsize=(15, 6.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.36)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── Panel A: ROC curves ──
ax1.set_facecolor(COLORS['bg'])
ax1.plot([0, 1], [0, 1], '--', color=COLORS['diag'], lw=1.2,
         alpha=0.7, label='Random (AUC=0.500)')

model_specs = [
    ('Logistic Regression', res_lr, COLORS['lr'], '-'),
    ('Random Forest',        res_rf, COLORS['rf'], '-'),
    ('Gradient Boosting',    res_gb, COLORS['gb'], '--'),
]
for name, res, col, ls in model_specs:
    ax1.plot(res['fpr'], res['tpr'], color=col, lw=2.2, linestyle=ls, alpha=0.92,
             label=f'{name}  (AUC = {res["auc"]:.3f})')

ax1.axhline(y=0.80, color='#404040', lw=0.6, ls=':', alpha=0.5)
ax1.text(0.65, 0.81, 'Sensitivity = 0.80', fontsize=8, color='#404040', alpha=0.7)

ax1.set_xlabel('1 − Specificity (False Positive Rate)', fontsize=10.5)
ax1.set_ylabel('Sensitivity (True Positive Rate)', fontsize=10.5)
ax1.set_title('A. ROC Curves — PARM Model Comparison\n'
              f'(5-fold CV, n={len(sub)}, n_pos={int(y.sum())}, prev={y.mean():.1%})',
              fontsize=11, fontweight='bold', color='#1F4E79', loc='left', pad=8)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.05)
ax1.set_xticks(np.arange(0, 1.1, 0.2))
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.95,
           edgecolor='#BDD7EE', title='Model', title_fontsize=9)

# Best AUC badge
best_model, best_auc = max(
    [('Logistic Regression', res_lr['auc']),
     ('Random Forest', res_rf['auc']),
     ('Gradient Boosting', res_gb['auc'])],
    key=lambda x: x[1]
)
ax1.text(0.02, 0.98,
         f'Best AUC = {best_auc:.3f}\n({best_model})',
         transform=ax1.transAxes, fontsize=9.5, va='top', fontweight='bold',
         color='#375623',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#E2EFDA',
                   edgecolor='#375623', linewidth=1.2, alpha=0.97))

# Performance inset table
perf_text = (
    f"{'Model':<22}{'AUC':>6}{'Sens':>6}{'Spec':>6}{'F1':>6}\n"
    f"{'─'*46}\n"
    f"{'Logistic Regression':<22}{res_lr['auc']:>6.3f}"
    f"{res_lr['sens']:>6.3f}{res_lr['spec']:>6.3f}{res_lr['f1']:>6.3f}\n"
    f"{'Random Forest':<22}{res_rf['auc']:>6.3f}"
    f"{res_rf['sens']:>6.3f}{res_rf['spec']:>6.3f}{res_rf['f1']:>6.3f}\n"
    f"{'Gradient Boosting':<22}{res_gb['auc']:>6.3f}"
    f"{res_gb['sens']:>6.3f}{res_gb['spec']:>6.3f}{res_gb['f1']:>6.3f}"
)
ax1.text(0.35, 0.40, perf_text, transform=ax1.transAxes,
         fontsize=8, fontfamily='monospace', va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#BDD7EE', linewidth=1, alpha=0.97))

# ── Panel B: Feature Importance ──
names = [n for n, _ in fi][::-1]
imps  = [v for _, v in fi][::-1]
y_pos = np.arange(len(names))

bar_colors = []
for imp in imps:
    if imp >= 0.20:   bar_colors.append('#C00000')
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
ax2.set_xlim(0, 0.35)
ax2.axvline(x=0, color='#404040', lw=0.8)
ax2.set_title('B. Random Forest Feature Importance\n'
              '(PARM — Predictor Ranking for High AMR Risk)',
              fontsize=11, fontweight='bold', color='#1F4E79', loc='left', pad=8)

legend_els = [
    Patch(facecolor='#C00000', label='High importance (>0.20)'),
    Patch(facecolor='#C55A11', label='Moderate (0.10–0.20)'),
    Patch(facecolor='#2E75B6', label='Low (<0.10)'),
]
ax2.legend(handles=legend_els, loc='lower right', fontsize=8.5,
           framealpha=0.95, edgecolor='#BDD7EE')

# ── Overall title & footnote ──
fig.suptitle(
    'Figure 6 — PARM: Poultry AMR Risk Prediction Model\n'
    'ROC Curves & Feature Importance (Proof-of-Concept, n=211)',
    fontsize=13, fontweight='bold', color='#1F4E79', y=1.02,
)
fig.text(
    0.01, -0.03,
    'Note: 5-fold stratified cross-validation. Outcome: High AMR Risk Index (≥5). '
    'AUC ≥0.80 = good discrimination. '
    'PARM is exploratory/proof-of-concept — external validation required before deployment.',
    fontsize=8.5, color='#767676',
)

plt.savefig(OUTPUT, dpi=300)
plt.close()
print(f'✓ Saved: {OUTPUT}')
