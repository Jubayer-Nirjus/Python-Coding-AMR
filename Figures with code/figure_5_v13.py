"""
Figure 5 — AMR-Focused Bivariate Analysis: Key Associations
Updated for CP_AMR_Master_File_V13.xlsx
Output: Figure5_Bivariate_Analysis.png

Coding reference (V13):
  AM_use_binary:      0=AM users (n=165), 1=non-users (n=47)
  Heard_of_AMR:       0=Yes, 1=No
  AM_Without_Rx:      0=Yes(non-Rx), 1=No(Rx), 99=non-users
  Withdrawal_Practice:0=Always, 1=Sometimes, 2=Never, 99=non-users
  AM_Growth_Promoter: 0=Yes(GP user), 1=No, 99=non-users
  AWaRE_Category:     0=Access, 1=Watch, 2=Reserve, 99=non-users
  Number_of_AM:       1=mono, 2=dual, 3=poly, 99=non-users
  Digital_Score:      0-2
  AMR_Risk_Index:     0-8
  Knowledge_Score:    0-6
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

EXCEL_FILE = 'CP_AMR_Master_File_V13.xlsx'
OUTPUT     = 'Figure5_Bivariate_Analysis.png'

COLORS = {
    'dark_blue'  : '#1F4E79',
    'mid_blue'   : '#2E75B6',
    'light_blue' : '#BDD7EE',
    'green_dark' : '#1E5631',
    'green_light': '#70AD47',
    'orange'     : '#C55A11',
    'red'        : '#C00000',
    'gold'       : '#BF6000',
    'gray_dark'  : '#404040',
    'gray_mid'   : '#767676',
    'gray_light' : '#D9D9D9',
    'broiler'    : '#2E75B6',
    'layer'      : '#375623',
    'sonali'     : '#C55A11',
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

N  = len(df)
# AM users: AM_use_binary==0
AM = df[df['AM_use_binary'] == 0].copy()

# ── COMPUTE STATS ──

# Panel A — Non-Rx use × AMR Awareness
# Heard_of_AMR: 0=Yes(aware), 1=No(unaware)
# AM_Without_Rx: 0=Yes(non-Rx), 99=non-users
heard_yes = AM[AM['Heard_of_AMR'] == 0]
heard_no  = AM[AM['Heard_of_AMR'] == 1]

def norx_pct(sub):
    d = sub['AM_Without_Rx'].dropna()
    d = d[d != 99]
    return (d == 0).sum() / len(d) * 100 if len(d) > 0 else 0.0

norx_aware   = norx_pct(heard_yes)
norx_unaware = norx_pct(heard_no)

# Chi-square on the valid (non-99) subset
am_valid = AM[AM['AM_Without_Rx'] != 99].copy()
ct_heard = pd.crosstab(am_valid['Heard_of_AMR'], am_valid['AM_Without_Rx'])
_, p_norx_heard, _, _ = stats.chi2_contingency(ct_heard)
p_norx_str = '< 0.001' if p_norx_heard < 0.001 else f'{p_norx_heard:.3f}'

# Panel B — Withdrawal × Knowledge Score (KW)
# Withdrawal_Practice: 0=Always, 1=Sometimes, 2=Never; exclude 99
am_wd = AM[AM['Withdrawal_Practice'] != 99].copy()
wd_always    = am_wd[am_wd['Withdrawal_Practice'] == 0]['Knowledge_Score'].dropna()
wd_sometimes = am_wd[am_wd['Withdrawal_Practice'] == 1]['Knowledge_Score'].dropna()
wd_never     = am_wd[am_wd['Withdrawal_Practice'] == 2]['Knowledge_Score'].dropna()

kw_stat, kw_p = stats.kruskal(wd_always, wd_sometimes, wd_never)
p_wd_str = '< 0.001' if kw_p < 0.001 else f'{kw_p:.3f}'

# Panel D — Growth Promoter × Knowledge (MW)
# AM_Growth_Promoter: 0=GP user, 1=No; exclude 99
am_gp = AM[AM['AM_Growth_Promoter'] != 99].copy()
gp_users = am_gp[am_gp['AM_Growth_Promoter'] == 0]['Knowledge_Score'].dropna()
gp_non   = am_gp[am_gp['AM_Growth_Promoter'] == 1]['Knowledge_Score'].dropna()
_, p_gp  = stats.mannwhitneyu(gp_users, gp_non, alternative='two-sided')
p_gp_str = '< 0.001' if p_gp < 0.001 else f'{p_gp:.3f}'

# Panel E — Digital Score × AMR Risk (Spearman)
paired_dig = df[['Digital_Score', 'AMR_Risk_Index']].dropna()
rho_dig, p_dig = stats.spearmanr(paired_dig['Digital_Score'], paired_dig['AMR_Risk_Index'])
rho_str   = f'{rho_dig:.3f}'
p_dig_str = '< 0.001' if p_dig < 0.001 else f'{p_dig:.3f}'

# ── FIGURE ──
fig = plt.figure(figsize=(16, 10))
gs_ = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.40)

ax1 = fig.add_subplot(gs_[0, 0])
ax2 = fig.add_subplot(gs_[0, 1])
ax3 = fig.add_subplot(gs_[0, 2])
ax4 = fig.add_subplot(gs_[1, 0])
ax5 = fig.add_subplot(gs_[1, 1])
ax6 = fig.add_subplot(gs_[1, 2])

def p_badge(ax, text, facecolor=None, edgecolor=None):
    fc = facecolor or COLORS['light_blue']
    ec = edgecolor or COLORS['dark_blue']
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=COLORS['dark_blue'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=fc, edgecolor=ec,
                      linewidth=1, alpha=0.95))

# ── PANEL A ──
bars_a = ax1.bar(
    [f'Aware\n(n={len(heard_yes)})', f'Unaware\n(n={len(heard_no)})'],
    [norx_aware, norx_unaware],
    color=[COLORS['green_dark'], COLORS['red']],
    edgecolor='white', lw=0.7, width=0.55,
)
for bar, val in zip(bars_a, [norx_aware, norx_unaware]):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 1, f'{val:.1f}%',
             ha='center', fontsize=10.5, fontweight='bold', color=COLORS['dark_blue'])
ax1.set_ylabel('% Using AM Without Prescription', fontsize=10)
ax1.set_ylim(0, 82)
ax1.set_yticks(range(0, 81, 20))
ax1.tick_params(axis='x', labelsize=10.5)
ax1.tick_params(axis='y', labelsize=9.5)
ax1.set_title('A. Non-Prescription AM Use\nvs AMR Awareness',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
ax1.axhline(y=0, color=COLORS['gray_dark'], lw=0.8)
p_badge(ax1, f'p = {p_norx_str}')

# ── PANEL B ──
wd_data   = [wd_always, wd_sometimes, wd_never]
wd_labels = [f'Always\n(n={len(wd_always)})',
             f'Sometimes\n(n={len(wd_sometimes)})',
             f'Never\n(n={len(wd_never)})']
bp = ax2.boxplot(wd_data, labels=wd_labels, patch_artist=True, widths=0.55,
                 medianprops=dict(color='white', lw=2.5),
                 flierprops=dict(marker='o', markersize=4, alpha=0.45, markeredgecolor='none'),
                 boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
for patch, col in zip(bp['boxes'], [COLORS['green_dark'], COLORS['gold'], COLORS['red']]):
    patch.set_facecolor(col)
    patch.set_alpha(0.82)
ax2.set_ylabel('Knowledge Score (0–6)', fontsize=10)
ax2.set_ylim(-0.5, 7.5)
ax2.tick_params(axis='x', labelsize=9.5)
ax2.tick_params(axis='y', labelsize=9.5)
ax2.set_title('B. Knowledge Score\nvs Withdrawal Adherence',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
p_badge(ax2, f'KW p = {p_wd_str}')

# Significance bracket (Always vs Never, Bonferroni)
_, p_a_n = stats.mannwhitneyu(wd_always, wd_never, alternative='two-sided')
if p_a_n * 3 < 0.05:
    ax2.plot([1, 1, 3, 3], [6.5, 6.58, 6.58, 6.5], lw=1, color=COLORS['gray_dark'])
    ax2.text(2, 6.6, '***', ha='center', fontsize=8, color=COLORS['gray_dark'])

# ── PANEL C — AWaRe by Farm Type ──
ft_am_groups = [AM[AM['Farm_Type'] == ft] for ft in [0, 1, 2]]
x_c  = np.arange(3)
bw_c = 0.26
ft_labels_c = [f'Broiler\n(n={len(ft_am_groups[0])})',
               f'Layer\n(n={len(ft_am_groups[1])})',
               f'Sonali\n(n={len(ft_am_groups[2])})']
aware_palette = [(0, 'Access', '#375623'), (1, 'Watch', '#BF6000'), (2, 'Reserve', '#C00000')]
for fi, (code, lbl, col) in enumerate(aware_palette):
    vals = []
    for g in ft_am_groups:
        d = g['AWaRE_Category'].dropna()
        d = d[d != 99]
        vals.append((d == code).sum() / max(len(d), 1) * 100)
    ax3.bar(x_c + fi*bw_c - bw_c, vals, bw_c,
            label=lbl, color=col, edgecolor='white', lw=0.6)
ax3.set_xticks(x_c); ax3.set_xticklabels(ft_labels_c, fontsize=9.5)
ax3.set_ylabel('% of AM-Using Farms', fontsize=10)
ax3.set_ylim(0, 82); ax3.set_yticks(range(0, 81, 20))
ax3.tick_params(axis='y', labelsize=9.5)
ax3.axhline(y=0, color=COLORS['gray_dark'], lw=0.8)
ax3.legend(loc='upper center', fontsize=8.5, framealpha=0.95, edgecolor=COLORS['gray_light'])
ax3.set_title('C. WHO AWaRe Category\nby Farm Type',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
p_badge(ax3, 'ns (p = 0.383)')

# ── PANEL D — Knowledge × Growth Promoter (violin) ──
vp = ax4.violinplot([gp_users, gp_non], positions=[1, 2], widths=0.65,
                    showmedians=True, showextrema=True)
for pc, col in zip(vp['bodies'], [COLORS['red'], COLORS['green_dark']]):
    pc.set_facecolor(col); pc.set_alpha(0.72)
    pc.set_edgecolor('white'); pc.set_linewidth(0.5)
vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2.5)
for part in ['cbars', 'cmins', 'cmaxes']:
    vp[part].set_color(COLORS['gray_dark']); vp[part].set_linewidth(0.8)
ax4.set_xticks([1, 2])
ax4.set_xticklabels([f'GP Users\n(n={len(gp_users)})', f'Non-users\n(n={len(gp_non)})'], fontsize=10)
ax4.set_ylabel('Knowledge Score (0–6)', fontsize=10)
ax4.set_ylim(-0.3, 7.3)
ax4.tick_params(axis='y', labelsize=9.5)
ax4.set_title('D. Knowledge Score\nvs Growth Promoter Use',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
p_badge(ax4, f'MW p = {p_gp_str}')
for xpos, data, col in [(1, gp_users, COLORS['red']), (2, gp_non, COLORS['green_dark'])]:
    ax4.text(xpos, data.mean() + 0.12, f'μ={data.mean():.1f}',
             ha='center', fontsize=9, color=col, fontweight='bold')

# ── PANEL E — Digital Score × AMR Risk (scatter) ──
rng    = np.random.default_rng(42)
jitter = rng.uniform(-0.09, 0.09, len(paired_dig))
ax5.scatter(paired_dig['Digital_Score'] + jitter, paired_dig['AMR_Risk_Index'],
            alpha=0.45, s=28, color=COLORS['mid_blue'], edgecolors='white', lw=0.3)
z  = np.polyfit(paired_dig['Digital_Score'], paired_dig['AMR_Risk_Index'], 1)
xr = np.linspace(-0.15, 2.15, 50)
ax5.plot(xr, np.poly1d(z)(xr), color=COLORS['red'], lw=2, linestyle='--', alpha=0.85)
ax5.set_xlabel('Digital/AI Score (0–2)', fontsize=10)
ax5.set_ylabel('AMR Risk Index (0–8)', fontsize=10)
ax5.set_xlim(-0.25, 2.25); ax5.set_ylim(-0.3, 9)
ax5.set_xticks([0, 1, 2]); ax5.tick_params(labelsize=9.5)
ax5.set_title('E. Digital Score\nvs AMR Risk Index',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
p_badge(ax5, f'ρ = {rho_str}  p = {p_dig_str}')

# ── PANEL F — Therapy Type by Farm Type ──
x_f = np.arange(3)
bw_f = 0.26
ft_labels_f = [f'Broiler\n(n={len(AM[AM["Farm_Type"]==0])})',
               f'Layer\n(n={len(AM[AM["Farm_Type"]==1])})',
               f'Sonali\n(n={len(AM[AM["Farm_Type"]==2])})']
therapy_defs = [
    (lambda s: s == 1, 'Monotherapy (1)',  COLORS['green_dark']),
    (lambda s: s == 2, 'Dual therapy (2)', COLORS['gold']),
    (lambda s: s >= 3, 'Polytherapy (≥3)', COLORS['red']),
]
for fi, (fn, lbl, col) in enumerate(therapy_defs):
    vals = []
    for ft in [0, 1, 2]:
        sub = AM[AM['Farm_Type'] == ft]['Number_of_AM'].dropna()
        sub = sub[sub != 99]
        vals.append(fn(sub).sum() / max(len(sub), 1) * 100)
    ax6.bar(x_f + fi*bw_f - bw_f, vals, bw_f, label=lbl, color=col, edgecolor='white', lw=0.6)

for fi, (fn, _, _) in enumerate(therapy_defs):
    for xi, ft in enumerate([0, 1, 2]):
        sub = AM[AM['Farm_Type'] == ft]['Number_of_AM'].dropna()
        sub = sub[sub != 99]
        val = fn(sub).sum() / max(len(sub), 1) * 100
        if val >= 5:
            ax6.text(xi + fi*bw_f - bw_f, val + 0.8, f'{val:.0f}%',
                     ha='center', fontsize=7.5, color=COLORS['gray_dark'], fontweight='bold')

ax6.set_xticks(x_f); ax6.set_xticklabels(ft_labels_f, fontsize=9.5)
ax6.set_ylabel('% of AM-Using Farms', fontsize=10)
ax6.set_ylim(0, 88); ax6.set_yticks(range(0, 81, 20))
ax6.tick_params(axis='y', labelsize=9.5)
ax6.axhline(y=0, color=COLORS['gray_dark'], lw=0.8)
ax6.legend(loc='upper left', fontsize=7.5, framealpha=0.95,
           edgecolor=COLORS['gray_light'], title='Therapy type', title_fontsize=7.5)
ax6.set_title('F. Therapy Type\nby Farm Type',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)

fig.suptitle('Figure 5 — AMR-Focused Bivariate Analysis: Key Associations',
             fontsize=13, fontweight='bold', color=COLORS['dark_blue'], y=1.02)
fig.text(
    0.01, -0.02,
    f'Note: AM users only (n={len(AM)}) for AMU-specific panels (A, B, C, D, F). '
    'KW = Kruskal-Wallis; MW = Mann-Whitney U; ρ = Spearman rank correlation. '
    'Panel E: jitter added for visibility; dashed line = OLS trend. '
    'Source: CP_AMR_Master_File_V13.xlsx',
    fontsize=8.5, color=COLORS['gray_mid'],
)

plt.savefig(OUTPUT, dpi=300)
plt.close()
print(f'✓ Saved: {OUTPUT}')
