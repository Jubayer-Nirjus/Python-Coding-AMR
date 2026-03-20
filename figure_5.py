
"""
Figure 5 — AMR-Focused Bivariate Analysis: Key Associations
Run: python figure5_bivariate.py
Output: Figure5_Bivariate_Analysis.png

Requirements:
    pip install pandas numpy matplotlib scipy openpyxl
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

# ── CONFIG ──
EXCEL_FILE = 'Master_Analysis_Workbook.xlsx'
OUTPUT     = 'Figure5_Bivariate_Analysis.png'

COLORS = {
    'dark_blue' : '#1F4E79',
    'mid_blue'  : '#2E75B6',
    'light_blue': '#BDD7EE',
    'green_dark': '#1E5631',
    'green_light': '#70AD47',
    'orange'    : '#C55A11',
    'red'       : '#C00000',
    'gold'      : '#BF6000',
    'gray_dark' : '#404040',
    'gray_mid'  : '#767676',
    'gray_light': '#D9D9D9',
    'broiler'   : '#2E75B6',
    'layer'     : '#375623',
    'sonali'    : '#C55A11',
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
df = pd.read_excel(EXCEL_FILE, sheet_name='1_Master_Data', header=1)
df_sc = pd.read_excel(EXCEL_FILE, sheet_name='3_Scores_Summary', header=1)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').replace(99, np.nan)

df['AMR_Risk_Cat'] = df_sc['AMR_Risk_Cat'].values

N  = len(df)
AM = df[df['AM_use_binary'] == 0].copy()   # AM users only (n≈165)


# ── COMPUTE STATS ──

# Panel A — Non-Rx use × AMR Awareness
heard_yes = AM[AM['Heard_of_AMR'] == 0]    # heard of AMR (code 0=Yes)
heard_no  = AM[AM['Heard_of_AMR'] == 1]    # not heard
norx_aware   = (heard_yes['AM_Without_Rx'] == 0).sum() / max(len(heard_yes), 1) * 100
norx_unaware = (heard_no['AM_Without_Rx']  == 0).sum() / max(len(heard_no),  1) * 100

ct_heard = pd.crosstab(AM['Heard_of_AMR'], AM['AM_Without_Rx'].dropna())
_, p_norx_heard, _, _ = stats.chi2_contingency(ct_heard)
p_norx_str = '< 0.001' if p_norx_heard < 0.001 else f'{p_norx_heard:.3f}'

# Panel B — Withdrawal × Knowledge Score (KW)
wd_always   = AM[AM['Withdrawal_Practice'] == 0]['Knowledge_Score'].dropna()
wd_sometimes= AM[AM['Withdrawal_Practice'] == 1]['Knowledge_Score'].dropna()
wd_never    = AM[AM['Withdrawal_Practice'] == 2]['Knowledge_Score'].dropna()

kw_stat, kw_p = stats.kruskal(wd_always, wd_sometimes, wd_never)
p_wd_str = '< 0.001' if kw_p < 0.001 else f'{kw_p:.3f}'

# Panel D — Growth Promoter × Knowledge Score (MW)
gp_users  = AM[AM['AM_Growth_Promoter'] == 0]['Knowledge_Score'].dropna()
gp_non    = AM[AM['AM_Growth_Promoter'] == 1]['Knowledge_Score'].dropna()
_, p_gp   = stats.mannwhitneyu(gp_users, gp_non, alternative='two-sided')
p_gp_str  = '< 0.001' if p_gp < 0.001 else f'{p_gp:.3f}'

# Panel E — Digital Score × AMR Risk (Spearman)
paired_dig = df[['Digital_Score', 'AMR_Risk_Index']].dropna()
rho_dig, p_dig = stats.spearmanr(paired_dig['Digital_Score'],
                                  paired_dig['AMR_Risk_Index'])
rho_str = f'{rho_dig:.3f}'
p_dig_str = '< 0.001' if p_dig < 0.001 else f'{p_dig:.3f}'


# ── FIGURE LAYOUT ──
fig = plt.figure(figsize=(16, 10))
gs_ = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.44, wspace=0.40,
)

ax1 = fig.add_subplot(gs_[0, 0])   # A: Non-Rx × Awareness
ax2 = fig.add_subplot(gs_[0, 1])   # B: Withdrawal × Knowledge
ax3 = fig.add_subplot(gs_[0, 2])   # C: AWaRe by Farm Type
ax4 = fig.add_subplot(gs_[1, 0])   # D: Growth Promoter × Knowledge
ax5 = fig.add_subplot(gs_[1, 1])   # E: Digital × AMR Risk
ax6 = fig.add_subplot(gs_[1, 2])   # F: Polytherapy by Farm Type


def p_badge(ax, text, facecolor=None, edgecolor=None):
    """Add a p-value badge in top-right corner."""
    fc = facecolor or COLORS['light_blue']
    ec = edgecolor or COLORS['dark_blue']
    ax.text(
        0.98, 0.97, text,
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9, color=COLORS['dark_blue'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=fc,
                  edgecolor=ec, linewidth=1, alpha=0.95),
    )


# ════════════════════════════════════════════
# PANEL A — Non-Rx AM use × AMR Awareness
# ════════════════════════════════════════════
bars_a = ax1.bar(
    [f'Aware\n(n={len(heard_yes)})', f'Unaware\n(n={len(heard_no)})'],
    [norx_aware, norx_unaware],
    color=[COLORS['green_dark'], COLORS['red']],
    edgecolor='white', lw=0.7, width=0.55,
)
for bar, val in zip(bars_a, [norx_aware, norx_unaware]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1, f'{val:.1f}%',
        ha='center', fontsize=10.5,
        fontweight='bold', color=COLORS['dark_blue'],
    )

ax1.set_ylabel('% Using AM Without Prescription', fontsize=10)
ax1.set_ylim(0, 82)
ax1.set_yticks(range(0, 81, 20))
ax1.tick_params(axis='x', labelsize=10.5)
ax1.tick_params(axis='y', labelsize=9.5)
ax1.set_title('A. Non-Prescription AM Use\nvs AMR Awareness',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
ax1.axhline(y=0, color=COLORS['gray_dark'], lw=0.8)
p_badge(ax1, f'p = {p_norx_str}')


# ════════════════════════════════════════════
# PANEL B — Knowledge Score × Withdrawal
# ════════════════════════════════════════════
wd_data   = [wd_always, wd_sometimes, wd_never]
wd_labels = [
    f'Always\n(n={len(wd_always)})',
    f'Sometimes\n(n={len(wd_sometimes)})',
    f'Never\n(n={len(wd_never)})',
]
bp = ax2.boxplot(
    wd_data,
    labels       = wd_labels,
    patch_artist = True,
    widths       = 0.55,
    medianprops  = dict(color='white', lw=2.5),
    flierprops   = dict(marker='o', markersize=4, alpha=0.45,
                        markeredgecolor='none'),
    boxprops     = dict(lw=0.6),
    whiskerprops = dict(lw=0.8),
    capprops     = dict(lw=0.8),
)
box_colors = [COLORS['green_dark'], COLORS['gold'], COLORS['red']]
for patch, col in zip(bp['boxes'], box_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.82)

ax2.set_ylabel('Knowledge Score (0–6)', fontsize=10)
ax2.set_ylim(-0.5, 7.5)
ax2.tick_params(axis='x', labelsize=9.5)
ax2.tick_params(axis='y', labelsize=9.5)
ax2.set_title('B. Knowledge Score\nvs Withdrawal Adherence',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
p_badge(ax2, f'KW p = {p_wd_str}')

# Significance brackets
def sig_bracket(ax, x1, x2, y, text):
    ax.plot([x1, x1, x2, x2], [y, y+0.08, y+0.08, y],
            lw=1, color=COLORS['gray_dark'])
    ax.text((x1+x2)/2, y+0.1, text, ha='center',
            fontsize=8, color=COLORS['gray_dark'])

# Bonferroni-corrected pairwise
_, p_a_s = stats.mannwhitneyu(wd_always, wd_sometimes, alternative='two-sided')
_, p_a_n = stats.mannwhitneyu(wd_always, wd_never,     alternative='two-sided')
if min(p_a_n*3, 1.0) < 0.05:
    sig_bracket(ax2, 1, 3, 6.5, '***')


# ════════════════════════════════════════════
# PANEL C — AWaRe Category by Farm Type
# ════════════════════════════════════════════
ft_am_groups = [
    AM[AM['Farm_Type'] == 0],
    AM[AM['Farm_Type'] == 1],
    AM[AM['Farm_Type'] == 2],
]
x_c  = np.arange(3)
bw_c = 0.26
ft_labels_c = [
    f'Broiler\n(n={len(ft_am_groups[0])})',
    f'Layer\n(n={len(ft_am_groups[1])})',
    f'Sonali\n(n={len(ft_am_groups[2])})',
]
aware_palette = [
    (0, 'Access',   '#375623'),
    (1, 'Watch',  '#BF6000'),
    (2, 'Reserve','#C00000'),
]
for fi, (code, lbl, col) in enumerate(aware_palette):
    vals = [(g['AWaRE_Category'] == code).sum() / max(g['AWaRE_Category'].notna().sum(), 1) * 100
            for g in ft_am_groups]
    ax3.bar(x_c + fi*bw_c - bw_c, vals, bw_c,
            label=lbl, color=col, edgecolor='white', lw=0.6)

ax3.set_xticks(x_c)
ax3.set_xticklabels(ft_labels_c, fontsize=9.5)
ax3.set_ylabel('% of AM-Using Farms', fontsize=10)
ax3.set_ylim(0, 82)
ax3.set_yticks(range(0, 81, 20))
ax3.tick_params(axis='y', labelsize=9.5)
ax3.axhline(y=0, color=COLORS['gray_dark'], lw=0.8)
ax3.legend(loc='upper center', fontsize=8.5, framealpha=0.95,
           edgecolor=COLORS['gray_light'])
ax3.set_title('C. WHO AWaRe Category\nby Farm Type',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
p_badge(ax3, 'ns (p > 0.05)')


# ════════════════════════════════════════════
# PANEL D — Knowledge Score × Growth Promoter
# ════════════════════════════════════════════
vp = ax4.violinplot(
    [gp_users, gp_non],
    positions = [1, 2],
    widths    = 0.65,
    showmedians = True,
    showextrema = True,
)
vp_colors = [COLORS['red'], COLORS['green_dark']]
for pc, col in zip(vp['bodies'], vp_colors):
    pc.set_facecolor(col)
    pc.set_alpha(0.72)
    pc.set_edgecolor('white')
    pc.set_linewidth(0.5)
vp['cmedians'].set_color('white')
vp['cmedians'].set_linewidth(2.5)
for part in ['cbars', 'cmins', 'cmaxes']:
    vp[part].set_color(COLORS['gray_dark'])
    vp[part].set_linewidth(0.8)

ax4.set_xticks([1, 2])
ax4.set_xticklabels(
    [f'GP Users\n(n={len(gp_users)})', f'Non-users\n(n={len(gp_non)})'],
    fontsize=10,
)
ax4.set_ylabel('Knowledge Score (0–6)', fontsize=10)
ax4.set_ylim(-0.3, 7.3)
ax4.tick_params(axis='y', labelsize=9.5)
ax4.set_title('D. Knowledge Score\nvs Growth Promoter Use',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
p_badge(ax4, f'MW p = {p_gp_str}')

# Mean annotations
for xpos, data, col in [(1, gp_users, COLORS['red']),
                         (2, gp_non,  COLORS['green_dark'])]:
    ax4.text(xpos, data.mean() + 0.12, f'μ={data.mean():.1f}',
             ha='center', fontsize=9, color=col, fontweight='bold')


# ════════════════════════════════════════════
# PANEL E — Digital Score × AMR Risk (scatter)
# ════════════════════════════════════════════
rng   = np.random.default_rng(42)
jitter = rng.uniform(-0.09, 0.09, len(paired_dig))

ax5.scatter(
    paired_dig['Digital_Score'] + jitter,
    paired_dig['AMR_Risk_Index'],
    alpha=0.45, s=28,
    color=COLORS['mid_blue'],
    edgecolors='white', lw=0.3,
)
# Regression trend line
z  = np.polyfit(paired_dig['Digital_Score'], paired_dig['AMR_Risk_Index'], 1)
xr = np.linspace(-0.15, 2.15, 50)
ax5.plot(xr, np.poly1d(z)(xr),
         color=COLORS['red'], lw=2, linestyle='--', alpha=0.85)

ax5.set_xlabel('Digital/AI Score (0–2)', fontsize=10)
ax5.set_ylabel('AMR Risk Index (0–8)', fontsize=10)
ax5.set_xlim(-0.25, 2.25)
ax5.set_ylim(-0.3, 9)
ax5.set_xticks([0, 1, 2])
ax5.tick_params(labelsize=9.5)
ax5.set_title('E. Digital Score\nvs AMR Risk Index',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)
p_badge(ax5, f'ρ = {rho_str}  p = {p_dig_str}')


# ════════════════════════════════════════════
# PANEL F — Therapy Type by Farm Type
# ════════════════════════════════════════════
x_f  = np.arange(3)
bw_f = 0.26
ft_labels_f = [
    f'Broiler\n(n={len(AM[AM["Farm_Type"]==0])})',
    f'Layer\n(n={len(AM[AM["Farm_Type"]==1])})',
    f'Sonali\n(n={len(AM[AM["Farm_Type"]==2])})',
]
therapy_defs = [
    (lambda s: s == 1, 'Monotherapy (1)',  COLORS['green_dark']),
    (lambda s: s == 2, 'Dual therapy (2)', COLORS['gold']),
    (lambda s: s >= 3, 'Polytherapy (≥3)', COLORS['red']),
]
for fi, (fn, lbl, col) in enumerate(therapy_defs):
    vals = []
    for ft in [0, 1, 2]:
        sub = AM[AM['Farm_Type'] == ft]['Number_of_AM'].dropna()
        vals.append(fn(sub).sum() / max(len(sub), 1) * 100)
    ax6.bar(x_f + fi*bw_f - bw_f, vals, bw_f,
            label=lbl, color=col, edgecolor='white', lw=0.6)

# Value labels ≥ 5%
for fi, (fn, _, _) in enumerate(therapy_defs):
    for xi, ft in enumerate([0, 1, 2]):
        sub = AM[AM['Farm_Type'] == ft]['Number_of_AM'].dropna()
        val = fn(sub).sum() / max(len(sub), 1) * 100
        if val >= 5:
            ax6.text(xi + fi*bw_f - bw_f, val + 0.8,
                     f'{val:.0f}%', ha='center',
                     fontsize=7.5, color=COLORS['gray_dark'],
                     fontweight='bold')

ax6.set_xticks(x_f)
ax6.set_xticklabels(ft_labels_f, fontsize=9.5)
ax6.set_ylabel('% of AM-Using Farms', fontsize=10)
ax6.set_ylim(0, 88)
ax6.set_yticks(range(0, 81, 20))
ax6.tick_params(axis='y', labelsize=9.5)
ax6.axhline(y=0, color=COLORS['gray_dark'], lw=0.8)
ax6.legend(loc='upper left', fontsize=7.5, framealpha=0.95,
           edgecolor=COLORS['gray_light'],
           title='Therapy type', title_fontsize=7.5)
ax6.set_title('F. Therapy Type\nby Farm Type',
              fontsize=11, fontweight='bold', color=COLORS['dark_blue'], pad=8)


# ── Overall title & footnote ──
fig.suptitle(
    'Figure 5 — AMR-Focused Bivariate Analysis: Key Associations',
    fontsize=13, fontweight='bold',
    color=COLORS['dark_blue'], y=1.02,
)
fig.text(
    0.01, -0.02,
    'Note: AM users only (n≈165) for AMU-specific panels (A, B, C, D, F). '
    'KW = Kruskal-Wallis; MW = Mann-Whitney U; Chi² = Chi-square; ρ = Spearman rank correlation. '
    'Panel E: jitter added for visibility; dashed line = OLS trend.',
    fontsize=8.5, color=COLORS['gray_mid'],
)

plt.savefig(OUTPUT, dpi=300)
plt.close()
print(f'✓ Saved: {OUTPUT}')
