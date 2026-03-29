"""
Figure 4 — AMR Risk Index: Item-Level Prevalence & Category Distribution
Updated for CP_AMR_Master_File_V13.xlsx
Output: Figure4_AMR_Risk_Index.png

Coding reference (V13):
  AM_use_binary:      0=AM users (n=165), 1=non-users (n=47)
  AWaRE_Category:     0=Access, 1=Watch, 2=Reserve, 99=non-users
  AM_Without_Rx:      0=Yes(non-Rx), 1=No(Rx), 99=non-users
  Number_of_AM:       1=mono, 2=dual, 3+=polytherapy, 99=non-users
  Withdrawal_Practice:0=Always, 1=Sometimes, 2=Never, 99=non-users
  AM_Growth_Promoter: 0=Yes(GP), 1=No, 99=non-users
  Reuse_Leftover:     0=Yes, 1=No, 99=non-users
  Prescriber_of_AM:   0=Vet, 1=Non-vet, 2=Self/feed agent, 99=non-users
  AMR_Risk_Cat:       'Low','Moderate','High'
  Farm_Type:          0=Broiler, 1=Layer, 2=Sonali

Ground truth AMR Risk (V13, n=212):
  Low(0-2)=114(53.8%), Moderate(3-4)=71(33.5%), High(>=5)=27(12.7%)
  By farm: Broiler Low=50(45.9%) Mod=43(39.4%) High=16(14.7%)
           Layer   Low=43(61.4%) Mod=20(28.6%) High=7(10.0%)
           Sonali  Low=21(63.6%) Mod=8(24.2%)  High=4(12.1%)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

EXCEL_FILE = 'CP_AMR_Master_File_V13.xlsx'
OUTPUT     = 'Figure4_AMR_Risk_Index.png'

COLORS = {
    'dark_blue'  : '#1F4E79',
    'mid_blue'   : '#2E75B6',
    'light_blue' : '#BDD7EE',
    'green_dark' : '#1E5631',
    'green_light': '#70AD47',
    'orange'     : '#C55A11',
    'red'        : '#C00000',
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

N  = len(df)   # 212
AM = df[df['AM_use_binary'] == 0].copy()   # n=165
BR = df[df['Farm_Type'] == 0]              # n=109
LA = df[df['Farm_Type'] == 1]              # n=70
SO = df[df['Farm_Type'] == 2]             # n=33

def pct(data, col, code):
    """% of AM-using farms with given code; excludes 99."""
    d = data[col].dropna()
    d = d[d != 99]
    return round((d == code).sum() / len(d) * 100, 1) if len(d) > 0 else 0.0

def pct_any(data, col, codes):
    """% matching any of the given codes; excludes 99."""
    d = data[col].dropna()
    d = d[d != 99]
    return round(d.isin(codes).sum() / len(d) * 100, 1) if len(d) > 0 else 0.0

# ── PANEL A DATA: Heatmap of risk behaviours ──
# Items reflect the 8 AMR Risk Index components (AM users only)
HEATMAP_ITEMS = [
    ('AM_Without_Rx',        [0],   'AM Without\nPrescription'),
    ('AWaRE_Category',       [1],   'Watch Category\nAntibiotics'),
    ('AWaRE_Category',       [2],   'Reserve Category\nAntibiotics'),
    ('Number_of_AM',         [2],   'Dual Therapy\n(2 drugs)'),
    ('Number_of_AM',         [3],   'Polytherapy\n(≥3 drugs)'),
    ('Withdrawal_Practice',  [1],   'Withdrawal\nNon-adherence'),
    ('Withdrawal_Practice',  [2],   'No Withdrawal\nPeriod'),
    ('AM_Growth_Promoter',   [0],   'Growth Promoter\nUse'),
    ('Reuse_Leftover',       [0],   'Reuse Leftover\nAntibiotics'),
    ('Prescriber_of_AM',     [1],   'Non-Vet\nPrescriber'),
    ('Prescriber_of_AM',     [2],   'Self-\nPrescription'),
]

hm_data = pd.DataFrame(
    [[pct_any(AM[AM['Farm_Type'] == ft], col, codes) for ft in [0, 1, 2]]
     for col, codes, _ in HEATMAP_ITEMS],
    index   = [lbl for _, _, lbl in HEATMAP_ITEMS],
    columns = [
        f'Broiler\n(n={len(AM[AM["Farm_Type"]==0])})',
        f'Layer\n(n={len(AM[AM["Farm_Type"]==1])})',
        f'Sonali\n(n={len(AM[AM["Farm_Type"]==2])})',
    ],
)

# ── PANEL B DATA: AMR Risk Category distribution ──
# Use AMR_Risk_Cat from master data directly
def cat_pct(subset, cat_val):
    valid = subset['AMR_Risk_Cat'].dropna()
    return (valid == cat_val).sum() / max(len(valid), 1) * 100

cats     = ['Low\n(0–2)', 'Moderate\n(3–4)', 'High\n(≥5)']
cat_vals = ['Low', 'Moderate', 'High']
overall  = [cat_pct(df, v) for v in cat_vals]
broiler  = [cat_pct(BR, v) for v in cat_vals]
layer    = [cat_pct(LA, v) for v in cat_vals]
sonali   = [cat_pct(SO, v) for v in cat_vals]

# ── FIGURE ──
fig, axes = plt.subplots(
    1, 2, figsize=(14, 6.5),
    gridspec_kw={'width_ratios': [2, 1.4], 'wspace': 0.12},
)
ax_hm  = axes[0]
ax_bar = axes[1]

# Panel A: Heatmap
cmap = sns.color_palette('RdYlGn_r', as_cmap=True)
hm = sns.heatmap(
    hm_data, ax=ax_hm, annot=True, fmt='.1f',
    cmap=cmap, vmin=0, vmax=80,
    linewidths=0.6, linecolor='white',
    annot_kws={'size': 10, 'weight': 'bold'},
    cbar_kws={'label': '% of AM-Using Farms', 'shrink': 0.75, 'pad': 0.02},
)

for text in ax_hm.texts:
    try:
        val = float(text.get_text())
        text.set_color('white' if (val > 55 or val < 8) else COLORS['gray_dark'])
    except ValueError:
        pass

cbar = hm.collections[0].colorbar
cbar.set_label('% of AM-Using Farms', fontsize=9, color=COLORS['gray_dark'])
cbar.ax.tick_params(labelsize=8)

ax_hm.set_title(
    'A. Risk Behaviour Prevalence by Farm Type\n(AM-using farms only)',
    loc='left', fontsize=11, fontweight='bold',
    color=COLORS['dark_blue'], pad=10,
)
ax_hm.tick_params(axis='y', rotation=0,  labelsize=9.5)
ax_hm.tick_params(axis='x', rotation=0,  labelsize=10.5)

# Panel B: Grouped bar chart
x   = np.arange(len(cats))
bw  = 0.2

ax_bar.bar(x - 0.30, overall, bw, label=f'Overall (n={N})',
           color=COLORS['dark_blue'], edgecolor='white', lw=0.6)
ax_bar.bar(x - 0.10, broiler, bw, label=f'Broiler (n={len(BR)})',
           color=COLORS['broiler'],   edgecolor='white', lw=0.6)
ax_bar.bar(x + 0.10, layer,   bw, label=f'Layer (n={len(LA)})',
           color=COLORS['layer'],     edgecolor='white', lw=0.6)
ax_bar.bar(x + 0.30, sonali,  bw, label=f'Sonali (n={len(SO)})',
           color=COLORS['sonali'],    edgecolor='white', lw=0.6)

for i, (ov, br, la, so) in enumerate(zip(overall, broiler, layer, sonali)):
    for xpos, val in [(i-0.30, ov), (i-0.10, br), (i+0.10, la), (i+0.30, so)]:
        if val >= 4:
            ax_bar.text(xpos, val + 0.8, f'{val:.0f}%',
                        ha='center', fontsize=7.5,
                        color=COLORS['gray_dark'], fontweight='bold')

ax_bar.axvspan(1.55, 2.55, alpha=0.06, color=COLORS['red'])

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(cats, fontsize=10.5)
ax_bar.set_ylabel('% of Farms', fontsize=10)
ax_bar.set_ylim(0, 78)
ax_bar.set_yticks(range(0, 71, 10))
ax_bar.tick_params(axis='y', labelsize=9)
ax_bar.axhline(y=0, color=COLORS['gray_dark'], lw=0.8)

# KW test across farm types on AMR Risk Index
_, p_kw = stats.kruskal(
    BR['AMR_Risk_Index'].dropna(),
    LA['AMR_Risk_Index'].dropna(),
    SO['AMR_Risk_Index'].dropna(),
)
p_str = '= 0.002' if p_kw > 0.001 else '< 0.001'
ax_bar.text(
    0.03, 0.97, f'KW p {p_str}',
    transform=ax_bar.transAxes,
    fontsize=9, va='top', color=COLORS['dark_blue'], fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_blue'],
              edgecolor=COLORS['dark_blue'], linewidth=1, alpha=0.95),
)

ax_bar.legend(loc='upper right', fontsize=8.5, framealpha=0.95,
              edgecolor=COLORS['gray_light'])
ax_bar.set_title(
    'B. AMR Risk Category Distribution\nby Farm Type',
    loc='left', fontsize=11, fontweight='bold',
    color=COLORS['dark_blue'], pad=10,
)

fig.suptitle(
    'Figure 4 — AMR Risk Index: Item-Level Prevalence & Category Distribution by Farm Type',
    fontsize=13, fontweight='bold', color=COLORS['dark_blue'], y=1.02,
)
fig.text(
    0.01, -0.03,
    f'Note: Panel A — % of AM-using farms (n={len(AM)}) exhibiting each risk behaviour. '
    'Red=high prevalence, Green=low. '
    f'Panel B — AMR Risk Index categories for all farms (n={N}): '
    'Low(0-2)=114(53.8%), Moderate(3-4)=71(33.5%), High(≥5)=27(12.7%). '
    'KW = Kruskal-Wallis. Source: CP_AMR_Master_File_V13.xlsx',
    fontsize=8.5, color=COLORS['gray_mid'],
)

plt.savefig(OUTPUT, dpi=300)
plt.close()
print(f'✓ Saved: {OUTPUT}')
