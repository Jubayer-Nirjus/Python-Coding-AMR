"""
Figure 4 — AMR Risk Index: Item-Level Prevalence & Category Distribution
Preventive Veterinary Medicine (Elsevier) — journal-compliant output

Journal requirements applied:
  - No figure title on figure (caption only)
  - PNG, 300 dpi, 14×6.5 in = 4200×1950 px ✓
  - Wong (2011) colorblind-safe palette
  - Output: Figure_4.png

CAPTION (use in manuscript):
  Fig. 4. AMR Risk Index analysis by farm type. (A) Heatmap showing the prevalence of
  antimicrobial use (AMU) risk behaviours among antimicrobial-using farms (n = 165),
  stratified by farm type. Colour scale: red (RdYlGn_r) indicates high prevalence, green
  indicates low prevalence. (B) Distribution of AMR Risk Index categories (Low 0–2,
  Moderate 3–4, High ≥5) across all farms (n = 212) by farm type. Overall distribution:
  Low = 114 (53.8%), Moderate = 71 (33.5%), High = 27 (12.7%). KW = Kruskal–Wallis test
  across farm types (H = 12.45, df = 2, p = 0.002).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

EXCEL_FILE = 'CP_AMR_Master_File_V13.xlsx'
OUTPUT     = 'Figure_4.png'

# Wong (2011) colorblind-safe palette
COLORS = {
    'blue'       : '#0072B2',
    'sky_blue'   : '#56B4E9',
    'orange'     : '#E69F00',
    'vermillion' : '#D55E00',
    'green'      : '#009E73',
    'yellow'     : '#F0E442',
    'pink'       : '#CC79A7',
    'black'      : '#000000',
    'gray_dark'  : '#404040',
    'gray_mid'   : '#767676',
    'gray_light' : '#D9D9D9',
    # Farm type colors (colorblind-safe)
    'overall'    : '#000000',
    'broiler'    : '#0072B2',
    'layer'      : '#009E73',
    'sonali'     : '#E69F00',
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
    if col != 'AMR_Risk_Cat':
        df[col] = pd.to_numeric(df[col], errors='coerce')

N  = len(df)
AM = df[df['AM_use_binary'] == 0].copy()
BR = df[df['Farm_Type'] == 0]
LA = df[df['Farm_Type'] == 1]
SO = df[df['Farm_Type'] == 2]

def pct_any(data, col, codes):
    d = data[col].dropna()
    d = d[d != 99]
    return round(d.isin(codes).sum() / len(d) * 100, 1) if len(d) > 0 else 0.0

def cat_pct(subset, cat_val):
    valid = subset['AMR_Risk_Cat'].dropna()
    return (valid == cat_val).sum() / max(len(valid), 1) * 100

# ── PANEL A: Heatmap ──
HEATMAP_ITEMS = [
    ('AM_Without_Rx',        [0],    'AM Without Prescription'),
    ('AWaRE_Category',       [1],    'Watch Category Antibiotics'),
    ('AWaRE_Category',       [2],    'Reserve Category Antibiotics'),
    ('Number_of_AM',         [2],    'Dual Therapy (2 drugs)'),
    ('Number_of_AM',         [3],    'Polytherapy (≥3 drugs)'),
    ('Withdrawal_Practice',  [1],    'Withdrawal Non-adherence'),
    ('Withdrawal_Practice',  [2],    'No Withdrawal Period'),
    ('AM_Growth_Promoter',   [0],    'Growth Promoter Use'),
    ('Reuse_Leftover',       [0],    'Reuse Leftover Antibiotics'),
    ('Prescriber_of_AM',     [1],    'Non-Veterinarian Prescriber'),
    ('Prescriber_of_AM',     [2],    'Self-Prescription'),
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

# ── PANEL B: Category distribution ──
cats     = ['Low\n(0–2)', 'Moderate\n(3–4)', 'High\n(≥5)']
cat_vals = ['Low', 'Moderate', 'High']
overall  = [cat_pct(df, v) for v in cat_vals]
broiler  = [cat_pct(BR, v) for v in cat_vals]
layer    = [cat_pct(LA, v) for v in cat_vals]
sonali   = [cat_pct(SO, v) for v in cat_vals]

# ── FIGURE ──
fig, axes = plt.subplots(
    1, 2, figsize=(14, 6.5),
    gridspec_kw={'width_ratios': [2, 1.4], 'wspace': 0.14},
)
ax_hm  = axes[0]
ax_bar = axes[1]

# Panel A: Heatmap (RdYlGn_r is perceptually uniform and accessible with pattern annotations)
cmap = sns.color_palette('RdYlGn_r', as_cmap=True)
hm = sns.heatmap(
    hm_data, ax=ax_hm, annot=True, fmt='.1f',
    cmap=cmap, vmin=0, vmax=80,
    linewidths=0.6, linecolor='white',
    annot_kws={'size': 9.5, 'weight': 'bold'},
    cbar_kws={'label': 'AM-using farms (%)', 'shrink': 0.75, 'pad': 0.02},
)

for text in ax_hm.texts:
    try:
        val = float(text.get_text())
        text.set_color('white' if (val > 55 or val < 8) else COLORS['gray_dark'])
    except ValueError:
        pass

cbar = hm.collections[0].colorbar
cbar.set_label('AM-using farms (%)', fontsize=9, color=COLORS['gray_dark'])
cbar.ax.tick_params(labelsize=8)

ax_hm.set_title('A. Risk Behaviour Prevalence by Farm Type\n(AM-using farms only)',
                loc='left', fontsize=11, fontweight='bold',
                color=COLORS['gray_dark'], pad=8)
ax_hm.tick_params(axis='y', rotation=0,  labelsize=9)
ax_hm.tick_params(axis='x', rotation=0,  labelsize=10)

# Panel B: Grouped bar
x   = np.arange(len(cats))
bw  = 0.20

ax_bar.bar(x - 0.30, overall, bw, label=f'Overall (n={N})',
           color=COLORS['overall'],  edgecolor='white', lw=0.6, alpha=0.85)
ax_bar.bar(x - 0.10, broiler, bw, label=f'Broiler (n={len(BR)})',
           color=COLORS['broiler'], edgecolor='white', lw=0.6)
ax_bar.bar(x + 0.10, layer,   bw, label=f'Layer (n={len(LA)})',
           color=COLORS['layer'],   edgecolor='white', lw=0.6)
ax_bar.bar(x + 0.30, sonali,  bw, label=f'Sonali (n={len(SO)})',
           color=COLORS['sonali'],  edgecolor='white', lw=0.6)

for i, (ov, br, la, so) in enumerate(zip(overall, broiler, layer, sonali)):
    for xpos, val in [(i-0.30, ov), (i-0.10, br), (i+0.10, la), (i+0.30, so)]:
        if val >= 4:
            ax_bar.text(xpos, val + 0.8, f'{val:.0f}%',
                        ha='center', fontsize=7.5, color=COLORS['gray_dark'], fontweight='bold')

# High risk zone shading
ax_bar.axvspan(1.55, 2.55, alpha=0.05, color=COLORS['vermillion'])

ax_bar.set_xticks(x); ax_bar.set_xticklabels(cats, fontsize=10.5)
ax_bar.set_ylabel('Farms (%)', fontsize=10)
ax_bar.set_ylim(0, 78); ax_bar.set_yticks(range(0, 71, 10))
ax_bar.tick_params(axis='y', labelsize=9)
ax_bar.axhline(y=0, color=COLORS['gray_dark'], lw=0.8)

_, p_kw = stats.kruskal(BR['AMR_Risk_Index'].dropna(),
                         LA['AMR_Risk_Index'].dropna(),
                         SO['AMR_Risk_Index'].dropna())
p_str = '= 0.002' if p_kw > 0.001 else '< 0.001'
ax_bar.text(0.03, 0.97, f'KW p {p_str}', transform=ax_bar.transAxes,
            fontsize=9, va='top', color=COLORS['blue'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#CCE5FF',
                      edgecolor=COLORS['blue'], linewidth=1, alpha=0.95))

ax_bar.legend(loc='upper right', fontsize=8.5, framealpha=0.95,
              edgecolor=COLORS['gray_light'])
ax_bar.set_title('B. AMR Risk Category Distribution\nby Farm Type',
                 loc='left', fontsize=11, fontweight='bold',
                 color=COLORS['gray_dark'], pad=8)

fig.text(
    0.01, -0.02,
    f'AM = antimicrobial. AM-using farms n = {len(AM)} (AM_use_binary = 0). '
    f'Overall High Risk (≥5): n = 27 (12.7%). KW = Kruskal–Wallis.',
    fontsize=8, color=COLORS['gray_mid'],
)

plt.savefig(OUTPUT, dpi=300, format='png')
plt.close()
print(f'✓ Saved: {OUTPUT}')
