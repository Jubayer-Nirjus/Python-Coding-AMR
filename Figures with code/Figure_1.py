"""
Figure 1 — AMR Knowledge Gap Heatmap
Preventive Veterinary Medicine (Elsevier) — journal-compliant output

Journal requirements applied:
  - Figure title NOT displayed on figure (caption only)
  - PNG, 300 dpi, full-page width (>=2244 px) -> figsize=(13,6) @ 300dpi = 3900px ✓
  - Color-blind accessible palette (Wong 2011)
  - Minimal text on figure
  - Output: Figure_1.png

CAPTION (use in manuscript):
  Fig. 1. AMR knowledge gap heatmap showing the percentage of commercial poultry
  farmers answering each knowledge item correctly, stratified by farm type (Broiler,
  Layer, Sonali) and education level (Graduate, College/SSC, Primary/Below). Values
  represent the percentage of farmers providing the correct response. Colour scale:
  red = low correct response rate, green = high correct response rate. Total n = 212.
  Source: Bangladesh commercial poultry AMR survey, 2023.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

EXCEL_FILE = 'CP_AMR_Master_File_V13.xlsx'
OUTPUT     = 'Figure_1.png'

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
    # Legacy aliases for compatibility
    'dark_blue'  : '#0072B2',
    'mid_blue'   : '#56B4E9',
    'gold'       : '#E69F00',
    'gold_light' : '#FFF3CD',
    'light_blue' : '#CCEEFF',
}

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
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

N  = len(df)
BR = df[df['Farm_Type'] == 0]
LA = df[df['Farm_Type'] == 1]
SO = df[df['Farm_Type'] == 2]

def pct(data, col, code=0):
    d = data[col].dropna()
    d = d[d != 99]
    return round((d == code).sum() / len(d) * 100, 1) if len(d) > 0 else 0.0

KNOW_ITEMS = [
    ('Heard_of_AMR',            0, 'Has Heard of AMR / Antibiotic Resistance'),
    ('Herbal_Drug_Knowledge',   0, 'Aware of Herbal / Alternative Drugs'),
    ('Perceived_Causes_AMR',    0, 'Can Identify Causes of AMR'),
    ('Withdrawal_Awareness_K',  0, 'Aware of Antibiotic Withdrawal Period'),
    ('AMR_Misuse_Impact',       0, 'Knows Misuse Impacts Human Health'),
    ('AMR_Risk_Less_Effective', 0, 'Knows Antibiotics Becoming Less Effective'),
]

COL_GROUPS = {
    f'Overall\n(n={N})'                              : df,
    f'Broiler\n(n={len(BR)})'                        : BR,
    f'Layer\n(n={len(LA)})'                          : LA,
    f'Sonali\n(n={len(SO)})'                         : SO,
    f'Graduate\n(n={len(df[df["Education"]==0])})'   : df[df['Education'] == 0],
    f'College\n(n={len(df[df["Education"]==1])})'    : df[df['Education'] == 1],
    f'Primary\n(n={len(df[df["Education"]==2])})'    : df[df['Education'] == 2],
}

hm_df = pd.DataFrame(
    [[pct(g, col, code) for g in COL_GROUPS.values()]
     for col, code, _ in KNOW_ITEMS],
    index   = [lbl for _, _, lbl in KNOW_ITEMS],
    columns = list(COL_GROUPS.keys()),
)

# ── FIGURE — no suptitle (caption in manuscript) ──
fig, ax = plt.subplots(figsize=(13, 6))

# Colorblind-accessible diverging palette (blue-white-orange)
cmap = sns.diverging_palette(220, 20, s=85, l=40, center='light', as_cmap=True)

hm = sns.heatmap(
    hm_df,
    ax         = ax,
    annot      = True,
    fmt        = '.1f',
    cmap       = cmap,
    vmin       = 0,
    vmax       = 100,
    linewidths = 0.6,
    linecolor  = '#FFFFFF',
    annot_kws  = {'size': 10, 'weight': 'bold'},
    cbar_kws   = {
        'label'  : '% Correct responses',
        'shrink' : 0.70,
        'aspect' : 16,
        'pad'    : 0.02,
    },
)

# Annotation text colour
for text in ax.texts:
    try:
        val = float(text.get_text())
        text.set_color('white' if (val < 28 or val > 72) else COLORS['gray_dark'])
    except ValueError:
        pass

cbar = hm.collections[0].colorbar
cbar.set_label('% Correct responses', fontsize=9, color=COLORS['gray_dark'])
cbar.ax.tick_params(labelsize=8.5)

# Vertical separator between Farm Type and Education columns
ax.axvline(x=4, color='white',           linewidth=4)
ax.axvline(x=4, color=COLORS['gray_mid'],linewidth=1.2, linestyle='--', alpha=0.6)

# Group labels (minimal text)
trans = ax.get_xaxis_transform()
ax.annotate('By Farm Type',
    xy=(1.95, 1.035), xycoords=trans,
    fontsize=9.5, fontweight='bold', color=COLORS['blue'],
    ha='center', va='bottom',
    bbox=dict(boxstyle='round,pad=0.22', facecolor=COLORS['light_blue'],
              edgecolor=COLORS['blue'], linewidth=1, alpha=0.85))
ax.annotate('By Education Level',
    xy=(5.5, 1.035), xycoords=trans,
    fontsize=9.5, fontweight='bold', color=COLORS['vermillion'],
    ha='center', va='bottom',
    bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFE8E0',
              edgecolor=COLORS['vermillion'], linewidth=1, alpha=0.85))

ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='y', rotation=0,  labelsize=9.5)
ax.tick_params(axis='x', rotation=18, labelsize=9.5)

# Footnote (minimal — full legend in caption)
fig.text(
    0.01, -0.02,
    f'Values = % correct. Farm type: Broiler n={len(BR)}, Layer n={len(LA)}, '
    f'Sonali n={len(SO)}. Education: Graduate n={len(df[df["Education"]==0])}, '
    f'College n={len(df[df["Education"]==1])}, Primary n={len(df[df["Education"]==2])}. '
    f'Total n={N}.',
    fontsize=8, color=COLORS['gray_mid'],
)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(OUTPUT, dpi=300, format='png')
plt.close()
print(f'✓ Saved: {OUTPUT}  ({13*300}×{6*300} px at 300 dpi)')
