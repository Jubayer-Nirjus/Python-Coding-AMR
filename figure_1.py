
"""
Figure 1 — AMR Knowledge Gap Heatmap
Run: python figure1_knowledge_heatmap.py
Output: Figure1_Knowledge_Gap_Heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──
EXCEL_FILE = 'AMR_Poultry_Analysis_Ready.xlsx'
OUTPUT     = 'Figure1_Knowledge_Gap_Heatmap.png'

COLORS = {
    'dark_blue'  : '#1F4E79',
    'mid_blue'   : '#2E75B6',
    'light_blue' : '#BDD7EE',
    'gold'       : '#BF6000',
    'gold_light' : '#FFF2CC',
    'gray_dark'  : '#404040',
    'gray_mid'   : '#767676',
    'gray_light' : '#D9D9D9',
}

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.linewidth'    : 0.8,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.25,
})

# ── LOAD ──
df = pd.read_excel(EXCEL_FILE, sheet_name='1_Master_Data', header=1)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').replace(99, np.nan)

N  = len(df)
BR = df[df['Farm_Type'] == 0]   # Broiler  n=109
LA = df[df['Farm_Type'] == 1]   # Layer    n=70
SO = df[df['Farm_Type'] == 2]   # Sonali   n=33

def pct(data, col, code=0):
    d = data[col].dropna()
    return round((d == code).sum() / len(d) * 100, 1) if len(d) > 0 else 0.0

# ── DATA ──
# Items ordered worst → best (bottom = hardest question, top = easiest)
KNOW_ITEMS = [
    ('Heard_of_AMR',            0, 'Has Heard of AMR / Antibiotic Resistance'),
    ('Herbal_Drug_Knowledge',   0, 'Aware of Herbal / Alternative Drugs'),
    ('Perceived_Causes_AMR',    0, 'Can Identify Causes of AMR'),
    ('Withdrawal_Awareness_K',  0, 'Aware of Antibiotic Withdrawal Period'),
    ('AMR_Misuse_Impact',       0, 'Knows Misuse Impacts Human Health'),
    ('AMR_Risk_Less_Effective', 0, 'Knows Antibiotics Becoming Less Effective'),
]

COL_GROUPS = {
    f'Overall\n(n={N})'  : df,
    f'Broiler\n(n={len(BR)})' : BR,
    f'Layer\n(n={len(LA)})'   : LA,
    f'Sonali\n(n={len(SO)})'  : SO,
    f'Graduate\n(n={len(df[df["Education"]==0])})' : df[df['Education'] == 0],
    f'College\n(n={len(df[df["Education"]==1])})'  : df[df['Education'] == 1],
    f'Primary\n(n={len(df[df["Education"]==2])})'  : df[df['Education'] == 2],
}

hm_df = pd.DataFrame(
    [[pct(g, col, code) for g in COL_GROUPS.values()]
     for col, code, _ in KNOW_ITEMS],
    index   = [lbl for _, _, lbl in KNOW_ITEMS],
    columns = list(COL_GROUPS.keys()),
)

# ── FIGURE ──
fig, ax = plt.subplots(figsize=(13, 6))

# Palette: deep red (0%) → white (50%) → deep green (100%)
cmap = sns.diverging_palette(10, 130, s=90, l=38, center='light', as_cmap=True)

hm = sns.heatmap(
    hm_df,
    ax         = ax,
    annot      = True,
    fmt        = '.1f',
    cmap       = cmap,
    vmin       = 0,
    vmax       = 100,
    linewidths = 0.7,
    linecolor  = '#FFFFFF',
    annot_kws  = {'size': 10.5, 'weight': 'bold'},
    cbar_kws   = {
        'label'  : '% Answering Correctly',
        'shrink' : 0.72,
        'aspect' : 16,
        'pad'    : 0.02,
    },
)

# Annotation text colour: white on dark cells, dark on light cells
for text in ax.texts:
    try:
        val = float(text.get_text())
        if val < 30 or val > 72:
            text.set_color('white')
        else:
            text.set_color(COLORS['gray_dark'])
    except ValueError:
        pass

# Colorbar styling
cbar = hm.collections[0].colorbar
cbar.set_label('% Answering Correctly', fontsize=9.5, color=COLORS['gray_dark'])
cbar.ax.tick_params(labelsize=9)

# Vertical separator between Farm Type (cols 1–4) and Education (cols 5–7)
ax.axvline(x=4, color='white',              linewidth=4.5)
ax.axvline(x=4, color=COLORS['dark_blue'],  linewidth=1.5,
           linestyle='--', alpha=0.5)

# Group banners — using axis fraction coordinates
trans = ax.get_xaxis_transform()   # x in data units, y in axes fraction

ax.annotate(
    '◄── By Farm Type ──►',
    xy=(1.95, 1.03), xycoords=trans,
    fontsize=10, fontweight='bold', color=COLORS['dark_blue'],
    ha='center', va='bottom',
    bbox=dict(
        boxstyle='round,pad=0.28',
        facecolor=COLORS['light_blue'],
        edgecolor=COLORS['dark_blue'],
        linewidth=1.2, alpha=0.85,
    ),
)
ax.annotate(
    '◄── By Education Level ──►',
    xy=(5.5, 1.03), xycoords=trans,
    fontsize=10, fontweight='bold', color=COLORS['gold'],
    ha='center', va='bottom',
    bbox=dict(
        boxstyle='round,pad=0.28',
        facecolor=COLORS['gold_light'],
        edgecolor=COLORS['gold'],
        linewidth=1.2, alpha=0.85,
    ),
)

# Axis labels & ticks
ax.set_title(
    'Figure 1 — AMR Knowledge Gap: % Answering Correctly\n'
    'by Farm Type and Education Level',
    loc='left', pad=32,
    fontsize=12, fontweight='bold', color=COLORS['dark_blue'],
)
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='y', rotation=0,  labelsize=10)
ax.tick_params(axis='x', rotation=20, labelsize=10)

# Footnote
fig.text(
    0.01, -0.04,
    f'Note: Values = % of farmers answering each item correctly.  '
    f'Broiler n={len(BR)}, Layer n={len(LA)}, Sonali n={len(SO)}.  '
    f'Education — Graduate n={len(df[df["Education"]==0])}, '
    f'College n={len(df[df["Education"]==1])}, '
    f'Primary/Below n={len(df[df["Education"]==2])}.  '
    f'Total n={N}.',
    fontsize=8.5, color=COLORS['gray_mid'],
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(OUTPUT, dpi=300)
plt.close()
print(f"✓ Saved: {OUTPUT}")
