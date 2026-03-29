"""
Figure 1 — AMR Knowledge Gap Heatmap
Updated for CP_AMR_Master_File_V13.xlsx
Output: Figure1_Knowledge_Gap_Heatmap.png

Coding reference (V13):
  Farm_Type:  0=Broiler, 1=Layer, 2=Sonali
  Education:  0=Graduate, 1=College/SSC, 2=Primary/Below
  Heard_of_AMR:         0=Yes(heard), 1=No
  Herbal_Drug_Knowledge:0=Yes(knows), 1=No
  Perceived_Causes_AMR: 0=Yes(can identify), 1=No
  Withdrawal_Awareness_K:0=Yes(aware), 1=No
  AMR_Misuse_Impact:    0=Yes(knows), 1=No
  AMR_Risk_Less_Effective:0=Yes(knows), 1=No
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
df = df.dropna(subset=['Unique_ID'])
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

N  = len(df)   # 212
BR = df[df['Farm_Type'] == 0]   # n=109
LA = df[df['Farm_Type'] == 1]   # n=70
SO = df[df['Farm_Type'] == 2]   # n=33

# In V13: correct answer code = 0 for all knowledge items
# pct() returns % who answered correctly (code==0)
def pct(data, col, code=0):
    d = data[col].dropna()
    # Exclude 99 (structural N/A)
    d = d[d != 99]
    return round((d == code).sum() / len(d) * 100, 1) if len(d) > 0 else 0.0

# ── KNOWLEDGE ITEMS ──
# All correct answers = 0 in V13
KNOW_ITEMS = [
    ('Heard_of_AMR',             0, 'Has Heard of AMR / Antibiotic Resistance'),
    ('Herbal_Drug_Knowledge',    0, 'Aware of Herbal / Alternative Drugs'),
    ('Perceived_Causes_AMR',     0, 'Can Identify Causes of AMR'),
    ('Withdrawal_Awareness_K',   0, 'Aware of Antibiotic Withdrawal Period'),
    ('AMR_Misuse_Impact',        0, 'Knows Misuse Impacts Human Health'),
    ('AMR_Risk_Less_Effective',  0, 'Knows Antibiotics Becoming Less Effective'),
]

edu_groups = {
    f'Graduate\n(n={len(df[df["Education"]==0])})' : df[df['Education'] == 0],
    f'College\n(n={len(df[df["Education"]==1])})'  : df[df['Education'] == 1],
    f'Primary\n(n={len(df[df["Education"]==2])})'  : df[df['Education'] == 2],
}

COL_GROUPS = {
    f'Overall\n(n={N})'           : df,
    f'Broiler\n(n={len(BR)})'     : BR,
    f'Layer\n(n={len(LA)})'       : LA,
    f'Sonali\n(n={len(SO)})'      : SO,
    **edu_groups,
}

hm_df = pd.DataFrame(
    [[pct(g, col, code) for g in COL_GROUPS.values()]
     for col, code, _ in KNOW_ITEMS],
    index   = [lbl for _, _, lbl in KNOW_ITEMS],
    columns = list(COL_GROUPS.keys()),
)

# ── FIGURE ──
fig, ax = plt.subplots(figsize=(13, 6))

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

for text in ax.texts:
    try:
        val = float(text.get_text())
        text.set_color('white' if (val < 30 or val > 72) else COLORS['gray_dark'])
    except ValueError:
        pass

cbar = hm.collections[0].colorbar
cbar.set_label('% Answering Correctly', fontsize=9.5, color=COLORS['gray_dark'])
cbar.ax.tick_params(labelsize=9)

# Separator between Farm Type and Education
ax.axvline(x=4, color='white',             linewidth=4.5)
ax.axvline(x=4, color=COLORS['dark_blue'], linewidth=1.5, linestyle='--', alpha=0.5)

trans = ax.get_xaxis_transform()
ax.annotate(
    '◄── By Farm Type ──►',
    xy=(1.95, 1.03), xycoords=trans,
    fontsize=10, fontweight='bold', color=COLORS['dark_blue'],
    ha='center', va='bottom',
    bbox=dict(boxstyle='round,pad=0.28', facecolor=COLORS['light_blue'],
              edgecolor=COLORS['dark_blue'], linewidth=1.2, alpha=0.85),
)
ax.annotate(
    '◄── By Education Level ──►',
    xy=(5.5, 1.03), xycoords=trans,
    fontsize=10, fontweight='bold', color=COLORS['gold'],
    ha='center', va='bottom',
    bbox=dict(boxstyle='round,pad=0.28', facecolor=COLORS['gold_light'],
              edgecolor=COLORS['gold'], linewidth=1.2, alpha=0.85),
)

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

fig.text(
    0.01, -0.04,
    f'Note: Values = % of farmers answering each item correctly. '
    f'Broiler n={len(BR)}, Layer n={len(LA)}, Sonali n={len(SO)}. '
    f'Education — Graduate n={len(df[df["Education"]==0])}, '
    f'College n={len(df[df["Education"]==1])}, '
    f'Primary/Below n={len(df[df["Education"]==2])}. '
    f'Total n={N}. Source: CP_AMR_Master_File_V13.xlsx',
    fontsize=8.5, color=COLORS['gray_mid'],
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(OUTPUT, dpi=300)
plt.close()
print(f'✓ Saved: {OUTPUT}')
