
"""
Figure 2 — WHO AWaRe Category Distribution
Run: python figure2_aware.py
Output: Figure2_AWaRe_Distribution.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──
EXCEL_FILE = 'AMR_Poultry_Analysis_Ready.xlsx'
OUTPUT     = 'Figure2_AWaRe_Distribution.png'

COLORS = {
    'dark_blue'  : '#1F4E79',
    'mid_blue'   : '#2E75B6',
    'light_blue' : '#BDD7EE',
    'access'     : '#375623',
    'watch'      : '#BF6000',
    'reserve'    : '#C00000',
    'gray_dark'  : '#404040',
    'gray_mid'   : '#767676',
    'gray_light' : '#D9D9D9',
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
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').replace(99, np.nan)

AM   = df[df['AM_use_binary'] == 0]
b_am = AM[AM['Farm_Type'] == 0]
l_am = AM[AM['Farm_Type'] == 1]
s_am = AM[AM['Farm_Type'] == 2]

def pct(data, col, code):
    d = data[col].dropna()
    return round((d == code).sum() / len(d) * 100, 1) if len(d) > 0 else 0.0

# ── DATA ──
GROUPS = {
    f'Overall\n(n={len(AM)})'  : AM,
    f'Broiler\n(n={len(b_am)})': b_am,
    f'Layer\n(n={len(l_am)})'  : l_am,
    f'Sonali\n(n={len(s_am)})' : s_am,
}
labels       = list(GROUPS.keys())
access_vals  = [pct(g, 'AWaRE_Category', 0) for g in GROUPS.values()]
watch_vals   = [pct(g, 'AWaRE_Category', 1) for g in GROUPS.values()]
reserve_vals = [pct(g, 'AWaRE_Category', 2) for g in GROUPS.values()]

# ── FIGURE ──
# Wide figure, left panel for chart, right panel for legend/annotation
fig, axes = plt.subplots(
    1, 2,
    figsize     = (13, 6),
    gridspec_kw = {'width_ratios': [2.6, 1], 'wspace': 0.05},
)
ax_bar = axes[0]
ax_ann = axes[1]

x  = np.arange(len(labels))
bw = 0.50

# Bars
b1 = ax_bar.bar(x, access_vals,  bw,
                color=COLORS['access'],  label='Access',
                edgecolor='white', linewidth=0.7)
b2 = ax_bar.bar(x, watch_vals,   bw,
                color=COLORS['watch'],   label='Watch',
                edgecolor='white', linewidth=0.7,
                bottom=access_vals)
b3 = ax_bar.bar(x, reserve_vals, bw,
                color=COLORS['reserve'], label='Reserve',
                edgecolor='white', linewidth=0.7,
                bottom=[a + w for a, w in zip(access_vals, watch_vals)])

# Value labels (only if segment ≥ 7%)
for i, (a, w, r) in enumerate(zip(access_vals, watch_vals, reserve_vals)):
    if a >= 7:
        ax_bar.text(i, a / 2,
                    f'{a:.1f}%', ha='center', va='center',
                    fontsize=9.5, color='white', fontweight='bold')
    if w >= 7:
        ax_bar.text(i, a + w / 2,
                    f'{w:.1f}%', ha='center', va='center',
                    fontsize=9.5, color='white', fontweight='bold')
    if r >= 7 or r < 7:
        ax_bar.text(i, a + w + r / 2,
                    f'{r:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=11)
ax_bar.set_ylabel('% of AM-Using Farms', fontsize=10.5)
ax_bar.set_ylim(0, 120)
ax_bar.set_yticks(range(0, 101, 20))
ax_bar.tick_params(axis='y', labelsize=9.5)
ax_bar.axhline(y=0, color=COLORS['gray_dark'], linewidth=0.8)

ax_bar.legend(
    loc='upper center', fontsize=7.5,
    framealpha=0.95, edgecolor=COLORS['gray_light'],
    title='WHO AWaRe Category', title_fontsize=8.5,
)

ax_bar.set_title(
    'Figure 2 — WHO AWaRe Antibiotic Category Distribution by Farm Type\n'
    '(AM-using farms only, n=165)',
    loc='left', fontsize=12, fontweight='bold',
    color=COLORS['dark_blue'], pad=12,
)

# ── RIGHT PANEL — definitions ──
ax_ann.set_xlim(0, 1)
ax_ann.set_ylim(0, 1)
ax_ann.axis('off')

AWARE_DEFS = [
    ('Access',       COLORS['access'],  '#E8F5E9',
     'Broad-spectrum first-line\nantibiotics. Appropriate\nfor common infections.'),
    ('Watch',    COLORS['watch'],   '#FFF8E1',
     'Higher resistance potential.\nShould be used only for\nspecific indications.'),
    ('Reserve', COLORS['reserve'], '#FFEBEE',
     'Last-resort antibiotics.\nCritical for human medicine.\nUse strictly limited.'),
]


# Fixed box heights and positions — no overlap
BOX_H   = 0.30
GAP     = 0.04
y_start = 0.97

for cat, border_c, fill_c, desc in AWARE_DEFS:
    y_box = y_start - BOX_H

    ax_ann.add_patch(mpatches.FancyBboxPatch(
        (0.04, y_box), 0.92, BOX_H,
        boxstyle    = 'round,pad=0.025',
        facecolor   = fill_c,
        edgecolor   = border_c,
        linewidth   = 2.0,
        clip_on     = False,
    ))
    # Category label
    ax_ann.text(
        0.50, y_start - 0.055,
        cat,
        ha='center', va='top',
        fontsize=10, fontweight='bold', color=border_c,
    )
    # Description
    ax_ann.text(
        0.50, y_start - 0.125,
        desc,
        ha='center', va='top',
        fontsize=8.5, color=COLORS['gray_dark'], linespacing=1.55,
    )

    y_start -= (BOX_H + GAP)

ax_ann.text(
    0.50, 0.01,
    'WHO AWaRe Classification 2021',
    ha='center', fontsize=7.5,
    color=COLORS['gray_mid'], style='italic',
)

# ── FOOTNOTE ──
fig.text(
    0.01, -0.03,
    'Note: Denominator = AM-using farms only (n=165). '
    'Farms using multiple antibiotic categories classified to highest-risk category (worst-case). '
    'Non-users (n=47) excluded.',
    fontsize=8, color=COLORS['gray_mid'],
)

plt.savefig(OUTPUT, dpi=300)
plt.close()
print(f"✓ Saved: {OUTPUT}")
