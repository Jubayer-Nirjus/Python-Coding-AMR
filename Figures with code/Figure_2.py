"""
Figure 2 — WHO AWaRe Antibiotic Category Distribution
Preventive Veterinary Medicine (Elsevier) — journal-compliant output

Journal requirements applied:
  - No figure title on the figure itself (caption only)
  - PNG, 300 dpi, 13×6 inches = 3900×1800 px ✓
  - Wong (2011) colorblind-safe palette
  - Output: Figure_2.png

CAPTION (use in manuscript):
  Fig. 2. WHO AWaRe antibiotic category distribution among antimicrobial-using poultry
  farms, stratified by farm type. Bars represent the percentage of antimicrobial-using
  farms (n = 165) classified into Access, Watch, or Reserve categories per WHO AWaRe
  2023. Reserve category includes colistin following reclassification (five farms).
  Non-antimicrobial-using farms (n = 47) are excluded. Panel definitions: Access —
  broad-spectrum first-line antibiotics; Watch — higher resistance potential, restricted
  indications; Reserve — last-resort antibiotics critical for human medicine.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

EXCEL_FILE = 'CP_AMR_Master_File_V13.xlsx'
OUTPUT     = 'Figure_2.png'

# Wong (2011) colorblind-safe palette
COLORS = {
    'blue'       : '#0072B2',
    'sky_blue'   : '#56B4E9',
    'orange'     : '#E69F00',
    'vermillion' : '#D55E00',
    'green'      : '#009E73',
    'pink'       : '#CC79A7',
    'black'      : '#000000',
    'gray_dark'  : '#404040',
    'gray_mid'   : '#767676',
    'gray_light' : '#D9D9D9',
    # AWaRe specific (colorblind-safe)
    'access'     : '#009E73',   # green — safe/accessible
    'watch'      : '#E69F00',   # orange — caution
    'reserve'    : '#D55E00',   # vermillion — restrict
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

AM   = df[df['AM_use_binary'] == 0].copy()
b_am = AM[AM['Farm_Type'] == 0]
l_am = AM[AM['Farm_Type'] == 1]
s_am = AM[AM['Farm_Type'] == 2]

def pct(data, col, code):
    d = data[col].dropna()
    d = d[d != 99]
    return round((d == code).sum() / len(d) * 100, 1) if len(d) > 0 else 0.0

GROUPS = {
    f'Overall\n(n={len(AM)})'   : AM,
    f'Broiler\n(n={len(b_am)})' : b_am,
    f'Layer\n(n={len(l_am)})'   : l_am,
    f'Sonali\n(n={len(s_am)})'  : s_am,
}
labels       = list(GROUPS.keys())
access_vals  = [pct(g, 'AWaRE_Category', 0) for g in GROUPS.values()]
watch_vals   = [pct(g, 'AWaRE_Category', 1) for g in GROUPS.values()]
reserve_vals = [pct(g, 'AWaRE_Category', 2) for g in GROUPS.values()]

# ── FIGURE ──
fig, axes = plt.subplots(
    1, 2, figsize=(13, 6),
    gridspec_kw={'width_ratios': [2.6, 1], 'wspace': 0.05},
)
ax_bar = axes[0]
ax_ann = axes[1]

x  = np.arange(len(labels))
bw = 0.52

ax_bar.bar(x, access_vals,  bw, color=COLORS['access'],
           label='Access',  edgecolor='white', linewidth=0.7)
ax_bar.bar(x, watch_vals,   bw, color=COLORS['watch'],
           label='Watch',   edgecolor='white', linewidth=0.7,
           bottom=access_vals)
ax_bar.bar(x, reserve_vals, bw, color=COLORS['reserve'],
           label='Reserve', edgecolor='white', linewidth=0.7,
           bottom=[a + w for a, w in zip(access_vals, watch_vals)])

for i, (a, w, r) in enumerate(zip(access_vals, watch_vals, reserve_vals)):
    if a >= 7:
        ax_bar.text(i, a / 2, f'{a:.1f}%',
                    ha='center', va='center', fontsize=9.5, color='white', fontweight='bold')
    if w >= 7:
        ax_bar.text(i, a + w / 2, f'{w:.1f}%',
                    ha='center', va='center', fontsize=9.5, color='white', fontweight='bold')
    ax_bar.text(i, a + w + r / 2, f'{r:.1f}%',
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=11)
ax_bar.set_ylabel('AM-using farms (%)', fontsize=10.5)
ax_bar.set_ylim(0, 115)
ax_bar.set_yticks(range(0, 101, 20))
ax_bar.tick_params(axis='y', labelsize=9.5)
ax_bar.axhline(y=0, color=COLORS['gray_dark'], linewidth=0.8)
ax_bar.legend(
    loc='upper center', fontsize=8.5, framealpha=0.95,
    edgecolor=COLORS['gray_light'],
    title='WHO AWaRe category', title_fontsize=8.5,
)

# ── RIGHT PANEL — definitions ──
ax_ann.set_xlim(0, 1)
ax_ann.set_ylim(0, 1)
ax_ann.axis('off')

AWARE_DEFS = [
    ('Access',   COLORS['access'],  '#E8F5EE',
     'Broad-spectrum first-line\nantibiotics. Appropriate\nfor common infections.'),
    ('Watch',    COLORS['watch'],   '#FFF8E1',
     'Higher resistance potential.\nFor specific indications\nonly.'),
    ('Reserve',  COLORS['reserve'], '#FFECE8',
     'Last-resort antibiotics.\nCritical for human medicine.\nUse strictly limited.'),
]

BOX_H   = 0.29
GAP     = 0.04
y_start = 0.97

for cat, border_c, fill_c, desc in AWARE_DEFS:
    y_box = y_start - BOX_H
    ax_ann.add_patch(mpatches.FancyBboxPatch(
        (0.04, y_box), 0.92, BOX_H,
        boxstyle='round,pad=0.025',
        facecolor=fill_c, edgecolor=border_c, linewidth=2.0, clip_on=False,
    ))
    ax_ann.text(0.50, y_start - 0.055, cat,
                ha='center', va='top', fontsize=10, fontweight='bold', color=border_c)
    ax_ann.text(0.50, y_start - 0.125, desc,
                ha='center', va='top', fontsize=8.5, color=COLORS['gray_dark'], linespacing=1.5)
    y_start -= (BOX_H + GAP)

ax_ann.text(0.50, 0.01, 'WHO AWaRe 2023\n(Colistin = Reserve)',
            ha='center', fontsize=7.5, color=COLORS['gray_mid'], style='italic')

fig.text(
    0.01, -0.02,
    f'AM = antimicrobial. n (AM-using) = {len(AM)}; non-users (n = {len(df[df["AM_use_binary"]==1])}) excluded. '
    'Farms using multiple antibiotic categories assigned to the highest-risk category.',
    fontsize=8, color=COLORS['gray_mid'],
)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, format='png')
plt.close()
print(f'✓ Saved: {OUTPUT}')
