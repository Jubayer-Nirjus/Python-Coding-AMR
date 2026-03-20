
"""
Figure 3 — AI & Digital Technology Adoption Landscape
Run: python figure3_ai_adoption.py
Output: Figure3_AI_Adoption_Landscape.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──
EXCEL_FILE = 'AMR_Poultry_Analysis_Ready.xlsx'
OUTPUT     = 'Figure3_AI_Adoption_Landscape.png'

COLORS = {
    'dark_blue'  : '#1F4E79',
    'mid_blue'   : '#2E75B6',
    'light_blue' : '#BDD7EE',
    'green_dark' : '#1E5631',
    'green_mid'  : '#375623',
    'green_light': '#70AD47',
    'red'        : '#C00000',
    'gray_dark'  : '#404040',
    'gray_mid'   : '#767676',
    'gray_light' : '#D9D9D9',
    'maybe_blue' : '#9DC3E6',
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

N  = len(df)
BR = df[df['Farm_Type'] == 0]
LA = df[df['Farm_Type'] == 1]
SO = df[df['Farm_Type'] == 2]
FT_GROUPS = [BR, LA, SO]

def pct(data, col, code):
    d = data[col].dropna()
    return round((d == code).sum() / len(d) * 100, 1) if len(d) > 0 else 0.0

# ── FIGURE — 3 tall subplots side by side ──
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3,
    figsize     = (15, 7.5),
    gridspec_kw = {'wspace': 0.42},
)

FT_LABELS = [f'Broiler\n(n={len(BR)})', f'Layer\n(n={len(LA)})', f'Sonali\n(n={len(SO)})']
x  = np.arange(3)
bw = 0.52

def label_bar(ax, x_pos, bottom, height, txt, min_h=5):
    """Inside label with adaptive font size."""

    if height >= min_h:
        fontsize = 9.5
    else:
        fontsize = 7  # 👈 small font but still visible

    ax.text(
        x_pos, bottom + height / 2,   # 👈 ALWAYS inside
        txt,
        ha='center', va='center',
        fontsize=fontsize,
        color='white',
        fontweight='bold'
    )
def overall_badge(ax, text, bg, border):
    """Floating 'Overall' badge at top of panel."""
    ax.text(
        1, 112, text,
        ha='center', fontsize=9, color=border, fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor=bg, edgecolor=border,
            linewidth=1.2, alpha=0.97,
        ),
    )

# ── PANEL A — Automation ──────────────────────────────────────────────────
auto_yes = [pct(g, 'Use_of_Automation', 1) for g in FT_GROUPS]
auto_no  = [100 - v for v in auto_yes]

ax1.bar(x, auto_yes, bw, color=COLORS['mid_blue'],   label='Any Automation', edgecolor='white', linewidth=0.7)
ax1.bar(x, auto_no,  bw, color=COLORS['gray_light'], label='No Automation',  edgecolor='white', linewidth=0.7, bottom=auto_yes)

for i in range(3):
    label_bar(ax1, i, 0,           auto_yes[i], f'{auto_yes[i]:.1f}%')
    label_bar(ax1, i, auto_yes[i], auto_no[i],  f'{auto_no[i]:.1f}%', min_h=8)
    if auto_no[i] < 5:
        # print outside bar if too small
        pass

n_auto_ov = int((df['Use_of_Automation'] == 1).sum())
overall_badge(ax1, f'Overall: {n_auto_ov/N*100:.1f}%', COLORS['light_blue'], COLORS['dark_blue'])

ax1.set_xticks(x); ax1.set_xticklabels(FT_LABELS, fontsize=10.5)
ax1.set_ylim(0, 125); ax1.set_yticks(range(0, 101, 20))
ax1.set_ylabel('% of Farms', fontsize=10.5)
ax1.set_title('A. Automation Technology\nAdoption', fontsize=11, fontweight='bold',
              color=COLORS['dark_blue'], pad=10)
ax1.legend(
    loc='upper right',
    bbox_to_anchor=(1.40, 1.00),
    fontsize=9,
    framealpha=0.95
)
ax1.axhline(y=0, color=COLORS['gray_dark'], linewidth=0.8)

# ── PANEL B — AI Use Frequency ────────────────────────────────────────────
ai_reg   = [pct(g, 'AI_Use_6mo', 0) for g in FT_GROUPS]
ai_some  = [pct(g, 'AI_Use_6mo', 1) for g in FT_GROUPS]
ai_never = [pct(g, 'AI_Use_6mo', 2) for g in FT_GROUPS]
bot_some = [r + s for r, s in zip(ai_reg, ai_some)]

ax2.bar(x, ai_reg,   bw, color=COLORS['green_mid'],  label='Regularly', edgecolor='white', linewidth=0.7)
ax2.bar(x, ai_some,  bw, color=COLORS['green_light'],label='Sometimes', edgecolor='white', linewidth=0.7, bottom=ai_reg)
ax2.bar(x, ai_never, bw, color=COLORS['gray_light'], label='Never',     edgecolor='white', linewidth=0.7, bottom=bot_some)

for i in range(3):
    label_bar(ax2, i, 0,          ai_reg[i],   f'{ai_reg[i]:.1f}%',  min_h=4)
    label_bar(ax2, i, ai_reg[i],  ai_some[i],  f'{ai_some[i]:.1f}%', min_h=4)
    label_bar(ax2, i, bot_some[i],ai_never[i], f'{ai_never[i]:.1f}%',min_h=4) 

n_reg_ov  = int((df['AI_Use_6mo'] == 0).sum())
n_some_ov = int((df['AI_Use_6mo'] == 1).sum())
overall_badge(ax2,
    f'Reg: {n_reg_ov/N*100:.1f}%  |  Some: {n_some_ov/N*100:.1f}%',
    '#E8F5E9', COLORS['green_mid'])

ax2.set_xticks(x); ax2.set_xticklabels(FT_LABELS, fontsize=10.5)
ax2.set_ylim(0, 125); ax2.set_yticks(range(0, 101, 20))
ax2.set_ylabel('% of Farms', fontsize=10.5)
ax2.set_title('B. AI / Digital Tool Use\n(Last 6 Months)', fontsize=11, fontweight='bold',
              color=COLORS['dark_blue'], pad=10)
ax2.legend(
    loc='upper right',
    bbox_to_anchor=(1.40, 1.00),
    fontsize=9,
    framealpha=0.95
)
ax2.axhline(y=0, color=COLORS['gray_dark'], linewidth=0.8)

# ── PANEL C — AI Adoption Willingness ────────────────────────────────────
will_yes   = [pct(g, 'AI_Adoption_Willingness', 0) for g in FT_GROUPS]
will_maybe = [pct(g, 'AI_Adoption_Willingness', 2) for g in FT_GROUPS]
will_no    = [pct(g, 'AI_Adoption_Willingness', 1) for g in FT_GROUPS]
bot_maybe  = [y + m for y, m in zip(will_yes, will_maybe)]

ax3.bar(x, will_yes,   bw, color=COLORS['mid_blue'],   label='Yes',   edgecolor='white', linewidth=0.7)
ax3.bar(x, will_maybe, bw, color=COLORS['maybe_blue'], label='Maybe', edgecolor='white', linewidth=0.7, bottom=will_yes)
ax3.bar(x, will_no,    bw, color=COLORS['red'],         label='No',    edgecolor='white', linewidth=0.7, bottom=bot_maybe)

for i in range(3):
    label_bar(ax3, i, 0,           will_yes[i],   f'{will_yes[i]:.1f}%',   min_h=5)
    label_bar(ax3, i, will_yes[i], will_maybe[i], f'{will_maybe[i]:.1f}%', min_h=5)
    label_bar(ax3, i, bot_maybe[i],will_no[i],    f'{will_no[i]:.1f}%',    min_h=5)

n_yes_ov = int((df['AI_Adoption_Willingness'] == 0).sum())
overall_badge(ax3, f'Overall Yes: {n_yes_ov/N*100:.1f}%', COLORS['light_blue'], COLORS['dark_blue'])

ax3.set_xticks(x); ax3.set_xticklabels(FT_LABELS, fontsize=10.5)
ax3.set_ylim(0, 125); ax3.set_yticks(range(0, 101, 20))
ax3.set_ylabel('% of Farms', fontsize=10.5)
ax3.set_title('C. AI Adoption Willingness\n(if ROI = 2 cycles)', fontsize=11, fontweight='bold',
              color=COLORS['dark_blue'], pad=10)
ax3.legend(
    loc='upper right',
    bbox_to_anchor=(1.30, 1.00),
    fontsize=9,
    framealpha=0.95
)
ax3.axhline(y=0, color=COLORS['gray_dark'], linewidth=0.8)

# ── OVERALL TITLE ──
fig.suptitle(
    'Figure 3 — AI & Digital Technology Adoption Landscape by Farm Type  (n=212)',
    fontsize=13, fontweight='bold', color=COLORS['dark_blue'],
    y=1.02,
)

# ── FOOTNOTE ──
fig.text(
    0.01, -0.03,
    'Note: Automation = any technology (CCTV, auto feeders/drinkers, climate sensors, flock apps). '
    'AI Use = any digital/AI tool used in the last 6 months. '
    'Willingness conditional on 2-cycle return-on-investment scenario.',
    fontsize=8.5, color=COLORS['gray_mid'],
)

plt.savefig(OUTPUT, dpi=300)
plt.close()
print(f"✓ Saved: {OUTPUT}")
