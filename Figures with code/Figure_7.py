"""
Figure 7 — Geographic Distribution: AMR Risk Index & Knowledge Score
Preventive Veterinary Medicine (Elsevier) — journal-compliant output

Journal requirements applied:
  - No figure title on figure (caption only)
  - PNG, 300 dpi, 16×12 in = 4800×3600 px ✓
  - Wong (2011) colorblind-safe palette for risk categories
  - Map note: lines delineate study areas, do not depict accepted national boundaries
  - Output: Figure_7.png

CAPTION (use in manuscript):
  Fig. 7. Geographic distribution of AMR Risk Index and AMR knowledge among commercial
  poultry farms surveyed in Bangladesh (n = 212; 197 farms with valid GPS coordinates
  plotted). (A) Farm locations colour-coded by AMR Risk category: Low (AMR Risk Index
  0–2), Moderate (3–4), High (≥5). (B) Farm locations colour-coded by AMR Knowledge
  Score (0–6; viridis scale). (C) Mean AMR Risk Index (±SD) by district. Dashed line
  indicates overall mean (2.40). Districts with fewer than five farms and those with
  unidentified GPS coordinates (n = 23, labelled "Unknown") are excluded from Panel C.
  Map lines delineate study areas and do not necessarily depict accepted national boundaries.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

EXCEL_FILE = 'CP_AMR_Master_File_V13.xlsx'
OUTPUT     = 'Figure_7.png'

# Wong (2011) colorblind-safe palette
# For risk: use blue (low), orange (moderate), vermillion (high) — avoids red-green conflict
COLORS = {
    'low'        : '#56B4E9',   # sky blue — low risk
    'moderate'   : '#E69F00',   # orange — moderate risk
    'high'       : '#D55E00',   # vermillion — high risk
    'blue'       : '#0072B2',
    'gray_dark'  : '#404040',
    'gray_mid'   : '#767676',
    'gray_light' : '#D9D9D9',
}

risk_colors = {
    'Low'      : COLORS['low'],
    'Moderate' : COLORS['moderate'],
    'High'     : COLORS['high'],
}

plt.rcParams.update({
    'font.family'   : 'DejaVu Sans',
    'font.size'     : 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.dpi'   : 300,
    'savefig.bbox'  : 'tight',
    'savefig.pad_inches': 0.15,
})

# ── LOAD ──
df = pd.read_excel(EXCEL_FILE, sheet_name='1_Master_Data', header=1)
df = df.dropna(subset=['Unique_ID'])
for col in ['Latitude', 'Longitude', 'AMR_Risk_Index', 'Knowledge_Score']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

N_total = len(df)
df_geo  = df.dropna(subset=['Latitude', 'Longitude']).copy()
df_geo  = df_geo[(df_geo['Latitude']  >= 20.5) & (df_geo['Latitude']  <= 26.5) &
                  (df_geo['Longitude'] >= 88.0) & (df_geo['Longitude'] <= 92.5)]
n_plotted = len(df_geo)
print(f'Farms plotted: {n_plotted} / {N_total}')

df_district = df_geo[~df_geo['District'].isin(['Unknown', '', None])].copy()
df_district = df_district.dropna(subset=['District'])

def risk_cat(risk):
    if   risk < 3:  return 'Low'
    elif risk < 5:  return 'Moderate'
    else:           return 'High'

df_geo['Risk_Category'] = df_geo['AMR_Risk_Index'].apply(risk_cat)
district_coords = df_geo[df_geo['District'] != 'Unknown'].groupby('District')[
    ['Longitude', 'Latitude']].mean().reset_index()
district_stats = df_district.groupby('District')['AMR_Risk_Index'].agg(
    ['mean', 'std', 'count']).reset_index()
district_stats = district_stats[district_stats['count'] >= 5]
district_stats = district_stats.sort_values('mean', ascending=False)
overall_mean   = df['AMR_Risk_Index'].dropna().mean()

# ── FIGURE ──
fig = plt.figure(figsize=(16, 12))
gs  = GridSpec(2, 2, height_ratios=[1, 0.72], width_ratios=[1, 1],
               hspace=0.28, wspace=0.30)

# ── Panel A: AMR Risk by Location ──
ax_a = fig.add_subplot(gs[0, 0])
for cat in ['Low', 'Moderate', 'High']:
    sub = df_geo[df_geo['Risk_Category'] == cat]
    ax_a.scatter(sub['Longitude'], sub['Latitude'],
                 c=risk_colors[cat], s=50, edgecolors='white', linewidth=0.8,
                 label=f'{cat} (n = {len(sub)})', alpha=0.85, zorder=3)

for _, row in district_coords.iterrows():
    ax_a.annotate(row['District'], xy=(row['Longitude'], row['Latitude']),
                  xytext=(3, 3), textcoords='offset points',
                  fontsize=7.5, ha='left', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            alpha=0.75, edgecolor='none'))

ax_a.set_xlim(88.3, 92.0); ax_a.set_ylim(21.5, 26.8)
ax_a.set_xlabel('Longitude', fontweight='bold')
ax_a.set_ylabel('Latitude', fontweight='bold')
ax_a.set_title('A. AMR Risk Category by Farm Location', fontweight='bold', pad=8)
ax_a.legend(title='Risk category', loc='upper right', frameon=True, fancybox=True,
            fontsize=9, title_fontsize=9)
ax_a.grid(True, linestyle='--', alpha=0.4)
ax_a.text(0.02, 0.02,
          f'n plotted = {n_plotted}/{N_total}\nOverall mean ± SD: {overall_mean:.2f} ± {df["AMR_Risk_Index"].std():.2f}',
          transform=ax_a.transAxes, fontsize=8, va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#AAAAAA', alpha=0.9))

# ── Panel B: Knowledge Score by Location ──
ax_b = fig.add_subplot(gs[0, 1])
valid_k = df_geo.dropna(subset=['Knowledge_Score'])
sc = ax_b.scatter(valid_k['Longitude'], valid_k['Latitude'],
                  c=valid_k['Knowledge_Score'], s=50, edgecolors='white',
                  linewidth=0.8, cmap='viridis', alpha=0.85, vmin=0, vmax=6, zorder=3)

for _, row in district_coords.iterrows():
    ax_b.annotate(row['District'], xy=(row['Longitude'], row['Latitude']),
                  xytext=(3, 3), textcoords='offset points',
                  fontsize=7.5, ha='left', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            alpha=0.75, edgecolor='none'))

ax_b.set_xlim(88.3, 92.0); ax_b.set_ylim(21.5, 26.8)
ax_b.set_xlabel('Longitude', fontweight='bold')
ax_b.set_ylabel('Latitude', fontweight='bold')
ax_b.set_title('B. AMR Knowledge Score by Farm Location', fontweight='bold', pad=8)
cbar = plt.colorbar(sc, ax=ax_b, shrink=0.80)
cbar.set_label('Knowledge Score (0–6)', fontsize=9)
cbar.ax.tick_params(labelsize=8)
ax_b.grid(True, linestyle='--', alpha=0.4)
ax_b.text(0.02, 0.02,
          f'Mean knowledge: {df["Knowledge_Score"].dropna().mean():.2f}\nRange: 0–6',
          transform=ax_b.transAxes, fontsize=8, va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#AAAAAA', alpha=0.9))

# ── Panel C: Mean AMR Risk by District ──
ax_c = fig.add_subplot(gs[1, :])
bar_colors_c = [COLORS['high'] if m >= 5 else
                COLORS['moderate'] if m >= 3 else
                COLORS['low'] for m in district_stats['mean']]

bars = ax_c.bar(district_stats['District'], district_stats['mean'],
                yerr=district_stats['std'].fillna(0), capsize=4,
                color=bar_colors_c, edgecolor='#333333', linewidth=0.8,
                alpha=0.82, zorder=2, error_kw={'linewidth': 1.2})

for bar, (_, row) in zip(bars, district_stats.iterrows()):
    sd_val = row['std'] if not np.isnan(row['std']) else 0
    ax_c.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + sd_val + 0.12,
              f'{row["mean"]:.1f}\n(n = {int(row["count"])})',
              ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333333')

ax_c.axhline(overall_mean, color=COLORS['blue'], linestyle='--', linewidth=2,
             label=f'Overall mean = {overall_mean:.2f}', zorder=3)

ax_c.axhspan(0, 3, alpha=0.04, color=COLORS['low'])
ax_c.axhspan(3, 5, alpha=0.04, color=COLORS['moderate'])
ax_c.axhspan(5, 8, alpha=0.04, color=COLORS['high'])

ax_c.set_xlabel('District', fontweight='bold')
ax_c.set_ylabel('Mean AMR Risk Index (±SD)', fontweight='bold')
ax_c.set_title('C. Mean AMR Risk Index by District  (districts with n ≥ 5 shown)',
               fontweight='bold', pad=8)
ax_c.legend(loc='upper right', frameon=True, fontsize=9)
ax_c.set_ylim(0, 5.5)
ax_c.grid(axis='y', linestyle='--', alpha=0.4)
plt.setp(ax_c.get_xticklabels(), rotation=35, ha='right')

fig.text(
    0.5, 0.01,
    f'GPS from field survey (ODK Collect). {n_plotted} of {N_total} farms have valid coordinates. '
    '23 farms with unidentified GPS location excluded from Panel C. '
    'Colour: sky blue = Low (0–2), orange = Moderate (3–4), vermillion = High (≥5). '
    'Map lines delineate study areas and do not necessarily depict accepted national boundaries.',
    ha='center', fontsize=8, style='italic',
)

plt.tight_layout(rect=[0, 0.04, 1, 0.99])
plt.savefig(OUTPUT, dpi=300, format='png', bbox_inches='tight')
plt.close()
print(f'✓ Saved: {OUTPUT}')
