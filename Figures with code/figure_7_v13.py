"""
Figure 7 — Geographic Distribution: AMR Risk Index & Knowledge Score
Updated for CP_AMR_Master_File_V13.xlsx
Output: Figure7_Geographic_Risk_Map.png

Notes (V13):
  - 197 farms have valid GPS coordinates (15 missing)
  - 'Unknown' district (n=23) excluded from district bar chart
  - AMR Risk categories: Low(<3), Moderate(3-4), High(>=5)
  - Ground-truth: Low=114(53.8%), Mod=71(33.5%), High=27(12.7%)
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
OUTPUT     = 'Figure7_Geographic_Risk_Map.png'

# ── LOAD ──
df = pd.read_excel(EXCEL_FILE, sheet_name='1_Master_Data', header=1)
df = df.dropna(subset=['Unique_ID'])
for col in ['Latitude', 'Longitude', 'AMR_Risk_Index', 'Knowledge_Score']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

N_total = len(df)

# Filter valid coordinates within Bangladesh bounds
df_geo = df.dropna(subset=['Latitude', 'Longitude']).copy()
df_geo = df_geo[
    (df_geo['Latitude']  >= 20.5) & (df_geo['Latitude']  <= 26.5) &
    (df_geo['Longitude'] >= 88.0) & (df_geo['Longitude'] <= 92.5)
]
n_plotted = len(df_geo)
print(f'Farms with valid coordinates: {n_plotted} / {N_total}')

# Exclude 'Unknown' district from district-level stats
df_district = df_geo[~df_geo['District'].isin(['Unknown', '', None])].copy()
df_district = df_district.dropna(subset=['District'])

# Risk category
def risk_cat(risk):
    if   risk < 3:  return 'Low'
    elif risk < 5:  return 'Moderate'
    else:           return 'High'

df_geo['Risk_Category']  = df_geo['AMR_Risk_Index'].apply(risk_cat)
risk_colors = {'Low': '#2ecc71', 'Moderate': '#f1c40f', 'High': '#e74c3c'}

# District centroids
district_coords = df_geo[df_geo['District'] != 'Unknown'].groupby('District')[
    ['Longitude', 'Latitude']].mean().reset_index()

# District-level stats (exclude Unknown)
district_stats = df_district.groupby('District')['AMR_Risk_Index'].agg(
    ['mean', 'std', 'count']).reset_index()
district_stats = district_stats.sort_values('mean', ascending=False)
overall_mean = df['AMR_Risk_Index'].dropna().mean()

print(f'Districts in chart: {list(district_stats["District"])}')
print(f'Overall AMR Risk Mean: {overall_mean:.2f}')

# ── STYLING ──
plt.rcParams.update({
    'font.family'  : 'DejaVu Sans',
    'font.size'    : 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.dpi'  : 300,
    'savefig.bbox' : 'tight',
})

# ── FIGURE ──
fig = plt.figure(figsize=(16, 12))
gs  = GridSpec(2, 2, height_ratios=[1, 0.7], width_ratios=[1, 1],
               hspace=0.28, wspace=0.30)

# ── Panel A: AMR Risk by Location ──
ax_a = fig.add_subplot(gs[0, 0])

for cat in ['Low', 'Moderate', 'High']:
    sub = df_geo[df_geo['Risk_Category'] == cat]
    ax_a.scatter(sub['Longitude'], sub['Latitude'],
                 c=risk_colors[cat], s=50, edgecolors='white', linewidth=0.8,
                 label=f'{cat} (n={len(sub)})', alpha=0.85, zorder=3)

# Annotate named districts only
for _, row in district_coords.iterrows():
    ax_a.annotate(row['District'],
                  xy=(row['Longitude'], row['Latitude']),
                  xytext=(3, 3), textcoords='offset points',
                  fontsize=7.5, ha='left', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            alpha=0.75, edgecolor='none'))

ax_a.set_xlim(88.3, 92.0)
ax_a.set_ylim(21.5, 26.8)
ax_a.set_xlabel('Longitude', fontweight='bold')
ax_a.set_ylabel('Latitude', fontweight='bold')
ax_a.set_title('A. AMR Risk Index by Farm Location', fontweight='bold')
ax_a.legend(title='Risk Level', loc='upper right', frameon=True, fancybox=True)
ax_a.grid(True, linestyle='--', alpha=0.4)

# Summary box
ax_a.text(0.02, 0.02,
          f'n_plotted={n_plotted}/{N_total}\n'
          f'High Risk: {len(df_geo[df_geo["Risk_Category"]=="High"])} farms\n'
          f'Mean Risk Index: {overall_mean:.2f}±{df["AMR_Risk_Index"].std():.2f}',
          transform=ax_a.transAxes, fontsize=8, va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#AAAAAA', alpha=0.9))

# ── Panel B: Knowledge Score by Location ──
ax_b = fig.add_subplot(gs[0, 1])
valid_geo_k = df_geo.dropna(subset=['Knowledge_Score'])
sc = ax_b.scatter(valid_geo_k['Longitude'], valid_geo_k['Latitude'],
                  c=valid_geo_k['Knowledge_Score'], s=50, edgecolors='white',
                  linewidth=0.8, cmap='viridis', alpha=0.85, vmin=0, vmax=6, zorder=3)

for _, row in district_coords.iterrows():
    ax_b.annotate(row['District'],
                  xy=(row['Longitude'], row['Latitude']),
                  xytext=(3, 3), textcoords='offset points',
                  fontsize=7.5, ha='left', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            alpha=0.75, edgecolor='none'))

ax_b.set_xlim(88.3, 92.0)
ax_b.set_ylim(21.5, 26.8)
ax_b.set_xlabel('Longitude', fontweight='bold')
ax_b.set_ylabel('Latitude', fontweight='bold')
ax_b.set_title('B. Knowledge Score by Farm Location', fontweight='bold')
cbar = plt.colorbar(sc, ax=ax_b, shrink=0.8)
cbar.set_label('Knowledge Score (0–6)', fontsize=9)
cbar.ax.tick_params(labelsize=8)
ax_b.grid(True, linestyle='--', alpha=0.4)

k_mean = df['Knowledge_Score'].dropna().mean()
ax_b.text(0.02, 0.02,
          f'Mean Knowledge: {k_mean:.2f}\nRange: 0–6',
          transform=ax_b.transAxes, fontsize=8, va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#AAAAAA', alpha=0.9))

# ── Panel C: Mean AMR Risk by District (bar chart) ──
ax_c = fig.add_subplot(gs[1, :])

# Colour bars by risk level
bar_colors = []
for mean_val in district_stats['mean']:
    if   mean_val >= 5: bar_colors.append('#e74c3c')
    elif mean_val >= 3: bar_colors.append('#f1c40f')
    else:               bar_colors.append('#2ecc71')

bars = ax_c.bar(district_stats['District'], district_stats['mean'],
                yerr=district_stats['std'].fillna(0), capsize=4,
                color=bar_colors, edgecolor='#333333', linewidth=0.8,
                alpha=0.82, zorder=2, error_kw={'linewidth': 1.2})

# n label on each bar
for bar, (_, row) in zip(bars, district_stats.iterrows()):
    ax_c.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + (row['std'] if not np.isnan(row['std']) else 0) + 0.12,
              f'{row["mean"]:.1f}\n(n={int(row["count"])})',
              ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333333')

# Overall mean line
ax_c.axhline(overall_mean, color='#C00000', linestyle='--', linewidth=2,
             label=f'Overall Mean = {overall_mean:.2f}', zorder=3)

# Risk zone bands
ax_c.axhspan(0,   3, alpha=0.05, color='#2ecc71')
ax_c.axhspan(3,   5, alpha=0.05, color='#f1c40f')
ax_c.axhspan(5, 8.5, alpha=0.05, color='#e74c3c')

ax_c.set_xlabel('District', fontweight='bold')
ax_c.set_ylabel('Mean AMR Risk Index (±SD)', fontweight='bold')
ax_c.set_title('C. Mean AMR Risk Index by District  (n ≥ 5 shown; "Unknown" GPS excluded)',
               fontweight='bold')
ax_c.legend(loc='upper right', frameon=True)
ax_c.set_ylim(0, 5.5)
ax_c.grid(axis='y', linestyle='--', alpha=0.4)
plt.setp(ax_c.get_xticklabels(), rotation=35, ha='right')

# ── TITLE & FOOTNOTE ──
fig.suptitle(
    'Figure 7 — Geographic Distribution: AMR Risk Index & Knowledge Score\n'
    f'Commercial Poultry Farms, Bangladesh (n={N_total})',
    fontsize=15, fontweight='bold', y=0.99,
)
fig.text(
    0.5, 0.01,
    f'Note: GPS from field survey. {n_plotted} of {N_total} farms have valid coordinates; '
    '23 farms with "Unknown" district excluded from Panel C. '
    'Risk bands: Green=Low(0–2), Yellow=Moderate(3–4), Red=High(≥5). '
    'Dashed line = overall mean AMR Risk Index. '
    'Source: CP_AMR_Master_File_V13.xlsx',
    ha='center', fontsize=8.5, style='italic',
)

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig(OUTPUT, dpi=300, bbox_inches='tight')
plt.close()
print(f'✓ Saved: {OUTPUT}')
