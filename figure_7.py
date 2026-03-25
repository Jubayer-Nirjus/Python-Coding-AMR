"""
Figure 7 — Geographic Distribution: AMR Risk Index & Knowledge Score
Commercial Poultry Farms, Bangladesh (n=212)
Enhanced version — saves figure as PNG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================================
# 1. LOAD DATA
# =============================================================================
file_path = 'Master_Analysis_Workbook.xlsx'

# Load master data (skip header row)
df = pd.read_excel(file_path, sheet_name='1_Master_Data', header=1)

# Ensure required columns are numeric
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['AMR_Risk_Index'] = pd.to_numeric(df['AMR_Risk_Index'], errors='coerce')
df['Knowledge_Score'] = pd.to_numeric(df['Knowledge_Score'], errors='coerce')

# Drop rows with missing coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])

# Filter to Bangladesh bounds (approx)
df = df[(df['Latitude'] >= 20.5) & (df['Latitude'] <= 26.5) &
        (df['Longitude'] >= 88.0) & (df['Longitude'] <= 92.5)]

# Remove rows where District is 'Unknown' or empty
df = df[~df['District'].isin(['Unknown', 'nan', None])]
df = df.dropna(subset=['District'])

print(f"Number of farms with valid coordinates: {len(df)}")
print("Districts present:", sorted(df['District'].unique()))

# =============================================================================
# 2. PREPARE DATA FOR PANELS A AND B
# =============================================================================
def risk_category(risk):
    if risk < 2:
        return 'Low'
    elif risk < 5:
        return 'Moderate'
    else:
        return 'High'

df['Risk_Category'] = df['AMR_Risk_Index'].apply(risk_category)
risk_colors = {'Low': '#2ecc71', 'Moderate': '#f1c40f', 'High': '#e74c3c'}

# District centroids for annotation
district_coords = df.groupby('District')[['Longitude', 'Latitude']].mean().reset_index()

# =============================================================================
# 3. PREPARE DATA FOR PANEL C
# =============================================================================
district_stats = df.groupby('District')['AMR_Risk_Index'].agg(['mean', 'std', 'count']).reset_index()
district_stats = district_stats.sort_values('mean', ascending=False)
overall_mean = df['AMR_Risk_Index'].mean()

# =============================================================================
# 4. STYLING
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# =============================================================================
# 5. CREATE FIGURE
# =============================================================================
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, height_ratios=[1, 0.7], width_ratios=[1, 1], hspace=0.25, wspace=0.3)

# ---- Panel A: AMR Risk Index by Farm Location (color by risk category) ----
ax_a = fig.add_subplot(gs[0, 0])

for cat, color in risk_colors.items():
    subset = df[df['Risk_Category'] == cat]
    ax_a.scatter(subset['Longitude'], subset['Latitude'],
                 c=color, s=45, edgecolors='white', linewidth=0.8,
                 label=cat, alpha=0.85, zorder=3)

# Annotate district names with background
for _, row in district_coords.iterrows():
    ax_a.annotate(row['District'], xy=(row['Longitude'], row['Latitude']),
                  xytext=(3, 3), textcoords='offset points',
                  fontsize=8, ha='left', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

ax_a.set_xlim(88, 93)
ax_a.set_ylim(21, 27)
ax_a.set_xlabel('Longitude', fontweight='bold')
ax_a.set_ylabel('Latitude', fontweight='bold')
ax_a.set_title('A. AMR Risk Index by Farm Location', fontweight='bold')
ax_a.legend(title='Risk Level', loc='upper right', frameon=True, fancybox=True)
ax_a.grid(True, linestyle='--', alpha=0.5)

# ---- Panel B: Knowledge Score by Farm Location (continuous colormap) ----
ax_b = fig.add_subplot(gs[0, 1])
sc = ax_b.scatter(df['Longitude'], df['Latitude'],
                  c=df['Knowledge_Score'], s=45, edgecolors='white',
                  linewidth=0.8, cmap='viridis', alpha=0.85, vmin=0, vmax=6, zorder=3)

for _, row in district_coords.iterrows():
    ax_b.annotate(row['District'], xy=(row['Longitude'], row['Latitude']),
                  xytext=(3, 3), textcoords='offset points',
                  fontsize=8, ha='left', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

ax_b.set_xlim(88, 93)
ax_b.set_ylim(21, 27)
ax_b.set_xlabel('Longitude', fontweight='bold')
ax_b.set_ylabel('Latitude', fontweight='bold')
ax_b.set_title('B. Knowledge Score by Farm Location', fontweight='bold')
cbar = plt.colorbar(sc, ax=ax_b, shrink=0.8)
cbar.set_label('Knowledge Score (0–6)', fontsize=10)
cbar.ax.tick_params(labelsize=9)
ax_b.grid(True, linestyle='--', alpha=0.5)

# ---- Panel C: Mean AMR Risk Index by District (bar chart with error bars) ----
ax_c = fig.add_subplot(gs[1, :])
districts = district_stats['District']
means = district_stats['mean']
stds = district_stats['std']

bars = ax_c.bar(districts, means, yerr=stds, capsize=5,
                color='steelblue', edgecolor='black', alpha=0.7, zorder=2)

# Add value labels on bars
for bar, val in zip(bars, means):
    ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Overall mean line
ax_c.axhline(overall_mean, color='red', linestyle='--', linewidth=2,
             label=f'Overall Mean = {overall_mean:.2f}', zorder=3)

ax_c.set_xlabel('District', fontweight='bold')
ax_c.set_ylabel('Mean AMR Risk Index', fontweight='bold')
ax_c.set_title('C. Mean AMR Risk Index by District', fontweight='bold')
ax_c.legend(loc='upper right', frameon=True)
ax_c.grid(axis='y', linestyle='--', alpha=0.5)
plt.setp(ax_c.get_xticklabels(), rotation=45, ha='right')

# =============================================================================
# 6. OVERALL TITLE AND FOOTNOTE
# =============================================================================
fig.suptitle('Figure 7 — Geographic Distribution: AMR Risk Index & Knowledge Score\n'
             'Commercial Poultry Farms, Bangladesh (n=212)',
             fontsize=16, fontweight='bold', y=0.98)

n_plotted = len(df)
fig.text(0.5, 0.02,
         f'Note: GPS from field survey (n=212). {n_plotted} farms with valid coordinates plotted. '
         'Red=high risk (≥5), Yellow=moderate, Green=low. Panel C: dashed line = overall mean.',
         ha='center', fontsize=9, style='italic')

# =============================================================================
# 7. SAVE THE FIGURE
# =============================================================================
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
output_file = 'Figure7_Geographic_Risk_Map_Enhanced.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Figure saved as: {output_file}")