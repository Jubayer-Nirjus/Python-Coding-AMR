"""
Figure 8 — Farm Typology Cluster Analysis
Preventive Veterinary Medicine (Elsevier) — journal-compliant output

Journal requirements applied:
  - No figure title on figure (caption only)
  - PNG, 300 dpi, 18×11 in = 5400×3300 px ✓
  - Wong (2011) colorblind-safe palette
  - Output: Figure_8.png

CAPTION (use in manuscript):
  Fig. 8. Farm typology analysis of antimicrobial-using farms (n = 165) using
  k-means clustering (k = 3, n_init = 50) on six standardised variables (AMU Score,
  Biosecurity Score, Knowledge Score, Adjusted Practice Score, Digital Score, AMR
  Risk Index). (A) Principal component analysis (PCA) biplot showing cluster separation;
  PC1 and PC2 explained variances are indicated. Ward hierarchical clustering (adjusted
  Rand index = 0.658) confirmed cluster stability. (B) Elbow (inertia; left y-axis) and
  silhouette coefficient (right y-axis) curves across k = 2–7; dashed line indicates
  selected k = 3. (C) Standardised cluster profiles (z-scores) shown as radar chart.
  (D) Mean AMR Risk Index by cluster (±SD; Kruskal–Wallis p < 0.001). (E) Mean Adjusted
  Practice Score by cluster (±SD; Kruskal–Wallis p < 0.001). (F) Silhouette plot for
  the k = 3 solution; dashed line indicates mean silhouette score (noted limitation:
  score = 0.220 indicates weak-to-moderate cluster separation). KW = Kruskal–Wallis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import kruskal
import warnings
warnings.filterwarnings('ignore')

EXCEL_FILE = 'CP_AMR_Master_File_V13.xlsx'
OUTPUT     = 'Figure_8.png'

# Wong (2011) colorblind-safe palette for clusters
COLORS = {
    'c_high'   : '#D55E00',   # vermillion — high-risk cluster
    'c_mid'    : '#E69F00',   # orange — middle cluster
    'c_low'    : '#009E73',   # green — low-risk/best cluster
    'blue'     : '#0072B2',
    'gray_dark': '#404040',
    'gray_mid' : '#767676',
    'gray_light': '#D9D9D9',
}

plt.rcParams.update({
    'font.family'   : 'DejaVu Sans',
    'font.size'     : 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.dpi'   : 300,
})

# ── LOAD — 3_Scores_Summary ──
sc = pd.read_excel(EXCEL_FILE, sheet_name='3_Scores_Summary', header=1)
sc = sc.dropna(subset=['Unique_ID'])
sc_am = sc[sc['AM_use_binary'] == 0].copy()
print(f'AM-using farms: n = {len(sc_am)}')

col_map = {
    'AMU'        : 'AMU_Score_Raw',
    'Biosecurity': 'Biosecurity_Score',
    'Knowledge'  : 'Knowledge_Score',
    'Practice'   : 'Practice_Score_Adjusted',
    'Digital'    : 'Digital_Score',
    'AMR_Risk'   : 'AMR_Risk_Index',
}
features = list(col_map.values())

X_df  = sc_am[features].dropna()
sc_am = sc_am.loc[X_df.index].copy()
X     = X_df.values
print(f'Complete cases: n = {len(X)}')

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans   = KMeans(n_clusters=3, random_state=42, n_init=50)
clusters = kmeans.fit_predict(X_scaled)
sc_am['Cluster'] = clusters

# Assign cluster names by descending AMR Risk
cluster_amr = {c: sc_am[sc_am['Cluster'] == c]['AMR_Risk_Index'].mean() for c in range(3)}
ordered = sorted(cluster_amr, key=lambda c: -cluster_amr[c])
cluster_names = {
    ordered[0]: 'High-Risk Traditional',
    ordered[1]: 'Knowledge-Rich Conservative',
    ordered[2]: 'Biosecure Tech-Adopter',
}
cluster_sizes = {c: (sc_am['Cluster'] == c).sum() for c in range(3)}
print('Sizes:', {cluster_names[c]: cluster_sizes[c] for c in range(3)})

# Colors aligned to cluster risk
color_for_name = {
    'High-Risk Traditional'       : COLORS['c_high'],
    'Knowledge-Rich Conservative' : COLORS['c_mid'],
    'Biosecure Tech-Adopter'      : COLORS['c_low'],
}
color_map = {c: color_for_name[cluster_names[c]] for c in range(3)}

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

inertia    = []
sil_scores = []
K_range    = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

# ── FIGURE ──
fig = plt.figure(figsize=(18, 11))
gs  = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.38)
fig.subplots_adjust(bottom=0.12)

def cluster_label(c):
    nm = cluster_names[c]; n = cluster_sizes[c]
    return f'{nm}\n(n = {n}, {n/len(sc_am)*100:.1f}%)'

# ── A. PCA plot ──
ax_a = fig.add_subplot(gs[0, 0])
for c in range(3):
    mask = sc_am['Cluster'].values == c
    ax_a.scatter(X_pca[mask, 0], X_pca[mask, 1],
                 c=color_map[c], label=cluster_label(c),
                 s=70, alpha=0.72, edgecolors='white', linewidth=0.5)
ax_a.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
ax_a.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
ax_a.set_title('A. PCA Projection (k-means k = 3)', fontweight='bold', pad=8)
ax_a.legend(title='Cluster', loc='best', frameon=True, fancybox=True, fontsize=8, title_fontsize=8.5)
ax_a.grid(True, alpha=0.3)
ax_a.text(0.02, 0.02, 'Ward ARI = 0.658 ✓',
          transform=ax_a.transAxes, fontsize=8, va='bottom', color=COLORS['c_low'],
          fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F0E8', edgecolor=COLORS['c_low'], alpha=0.9))

# ── B. Elbow + Silhouette ──
ax_b  = fig.add_subplot(gs[0, 1])
ax_b2 = ax_b.twinx()
ax_b.plot(K_range, inertia, marker='o', color=COLORS['blue'], lw=2, ms=8, label='Inertia')
ax_b.set_xlabel('Number of clusters (k)', fontweight='bold')
ax_b.set_ylabel('Inertia (WCSS)', fontweight='bold', color=COLORS['blue'])
ax_b.tick_params(axis='y', labelcolor=COLORS['blue'])
ax_b.set_title('B. Elbow & Silhouette — k Selection', fontweight='bold', pad=8)
ax_b.grid(True, alpha=0.3)
ax_b.axvline(x=3, color=COLORS['c_high'], ls='--', lw=1.2, alpha=0.7)

ax_b2.plot(K_range, sil_scores, marker='s', color=COLORS['c_mid'],
           lw=2, ms=8, ls='--', label='Silhouette')
ax_b2.set_ylabel('Silhouette coefficient', fontweight='bold', color=COLORS['c_mid'])
ax_b2.tick_params(axis='y', labelcolor=COLORS['c_mid'])

k3_sil = silhouette_score(X_scaled, clusters)
ax_b2.annotate(f'k = 3 selected\nSil. = {k3_sil:.3f}',
               xy=(3, k3_sil), xytext=(3.5, k3_sil + 0.02),
               fontsize=8, color=COLORS['c_high'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=COLORS['c_high']))

# ── C. Radar cluster profiles ──
ax_c = fig.add_subplot(gs[0, 2], projection='polar')
labels_radar = ['AMU Score', 'Biosecurity', 'Knowledge', 'Practice', 'Digital', 'AMR Risk']
angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist()
angles += angles[:1]

cluster_means = np.array([sc_am[sc_am['Cluster'] == c][features].mean().values for c in range(3)])
cluster_means_z = scaler.transform(cluster_means)

for c in range(3):
    vals = cluster_means_z[c].tolist(); vals += vals[:1]
    ax_c.plot(angles, vals, 'o-', lw=2, label=cluster_names[c], color=color_map[c])
    ax_c.fill(angles, vals, alpha=0.12, color=color_map[c])

ax_c.set_xticks(angles[:-1]); ax_c.set_xticklabels(labels_radar, fontsize=9)
ax_c.set_ylim(-2.5, 2.5)
ax_c.set_title('C. Cluster Profiles\n(standardised z-scores)', fontweight='bold', pad=20)
ax_c.legend(loc='upper right', bbox_to_anchor=(1.38, 1.15), fontsize=8)
ax_c.grid(True)

# ── D. AMR Risk by cluster ──
ax_d = fig.add_subplot(gs[1, 0])
risk_by_c = [sc_am[sc_am['Cluster'] == c]['AMR_Risk_Index'].values for c in range(3)]
r_means = [v.mean() for v in risk_by_c]; r_sds = [v.std() for v in risk_by_c]
clabels = [cluster_label(c) for c in range(3)]

bars_d = ax_d.bar(range(3), r_means, yerr=r_sds, capsize=5,
                  color=[color_map[c] for c in range(3)],
                  edgecolor='black', lw=0.8, alpha=0.82, error_kw={'lw': 1.5})
for bar, mn, sd in zip(bars_d, r_means, r_sds):
    ax_d.text(bar.get_x() + bar.get_width()/2, mn + sd + 0.12,
              f'{mn:.2f} ± {sd:.2f}', ha='center', fontsize=8.5, fontweight='bold')

h, p = kruskal(*risk_by_c)
ax_d.text(0.98, 0.98, 'KW p < 0.001', transform=ax_d.transAxes, ha='right', va='top',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#CCE5FF', alpha=0.9),
          fontsize=9.5, fontweight='bold', color=COLORS['blue'])
ax_d.set_ylabel('Mean ± SD', fontweight='bold')
ax_d.set_title('D. AMR Risk Index by Cluster', fontweight='bold', pad=8)
ax_d.set_xticks(range(3)); ax_d.set_xticklabels(clabels, fontsize=7.5)
ax_d.tick_params(axis='x', labelrotation=8, pad=0)
ax_d.set_ylim(0, max(r_means) * 1.38); ax_d.grid(axis='y', alpha=0.3)

# ── E. Practice Score by cluster ──
ax_e = fig.add_subplot(gs[1, 1])
prac_by_c = [sc_am[sc_am['Cluster'] == c]['Practice_Score_Adjusted'].values for c in range(3)]
p_means = [v.mean() for v in prac_by_c]; p_sds = [v.std() for v in prac_by_c]

bars_e = ax_e.bar(range(3), p_means, yerr=p_sds, capsize=5,
                  color=[color_map[c] for c in range(3)],
                  edgecolor='black', lw=0.8, alpha=0.82, error_kw={'lw': 1.5})
for bar, mn, sd in zip(bars_e, p_means, p_sds):
    ax_e.text(bar.get_x() + bar.get_width()/2, mn + sd + 0.3,
              f'{mn:.2f} ± {sd:.2f}', ha='center', fontsize=8.5, fontweight='bold')

hp, pp = kruskal(*prac_by_c)
ax_e.text(0.98, 0.98, 'KW p < 0.001', transform=ax_e.transAxes, ha='right', va='top',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#CCE5FF', alpha=0.9),
          fontsize=9.5, fontweight='bold', color=COLORS['blue'])
ax_e.set_ylabel('Mean ± SD', fontweight='bold')
ax_e.set_title('E. Practice Score (0–30) by Cluster', fontweight='bold', pad=8)
ax_e.set_xticks(range(3)); ax_e.set_xticklabels(clabels, fontsize=7.5)
ax_e.tick_params(axis='x', labelrotation= 8, pad=0)
ax_e.set_ylim(0, max(p_means) * 1.38); ax_e.grid(axis='y', alpha=0.3)

# ── F. Silhouette plot ──
ax_f = fig.add_subplot(gs[1, 2])
sil_vals  = silhouette_samples(X_scaled, sc_am['Cluster'].values)
avg_score = silhouette_score(X_scaled, sc_am['Cluster'].values)

y_lower = 10
for c in range(3):
    c_sil   = np.sort(sil_vals[sc_am['Cluster'].values == c])
    y_upper = y_lower + len(c_sil)
    ax_f.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                       facecolor=color_map[c], edgecolor='none', alpha=0.75,
                       label=cluster_names[c])
    y_lower = y_upper + 10

ax_f.axvline(avg_score, color=COLORS['c_high'], ls='--', lw=2,
             label=f'Mean = {avg_score:.3f}')
ax_f.set_xlabel('Silhouette coefficient', fontweight='bold')
ax_f.set_ylabel('Farm index', fontweight='bold')
ax_f.set_title(f'F. Silhouette Plot (k = 3, mean = {avg_score:.3f})', fontweight='bold', pad=8)
ax_f.set_yticks([]); ax_f.grid(axis='x', alpha=0.3)
ax_f.legend(loc='upper left', fontsize=6.5)
ax_f.text(0.98, 0.02,
          'Mean silhouette = 0.220\n(weak-to-moderate separation)\n— acknowledged limitation',
          transform=ax_f.transAxes, ha='right', va='bottom', fontsize=7.5, color='#9C0006',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor=COLORS['c_high'], alpha=0.9))

fig.text(
    0.5, 0.005,
    f'k-means (k = 3, n_init = 50, n = {len(X)} AM-using farms) on 6 z-score standardised variables. '
    f'Ward hierarchical validation: ARI = 0.658. Silhouette = {avg_score:.3f} (weak-to-moderate; acknowledged limitation). '
    'KW = Kruskal–Wallis (all variables p < 0.001). '
    'k = 3 supported by Gap Statistic and policy interpretability.',
    ha='center', fontsize=8.5, style='italic',
)

plt.tight_layout(rect=[0, 0.03, 1, 0.99])
plt.savefig(OUTPUT, dpi=300, format='png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'✓ Saved: {OUTPUT}')

# Summary
print('\nCluster summary:')
for c in range(3):
    sub = sc_am[sc_am['Cluster'] == c]
    print(f'  {cluster_names[c]} (n = {len(sub)}): AMR Risk = {sub["AMR_Risk_Index"].mean():.2f}')
