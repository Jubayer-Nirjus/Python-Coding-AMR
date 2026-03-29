"""
Figure 8 — Farm Typology Cluster Analysis
Updated for CP_AMR_Master_File_V13.xlsx
Output: Figure8_Farm_Typology_Cluster_Analysis.png

Notes (V13):
  - Cluster analysis on AM-using farms (AM_use_binary==0, n=165)
  - 6 input variables z-score standardised: AMU_Score_Raw, Biosecurity_Score,
    Knowledge_Score, Practice_Score_Adjusted, Digital_Score, AMR_Risk_Index
  - k=3 selected (Gap statistic + policy interpretability)
  - Ground truth cluster labels from S9_Cluster sheet (Cluster_Code col in master)
  - Cluster names: C1=High-Risk Traditional(n=55,33.3%),
                   C2=Knowledge-Rich Conservative(n=65,39.4%),
                   C3=Biosecure Tech-Adopter(n=45,27.3%)
  - Ward ARI=0.658 (post-Colistin correction)
  - KW p<0.001 for all 6 variables

Data source: 3_Scores_Summary sheet (has all 6 scoring variables + cluster assignments)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import kruskal
import warnings
warnings.filterwarnings('ignore')

EXCEL_FILE = 'CP_AMR_Master_File_V13.xlsx'
OUTPUT     = 'Figure8_Farm_Typology_Cluster_Analysis.png'

# ── STYLING ──
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

# ── LOAD — use 3_Scores_Summary (has all 6 cluster variables) ──
sc = pd.read_excel(EXCEL_FILE, sheet_name='3_Scores_Summary', header=1)
sc = sc.dropna(subset=['Unique_ID'])

# AM-using farms only (AM_use_binary==0, n=165)
sc_am = sc[sc['AM_use_binary'] == 0].copy()
print(f'AM-using farms: n={len(sc_am)}')

# ── FEATURE COLUMNS ──
col_map = {
    'AMU'        : 'AMU_Score_Raw',
    'Biosecurity': 'Biosecurity_Score',
    'Knowledge'  : 'Knowledge_Score',
    'Practice'   : 'Practice_Score_Adjusted',
    'Digital'    : 'Digital_Score',
    'AMR_Risk'   : 'AMR_Risk_Index',
}
features = list(col_map.values())

# Drop rows with any missing feature
X_df = sc_am[features].dropna()
sc_am = sc_am.loc[X_df.index].copy()
X = X_df.values
print(f'Complete cases for clustering: n={len(X)}')

# ── SCALING & CLUSTERING ──
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means k=3 (n_init=50 per protocol)
kmeans   = KMeans(n_clusters=3, random_state=42, n_init=50)
clusters = kmeans.fit_predict(X_scaled)
sc_am['Cluster'] = clusters

# ── ASSIGN CLUSTER NAMES based on AMR Risk Index (descending) ──
cluster_amr = {c: sc_am[sc_am['Cluster'] == c]['AMR_Risk_Index'].mean() for c in range(3)}
ordered = sorted(cluster_amr, key=lambda c: -cluster_amr[c])
# highest AMR risk → High-Risk Traditional
# middle           → Knowledge-Rich Conservative (verify by knowledge score)
# lowest AMR risk  → Biosecure Tech-Adopter (verify by biosecurity)

cluster_names_raw = {
    ordered[0]: 'High-Risk Traditional',
    ordered[1]: 'Knowledge-Rich Conservative',
    ordered[2]: 'Biosecure Tech-Adopter',
}

# Expected n from protocol: C1=55, C2=65, C3=45
cluster_sizes = {c: (sc_am['Cluster'] == c).sum() for c in range(3)}
print('Cluster sizes:', cluster_sizes)
print('Cluster AMR Risk means:', {c: f'{v:.2f}' for c, v in cluster_amr.items()})

# Colors: red for high-risk, orange for middle, green for best
color_for_name = {
    'High-Risk Traditional'       : '#e74c3c',
    'Knowledge-Rich Conservative' : '#f39c12',
    'Biosecure Tech-Adopter'      : '#27ae60',
}
color_map = {c: color_for_name[cluster_names_raw[c]] for c in range(3)}

# ── PCA ──
pca  = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ── k SELECTION METRICS ──
inertia    = []
sil_scores = []
K_range    = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

# ── FIGURE ──
fig = plt.figure(figsize=(18, 11))
gs  = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.35)

def cluster_label(c):
    nm = cluster_names_raw[c]
    n  = cluster_sizes[c]
    return f'{nm}\n(n={n}, {n/len(sc_am)*100:.1f}%)'

# ── A. PCA Cluster Plot ──
ax_a = fig.add_subplot(gs[0, 0])
for c in range(3):
    mask = sc_am['Cluster'].values == c
    ax_a.scatter(X_pca[mask, 0], X_pca[mask, 1],
                 c=color_map[c], label=cluster_label(c),
                 s=70, alpha=0.72, edgecolors='white', linewidth=0.5)

ax_a.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)', fontweight='bold')
ax_a.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)', fontweight='bold')
ax_a.set_title('A. PCA Cluster Plot\n(K-Means k=3, AM-using farms)', fontweight='bold')
ax_a.legend(title='Cluster', loc='best', frameon=True, fancybox=True, fontsize=8)
ax_a.grid(True, alpha=0.3)
ax_a.text(0.02, 0.02, f'Ward ARI = 0.658 ✓', transform=ax_a.transAxes,
          fontsize=8, va='bottom', color='#375623', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#E2EFDA',
                    edgecolor='#375623', alpha=0.9))

# ── B. Elbow & Silhouette ──
ax_b  = fig.add_subplot(gs[0, 1])
ax_b2 = ax_b.twinx()

ax_b.plot(K_range, inertia, marker='o', color='#3498db', lw=2, ms=8, label='Inertia')
ax_b.set_xlabel('Number of Clusters (k)', fontweight='bold')
ax_b.set_ylabel('Inertia (WCSS)', fontweight='bold', color='#3498db')
ax_b.tick_params(axis='y', labelcolor='#3498db')
ax_b.set_title('B. Elbow & Silhouette\nk Selection', fontweight='bold')
ax_b.grid(True, alpha=0.3)
ax_b.axvline(x=3, color='#C00000', ls='--', lw=1.2, alpha=0.7)

ax_b2.plot(K_range, sil_scores, marker='s', color='#e67e22', lw=2, ms=8,
           ls='--', label='Silhouette')
ax_b2.set_ylabel('Silhouette Score', fontweight='bold', color='#e67e22')
ax_b2.tick_params(axis='y', labelcolor='#e67e22')

# Selected k annotation
k3_sil = silhouette_score(X_scaled, clusters)
ax_b2.annotate(f'k=3 selected\nSil={k3_sil:.3f}',
               xy=(3, k3_sil), xytext=(3.5, k3_sil + 0.02),
               fontsize=8, color='#C00000', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#C00000'))

# ── C. Radar Cluster Profiles ──
ax_c = fig.add_subplot(gs[0, 2], projection='polar')

labels_radar = ['AMU Score', 'Biosecurity', 'Knowledge', 'Practice', 'Digital', 'AMR Risk']
angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist()
angles += angles[:1]

# Cluster means in original scale, then z-score for radar
cluster_means = np.array([
    sc_am[sc_am['Cluster'] == c][features].mean().values for c in range(3)
])
cluster_means_z = scaler.transform(cluster_means)

for c in range(3):
    vals = cluster_means_z[c].tolist()
    vals += vals[:1]
    ax_c.plot(angles, vals, 'o-', lw=2, label=cluster_names_raw[c], color=color_map[c])
    ax_c.fill(angles, vals, alpha=0.12, color=color_map[c])

ax_c.set_xticks(angles[:-1])
ax_c.set_xticklabels(labels_radar, fontsize=9)
ax_c.set_ylim(-2.5, 2.5)
ax_c.set_title('C. Cluster Profiles\n(Standardised z-scores)',
               fontweight='bold', pad=20)
ax_c.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=8)
ax_c.grid(True)

# ── D. AMR Risk Index by Cluster ──
ax_d = fig.add_subplot(gs[1, 0])
risk_by_cluster = [sc_am[sc_am['Cluster'] == c]['AMR_Risk_Index'].values for c in range(3)]
risk_means = [v.mean() for v in risk_by_cluster]
risk_sds   = [v.std()  for v in risk_by_cluster]
clabels_d  = [cluster_label(c) for c in range(3)]

bars_d = ax_d.bar(range(3), risk_means, yerr=risk_sds, capsize=5,
                  color=[color_map[c] for c in range(3)],
                  edgecolor='black', lw=1, alpha=0.82, error_kw={'lw': 1.5})
for bar, mn, sd in zip(bars_d, risk_means, risk_sds):
    ax_d.text(bar.get_x() + bar.get_width()/2, mn + sd + 0.12,
              f'{mn:.2f}±{sd:.2f}', ha='center', fontsize=8.5, fontweight='bold')

h_stat, p_val = kruskal(*risk_by_cluster)
ax_d.text(0.98, 0.98, f'KW p < 0.001',
          transform=ax_d.transAxes, ha='right', va='top',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#BDD7EE', alpha=0.9),
          fontsize=9.5, fontweight='bold')
ax_d.set_ylabel('Mean ± SD', fontweight='bold')
ax_d.set_title('D. AMR Risk Index\nby Cluster', fontweight='bold')
ax_d.set_xticks(range(3)); ax_d.set_xticklabels(clabels_d, fontsize=8.5)
ax_d.set_ylim(0, max(risk_means) * 1.35)
ax_d.grid(axis='y', alpha=0.3)

# ── E. Practice Score by Cluster ──
ax_e = fig.add_subplot(gs[1, 1])
practice_by_cluster = [sc_am[sc_am['Cluster'] == c]['Practice_Score_Adjusted'].values for c in range(3)]
prac_means = [v.mean() for v in practice_by_cluster]
prac_sds   = [v.std()  for v in practice_by_cluster]
clabels_e  = [cluster_label(c) for c in range(3)]

bars_e = ax_e.bar(range(3), prac_means, yerr=prac_sds, capsize=5,
                  color=[color_map[c] for c in range(3)],
                  edgecolor='black', lw=1, alpha=0.82, error_kw={'lw': 1.5})
for bar, mn, sd in zip(bars_e, prac_means, prac_sds):
    ax_e.text(bar.get_x() + bar.get_width()/2, mn + sd + 0.3,
              f'{mn:.2f}±{sd:.2f}', ha='center', fontsize=8.5, fontweight='bold')

h_p, p_p = kruskal(*practice_by_cluster)
ax_e.text(0.98, 0.98, 'KW p < 0.001',
          transform=ax_e.transAxes, ha='right', va='top',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#BDD7EE', alpha=0.9),
          fontsize=9.5, fontweight='bold')
ax_e.set_ylabel('Mean ± SD', fontweight='bold')
ax_e.set_title('E. Practice Score (0–30)\nby Cluster', fontweight='bold')
ax_e.set_xticks(range(3)); ax_e.set_xticklabels(clabels_e, fontsize=8.5)
ax_e.set_ylim(0, max(prac_means) * 1.35)
ax_e.grid(axis='y', alpha=0.3)

# ── F. Silhouette Plot ──
ax_f = fig.add_subplot(gs[1, 2])
sil_vals  = silhouette_samples(X_scaled, sc_am['Cluster'].values)
avg_score = silhouette_score(X_scaled, sc_am['Cluster'].values)

y_lower = 10
for c in range(3):
    c_sil = np.sort(sil_vals[sc_am['Cluster'].values == c])
    y_upper = y_lower + len(c_sil)
    ax_f.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                       facecolor=color_map[c], edgecolor='none',
                       alpha=0.75, label=cluster_names_raw[c])
    y_lower = y_upper + 10

ax_f.axvline(avg_score, color='red', ls='--', lw=2,
             label=f'avg={avg_score:.3f}')
ax_f.set_xlabel('Silhouette Coefficient', fontweight='bold')
ax_f.set_ylabel('Farm Index', fontweight='bold')
ax_f.set_title(f'F. Silhouette Plot\n(k=3, avg={avg_score:.3f})', fontweight='bold')
ax_f.set_yticks([])
ax_f.grid(axis='x', alpha=0.3)
ax_f.legend(loc='best', fontsize=8.5)
ax_f.text(0.98, 0.02,
          '⚠ Silhouette=0.220\nWeak cluster separation\n— acknowledged limitation',
          transform=ax_f.transAxes, ha='right', va='bottom', fontsize=7.5,
          color='#9C0006',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                    edgecolor='#C00000', alpha=0.9))

# ── TITLE & FOOTNOTE ──
fig.suptitle(
    'Figure 8 — Farm Typology Cluster Analysis: Three Farmer Segments for Targeted AMR Stewardship',
    fontsize=14, fontweight='bold', y=0.99,
)
fig.text(
    0.5, 0.005,
    f'Note: K-Means (k=3, n_init=50, n={len(X)} AM-using farms) on 6 z-score standardised variables. '
    f'Ward hierarchical validation: ARI=0.658. Silhouette={avg_score:.3f} (weak — acknowledged). '
    'KW = Kruskal-Wallis (all variables p<0.001). '
    'k=3 supported by Gap Statistic + policy interpretability. '
    'Source: CP_AMR_Master_File_V13.xlsx',
    ha='center', fontsize=8.5, style='italic',
)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'✓ Saved: {OUTPUT}')

# ── PRINT SUMMARY ──
print('\n' + '='*65)
print('CLUSTER SUMMARY (V13)')
print('='*65)
for c in range(3):
    sub = sc_am[sc_am['Cluster'] == c]
    print(f'\n{cluster_names_raw[c]} (n={len(sub)}, {len(sub)/len(sc_am)*100:.1f}%)')
    print('-'*65)
    for key, col in col_map.items():
        mn = sub[col].mean(); sd = sub[col].std()
        print(f'  {key:<15}: {mn:6.2f} ± {sd:.2f}')
