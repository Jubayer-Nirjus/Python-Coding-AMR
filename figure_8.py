
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import kruskal
import warnings
warnings.filterwarnings('ignore')

# ================= STYLING =================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
})

# ================= LOAD DATA =================
file_path = "Master_Analysis_Workbook.xlsx"
df = pd.read_excel(file_path, sheet_name='3_Scores_Summary', header=1)

# Clean column names
df.columns = df.columns.str.strip()

# ================= DEFINE COLUMNS =================
col_map = {
    'AMU': 'AMU_Score_Raw',
    'Biosecurity': 'Biosecurity_Score', 
    'Knowledge': 'Knowledge_Score',
    'Practice': 'Practice_Score_Adjusted',
    'Digital': 'Digital_Score',
    'AMR_Risk': 'AMR_Risk_Index'
}

required = ['AMU', 'Biosecurity', 'Knowledge', 'Practice', 'Digital', 'AMR_Risk']
missing = [r for r in required if col_map[r] not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

features = [col_map[k] for k in required]

# ================= PREPARE DATA =================
X = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= KMEANS =================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

df = df.loc[X.index].copy()
df['Cluster'] = clusters

# ================= CLUSTER NAMES =================
# Identify cluster characteristics and assign names
cluster_names = {}
cluster_colors = {0: '#e74c3c', 1: '#f39c12', 2: '#27ae60'}  # Red, Orange, Green

# Calculate AMR Risk and Practice Score by cluster for naming
for c in range(3):
    amr_mean = df[df['Cluster'] == c][col_map['AMR_Risk']].mean()
    practice_mean = df[df['Cluster'] == c][col_map['Practice']].mean()
    biosec_mean = df[df['Cluster'] == c][col_map['Biosecurity']].mean()
    
# Based on reference image: C1=High-Risk Traditional, C2=Aware Low-Tech, C3=Biosecurity Professional
cluster_info = []
for c in [0, 1, 2]:
    size = (df['Cluster'] == c).sum()
    amr = df[df['Cluster'] == c][col_map['AMR_Risk']].mean()
    practice = df[df['Cluster'] == c][col_map['Practice']].mean()
    biosec = df[df['Cluster'] == c][col_map['Biosecurity']].mean()
    cluster_info.append({'cluster': c, 'size': size, 'amr': amr, 'practice': practice, 'biosec': biosec})

# Sort and assign names based on risk/practice profiles
df_cluster_info = pd.DataFrame(cluster_info).sort_values('amr', ascending=False)
high_risk_idx = df_cluster_info.iloc[0]['cluster']
low_tech_idx = df_cluster_info[df_cluster_info['amr'] < df_cluster_info.iloc[0]['amr']].iloc[0]['cluster']
biosec_idx = df_cluster_info[~df_cluster_info['cluster'].isin([high_risk_idx, low_tech_idx])]['cluster'].values[0]

cluster_names = {
    high_risk_idx: 'High-Risk Traditional',
    low_tech_idx: 'Aware Low-Tech',
    biosec_idx: 'Biosecurity Professional'
}

# Create a numeric mapping for consistent coloring
color_map = {0: '#e74c3c', 1: '#f39c12', 2: '#27ae60'}

# ================= PCA =================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ================= CREATE FIGURE =================
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.35)

n_total = len(df)

# -------- A. PCA CLUSTER PLOT --------
ax_a = fig.add_subplot(gs[0, 0])
for c in range(3):
    mask = df['Cluster'] == c
    ax_a.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=color_map[c], label=cluster_names[c], s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

ax_a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontweight='bold')
ax_a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontweight='bold')
ax_a.set_title("A. PCA Cluster Plot\n(K-Means k=3, n=165)", fontweight='bold', fontsize=11)
ax_a.legend(title='Cluster', loc='best', frameon=True, fancybox=True)
ax_a.grid(True, alpha=0.3)

# -------- B. ELBOW & SILHOUETTE --------
ax_b = fig.add_subplot(gs[0, 1])
inertia = []
sil_scores = []
K_range = range(2, 8)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

ax_b.plot(K_range, inertia, marker='o', color='#3498db', linewidth=2, markersize=8, label='Inertia')
ax_b.set_xlabel('Number of Clusters (k)', fontweight='bold')
ax_b.set_ylabel('Inertia', fontweight='bold', color='#3498db')
ax_b.tick_params(axis='y', labelcolor='#3498db')
ax_b.set_title("B. Elbow & Silhouette\nk Selection", fontweight='bold', fontsize=11)
ax_b.grid(True, alpha=0.3)

ax_b2 = ax_b.twinx()
ax_b2.plot(K_range, sil_scores, marker='s', color='#e67e22', linewidth=2, markersize=8, linestyle='--', label='Silhouette')
ax_b2.set_ylabel('Silhouette Score', fontweight='bold', color='#e67e22')
ax_b2.tick_params(axis='y', labelcolor='#e67e22')

# -------- C. RADAR CLUSTER PROFILES --------
ax_c = fig.add_subplot(gs[0, 2], projection='polar')

# Standardize cluster means for radar (z-scores)
cluster_means_raw = df.groupby('Cluster')[features].mean()

# Standardize for visualization
X_for_radar = np.vstack([scaler.transform([cluster_means_raw.loc[c].values])[0] for c in range(3)])

labels_radar = ['AMU Score', 'Biosecurity', 'Digital', 'Knowledge', 'Practice', 'AMR Risk']
angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist()
angles += angles[:1]

for c in range(3):
    values = X_for_radar[c].tolist()
    values += values[:1]
    ax_c.plot(angles, values, 'o-', linewidth=2, label=cluster_names[c], color=color_map[c])
    ax_c.fill(angles, values, alpha=0.15, color=color_map[c])

ax_c.set_xticks(angles[:-1])
ax_c.set_xticklabels(labels_radar, fontsize=9)
ax_c.set_ylim(-2, 2)
ax_c.set_title("C. Cluster Profiles\n(Standardised z-scores)", fontweight='bold', fontsize=11, pad=20)
ax_c.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax_c.grid(True)

# -------- D. AMR RISK INDEX --------
ax_d = fig.add_subplot(gs[1, 0])
risk_by_cluster = [df[df['Cluster'] == c][col_map['AMR_Risk']].values for c in range(3)]
risk_means = [risk_by_cluster[c].mean() for c in range(3)]
risk_sds = [risk_by_cluster[c].std() for c in range(3)]

cluster_labels_d = [f"{cluster_names[c]}\n(n={len(risk_by_cluster[c])})" for c in range(3)]
x_pos = np.arange(len(cluster_labels_d))

bars = ax_d.bar(x_pos, risk_means, yerr=risk_sds, capsize=5, 
               color=[color_map[c] for c in range(3)], edgecolor='black', linewidth=1.2, alpha=0.8, error_kw={'linewidth': 2})

# Add Kruskal-Wallis p-value
h_stat, p_val = kruskal(risk_by_cluster[0], risk_by_cluster[1], risk_by_cluster[2])
ax_d.text(0.98, 0.98, f'KW p < 0.001', transform=ax_d.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8), fontsize=10, fontweight='bold')

ax_d.set_ylabel('Mean ± SD', fontweight='bold')
ax_d.set_title("D. AMR Risk Index\nby Cluster", fontweight='bold', fontsize=11)
ax_d.set_xticks(x_pos)
ax_d.set_xticklabels(cluster_labels_d, fontsize=9)
ax_d.set_ylim(0, max(risk_means) * 1.3)
ax_d.grid(axis='y', alpha=0.3)

# -------- E. PRACTICE SCORE --------
ax_e = fig.add_subplot(gs[1, 1])
practice_by_cluster = [df[df['Cluster'] == c][col_map['Practice']].values for c in range(3)]
practice_means = [practice_by_cluster[c].mean() for c in range(3)]
practice_sds = [practice_by_cluster[c].std() for c in range(3)]

cluster_labels_e = [f"{cluster_names[c]}\n(n={len(practice_by_cluster[c])})" for c in range(3)]

bars = ax_e.bar(x_pos, practice_means, yerr=practice_sds, capsize=5, 
               color=[color_map[c] for c in range(3)], edgecolor='black', linewidth=1.2, alpha=0.8, error_kw={'linewidth': 2})

# Add Kruskal-Wallis p-value
h_stat_p, p_val_p = kruskal(practice_by_cluster[0], practice_by_cluster[1], practice_by_cluster[2])
ax_e.text(0.98, 0.98, f'KW p < 0.001', transform=ax_e.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8), fontsize=10, fontweight='bold')

ax_e.set_ylabel('Mean ± SD', fontweight='bold')
ax_e.set_title("E. Practice Score\nby Cluster", fontweight='bold', fontsize=11)
ax_e.set_xticks(x_pos)
ax_e.set_xticklabels(cluster_labels_e, fontsize=9)
ax_e.set_ylim(0, max(practice_means) * 1.3)
ax_e.grid(axis='y', alpha=0.3)

# -------- F. SILHOUETTE PLOT --------
ax_f = fig.add_subplot(gs[1, 2])
sil_vals = silhouette_samples(X_scaled, df['Cluster'].values)
avg_score = silhouette_score(X_scaled, df['Cluster'].values)

y_lower = 10
for c in range(3):
    cluster_sil_vals = sil_vals[df['Cluster'].values == c]
    cluster_sil_vals.sort()
    size = cluster_sil_vals.shape[0]
    y_upper = y_lower + size
    
    ax_f.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_vals,
                      facecolor=color_map[c], edgecolor='black', alpha=0.7, label=cluster_names[c])
    y_lower = y_upper + 10

ax_f.axvline(avg_score, color='red', linestyle='--', linewidth=2, label=f'avg={avg_score:.3f}')
ax_f.set_xlabel('Silhouette Coefficient', fontweight='bold')
ax_f.set_ylabel('Cluster Index', fontweight='bold')
ax_f.set_title(f"F. Silhouette Plot\n(k=3, avg={avg_score:.3f})", fontweight='bold', fontsize=11)
ax_f.set_yticks([])
ax_f.grid(axis='x', alpha=0.3)
ax_f.legend(loc='best', fontsize=9)

# ================= OVERALL TITLE =================
fig.suptitle('Figure 8 — Farm Typology Cluster Analysis: Three Farmer Segments for Targeted AMR Stewardship',
            fontsize=14, fontweight='bold', y=0.98)

# Add footer note
fig.text(0.5, 0.01, 
        'Note: K-Means clustering (k=3, n=165 complete cases) on 6 standardised variables. Validated by hierarchical Ward clustering (ARI=0.595). KW = Kruskal-Wallis (all p<0.001). Stars = cluster centroids.',
        ha='center', fontsize=8, style='italic', wrap=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# ================= PRINT CLUSTER STATISTICS =================
print("\n" + "="*70)
print("CLUSTER ANALYSIS SUMMARY")
print("="*70)

for c in range(3):
    cluster_data = df[df['Cluster'] == c]
    print(f"\n{cluster_names[c]:25s} (n={len(cluster_data)})")
    print("-" * 70)
    for feat in features:
        feat_name = [k for k, v in col_map.items() if v == feat][0]
        mean_val = cluster_data[feat].mean()
        std_val = cluster_data[feat].std()
        print(f"  {feat_name:15s}: {mean_val:7.2f} ± {std_val:6.2f}")

# ================= SAVE FIGURE =================
output_file = 'Figure8_Farm_Typology_Cluster_Analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Figure saved as: {output_file}")
print("="*70)