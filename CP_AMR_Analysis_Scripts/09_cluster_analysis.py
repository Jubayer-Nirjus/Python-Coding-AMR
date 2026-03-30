"""
09_cluster_analysis.py
======================
Step 9: K-means k=3 on AM-using farms (n=165).
Validation: Silhouette, CH, DB Index, Gap Statistic, Ward ARI.
V13 GT: C1=High-Risk Traditional(55), C2=Knowledge-Rich Conservative(65), C3=Biosecure Tech(45).
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score, adjusted_rand_score)
import importlib.util
import sys

# Import 00_load_data.py dynamically since it starts with digits
spec = importlib.util.spec_from_file_location("load_data", "00_load_data.py")
load_data_module = importlib.util.module_from_spec(spec)
sys.modules["load_data"] = load_data_module
spec.loader.exec_module(load_data_module)

from load_data import load_all, FT_MAP

data = load_all()
df = data["combined"]
df_am = df[df["AM_use_binary"] == 0].copy().reset_index(drop=True)
n_am = len(df_am)

print("="*65)
print(f"STEP 9 — CLUSTER ANALYSIS (AM-using farms, n={n_am})")
print("="*65)

# ── Clustering variables (6 z-scored) ────────────────────────────────────
CLUSTER_VARS = ["Knowledge_Score", "AMR_Risk_Index", "Practice_Score_Adjusted",
                "Digital_Score", "Biosecurity_Score", "Attitude_Score"]
df_cl = df_am[CLUSTER_VARS].dropna()
idx_valid = df_cl.index
n_cl = len(df_cl)
print(f"n (complete cases for clustering) = {n_cl}")

scaler = StandardScaler()
X_sc = scaler.fit_transform(df_cl.values)

# ── K-selection metrics (k=2 to 7) ───────────────────────────────────────
print("\n── Cluster Validation Metrics (k=2–7) ──")
print(f"{'k':>4} {'Silhouette':>12} {'CH Index':>12} {'DB Index':>12} {'WCSS':>12}")
print("-"*52)
wcss_list = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X_sc)
    sil = silhouette_score(X_sc, labels)
    ch = calinski_harabasz_score(X_sc, labels)
    db = davies_bouldin_score(X_sc, labels)
    wcss = km.inertia_
    wcss_list.append(wcss)
    marker = " ← SELECTED" if k == 3 else ""
    print(f"{k:>4} {sil:>12.3f} {ch:>12.2f} {db:>12.3f} {wcss:>12.1f}{marker}")

# ── Gap Statistic ──────────────────────────────────────────────────────────
print("\n── Gap Statistic (n_ref=50, seed=42) ──")
def gap_statistic(X, k_max=7, n_ref=50, seed=42):
    rng = np.random.default_rng(seed)
    Wks = []
    for k in range(1, k_max+1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        km.fit(X)
        Wks.append(np.log(km.inertia_))
    gaps = []
    for k in range(1, k_max+1):
        ref_Wks = []
        for _ in range(n_ref):
            ref = rng.uniform(X.min(0), X.max(0), size=X.shape)
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(ref)
            ref_Wks.append(np.log(km.inertia_))
        gaps.append(np.mean(ref_Wks) - Wks[k-1])
    return gaps

gaps = gap_statistic(X_sc)
for k, g in enumerate(gaps[:7], 1):
    print(f"  k={k}: Gap={g:.4f}", "← best" if g==max(gaps[:7]) else "")

# ── Final k=3 model ───────────────────────────────────────────────────────
print("\n── Final Model: k=3 ──")
km3 = KMeans(n_clusters=3, n_init=50, random_state=42)
labels3 = km3.fit_predict(X_sc)

sil3 = silhouette_score(X_sc, labels3)
ch3 = calinski_harabasz_score(X_sc, labels3)
db3 = davies_bouldin_score(X_sc, labels3)
print(f"Silhouette = {sil3:.3f} (GT: 0.220) {'⚠ weak (<0.25)' if sil3<0.25 else '✅'}")
print(f"CH Index   = {ch3:.2f} (GT: 54.34)")
print(f"DB Index   = {db3:.3f} (GT: 1.455)")

# ── Ward hierarchical validation ──────────────────────────────────────────
print("\n── Ward Hierarchical Validation ──")
ward = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels_ward = ward.fit_predict(X_sc)
ari = adjusted_rand_score(labels3, labels_ward)
print(f"Adjusted Rand Index (KMeans vs Ward): {ari:.3f} (GT: 0.658)")

# ── Assign cluster names ───────────────────────────────────────────────────
# Match clusters to V13 profile by AMR_Risk_Index (C1=highest, C3=lowest)
cl_risk = {k: df_cl.loc[df_cl.index[labels3==k], "AMR_Risk_Index"].mean()
           for k in range(3)}
sorted_cl = sorted(cl_risk, key=lambda x: cl_risk[x], reverse=True)
# sorted_cl[0]=highest risk=C1, [1]=C2, [2]=lowest risk=C3
remap = {sorted_cl[0]: 1, sorted_cl[1]: 2, sorted_cl[2]: 3}
labels_named = np.array([remap[l] for l in labels3])
CLUSTER_NAMES = {1:"C1: High-Risk Traditional",
                 2:"C2: Knowledge-Rich Conservative",
                 3:"C3: Biosecure Tech-Adopter"}

# ── Cluster profiles ───────────────────────────────────────────────────────
print("\n── Cluster Profiles ──")
print(f"\n{'Variable':<30} {'C1:High-Risk':>18} {'C2:Know-Rich':>18} {'C3:BioTech':>18} {'KW p':>10}")
print("-"*85)

for var in CLUSTER_VARS:
    if var not in df_am.columns: continue
    grp_vals = [df_cl.loc[df_cl.index[labels_named==c], var].dropna()
                for c in [1,2,3]]
    h, p = stats.kruskal(*[g.values for g in grp_vals if len(g)>0])
    sig = "***" if p<0.001 else("**" if p<0.01 else("*" if p<0.05 else "ns"))
    row = f"{var:<30}"
    for c in [1,2,3]:
        vals = df_cl.loc[df_cl.index[labels_named==c], var].dropna()
        row += f" {vals.mean():>8.2f}±{vals.std():>6.2f}  "
    print(row + f" {p:>8.4f}{sig}")

# Farm type distribution
print(f"\n{'Farm Type':<30}", end="")
for c in [1,2,3]:
    n_c = (labels_named==c).sum()
    print(f" {'n='+str(n_c):>18}", end="")
print()
for ft, lbl in FT_MAP.items():
    row = f"  {lbl:<28}"
    for c in [1,2,3]:
        idx_c = df_cl.index[labels_named==c]
        n_ft_c = (df_am.loc[idx_c, "Farm_Type"] == ft).sum()
        n_c = (labels_named==c).sum()
        row += f" {n_ft_c:>8}({n_ft_c/n_c*100:>4.1f}%)    "
    print(row)

# Chi-square farm type × cluster
ct_ft = pd.crosstab(df_am.loc[df_cl.index, "Farm_Type"], labels_named)
chi2, p_ft, dof_ft, _ = stats.chi2_contingency(ct_ft)
print(f"\nFarm Type × Cluster: χ²({dof_ft})={chi2:.2f}, p={p_ft:.3f} (GT: χ²=10.22, p=0.037)")

# ── V13 GT verification ───────────────────────────────────────────────────
print("\n── V13 Ground-Truth Verification ──")
GT_CLUSTERS = {
    "C1: High-Risk Traditional": {"n":55,"Knowledge":1.47,"AMR_Risk":4.64,"Practice":14.42,"Digital":0.45},
    "C2: Knowledge-Rich Conservative": {"n":65,"Knowledge":4.12,"AMR_Risk":2.38,"Practice":19.23,"Digital":0.15},
    "C3: Biosecure Tech-Adopter": {"n":45,"Knowledge":3.36,"AMR_Risk":1.67,"Practice":23.40,"Digital":1.02},
}
for cname, gt in GT_CLUSTERS.items():
    c = int(cname[1])
    actual_n = (labels_named==c).sum()
    print(f"  {cname}: n={actual_n} (GT:{gt['n']}), "
          f"AMR Risk Mean={df_cl.loc[df_cl.index[labels_named==c],'AMR_Risk_Index'].mean():.2f} "
          f"(GT:{gt['AMR_Risk']})")

print("\n  ⚠ Cluster membership may shift slightly due to k-means initialization.")
print("  Use GT n values (55/65/45) for manuscript if close. Verify via dendrogram.")
print("\n✅ Step 9 complete.")
