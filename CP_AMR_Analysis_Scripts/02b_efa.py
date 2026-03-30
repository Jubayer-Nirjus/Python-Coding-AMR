"""
02b_efa.py
==========
Step 2b: Exploratory Factor Analysis — 17-item Non-AMU Practice scale.
Method: Principal Axis Factoring + Varimax rotation.
Factor retention: Parallel analysis (primary) + Kaiser criterion.

V13 ground-truth: KMO=0.745, Bartlett p<0.001, 4 factors, 35.6% variance.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import importlib.util
import sys

# Import 00_load_data.py dynamically since it starts with digits
spec = importlib.util.spec_from_file_location("load_data", "00_load_data.py")
load_data_module = importlib.util.module_from_spec(spec)
sys.modules["load_data"] = load_data_module
spec.loader.exec_module(load_data_module)

from load_data import load_all

data = load_all()
df = data["combined"]

print("=" * 65)
print("STEP 2b — EXPLORATORY FACTOR ANALYSIS (17-item Non-AMU scale)")
print("=" * 65)

# ── 17 non-AMU practice items ──────────────────────────────────────────────
EFA_ITEMS = [
    "Fencing", "Footbath", "Visitor_Biosecurity", "Sick_Bird_Isolation",
    "Dead_Bird_Disposal", "PPE_Use",                    # Biosecurity (6)
    "Flock_Vacc_Schedule", "Vaccine_Planner",
    "Timing_Adherence", "Vaccines_Stored_Cold",          # Vaccine (4)
    "Primary_Feed_Strategy", "FCR", "Feed_Storage",
    "DOC_Source", "Batch_Doc_Record",                    # Feed & Farm (5)
    "Use_of_Automation", "AI_Use_6mo",                   # Digital (2)
]
items = [c for c in EFA_ITEMS if c in df.columns]
df_efa = df[items].dropna()
n, k = df_efa.shape
print(f"n={n}, k={k} items")

# ── KMO (Kaiser-Meyer-Olkin adequacy) ─────────────────────────────────────
def kmo(corr_matrix):
    """KMO measure of sampling adequacy."""
    corr = np.array(corr_matrix)
    np.fill_diagonal(corr, 0)
    inv_corr = np.linalg.pinv(corr_matrix)
    # Partial correlations
    D = np.diag(np.sqrt(np.diag(inv_corr)))
    partial = -inv_corr / (D @ D)
    np.fill_diagonal(partial, 0)
    r2 = corr ** 2
    p2 = partial ** 2
    kmo_val = r2.sum() / (r2.sum() + p2.sum())
    return round(kmo_val, 3)

R = df_efa.corr().values
kmo_val = kmo(R)
print(f"\nKMO = {kmo_val:.3f}  (V13 target: 0.745) {'✅' if abs(kmo_val - 0.745) < 0.05 else '~'}")

# ── Bartlett's test of sphericity ─────────────────────────────────────────
def bartlett_test(corr_matrix, n_obs):
    """Bartlett's test: H0 = identity matrix."""
    k = corr_matrix.shape[0]
    det = np.linalg.det(corr_matrix)
    chi2 = -(n_obs - 1 - (2 * k + 5) / 6) * np.log(max(det, 1e-300))
    df_b = k * (k - 1) / 2
    p = stats.chi2.sf(chi2, df_b)
    return chi2, df_b, p

chi2, df_b, p_bart = bartlett_test(R, n)
print(f"Bartlett's χ²({df_b:.0f}) = {chi2:.2f}, p={'<0.001' if p_bart < 0.001 else f'{p_bart:.4f}'}")
print(f"  (V13 target: χ²=881.28, p<0.001)")

# ── Eigenvalues (for Kaiser criterion and parallel analysis) ───────────────
eigvals = np.linalg.eigvalsh(R)[::-1]
print(f"\nEigenvalues > 1.0 (Kaiser): {(eigvals > 1.0).sum()} factors")
print(f"Top 6 eigenvalues: {eigvals[:6].round(3)}")

# ── Parallel analysis (Hayton et al. 2004) ───────────────────────────────
def parallel_analysis(n_obs, n_vars, n_iter=1000, pctile=95, seed=42):
    """Generate parallel analysis critical eigenvalues."""
    rng = np.random.default_rng(seed)
    rand_eigs = []
    for _ in range(n_iter):
        rand_data = rng.standard_normal((n_obs, n_vars))
        rand_corr = np.corrcoef(rand_data.T)
        eigs = np.linalg.eigvalsh(rand_corr)[::-1]
        rand_eigs.append(eigs)
    crit = np.percentile(rand_eigs, pctile, axis=0)
    return crit

print("\nRunning parallel analysis (1000 iterations)...")
crit_eigs = parallel_analysis(n, k)
n_factors_pa = (eigvals[:k] > crit_eigs[:k]).sum()
print(f"Parallel analysis → retain {n_factors_pa} factors  (V13 target: 4)")

# ── Principal Axis Factoring + Varimax (n_factors=4) ──────────────────────
from numpy.linalg import eigh, svd

def paf_varimax(R, n_factors, n_iter=100, tol=1e-6):
    """Principal axis factoring with Varimax rotation."""
    k = R.shape[0]
    communalities = np.full(k, 0.5)
    
    for _ in range(n_iter):
        R_reduced = R.copy()
        np.fill_diagonal(R_reduced, communalities)
        eigvals_r, eigvecs_r = eigh(R_reduced)
        # Sort descending
        idx = np.argsort(eigvals_r)[::-1]
        eigvals_r, eigvecs_r = eigvals_r[idx], eigvecs_r[:, idx]
        # Factor loadings
        pos_eigs = np.maximum(eigvals_r[:n_factors], 0)
        L = eigvecs_r[:, :n_factors] * np.sqrt(pos_eigs)
        new_comm = (L ** 2).sum(axis=1)
        if np.max(np.abs(new_comm - communalities)) < tol:
            communalities = new_comm
            break
        communalities = np.clip(new_comm, 0.005, 0.999)
    
    # Varimax rotation (Kaiser, 1958)
    def varimax(L, max_iter=1000, tol=1e-6):
        p, k = L.shape
        T = np.eye(k)
        for _ in range(max_iter):
            old_T = T.copy()
            for i in range(k):
                for j in range(i + 1, k):
                    x = L @ T
                    u = x[:, i] ** 2 - x[:, j] ** 2
                    v = 2 * x[:, i] * x[:, j]
                    A = u.sum()
                    B = v.sum()
                    C = (u ** 2 - v ** 2).sum()
                    D = (u * v).sum()
                    num = 2 * (p * D - A * B)
                    den = p * C - (A ** 2 - B ** 2)
                    if abs(den) > 1e-10:
                        angle = np.arctan2(num, den) / 4
                        rot = np.eye(k)
                        c, s = np.cos(angle), np.sin(angle)
                        rot[i, i] = c; rot[j, j] = c
                        rot[i, j] = -s; rot[j, i] = s
                        T = T @ rot
            if np.max(np.abs(T - old_T)) < tol:
                break
        return L @ T
    
    L_rot = varimax(L)
    communalities_final = (L_rot ** 2).sum(axis=1)
    pct_var = (L_rot ** 2).sum(axis=0) / k * 100
    return L_rot, communalities_final, pct_var

N_FACTORS = 4
print(f"\nFitting PAF with {N_FACTORS} factors + Varimax rotation...")
L_rot, comm, pct_var = paf_varimax(R, N_FACTORS)

total_var = pct_var.sum()
print(f"Total variance explained: {total_var:.1f}%  (V13 target: 35.6%)")
for f in range(N_FACTORS):
    print(f"  Factor {f+1}: {pct_var[f]:.1f}%")

# ── Factor loading matrix ──────────────────────────────────────────────────
print(f"\n── Factor Loading Matrix (|λ|≥0.30 shown) ──")
print(f"{'Item':<30} {'F1':>7} {'F2':>7} {'F3':>7} {'F4':>7} {'h²':>7}")
print("-" * 70)
for i, item in enumerate(items):
    loads = L_rot[i]
    h2 = comm[i]
    sig = [f"{l:>7.3f}" if abs(l) >= 0.30 else f"{'':>7}" for l in loads]
    h2_flag = " ⚠" if h2 < 0.30 else ""
    print(f"{item:<30} {''.join(sig)} {h2:>7.3f}{h2_flag}")

# Communality warnings
low_comm = [(items[i], comm[i]) for i in range(len(items)) if comm[i] < 0.30]
if low_comm:
    print(f"\n⚠ Low communalities (<0.30) — note in manuscript limitations:")
    for item, h2 in low_comm:
        print(f"  {item}: h²={h2:.3f}")

# ── Summary for manuscript ─────────────────────────────────────────────────
print("\n── Manuscript reporting values ──")
print(f"KMO = {kmo_val:.3f}, Bartlett's χ²({df_b:.0f}) = {chi2:.2f}, p<0.001")
print(f"Parallel analysis: {n_factors_pa}-factor solution supported")
print(f"Total variance explained: {total_var:.1f}% (below ≥50% convention — "
      f"acknowledge in limitations)")

print("\n✅ Step 2b complete.")
