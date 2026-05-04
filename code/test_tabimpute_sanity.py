"""Sanity check: does TabImpute work on the Merged dataset?

1. Load the Merged score matrix (118 x 114, 31.1% observed).
2. Impute all missing values with TabImpute V2.
3. Hold out 10% of observed entries (ensuring ≥1 per row remains),
   mask them, re-impute, and measure R² on the held-out entries.
"""

import pathlib
import sys
import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir

DATA = data_dir()

SEED = 42

# ── Load data ─────────────────────────────────────────────────────

d = np.load(DATA / "merged.matrix.npz", allow_pickle=True)
B = d["B"].astype(np.float64)
O = d["O"].astype(np.float64)
model_names = d["model_names"]
benchmark_names = d["benchmark_names"]
M, N = B.shape
print(f"Merged: {M} models × {N} benchmarks, {O.sum()/(M*N):.1%} observed")
print(f"Total observed entries: {int(O.sum())}")

# ── Step 1: Try full imputation ───────────────────────────────────

# Convert to NaN format expected by TabImpute
X = np.where(O == 1, B, np.nan)
print(f"\nMatrix has {np.isnan(X).sum()} NaN entries out of {M*N}")

print("\nImporting TabImpute V2...")
from tabimpute.tabimpute_v2 import TabImputeV2

print("Loading TabImputeV2 (downloads weights on first use)...")
imputer = TabImputeV2(device='cpu')

print("Running full imputation...")
X_full = imputer.impute(X)
print(f"Imputation complete. Result shape: {X_full.shape}")
print(f"Any NaN remaining: {np.isnan(X_full).any()}")

# Check that observed entries are preserved
obs_diff = np.abs(X_full[O == 1] - B[O == 1])
print(f"Max diff on observed entries: {obs_diff.max():.6f}")
print(f"Mean diff on observed entries: {obs_diff.mean():.6f}")

# Basic stats on imputed values
imputed_vals = X_full[O == 0]
print(f"\nImputed values: min={imputed_vals.min():.2f}, "
      f"max={imputed_vals.max():.2f}, mean={imputed_vals.mean():.2f}")

# ── Step 2: Hold-out test ─────────────────────────────────────────

print("\n" + "="*60)
print("Hold-out test: mask 10% of observed entries, re-impute")
print("="*60)

rng = np.random.default_rng(SEED)

# Find all observed entries
obs_rows, obs_cols = np.where(O == 1)
n_obs = len(obs_rows)
print(f"Total observed: {n_obs}")

# Count observations per row
row_counts = O.sum(axis=1).astype(int)
print(f"Obs per row: min={row_counts.min()}, max={row_counts.max()}, "
      f"mean={row_counts.mean():.1f}")

# Select 10% to hold out, but ensure each row keeps at least 1
n_holdout_target = n_obs // 10
print(f"Target holdout: {n_holdout_target} entries")

holdout_mask = np.zeros(n_obs, dtype=bool)
perm = rng.permutation(n_obs)
row_remaining = row_counts.copy()

n_held = 0
for idx in perm:
    if n_held >= n_holdout_target:
        break
    r = obs_rows[idx]
    if row_remaining[r] > 1:
        holdout_mask[idx] = True
        row_remaining[r] -= 1
        n_held += 1

print(f"Actually held out: {n_held} entries")
print(f"Min obs remaining per row: {row_remaining.min()}")

# Build masked matrix
O_masked = O.copy()
B_masked = B.copy()
for idx in np.where(holdout_mask)[0]:
    r, c = obs_rows[idx], obs_cols[idx]
    O_masked[r, c] = 0.0
    B_masked[r, c] = 0.0

X_masked = np.where(O_masked == 1, B_masked, np.nan)
print(f"Masked matrix: {np.isnan(X_masked).sum()} NaN entries")

# Impute
print("Running imputation on masked matrix...")
X_imputed = imputer.impute(X_masked)
print("Done.")

# Evaluate on held-out entries
true_vals = []
pred_vals = []
for idx in np.where(holdout_mask)[0]:
    r, c = obs_rows[idx], obs_cols[idx]
    true_vals.append(B[r, c])
    pred_vals.append(X_imputed[r, c])

true_vals = np.array(true_vals)
pred_vals = np.array(pred_vals)

ss_res = np.sum((true_vals - pred_vals) ** 2)
ss_tot = np.sum((true_vals - true_vals.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot
rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
mae = np.mean(np.abs(true_vals - pred_vals))

print(f"\n{'='*60}")
print(f"Hold-out results ({n_held} entries):")
print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  MAE  = {mae:.4f}")
print(f"  True range: [{true_vals.min():.2f}, {true_vals.max():.2f}]")
print(f"  Pred range: [{pred_vals.min():.2f}, {pred_vals.max():.2f}]")
print(f"{'='*60}")

# Per-model correlation for models with ≥3 held-out entries
print("\nPer-model stats (models with ≥3 held-out entries):")
for r in range(M):
    mask_r = holdout_mask & (obs_rows == r)
    if mask_r.sum() >= 3:
        t_r = B[r, obs_cols[mask_r]]
        p_r = X_imputed[r, obs_cols[mask_r]]
        if t_r.std() > 0:
            corr = np.corrcoef(t_r, p_r)[0, 1]
            r2_r = 1.0 - np.sum((t_r - p_r)**2) / np.sum((t_r - t_r.mean())**2)
            print(f"  {model_names[r]}: {mask_r.sum()} entries, "
                  f"corr={corr:.3f}, R²={r2_r:.3f}")
