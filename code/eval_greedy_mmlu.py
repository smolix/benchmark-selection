"""Evaluate greedy benchmark selection on MMLU with cross-validation.

For each holdout fraction and fold:
  1. Split models into train/test
  2. Estimate correlation matrix from training models
  3. Run greedy entropy selection for k=1..K_MAX
  4. For each k, impute test models' unselected scores from selected ones
  5. Report: entropy, residual variance, R², RMSE on held-out data
"""

import pathlib
import sys
import numpy as np
from collections import defaultdict

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir

DATA = data_dir()
OUT = figures_dir()
OUT.mkdir(parents=True, exist_ok=True)

from greedy_select import greedy_entropy
from cv_split import cv_folds, cv_train_val

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRACS = [0.1, 0.2, 0.5, 0.9]
SEED = 42

# ── Load MMLU data ─────────────────────────────────────────────────
d = np.load(DATA / "mmlu.matrix.npz", allow_pickle=True)
B = d["B"].astype(np.float64)
O = d["O"].astype(np.float64)
benchmark_names = d["benchmark_names"]
M, N = B.shape
print(f"MMLU: {M} models × {N} tasks, {O.sum()/(M*N)*100:.1f}% observed")


def standardize_and_corr(B_train, O_train):
    """Estimate correlation matrix from training data."""
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))

    # Standardize
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]

    # Pairwise-complete covariance (= correlation since standardized)
    gram = B_std.T @ B_std
    pair_counts = O_train.T @ O_train
    denom = np.maximum(pair_counts - 1.0, 1.0)
    R = gram / denom

    # Project PSD
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = (eigvecs * eigvals[None, :]) @ eigvecs.T

    # Force unit diagonal (correlation)
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(R), 1e-12))
    R = R * np.outer(d_inv, d_inv)

    return R, col_mean, col_std


def impute_and_evaluate(B_test, O_test, selected, R, col_mean, col_std):
    """Impute unselected benchmarks for test models and compute metrics.

    Returns dict with R², RMSE averaged over test models.
    """
    A = np.array(selected)
    all_idx = np.arange(N)
    A_bar = np.setdiff1d(all_idx, A)

    if len(A_bar) == 0:
        return {"r2": 1.0, "rmse": 0.0}

    R_AA = R[np.ix_(A, A)]
    R_bA = R[np.ix_(A_bar, A)]

    # Solve R_AA^{-1} once
    try:
        L = np.linalg.cholesky(R_AA)
        R_AA_inv_R_Ab = np.linalg.solve(L, R_bA.T)
        R_AA_inv_R_Ab = np.linalg.solve(L.T, R_AA_inv_R_Ab)  # (|A|, |A_bar|)
        W = R_bA @ np.linalg.inv(R_AA)  # (|A_bar|, |A|) regression weights
    except np.linalg.LinAlgError:
        R_AA_reg = R_AA + 1e-6 * np.eye(len(A))
        W = R_bA @ np.linalg.inv(R_AA_reg)

    # Standardize test data with training stats
    B_test_std = (B_test - col_mean[None, :]) / col_std[None, :]

    # For each test model, impute unselected benchmarks
    # In standardized space: E[x_Abar | x_A] = mu_Abar + W (x_A - mu_A)
    # Since standardized, mu = 0 (approximately)
    X_A = B_test_std[:, A]       # (M_test, |A|)
    X_Abar_true = B_test_std[:, A_bar]  # (M_test, |A_bar|)
    X_Abar_pred = X_A @ W.T      # (M_test, |A_bar|)

    # Only evaluate on observed entries in test set
    O_Abar = O_test[:, A_bar]
    residuals = O_Abar * (X_Abar_true - X_Abar_pred)

    n_eval = O_Abar.sum()
    if n_eval == 0:
        return {"r2": np.nan, "rmse": np.nan}

    ss_res = (residuals ** 2).sum()
    ss_tot = (O_Abar * X_Abar_true ** 2).sum()

    rmse = np.sqrt(ss_res / n_eval)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {"r2": r2, "rmse": rmse}


# ── Main CV loop ───────────────────────────────────────────────────
folds = cv_folds(M, N_FOLDS, SEED)

# results[p][metric] = (N_FOLDS, K_MAX)
results = {}

for p in HOLDOUT_FRACS:
    metrics = defaultdict(list)

    for fold_idx in range(N_FOLDS):
        train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)

        B_train, O_train = B[train_idx], O[train_idx]
        B_test, O_test = B[val_idx], O[val_idx]
        M_train = len(train_idx)
        M_test = len(val_idx)

        # Estimate correlation from training data
        R, col_mean, col_std = standardize_and_corr(B_train, O_train)

        # Run greedy
        result = greedy_entropy(R, K_MAX, benchmark_names=benchmark_names)

        # Evaluate at each k
        fold_entropy = []
        fold_resid_var = []
        fold_r2 = []
        fold_rmse = []

        for k in range(1, K_MAX + 1):
            selected = result["selected"][:k]

            # Entropy = log-det of selected submatrix
            R_AA = R[np.ix_(selected, selected)]
            sign, logdet = np.linalg.slogdet(R_AA)
            entropy = logdet if sign > 0 else -np.inf
            fold_entropy.append(entropy)

            # Residual variance (theoretical, from covariance)
            fold_resid_var.append(result["residual_variance"][k - 1])

            # Imputation quality on test data
            imp = impute_and_evaluate(B_test, O_test, selected, R, col_mean, col_std)
            fold_r2.append(imp["r2"])
            fold_rmse.append(imp["rmse"])

        metrics["entropy"].append(fold_entropy)
        metrics["resid_var"].append(fold_resid_var)
        metrics["r2"].append(fold_r2)
        metrics["rmse"].append(fold_rmse)

    # Convert to arrays: (N_FOLDS, K_MAX)
    results[p] = {k: np.array(v) for k, v in metrics.items()}

    # Print summary
    mean_r2 = results[p]["r2"].mean(axis=0)
    print(f"\nHoldout {p:.0%}:")
    for k in [1, 3, 5, 10, 15]:
        print(f"  k={k:2d}: R²={mean_r2[k-1]:.4f}  "
              f"resid_var={results[p]['resid_var'].mean(axis=0)[k-1]:.4f}  "
              f"entropy={results[p]['entropy'].mean(axis=0)[k-1]:.3f}")


# ── Plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ks = np.arange(1, K_MAX + 1)

colors = {0.1: "#4477AA", 0.2: "#66CCEE", 0.5: "#EE7733", 0.9: "#CC3311"}

for p in HOLDOUT_FRACS:
    c = colors[p]
    label = f"{p:.0%} holdout"
    mean_r2 = results[p]["r2"].mean(axis=0)
    std_r2 = results[p]["r2"].std(axis=0)
    mean_rv = results[p]["resid_var"].mean(axis=0)
    mean_ent = results[p]["entropy"].mean(axis=0)

    # Panel 1: R²
    axes[0].plot(ks, mean_r2, "o-", color=c, label=label, markersize=4, linewidth=1.5)
    axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                         color=c, alpha=0.15)

    # Panel 2: Residual variance
    axes[1].semilogy(ks, mean_rv, "o-", color=c, label=label,
                     markersize=4, linewidth=1.5)

    # Panel 3: Entropy
    axes[2].plot(ks, mean_ent, "o-", color=c, label=label,
                 markersize=4, linewidth=1.5)

axes[0].set_ylabel("Imputation $R^2$")
axes[0].set_ylim(0.75, 1.0)
axes[1].set_ylabel("Residual variance fraction")
axes[2].set_ylabel("Entropy $\\log\\det(R_{\\mathcal{A}})$")

for ax in axes:
    ax.set_xlabel("Selected benchmarks $k$")
    ax.set_xlim(0.5, K_MAX + 0.5)
    ax.set_xticks([1, 3, 5, 7, 10, 13, 15])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
out_path = OUT / "greedy_cv_mmlu.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"\nSaved to {out_path}")
plt.close()
