"""Evaluate greedy benchmark selection on Merged with cross-validation.

Same protocol as eval_greedy_mmlu.py but uses EM for covariance estimation
since the merged matrix is sparse.
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
from em_cov import em_cov
from cv_split import cv_folds, cv_train_val

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRACS = [0.1, 0.2, 0.5, 0.9]
SEED = 42

# ── Load Merged data ───────────────────────────────────────────────
d = np.load(DATA / "merged.matrix.npz", allow_pickle=True)
B = d["B"].astype(np.float64)
O = d["O"].astype(np.float64)
benchmark_names = d["benchmark_names"]
M, N = B.shape
print(f"Merged: {M} models × {N} tasks, {O.sum()/(M*N)*100:.1f}% observed")


def estimate_corr_em(B_train, O_train):
    """Estimate correlation matrix from training data via EM."""
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)

    # Standardize observed entries
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    # Relative floor: at least 1% of (|mean| + 1) to handle mixed scales
    col_std_floor = 0.01 * (np.abs(col_mean) + 1.0)
    col_std = np.sqrt(np.maximum(col_var, col_std_floor ** 2))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]

    # Run EM with stronger regularization for rank-deficient or sparse cases.
    regularized = M_tr < N_tr or obs_frac < 0.5
    use_shrink = "auto" if M_tr < N_tr else 0.0
    em_eps = 1e-3 if regularized else 1e-6
    em_iter = 1000 if regularized else 500
    result = em_cov(B_std, O_train, max_iter=em_iter, tol=5e-4,
                    shrinkage=use_shrink, eps_psd=em_eps, verbose=False)
    Sigma = result["Sigma"]

    # Convert to correlation
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)

    return R, col_mean, col_std


def impute_and_evaluate(B_test, O_test, selected, R, col_mean, col_std):
    """Impute unselected benchmarks for test models and compute metrics.

    For each test model, use whichever selected benchmarks are observed
    (per-model conditional Gaussian) to predict the unselected ones.
    Ridge-regularized to prevent catastrophic predictions with sparse data.
    """
    A = set(selected)
    M_test = B_test.shape[0]

    # Standardize and clip to prevent outliers from noisy col_std
    B_test_std = O_test * (B_test - col_mean[None, :]) / col_std[None, :]
    B_test_std = np.clip(B_test_std, -10, 10) * O_test

    RIDGE = 0.01  # regularization on R_oo

    ss_res = 0.0
    ss_tot = 0.0
    n_eval = 0

    for i in range(M_test):
        # Which selected benchmarks does this model have?
        obs_A = [j for j in selected if O_test[i, j] == 1]
        # Which unselected benchmarks are observed (for evaluation)?
        eval_idx = [j for j in range(N) if j not in A and O_test[i, j] == 1]

        if len(obs_A) == 0 or len(eval_idx) == 0:
            continue

        obs_A = np.array(obs_A)
        eval_idx = np.array(eval_idx)

        # Ridge-regularized conditional Gaussian
        R_oo = R[np.ix_(obs_A, obs_A)] + RIDGE * np.eye(len(obs_A))
        R_eo = R[np.ix_(eval_idx, obs_A)]
        W = np.linalg.solve(R_oo, R_eo.T).T  # more stable than inv

        x_obs = B_test_std[i, obs_A]
        x_true = B_test_std[i, eval_idx]
        x_pred = W @ x_obs

        ss_res += np.sum((x_true - x_pred) ** 2)
        ss_tot += np.sum(x_true ** 2)
        n_eval += len(eval_idx)

    if n_eval == 0:
        return {"r2": np.nan, "rmse": np.nan}

    rmse = np.sqrt(ss_res / n_eval)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {"r2": r2, "rmse": rmse}


# ── Main CV loop ───────────────────────────────────────────────────
folds = cv_folds(M, N_FOLDS, SEED)

results = {}

for p in HOLDOUT_FRACS:
    metrics = defaultdict(list)

    print(f"\n{'='*60}")
    print(f"Holdout {p:.0%}")
    print(f"{'='*60}")

    for fold_idx in range(N_FOLDS):
        train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)

        B_train, O_train = B[train_idx], O[train_idx]
        B_test, O_test = B[val_idx], O[val_idx]
        M_train = len(train_idx)
        M_test = len(val_idx)

        print(f"  Fold {fold_idx}: {M_train} train / {M_test} test", end="")

        # Estimate correlation from training data via EM
        try:
            R, col_mean, col_std = estimate_corr_em(B_train, O_train)
        except Exception as e:
            print(f" — EM failed: {e}")
            # Fill with NaN
            metrics["entropy"].append([np.nan] * K_MAX)
            metrics["resid_var"].append([np.nan] * K_MAX)
            metrics["r2"].append([np.nan] * K_MAX)
            metrics["rmse"].append([np.nan] * K_MAX)
            continue

        # Run greedy
        result = greedy_entropy(R, K_MAX, benchmark_names=benchmark_names)

        fold_entropy = []
        fold_resid_var = []
        fold_r2 = []
        fold_rmse = []

        for k in range(1, K_MAX + 1):
            selected = result["selected"][:k]

            R_AA = R[np.ix_(selected, selected)]
            sign, logdet = np.linalg.slogdet(R_AA)
            entropy = logdet if sign > 0 else -np.inf
            fold_entropy.append(entropy)

            fold_resid_var.append(result["residual_variance"][k - 1])

            imp = impute_and_evaluate(B_test, O_test, selected, R,
                                      col_mean, col_std)
            fold_r2.append(imp["r2"])
            fold_rmse.append(imp["rmse"])

        metrics["entropy"].append(fold_entropy)
        metrics["resid_var"].append(fold_resid_var)
        metrics["r2"].append(fold_r2)
        metrics["rmse"].append(fold_rmse)

        print(f"  R²@5={fold_r2[4]:.3f}  R²@10={fold_r2[9]:.3f}")

    results[p] = {k: np.array(v) for k, v in metrics.items()}

    # Print summary (ignoring NaN folds)
    valid = ~np.all(np.isnan(results[p]["r2"]), axis=1)
    if valid.any():
        mean_r2 = np.nanmean(results[p]["r2"][valid], axis=0)
        print(f"\n  Summary (holdout {p:.0%}, {valid.sum()} valid folds):")
        for k in [1, 3, 5, 10, 15]:
            rv = np.nanmean(results[p]["resid_var"][valid], axis=0)[k-1]
            ent = np.nanmean(results[p]["entropy"][valid], axis=0)[k-1]
            print(f"    k={k:2d}: R²={mean_r2[k-1]:.4f}  "
                  f"resid_var={rv:.4f}  entropy={ent:.3f}")


# ── Plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ks = np.arange(1, K_MAX + 1)

colors = {0.1: "#4477AA", 0.2: "#66CCEE", 0.5: "#EE7733", 0.9: "#CC3311"}

for p in HOLDOUT_FRACS:
    c = colors[p]
    label = f"{p:.0%} holdout"
    valid = ~np.all(np.isnan(results[p]["r2"]), axis=1)
    if not valid.any():
        continue

    mean_r2 = np.nanmean(results[p]["r2"][valid], axis=0)
    std_r2 = np.nanstd(results[p]["r2"][valid], axis=0)
    mean_rv = np.nanmean(results[p]["resid_var"][valid], axis=0)
    mean_ent = np.nanmean(results[p]["entropy"][valid], axis=0)

    axes[0].plot(ks, mean_r2, "o-", color=c, label=label,
                 markersize=4, linewidth=1.5)
    axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                         color=c, alpha=0.15)

    axes[1].semilogy(ks, mean_rv, "o-", color=c, label=label,
                     markersize=4, linewidth=1.5)

    axes[2].plot(ks, mean_ent, "o-", color=c, label=label,
                 markersize=4, linewidth=1.5)

axes[0].set_ylabel("Imputation $R^2$")
axes[1].set_ylabel("Residual variance fraction")
axes[2].set_ylabel("Entropy $\\log\\det(R_{\\mathcal{A}})$")

for ax in axes:
    ax.set_xlabel("Selected benchmarks $k$")
    ax.set_xlim(0.5, K_MAX + 0.5)
    ax.set_xticks([1, 3, 5, 7, 10, 13, 15])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
out_path = OUT / "greedy_cv_merged.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"\nSaved to {out_path}")
plt.close()
