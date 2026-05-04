"""Evaluate greedy benchmark selection on all datasets with cross-validation.

For each dataset (MMLU, MTEB, Merged) and holdout fraction:
  1. Split models into train/test (10-fold CV)
  2. Estimate correlation matrix from training models
  3. Run greedy entropy selection for k=1..K_MAX
  4. For each k, impute test models' unselected scores from selected ones
  5. Report: entropy, residual variance, R², RMSE on held-out data
  6. Log full selection order per fold

Saves detailed results (including selection orders) to npz files for later use.
"""

import pathlib
import sys
import time
import json
import numpy as np
from collections import defaultdict

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import greedy_entropy, random_select
from em_cov import em_cov
from cv_split import cv_folds, cv_train_val

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRACS = [0.1, 0.2, 0.5, 0.9]
SEED = 42

DATASETS = ["mmlu", "mteb", "merged"]


# ── Covariance estimation ────────────────────────────────────────

def estimate_corr_pairwise(B_train, O_train):
    """Estimate correlation matrix via pairwise-complete covariance.
    Suitable for nearly-complete or complete data.
    """
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))

    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]

    gram = B_std.T @ B_std
    pair_counts = O_train.T @ O_train
    denom = np.maximum(pair_counts - 1.0, 1.0)
    R = gram / denom

    # Project PSD
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = (eigvecs * eigvals[None, :]) @ eigvecs.T

    # Force unit diagonal
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(R), 1e-12))
    R = R * np.outer(d_inv, d_inv)

    return R, col_mean, col_std


def estimate_corr_em(B_train, O_train):
    """Estimate correlation matrix via EM. For sparse/rank-deficient data."""
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)

    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std_floor = 0.01 * (np.abs(col_mean) + 1.0)
    col_std = np.sqrt(np.maximum(col_var, col_std_floor ** 2))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]

    regularized = M_tr < N_tr or obs_frac < 0.5
    use_shrink = "auto" if M_tr < N_tr else 0.0
    em_eps = 1e-3 if regularized else 1e-6
    em_iter = 1000 if regularized else 500
    result = em_cov(B_std, O_train, max_iter=em_iter, tol=5e-4,
                    shrinkage=use_shrink, eps_psd=em_eps, verbose=False)
    Sigma = result["Sigma"]

    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)

    return R, col_mean, col_std


def estimate_corr(B_train, O_train, method="auto"):
    """Choose estimation method based on data characteristics."""
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)

    if method == "auto":
        # Use EM for sparse data or rank-deficient case
        if obs_frac < 0.9 or M_tr < N_tr:
            return estimate_corr_em(B_train, O_train)
        else:
            return estimate_corr_pairwise(B_train, O_train)
    elif method == "em":
        return estimate_corr_em(B_train, O_train)
    else:
        return estimate_corr_pairwise(B_train, O_train)


# ── Imputation and evaluation ────────────────────────────────────

def impute_and_evaluate(B_test, O_test, selected, R, col_mean, col_std, N):
    """Impute unselected benchmarks for test models and compute metrics.

    Uses conditional Gaussian with ridge regularization.
    Vectorized: when all test models share the same observation pattern
    (common for fully-observed data), a single matrix solve covers all models.
    """
    A = set(selected)
    M_test = B_test.shape[0]

    B_test_std = O_test * (B_test - col_mean[None, :]) / col_std[None, :]
    B_test_std = np.clip(B_test_std, -10, 10) * O_test

    RIDGE = 0.01

    # Check if all models share the same observation pattern (fast path)
    obs_A = np.array([j for j in selected if O_test[0, j] == 1])
    eval_idx = np.array([j for j in range(N) if j not in A and O_test[0, j] == 1])
    all_same = (len(obs_A) > 0 and len(eval_idx) > 0 and
                np.all(O_test[:, list(selected)] == O_test[0, list(selected)]) and
                np.all(O_test[:, eval_idx] == O_test[0, eval_idx]))

    if all_same:
        # Vectorized: single solve for all models
        R_oo = R[np.ix_(obs_A, obs_A)] + RIDGE * np.eye(len(obs_A))
        R_eo = R[np.ix_(eval_idx, obs_A)]
        W = np.linalg.solve(R_oo, R_eo.T).T  # (n_eval, n_obs)

        X_obs = B_test_std[:, obs_A]          # (M_test, n_obs)
        X_true = B_test_std[:, eval_idx]      # (M_test, n_eval)
        X_pred = X_obs @ W.T                  # (M_test, n_eval)

        ss_res = np.sum((X_true - X_pred) ** 2)
        ss_tot = np.sum(X_true ** 2)
        n_eval = M_test * len(eval_idx)
    else:
        # Per-model fallback for heterogeneous observation patterns
        ss_res = 0.0
        ss_tot = 0.0
        n_eval = 0

        for i in range(M_test):
            obs_A_i = np.array([j for j in selected if O_test[i, j] == 1])
            eval_i = np.array([j for j in range(N) if j not in A and O_test[i, j] == 1])

            if len(obs_A_i) == 0 or len(eval_i) == 0:
                continue

            R_oo = R[np.ix_(obs_A_i, obs_A_i)] + RIDGE * np.eye(len(obs_A_i))
            R_eo = R[np.ix_(eval_i, obs_A_i)]
            W = np.linalg.solve(R_oo, R_eo.T).T

            x_obs = B_test_std[i, obs_A_i]
            x_true = B_test_std[i, eval_i]
            x_pred = W @ x_obs

            ss_res += np.sum((x_true - x_pred) ** 2)
            ss_tot += np.sum(x_true ** 2)
            n_eval += len(eval_i)

    if n_eval == 0:
        return {"r2": np.nan, "rmse": np.nan}

    rmse = np.sqrt(ss_res / n_eval)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {"r2": r2, "rmse": rmse}


# ── Main ─────────────────────────────────────────────────────────

def run_dataset(ds_name):
    """Run full CV evaluation for one dataset."""
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    obs_frac = O.sum() / (M * N)
    print(f"{M} models × {N} tasks, {obs_frac*100:.1f}% observed")

    folds = cv_folds(M, N_FOLDS, SEED)

    results = {}
    all_logs = []  # detailed per-fold logs

    for p in HOLDOUT_FRACS:
        metrics = defaultdict(list)
        fold_selections = []  # selection order per fold

        print(f"\n{'='*60}")
        print(f"Holdout {p:.0%}")
        print(f"{'='*60}")

        for fold_idx in range(N_FOLDS):
            t0 = time.time()
            train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)

            B_train, O_train = B[train_idx], O[train_idx]
            B_test, O_test = B[val_idx], O[val_idx]
            M_train = len(train_idx)
            M_test = len(val_idx)

            print(f"  Fold {fold_idx}: {M_train} train / {M_test} test",
                  end="", flush=True)

            # Estimate correlation
            try:
                R, col_mean, col_std = estimate_corr(B_train, O_train)
            except Exception as e:
                print(f" — estimation failed: {e}", flush=True)
                metrics["entropy"].append([np.nan] * K_MAX)
                metrics["resid_var"].append([np.nan] * K_MAX)
                metrics["r2"].append([np.nan] * K_MAX)
                metrics["rmse"].append([np.nan] * K_MAX)
                metrics["rand_entropy"].append([np.nan] * K_MAX)
                metrics["rand_resid_var"].append([np.nan] * K_MAX)
                metrics["rand_r2"].append([np.nan] * K_MAX)
                metrics["rand_rmse"].append([np.nan] * K_MAX)
                fold_selections.append({"selected_idx": [],
                                        "selected_names": []})
                continue

            # Run greedy
            result = greedy_entropy(R, K_MAX, benchmark_names=benchmark_names)

            # Log selection order
            sel_info = {
                "selected_idx": result["selected"],
                "selected_names": result.get("names", []),
            }
            fold_selections.append(sel_info)

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
                                          col_mean, col_std, N)
                fold_r2.append(imp["r2"])
                fold_rmse.append(imp["rmse"])

            metrics["entropy"].append(fold_entropy)
            metrics["resid_var"].append(fold_resid_var)
            metrics["r2"].append(fold_r2)
            metrics["rmse"].append(fold_rmse)

            # Run random baseline
            rand_seed = (SEED + 7, fold_idx, int(p * 10000))
            res_rand = random_select(R, K_MAX, seed=rand_seed,
                                     benchmark_names=benchmark_names)
            rand_entropy, rand_rv, rand_r2, rand_rmse = [], [], [], []
            for k in range(1, K_MAX + 1):
                sel_rand = res_rand["selected"][:k]
                R_AA_r = R[np.ix_(sel_rand, sel_rand)]
                s_r, ld_r = np.linalg.slogdet(R_AA_r)
                rand_entropy.append(ld_r if s_r > 0 else -np.inf)
                rand_rv.append(res_rand["residual_variance"][k - 1])
                imp_r = impute_and_evaluate(B_test, O_test, sel_rand, R,
                                            col_mean, col_std, N)
                rand_r2.append(imp_r["r2"])
                rand_rmse.append(imp_r["rmse"])
            metrics["rand_entropy"].append(rand_entropy)
            metrics["rand_resid_var"].append(rand_rv)
            metrics["rand_r2"].append(rand_r2)
            metrics["rand_rmse"].append(rand_rmse)

            elapsed = time.time() - t0
            print(f"  R²@5={fold_r2[4]:.3f}  R²@10={fold_r2[9]:.3f}"
                  f"  Rand@5={rand_r2[4]:.3f}"
                  f"  ({elapsed:.1f}s)", flush=True)

            # Print selection order for this fold
            names = result.get("names", [str(i) for i in result["selected"]])
            top5 = ", ".join(names[:5])
            print(f"         selected: {top5}, ...", flush=True)

        results[p] = {k: np.array(v) for k, v in metrics.items()}

        # Store fold logs
        for fi, sel in enumerate(fold_selections):
            all_logs.append({
                "dataset": ds_name,
                "holdout_frac": p,
                "fold": fi,
                "selected_idx": sel["selected_idx"],
                "selected_names": sel["selected_names"],
            })

        # Print summary
        valid = ~np.all(np.isnan(results[p]["r2"]), axis=1)
        if valid.any():
            mean_r2 = np.nanmean(results[p]["r2"][valid], axis=0)
            print(f"\n  Summary (holdout {p:.0%}, {valid.sum()} valid folds):",
                  flush=True)
            for k in [1, 3, 5, 10, 15]:
                rv = np.nanmean(results[p]["resid_var"][valid], axis=0)[k-1]
                ent = np.nanmean(results[p]["entropy"][valid], axis=0)[k-1]
                rmse = np.nanmean(results[p]["rmse"][valid], axis=0)[k-1]
                print(f"    k={k:2d}: R²={mean_r2[k-1]:.4f}  "
                      f"resid_var={rv:.4f}  entropy={ent:.3f}  "
                      f"rmse={rmse:.4f}", flush=True)

    return results, all_logs, benchmark_names


# ── Plotting helper ──────────────────────────────────────────────

def plot_results(ds_name, results):
    """Generate 3-panel CV plot for one dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = np.arange(1, K_MAX + 1)

    colors = {0.1: "#4477AA", 0.2: "#66CCEE",
              0.5: "#EE7733", 0.9: "#CC3311"}

    for p in HOLDOUT_FRACS:
        if p not in results:
            continue
        c = colors[p]
        label_g = f"{p:.0%} holdout"
        valid = ~np.all(np.isnan(results[p]["r2"]), axis=1)
        if not valid.any():
            continue

        mean_r2 = np.nanmean(results[p]["r2"][valid], axis=0)
        std_r2 = np.nanstd(results[p]["r2"][valid], axis=0)
        mean_rv = np.nanmean(results[p]["resid_var"][valid], axis=0)
        mean_ent = np.nanmean(results[p]["entropy"][valid], axis=0)

        axes[0].plot(ks, mean_r2, "o-", color=c, label=label_g,
                     markersize=4, linewidth=1.5)
        axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                             color=c, alpha=0.15)
        axes[1].semilogy(ks, mean_rv, "o-", color=c, label=label_g,
                         markersize=4, linewidth=1.5)
        axes[2].plot(ks, mean_ent, "o-", color=c, label=label_g,
                     markersize=4, linewidth=1.5)

        # Random baseline (dashed, same color)
        if "rand_r2" in results[p]:
            valid_r = ~np.all(np.isnan(results[p]["rand_r2"]), axis=1)
            if valid_r.any():
                label_r = f"Random {p:.0%}" if p == 0.1 else None
                mr2_r = np.nanmean(results[p]["rand_r2"][valid_r], axis=0)
                mrv_r = np.nanmean(results[p]["rand_resid_var"][valid_r], axis=0)
                ment_r = np.nanmean(results[p]["rand_entropy"][valid_r], axis=0)
                axes[0].plot(ks, mr2_r, "x--", color=c, label=label_r,
                             markersize=3, linewidth=1.0, alpha=0.6)
                axes[1].semilogy(ks, mrv_r, "x--", color=c,
                                 markersize=3, linewidth=1.0, alpha=0.6)
                axes[2].plot(ks, ment_r, "x--", color=c,
                             markersize=3, linewidth=1.0, alpha=0.6)

    # Add a single legend entry for "Random" via a dummy artist
    from matplotlib.lines import Line2D
    axes[0].plot([], [], "x--", color="gray", linewidth=1.0,
                 alpha=0.6, label="Random")

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
    out_path = OUT / f"greedy_cv_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}", flush=True)
    plt.close()


# ── Run all datasets ─────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}
    all_logs = []

    total_t0 = time.time()

    for ds_name in DATASETS:
        ds_t0 = time.time()
        results, logs, bench_names = run_dataset(ds_name)
        all_results[ds_name] = results
        all_logs.extend(logs)
        plot_results(ds_name, results)

        # Save per-dataset results
        save_dict = {}
        for p, metrics in results.items():
            prefix = f"p{int(p*100)}"
            for metric_name, arr in metrics.items():
                save_dict[f"{prefix}_{metric_name}"] = arr
        save_dict["benchmark_names"] = bench_names
        save_dict["holdout_fracs"] = np.array(HOLDOUT_FRACS)
        np.savez(LOGS / f"greedy_cv_{ds_name}.npz", **save_dict)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{ds_name.upper()} completed in {ds_elapsed:.0f}s", flush=True)

    # Save full selection logs as JSON
    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_logs = []
    for log in all_logs:
        json_logs.append({k: convert(v) for k, v in log.items()})

    log_path = LOGS / "greedy_cv_selections.json"
    with open(log_path, "w") as f:
        json.dump(json_logs, f, indent=2, default=convert)
    print(f"\nSelection logs saved to {log_path}", flush=True)

    total_elapsed = time.time() - total_t0
    print(f"\nAll datasets completed in {total_elapsed:.0f}s", flush=True)
