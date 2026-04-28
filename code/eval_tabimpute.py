"""Evaluate benchmark selection with TabImpute V2 as the imputation backend.

Same CV protocol as eval_entropy_vs_mi.py (10-fold, 10% holdout), but replaces
Gaussian conditional mean with TabImpute V2 for imputing unselected benchmarks.

Runs entropy, MI, and random selection on all 4 datasets, reports R².
"""

import pathlib
import sys
import time
import numpy as np
from collections import defaultdict

CODE = pathlib.Path(__file__).resolve().parent
ROOT = CODE.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
LOGS = ROOT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))

from greedy_select import greedy_entropy, greedy_mi, random_select
from em_cov import em_cov
from cv_split import cv_folds, cv_train_val

import torch
from tabimpute.tabimpute_v2 import TabImputeV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42
MAX_ROWS = 4000    # max total rows per TabImpute call (GPU memory limit)
TEST_BATCH = 200   # test rows per batch; context rows = MAX_ROWS - TEST_BATCH

DATASETS = ["mmlu"]

DISPLAY_NAME = {
    "mmlu": "MMLU", "mteb": "MTEB",
    "merged": "Merged", "benchpress": "BenchPress",
}

# ── Covariance estimation (same as other scripts) ─────────────────

def estimate_corr_pairwise(B_train, O_train):
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
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = (eigvecs * eigvals[None, :]) @ eigvecs.T
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(R), 1e-12))
    R = R * np.outer(d_inv, d_inv)
    return R, col_mean, col_std


def estimate_corr_em(B_train, O_train):
    M_tr, N_tr = B_train.shape
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std_floor = 0.01 * (np.abs(col_mean) + 1.0)
    col_std = np.sqrt(np.maximum(col_var, col_std_floor ** 2))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]
    use_shrink = "auto" if M_tr < N_tr else 0.0
    em_eps = 1e-3 if M_tr < N_tr else 1e-6
    result = em_cov(B_std, O_train, max_iter=500, tol=5e-4,
                    shrinkage=use_shrink, eps_psd=em_eps, verbose=False)
    Sigma = result["Sigma"]
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    return R, col_mean, col_std


def estimate_corr(B_train, O_train):
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)
    if obs_frac < 0.9 or M_tr < N_tr:
        return estimate_corr_em(B_train, O_train)
    else:
        return estimate_corr_pairwise(B_train, O_train)


# ── TabImpute-based imputation ────────────────────────────────────

def impute_tabimpute(imputer, B_train, O_train, B_test, O_test,
                     selected, col_mean, col_std, N):
    """Impute test models' unselected benchmarks using TabImpute V2.

    Constructs a matrix with training rows (context) and test rows
    (only selected benchmarks observed), runs TabImpute, evaluates R²
    in standardized (z-scored) space for comparability with the Gaussian
    conditional mean baseline.

    All training rows are used as context (subsampled only if needed
    to fit GPU memory). Test rows are batched so that each batch sees
    the maximum number of training rows.
    """
    A = set(selected)
    M_train = B_train.shape[0]
    M_test = B_test.shape[0]

    # Context rows: all training data with observed entries
    X_ctx = np.where(O_train == 1, B_train, np.nan)

    # Subsample training rows if needed to leave room for test batches
    max_ctx = MAX_ROWS - TEST_BATCH
    if M_train > max_ctx:
        rng = np.random.default_rng(42)
        ctx_idx = rng.choice(M_train, max_ctx, replace=False)
        X_ctx = X_ctx[ctx_idx]
        n_ctx = max_ctx
    else:
        n_ctx = M_train

    # Test rows: only reveal selected benchmarks
    X_test = np.full((M_test, N), np.nan)
    for j in selected:
        for i in range(M_test):
            if O_test[i, j] == 1:
                X_test[i, j] = B_test[i, j]

    # Process test rows in batches, each seeing all context rows
    max_test_per_batch = max(1, min(TEST_BATCH, MAX_ROWS - n_ctx))
    X_test_imputed = np.empty((M_test, N), dtype=np.float64)
    for start in range(0, M_test, max_test_per_batch):
        end = min(start + max_test_per_batch, M_test)
        X_batch = np.vstack([X_ctx, X_test[start:end]])
        X_batch_imp = imputer.impute(X_batch)
        X_test_imputed[start:end] = X_batch_imp[n_ctx:]

    # Evaluate in standardized space (same as Gaussian method):
    # z-score using training col_mean/col_std, then compute R²
    ss_res = 0.0
    ss_tot = 0.0
    n_eval = 0

    for i in range(M_test):
        eval_idx = np.array([j for j in range(N)
                             if j not in A and O_test[i, j] == 1])
        if len(eval_idx) == 0:
            continue

        true_std = (B_test[i, eval_idx] - col_mean[eval_idx]) / col_std[eval_idx]
        pred_std = (X_test_imputed[i, eval_idx] - col_mean[eval_idx]) / col_std[eval_idx]
        true_std = np.clip(true_std, -10, 10)
        pred_std = np.clip(pred_std, -10, 10)

        ss_res += np.sum((true_std - pred_std) ** 2)
        ss_tot += np.sum(true_std ** 2)  # mean is 0 in standardized space
        n_eval += len(eval_idx)

    if n_eval == 0 or ss_tot < 1e-12:
        return {"r2": np.nan, "rmse": np.nan, "n_eval": 0}

    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / n_eval)
    return {"r2": r2, "rmse": rmse, "n_eval": n_eval}


# ── Run one dataset ───────────────────────────────────────────────

def run_dataset(ds_name, imputer):
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    obs_frac = O.sum() / (M * N)
    print(f"{M} models × {N} tasks, {obs_frac*100:.1f}% observed", flush=True)

    folds = cv_folds(M, N_FOLDS, SEED)
    p = HOLDOUT_FRAC

    metrics = {"entropy": defaultdict(list), "mi": defaultdict(list),
               "random": defaultdict(list)}

    for fold_idx in range(N_FOLDS):
        t0 = time.time()
        train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)

        B_train, O_train = B[train_idx], O[train_idx]
        B_test, O_test = B[val_idx], O[val_idx]
        M_train = len(train_idx)
        M_test = len(val_idx)

        print(f"  Fold {fold_idx}: {M_train} train / {M_test} test",
              end="", flush=True)

        # Estimate correlation (still needed for selection)
        try:
            R, col_mean, col_std = estimate_corr(B_train, O_train)
        except Exception as e:
            print(f" — estimation failed: {e}", flush=True)
            for method in ["entropy", "mi", "random"]:
                metrics[method]["r2"].append([np.nan] * K_MAX)
                metrics[method]["rmse"].append([np.nan] * K_MAX)
            continue

        # Run all three selection methods
        res_ent = greedy_entropy(R, K_MAX, benchmark_names=benchmark_names)
        res_mi = greedy_mi(R, K_MAX, benchmark_names=benchmark_names)
        rand_seed = (SEED + 7, fold_idx, int(p * 10000))
        res_rand = random_select(R, K_MAX, seed=rand_seed,
                                 benchmark_names=benchmark_names)

        for method, result in [("entropy", res_ent), ("mi", res_mi),
                               ("random", res_rand)]:
            fold_r2, fold_rmse = [], []
            for k in range(1, K_MAX + 1):
                selected = result["selected"][:k]
                imp = impute_tabimpute(imputer, B_train, O_train,
                                       B_test, O_test, selected,
                                       col_mean, col_std, N)
                fold_r2.append(imp["r2"])
                fold_rmse.append(imp["rmse"])

            metrics[method]["r2"].append(fold_r2)
            metrics[method]["rmse"].append(fold_rmse)

        elapsed = time.time() - t0
        ent5 = metrics['entropy']['r2'][-1][4]
        mi5 = metrics['mi']['r2'][-1][4]
        rnd5 = metrics['random']['r2'][-1][4]
        print(f"  Ent R²@5={ent5:.3f}"
              f"  MI R²@5={mi5:.3f}"
              f"  Rand R²@5={rnd5:.3f}"
              f"  ({elapsed:.1f}s)", flush=True)

    # Convert to arrays
    for method in ["entropy", "mi", "random"]:
        for key in metrics[method]:
            metrics[method][key] = np.array(metrics[method][key])

    return metrics, benchmark_names


# ── Plotting ──────────────────────────────────────────────────────

def plot_comparison(ds_name, metrics):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ks = np.arange(1, K_MAX + 1)

    colors = {"entropy": "#4477AA", "mi": "#CC3311", "random": "#888888"}
    labels = {"entropy": "Entropy (Alg. 1)", "mi": "Mutual Info (Alg. 2)",
              "random": "Random"}
    styles = {"entropy": "o-", "mi": "o-", "random": "x--"}

    for method in ["entropy", "mi", "random"]:
        c = colors[method]
        ls = styles[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        valid = ~np.all(np.isnan(metrics[method]["r2"]), axis=1)
        if not valid.any():
            continue

        mean_r2 = np.nanmean(metrics[method]["r2"][valid], axis=0)
        std_r2 = np.nanstd(metrics[method]["r2"][valid], axis=0)

        ax.plot(ks, mean_r2, ls, color=c, label=labels[method],
                markersize=ms, linewidth=lw)
        ax.fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                        color=c, alpha=0.15)

    ax.set_ylabel("Imputation $R^2$ (TabImpute V2)")
    ax.set_xlabel("Selected benchmarks $k$")
    ax.set_xlim(0.5, K_MAX + 0.5)
    ax.set_xticks([1, 3, 5, 7, 10, 13, 15])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{DISPLAY_NAME[ds_name]}")

    plt.tight_layout()
    out_path = OUT / f"tabimpute_r2_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}", flush=True)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_t0 = time.time()

    print("Loading TabImputeV2...")
    imputer = TabImputeV2(device='cpu')
    print("Model loaded.\n")

    all_metrics = {}
    for ds_name in DATASETS:
        ds_t0 = time.time()
        metrics, bench_names = run_dataset(ds_name, imputer)
        all_metrics[ds_name] = metrics
        plot_comparison(ds_name, metrics)

        # Save results
        save_dict = {"benchmark_names": bench_names}
        for method in ["entropy", "mi", "random"]:
            for key, arr in metrics[method].items():
                save_dict[f"{method}_{key}"] = arr
        np.savez(LOGS / f"tabimpute_{ds_name}.npz", **save_dict)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{ds_name.upper()} completed in {ds_elapsed:.0f}s", flush=True)

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Mean R² at k=5 and k=15 (TabImpute V2 imputation)")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Method':<10} {'R²@5':>8} {'R²@10':>8} {'R²@15':>8}")
    print("-" * 50)
    for ds_name in DATASETS:
        for method in ["entropy", "mi", "random"]:
            valid = ~np.all(np.isnan(all_metrics[ds_name][method]["r2"]), axis=1)
            if valid.any():
                mr2 = np.nanmean(all_metrics[ds_name][method]["r2"][valid], axis=0)
                print(f"{DISPLAY_NAME[ds_name]:<12} {method:<10} "
                      f"{mr2[4]:>8.4f} {mr2[9]:>8.4f} {mr2[14]:>8.4f}")
        print("-" * 50)

    total_elapsed = time.time() - total_t0
    print(f"\nAll datasets completed in {total_elapsed:.0f}s")
