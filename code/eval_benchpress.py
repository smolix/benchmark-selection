"""Run both greedy CV and entropy-vs-MI experiments on the BenchPress dataset.

Produces:
  figures/greedy_cv_benchpress.pdf
  figures/entropy_vs_mi_benchpress.pdf
"""

import pathlib
import sys
import time
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

from greedy_select import greedy_entropy, greedy_mi, random_select
from em_cov import em_cov
from cv_split import cv_folds, cv_train_val

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRACS = [0.1, 0.2, 0.5, 0.9]
SEED = 42
DS_NAME = "benchpress"


# ── Covariance estimation (EM, since data is sparse) ──────────────

def estimate_corr_em(B_train, O_train):
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


# ── Imputation (vectorized with per-model fallback) ───────────────

def impute_and_evaluate(B_test, O_test, selected, R, col_mean, col_std, N):
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
        R_oo = R[np.ix_(obs_A, obs_A)] + RIDGE * np.eye(len(obs_A))
        R_eo = R[np.ix_(eval_idx, obs_A)]
        W = np.linalg.solve(R_oo, R_eo.T).T
        X_obs = B_test_std[:, obs_A]
        X_true = B_test_std[:, eval_idx]
        X_pred = X_obs @ W.T
        ss_res = np.sum((X_true - X_pred) ** 2)
        ss_tot = np.sum(X_true ** 2)
        n_eval = M_test * len(eval_idx)
    else:
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
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "rmse": np.sqrt(ss_res / n_eval)}


# ── Part 1: Greedy CV (like eval_greedy_all.py) ──────────────────

def run_greedy_cv():
    print(f"\n{'#'*70}")
    print(f"# BenchPress: Greedy CV (figures 2-4 equivalent)")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{DS_NAME}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    print(f"{M} models × {N} tasks, {O.sum()/(M*N)*100:.1f}% observed")

    folds = cv_folds(M, N_FOLDS, SEED)
    results = {}

    for p in HOLDOUT_FRACS:
        metrics = defaultdict(list)
        print(f"\n{'='*60}\nHoldout {p:.0%}\n{'='*60}")

        for fold_idx in range(N_FOLDS):
            t0 = time.time()
            train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)
            B_train, O_train = B[train_idx], O[train_idx]
            B_test, O_test = B[val_idx], O[val_idx]

            print(f"  Fold {fold_idx}: {len(train_idx)} train / {len(val_idx)} test",
                  end="", flush=True)

            try:
                R, col_mean, col_std = estimate_corr_em(B_train, O_train)
            except Exception as e:
                print(f" — failed: {e}", flush=True)
                for key in ["entropy", "resid_var", "r2", "rmse",
                            "rand_entropy", "rand_resid_var", "rand_r2", "rand_rmse"]:
                    metrics[key].append([np.nan] * K_MAX)
                continue

            # Greedy entropy
            result = greedy_entropy(R, K_MAX, benchmark_names=benchmark_names)

            fold_ent, fold_rv, fold_r2, fold_rmse = [], [], [], []
            for k in range(1, K_MAX + 1):
                selected = result["selected"][:k]
                R_AA = R[np.ix_(selected, selected)]
                sign, logdet = np.linalg.slogdet(R_AA)
                fold_ent.append(logdet if sign > 0 else -np.inf)
                fold_rv.append(result["residual_variance"][k - 1])
                imp = impute_and_evaluate(B_test, O_test, selected, R,
                                          col_mean, col_std, N)
                fold_r2.append(imp["r2"])
                fold_rmse.append(imp["rmse"])

            metrics["entropy"].append(fold_ent)
            metrics["resid_var"].append(fold_rv)
            metrics["r2"].append(fold_r2)
            metrics["rmse"].append(fold_rmse)

            # Random baseline
            rand_seed = (SEED + 7, fold_idx, int(p * 10000))
            res_rand = random_select(R, K_MAX, seed=rand_seed,
                                     benchmark_names=benchmark_names)
            rand_ent, rand_rv, rand_r2, rand_rmse = [], [], [], []
            for k in range(1, K_MAX + 1):
                sel_rand = res_rand["selected"][:k]
                R_AA_r = R[np.ix_(sel_rand, sel_rand)]
                s_r, ld_r = np.linalg.slogdet(R_AA_r)
                rand_ent.append(ld_r if s_r > 0 else -np.inf)
                rand_rv.append(res_rand["residual_variance"][k - 1])
                imp_r = impute_and_evaluate(B_test, O_test, sel_rand, R,
                                            col_mean, col_std, N)
                rand_r2.append(imp_r["r2"])
                rand_rmse.append(imp_r["rmse"])
            metrics["rand_entropy"].append(rand_ent)
            metrics["rand_resid_var"].append(rand_rv)
            metrics["rand_r2"].append(rand_r2)
            metrics["rand_rmse"].append(rand_rmse)

            elapsed = time.time() - t0
            print(f"  R²@5={fold_r2[4]:.3f}  Rand@5={rand_r2[4]:.3f}"
                  f"  ({elapsed:.1f}s)", flush=True)

        results[p] = {k: np.array(v) for k, v in metrics.items()}

        valid = ~np.all(np.isnan(results[p]["r2"]), axis=1)
        if valid.any():
            mean_r2 = np.nanmean(results[p]["r2"][valid], axis=0)
            print(f"\n  Summary (holdout {p:.0%}, {valid.sum()} valid folds):")
            for k in [1, 3, 5, 10, 15]:
                rv = np.nanmean(results[p]["resid_var"][valid], axis=0)[k-1]
                print(f"    k={k:2d}: R²={mean_r2[k-1]:.4f}  resid_var={rv:.4f}")

    return results


def plot_greedy_cv(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = np.arange(1, K_MAX + 1)
    colors = {0.1: "#4477AA", 0.2: "#66CCEE", 0.5: "#EE7733", 0.9: "#CC3311"}

    for p in HOLDOUT_FRACS:
        if p not in results:
            continue
        c = colors[p]
        valid = ~np.all(np.isnan(results[p]["r2"]), axis=1)
        if not valid.any():
            continue

        mean_r2 = np.nanmean(results[p]["r2"][valid], axis=0)
        std_r2 = np.nanstd(results[p]["r2"][valid], axis=0)
        mean_rv = np.nanmean(results[p]["resid_var"][valid], axis=0)
        mean_ent = np.nanmean(results[p]["entropy"][valid], axis=0)

        axes[0].plot(ks, mean_r2, "o-", color=c, label=f"{p:.0%} holdout",
                     markersize=4, linewidth=1.5)
        axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                             color=c, alpha=0.15)
        axes[1].semilogy(ks, mean_rv, "o-", color=c, label=f"{p:.0%} holdout",
                         markersize=4, linewidth=1.5)
        axes[2].plot(ks, mean_ent, "o-", color=c, label=f"{p:.0%} holdout",
                     markersize=4, linewidth=1.5)

        # Random baseline
        if "rand_r2" in results[p]:
            valid_r = ~np.all(np.isnan(results[p]["rand_r2"]), axis=1)
            if valid_r.any():
                mr2_r = np.nanmean(results[p]["rand_r2"][valid_r], axis=0)
                mrv_r = np.nanmean(results[p]["rand_resid_var"][valid_r], axis=0)
                ment_r = np.nanmean(results[p]["rand_entropy"][valid_r], axis=0)
                axes[0].plot(ks, mr2_r, "x--", color=c,
                             markersize=3, linewidth=1.0, alpha=0.6)
                axes[1].semilogy(ks, mrv_r, "x--", color=c,
                                 markersize=3, linewidth=1.0, alpha=0.6)
                axes[2].plot(ks, ment_r, "x--", color=c,
                             markersize=3, linewidth=1.0, alpha=0.6)

    axes[0].plot([], [], "x--", color="gray", linewidth=1.0, alpha=0.6, label="Random")

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
    out_path = OUT / f"greedy_cv_{DS_NAME}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ── Part 2: Entropy vs MI (like eval_entropy_vs_mi.py) ───────────

def run_entropy_vs_mi():
    print(f"\n{'#'*70}")
    print(f"# BenchPress: Entropy vs MI vs Random (figures 8-10 equivalent)")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{DS_NAME}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    print(f"{M} models × {N} tasks, {O.sum()/(M*N)*100:.1f}% observed")

    folds = cv_folds(M, N_FOLDS, SEED)
    p = 0.1  # 10% holdout
    metrics = {"entropy": defaultdict(list), "mi": defaultdict(list),
               "random": defaultdict(list)}

    for fold_idx in range(N_FOLDS):
        t0 = time.time()
        train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)
        B_train, O_train = B[train_idx], O[train_idx]
        B_test, O_test = B[val_idx], O[val_idx]

        print(f"  Fold {fold_idx}: {len(train_idx)} train / {len(val_idx)} test",
              end="", flush=True)

        try:
            R, col_mean, col_std = estimate_corr_em(B_train, O_train)
        except Exception as e:
            print(f" — failed: {e}", flush=True)
            for method in ["entropy", "mi", "random"]:
                for metric in ["r2", "rmse", "resid_var", "mi_val"]:
                    metrics[method][metric].append([np.nan] * K_MAX)
            continue

        # Run all three methods
        res_ent = greedy_entropy(R, K_MAX, benchmark_names=benchmark_names)
        res_mi = greedy_mi(R, K_MAX, benchmark_names=benchmark_names)
        rand_seed = (SEED + 7, fold_idx, int(p * 10000))
        res_rand = random_select(R, K_MAX, seed=rand_seed,
                                 benchmark_names=benchmark_names)

        sign_full, logdet_full = np.linalg.slogdet(R)
        if sign_full <= 0:
            eigv = np.linalg.eigvalsh(R)
            logdet_full = np.sum(np.log(np.maximum(eigv, 1e-300)))

        for method, result in [("entropy", res_ent), ("mi", res_mi),
                               ("random", res_rand)]:
            fold_r2, fold_rmse, fold_rv, fold_mi = [], [], [], []
            for k in range(1, K_MAX + 1):
                selected = result["selected"][:k]
                fold_rv.append(result["residual_variance"][k - 1])
                imp = impute_and_evaluate(B_test, O_test, selected, R,
                                          col_mean, col_std, N)
                fold_r2.append(imp["r2"])
                fold_rmse.append(imp["rmse"])

                R_AA = R[np.ix_(selected, selected)]
                s_a, ld_a = np.linalg.slogdet(R_AA)
                ld_a = ld_a if s_a > 0 else -1e30
                comp = sorted(set(range(N)) - set(selected))
                R_CC = R[np.ix_(comp, comp)]
                s_c, ld_c = np.linalg.slogdet(R_CC)
                ld_c = ld_c if s_c > 0 else -1e30
                fold_mi.append(0.5 * (ld_a + ld_c - logdet_full))

            metrics[method]["r2"].append(fold_r2)
            metrics[method]["rmse"].append(fold_rmse)
            metrics[method]["resid_var"].append(fold_rv)
            metrics[method]["mi_val"].append(fold_mi)

        elapsed = time.time() - t0
        print(f"  Ent R²@5={metrics['entropy']['r2'][-1][4]:.3f}"
              f"  MI R²@5={metrics['mi']['r2'][-1][4]:.3f}"
              f"  Rand R²@5={metrics['random']['r2'][-1][4]:.3f}"
              f"  ({elapsed:.1f}s)", flush=True)

    for method in ["entropy", "mi", "random"]:
        for key in metrics[method]:
            metrics[method][key] = np.array(metrics[method][key])

    return metrics


def plot_entropy_vs_mi(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
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
        mean_rv = np.nanmean(metrics[method]["resid_var"][valid], axis=0)

        axes[0].plot(ks, mean_r2, ls, color=c, label=labels[method],
                     markersize=ms, linewidth=lw)
        axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                             color=c, alpha=0.15)
        axes[1].semilogy(ks, mean_rv, ls, color=c, label=labels[method],
                         markersize=ms, linewidth=lw)

    for method in ["entropy", "mi", "random"]:
        c = colors[method]
        ls = styles[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        if "mi_val" in metrics[method]:
            valid = ~np.all(np.isnan(metrics[method]["mi_val"]), axis=1)
            if valid.any():
                mean_mi = np.nanmean(metrics[method]["mi_val"][valid], axis=0)
                axes[2].plot(ks, mean_mi, ls, color=c, label=labels[method],
                             markersize=ms, linewidth=lw)

    axes[0].set_ylabel("Imputation $R^2$")
    axes[1].set_ylabel("Residual variance fraction")
    axes[2].set_ylabel("Mutual information $I(X_{\\mathcal{A}}; X_{\\bar{\\mathcal{A}}})$")
    for ax in axes:
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xlim(0.5, K_MAX + 0.5)
        ax.set_xticks([1, 3, 5, 7, 10, 13, 15])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT / f"entropy_vs_mi_{DS_NAME}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_t0 = time.time()

    # Part 1: Greedy CV
    cv_results = run_greedy_cv()
    plot_greedy_cv(cv_results)

    # Part 2: Entropy vs MI
    mi_metrics = run_entropy_vs_mi()
    plot_entropy_vs_mi(mi_metrics)

    # Save results
    save_dict = {"benchmark_names": np.load(DATA / f"{DS_NAME}.matrix.npz",
                                            allow_pickle=True)["benchmark_names"]}
    for p, m in cv_results.items():
        prefix = f"cv_p{int(p*100)}"
        for key, arr in m.items():
            save_dict[f"{prefix}_{key}"] = arr
    for method in ["entropy", "mi", "random"]:
        for key, arr in mi_metrics[method].items():
            save_dict[f"emi_{method}_{key}"] = arr
    np.savez(LOGS / f"benchpress_results.npz", **save_dict)

    total_elapsed = time.time() - total_t0
    print(f"\nBenchPress experiments completed in {total_elapsed:.0f}s")
