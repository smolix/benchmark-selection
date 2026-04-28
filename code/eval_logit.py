"""Run all experiments in logit-transformed score space.

Applies s -> logit(s / max_j) before covariance estimation and selection.
Predictions are inverted via sigmoid and evaluated in raw-score space.

Produces:
  figures_logit/eigenspectrum_logit.pdf
  figures_logit/greedy_cv_{dataset}_logit.pdf        (4 datasets)
  figures_logit/entropy_vs_mi_{dataset}_logit.pdf     (4 datasets)
  logs_logit/*.npz
"""

import pathlib
import sys
import time
import json
import numpy as np
from collections import defaultdict

CODE = pathlib.Path(__file__).resolve().parent
ROOT = CODE.parent
DATA = ROOT / "data"
OUT = ROOT / "figures_logit"
LOGS = ROOT / "logs_logit"
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))

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
EPS = 1e-3  # clipping bound for logit

DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
DISPLAY = {"mmlu": "MMLU", "mteb": "MTEB",
           "merged": "Merged", "benchpress": "BenchPress"}


# ── Logit transform ───────────────────────────────────────────────

def logit_transform(B, O, col_max=None):
    """Transform scores to logit space using 0-max normalization.

    Parameters
    ----------
    B : (M, N) score matrix (0 for unobserved entries).
    O : (M, N) observation mask.
    col_max : (N,) per-benchmark max. If None, computed from B, O.

    Returns
    -------
    B_logit : (M, N) logit-transformed scores (0 for unobserved).
    col_max : (N,) the max values used.
    """
    N = B.shape[1]
    if col_max is None:
        col_max = np.zeros(N)
        for j in range(N):
            obs = B[:, j][O[:, j] == 1]
            col_max[j] = obs.max() if len(obs) > 0 else 1.0
        # Ensure col_max > 0
        col_max = np.maximum(col_max, EPS)

    B_logit = np.zeros_like(B)
    for j in range(N):
        s = B[:, j] / col_max[j]              # normalize to [0, 1]
        s = np.clip(s, EPS, 1.0 - EPS)        # avoid log(0)
        f = np.log(s / (1.0 - s))             # logit
        B_logit[:, j] = f * O[:, j]           # mask unobserved
    return B_logit, col_max


def sigmoid(f):
    return 1.0 / (1.0 + np.exp(-f))


# ── Covariance estimation ────────────────────────────────────────

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


# ── Imputation in logit space, R² in raw space ───────────────────

def impute_and_evaluate_logit(B_test_logit, O_test, selected, R,
                              col_mean_logit, col_std_logit,
                              B_test_raw, col_mean_raw, col_std_raw,
                              col_max, N):
    """Impute in logit space, invert via sigmoid, evaluate R² in
    standardized raw-score space (for comparability with non-logit results).
    """
    A = set(selected)
    M_test = B_test_logit.shape[0]
    RIDGE = 0.01

    # Standardize logit scores for imputation
    B_test_std = O_test * (B_test_logit - col_mean_logit[None, :]) / col_std_logit[None, :]
    B_test_std = np.clip(B_test_std, -10, 10) * O_test

    # Check if all models share the same observation pattern (fast path)
    obs_A = np.array([j for j in selected if O_test[0, j] == 1])
    eval_idx = np.array([j for j in range(N) if j not in A and O_test[0, j] == 1])
    all_same = (len(obs_A) > 0 and len(eval_idx) > 0 and
                np.all(O_test[:, list(selected)] == O_test[0, list(selected)]) and
                np.all(O_test[:, eval_idx] == O_test[0, eval_idx]))

    ss_res = 0.0
    ss_tot = 0.0
    n_eval = 0

    if all_same:
        R_oo = R[np.ix_(obs_A, obs_A)] + RIDGE * np.eye(len(obs_A))
        R_eo = R[np.ix_(eval_idx, obs_A)]
        W = np.linalg.solve(R_oo, R_eo.T).T
        X_obs = B_test_std[:, obs_A]
        # Predict in standardized logit space
        X_pred_std_logit = X_obs @ W.T

        # Convert predictions to raw score space
        # Unstandardize logit: f_pred = pred_std * col_std_logit + col_mean_logit
        F_pred = X_pred_std_logit * col_std_logit[eval_idx] + col_mean_logit[eval_idx]
        # Sigmoid and rescale: s_pred = sigmoid(f) * col_max
        S_pred = sigmoid(F_pred) * col_max[eval_idx]

        # True raw scores, standardized using raw training stats
        S_true = B_test_raw[:, eval_idx]
        true_std = (S_true - col_mean_raw[eval_idx]) / col_std_raw[eval_idx]
        pred_std = (S_pred - col_mean_raw[eval_idx]) / col_std_raw[eval_idx]
        true_std = np.clip(true_std, -10, 10)
        pred_std = np.clip(pred_std, -10, 10)

        ss_res = np.sum((true_std - pred_std) ** 2)
        ss_tot = np.sum(true_std ** 2)
        n_eval = M_test * len(eval_idx)
    else:
        for i in range(M_test):
            obs_A_i = np.array([j for j in selected if O_test[i, j] == 1])
            eval_i = np.array([j for j in range(N) if j not in A and O_test[i, j] == 1])
            if len(obs_A_i) == 0 or len(eval_i) == 0:
                continue

            R_oo = R[np.ix_(obs_A_i, obs_A_i)] + RIDGE * np.eye(len(obs_A_i))
            R_eo = R[np.ix_(eval_i, obs_A_i)]
            W = np.linalg.solve(R_oo, R_eo.T).T
            x_obs = B_test_std[i, obs_A_i]
            f_pred_std = W @ x_obs

            # To raw space
            f_pred = f_pred_std * col_std_logit[eval_i] + col_mean_logit[eval_i]
            s_pred = sigmoid(f_pred) * col_max[eval_i]

            s_true = B_test_raw[i, eval_i]
            t_std = (s_true - col_mean_raw[eval_i]) / col_std_raw[eval_i]
            p_std = (s_pred - col_mean_raw[eval_i]) / col_std_raw[eval_i]
            t_std = np.clip(t_std, -10, 10)
            p_std = np.clip(p_std, -10, 10)

            ss_res += np.sum((t_std - p_std) ** 2)
            ss_tot += np.sum(t_std ** 2)
            n_eval += len(eval_i)

    if n_eval == 0:
        return {"r2": np.nan, "rmse": np.nan}
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "rmse": np.sqrt(ss_res / n_eval)}


# ── Raw-space col_mean and col_std (for R² evaluation) ───────────

def raw_stats(B, O):
    col_count = np.maximum(O.sum(axis=0), 1.0)
    col_mean = (B * O).sum(axis=0) / col_count
    B_c = O * (B - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))
    return col_mean, col_std


# ══════════════════════════════════════════════════════════════════
# Part 1: Eigenspectrum
# ══════════════════════════════════════════════════════════════════

def plot_eigenspectrum_logit():
    print("\n" + "#"*70)
    print("# Eigenspectrum (logit space)")
    print("#"*70)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"mmlu": "#4477AA", "mteb": "#CC3311",
              "merged": "#EE7733", "benchpress": "#228833"}

    for ds in DATASETS:
        d = np.load(DATA / f"{ds}.matrix.npz", allow_pickle=True)
        B = d["B"].astype(np.float64)
        O = d["O"].astype(np.float64)
        B_logit, col_max = logit_transform(B, O)

        R, _, _ = estimate_corr(B_logit, O)

        eigvals = np.linalg.eigvalsh(R)[::-1]
        N = len(eigvals)
        cumvar = np.cumsum(eigvals) / eigvals.sum()
        resid = 1.0 - cumvar
        ks = np.arange(1, N + 1)
        ax.semilogy(ks[:30], resid[:30], "o-", color=colors[ds],
                     label=DISPLAY[ds], markersize=3, linewidth=1.5)
        print(f"  {DISPLAY[ds]}: 90% at k={np.searchsorted(cumvar, 0.9)+1}, "
              f"95% at k={np.searchsorted(cumvar, 0.95)+1}")

    ax.axhline(0.10, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.axhline(0.05, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.axhline(0.01, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.set_xlabel("Number of components $k$")
    ax.set_ylabel("Residual variance fraction $1 - \\rho(k)$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT / "eigenspectrum_logit.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Part 2: Greedy CV (entropy + random, 4 holdout fracs)
# ══════════════════════════════════════════════════════════════════

def run_greedy_cv(ds_name):
    print(f"\n{'#'*70}")
    print(f"# {DISPLAY[ds_name]}: Greedy CV (logit)")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B_raw = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B_raw.shape
    print(f"{M} models × {N} tasks, {O.sum()/(M*N)*100:.1f}% observed")

    folds = cv_folds(M, N_FOLDS, SEED)
    results = {}

    for p in HOLDOUT_FRACS:
        metrics = defaultdict(list)
        print(f"\n  Holdout {p:.0%}")

        for fold_idx in range(N_FOLDS):
            t0 = time.time()
            train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)
            B_train_raw, O_train = B_raw[train_idx], O[train_idx]
            B_test_raw, O_test = B_raw[val_idx], O[val_idx]

            # Logit transform using training col_max
            B_train_logit, col_max = logit_transform(B_train_raw, O_train)
            B_test_logit, _ = logit_transform(B_test_raw, O_test, col_max=col_max)

            # Raw stats for R² evaluation
            cm_raw, cs_raw = raw_stats(B_train_raw, O_train)

            try:
                R, cm_logit, cs_logit = estimate_corr(B_train_logit, O_train)
            except Exception as e:
                print(f"    Fold {fold_idx}: failed ({e})")
                for key in ["entropy", "resid_var", "r2", "rmse",
                            "rand_entropy", "rand_resid_var", "rand_r2", "rand_rmse"]:
                    metrics[key].append([np.nan] * K_MAX)
                continue

            # Greedy entropy
            result = greedy_entropy(R, K_MAX, benchmark_names=benchmark_names)

            fold_ent, fold_rv, fold_r2, fold_rmse = [], [], [], []
            for k in range(1, K_MAX + 1):
                sel = result["selected"][:k]
                R_AA = R[np.ix_(sel, sel)]
                sign, logdet = np.linalg.slogdet(R_AA)
                fold_ent.append(logdet if sign > 0 else -np.inf)
                fold_rv.append(result["residual_variance"][k - 1])
                imp = impute_and_evaluate_logit(
                    B_test_logit, O_test, sel, R, cm_logit, cs_logit,
                    B_test_raw, cm_raw, cs_raw, col_max, N)
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
                sel_r = res_rand["selected"][:k]
                R_AA_r = R[np.ix_(sel_r, sel_r)]
                s_r, ld_r = np.linalg.slogdet(R_AA_r)
                rand_ent.append(ld_r if s_r > 0 else -np.inf)
                rand_rv.append(res_rand["residual_variance"][k - 1])
                imp_r = impute_and_evaluate_logit(
                    B_test_logit, O_test, sel_r, R, cm_logit, cs_logit,
                    B_test_raw, cm_raw, cs_raw, col_max, N)
                rand_r2.append(imp_r["r2"])
                rand_rmse.append(imp_r["rmse"])
            metrics["rand_entropy"].append(rand_ent)
            metrics["rand_resid_var"].append(rand_rv)
            metrics["rand_r2"].append(rand_r2)
            metrics["rand_rmse"].append(rand_rmse)

            elapsed = time.time() - t0
            print(f"    Fold {fold_idx}: R²@5={fold_r2[4]:.3f}  "
                  f"Rand@5={rand_r2[4]:.3f}  ({elapsed:.1f}s)", flush=True)

        results[p] = {k: np.array(v) for k, v in metrics.items()}

    return results


def plot_greedy_cv(ds_name, results):
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
        mr2 = np.nanmean(results[p]["r2"][valid], axis=0)
        sr2 = np.nanstd(results[p]["r2"][valid], axis=0)
        mrv = np.nanmean(results[p]["resid_var"][valid], axis=0)
        ment = np.nanmean(results[p]["entropy"][valid], axis=0)

        axes[0].plot(ks, mr2, "o-", color=c, label=f"{p:.0%} holdout",
                     markersize=4, linewidth=1.5)
        axes[0].fill_between(ks, mr2 - sr2, mr2 + sr2, color=c, alpha=0.15)
        axes[1].semilogy(ks, mrv, "o-", color=c, label=f"{p:.0%} holdout",
                         markersize=4, linewidth=1.5)
        axes[2].plot(ks, ment, "o-", color=c, label=f"{p:.0%} holdout",
                     markersize=4, linewidth=1.5)

        # Random baseline
        if "rand_r2" in results[p]:
            vr = ~np.all(np.isnan(results[p]["rand_r2"]), axis=1)
            if vr.any():
                axes[0].plot(ks, np.nanmean(results[p]["rand_r2"][vr], axis=0),
                             "x--", color=c, markersize=3, linewidth=1.0, alpha=0.6)
                axes[1].semilogy(ks, np.nanmean(results[p]["rand_resid_var"][vr], axis=0),
                                 "x--", color=c, markersize=3, linewidth=1.0, alpha=0.6)
                axes[2].plot(ks, np.nanmean(results[p]["rand_entropy"][vr], axis=0),
                             "x--", color=c, markersize=3, linewidth=1.0, alpha=0.6)

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
    out = OUT / f"greedy_cv_{ds_name}_logit.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Part 3: Entropy vs MI vs Random (10% holdout)
# ══════════════════════════════════════════════════════════════════

def run_entropy_vs_mi(ds_name):
    print(f"\n{'#'*70}")
    print(f"# {DISPLAY[ds_name]}: Entropy vs MI (logit)")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B_raw = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B_raw.shape

    folds = cv_folds(M, N_FOLDS, SEED)
    p = 0.1
    metrics = {"entropy": defaultdict(list), "mi": defaultdict(list),
               "random": defaultdict(list)}

    for fold_idx in range(N_FOLDS):
        t0 = time.time()
        train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)
        B_train_raw, O_train = B_raw[train_idx], O[train_idx]
        B_test_raw, O_test = B_raw[val_idx], O[val_idx]

        B_train_logit, col_max = logit_transform(B_train_raw, O_train)
        B_test_logit, _ = logit_transform(B_test_raw, O_test, col_max=col_max)
        cm_raw, cs_raw = raw_stats(B_train_raw, O_train)

        try:
            R, cm_logit, cs_logit = estimate_corr(B_train_logit, O_train)
        except Exception as e:
            print(f"  Fold {fold_idx}: failed ({e})")
            for method in ["entropy", "mi", "random"]:
                for key in ["r2", "rmse", "resid_var", "mi_val"]:
                    metrics[method][key].append([np.nan] * K_MAX)
            continue

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
                sel = result["selected"][:k]
                fold_rv.append(result["residual_variance"][k - 1])
                imp = impute_and_evaluate_logit(
                    B_test_logit, O_test, sel, R, cm_logit, cs_logit,
                    B_test_raw, cm_raw, cs_raw, col_max, N)
                fold_r2.append(imp["r2"])
                fold_rmse.append(imp["rmse"])

                R_AA = R[np.ix_(sel, sel)]
                s_a, ld_a = np.linalg.slogdet(R_AA)
                ld_a = ld_a if s_a > 0 else -1e30
                comp = sorted(set(range(N)) - set(sel))
                R_CC = R[np.ix_(comp, comp)]
                s_c, ld_c = np.linalg.slogdet(R_CC)
                ld_c = ld_c if s_c > 0 else -1e30
                fold_mi.append(0.5 * (ld_a + ld_c - logdet_full))

            metrics[method]["r2"].append(fold_r2)
            metrics[method]["rmse"].append(fold_rmse)
            metrics[method]["resid_var"].append(fold_rv)
            metrics[method]["mi_val"].append(fold_mi)

        elapsed = time.time() - t0
        print(f"  Fold {fold_idx}: Ent={metrics['entropy']['r2'][-1][4]:.3f}  "
              f"MI={metrics['mi']['r2'][-1][4]:.3f}  "
              f"Rand={metrics['random']['r2'][-1][4]:.3f}  ({elapsed:.1f}s)",
              flush=True)

    for method in ["entropy", "mi", "random"]:
        for key in metrics[method]:
            metrics[method][key] = np.array(metrics[method][key])

    return metrics


def plot_entropy_vs_mi(ds_name, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = np.arange(1, K_MAX + 1)
    colors = {"entropy": "#4477AA", "mi": "#CC3311", "random": "#888888"}
    labels = {"entropy": "Entropy (Alg. 1)", "mi": "Mutual Info (Alg. 2)",
              "random": "Random"}
    styles = {"entropy": "o-", "mi": "o-", "random": "x--"}

    for method in ["entropy", "mi", "random"]:
        c, ls = colors[method], styles[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        valid = ~np.all(np.isnan(metrics[method]["r2"]), axis=1)
        if not valid.any():
            continue
        mr2 = np.nanmean(metrics[method]["r2"][valid], axis=0)
        sr2 = np.nanstd(metrics[method]["r2"][valid], axis=0)
        mrv = np.nanmean(metrics[method]["resid_var"][valid], axis=0)

        axes[0].plot(ks, mr2, ls, color=c, label=labels[method],
                     markersize=ms, linewidth=lw)
        axes[0].fill_between(ks, mr2 - sr2, mr2 + sr2, color=c, alpha=0.15)
        axes[1].semilogy(ks, mrv, ls, color=c, label=labels[method],
                         markersize=ms, linewidth=lw)

    for method in ["entropy", "mi", "random"]:
        c, ls = colors[method], styles[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        if "mi_val" in metrics[method]:
            valid = ~np.all(np.isnan(metrics[method]["mi_val"]), axis=1)
            if valid.any():
                mmi = np.nanmean(metrics[method]["mi_val"][valid], axis=0)
                axes[2].plot(ks, mmi, ls, color=c, label=labels[method],
                             markersize=ms, linewidth=lw)

    axes[0].set_ylabel("Imputation $R^2$")
    axes[1].set_ylabel("Residual variance fraction")
    axes[2].set_ylabel("Mutual information")
    for ax in axes:
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xlim(0.5, K_MAX + 0.5)
        ax.set_xticks([1, 3, 5, 7, 10, 13, 15])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUT / f"entropy_vs_mi_{ds_name}_logit.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_t0 = time.time()

    # Eigenspectrum
    plot_eigenspectrum_logit()

    # Per-dataset experiments
    for ds in DATASETS:
        ds_t0 = time.time()

        # Greedy CV
        cv_results = run_greedy_cv(ds)
        plot_greedy_cv(ds, cv_results)
        save = {}
        for p, m in cv_results.items():
            prefix = f"p{int(p*100)}"
            for key, arr in m.items():
                save[f"{prefix}_{key}"] = arr
        np.savez(LOGS / f"greedy_cv_{ds}_logit.npz", **save)

        # Entropy vs MI
        mi_metrics = run_entropy_vs_mi(ds)
        plot_entropy_vs_mi(ds, mi_metrics)
        save2 = {}
        for method in ["entropy", "mi", "random"]:
            for key, arr in mi_metrics[method].items():
                save2[f"{method}_{key}"] = arr
        np.savez(LOGS / f"entropy_vs_mi_{ds}_logit.npz", **save2)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{DISPLAY[ds]} completed in {ds_elapsed:.0f}s", flush=True)

    total = time.time() - total_t0
    print(f"\nAll experiments completed in {total:.0f}s")
