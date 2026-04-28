"""Plot MI-based greedy selection order for MMLU, MTEB, and Merged.

Loads each dataset, runs 10-fold CV with greedy_mi selection, and produces
a selection-order plot (blue dots per fold, red diamonds for mean, frequency
annotation on the right margin).  Uses EM covariance for datasets with
missing data (MTEB, Merged) and pairwise for fully observed (MMLU).
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
OUT = ROOT / "figures"
LOGS = ROOT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))

from greedy_select import greedy_mi
from em_cov import em_cov
from cv_split import cv_folds, cv_train_val

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────────
K_MAX = 15
N_FOLDS = 10
SEED = 42
HOLDOUT_FRAC = 0.1

DATASETS = ["mmlu", "mteb", "merged"]


# ── Covariance estimation ───────────────────────────────────────

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


# ── Run one dataset ─────────────────────────────────────────────

def run_dataset(ds_name):
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    obs_frac = O.sum() / (M * N)
    print(f"{M} models x {N} tasks, {obs_frac*100:.1f}% observed")

    folds = cv_folds(M, N_FOLDS, SEED)
    print(f"Holdout {HOLDOUT_FRAC:.0%}\n")

    all_logs = []

    for fold_idx in range(N_FOLDS):
        t0 = time.time()
        train_idx, val_idx = cv_train_val(folds, fold_idx, HOLDOUT_FRAC, M, SEED)

        B_train, O_train = B[train_idx], O[train_idx]
        M_train = len(train_idx)
        M_test = len(val_idx)
        print(f"  Fold {fold_idx}: {M_train} train / {M_test} test", end="", flush=True)

        try:
            R, col_mean, col_std = estimate_corr(B_train, O_train)
        except Exception as e:
            print(f" -- estimation failed: {e}", flush=True)
            continue

        result = greedy_mi(R, K_MAX, benchmark_names=benchmark_names)

        names = result.get("names", [str(i) for i in result["selected"]])
        top5 = ", ".join(names[:5])
        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)  top-5: {top5}", flush=True)

        all_logs.append({
            "dataset": ds_name,
            "holdout_frac": HOLDOUT_FRAC,
            "fold": fold_idx,
            "selected_idx": result["selected"],
            "selected_names": names,
        })

    return all_logs, benchmark_names


# ── Plotting ────────────────────────────────────────────────────

def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def plot_selection_order(ds_name, all_logs):
    NOT_SELECTED = K_MAX + 1
    n_folds_actual = len(all_logs)
    positions = defaultdict(list)
    for entry in all_logs:
        fold = entry["fold"]
        for pos, name in enumerate(entry["selected_names"]):
            positions[name].append((fold, pos + 1))

    stats = []
    for name, pos_list in positions.items():
        ranks = [p for _, p in pos_list]
        n_missing = n_folds_actual - len(pos_list)
        all_ranks = ranks + [NOT_SELECTED] * n_missing
        stats.append({
            "name": name,
            "count": len(pos_list),
            "mean_rank": np.mean(all_ranks),
            "min_rank": min(ranks),
            "max_rank": max(ranks),
            "positions": pos_list,
        })
    stats.sort(key=lambda s: s["mean_rank"])

    shown = [s for s in stats if s["count"] >= 2][:25]
    n_shown = len(shown)
    names = [s["name"] for s in shown]

    fig_height = max(4, 0.35 * n_shown + 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(8, fig_height))

    for i, s in enumerate(shown):
        ranks = [p for _, p in s["positions"]]
        y_vals = np.full(len(ranks), i)
        ax.scatter(ranks, y_vals, color="#4477AA", alpha=0.5, s=25,
                   zorder=3, edgecolors="none")
        ax.scatter([s["mean_rank"]], [i], color="#CC3311", s=60,
                   zorder=4, edgecolors="white", linewidths=0.5,
                   marker="D")
        ax.text(K_MAX + 0.8, i, f"{s['count']}/{N_FOLDS}",
                fontsize=7, va="center", ha="left", color="#666666")

    ax.set_yticks(range(n_shown))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Selection position $k$", fontsize=10)
    ax.set_xlim(0.3, K_MAX + 2.5)
    ax.set_ylim(-0.7, n_shown - 0.3)
    ax.set_xticks(range(1, K_MAX + 1))
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    ax.scatter([], [], color="#4477AA", alpha=0.5, s=25, label="Per-fold position")
    ax.scatter([], [], color="#CC3311", s=60, marker="D",
               edgecolors="white", linewidths=0.5, label="Mean position")
    ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    out_path = OUT / f"selection_order_mi_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close()

    # Print summary
    print(f"\n  Top-10 by mean rank:")
    for s in stats[:10]:
        print(f"    {s['name']:40s}  count={s['count']:2d}/{N_FOLDS}"
              f"  mean_rank={s['mean_rank']:.1f}")


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_t0 = time.time()
    all_dataset_logs = {}

    for ds_name in DATASETS:
        ds_t0 = time.time()
        logs, bench_names = run_dataset(ds_name)
        plot_selection_order(ds_name, logs)
        all_dataset_logs[ds_name] = logs

        ds_elapsed = time.time() - ds_t0
        print(f"\n{ds_name.upper()} completed in {ds_elapsed:.0f}s\n")

    # Save all selection logs
    combined = []
    for ds_name in DATASETS:
        combined.extend(all_dataset_logs[ds_name])
    json_logs = [{k: convert(v) for k, v in log.items()} for log in combined]
    log_path = LOGS / "greedy_mi_cv_selections_all.json"
    with open(log_path, "w") as f:
        json.dump(json_logs, f, indent=2, default=convert)
    print(f"\nAll selection logs saved to {log_path}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.0f}s")
