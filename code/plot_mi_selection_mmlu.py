"""Plot MI-based greedy selection order for MMLU (10% holdout, 10-fold CV).

Loads MMLU data, runs 10-fold CV with greedy_mi selection, and produces
a selection-order plot (blue dots per fold, red diamonds for mean, frequency
annotation on the right margin).  Saves the plot and selection logs.
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

from greedy_select import greedy_mi
from cv_split import cv_folds, cv_train_val

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────────
K_MAX = 15
N_FOLDS = 10
SEED = 42
HOLDOUT_FRAC = 0.1


# ── Covariance estimation (pairwise-complete) ──────────────────
def estimate_corr_pairwise(B_train, O_train):
    """Estimate correlation matrix via pairwise-complete covariance."""
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


# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    total_t0 = time.time()

    # Load data
    d = np.load(DATA / "mmlu.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    obs_frac = O.sum() / (M * N)
    print(f"MMLU: {M} models x {N} tasks, {obs_frac*100:.1f}% observed")

    # Balanced k-fold split
    folds = cv_folds(M, N_FOLDS, SEED)
    print(f"Holdout {HOLDOUT_FRAC:.0%}\n")

    # ── Run CV ──────────────────────────────────────────────────
    all_logs = []

    for fold_idx in range(N_FOLDS):
        t0 = time.time()
        train_idx, val_idx = cv_train_val(folds, fold_idx, HOLDOUT_FRAC, M, SEED)

        B_train, O_train = B[train_idx], O[train_idx]
        M_train = len(train_idx)
        M_test = len(val_idx)
        print(f"  Fold {fold_idx}: {M_train} train / {M_test} test", end="", flush=True)

        # Estimate correlation
        R, col_mean, col_std = estimate_corr_pairwise(B_train, O_train)

        # Run greedy MI
        result = greedy_mi(R, K_MAX, benchmark_names=benchmark_names)

        names = result.get("names", [str(i) for i in result["selected"]])
        top5 = ", ".join(names[:5])
        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)  top-5: {top5}", flush=True)

        all_logs.append({
            "dataset": "mmlu",
            "holdout_frac": HOLDOUT_FRAC,
            "fold": fold_idx,
            "selected_idx": result["selected"],
            "selected_names": names,
        })

    # ── Save selection logs ─────────────────────────────────────
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_logs = [{k: convert(v) for k, v in log.items()} for log in all_logs]
    log_path = LOGS / "greedy_mi_cv_selections.json"
    with open(log_path, "w") as f:
        json.dump(json_logs, f, indent=2, default=convert)
    print(f"\nSelection logs saved to {log_path}")

    # ── Build selection statistics ──────────────────────────────
    NOT_SELECTED = K_MAX + 1
    n_folds_actual = len(json_logs)
    positions = defaultdict(list)  # name -> [(fold, 1-indexed position)]
    for entry in json_logs:
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

    # ── Plot ────────────────────────────────────────────────────
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
    # Title omitted — provided by the figure caption in the paper.

    ax.scatter([], [], color="#4477AA", alpha=0.5, s=25, label="Per-fold position")
    ax.scatter([], [], color="#CC3311", s=60, marker="D",
               edgecolors="white", linewidths=0.5, label="Mean position")
    ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    out_path = OUT / "selection_order_mi_mmlu.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close()

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.1f}s")
