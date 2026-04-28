"""Plot benchmark selection order for the 10% holdout (most training data).

For each dataset, shows which benchmarks are selected and at which position
across the 10 CV folds. Benchmarks ordered by mean selection rank.
"""

import json
import pathlib
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

K_MAX = 15
N_FOLDS = 10

with open(LOGS / "greedy_cv_selections.json") as f:
    all_logs = json.load(f)

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "mteb": "MTEB",
    "merged": "Merged",
}


def get_selection_stats(ds_name, holdout=0.1):
    """Get selection frequency and positions for each benchmark.

    Benchmarks not selected in a fold receive position K_MAX + 1 so that
    the mean rank penalizes infrequent selection.
    """
    entries = [l for l in all_logs
               if l["dataset"] == ds_name and l["holdout_frac"] == holdout]
    n_folds_actual = len(entries)

    # For each benchmark, collect the positions at which it was selected
    positions = defaultdict(list)  # name -> list of (fold, position)
    for entry in entries:
        fold = entry["fold"]
        for pos, name in enumerate(entry["selected_names"]):
            positions[name].append((fold, pos + 1))  # 1-indexed position

    # Compute stats; assign K_MAX+1 for folds where not selected
    NOT_SELECTED = K_MAX + 1
    stats = []
    for name, pos_list in positions.items():
        ranks = [p for _, p in pos_list]
        n_missing = n_folds_actual - len(pos_list)
        all_ranks = ranks + [NOT_SELECTED] * n_missing
        stats.append({
            "name": name,
            "count": len(pos_list),      # how many folds selected this
            "mean_rank": np.mean(all_ranks),
            "min_rank": min(ranks),
            "max_rank": max(ranks),
            "positions": pos_list,        # (fold, position) pairs
        })

    stats.sort(key=lambda s: s["mean_rank"])
    return stats


def plot_selection(ds_name, ax):
    """Plot selection order for one dataset."""
    stats = get_selection_stats(ds_name)

    # Show top benchmarks (those selected in at least 2 folds, or top 20)
    # Already sorted by mean_rank (with K_MAX+1 penalty for missing folds)
    shown = [s for s in stats if s["count"] >= 2][:25]

    n_shown = len(shown)
    names = [s["name"] for s in shown]

    # Plot individual fold selections as dots
    for i, s in enumerate(shown):
        ranks = [p for _, p in s["positions"]]
        # Jitter y slightly for overlapping dots
        y_vals = np.full(len(ranks), i)
        ax.scatter(ranks, y_vals, color="#4477AA", alpha=0.5, s=25,
                   zorder=3, edgecolors="none")
        # Mean rank as a larger marker
        ax.scatter([s["mean_rank"]], [i], color="#CC3311", s=60,
                   zorder=4, edgecolors="white", linewidths=0.5,
                   marker="D")
        # Frequency annotation
        ax.text(K_MAX + 0.8, i, f"{s['count']}/{N_FOLDS}",
                fontsize=7, va="center", ha="left", color="#666666")

    ax.set_yticks(range(n_shown))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Selection position $k$", fontsize=10)
    ax.set_xlim(0.3, K_MAX + 2.5)
    ax.set_ylim(-0.7, n_shown - 0.3)
    ax.set_xticks(range(1, K_MAX + 1))
    ax.invert_yaxis()  # best (earliest) at top
    ax.grid(True, axis="x", alpha=0.3)
    # Title omitted — provided by the figure caption in the paper.

    # Add legend
    ax.scatter([], [], color="#4477AA", alpha=0.5, s=25, label="Per-fold position")
    ax.scatter([], [], color="#CC3311", s=60, marker="D",
               edgecolors="white", linewidths=0.5, label="Mean position")
    ax.legend(fontsize=7, loc="lower left")


# Create one figure per dataset
for ds_name in ["mmlu", "mteb", "merged"]:
    stats = get_selection_stats(ds_name)
    shown = [s for s in stats if s["count"] >= 2][:25]  # already sorted
    n_shown = len(shown)

    fig_height = max(4, 0.35 * n_shown + 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(8, fig_height))
    plot_selection(ds_name, ax)
    plt.tight_layout()

    out_path = OUT / f"selection_order_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved {out_path}")
    plt.close()

print("\nDone.")
