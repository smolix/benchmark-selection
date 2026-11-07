"""Plot benchmark selection order: selection stability over the matched splits.

For each dataset, at the 10% holdout (most training data; --holdout), shows
at which position k = 1..K_MAX (= 15; fewer for a dataset whose splits
have fewer picks, figure_k_max) each benchmark is selected across the
matched splits of cv_split.cv_splits: --repeats repetitions of 10-fold CV
(default N_REPEATS = 5, i.e. 50 splits; repetition 0 is the original
seed-42 folds, so --repeats 1 is the historical 10-fold figure).  Only
benchmarks observed for at least MIN_TRAIN_OBS training models could be
selected on a split (cv_split.eligible_mask); where fewer than K_MAX were
eligible a split has fewer picks.

Benchmarks are ordered by mean selection position, where a split that does
not select a benchmark counts as position K_MAX + 1.  Blue dots: dot area
is proportional to the number of splits selecting the benchmark at that
position (DOT_AREA per tenth of the splits, so one fold of 10 is the
historical single dot); red diamonds: mean position; right margin: number
of splits selecting the benchmark at all.  Rows: benchmarks selected in at
least max(2, ceil(MIN_ROW_SHARE * n_splits)) splits, at most MAX_ROWS, by
mean position (for 10 folds, as before: >= 2 folds, 25 rows).

Selection logs (LOGS = path_config.logs_dir()), the first one found with
the method and holdout is used:
  entropy, random:
    greedy_cv_splits_{ds}.json    eval_greedy_all.py (one entry per
                                  holdout, split and method)
    greedy_cv_{ds}.npz            eval_greedy_all.py, p{pct}_{pre}sel
    {ds}_results.npz              eval_benchpress.py, cv_p{pct}_{pre}sel
    entropy_vs_mi_{ds}.npz        eval_entropy_vs_mi.py, {method}_selected
                                  (its holdout only; also {ds}_results.npz
                                  emi_{method}_selected)
    greedy_cv_selections.json     historical rep-0 entropy log (10 folds)
  mi:
    greedy_mi_cv_splits_{ds}.json plot_mi_selection_all.py
    greedy_mi_cv_{ds}.npz         plot_mi_selection_all.py, p{pct}_sel
    entropy_vs_mi_{ds}.npz, {ds}_results.npz (emi_mi_selected)
    greedy_mi_cv_selections_all.json  historical rep-0 MI log
  facloc (facility location, the headline method since 2026-10-19) and
  trace (greedy trace): the sources of entropy and random (their entries
  / keys trace_ and facloc_, which eval_greedy_all.py writes since
  2026-10-19), then
    baselines_{ds}.npz            eval_baselines.py, {method}_selected with
                                  split_rep / split_fold / split_id /
                                  n_eligible (its holdout only)
  hub and the facility-location similarity variants facloc_abs, facloc_sq:
    baselines_{ds}.npz            only
  (BASELINE_METHODS: the methods with a baselines_{ds}.npz source; all are
  nested, so position k is the k-th pick; pca's per-k sets are not, so it
  has no selection order).  Every source is a selection on the same
  matched splits with the same estimator and eligibility mask, so the
  first one found decides only which file is read.
Splits without any selection (failed covariance estimate, no eligible
benchmark) are left out and reported.

Figures (OUT = path_config.figures_dir()), {sfx} = "" at the 10% holdout,
else _p{pct}:
  selection_order_{ds}{sfx}.pdf           entropy (the paper's figure)
  selection_order_mi_{ds}{sfx}.pdf        mi (plot_mi_selection_all.py
                                          writes the same file)
  selection_order_{method}_{ds}{sfx}.pdf  any other method (e.g.
                                          selection_order_facloc_{ds}.pdf)

Usage:
    python plot_selection_order.py [--datasets mmlu mteb merged]
                                   [--repeats 5] [--holdout 0.1]
                                   [--methods entropy [random mi facloc
                                    trace ...]]

--datasets accepts any name (its logs must exist).  Exits with status 1 if
a requested (dataset, method) has no selection log.
"""

import argparse
import json
import math
import pathlib
import sys
from collections import Counter, OrderedDict

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from path_config import figures_dir, logs_dir
from cv_split import N_REPEATS

LOGS = logs_dir()
OUT = figures_dir()
OUT.mkdir(parents=True, exist_ok=True)

K_MAX = 15
HOLDOUT_FRAC = 0.1
DATASETS = ["mmlu", "mteb", "merged"]

DOT_AREA = 25.0        # marker area (pt²) of one tenth of the splits
MIN_ROW_SHARE = 0.2    # rows: picked in >= max(2, 20% of the) splits
MAX_ROWS = 25

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "mteb": "MTEB",
    "merged": "Merged",
    "benchpress": "BenchPress",
    # data release (data/build_*.py)
    "olv2sub": "OLL v2 sub-tasks",
    "olv1items": "OLL v1 tasks",
    "epoch": "Epoch AI",
    "helm_lite": "HELM Lite",
    "helm_mmlu": "HELM MMLU",
    "helm_classic": "HELM Classic",
    "mteb2026": "MTEB 2026",
}

# Key prefix of a method's per-split arrays in greedy_cv_{ds}.npz /
# {ds}_results.npz (cv_ ...), as written by eval_greedy_all.py (greedy
# trace and facility location joined it on 2026-10-19)
METHOD_PREFIX = {"entropy": "", "random": "rand_", "trace": "trace_",
                 "facloc": "facloc_"}
# Methods with eval_baselines.py's log as a source (the last one; the
# only one of hub, facloc_abs and facloc_sq)
BASELINE_METHODS = ("facloc", "trace", "hub", "facloc_abs", "facloc_sq")
# Split metadata keys of baselines_{ds}.npz (_from_npz meta_keys)
BASELINES_META = {"rep": "split_rep", "fold": "split_fold",
                  "split_id": "split_id", "n_eligible": "n_eligible"}
# (entropy_rf was dropped on 2026-10-18, protocol rule D3)
METHODS = ["entropy", "random", "mi", "facloc", "trace", "hub",
           "facloc_abs", "facloc_sq"]


def display_name(ds_name):
    """Display name of a dataset (the raw name if it has none)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def figure_path(ds_name, method="entropy", holdout=HOLDOUT_FRAC):
    """Output path of a selection-order figure (historical names at the
    10% holdout: selection_order_{ds}.pdf, selection_order_mi_{ds}.pdf)."""
    tag = "" if method == "entropy" else f"{method}_"
    sfx = "" if _same_p(holdout, HOLDOUT_FRAC) else f"_p{int(holdout * 100)}"
    return OUT / f"selection_order_{tag}{ds_name}{sfx}.pdf"


def _same_p(a, b):
    return abs(float(a) - float(b)) < 1e-9


# ── Selection logs ───────────────────────────────────────────────

def _entry(rep, fold, split_id, n_eligible, selected_idx, selected_names,
           n_index_ties=None):
    """One split's selection, the common format of every log source."""
    return {"rep": int(rep), "fold": int(fold), "split_id": int(split_id),
            "n_eligible": None if n_eligible is None else int(n_eligible),
            "selected_idx": [int(j) for j in selected_idx],
            "selected_names": [str(n) for n in selected_names],
            "n_index_ties": n_index_ties}


def _index_ties_json(e):
    """Picks among the first K_MAX decided by the lowest index, from a
    split log's index_ties list of (step, n_tied); None if not logged."""
    ties = e.get("index_ties")
    if ties is None:
        return None
    return sum(1 for step, _ in ties if step < K_MAX)


def _from_split_json(path, ds_name, method, holdout):
    """Entries of a per-split log (greedy_cv_splits_{ds}.json,
    greedy_mi_cv_splits_{ds}.json): one entry per (holdout, split,
    method)."""
    out = []
    for e in json.loads(path.read_text()):
        if (e.get("method") != method or e.get("dataset", ds_name) != ds_name
                or not _same_p(e["holdout_frac"], holdout)):
            continue
        out.append(_entry(e["rep"], e["fold"], e["split_id"],
                          e.get("n_eligible"), e["selected_idx"],
                          e["selected_names"], _index_ties_json(e)))
    return out


def _from_legacy_json(path, ds_name, holdout):
    """Entries of a historical per-fold log (greedy_cv_selections.json,
    greedy_mi_cv_selections_all.json): rep 0 only; entries without rep /
    split_id predate the matched design and are rep 0, split_id = fold."""
    out = []
    for e in json.loads(path.read_text()):
        if e["dataset"] != ds_name or not _same_p(e["holdout_frac"], holdout):
            continue
        out.append(_entry(e.get("rep", 0), e["fold"],
                          e.get("split_id", e["fold"]), e.get("n_eligible"),
                          e["selected_idx"], e["selected_names"],
                          _index_ties_json(e)))
    return out


def _from_npz(path, sel_key, meta, ties_key=None, holdout_key=None,
              holdout=None, meta_keys=None):
    """Entries of a per-split npz log: sel_key (n_splits, k) int, -1
    padded; split metadata {meta}rep, {meta}fold, {meta}split_id,
    {meta}n_eligible, or the keys meta_keys maps these four names to
    (BASELINES_META: split_rep, ...).  With holdout_key, the file must be
    at `holdout`.  None if the file lacks these keys (e.g. a
    pre-matched-design log)."""
    mk = (meta_keys if meta_keys is not None else
          {k: meta + k for k in ("rep", "fold", "split_id", "n_eligible")})
    d = np.load(path, allow_pickle=True)
    need = [sel_key] + [mk[k] for k in ("rep", "fold", "split_id")]
    if any(k not in d.files for k in need) or "benchmark_names" not in d.files:
        return None
    if holdout_key is not None and (holdout_key not in d.files
                                    or not _same_p(d[holdout_key], holdout)):
        return None
    names = d["benchmark_names"]
    sel = d[sel_key]
    n_el = d[mk["n_eligible"]] if mk["n_eligible"] in d.files else None
    ties = d[ties_key] if ties_key and ties_key in d.files else None
    out = []
    for i in range(sel.shape[0]):
        idx = [int(j) for j in sel[i, :K_MAX] if j >= 0]
        out.append(_entry(d[mk["rep"]][i], d[mk["fold"]][i],
                          d[mk["split_id"]][i],
                          None if n_el is None else n_el[i], idx,
                          [names[j] for j in idx],
                          None if ties is None
                          else int(ties[i, min(K_MAX, ties.shape[1]) - 1])))
    return out


def selection_sources(ds_name, method, holdout=HOLDOUT_FRAC):
    """Candidate logs of (dataset, method, holdout), in order of preference:
    list of (path, loader) with loader(path) -> entries or None."""
    pct = int(holdout * 100)
    p_ = f"p{pct}_"
    baselines = (LOGS / f"baselines_{ds_name}.npz",
                 lambda f: _from_npz(f, f"{method}_selected", "",
                                     f"{method}_index_ties", "holdout_frac",
                                     holdout, BASELINES_META))
    if method in BASELINE_METHODS and method not in METHOD_PREFIX:
        return [baselines]
    sources = []
    if method == "mi":
        sources.append((LOGS / f"greedy_mi_cv_splits_{ds_name}.json",
                        lambda f: _from_split_json(f, ds_name, "mi", holdout)))
        sources.append((LOGS / f"greedy_mi_cv_{ds_name}.npz",
                        lambda f: _from_npz(f, p_ + "sel", p_,
                                            p_ + "index_ties")))
    else:
        pre = METHOD_PREFIX.get(method)
        sources.append((LOGS / f"greedy_cv_splits_{ds_name}.json",
                        lambda f: _from_split_json(f, ds_name, method,
                                                   holdout)))
        if pre is not None:
            sources.append((LOGS / f"greedy_cv_{ds_name}.npz",
                            lambda f: _from_npz(f, f"{p_}{pre}sel", p_,
                                                f"{p_}{pre}index_ties")))
            sources.append((LOGS / f"{ds_name}_results.npz",
                            lambda f: _from_npz(f, f"cv_{p_}{pre}sel",
                                                "cv_" + p_,
                                                f"cv_{p_}{pre}index_ties")))
    # eval_entropy_vs_mi.py / eval_benchpress.py Part 2 (one holdout)
    sources.append((LOGS / f"entropy_vs_mi_{ds_name}.npz",
                    lambda f: _from_npz(f, f"{method}_selected", "",
                                        f"{method}_index_ties",
                                        "holdout_frac", holdout)))
    sources.append((LOGS / f"{ds_name}_results.npz",
                    lambda f: _from_npz(f, f"emi_{method}_selected", "emi_",
                                        f"emi_{method}_index_ties",
                                        "emi_holdout_frac", holdout)))
    legacy = {"entropy": "greedy_cv_selections.json",
              "mi": "greedy_mi_cv_selections_all.json"}.get(method)
    if legacy:
        sources.append((LOGS / legacy,
                        lambda f: _from_legacy_json(f, ds_name, holdout)))
    if method in BASELINE_METHODS:
        sources.append(baselines)
    return sources


def load_selections(ds_name, method="entropy", holdout=HOLDOUT_FRAC,
                    repeats=N_REPEATS):
    """Selections of the first available log of (dataset, method, holdout),
    restricted to reps < repeats, in split order.

    Returns (entries, path, n_dropped): entries as _entry (splits without
    picks left out, n_dropped of them), path the log read; (None, None, 0)
    if no log has the method at this holdout.
    """
    for path, loader in selection_sources(ds_name, method, holdout):
        if not path.is_file():
            continue
        entries = loader(path)
        if not entries:
            continue
        entries = sorted((e for e in entries if e["rep"] < repeats),
                         key=lambda e: e["split_id"])
        kept = [e for e in entries if e["selected_names"]]
        return kept, path, len(entries) - len(kept)
    return None, None, 0


# ── Selection statistics ─────────────────────────────────────────

def figure_k_max(entries):
    """Positions shown: K_MAX, or fewer when no split has K_MAX picks (a
    dataset with fewer benchmarks, e.g. HELM Lite's 10)."""
    return min(K_MAX, max((len(e["selected_names"]) for e in entries),
                          default=K_MAX))


def selection_stats(entries, k_max=K_MAX):
    """Per-benchmark selection statistics over the splits of `entries`.

    Benchmarks not selected in a split receive position k_max + 1 so that
    the mean rank penalizes infrequent selection.  Returns the list of
    dicts (name, count, mean_rank, min_rank, max_rank, positions =
    [(split_id, position)], position_counts = Counter of positions) sorted
    by mean rank (stable: first appearance in split order breaks ties).
    """
    n_splits = len(entries)
    positions = OrderedDict()  # name -> list of (split_id, position)
    for entry in entries:
        sid = entry.get("split_id", entry.get("fold"))
        for pos, name in enumerate(entry["selected_names"][:k_max], 1):
            positions.setdefault(name, []).append((sid, pos))

    not_selected = k_max + 1
    stats = []
    for name, pos_list in positions.items():
        ranks = [p for _, p in pos_list]
        all_ranks = ranks + [not_selected] * (n_splits - len(pos_list))
        stats.append({
            "name": name,
            "count": len(pos_list),      # how many splits selected this
            "mean_rank": np.mean(all_ranks),
            "min_rank": min(ranks),
            "max_rank": max(ranks),
            "positions": pos_list,       # (split_id, position) pairs
            "position_counts": Counter(ranks),
        })
    stats.sort(key=lambda s: s["mean_rank"])
    return stats


def min_row_count(n_splits):
    """Fewest splits a benchmark must be selected in to get a row."""
    return max(2, math.ceil(MIN_ROW_SHARE * n_splits - 1e-9))


def shown_rows(stats, n_splits):
    """Rows of the figure (stats already sorted by mean rank)."""
    lo = min_row_count(n_splits)
    return [s for s in stats if s["count"] >= lo][:MAX_ROWS]


def _dot_area(count, n_splits):
    return DOT_AREA * count * 10.0 / max(n_splits, 1)


def plot_selection(stats, n_splits, ax, k_max=K_MAX):
    """Plot the selection order of one (dataset, method) into ax."""
    shown = shown_rows(stats, n_splits)
    n_shown = len(shown)
    names = [s["name"] for s in shown]

    for i, s in enumerate(shown):
        pos = sorted(s["position_counts"])
        counts = np.array([s["position_counts"][p] for p in pos])
        ax.scatter(pos, np.full(len(pos), i), color="#4477AA", alpha=0.5,
                   s=_dot_area(counts, n_splits), zorder=3,
                   edgecolors="none")
        # Mean rank as a larger marker
        ax.scatter([s["mean_rank"]], [i], color="#CC3311", s=60,
                   zorder=4, edgecolors="white", linewidths=0.5,
                   marker="D")
        # Frequency annotation
        ax.text(k_max + 0.8, i, f"{s['count']}/{n_splits}",
                fontsize=7, va="center", ha="left", color="#666666")

    ax.set_yticks(range(n_shown))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Selection position $k$", fontsize=10)
    ax.set_xlim(0.3, k_max + 2.5)
    ax.set_ylim(-0.7, n_shown - 0.3)
    ax.set_xticks(range(1, k_max + 1))
    ax.invert_yaxis()  # best (earliest) at top
    ax.grid(True, axis="x", alpha=0.3)
    # Title omitted — provided by the figure caption in the paper.

    unit = "fold" if n_splits <= 10 else "split"
    ax.scatter([], [], color="#4477AA", alpha=0.5, s=_dot_area(1, n_splits),
               label=f"Per-{unit} position (area ∝ {unit}s)")
    ax.scatter([], [], color="#CC3311", s=60, marker="D",
               edgecolors="white", linewidths=0.5, label="Mean position")
    ax.legend(fontsize=7, loc="lower left")


def save_selection_figure(stats, n_splits, out_path, k_max=K_MAX):
    """One selection-order figure (sized by its number of rows)."""
    n_shown = len(shown_rows(stats, n_splits))
    fig_height = max(4, 0.35 * n_shown + 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(8, fig_height))
    plot_selection(stats, n_splits, ax, k_max)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def describe_splits(entries, n_dropped=0):
    """One-line summary of the splits behind a figure: count, reps,
    eligible-set sizes and lowest-index tie decisions where logged."""
    reps = sorted({e["rep"] for e in entries})
    parts = [f"{len(entries)} splits (reps {', '.join(map(str, reps))})"]
    n_el = [e["n_eligible"] for e in entries if e["n_eligible"] is not None]
    if n_el:
        parts.append(f"eligible {min(n_el)}–{max(n_el)}")
    ties = [e["n_index_ties"] for e in entries
            if e["n_index_ties"] is not None]
    if ties:
        parts.append(f"index-decided picks (k <= {K_MAX}) {sum(ties)} "
                     f"in {sum(t > 0 for t in ties)} splits")
    if n_dropped:
        parts.append(f"{n_dropped} split(s) without picks left out")
    return ", ".join(parts)


def print_top(stats, n_splits, n_top=10):
    print(f"  Top-{n_top} by mean rank:")
    for s in stats[:n_top]:
        print(f"    {s['name']:40s}  count={s['count']:2d}/{n_splits}"
              f"  mean_rank={s['mean_rank']:.1f}")


# ── Main ─────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Selection-order figures over the matched splits "
                    "(see module docstring).")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names; default: %(default)s")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="use CV repetitions 0..repeats-1 of the logs "
                             "(1: the historical 10 folds); default: "
                             "%(default)s")
    parser.add_argument("--holdout", type=float, default=HOLDOUT_FRAC,
                        help="holdout fraction; default: %(default)s")
    parser.add_argument("--methods", nargs="+", default=["entropy"],
                        choices=METHODS,
                        help="selection methods; default: %(default)s")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    missing = []
    for ds_name in args.datasets:
        for method in args.methods:
            entries, path, n_dropped = load_selections(
                ds_name, method, args.holdout, args.repeats)
            if not entries:
                print(f"\n{display_name(ds_name)} ({method}): no selection "
                      f"log at holdout {args.holdout:g}", file=sys.stderr)
                missing.append((ds_name, method))
                continue
            n_splits = len(entries)
            n_reps = len({e["rep"] for e in entries})
            print(f"\n{display_name(ds_name)} ({method}, holdout "
                  f"{args.holdout:.0%}): {path.name}")
            print("  " + describe_splits(entries, n_dropped))
            if n_reps < args.repeats:
                print(f"  note: {n_reps} of {args.repeats} requested "
                      f"repetitions in the log")
            k_max = figure_k_max(entries)
            stats = selection_stats(entries, k_max)
            save_selection_figure(stats, n_splits,
                                  figure_path(ds_name, method, args.holdout),
                                  k_max)
            print_top(stats, n_splits)

    print("\nDone.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
