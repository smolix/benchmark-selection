"""Plot MI-based greedy selection order over the matched splits.

Loads each dataset (any data/<name>.matrix.npz; default MMLU, MTEB,
Merged), runs greedy_mi on every matched split of cv_split.cv_splits at the
10% holdout (--holdouts): --repeats repetitions of 10-fold CV (default
N_REPEATS = 5, i.e. 50 splits; repetition 0 is the original seed-42 folds
and training subsamples, so its rows do not depend on --repeats and
--repeats 1 is the historical run).  On every split only benchmarks
observed for at least MIN_TRAIN_OBS training models can be selected
(cv_split.eligible_mask, the mask every method of the split shares);
ineligible benchmarks stay in the correlation estimate and in the MI
complement.  Near-ties of the MI gains (greedy_select.TIE_RTOL) are broken
by explained variance, then by the lowest index; index_ties counts the
picks the lowest index decided.  Uses EM covariance for datasets with
missing data (MTEB, Merged) and pairwise for fully observed (MMLU), the
estimator of eval_greedy_all.py and eval_entropy_vs_mi.py, so the picks
equal eval_entropy_vs_mi.py's mi_selected split by split.  Up to k_max =
min(K_MAX, N) picks (K_MAX = 15; HELM Lite has N = 10 benchmarks).

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir()):

  OUT/selection_order_mi_{ds}.pdf  selection-order plot over all splits
      (plot_selection_order.py: dot area ∝ number of splits picking a
      benchmark at a position, red diamonds = mean position, right margin
      = splits picking it); other holdouts: selection_order_mi_{ds}_p{pct}.pdf
  LOGS/greedy_mi_cv_{ds}.npz  per holdout p, keys prefixed p{100p}_ (p10_);
      rows are the splits in cv_splits order (rep-major, folds in order:
      the first 10 rows are rep 0, the historical folds), the layout of
      eval_greedy_all.py's greedy_cv_{ds}.npz:
        rep, fold, split_id, n_train, n_test, n_eligible   (n_splits,) int
        eligible     (n_splits, N) bool, the split's selectable benchmarks
        failed       (n_splits,) bool, covariance estimation failed (the
                     split's rows are NaN / -1)
        sel          (n_splits, k_max) int, MI selection order, -1 padded
        index_ties   (n_splits, k_max) int, picks among the first k
                     decided by the lowest column index
                     (greedy_select.index_tie_count(result, k))
        tie_breaks   (n_splits, k_max) int, picks among the first k made
                     among near-tied MI gains
                     (greedy_select.tie_break_count(result, k); the layout
                     of index_ties, as in every log)
        mi_val       (n_splits, k_max) float, I(X_A; X_rest) after k picks
        gain         (n_splits, k_max) float, MI gain of the k-th pick
        resid_var    (n_splits, k_max) float, residual-variance fraction
        log_det      (n_splits, k_max) float, log det R_AA (Cholesky)
                     (NaN beyond the last eligible pick)
      plus benchmark_names, dataset, holdout_fracs, methods (["mi"]),
      method_prefixes ([""]), n_folds, n_repeats, seed, min_train_obs,
      k_max.  Pair with other methods row by row (same split_id).
  LOGS/greedy_mi_cv_splits_{ds}.json  one entry per (holdout, split):
      dataset, holdout_frac, rep, fold, split_id, n_train, n_test,
      n_eligible, ineligible_idx, method ("mi"), failed, selected_idx,
      selected_names, tie_breaks, index_ties (the entry format of
      eval_greedy_all.py's greedy_cv_splits_{ds}.json).
  LOGS/greedy_mi_cv_selections_all.json  (--selections-json) rep-0 MI
      selection order per (dataset, holdout, fold) of this run's datasets
      in the historical format (dataset, holdout_frac, fold, selected_idx,
      selected_names; failed folds left out, as before), with rep,
      split_id, n_eligible, tie_breaks and index_ties added
      (make_numbers.py).

Usage:
    python plot_mi_selection_all.py [--datasets mmlu mteb merged]
                                    [--repeats 5] [--holdouts 0.1]
                                    [--workers 1]
                                    [--selections-json greedy_mi_cv_selections_all.json]
                                    [--selections-from-logs]

--datasets accepts any name with a data/<name>.matrix.npz.  Datasets share
no state, so per-dataset runs reproduce a joint run: run them with
--selections-json none, then

    python plot_mi_selection_all.py --datasets mmlu mteb merged --selections-from-logs

writes greedy_mi_cv_selections_all.json from their greedy_mi_cv_splits_{ds}.json,
identical to the joint run's.

--workers W evaluates the splits of all holdouts in W spawned
single-threaded processes (eval_entropy_vs_mi.map_splits): every split is a
pure function of the data and the split, so the logs do not depend on W,
and the per-split progress lines are printed in split order (stdout
differs from the serial run's in the timings only).  The default W = 1
runs the serial loop, as before.
"""

import argparse
import pathlib
import sys
import time
import json
import numpy as np
from collections import defaultdict
from contextlib import closing

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import greedy_mi, index_tie_count, tie_break_count
from em_cov import em_cov
from cv_split import cv_splits, eligible_mask, MIN_TRAIN_OBS, N_REPEATS
from eval_entropy_vs_mi import map_splits, say_now
from plot_selection_order import (display_name, figure_path, figure_k_max,
                                  selection_stats, save_selection_figure,
                                  describe_splits, print_top)

# ── Parameters ──────────────────────────────────────────────────
K_MAX = 15
N_FOLDS = 10
SEED = 42
HOLDOUT_FRAC = 0.1

DATASETS = ["mmlu", "mteb", "merged"]
METHOD = "mi"
# Per-split float arrays: npz key -> greedy_mi result key
CURVES = {"mi_val": "mi_values", "gain": "gains",
          "resid_var": "residual_variance", "log_det": "log_dets"}


# ── Covariance estimation (same as eval_greedy_all.py) ───────────

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


def estimate_corr(B_train, O_train):
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)
    if obs_frac < 0.9 or M_tr < N_tr:
        return estimate_corr_em(B_train, O_train)
    else:
        return estimate_corr_pairwise(B_train, O_train)


# ── Run one dataset ─────────────────────────────────────────────

def _pad_index(selected, k_max=K_MAX):
    """Selection order as a length-k_max int list, -1 padded."""
    sel = [int(j) for j in selected][:k_max]
    return sel + [-1] * (k_max - len(sel))


def mi_order_split(ctx, s, say=say_now):
    """One matched split of run_dataset at holdout s.holdout_frac (its loop
    body): {"rows": [(npz key suffix, the split's row)] in the order the
    keys enter results[p], "log": the split's selection entry}; progress
    via say (eval_entropy_vs_mi.map_splits)."""
    t0 = time.time()
    B, O, N, k_max = ctx["B"], ctx["O"], ctx["N"], ctx["k_max"]
    p = s.holdout_frac
    B_train, O_train = B[s.train_idx], O[s.train_idx]
    M_train = len(s.train_idx)
    M_test = len(s.val_idx)

    # One eligibility mask per split, shared by every method
    el = eligible_mask(O_train)
    n_el = int(el.sum())
    rows = [("rep", s.rep), ("fold", s.fold), ("split_id", s.split_id),
            ("n_train", M_train), ("n_test", M_test), ("n_eligible", n_el),
            ("eligible", el)]
    split_info = {
        "dataset": ctx["ds_name"], "holdout_frac": p, "rep": s.rep,
        "fold": s.fold, "split_id": s.split_id, "n_train": M_train,
        "n_test": M_test, "n_eligible": n_el,
        "ineligible_idx": np.flatnonzero(~el).tolist(),
        "method": METHOD,
    }
    say(f"  Rep {s.rep} fold {s.fold}: {M_train} train / "
        f"{M_test} test, {n_el}/{N} eligible", end="")

    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:
        say(f" -- estimation failed: {e}")
        rows.append(("failed", True))
        for key in CURVES:
            rows.append((key, [np.nan] * k_max))
        rows.append(("sel", [-1] * k_max))
        rows.append(("index_ties", [0] * k_max))
        rows.append(("tie_breaks", [0] * k_max))
        return {"rows": rows,
                "log": {**split_info, "failed": True, "selected_idx": [],
                        "selected_names": [], "tie_breaks": [],
                        "index_ties": []}}
    rows.append(("failed", False))

    result = greedy_mi(R, k_max, benchmark_names=ctx["names"], eligible=el)

    for key, res_key in CURVES.items():
        rows.append((key, result[res_key]))
    rows.append(("sel", _pad_index(result["selected"], k_max)))
    rows.append(("index_ties",
                 [index_tie_count(result, k) for k in range(1, k_max + 1)]))
    rows.append(("tie_breaks",
                 [tie_break_count(result, k) for k in range(1, k_max + 1)]))
    names = result.get("names", [str(i) for i in result["selected"]])
    log = {**split_info, "failed": False,
           "selected_idx": result["selected"], "selected_names": names,
           "tie_breaks": result["tie_breaks"],
           "index_ties": result["index_ties"]}

    top5 = ", ".join(names[:5])
    ties = (f"  [index ties {len(result['index_ties'])}]"
            if result["index_ties"] else "")
    elapsed = time.time() - t0
    say(f"  ({elapsed:.1f}s)  top-5: {top5}{ties}")
    return {"rows": rows, "log": log}


def run_dataset(ds_name, n_repeats=N_REPEATS, holdout_fracs=(HOLDOUT_FRAC,),
                workers=1):
    """greedy_mi on every matched split of one dataset.

    Returns (results, split_logs, benchmark_names): results[p] maps the npz
    key suffixes (module docstring) to arrays over the splits of holdout p;
    split_logs has one selection entry per (holdout, split).  workers > 1
    evaluates the splits of all holdouts in that many processes
    (eval_entropy_vs_mi.map_splits; the arrays, logs and progress lines
    are those of the serial run).
    """
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
    print(f"{n_repeats} x {N_FOLDS}-fold matched splits; selectable: "
          f">= {MIN_TRAIN_OBS} observed training models")
    # At most N picks (helm_lite: N = 10)
    k_max = min(K_MAX, N)

    results = {}
    split_logs = []

    ctx = dict(B=B, O=O, N=N, k_max=k_max, names=benchmark_names,
               ds_name=ds_name)
    holdout_fracs = list(holdout_fracs)     # iterated twice below
    splits = {p: list(cv_splits(M, N_FOLDS, n_repeats, p, SEED))
              for p in holdout_fracs}
    # One pass over the splits of every holdout (the serial path evaluates
    # each split when its row is taken, after its holdout's header)
    parts = map_splits(mi_order_split, ctx,
                       [s for p in holdout_fracs for s in splits[p]], workers)
    with closing(parts):
        for p in holdout_fracs:
            print(f"Holdout {p:.0%}\n")
            metrics = defaultdict(list)

            # zip takes the next split first: exactly one part per split
            for _, part in zip(splits[p], parts):
                for key, val in part["rows"]:
                    metrics[key].append(val)
                split_logs.append(part["log"])

            results[p] = {k: np.array(v) for k, v in metrics.items()}

    return results, split_logs, benchmark_names


# ── Output ──────────────────────────────────────────────────────

def convert(obj):
    """numpy scalars/arrays and tuples to JSON-native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (tuple, list)):
        return [convert(v) for v in obj]
    return obj


def legacy_selection(entry):
    """Historical greedy_mi_cv_selections_all.json entry of a rep-0 split
    (first five keys as before, then the matched-split metadata)."""
    keys = ["dataset", "holdout_frac", "fold", "selected_idx",
            "selected_names", "rep", "split_id", "n_eligible", "tie_breaks",
            "index_ties"]
    return {k: entry[k] for k in keys}


def write_selections(path, split_logs):
    """Write the historical selection log: the rep-0 entries of split_logs
    (in their order) that have a selection, in legacy_selection format."""
    json_logs = [legacy_selection({k: convert(v) for k, v in e.items()})
                 for e in split_logs
                 if e["rep"] == 0 and not e.get("failed", False)]
    with open(path, "w") as f:
        json.dump(json_logs, f, indent=2, default=convert)
    print(f"\nAll selection logs saved to {path}", flush=True)


def save_dataset(ds_name, results, split_logs, bench_names, n_repeats):
    """Write greedy_mi_cv_{ds}.npz and greedy_mi_cv_splits_{ds}.json."""
    save_dict = {}
    for p, metrics in results.items():
        prefix = f"p{int(p*100)}"
        for metric_name, arr in metrics.items():
            save_dict[f"{prefix}_{metric_name}"] = arr
    save_dict["benchmark_names"] = bench_names
    save_dict["dataset"] = np.array(ds_name)
    save_dict["holdout_fracs"] = np.array(list(results))
    save_dict["methods"] = np.array([METHOD])
    save_dict["method_prefixes"] = np.array([""])
    save_dict["n_folds"] = np.array(N_FOLDS)
    save_dict["n_repeats"] = np.array(n_repeats)
    save_dict["seed"] = np.array(SEED)
    save_dict["min_train_obs"] = np.array(MIN_TRAIN_OBS)
    save_dict["k_max"] = np.array(            # min(K_MAX, N)
        next(iter(results.values()))["sel"].shape[1])
    np.savez(LOGS / f"greedy_mi_cv_{ds_name}.npz", **save_dict)

    with open(LOGS / f"greedy_mi_cv_splits_{ds_name}.json", "w") as f:
        json.dump([{k: convert(v) for k, v in e.items()} for e in split_logs],
                  f, indent=1, default=convert)


def plot_dataset(ds_name, split_logs):
    """selection_order_mi_{ds}.pdf (per holdout) over all splits with a
    selection, via plot_selection_order.py."""
    for p in sorted({e["holdout_frac"] for e in split_logs}):
        entries = [{**e, "n_index_ties": sum(1 for step, _ in e["index_ties"]
                                             if step < K_MAX)}
                   for e in split_logs
                   if e["holdout_frac"] == p and e["selected_names"]]
        n_dropped = sum(1 for e in split_logs
                        if e["holdout_frac"] == p and not e["selected_names"])
        if not entries:
            print(f"  {display_name(ds_name)}: no split with a selection at "
                  f"holdout {p:.0%}, no figure", flush=True)
            continue
        k_max = figure_k_max(entries)
        stats = selection_stats(entries, k_max)
        print(f"\n  {display_name(ds_name)} MI, holdout {p:.0%}: "
              + describe_splits(entries, n_dropped))
        save_selection_figure(stats, len(entries),
                              figure_path(ds_name, METHOD, p), k_max)
        print_top(stats, len(entries))


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Greedy MI selection order over the matched splits "
                    "(see module docstring).")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names (data/<name>.matrix.npz); "
                             "default: %(default)s")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions (rep 0 = the historical folds); "
                             "default: %(default)s")
    parser.add_argument("--holdouts", nargs="+", type=float,
                        default=[HOLDOUT_FRAC],
                        help="holdout fractions; default: %(default)s")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes over the splits (results do not "
                             "depend on it); default: %(default)s")
    parser.add_argument("--selections-json",
                        default="greedy_mi_cv_selections_all.json",
                        help="file name under the logs directory of the "
                             "rep-0 MI selection log of this run's datasets "
                             "('none' skips it); default: %(default)s")
    parser.add_argument("--selections-from-logs", action="store_true",
                        help="run nothing; write --selections-json from "
                             "the greedy_mi_cv_splits_{ds}.json of "
                             "--datasets in the logs directory (combines "
                             "per-dataset runs into the joint run's file)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    write_sel = args.selections_json.lower() != "none"

    if args.selections_from_logs:
        if not write_sel:
            parser.error("--selections-from-logs needs --selections-json")
        all_logs = []
        for ds_name in args.datasets:
            path = LOGS / f"greedy_mi_cv_splits_{ds_name}.json"
            if not path.is_file():
                parser.error(f"missing {path}")
            all_logs.extend(json.loads(path.read_text()))
        write_selections(LOGS / args.selections_json, all_logs)
        sys.exit(0)

    missing = [ds for ds in args.datasets
               if not (DATA / f"{ds}.matrix.npz").is_file()]
    if missing:
        parser.error("no matrix for dataset(s) "
                     + ", ".join(f"{ds} ({DATA / f'{ds}.matrix.npz'})"
                                 for ds in missing))

    total_t0 = time.time()
    all_logs = []

    for ds_name in args.datasets:
        ds_t0 = time.time()
        results, logs, bench_names = run_dataset(
            ds_name, args.repeats, args.holdouts, args.workers)
        save_dataset(ds_name, results, logs, bench_names, args.repeats)
        plot_dataset(ds_name, logs)
        all_logs.extend(logs)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{ds_name.upper()} completed in {ds_elapsed:.0f}s\n")

    # Rep-0 selection order in the historical format
    if write_sel:
        write_selections(LOGS / args.selections_json, all_logs)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.0f}s")
