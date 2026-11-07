"""Evaluate greedy benchmark selection on all datasets with cross-validation.

For each dataset (any data/<name>.matrix.npz; default MMLU, MTEB, Merged)
and holdout fraction:
  1. Split models into train/test: --repeats repetitions of 10-fold CV, the
     matched splits of cv_split.cv_splits (repetition 0 is the original
     seed-42 folds and training subsamples)
  2. Estimate correlation matrix from training models
  3. Eligible benchmarks: observed for at least MIN_TRAIN_OBS training
     models (cv_split.eligible_mask); one mask per split, shared by every
     method.  Ineligible benchmarks stay in R and remain imputation targets
     if observed for at least MIN_TARGET_OBS training models
     (cv_split.target_mask, protocol rule D1; the cells of the others are
     never scored)
  4. Select k = 1..k_max benchmarks (k_max = min(--kmax, N), default 15;
     at k = N nothing is left to impute and R² is NaN) by greedy entropy
     and random selection, each from the split's common random numbers
     (Split.seed; entropy_rf was dropped on 2026-10-18, protocol rule D3),
     and since 2026-10-19 by greedy trace minimization
     (baseline_select.greedy_trace_select) and greedy facility location
     on the correlation matrix clipped at 0
     (baseline_select.facility_location_select, the paper's headline
     method).  Both are deterministic and draw no random numbers, so the
     entropy and random rows are those of a run without them, bit for bit
  5. For each k, impute test models' unselected target scores from the
     selected ones, predictions clipped to the valid score range
     (score_bounds.py, rule D2)
  6. Report: entropy, residual variance, R², RMSE on held-out data, and
     paired R² differences between methods over the matched splits
  7. Log full selection order per split and method

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir()):

  LOGS/greedy_cv_{ds}.npz  per holdout p, keys prefixed p{100p}_ (p10_ ...);
      rows are the splits in cv_splits order (rep-major, folds in order:
      the first 10 rows are rep 0, i.e. the historical folds):
        {pre}{metric}   (n_splits, k_max) float, metric in entropy (log det
                        R_AA), resid_var, r2, rmse; pre = "" greedy entropy,
                        "rand_" random (the historical keys keep their
                        names), "trace_" greedy trace, "facloc_" facility
                        location; NaN for k > n_eligible
        rep, fold, split_id, n_train, n_test, n_eligible   (n_splits,) int
        eligible        (n_splits, N) bool, the split's selectable benchmarks
        failed          (n_splits,) bool, covariance estimation failed (the
                        split's metric rows are NaN)
        {pre}sel        (n_splits, k_max) int, selection order, -1 padded
        {pre}index_ties (n_splits, k_max) int, picks among the first k
                        decided by the lowest column index
                        (greedy_select.index_tie_count(result, k))
        {pre}tie_breaks (n_splits, k_max) int, picks among the first k made
                        among near-tied scores
                        (greedy_select.tie_break_count(result, k); the
                        layout of {pre}index_ties, as in every log)
      plus benchmark_names, holdout_fracs, methods, method_prefixes
      (aligned with methods), n_folds, n_repeats, seed, min_train_obs,
      min_target_obs, pred_clip, k_max.  Pair methods row by row (same
      split_id).
  LOGS/greedy_cv_splits_{ds}.json  one entry per (holdout, split, method):
      dataset, holdout_frac, rep, fold, split_id, n_train, n_test,
      n_eligible, ineligible_idx, method, selected_idx, selected_names,
      tie_breaks, index_ties.  The entries of entropy and random come
      first, per holdout and split as before; those of the methods added
      on 2026-10-19 (APPENDED_METHODS: trace, facloc) follow all of them,
      in the same (holdout, split, method) order, so every earlier entry
      keeps its position in the list.
  LOGS/greedy_cv_paired_{ds}.csv  paired R² differences of the method
      pairs PAIRS per (holdout, k) over the matched splits
      (paired_stats.paired_stats, oriented as paired_stats.orient; columns
      dataset, holdout_frac, k, method_a, method_b, then
      paired_stats.STAT_FIELDS): entropy − random, then the pairs of
      trace and facloc (trace − random, facloc − random, entropy − trace,
      entropy − facloc, trace − facloc); rows pair by pair (then holdout,
      k), so the entropy − random rows are those of a run without the
      other two methods, in the same places.
  LOGS/greedy_cv_selections.json  (--selections-json) rep-0 greedy-entropy
      selection order per (dataset, holdout, fold) of the datasets of this
      run, in the historical format (plot_selection_order.py,
      make_numbers.py), with rep, split_id, n_eligible, tie_breaks and
      index_ties added.
  OUT/greedy_cv_{ds}.pdf  3-panel CV plot: greedy entropy (± std band for
      R²), random, greedy trace and facility location per holdout (color),
      means over all splits.  eval_benchpress.py writes
      greedy_cv_benchpress.pdf itself; do not run both for benchpress into
      one experiment root.

Usage:
    python eval_greedy_all.py [--datasets mmlu mteb merged] [--repeats 5]
                              [--holdouts 0.1 0.2 0.5 0.9]
                              [--kmax 15]
                              [--workers 1]
                              [--selections-json greedy_cv_selections.json]
                              [--selections-from-logs]

--datasets accepts any name with a data/<name>.matrix.npz.  --repeats 1
reproduces the single-repetition run (rep 0 only); rep-0 rows do not depend
on --repeats.  Against the historical logs/greedy_cv_{ds}.npz, rep 0
reproduces the splits and random draws exactly, and R² to ~1e-13 except
Merged at the 90% holdout (12 training models for 114 benchmarks), where
random R² differs by up to 1.5e-2 even with identical splits and picks:
EM there is not bitwise reproducible against the platform of the
historical log (1 vs 8 BLAS threads moves the estimate by only ~1e-13, so
likely a different EM stopping iteration at tolerance 5e-4).  Pairing is
unaffected.  Runs on disjoint dataset subsets share no state and can run
in parallel, but each rewrites --selections-json with its own datasets: run
them with --selections-json none, then

    python eval_greedy_all.py --datasets mmlu mteb merged --selections-from-logs

writes greedy_cv_selections.json from their greedy_cv_splits_{ds}.json,
identical to the joint run's.

--workers W evaluates the splits of all holdouts in W spawned
single-threaded processes (eval_entropy_vs_mi.map_splits): every split is a
pure function of the data and the split (common random numbers from
Split.seed), so the logs do not depend on W, and the per-split progress
lines are printed in split order (stdout differs from the serial run's in
the timings only).  The default W = 1 runs the serial loop, as before.
"""

import argparse
import pathlib
import sys
import time
import json
import warnings
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

from greedy_select import (greedy_entropy, random_select, selection_at_k,
                           index_tie_count, tie_break_count)
from baseline_select import facility_location_select, greedy_trace_select
from em_cov import em_cov
from cv_split import (cv_splits, eligible_mask, target_mask, MIN_TRAIN_OBS,
                      N_REPEATS, RANDOM_SEED_OFFSET)
from paired_stats import STAT_FIELDS, orient_pair, paired_stats, write_csv
from eval_entropy_vs_mi import (impute_and_evaluate, map_splits,
                                protocol_config, say_now)
from score_bounds import dataset_bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRACS = [0.1, 0.2, 0.5, 0.9]
SEED = 42

DATASETS = ["mmlu", "mteb", "merged"]

# Methods and the key prefix of their per-split arrays in greedy_cv_{ds}.npz
# ("" and "rand_" are the historical keys of greedy entropy and random;
# greedy trace and facility location joined on 2026-10-19).
METHODS = ["entropy", "random", "trace", "facloc"]
METHOD_PREFIX = {"entropy": "", "random": "rand_", "trace": "trace_",
                 "facloc": "facloc_"}
METRICS = ["entropy", "resid_var", "r2", "rmse"]
# Methods whose greedy_cv_splits_{ds}.json entries follow those of the
# original two (module docstring), so that no earlier entry moves
APPENDED_METHODS = ("trace", "facloc")

# Paired comparisons (method_a − method_b) in greedy_cv_paired_{ds}.csv,
# in paired_stats' orientation; the original pair first (paired_rows
# writes pair by pair, so its rows come first, as in a run without the
# others)
PAIRS = [orient_pair(a, b) for a, b in [
    ("entropy", "random"), ("trace", "random"), ("facloc", "random"),
    ("entropy", "trace"), ("entropy", "facloc"), ("trace", "facloc")]]
PAIRED_CSV_FIELDS = (["dataset", "holdout_frac", "k", "method_a",
                      "method_b"] + STAT_FIELDS)

# Line styles of the methods in the CV figure (the holdout sets the color)
PLOT_STYLE = {"entropy": dict(fmt="o-", markersize=4, linewidth=1.5),
              "random": dict(fmt="x--", markersize=3, linewidth=1.0,
                             alpha=0.6),
              "trace": dict(fmt="^:", markersize=3, linewidth=1.0,
                            alpha=0.8),
              "facloc": dict(fmt="v-.", markersize=3, linewidth=1.2,
                             alpha=0.9)}
PLOT_LABEL = {"entropy": "Entropy (Alg. 1)", "random": "Random",
              "trace": "Greedy trace min.", "facloc": "Facility location"}
PLOT_SHORT = {"entropy": "R²", "random": "Rand", "trace": "Trace",
              "facloc": "FL"}       # per-split progress lines


def run_selectors(R, k_max, benchmark_names, eligible, split):
    """Every method's selection on one split, in METHODS order: the split's
    eligibility mask for all of them, its common random numbers for random
    (trace and facloc draw none)."""
    return {
        "entropy": greedy_entropy(
            R, k_max, benchmark_names=benchmark_names, eligible=eligible),
        "random": random_select(
            R, k_max, seed=split.seed(SEED + RANDOM_SEED_OFFSET),
            benchmark_names=benchmark_names, eligible=eligible),
        "trace": greedy_trace_select(
            R, k_max, benchmark_names=benchmark_names, eligible=eligible),
        "facloc": facility_location_select(
            R, k_max, benchmark_names=benchmark_names, eligible=eligible),
    }


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

# impute_and_evaluate is eval_entropy_vs_mi's pooled R² (imported above;
# this script had a verbatim copy until 2026-10-18).  Its vectorized path
# now requires identical observation rows of all validation models: the
# old condition dropped target cells on HELM Classic and HELM Lite
# (protocol_changes.md H1) and chose the same path as the new one on every
# other dataset, so their greedy_cv logs are unchanged.


def evaluate_selection(result, R, B_test, O_test, col_mean, col_std, N,
                       k_max=K_MAX, target=None, bounds=None):
    """Per-k metrics (lists of length k_max) of one selector's result:
    entropy = log det R_AA, resid_var, r2 and rmse of the test models
    (impute_and_evaluate with the split's D1 targets and the dataset's
    D2 bounds).  NaN for k beyond the eligible benchmarks (selection_at_k
    is None)."""
    out = {m: [] for m in METRICS}
    for k in range(1, k_max + 1):
        out["resid_var"].append(result["residual_variance"][k - 1])
        selected = selection_at_k(result, k)
        if selected is None:
            for m in ("entropy", "r2", "rmse"):
                out[m].append(np.nan)
            continue
        R_AA = R[np.ix_(selected, selected)]
        sign, logdet = np.linalg.slogdet(R_AA)
        out["entropy"].append(logdet if sign > 0 else -np.inf)
        imp = impute_and_evaluate(B_test, O_test, selected, R,
                                  col_mean, col_std, N, target=target,
                                  bounds=bounds)
        out["r2"].append(imp["r2"])
        out["rmse"].append(imp["rmse"])
    return out


# ── Paired statistics over matched splits ────────────────────────

def paired_rows(ds_name, results):
    """CSV rows (dicts over PAIRED_CSV_FIELDS) of the R² differences of
    every pair in PAIRS (paired_stats.paired_stats), per holdout and
    k = 1..k_max (the width of the r2 arrays).  Rows run pair by pair
    (then holdout, then k), so the original entropy − random rows come
    first, exactly as in a run without the methods added later."""
    rows = []
    for m_a, m_b in PAIRS:
        for p, res in results.items():
            r2_a = res[METHOD_PREFIX[m_a] + "r2"]
            r2_b = res[METHOD_PREFIX[m_b] + "r2"]
            for k in range(1, r2_a.shape[1] + 1):
                st = paired_stats(r2_a[:, k - 1], r2_b[:, k - 1],
                                  res["n_train"], res["n_test"], res["rep"])
                rows.append({"dataset": ds_name, "holdout_frac": p, "k": k,
                             "method_a": m_a, "method_b": m_b, **st})
    return rows


def write_paired_csv(path, rows):
    write_csv(rows, path, PAIRED_CSV_FIELDS)


def print_paired(rows, ks=(1, 5, 10)):
    print("\n  Paired R² differences over matched splits "
          "(mean, Nadeau–Bengio 95% CI, win rate):", flush=True)
    for row in rows:
        if row["k"] not in ks:
            continue
        print(f"    p={row['holdout_frac']:.0%} k={row['k']:2d} "
              f"{row['method_a']:>10} − {row['method_b']:<10} "
              f"Δ={row['mean_diff']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
              f"win={row['win_rate']:.2f} (n={row['n_splits']})",
              flush=True)


# ── Main ─────────────────────────────────────────────────────────

def _pad_index(selected, k_max=K_MAX):
    """Selection order as a length-k_max int list, -1 padded."""
    sel = [int(j) for j in selected][:k_max]
    return sel + [-1] * (k_max - len(sel))


def _r2_at(r2_row, k):
    """Formatted R² of the size-k selection ("-" beyond k_max)."""
    return f"{r2_row[k - 1]:.3f}" if k <= len(r2_row) else "-"


def _nanmean(a, axis=0):
    """np.nanmean without the all-NaN-column warning (k > n_eligible, or
    k = N with no imputation target)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis)


def greedy_cv_split(ctx, s, say=say_now):
    """One matched split of run_dataset at holdout s.holdout_frac (its loop
    body): {"rows": [(npz key suffix, the split's row)] in the order the
    keys enter results[p], "logs": [selection log entry of every method]};
    progress via say (eval_entropy_vs_mi.map_splits)."""
    t0 = time.time()
    B, O, N, k_max = ctx["B"], ctx["O"], ctx["N"], ctx["k_max"]
    methods = ctx["methods"]
    p = s.holdout_frac
    B_train, O_train = B[s.train_idx], O[s.train_idx]
    B_test, O_test = B[s.val_idx], O[s.val_idx]
    M_train = len(s.train_idx)
    M_test = len(s.val_idx)

    # One eligibility mask per split, shared by every method
    el = eligible_mask(O_train)
    target = target_mask(O_train)    # D1, shared by every method
    n_el = int(el.sum())
    rows = [("rep", s.rep), ("fold", s.fold), ("split_id", s.split_id),
            ("n_train", M_train), ("n_test", M_test), ("n_eligible", n_el),
            ("eligible", el)]
    logs = []
    split_info = {
        "dataset": ctx["ds_name"], "holdout_frac": p, "rep": s.rep,
        "fold": s.fold, "split_id": s.split_id, "n_train": M_train,
        "n_test": M_test, "n_eligible": n_el,
        "ineligible_idx": np.flatnonzero(~el).tolist(),
    }

    say(f"  Rep {s.rep} fold {s.fold}: {M_train} train / "
        f"{M_test} test, {n_el}/{N} eligible", end="")

    # Estimate correlation
    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:
        say(f" — estimation failed: {e}")
        rows.append(("failed", True))
        for m in methods:
            pre = METHOD_PREFIX[m]
            for metric in METRICS:
                rows.append((pre + metric, [np.nan] * k_max))
            rows.append((pre + "sel", [-1] * k_max))
            rows.append((pre + "index_ties", [0] * k_max))
            rows.append((pre + "tie_breaks", [0] * k_max))
            logs.append({**split_info, "method": m,
                         "selected_idx": [], "selected_names": [],
                         "tie_breaks": [], "index_ties": []})
        return {"rows": rows, "logs": logs}
    rows.append(("failed", False))

    # Same split, eligible set and common random numbers for all
    runs = run_selectors(R, k_max, ctx["names"], el, s)
    evals = {}
    for m, res in runs.items():
        pre = METHOD_PREFIX[m]
        evals[m] = evaluate_selection(res, R, B_test, O_test,
                                      col_mean, col_std, N, k_max,
                                      target, ctx["bounds"])
        for metric in METRICS:
            rows.append((pre + metric, evals[m][metric]))
        rows.append((pre + "sel", _pad_index(res["selected"], k_max)))
        rows.append((pre + "index_ties",
                     [index_tie_count(res, k) for k in range(1, k_max + 1)]))
        rows.append((pre + "tie_breaks",
                     [tie_break_count(res, k) for k in range(1, k_max + 1)]))
        logs.append({**split_info, "method": m,
                     "selected_idx": res["selected"],
                     "selected_names": res.get("names", []),
                     "tie_breaks": res["tie_breaks"],
                     "index_ties": res["index_ties"]})

    fold_r2 = evals["entropy"]["r2"]
    elapsed = time.time() - t0
    say(f"  R²@5={_r2_at(fold_r2, 5)}  R²@10={_r2_at(fold_r2, 10)}"
        + "".join(f"  {PLOT_SHORT[m]}@5={_r2_at(evals[m]['r2'], 5)}"
                  for m in methods[1:])
        + f"  ({elapsed:.1f}s)")

    # Print selection order for this split
    result = runs["entropy"]
    names = result.get("names", [str(i) for i in result["selected"]])
    top5 = ", ".join(names[:5])
    say(f"         selected: {top5}, ...")
    return {"rows": rows, "logs": logs}


def run_dataset(ds_name, n_repeats=N_REPEATS, holdout_fracs=HOLDOUT_FRACS,
                k_max=K_MAX, workers=1):
    """Run the matched-split CV evaluation for one dataset.

    Returns (results, split_logs, benchmark_names): results[p] maps the npz
    key suffixes (module docstring) to arrays over the splits of holdout p
    (per-k arrays with min(k_max, N) columns); split_logs has one selection
    entry per (holdout, split, method), those of APPENDED_METHODS after all
    others.  workers > 1 evaluates the splits of all holdouts in that many
    processes (eval_entropy_vs_mi.map_splits; the arrays, logs and
    progress lines are those of the serial run).
    """
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)     # D2
    obs_frac = O.sum() / (M * N)
    print(f"{M} models × {N} tasks, {obs_frac*100:.1f}% observed")
    print(f"{n_repeats} × {N_FOLDS}-fold matched splits; selectable: "
          f">= {MIN_TRAIN_OBS} observed training models")
    # At most N picks (k = N leaves no imputation target: R² is NaN there)
    k_max = min(k_max, N)
    ks_report = [k for k in [1, 3, 5, 10, 15] if k <= k_max]

    results = {}
    split_logs = []  # selection order per (split, method)
    appended_logs = []  # ... of APPENDED_METHODS, written after split_logs

    def log_entry(entry):
        (appended_logs if entry["method"] in APPENDED_METHODS
         else split_logs).append(entry)

    ctx = dict(B=B, O=O, N=N, k_max=k_max, names=benchmark_names,
               bounds=bounds, ds_name=ds_name, methods=list(METHODS))
    holdout_fracs = list(holdout_fracs)     # iterated twice below
    splits = {p: list(cv_splits(M, N_FOLDS, n_repeats, p, SEED))
              for p in holdout_fracs}
    # One pass over the splits of every holdout (the serial path evaluates
    # each split when its row is taken, after its holdout's header)
    parts = map_splits(greedy_cv_split, ctx,
                       [s for p in holdout_fracs for s in splits[p]], workers)
    with closing(parts):
        for p in holdout_fracs:
            metrics = defaultdict(list)

            print(f"\n{'='*60}")
            print(f"Holdout {p:.0%}")
            print(f"{'='*60}")

            # zip takes the next split first: exactly one part per split
            for _, part in zip(splits[p], parts):
                for key, val in part["rows"]:
                    metrics[key].append(val)
                for entry in part["logs"]:
                    log_entry(entry)

            results[p] = {k: np.array(v) for k, v in metrics.items()}

            # Print summary
            valid = ~np.all(np.isnan(results[p]["r2"]), axis=1)
            if valid.any():
                mean_r2 = _nanmean(results[p]["r2"][valid])
                print(f"\n  Summary (holdout {p:.0%}, {valid.sum()} valid "
                      f"splits):", flush=True)
                for k in ks_report:
                    rv = _nanmean(results[p]["resid_var"][valid])[k-1]
                    ent = _nanmean(results[p]["entropy"][valid])[k-1]
                    rmse = _nanmean(results[p]["rmse"][valid])[k-1]
                    print(f"    k={k:2d}: R²={mean_r2[k-1]:.4f}  "
                          f"resid_var={rv:.4f}  entropy={ent:.3f}  "
                          f"rmse={rmse:.4f}", flush=True)
                for m in METHODS[1:]:
                    r2_m = _nanmean(results[p][METHOD_PREFIX[m] + "r2"][valid])
                    print(f"    {m:<10} R²: " + "  ".join(
                        f"k={k}: {r2_m[k-1]:.4f}" for k in ks_report),
                        flush=True)
                n_ties = {m: int(results[p][METHOD_PREFIX[m] + "index_ties"]
                                 [:, -1].sum()) for m in METHODS}
                print(f"    index-decided picks (all splits, k={k_max}): "
                      + ", ".join(f"{m} {n}" for m, n in n_ties.items()),
                      flush=True)

    return results, split_logs + appended_logs, benchmark_names


# ── Plotting helper ──────────────────────────────────────────────

def plot_results(ds_name, results):
    """Generate 3-panel CV plot for one dataset: per holdout (color) greedy
    entropy (solid, ± std band for R²) and the other METHODS in their
    PLOT_STYLE; one gray legend entry per other method."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    k_max = next(iter(results.values()))["r2"].shape[1]
    ks = np.arange(1, k_max + 1)

    colors = {0.1: "#4477AA", 0.2: "#66CCEE",
              0.5: "#EE7733", 0.9: "#CC3311"}

    for i, p in enumerate(results):
        c = colors.get(p, f"C{i}")
        label_g = f"{p:.0%} holdout"
        valid = ~np.all(np.isnan(results[p]["r2"]), axis=1)
        if not valid.any():
            continue

        mean_r2 = _nanmean(results[p]["r2"][valid])
        std_r2 = _nanstd(results[p]["r2"][valid])
        mean_rv = _nanmean(results[p]["resid_var"][valid])
        mean_ent = _nanmean(results[p]["entropy"][valid])

        axes[0].plot(ks, mean_r2, "o-", color=c, label=label_g,
                     markersize=4, linewidth=1.5)
        axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                             color=c, alpha=0.15)
        axes[1].semilogy(ks, mean_rv, "o-", color=c, label=label_g,
                         markersize=4, linewidth=1.5)
        axes[2].plot(ks, mean_ent, "o-", color=c, label=label_g,
                     markersize=4, linewidth=1.5)

        # Random (dashed), trace and facility location in the same color
        for m in METHODS[1:]:
            pre = METHOD_PREFIX[m]
            if pre + "r2" not in results[p]:
                continue
            valid_m = ~np.all(np.isnan(results[p][pre + "r2"]), axis=1)
            if not valid_m.any():
                continue
            style = dict(PLOT_STYLE[m])
            fmt = style.pop("fmt")
            label_m = f"Random {p:.0%}" if m == "random" and p == 0.1 \
                else None
            axes[0].plot(ks, _nanmean(results[p][pre + "r2"][valid_m]), fmt,
                         color=c, label=label_m, **style)
            axes[1].semilogy(ks, _nanmean(
                results[p][pre + "resid_var"][valid_m]), fmt, color=c,
                **style)
            axes[2].plot(ks, _nanmean(results[p][pre + "entropy"][valid_m]),
                         fmt, color=c, **style)

    # One legend entry per method other than entropy via dummy artists
    for m in METHODS[1:]:
        style = dict(PLOT_STYLE[m])
        fmt = style.pop("fmt")
        axes[0].plot([], [], fmt, color="gray", label=PLOT_LABEL[m], **style)

    axes[0].set_ylabel("Imputation $R^2$")
    axes[1].set_ylabel("Residual variance fraction")
    axes[2].set_ylabel("Entropy $\\log\\det(R_{\\mathcal{A}})$")

    for ax in axes:
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xlim(0.5, k_max + 0.5)
        ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= k_max])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT / f"greedy_cv_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}", flush=True)
    plt.close()


# ── Run all datasets ─────────────────────────────────────────────

def convert(obj):
    """numpy scalars/arrays and tuples to JSON-native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [convert(v) for v in obj]
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    return obj


def legacy_selection(entry):
    """Historical greedy_cv_selections.json entry of a rep-0 entropy log
    (first five keys as before, then the matched-split metadata)."""
    keys = ["dataset", "holdout_frac", "fold", "selected_idx",
            "selected_names", "rep", "split_id", "n_eligible", "tie_breaks",
            "index_ties"]
    return {k: entry[k] for k in keys}


def write_selections(path, split_logs):
    """Write the historical selection log: the rep-0 greedy-entropy entries
    of split_logs (in their order) in legacy_selection format."""
    json_logs = [legacy_selection({k: convert(v) for k, v in e.items()})
                 for e in split_logs
                 if e["method"] == "entropy" and e["rep"] == 0]
    with open(path, "w") as f:
        json.dump(json_logs, f, indent=2, default=convert)
    print(f"\nSelection logs saved to {path}", flush=True)


def save_dataset(ds_name, results, split_logs, bench_names, n_repeats):
    """Write greedy_cv_{ds}.npz, greedy_cv_splits_{ds}.json and
    greedy_cv_paired_{ds}.csv; returns the paired rows."""
    save_dict = {}
    for p, metrics in results.items():
        prefix = f"p{int(p*100)}"
        for metric_name, arr in metrics.items():
            save_dict[f"{prefix}_{metric_name}"] = arr
    save_dict["benchmark_names"] = bench_names
    save_dict["holdout_fracs"] = np.array(list(results))
    save_dict["methods"] = np.array(METHODS)
    save_dict["method_prefixes"] = np.array([METHOD_PREFIX[m]
                                             for m in METHODS])
    save_dict["n_folds"] = np.array(N_FOLDS)
    save_dict["n_repeats"] = np.array(n_repeats)
    save_dict["seed"] = np.array(SEED)
    proto = protocol_config()
    save_dict["min_train_obs"] = np.array(proto["min_train_obs"])
    save_dict["min_target_obs"] = np.array(proto["min_target_obs"])
    save_dict["pred_clip"] = np.array(proto["pred_clip"])
    save_dict["k_max"] = np.array(            # min(--kmax, N)
        next(iter(results.values()))["r2"].shape[1])
    np.savez(LOGS / f"greedy_cv_{ds_name}.npz", **save_dict)

    with open(LOGS / f"greedy_cv_splits_{ds_name}.json", "w") as f:
        json.dump([{k: convert(v) for k, v in e.items()} for e in split_logs],
                  f, indent=1, default=convert)

    rows = paired_rows(ds_name, results)
    write_paired_csv(LOGS / f"greedy_cv_paired_{ds_name}.csv", rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Greedy entropy / random / greedy trace / facility "
                    "location selection under matched repeated CV (see "
                    "module docstring).")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names (data/<name>.matrix.npz); "
                             "default: %(default)s")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions (rep 0 = the historical folds); "
                             "default: %(default)s")
    parser.add_argument("--holdouts", nargs="+", type=float,
                        default=HOLDOUT_FRACS,
                        help="holdout fractions; default: %(default)s")
    parser.add_argument("--kmax", type=int, default=K_MAX,
                        help="largest k (capped at the number of "
                             "benchmarks); default: %(default)s")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes over the splits (results do not "
                             "depend on it); default: %(default)s")
    parser.add_argument("--selections-json",
                        default="greedy_cv_selections.json",
                        help="file name under the logs directory of the "
                             "rep-0 entropy selection log of this run's "
                             "datasets ('none' skips it); default: "
                             "%(default)s")
    parser.add_argument("--selections-from-logs", action="store_true",
                        help="run nothing; write --selections-json from "
                             "the greedy_cv_splits_{ds}.json of --datasets "
                             "in the logs directory (combines per-dataset "
                             "runs into the joint run's file)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.kmax < 1:
        parser.error("--kmax must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    write_sel = args.selections_json.lower() != "none"

    if args.selections_from_logs:
        if not write_sel:
            parser.error("--selections-from-logs needs --selections-json")
        all_logs = []
        for ds_name in args.datasets:
            path = LOGS / f"greedy_cv_splits_{ds_name}.json"
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

    all_logs = []
    total_t0 = time.time()

    for ds_name in args.datasets:
        ds_t0 = time.time()
        results, logs, bench_names = run_dataset(ds_name, args.repeats,
                                                 args.holdouts, args.kmax,
                                                 args.workers)
        all_logs.extend(logs)
        plot_results(ds_name, results)

        # Save per-dataset results
        rows = save_dataset(ds_name, results, logs, bench_names, args.repeats)
        print_paired(rows)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{ds_name.upper()} completed in {ds_elapsed:.0f}s", flush=True)

    # Rep-0 greedy-entropy selection order in the historical format
    if write_sel:
        write_selections(LOGS / args.selections_json, all_logs)

    total_elapsed = time.time() - total_t0
    print(f"\nAll datasets completed in {total_elapsed:.0f}s", flush=True)
