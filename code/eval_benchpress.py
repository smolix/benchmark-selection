"""Greedy CV and entropy-vs-MI experiments on the matched splits, one
results file per dataset (the BenchPress results layout).

BenchPress (data/benchpress.matrix.npz) is an ordinary dataset: the
experiments here are those of eval_greedy_all.py (Part 1) and
eval_entropy_vs_mi.py (Part 2), on the same matched splits, eligibility
masks and seeds, with the same covariance estimator rule (--estimator
auto, which picks EM for every BenchPress split since M_train < N or
fewer than 90% of the entries are observed) and the same imputation.
Where those scripts cover BenchPress via --datasets, this script is only
needed for logs/benchpress_results.npz, the file (keys cv_p*_*, emi_*)
from which make_numbers.py reads the BenchPress numbers.  It accepts any
dataset name (--datasets, data/<name>.matrix.npz) and writes
logs/<name>_results.npz; its figure names are those of eval_greedy_all.py
and eval_entropy_vs_mi.py, so for a dataset that both cover, run only one
of them in an experiment root.

  Part 1  greedy CV at holdout 10/20/50/90%: greedy entropy, random,
          greedy trace and facility location
  Part 2  entropy vs MI vs random (and greedy trace, facility location)
          at the 10% holdout, on the splits and correlation estimates of
          Part 1
(entropy_rf, greedy entropy from a random first pivot, was dropped on
2026-10-18, protocol rule D3.)  Greedy trace minimization
(baseline_select.greedy_trace_select) and greedy facility location on the
correlation matrix clipped at 0 (baseline_select.facility_location_select,
the paper's headline method) joined both parts on 2026-10-19, after the
original methods; both are deterministic and draw no random numbers, so
every entropy, random and MI array is that of a run without them, bit for
bit.

Matched design (cv_split.cv_splits): n_repeats x 10-fold CV, default
N_REPEATS = 5, i.e. 50 splits per holdout fraction.  Repetition 0 is the
original seed-42 folds and training subsamples, so its rows reproduce the
single-repetition runs (--repeats 1 runs rep 0 only; rep-0 rows do not
depend on --repeats).  In every split only benchmarks observed in at
least MIN_TRAIN_OBS training models can be selected (one eligibility mask
per split, shared by every method); ineligible benchmarks stay in the
correlation estimate, the MI complement and the residual variance, and
remain imputation targets if observed for at least MIN_TARGET_OBS training
models (cv_split.target_mask, protocol rule D1: the cells of the others are
never scored); every prediction is clipped to its benchmark's valid score
range (score_bounds.py, rule D2).  The random baseline draws from
Split.seed(SEED + RANDOM_SEED_OFFSET) (rep 0: the original (SEED + 7, fold,
int(p * 10000))).  Near-ties of the greedy scores are broken by
explained variance, then by the lowest index (greedy_select); index_ties
counts the picks the lowest index decided.

Produces (default --datasets benchpress):
  logs/benchpress_results.npz
  logs/benchpress_results_paired.csv    paired R² differences
  figures/greedy_cv_benchpress.pdf      Part 1: entropy, random, greedy
                                        trace and facility location
  figures/entropy_vs_mi_benchpress.pdf  Part 2: entropy, MI, random,
                                        greedy trace and facility location

logs/<name>_results.npz is the greedy_cv_<name>.npz layout of
eval_greedy_all.py with keys prefixed "cv_", plus the
entropy_vs_mi_<name>.npz layout of eval_entropy_vs_mi.py prefixed "emi_".
S splits per holdout fraction; rows are the splits in cv_splits order
(rep-major, folds in order: the first n_folds rows are rep 0) and pair
across methods (equal rows = same split_id).  Values at k beyond the
number of eligible benchmarks are NaN.
  cv_p{pct}_{pre}{metric}    (S, k_max) float, pct in 10, 20, 50, 90;
      pre "" greedy entropy, "rand_" random, "trace_" greedy trace,
      "facloc_" facility location; metric
      entropy (log det R_AA), resid_var, r2, rmse
  cv_p{pct}_{pre}sel         (S, k_max) int, selection order, -1 padded
  cv_p{pct}_{pre}index_ties  (S, k_max) int, picks among the first k
      decided by the lowest index (greedy_select.index_tie_count)
  cv_p{pct}_{pre}tie_breaks  (S, k_max) int, picks among the first k made
      among near-tied scores (the layout of index_ties, as in every log)
  cv_p{pct}_{rep,fold,split_id,n_train,n_test,n_eligible}  (S,) int
  cv_p{pct}_eligible         (S, N) bool, the split's selectable benchmarks
  cv_p{pct}_failed           (S,) bool, estimation failed (rows NaN)
  cv_p{pct}_estimator        (S,) str, "em", "pairwise" or "failed"
  emi_{method}_{metric}      (S, k_max) float at the 10% holdout, method in
      entropy, mi, random, trace, facloc; metric r2, rmse, resid_var,
      mi_val
      (I(X_A; X_rest))
  emi_{method}_selected      (S, k_max) int, selection order, -1 padded
  emi_{method}_index_ties    (S, k_max) int, as for cv_
  emi_{method}_tie_breaks    (S, k_max) int, picks among the first k made
      among near-tied scores
  emi_{rep,fold,split_id,n_train,n_val,n_eligible}  (S,) int
  emi_eligible (S, N) bool, emi_estimation_failed (S,) bool
  benchmark_names, dataset, holdout_fracs, emi_holdout_frac, cv_methods,
  cv_method_prefixes, emi_methods, k_max, n_folds, n_repeats, seed,
  min_train_obs, min_target_obs, pred_clip, estimator
(The emi_ arrays of entropy, random, trace and facloc equal the cv_p10
ones: one selection per split.)

logs/<name>_results_paired.csv: for each holdout, k = 1..k_max and method
pair (CV_PAIRS; EMI_PAIRS too at the 10% holdout) the paired R²
difference method_a − method_b over the matched splits
(paired_stats.paired_stats: mean, Nadeau–Bengio corrected standard error,
95% t-interval, win rate; pairs oriented as paired_stats.orient, e.g.
mi − entropy), columns as eval_greedy_all.py's greedy_cv_paired_<name>.csv.
The rows of the pairs of trace and facloc (ADDED_CV_PAIRS, and
ADDED_EMI_PAIRS at the 10% holdout) follow all the others, which are
those of a run without the two methods, in the same order.

The eligibility threshold is cv_split.MIN_TRAIN_OBS, shared by every
script (no per-script override, so the eligible mask of a split is the
same in every log).

Reproduction.  The historical logs/benchpress_results.npz (April 2026)
predates the current data/benchpress.matrix.npz (May 2026) and is not a
rep-0 reference: its R² differ even where the split and the random picks
are identical.  logs/baselines_benchpress.npz, written from the current
matrix, is (eval_baselines.py rep 0 reproduces it to ~1e-12).

Usage:
    python eval_benchpress.py [--datasets benchpress ...] [--repeats 5]
                              [--folds 10] [--kmax 15]
                              [--estimator auto|em|pairwise]
                              [--workers 1]

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

from greedy_select import (greedy_entropy, greedy_mi,
                           index_tie_count, random_select, selection_at_k,
                           tie_break_count)
from baseline_select import facility_location_select, greedy_trace_select
from em_cov import em_cov
from cv_split import (MIN_TRAIN_OBS, N_REPEATS, target_mask,
                      RANDOM_SEED_OFFSET, cv_splits, eligible_mask)
from paired_stats import STAT_FIELDS, orient_pair, paired_stats, write_csv
from eval_entropy_vs_mi import (METHOD_COLOR, METHOD_LABEL, METHOD_STYLE,
                                impute_and_evaluate, map_splits,
                                protocol_config, say_now)
from score_bounds import dataset_bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRACS = [0.1, 0.2, 0.5, 0.9]
EMI_HOLDOUT = 0.1          # Part 2 runs on Part 1's 10% holdout splits
SEED = 42
DEFAULT_DATASETS = ["benchpress"]

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "mteb": "MTEB",
    "merged": "Merged",
    "benchpress": "BenchPress",
}

# Part 1 methods -> key prefix of their cv_p{pct}_ arrays (as in
# eval_greedy_all.py: greedy entropy unprefixed, random "rand_"; greedy
# trace and facility location joined on 2026-10-19)
CV_METHODS = {"entropy": "", "random": "rand_", "trace": "trace_",
              "facloc": "facloc_"}
CV_METRICS = ["entropy", "resid_var", "r2", "rmse"]
EMI_METHODS = ["entropy", "mi", "random", "trace", "facloc"]
EMI_METRICS = ["r2", "rmse", "resid_var", "mi_val"]
PLOT_EMI_METHODS = ["entropy", "mi", "random", "trace", "facloc"]

# Paired R² comparisons (method_a − method_b, in paired_stats'
# orientation): CV_PAIRS at every holdout, EMI_PAIRS in addition at
# EMI_HOLDOUT; then (paired_rows writes them after all of those) the
# pairs of greedy trace and facility location
CV_PAIRS = [orient_pair(a, b) for a, b in [("entropy", "random")]]
EMI_PAIRS = [orient_pair(a, b) for a, b in [("mi", "entropy"),
                                             ("mi", "random")]]
ADDED_CV_PAIRS = [orient_pair(a, b) for a, b in [
    ("trace", "random"), ("facloc", "random"), ("entropy", "trace"),
    ("entropy", "facloc"), ("trace", "facloc")]]
ADDED_EMI_PAIRS = [orient_pair(a, b) for a, b in [("mi", "trace"),
                                                   ("mi", "facloc")]]
PAIRED_CSV_FIELDS = (["dataset", "holdout_frac", "k", "method_a",
                      "method_b"] + STAT_FIELDS)

# Part 1 figure: line style of every method but greedy entropy (drawn
# solid with a band) in its holdout's color, as in eval_greedy_all.py
CV_PLOT_STYLE = {"random": dict(fmt="x--", markersize=3, linewidth=1.0,
                                alpha=0.6),
                 "trace": dict(fmt="^:", markersize=3, linewidth=1.0,
                               alpha=0.8),
                 "facloc": dict(fmt="v-.", markersize=3, linewidth=1.2,
                                alpha=0.9)}


def display_name(ds_name):
    return DISPLAY_NAME.get(ds_name, ds_name)


def _nanmean(a, axis=0):
    """np.nanmean without the empty-slice warning (k beyond the eligible
    benchmarks of every split gives an all-NaN column, mean NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(a, axis=axis)


# ── Covariance estimation (same as eval_greedy_all.py) ───────────

def estimate_corr_pairwise(B_train, O_train):
    """Pairwise-complete correlation, projected PSD, unit diagonal.
    Suitable for nearly-complete or complete data."""
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
    """Correlation via EM. For sparse/rank-deficient data."""
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
    """(R, col_mean, col_std, name of the estimator used).  "auto" is the
    rule of eval_greedy_all.py / eval_entropy_vs_mi.py: EM for sparse
    (< 90% observed) or rank-deficient (M_train < N) training data, else
    pairwise."""
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)
    if method == "auto":
        method = "em" if (obs_frac < 0.9 or M_tr < N_tr) else "pairwise"
    if method == "em":
        return (*estimate_corr_em(B_train, O_train), "em")
    return (*estimate_corr_pairwise(B_train, O_train), "pairwise")


# ── Imputation ───────────────────────────────────────────────────

# impute_and_evaluate is eval_entropy_vs_mi's pooled R² (imported above).
# Until 2026-10-18 this script had a verbatim copy whose vectorized path
# was taken when the validation rows agreed only on the selected
# benchmarks and on those validation model 0 observes, and then dropped
# the cells other models observe and model 0 does not (protocol_changes.md
# H1).  The shared function takes that path only for identical observation
# rows; on every split where the old copy dropped no cell (BenchPress: all
# 50) both choose the same path, so benchpress_results.npz is unchanged.


# ── Selection and per-k evaluation on one split ───────────────────

def full_logdet(R):
    """log det R, the constant of the MI formula (eigenvalue fallback)."""
    sign_full, logdet_full = np.linalg.slogdet(R)
    if sign_full <= 0:
        eigv = np.linalg.eigvalsh(R)
        logdet_full = np.sum(np.log(np.maximum(eigv, 1e-300)))
    return logdet_full


def run_selectors(R, k_max, benchmark_names, eligible, split, with_mi):
    """Every method's selection on one split: the split's eligibility mask
    for all of them, its common random numbers for random (greedy trace
    and facility location are deterministic and draw none)."""
    results = {
        "entropy": greedy_entropy(R, k_max, benchmark_names=benchmark_names,
                                  eligible=eligible),
        "random": random_select(
            R, k_max, seed=split.seed(SEED + RANDOM_SEED_OFFSET),
            benchmark_names=benchmark_names, eligible=eligible),
        "trace": greedy_trace_select(R, k_max,
                                     benchmark_names=benchmark_names,
                                     eligible=eligible),
        "facloc": facility_location_select(R, k_max,
                                           benchmark_names=benchmark_names,
                                           eligible=eligible),
    }
    if with_mi:
        results["mi"] = greedy_mi(R, k_max, benchmark_names=benchmark_names,
                                  eligible=eligible)
    return results


def _empty_eval(k_max):
    out = {m: np.full(k_max, np.nan)
           for m in ("entropy", "resid_var", "r2", "rmse", "mi_val")}
    out["index_ties"] = np.zeros(k_max, dtype=np.int64)
    out["tie_breaks"] = np.zeros(k_max, dtype=np.int64)
    out["selected"] = np.full(k_max, -1, dtype=np.int64)
    return out


def evaluate_selection(result, R, B_test, O_test, col_mean, col_std, k_max,
                       logdet_full=None, target=None, bounds=None):
    """Per-k metrics of one (nested) selector's result on one split.

    (k_max,) arrays: entropy (log det R_AA, -inf if singular), resid_var,
    r2, rmse and, given logdet_full = log det R, mi_val = I(X_A; X_rest)
    (singular blocks at -1e30, as before); index_ties / tie_breaks (picks
    among the first k decided by the lowest index / made among near-tied
    scores) and selected (selection order, -1 padded).  NaN where fewer
    than k benchmarks were eligible.  R² and RMSE: impute_and_evaluate
    with the split's D1 targets and the dataset's D2 bounds.
    """
    N = R.shape[0]
    out = _empty_eval(k_max)
    out["resid_var"][:] = result["residual_variance"][:k_max]
    for k in range(1, k_max + 1):
        out["index_ties"][k - 1] = index_tie_count(result, k)
        out["tie_breaks"][k - 1] = tie_break_count(result, k)
        selected = selection_at_k(result, k)
        if selected is None:
            continue
        R_AA = R[np.ix_(selected, selected)]
        sign, logdet = np.linalg.slogdet(R_AA)
        out["entropy"][k - 1] = logdet if sign > 0 else -np.inf
        imp = impute_and_evaluate(B_test, O_test, selected, R,
                                  col_mean, col_std, N, target=target,
                                  bounds=bounds)
        out["r2"][k - 1] = imp["r2"]
        out["rmse"][k - 1] = imp["rmse"]
        if logdet_full is not None:
            ld_a = logdet if sign > 0 else -1e30
            comp = sorted(set(range(N)) - set(selected))
            R_CC = R[np.ix_(comp, comp)]
            s_c, ld_c = np.linalg.slogdet(R_CC)
            ld_c = ld_c if s_c > 0 else -1e30
            out["mi_val"][k - 1] = 0.5 * (ld_a + ld_c - logdet_full)
    picks = list(result["selected"])[:k_max]
    out["selected"][:len(picks)] = picks
    return out


# ── Run one dataset ──────────────────────────────────────────────

def benchpress_split(ctx, s, say=say_now):
    """One matched split of run_dataset at holdout s.holdout_frac (its loop
    body): {"meta": [(key, value)] of the split (failed and estimator
    last), "evals": {method: evaluate_selection(...)} of ctx's CV methods
    plus "mi" at EMI_HOLDOUT (_empty_eval rows if the estimate failed)};
    progress via say (eval_entropy_vs_mi.map_splits)."""
    t0 = time.time()
    B, O, N, k_max = ctx["B"], ctx["O"], ctx["N"], ctx["k_max"]
    at5 = min(5, k_max) - 1
    with_mi = s.holdout_frac == EMI_HOLDOUT
    methods = list(ctx["cv_methods"]) + (["mi"] if with_mi else [])
    B_train, O_train = B[s.train_idx], O[s.train_idx]
    B_test, O_test = B[s.val_idx], O[s.val_idx]
    el = eligible_mask(O_train)
    target = target_mask(O_train)    # D1, shared by every method
    meta = [("rep", s.rep), ("fold", s.fold), ("split_id", s.split_id),
            ("n_train", len(s.train_idx)), ("n_val", len(s.val_idx)),
            ("n_eligible", int(el.sum())), ("eligible", el)]

    say(f"  Rep {s.rep} fold {s.fold}: {len(s.train_idx)} train / "
        f"{len(s.val_idx)} test, {int(el.sum())}/{N} eligible", end="")

    try:
        R, col_mean, col_std, est = estimate_corr(B_train, O_train,
                                                  ctx["estimator"])
    except Exception as e:
        say(f" — failed: {e}")
        meta += [("failed", True), ("estimator", "failed")]
        return {"meta": meta,
                "evals": {m: _empty_eval(k_max) for m in methods}}
    meta += [("failed", False), ("estimator", est)]

    results = run_selectors(R, k_max, ctx["names"], el, s, with_mi)
    logdet_full = full_logdet(R) if with_mi else None
    evals = {m: evaluate_selection(results[m], R, B_test, O_test, col_mean,
                                   col_std, k_max, logdet_full, target,
                                   ctx["bounds"])
             for m in methods}

    elapsed = time.time() - t0
    line = f"  R²@{at5 + 1}={evals['entropy']['r2'][at5]:.3f}"
    if with_mi:
        line += f"  MI@{at5 + 1}={evals['mi']['r2'][at5]:.3f}"
    line += f"  Rand@{at5 + 1}={evals['random']['r2'][at5]:.3f}"
    for m, short in (("trace", "Trace"), ("facloc", "FL")):
        if m in evals:
            line += f"  {short}@{at5 + 1}={evals[m]['r2'][at5]:.3f}"
    ties = [f"{m}:{int(evals[m]['index_ties'][-1])}"
            for m in methods if evals[m]["index_ties"][-1]]
    if ties:
        line += "  index ties " + " ".join(ties)
    say(line + f"  ({elapsed:.1f}s)")
    return {"meta": meta, "evals": evals}


def run_dataset(ds_name, k_max=K_MAX, n_folds=N_FOLDS, n_repeats=N_REPEATS,
                estimator="auto", workers=1):
    """Part 1 at every holdout fraction (and Part 2 at EMI_HOLDOUT) over
    the matched splits.  Returns ({p: {"meta": {key: (S, ...)}, "evals":
    {method: {metric: (S, ...)}}}}, benchmark_names); "mi" is evaluated
    at EMI_HOLDOUT only.  workers > 1 evaluates the splits of all
    holdouts in that many processes (eval_entropy_vs_mi.map_splits; the
    arrays and progress lines are those of the serial run)."""
    print(f"\n{'#'*70}")
    print(f"# {display_name(ds_name)}: greedy CV and entropy vs MI "
          f"({n_repeats} x {n_folds}-fold matched splits)")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)     # D2
    print(f"{M} models × {N} tasks, {O.sum()/(M*N)*100:.1f}% observed; "
          f"selectable: >= {MIN_TRAIN_OBS} training observations",
          flush=True)

    cv = {}
    ctx = dict(B=B, O=O, N=N, k_max=k_max, names=benchmark_names,
               bounds=bounds, estimator=estimator,
               cv_methods=list(CV_METHODS))
    splits = {p: list(cv_splits(M, n_folds, n_repeats, p, SEED))
              for p in HOLDOUT_FRACS}
    # One pass over the splits of every holdout (the serial path evaluates
    # each split when its row is taken, after its holdout's header)
    parts = map_splits(benchpress_split, ctx,
                       [s for p in HOLDOUT_FRACS for s in splits[p]], workers)
    with closing(parts):
        for p in HOLDOUT_FRACS:
            with_mi = p == EMI_HOLDOUT
            methods = list(CV_METHODS) + (["mi"] if with_mi else [])
            evals = {m: [] for m in methods}
            meta = defaultdict(list)
            print(f"\n{'='*60}\nHoldout {p:.0%}\n{'='*60}")

            # zip takes the next split first: exactly one part per split
            for _, part in zip(splits[p], parts):
                for key, val in part["meta"]:
                    meta[key].append(val)
                for m in methods:
                    evals[m].append(part["evals"][m])

            cv[p] = {
                "meta": {key: np.array(v) for key, v in meta.items()},
                "evals": {m: {key: np.array([e[key] for e in evals[m]])
                              for key in evals[m][0]} for m in methods},
            }
            print_summary(cv[p], p, k_max)
            print_paired(paired_rows(ds_name, {p: cv[p]}, k_max), k_max)

    return cv, benchmark_names


def print_summary(res, p, k_max):
    """Mean R² (and resid_var) over valid splits, eligibility range and
    index-tie counts of one holdout fraction."""
    meta, evals = res["meta"], res["evals"]
    n = len(meta["split_id"])
    ks = [k for k in [1, 3, 5, 10, 15] if k <= k_max]
    print(f"\n  Summary (holdout {p:.0%}, {n} splits, "
          f"{meta['n_eligible'].min()}–{meta['n_eligible'].max()} eligible):")
    for m, ev in evals.items():
        valid = ~np.all(np.isnan(ev["r2"]), axis=1)
        if not valid.any():
            print(f"    {m:<11} no valid splits")
            continue
        mean_r2 = _nanmean(ev["r2"][valid], axis=0)
        mean_rv = _nanmean(ev["resid_var"][valid], axis=0)
        cells = "  ".join(f"k={k}: {mean_r2[k-1]:.4f}/{mean_rv[k-1]:.4f}"
                          for k in ks)
        n_tied = int(np.count_nonzero(ev["index_ties"][:, -1]))
        print(f"    {m:<11} R²/resid_var  {cells}  "
              f"[{valid.sum()} valid, index ties in {n_tied}]", flush=True)


# ── Paired statistics over matched splits ────────────────────────

def paired_rows(ds_name, cv, k_max):
    """CSV rows (dicts over PAIRED_CSV_FIELDS) of the R² differences of
    CV_PAIRS (and EMI_PAIRS at EMI_HOLDOUT), per holdout and k
    (paired_stats.paired_stats); then, per holdout and k again, those of
    ADDED_CV_PAIRS (and ADDED_EMI_PAIRS), so the first rows are those of
    a run without greedy trace and facility location."""
    rows = []
    for cv_pairs, emi_pairs in ((CV_PAIRS, EMI_PAIRS),
                                (ADDED_CV_PAIRS, ADDED_EMI_PAIRS)):
        rows += _paired_rows(ds_name, cv, k_max, cv_pairs, emi_pairs)
    return rows


def _paired_rows(ds_name, cv, k_max, cv_pairs, emi_pairs):
    """paired_rows of the pairs cv_pairs (every holdout) and emi_pairs
    (holdouts with MI)."""
    rows = []
    for p, res in cv.items():
        evals, meta = res["evals"], res["meta"]
        pairs = cv_pairs + (emi_pairs if "mi" in evals else [])
        for m_a, m_b in pairs:
            for k in range(1, k_max + 1):
                st = paired_stats(evals[m_a]["r2"][:, k - 1],
                                  evals[m_b]["r2"][:, k - 1],
                                  meta["n_train"], meta["n_val"],
                                  meta["rep"])
                rows.append({"dataset": ds_name, "holdout_frac": p, "k": k,
                             "method_a": m_a, "method_b": m_b, **st})
    return rows


def write_paired_csv(path, rows):
    write_csv(rows, path, PAIRED_CSV_FIELDS)
    print(f"Paired statistics saved to {path}", flush=True)


def print_paired(rows, k_max, ks=(1, 5, 10)):
    print("\n  Paired R² differences over matched splits "
          "(mean, Nadeau–Bengio 95% CI, win rate):", flush=True)
    for row in rows:
        if row["k"] not in ks or row["k"] > k_max:
            continue
        print(f"    k={row['k']:<2} {row['method_a']:>10} − "
              f"{row['method_b']:<10} {row['mean_diff']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  "
              f"win {row['win_rate']:.2f}  (n={row['n_splits']})",
              flush=True)


# ── Plotting ─────────────────────────────────────────────────────

def _style_axis(ax, k_max):
    ax.set_xlabel("Selected benchmarks $k$")
    ax.set_xlim(0.5, k_max + 0.5)
    ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= k_max])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_greedy_cv(ds_name, cv, k_max):
    """Part 1 figure: greedy entropy (solid) and random (dashed), greedy
    trace and facility location (CV_PLOT_STYLE) per holdout fraction
    (color); mean (± std for R² of greedy entropy) over valid splits."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = np.arange(1, k_max + 1)
    colors = {0.1: "#4477AA", 0.2: "#66CCEE", 0.5: "#EE7733", 0.9: "#CC3311"}

    for p in HOLDOUT_FRACS:
        if p not in cv:
            continue
        c = colors[p]
        ent = cv[p]["evals"]["entropy"]
        valid = ~np.all(np.isnan(ent["r2"]), axis=1)
        if not valid.any():
            continue

        mean_r2 = _nanmean(ent["r2"][valid], axis=0)
        std_r2 = _nanstd(ent["r2"][valid], axis=0)
        mean_rv = _nanmean(ent["resid_var"][valid], axis=0)
        mean_ent = _nanmean(ent["entropy"][valid], axis=0)

        axes[0].plot(ks, mean_r2, "o-", color=c, label=f"{p:.0%} holdout",
                     markersize=4, linewidth=1.5)
        axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                             color=c, alpha=0.15)
        axes[1].semilogy(ks, mean_rv, "o-", color=c, label=f"{p:.0%} holdout",
                         markersize=4, linewidth=1.5)
        axes[2].plot(ks, mean_ent, "o-", color=c, label=f"{p:.0%} holdout",
                     markersize=4, linewidth=1.5)

        # Random baseline, greedy trace and facility location
        for m, style in CV_PLOT_STYLE.items():
            if m not in cv[p]["evals"]:
                continue
            ev = cv[p]["evals"][m]
            valid_m = ~np.all(np.isnan(ev["r2"]), axis=1)
            if not valid_m.any():
                continue
            style = dict(style)
            fmt = style.pop("fmt")
            axes[0].plot(ks, _nanmean(ev["r2"][valid_m], axis=0), fmt,
                         color=c, **style)
            axes[1].semilogy(ks, _nanmean(ev["resid_var"][valid_m], axis=0),
                             fmt, color=c, **style)
            axes[2].plot(ks, _nanmean(ev["entropy"][valid_m], axis=0), fmt,
                         color=c, **style)

    # One gray legend entry per method drawn in the holdout colors
    for m, style in CV_PLOT_STYLE.items():
        style = dict(style)
        fmt = style.pop("fmt")
        axes[0].plot([], [], fmt, color="gray", label=METHOD_LABEL[m],
                     **style)

    axes[0].set_ylabel("Imputation $R^2$")
    axes[1].set_ylabel("Residual variance fraction")
    axes[2].set_ylabel("Entropy $\\log\\det(R_{\\mathcal{A}})$")
    for ax in axes:
        _style_axis(ax, k_max)

    plt.tight_layout()
    out_path = OUT / f"greedy_cv_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


def plot_entropy_vs_mi(ds_name, evals, k_max):
    """Part 2 figure: entropy, MI, random, greedy trace and facility
    location at the 10% holdout (labels, colors and markers of
    eval_entropy_vs_mi.py)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = np.arange(1, k_max + 1)
    colors, labels, styles = METHOD_COLOR, METHOD_LABEL, METHOD_STYLE

    for method in PLOT_EMI_METHODS:
        c = colors[method]
        ls = styles[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        valid = ~np.all(np.isnan(evals[method]["r2"]), axis=1)
        if not valid.any():
            continue

        mean_r2 = _nanmean(evals[method]["r2"][valid], axis=0)
        std_r2 = _nanstd(evals[method]["r2"][valid], axis=0)
        mean_rv = _nanmean(evals[method]["resid_var"][valid], axis=0)

        axes[0].plot(ks, mean_r2, ls, color=c, label=labels[method],
                     markersize=ms, linewidth=lw)
        axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                             color=c, alpha=0.15)
        axes[1].semilogy(ks, mean_rv, ls, color=c, label=labels[method],
                         markersize=ms, linewidth=lw)

    for method in PLOT_EMI_METHODS:
        c = colors[method]
        ls = styles[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        valid = ~np.all(np.isnan(evals[method]["mi_val"]), axis=1)
        if valid.any():
            mean_mi = _nanmean(evals[method]["mi_val"][valid], axis=0)
            axes[2].plot(ks, mean_mi, ls, color=c, label=labels[method],
                         markersize=ms, linewidth=lw)

    axes[0].set_ylabel("Imputation $R^2$")
    axes[1].set_ylabel("Residual variance fraction")
    axes[2].set_ylabel("Mutual information $I(X_{\\mathcal{A}}; X_{\\bar{\\mathcal{A}}})$")
    for ax in axes:
        _style_axis(ax, k_max)

    plt.tight_layout()
    out_path = OUT / f"entropy_vs_mi_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ── Results file ─────────────────────────────────────────────────

def save_results(ds_name, cv, benchmark_names, k_max, n_folds, n_repeats,
                 estimator):
    """logs/{ds_name}_results.npz in the layout of the module docstring
    (the original cv_p*_* / emi_* keys first, new keys after them) and
    logs/{ds_name}_results_paired.csv."""
    save_dict = {"benchmark_names": benchmark_names}
    for p, res in cv.items():
        prefix = f"cv_p{int(p*100)}"
        evals, meta = res["evals"], res["meta"]
        for method, pre in CV_METHODS.items():
            for metric in CV_METRICS:
                save_dict[f"{prefix}_{pre}{metric}"] = evals[method][metric]
        for method, pre in CV_METHODS.items():
            save_dict[f"{prefix}_{pre}sel"] = evals[method]["selected"]
            save_dict[f"{prefix}_{pre}index_ties"] = evals[method]["index_ties"]
            save_dict[f"{prefix}_{pre}tie_breaks"] = evals[method]["tie_breaks"]
        for key, src in [("rep", "rep"), ("fold", "fold"),
                         ("split_id", "split_id"), ("n_train", "n_train"),
                         ("n_test", "n_val"), ("n_eligible", "n_eligible"),
                         ("eligible", "eligible"), ("failed", "failed"),
                         ("estimator", "estimator")]:
            save_dict[f"{prefix}_{key}"] = meta[src]
    emi = cv[EMI_HOLDOUT]
    for method in EMI_METHODS:
        for metric in EMI_METRICS:
            save_dict[f"emi_{method}_{metric}"] = emi["evals"][method][metric]
    for method in EMI_METHODS:
        for key in ["selected", "index_ties", "tie_breaks"]:
            save_dict[f"emi_{method}_{key}"] = emi["evals"][method][key]
    for key in ["rep", "fold", "split_id", "n_train", "n_val", "n_eligible",
                "eligible"]:
        save_dict[f"emi_{key}"] = emi["meta"][key]
    save_dict["emi_estimation_failed"] = emi["meta"]["failed"]
    save_dict.update(
        dataset=np.array(ds_name), holdout_fracs=np.array(list(cv)),
        emi_holdout_frac=np.array(EMI_HOLDOUT),
        cv_methods=np.array(list(CV_METHODS)),
        cv_method_prefixes=np.array(list(CV_METHODS.values())),
        emi_methods=np.array(EMI_METHODS), k_max=np.array(k_max),
        n_folds=np.array(n_folds), n_repeats=np.array(n_repeats),
        seed=np.array(SEED), **{key: np.array(val) for key, val in
                                protocol_config().items()},
        estimator=np.array(estimator))
    out_path = LOGS / f"{ds_name}_results.npz"
    np.savez(out_path, **save_dict)
    print(f"Results saved to {out_path}", flush=True)
    write_paired_csv(LOGS / f"{ds_name}_results_paired.csv",
                     paired_rows(ds_name, cv, k_max))


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Greedy CV and entropy vs MI on matched splits "
                    "(logs/<dataset>_results.npz).")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="dataset names, data/<name>.matrix.npz "
                             "(default: benchpress)")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help=f"CV repetitions (default {N_REPEATS}; "
                             f"1 = the original seed-42 folds only)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--kmax", type=int, default=K_MAX)
    parser.add_argument("--estimator", choices=["auto", "em", "pairwise"],
                        default="auto",
                        help="correlation estimator (auto: EM if < 90%% "
                             "observed or M_train < N, else pairwise; "
                             "EM for every BenchPress split)")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes over the splits (results do not "
                             "depend on it; default %(default)s)")
    args = parser.parse_args()
    if args.repeats < 1 or args.folds < 2 or args.kmax < 1:
        parser.error("need --repeats >= 1, --folds >= 2, --kmax >= 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    for ds_name in args.datasets:
        if not (DATA / f"{ds_name}.matrix.npz").is_file():
            parser.error(f"no data file {DATA / f'{ds_name}.matrix.npz'}")

    total_t0 = time.time()
    for ds_name in args.datasets:
        ds_t0 = time.time()
        cv, bench_names = run_dataset(ds_name, args.kmax, args.folds,
                                      args.repeats, args.estimator,
                                      args.workers)
        plot_greedy_cv(ds_name, cv, args.kmax)
        plot_entropy_vs_mi(ds_name, cv[EMI_HOLDOUT]["evals"], args.kmax)
        save_results(ds_name, cv, bench_names, args.kmax, args.folds,
                     args.repeats, args.estimator)
        print(f"\n{display_name(ds_name)} completed in "
              f"{time.time() - ds_t0:.0f}s", flush=True)

    total_elapsed = time.time() - total_t0
    print(f"\nAll datasets completed in {total_elapsed:.0f}s", flush=True)
