"""Three reviewer-requested diagnostics (NeurIPS rebuttal, camera-ready rerun).

1. clip   — How often do raw-score imputations leave the valid score
            range, and does clipping them change R²?  (QZ19 Q3; since
            2026-10-18 every script clips, protocol rule D2, and this
            check is the one source of unclipped numbers: the appendix
            table, every main method, k = 1..15)
2. contam — Inflate one benchmark's scores for a subset of models
            (simulated contamination) and observe how the estimated
            correlation and the selections react.  (QZ19 Q2)
3. mono   — Are the greedy MI marginal gains empirically positive
            (monotone regime), and how does Random Greedy compare with
            deterministic greedy?  (QZ19 weakness 2)

Matched design.  clip and mono run on the matched splits of
cv_split.cv_splits(M, folds, repeats, HOLDOUT_FRAC, SEED): --repeats
repetitions (default N_REPEATS = 5) of --folds-fold CV (default 10), i.e.
50 splits per dataset.  Repetition 0 is the original single 10-fold run
(cv_folds(M, 10, 42) and the seed-42 training subsamples), so its
deterministic rows (clip; mono's greedy MI) reproduce the
single-repetition results and --repeats 1 is the original protocol.
Random Greedy is not a reproduction: it replaces the rebuttal-era top-3
heuristic, and its seeds (rg_seed) are new.  On a split every method sees the same training / validation
models, the same estimated correlation matrix (one EM fit per split,
shared by clip and mono within a process), one eligibility mask
(cv_split.eligible_mask: only benchmarks observed for at least
MIN_TRAIN_OBS training models can be selected; ineligible ones stay in the
correlation matrix and the MI complement and remain imputation targets),
one target mask (cv_split.target_mask, protocol rule D1: the cells of
benchmarks observed by fewer than MIN_TARGET_OBS = 2 training models are
never scored) and the split's common random numbers (Split.seed).  Where
fewer than k benchmarks are eligible, the values at that k are NaN (null
in the JSON).
Results carry their split (rep, fold, split_id) so methods pair split by
split.  contam uses the full matrix, with the eligibility mask of all
models.

Methods.  clip: the seven selectors of eval_baselines.run_selection (mi,
entropy, trace, facloc, pca, hub, random: same eligible set and common
random numbers as there).  contam and mono: entropy (greedy_entropy) and mi
(greedy_mi) as in the paper (entropy_rf, greedy entropy from a random
first pivot, was dropped on 2026-10-18, protocol rule D3: no run, key or
macro of it remains, and no other stream changed); contam also ranks the
target in the orders of three more selectors (baseline_select): trace
(greedy_trace_select, maximum explained variance of the unselected
benchmarks), hub (hub_select, mean |correlation|) and facloc
(facility_location_select on the correlation clipped at 0, the paper's
headline method since 2026-10-19; added last, deterministic, so every
earlier key and value of the check is unchanged).  mono concerns MI
alone: facility location is monotone submodular by construction (its
greedy gains are non-negative and non-increasing, test_facloc.py), so it
has no monotonicity question and no Random Greedy counterpart there.

Contamination on EM datasets (rebuttal promise A12).  --contam-dataset
takes several datasets; by default (default_contam_datasets) MMLU and those
of the EM datasets CONTAM_EM_DATASETS (mteb, merged) that are among
--datasets, so --check all writes "contam", "contam_mteb" and
"contam_merged".  estimate_corr uses EM unless the matrix is at least 90%
observed with M >= N (so MMLU is pairwise, MTEB and Merged EM), and the
result records which estimator ran ("estimator", verified by recomputing
it).  The target is the top MI pick among the eligible benchmarks observed
for at least CONTAM_MIN_OBS (= MIN_TRAIN_OBS / 0.1 = 30) models, so that
the 10% contamination is expected to touch MIN_TRAIN_OBS of its scores;
the higher MI picks it passes over are listed under "skipped_mi_picks".
On MMLU (every benchmark observed for all 5,452 models) and MTEB (every
benchmark observed for at least 175) nothing is skipped: the target is the
plain top MI pick.  On Merged, 19 of the 114 columns are observed for
fewer than 30 models (the 8 mt_bench/* columns for 3, 11 helm/* columns
for 17), and the rule skips the first three MI picks, mt_bench/math,
mt_bench/writing (3 models each) and helm/NarrativeQA (17): the target is
arena_hard/score (52 models, MI rank 4).  --contam-min-obs 3 restores the
plain top MI pick.  "configs" holds the fixed contaminated subset (seed
SEED); "draws" repeats every (frac, delta) for --contam-draws random
subsets (draw 0 = that fixed subset, draw d >= 1 default_rng((SEED +
CONTAM_DRAW_SEED_OFFSET, d)), offset 25) and reports the distribution of
the target's rank per method.

Imputation.  impute_rows (clip) solves every validation model separately:
it has no vectorized fast path, so every observed unselected target
benchmark of every validation model with at least one observed selected
benchmark is scored (the H1 fast-path flaw of protocol_changes.md cannot
occur here); it returns the unclipped predictions, and the check scores
them unclipped and clipped to the valid range (score_bounds.clip_z with
the bounds of score_bounds.dataset_bounds: metadata or the documented
per-dataset rule, never observed scores; until 2026-10-18 the range was
inferred from the training models' scores).  mono scores with
eval_entropy_vs_mi.impute_and_evaluate, imported, not copied, with the
split's targets and the dataset's bounds (rules D1 and D2), so its R² has
that function's cell set: every observed unselected target benchmark of
every validation model with at least one observed selected benchmark.
Its vectorized path is taken only when all validation models have
identical observation rows (same_observation_rows), so the H1 flaw (cells
that validation model 0 lacks but others observe went unscored) is fixed
there and mono needs no change of its own.

Random Greedy is Algorithm 1 of Buchbinder, Feldman, Naor & Schwartz,
"Submodular Maximization with Cardinality Constraints" (SODA 2014).  The
ground set is padded with 2k dummy elements of zero marginal gain; at each
of k steps, M_i is a set of k elements (real or dummy) maximizing the sum
of marginal gains w.r.t. the current set, and one element of M_i, drawn
uniformly at random, is added (a dummy adds nothing).  For non-negative
submodular f under |S| ≤ k this guarantees E[f(S_k)] ≥ f(OPT)/e.
f(S) = I(X_S; X_{V\\S}) is non-negative and submodular (not monotone), so
the guarantee applies.  It holds for a fixed target k, so Random Greedy is
run separately for each k in K_RG rather than read off prefixes of a single
run.  Only eligible benchmarks can enter M_i.  NOTE: the rebuttal-era
version of this script instead drew uniformly among the top-3
positive-gain candidates; that heuristic carries no approximation
guarantee, and its numbers are not Random Greedy results.

Ties.  Scores that agree to within TIE_RTOL (greedy_select._near_max) are
tied; every selector breaks such ties by the explained variance of the
unselected benchmarks, then by the lowest column index (greedy_select).
Random Greedy orders tied gains the same way, so the data, and failing
that the column order, never rounding noise, decide which tied element
enters M_i or which one a draw picks.  Such ties occur between
near-duplicate columns, e.g. merged's mt_bench/* (observed for only 3
models; their gains agree to ~1e-14).  n_tie_breaks counts the picks made
among near-tied scores, n_index_ties those the lowest index decided.

Paired statistics.  For each dataset, k and method pair, the R² difference
a − b over the matched splits: mean, Nadeau–Bengio corrected standard
error sqrt(var · (1/n + n_val/n_train)), 95% t-interval and win rate
(paired_stats.paired_stats, fields paired_stats.STAT_FIELDS; pairs
oriented by paired_stats.orient_pair; clip: CLIP_PAIRS at K_EVAL, their
_clipped analogues and m_clipped − m at every k), under the key "paired"
and in one CSV per check and dataset.  Random Greedy
enters with its mean over the N_SEEDS runs of a split.

Outputs (LOGS = path_config.logs_dir(); NAME = reviewer_checks, or
reviewer_checks_TAG with --tag TAG):
  LOGS/NAME.json   every check under its own top-level key ("clip",
                   "contam", "mono"); running one check replaces only that
                   key, so checks can be run separately or in parallel.
                   The contamination check of CONTAM_DATASET (mmlu) is
                   "contam"; that of any other dataset ds is "contam_ds"
                   (e.g. "contam_mteb", written right after "contam"), so
                   an EM-dataset run never replaces the MMLU result.
  LOGS/NAME_clip_{ds}_paired.csv, LOGS/NAME_mono_{ds}_paired.csv
                   paired R² differences (columns PAIRED_CSV_FIELDS)
  LOGS/NAME_clip_{ds}.npz
                   the clip check's per-split arrays (check_clip)

Datasets are addressed by name: --datasets NAME reads data/NAME.matrix.npz
(B with missing = 0, mask O, benchmark_names).

Usage:
    python eval_reviewer_checks.py [--check all|clip|contam|mono]
                                   [--datasets mmlu mteb merged benchpress]
                                   [--repeats 5] [--folds 10]
                                   [--contam-dataset mmlu [mteb merged]]
                                   (default: mmlu, then mteb / merged if
                                   they are among --datasets)
                                   [--contam-draws 20] [--contam-min-obs 30]
                                   [--tag TAG] [--workers 1]

--workers W runs the expensive units in W spawned single-threaded
processes (map_tasks; keep OMP / OpenBLAS / MKL at one thread per
process): the matched splits of clip and mono (clip_split, mono_split;
all datasets of a check in one task list) and the contaminated estimates
of contam (contam_draw, one per draw and (frac, delta)).  Every unit is a
pure function of its dataset and its split or subset, with its own common
random numbers (Split.seed, rg_seed, the draw's permutation stream), and
the results are accumulated and printed in the serial order, so every
output equals the serial run's (the JSON has no workers field: make_numbers
turns all of it into \\Rev macros); test_eval_reviewer_checks checks it.
The clip workers return each split's estimate and the mono workers get
it, so clip and mono still share one EM fit per split.  The checks run
one after another, and contam's uncontaminated estimate stays serial.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import pathlib
import re
import sys
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

try:
    import fcntl
except ImportError:  # non-POSIX: results are still written, without a lock
    fcntl = None

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, logs_dir

DATA = data_dir()
LOGS = logs_dir()
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import (greedy_entropy, greedy_mi, index_tie_count,
                           selection_at_k, tie_break_count, TIE_RTOL,
                           _near_max, _explained_variance,
                           _complement_precision_diag)
import cv_split
import score_bounds
from cv_split import (cv_splits, eligible_mask, target_mask, MIN_TRAIN_OBS,
                      N_REPEATS)
from baseline_select import (facility_location_select, greedy_trace_select,
                             hub_select)
from eval_entropy_vs_mi import (estimate_corr, estimate_corr_em,
                                estimate_corr_pairwise, impute_and_evaluate)
from eval_baselines import run_selection
from score_bounds import clip_z, dataset_bounds
from paired_stats import STAT_FIELDS, orient_pair, paired_stats, write_csv

N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42
K_EVAL = 5
K_MAX = 15
K_RG = (5, 10, 15)    # target sizes for Random Greedy (one run per k)
N_SEEDS = 5           # Random Greedy seeds per split and k
RG_RUN_KEYS = ("fold", "rep", "split_id", "seed", "r2", "n_real", "mi",
               "n_tie_breaks", "n_index_ties", "selected")

DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
CONTAM_DATASET = "mmlu"
# The clip check's methods (eval_baselines.run_selection) and its paired
# comparisons a − b at K_EVAL in paired_stats' orientation (mi − entropy,
# ...); entropy_rf was dropped (D3)
CLIP_METHODS = ("mi", "entropy", "trace", "facloc", "pca", "hub", "random")
CLIP_PAIRS = tuple(orient_pair(a, b) for a, b in
                   (("mi", "entropy"), ("mi", "trace"), ("trace", "facloc"),
                    ("mi", "random"), ("trace", "random")))

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "mteb": "MTEB",
    "merged": "Merged",
    "benchpress": "BenchPress",
}

RESULTS_STEM = "reviewer_checks"
RESULTS_FILE = RESULTS_STEM + ".json"
CHECK_ORDER = ("clip", "contam", "mono")   # top-level key order in the file

PAIRED_CSV_FIELDS = (["dataset", "holdout_frac", "k", "method_a",
                      "method_b"] + STAT_FIELDS)


def display_name(ds):
    """Display name of a dataset (the raw name if none is registered)."""
    return DISPLAY_NAME.get(ds, ds)


def matrix_path(ds):
    return DATA / f"{ds}.matrix.npz"


def load(ds):
    d = np.load(matrix_path(ds), allow_pickle=True)
    return (d["B"].astype(np.float64), d["O"].astype(np.float64),
            d["benchmark_names"])


def _warn(msg):
    print(f"warning: {msg}", file=sys.stderr, flush=True)


# ── Result persistence ───────────────────────────────────────────

def _jsonable(obj):
    """Recursively convert numpy types to native Python for strict JSON.

    Non-finite floats become None (JSON has no NaN/inf); dict keys become
    strings.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if np.isfinite(x) else None
    return obj


def _key_rank(key):
    """Sort key: the checks in CHECK_ORDER first, each followed by its
    per-dataset keys ({check}_{ds}, e.g. contam_mteb) alphabetically, any
    other key after them alphabetically."""
    if key in CHECK_ORDER:
        return CHECK_ORDER.index(key), ""
    for i, check in enumerate(CHECK_ORDER):
        if key.startswith(check + "_"):
            return i, key
    return len(CHECK_ORDER), key


def contam_key(ds):
    """Top-level results key of the contamination check of dataset ds:
    "contam" for CONTAM_DATASET (the MMLU result that make_numbers reads as
    \\RevContam...), "contam_{ds}" for any other dataset."""
    return "contam" if ds == CONTAM_DATASET else f"contam_{ds}"


def save_results(key, payload, name=RESULTS_FILE):
    """Store one check's results under `key` in LOGS/`name` (default
    reviewer_checks.json), keeping the keys written by other checks.

    The read-modify-write holds an exclusive flock on the LOGS directory
    and replaces the file atomically, so checks running as parallel
    processes do not drop each other's keys.  Where flock is unsupported
    (e.g. FUSE / network mounts) it warns and writes without the lock.
    The whole file is re-sanitized (NaN/inf in existing keys become null)
    and its top-level keys are written in CHECK_ORDER, so its bytes do not
    depend on the order in which the checks finished.  A file that is not
    a JSON object is kept as `name`.corrupt (then .corrupt.1, .corrupt.2,
    ...) and a fresh one started.  Returns the path written.
    """
    path = LOGS / name
    tmp = LOGS / f".{name}.{os.getpid()}.tmp"
    fd = os.open(LOGS, os.O_RDONLY)
    try:
        if fcntl is None:
            _warn(f"no fcntl on this platform: writing {path} without a "
                  f"lock; do not run checks in parallel here")
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError as e:
                _warn(f"cannot lock {LOGS} ({e}): writing {path} without "
                      f"a lock; do not run checks in parallel here")
        results = {}
        if path.exists():
            try:
                results = json.loads(path.read_text())
            except ValueError:  # JSONDecodeError, UnicodeDecodeError
                results = None
            if not isinstance(results, dict):
                bad = path.with_name(name + ".corrupt")
                n = 0
                while bad.exists():
                    n += 1
                    bad = path.with_name(f"{name}.corrupt.{n}")
                os.replace(path, bad)
                _warn(f"{path} was not a JSON object; moved to {bad}")
                results = {}
        results[key] = payload
        results = _jsonable({k: results[k]
                             for k in sorted(results, key=_key_rank)})
        try:
            with open(tmp, "w") as f:
                json.dump(results, f, indent=2, allow_nan=False)
                f.write("\n")
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    finally:
        os.close(fd)  # releases the lock
    return path


def write_paired_csv(path, rows):
    """One row per paired comparison (PAIRED_CSV_FIELDS; NaN for undefined
    values), written atomically (paired_stats.write_csv)."""
    write_csv(rows, path, PAIRED_CSV_FIELDS)
    print(f"  paired statistics saved to {path}", flush=True)


def _sha16(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _config(datasets=(), n_folds=N_FOLDS, n_repeats=None, extra_sources=(),
            **settings):
    """Settings and input hashes stored with each check's results.

    source_sha256 covers the code that produces the numbers (the hash of
    greedy_select.py identifies which tie-break produced the selections;
    extra_sources adds further modules, e.g. baseline_select.py for
    contam); data_sha256 covers the score matrices read.  n_repeats (and
    the split layout) is given for the checks that run on the matched
    splits.
    """
    sources = ["greedy_select.py", "eval_entropy_vs_mi.py", "em_cov.py",
               "cv_split.py", pathlib.Path(__file__).name, *extra_sources]
    splits = {}
    if n_repeats is not None:
        splits = {"n_repeats": n_repeats, "n_splits": n_repeats * n_folds,
                  "splits": "cv_split.cv_splits(M, n_folds, n_repeats, "
                            "holdout_frac, seed); rep 0 = cv_folds(M, "
                            "n_folds, seed) with the seed-42 training "
                            "subsamples; split_id = rep * n_folds + fold"}
    return {
        "n_folds": n_folds, "holdout_frac": HOLDOUT_FRAC, "seed": SEED,
        **splits,
        "tie_rtol": TIE_RTOL,
        "min_train_obs": MIN_TRAIN_OBS,
        **settings,
        "source_sha256": {f: _sha16(CODE / f) for f in sources},
        "data_sha256": {f"{ds}.matrix.npz": _sha16(matrix_path(ds))
                        for ds in datasets},
        "numpy": np.__version__,
    }


# ── Matched splits ───────────────────────────────────────────────

_CORR_CACHE = {}


def _corr_key(ds, split):
    return ds, split.train_idx.tobytes()


def split_corr(ds, B, O, split):
    """estimate_corr on the split's training models, memoized per dataset
    and training set, so that clip and mono run in one process (--check
    all) share one EM fit per split.  An estimation failure is memoized
    too and re-raised.  With --workers the clip workers send their
    estimates back and the mono workers receive them (_clip_task,
    _mono_task), so the fits are still shared."""
    key = _corr_key(ds, split)
    if key not in _CORR_CACHE:
        try:
            _CORR_CACHE[key] = estimate_corr(B[split.train_idx],
                                             O[split.train_idx])
        except Exception as e:
            _CORR_CACHE[key] = e
    out = _CORR_CACHE[key]
    if isinstance(out, Exception):
        raise out
    return out


def _add_split(table, split, n_eligible, failed):
    """Append one split's metadata to `table` (one list per field, in
    cv_splits order)."""
    for key, val in [("rep", split.rep), ("fold", split.fold),
                     ("split_id", split.split_id),
                     ("n_train", len(split.train_idx)),
                     ("n_val", len(split.val_idx)),
                     ("n_eligible", n_eligible),
                     ("estimation_failed", failed)]:
        table[key].append(val)


def _ineligible_counts(counts, names):
    """{benchmark name: number of splits in which it was ineligible},
    nonzero entries only."""
    return {names[j]: int(c) for j, c in enumerate(counts) if c}


# ── Worker processes (--workers; spawn) ──────────────────────────

# Worker state: the contexts of the check's datasets, {ds: dict}
_CTX = {}


def _init_worker(ctx):
    _CTX.clear()
    _CTX.update(ctx)


def map_tasks(tasks, serial, worker, ctx, workers=1):
    """The results of every task, in task order: serial(task) in this
    process (workers <= 1), else worker(task) in `workers` spawned
    single-threaded processes that hold ctx (ProcessPoolExecutor.map keeps
    the order).  Every task is a pure function of its dataset's context
    and its split (or contaminated subset), with its own common random
    numbers, so the results do not depend on workers.  A generator: the
    pool shuts down when it is exhausted or closed."""
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield serial(task)
        return
    mp = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                             mp_context=mp, initializer=_init_worker,
                             initargs=(ctx,)) as pool:
        yield from pool.map(worker, tasks)


# ── Paired statistics over matched splits ────────────────────────

def _paired_line(label, st):
    """One printed line of a paired comparison: mean [95% CI], marked '*'
    when the interval excludes 0.  Differences that +.4f would print as
    ±0.0000 (all three values nonzero and below 5e-5 in magnitude, e.g.
    MMLU's mi_clipped − mi, +2e-6 with a CI above 0) are printed as
    +.1e, so a nonzero interval never reads as [+0.0000, +0.0000]."""
    if st["n_splits"] == 0:
        return f"    {label:<34} no common splits"
    vals = (st["mean_diff"], st["ci_low"], st["ci_high"])
    big = max(abs(v) for v in vals)
    fmt = "+.1e" if np.isfinite(big) and 0 < big < 5e-5 else "+.4f"
    mean, lo, hi = (format(v, fmt) for v in vals)
    excl = " *" if st["ci_low"] > 0 or st["ci_high"] < 0 else "  "
    return (f"    {label:<34} {mean} [{lo}, {hi}]{excl} "
            f"win {100 * st['win_rate']:5.1f}%  (n={st['n_splits']})")


def _nan_mean(x):
    """np.nanmean without the all-NaN / empty warning (NaN then)."""
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _nan_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.isfinite(x).any() else np.nan


# ── 1. Clipping ──────────────────────────────────────────────────

def bound_kinds(bounds):
    """(N,) object array: the kind of each benchmark's valid range
    (score_bounds.BOUND_KIND names: unit_interval [0, 1], percent [0, 100],
    correlation [-1, 1], judge_1_10 [1, 10], wb_score [-80, 100],
    unbounded)."""
    return np.array([score_bounds.bound_kind(lo, hi)
                     for lo, hi in zip(bounds.lo, bounds.hi)], dtype=object)


def impute_rows(B_test, O_test, selected, R, col_mean, col_std, N,
                target=None):
    """Per-row conditional-mean imputation; returns flat arrays of
    (z_true, z_pred, benchmark index) over all evaluated cells (None if
    there are none).  z_pred is UNCLIPPED (the clip check applies the
    valid-range clip itself).

    The cells are those of eval_entropy_vs_mi.impute_and_evaluate's
    definition: for every validation model with at least one observed
    selected benchmark, every observed unselected target benchmark
    (target: the split's D1 mask, cv_split.target_mask; None: every
    column), predicted from that model's observed selected benchmarks
    (ridge 0.01, standardized scores clipped at ±10).  Every row is solved
    separately: there is deliberately no vectorized fast path, so no
    model's cells can be dropped for following another model's
    observation pattern (the H1 flaw of protocol_changes.md)."""
    A = set(selected)
    RIDGE = 0.01
    is_target = (np.ones(N, dtype=bool) if target is None
                 else np.asarray(target, dtype=bool))
    B_std = O_test * (B_test - col_mean[None, :]) / col_std[None, :]
    B_std = np.clip(B_std, -10, 10) * O_test

    zt, zp, jj = [], [], []
    for i in range(B_test.shape[0]):
        obs_A = np.array([j for j in selected if O_test[i, j] == 1])
        ev = np.array([j for j in range(N)
                       if j not in A and O_test[i, j] == 1 and is_target[j]])
        if len(obs_A) == 0 or len(ev) == 0:
            continue
        R_oo = R[np.ix_(obs_A, obs_A)] + RIDGE * np.eye(len(obs_A))
        R_eo = R[np.ix_(ev, obs_A)]
        W = np.linalg.solve(R_oo, R_eo.T).T
        zt.append(B_std[i, ev])
        zp.append(W @ B_std[i, obs_A])
        jj.append(ev)
    if not zt:
        return None
    return np.concatenate(zt), np.concatenate(zp), np.concatenate(jj)


def r2(zt, zp):
    return 1.0 - np.sum((zt - zp) ** 2) / max(np.sum(zt ** 2), 1e-12)


def pct(n, total):
    return 100.0 * n / total if total else None


def clip_selections(R, k, split, eligible, names=None):
    """The CLIP_METHODS selections of one split (k picks each), all with
    the split's eligibility mask and common random numbers
    (eval_baselines.run_selection)."""
    return {m: run_selection(m, R, k, names, split, eligible)
            for m in CLIP_METHODS}


def _clip_stats_split(zt, zp, jj, lo_z, hi_z):
    """(r2 unclipped, r2 clipped, n outside the valid range, outside mask)
    of one set of scored cells; the clip is score_bounds.clip_z with the
    split's standardized bounds (lo_z, hi_z: (N,))."""
    zp_clip, n_out = clip_z(zp, lo_z[jj], hi_z[jj])
    outside = (zp < lo_z[jj]) | (zp > hi_z[jj])
    return r2(zt, zp), r2(zt, zp_clip), n_out, outside


def clip_split(c, s):
    """One matched split s of check_clip on the dataset context c (ds, B,
    O, K, bounds, kind, kinds; check_clip).  Returns the split's
    "eligible" mask and "n_eligible", "error" (repr of the exception if
    the covariance estimation failed, else None) and "rows": per method
    of CLIP_METHODS one entry per k = 1..K, None where the selection or
    its scored cells are empty, else r2_raw, r2_clip, n_outside and
    n_cells, at K_EVAL also the outside / total cell counts, per bound
    kind (outside, total), the benchmarks of the outside cells
    (outside_idx), the selection and its tie counts."""
    B, O, K = c["B"], c["O"], c["K"]
    bounds, kind, kinds = c["bounds"], c["kind"], c["kinds"]
    N = B.shape[1]
    eligible = eligible_mask(O[s.train_idx])
    target = target_mask(O[s.train_idx])
    part = {"eligible": eligible, "n_eligible": int(eligible.sum()),
            "error": None, "rows": {}}
    try:
        R, cm, cs = split_corr(c["ds"], B, O, s)
    except Exception as e:
        part["error"] = repr(e)
        return part
    lo_z, hi_z = (np.asarray(b, dtype=np.float64) for b in
                  ((bounds.lo - cm) / cs, (bounds.hi - cm) / cs))
    res = clip_selections(R, K, s, eligible)
    for m in CLIP_METHODS:
        rows = part["rows"][m] = []
        for k in range(1, K + 1):
            sel = selection_at_k(res[m], k)
            out = (None if sel is None else
                   impute_rows(B[s.val_idx], O[s.val_idx], sel, R,
                               cm, cs, N, target))
            if out is None:
                rows.append(None)
                continue
            zt, zp, jj = out
            r2_raw, r2_clip, n_out, outside = _clip_stats_split(
                zt, zp, jj, lo_z, hi_z)
            row = {"r2_raw": r2_raw, "r2_clip": r2_clip, "n_outside": n_out,
                   "n_cells": len(zp)}
            if k == K_EVAL:
                by_kind = {}
                for b in kinds:
                    in_b = kind[jj] == b
                    by_kind[b] = (int((outside & in_b).sum()),
                                  int(in_b.sum()))
                row.update(outside=int(outside.sum()), total=len(zp),
                           by_kind=by_kind, outside_idx=jj[outside],
                           selected=sel,
                           n_tie_breaks=tie_break_count(res[m], k),
                           n_index_ties=index_tie_count(res[m], k))
            rows.append(row)
    return part


def _clip_task(task):
    """clip_split in a worker process; the split's estimate (or its
    failure) goes back with the result, for check_mono."""
    ds, s = task
    part = clip_split(_CTX[ds], s)
    part["corr"] = _CORR_CACHE.pop(_corr_key(ds, s))
    return part


def check_clip(datasets=DATASETS, n_repeats=N_REPEATS, n_folds=N_FOLDS,
               csv_prefix=None, k_max=K_MAX, workers=1):
    """Returns {"config", "datasets": {ds: ...}}: the effect of protocol
    rule D2 (every prediction clipped to its benchmark's valid score
    range, score_bounds.dataset_bounds: metadata or documented rule, never
    observed scores) on the Gaussian conditional mean, the one place where
    unclipped numbers are reported (appendix).

    For every method of CLIP_METHODS (mi, entropy, trace, facloc, hub,
    pca, random; eval_baselines.run_selection on the split's eligible set
    and common random numbers) and k = 1..min(k_max, N): the pooled R² of
    the scored cells (impute_and_evaluate's, with the split's D1 targets)
    unclipped ("raw") and clipped, and the number of predictions outside
    the valid range.  Per split these are written to
    LOGS/NAME_clip_{ds}.npz ({m}_r2_raw, {m}_r2_clip, {m}_n_outside,
    {m}_n_cells (S, K); split metadata rep, fold, split_id, n_train,
    n_val, n_eligible, estimation_failed); the JSON keeps per dataset:

      "bounds" (number of benchmarks per bound kind), "bounds_per_benchmark"
      (kind / lo / hi / n_obs, benchmark order; null = unbounded; the
      source of each bound is in data/score_bounds.json),
      "splits", "ineligible_splits", "skipped_folds",
      "n_splits_too_few_eligible" (at K_EVAL);
      per method at K_EVAL = 5 (the rebuttal-era layout): the outside
      count / total / fraction / percent, "outside_by_kind",
      "outside_fixed_scale[_pct]" (unit_interval and percent kinds),
      "outside_by_benchmark" (nonzero only), per evaluated split "folds" /
      "reps" / "split_ids" / "selected" / "r2_raw" / "r2_clip" /
      "n_tie_breaks" / "n_index_ties", "r2_raw_by_split" /
      "r2_clip_by_split" (aligned with "splits"), and their means / stds;
      "by_k": {method: {"k", "r2_raw_mean", "r2_clip_mean",
      "outside_pct"}} (lists over k; counts in the npz);
      "paired": paired_stats at K_EVAL for the pairs of CLIP_PAIRS,
      unclipped and clipped ({a}_clipped), and of clipped − unclipped per
      method; the CSV NAME_clip_{ds}_paired.csv has clipped − unclipped
      per method and k = 1..K and the method pairs at K_EVAL.

    Each method's cells are its own targets (observed unselected target
    benchmarks of rows with ≥ 1 observed selected benchmark), as in
    impute_and_evaluate, so the methods' totals differ.

    workers > 1 evaluates the splits of all datasets (clip_split) in that
    many processes (map_tasks); the per-split results are accumulated in
    split order, so every output equals the serial run's.
    """
    print("\n" + "=" * 70)
    print(f"CHECK 1: valid-range violations of the imputations and the "
          f"effect of the clip (rule D2), k = 1..{k_max}")
    print("=" * 70)
    results = {"config": _config(datasets, n_folds, n_repeats, k=K_EVAL,
                                 k_max=k_max, methods=list(CLIP_METHODS),
                                 extra_sources=("score_bounds.py",
                                                "eval_baselines.py",
                                                "baseline_select.py"),
                                 bounds_from="score_bounds.dataset_bounds "
                                             "(data/<ds>.meta.json or the "
                                             "documented per-dataset rule; "
                                             "never observed scores)",
                                 min_target_obs=int(
                                     cv_split.MIN_TARGET_OBS)),
               "datasets": {}}
    # Every dataset's splits in one task list, so that --workers keeps all
    # processes busy across datasets; the results come back in order
    ctx, tasks = {}, []
    for ds in datasets:
        B, O, bn = load(ds)
        bn = [str(b) for b in bn]
        M, N = B.shape
        bounds = dataset_bounds(ds, bn, DATA)
        ctx[ds] = {"ds": ds, "B": B, "O": O, "bn": bn, "K": min(k_max, N),
                   "bounds": bounds, "kind": bound_kinds(bounds),
                   "kinds": list(dict.fromkeys(
                       score_bounds.BOUND_KIND.values())),
                   "splits": list(cv_splits(M, n_folds, n_repeats,
                                            HOLDOUT_FRAC, SEED))}
        tasks += [(ds, s) for s in ctx[ds]["splits"]]
    parts = map_tasks(tasks, lambda t: clip_split(ctx[t[0]], t[1]),
                      _clip_task, ctx, workers)
    for ds in datasets:
        dctx = ctx[ds]
        B, O, bn, K = dctx["B"], dctx["O"], dctx["bn"], dctx["K"]
        bounds, kind, kinds = dctx["bounds"], dctx["kind"], dctx["kinds"]
        M, N = B.shape

        stats = {m: {"outside": 0, "total": 0,
                     "outside_by_kind": {b: {"outside": 0, "total": 0}
                                         for b in kinds},
                     "outside_by_benchmark": np.zeros(N, dtype=int),
                     "folds": [], "reps": [], "split_ids": [],
                     "selected": [], "r2_raw": [], "r2_clip": [],
                     "n_tie_breaks": [], "n_index_ties": [],
                     "r2_raw_by_split": [], "r2_clip_by_split": []}
                 for m in CLIP_METHODS}
        splits = defaultdict(list)
        n_ineligible = np.zeros(N, dtype=int)
        skipped, too_few = {}, 0
        S_total = n_repeats * n_folds
        arr = {}
        for m in CLIP_METHODS:
            for key in ("r2_raw", "r2_clip"):
                arr[f"{m}_{key}"] = np.full((S_total, K), np.nan)
            for key in ("n_outside", "n_cells"):
                arr[f"{m}_{key}"] = np.zeros((S_total, K), dtype=np.int64)
        for i, s in enumerate(dctx["splits"]):
            part = next(parts)
            if "corr" in part:          # from a worker: kept for check_mono
                _CORR_CACHE.setdefault(_corr_key(ds, s), part["corr"])
            n_eligible = part["n_eligible"]
            n_ineligible += ~part["eligible"]
            if part["error"] is not None:
                skipped[s.split_id] = part["error"]
                _add_split(splits, s, n_eligible, True)
                for m in CLIP_METHODS:
                    stats[m]["r2_raw_by_split"].append(np.nan)
                    stats[m]["r2_clip_by_split"].append(np.nan)
                continue
            _add_split(splits, s, n_eligible, False)
            too_few += n_eligible < K_EVAL
            for m in CLIP_METHODS:
                for k, row in enumerate(part["rows"][m], start=1):
                    if row is None:
                        if k == K_EVAL:
                            stats[m]["r2_raw_by_split"].append(np.nan)
                            stats[m]["r2_clip_by_split"].append(np.nan)
                        continue
                    for key in ("r2_raw", "r2_clip", "n_outside", "n_cells"):
                        arr[f"{m}_{key}"][i, k - 1] = row[key]
                    if k != K_EVAL:
                        continue
                    s_m = stats[m]
                    s_m["outside"] += row["outside"]
                    s_m["total"] += row["total"]
                    for b in kinds:
                        n_b, t_b = row["by_kind"][b]
                        s_m["outside_by_kind"][b]["outside"] += n_b
                        s_m["outside_by_kind"][b]["total"] += t_b
                    np.add.at(s_m["outside_by_benchmark"], row["outside_idx"],
                              1)
                    s_m["folds"].append(s.fold)
                    s_m["reps"].append(s.rep)
                    s_m["split_ids"].append(s.split_id)
                    s_m["selected"].append(row["selected"])
                    s_m["r2_raw"].append(row["r2_raw"])
                    s_m["r2_clip"].append(row["r2_clip"])
                    s_m["r2_raw_by_split"].append(row["r2_raw"])
                    s_m["r2_clip_by_split"].append(row["r2_clip"])
                    s_m["n_tie_breaks"].append(row["n_tie_breaks"])
                    s_m["n_index_ties"].append(row["n_index_ties"])
                if K < K_EVAL:              # K_EVAL beyond N: no entry yet
                    stats[m]["r2_raw_by_split"].append(np.nan)
                    stats[m]["r2_clip_by_split"].append(np.nan)

        n_splits = len(splits["split_id"])
        for key in ("r2_raw", "r2_clip", "n_outside", "n_cells"):
            for m in CLIP_METHODS:
                arr[f"{m}_{key}"] = arr[f"{m}_{key}"][:n_splits]
        print(f"\n{ds}: {M} models × {N} benchmarks, {n_splits} splits "
              f"({n_repeats} × {n_folds}-fold), eligible per split "
              f"{min(splits['n_eligible'])}–{max(splits['n_eligible'])} "
              f"of {N}; valid ranges: "
              + ", ".join(f"{b} {int((kind == b).sum())}" for b in kinds
                          if (kind == b).any()))
        by_k = {}
        for m in CLIP_METHODS:
            s_m = stats[m]
            s_m["n_splits"] = len(s_m["split_ids"])
            s_m["outside_frac"] = (s_m["outside"] / s_m["total"]
                                   if s_m["total"] else None)
            s_m["outside_pct"] = pct(s_m["outside"], s_m["total"])
            for c in s_m["outside_by_kind"].values():
                c["outside_pct"] = pct(c["outside"], c["total"])
            s_m["outside_fixed_scale"] = sum(
                s_m["outside_by_kind"][b]["outside"]
                for b in ["unit_interval", "percent"])
            s_m["outside_fixed_scale_pct"] = pct(s_m["outside_fixed_scale"],
                                                 s_m["total"])
            s_m["outside_by_benchmark"] = {
                bn[j]: int(c)
                for j, c in enumerate(s_m["outside_by_benchmark"]) if c}
            s_m["r2_raw_mean"] = (float(np.mean(s_m["r2_raw"]))
                                  if s_m["r2_raw"] else None)
            s_m["r2_clip_mean"] = (float(np.mean(s_m["r2_clip"]))
                                   if s_m["r2_clip"] else None)
            s_m["r2_raw_std"] = (float(np.std(s_m["r2_raw"]))
                                 if s_m["r2_raw"] else None)
            s_m["r2_clip_std"] = (float(np.std(s_m["r2_clip"]))
                                  if s_m["r2_clip"] else None)
            n_out = arr[f"{m}_n_outside"].sum(axis=0)
            n_tot = arr[f"{m}_n_cells"].sum(axis=0)
            by_k[m] = {
                "k": list(range(1, K + 1)),
                "r2_raw_mean": [_nan_mean(arr[f"{m}_r2_raw"][:, k])
                                for k in range(K)],
                "r2_clip_mean": [_nan_mean(arr[f"{m}_r2_clip"][:, k])
                                 for k in range(K)],
                "outside_pct": [pct(int(o), int(t))
                                for o, t in zip(n_out, n_tot)]}
            if s_m["total"] == 0:
                continue
            print(f"  {m:<8} k={K_EVAL}: outside the valid range "
                  f"{s_m['outside_pct']:5.2f}% of {s_m['total']} imputations "
                  f"({s_m['outside_fixed_scale_pct']:.2f}% on a 0-1 / 0-100 "
                  f"scale) | R² unclipped {s_m['r2_raw_mean']:.4f} → "
                  f"clipped {s_m['r2_clip_mean']:.4f}")
        print("  mean R² unclipped → clipped (k = 1, 5, 10, "
              f"{K}; % of predictions outside):")
        for m in CLIP_METHODS:
            bk = by_k[m]
            cells = []
            for k in sorted({1, 5, 10, K} & set(bk["k"])):
                o = bk["outside_pct"][k - 1]
                cells.append(f"k={k}: {bk['r2_raw_mean'][k - 1]:+.4f} → "
                             f"{bk['r2_clip_mean'][k - 1]:+.4f} "
                             f"({0.0 if o is None else o:.2f}%)")
            print(f"    {m:<8} " + "  ".join(cells))

        # Paired R² differences over the matched splits
        n_train = np.array(splits["n_train"])
        n_val = np.array(splits["n_val"])
        split_rep = np.array(splits["rep"])
        vals = {}
        for m in CLIP_METHODS:
            vals[m] = np.array(stats[m]["r2_raw_by_split"], dtype=float)
            vals[m + "_clipped"] = np.array(stats[m]["r2_clip_by_split"],
                                            dtype=float)
        pairs = (list(CLIP_PAIRS)
                 + [(a + "_clipped", b + "_clipped") for a, b in CLIP_PAIRS]
                 + [(m + "_clipped", m) for m in CLIP_METHODS])
        paired, rows = {}, []
        print(f"  paired R² differences a − b (k={K_EVAL}; mean "
              f"[Nadeau–Bengio 95% CI], * = CI excludes 0, win rate):")
        for a, b in pairs:
            if K < K_EVAL:
                break
            st = paired_stats(vals[a], vals[b], n_train, n_val, split_rep)
            paired[f"{a}_vs_{b}"] = {"k": K_EVAL, "method_a": a,
                                     "method_b": b, **st}
            rows.append({"dataset": ds, "holdout_frac": HOLDOUT_FRAC,
                         "k": K_EVAL, "method_a": a, "method_b": b, **st})
            print(_paired_line(f"{a} − {b}", st))
        for m in CLIP_METHODS:
            for k in range(1, K + 1):
                if k == K_EVAL:
                    continue
                st = paired_stats(arr[f"{m}_r2_clip"][:, k - 1],
                                  arr[f"{m}_r2_raw"][:, k - 1], n_train,
                                  n_val, split_rep)
                rows.append({"dataset": ds, "holdout_frac": HOLDOUT_FRAC,
                             "k": k, "method_a": m + "_clipped",
                             "method_b": m, **st})
        if csv_prefix is not None:
            write_paired_csv(pathlib.Path(f"{csv_prefix}_clip_{ds}"
                                          f"_paired.csv"), rows)
            npz_path = pathlib.Path(f"{csv_prefix}_clip_{ds}.npz")
            meta = {key: np.array(splits[key]) for key in
                    ("rep", "fold", "split_id", "n_train", "n_val",
                     "n_eligible", "estimation_failed")}
            np.savez(npz_path, **arr, **meta,
                     methods=np.array(CLIP_METHODS),
                     ks=np.arange(1, K + 1), benchmark_names=np.array(bn),
                     bound_lo=bounds.lo, bound_hi=bounds.hi,
                     holdout_frac=np.float64(HOLDOUT_FRAC),
                     min_target_obs=np.int64(cv_split.MIN_TARGET_OBS))
            print(f"  per-split clip arrays saved to {npz_path}", flush=True)
        if too_few:
            print(f"  {too_few} splits with fewer than {K_EVAL} eligible "
                  f"benchmarks (values NaN)")
        if skipped:
            print(f"  skipped splits (covariance estimation failed): "
                  f"{sorted(skipped)}")
        results["datasets"][ds] = {
            "bounds": {b: int((kind == b).sum()) for b in kinds},
            "bounds_per_benchmark": {"kind": list(kind),
                                     "lo": [_finite_or_none(v)
                                            for v in bounds.lo],
                                     "hi": [_finite_or_none(v)
                                            for v in bounds.hi],
                                     "n_obs": O.sum(axis=0).astype(int)},
            "skipped_folds": skipped,
            "splits": dict(splits),
            "ineligible_splits": _ineligible_counts(n_ineligible, bn),
            "n_splits_too_few_eligible": too_few,
            **stats,
            "by_k": by_k,
            "paired": paired,
        }
    parts.close()                       # (shuts the worker pool down)
    return results


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


# ── 2. Contamination ─────────────────────────────────────────────

def selection_rank(order, j):
    return order.index(j) + 1 if j in order else None


def jaccard(a, b, k):
    return len(set(a[:k]) & set(b[:k])) / len(set(a[:k]) | set(b[:k]))


def step0_mi_gain(R, j, eligible=None):
    """Greedy MI's first-step gain of benchmark j,
        Δ(j | ∅) = ½[log R_jj + log (R⁻¹)_jj] = −½ log(1 − R²_{j|rest}),
    with its rank among the eligible benchmarks (all N if eligible is
    None; 1 = greedy MI's first pick) and R²_{j|rest}, the variance of j
    explained by all other benchmarks."""
    N = R.shape[0]
    P = _complement_precision_diag(R, np.arange(N))
    g = 0.5 * (np.log(np.diag(R)) + np.log(P))
    pool = g if eligible is None else g[eligible]
    return {"gain": g[j], "rank": int(1 + (pool > g[j]).sum()),
            "r2_given_rest": 1.0 - 1.0 / (R[j, j] * P[j])}


def contamination_scale(kind, lo, hi):
    """Score unit in which the contamination delta is given, and the cap:
    (1, 1) on a [0,1] scale, (100, 100) on [0,100], (hi − lo, hi) on any
    other bounded range (kind and range of score_bounds.dataset_bounds;
    until 2026-10-18 inferred from the observed scores, the same kinds for
    the three targets: MMLU miscellaneous and MTEB BiorxivClusteringS2S
    [0, 1], Merged arena_hard/score [0, 100]).  MMLU (unit interval) keeps
    the original +delta capped at 1.0.  An unbounded scale has no unit:
    ValueError."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("contamination needs a bounded score scale")
    if kind == "unit_interval":
        return 1.0, 1.0
    if kind == "percent":
        return 100.0, 100.0
    return float(hi - lo), float(hi)


# Contamination grid: fraction of models, delta in score-scale units
CONTAM_FRACS = (0.1, 0.3)
CONTAM_DELTAS = (0.1, 0.2)
# The rebuttal-era methods (their keys are unchanged; entropy_rf, the third
# of them, was dropped on 2026-10-18, rule D3) and the selectors added for
# the camera-ready: trace and hub (promise A12), then facloc (facility
# location, the headline method since 2026-10-19).  Their keys come after
# those of the earlier methods, facloc's last.
CONTAM_METHODS = ("mi", "entropy")
CONTAM_BASELINES = ("trace", "hub", "facloc")
CONTAM_ALL = CONTAM_METHODS + CONTAM_BASELINES
# The target must be observed for at least this many models, so that the
# smallest contaminated fraction is expected to touch MIN_TRAIN_OBS of its
# scores: ceil(3 / 0.1) = 30.  Every MMLU (5,452) and MTEB (>= 175)
# benchmark passes.  On Merged 19 columns fail (8 mt_bench/* with 3
# models, 11 helm/* with 17), and the rule passes over the first three MI
# picks: mt_bench/math, mt_bench/writing (3 models each) and
# helm/NarrativeQA (17); "skipped_mi_picks" lists them.
CONTAM_MIN_OBS = int(np.ceil(round(MIN_TRAIN_OBS / min(CONTAM_FRACS), 9)))
# Random contaminated subsets of the "draws" summary: draw 0 is the fixed
# permutation default_rng(SEED) of "configs"; draw d >= 1 permutes with
# default_rng((SEED + CONTAM_DRAW_SEED_OFFSET, d)).  The tag 25 is distinct
# from every other seed tag in code/: RANDOM_SEED_OFFSET 7,
# ENTROPY_RF_SEED_OFFSET 11, 13-16 (eval_prospective), 18
# (eval_item_budget), 19 (analyze_objectives), 20 (reserved for the
# TabImpute context), 21 (eval_imputers) and 23-24
# (eval_selection_stability).  It was 23 in the first camera-ready version
# of this check, which collided with eval_selection_stability's
# SUB_SEED_OFFSET; draw 0 (the "configs") does not depend on the tag, the
# draws d >= 1 of that version differ from these.
CONTAM_N_DRAWS = 20
CONTAM_DRAW_SEED_OFFSET = 25
# Datasets of the contamination check by default (default_contam_datasets):
# CONTAM_DATASET, then these EM datasets (promise A12) where they are among
# --datasets
CONTAM_EM_DATASETS = ("mteb", "merged")


def default_contam_datasets(datasets):
    """Default --contam-dataset: CONTAM_DATASET (mmlu, always, as before),
    then the CONTAM_EM_DATASETS among `datasets` (--datasets), in
    CONTAM_EM_DATASETS order.  --check all with the default --datasets
    (as run_all.sh's reviewer_checks job calls it) thus writes "contam",
    "contam_mteb" and "contam_merged", and a run restricted to fewer
    datasets (e.g. the driver's DATASETS=mmlu) adds no contamination run
    outside them."""
    return [CONTAM_DATASET] + [ds for ds in CONTAM_EM_DATASETS
                               if ds in datasets and ds != CONTAM_DATASET]


def contam_orders(R, eligible):
    """Complete selection orders (all eligible benchmarks) of every
    CONTAM_ALL method on R: {method: selector result}.  Greedy is
    prefix-consistent, so order[:K_MAX] is the K_MAX-step selection.
    hub is one-shot (mean |corr|), trace adaptive (explained variance),
    facloc adaptive (coverage by the best clipped correlation; draws no
    random numbers)."""
    N = R.shape[0]
    selectors = {
        "mi": greedy_mi, "entropy": greedy_entropy,
        "trace": greedy_trace_select, "hub": hub_select,
        "facloc": facility_location_select,
    }
    return {m: selectors[m](R, N, eligible=eligible) for m in CONTAM_ALL}


def contaminate(B, O, j, models, delta, scale, cap):
    """Copy of B with delta·scale added to benchmark j's observed scores
    of `models`, capped at `cap`.  Missing cells are left as they are (the
    estimators never read them).  Returns (Bc, increase, n_capped):
    the effective increase of each observed contaminated score (models
    order) and the number of those that hit the cap."""
    models = np.asarray(models, dtype=int)
    rows = models[O[models, j] == 1]
    Bc = B.copy()
    Bc[rows, j] = np.minimum(B[rows, j] + delta * scale, cap)
    n_capped = int((B[rows, j] + delta * scale > cap).sum())
    return Bc, Bc[rows, j] - B[rows, j], n_capped


def corr_estimator(B, O, R):
    """Which estimator estimate_corr(B, O) used to produce R, verified by
    recomputing it: "pairwise" if R equals estimate_corr_pairwise(B, O)
    bit for bit, else "em" if it equals estimate_corr_em(B, O), else
    "unknown".  The cheap pairwise estimate is tried first, so a pairwise
    dataset costs no EM fit and an EM dataset one extra fit."""
    M, N = B.shape
    if np.array_equal(estimate_corr_pairwise(B, O)[0], R):
        name = "pairwise"
    elif np.array_equal(estimate_corr_em(B, O)[0], R):
        name = "em"
    else:
        name = "unknown"
    return {"name": name, "verified": name != "unknown",
            "obs_frac": float(O.sum() / (M * N)), "n_models": M,
            "n_benchmarks": N,
            "rule": "eval_entropy_vs_mi.estimate_corr: EM if the observed "
                    "fraction < 0.9 or M < N, else pairwise"}


def _rank(order, j):
    """selection_rank as a float (NaN for a benchmark not in the order)."""
    r = selection_rank(order, j)
    return np.nan if r is None else float(r)


def _draw_record(order0, order1, b_star, n_observed, abs1, N):
    """One contaminated draw of one (frac, delta): the target's rank after
    contamination and the top-5 Jaccard overlap, per CONTAM_ALL method."""
    rec = {"n_models_observed": n_observed,
           "mean_abs_corr_after": abs1 / (N - 1)}
    for m in CONTAM_ALL:
        rec[m + "_rank_after"] = _rank(order1[m], b_star)
        rec[m + "_jaccard_top5"] = jaccard(order0[m], order1[m], 5)
    return rec


def _summarize_draws(frac, delta, n_models, records, ranks_before):
    """Distribution over the draws of one (frac, delta): per method the
    rank after (list), its median / min / max, the fraction of draws in
    which the target moved up (promoted) or down (demoted), ended in the
    top 5 / top K_MAX, and the mean top-5 Jaccard overlap."""
    out = {"frac": frac, "delta": delta, "n_models": n_models,
           "n_models_observed": [r["n_models_observed"] for r in records],
           "mean_abs_corr_after": [r["mean_abs_corr_after"]
                                   for r in records]}
    for m in CONTAM_ALL:
        rk = np.array([r[m + "_rank_after"] for r in records], dtype=float)
        r0 = ranks_before[m]
        r0 = np.nan if r0 is None else float(r0)
        out.update({
            m + "_rank_before": ranks_before[m],
            m + "_rank_after": [None if np.isnan(x) else int(x)
                                for x in rk],
            m + "_rank_after_median": _nan_median(rk),
            m + "_rank_after_min": _nan_int(np.min, rk),
            m + "_rank_after_max": _nan_int(np.max, rk),
            m + "_promoted_frac": float(np.mean(rk < r0)),
            m + "_demoted_frac": float(np.mean(rk > r0)),
            m + "_top5_after_frac": float(np.mean(rk <= 5)),
            m + "_top15_after_frac": float(np.mean(rk <= K_MAX)),
            m + "_jaccard_top5_mean": float(np.mean(
                [r[m + "_jaccard_top5"] for r in records])),
        })
    return out


def _nan_median(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def _nan_int(f, x):
    """f (np.min / np.max) over the finite values of x as an int; None if
    there are none."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return int(f(x)) if x.size else None


def contam_draw(c, subset, delta, full=False):
    """One contaminated estimate of check_contam on the context c (B, O,
    b_star, scale, cap, eligible): the CONTAM_ALL orders on the
    correlation estimated after contaminating `subset` by delta ("res1",
    contam_orders) and the target's |corr| sum ("abs1"); with full (draw
    0, the "configs") also the effective increases, the number capped and
    the target's MI step-0 gain after ("increase", "n_capped", "g1")."""
    B, O, b_star, eligible = c["B"], c["O"], c["b_star"], c["eligible"]
    Bc, increase, n_capped = contaminate(B, O, b_star, subset, delta,
                                         c["scale"], c["cap"])
    Rc, _, _ = estimate_corr(Bc, O)
    out = {"res1": contam_orders(Rc, eligible),
           "abs1": np.abs(Rc[b_star]).sum() - 1.0}
    if full:
        out.update(increase=increase, n_capped=n_capped,
                   g1=step0_mi_gain(Rc, b_star, eligible))
    return out


def _contam_unit(ctx, task):
    """contam_draw of one task (ds, draw d, frac, delta, subset) of
    check_contam; draw 0 is the "configs" one (full)."""
    ds, d, frac, delta, subset = task
    return contam_draw(ctx[ds], subset, delta, full=d == 0)


def _contam_task(task):
    return _contam_unit(_CTX, task)


def check_contam(ds=CONTAM_DATASET, n_draws=CONTAM_N_DRAWS,
                 min_obs=CONTAM_MIN_OBS, workers=1):
    """Returns {"config", "dataset", "benchmark", "benchmark_index",
    "mean_abs_corr_before", "mi_top15_before", "entropy_top15_before",
    "mi_step0_before", "configs": [...]}, one
    config per (frac, delta) with the contaminated models (indices, number
    capped at the scale maximum, mean effective delta), the mean |corr|
    after, the benchmark's MI step-0 gain after, its MI / entropy rank
    before and after, top-5 and top-15 Jaccard overlaps,
    and the top-15 selections before and after.

    Added after those keys (camera-ready, promise A12; every key above and
    its value is unchanged): "estimator" (which estimator estimate_corr
    used, verified by corr_estimator; EM on MTEB / Merged, pairwise on
    MMLU), the target rule ("target_min_obs", "target_n_obs",
    "top_mi_pick", "top_mi_pick_n_obs"; at the end, "skipped_mi_picks":
    the MI picks ahead of the target, each {"benchmark", "mi_rank",
    "n_obs"}, empty where the target is the top MI pick), "ranks_before"
    {method: rank}
    for every CONTAM_ALL method, the trace and hub top-15 orders and their
    tie_breaks (hub also index_ties) before, then the same for facloc
    (facloc_top15_before, facloc_tie_breaks_before,
    facloc_index_ties_before; added 2026-10-19, after every earlier key);
    in each config "n_models_observed" (contaminated models that observe
    the target) and for trace, hub and facloc the same rank / Jaccard /
    top-15 keys as for the other methods plus their ties after; and
    "draws": {"n_draws",
    "seeds", "configs": one _summarize_draws entry per (frac, delta)}, the
    distribution of the target's rank over n_draws contaminated subsets
    (draw 0 = the subset of "configs").

    The analysis uses the full matrix (all models).  Only benchmarks
    observed for at least MIN_TRAIN_OBS models are eligible (eligible_mask
    of all models; contamination changes scores, not the mask), and ranks
    are 1-based positions in the full greedy order over the eligible
    benchmarks (null for an ineligible benchmark); a rank > k_max means
    the benchmark is not among the top-k_max picks (entropy_rf and its
    keys were dropped on 2026-10-18, rule D3).  The contaminated models
    are the first frac·M of one random permutation, so the 10% subset is
    contained in the 30% one and both deltas use the same models: the
    delta and frac effects are not confounded with which models were
    drawn.  Only their observed scores change.  delta is in units of the
    benchmark's score scale (contamination_scale on the valid range of
    score_bounds.dataset_bounds; MMLU: +delta capped at 1.0).  The target
    is the top MI pick among the benchmarks observed for at least min_obs
    models (default CONTAM_MIN_OBS; with min_obs <= MIN_TRAIN_OBS, and on
    MMLU and MTEB, the plain top MI pick).

    workers > 1 evaluates the contaminated estimates (contam_draw, one per
    draw and (frac, delta); every draw has its own permutation stream) in
    that many processes (map_tasks), in order, so the result equals the
    serial run's; the uncontaminated estimate before them stays serial.
    """
    print("\n" + "=" * 70)
    print(f"CHECK 2: benchmark contamination ({display_name(ds)})")
    print("=" * 70)
    B, O, bn = load(ds)
    M, N = B.shape
    bn = [str(b) for b in bn]
    eligible = eligible_mask(O)
    n_obs = O.sum(axis=0).astype(int)

    # Full greedy orders, so that every rank is defined; greedy is
    # prefix-consistent, so order[:K_MAX] is the K_MAX-step selection
    R0, cm, cs = estimate_corr(B, O)
    estimator = corr_estimator(B, O, R0)
    if not estimator["verified"]:
        _warn(f"{ds}: estimate_corr matches neither the pairwise nor the "
              f"EM estimator")
    res0 = contam_orders(R0, eligible)
    ent_res0, mi_res0 = res0["entropy"], res0["mi"]
    order0 = {m: res0[m]["selected"] for m in CONTAM_ALL}
    ent0, mi0 = order0["entropy"], order0["mi"]
    targets = [j for j in mi0 if n_obs[j] >= min_obs]
    if not targets:
        raise ValueError(f"{ds}: no eligible benchmark is observed for at "
                         f"least {min_obs} models")
    b_star = targets[0]
    # MI picks ahead of the target (observed for fewer than min_obs models)
    skipped = [{"benchmark": bn[j], "mi_rank": r + 1, "n_obs": int(n_obs[j])}
               for r, j in enumerate(mi0[:mi0.index(b_star)])]
    abs0 = np.abs(R0[b_star]).sum() - 1.0
    g0 = step0_mi_gain(R0, b_star, eligible)
    bounds = dataset_bounds(ds, bn, DATA)
    kind = bound_kinds(bounds)
    scale, cap = contamination_scale(kind[b_star], bounds.lo[b_star],
                                     bounds.hi[b_star])
    verified = ("verified by recomputation" if estimator["verified"]
                else "NOT VERIFIED")
    print(f"{M} models × {N} benchmarks, {100 * estimator['obs_frac']:.1f}% "
          f"observed; estimator: {estimator['name']} ({verified})")
    if b_star == mi0[0]:
        selected_as = "top MI pick on the uncontaminated matrix"
        how = "top MI pick"
    else:
        mi_rank = selection_rank(mi0, b_star)
        selected_as = (f"top MI pick among the benchmarks observed for at "
                       f"least {min_obs} models (MI rank {mi_rank}; the top "
                       f"MI pick {bn[mi0[0]]} is observed for "
                       f"{n_obs[mi0[0]]})")
        how = (f"MI rank {mi_rank}, top MI pick observed for >= {min_obs} "
               f"models; {bn[mi0[0]]} has {n_obs[mi0[0]]}")
    print(f"contaminated benchmark: {bn[b_star]} "
          f"({how}; mean |corr| = {abs0/(N-1):.3f}, "
          f"R² given the rest = {g0['r2_given_rest']:.3f})")
    if skipped:
        print(f"  skipped higher MI picks (observed for < {min_obs} "
              f"models): " + ", ".join(
                  f"{s['benchmark']} (rank {s['mi_rank']}, {s['n_obs']})"
                  for s in skipped))

    def names(order):
        return [bn[j] for j in order[:K_MAX]]

    results = {
        "config": _config([ds], k_max=K_MAX,
                          extra_sources=("baseline_select.py",),
                          rank="1-based position in the full greedy order "
                               "over the eligible benchmarks",
                          eligibility="eligible_mask of all models",
                          contaminated_models="first frac*M models of "
                                              "rng(seed).permutation(M), "
                                              "shared by both deltas",
                          methods=list(CONTAM_ALL),
                          fracs=list(CONTAM_FRACS),
                          deltas=list(CONTAM_DELTAS),
                          target="top MI pick among the eligible "
                                 "benchmarks observed for at least "
                                 "target_min_obs models",
                          target_min_obs=min_obs,
                          n_draws=n_draws,
                          draw_seeds="draw 0: default_rng(SEED) (the "
                                     "configs); draw d >= 1: "
                                     "default_rng((SEED + "
                                     f"{CONTAM_DRAW_SEED_OFFSET}, d))"),
        "dataset": ds,
        "benchmark": bn[b_star],
        "benchmark_index": b_star,
        "selected_as": selected_as,
        "score_kind": kind[b_star],
        "delta_unit": scale,
        "cap": cap,
        "n_eligible": int(eligible.sum()),
        "mean_abs_corr_before": abs0 / (N - 1),
        "mi_top15_before": names(mi0),
        "entropy_top15_before": names(ent0),
        "entropy_tie_breaks_before": ent_res0.get("tie_breaks"),
        "mi_tie_breaks_before": mi_res0.get("tie_breaks"),
        "mi_index_ties_before": mi_res0.get("index_ties"),
        "mi_step0_before": g0,
        "configs": [],
        # ── added for the camera-ready (A12) ──
        "estimator": estimator,
        "target_min_obs": min_obs,
        "target_n_obs": int(n_obs[b_star]),
        "top_mi_pick": bn[mi0[0]],
        "top_mi_pick_n_obs": int(n_obs[mi0[0]]),
        "ranks_before": {m: selection_rank(order0[m], b_star)
                         for m in CONTAM_ALL},
        "trace_top15_before": names(order0["trace"]),
        "hub_top15_before": names(order0["hub"]),
        "trace_tie_breaks_before": res0["trace"]["tie_breaks"],
        "hub_tie_breaks_before": res0["hub"]["tie_breaks"],
        "hub_index_ties_before": res0["hub"]["index_ties"],
    }
    # ── added 2026-10-19: facility location (headline method) ──
    if "facloc" in CONTAM_ALL:
        results.update({
            "facloc_top15_before": names(order0["facloc"]),
            "facloc_tie_breaks_before": res0["facloc"]["tie_breaks"],
            "facloc_index_ties_before": res0["facloc"]["index_ties"]})

    # Every contaminated estimate (contam_draw) in one task list, in the
    # order of the loops below: draw 0 (the "configs", permutation
    # default_rng(SEED)), then the draws d >= 1, each with its own
    # permutation stream; with --workers they run in parallel and come
    # back in this order
    perms = [np.random.default_rng(SEED).permutation(M)]
    perms += [np.random.default_rng(
        (SEED + CONTAM_DRAW_SEED_OFFSET, d)).permutation(M)
        for d in range(1, n_draws)]
    tasks = [(ds, d, frac, delta, np.sort(perm_d[:int(frac * M)]))
             for d, perm_d in enumerate(perms)
             for frac in CONTAM_FRACS for delta in CONTAM_DELTAS]
    ctx = {ds: {"B": B, "O": O, "b_star": b_star, "scale": scale,
                "cap": cap, "eligible": eligible}}
    units = map_tasks(tasks, lambda t: _contam_unit(ctx, t), _contam_task,
                      ctx, workers)
    perm = perms[0]
    draw0 = {}
    for frac in CONTAM_FRACS:
        subset = np.sort(perm[:int(frac * M)])
        obs = O[subset, b_star] == 1
        for delta in CONTAM_DELTAS:
            unit = next(units)
            increase, n_capped = unit["increase"], unit["n_capped"]
            res1 = unit["res1"]
            ent_res1, mi_res1 = res1["entropy"], res1["mi"]
            order1 = {m: res1[m]["selected"] for m in CONTAM_ALL}
            ent1, mi1 = order1["entropy"], order1["mi"]
            abs1 = unit["abs1"]
            g1 = unit["g1"]

            cfg = {
                "frac": frac, "delta": delta, "n_models": len(subset),
                "model_indices": subset,
                "n_capped": n_capped,
                "effective_delta_mean": (float(increase.mean())
                                         if increase.size else None),
                "mean_abs_corr_before": abs0 / (N - 1),
                "mean_abs_corr_after": abs1 / (N - 1),
                "mi_step0_after": g1,
                "mi_rank_before": selection_rank(mi0, b_star),
                "mi_rank_after": selection_rank(mi1, b_star),
                "entropy_rank_before": selection_rank(ent0, b_star),
                "entropy_rank_after": selection_rank(ent1, b_star),
                "mi_jaccard_top5": jaccard(mi0, mi1, 5),
                "mi_jaccard_top15": jaccard(mi0, mi1, K_MAX),
                "entropy_jaccard_top5": jaccard(ent0, ent1, 5),
                "entropy_jaccard_top15": jaccard(ent0, ent1, K_MAX),
                "mi_top15_before": names(mi0),
                "mi_top15_after": names(mi1),
                "entropy_top15_before": names(ent0),
                "entropy_top15_after": names(ent1),
                "entropy_tie_breaks_after": ent_res1.get("tie_breaks"),
                "mi_tie_breaks_after": mi_res1.get("tie_breaks"),
                "mi_index_ties_after": mi_res1.get("index_ties"),
            }
            # ── added for the camera-ready (A12) ──
            cfg["n_models_observed"] = int(obs.sum())
            for m in CONTAM_BASELINES:
                cfg.update({
                    m + "_rank_before": selection_rank(order0[m], b_star),
                    m + "_rank_after": selection_rank(order1[m], b_star),
                    m + "_jaccard_top5": jaccard(order0[m], order1[m], 5),
                    m + "_jaccard_top15": jaccard(order0[m], order1[m],
                                                  K_MAX),
                    m + "_top15_before": names(order0[m]),
                    m + "_top15_after": names(order1[m]),
                })
            cfg["trace_tie_breaks_after"] = res1["trace"]["tie_breaks"]
            cfg["hub_tie_breaks_after"] = res1["hub"]["tie_breaks"]
            cfg["hub_index_ties_after"] = res1["hub"]["index_ties"]
            if "facloc" in CONTAM_ALL:
                cfg["facloc_tie_breaks_after"] = res1["facloc"]["tie_breaks"]
                cfg["facloc_index_ties_after"] = res1["facloc"]["index_ties"]
            results["configs"].append(cfg)
            draw0[frac, delta] = _draw_record(order0, order1, b_star,
                                              int(obs.sum()), abs1, N)

            eff = cfg["effective_delta_mean"]
            print(f"\n  {int(frac*100)}% of models +{delta:.2f} on "
                  f"{bn[b_star]} ({n_capped}/{int(obs.sum())} capped at "
                  f"{cap:g}; mean effective "
                  f"+{(np.nan if eff is None else eff) / scale:.3f}):")
            print(f"    mean |corr| of {bn[b_star]}: "
                  f"{abs0/(N-1):.3f} → {abs1/(N-1):.3f}")
            print(f"    MI step-0 gain:        {g0['gain']:.3f} "
                  f"(rank {g0['rank']}) → {g1['gain']:.3f} "
                  f"(rank {g1['rank']});  R² given the rest "
                  f"{g0['r2_given_rest']:.3f} → {g1['r2_given_rest']:.3f}")
            for m in CONTAM_ALL:
                label = "MI" if m == "mi" else m
                print(f"    rank in {label + ' order:':<19}"
                      f"{cfg[m + '_rank_before']} → "
                      f"{cfg[m + '_rank_after']}"
                      f"   (top-5 Jaccard {cfg[m + '_jaccard_top5']:.2f}, "
                      f"top-15 {cfg[m + '_jaccard_top15']:.2f})")
    print(f"\n  (ranks are positions in the full greedy order over the "
          f"{int(eligible.sum())} eligible of {N} benchmarks; > {K_MAX} = "
          f"not among the top {K_MAX})")

    # Rank distribution over n_draws contaminated subsets (draw 0 above)
    records = {key: [rec] for key, rec in draw0.items()}
    n0 = len(CONTAM_FRACS) * len(CONTAM_DELTAS)
    for (_, d, frac, delta, subset), unit in zip(tasks[n0:], units):
        n_observed = int((O[subset, b_star] == 1).sum())
        order1 = {m: unit["res1"][m]["selected"] for m in CONTAM_ALL}
        records[frac, delta].append(_draw_record(
            order0, order1, b_star, n_observed, unit["abs1"], N))
    units.close()                       # (shuts the worker pool down)
    results["draws"] = {
        "n_draws": n_draws,
        "seeds": results["config"]["draw_seeds"],
        "configs": [_summarize_draws(frac, delta, int(frac * M),
                                     records[frac, delta],
                                     results["ranks_before"])
                    for frac in CONTAM_FRACS for delta in CONTAM_DELTAS],
    }
    results["skipped_mi_picks"] = skipped
    print(f"\n  rank of {bn[b_star]} over {n_draws} contaminated subsets "
          f"(before → median [min–max] after; % of draws promoted / "
          f"demoted / in the top 5 / top {K_MAX}):")
    for summ in results["draws"]["configs"]:
        n_ob = summ["n_models_observed"]
        print(f"    {int(summ['frac'] * 100)}% +{summ['delta']:.2f} "
              f"({min(n_ob)}–{max(n_ob)} contaminated scores):")
        for m in CONTAM_ALL:
            print(f"      {m:<11}{summ[m + '_rank_before']!s:>4} → "
                  f"{summ[m + '_rank_after_median']:5.1f} "
                  f"[{summ[m + '_rank_after_min']}–"
                  f"{summ[m + '_rank_after_max']}]   "
                  f"promoted {100 * summ[m + '_promoted_frac']:5.1f}%  "
                  f"demoted {100 * summ[m + '_demoted_frac']:5.1f}%  "
                  f"top-5 {100 * summ[m + '_top5_after_frac']:5.1f}%  "
                  f"top-{K_MAX} {100 * summ[m + '_top15_after_frac']:5.1f}%")
    return results


# ── 3. MI monotonicity + Random Greedy ───────────────────────────

def _mi_gains(Sigma, d, selected_set, eligible=None):
    """MI marginal gains Δ(v | S) = ½[log d_v + log P_vv] for v ∉ S, with
    greedy_mi's arithmetic (-inf where d_v or P_vv underflows, for v ∈ S
    and for ineligible v); d holds the conditional variances given S.  The
    complement V \\ S of P_vv contains every unselected benchmark,
    eligible or not."""
    N = len(d)
    comp = np.array(sorted(set(range(N)) - selected_set))
    P = np.zeros(N)
    P[comp] = _complement_precision_diag(Sigma, comp)
    gains = np.full(N, -np.inf)
    for j in comp:
        if d[j] > 1e-300 and P[j] > 1e-300:
            gains[j] = 0.5 * (np.log(d[j]) + np.log(P[j]))
    if eligible is not None:
        gains[~eligible] = -np.inf
    return gains


def _cholesky_step(Sigma, L, d, t, j_star, selected_set):
    """Pivot j_star into column t of the forward Cholesky state (L, d), as
    in greedy_mi; selected_set must already contain j_star."""
    sqrt_d = np.sqrt(max(d[j_star], 1e-300))
    for j in range(len(d)):
        if j in selected_set and j != j_star:
            continue
        L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
        if j != j_star:
            d[j] -= L[j, t] ** 2


def _order_tied(Sigma, L, d, t, cand, unsel):
    """Order near-tied candidates `cand` (ascending indices) as repeated
    greedy_select._tie_break calls would at this state: by the residual
    variance of the unselected set `unsel` each explains (descending;
    values within TIE_RTOL · max(ev_max, max_i Σ_ii) tied), ties of that
    by lowest index.  Returns (ordered indices, n_index per element), where
    n_index > 1 is the size of the explained-variance tie group whose
    order the column index decided."""
    scale = float(np.max(np.diag(Sigma)))
    ev = _explained_variance(Sigma, L, d, t, cand, unsel)
    rest = np.arange(len(cand))
    order, n_index = [], []
    while rest.size:
        tied = _near_max(ev[rest], scale)
        group = rest[tied]
        order.extend(int(j) for j in cand[group])
        n_index.extend([len(group)] * len(group))
        rest = rest[~tied]
    return order, n_index


def _rg_candidates(gains, k, order_tied=None):
    """Real elements of Random Greedy's M_i, with the size of each one's
    tie group of gains and of its index-tie group.

    Elements with gain > TIE_RTOL, best first, at most k of them.  Gains
    within TIE_RTOL (absolute below 1, relative above; see _near_max) of
    the best remaining one form a tie group; a gain within TIE_RTOL of 0 is
    no better than a dummy's.  A tie group is ordered by order_tied(group)
    -> (ordered group, n_index) when given (explained variance, then
    index: _order_tied), else by index.  The order, and so which tied
    elements fill the last places of M_i, is fixed by the data and the
    column order, not by rounding noise.
    """
    cand = np.flatnonzero(gains > TIE_RTOL)
    real, n_tied, n_index = [], [], []
    while cand.size and len(real) < k:
        tied = _near_max(gains[cand], scale=1.0)
        group = [int(j) for j in cand[tied]]
        if len(group) == 1:
            n_idx = [1]
        elif order_tied is not None:
            group, n_idx = order_tied(cand[tied])
        else:
            n_idx = [len(group)] * len(group)
        real.extend(group)
        n_tied.extend([len(group)] * len(group))
        n_index.extend(n_idx)
        cand = cand[~tied]
    return real[:k], n_tied[:k], n_index[:k]


def randomized_greedy_mi(Sigma, k, *, seed=0, eligible=None,
                         return_ties=False):
    """Random Greedy (Buchbinder, Feldman, Naor & Schwartz, SODA 2014,
    Alg. 1) for f(S) = I(X_S; X_{V\\S}) under |S| ≤ k.

    The ground set is padded with 2k dummy elements whose marginal gain is
    always 0.  At step i, M_i is a set of k elements maximizing the sum of
    marginal gains Δ(v | S_{i-1}) = ½[log σ²_{v|S} + log P_vv] (as in
    greedy_mi): the real elements with positive gain, best first, at most
    k of them, topped up with dummies.  Real elements with gain ≤ 0 never
    beat a dummy, and at most i-1 of the 2k dummies are used, so M_i can
    always be filled.  u_i is drawn uniformly from M_i; drawing a dummy
    leaves the selection unchanged.  For non-negative submodular f this
    gives E[f(S_k)] ≥ f(OPT)/e for this k; prefixes of a run carry no
    guarantee for smaller k.

    With an eligibility mask only eligible benchmarks can enter M_i (the
    ground set is the eligible benchmarks; ineligible ones stay in the
    complement V \\ S of the objective).

    Gains equal to within TIE_RTOL count as tied and are ordered by
    explained variance, then column index (_rg_candidates, _order_tied),
    so M_i maximizes the sum of gains to within k·TIE_RTOL and does not
    depend on rounding noise.  Any maximizing set is a valid M_i, and the
    uniform draw does not depend on its order, so the guarantee is
    unaffected.

    seed is keyword-only (the rebuttal-era top-3 heuristic took top_m as
    the third argument).

    Returns
    -------
    list of the real indices selected, in the order added
    (length ≤ k; k - length dummies were drawn).  With return_ties,
    (selected, index_ties, tie_breaks): tie_breaks lists (step, n_tied)
    for each step whose drawn element was one of n_tied > 1 elements with
    near-tied gains, index_ties (step, n_index) for those at which n_index
    > 1 of them also tied on explained variance, so the column order
    decided which of them was added.
    """
    N = Sigma.shape[0]
    assert Sigma.shape == (N, N)
    assert k >= 1
    if eligible is not None:
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.shape != (N,):
            raise ValueError(f"eligible has shape {eligible.shape}, "
                             f"expected ({N},)")
    rng = np.random.default_rng(seed)
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, k))
    selected, selected_set = [], set()
    index_ties, tie_breaks = [], []

    def order_tied(cand):
        unsel = np.array(sorted(set(range(N)) - selected_set))
        return _order_tied(Sigma, L, d, len(selected), cand, unsel)

    for i in range(k):
        gains = _mi_gains(Sigma, d, selected_set, eligible)
        # M_i = positive-gain real elements (best first, ties by explained
        # variance, then index) followed by dummies at positions
        # len(real) .. k-1
        real, n_tied, n_index = _rg_candidates(gains, k, order_tied)
        u = int(rng.integers(k))
        if u >= len(real):
            continue  # dummy drawn: f(S + dummy) = f(S)
        j_star = real[u]
        if n_tied[u] > 1:
            tie_breaks.append((i, n_tied[u]))
        if n_index[u] > 1:
            index_ties.append((i, n_index[u]))

        # Forward Cholesky update (column = number of real picks so far)
        selected_set.add(j_star)
        _cholesky_step(Sigma, L, d, len(selected), j_star, selected_set)
        selected.append(j_star)
    if return_ties:
        return selected, index_ties, tie_breaks
    return selected


def rg_seed(split, k, seed_idx):
    """Random Greedy seed of one run: (SEED, fold, k, seed_idx) in rep 0,
    (SEED, rep, fold, k, seed_idx) in rep r > 0 (rep 0 omits the rep, as
    cv_split.split_seed does).

    These seeds are new, also in rep 0: the rebuttal-era script seeded its
    top-3 heuristic (not Random Greedy) with (SEED, fold, s), s < 3, one
    run for all k, so rep-0 Random Greedy draws do not reproduce any
    earlier log.  They are the same across scripts, splits and reps by
    construction, and distinct per (split, k, seed_idx)."""
    if split.rep == 0:
        return (SEED, int(split.fold), int(k), int(seed_idx))
    return (SEED, int(split.rep), int(split.fold), int(k), int(seed_idx))


def _logdet(A):
    """log det of a PSD block; falls back to eigenvalues (clipped at
    1e-300) when slogdet's sign is not positive, as greedy_mi does."""
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        logdet = np.sum(np.log(np.maximum(np.linalg.eigvalsh(A), 1e-300)))
    return logdet


def mi_value(Sigma, selected):
    """I(X_S; X_{V\\S}) = ½[log det Σ_S + log det Σ_{V\\S} − log det Σ];
    0 for S = ∅ or S = V."""
    N = Sigma.shape[0]
    sel = list(selected)
    comp = sorted(set(range(N)) - set(sel))
    if not sel or not comp:
        return 0.0
    return 0.5 * (_logdet(Sigma[np.ix_(sel, sel)])
                  + _logdet(Sigma[np.ix_(comp, comp)]) - _logdet(Sigma))


def mono_split(c, s):
    """One matched split s of check_mono on the dataset context c (ds, B,
    O, ks, k_max, bounds; check_mono).  Returns the split's "eligible"
    mask and "n_eligible", "error" (repr of the exception if the
    covariance estimation failed, else None) and otherwise "greedy" (greedy
    MI's selected / gains / mi_values / tie_breaks / index_ties), "det"
    {k: the deterministic n_index_ties, n_tie_breaks, r2, mi} and "rnd"
    {k: one dict per Random Greedy run, keys RG_RUN_KEYS}, in the order in
    which check_mono appends them.  Random Greedy draws from the split's
    own seeds (rg_seed)."""
    B, O, ks, bounds = c["B"], c["O"], c["ks"], c["bounds"]
    N = B.shape[1]
    eligible = eligible_mask(O[s.train_idx])
    target = target_mask(O[s.train_idx])         # D1
    part = {"eligible": eligible, "n_eligible": int(eligible.sum()),
            "error": None}
    try:
        R, cm, cs = split_corr(c["ds"], B, O, s)
    except Exception as e:
        part["error"] = repr(e)
        return part
    res = greedy_mi(R, c["k_max"], eligible=eligible)
    part["greedy"] = {key: res[key] for key in
                      ("selected", "gains", "mi_values", "tie_breaks",
                       "index_ties")}
    B_val, O_val = B[s.val_idx], O[s.val_idx]
    det, rnd = {}, {}
    part.update(det=det, rnd=rnd)

    for k in ks:
        # Deterministic greedy is prefix-consistent: top k of the
        # k_max run is the k-step greedy solution
        sel = selection_at_k(res, k)
        dk = det[k] = {"n_index_ties": index_tie_count(res, k),
                       "n_tie_breaks": sum(1 for t, _ in res["tie_breaks"]
                                           if t < k)}
        runs = rnd[k] = []
        if sel is None:   # fewer than k eligible benchmarks
            dk["r2"], dk["mi"] = np.nan, np.nan
            continue
        imp = impute_and_evaluate(B_val, O_val, sel, R, cm, cs, N,
                                  target=target, bounds=bounds)
        dk["r2"] = imp["r2"]
        dk["mi"] = mi_value(R, sel)

        for si in range(N_SEEDS):
            sel, rg_index, rg_breaks = randomized_greedy_mi(
                R, k, seed=rg_seed(s, k, si), eligible=eligible,
                return_ties=True)
            imp = impute_and_evaluate(B_val, O_val, sel, R, cm, cs,
                                      N, target=target,
                                      bounds=bounds)
            runs.append({"fold": s.fold, "rep": s.rep,
                         "split_id": s.split_id, "seed": si,
                         "r2": imp["r2"], "n_real": len(sel),
                         "mi": mi_value(R, sel),
                         "n_tie_breaks": len(rg_breaks),
                         "n_index_ties": len(rg_index), "selected": sel})
    return part


def _mono_task(task):
    """mono_split in a worker process, with the split's estimate from the
    clip workers (None: estimated here)."""
    ds, s, corr = task
    key = _corr_key(ds, s)
    if corr is not None:
        _CORR_CACHE[key] = corr
    try:
        return mono_split(_CTX[ds], s)
    finally:
        _CORR_CACHE.pop(key, None)


def check_mono(datasets=DATASETS, n_repeats=N_REPEATS, n_folds=N_FOLDS,
               csv_prefix=None, workers=1):
    """Returns {"config", "datasets": {ds: ...}}.  Per dataset: "splits"
    (one entry per matched split: rep, fold, split_id, n_train, n_val,
    n_eligible, estimation_failed), "ineligible_splits"; for each split
    whose estimation succeeded (in "folds" / "reps" / "split_ids") the
    greedy MI selection (eligible benchmarks only), marginal gains and MI
    values (k ≤ K_MAX; NaN beyond the eligible benchmarks) and its
    tie_breaks / index_ties; the minimum gain and the number of steps with
    gain ≤ 0; and for each k in K_RG the deterministic-greedy R² / MI
    per split (aligned with "split_ids"; NaN where fewer than k benchmarks
    are eligible) and the Random Greedy R² / |S| / MI / selection per
    (split, seed) run (only on splits with at least k eligible benchmarks,
    so both methods cover the same splits), the seed-mean per split
    ("r2_split_mean", "mi_split_mean", aligned with "split_ids"), and
    summaries.  "paired"[k] holds paired_stats of deterministic minus
    seed-mean Random Greedy R² ("r2") and MI ("mi") over the splits.

    Ties: per split, n_tie_breaks / n_index_ties count the deterministic
    steps < k made among near-tied gains / decided by the lowest index
    (greedy_mi's own tie-break); per Random Greedy run, the drawn elements
    that were near-tied / ordered by index (randomized_greedy_mi).

    workers > 1 evaluates the splits of all datasets (mono_split) in that
    many processes (map_tasks), with the estimates check_clip cached; the
    per-split results are appended in split order, so every output equals
    the serial run's."""
    print("\n" + "=" * 70)
    print(f"CHECK 3: MI gain positivity (k≤{K_MAX}) and Random Greedy "
          f"(Buchbinder et al. 2014)")
    print("=" * 70)

    results = {"config": _config(datasets, n_folds, n_repeats, k_max=K_MAX,
                                 k_random_greedy=list(K_RG),
                                 n_seeds=N_SEEDS,
                                 rg_seed="(SEED, fold, k, seed_idx) in rep "
                                         "0, (SEED, rep, fold, k, seed_idx) "
                                         "in rep r > 0; new seeds (the "
                                         "rebuttal-era top-3 heuristic used "
                                         "(SEED, fold, s)), so rep 0 does "
                                         "not reproduce earlier draws"),
               "datasets": {}}
    # Every dataset's splits in one task list (as in check_clip), each
    # with the estimate check_clip left in the cache, if any
    ctx, tasks = {}, []
    for ds in datasets:
        B, O, bn = load(ds)
        bn = [str(b) for b in bn]
        M, N = B.shape
        ctx[ds] = {"ds": ds, "B": B, "O": O, "bn": bn,
                   "ks": [k for k in K_RG if k <= N], "k_max": min(K_MAX, N),
                   "bounds": dataset_bounds(ds, bn, DATA),          # D2
                   "splits": list(cv_splits(M, n_folds, n_repeats,
                                            HOLDOUT_FRAC, SEED))}
        tasks += [(ds, s, _CORR_CACHE.get(_corr_key(ds, s)))
                  for s in ctx[ds]["splits"]]
    parts = map_tasks(tasks, lambda t: mono_split(ctx[t[0]], t[1]),
                      _mono_task, ctx, workers)
    for ds in datasets:
        dctx = ctx[ds]
        B, O, bn, ks = dctx["B"], dctx["O"], dctx["bn"], dctx["ks"]
        M, N = B.shape

        out = {"benchmark_names": bn, "folds": [], "reps": [],
               "split_ids": [], "skipped_folds": {}, "greedy_selected": [],
               "greedy_gains": [], "greedy_mi_values": [],
               "greedy_tie_breaks": [], "greedy_index_ties": []}
        splits = defaultdict(list)
        n_ineligible = np.zeros(N, dtype=int)
        det = {k: defaultdict(list) for k in ks}
        rnd = {k: defaultdict(list) for k in ks}
        for s in dctx["splits"]:
            part = next(parts)
            n_eligible = part["n_eligible"]
            n_ineligible += ~part["eligible"]
            if part["error"] is not None:
                out["skipped_folds"][s.split_id] = part["error"]
                _add_split(splits, s, n_eligible, True)
                continue
            _add_split(splits, s, n_eligible, False)
            out["folds"].append(s.fold)
            out["reps"].append(s.rep)
            out["split_ids"].append(s.split_id)
            for key in ("selected", "gains", "mi_values", "tie_breaks",
                        "index_ties"):
                out[f"greedy_{key}"].append(part["greedy"][key])
            for k in ks:
                for key, v in part["det"][k].items():
                    det[k][key].append(v)
                for run in part["rnd"][k]:
                    for key, v in run.items():
                        rnd[k][key].append(v)

        out["splits"] = dict(splits)
        out["ineligible_splits"] = _ineligible_counts(n_ineligible, bn)
        if not out["folds"]:
            print(f"\n{ds}: all splits skipped (covariance estimation "
                  f"failed)")
            results["datasets"][ds] = out
            continue

        gains = np.concatenate(out["greedy_gains"])
        gains = gains[~np.isnan(gains)]   # NaN: beyond the eligible ones
        out["min_gain"] = float(gains.min())
        out["n_nonpositive_steps"] = int((gains <= 0).sum())
        out["n_steps"] = len(gains)
        out["deterministic"], out["random_greedy"] = {}, {}
        out["paired"] = {}
        sid_row = {sid: i for i, sid in enumerate(splits["split_id"])}
        rows_of = [sid_row[sid] for sid in out["split_ids"]]
        n_train = np.array(splits["n_train"])[rows_of]
        n_val = np.array(splits["n_val"])[rows_of]
        split_rep = np.array(splits["rep"])[rows_of]
        csv_rows = []
        for k in ks:
            dk, rk = dict(det[k]), dict(rnd[k])
            for key in RG_RUN_KEYS:   # k above every split's eligible set
                rk.setdefault(key, [])
            run_sid = np.array(rk["split_id"], dtype=int)
            for key in ("r2", "mi"):
                x = np.array(rk[key], dtype=float)
                rk[f"{key}_split_mean"] = [
                    _nan_mean(x[run_sid == sid]) for sid in out["split_ids"]]
            n_runs = len(rk["r2"])
            dk.update(r2_mean=_nan_mean(dk["r2"]), r2_std=_nan_std(dk["r2"]),
                      mi_mean=_nan_mean(dk["mi"]),
                      n_splits=int(np.isfinite(dk["r2"]).sum()),
                      n_folds_with_index_ties=int(
                          np.count_nonzero(dk["n_index_ties"])),
                      n_folds_with_tie_breaks=int(
                          np.count_nonzero(dk["n_tie_breaks"])))
            rk_mi = _nan_mean(rk["mi"])
            rk.update(r2_mean=_nan_mean(rk["r2"]), r2_std=_nan_std(rk["r2"]),
                      mi_mean=rk_mi,
                      # det MI is 0 when k = N (S = V)
                      mi_ratio_to_deterministic=(rk_mi / dk["mi_mean"]
                                                 if dk["mi_mean"] != 0
                                                 else np.nan),
                      n_real_mean=_nan_mean(rk["n_real"]),
                      n_real_min=(int(np.min(rk["n_real"])) if n_runs
                                  else None),
                      n_runs=n_runs,
                      n_r2_nan=int(np.isnan(np.array(rk["r2"], dtype=float))
                                   .sum()),
                      n_runs_with_index_ties=int(
                          np.count_nonzero(rk["n_index_ties"])),
                      n_runs_with_tie_breaks=int(
                          np.count_nonzero(rk["n_tie_breaks"])))
            out["deterministic"][k], out["random_greedy"][k] = dk, rk
            out["paired"][k] = {}
            for key in ("r2", "mi"):
                st = paired_stats(dk[key], rk[f"{key}_split_mean"],
                                  n_train, n_val, split_rep)
                out["paired"][k][key] = {"method_a": "mi",
                                         "method_b": "random_greedy", **st}
                if key == "r2":
                    csv_rows.append({"dataset": ds,
                                     "holdout_frac": HOLDOUT_FRAC, "k": k,
                                     "method_a": "mi",
                                     "method_b": "random_greedy", **st})
        results["datasets"][ds] = out

        n_el = splits["n_eligible"]
        print(f"\n{ds}: {len(out['folds'])} splits ({n_repeats} × "
              f"{n_folds}-fold), eligible per split {min(n_el)}–{max(n_el)} "
              f"of {N} | {out['n_steps']} greedy MI steps | "
              f"min gain {out['min_gain']:+.4f} | "
              f"non-positive-gain steps: "
              f"{out['n_nonpositive_steps']}/{out['n_steps']}")
        print("   k | det R²         RG R²          | det MI   RG MI    "
              "RG/det | RG |S| mean (min) | ties det / RG")
        for k in ks:
            dk, rk = out["deterministic"][k], out["random_greedy"][k]
            n_real_min = "-" if rk["n_real_min"] is None else rk["n_real_min"]
            print(f"  {k:>2} | {dk['r2_mean']:.3f} ± {dk['r2_std']:.3f}  "
                  f"{rk['r2_mean']:.3f} ± {rk['r2_std']:.3f}  | "
                  f"{dk['mi_mean']:7.3f}  {rk['mi_mean']:7.3f}  "
                  f"{rk['mi_ratio_to_deterministic']:.3f}  | "
                  f"{rk['n_real_mean']:5.2f} ({n_real_min})"
                  f"         | {dk['n_folds_with_index_ties']:>2} / "
                  f"{rk['n_runs_with_index_ties']:>2}")
        print(f"  (deterministic: {len(out['folds'])} splits; Random Greedy: "
              f"{N_SEEDS} seeds × each split with ≥ k eligible benchmarks, "
              f"run separately for each k; ties = splits / runs in which "
              f"the lowest index decided a pick)")
        print("  paired R² differences deterministic − Random Greedy "
              "(seed mean per split; mean [Nadeau–Bengio 95% CI], "
              "* = CI excludes 0, win rate):")
        for k in ks:
            print(_paired_line(f"k={k}", out["paired"][k]["r2"]))
        if csv_prefix is not None:
            write_paired_csv(pathlib.Path(f"{csv_prefix}_mono_{ds}"
                                          f"_paired.csv"), csv_rows)
        if out["skipped_folds"]:
            print(f"  skipped splits (covariance estimation failed): "
                  f"{sorted(out['skipped_folds'])}")
    parts.close()                       # (shuts the worker pool down)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reviewer checks (clipping, contamination, MI gain "
                    "positivity / Random Greedy) on the matched "
                    "repeated-CV splits.")
    parser.add_argument("--check", default="all",
                        choices=["all", *CHECK_ORDER])
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="datasets of clip and mono, each read from "
                             "{name}.matrix.npz in the data directory "
                             f"(default: {' '.join(DATASETS)}); they also "
                             "set the default --contam-dataset")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions of clip and mono (default "
                             "%(default)s; 1 = repetition 0 only, the "
                             "original folds)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--contam-dataset", nargs="+", default=None,
                        help="dataset(s) of the contamination check "
                             f"(default: {CONTAM_DATASET}, then those of "
                             f"the EM datasets {' '.join(CONTAM_EM_DATASETS)}"
                             " that are among --datasets, i.e. "
                             f"{' '.join(default_contam_datasets(DATASETS))}"
                             " with the default --datasets; "
                             f"--contam-dataset {CONTAM_DATASET} runs only "
                             "the MMLU check, as the rebuttal-era script "
                             f"did); the result of {CONTAM_DATASET} is "
                             "stored under 'contam', that of any other "
                             "dataset DS under 'contam_DS'")
    parser.add_argument("--contam-draws", type=int, default=CONTAM_N_DRAWS,
                        help="random contaminated subsets of the 'draws' "
                             "rank summary (default %(default)s; draw 0 is "
                             "the fixed subset of 'configs', 1 = that one "
                             "only)")
    parser.add_argument("--contam-min-obs", type=int, default=CONTAM_MIN_OBS,
                        help="the contaminated benchmark is the top MI pick "
                             "among the benchmarks observed for at least "
                             "this many models (default %(default)s = "
                             "MIN_TRAIN_OBS / 0.1; 3 = the plain top MI "
                             "pick)")
    parser.add_argument("--tag", default="",
                        help="write reviewer_checks_TAG.json and "
                             "reviewer_checks_TAG_*_paired.csv instead of "
                             "reviewer_checks.json / reviewer_checks_*.csv")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes evaluating the splits of clip and "
                             "mono and the contaminated estimates of contam "
                             "(spawn, single-threaded each; the results do "
                             "not depend on it; default %(default)s)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.tag):
        parser.error("--tag must be letters, digits, '_' or '-'")
    if args.contam_draws < 1:
        parser.error("--contam-draws must be at least 1")
    if args.contam_min_obs < 0:
        parser.error("--contam-min-obs must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    datasets = list(dict.fromkeys(args.datasets))
    contam_datasets = list(dict.fromkeys(
        args.contam_dataset if args.contam_dataset is not None
        else default_contam_datasets(datasets)))
    needed = ((datasets if args.check in ("all", "clip", "mono") else [])
              + (contam_datasets if args.check in ("all", "contam")
                 else []))
    missing = [ds for ds in dict.fromkeys(needed)
               if not matrix_path(ds).is_file()]
    if missing:
        parser.error("no data matrix for " + ", ".join(
            f"{ds} ({matrix_path(ds)})" for ds in missing))
    stem = RESULTS_STEM + (f"_{args.tag}" if args.tag else "")
    csv_prefix = LOGS / stem
    # (results key, check) in CHECK_ORDER; contam once per dataset
    jobs = {
        "clip": [("clip", lambda: check_clip(datasets, args.repeats,
                                             args.folds, csv_prefix,
                                             workers=args.workers))],
        "contam": [(contam_key(ds),
                    lambda ds=ds: check_contam(ds, args.contam_draws,
                                               args.contam_min_obs,
                                               args.workers))
                   for ds in contam_datasets],
        "mono": [("mono", lambda: check_mono(datasets, args.repeats,
                                             args.folds, csv_prefix,
                                             args.workers))],
    }
    for name in CHECK_ORDER:
        if args.check in ("all", name):
            for key, run in jobs[name]:
                path = save_results(key, run(), stem + ".json")
                print(f"\n[{key}] results written to {path}", flush=True)
