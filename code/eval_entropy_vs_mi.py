"""Compare greedy entropy vs mutual information selection via repeated CV
(with random, greedy trace and facility location alongside).

For each dataset, uses the 10% holdout (most training data) setting on the
matched splits of cv_split.cv_splits: --repeats repetitions (default
N_REPEATS = 5) of 10-fold CV, i.e. 50 splits.  Repetition 0 is the
original single 10-fold run (seed-42 folds and training subsamples).  On a
split every method sees the same training / validation models, the same
estimated correlation matrix, the same eligible set (benchmarks observed
in at least MIN_TRAIN_OBS training models, cv_split.eligible_mask) and the
same imputation targets (cv_split.target_mask) and the same common random
numbers (Split.seed), so results pair split by split (rep, fold, split_id
in the log).

Methods:
  entropy     Algorithm 1, greedy entropy (pivoted Cholesky)
  mi          Algorithm 2, greedy mutual information
  random      uniformly random order of the eligible benchmarks,
              seed Split.seed(SEED + RANDOM_SEED_OFFSET)
  trace       greedy trace minimization, forward selection minimizing the
              total residual variance (baseline_select.greedy_trace_select)
  facloc      greedy facility location on the correlation matrix clipped
              at 0, F(A) = sum_j max_{i in A} max(R_ij, 0) over all N
              benchmarks (baseline_select.facility_location_select, the
              paper's headline method since 2026-10-19)
(entropy_rf, greedy entropy from a random first pivot, was dropped on
2026-10-18, protocol rule D3; its seed tag SEED + 11 is retired and no
other stream changed.)  trace and facloc joined on 2026-10-19, after
random: both are deterministic and draw no random numbers, so the
entropy, mi and random rows are those of a run without them, bit for bit.

Scoring (impute_and_evaluate, used by every script): protocol rules D1
(a benchmark observed by fewer than MIN_TARGET_OBS = 2 training models is
not an imputation target; its cells are never scored) and D2 (the
predictions are clipped to the benchmark's valid score range,
score_bounds.py, in train-standardized units).

Near-ties of the greedy scores (TIE_RTOL) are broken by explained variance,
then by the lowest index (greedy_select); {method}_index_ties counts the
picks the lowest index decided.  Ineligible benchmarks are never selected
but remain imputation targets (if they meet MIN_TARGET_OBS), part of the
MI complement and of the residual variance.  Where fewer than k
benchmarks are eligible, every value at that k is NaN.  --kmax is capped
at the number of benchmarks N (HELM Lite: N = 10; at k = N no benchmark is
left to impute and R² is NaN), so the per-k arrays have min(kmax, N)
columns.

Usage:
    python eval_entropy_vs_mi.py [--datasets mmlu mteb merged] [--repeats 5]
                                 [--folds 10] [--kmax 15] [--workers 1]

--datasets accepts any name with a {name}.matrix.npz in the data directory
(path_config.data_dir()); --repeats 1 runs repetition 0 only.

--workers W evaluates the splits in W spawned single-threaded processes
(map_splits, which eval_greedy_all, eval_baselines, eval_benchpress and
plot_mi_selection_all use too): every split is a pure function of the data
and the split (its common random numbers come from Split.seed), so the
logs do not depend on W, and the per-split progress lines are printed in
split order (stdout differs from the serial run's in the timings only).
The default W = 1 runs the serial loop in this process, as before.

Outputs, per dataset:
  logs/entropy_vs_mi_{ds}.npz
    {method}_r2, {method}_rmse     (n_splits, kmax) imputation R² / RMSE on
                                   the validation models given the first k
                                   picks (method in entropy, mi, random,
                                   trace, facloc)
    {method}_resid_var             (n_splits, kmax) residual-variance fraction
    {method}_mi_val                (n_splits, kmax) I(X_A; X_rest)
    {method}_selected              (n_splits, kmax) int, picks in order, -1
                                   beyond the last (or on a failed split)
    {method}_index_ties            (n_splits, kmax) int, picks among the
                                   first k decided by the lowest index
    {method}_tie_breaks            (n_splits, kmax) int, picks among the
                                   first k made among near-tied scores
                                   (greedy_select.tie_break_count).  Every
                                   log stores {m}_tie_breaks in the layout
                                   of its {m}_index_ties (greedy_cv_*,
                                   baselines_*, {ds}_results cv_/emi_,
                                   tabimpute_*, family_holdout_*,
                                   greedy_mi_cv_*: (n_splits, kmax);
                                   cost_budget_*: (n_budgets, n_splits))
    rep, fold, split_id            (n_splits,) int, the split of each row
                                   (rows rep-major, folds in order)
    n_train, n_val, n_eligible     (n_splits,) int
    eligible                       (n_splits, N) bool, the selectable set
    estimation_failed              (n_splits,) bool (rows all NaN)
    holdout_frac, n_folds, n_repeats, min_train_obs, min_target_obs,
    pred_clip, seed                scalars
    benchmark_names                (N,)
  figures/entropy_vs_mi_{ds}.pdf   R² (mean ± std over splits), residual
                                   variance and MI (means) of entropy, MI,
                                   random, greedy trace and facility
                                   location
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

from greedy_select import (greedy_entropy, greedy_mi, random_select,
                           selection_at_k, index_tie_count, tie_break_count)
from baseline_select import facility_location_select, greedy_trace_select
from em_cov import em_cov
import cv_split
import score_bounds
from cv_split import (cv_splits, eligible_mask, target_mask, MIN_TRAIN_OBS,
                      N_REPEATS, RANDOM_SEED_OFFSET)
from score_bounds import clip_z, dataset_bounds, z_limits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42

DATASETS = ["mmlu", "mteb", "merged"]

# Log order; the three paper methods keep their original key order, greedy
# trace and facility location (2026-10-19) follow them.
METHODS = ["entropy", "mi", "random", "trace", "facloc"]
PLOT_METHODS = ["entropy", "mi", "random", "trace", "facloc"]
METRICS = ["r2", "rmse", "resid_var", "mi_val"]
Z_CLIP = 10.0        # clip of the true standardized scores (and inputs)
RIDGE = 0.01         # ridge on R_AA of the conditional mean

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

# Labels, colors and markers of eval_baselines.py
METHOD_LABEL = {
    "entropy": "Entropy (Alg. 1)",
    "mi": "Mutual Info (Alg. 2)",
    "random": "Random",
    "trace": "Greedy trace min.",
    "facloc": "Facility location",
}
METHOD_COLOR = {"entropy": "#4477AA", "mi": "#CC3311", "random": "#888888",
                "trace": "#EE7733", "facloc": "#009988"}
METHOD_STYLE = {"entropy": "o-", "mi": "o-", "random": "x--",
                "trace": "^-", "facloc": "v-"}
METHOD_SHORT = {"entropy": "Ent", "mi": "MI", "random": "Rand",
                "trace": "Trace", "facloc": "FL"}


def display_name(ds_name):
    """Display name of a dataset (the raw name if it has none)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def matrix_path(ds_name):
    return DATA / f"{ds_name}.matrix.npz"


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


# ── Imputation ───────────────────────────────────────────────────

def same_observation_rows(O_test):
    """True if every validation model has validation model 0's observation
    row (the condition of impute_and_evaluate's vectorized path)."""
    O_test = np.asarray(O_test)
    return bool(np.all(O_test == O_test[:1]))


def impute_and_evaluate(B_test, O_test, selected, R, col_mean, col_std, N,
                        target=None, bounds=None):
    """Pooled imputation R² and RMSE of the size-k selection `selected`.

    Every validation model that observes at least one selected benchmark
    and at least one unselected target is imputed from its observed
    selected benchmarks (conditional Gaussian mean in training-standardized
    units, inputs and true z clipped to ±Z_CLIP, ridge RIDGE); R² = 1 −
    SS_res / SS_tot pooled over those models' observed unselected target
    cells (SS_tot about the training mean), RMSE over the same cells.  NaN
    if no cell is scored.

    Protocol rules (every script passes both; None, for synthetic tests,
    means every column is a target / no clip):
      target  (N,) bool, the split's imputation targets
              (cv_split.target_mask(O_train), rule D1): the cells of the
              other benchmarks are never scored;
      bounds  the benchmarks' valid raw-score ranges (score_bounds.
              dataset_bounds, rule D2): every prediction is clipped to its
              range in train-standardized units (score_bounds.z_limits,
              clip_z) before it is scored.
    Returns {"r2", "rmse", "n_eval" (cells scored), "n_clipped"
    (predictions the D2 clip changed)}.

    Vectorized path: when every validation model has the same
    observation row (fully observed data such as MMLU), one solve covers
    all models; it scores exactly the cells of the per-model loop.  Until
    2026-10-18 it was also taken when the rows agreed only on the selected
    benchmarks and on the benchmarks validation model 0 observes, and
    then dropped the cells that other models observe and model 0 does not
    (protocol_changes.md H1).  On the 50 matched splits at the 10%
    holdout, with the selections of random, MI and greedy trace at
    k <= 15, that dropped cells in 41 evaluations on HELM Classic (18
    splits at risk, 516 cells) and 11 on HELM Lite (1 split, 88 cells).
    No split of the other data/ matrices is at risk: there the old and
    the new function take the same path and agree bit for bit.  This is
    the one implementation; eval_greedy_all and eval_benchpress import it
    (both had verbatim copies of the old function).
    """
    A = set(selected)
    M_test = B_test.shape[0]
    B_test_std = O_test * (B_test - col_mean[None, :]) / col_std[None, :]
    B_test_std = np.clip(B_test_std, -Z_CLIP, Z_CLIP) * O_test
    is_target = (np.ones(N, dtype=bool) if target is None
                 else np.asarray(target, dtype=bool))
    z_lo, z_hi = z_limits(bounds, col_mean, col_std)

    # Vectorized path only for identical observation rows (see docstring)
    obs_A = np.array([j for j in selected if O_test[0, j] == 1])
    eval_idx = np.array([j for j in range(N) if j not in A
                         and O_test[0, j] == 1 and is_target[j]])
    all_same = (len(obs_A) > 0 and len(eval_idx) > 0 and
                same_observation_rows(O_test))

    n_clipped = 0
    if all_same:
        R_oo = R[np.ix_(obs_A, obs_A)] + RIDGE * np.eye(len(obs_A))
        R_eo = R[np.ix_(eval_idx, obs_A)]
        W = np.linalg.solve(R_oo, R_eo.T).T
        X_obs = B_test_std[:, obs_A]
        X_true = B_test_std[:, eval_idx]
        X_pred, n_clipped = clip_z(X_obs @ W.T, z_lo[eval_idx],
                                   z_hi[eval_idx])
        ss_res = np.sum((X_true - X_pred) ** 2)
        ss_tot = np.sum(X_true ** 2)
        n_eval = M_test * len(eval_idx)
    else:
        ss_res = 0.0
        ss_tot = 0.0
        n_eval = 0
        for i in range(M_test):
            obs_A_i = np.array([j for j in selected if O_test[i, j] == 1])
            eval_i = np.array([j for j in range(N) if j not in A
                               and O_test[i, j] == 1 and is_target[j]])
            if len(obs_A_i) == 0 or len(eval_i) == 0:
                continue
            R_oo = R[np.ix_(obs_A_i, obs_A_i)] + RIDGE * np.eye(len(obs_A_i))
            R_eo = R[np.ix_(eval_i, obs_A_i)]
            W = np.linalg.solve(R_oo, R_eo.T).T
            x_obs = B_test_std[i, obs_A_i]
            x_true = B_test_std[i, eval_i]
            x_pred, n_c = clip_z(W @ x_obs, z_lo[eval_i], z_hi[eval_i])
            n_clipped += n_c
            ss_res += np.sum((x_true - x_pred) ** 2)
            ss_tot += np.sum(x_true ** 2)
            n_eval += len(eval_i)

    if n_eval == 0:
        return {"r2": np.nan, "rmse": np.nan, "n_eval": 0, "n_clipped": 0}
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "rmse": np.sqrt(ss_res / n_eval), "n_eval": int(n_eval),
            "n_clipped": int(n_clipped)}


def protocol_config():
    """The protocol constants of a run, for the logs (npz scalars / JSON
    config): min_train_obs, min_target_obs (D1) and pred_clip (D2), read
    at call time (test-only overrides show up in the log)."""
    return {"min_train_obs": int(cv_split.MIN_TRAIN_OBS),
            "min_target_obs": int(cv_split.MIN_TARGET_OBS),
            "pred_clip": bool(score_bounds.PRED_CLIP)}


# ── Selection and per-split evaluation ───────────────────────────

def run_selection(method, R, k_max, benchmark_names, split, eligible):
    """Run one method on split `split` with the split's eligible mask and
    common random numbers (identical for every script on that split)."""
    if method == "entropy":
        return greedy_entropy(R, k_max, benchmark_names=benchmark_names,
                              eligible=eligible)
    if method == "mi":
        return greedy_mi(R, k_max, benchmark_names=benchmark_names,
                         eligible=eligible)
    if method == "random":
        return random_select(R, k_max,
                             seed=split.seed(SEED + RANDOM_SEED_OFFSET),
                             benchmark_names=benchmark_names,
                             eligible=eligible)
    if method == "trace":
        return greedy_trace_select(R, k_max, benchmark_names=benchmark_names,
                                   eligible=eligible)
    if method == "facloc":
        return facility_location_select(R, k_max,
                                        benchmark_names=benchmark_names,
                                        eligible=eligible)
    raise ValueError(method)


def full_logdet(R):
    """log det R, via eigenvalues when slogdet reports a non-positive sign."""
    sign_full, logdet_full = np.linalg.slogdet(R)
    if sign_full <= 0:
        eigv = np.linalg.eigvalsh(R)
        logdet_full = np.sum(np.log(np.maximum(eigv, 1e-300)))
    return logdet_full


def subset_mi(R, selected, logdet_full, N):
    """I(X_A; X_rest) = ½[log det R_AA + log det R_CC − log det R] of the
    subset A = selected against all other benchmarks (eligible or not)."""
    R_AA = R[np.ix_(selected, selected)]
    s_a, ld_a = np.linalg.slogdet(R_AA)
    ld_a = ld_a if s_a > 0 else -1e30
    comp = sorted(set(range(N)) - set(selected))
    R_CC = R[np.ix_(comp, comp)]
    s_c, ld_c = np.linalg.slogdet(R_CC)
    ld_c = ld_c if s_c > 0 else -1e30
    return 0.5 * (ld_a + ld_c - logdet_full)


def evaluate_result(result, k_max, B_test, O_test, R, col_mean, col_std, N,
                    logdet_full, target=None, bounds=None):
    """Per-k metrics of one selection (NaN where fewer than k picks); R²
    and RMSE under the split's targets and the dataset's bounds
    (impute_and_evaluate)."""
    out = {key: [] for key in METRICS + ["index_ties", "tie_breaks"]}
    for k in range(1, k_max + 1):
        out["resid_var"].append(result["residual_variance"][k - 1])
        selected = selection_at_k(result, k)
        if selected is None:
            for metric in ["r2", "rmse", "mi_val"]:
                out[metric].append(np.nan)
        else:
            imp = impute_and_evaluate(B_test, O_test, selected, R,
                                      col_mean, col_std, N, target=target,
                                      bounds=bounds)
            out["r2"].append(imp["r2"])
            out["rmse"].append(imp["rmse"])
            out["mi_val"].append(subset_mi(R, selected, logdet_full, N))
        out["index_ties"].append(index_tie_count(result, k))
        out["tie_breaks"].append(tie_break_count(result, k))
    sel = list(result["selected"])[:k_max]
    out["selected"] = sel + [-1] * (k_max - len(sel))
    return out


# ── Split-level parallelism (--workers) ──────────────────────────

def say_now(text, end="\n"):
    """Progress output of a split evaluated in this process: printed at
    once (map_splits' serial path, the output of the serial loop)."""
    print(text, end=end, flush=True)


# Worker state of the split processes (map_splits; spawn)
_SPLIT_WORKER = {}


def _init_split_worker(split_fn, ctx):
    _SPLIT_WORKER.clear()
    _SPLIT_WORKER.update(split_fn=split_fn, ctx=ctx)


def _run_split_task(split):
    """split_fn on one split in a worker process; its progress output
    (say) is collected and returned with the result."""
    w = _SPLIT_WORKER
    text = []
    res = w["split_fn"](w["ctx"], split,
                        lambda line, end="\n": text.append(line + end))
    return res, "".join(text)


def map_splits(split_fn, ctx, splits, workers=1):
    """split_fn(ctx, split, say) of every split, in order (a generator).

    workers <= 1: in this process, the split's progress output printed as
    it runs (say_now), i.e. the serial loop as before.  Otherwise in
    `workers` spawned single-threaded processes (ProcessPoolExecutor.map
    keeps the order): each split's progress output is printed when its
    result is taken, so stdout equals the serial run's apart from the
    timings.  Every split is a pure function of ctx and the split (its
    random numbers come from Split.seed), so the results do not depend on
    workers.  split_fn and ctx are pickled to the workers, which import
    split_fn's module afresh: module state patched at run time (tests)
    does not reach them.  Keep OMP / OpenBLAS / MKL at one thread per
    process (run_all.sh does).  eval_greedy_all, eval_baselines,
    eval_benchpress and plot_mi_selection_all use it too."""
    if workers <= 1 or len(splits) <= 1:
        for s in splits:
            yield split_fn(ctx, s, say_now)
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    mp = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(splits)),
                             mp_context=mp, initializer=_init_split_worker,
                             initargs=(split_fn, ctx)) as pool:
        results = pool.map(_run_split_task, splits)
        try:
            for res, text in results:
                sys.stdout.write(text)
                sys.stdout.flush()
                yield res
        finally:
            # closed early (an error downstream): cancel the splits not
            # yet started, so the pool's shutdown waits for running ones
            results.close()


def emi_split(ctx, s, say=say_now):
    """One matched split of run_dataset (its loop body): {"meta": [(key,
    value)] of the split's metadata, "evals": {method: evaluate_result(...)}
    of ctx["methods"], None if the estimate failed}; progress via say."""
    t0 = time.time()
    B, O, N, k_max = ctx["B"], ctx["O"], ctx["N"], ctx["k_max"]
    at5 = min(5, k_max) - 1
    B_train, O_train = B[s.train_idx], O[s.train_idx]
    B_test, O_test = B[s.val_idx], O[s.val_idx]
    M_train = len(s.train_idx)
    M_test = len(s.val_idx)

    # One eligibility mask per split, shared by every method
    eligible = eligible_mask(O_train)
    target = target_mask(O_train)        # D1, shared by every method
    n_eligible = int(eligible.sum())
    meta = [("rep", s.rep), ("fold", s.fold), ("split_id", s.split_id),
            ("n_train", M_train), ("n_val", M_test),
            ("n_eligible", n_eligible), ("eligible", eligible)]

    say(f"  Rep {s.rep} fold {s.fold}: {M_train} train / {M_test} test, "
        f"{n_eligible}/{N} eligible", end="")

    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:
        say(f" — estimation failed: {e}")
        return {"meta": meta, "evals": None}

    logdet_full = full_logdet(R)

    evals = {}
    for method in ctx["methods"]:
        result = run_selection(method, R, k_max, ctx["names"], s, eligible)
        evals[method] = evaluate_result(result, k_max, B_test, O_test, R,
                                        col_mean, col_std, N, logdet_full,
                                        target=target, bounds=ctx["bounds"])

    elapsed = time.time() - t0
    say("  " + "  ".join(
        f"{METHOD_SHORT[m]} R²@{at5 + 1}={evals[m]['r2'][at5]:.3f}"
        for m in ctx["methods"]) + f"  ({elapsed:.1f}s)")
    return {"meta": meta, "evals": evals}


def run_dataset(ds_name, n_repeats=N_REPEATS, n_folds=N_FOLDS, k_max=K_MAX,
                workers=1):
    """All methods on every matched split of one dataset.

    Returns (metrics, split_meta, benchmark_names): metrics[method][key] is
    an array with one row per split (key in METRICS, selected, index_ties,
    tie_breaks), split_meta[key] one entry per split
    (rep, fold, split_id, n_train, n_val, n_eligible, eligible,
    estimation_failed), rows in cv_splits order.  workers > 1 evaluates
    the splits in that many processes (map_splits; the arrays and the
    progress lines are those of the serial run).
    """
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")

    d = np.load(matrix_path(ds_name), allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)
    obs_frac = O.sum() / (M * N)
    print(f"{M} models × {N} tasks, {obs_frac*100:.1f}% observed", flush=True)
    print(f"{n_repeats} × {n_folds}-fold CV, {HOLDOUT_FRAC:.0%} holdout, "
          f"selectable: benchmarks with ≥ {MIN_TRAIN_OBS} training models; "
          f"targets: ≥ {cv_split.MIN_TARGET_OBS}; predictions clipped to "
          f"the valid range: {score_bounds.PRED_CLIP}", flush=True)
    # At most N picks (k = N leaves no imputation target: R² is NaN there)
    k_max = min(k_max, N)

    p = HOLDOUT_FRAC
    metrics = {m: defaultdict(list) for m in METHODS}
    split_meta = defaultdict(list)
    ctx = dict(B=B, O=O, N=N, k_max=k_max, names=benchmark_names,
               bounds=bounds, methods=list(METHODS))
    splits = list(cv_splits(M, n_folds, n_repeats, p, SEED))

    with closing(map_splits(emi_split, ctx, splits, workers)) as parts:
        for part in parts:
            for key, val in part["meta"]:
                split_meta[key].append(val)
            if part["evals"] is None:
                split_meta["estimation_failed"].append(True)
                for method in METHODS:
                    for metric in METRICS:
                        metrics[method][metric].append([np.nan] * k_max)
                    for key in ["index_ties", "tie_breaks"]:
                        metrics[method][key].append([0] * k_max)
                    metrics[method]["selected"].append([-1] * k_max)
                continue
            split_meta["estimation_failed"].append(False)
            for method in METHODS:
                for key, vals in part["evals"][method].items():
                    metrics[method][key].append(vals)

    # Convert to arrays
    for method in METHODS:
        for key in metrics[method]:
            metrics[method][key] = np.array(metrics[method][key])
    for key in split_meta:
        split_meta[key] = np.array(split_meta[key])

    return metrics, split_meta, benchmark_names


def save_results(ds_name, metrics, split_meta, bench_names, n_repeats,
                 n_folds):
    save_dict = {"benchmark_names": bench_names}
    for method in METHODS:
        for key, arr in metrics[method].items():
            save_dict[f"{method}_{key}"] = arr
    for key, arr in split_meta.items():
        save_dict[key] = arr
    save_dict.update(holdout_frac=np.float64(HOLDOUT_FRAC),
                     n_folds=np.int64(n_folds), n_repeats=np.int64(n_repeats),
                     min_train_obs=np.int64(MIN_TRAIN_OBS),
                     min_target_obs=np.int64(cv_split.MIN_TARGET_OBS),
                     pred_clip=np.bool_(score_bounds.PRED_CLIP),
                     seed=np.int64(SEED))
    path = LOGS / f"entropy_vs_mi_{ds_name}.npz"
    np.savez(path, **save_dict)
    print(f"Results saved to {path}", flush=True)


def print_summary(ds_name, metrics, split_meta, k_max):
    ks_report = [k for k in [1, 3, 5, 10, 15] if k <= k_max]
    n_splits = len(split_meta["split_id"])
    print(f"\n== {display_name(ds_name)}: imputation R² "
          f"(mean ± std over {n_splits} splits) ==")
    print(f"{'method':<24}" + "".join(f"k={k:<12}" for k in ks_report))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for m in METHODS:
            r2 = metrics[m]["r2"]
            cells = [f"{np.nanmean(r2[:, k - 1]):.3f}"
                     f"±{np.nanstd(r2[:, k - 1]):.3f}" for k in ks_report]
            print(f"{METHOD_LABEL[m]:<24}" + "".join(f"{c:<14}" for c in cells))
    n_el = split_meta["n_eligible"]
    N = split_meta["eligible"].shape[1]
    print(f"eligible benchmarks per split: {n_el.min()}–{n_el.max()} of {N} "
          f"({int(np.sum(n_el < N))}/{n_splits} splits restricted)")
    for m in METHODS:
        ties = metrics[m]["index_ties"][:, -1]
        print(f"index ties ({m}, k ≤ {k_max}): {int(ties.sum())} picks "
              f"in {int(np.count_nonzero(ties))}/{n_splits} splits")


# ── Plotting ─────────────────────────────────────────────────────

def plot_comparison(ds_name, metrics, k_max=K_MAX, methods=PLOT_METHODS):
    """3-panel comparison: R², residual variance, MI value."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = np.arange(1, k_max + 1)

    with warnings.catch_warnings():
        # All-NaN columns (fewer than k eligible benchmarks) plot as gaps
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for method in methods:
            c = METHOD_COLOR[method]
            lab = METHOD_LABEL[method]
            ls = METHOD_STYLE[method]
            lw = 1.5 if method != "random" else 1.0
            ms = 4 if method != "random" else 3
            valid = ~np.all(np.isnan(metrics[method]["r2"]), axis=1)
            if not valid.any():
                continue

            mean_r2 = np.nanmean(metrics[method]["r2"][valid], axis=0)
            std_r2 = np.nanstd(metrics[method]["r2"][valid], axis=0)
            mean_rv = np.nanmean(metrics[method]["resid_var"][valid], axis=0)

            axes[0].plot(ks, mean_r2, ls, color=c, label=lab,
                         markersize=ms, linewidth=lw)
            axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                                 color=c, alpha=0.15)
            axes[1].semilogy(ks, mean_rv, ls, color=c, label=lab,
                             markersize=ms, linewidth=lw)

        # MI values for all methods
        for method in methods:
            c = METHOD_COLOR[method]
            lab = METHOD_LABEL[method]
            ls = METHOD_STYLE[method]
            lw = 1.5 if method != "random" else 1.0
            ms = 4 if method != "random" else 3
            if "mi_val" in metrics[method]:
                valid = ~np.all(np.isnan(metrics[method]["mi_val"]), axis=1)
                if valid.any():
                    mean_mi = np.nanmean(metrics[method]["mi_val"][valid],
                                         axis=0)
                    axes[2].plot(ks, mean_mi, ls, color=c, label=lab,
                                 markersize=ms, linewidth=lw)

    axes[0].set_ylabel("Imputation $R^2$")
    axes[1].set_ylabel("Residual variance fraction")
    axes[2].set_ylabel("Mutual information $I(X_{\\mathcal{A}}; X_{\\bar{\\mathcal{A}}})$")

    for ax in axes:
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xlim(0.5, k_max + 0.5)
        ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= k_max])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT / f"entropy_vs_mi_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}", flush=True)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entropy vs MI vs random (and greedy trace, facility "
                    "location) selection on the matched repeated-CV "
                    "splits.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names, each read from "
                             "{name}.matrix.npz in the data directory "
                             f"(default: {' '.join(DATASETS)})")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions (default %(default)s; 1 = "
                             "repetition 0 only, the original folds)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--kmax", type=int, default=K_MAX)
    parser.add_argument("--workers", type=int, default=1,
                        help="processes over the splits (results do not "
                             "depend on it; default %(default)s)")
    parser.add_argument("--plot-entropy-rf", action="store_true",
                        help="ignored: entropy_rf was dropped (protocol "
                             "rule D3); kept so old command lines run")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.kmax < 1:
        parser.error("--kmax must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    missing = [ds for ds in args.datasets if not matrix_path(ds).is_file()]
    if missing:
        parser.error("no data matrix for " + ", ".join(
            f"{ds} ({matrix_path(ds)})" for ds in missing))
    if args.plot_entropy_rf:
        print("note: --plot-entropy-rf is ignored (entropy_rf was dropped, "
              "protocol rule D3)", flush=True)
    plot_methods = PLOT_METHODS

    total_t0 = time.time()

    for ds_name in args.datasets:
        ds_t0 = time.time()
        metrics, split_meta, bench_names = run_dataset(
            ds_name, args.repeats, args.folds, args.kmax, args.workers)
        k_max = metrics["entropy"]["r2"].shape[1]     # min(--kmax, N)
        print_summary(ds_name, metrics, split_meta, k_max)
        plot_comparison(ds_name, metrics, k_max, plot_methods)
        save_results(ds_name, metrics, split_meta, bench_names,
                     args.repeats, args.folds)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{ds_name.upper()} completed in {ds_elapsed:.0f}s", flush=True)

    total_elapsed = time.time() - total_t0
    print(f"\nAll datasets completed in {total_elapsed:.0f}s", flush=True)
