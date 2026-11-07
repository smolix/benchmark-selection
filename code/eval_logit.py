"""Run all experiments in logit-transformed score space.

Applies s -> logit(s / max_j) before covariance estimation and selection.
Predictions are inverted via sigmoid and evaluated in raw-score space.

Matched (paired) design: every method is evaluated on the same
n_repeats × 10 train/validation splits (cv_split.cv_splits; repetition 0
is the original seed-42 10-fold CV with the same training subsamples), on
the same eligible set (cv_split.eligible_mask: benchmarks observed in at
least MIN_TRAIN_OBS training models may be selected; all benchmarks stay
in the correlation matrix, the MI complement and the imputation targets)
and with the same common random numbers (Split.seed: the random baseline
uses SEED + RANDOM_SEED_OFFSET).  Methods: entropy (Alg. 1), mi (Alg. 2),
random (entropy_rf was dropped on 2026-10-18, protocol rule D3), and since
2026-10-19 greedy trace minimization (baseline_select.greedy_trace_select)
and greedy facility location on the correlation matrix clipped at 0
(baseline_select.facility_location_select, the paper's headline method),
on the logit-space correlation estimate like the others; both draw no
random numbers, so every array of the other methods is that of a run
without them, bit for bit.  The greedy CV runs every method but MI, the
entropy-vs-MI comparison all five.  Where
fewer than k benchmarks are eligible, all eligible ones are selected and
the metrics are NaN beyond.
k runs to min(K_MAX, N) (dataset_k_max; HELM Lite: N = 10, and at k = N
nothing is left to impute, so R² is NaN there).

Applicability.  The transform needs bounded, accuracy-like scores: s /
(training column max) is clipped to [EPS, 1 - EPS] before the logit, so
every zero or negative score lands on one floor value, logit(EPS) = -6.91,
a point mass far below a column's other logits.  Each run prints the share
of observed cells at that floor ("logit floor: ...", full-data column
maxima).  The paper's datasets have at most 0.4% (Merged; MMLU none) and
so do HELM Lite / MMLU / Classic (0, 0, 0.1%), MTEB 2026 (0.1%) and OLL
v1 tasks (0.05%): the logit variant applies.  OLL
v2 sub-tasks (1.3% overall, up to 15% of a MATH-hard column) and Epoch AI
(3.4%, up to 38% of a FrontierMath column) have floor masses from tasks
most models fail outright; there the logit run is a sensitivity check
only and the raw-score results are primary.

Protocol rules.  R² is pooled over the cells of the raw-score pipeline
(eval_entropy_vs_mi.impute_and_evaluate) in its raw standardization
(raw_stats), with its rules: the targets of a split are the benchmarks
observed for at least MIN_TARGET_OBS = 2 training models (rule D1,
cv_split.target_mask; the cells of the others are never scored), and the
raw-space predictions (sigmoid of the logit-space prediction times the
column maximum) are clipped to the benchmark's valid score range
(score_bounds.py, rule D2) before they are standardized.  That is the
only clip of the predictions, as for the Gaussian and TabImpute scorers;
the truths are clipped to +-10 as everywhere.  (Until 2026-10-19 the
standardized logit predictions were also clipped to +-10, which only
ever shrank a residual and favoured this pipeline; the clip now applies
only when the test-only override turns score_bounds.PRED_CLIP off, which
reproduces the scores before rule D2.)

Targets without training variance (before rule D1).  A target benchmark
with fewer than NO_INFO_MIN_OBS = 2 training observations has no training
variance: the
raw pipeline floors its SD and its conditional-mean prediction there is
exactly the training mean (R̂ carries no correlation with such a column;
its one observation, if any, standardizes to 0), so the cell adds the
same amount to SS_res and SS_tot.  The logit pipeline predicts the raw
training mean there as well; before 2026-10-18 it scored the sigmoid of
the logit-space prediction against an SD of 1e-6, which clipped truth and
prediction to the same ±10 and counted the cell as a perfect prediction
(HELM Classic at 90% holdout: R² = 1.0 exactly, RMSE 0).  Such cells
occur only on sparse matrices and mostly at large holdouts: at 10% none
on the data-release datasets, 16 cells (1/50 splits) on Merged and 22
(17/50) on BenchPress; at 90% 14.3% of Merged's validation cells, 17.1%
of BenchPress's, 9.4% of Epoch's, 2.0% of HELM Classic's.  Their count
per split is logged (n_no_info_cells).  Since rule D1 (MIN_TARGET_OBS =
NO_INFO_MIN_OBS = 2) these cells are not scored at all; the training-mean
rule only acts if MIN_TARGET_OBS is lowered (test-only override), which
reproduces the scores of 2026-10-18 before D1.  The raw standardization
(raw_stats) also changed on 2026-10-18 for near-constant training
columns; with both changes the logit R² of MMLU is unchanged, and the
rep-0 means at k = 5
(entropy_vs_mi, 10% holdout) move on MTEB (MI 0.718 -> 0.720), Merged
(random 0.448 -> 0.415) and BenchPress (MI -0.217 -> -0.148).

Usage:
    python eval_logit.py [--datasets mmlu mteb merged benchpress]
                         [--repeats 5]
                         [--no-eigenspectrum | --eigenspectrum-only]
                         [--workers 1]

--datasets accepts any name with a data/<name>.matrix.npz (default: the
comma-separated BENCHSELECT_DATASETS, else mmlu mteb merged benchpress);
--repeats 1 runs repetition 0 only.  Datasets share no state, so
`--datasets X --no-eigenspectrum` per dataset plus one
`--eigenspectrum-only` run (the figure combines all --datasets) reproduce
a joint run exactly.

--workers W evaluates the splits (every holdout of the greedy CV and the
entropy-vs-MI splits) in W spawned single-threaded processes, one split
per task (the BLAS thread variables default to 1 for them; a worker
refuses to start if the parent overrode the protocol constants, test-only
module state that a spawned process does not see); the process of a
10%-holdout split runs its greedy-CV and entropy-vs-MI parts on one
estimate, as the serial run shares it through its cache.  Every split is
a pure function of the data and the split (random draws its common random
numbers), so the npz files and figures do not depend on W, and the rows
and progress lines come in split order (stdout equals the serial run's
apart from the per-split seconds); the default W = 1 runs the splits in
this process, as before.  The eigenspectrum is one estimate per dataset
and stays in this process.

Produces:
  figures_logit/eigenspectrum_logit.pdf
  figures_logit/greedy_cv_{dataset}_logit.pdf        (per dataset)
  figures_logit/entropy_vs_mi_{dataset}_logit.pdf     (per dataset)
  logs_logit/greedy_cv_{dataset}_logit.npz
  logs_logit/entropy_vs_mi_{dataset}_logit.npz

Per-split arrays have one row per split in split_id order (rep-major,
folds in order; rows 0..9 are repetition 0) and one column per k:

  greedy_cv_{ds}_logit.npz, for each holdout p (prefix p{pct}_, e.g. p10_)
    p{pct}_{metric}             greedy entropy: entropy (log det R_AA),
                                resid_var, r2, rmse
    p{pct}_rand_{metric}        random baseline, same metrics
    p{pct}_trace_{metric}, p{pct}_facloc_{metric}
                                greedy trace, facility location, same
                                metrics
    p{pct}_{pre}selected        (splits, k_max) int, -1 padded (pre ""
                                entropy, rand_, trace_, facloc_)
    p{pct}_{pre}index_ties      (splits, k_max) int: picks among
                                the first k decided by the lowest index
                                (greedy_select.index_tie_count; -1 if the
                                split failed)
    p{pct}_split_rep, _split_fold, _split_id, _n_eligible, _n_train,
    p{pct}_n_val                (splits,) int split metadata
    p{pct}_n_no_info_cells      (splits,) int, observed validation cells
                                of benchmarks with < NO_INFO_MIN_OBS
                                training observations (predicted by the
                                training mean before D1; not scored
                                since D1; module docstring)
    methods, method_prefixes    method names and their key prefixes
                                ("", "rand_", "trace_", "facloc_")
  entropy_vs_mi_{ds}_logit.npz (10% holdout; method in entropy, mi,
  random, trace, facloc)
    {method}_{r2,rmse,resid_var,mi_val}    (splits, k_max) float
    {method}_selected, {method}_index_ties (splits, k_max) int, as above
    split_rep, split_fold, split_id, n_eligible, n_train, n_val,
    n_no_info_cells, methods
  both: holdout_frac(s), n_folds, n_repeats, seed, min_train_obs,
        min_target_obs, pred_clip, k_max, benchmark_names
"""

import argparse
import pathlib
import sys
import time
import json
import multiprocessing
import os
import warnings
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, logit_figures_dir, logit_logs_dir

DATA = data_dir()
OUT = logit_figures_dir()
LOGS = logit_logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import (greedy_entropy, greedy_mi, random_select,
                           selection_at_k, index_tie_count)
from baseline_select import facility_location_select, greedy_trace_select
from em_cov import em_cov
import cv_split
import score_bounds
from cv_split import (cv_splits, eligible_mask, target_mask, N_REPEATS,
                      MIN_TRAIN_OBS, RANDOM_SEED_OFFSET)
from score_bounds import clip_raw, dataset_bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRACS = [0.1, 0.2, 0.5, 0.9]
ENT_MI_HOLDOUT = 0.1  # holdout of the entropy-vs-MI comparison
SEED = 42
EPS = 1e-3  # clipping bound for logit
K_PRINT = 5  # k of the per-split progress lines
# Targets with fewer training observations have no training variance and
# are predicted by the raw training mean (module docstring)
NO_INFO_MIN_OBS = 2

DEFAULT_DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
DATASETS = os.environ.get("BENCHSELECT_DATASETS")
DATASETS = DATASETS.split(",") if DATASETS else DEFAULT_DATASETS
DISPLAY = {"mmlu": "MMLU", "mteb": "MTEB",
           "merged": "Merged", "benchpress": "BenchPress"}
COLORS = {"mmlu": "#4477AA", "mteb": "#CC3311",
          "merged": "#EE7733", "benchpress": "#228833"}
# Colors of further datasets, in order of appearance
EXTRA_COLORS = ["#AA3377", "#66CCEE", "#CCBB44", "#009988", "#EE3377",
                "#332288", "#BBBBBB", "#882255", "#999933", "#000000"]

# greedy_cv: method -> key prefix within p{pct}_ (greedy entropy unprefixed;
# greedy trace and facility location joined on 2026-10-19)
CV_METHODS = {"entropy": "", "random": "rand_", "trace": "trace_",
              "facloc": "facloc_"}
CV_METRICS = ["entropy", "resid_var", "r2", "rmse"]
# entropy_vs_mi (the original three first)
EMI_METHODS = ["entropy", "mi", "random", "trace", "facloc"]
EMI_METRICS = ["r2", "rmse", "resid_var", "mi_val"]
# Figure styles (labels, colors and markers of eval_entropy_vs_mi.py /
# eval_baselines.py); greedy CV: every method but greedy entropy in its
# holdout's color with this line style
METHOD_LABEL = {"entropy": "Entropy (Alg. 1)", "mi": "Mutual Info (Alg. 2)",
                "random": "Random", "trace": "Greedy trace min.",
                "facloc": "Facility location"}
METHOD_COLOR = {"entropy": "#4477AA", "mi": "#CC3311", "random": "#888888",
                "trace": "#EE7733", "facloc": "#009988"}
METHOD_STYLE = {"entropy": "o-", "mi": "o-", "random": "x--",
                "trace": "^-", "facloc": "v-"}
CV_PLOT_STYLE = {"random": dict(fmt="x--", markersize=3, linewidth=1.0,
                                alpha=0.6),
                 "trace": dict(fmt="^:", markersize=3, linewidth=1.0,
                               alpha=0.8),
                 "facloc": dict(fmt="v-.", markersize=3, linewidth=1.2,
                                alpha=0.9)}


def run_selectors(R, k_max, benchmark_names, eligible, split, methods):
    """Selections of `methods` on one split, in that order: the split's
    eligibility mask for all of them, its common random numbers for random
    (greedy trace and facility location are deterministic and draw
    none)."""
    out = {}
    for m in methods:
        if m == "entropy":
            out[m] = greedy_entropy(R, k_max, benchmark_names=benchmark_names,
                                    eligible=eligible)
        elif m == "mi":
            out[m] = greedy_mi(R, k_max, benchmark_names=benchmark_names,
                               eligible=eligible)
        elif m == "random":
            out[m] = random_select(
                R, k_max, seed=split.seed(SEED + RANDOM_SEED_OFFSET),
                benchmark_names=benchmark_names, eligible=eligible)
        elif m == "trace":
            out[m] = greedy_trace_select(R, k_max,
                                         benchmark_names=benchmark_names,
                                         eligible=eligible)
        elif m == "facloc":
            out[m] = facility_location_select(
                R, k_max, benchmark_names=benchmark_names, eligible=eligible)
        else:
            raise ValueError(m)
    return out


def display_name(ds):
    """Display name of a dataset (the raw name if it has none)."""
    return DISPLAY.get(ds, ds)


# Figure legends of the data-release datasets.  Printed lines keep
# display_name (the raw name for these): make_numbers.py parses the
# eigenspectrum block's "  NAME: 90% at k=..." lines, NAME one token.
LEGEND = {"olv2sub": "OLL v2 sub-tasks", "olv1items": "OLL v1 tasks",
          "epoch": "Epoch AI", "helm_lite": "HELM Lite",
          "helm_mmlu": "HELM MMLU", "helm_classic": "HELM Classic",
          "mteb2026": "MTEB 2026"}


def legend_name(ds):
    """Figure-legend name of a dataset."""
    return LEGEND.get(ds, display_name(ds))


def dataset_k_max(N):
    """Largest k for a dataset of N benchmarks, min(K_MAX, N): at k = N
    nothing is left to impute (R² NaN), beyond it there is no selection."""
    return min(K_MAX, N)


def floor_counts(B, O):
    """Per column, (observed cells, cells at the logit floor) under the
    full-data column maxima: s = B / max <= EPS (zero or negative scores),
    which logit_transform clips to logit(EPS)."""
    N = B.shape[1]
    n_obs = np.zeros(N, dtype=np.int64)
    n_floor = np.zeros(N, dtype=np.int64)
    for j in range(N):
        obs = B[:, j][O[:, j] == 1]
        if len(obs) == 0:
            continue
        col_max = max(obs.max(), EPS)
        n_obs[j] = len(obs)
        n_floor[j] = int(np.count_nonzero(obs / col_max <= EPS))
    return n_obs, n_floor


def print_clip_report(B, O, benchmark_names):
    """One line: share of observed cells at the logit floor (applicability
    of the transform, see the module docstring)."""
    n_obs, n_floor = floor_counts(B, O)
    total, fl = int(n_obs.sum()), int(n_floor.sum())
    share = n_floor / np.maximum(n_obs, 1)
    j = int(np.argmax(share))
    worst = (f"; largest share {benchmark_names[j]} {share[j]:.1%}"
             if fl else "")
    print(f"logit floor (score / column max <= {EPS:g}, clipped to "
          f"{np.log(EPS / (1 - EPS)):.2f}): {fl} of {total} observed cells "
          f"({fl / max(total, 1):.2%}) in {int(np.count_nonzero(n_floor))} "
          f"of {len(n_obs)} columns{worst}")


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


# ── Imputation in logit space, R² in raw space ───────────────────

def same_observation_rows(O_test):
    """True if every validation model has validation model 0's observation
    row (as eval_entropy_vs_mi.same_observation_rows)."""
    O_test = np.asarray(O_test)
    return bool(np.all(O_test == O_test[:1]))


def impute_and_evaluate_logit(B_test_logit, O_test, selected, R,
                              col_mean_logit, col_std_logit,
                              B_test_raw, col_mean_raw, col_std_raw,
                              col_max, N, no_info=None, target=None,
                              bounds=None):
    """Impute in logit space, invert via sigmoid, evaluate R² in
    standardized raw-score space (for comparability with non-logit results).

    The cells are those of eval_entropy_vs_mi.impute_and_evaluate (the
    observed unselected target benchmarks of every validation model
    observing a selected one) in the same raw standardization (raw_stats).
    target: (N,) bool, the split's D1 targets (cv_split.target_mask; None:
    every column); bounds: the D2 valid raw-score ranges (lo, hi) of
    score_bounds.dataset_bounds, applied to the raw-space predictions
    (score_bounds.clip_raw) before they are standardized (None: no clip);
    the standardized predictions are clipped to +-10 only when
    score_bounds.PRED_CLIP is off (test-only, the pre-D2 behaviour).
    no_info:
    (N,) bool, benchmarks with fewer than NO_INFO_MIN_OBS training
    observations; their prediction is the raw training mean (standardized
    0), which is what the raw-score pipeline predicts there (module
    docstring, "Targets without training variance").  None: no such
    benchmark.  The vectorized path requires identical observation rows
    (eval_entropy_vs_mi.same_observation_rows; protocol_changes.md H1).
    """
    A = set(selected)
    M_test = B_test_logit.shape[0]
    RIDGE = 0.01
    if no_info is None:
        no_info = np.zeros(N, dtype=bool)
    is_target = (np.ones(N, dtype=bool) if target is None
                 else np.asarray(target, dtype=bool))
    if bounds is None:
        lo, hi = np.full(N, -np.inf), np.full(N, np.inf)
    else:
        lo, hi = (np.asarray(b, dtype=np.float64) for b in tuple(bounds)[:2])

    # Standardize logit scores for imputation
    B_test_std = O_test * (B_test_logit - col_mean_logit[None, :]) / col_std_logit[None, :]
    B_test_std = np.clip(B_test_std, -10, 10) * O_test

    # Vectorized path only for identical observation rows
    obs_A = np.array([j for j in selected if O_test[0, j] == 1])
    eval_idx = np.array([j for j in range(N) if j not in A
                         and O_test[0, j] == 1 and is_target[j]])
    all_same = (len(obs_A) > 0 and len(eval_idx) > 0 and
                same_observation_rows(O_test))

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
        # D2: the valid score range, in raw units
        S_pred, _ = clip_raw(S_pred, lo[eval_idx], hi[eval_idx])

        # True raw scores, standardized using raw training stats
        S_true = B_test_raw[:, eval_idx]
        true_std = (S_true - col_mean_raw[eval_idx]) / col_std_raw[eval_idx]
        pred_std = (S_pred - col_mean_raw[eval_idx]) / col_std_raw[eval_idx]
        true_std = np.clip(true_std, -10, 10)
        if not score_bounds.PRED_CLIP:           # pre-D2 (test-only)
            pred_std = np.clip(pred_std, -10, 10)
        pred_std[:, no_info[eval_idx]] = 0.0     # raw training mean

        ss_res = np.sum((true_std - pred_std) ** 2)
        ss_tot = np.sum(true_std ** 2)
        n_eval = M_test * len(eval_idx)
    else:
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
            f_pred_std = W @ x_obs

            # To raw space, clipped to the valid range (D2)
            f_pred = f_pred_std * col_std_logit[eval_i] + col_mean_logit[eval_i]
            s_pred = sigmoid(f_pred) * col_max[eval_i]
            s_pred, _ = clip_raw(s_pred, lo[eval_i], hi[eval_i])

            s_true = B_test_raw[i, eval_i]
            t_std = (s_true - col_mean_raw[eval_i]) / col_std_raw[eval_i]
            p_std = (s_pred - col_mean_raw[eval_i]) / col_std_raw[eval_i]
            t_std = np.clip(t_std, -10, 10)
            if not score_bounds.PRED_CLIP:        # pre-D2 (test-only)
                p_std = np.clip(p_std, -10, 10)
            p_std[no_info[eval_i]] = 0.0          # raw training mean

            ss_res += np.sum((t_std - p_std) ** 2)
            ss_tot += np.sum(t_std ** 2)
            n_eval += len(eval_i)

    if n_eval == 0:
        return {"r2": np.nan, "rmse": np.nan}
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "rmse": np.sqrt(ss_res / n_eval)}


# ── Raw-space col_mean and col_std (for R² evaluation) ───────────

def raw_stats(B, O):
    """Training mean and SD of the raw scores, exactly as the raw-score
    pipeline standardizes them (eval_entropy_vs_mi.estimate_corr): the SD
    is floored at 0.01 (|mean| + 1) where that pipeline estimates by EM
    (training matrix < 90% observed or fewer models than benchmarks), the
    variance at 1e-12 where it uses pairwise correlations.  So logit and
    raw R² weight every validation cell alike.  (Until 2026-10-18 the
    variance was floored at 1e-12 everywhere: a benchmark without or with
    almost no training variance got an SD down to 1e-6, its standardized
    truths and predictions clipped to ±10 and the cells weighed up to 100
    times a typical cell, with residual 0 or 400 by the signs alone -- e.g.
    the R² = 1 cells of module docstring "Targets without training
    variance".  MMLU, where the raw pipeline uses pairwise correlations,
    is unchanged; on the EM datasets the near-constant columns change.)"""
    M_tr, N_tr = B.shape
    obs_frac = O.sum() / (M_tr * N_tr)
    col_count = np.maximum(O.sum(axis=0), 1.0)
    col_mean = (B * O).sum(axis=0) / col_count
    B_c = O * (B - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    if obs_frac < 0.9 or M_tr < N_tr:          # estimate_corr_em's floor
        col_std_floor = 0.01 * (np.abs(col_mean) + 1.0)
        col_std = np.sqrt(np.maximum(col_var, col_std_floor ** 2))
    else:                                      # estimate_corr_pairwise's
        col_std = np.sqrt(np.maximum(col_var, 1e-12))
    return col_mean, col_std


def no_info_mask(O_train):
    """(N,) bool: benchmarks with fewer than NO_INFO_MIN_OBS training
    observations (no training variance; predicted by the training mean)."""
    return np.asarray(O_train).sum(axis=0) < NO_INFO_MIN_OBS


# ── Matched splits ───────────────────────────────────────────────

def load_dataset(ds_name):
    """(B_raw, O, benchmark_names) of data/<ds_name>.matrix.npz."""
    path = DATA / f"{ds_name}.matrix.npz"
    if not path.is_file():
        raise SystemExit(f"eval_logit.py: no dataset {ds_name!r} "
                         f"({path} does not exist)")
    d = np.load(path, allow_pickle=True)
    return (d["B"].astype(np.float64), d["O"].astype(np.float64),
            d["benchmark_names"])


def split_label(s):
    """'Fold f' for repetition 0 (as in the single-repetition logs),
    'Rep r Fold f' otherwise."""
    return f"Fold {s.fold}" if s.rep == 0 else f"Rep {s.rep} Fold {s.fold}"


def prepare_split(B_raw, O, s, bounds=None):
    """Logit-space data of one split; R etc. from the training models.

    Returns a dict with the validation data, the training statistics, the
    split's D1 targets (target_mask of the training rows), the dataset's
    D2 bounds (as given) and R, cm_logit, cs_logit (estimate_corr on the
    logit training scores).  estimate_corr's exceptions propagate.
    """
    B_train_raw, O_train = B_raw[s.train_idx], O[s.train_idx]
    B_test_raw, O_test = B_raw[s.val_idx], O[s.val_idx]

    # Logit transform using training col_max
    B_train_logit, col_max = logit_transform(B_train_raw, O_train)
    B_test_logit, _ = logit_transform(B_test_raw, O_test, col_max=col_max)

    # Raw stats for R² evaluation
    cm_raw, cs_raw = raw_stats(B_train_raw, O_train)
    no_info = no_info_mask(O_train)

    R, cm_logit, cs_logit = estimate_corr(B_train_logit, O_train)
    return {"B_test_logit": B_test_logit, "O_test": O_test,
            "B_test_raw": B_test_raw, "col_max": col_max,
            "cm_raw": cm_raw, "cs_raw": cs_raw, "no_info": no_info,
            "target": target_mask(O_train), "bounds": bounds,
            "R": R, "cm_logit": cm_logit, "cs_logit": cs_logit}


def cached_split(B_raw, O, s, cache=None, bounds=None):
    """prepare_split, memoized in `cache` (dict keyed by (holdout tag,
    split_id)) so the 10%-holdout splits are estimated once for both the
    greedy CV and the entropy-vs-MI comparison; a failure is cached and
    re-raised."""
    key = (int(s.holdout_frac * 10000), s.split_id)
    if cache is not None and key in cache:
        sd = cache[key]
    else:
        try:
            sd = prepare_split(B_raw, O, s, bounds)
        except Exception as e:
            sd = e
        if cache is not None:
            cache[key] = sd
    if isinstance(sd, Exception):
        raise sd
    return sd


def impute_split(sd, sel):
    """impute_and_evaluate_logit of selection `sel` on split data sd."""
    return impute_and_evaluate_logit(
        sd["B_test_logit"], sd["O_test"], sel, sd["R"], sd["cm_logit"],
        sd["cs_logit"], sd["B_test_raw"], sd["cm_raw"], sd["cs_raw"],
        sd["col_max"], sd["R"].shape[0], no_info=sd["no_info"],
        target=sd.get("target"), bounds=sd.get("bounds"))


def selected_row(result, k_max):
    """(k_max,) int array of a nested selection, -1 beyond the last pick."""
    row = np.full(k_max, -1, dtype=np.int64)
    sel = result["selected"][:k_max]
    row[:len(sel)] = sel
    return row


def index_tie_row(result, k_max):
    """(k_max,) int: index-decided picks affecting the size-k selection."""
    return np.array([index_tie_count(result, k) for k in range(1, k_max + 1)],
                    dtype=np.int64)


def append_meta(meta, s, n_eligible, O=None):
    """Split metadata of split s; with the observation mask O also
    n_no_info_cells (validation cells of no_info_mask benchmarks)."""
    meta["split_rep"].append(s.rep)
    meta["split_fold"].append(s.fold)
    meta["split_id"].append(s.split_id)
    meta["n_eligible"].append(n_eligible)
    meta["n_train"].append(len(s.train_idx))
    meta["n_val"].append(len(s.val_idx))
    if O is not None:
        no_info = no_info_mask(O[s.train_idx])
        meta["n_no_info_cells"].append(
            int((O[s.val_idx][:, no_info] > 0).sum()))


def _nanmean(a, axis=0):
    """np.nanmean without the all-NaN-column warning (k > n_eligible)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis)


def print_split_summary(meta, ties, N, k_max=K_MAX):
    """Eligibility and lowest-index tie counts over a dataset's splits."""
    n_el = np.asarray(meta["n_eligible"])
    if (n_el < N).any():
        print(f"    eligible benchmarks (>= {MIN_TRAIN_OBS} training "
              f"models): {n_el.min()}-{n_el.max()} of {N} "
              f"({np.count_nonzero(n_el < N)}/{len(n_el)} splits restricted)")
    n_ni = np.asarray(meta.get("n_no_info_cells", []))
    if n_ni.any():
        print(f"    validation cells of benchmarks with < {NO_INFO_MIN_OBS} "
              f"training observations (predicted by the training mean): "
              f"{int(n_ni.sum())} in {np.count_nonzero(n_ni)}/{len(n_ni)} "
              f"splits")
    parts = []
    for method, rows in ties.items():
        last = np.asarray(rows, dtype=np.int64).reshape(len(rows), -1)[:, -1]
        ok = last >= 0                       # -1: the split failed
        parts.append(f"{method} {np.count_nonzero(last[ok] > 0)}"
                     f"/{np.count_nonzero(ok)}")
    print(f"    splits with lowest-index tie-breaks (k <= {k_max}): "
          f"{', '.join(parts)}")


# ══════════════════════════════════════════════════════════════════
# Part 1: Eigenspectrum
# ══════════════════════════════════════════════════════════════════

def plot_eigenspectrum_logit(datasets=None):
    print("\n" + "#"*70)
    print("# Eigenspectrum (logit space)")
    print("#"*70)

    datasets = DATASETS if datasets is None else datasets
    fig, ax = plt.subplots(figsize=(6, 4))
    extra = iter(EXTRA_COLORS * (1 + len(datasets) // len(EXTRA_COLORS)))

    for ds in datasets:
        B, O, _ = load_dataset(ds)
        B_logit, col_max = logit_transform(B, O)

        R, _, _ = estimate_corr(B_logit, O)

        eigvals = np.linalg.eigvalsh(R)[::-1]
        N = len(eigvals)
        cumvar = np.cumsum(eigvals) / eigvals.sum()
        resid = 1.0 - cumvar
        ks = np.arange(1, N + 1)
        color = COLORS[ds] if ds in COLORS else next(extra)
        # At most 30 components; never k = N, whose residual is 0 up to
        # rounding (only matters for datasets with N <= 30)
        n_plot = min(30, max(N - 1, 1))
        ax.semilogy(ks[:n_plot], resid[:n_plot], "o-", color=color,
                     label=legend_name(ds), markersize=3, linewidth=1.5)
        print(f"  {display_name(ds)}: 90% at k={np.searchsorted(cumvar, 0.9)+1}, "
              f"95% at k={np.searchsorted(cumvar, 0.95)+1}")

    ax.axhline(0.10, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.axhline(0.05, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.axhline(0.01, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.set_xlabel("Number of components $k$")
    ax.set_ylabel("Residual variance fraction $1 - \\rho(k)$")
    many = len(datasets) > 6         # e.g. the data-release datasets
    ax.legend(fontsize=7 if many else 9, ncol=2 if many else 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT / "eigenspectrum_logit.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Part 2: Greedy CV (entropy, random, trace, facloc; 4 holdout fracs)
# ══════════════════════════════════════════════════════════════════

def cv_curves(result, sd, k_max):
    """Per-k log det R_AA ("entropy"), resid_var, r2, rmse of a selection;
    NaN where fewer than k benchmarks were eligible."""
    R = sd["R"]
    curves = {m: [] for m in CV_METRICS}
    for k in range(1, k_max + 1):
        sel = selection_at_k(result, k)
        curves["resid_var"].append(result["residual_variance"][k - 1])
        if sel is None:
            for m in ["entropy", "r2", "rmse"]:
                curves[m].append(np.nan)
            continue
        R_AA = R[np.ix_(sel, sel)]
        sign, logdet = np.linalg.slogdet(R_AA)
        curves["entropy"].append(logdet if sign > 0 else -np.inf)
        imp = impute_split(sd, sel)
        curves["r2"].append(imp["r2"])
        curves["rmse"].append(imp["rmse"])
    return curves


def greedy_cv_split(B_raw, O, benchmark_names, s, k_max, bounds, cache,
                    metrics, meta, ties):
    """One split s of the greedy CV (holdout s.holdout_frac): appends its
    rows to run_greedy_cv's metrics, meta and ties and returns its
    progress line.  A pure function of the data and the split (random
    draws the split's common random numbers; cache only memoizes
    prepare_split), so the rows do not depend on the process that
    computes them (--workers)."""
    N = B_raw.shape[1]
    i_print = min(K_PRINT, k_max) - 1
    t0 = time.time()
    el = eligible_mask(O[s.train_idx])
    append_meta(meta, s, int(el.sum()), O)

    try:
        sd = cached_split(B_raw, O, s, cache, bounds)
    except Exception as e:
        for method, prefix in CV_METHODS.items():
            for key in CV_METRICS:
                metrics[prefix + key].append([np.nan] * k_max)
            metrics[prefix + "selected"].append(
                np.full(k_max, -1, dtype=np.int64))
            metrics[prefix + "index_ties"].append(
                np.full(k_max, -1, dtype=np.int64))
            ties[method].append(np.full(k_max, -1, dtype=np.int64))
        return f"    {split_label(s)}: failed ({e})"
    res = run_selectors(sd["R"], k_max, benchmark_names, el, s,
                        CV_METHODS)
    for method, prefix in CV_METHODS.items():
        curves = cv_curves(res[method], sd, k_max)
        for key in CV_METRICS:
            metrics[prefix + key].append(curves[key])
        metrics[prefix + "selected"].append(
            selected_row(res[method], k_max))
        tie_row = index_tie_row(res[method], k_max)
        metrics[prefix + "index_ties"].append(tie_row)
        ties[method].append(tie_row)

    elapsed = time.time() - t0
    elig = f"  elig={int(el.sum())}/{N}" if el.sum() < N else ""
    extra = "".join(
        f"  {short}@{i_print + 1}="
        f"{metrics[CV_METHODS[m] + 'r2'][-1][i_print]:.3f}"
        for m, short in (("trace", "Trace"), ("facloc", "FL"))
        if m in CV_METHODS)
    return (f"    {split_label(s)}: "
            f"R²@{i_print + 1}={metrics['r2'][-1][i_print]:.3f}  "
            f"Rand@{i_print + 1}={metrics['rand_r2'][-1][i_print]:.3f}"
            f"{extra}{elig}  ({elapsed:.1f}s)")


def run_greedy_cv(ds_name, n_repeats=N_REPEATS, cache=None, workers=1):
    """The greedy CV of one dataset at every holdout (module docstring).
    cache: dict shared with run_entropy_vs_mi (its 10%-holdout splits are
    estimated once for both).  workers > 1: the splits of every holdout
    run in that many spawned single-threaded processes (map_units), their
    rows and progress lines assembled in split order; with a cache, the
    process of a 10%-holdout split also runs its entropy-vs-MI split on
    the same estimate and keeps it in the cache for run_entropy_vs_mi."""
    print(f"\n{'#'*70}")
    print(f"# {display_name(ds_name)}: Greedy CV (logit)")
    print(f"{'#'*70}")

    B_raw, O, benchmark_names = load_dataset(ds_name)
    M, N = B_raw.shape
    k_max = dataset_k_max(N)
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)     # D2
    print(f"{M} models × {N} tasks, {O.sum()/(M*N)*100:.1f}% observed")
    print(f"{n_repeats} × {N_FOLDS} matched splits per holdout")
    print_clip_report(B_raw, O, benchmark_names)

    results = {}
    splits = {p: list(cv_splits(M, N_FOLDS, n_repeats, p, SEED))
              for p in HOLDOUT_FRACS}
    parts = None
    if workers > 1:
        both = cache is not None
        units = [("cv+emi" if both and p == ENT_MI_HOLDOUT else "cv", s)
                 for p in HOLDOUT_FRACS for s in splits[p]]
        parts = map_units(split_context(B_raw, O, benchmark_names, k_max,
                                        bounds), units, workers)

    for p in HOLDOUT_FRACS:
        metrics = defaultdict(list)
        meta = defaultdict(list)
        ties = {m: [] for m in CV_METHODS}
        print(f"\n  Holdout {p:.0%}")

        for s in splits[p]:
            if parts is None:
                line = greedy_cv_split(
                    B_raw, O, benchmark_names, s, k_max, bounds,
                    cache if p == ENT_MI_HOLDOUT else None,
                    metrics, meta, ties)
            else:
                part = next(parts)
                line = absorb(part["cv"], metrics, meta, ties)
                if "emi" in part:
                    cache[EMI_PART, s.split_id] = part["emi"]
            print(line, flush=True)

        print_split_summary(meta, ties, N, k_max)
        results[p] = {k: np.array(v) for k, v in metrics.items()}
        results[p].update({k: np.array(v, dtype=np.int64)
                           for k, v in meta.items()})
    if parts is not None:
        next(parts, None)            # all consumed: ends the worker pool

    return results, benchmark_names


def plot_greedy_cv(ds_name, results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    k_max = results[next(iter(results))]["r2"].shape[1]
    ks = np.arange(1, k_max + 1)
    colors = {0.1: "#4477AA", 0.2: "#66CCEE", 0.5: "#EE7733", 0.9: "#CC3311"}

    for p in HOLDOUT_FRACS:
        if p not in results:
            continue
        c = colors[p]
        valid = ~np.all(np.isnan(results[p]["r2"]), axis=1)
        if not valid.any():
            continue
        mr2 = _nanmean(results[p]["r2"][valid], axis=0)
        sr2 = _nanstd(results[p]["r2"][valid], axis=0)
        mrv = _nanmean(results[p]["resid_var"][valid], axis=0)
        ment = _nanmean(results[p]["entropy"][valid], axis=0)

        axes[0].plot(ks, mr2, "o-", color=c, label=f"{p:.0%} holdout",
                     markersize=4, linewidth=1.5)
        axes[0].fill_between(ks, mr2 - sr2, mr2 + sr2, color=c, alpha=0.15)
        axes[1].semilogy(ks, mrv, "o-", color=c, label=f"{p:.0%} holdout",
                         markersize=4, linewidth=1.5)
        axes[2].plot(ks, ment, "o-", color=c, label=f"{p:.0%} holdout",
                     markersize=4, linewidth=1.5)

        # Random baseline, greedy trace and facility location
        for m, style in CV_PLOT_STYLE.items():
            pre = CV_METHODS[m]
            if pre + "r2" not in results[p]:
                continue
            vr = ~np.all(np.isnan(results[p][pre + "r2"]), axis=1)
            if not vr.any():
                continue
            style = dict(style)
            fmt = style.pop("fmt")
            axes[0].plot(ks, _nanmean(results[p][pre + "r2"][vr], axis=0),
                         fmt, color=c, **style)
            axes[1].semilogy(ks, _nanmean(results[p][pre + "resid_var"][vr],
                                          axis=0), fmt, color=c, **style)
            axes[2].plot(ks, _nanmean(results[p][pre + "entropy"][vr],
                                      axis=0), fmt, color=c, **style)

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
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xlim(0.5, k_max + 0.5)
        ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= k_max])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUT / f"greedy_cv_{ds_name}_logit.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Part 3: Entropy vs MI vs Random, trace, facloc (10% holdout)
# ══════════════════════════════════════════════════════════════════

def emi_curves(result, sd, k_max, logdet_full):
    """Per-k r2, rmse, resid_var and MI I(X_A; X_rest) of a selection;
    NaN where fewer than k benchmarks were eligible.  The MI complement
    is every unselected benchmark, eligible or not."""
    R = sd["R"]
    N = R.shape[0]
    curves = {m: [] for m in EMI_METRICS}
    for k in range(1, k_max + 1):
        sel = selection_at_k(result, k)
        curves["resid_var"].append(result["residual_variance"][k - 1])
        if sel is None:
            for m in ["r2", "rmse", "mi_val"]:
                curves[m].append(np.nan)
            continue
        imp = impute_split(sd, sel)
        curves["r2"].append(imp["r2"])
        curves["rmse"].append(imp["rmse"])

        R_AA = R[np.ix_(sel, sel)]
        s_a, ld_a = np.linalg.slogdet(R_AA)
        ld_a = ld_a if s_a > 0 else -1e30
        comp = sorted(set(range(N)) - set(sel))
        R_CC = R[np.ix_(comp, comp)]
        s_c, ld_c = np.linalg.slogdet(R_CC)
        ld_c = ld_c if s_c > 0 else -1e30
        curves["mi_val"].append(0.5 * (ld_a + ld_c - logdet_full))
    return curves


def entropy_vs_mi_split(B_raw, O, benchmark_names, s, k_max, bounds, cache,
                        metrics, meta, ties):
    """One split s of the entropy-vs-MI comparison: appends its rows to
    run_entropy_vs_mi's metrics, meta and ties and returns its progress
    line (a pure function of the data and the split, as
    greedy_cv_split)."""
    N = B_raw.shape[1]
    i_print = min(K_PRINT, k_max) - 1
    t0 = time.time()
    el = eligible_mask(O[s.train_idx])
    append_meta(meta, s, int(el.sum()), O)

    try:
        sd = cached_split(B_raw, O, s, cache, bounds)
    except Exception as e:
        for method in EMI_METHODS:
            for key in EMI_METRICS:
                metrics[method][key].append([np.nan] * k_max)
            for key in ["selected", "index_ties"]:
                metrics[method][key].append(
                    np.full(k_max, -1, dtype=np.int64))
            ties[method].append(np.full(k_max, -1, dtype=np.int64))
        return f"  {split_label(s)}: failed ({e})"
    R = sd["R"]
    res = run_selectors(R, k_max, benchmark_names, el, s, EMI_METHODS)

    sign_full, logdet_full = np.linalg.slogdet(R)
    if sign_full <= 0:
        eigv = np.linalg.eigvalsh(R)
        logdet_full = np.sum(np.log(np.maximum(eigv, 1e-300)))

    for method in EMI_METHODS:
        curves = emi_curves(res[method], sd, k_max, logdet_full)
        for key in EMI_METRICS:
            metrics[method][key].append(curves[key])
        metrics[method]["selected"].append(
            selected_row(res[method], k_max))
        tie_row = index_tie_row(res[method], k_max)
        metrics[method]["index_ties"].append(tie_row)
        ties[method].append(tie_row)

    elapsed = time.time() - t0
    elig = f"  elig={int(el.sum())}/{N}" if el.sum() < N else ""
    return (f"  {split_label(s)}: "
            f"Ent={metrics['entropy']['r2'][-1][i_print]:.3f}  "
            f"MI={metrics['mi']['r2'][-1][i_print]:.3f}  "
            f"Rand={metrics['random']['r2'][-1][i_print]:.3f}"
            + "".join(f"  {short}={metrics[m]['r2'][-1][i_print]:.3f}"
                      for m, short in (("trace", "Trace"), ("facloc", "FL"))
                      if m in metrics)
            + f"{elig}  ({elapsed:.1f}s)")


def run_entropy_vs_mi(ds_name, n_repeats=N_REPEATS, cache=None, workers=1):
    """The entropy-vs-MI comparison of one dataset (10% holdout).  cache:
    run_greedy_cv's (estimates of these splits, or with workers > 1 the
    splits' results computed alongside its own).  workers > 1: the splits
    not found in the cache run in that many spawned single-threaded
    processes (map_units); rows and progress lines in split order."""
    print(f"\n{'#'*70}")
    print(f"# {display_name(ds_name)}: Entropy vs MI (logit)")
    print(f"{'#'*70}")

    B_raw, O, benchmark_names = load_dataset(ds_name)
    M, N = B_raw.shape
    k_max = dataset_k_max(N)
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)     # D2

    p = ENT_MI_HOLDOUT
    metrics = {m: defaultdict(list) for m in EMI_METHODS}
    meta = defaultdict(list)
    ties = {m: [] for m in EMI_METHODS}
    splits = list(cv_splits(M, N_FOLDS, n_repeats, p, SEED))
    parts = None
    if workers > 1:
        done = [(cache or {}).pop((EMI_PART, s.split_id), None)
                for s in splits]
        if all(part is not None for part in done):
            parts = iter(done)
        else:
            parts = (part["emi"] for part in map_units(
                split_context(B_raw, O, benchmark_names, k_max, bounds),
                [("emi", s) for s in splits], workers))

    for s in splits:
        if parts is None:
            line = entropy_vs_mi_split(B_raw, O, benchmark_names, s, k_max,
                                       bounds, cache, metrics, meta, ties)
        else:
            line = absorb(next(parts), metrics, meta, ties)
        print(line, flush=True)
    if parts is not None:
        next(parts, None)            # all consumed: ends the worker pool

    print_split_summary(meta, ties, N, k_max)
    for method in EMI_METHODS:
        for key in metrics[method]:
            metrics[method][key] = np.array(metrics[method][key])
    meta = {k: np.array(v, dtype=np.int64) for k, v in meta.items()}

    return metrics, meta, benchmark_names


def plot_entropy_vs_mi(ds_name, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    k_max = metrics["entropy"]["r2"].shape[1]
    ks = np.arange(1, k_max + 1)
    colors, labels, styles = METHOD_COLOR, METHOD_LABEL, METHOD_STYLE
    widths = {m: 1.0 if m == "random" else 1.5 for m in EMI_METHODS}
    sizes = {m: 3 if m == "random" else 4 for m in EMI_METHODS}
    band = {m: True for m in EMI_METHODS}
    methods = [m for m in EMI_METHODS if m in metrics]

    for method in methods:
        c, ls = colors[method], styles[method]
        lw, ms = widths[method], sizes[method]
        valid = ~np.all(np.isnan(metrics[method]["r2"]), axis=1)
        if not valid.any():
            continue
        mr2 = _nanmean(metrics[method]["r2"][valid], axis=0)
        sr2 = _nanstd(metrics[method]["r2"][valid], axis=0)
        mrv = _nanmean(metrics[method]["resid_var"][valid], axis=0)

        axes[0].plot(ks, mr2, ls, color=c, label=labels[method],
                     markersize=ms, linewidth=lw)
        if band[method]:
            axes[0].fill_between(ks, mr2 - sr2, mr2 + sr2, color=c,
                                 alpha=0.15)
        axes[1].semilogy(ks, mrv, ls, color=c, label=labels[method],
                         markersize=ms, linewidth=lw)

    for method in methods:
        c, ls = colors[method], styles[method]
        lw, ms = widths[method], sizes[method]
        if "mi_val" in metrics[method]:
            valid = ~np.all(np.isnan(metrics[method]["mi_val"]), axis=1)
            if valid.any():
                mmi = _nanmean(metrics[method]["mi_val"][valid], axis=0)
                axes[2].plot(ks, mmi, ls, color=c, label=labels[method],
                             markersize=ms, linewidth=lw)

    axes[0].set_ylabel("Imputation $R^2$")
    axes[1].set_ylabel("Residual variance fraction")
    axes[2].set_ylabel("Mutual information")
    for ax in axes:
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xlim(0.5, k_max + 0.5)
        ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= k_max])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUT / f"entropy_vs_mi_{ds_name}_logit.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Split-level worker processes (--workers)
# ══════════════════════════════════════════════════════════════════

# Cache key tag of an entropy-vs-MI split computed by run_greedy_cv's
# worker processes: cache[EMI_PART, split_id] = (metrics, meta, ties, line)
EMI_PART = "emi"
# BLAS thread variables of the worker processes: 1 unless set (run_all.sh
# sets them for every job)
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS")
# Worker state of the split processes (spawn)
_CTX = {}


def _protocol():
    """The protocol constants read at call time (D1 threshold, D2 clip)."""
    return int(cv_split.MIN_TARGET_OBS), bool(score_bounds.PRED_CLIP)


def split_context(B_raw, O, benchmark_names, k_max, bounds):
    """What a worker process needs for its splits: the data, k_max, the D2
    bounds, and the protocol constants of this process (checked by the
    workers)."""
    return {"B": B_raw, "O": O, "names": benchmark_names, "k_max": k_max,
            "bounds": bounds, "protocol": _protocol()}


def _init_worker(ctx):
    _CTX.clear()
    _CTX.update(ctx)
    # test-only overrides of the protocol constants (module state of the
    # parent) do not reach a spawned process: refuse rather than differ
    if _protocol() != ctx["protocol"]:
        raise RuntimeError(f"protocol {ctx['protocol']} overridden in the "
                           f"parent process: run with --workers 1")


def _run_unit(unit):
    """One split in a worker process.  unit = (kind, split): "cv" its
    greedy-CV split, "emi" its entropy-vs-MI split, "cv+emi" both on one
    estimate (as the serial run shares it through its cache).  Returns
    {"cv" / "emi": (metrics, meta, ties, line)} of the split (absorb)."""
    kind, s = unit
    c = _CTX
    cache = {} if kind == "cv+emi" else None
    data = (c["B"], c["O"], c["names"], s, c["k_max"], c["bounds"], cache)
    out = {}
    if kind in ("cv", "cv+emi"):
        acc = (defaultdict(list), defaultdict(list),
               {m: [] for m in CV_METHODS})
        line = greedy_cv_split(*data, *acc)
        out["cv"] = (*acc, line)
    if kind in ("emi", "cv+emi"):
        acc = ({m: defaultdict(list) for m in EMI_METHODS},
               defaultdict(list), {m: [] for m in EMI_METHODS})
        line = entropy_vs_mi_split(*data, *acc)
        out["emi"] = (*acc, line)
    return out


def map_units(ctx, units, workers):
    """_run_unit of every unit, in order, in `workers` spawned
    single-threaded processes (ProcessPoolExecutor.map keeps the order;
    every split is a pure function of the data and the split, so its rows
    do not depend on workers)."""
    for var in BLAS_THREAD_VARS:
        os.environ.setdefault(var, "1")
    mp = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(units)),
                             mp_context=mp, initializer=_init_worker,
                             initargs=(ctx,)) as pool:
        yield from pool.map(_run_unit, units)


def _extend(dst, src):
    for key, val in src.items():
        if isinstance(val, dict):
            _extend(dst[key], val)
        else:
            dst[key].extend(val)


def absorb(part, metrics, meta, ties):
    """Append the rows of one split computed by a worker process (part =
    (metrics, meta, ties, line) of _run_unit) to the run's metrics, meta
    and ties, as the serial run appends them; returns its progress
    line."""
    rows, meta_rows, tie_rows, line = part
    _extend(metrics, rows)
    _extend(meta, meta_rows)
    _extend(ties, tie_rows)
    return line


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def design_info(n_repeats, benchmark_names, k_max=K_MAX):
    """Scalars describing the matched design, saved in both npz files."""
    return {"n_folds": np.int64(N_FOLDS), "n_repeats": np.int64(n_repeats),
            "seed": np.int64(SEED), "min_train_obs": np.int64(MIN_TRAIN_OBS),
            "min_target_obs": np.int64(cv_split.MIN_TARGET_OBS),
            "pred_clip": np.bool_(score_bounds.PRED_CLIP),
            "k_max": np.int64(k_max),
            "benchmark_names": np.array([str(b) for b in benchmark_names])}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Logit-space greedy CV and entropy-vs-MI comparison "
                    "on matched repeated-CV splits.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names (data/<name>.matrix.npz); "
                             "default: $BENCHSELECT_DATASETS (comma-"
                             "separated) or %(default)s")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions (rep 0 = the original seed-42 "
                             "folds); default %(default)s")
    part = parser.add_mutually_exclusive_group()
    part.add_argument("--no-eigenspectrum", action="store_true",
                      help="skip the eigenspectrum figure (e.g. to run "
                           "datasets as separate jobs)")
    part.add_argument("--eigenspectrum-only", action="store_true",
                      help="only the eigenspectrum figure of --datasets")
    parser.add_argument("--workers", type=int, default=1,
                        help="evaluate the splits in this many spawned "
                             "single-threaded processes; the logs do not "
                             "depend on it (default %(default)s)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    missing = [ds for ds in args.datasets
               if not (DATA / f"{ds}.matrix.npz").is_file()]
    if missing:
        parser.error(f"no {', '.join(f'{ds}.matrix.npz' for ds in missing)}"
                     f" in {DATA}")
    return args


if __name__ == "__main__":
    args = parse_args()
    total_t0 = time.time()

    # Eigenspectrum
    if not args.no_eigenspectrum:
        plot_eigenspectrum_logit(args.datasets)

    # Per-dataset experiments
    for ds in [] if args.eigenspectrum_only else args.datasets:
        ds_t0 = time.time()
        # 10%-holdout split estimates shared by greedy CV and entropy vs MI
        cache = {}

        # Greedy CV
        cv_results, bench_names = run_greedy_cv(ds, args.repeats, cache,
                                                args.workers)
        plot_greedy_cv(ds, cv_results)
        save = design_info(args.repeats, bench_names,
                           dataset_k_max(len(bench_names)))
        save["holdout_fracs"] = np.array(HOLDOUT_FRACS)
        save["methods"] = np.array(list(CV_METHODS))
        save["method_prefixes"] = np.array(list(CV_METHODS.values()))
        for p, m in cv_results.items():
            prefix = f"p{int(p*100)}"
            for key, arr in m.items():
                save[f"{prefix}_{key}"] = arr
        np.savez(LOGS / f"greedy_cv_{ds}_logit.npz", **save)

        # Entropy vs MI
        mi_metrics, mi_meta, bench_names = run_entropy_vs_mi(
            ds, args.repeats, cache, args.workers)
        cache.clear()
        plot_entropy_vs_mi(ds, mi_metrics)
        save2 = design_info(args.repeats, bench_names,
                            dataset_k_max(len(bench_names)))
        save2["holdout_frac"] = np.float64(ENT_MI_HOLDOUT)
        save2["methods"] = np.array(EMI_METHODS)
        save2.update(mi_meta)
        for method in EMI_METHODS:
            for key, arr in mi_metrics[method].items():
                save2[f"{method}_{key}"] = arr
        np.savez(LOGS / f"entropy_vs_mi_{ds}_logit.npz", **save2)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{display_name(ds)} completed in {ds_elapsed:.0f}s", flush=True)

    total = time.time() - total_t0
    print(f"\nAll experiments completed in {total:.0f}s")
