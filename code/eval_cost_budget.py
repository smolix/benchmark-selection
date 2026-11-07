"""Cost-weighted (knapsack) benchmark selection under an example budget.

Benchmarks differ enormously in evaluation cost; on MMLU the per-subject
test sizes range from 100 (abstract_algebra) to 1,534 (professional_law).
This experiment instantiates the budgeted greedy of Appendix B.3 with
cost c_j = number of test questions of subject j and budget C = total
number of questions the practitioner is willing to run.  Budgets are
given as fractions of the total cost of all benchmarks of the dataset.

Methods (all evaluated on the same matched splits, see below):
  knap_entropy     cost-effective greedy on shifted entropy gains
                   (gain/cost rule) + best-singleton fallback
  knap_mi          same gain-per-cost rule with MI gains (heuristic)
  card_entropy     cost-blind greedy entropy order, adding benchmarks in
                   that order while they fit the budget (skip-and-continue)
  card_mi          same for the MI order
  cheap_first      cheapest benchmarks first (maximizes count)
  random           random order, adding while budget allows
  knap_trace       same gain-per-cost rule with the trace objective: gain =
                   explained variance of the unselected benchmarks (the
                   drop in their total conditional variance,
                   greedy_select._explained_variance), + best singleton
  card_trace       cost-blind greedy trace order
                   (baseline_select.greedy_trace_select), filled like
                   card_entropy
  knap_facloc      same gain-per-cost rule with facility location on the
                   clipped correlation matrix (the headline method, user
                   decision 2026-10-19): gain = sum_j max(s_ij − c_j, 0),
                   c_j = max_{a ∈ A} s_ja the current cover, S built as
                   baseline_select.facility_location_select builds it,
                   + best singleton (budgeted_select)
  card_facloc      cost-blind greedy facility-location order
                   (baseline_select.facility_location_select), filled like
                   card_entropy
The trace methods were added after the paper's six; they are appended to
METHODS, so every array of the other methods is bit-identical to a run
without them (only the `methods` list grows).  The facility-location pair
is appended after the trace pair in the same way: it draws no random
numbers, so no other method's selections, R² or ties change.  The *_rf pair
(knap_entropy_rf, card_entropy_rf: entropy from a random first pick) was
dropped on 2026-10-18 (protocol rule D3); no other method's stream
changed.

Matched design.  The splits are cv_splits(M, folds, repeats, 0.1, SEED):
--repeats repetitions of --folds-fold CV (default N_REPEATS = 5 x 10 = 50
splits).  Repetition 0 is the original single run (cv_folds(M, 10, 42)
and the seed-42 training subsample), so --repeats 1 is the original
protocol.  On every split all methods and budgets share
  * the train / validation models and the estimated correlation matrix,
  * one eligibility mask, eligible_mask(O_train): only benchmarks observed
    for at least MIN_TRAIN_OBS training models can be selected.
    Ineligible benchmarks stay in the correlation matrix (MI complement,
    explained variance of the tie-breaks) and remain imputation targets
    if observed for at least MIN_TARGET_OBS training models
    (cv_split.target_mask, protocol rule D1; the cells of the others are
    never scored); every prediction is clipped to its benchmark's valid
    score range (score_bounds.py, rule D2),
  * common random numbers: random's permutation is random_select's,
    seeded with Split.seed(SEED + RANDOM_SEED_OFFSET) (rep 0: (49, fold,
    1000)), the same order as the random baseline of every other script
    on that split; one permutation per split serves all budgets.
The budgets C = frac · (sum of all costs) are the same on every split.

Reproducing the historical cost_budget logs.  Before the matched design
this script drew random from rng = default_rng((SEED + 7, fold)), a fresh
rng.permutation(N) for each budget in order, so its random R² cannot be
compared with the shared-seed default (rep 0 differs by up to ~0.5 in a
cell).  --legacy-random restores exactly those draws (rep 0: seed
(SEED + 7, fold), the b-th permutation for the b-th budget, restricted to
the eligible benchmarks as random_select restricts its permutation; rep
r > 0: seed (SEED + 7, r, fold)), so rep 0 reproduces the historical
random_r2 wherever every benchmark is eligible and the cost table is the
historical one (verified to ~1e-13 on MMLU and, with the HEAD cost table
via --counts, on MTEB); the log records random_legacy = True.  That
random baseline is not shared with the other scripts, so use it only to
compare with the historical logs, never for paired statistics across
scripts.  Independently of --legacy-random, knap_entropy / card_entropy
differ from the historical logs by design (entropy tie-break), as does
cheap_first wherever costs tie (equal-cost tie-break, below), and every
method changes with the cost table itself: data/mteb_task_sizes.json was
rebuilt after the historical MTEB log (counts_sha256 in the log
identifies the table used).

Ties.  Near-ties (TIE_RTOL, greedy_select._near_max) of every selector's
score are broken by explained variance of the unselected benchmarks given
the current selection, then by the lowest index: gain/cost ratios of the
knapsack (every objective; relative tolerance, explained variance per unit
cost), the entropy singleton fallback, the greedy entropy / MI orders
(greedy_select), the greedy trace order (baseline_select; its score is
the explained variance, so the lowest index decides), the greedy
facility-location order (baseline_select.facility_location_select: gains
within TIE_RTOL · max(|best|, 1)) and equal costs in
cheap_first (before: np.argsort's order among equal costs, which depends
on the CPU's sorting kernel; laptop and 4090 runs picked different
100-question MMLU subjects).  {m}_index_ties
counts the picks the lowest index decided.  Without near-ties the
selections are bit-identical to the plain argmax / argsort rules.

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir()):
  LOGS/cost_budget_{ds}.npz
    budgets, budgets_frac              (n_budgets,) absolute budgets C and
                                       their fractions of total_cost
    costs, benchmark_names             (N,)
    {m}_r2, {m}_size                   (n_budgets, n_splits) imputation R²
                                       on the validation models / number of
                                       benchmarks selected (NaN R²: nothing
                                       affordable; NaN both: estimation
                                       failed); column i is split i
    {m}_rmse, {m}_cost                 (n_budgets, n_splits) RMSE / total
                                       cost of the selection
    {m}_selected                       (n_budgets, n_splits, N) int, picks
                                       in selection order, -1 padded
    {m}_index_ties, {m}_tie_breaks     (n_budgets, n_splits) int, picks of
                                       the selection decided by the lowest
                                       index / made among near-tied
                                       candidates (order-filled methods:
                                       steps of the order up to the last
                                       pick)
    knap_*_singleton                   (n_budgets, n_splits) bool, the
                                       best-singleton fallback won
    rep, fold, split_id                (n_splits,) int; pair methods (and
                                       scripts) on split_id = rep·folds+fold
    n_train, n_val, n_eligible         (n_splits,) int
    eligible                           (n_splits, N) bool selectable mask
    estimation_failed                  (n_splits,) bool
    methods                            (n_methods,) method names
    holdout_frac, n_folds, n_repeats, min_train_obs, min_target_obs,
    pred_clip, seed, total_cost, counts_file, counts_sha256,
    random_legacy                      scalars
  LOGS/cost_budget_{ds}_paired.csv
    paired R² differences method_a − method_b for every method pair and
    budget over the matched splits (paired_stats.paired_stats: mean,
    Nadeau–Bengio corrected standard error, 95% t-interval, win rate;
    pairs oriented by paired_stats.orient: mi before entropy before
    trace, knap_ before card_ for the same objective, then facloc and
    cheap_first (both unranked by paired_stats.METHOD_ORDER, so between
    hub and random, equal ranks in METHODS order: cheap_first −
    knap_facloc) and random last, e.g. knap_mi − knap_entropy,
    knap_trace − knap_facloc, knap_facloc − random; columns dataset,
    holdout_frac, budget_frac, budget, method_a, method_b, then
    paired_stats.STAT_FIELDS)
  OUT/cost_budget_{ds}.pdf            mean R² (± std over splits) vs budget
                                      (PLOT_METHODS: the paper's six and
                                      the facility-location pair;
                                      --plot-trace adds the trace pair)

Fixed per-benchmark overhead (opt-in, --fixed-cost RATIO ...; without the
flag nothing above changes).  Running a benchmark also costs a fixed
amount (integration, harness setup, one model load per suite) that does not
scale with its number of examples: the true cost of benchmark j is
    cost_j = c0 + c_j,   c0 = RATIO · median_j c_j
(the median of the dataset's whole cost table, eligible or not), one run
per RATIO (default 0 0.1 0.5 1 2; RATIO 0 is the run above, bit for bit:
test_eval_cost_budget.py).  Budgets are fractions of the full suite's TRUE
cost, C = frac · sum_j (c0 + c_j) = frac · (total_cost + N · c0), so
frac = 1 still buys everything and, as c0 grows, the problem tends to the
cardinality-constrained one (C / c0 ≈ frac · N benchmarks).  Every method
is held to the true budget: the knapsack methods rank by gain / (c0 + c_j)
(budgeted_select on the true costs, so their ratio tie-break and
best-singleton fallback see c0 too), cheapest first orders by the true
cost (the same order, since c0 is constant), and the cost-blind orders and
random (the same shared permutation, Split.seed(SEED + RANDOM_SEED_OFFSET):
no new random-number stream) are filled skip-and-continue under the true
costs.  The cost of every selection is accumulated in pick order (the
selectors' feasibility arithmetic) and {m}_violations (cost > budget) is
logged; expected all False.  Human-evaluation burden (annotators, review
of open-ended outputs) is a further per-benchmark cost for which there are
no data; it is not modelled.  Metrics: r2 as above (unselected observed
benchmarks; NaN if nothing is affordable) and r2_suite (suite_r2: pooled
over ALL observed validation cells, a bought cell counting as known, the
empty selection scoring 0; one target set for every method and budget, as
in eval_token_costs.py).  Outputs (instead of the cost_budget_* files;
cost_fixed_* is not matched by make_numbers' cost_budget_*.npz glob):
  LOGS/cost_fixed_{ds}.npz  (savez_compressed; first axis = split)
    {m}_r2, {m}_r2_suite, {m}_rmse, {m}_cost   (n_splits, n_levels,
                                       n_budgets) float, cost = true cost
                                       of the selection
    {m}_size, {m}_index_ties, {m}_tie_breaks   (n_splits, n_levels,
                                       n_budgets) int
    {m}_violations                     (n_splits, n_levels, n_budgets) bool
    {m}_selected                       (n_splits, n_levels, n_budgets, N)
                                       int, picks in order, -1 padded
    knap_*_singleton                   (n_splits, n_levels, n_budgets) bool
    rep, fold, split_id, n_train, n_val, n_eligible, eligible,
    estimation_failed                  split metadata as above
    fixed_ratios, fixed_costs, total_cost   (n_levels,) RATIO, c0, true
                                       suite cost
    budgets (n_levels, n_budgets), budgets_frac (n_budgets,), costs (N,)
    variable costs c_j, costs_true (n_levels, N), median_cost, methods,
    benchmark_names, holdout_frac, n_folds, n_repeats, min_train_obs,
    seed, counts_file, counts_sha256, matrix_sha256
  LOGS/cost_fixed_{ds}_paired.csv  every oriented method pair (orient), level,
                                   budget and metric (columns dataset,
                                   holdout_frac, fixed_ratio, fixed_cost,
                                   budget_frac, budget, metric, method_a,
                                   method_b, then STAT_FIELDS)
  LOGS/cost_fixed_{ds}.json   config, notes, summary tables, FIXED_PAIRS
                              rows, checks, runtime and a flat "macros"
                              dict of finite numbers ("macros_undefined":
                              the NaN statistics, not emitted):
      CostFixed{T}Rsq{Ds}{M}F{r}B{b}, CostFixed{T}RsqStd...   R² split mean
                              / std; {T} = '' (r2) or Suite (r2_suite);
                              {r} = num_words(RATIO) (Fzero,
                              Fzeropointone, Fzeropointfive, Fone, Ftwo),
                              {b} = num_words(100 · frac) as in CostRsq
      CostFixedSize..., CostFixedSpent...   mean benchmarks selected / mean
                              true cost over budget
      CostFixed{T}Pair{S}{Ds}{A}Vs{B}F{r}B{b}   FIXED_PAIRS, S in Diff, Se,
                              Lo, Hi, Win, N; CostFixed{T}SigPos /
                              SigNeg{Ds}{A}Vs{B}F{r}: budgets whose 95%
                              interval lies above / below 0
      CostFixedOverhead{Ds}F{r} (c0), CostFixedTotal{Ds}F{r},
      CostFixedBudget{Ds}F{r}B{b}, CostFixedEmpty{Ds}F{r}B{b} (splits on
      which nothing is affordable), CostFixedMedianCost{Ds},
      CostFixedNBench{Ds}, CostFixedNSplits{Ds}, CostFixedViolations{Ds}
  OUT/cost_fixed_{ds}.pdf     one column per level: mean R² vs budget (the
                              six paper methods, knap_trace and
                              knap_facloc), and the paired R² difference
                              vs random (95% Nadeau–Bengio band)

Usage:
    python eval_cost_budget.py [--dataset mmlu | --datasets NAME ...]
                               [--counts counts.json ...] [--budgets F ...]
                               [--repeats 5] [--folds 10] [--plot-entropy-rf]
                               [--plot-trace] [--legacy-random]
                               [--fixed-cost [RATIO ...]]

Macros of the facility-location pair follow the trace pair's names:
CostRsq{Ds}KnapFacloc..., CostRsq{Ds}CardFacloc... (make_numbers.py reads
every {m}_r2 of the npz), CostFixed...KnapFacloc... / ...CardFacloc...
and the FIXED_PAIRS pairs KnapFaclocVsRandom, CardFaclocVsRandom,
CheapFirstVsKnapFacloc, KnapFaclocVsCardFacloc, KnapMiVsKnapFacloc and
KnapTraceVsKnapFacloc (JSON of --fixed-cost).

Datasets are addressed by name (data/<name>.matrix.npz); --counts maps
each benchmark name to its cost (defaults: mmlu, mteb, see --help).
"""

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time
import warnings
import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import (greedy_entropy, greedy_mi, index_tie_count,
                           random_select, TIE_RTOL, _eligible,
                           _explained_variance, _near_max, _tie_break)
from baseline_select import facility_location_select, greedy_trace_select
from cv_split import (cv_splits, eligible_mask, target_mask, MIN_TRAIN_OBS,
                      N_REPEATS, RANDOM_SEED_OFFSET)
from eval_entropy_vs_mi import (estimate_corr, impute_and_evaluate,
                                protocol_config)
from score_bounds import clip_z, dataset_bounds, z_limits
from paired_stats import (STAT_FIELDS, orient, orient_pair, paired_stats,
                          write_csv)
from make_numbers import camel, num_words

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42
D_FLOOR = 1e-4  # conditional-variance floor defining the entropy shift

DEFAULT_BUDGETS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]
# Cost tables (data/build_cost_tables.py) used when --counts is not given
DEFAULT_COUNTS = {"mmlu": "mmlu_test_counts.json",
                  "mteb": "mteb_task_sizes.json"}

# Log order: the six paper methods keep their original key order; later
# additions are appended (the trace pair, then the facility-location pair).
METHODS = ["knap_entropy", "knap_mi", "card_entropy", "card_mi",
           "cheap_first", "random", "knap_trace", "card_trace",
           "knap_facloc", "card_facloc"]
# Default figure: the paper's six and the headline facility-location pair
# (--plot-trace adds the trace pair)
PLOT_METHODS = METHODS[:6] + ["knap_facloc", "card_facloc"]
KNAP_OBJECTIVE = {"knap_entropy": "entropy", "knap_mi": "mi",
                  "knap_trace": "trace", "knap_facloc": "facloc"}
# budgeted_select objectives
OBJECTIVES = ("entropy", "mi", "trace", "facloc")
TRACE_D_FLOOR = 1e-12  # conditional variance below which a trace gain is 0
                       # (greedy_trace_select's floor)

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "mteb": "MTEB",
    "merged": "Merged",
    "benchpress": "BenchPress",
}

METHOD_LABEL = {
    "knap_entropy": "Knapsack entropy (B.3)",
    "knap_mi": "Knapsack MI (heuristic)",
    "card_entropy": "Cost-blind entropy",
    "card_mi": "Cost-blind MI",
    "cheap_first": "Cheapest first",
    "random": "Random",
    "knap_trace": "Knapsack trace",
    "card_trace": "Cost-blind trace",
    "knap_facloc": "Knapsack facility location",
    "card_facloc": "Cost-blind facility location",
}

# facility location: eval_baselines' teal (#009988) and Paul Tol's mint
METHOD_COLOR = {
    "knap_entropy": "#4477AA",
    "knap_mi": "#CC3311",
    "card_entropy": "#66AACC",
    "card_mi": "#EE9988",
    "cheap_first": "#228833",
    "random": "#888888",
    "knap_trace": "#AA3377",
    "card_trace": "#CC99BB",
    "knap_facloc": "#009988",
    "card_facloc": "#44BB99",
}

METHOD_STYLE = {
    "knap_entropy": "o-",
    "knap_mi": "o-",
    "card_entropy": "s--",
    "card_mi": "s--",
    "cheap_first": "^-",
    "random": "x--",
    "knap_trace": "D-",
    "card_trace": "d--",
    "knap_facloc": "v-",
    "card_facloc": "v--",
}

PAIRED_FIELDS = (["dataset", "holdout_frac", "budget_frac", "budget",
                  "method_a", "method_b"] + STAT_FIELDS)

# Pairs printed in the stdout summary (all pairs go to the CSV), in
# paired_stats' orientation (knap_mi − knap_entropy, ...)
PRINT_PAIRS = [orient_pair(a, b) for a, b in [
    ("knap_mi", "knap_entropy"), ("knap_entropy", "card_entropy"),
    ("knap_entropy", "cheap_first"), ("knap_entropy", "random"),
    ("knap_mi", "card_mi"), ("knap_mi", "cheap_first"),
    ("knap_mi", "random"), ("knap_mi", "knap_trace"),
    ("knap_trace", "card_trace"), ("knap_trace", "random"),
    ("knap_facloc", "card_facloc"), ("knap_facloc", "random"),
    ("cheap_first", "knap_facloc"), ("knap_mi", "knap_facloc"),
    ("knap_trace", "knap_facloc")]]

# Fixed per-benchmark overhead (--fixed-cost; module docstring)
DEFAULT_FIXED_RATIOS = [0.0, 0.1, 0.5, 1.0, 2.0]   # c0 / median cost
FIXED_STEM = "cost_fixed"        # LOGS/cost_fixed_{ds}.*, OUT/cost_fixed_{ds}.pdf
FIXED_MACRO_PREFIX = "CostFixed"
FIXED_METRICS = {"r2": "", "r2_suite": "Suite"}   # log key -> macro tag
# Pairs printed, kept in the JSON and given macros (all oriented pairs go
# to the CSV): every knapsack objective and the count maximizer against
# random (the reviewer's comparison), knapsack vs cheapest first and vs
# its cost-blind order, and the objectives against each other.  The
# facility-location pairs are appended (the trace pairs' pattern, plus
# knap_trace − knap_facloc); cheap_first and knap_facloc have equal
# paired_stats ranks, so that pair is listed in METHODS order, the
# orientation of orient(METHODS).
FIXED_PAIRS = [orient_pair(a, b) for a, b in [
    ("knap_entropy", "random"), ("knap_mi", "random"),
    ("knap_trace", "random"), ("card_trace", "random"),
    ("cheap_first", "random"), ("knap_trace", "cheap_first"),
    ("knap_entropy", "card_entropy"), ("knap_mi", "card_mi"),
    ("knap_trace", "card_trace"), ("knap_mi", "knap_trace"),
    ("knap_facloc", "random"), ("card_facloc", "random"),
    ("cheap_first", "knap_facloc"), ("knap_facloc", "card_facloc"),
    ("knap_mi", "knap_facloc"), ("knap_trace", "knap_facloc")]]
FIXED_PAIR_STATS = [("Diff", "mean_diff"), ("Se", "se_nb"),
                    ("Lo", "ci_low"), ("Hi", "ci_high"),
                    ("Win", "win_rate"), ("N", "n_splits")]
FIXED_PAIRED_FIELDS = (["dataset", "holdout_frac", "fixed_ratio",
                        "fixed_cost", "budget_frac", "budget", "metric",
                        "method_a", "method_b"] + STAT_FIELDS)
FIXED_PLOT_METHODS = METHODS[:6] + ["knap_trace", "knap_facloc"]
FIXED_DIFF_METHODS = ["knap_entropy", "knap_mi", "knap_trace",
                      "knap_facloc", "cheap_first"]   # − random, lower row


def display_name(ds_name):
    """Display name of a dataset (the raw name if none is registered)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def matrix_path(ds_name):
    return DATA / f"{ds_name}.matrix.npz"


# ── Budgeted greedy (Appendix B.3) ───────────────────────────────

def _chol_state_init(Sigma):
    N = Sigma.shape[0]
    return {"d": np.diag(Sigma).copy().astype(np.float64),
            "L": np.zeros((N, N)), "t": 0, "selected": []}


def _chol_state_add(state, Sigma, j_star):
    """Add j_star to the selection, updating conditional variances."""
    d, L, t = state["d"], state["L"], state["t"]
    N = len(d)
    sel_set = set(state["selected"])
    sqrt_d = np.sqrt(max(d[j_star], 1e-300))
    for j in range(N):
        if j in sel_set:
            continue
        L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
        if j != j_star:
            d[j] -= L[j, t] ** 2
    state["t"] = t + 1
    state["selected"].append(j_star)


def _entropy_gain(state, j):
    """Shifted entropy marginal: ½ log(d_j / D_FLOOR), clipped at 0."""
    return max(0.5 * np.log(max(state["d"][j], 1e-300) / D_FLOOR), 0.0)


def _mi_gain(state, Sigma, P_full, j):
    """MI marginal ½[log d_j + log P_jj]; clipped at 0 for the ratio rule."""
    dj = max(state["d"][j], 1e-300)
    pj = max(P_full[j], 1e-300)
    return max(0.5 * (np.log(dj) + np.log(pj)), 0.0)


def _trace_gains(state, Sigma, cand):
    """Trace-objective marginals of the candidates `cand` (array of
    unselected indices): the explained variance
        sum_{i unselected} r_ij² / d_j
    of greedy_select._explained_variance on the current Cholesky state,
    i.e. the drop in the total conditional variance of the unselected
    benchmarks if j were added (the i = j term removes d_j itself; the
    score of baseline_select.greedy_trace_select).  Gains of candidates
    with d_j <= TRACE_D_FLOOR (an essentially singular residual) are 0,
    like greedy_trace_select's -inf; gains are >= 0."""
    cand = np.asarray(cand, dtype=int)
    N = len(state["d"])
    sel_set = set(state["selected"])
    unsel = np.array([j for j in range(N) if j not in sel_set])
    ev = _explained_variance(Sigma, state["L"], state["d"], state["t"],
                             cand, unsel)
    ev[state["d"][cand] <= TRACE_D_FLOOR] = 0.0
    return np.maximum(ev, 0.0)


def _facloc_similarity(Sigma):
    """Similarity matrix S of baseline_select.facility_location_select
    (similarity "clipped"), with its arithmetic: R = Sigma rescaled to a
    unit diagonal, s_ij = max(R_ij, 0), s_jj = 1."""
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    S = np.maximum(R, 0.0)
    np.fill_diagonal(S, 1.0)
    return S


def _facloc_gains(S, cover, cand):
    """Facility-location marginals of the candidates `cand` (array of
    unselected indices) given the cover c_j = max_{a ∈ A} s_ja of the
    current selection (0 for A = ∅):
        gain_i = sum_{j=1..N} max(s_ji − c_j, 0),
    over ALL N targets, selected and ineligible ones included
    (facility_location_select's gain, the same arithmetic; S symmetric).
    Gains are >= 0, and the gain of an unselected i is at least its own
    term 1 − c_i, positive unless a selected benchmark covers i perfectly
    (s_ia = 1, e.g. a duplicated column)."""
    cand = np.asarray(cand, dtype=int)
    return np.sum(np.maximum(S[:, cand] - cover[:, None], 0.0), axis=0)


def _complement_precision_full(Sigma, selected):
    from greedy_select import _complement_precision_diag
    N = Sigma.shape[0]
    comp = np.array(sorted(set(range(N)) - set(selected)))
    P_full = np.zeros(N)
    P_full[comp] = _complement_precision_diag(Sigma, comp)
    return P_full


def _tiebreak_by_explained(state, Sigma, cand, weights):
    """Among tied candidates, pick the max explained variance × weight.

    Explained variance is greedy_select._explained_variance on the current
    Cholesky state (rows of unselected benchmarks are maintained up to
    column t−1, the same layout as greedy_entropy).  Weighted values within
    TIE_RTOL · max(|best|, s · max weight), s = max_i Σ_ii, are tied (the
    variance-scale noise floor of greedy_entropy, carried through the
    weights, so the result does not depend on the cost units) and go to
    the lowest index.

    Returns (j_star, n_candidates, n_index_tied) like
    greedy_select._tie_break: n_index_tied > 1 is the number of candidates
    still tied when the lowest index decided, else 1.
    """
    cand = np.asarray(cand, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if len(cand) == 1:
        return int(cand[0]), 1, 1
    N = len(state["d"])
    sel_set = set(state["selected"])
    unsel = np.array([j for j in range(N) if j not in sel_set])
    ev = _explained_variance(Sigma, state["L"], state["d"], state["t"],
                             cand, unsel)
    scale = float(np.max(np.diag(Sigma))) * float(np.max(weights))
    tied = _near_max(ev * weights, scale)
    return int(cand[np.argmax(tied)]), len(cand), int(tied.sum())


def budgeted_select(Sigma, costs, budget, objective="entropy",
                    eligible=None):
    """Cost-effective greedy + best-singleton fallback (Appendix B.3).

    Both strategies are scored with the SAME shifted objective the
    gain/cost rule uses (sum of shifted marginals), as in Krause &
    Guestrin: comparing unshifted log-dets would trivially favor the
    singleton on a correlation matrix.

    Ratio ties: gain/cost ratios tie whenever costs tie (many MMLU
    subjects have exactly 100 questions) and the gains tie, which for the
    entropy objective on a correlation matrix they do at step 1.  Ratios
    within a relative TIE_RTOL of the best (TIE_RTOL · best_ratio, so the
    tie set does not depend on the cost units) are treated as tied and
    broken by explained variance / cost (the deterministic tie-break of
    greedy_select, per unit cost), then by the lowest index.  This holds
    for every objective; without near-tied ratios the pick is the plain
    argmax with identical arithmetic.

    Entropy objective only: the singleton is chosen among value-tied
    affordable benchmarks by explained variance, and replaces the greedy
    set only if its value exceeds greedy_value by more than
    TIE_RTOL · max(1, greedy_value) (otherwise ~1e-16 rounding in the
    diagonal decides).  The MI fallback keeps the strict comparison and
    Python's max (single values ½ log P_jj differ generically).

    objective : "entropy" (shifted entropy gains ½ log(d_j / D_FLOOR)),
        "mi" (MI gains, heuristic) or "trace" (explained variance of the
        unselected benchmarks, _trace_gains: f(A) = tr Σ − tr Σ_{Ā|A}, the
        objective of baseline_select.greedy_trace_select; unshifted, since
        f(∅) = 0 and f is monotone).  Trace ratio ties use the same rule;
        as the gain already is the explained variance, the explained-
        variance tie-break re-ties them and the lowest index decides
        (tie_breaks == index_ties, as in greedy_trace_select).  The trace
        singleton is the affordable benchmark of largest unconditional
        explained variance sum_i Σ_ij² / Σ_jj (near-ties at the variance
        scale s = max_i Σ_ii: lowest index) and replaces the greedy set
        only if its value exceeds greedy_value by more than
        TIE_RTOL · max(s, greedy_value).  The entropy and MI arithmetic is
        unchanged by the trace objective.
        "facloc" (the headline method): facility location on the clipped
        correlation matrix, F(A) = sum_{j=1..N} max_{i ∈ A} s_ij with S
        built as baseline_select.facility_location_select builds it
        (_facloc_similarity: unit-diagonal rescaling, clipped at 0,
        s_jj = 1), targets j over all N benchmarks (ineligible ones
        included), gain of i given the cover c_j = max_{a ∈ A} s_ja:
        sum_j max(s_ji − c_j, 0) (_facloc_gains); unshifted, since
        F(∅) = 0 and F is monotone.  Ratio ties use the rule of every
        objective (relative TIE_RTOL, explained variance per unit cost,
        then the lowest index).  The facloc singleton is the affordable
        benchmark of largest F({j}) = sum_i s_ij (near-ties within
        TIE_RTOL · max(|best|, 1), facility_location_select's step-1
        rule: explained variance, then the lowest index) and replaces the
        greedy set only if its value exceeds greedy_value by more than
        TIE_RTOL · max(1, greedy_value).  With unit costs and budget k
        the greedy part is facility_location_select(Sigma, k)'s selection
        (test_eval_cost_budget.py), except that its gain ties are absolute
        below 1 (TIE_RTOL · max(|best|, 1)) where the ratio ties here are
        relative, and that it stops once every gain is 0 (F saturated,
        e.g. by duplicated columns) where facility_location_select goes on
        by explained variance; the singleton then never wins (it is the
        first greedy pick).  Guarantee: s_ij >= 0, so F is normalized,
        monotone and submodular, and the better of the cost-effective
        greedy (candidates restricted to those that still fit, as here)
        and the best affordable singleton is within (1 − 1/e)/2 ≈ 0.316
        of the best affordable set of eligible benchmarks (Krause &
        Guestrin 2005, KraGue05b in the paper), up to the TIE_RTOL
        tolerances of the ratio ties and the singleton margin.  (For the
        other objectives the bound needs assumptions: MI is not monotone,
        the trace objective is not submodular in general, and the shifted
        entropy gains are clipped at 0.)  No other objective's arithmetic
        changes.

    eligible : optional (N,) bool mask; only eligible benchmarks can be
        picked (greedy steps and the singleton), while all of them stay in
        Σ, in the MI complement and in the explained variance.
    (The first_seed option of knap_entropy_rf, a random first pick from
    the step-1 tie set, was removed with that method, protocol rule D3.)

    `costs` may be any sequence of positive per-benchmark costs.

    Returns a dict:
        selected   list of picks (greedy order, or [singleton])
        singleton  True if the best-singleton fallback replaced the greedy
                   set
        value      shifted objective of `selected`
        tie_breaks [(step, n_candidates)] picks of `selected` made among
                   near-tied candidates
        index_ties [(step, n_tied)] the subset decided by the lowest index
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective {objective!r} not in {OBJECTIVES}")
    costs = np.asarray(costs, dtype=float)
    N = Sigma.shape[0]
    el = _eligible(eligible, N)
    state = _chol_state_init(Sigma)
    spent = 0.0
    greedy_value = 0.0  # accumulated shifted objective
    tie_breaks, index_ties = [], []
    if objective == "facloc":
        S_fl = _facloc_similarity(Sigma)
        cover = np.zeros(N)          # max_{a ∈ A} s_ja per target j

    while True:
        best_j, best_ratio = -1, -np.inf
        if objective == "mi":
            P_full = _complement_precision_full(Sigma, state["selected"])
        elif objective == "trace":
            # Explained variance of every unselected benchmark (one call)
            sel_set = set(state["selected"])
            unsel = np.array([j for j in range(N) if j not in sel_set],
                             dtype=int)
            trace_gain = np.zeros(N)
            if len(unsel):
                trace_gain[unsel] = _trace_gains(state, Sigma, unsel)
        elif objective == "facloc":
            # Facility-location gain of every unselected benchmark
            sel_set = set(state["selected"])
            unsel = np.array([j for j in range(N) if j not in sel_set],
                             dtype=int)
            fl_gain = np.zeros(N)
            if len(unsel):
                fl_gain[unsel] = _facloc_gains(S_fl, cover, unsel)
        cand_j, cand_ratio = [], []  # all feasible (j, ratio)
        for j in range(N):
            if (j in state["selected"] or spent + costs[j] > budget
                    or (el is not None and not el[j])):
                continue
            if objective == "entropy":
                gain = _entropy_gain(state, j)
            elif objective == "mi":
                gain = _mi_gain(state, Sigma, P_full, j)
            elif objective == "trace":
                gain = float(trace_gain[j])
            else:
                gain = float(fl_gain[j])
            ratio = gain / costs[j]
            cand_j.append(j)
            cand_ratio.append(ratio)
            if ratio > best_ratio:
                best_ratio = ratio
                best_j = j
        if best_j < 0 or best_ratio <= 0:
            break
        cand_j, cand_ratio = np.array(cand_j), np.array(cand_ratio)
        # Purely relative: ratio magnitudes scale with the cost units
        tied = cand_j[_near_max(cand_ratio, 0.0) & (cand_ratio > 0)]
        step = state["t"]
        best_j, n_cand, n_index = _tiebreak_by_explained(
            state, Sigma, tied, 1.0 / costs[tied])
        if n_cand > 1:
            tie_breaks.append((step, n_cand))
        if n_index > 1:
            index_ties.append((step, n_index))
        if objective == "entropy":
            best_gain = _entropy_gain(state, best_j)
        elif objective == "mi":
            best_gain = _mi_gain(state, Sigma, P_full, best_j)
        elif objective == "trace":
            best_gain = float(trace_gain[best_j])
        else:
            best_gain = float(fl_gain[best_j])
        spent += costs[best_j]
        greedy_value += best_gain
        _chol_state_add(state, Sigma, best_j)
        if objective == "facloc":
            cover = np.maximum(cover, S_fl[:, best_j])

    result = {"selected": list(state["selected"]), "singleton": False,
              "value": greedy_value, "tie_breaks": tie_breaks,
              "index_ties": index_ties}

    # Best-singleton fallback under the same shifted objective
    afford = [j for j in range(N)
              if costs[j] <= budget and (el is None or el[j])]
    if afford:
        n_cand = n_index = 1
        if objective == "entropy":
            def single_value(j):
                return max(0.5 * np.log(max(Sigma[j, j], 1e-300) / D_FLOOR),
                           0.0)
            # Value ties -> explained variance; noise must not flip fallback
            # (values are dimensionless log-gains: noise floor 1)
            vals = np.array([single_value(j) for j in afford])
            tied = np.array(afford)[_near_max(vals, 1.0)]
            best_single, n_cand, n_index = _tiebreak_by_explained(
                _chol_state_init(Sigma), Sigma, tied, np.ones(len(tied)))
            margin = TIE_RTOL * max(1.0, abs(greedy_value))
        elif objective == "trace":
            # Unconditional explained variance sum_i Σ_ij² / Σ_jj
            state0 = _chol_state_init(Sigma)
            vals = _trace_gains(state0, Sigma, np.array(afford, dtype=int))
            single_vals = dict(zip(afford, vals.tolist()))

            def single_value(j):
                return single_vals[j]
            scale = float(np.max(np.diag(Sigma)))
            tied = np.array(afford)[_near_max(vals, scale)]
            best_single, n_cand, n_index = _tiebreak_by_explained(
                state0, Sigma, tied, np.ones(len(tied)))
            margin = TIE_RTOL * max(scale, abs(greedy_value))
        elif objective == "facloc":
            # F({j}) = sum_i s_ij, the step-1 gain (dimensionless: scale 1)
            vals = _facloc_gains(S_fl, np.zeros(N),
                                 np.array(afford, dtype=int))
            single_vals = dict(zip(afford, vals.tolist()))

            def single_value(j):
                return single_vals[j]
            tied = np.array(afford)[_near_max(vals, 1.0)]
            best_single, n_cand, n_index = _tiebreak_by_explained(
                _chol_state_init(Sigma), Sigma, tied, np.ones(len(tied)))
            margin = TIE_RTOL * max(1.0, abs(greedy_value))
        else:
            P0 = _complement_precision_full(Sigma, [])
            def single_value(j):
                dj = max(Sigma[j, j], 1e-300)
                pj = max(P0[j], 1e-300)
                return max(0.5 * (np.log(dj) + np.log(pj)), 0.0)
            best_single = max(afford, key=single_value)
            margin = 0.0
        if single_value(best_single) > greedy_value + margin:
            result.update(
                selected=[best_single], singleton=True,
                value=single_value(best_single),
                tie_breaks=[(0, n_cand)] if n_cand > 1 else [],
                index_ties=[(0, n_index)] if n_index > 1 else [])
    return result


def budgeted_greedy(Sigma, costs, budget, objective="entropy",
                    eligible=None):
    """Selected benchmarks of budgeted_select (a list)."""
    return budgeted_select(Sigma, costs, budget, objective,
                           eligible=eligible)["selected"]


def order_fill(order, costs, budget):
    """Add benchmarks in the given order while they fit (skip oversized)."""
    sel, spent = [], 0.0
    for j in order:
        if spent + costs[j] <= budget:
            sel.append(int(j))
            spent += costs[j]
    return sel


def cheap_first_order(Sigma, costs, eligible=None):
    """Cheapest-first order of the eligible benchmarks (cheap_first).

    Ascending cost.  Costs within a relative TIE_RTOL of the cheapest
    remaining one (in practice exact ties, e.g. the ten 100-question MMLU
    subjects) are near-tied and broken like every selector's ties: by the
    explained variance of the not yet ordered benchmarks given the ones
    ordered so far (greedy_select._tie_break), then by the lowest index.
    Since costs never decrease along the order, order_fill of it takes a
    prefix.  Returns {"selected", "tie_breaks", "index_ties"} in
    greedy_select's conventions (steps are positions in the order).
    """
    costs = np.asarray(costs, dtype=float)
    N = Sigma.shape[0]
    el = _eligible(eligible, N)
    remaining = np.arange(N) if el is None else np.flatnonzero(el)
    state = _chol_state_init(Sigma)
    tie_breaks, index_ties = [], []
    while len(remaining):
        cand = remaining[_near_max(-costs[remaining], 0.0)]
        t = state["t"]
        if len(cand) > 1:
            sel_set = set(state["selected"])
            unsel = np.array([j for j in range(N) if j not in sel_set])
            j_star, n_cand, n_index = _tie_break(Sigma, state["L"],
                                                 state["d"], t, cand, unsel)
        else:
            j_star, n_cand, n_index = int(cand[0]), 1, 1
        if n_cand > 1:
            tie_breaks.append((t, n_cand))
        if n_index > 1:
            index_ties.append((t, n_index))
        _chol_state_add(state, Sigma, j_star)
        remaining = remaining[remaining != j_star]
    return {"selected": list(state["selected"]), "tie_breaks": tie_breaks,
            "index_ties": index_ties}


def legacy_random_seed(split):
    """Seed of the pre-matched-design random baseline of this script:
    (SEED + 7, fold) in rep 0, (SEED + 7, rep, fold) in rep r > 0."""
    tag = SEED + RANDOM_SEED_OFFSET
    if split.rep == 0:
        return (tag, int(split.fold))
    return (tag, int(split.rep), int(split.fold))


def legacy_random_orders(N, split, n_budgets, eligible):
    """--legacy-random: one order per budget, the b-th rng.permutation(N)
    of rng = default_rng(legacy_random_seed(split)) (the historical draw
    sequence), restricted to the eligible benchmarks like random_select's
    permutation."""
    rng = np.random.default_rng(legacy_random_seed(split))
    el = _eligible(eligible, N)
    orders = []
    for _ in range(n_budgets):
        perm = rng.permutation(N)
        if el is not None:
            perm = perm[el[perm]]
        orders.append(perm.tolist())
    return orders


def split_orders(R, costs, split, eligible, n_budgets=None,
                 legacy_random=False):
    """Budget-independent orders of the order-filled methods on one split
    (shared eligibility mask and the split's common random numbers).
    With legacy_random, "random" is instead the list of n_budgets
    historical per-budget orders (legacy_random_orders).  card_facloc is
    facility_location_select's full order (deterministic, no random
    numbers)."""
    N = R.shape[0]
    orders = {
        "card_entropy": greedy_entropy(R, N, eligible=eligible),
        "card_mi": greedy_mi(R, N, eligible=eligible),
        "cheap_first": cheap_first_order(R, costs, eligible=eligible),
        "card_trace": greedy_trace_select(R, N, eligible=eligible),
        "card_facloc": facility_location_select(R, N, eligible=eligible),
    }
    if legacy_random:
        orders["random"] = legacy_random_orders(N, split, n_budgets,
                                                eligible)
    else:
        orders["random"] = random_select(
            R, N, seed=split.seed(SEED + RANDOM_SEED_OFFSET),
            eligible=eligible)
    return orders


def select_for_budget(method, R, costs, budget, orders, split, eligible,
                      b_idx=None):
    """Selection of `method` under `budget` (the b_idx-th budget) on one
    split.

    Returns {"selected", "tie_breaks", "index_ties" (counts),
    "singleton"}; for order-filled methods the
    counts cover the steps of the order up to the last pick.
    """
    if method in KNAP_OBJECTIVE:
        res = budgeted_select(R, costs, budget, KNAP_OBJECTIVE[method],
                              eligible=eligible)
        return {"selected": res["selected"],
                "tie_breaks": len(res["tie_breaks"]),
                "index_ties": len(res["index_ties"]),
                "singleton": res["singleton"]}
    if method not in orders:
        raise ValueError(method)
    res = orders[method]
    if isinstance(res, list):          # --legacy-random: per-budget orders
        return {"selected": order_fill(res[b_idx], costs, budget),
                "tie_breaks": 0, "index_ties": 0, "singleton": False}
    order = res["selected"]
    sel = order_fill(order, costs, budget)
    n_used = order.index(sel[-1]) + 1 if sel else 0
    return {"selected": sel,
            "tie_breaks": sum(1 for step, _ in res["tie_breaks"]
                              if step < n_used),
            "index_ties": index_tie_count(res, n_used) if n_used else 0,
            "singleton": False}


def sequential_cost(sel, costs):
    """True cost of a selection summed in pick order (the arithmetic of the
    selectors' feasibility checks, so cost <= budget holds exactly)."""
    spent = 0.0
    for j in sel:
        spent += float(costs[j])
    return spent


def suite_r2(B_test, O_test, selected, R, col_mean, col_std, target=None,
             bounds=None):
    """Full-suite R² of a selection on the validation models.

    impute_and_evaluate's predictions (train-standardized units clipped to
    ±10, ridge 0.01 on R_AA, each model's observed selected benchmarks
    predicting its observed unselected ones), but pooled over ALL observed
    validation cells: a selected cell counts as known (residual 0), a model
    without an observed selected benchmark is predicted by the training
    mean (0), so
        r2_suite = 1 − SS_res(unselected) / SS_tot(all observed).
    Unlike the paper's R² (pooled over the unselected benchmarks only),
    the target set is the same for every selection, so selections of
    different size compare directly; the empty selection scores 0.
    With complete validation data SS_res is impute_and_evaluate's, i.e.
    (1 − r2_suite) · SS_tot(all) = (1 − r2) · SS_tot(unselected).
    The protocol rules of impute_and_evaluate apply: only the cells of the
    targets count (target: the split's cv_split.target_mask, rule D1; a
    non-target is never selected, so it leaves SS_res and SS_tot alike),
    and the predictions are clipped to the valid score range (bounds:
    score_bounds.dataset_bounds, rule D2; None for either: every column /
    no clip).
    (Defined here for the --fixed-cost runs; eval_token_costs.py imports
    it.)
    """
    RIDGE = 0.01
    X = O_test * (B_test - col_mean[None, :]) / col_std[None, :]
    X = np.clip(X, -10, 10) * O_test
    obs = O_test > 0
    N = B_test.shape[1]
    # scored cells: observed targets (all observed cells without a mask)
    obs_t = (obs if target is None
             else obs & np.asarray(target, dtype=bool)[None, :])
    z_lo, z_hi = z_limits(bounds, col_mean, col_std)
    in_sel = np.zeros(N, dtype=bool)
    in_sel[np.asarray(list(selected), dtype=int)] = True
    ss_res = ss_tot = 0.0
    cache = {}
    for i in range(B_test.shape[0]):
        o = obs_t[i]
        ss_tot += float(np.sum(X[i, o] ** 2))
        a = np.flatnonzero(obs[i] & in_sel)
        e = np.flatnonzero(o & ~in_sel)
        if len(e) == 0:
            continue
        if len(a) == 0:
            ss_res += float(np.sum(X[i, e] ** 2))
            continue
        key = (a.tobytes(), e.tobytes())
        if key not in cache:
            R_oo = R[np.ix_(a, a)] + RIDGE * np.eye(len(a))
            cache[key] = np.linalg.solve(R_oo, R[np.ix_(e, a)].T).T
        pred, _ = clip_z(cache[key] @ X[i, a], z_lo[e], z_hi[e])
        ss_res += float(np.sum((X[i, e] - pred) ** 2))
    if ss_tot <= 0:
        return math.nan
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# ── Main experiment ──────────────────────────────────────────────

def load_costs(counts_path, benchmark_names):
    """Costs of the benchmarks (in matrix column order) from a JSON
    {benchmark_name: cost} file, plus the file's SHA-256."""
    raw = pathlib.Path(counts_path).read_bytes()
    counts = json.loads(raw)
    missing = [b for b in benchmark_names if b not in counts]
    if missing:
        raise ValueError(f"{counts_path}: no counts for {len(missing)} "
                         f"benchmarks, e.g. {missing[:5]}")
    costs = np.array([float(counts[b]) for b in benchmark_names])
    bad = [b for b, c in zip(benchmark_names, costs)
           if not (np.isfinite(c) and c > 0)]
    if bad:
        raise ValueError(f"{counts_path}: costs must be positive, "
                         f"e.g. {bad[:5]}")
    return costs, hashlib.sha256(raw).hexdigest()


def _budget_label(budget, frac):
    return f"{int(budget)} ({100 * frac:g}%)"


def _nan_mean_std(a, axis=None):
    """np.nanmean / np.nanstd (ddof=0) without all-NaN-slice warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)


def run(ds_name, counts_path, budgets_frac, n_repeats=N_REPEATS,
        n_folds=N_FOLDS, legacy_random=False):
    """Every method and budget on the matched splits of one dataset
    (legacy_random: the historical random draws, see the module
    docstring).

    Returns the save dict (see the module docstring for its keys)."""
    d = np.load(matrix_path(ds_name), allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = [str(b) for b in d["benchmark_names"]]
    M, N = B.shape
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)     # D2

    costs, counts_sha = load_costs(counts_path, benchmark_names)
    total_cost = costs.sum()
    budgets = [f * total_cost for f in budgets_frac]

    print(f"{display_name(ds_name)}: {M} models × {N} tasks, total cost "
          f"{int(total_cost)} examples")
    print(f"cost range: {int(costs.min())} ({benchmark_names[int(np.argmin(costs))]})"
          f" – {int(costs.max())} ({benchmark_names[int(np.argmax(costs))]})")
    print(f"budgets: {[int(b) for b in budgets]}", flush=True)

    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED))
    S, nb = len(splits), len(budgets)
    print(f"{n_repeats} × {n_folds}-fold CV = {S} matched splits, "
          f"{HOLDOUT_FRAC:.0%} holdout, selectable: benchmarks with "
          f">= {MIN_TRAIN_OBS} training models", flush=True)
    if legacy_random:
        print("random: historical per-budget draws (--legacy-random), not "
              "the random baseline shared with the other scripts",
              flush=True)

    out = {"budgets": np.array(budgets),
           "budgets_frac": np.array(budgets_frac),
           "costs": costs,
           "benchmark_names": np.array(benchmark_names)}
    for m in METHODS:
        out[f"{m}_r2"] = np.full((nb, S), np.nan)
        out[f"{m}_size"] = np.full((nb, S), np.nan)
    for m in METHODS:
        out[f"{m}_rmse"] = np.full((nb, S), np.nan)
        out[f"{m}_cost"] = np.full((nb, S), np.nan)
        out[f"{m}_selected"] = np.full((nb, S, N), -1, dtype=np.int64)
        out[f"{m}_index_ties"] = np.zeros((nb, S), dtype=np.int64)
        out[f"{m}_tie_breaks"] = np.zeros((nb, S), dtype=np.int64)
    for m in KNAP_OBJECTIVE:
        out[f"{m}_singleton"] = np.zeros((nb, S), dtype=bool)
    out.update(
        rep=np.array([s.rep for s in splits], dtype=np.int64),
        fold=np.array([s.fold for s in splits], dtype=np.int64),
        split_id=np.array([s.split_id for s in splits], dtype=np.int64),
        n_train=np.array([len(s.train_idx) for s in splits], dtype=np.int64),
        n_val=np.array([len(s.val_idx) for s in splits], dtype=np.int64),
        n_eligible=np.zeros(S, dtype=np.int64),
        eligible=np.zeros((S, N), dtype=bool),
        estimation_failed=np.zeros(S, dtype=bool))

    for i, s in enumerate(splits):
        t0 = time.time()
        B_train, O_train = B[s.train_idx], O[s.train_idx]
        B_test, O_test = B[s.val_idx], O[s.val_idx]

        # One eligibility mask per split, shared by every method and budget
        el = eligible_mask(O_train)
        target = target_mask(O_train)        # D1, shared by every method
        n_el = int(el.sum())
        out["eligible"][i] = el
        out["n_eligible"][i] = n_el

        try:
            R, col_mean, col_std = estimate_corr(B_train, O_train)
        except Exception as e:
            print(f"  rep {s.rep} fold {s.fold}: estimation failed: {e}",
                  flush=True)
            out["estimation_failed"][i] = True
            continue

        orders = split_orders(R, costs, s, el, nb, legacy_random)
        imputed = {}  # identical selections recur across methods/budgets
        for method in METHODS:
            for b_idx, budget in enumerate(budgets):
                res = select_for_budget(method, R, costs, budget, orders, s,
                                        el, b_idx)
                sel = res["selected"]
                if sel:
                    key = tuple(sel)
                    if key not in imputed:
                        imputed[key] = impute_and_evaluate(
                            B_test, O_test, sel, R, col_mean, col_std, N,
                            target=target, bounds=bounds)
                    out[f"{method}_r2"][b_idx, i] = imputed[key]["r2"]
                    out[f"{method}_rmse"][b_idx, i] = imputed[key]["rmse"]
                out[f"{method}_size"][b_idx, i] = len(sel)
                out[f"{method}_cost"][b_idx, i] = costs[sel].sum()
                out[f"{method}_selected"][b_idx, i, :len(sel)] = sel
                out[f"{method}_index_ties"][b_idx, i] = res["index_ties"]
                out[f"{method}_tie_breaks"][b_idx, i] = res["tie_breaks"]
                if method in KNAP_OBJECTIVE:
                    out[f"{method}_singleton"][b_idx, i] = res["singleton"]
        print(f"  rep {s.rep} fold {s.fold} done ({time.time()-t0:.1f}s), "
              f"{n_el}/{N} eligible", flush=True)

    out.update(methods=np.array(METHODS),
               holdout_frac=np.float64(HOLDOUT_FRAC),
               n_folds=np.int64(n_folds), n_repeats=np.int64(n_repeats),
               **{k: (np.bool_(v) if isinstance(v, bool) else np.int64(v))
                  for k, v in protocol_config().items()},
               seed=np.int64(SEED),
               total_cost=np.float64(total_cost),
               counts_file=np.array(pathlib.Path(counts_path).name),
               counts_sha256=np.array(counts_sha),
               random_legacy=np.bool_(legacy_random))
    return out


# ── Paired statistics over matched splits ────────────────────────

def paired_rows(ds_name, out):
    """paired_stats.paired_stats of R² for every method pair (oriented by
    paired_stats.orient) and every budget, as CSV rows (PAIRED_FIELDS)."""
    rows = []
    for b_idx, (budget, frac) in enumerate(zip(out["budgets"],
                                               out["budgets_frac"])):
        for ma, mb in orient(METHODS):
            st = paired_stats(out[f"{ma}_r2"][b_idx], out[f"{mb}_r2"][b_idx],
                              out["n_train"], out["n_val"], out["rep"])
            rows.append({"dataset": ds_name, "holdout_frac": HOLDOUT_FRAC,
                         "budget_frac": float(frac), "budget": float(budget),
                         "method_a": ma, "method_b": mb, **st})
    return rows


def write_paired_csv(path, rows):
    write_csv(rows, path, PAIRED_FIELDS)
    print(f"Paired statistics saved to {path}", flush=True)


# ── Reporting ────────────────────────────────────────────────────

def print_summary(ds_name, out, paired):
    budgets, fracs = out["budgets"], out["budgets_frac"]
    S = len(out["split_id"])
    print(f"\n== {display_name(ds_name)}: imputation R² vs example budget "
          f"(mean over {S} splits) ==")
    print(f"{'method':<30}" + "".join(
        f"{_budget_label(b, f):>16}" for b, f in zip(budgets, fracs)))
    for m in METHODS:
        cells = []
        for b_idx in range(len(budgets)):
            mu, _ = _nan_mean_std(out[f"{m}_r2"][b_idx])
            nsel, _ = _nan_mean_std(out[f"{m}_size"][b_idx])
            cells.append(f"{mu:.3f}/{nsel:.0f}")
        print(f"{METHOD_LABEL[m]:<30}" + "".join(f"{c:>16}" for c in cells))
    print("(cell = mean R² / mean #benchmarks selected)")

    n_el = out["n_eligible"]
    N = out["eligible"].shape[1]
    print(f"eligible benchmarks per split: {n_el.min()}–{n_el.max()} of {N} "
          f"({int(np.sum(n_el < N))}/{S} splits restricted)"
          + (f"; estimation failed on {int(out['estimation_failed'].sum())}"
             " splits" if out["estimation_failed"].any() else ""))
    print("index ties (picks decided by column order), cells affected / "
          "total picks over budgets × splits: " + ", ".join(
              f"{m} {int(np.count_nonzero(out[f'{m}_index_ties']))}/"
              f"{int(out[f'{m}_index_ties'].sum())}" for m in METHODS))
    wins = [f"{m} {int(out[f'{m}_singleton'].sum())}" for m in KNAP_OBJECTIVE]
    print("best-singleton fallback won (budget × split cells): "
          + ", ".join(wins))

    lookup = {(r["budget_frac"], r["method_a"], r["method_b"]): r
              for r in paired}
    print(f"\n== {display_name(ds_name)}: paired R² differences a − b "
          f"(mean; * = Nadeau–Bengio 95% CI excludes 0) ==")
    print(f"{'pair':<34}" + "".join(f"{100 * f:>9g}%" for f in fracs))
    for ma, mb in PRINT_PAIRS:
        cells = []
        for f in fracs:
            r = lookup[(float(f), ma, mb)]
            if r["n_splits"] == 0:
                cells.append("n/a")
                continue
            sig = (np.isfinite(r["ci_low"])
                   and (r["ci_low"] > 0 or r["ci_high"] < 0))
            cells.append(f"{r['mean_diff']:+.3f}{'*' if sig else ' '}")
        print(f"{ma + ' − ' + mb:<34}" + "".join(f"{c:>10}" for c in cells))


def plot_budget(ds_name, out, methods=PLOT_METHODS):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = 100 * np.asarray(out["budgets_frac"], dtype=float)
    for m in methods:
        mus, sds = _nan_mean_std(out[f"{m}_r2"], axis=1)
        lw = 1.5 if m.startswith("knap") else 1.0
        ax.plot(x, mus, METHOD_STYLE[m], color=METHOD_COLOR[m],
                label=METHOD_LABEL[m], markersize=4, linewidth=lw)
        ax.fill_between(x, mus - sds, mus + sds, color=METHOD_COLOR[m],
                        alpha=0.10)
    ax.set_xlabel("Example budget (% of full suite)")
    ax.set_ylabel("Imputation $R^2$")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    out_path = OUT / f"cost_budget_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}", flush=True)


# ── Fixed per-benchmark overhead (--fixed-cost) ──────────────────

def fixed_overhead(costs, ratio):
    """(c0, true costs) of overhead level `ratio`: c0 = ratio ·
    median(costs) and the true cost c0 + costs[j] of every benchmark.
    Ratio 0 returns (0.0, costs) with `costs` itself, so the plain run's
    arithmetic is untouched."""
    costs = np.asarray(costs, dtype=float)
    ratio = float(ratio)
    if not (math.isfinite(ratio) and ratio >= 0.0):
        raise ValueError(f"fixed-cost ratio must be finite and >= 0, got "
                         f"{ratio!r}")
    if ratio == 0.0:
        return 0.0, costs
    c0 = ratio * float(np.median(costs))
    return c0, costs + c0


def fixed_levels(costs, ratios, budgets_frac):
    """([(c0, true costs)] per ratio, (n_levels, n_budgets) budgets): the
    budget of fraction f at a level is f · sum_j true cost (ratio 0: the
    plain run's f · total_cost, same arithmetic)."""
    levels = [fixed_overhead(costs, r) for r in ratios]
    budgets = np.array([[f * true.sum() for f in budgets_frac]
                        for _, true in levels], dtype=np.float64)
    return levels, budgets


def evaluate_fixed_split(B, O, split, levels, budgets, bounds=None):
    """Every method, overhead level and budget on one matched split.

    levels, budgets: fixed_levels'.  The selections see the training
    models (correlation matrix, eligibility mask) and the true costs only;
    the validation models enter the evaluation alone.  All levels share
    the split's eligibility mask, correlation estimate, cost-blind greedy
    orders and random permutation (common random numbers); cheapest first
    is ordered by each level's true costs.  Returns {"failed": bool,
    "eligible": (N,) bool, plus, unless the estimate failed, the per-split
    arrays of run_fixed_arrays without the split axis}; "error" holds the
    estimation error.
    """
    N = B.shape[1]
    nf, nb = budgets.shape
    B_train, O_train = B[split.train_idx], O[split.train_idx]
    B_test, O_test = B[split.val_idx], O[split.val_idx]
    el = eligible_mask(O_train)
    target = target_mask(O_train)            # D1; bounds: D2 (None: no clip)
    row = {"failed": False, "eligible": el}
    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:  # logged by the caller, row left NaN
        row.update(failed=True, error=str(e))
        return row
    shared = split_orders(R, levels[0][1], split, el)
    for m in METHODS:
        for key in ("r2", "r2_suite", "rmse", "cost"):
            row[f"{m}_{key}"] = np.full((nf, nb), np.nan)
        for key in ("size", "index_ties", "tie_breaks"):
            row[f"{m}_{key}"] = np.zeros((nf, nb), dtype=np.int64)
        row[f"{m}_violations"] = np.zeros((nf, nb), dtype=bool)
        row[f"{m}_selected"] = np.full((nf, nb, N), -1, dtype=np.int64)
    for m in KNAP_OBJECTIVE:
        row[f"{m}_singleton"] = np.zeros((nf, nb), dtype=bool)
    imputed = {}  # identical selections recur across methods, levels, budgets
    for f, (_, true) in enumerate(levels):
        orders = dict(shared, cheap_first=cheap_first_order(R, true,
                                                            eligible=el))
        for m in METHODS:
            for b in range(nb):
                budget = budgets[f, b]
                res = select_for_budget(m, R, true, budget, orders, split,
                                        el, b)
                sel = res["selected"]
                key = tuple(sel)
                if key not in imputed:
                    r2 = rmse = np.nan
                    if sel:
                        imp = impute_and_evaluate(B_test, O_test, sel, R,
                                                  col_mean, col_std, N,
                                                  target=target,
                                                  bounds=bounds)
                        r2, rmse = imp["r2"], imp["rmse"]
                    imputed[key] = (r2, rmse, suite_r2(
                        B_test, O_test, sel, R, col_mean, col_std,
                        target=target, bounds=bounds))
                r2, rmse, r2s = imputed[key]
                cost = sequential_cost(sel, true)
                row[f"{m}_r2"][f, b] = r2
                row[f"{m}_r2_suite"][f, b] = r2s
                row[f"{m}_rmse"][f, b] = rmse
                row[f"{m}_cost"][f, b] = cost
                row[f"{m}_violations"][f, b] = cost > budget
                row[f"{m}_size"][f, b] = len(sel)
                row[f"{m}_selected"][f, b, :len(sel)] = sel
                row[f"{m}_index_ties"][f, b] = res["index_ties"]
                row[f"{m}_tie_breaks"][f, b] = res["tie_breaks"]
                if m in KNAP_OBJECTIVE:
                    row[f"{m}_singleton"][f, b] = res["singleton"]
    return row


def run_fixed_arrays(B, O, benchmark_names, costs, budgets_frac, ratios,
                     n_repeats=N_REPEATS, n_folds=N_FOLDS, verbose=True,
                     bounds=None):
    """The fixed-overhead experiment on one dataset given as arrays (B with
    missing = 0, O mask, positive variable costs c_j; bounds: the valid
    score ranges of rule D2, None: no clip).  Returns the save dict of
    LOGS/cost_fixed_{ds}.npz (module docstring)."""
    B = np.asarray(B, dtype=np.float64)
    O = np.asarray(O, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    M, N = B.shape
    if costs.shape != (N,) or not np.all(np.isfinite(costs) & (costs > 0)):
        raise ValueError(f"need {N} positive finite costs")
    ratios = [float(r) for r in ratios]
    levels, budgets = fixed_levels(costs, ratios, budgets_frac)
    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED))
    S = len(splits)
    if verbose:
        for r, (c0, true), bud in zip(ratios, levels, budgets):
            print(f"  overhead {r:g} x median: c0 = {c0:.6g} per benchmark, "
                  f"suite {true.sum():.6g}, budgets "
                  f"{[float(f'{x:.6g}') for x in bud]}", flush=True)
        print(f"{n_repeats} × {n_folds}-fold CV = {S} matched splits, "
              f"{HOLDOUT_FRAC:.0%} holdout, selectable: benchmarks with "
              f">= {MIN_TRAIN_OBS} training models", flush=True)
    rows = []
    for s in splits:
        t0 = time.time()
        r = evaluate_fixed_split(B, O, s, levels, budgets, bounds)
        rows.append(r)
        if verbose:
            if r["failed"]:
                print(f"  rep {s.rep} fold {s.fold}: estimation failed: "
                      f"{r['error']}", flush=True)
            else:
                print(f"  rep {s.rep} fold {s.fold} done "
                      f"({time.time() - t0:.1f}s), "
                      f"{int(r['eligible'].sum())}/{N} eligible", flush=True)
    template = next((r for r in rows if not r["failed"]), None)
    if template is None:
        raise RuntimeError("estimation failed on every split")
    out = {}
    for key, val in template.items():
        if key in ("failed", "error"):
            continue
        val = np.asarray(val)
        fill = (np.nan if val.dtype.kind == "f"
                else -1 if key.endswith("_selected") else 0)
        arr = np.full((S,) + val.shape, fill, dtype=val.dtype)
        for i, r in enumerate(rows):
            if key in r:
                arr[i] = r[key]
        out[key] = arr
    out.update(
        rep=np.array([s.rep for s in splits], dtype=np.int64),
        fold=np.array([s.fold for s in splits], dtype=np.int64),
        split_id=np.array([s.split_id for s in splits], dtype=np.int64),
        n_train=np.array([len(s.train_idx) for s in splits], dtype=np.int64),
        n_val=np.array([len(s.val_idx) for s in splits], dtype=np.int64),
        n_eligible=out["eligible"].sum(axis=1).astype(np.int64),
        estimation_failed=np.array([r["failed"] for r in rows]),
        fixed_ratios=np.array(ratios, dtype=np.float64),
        fixed_costs=np.array([c0 for c0, _ in levels], dtype=np.float64),
        total_cost=np.array([true.sum() for _, true in levels],
                            dtype=np.float64),
        budgets=budgets, budgets_frac=np.array(budgets_frac, dtype=float),
        costs=costs, costs_true=np.array([t for _, t in levels]),
        median_cost=np.float64(np.median(costs)),
        benchmark_names=np.array(benchmark_names), methods=np.array(METHODS),
        holdout_frac=np.float64(HOLDOUT_FRAC), n_folds=np.int64(n_folds),
        n_repeats=np.int64(n_repeats),
        **{k: (np.bool_(v) if isinstance(v, bool) else np.int64(v))
           for k, v in protocol_config().items()},
        seed=np.int64(SEED))
    return out


def run_fixed(ds_name, counts_path, budgets_frac, ratios, n_repeats=N_REPEATS,
              n_folds=N_FOLDS):
    """run_fixed_arrays on data/<ds>.matrix.npz with the cost table
    `counts_path` (load_costs)."""
    path = matrix_path(ds_name)
    d = np.load(path, allow_pickle=False)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    names = [str(b) for b in d["benchmark_names"]]
    M, N = B.shape
    costs, counts_sha = load_costs(counts_path, names)
    print(f"{display_name(ds_name)}: {M} models × {N} tasks, variable cost "
          f"{costs.sum():.6g} (median {np.median(costs):.6g}, range "
          f"{costs.min():.6g}–{costs.max():.6g}); fixed overhead per "
          f"benchmark c0 = ratio × median, ratios {list(ratios)}",
          flush=True)
    out = run_fixed_arrays(B, O, names, costs, budgets_frac, ratios,
                           n_repeats, n_folds,
                           bounds=dataset_bounds(ds_name, names, DATA))
    out.update(counts_file=np.array(pathlib.Path(counts_path).name),
               counts_sha256=np.array(counts_sha),
               matrix_sha256=np.array(
                   hashlib.sha256(path.read_bytes()).hexdigest()))
    return out


def fixed_paired_rows(ds_name, out):
    """paired_stats of both FIXED_METRICS for every oriented method pair,
    overhead level and budget, as CSV rows (FIXED_PAIRED_FIELDS)."""
    rows = []
    for f, (ratio, c0) in enumerate(zip(out["fixed_ratios"],
                                        out["fixed_costs"])):
        for b, frac in enumerate(out["budgets_frac"]):
            for metric in FIXED_METRICS:
                for ma, mb in orient(METHODS):
                    st = paired_stats(out[f"{ma}_{metric}"][:, f, b],
                                      out[f"{mb}_{metric}"][:, f, b],
                                      out["n_train"], out["n_val"],
                                      out["rep"])
                    rows.append({"dataset": ds_name,
                                 "holdout_frac": HOLDOUT_FRAC,
                                 "fixed_ratio": float(ratio),
                                 "fixed_cost": float(c0),
                                 "budget_frac": float(frac),
                                 "budget": float(out["budgets"][f, b]),
                                 "metric": metric, "method_a": ma,
                                 "method_b": mb, **st})
    return rows


def _fixed_lookup(paired):
    return {(r["fixed_ratio"], r["budget_frac"], r["metric"], r["method_a"],
             r["method_b"]): r for r in paired}


def _significant(r):
    return bool(np.isfinite(r["ci_low"])
                and (r["ci_low"] > 0 or r["ci_high"] < 0))


def fixed_summary(out):
    """{ratio: {method: {r2_mean, r2_std, r2_n, r2_suite_mean, ...,
    size_mean, spent_mean, n_empty}}} (lists over budgets; over the
    estimated splits; spent = true cost / budget)."""
    ok = ~out["estimation_failed"]
    tab = {}
    for f, ratio in enumerate(out["fixed_ratios"]):
        level = {}
        for m in METHODS:
            entry = {}
            for metric in FIXED_METRICS:
                arr = out[f"{m}_{metric}"][ok, f]
                mu, sd = _nan_mean_std(arr, axis=0)
                entry[f"{metric}_mean"] = np.asarray(mu).tolist()
                entry[f"{metric}_std"] = np.asarray(sd).tolist()
                entry[f"{metric}_n"] = np.isfinite(arr).sum(axis=0).tolist()
            size = out[f"{m}_size"][ok, f].astype(float)
            spent = out[f"{m}_cost"][ok, f] / out["budgets"][f][None, :]
            entry.update(size_mean=size.mean(axis=0).tolist(),
                         spent_mean=_nan_mean_std(spent, axis=0)[0].tolist(),
                         n_empty=(size == 0).sum(axis=0).astype(int).tolist())
            level[m] = entry
        tab[f"{float(ratio):g}"] = level
    return tab


def fixed_macros(ds_name, out, paired, undefined=None):
    """Flat {MacroName: finite number} of a --fixed-cost run (module
    docstring).  Undefined statistics (NaN: nothing affordable, no finite
    pair) are not emitted; their names go to `undefined` if given."""
    P, D = FIXED_MACRO_PREFIX, camel(ds_name)
    ok = ~out["estimation_failed"]
    F = ["F" + num_words(float(r)) for r in out["fixed_ratios"]]
    W = ["B" + num_words(100.0 * float(f)) for f in out["budgets_frac"]]
    mac = {}

    def put(name, v):
        if not name.isascii() or not name.isalpha():
            raise ValueError(f"macro name {name!r} is not letters only")
        if name in mac:
            raise ValueError(f"duplicate macro {name}")
        v = int(v) if isinstance(v, (bool, np.bool_, np.integer)) else v
        if isinstance(v, int) or (isinstance(v, float) and math.isfinite(v)):
            mac[name] = v
        elif undefined is not None:
            undefined.append(name)

    put(f"{P}MedianCost{D}", float(out["median_cost"]))
    put(f"{P}NBench{D}", int(len(out["costs"])))
    put(f"{P}NSplits{D}", int(ok.sum()))
    put(f"{P}Violations{D}", int(sum(int(out[f"{m}_violations"].sum())
                                     for m in METHODS)))
    lookup = _fixed_lookup(paired)
    for f, Fw in enumerate(F):
        put(f"{P}Overhead{D}{Fw}", float(out["fixed_costs"][f]))
        put(f"{P}Total{D}{Fw}", float(out["total_cost"][f]))
        for b, Bw in enumerate(W):
            put(f"{P}Budget{D}{Fw}{Bw}", float(out["budgets"][f, b]))
            put(f"{P}Empty{D}{Fw}{Bw}",
                int(np.sum(out["cheap_first_size"][ok, f, b] == 0)))
        for m in METHODS:
            M_ = camel(m)
            size = out[f"{m}_size"][ok, f].astype(float).mean(axis=0)
            spent = _nan_mean_std(out[f"{m}_cost"][ok, f]
                                  / out["budgets"][f][None, :], axis=0)[0]
            for metric, T in FIXED_METRICS.items():
                mu, sd = _nan_mean_std(out[f"{m}_{metric}"][ok, f], axis=0)
                for b, Bw in enumerate(W):
                    put(f"{P}{T}Rsq{D}{M_}{Fw}{Bw}", float(mu[b]))
                    put(f"{P}{T}RsqStd{D}{M_}{Fw}{Bw}", float(sd[b]))
            for b, Bw in enumerate(W):
                put(f"{P}Size{D}{M_}{Fw}{Bw}", float(size[b]))
                put(f"{P}Spent{D}{M_}{Fw}{Bw}", float(spent[b]))
        ratio = float(out["fixed_ratios"][f])
        for ma, mb in FIXED_PAIRS:
            pair = f"{camel(ma)}Vs{camel(mb)}"
            for metric, T in FIXED_METRICS.items():
                n_pos = n_neg = 0
                for b, Bw in enumerate(W):
                    r = lookup[(ratio, float(out["budgets_frac"][b]), metric,
                                ma, mb)]
                    for sname, field in FIXED_PAIR_STATS:
                        v = r[field]
                        put(f"{P}{T}Pair{sname}{D}{pair}{Fw}{Bw}",
                            int(v) if field == "n_splits" else float(v))
                    if _significant(r):
                        n_pos += int(r["ci_low"] > 0)
                        n_neg += int(r["ci_high"] < 0)
                put(f"{P}{T}SigPos{D}{pair}{Fw}", n_pos)
                put(f"{P}{T}SigNeg{D}{pair}{Fw}", n_neg)
    return mac


def _json_safe(x):
    """Floats/ints/lists for json.dumps (NaN and inf as null)."""
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


FIXED_NOTES = {
    "cost_model": "true cost of benchmark j = c0 + c_j, c0 = fixed_ratio x "
                  "median_j c_j (whole cost table); c_j = the dataset's "
                  "cost table (examples)",
    "budgets": "fractions of the full suite's true cost sum_j (c0 + c_j), "
               "the same on every split; fixed_ratio 0 = the plain "
               "cost_budget run (same budgets, selections and R²)",
    "methods": "knapsack: gain / (c0 + c_j) with budgeted_select's tie rule "
               "and best-singleton fallback; cheapest first by true cost; "
               "cost-blind orders and random (shared permutation, seed tag "
               "SEED + 7) filled skip-and-continue under the true costs; "
               "every selection's true cost <= budget (violations)",
    "metrics": "r2: pooled over the unselected observed benchmarks (the "
               "paper's R², NaN if nothing is affordable); r2_suite: pooled "
               "over all observed validation cells, a bought cell counting "
               "as known, empty selection = 0",
    "random_streams": "no new stream: random = Split.seed(SEED + 7), as in "
                      "the plain run (the retired SEED + 11 stream stays "
                      "unused, rule D3)",
    "facloc": "knap_facloc: the knapsack rule with facility-location gains "
              "on the clipped correlation matrix (budgeted_select objective "
              "facloc, S as baseline_select.facility_location_select); "
              "card_facloc: facility_location_select's order filled "
              "skip-and-continue; deterministic, no random numbers",
    "human_evaluation": "the burden of human evaluation (annotators, review "
                        "of open-ended outputs) is a further per-benchmark "
                        "cost; no data, not modelled",
}


def fixed_json(ds_name, out, paired, macros, config, undefined=None):
    """The LOGS/cost_fixed_{ds}.json payload."""
    key_rows = [r for r in paired
                if (r["method_a"], r["method_b"]) in FIXED_PAIRS]
    viol = {m: int(out[f"{m}_violations"].sum()) for m in METHODS}
    checks = {"violations": viol, "violations_total": sum(viol.values()),
              "estimation_failed": int(out["estimation_failed"].sum()),
              "n_eligible_min": int(out["n_eligible"].min()),
              "n_eligible_max": int(out["n_eligible"].max())}
    return _json_safe({
        "script": "eval_cost_budget.py --fixed-cost", "dataset": ds_name,
        "config": config, "notes": FIXED_NOTES,
        "fixed_ratios": out["fixed_ratios"], "fixed_costs": out["fixed_costs"],
        "median_cost": float(out["median_cost"]),
        "total_cost": out["total_cost"], "budgets": out["budgets"],
        "budgets_frac": out["budgets_frac"],
        "summary": fixed_summary(out), "paired_key_pairs": key_rows,
        "checks": checks, "runtime_s": config.get("runtime_s"),
        "n_macros": len(macros), "macros": macros,
        "macros_undefined": sorted(undefined or []),
        "macros_undefined_note": "statistics without a finite value "
                                 "(nothing affordable, or no split where "
                                 "both values of a pair are finite); not "
                                 "emitted as macros"})


def print_fixed_summary(ds_name, out, paired):
    ok = ~out["estimation_failed"]
    S, S_ok = len(out["split_id"]), int(ok.sum())
    fracs = np.asarray(out["budgets_frac"], dtype=float)
    lookup = _fixed_lookup(paired)
    for f, ratio in enumerate(out["fixed_ratios"]):
        c0, total = out["fixed_costs"][f], out["total_cost"][f]
        print(f"\n== {display_name(ds_name)}, fixed overhead c0 = "
              f"{ratio:g} × median = {c0:.6g} per benchmark (suite "
              f"{total:.6g}): R² / full-suite R² / #selected vs budget "
              f"(means over {S_ok} of {S} splits) ==")
        print(f"{'method':<30}" + "".join(f"{100 * x:>17g}%" for x in fracs))
        for m in METHODS:
            cells = []
            for b in range(len(fracs)):
                mu = _nan_mean_std(out[f"{m}_r2"][ok, f, b])[0]
                ms = _nan_mean_std(out[f"{m}_r2_suite"][ok, f, b])[0]
                sz = out[f"{m}_size"][ok, f, b].mean()
                cells.append(f"{mu:.3f}/{ms:.3f}/{sz:.1f}")
            print(f"{METHOD_LABEL[m]:<30}" + "".join(f"{c:>18}"
                                                     for c in cells))
        for metric in FIXED_METRICS:
            print(f"paired {metric} differences a − b (mean; * = "
                  f"Nadeau–Bengio 95% CI excludes 0; (n) = paired splits "
                  f"where fewer than {S_ok}):")
            for ma, mb in FIXED_PAIRS:
                cells = []
                for x in fracs:
                    r = lookup[(float(ratio), float(x), metric, ma, mb)]
                    n = int(r["n_splits"])
                    cells.append("n/a" if n == 0 else
                                 f"{r['mean_diff']:+.3f}"
                                 f"{'*' if _significant(r) else ' '}"
                                 + (f"({n})" if n < S_ok else ""))
                print(f"  {ma + ' − ' + mb:<30}"
                      + "".join(f"{c:>12}" for c in cells))
    viol = sum(int(out[f"{m}_violations"].sum()) for m in METHODS)
    n_el = out["n_eligible"]
    print(f"\nchecks: budget violations {viol} (every method, level, budget "
          f"and split); eligible {n_el.min()}–{n_el.max()} of "
          f"{len(out['costs'])}; estimation failed on {S - S_ok} splits")


def plot_fixed(ds_name, out, paired, path=None):
    """One column per overhead level; rows: mean R² vs budget
    (FIXED_PLOT_METHODS), its paired difference vs random
    (FIXED_DIFF_METHODS, 95% Nadeau–Bengio band), then the same two rows
    for the full-suite R²."""
    ratios = np.asarray(out["fixed_ratios"], dtype=float)
    nf = len(ratios)
    x = 100 * np.asarray(out["budgets_frac"], dtype=float)
    ok = ~out["estimation_failed"]
    lookup = _fixed_lookup(paired)
    fig, axes = plt.subplots(4, nf, figsize=(2.6 * nf, 8.4), squeeze=False,
                             sharey="row")
    for f, ratio in enumerate(ratios):
        for r0, (metric, label) in enumerate(
                (("r2", "Imputation $R^2$"),
                 ("r2_suite", "Full-suite $R^2$"))):
            ax = axes[2 * r0, f]
            for m in FIXED_PLOT_METHODS:
                mu = _nan_mean_std(out[f"{m}_{metric}"][ok, f], axis=0)[0]
                ax.plot(x, mu, METHOD_STYLE[m], color=METHOD_COLOR[m],
                        label=METHOD_LABEL[m], markersize=3,
                        linewidth=1.5 if m.startswith("knap") else 1.0)
            if f == 0:
                ax.set_ylabel(label, fontsize=8)
            if r0 == 0:
                ax.set_title(f"$c_0$ = {ratio:g} × median", fontsize=8)
            ax = axes[2 * r0 + 1, f]
            ax.axhline(0.0, color="#888888", linewidth=0.8)
            for m in FIXED_DIFF_METHODS:
                rows = [lookup[(float(ratio), float(fr), metric, m,
                                "random")] for fr in out["budgets_frac"]]
                mu = np.array([p["mean_diff"] for p in rows], dtype=float)
                lo = np.array([p["ci_low"] for p in rows], dtype=float)
                hi = np.array([p["ci_high"] for p in rows], dtype=float)
                ax.plot(x, mu, METHOD_STYLE[m], color=METHOD_COLOR[m],
                        markersize=3, linewidth=1.2, label=METHOD_LABEL[m])
                ax.fill_between(x, lo, hi, color=METHOD_COLOR[m], alpha=0.12)
            if f == 0:
                ax.set_ylabel(f"$\\Delta$ vs random", fontsize=8)
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Budget (% of full suite, true cost)", fontsize=7)
    axes[0, 0].legend(fontsize=5, loc="lower right")
    axes[1, 0].legend(fontsize=5, loc="best")
    fig.suptitle(f"{display_name(ds_name)}: fixed per-benchmark overhead "
                 f"$c_0$ (cost $c_0 + c_j$)", fontsize=9)
    plt.tight_layout()
    path = OUT / f"{FIXED_STEM}_{ds_name}.pdf" if path is None else path
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)
    return path


def main_fixed(datasets, counts, args, ratios):
    """--fixed-cost: run_fixed per dataset, then its CSV, JSON and figure."""
    for ds_name, counts_path in zip(datasets, counts):
        t0 = time.time()
        out = run_fixed(ds_name, counts_path, args.budgets, ratios,
                        args.repeats, args.folds)
        paired = fixed_paired_rows(ds_name, out)
        print_fixed_summary(ds_name, out, paired)
        undefined = []
        macros = fixed_macros(ds_name, out, paired, undefined)
        stem = f"{FIXED_STEM}_{ds_name}"
        np.savez_compressed(LOGS / f"{stem}.npz", **out)
        print(f"Results saved to {LOGS / f'{stem}.npz'}", flush=True)
        write_csv(paired, LOGS / f"{stem}_paired.csv", FIXED_PAIRED_FIELDS)
        print(f"Paired statistics saved to {LOGS / f'{stem}_paired.csv'}",
              flush=True)
        plot_fixed(ds_name, out, paired)
        config = {"dataset": ds_name, "repeats": args.repeats,
                  "folds": args.folds, "holdout_frac": HOLDOUT_FRAC,
                  "seed": SEED, **protocol_config(),
                  "bounds": "score_bounds.dataset_bounds "
                            "(data/score_bounds.json)",
                  "random_seed_tag": SEED + RANDOM_SEED_OFFSET,
                  "budgets_frac": list(args.budgets),
                  "fixed_ratios": list(ratios), "methods": METHODS,
                  "metrics": list(FIXED_METRICS),
                  "fixed_pairs": [list(p) for p in FIXED_PAIRS],
                  "counts_file": str(out["counts_file"]),
                  "counts_sha256": str(out["counts_sha256"]),
                  "matrix": matrix_path(ds_name).name,
                  "matrix_sha256": str(out["matrix_sha256"]),
                  "runtime_s": round(time.time() - t0, 1)}
        jpath = LOGS / f"{stem}.json"
        report = fixed_json(ds_name, out, paired, macros, config, undefined)
        jpath.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
        print(f"Summary saved to {jpath} ({len(macros)} macros, "
              f"{len(undefined)} undefined statistics left out)", flush=True)
        print(f"{display_name(ds_name)} completed in {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cost-budgeted benchmark selection (knapsack greedy vs "
                    "cost-blind orders, cheapest first and random) on the "
                    "matched repeated-CV splits.")
    parser.add_argument("--dataset", default=None,
                        help="one dataset name (data/<name>.matrix.npz); "
                             "same as --datasets NAME (default: mmlu)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="dataset names, each read from "
                             "data/<name>.matrix.npz")
    parser.add_argument(
        "--counts", nargs="+", default=None,
        help="JSON file(s) {benchmark_name: cost} with a positive cost for "
             "every column of the dataset: one file per dataset (in "
             "--datasets order) or one file for all.  Without --counts: "
             + ", ".join(f"{ds} -> data/{f}"
                         for ds, f in DEFAULT_COUNTS.items())
             + " (data/build_cost_tables.py).  Cost scope: mmlu = test "
             "questions per subject; mteb = mteb num_samples per task, "
             "whose unit depends on the task type (retrieval: corpus "
             "documents + queries).  The checked-in mteb table counts only "
             "the eval splits and language subsets scored by "
             "MTEB(eng, classic) (build_cost_tables.py --mteb-scope "
             "benchmark, total 40,232,830); the July 2026 rebuttal logs "
             "used --mteb-scope task (all language subsets, total "
             "40,858,920), which changes eight multilingual tasks and "
             "hence the MTEB results")
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS,
        help="budgets as fractions of the dataset's total cost (sum of the "
             "costs of all its benchmarks, eligible or not), e.g. 0.05 = "
             "5%%; each gives one absolute budget C = frac · total, the same "
             "on every split.  Default: %(default)s (the MMLU grid; "
             "run_all.sh uses 0.002 0.005 0.01 0.02 0.05 0.10 0.20 for "
             "mteb)")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions; splits = repeats × folds, "
                             "rep 0 = the original seed-42 folds "
                             "(default: %(default)s)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--plot-entropy-rf", action="store_true",
                        help="ignored: the entropy_rf pair was dropped "
                             "(protocol rule D3); kept so old command lines "
                             "run")
    parser.add_argument("--plot-trace", action="store_true",
                        help="also plot knap_trace and card_trace (always "
                             "logged)")
    parser.add_argument("--legacy-random", action="store_true",
                        help="draw random as before the matched design (a "
                             "fresh permutation per budget from "
                             "default_rng((SEED + 7, fold)); rep 0 "
                             "reproduces the historical cost_budget logs' "
                             "random_r2), instead of the random baseline "
                             "shared with the other scripts; for comparisons "
                             "with old logs only")
    parser.add_argument(
        "--fixed-cost", nargs="*", type=float, default=None,
        metavar="RATIO",
        help="opt-in fixed per-benchmark overhead: true cost c0 + c_j with "
             "c0 = RATIO x the median cost of the dataset's cost table, one "
             "level per RATIO (flag without values: "
             + " ".join(f"{r:g}" for r in DEFAULT_FIXED_RATIOS)
             + "); --budgets are then fractions of the suite's true cost "
             "sum_j (c0 + c_j), every method is held to it, and the run "
             "writes LOGS/cost_fixed_<ds>.{npz,json}, "
             "LOGS/cost_fixed_<ds>_paired.csv and OUT/cost_fixed_<ds>.pdf "
             "instead of the cost_budget_* outputs (default: off)")
    args = parser.parse_args()
    if args.repeats < 1 or args.folds < 2:
        parser.error("need --repeats >= 1 and --folds >= 2")
    if args.fixed_cost is not None:
        if args.legacy_random:
            parser.error("--fixed-cost uses the shared random baseline; "
                         "drop --legacy-random")
        if any(not (np.isfinite(r) and r >= 0) for r in args.fixed_cost):
            parser.error("--fixed-cost ratios must be finite and >= 0")
        if len(set(args.fixed_cost)) != len(args.fixed_cost):
            parser.error("--fixed-cost ratios must be distinct")
    if any(not (np.isfinite(f) and f > 0) for f in args.budgets):
        parser.error("--budgets must be positive fractions")
    if args.dataset is not None and args.datasets is not None:
        parser.error("give --dataset or --datasets, not both")
    datasets = args.datasets or [args.dataset or "mmlu"]
    missing = [ds for ds in datasets if not matrix_path(ds).is_file()]
    if missing:
        avail = sorted(p.name[:-len(".matrix.npz")]
                       for p in DATA.glob("*.matrix.npz"))
        parser.error(f"no {DATA}/<name>.matrix.npz for {missing}; "
                     f"available: {' '.join(avail)}")
    if args.counts is None:
        no_default = [ds for ds in datasets if ds not in DEFAULT_COUNTS]
        if no_default:
            parser.error(f"no default cost table for {no_default}; "
                         "pass --counts")
        counts = [DATA / DEFAULT_COUNTS[ds] for ds in datasets]
    elif len(args.counts) == 1:
        counts = args.counts * len(datasets)
    elif len(args.counts) == len(datasets):
        counts = args.counts
    else:
        parser.error(f"--counts: give 1 or {len(datasets)} files "
                     f"(one per dataset), not {len(args.counts)}")
    if args.plot_entropy_rf:
        print("note: --plot-entropy-rf is ignored (the entropy_rf pair was "
              "dropped, protocol rule D3)", flush=True)
    plot_methods = (PLOT_METHODS
                    + (["knap_trace", "card_trace"] if args.plot_trace
                       else []))
    if args.fixed_cost is not None:
        main_fixed(datasets, counts, args,
                   sorted(args.fixed_cost or DEFAULT_FIXED_RATIOS))
        sys.exit(0)

    for ds_name, counts_path in zip(datasets, counts):
        t0 = time.time()
        out = run(ds_name, counts_path, args.budgets, args.repeats,
                  args.folds, args.legacy_random)
        paired = paired_rows(ds_name, out)
        print_summary(ds_name, out, paired)
        plot_budget(ds_name, out, plot_methods)
        np.savez(LOGS / f"cost_budget_{ds_name}.npz", **out)
        write_paired_csv(LOGS / f"cost_budget_{ds_name}_paired.csv", paired)
        print(f"{display_name(ds_name)} completed in {time.time() - t0:.0f}s",
              flush=True)
