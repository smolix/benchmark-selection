r"""Selection stability: is low overlap between selections harmless?

Reviewer QSWq (follow-up, request B4) noted that the selections of
different splits overlap little and that the paper does not show directly
that this is harmless.  This script tests it.  On every matched split
(cv_split.cv_splits, 10% holdout, --repeats x --folds splits) and for each
method of METHODS (mi, trace, entropy, hub, pca, facloc) it compares the
split's selection with alternative selections that the same method makes
from slightly different training data, all scored on the same validation
fold with the same imputation model.  facloc is greedy facility location
on the clipped correlation matrix (baseline_select.facility_location_select
via eval_baselines.run_selection), the paper's headline method since
2026-10-19; like the other methods it is deterministic, every random draw
below is shared by all methods and drawn before any selection, and every
summary is per method, so adding it changed no other method's values.

  full selection   A_k: the method run on the split's training estimate
                   R = estimate_corr(B_train, O_train) (the paper's
                   estimator) with the split's eligibility mask
                   eligible_mask(O_train), for k in KS = (3, 5, 10).  These
                   are the selections of eval_baselines.py on that split
                   (--check-logs compares them).
  alternatives     A_k^b, b = 1..N_SUB (10): the same method on
                   R_b = estimate_corr(B_sub, O_sub), the same estimator
                   rule applied to a subsample of SUB_FRAC (80%) of the
                   split's TRAINING models (subsample_positions: positions
                   into train_idx, so no validation model can enter), with
                   eligibility eligible_mask(O_sub) (a subset of the
                   split's mask; --sub-eligible split keeps the split's
                   mask instead).  Common random numbers
                   Split.seed(SEED + SUB_SEED_OFFSET), offset 23: draw b is
                   the b-th rng.permutation(n_train)[:n_keep] of one
                   generator, n_keep = round(SUB_FRAC * n_train), rows
                   sorted, so the subsamples are shared by all methods.
  random subsets   N_RANDOM (10) uniformly random k-subsets of the split's
                   eligible benchmarks (random_subsets): draw r is a
                   permutation of all N benchmarks restricted to the
                   eligible ones (as greedy_select.random_select), its
                   first k entries for every k (nested across k).  Common
                   random numbers Split.seed(SEED +
                   RANDOM_SUBSETS_SEED_OFFSET), offset 24.
  random s-swaps   the matched null of the direct test (random_swap): for
                   s = 1..k and draw d = 1..N_SWAP (10), swap_d(A_k, s) is
                   A_k without s of its members plus s eligible benchmarks
                   outside A_k (the split's mask), both chosen by the d-th
                   permutation rng.permutation(N) of rng =
                   default_rng(Split.seed(SEED + SWAP_SEED_OFFSET)), offset
                   22, shared by every method and k and nested in s.  Over
                   the permutation it is a uniformly random eligible
                   k-subset with exactly k - s members of A_k, i.e. a random
                   subset conditioned on its overlap with the full
                   selection.
  random baseline  the paper's random baseline on the split (random_select,
                   Split.seed(SEED + RANDOM_SEED_OFFSET)), for reference.

Seed tags.  The three streams of this script, offsets 22 (s-swaps), 23
(subsamples) and 24 (random subsets), are distinct from every other seed
tag of code/: RANDOM_SEED_OFFSET 7, ENTROPY_RF_SEED_OFFSET 11,
eval_prospective's 13-16, eval_item_budget's 18, analyze_objectives' 19,
the TabImpute context's reserved 20, eval_imputers' 21 and
eval_reviewer_checks' CONTAM_DRAW_SEED_OFFSET 25
(test_seed_tags_unique_in_code scans code/ for *_SEED_OFFSET assignments
and for SEED-plus-number seeds).  The contamination draws used 23 for a
while; they moved to 25 (their seed tuples had another length, so the two
streams never produced the same numbers), and the subsamples kept 23, so
the alternatives are those of this script's first version.

Every set is evaluated on the split's validation fold with the FULL
training estimate (R, col_mean, col_std) for imputation, so only the
selection varies:

  r2    realized pooled R^2 (eval_entropy_vs_mi.impute_and_evaluate,
        imported, not copied: test models x unselected observed target
        benchmarks (rule D1), train-standardized units, ridge 0.01,
        predictions clipped to the valid range (rule D2)); NaN where no
        validation model observes a selected benchmark (its skip rule;
        rare, e.g. 4 of 7500 hub alternatives on Merged), left out of
        every mean
  pop   population R^2 of the set under the full training R:
        pop(A) = tr(R_CA R_AA^{-1} R_AC) / tr(R_CC), C = V \ A, i.e. the
        trace objective's explained variance of the complement,
        (F_tr(A) - k) / (N - k) on a correlation matrix (analyze_objectives
        R2_pop).  At fixed k, pop(A_k^b) / pop(A_k) is the ratio of the
        explained variances, the "population explained variance relative to
        the full selection" (pop_rel).
  jacc  Jaccard |A n A'| / |A u A'| of an alternative with the full
        selection and of every pair of alternatives (jacc_pair: mean over
        the N_SUB (N_SUB - 1) / 2 pairs); n_common = |A n A_k|, the number
        of the full selection's benchmarks an alternative keeps, and
        s = k - n_common the number it replaces.  The random subsets give
        the chance level of both.
  cells share of the validation fold's observed cells that the set is
        scored on (eval_baselines.evaluated_share): the observed cells of
        the unselected benchmarks of the models that observe at least one
        selected benchmark, over those of all validation models.

The direct test.  Per method and k, over the splits:

  cost          R^2(A_k) - mean_b R^2(A_k^b) (split means; what selecting
                from 80% of the training models costs), with its paired
                interval alt_vs_full below
  gap           R^2(A_k) - mean_r R^2(random subset r); gap_significant:
                the full_vs_random interval lies above 0
  sd_over_gap, range_over_gap, cost_over_gap   within-split SD (ddof 1)
                and range of the alternatives' R^2 and the cost, over the
                gap (ratios of the split means)
  worst_cost_over_gap   -ci_low(alt_vs_full) / gap: the largest share of
                the gap to random that the 95% interval allows the
                alternatives to lose (negative: they gain even then)
  swap_loss     sum_b cost_b / sum_b s_b over the swapped alternatives
                (s_b >= 1): R^2 lost per replaced benchmark
  null_swap_loss  the same for the matched null of the same alternatives:
                null_b = R^2(A_k) - mean_d R^2(swap_d(A_k, s_b)), the cost of
                replacing s_b benchmarks of the same split's A_k at random
  harm          sum_b cost_b / sum_b null_b = swap_loss / null_swap_loss:
                0 if the alternatives' replacements cost nothing, 1 if they
                cost as much as replacing the same number of benchmarks of
                the same full selections at random
  harm_pop      the same ratio of the population R^2 losses
                (pop(A_k) - pop(A_k^b) over pop(A_k) - mean_d
                pop(swap_d(A_k, s_b))), free of validation noise
  harm_subsets  the first version's harm, swap_loss / rand_swap_loss, with
                rand_swap_loss the random k-subsets' cost per replaced
                benchmark against A_k.  The subsets replace nearly all k
                benchmarks, the alternatives one or two, and random
                replacement cost is concave in s, so this ratio is biased
                upward (MTEB trace at k = 3: 0.39 against the matched
                0.22); kept in the tables for comparison only (no macro),
                NaN where harm is
  by_swap       per s: the number of alternatives, their mean cost and the
                mean null cost of the same alternatives
  low_share, low_cost   share of the alternatives with Jaccard below
                LOW_JACCARD (0.5) with A_k and their mean cost
  cross_jacc    mean Jaccard of the full selections A_k of two different
                splits (all pairs): the overlap the reviewer observed.
                Alternatives with jacc_pair at or below it perturb the
                selection at least as much as the change of CV split does.
  cells_full, cells_alt   mean cell shares of A_k and of the alternatives;
                cells_shift the share of alternatives whose cell share
                differs from A_k's by more than COV_SHIFT (0.1),
                cost_cov_matched the mean cost of the others and
                cov_cost_corr the correlation of the change in cell share
                with the cost over the swapped alternatives

Rules.  The ratios to the gap and harm are NaN unless gap_significant: a
gap near 0 makes them meaningless (MMLU MI at k = 10 gave -8).  swap_loss,
null_swap_loss, harm and harm_pop rest on n_swapped alternatives from
n_swap_splits splits, low_cost on n_low alternatives from n_low_splits
splits; each is NaN below MIN_N_ALT (30) alternatives or MIN_N_SPLITS (10)
splits (MMLU trace at k = 10 has a single swapped alternative of 500).
harm and harm_pop are also NaN where their interval is unbounded (the
matched null cost is not significantly different from 0, e.g. MMLU PCA at
k = 5), and harm_pop unless its denominator is positive.
Their 95% intervals (*_lo, *_hi) come from ratio_ci: the Fieller interval
of a ratio of per-split sums, sum_j num_j / sum_j den_j, under the
Nadeau-Bengio corrected variance of paired_stats (the values H that a
corrected t test of mean_j(num_j - H den_j) = 0 does not reject; with
den_j = 1 it is paired_stats' interval of the mean), over the splits that
contribute; NaN where the interval is unbounded.  A plain split bootstrap
would ignore the overlap of the training sets of different splits, which
the correction accounts for.  status is "uninformative" without a
significant gap (there is nothing to lose: no statement about harm is
possible), "stable" with too few swapped alternatives (the selection
hardly changes, so low overlap does not arise), else "tested".

Reading the result.  It is a statement per dataset, method and k, made by
the interval bounds rather than the means: alt_vs_full (is there a cost at
all), worst_cost_over_gap (how much of the method's advantage over random
the interval allows to lose), alt_vs_random (do the alternatives still
beat random) and harm with its interval (do the alternatives' replacements
cost less than random replacements of as many benchmarks).  "stable"
cells answer the reviewer by cross_jacc and jacc_pair (no low overlap to
test), "uninformative" ones not at all.

Coverage.  Every set is scored with the same R, col_mean and col_std, but
on its own targets: the observed validation cells of the benchmarks it
does not select, and on sparse data impute_and_evaluate skips validation
models that observe none of its benchmarks.  Both differ between sets by
construction, exactly as between the methods of eval_baselines (the
paper's metric), so on sparse data (Merged, BenchPress) part of a cost can
be a change of the evaluated population; cells_* and cost_cov_matched
measure it.  MTEB and MMLU are scored on (nearly) every cell.

Paired statistics (paired_stats.paired_stats over the matched splits,
Nadeau-Bengio corrected 95% interval, win rate), per method and k:

  alt_vs_full       {m}_alt - {m}: mean_b R^2(A_k^b) - R^2(A_k)
  full_vs_random    {m} - random_sets: R^2(A_k) - mean_r R^2(random r)
  alt_vs_random     {m}_alt - random_sets
  altmin_vs_random  {m}_altmin - random_sets: the worst alternative of the
                    split against the random-subset mean

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir()):

  LOGS/selection_stability_{ds}.npz      S splits, K = len(ks)
    rep, fold, split_id, n_train, n_val, n_eligible   (S,) int
    eligible (S, N) bool; estimation_failed (S,) bool (rows NaN / -1)
    n_sub_models (S,) int, training models per subsample
    sub_failed (S, n_sub) bool (subsample estimate failed: NaN / -1)
    sub_n_eligible (S, n_sub) int
    est_regime (S,), sub_est_regime (S, n_sub) int: estimator branch of
        estimate_corr (estimator_regime: 0 pairwise, 1 EM, 2 regularized
        EM, 3 regularized EM with auto shrinkage), diagnostic
    {m}_selected        (S, K, k_max) int, A_k in selection order, -1 pad
    {m}_alt_selected    (S, K, n_sub, k_max) int, A_k^b
    {m}_r2_full, {m}_pop_full, {m}_cells_full     (S, K)
    {m}_r2_alt, {m}_pop_alt, {m}_cells_alt        (S, K, n_sub)
    {m}_jacc_full, {m}_n_common        (S, K, n_sub) (n_common -1: none)
    {m}_jacc_pair                      (S, K)
    {m}_rand_jacc_full, {m}_rand_n_common   (S, K, n_random): random
        subsets against A_k
    {m}_swap_r2, {m}_swap_pop          (S, K, n_swap, k_max): random
        s-swap of A_k of draw d in column s - 1 (NaN for s > k)
    swap_order          (S, n_swap, N) int, draw d's permutation
    rand_order          (S, n_random, k_max) int, draw r's eligible
                        permutation prefix (subset of size k = [:k])
    rand_r2, rand_pop, rand_cells   (S, K, n_random); rand_jacc_pair (S, K)
    random_baseline_r2  (S, K); random_baseline_selected (S, k_max)
    time_s              (S,) seconds per split (one thread)
    ks, methods, benchmark_names, n_sub, sub_frac, n_random, n_swap,
    sub_eligible, holdout_frac, n_folds, n_repeats, seed, min_train_obs,
    min_target_obs, pred_clip, k_max, sub_seed_offset,
    random_subsets_seed_offset, swap_seed_offset, low_jaccard
  LOGS/selection_stability_{ds}.json     config (runtime_s wall clock,
    split_time_s single-threaded sum, the thresholds), summary tables,
    check_logs, and the flat "macros" dict
  LOGS/selection_stability_{ds}_paired.csv   columns dataset,
    holdout_frac, k, comparison, method_a, method_b + paired_stats.STAT_FIELDS
  OUT/selection_stability_{ds}.pdf       four panels: Jaccard (pairwise
    among the alternatives, cross_jacc dotted), R^2 of full selection /
    alternatives / random, the direct test (cost by number of replaced
    benchmarks at k = FIG_K, each method against its own matched random
    s-swaps; methods without a significant gap to random faded) and
    pop_rel
  OUT/selection_stability_all.pdf        the direct-test panel of every
    dataset: written by runs with more than one dataset and by
    --all-figure, which reads the logs of --datasets (the aggregate step
    after one job per dataset)

Macros ("macros" in the JSON; letters only, make_numbers.camel names, {Ds}
dataset, {M} method, K{k} spelled k): Stab{Field}{Ds}{M}K{k} for Field in
JaccPair, JaccFull, CrossJacc, Ident (share of alternatives equal to A_k),
Distinct (distinct alternative sets per split), RsqFull, RsqAlt, RsqAltSd,
RsqAltRange, RsqAltMin, Cost, PopFull, PopRel, PopRelMin, Gap, SdOverGap,
RangeOverGap, CostOverGap, WorstCostOverGap, NSwapped, NSwapSplits,
SwapLoss, SwapLossLo, SwapLossHi, NullSwapLoss, RandSwapLoss, Harm,
HarmLo, HarmHi, HarmPop, NLow, LowShare, LowCost, LowCostLo, LowCostHi,
CellsFull, CellsAlt, CellsShift, CostCovMatched; Stab{Field}{Ds}RandomK{k}
for JaccPair, Rsq, RsqSd, RsqRange, Pop, and StabRsq{Ds}RandomBaselineK{k};
StabPair{S}{Ds}{M}{Cmp}K{k} for S in Diff, Lo, Hi, Win and Cmp in
AltVsFull, FullVsRandom, AltVsRandom, AltMinVsRandom; StabNSplits{Ds},
StabNReps{Ds}, StabNSub{Ds}, StabNRandom{Ds}, StabNSwap{Ds},
StabSubFrac{Ds}, StabSubModels{Ds}, StabSubFailed{Ds},
StabRegimeChanged{Ds}, StabSubEligDrop{Ds}.  NaN values are null.

Usage:
    python eval_selection_stability.py [--datasets mmlu mteb merged benchpress]
        [--repeats 5] [--folds 10] [--ks 3 5 10] [--n-sub 10]
        [--sub-frac 0.8] [--n-random 10] [--n-swap 10]
        [--sub-eligible subsample|split] [--workers 1] [--check-logs DIR]
        [--from-logs | --all-figure]

--workers W evaluates the splits in W processes (spawn); every split is
independent and seeded by its own common random numbers, so the logs do
not depend on W (keep OMP / OpenBLAS / MKL at one thread per process).
run_all integration: one job per dataset (--datasets <ds>), then one
aggregate job --all-figure --datasets <all of them> for
selection_stability_all.pdf.

Runtime (single-threaded, --repeats 5, defaults, 4090-0 shared with other
jobs; config.split_time_s in the JSON is the sum of the per-split times):
MMLU 47 s, MTEB 27 min (33 s per split), Merged 1.8 h (130 s per split),
BenchPress 2.6 h (190 s per split), dominated by the eleven EM estimates
per split; the random s-swaps add 900 imputations per split (MMLU 18 s ->
47 s, MTEB 24 -> 33 s per split, within the host-load noise on Merged and
BenchPress).  --workers 10 brings MTEB, Merged and BenchPress to 4, 12
and 17 min of wall time.
"""

import argparse
import json
import math
import multiprocessing
import pathlib
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import selection_at_k
import cv_split
import score_bounds
from cv_split import (ENTROPY_RF_SEED_OFFSET, MIN_TRAIN_OBS, N_REPEATS,
                      RANDOM_SEED_OFFSET, cv_splits, eligible_mask,
                      target_mask)
from eval_entropy_vs_mi import estimate_corr, impute_and_evaluate
from score_bounds import dataset_bounds
from eval_baselines import (METHOD_COLOR, METHOD_LABEL, METHOD_STYLE,
                            display_name, evaluated_share, run_selection)
from paired_stats import LEVEL, STAT_FIELDS, paired_stats, write_csv
from make_numbers import camel, int_words

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42                       # eval_baselines.SEED (selections must match)
KS = [3, 5, 10]
# facloc (facility location, the paper's headline method since 2026-10-19)
# appended: the other methods' arrays, rows and macros are unchanged
METHODS = ["mi", "trace", "entropy", "hub", "pca", "facloc"]
DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
N_SUB = 10                      # training subsamples per split
SUB_FRAC = 0.8                  # share of the training models per subsample
N_RANDOM = 10                   # random eligible k-subsets per split
N_SWAP = 10                     # random s-swap draws per split
# Common random numbers Split.seed(SEED + offset): the random s-swaps 22,
# the training subsamples 23 and the random k-subsets 24.  Distinct from
# RANDOM_SEED_OFFSET = 7, ENTROPY_RF_SEED_OFFSET = 11, eval_prospective's
# 13-16, eval_item_budget's 18, analyze_objectives' 19, the reserved 20
# (TabImpute context), eval_imputers' 21 and eval_reviewer_checks'
# CONTAM_DRAW_SEED_OFFSET = 25 (it used 23 for a while); grep code/ for
# *_SEED_OFFSET and SEED-plus-number seeds before adding a tag.
SWAP_SEED_OFFSET = 22
SUB_SEED_OFFSET = 23
RANDOM_SUBSETS_SEED_OFFSET = 24
assert len({SUB_SEED_OFFSET, RANDOM_SUBSETS_SEED_OFFSET, SWAP_SEED_OFFSET,
            RANDOM_SEED_OFFSET, ENTROPY_RF_SEED_OFFSET}) == 5
POP_RIDGE = 1e-8                # baseline_select.RIDGE (trace objective)
LOW_JACCARD = 0.5               # "low overlap" threshold of the summary
MIN_N_ALT = 30                  # fewest alternatives behind a loss / harm
MIN_N_SPLITS = 10               # fewest splits contributing to them
COV_SHIFT = 0.1                 # |change of cell share| of a shifted set
FIG_K = 5                       # k of the direct-test panel
FIG_MIN_BIN = 3                 # fewest sets behind a plotted bin mean
MACRO_PREFIX = "Stab"
COMPARISONS = ["alt_vs_full", "full_vs_random", "alt_vs_random",
               "altmin_vs_random"]
PAIR_STATS = {"Diff": "mean_diff", "Lo": "ci_low", "Hi": "ci_high",
              "Win": "win_rate"}
PAIRED_FIELDS = (["dataset", "holdout_frac", "k", "comparison", "method_a",
                  "method_b"] + STAT_FIELDS)
RANDOM_COLOR = "#888888"


def matrix_path(ds_name):
    return DATA / f"{ds_name}.matrix.npz"


# ── Sets and set measures ────────────────────────────────────────

def subsample_positions(split, n_train, n_sub=N_SUB, sub_frac=SUB_FRAC):
    """(n_sub, n_keep) int: row positions into split.train_idx of the
    split's training subsamples, n_keep = round(sub_frac * n_train)
    (between 1 and n_train).  Row b is the b-th draw
    np.sort(rng.permutation(n_train)[:n_keep]) of one generator rng =
    default_rng(split.seed(SEED + SUB_SEED_OFFSET)), so the draws are
    shared by every method and the first b draws do not depend on n_sub.
    Positions index the training rows only: no validation model can enter
    a subsample."""
    n_keep = min(max(int(round(sub_frac * n_train)), 1), n_train)
    rng = np.random.default_rng(split.seed(SEED + SUB_SEED_OFFSET))
    return np.array([np.sort(rng.permutation(n_train)[:n_keep])
                     for _ in range(n_sub)], dtype=np.int64).reshape(
                         n_sub, n_keep)


def random_subsets(split, eligible, k_max, n_random=N_RANDOM):
    """(n_random, k_max) int: row r is the first k_max entries of the r-th
    permutation rng.permutation(N) restricted to the eligible benchmarks
    (greedy_select.random_select's rule), rng = default_rng(split.seed(SEED
    + RANDOM_SUBSETS_SEED_OFFSET)); the random k-subset of draw r is
    row[:k] (nested across k).  -1 pads rows with fewer eligible
    benchmarks than k_max."""
    eligible = np.asarray(eligible, dtype=bool)
    rng = np.random.default_rng(split.seed(SEED + RANDOM_SUBSETS_SEED_OFFSET))
    out = np.full((n_random, k_max), -1, dtype=np.int64)
    for r in range(n_random):
        perm = rng.permutation(len(eligible))
        perm = perm[eligible[perm]][:k_max]
        out[r, :len(perm)] = perm
    return out


def swap_orders(split, N, n_swap=N_SWAP):
    """(n_swap, N) int: row d is the d-th rng.permutation(N) of rng =
    default_rng(split.seed(SEED + SWAP_SEED_OFFSET)), the ranking of the
    benchmarks behind draw d of every random_swap of the split (shared by
    all methods and k; the first d rows do not depend on n_swap)."""
    rng = np.random.default_rng(split.seed(SEED + SWAP_SEED_OFFSET))
    return np.array([rng.permutation(N) for _ in range(n_swap)],
                    dtype=np.int64).reshape(n_swap, N)


def random_swap(selected, eligible, order, s):
    """Random s-swap of `selected` under the benchmark permutation `order`
    (a row of swap_orders): `selected` without the s members that `order`
    ranks first, followed by the s eligible non-members it ranks first.
    None if `selected` is None, s is not in 0..k or fewer than s eligible
    non-members exist.  Nested in s.  Over a uniformly random `order` the
    dropped members and the added benchmarks are independent uniform draws,
    so the result is a uniformly random eligible k-subset with exactly
    k - s members of `selected` (whose members are eligible)."""
    if selected is None:
        return None
    A = [int(j) for j in selected]
    k = len(A)
    if not 0 <= s <= k:
        return None
    order = np.asarray(order, dtype=np.int64)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    outside = np.asarray(eligible, dtype=bool).copy()
    outside[A] = False
    cand = np.flatnonzero(outside)
    if len(cand) < s:
        return None
    drop = set(sorted(A, key=lambda j: rank[j])[:s])
    add = cand[np.argsort(rank[cand], kind="stable")][:s]
    return [j for j in A if j not in drop] + [int(j) for j in add]


def jaccard(a, b):
    """|A n B| / |A u B| of two index collections (NaN if either is None)."""
    if a is None or b is None:
        return np.nan
    a, b = set(int(j) for j in a), set(int(j) for j in b)
    union = len(a | b)
    return len(a & b) / union if union else np.nan


def n_common(a, b):
    """|A n B| (-1 if either is None)."""
    if a is None or b is None:
        return -1
    return len(set(int(j) for j in a) & set(int(j) for j in b))


def mean_pairwise_jaccard(sets):
    """Mean Jaccard over all pairs of the sets that are not None (NaN with
    fewer than two)."""
    valid = [s for s in sets if s is not None]
    vals = [jaccard(a, b) for a, b in combinations(valid, 2)]
    return float(np.mean(vals)) if vals else np.nan


def pop_r2(R, selected, ridge=POP_RIDGE):
    """Population R^2 of the selection under R: explained over total
    variance of the unselected benchmarks C, tr(R_CA R_AA^{-1} R_AC) /
    tr(R_CC) (ridge on R_AA as in baseline_select), i.e. (F_tr(A) - k) /
    (N - k) on a correlation matrix.  NaN for None or an empty
    complement."""
    if selected is None:
        return np.nan
    A = [int(j) for j in selected]
    C = np.setdiff1d(np.arange(R.shape[0]), A)
    if len(C) == 0 or len(A) == 0:
        return np.nan
    R_AA = R[np.ix_(A, A)] + ridge * np.eye(len(A))
    R_AC = R[np.ix_(A, C)]
    explained = float(np.sum(R_AC * np.linalg.solve(R_AA, R_AC)))
    return explained / float(np.trace(R[np.ix_(C, C)]))


def estimator_regime(O_train):
    """Branch estimate_corr takes on these training rows: 0 pairwise, 1 EM,
    2 regularized EM (fewer models than benchmarks or < 50% observed),
    3 regularized EM with auto shrinkage (fewer models than benchmarks).
    Mirrors eval_entropy_vs_mi.estimate_corr / estimate_corr_em; a
    diagnostic only (does the subsample switch branch?)."""
    M_tr, N_tr = O_train.shape
    obs_frac = float(np.sum(O_train)) / max(M_tr * N_tr, 1)
    if not (obs_frac < 0.9 or M_tr < N_tr):
        return 0
    if M_tr < N_tr:
        return 3
    return 2 if obs_frac < 0.5 else 1


def _padded(indices, length):
    """Integer array of `indices` (None: none), padded with -1."""
    out = np.full(length, -1, dtype=np.int64)
    if indices is not None:
        out[:len(indices)] = indices
    return out


# ── Per-split evaluation ─────────────────────────────────────────

def method_sets(method, R, eligible, ks, benchmark_names, split):
    """[A_k for k in ks] of `method` on R with the given eligibility mask
    (eval_baselines.run_selection with k_max = max(ks); nested selectors
    give prefixes, PCA its per-k sets); None where fewer than k
    benchmarks are eligible."""
    result = run_selection(method, R, max(ks), benchmark_names, split,
                           eligible)
    return [selection_at_k(result, k) for k in ks]


def evaluate_split(split, B, O, benchmark_names, ks, n_sub=N_SUB,
                   sub_frac=SUB_FRAC, n_random=N_RANDOM,
                   sub_eligible="subsample", n_swap=N_SWAP, bounds=None):
    """Everything of one matched split (module docstring): a dict of
    arrays without the split dimension (keys as in the npz, minus the
    split metadata).  The realized R^2 score the split's targets
    (cv_split.target_mask, rule D1) with predictions clipped to `bounds`
    (score_bounds.dataset_bounds, rule D2; None: no clip)."""
    t0 = time.time()
    N = B.shape[1]
    K, k_max = len(ks), max(ks)
    B_train, O_train = B[split.train_idx], O[split.train_idx]
    B_test, O_test = B[split.val_idx], O[split.val_idx]
    n_train = len(split.train_idx)
    eligible = eligible_mask(O_train)
    target = target_mask(O_train)            # D1, shared by every method
    pos = subsample_positions(split, n_train, n_sub, sub_frac)
    rand = random_subsets(split, eligible, k_max, n_random)
    orders = swap_orders(split, N, n_swap)

    out = {
        "n_eligible": int(eligible.sum()), "eligible": eligible,
        "n_sub_models": int(pos.shape[1]),
        "estimation_failed": False,
        "sub_failed": np.zeros(n_sub, dtype=bool),
        "sub_n_eligible": np.zeros(n_sub, dtype=np.int64),
        "est_regime": estimator_regime(O_train),
        "sub_est_regime": np.array([estimator_regime(O_train[p])
                                    for p in pos], dtype=np.int64),
        "rand_order": rand,
        "swap_order": orders,
        "rand_r2": np.full((K, n_random), np.nan),
        "rand_pop": np.full((K, n_random), np.nan),
        "rand_cells": np.full((K, n_random), np.nan),
        "rand_jacc_pair": np.full(K, np.nan),
        "random_baseline_r2": np.full(K, np.nan),
        "random_baseline_selected": np.full(k_max, -1, dtype=np.int64),
    }
    for m in METHODS:
        out[f"{m}_selected"] = np.full((K, k_max), -1, dtype=np.int64)
        out[f"{m}_alt_selected"] = np.full((K, n_sub, k_max), -1,
                                           dtype=np.int64)
        for key in ("r2_full", "pop_full", "cells_full", "jacc_pair"):
            out[f"{m}_{key}"] = np.full(K, np.nan)
        for key in ("r2_alt", "pop_alt", "cells_alt", "jacc_full"):
            out[f"{m}_{key}"] = np.full((K, n_sub), np.nan)
        out[f"{m}_n_common"] = np.full((K, n_sub), -1, dtype=np.int64)
        out[f"{m}_rand_jacc_full"] = np.full((K, n_random), np.nan)
        out[f"{m}_rand_n_common"] = np.full((K, n_random), -1,
                                            dtype=np.int64)
        for key in ("swap_r2", "swap_pop"):
            out[f"{m}_{key}"] = np.full((K, n_swap, k_max), np.nan)

    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:
        print(f"  split {split.split_id}: estimation failed: {e}",
              flush=True)
        out["estimation_failed"] = True
        out["time_s"] = time.time() - t0
        return out

    def r2_of(sel):
        if sel is None:
            return np.nan
        return impute_and_evaluate(B_test, O_test, sel, R, col_mean,
                                   col_std, N, target=target,
                                   bounds=bounds)["r2"]

    def cells_of(sel):
        # share of the observed validation cells r2_of scores
        return (np.nan if sel is None
                else evaluated_share(O_test, sel, target))

    # Random subsets (shared by all methods) and the random baseline
    rand_sets = [[(list(row[:k]) if row[k - 1] >= 0 else None)
                  for row in rand] for k in ks]
    for i, k in enumerate(ks):
        for r, sel in enumerate(rand_sets[i]):
            out["rand_r2"][i, r] = r2_of(sel)
            out["rand_pop"][i, r] = pop_r2(R, sel)
            out["rand_cells"][i, r] = cells_of(sel)
        out["rand_jacc_pair"][i] = mean_pairwise_jaccard(rand_sets[i])
    base = run_selection("random", R, k_max, benchmark_names, split, eligible)
    out["random_baseline_selected"] = _padded(base["selected"], k_max)
    for i, k in enumerate(ks):
        out["random_baseline_r2"][i] = r2_of(selection_at_k(base, k))

    # Full-training selections
    full = {m: method_sets(m, R, eligible, ks, benchmark_names, split)
            for m in METHODS}

    # Alternatives: subsample estimate, same estimator rule
    alt = {m: [[None] * n_sub for _ in ks] for m in METHODS}
    for b, p in enumerate(pos):
        O_sub = O_train[p]
        el_b = (eligible_mask(O_sub) if sub_eligible == "subsample"
                else eligible)
        out["sub_n_eligible"][b] = int(el_b.sum())
        try:
            R_b = estimate_corr(B_train[p], O_sub)[0]
        except Exception as e:
            print(f"  split {split.split_id} subsample {b}: estimation "
                  f"failed: {e}", flush=True)
            out["sub_failed"][b] = True
            continue
        for m in METHODS:
            sets = method_sets(m, R_b, el_b, ks, benchmark_names, split)
            for i in range(K):
                alt[m][i][b] = sets[i]

    for m in METHODS:
        for i, k in enumerate(ks):
            A = full[m][i]
            out[f"{m}_selected"][i] = _padded(A, k_max)
            out[f"{m}_r2_full"][i] = r2_of(A)
            out[f"{m}_pop_full"][i] = pop_r2(R, A)
            out[f"{m}_cells_full"][i] = cells_of(A)
            for b, A_b in enumerate(alt[m][i]):
                out[f"{m}_alt_selected"][i, b] = _padded(A_b, k_max)
                out[f"{m}_r2_alt"][i, b] = r2_of(A_b)
                out[f"{m}_pop_alt"][i, b] = pop_r2(R, A_b)
                out[f"{m}_cells_alt"][i, b] = cells_of(A_b)
                out[f"{m}_jacc_full"][i, b] = jaccard(A_b, A)
                out[f"{m}_n_common"][i, b] = n_common(A_b, A)
            out[f"{m}_jacc_pair"][i] = mean_pairwise_jaccard(alt[m][i])
            for r, sel in enumerate(rand_sets[i]):
                out[f"{m}_rand_jacc_full"][i, r] = jaccard(sel, A)
                out[f"{m}_rand_n_common"][i, r] = n_common(sel, A)
            # Matched null: random s-swaps of A_k, s = 1..k
            if A is None:
                continue
            for d, order in enumerate(orders):
                for s in range(1, k + 1):
                    sw = random_swap(A, eligible, order, s)
                    out[f"{m}_swap_r2"][i, d, s - 1] = r2_of(sw)
                    out[f"{m}_swap_pop"][i, d, s - 1] = pop_r2(R, sw)
    out["time_s"] = time.time() - t0
    return out


_WORKER = {}


def _init_worker(B, O, benchmark_names, kwargs):
    _WORKER.update(B=B, O=O, names=benchmark_names, kwargs=kwargs)


def _run_split(split):
    w = _WORKER
    return evaluate_split(split, w["B"], w["O"], w["names"], **w["kwargs"])


def _split_line(split, res, ks, N):
    i = ks.index(FIG_K) if FIG_K in ks else len(ks) // 2
    head = (f"  Split {split.split_id} (rep {split.rep}, fold "
            f"{split.fold}): {len(split.train_idx)} train / "
            f"{len(split.val_idx)} test, {res['n_eligible']}/{N} eligible")
    if res["estimation_failed"]:
        return head + " — estimation failed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cells = [f"{m}:J{np.nanmean(res[f'{m}_jacc_full'][i]):.2f} "
                 f"{res[f'{m}_r2_full'][i]:.3f}/"
                 f"{np.nanmean(res[f'{m}_r2_alt'][i]):.3f}" for m in METHODS]
        cells.append(f"rand {np.nanmean(res['rand_r2'][i]):.3f}")
    return (head + f"  k={ks[i]} " + "  ".join(cells)
            + f"  ({res['time_s']:.1f}s)")


def run_dataset(ds_name, ks=KS, n_folds=N_FOLDS, n_repeats=N_REPEATS,
                n_sub=N_SUB, sub_frac=SUB_FRAC, n_random=N_RANDOM,
                sub_eligible="subsample", workers=1, n_swap=N_SWAP):
    """All matched splits of one dataset: dict of (S, ...) arrays plus the
    split metadata and run scalars (module docstring)."""
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")
    d = np.load(matrix_path(ds_name), allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    ks = sorted(int(k) for k in ks)
    print(f"{M} models × {N} tasks, {O.mean()*100:.1f}% observed", flush=True)
    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED))
    print(f"{n_repeats} × {n_folds}-fold CV = {len(splits)} matched splits; "
          f"{n_sub} subsamples of {sub_frac:.0%} of the training models "
          f"(Split.seed(SEED + {SUB_SEED_OFFSET})), {n_random} random "
          f"subsets (Split.seed(SEED + {RANDOM_SUBSETS_SEED_OFFSET})), "
          f"{n_swap} random s-swap draws (Split.seed(SEED + "
          f"{SWAP_SEED_OFFSET})); k = {ks}; workers = {workers}", flush=True)
    kwargs = dict(ks=ks, n_sub=n_sub, sub_frac=sub_frac, n_random=n_random,
                  sub_eligible=sub_eligible, n_swap=n_swap,
                  bounds=dataset_bounds(ds_name, benchmark_names, DATA))
    results = []
    if workers <= 1:
        for s in splits:
            res = evaluate_split(s, B, O, benchmark_names, **kwargs)
            print(_split_line(s, res, ks, N), flush=True)
            results.append(res)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=(B, O, benchmark_names,
                                           kwargs)) as pool:
            for s, res in zip(splits, pool.map(_run_split, splits)):
                print(_split_line(s, res, ks, N), flush=True)
                results.append(res)

    arr = {key: np.array([res[key] for res in results])
           for key in results[0]}
    arr.update(
        rep=np.array([s.rep for s in splits], dtype=np.int64),
        fold=np.array([s.fold for s in splits], dtype=np.int64),
        split_id=np.array([s.split_id for s in splits], dtype=np.int64),
        n_train=np.array([len(s.train_idx) for s in splits], dtype=np.int64),
        n_val=np.array([len(s.val_idx) for s in splits], dtype=np.int64),
        ks=np.array(ks, dtype=np.int64), methods=np.array(METHODS),
        benchmark_names=benchmark_names, n_sub=np.int64(n_sub),
        sub_frac=np.float64(sub_frac), n_random=np.int64(n_random),
        n_swap=np.int64(n_swap),
        sub_eligible=np.array(sub_eligible), holdout_frac=np.float64(
            HOLDOUT_FRAC), n_folds=np.int64(n_folds),
        n_repeats=np.int64(n_repeats), seed=np.int64(SEED),
        min_train_obs=np.int64(MIN_TRAIN_OBS),
        min_target_obs=np.int64(cv_split.MIN_TARGET_OBS),
        pred_clip=np.bool_(score_bounds.PRED_CLIP), k_max=np.int64(max(ks)),
        sub_seed_offset=np.int64(SUB_SEED_OFFSET),
        random_subsets_seed_offset=np.int64(RANDOM_SUBSETS_SEED_OFFSET),
        swap_seed_offset=np.int64(SWAP_SEED_OFFSET),
        low_jaccard=np.float64(LOW_JACCARD))
    for key in ("n_eligible", "n_sub_models", "est_regime"):
        arr[key] = arr[key].astype(np.int64)
    arr["estimation_failed"] = arr["estimation_failed"].astype(bool)
    return arr


# ── Summaries, paired statistics and macros ──────────────────────

def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=None, ddof=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis, ddof=ddof)


def _num(x):
    """JSON number (None for NaN / inf)."""
    if isinstance(x, (bool, np.bool_, int, np.integer)):
        return int(x)
    x = float(x)
    return x if math.isfinite(x) else None


def _ratio(num, den):
    return float(num / den) if np.isfinite(num) and np.isfinite(den) \
        and den > 0 else np.nan


def ratio_ci(num, den, n_train, n_val, level=LEVEL):
    """(estimate, lo, hi) of the ratio of sums sum_j num_j / sum_j den_j
    over the splits j where both are finite, with its Fieller interval
    under paired_stats' Nadeau-Bengio corrected variance: the H for which
    (mean(num) - H mean(den))^2 <= t^2 (1/n + n_val/n_train)
    var(num - H den), t the (1 + level) / 2 quantile of Student's t with
    n - 1 degrees of freedom and n_val/n_train the mean ratio of those
    splits.  With den_j = 1 this is paired_stats' interval of mean(num).
    lo and hi are NaN for fewer than two splits and where the interval is
    unbounded (mean(den) not significantly nonzero)."""
    from scipy import stats

    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den)
    n = int(ok.sum())
    if n == 0:
        return np.nan, np.nan, np.nan
    a, w = num[ok], den[ok]
    a_bar, w_bar = float(np.mean(a)), float(np.mean(w))
    est = a_bar / w_bar if w_bar != 0 else np.nan
    if n < 2:
        return est, np.nan, np.nan
    rho = float(np.mean(np.asarray(n_val, dtype=float)[ok])
                / np.mean(np.asarray(n_train, dtype=float)[ok]))
    t = float(stats.t.ppf(0.5 + level / 2, n - 1))
    c = t ** 2 * (1.0 / n + rho)
    cov = np.cov(a, w, ddof=1)
    alpha = w_bar ** 2 - c * cov[1, 1]
    if not alpha > 0:
        return est, np.nan, np.nan
    beta = a_bar * w_bar - c * cov[0, 1]
    gamma = a_bar ** 2 - c * cov[0, 0]
    root = math.sqrt(max(beta ** 2 - alpha * gamma, 0.0))
    return est, (beta - root) / alpha, (beta + root) / alpha


def split_values(arr, m, i):
    """Per-split (S,) values of method m at ks[i] (NaN on failed splits):
    full, alt_mean, alt_min, alt_max, alt_sd (ddof 1), rand_mean."""
    ok = ~arr["estimation_failed"]
    r2a = np.where(ok[:, None], arr[f"{m}_r2_alt"][:, i], np.nan)
    rr = np.where(ok[:, None], arr["rand_r2"][:, i], np.nan)
    has = np.isfinite(r2a).any(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "full": np.where(ok, arr[f"{m}_r2_full"][:, i], np.nan),
            "alt_mean": np.nanmean(r2a, axis=1),
            "alt_min": np.where(has, np.nanmin(np.where(has[:, None], r2a,
                                                        0.0), axis=1), np.nan),
            "alt_max": np.where(has, np.nanmax(np.where(has[:, None], r2a,
                                                        0.0), axis=1), np.nan),
            "alt_sd": np.nanstd(r2a, axis=1, ddof=1),
            "rand_mean": np.nanmean(rr, axis=1),
        }


def alt_arrays(arr, m, i):
    """Per-alternative (S_ok, n_sub) arrays of method m at ks[i] on the
    splits with a training estimate (S_ok rows, split metadata n_train,
    n_val): cost = R^2(A_k) - R^2(A_k^b), s = k - n_common (-1: no
    alternative), null = the matched null cost R^2(A_k) - mean_d
    R^2(swap_d(A_k, s)) (0 for s = 0), pop_cost / pop_null the same for
    the population R^2, dcells = cells(A_k^b) - cells(A_k), jacc, valid
    (an alternative with finite cost) and swapped (valid, s >= 1)."""
    k = int(arr["ks"][i])
    ok = ~arr["estimation_failed"]
    nc = arr[f"{m}_n_common"][ok, i]
    s = np.where(nc >= 0, k - nc, -1)
    full = arr[f"{m}_r2_full"][ok, i]
    pfull = arr[f"{m}_pop_full"][ok, i]
    cost = full[:, None] - arr[f"{m}_r2_alt"][ok, i]
    pop_cost = pfull[:, None] - arr[f"{m}_pop_alt"][ok, i]
    jacc = arr[f"{m}_jacc_full"][ok, i]
    valid = np.isfinite(jacc) & np.isfinite(cost) & (nc >= 0)
    # Cost of the random s-swaps, mean over the draws: column s - 1
    null_s = full[:, None] - _nanmean(arr[f"{m}_swap_r2"][ok, i], axis=1)
    pnull_s = pfull[:, None] - _nanmean(arr[f"{m}_swap_pop"][ok, i], axis=1)
    idx = np.clip(s - 1, 0, null_s.shape[1] - 1)
    null = np.where(s >= 1, np.take_along_axis(null_s, idx, axis=1),
                    np.where(s == 0, 0.0, np.nan))
    pop_null = np.where(s >= 1, np.take_along_axis(pnull_s, idx, axis=1),
                        np.where(s == 0, 0.0, np.nan))
    dcells = (arr[f"{m}_cells_alt"][ok, i]
              - arr[f"{m}_cells_full"][ok, i, None])
    return {"k": k, "cost": cost, "s": s, "null": null,
            "pop_cost": pop_cost, "pop_null": pop_null, "dcells": dcells,
            "jacc": jacc, "valid": valid, "swapped": valid & (s >= 1),
            "n_train": arr["n_train"][ok], "n_val": arr["n_val"][ok]}


def _split_sums(values, weights, mask):
    """Per-split sums of values and weights over the alternatives in mask
    ((S,) each, NaN on splits without one) and the number of splits that
    have one."""
    has = mask.any(axis=1)
    num = np.where(has, np.sum(np.where(mask, values, 0.0), axis=1), np.nan)
    den = np.where(has, np.sum(np.where(mask, weights, 0.0), axis=1), np.nan)
    return num, den, int(has.sum())


def _enough(n_alt, n_splits):
    return n_alt >= MIN_N_ALT and n_splits >= MIN_N_SPLITS


def _distinct(sel):
    """Number of distinct sets among the rows of (n_sub, k_max) sel (rows
    of -1 only: none)."""
    sets = {tuple(sorted(int(j) for j in row if j >= 0)) for row in sel}
    sets.discard(())
    return len(sets)


def cross_split_jaccard(arr, m, i):
    """Mean Jaccard of the full selections A_k of two different splits
    (all pairs of the splits with a training estimate and a size-k
    selection)."""
    k = int(arr["ks"][i])
    ok = ~arr["estimation_failed"]
    sel = arr[f"{m}_selected"][ok, i, :k]
    return mean_pairwise_jaccard([row.tolist() if np.all(row >= 0) else None
                                  for row in sel])


def method_table(arr, m, i, gap_significant=True, alt_ci_low=np.nan):
    """Summary of method m at k = ks[i] (module docstring).  The ratios to
    the gap and harm are NaN unless gap_significant (the full selection
    beats the random subsets: full_vs_random interval above 0);
    alt_ci_low, the lower bound of the alt_vs_full interval, gives
    worst_cost_over_gap.  Losses and harms need MIN_N_ALT alternatives
    from MIN_N_SPLITS splits, harms a bounded interval."""
    a = alt_arrays(arr, m, i)
    k = a["k"]
    ok = ~arr["estimation_failed"]
    v = split_values(arr, m, i)
    cost_b, valid, swapped, s = a["cost"], a["valid"], a["swapped"], a["s"]
    jf = a["jacc"]
    meta = (a["n_train"], a["n_val"])
    full = v["full"][ok]
    pop_rel = arr[f"{m}_pop_alt"][ok, i] / arr[f"{m}_pop_full"][ok, i, None]

    # Cost per replaced benchmark, and against the matched null
    n_swapped = int(swapped.sum())
    num, den, n_swap_splits = _split_sums(cost_b, s, swapped)
    swap_loss, swap_lo, swap_hi = ratio_ci(num, den, *meta)
    enough = _enough(n_swapped, n_swap_splits)
    hmask = swapped & np.isfinite(a["null"])
    num, den, _ = _split_sums(cost_b, a["null"], hmask)
    harm, harm_lo, harm_hi = ratio_ci(num, den, *meta)
    null_swap_loss = (float(np.sum(a["null"][hmask]) / np.sum(s[hmask]))
                      if hmask.any() else np.nan)
    pmask = swapped & np.isfinite(a["pop_null"]) & np.isfinite(a["pop_cost"])
    num, den, _ = _split_sums(a["pop_cost"], a["pop_null"], pmask)
    harm_pop, hp_lo, hp_hi = ratio_ci(num, den, *meta)
    # a bounded interval with a positive denominator: significantly > 0
    if not (enough and np.isfinite(hp_lo) and np.nansum(den) > 0):
        harm_pop = hp_lo = hp_hi = np.nan
    # The first version's reference: the random k-subsets against A_k
    rnc = arr[f"{m}_rand_n_common"][ok, i]
    rcost = full[:, None] - arr["rand_r2"][ok, i]
    rswap = (rnc >= 0) & (rnc < k) & np.isfinite(rcost)
    rand_swap_loss = (float(np.sum(rcost[rswap]) / np.sum((k - rnc)[rswap]))
                      if rswap.any() else np.nan)

    low = valid & (jf < LOW_JACCARD)
    num, den, n_low_splits = _split_sums(cost_b, np.ones_like(cost_b), low)
    low_cost, low_lo, low_hi = ratio_ci(num, den, *meta)
    n_low = int(low.sum())
    if not _enough(n_low, n_low_splits):
        low_cost = low_lo = low_hi = np.nan

    # Coverage: the cells each set is scored on
    dc = a["dcells"]
    shifted = valid & np.isfinite(dc) & (np.abs(dc) > COV_SHIFT)
    matched = valid & np.isfinite(dc) & (np.abs(dc) <= COV_SHIFT)
    cm = swapped & np.isfinite(dc)
    cov_corr = np.nan
    if cm.sum() >= 3 and np.std(dc[cm]) > 0 and np.std(cost_b[cm]) > 0:
        cov_corr = float(np.corrcoef(dc[cm], cost_b[cm])[0, 1])

    gap = v["full"] - v["rand_mean"]
    cost = v["full"] - v["alt_mean"]
    by_swap = {}
    for n_sw in range(k + 1):
        sel = valid & (s == n_sw)
        if sel.any():
            nsel = sel & np.isfinite(a["null"])
            by_swap[str(n_sw)] = {
                "n": int(sel.sum()), "cost": _num(np.mean(cost_b[sel])),
                "null_cost": (_num(np.mean(a["null"][nsel])) if nsel.any()
                              else None)}
    alt_sel = arr[f"{m}_alt_selected"][ok, i]
    tab = {
        "n_splits": int(np.sum(np.isfinite(v["full"]))),
        "n_alternatives": int(valid.sum()),
        "jacc_pair": _nanmean(arr[f"{m}_jacc_pair"][ok, i]),
        "jacc_full": _nanmean(jf),
        "cross_jacc": cross_split_jaccard(arr, m, i),
        "ident": (float(np.mean(jf[valid] == 1.0)) if valid.any()
                  else np.nan),
        "distinct": (float(np.mean([_distinct(x) for x in alt_sel]))
                     if len(alt_sel) else np.nan),
        "r2_full": _nanmean(v["full"]),
        "r2_full_std": _nanstd(v["full"]),
        "r2_alt": _nanmean(v["alt_mean"]),
        "r2_alt_sd": _nanmean(v["alt_sd"]),
        "r2_alt_range": _nanmean(v["alt_max"] - v["alt_min"]),
        "r2_alt_min": _nanmean(v["alt_min"]),
        "cost": _nanmean(cost),
        "cost_std": _nanstd(cost),
        "pop_full": _nanmean(arr[f"{m}_pop_full"][ok, i]),
        "pop_alt": _nanmean(arr[f"{m}_pop_alt"][ok, i]),
        "pop_rel": _nanmean(pop_rel),
        "pop_rel_min": _nanmean(np.where(np.isfinite(pop_rel).any(axis=1),
                                         np.nanmin(np.where(
                                             np.isfinite(pop_rel), pop_rel,
                                             np.inf), axis=1), np.nan)),
        "gap": _nanmean(gap),
        "n_swapped": n_swapped,
        "n_swap_splits": n_swap_splits,
        "swap_loss": swap_loss if enough else np.nan,
        "swap_loss_lo": swap_lo if enough else np.nan,
        "swap_loss_hi": swap_hi if enough else np.nan,
        "null_swap_loss": null_swap_loss if enough else np.nan,
        "rand_swap_loss": rand_swap_loss,
        "harm_pop": harm_pop,
        "harm_pop_lo": hp_lo,
        "harm_pop_hi": hp_hi,
        "n_low": n_low,
        "n_low_splits": n_low_splits,
        "low_share": (float(np.mean(low[valid])) if valid.any()
                      else np.nan),
        "low_cost": low_cost,
        "low_cost_lo": low_lo,
        "low_cost_hi": low_hi,
        "cells_full": _nanmean(arr[f"{m}_cells_full"][ok, i]),
        "cells_alt": (_nanmean(arr[f"{m}_cells_alt"][ok, i][valid])
                      if valid.any() else np.nan),
        "cells_shift": (float(np.mean(shifted[valid])) if valid.any()
                        else np.nan),
        "cost_cov_matched": (float(np.mean(cost_b[matched])) if matched.any()
                             else np.nan),
        "cov_cost_corr": cov_corr,
        "rand_jacc_full": _nanmean(arr[f"{m}_rand_jacc_full"][ok, i]),
        "gap_significant": bool(gap_significant),
    }
    g = tab["gap"] if gap_significant else np.nan
    tab["sd_over_gap"] = _ratio(tab["r2_alt_sd"], g)
    tab["range_over_gap"] = _ratio(tab["r2_alt_range"], g)
    tab["cost_over_gap"] = (float(tab["cost"] / g)
                            if np.isfinite(g) and g > 0 else np.nan)
    tab["worst_cost_over_gap"] = (float(-alt_ci_low / g)
                                  if np.isfinite(g) and g > 0
                                  and np.isfinite(alt_ci_low) else np.nan)
    use = gap_significant and enough
    # an unbounded interval: the null cost is not significantly nonzero
    bounded = use and np.isfinite(harm_lo)
    tab["harm"] = harm if bounded else np.nan
    tab["harm_lo"] = harm_lo if bounded else np.nan
    tab["harm_hi"] = harm_hi if bounded else np.nan
    tab["harm_subsets"] = (float(swap_loss / rand_swap_loss)
                           if bounded and np.isfinite(rand_swap_loss)
                           and rand_swap_loss > 0 else np.nan)
    tab = {key: (_num(val) if not isinstance(val, dict) else val)
           for key, val in tab.items()}
    tab["status"] = ("uninformative" if not gap_significant
                     else "stable" if not enough else "tested")
    tab["by_swap"] = by_swap
    return tab


def random_table(arr, i):
    """Summary of the random subsets (and random baseline) at ks[i]."""
    ok = ~arr["estimation_failed"]
    rr = arr["rand_r2"][ok, i]
    has = np.isfinite(rr).any(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rng_ = (np.nanmax(np.where(has[:, None], rr, 0.0), axis=1)
                - np.nanmin(np.where(has[:, None], rr, 0.0), axis=1))
    rng_ = np.where(has, rng_, np.nan)
    return {key: _num(val) for key, val in {
        "n_splits": int(has.sum()),
        "jacc_pair": _nanmean(arr["rand_jacc_pair"][ok, i]),
        "r2": _nanmean(_nanmean(rr, axis=1)),
        "r2_sd": _nanmean(_nanstd(rr, axis=1, ddof=1)),
        "r2_range": _nanmean(rng_),
        "pop": _nanmean(arr["rand_pop"][ok, i]),
        "cells": _nanmean(arr["rand_cells"][ok, i]),
        "random_baseline_r2": _nanmean(arr["random_baseline_r2"][ok, i]),
    }.items()}


def paired_rows(ds_name, arr):
    """paired_stats rows of COMPARISONS for every method and k
    (PAIRED_FIELDS)."""
    rows = []
    meta = (arr["n_train"], arr["n_val"], arr["rep"])
    for i, k in enumerate(arr["ks"]):
        for m in METHODS:
            v = split_values(arr, m, i)
            pairs = {
                "alt_vs_full": (f"{m}_alt", m, v["alt_mean"], v["full"]),
                "full_vs_random": (m, "random_sets", v["full"],
                                   v["rand_mean"]),
                "alt_vs_random": (f"{m}_alt", "random_sets", v["alt_mean"],
                                  v["rand_mean"]),
                "altmin_vs_random": (f"{m}_altmin", "random_sets",
                                     v["alt_min"], v["rand_mean"]),
            }
            for cmp_ in COMPARISONS:
                ma, mb, a, b = pairs[cmp_]
                # "method" keys the macros; write_csv keeps PAIRED_FIELDS
                rows.append({"dataset": ds_name,
                             "holdout_frac": float(arr["holdout_frac"]),
                             "k": int(k), "comparison": cmp_,
                             "method_a": ma, "method_b": mb, "method": m,
                             **paired_stats(a, b, *meta)})
    return rows


METHOD_FIELDS = {
    "JaccPair": "jacc_pair", "JaccFull": "jacc_full",
    "CrossJacc": "cross_jacc", "Ident": "ident",
    "Distinct": "distinct", "RsqFull": "r2_full", "RsqAlt": "r2_alt",
    "RsqAltSd": "r2_alt_sd", "RsqAltRange": "r2_alt_range",
    "RsqAltMin": "r2_alt_min", "Cost": "cost", "PopFull": "pop_full",
    "PopRel": "pop_rel", "PopRelMin": "pop_rel_min", "Gap": "gap",
    "SdOverGap": "sd_over_gap", "RangeOverGap": "range_over_gap",
    "CostOverGap": "cost_over_gap",
    "WorstCostOverGap": "worst_cost_over_gap",
    "NSwapped": "n_swapped", "NSwapSplits": "n_swap_splits",
    "SwapLoss": "swap_loss", "SwapLossLo": "swap_loss_lo",
    "SwapLossHi": "swap_loss_hi", "NullSwapLoss": "null_swap_loss",
    "RandSwapLoss": "rand_swap_loss", "Harm": "harm", "HarmLo": "harm_lo",
    "HarmHi": "harm_hi", "HarmPop": "harm_pop", "NLow": "n_low",
    "LowShare": "low_share", "LowCost": "low_cost",
    "LowCostLo": "low_cost_lo", "LowCostHi": "low_cost_hi",
    "CellsFull": "cells_full", "CellsAlt": "cells_alt",
    "CellsShift": "cells_shift", "CostCovMatched": "cost_cov_matched",
}
RANDOM_FIELDS = {"JaccPair": "jacc_pair", "Rsq": "r2", "RsqSd": "r2_sd",
                 "RsqRange": "r2_range", "Pop": "pop"}


def summarize(ds_name, arr, paired):
    """(tables, macros): tables {"k": {method | "random": summary}} and the
    flat letters-only macros dict (module docstring)."""
    D = camel(ds_name)
    P = MACRO_PREFIX
    macros = {}

    def put(name, value):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name!r} is not letters only")
        if name in macros:
            raise ValueError(f"duplicate macro {name}")
        macros[name] = _num(value) if value is not None else None

    lookup = {(r["k"], r["comparison"], r["method"]): r for r in paired}
    tables = {}
    for i, k in enumerate(arr["ks"]):
        K_ = "K" + int_words(k)
        tk = {m: method_table(
            arr, m, i,
            bool(lookup[(int(k), "full_vs_random", m)]["ci_low"] > 0),
            lookup[(int(k), "alt_vs_full", m)]["ci_low"])
            for m in METHODS}
        tk["random"] = random_table(arr, i)
        tables[str(int(k))] = tk
        for m in METHODS:
            for field, key in METHOD_FIELDS.items():
                put(f"{P}{field}{D}{camel(m)}{K_}", tk[m][key])
        for field, key in RANDOM_FIELDS.items():
            put(f"{P}{field}{D}Random{K_}", tk["random"][key])
        put(f"{P}Rsq{D}RandomBaseline{K_}", tk["random"]["random_baseline_r2"])
    for k in arr["ks"]:
        K_ = "K" + int_words(k)
        for m in METHODS:
            for cmp_ in COMPARISONS:
                r = lookup[(int(k), cmp_, m)]
                for S_, field in PAIR_STATS.items():
                    put(f"{P}Pair{S_}{D}{camel(m)}{camel(cmp_)}{K_}",
                        r[field])
    ok = ~arr["estimation_failed"]
    put(f"{P}NSplits{D}", int(ok.sum()))
    put(f"{P}NReps{D}", int(np.unique(arr["rep"][ok]).size))
    put(f"{P}NSub{D}", int(arr["n_sub"]))
    put(f"{P}NRandom{D}", int(arr["n_random"]))
    put(f"{P}NSwap{D}", int(arr["n_swap"]))
    put(f"{P}SubFrac{D}", float(arr["sub_frac"]))
    put(f"{P}SubModels{D}", float(np.mean(arr["n_sub_models"])))
    put(f"{P}SubFailed{D}", int(np.sum(arr["sub_failed"][ok])))
    put(f"{P}RegimeChanged{D}", int(np.sum(
        arr["sub_est_regime"][ok] != arr["est_regime"][ok, None])))
    put(f"{P}SubEligDrop{D}", float(np.mean(
        arr["n_eligible"][ok, None] - arr["sub_n_eligible"][ok]))
        if ok.any() else np.nan)
    return tables, macros


def print_summary(ds_name, tables, paired, n_splits=None):
    """Stability table per k (method rows, random subsets), the direct
    test per k and the alt_vs_full / full_vs_random / alt_vs_random paired
    differences; n_splits: splits with a training estimate (default:
    those where mi's first set was scored)."""
    if n_splits is None:
        n_splits = next(iter(tables.values()))["mi"]["n_splits"]
    print(f"\n== {display_name(ds_name)}: selection stability over "
          f"{n_splits} splits (alternatives: training subsamples; R² with "
          f"the full training estimate; R² is NaN where no validation "
          f"model observes a selected benchmark) ==")
    f = lambda x, d=3: "  n/a" if x is None else f"{x:.{d}f}"
    for k, tk in tables.items():
        print(f"k={k}  {'method':<9}{'Jpair':>6}{'Jfull':>6}{'Jcross':>7}"
              f"{'ident':>6}{'R2full':>8}{'R2alt':>8}{'sd':>7}{'range':>7}"
              f"{'cost':>8}{'popRel':>8}{'gap':>7}{'sd/gap':>7}"
              f"{'lowSh':>6}")
        for m in METHODS:
            t = tk[m]
            print(f"     {m:<9}{f(t['jacc_pair'], 2):>6}"
                  f"{f(t['jacc_full'], 2):>6}{f(t['cross_jacc'], 2):>7}"
                  f"{f(t['ident'], 2):>6}"
                  f"{f(t['r2_full']):>8}{f(t['r2_alt']):>8}"
                  f"{f(t['r2_alt_sd']):>7}{f(t['r2_alt_range']):>7}"
                  f"{f(t['cost']):>8}{f(t['pop_rel']):>8}{f(t['gap']):>7}"
                  f"{f(t['sd_over_gap'], 2):>7}{f(t['low_share'], 2):>6}")
        t = tk["random"]
        print(f"     {'random':<9}{f(t['jacc_pair'], 2):>6}{'':>19}"
              f"{f(t['r2']):>16}{f(t['r2_sd']):>7}{f(t['r2_range']):>7}"
              f"   (baseline seed+7: {f(t['random_baseline_r2'])})")
    print(f"\n== {display_name(ds_name)}: direct test (swapped "
          f"alternatives vs random s-swaps of the same A_k; 95% Fieller "
          f"intervals, NB-corrected; n/a below {MIN_N_ALT} alternatives or "
          f"{MIN_N_SPLITS} splits) ==")
    ci = lambda t, key, d=4: (f"{f(t[key], d)} [{f(t[key + '_lo'], d)}, "
                              f"{f(t[key + '_hi'], d)}]")
    for k, tk in tables.items():
        print(f"k={k}  {'method':<9}{'nSw':>5}{'spl':>4}  "
              f"{'loss/swap [CI]':<27}{'null/swap':>9}  "
              f"{'harm [CI]':<22}{'hPop':>6}{'worst/gap':>10}"
              f"{'cells f/a':>11}{'shift':>6}  status")
        for m in METHODS:
            t = tk[m]
            print(f"     {m:<9}{t['n_swapped']:>5}{t['n_swap_splits']:>4}  "
                  f"{ci(t, 'swap_loss'):<27}{f(t['null_swap_loss'], 4):>9}  "
                  f"{ci(t, 'harm', 2):<22}{f(t['harm_pop'], 2):>6}"
                  f"{f(t['worst_cost_over_gap'], 2):>10}"
                  f"{f(t['cells_full'], 2):>6}/{f(t['cells_alt'], 2):<4}"
                  f"{f(t['cells_shift'], 2):>6}  {t['status']}")
    print(f"\n== {display_name(ds_name)}: paired R² differences "
          f"(Nadeau–Bengio 95% CI, win rate) ==")
    for r in paired:
        if r["comparison"] not in ("alt_vs_full", "full_vs_random",
                                   "alt_vs_random"):
            continue
        print(f"  k={r['k']:<3}{r['method_a']:>12} − {r['method_b']:<12}"
              f"{r['mean_diff']:+.4f}  [{r['ci_low']:+.4f}, "
              f"{r['ci_high']:+.4f}]  win {r['win_rate']:.2f}  "
              f"(n={r['n_splits']})")


# ── Plotting ─────────────────────────────────────────────────────

def direct_curves(arr, m, k=FIG_K):
    """(xs, cost, null, n) of the direct-test panel for method m at k
    (FIG_K, or the middle k of the run): per number of replaced
    benchmarks x = 0..k the mean cost of the alternatives that replace x
    benchmarks and the mean matched-null cost of the same alternatives
    (random x-swaps of the same split's A_k); NaN for bins with fewer
    than FIG_MIN_BIN alternatives."""
    ks = [int(x) for x in arr["ks"]]
    i = ks.index(k) if k in ks else len(ks) // 2
    a = alt_arrays(arr, m, i)
    xs = np.arange(a["k"] + 1)
    cost, null, n = (np.full(len(xs), np.nan) for _ in range(3))
    for x in xs:
        sel = a["valid"] & (a["s"] == x) & np.isfinite(a["null"])
        n[x] = sel.sum()
        if sel.sum() >= FIG_MIN_BIN:
            cost[x] = np.mean(a["cost"][sel])
            null[x] = np.mean(a["null"][sel])
    return xs, cost, null, n


def _panel_direct(ax, arr, k=FIG_K, ds_name=None):
    """Cost R²(A_k) − R²(A) by number of replaced benchmarks k − |A n A_k|:
    the alternatives of each method (solid) and, dashed in the method's
    color, the random swaps of as many benchmarks of the same splits' A_k
    (the matched null of the same alternatives; each method against its
    own reference).  Methods whose full selection does not beat the
    random subsets significantly at this k (full_vs_random interval not
    above 0: status "uninformative") are faded.  Bins with fewer than
    FIG_MIN_BIN alternatives are not drawn."""
    ks = [int(x) for x in arr["ks"]]
    i = ks.index(k) if k in ks else len(ks) // 2
    xs = None
    for j, m in enumerate(METHODS):
        xs, cost, null, _ = direct_curves(arr, m, k)
        v = split_values(arr, m, i)
        gap = paired_stats(v["full"], v["rand_mean"], arr["n_train"],
                           arr["n_val"])
        alpha = 1.0 if gap["ci_low"] > 0 else 0.3
        off = (j - (len(METHODS) - 1) / 2) * 0.06
        ax.plot(xs + off, cost, METHOD_STYLE[m], color=METHOD_COLOR[m],
                label=METHOD_LABEL[m], markersize=4, linewidth=1.2,
                alpha=alpha)
        ax.plot(xs + off, null, "--", color=METHOD_COLOR[m], linewidth=0.9,
                alpha=0.8 * alpha)
    ax.plot([], [], "k-", linewidth=1.2, label="alternatives")
    ax.plot([], [], "k--", linewidth=0.9,
            label="random swaps of $A_k$ (matched)")
    ax.plot([], [], "k-", linewidth=1.2, alpha=0.3,
            label="faded: no significant gap to random")
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_xlabel(f"Benchmarks replaced (of $k$ = {len(xs) - 1})")
    ax.set_ylabel(r"$R^2$(full) $-$ $R^2$(alternative)")
    ax.set_xticks(xs)
    if ds_name is not None:
        ax.set_title(display_name(ds_name), fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_dataset(ds_name, arr, tables):
    ks = [int(k) for k in arr["ks"]]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for j, m in enumerate(METHODS):
        c, ls = METHOD_COLOR[m], METHOD_STYLE[m]
        off = (j - (len(METHODS) - 1) / 2) * 0.12
        tk = [tables[str(k)][m] for k in ks]
        val = lambda key: np.array([np.nan if t[key] is None else t[key]
                                    for t in tk], dtype=float)
        axes[0].plot(ks, val("jacc_pair"), ls, color=c, markersize=4,
                     label=f"{METHOD_LABEL[m]}")
        axes[0].plot(ks, val("cross_jacc"), ":", color=c, linewidth=1.0)
        axes[1].plot(np.array(ks) + off, val("r2_full"), ls, color=c,
                     markersize=4, label=METHOD_LABEL[m])
        axes[1].errorbar(np.array(ks) + off, val("r2_alt"),
                         yerr=val("r2_alt_sd"), fmt="none", ecolor=c,
                         elinewidth=1.0, capsize=2)
        axes[1].plot(np.array(ks) + off, val("r2_alt"), "o", color=c,
                     markerfacecolor="none", markersize=5)
        axes[3].plot(ks, val("pop_rel"), ls, color=c, markersize=4,
                     label=METHOD_LABEL[m])
        axes[3].plot(ks, val("pop_rel_min"), ":", color=c, linewidth=1.0)
    rt = [tables[str(k)]["random"] for k in ks]
    rval = lambda key: np.array([np.nan if t[key] is None else t[key]
                                 for t in rt], dtype=float)
    axes[0].plot(ks, rval("jacc_pair"), "x--", color=RANDOM_COLOR,
                 label="Random subsets")
    axes[1].plot(ks, rval("r2"), "x--", color=RANDOM_COLOR,
                 label="Random subsets")
    axes[0].set_ylabel("Jaccard (solid: alternatives pairwise; "
                       "dotted: full selections across splits)", fontsize=8)
    axes[1].set_ylabel(r"$R^2$ (filled: full; open: alternatives ± sd)",
                       fontsize=8)
    _panel_direct(axes[2], arr)
    axes[3].set_ylabel("Population $R^2$ of alternatives / full "
                       "(dotted: min)", fontsize=8)
    for ax in (axes[0], axes[1], axes[3]):
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
    for ax in axes:
        ax.legend(fontsize=6)
    fig.suptitle(f"{display_name(ds_name)}: selection stability "
                 f"({int(arr['n_sub'])} subsamples of "
                 f"{float(arr['sub_frac']):.0%} of the training models)",
                 fontsize=10)
    plt.tight_layout()
    path = OUT / f"selection_stability_{ds_name}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)


def plot_all(all_arr):
    fig, axes = plt.subplots(1, len(all_arr), figsize=(4.2 * len(all_arr), 3.6),
                             squeeze=False)
    for ax, (ds_name, arr) in zip(axes[0], all_arr.items()):
        _panel_direct(ax, arr, ds_name=ds_name)
    axes[0, 0].legend(fontsize=6)
    plt.tight_layout()
    path = OUT / "selection_stability_all.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)


def all_figure(ds_names):
    """OUT/selection_stability_all.pdf from the existing logs of ds_names
    (--all-figure: the aggregate step after one job per dataset; reads
    nothing else and writes nothing else)."""
    plot_all({ds: load_log(ds)[0] for ds in ds_names})


# ── Consistency with the baseline logs ───────────────────────────

def check_logs(ds_name, arr, log_dir):
    """Compare the full-training selections and R² (and the random
    baseline) with an eval_baselines.py log of the same splits: per method
    whether the size-k selections agree and the max |R² difference| (None
    if the log is missing or has other splits; a method the log lacks, e.g.
    facloc in a log written before 2026-10-19, is left out)."""
    path = pathlib.Path(log_dir) / f"baselines_{ds_name}.npz"
    if not path.is_file():
        print(f"  check: {path} not found; skipped")
        return None
    d = np.load(path, allow_pickle=True)
    if not np.array_equal(d["split_id"], arr["split_id"]):
        print(f"  check: {path} has other splits; skipped")
        return None
    k_log = int(d["k_max"])
    out = {}
    for m in METHODS + ["random"]:
        if f"{m}_r2" not in d or (m != "pca" and f"{m}_selected" not in d):
            continue
        same, diff = True, 0.0
        for i, k in enumerate(int(x) for x in arr["ks"]):
            if k > k_log:
                continue
            if m == "pca":
                ref = d["pca_selected_per_k"][:, k - 1, :k]
            else:
                ref = d[f"{m}_selected"][:, :k]
            if m == "random":
                ours = arr["random_baseline_selected"][:, :k]
                r2 = arr["random_baseline_r2"][:, i]
            else:
                ours = arr[f"{m}_selected"][:, i, :k]
                r2 = arr[f"{m}_r2_full"][:, i]
            same &= bool(np.array_equal(ref, ours))
            dd = np.abs(d[f"{m}_r2"][:, k - 1] - r2)
            if np.isfinite(dd).any():
                diff = max(diff, float(np.nanmax(dd)))
        out[m] = {"same_selected": same, "max_abs_r2_diff": diff}
    print("  check vs " + str(path) + ": " + ", ".join(
        f"{m} {'=' if v['same_selected'] else '≠'}sel "
        f"Δ{v['max_abs_r2_diff']:.1e}" for m, v in out.items()), flush=True)
    return out


# ── Logs ─────────────────────────────────────────────────────────

def load_log(ds_name):
    """(arr, config) of an earlier run: LOGS/selection_stability_{ds}.npz
    and the config of its JSON (runtime of the original run kept).  Logs
    without the random s-swaps (this script's first version: no matched
    null, no cell shares) or with other seed tags are refused: rerun."""
    path = LOGS / f"selection_stability_{ds_name}.npz"
    with np.load(path, allow_pickle=True) as d:
        arr = {key: d[key] for key in d.files}
    tags = tuple(int(arr[key]) if key in arr else None
                 for key in ("sub_seed_offset", "swap_seed_offset"))
    if tags != (SUB_SEED_OFFSET, SWAP_SEED_OFFSET) or "swap_order" not in arr:
        raise ValueError(f"{path} was written by an earlier version of "
                         f"eval_selection_stability.py (subsample / swap "
                         f"seed tags {tags}, random s-swaps "
                         f"{'swap_order' in arr}); rerun it")
    path = LOGS / f"selection_stability_{ds_name}.json"
    config = {}
    if path.is_file():
        with open(path) as f:
            config = json.load(f).get("config", {})
    config["from_logs"] = True
    return arr, config


def save(ds_name, arr, tables, macros, paired, config, check):
    np.savez(LOGS / f"selection_stability_{ds_name}.npz", **arr)
    write_csv(paired, LOGS / f"selection_stability_{ds_name}_paired.csv",
              PAIRED_FIELDS)
    doc = {"script": "eval_selection_stability.py", "dataset": ds_name,
           "description": "alternative selections of each method from "
                          "subsamples of the training models of every "
                          "matched split, scored on the split's validation "
                          "fold with the full training estimate; random "
                          "eligible k-subsets and random s-swaps of the "
                          "full selection for reference",
           "config": config, "tables": tables, "check_logs": check,
           "macros": macros}
    path = LOGS / f"selection_stability_{ds_name}.json"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)
    print(f"Results saved to {LOGS}/selection_stability_{ds_name}."
          f"{{npz,json}} and _paired.csv ({len(macros)} macros)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Selection stability on the matched repeated-CV splits: "
                    "alternative selections from training subsamples, "
                    "scored with the full training estimate.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names, read from "
                             "data/<name>.matrix.npz (default: %(default)s)")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions (default %(default)s)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--ks", type=int, nargs="+", default=KS,
                        help="selection sizes (default %(default)s)")
    parser.add_argument("--n-sub", type=int, default=N_SUB,
                        help="training subsamples per split "
                             "(default %(default)s)")
    parser.add_argument("--sub-frac", type=float, default=SUB_FRAC,
                        help="share of the training models per subsample "
                             "(default %(default)s)")
    parser.add_argument("--n-random", type=int, default=N_RANDOM,
                        help="random eligible k-subsets per split "
                             "(default %(default)s)")
    parser.add_argument("--n-swap", type=int, default=N_SWAP,
                        help="random s-swap draws per split, the matched "
                             "null of the direct test (default %(default)s)")
    parser.add_argument("--sub-eligible", choices=["subsample", "split"],
                        default="subsample",
                        help="eligibility of the alternatives: "
                             "eligible_mask of the subsample (default) or "
                             "the split's mask")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes over the splits (results do not "
                             "depend on it; default %(default)s)")
    parser.add_argument("--check-logs", default=None,
                        help="directory with baselines_<ds>.npz to compare "
                             "the full-training selections and R² with")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--from-logs", action="store_true",
                      help="recompute summaries, paired CSVs and figures "
                           "from existing LOGS/selection_stability_<ds>."
                           "npz (no selection runs; run options from the "
                           "logs)")
    mode.add_argument("--all-figure", action="store_true",
                      help="only write OUT/selection_stability_all.pdf from "
                           "the existing LOGS/selection_stability_<ds>.npz "
                           "of --datasets (the aggregate step after one job "
                           "per dataset; no runs, no other outputs)")
    args = parser.parse_args()
    if args.folds < 2 or args.repeats < 1:
        parser.error("need --folds >= 2 and --repeats >= 1")
    if (min(args.ks) < 1 or args.n_sub < 2 or args.n_random < 2
            or args.n_swap < 1):
        parser.error("need --ks >= 1, --n-sub >= 2, --n-random >= 2 and "
                     "--n-swap >= 1")
    if not 0.0 < args.sub_frac <= 1.0 or args.workers < 1:
        parser.error("need 0 < --sub-frac <= 1 and --workers >= 1")
    if args.all_figure or args.from_logs:
        missing = [ds for ds in args.datasets if not (
            LOGS / f"selection_stability_{ds}.npz").is_file()]
        if missing:
            parser.error(f"no {LOGS}/selection_stability_<ds>.npz for "
                         f"{missing}")
    else:
        missing = [ds for ds in args.datasets
                   if not matrix_path(ds).is_file()]
        if missing:
            parser.error(f"no {DATA}/<name>.matrix.npz for {missing}")
    if args.all_figure:
        all_figure(args.datasets)
        sys.exit(0)

    config = {"repeats": args.repeats, "folds": args.folds,
              "ks": sorted(args.ks), "holdout_frac": HOLDOUT_FRAC,
              "seed": SEED, "min_train_obs": MIN_TRAIN_OBS,
              "min_target_obs": int(cv_split.MIN_TARGET_OBS),
              "pred_clip": bool(score_bounds.PRED_CLIP),
              "methods": METHODS, "n_sub": args.n_sub,
              "sub_frac": args.sub_frac, "n_random": args.n_random,
              "n_swap": args.n_swap,
              "sub_eligible": args.sub_eligible,
              "sub_seed_tag": SEED + SUB_SEED_OFFSET,
              "random_subsets_seed_tag": SEED + RANDOM_SUBSETS_SEED_OFFSET,
              "swap_seed_tag": SEED + SWAP_SEED_OFFSET,
              "random_baseline_seed_tag": SEED + RANDOM_SEED_OFFSET,
              "low_jaccard": LOW_JACCARD, "pop_ridge": POP_RIDGE,
              "min_n_alt": MIN_N_ALT, "min_n_splits": MIN_N_SPLITS,
              "cov_shift": COV_SHIFT,
              "workers": args.workers, "argv": sys.argv[1:]}
    total_t0 = time.time()
    all_arr = {}
    for ds_name in args.datasets:
        ds_t0 = time.time()
        if args.from_logs:
            arr, cfg = load_log(ds_name)
        else:
            arr = run_dataset(ds_name, args.ks, args.folds, args.repeats,
                              args.n_sub, args.sub_frac, args.n_random,
                              args.sub_eligible, args.workers, args.n_swap)
            cfg = dict(config, runtime_s=round(time.time() - ds_t0, 1),
                       split_time_s=round(float(np.sum(arr["time_s"])), 1))
            # per-split arrays first, so a failing summary loses nothing
            # (rerun the rest with --from-logs)
            np.savez(LOGS / f"selection_stability_{ds_name}.npz", **arr)
        paired = paired_rows(ds_name, arr)
        tables, macros = summarize(ds_name, arr, paired)
        print_summary(ds_name, tables, paired,
                      int(np.sum(~arr["estimation_failed"])))
        check = (check_logs(ds_name, arr, args.check_logs)
                 if args.check_logs else None)
        plot_dataset(ds_name, arr, tables)
        save(ds_name, arr, tables, macros, paired, cfg, check)
        all_arr[ds_name] = arr
        print(f"{ds_name.upper()} completed in {time.time() - ds_t0:.0f}s",
              flush=True)
    if len(all_arr) > 1:
        plot_all(all_arr)
    print(f"\nAll datasets completed in {time.time() - total_t0:.0f}s",
          flush=True)
