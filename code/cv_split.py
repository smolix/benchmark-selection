"""Balanced k-fold cross-validation with training subsampling.

Usage:
    from cv_split import cv_folds, cv_train_val

    folds = cv_folds(M, n_folds=10, seed=42)
    for fold_idx in range(n_folds):
        for p in holdout_fracs:
            train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, seed=42)

Matched (paired) design with repeated CV:
    from cv_split import (cv_splits, eligible_mask, target_mask,
                          RANDOM_SEED_OFFSET)

    for s in cv_splits(M, n_folds=10, n_repeats=5, holdout_frac=p, seed=42):
        O_train = O[s.train_idx]
        elig = eligible_mask(O_train)          # shared by every method
        target = target_mask(O_train)          # shared by every scorer
        rand_seed = s.seed(SEED + RANDOM_SEED_OFFSET)
        ...

Repetition 0 of cv_splits is exactly cv_folds(M, n_folds, seed) with
cv_train_val(..., seed) subsampling, so rep-0 results reproduce the
single-repetition runs.  Every method evaluated on a split sees the same
train/validation indices, the same eligible set, the same imputation
targets and the same seeds.  SEED_TAGS lists the seed-tag offset of
every random stream in code/.

Protocol rules shared by every script (2026-10-18):
  eligibility  a benchmark is selectable in a split only if at least
               MIN_TRAIN_OBS = 3 training models observed it
               (eligible_mask);
  targets (D1) a benchmark observed by fewer than MIN_TARGET_OBS = 2
               training models is not an imputation target: no scorer
               scores its validation cells (target_mask).  With 0 or 1
               training observation it has no training variance; the
               correlation estimate carries no information about it
               (max |R_ij|, i a non-target, j != i, over the 50 matched
               splits of every data/ matrix at holdout 0.1 and 0.9:
               2.0e-11, BenchPress at 0.1; EM estimate of
               eval_entropy_vs_mi.estimate_corr; MMLU, MTEB, MTEB-2026,
               OLL v1 / v2, HELM Lite / MMLU have none), so it is left
               in the objectives (MI, trace, facility location, hub,
               PCA) and only its cells leave the score.  Non-targets
               are never eligible (MIN_TARGET_OBS <= MIN_TRAIN_OBS);
  clipping (D2) every imputer's predictions are clipped to the
               benchmark's valid score range (score_bounds.py).
"""

from typing import Iterator, NamedTuple, Sequence, Union

import numpy as np
from numpy.typing import NDArray

# A benchmark may be selected only if at least this many TRAINING models
# observed it; with fewer, its estimated correlations rest on one or two
# models (exchangeable columns, e.g. merged's mt_bench/*).  Ineligible
# benchmarks stay in the covariance and remain imputation targets if they
# meet MIN_TARGET_OBS.
MIN_TRAIN_OBS = 3

# Protocol rule D1: a benchmark is an imputation target in a split only if
# at least this many TRAINING models observed it (target_mask); the cells
# of the others are never scored, by any scorer.  A column with 0 or 1
# training observation has no training variance (its standardization is
# the SD floor and its R entries are 0 up to rounding), so its cells only
# added training-mean predictions of floored, +-10-clipped z-scores to the
# pooled sums.  Test-only override: setting it to 0 makes every column a
# target (the pre-D1 cells); scripts read it through target_mask at call
# time.
MIN_TARGET_OBS = 2
assert MIN_TARGET_OBS <= MIN_TRAIN_OBS   # non-targets are never eligible

# Repetitions of n-fold CV in the matched design (50 splits for 10 folds).
N_REPEATS = 5

# Offsets added to the base seed to form the seed tag of each stream of
# common random numbers (split_seed): the random baseline's permutation.
RANDOM_SEED_OFFSET = 7
# Retired (protocol rule D3, 2026-10-18): the stream of entropy_rf's random
# first pivot.  entropy_rf is no longer run; the tag stays reserved so that
# no new stream reuses it, and no other stream depended on it.
ENTROPY_RF_SEED_OFFSET = 11

# Registry of every seed-tag offset in code/ (documentation; each script
# defines its own constant).  The seed of a stream is Split.seed(SEED +
# offset) unless the entry says otherwise; SEED = 42 in every script, so
# offset 7 is the tag 49 and offset 20 the tag 62.  Before adding a
# stream, grep code/ for *_SEED_OFFSET assignments and SEED-plus-number
# seeds, take an unused offset and add it here (test_eval_tabimpute checks
# that every offset found in code/ is listed; the test_* files of the
# scripts check their own offsets for collisions).  Offsets 12 and 17 are
# unused.
SEED_TAGS = {
    0: "the tag SEED itself: cv_train_val's training subsample; fold "
       "permutation SEED (rep 0) or (SEED, rep); eval_reviewer_checks' "
       "Random Greedy seeds (SEED, [rep,] fold, k, seed_idx) and "
       "contamination subset default_rng(SEED)",
    7: "RANDOM_SEED_OFFSET: the random baseline's permutation (every CV "
       "script; eval_family_holdout: (tag, [rep,] family))",
    11: "ENTROPY_RF_SEED_OFFSET: retired (D3, 2026-10-18); was "
        "entropy_rf's first pivot in every CV script (eval_family_holdout "
        "as for 7); no longer drawn, kept reserved",
    13: "eval_prospective PROSP_RANDOM_SEED_OFFSET: random's permutation, "
        "draw seed (tag, d)",
    14: "eval_prospective PROSP_ENTROPY_RF_SEED_OFFSET: retired (D3); was "
        "entropy_rf's first pivot, draw seed (tag, d); kept reserved",
    15: "eval_prospective CONTROL_SPLIT_SEED_OFFSET: control split d's row "
        "permutation, (tag, d)",
    16: "eval_prospective BOOT_SEED_OFFSET: bootstrap resamples, "
        "default_rng(tag) per variant",
    18: "eval_item_budget ITEM_SEED_OFFSET: the item sample",
    19: "analyze_objectives RANDOM_SETS_SEED_OFFSET: random eligible "
        "k-subsets",
    20: "eval_tabimpute CONTEXT_SEED_OFFSET: TabImpute's context "
        "subsample when n_train exceeds 3,800 rows (one draw per split)",
    21: "eval_imputers INNER_SEED_OFFSET: permutation of the training "
        "models for the inner holdout",
    22: "reserved for eval_benchfamily_holdout; eval_selection_stability "
        "SWAP_SEED_OFFSET (random s-swaps) assigns it as well",
    23: "eval_selection_stability SUB_SEED_OFFSET: training subsamples",
    24: "eval_selection_stability RANDOM_SUBSETS_SEED_OFFSET: random "
        "eligible k-subsets",
    25: "eval_reviewer_checks CONTAM_DRAW_SEED_OFFSET: contamination "
        "subsets of draws d >= 1, default_rng((tag, d))",
}

Seed = Union[int, tuple]


def cv_folds(M: int, n_folds: int = 10, seed: Seed = 42) -> list[np.ndarray]:
    """Partition M items into n_folds balanced groups via random permutation.

    Returns a list of n_folds arrays of indices.  Sizes differ by at most 1.
    seed may be an int or a tuple (see fold_seed).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    return list(np.array_split(perm, n_folds))


def split_seed(tag: int, rep: int, fold: int, holdout_frac: float) -> tuple:
    """Common-random-number seed of one split for the stream `tag`.

    rep 0:  (tag, fold, int(holdout_frac * 10000))
    rep r:  (tag, r, fold, int(holdout_frac * 10000))   for r > 0

    Rep 0 reproduces the seeds of the single-repetition code: tag = SEED
    gives cv_train_val's training subsample, tag = SEED + 7 the random
    baseline's permutation.  The holdout tag is int(p * 10000) (truncated,
    not rounded), exactly as before.
    """
    p_tag = int(holdout_frac * 10000)
    if rep == 0:
        return (int(tag), int(fold), p_tag)
    return (int(tag), int(rep), int(fold), p_tag)


def fold_seed(rep: int, seed: int = 42) -> Seed:
    """Seed of the fold permutation of repetition `rep`: seed for rep 0
    (cv_folds(M, n_folds, seed)), (seed, rep) for rep > 0."""
    return int(seed) if rep == 0 else (int(seed), int(rep))


def repeat_folds(M: int, n_folds: int = 10, rep: int = 0,
                 seed: int = 42) -> list[np.ndarray]:
    """Folds of repetition `rep` (rep 0 == cv_folds(M, n_folds, seed))."""
    return cv_folds(M, n_folds, fold_seed(rep, seed))


def cv_train_val(
    folds: list[np.ndarray],
    fold_idx: int,
    holdout_frac: float,
    M: int,
    seed: int = 42,
    rep: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) for one fold and holdout fraction.

    Parameters
    ----------
    folds : list of arrays from cv_folds() / repeat_folds().
    fold_idx : which fold to hold out as the validation set.
    holdout_frac : fraction of *total* data that is NOT used for training.
        The validation set is always one fold (~1/n_folds of M).
        The training set is subsampled from the remaining folds so that
        |train| = round((1 - holdout_frac) * M), capped at the pool size.
        For holdout_frac = 1/n_folds this uses the entire pool.
    M : total number of items (for computing target train size).
    seed : base seed; combined with fold_idx and holdout_frac for
        reproducible subsampling independent of iteration order.
    rep : CV repetition (default 0).  The subsampling seed is
        split_seed(seed, rep, fold_idx, holdout_frac), which for rep 0 is
        the original (seed, fold_idx, int(holdout_frac * 10000)).
    """
    n_folds = len(folds)
    val_idx = folds[fold_idx]
    pool_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

    n_train = min(round((1 - holdout_frac) * M), len(pool_idx))

    if n_train >= len(pool_idx):
        train_idx = pool_idx
    else:
        # Deterministic sub-seed so result is independent of iteration order
        sub_seed = split_seed(seed, rep, fold_idx, holdout_frac)
        rng = np.random.default_rng(sub_seed)
        train_idx = rng.permutation(pool_idx)[:n_train]

    return train_idx, val_idx


class Split(NamedTuple):
    """One matched train/validation split of the repeated-CV design.

    split_id = rep * n_folds + fold numbers the splits 0 .. n_repeats·n_folds−1
    consistently across methods and scripts (pair results on it).
    """
    rep: int
    fold: int
    split_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    holdout_frac: float

    def seed(self, tag: int) -> tuple:
        """Common-random-number seed of this split for stream `tag`
        (split_seed(tag, rep, fold, holdout_frac)); e.g. tag = SEED + 7
        for the random baseline (SEED + 11, entropy_rf, is retired)."""
        return split_seed(tag, self.rep, self.fold, self.holdout_frac)


def cv_splits(
    M: int,
    n_folds: int = 10,
    n_repeats: int = N_REPEATS,
    holdout_frac: float = 0.1,
    seed: int = 42,
    reps: Union[Sequence[int], None] = None,
) -> Iterator[Split]:
    """Iterate over the n_repeats × n_folds matched splits of M items.

    Repetition r partitions the items with repeat_folds(M, n_folds, r, seed)
    (permutation seed `seed` for r = 0, (seed, r) for r > 0) and subsamples
    the training pool with cv_train_val(..., seed, rep=r).  Repetition 0
    is identical to cv_folds(M, n_folds, seed) + cv_train_val(..., seed).
    Splits are yielded rep-major, folds in order.  `reps` restricts the
    iteration to the given repetitions (e.g. [0] for a rep-0-only GPU run);
    split_id is unaffected.
    """
    rep_list = range(n_repeats) if reps is None else [int(r) for r in reps]
    for rep in rep_list:
        if not 0 <= rep < n_repeats:
            raise ValueError(f"rep {rep} outside 0..{n_repeats - 1}")
        folds = repeat_folds(M, n_folds, rep, seed)
        for fold in range(n_folds):
            train_idx, val_idx = cv_train_val(folds, fold, holdout_frac, M,
                                              seed, rep=rep)
            yield Split(rep, fold, rep * n_folds + fold, train_idx, val_idx,
                        holdout_frac)


def eligible_mask(O_train: NDArray, min_obs: int = MIN_TRAIN_OBS) -> NDArray:
    """(N,) bool mask of benchmarks selectable in a split: those observed
    for at least min_obs of the TRAINING models (rows of O_train, any
    nonzero entry counts as observed).  One mask per split, shared by every
    method."""
    O_train = np.asarray(O_train)
    return np.count_nonzero(O_train, axis=0) >= min_obs


def target_mask(O_train: NDArray, min_obs: Union[int, None] = None) -> NDArray:
    """(N,) bool mask of the imputation targets of a split (protocol rule
    D1): the benchmarks observed for at least min_obs of the TRAINING
    models (default MIN_TARGET_OBS, read at call time; any nonzero entry
    counts as observed).  The validation cells of the other benchmarks are
    never scored.  One mask per split, shared by every scorer; a column
    that is not a target is never eligible (MIN_TARGET_OBS <=
    MIN_TRAIN_OBS)."""
    if min_obs is None:
        min_obs = MIN_TARGET_OBS
    O_train = np.asarray(O_train)
    return np.count_nonzero(O_train, axis=0) >= min_obs
