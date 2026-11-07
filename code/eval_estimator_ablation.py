r"""Covariance-estimator ablation: stability of selection and imputation R^2
to the estimation choices under missing data (reviewer request B5).

Question.  The paper estimates the benchmark correlation matrix R from
the training models with eval_entropy_vs_mi.estimate_corr (Section 5.1,
Appendix app:em-details).  Reviewers asked how stable the results are to
the covariance-estimation choices under missing data (QSWq) and to the
shrinkage (PAT).  This script reruns selection and imputation on the
matched splits with one estimation choice changed at a time and reports,
per setting, the mean R^2, the paired difference to the paper's default
and the overlap (Jaccard) of the selected sets with the default's.

The knobs of estimate_corr (eval_entropy_vs_mi.py; em_cov.em_cov)
-----------------------------------------------------------------
  estimator rule   pairwise-complete correlation (estimate_corr_pairwise)
                   if the TRAINING matrix is >= 90% observed and n_train >=
                   N, else EM (estimate_corr_em)
  EM regime        "regularized" if n_train < N or < 50% observed: PSD
                   floor eps = 1e-3 and at most 1000 iterations, else eps
                   = 1e-6 and at most 500; tolerance 5e-4 on the relative
                   Frobenius change of Sigma (em_cov stops at the first
                   iteration t > 1 below it)
  shrinkage        em_cov(shrinkage="auto") if n_train < N, i.e. linear
                   shrinkage toward the scaled identity, Sigma <- (1 -
                   alpha) Sigma + alpha tr(Sigma)/N I, with the fixed
                   intensity alpha_0 = (N - n_train)/N, applied to the
                   initial estimate and once more after the last EM
                   iteration (not at every M-step); alpha = 0 otherwise.
                   The pairwise branch has no shrinkage.
  PSD floor        eigenvalue clamp eps: of the pairwise matrix (1e-6), of
                   the EM initial estimate and of every M-step (em_cov
                   eps_psd: 1e-6 or 1e-3, see above)
  eligibility      a benchmark is selectable if >= MIN_TRAIN_OBS = 3
                   training models observe it (cv_split.eligible_mask);
                   it does not enter the estimate
Fixed (not ablated): the column standardization (training means and SDs;
the EM branch floors the SD at 0.01 (|mean| + 1)), the pairwise
denominator floor max(n_jk - 1, 1), em_cov's 1e-6 Cholesky ridge, the
final renormalization to unit diagonal, and the imputation
(impute_and_evaluate: ridge 0.01, clip +-10 of the true scores, the
split's targets of rule D1 and the valid-range clip of the predictions of
rule D2).

Settings (one factor at a time from the default; SETTINGS)
---------------------------------------------------------
  default        the paper's estimate_corr (pairwise on MMLU; unregularized
                 EM on MTEB; regularized EM on Merged (alpha_0 = 8/114 at
                 n_train = 106) and BenchPress (alpha_0 = 0, n_train > N))
  estimator      pairwise | em: the branch forced on every split, with the
                 branch's own default settings (EM regime rule as above)
  shrinkage      shrink_x{0, 0.25, 4}: alpha = min(1, c alpha_0), c in {0,
                 1/4, 4} (the default rule scaled; differs from the
                 default only where n_train < N, i.e. on Merged);
                 alpha_{0.01, 0.1, 0.3}: that fixed intensity on every
                 split and branch (EM: em_cov(shrinkage=alpha); pairwise:
                 R <- (1 - alpha) R + alpha I after the unit-diagonal
                 renormalization)
  psd_floor      psd_{1e-6, 1e-4, 1e-3, 1e-2}: eps of the branch in use,
                 replacing the regime rule (the value equal to the default
                 is an exact alias)
  em_convergence tol_5e-3: tolerance x10; tol_5e-5: tolerance / 10 with 4x
                 the iteration cap; iter_x4: 4x the iteration cap (EM
                 only; on the pairwise branch these are aliases)
  eligibility    elig_{1, 2, 5, 10}: eligible_mask(O_train, m) with the
                 default estimate (m = 3 is the default)
A setting whose resolved estimator parameters (branch, eps, alpha,
iteration cap, tolerance) equal the default's on a split reuses the
default estimate (e.g. every shrinkage-scale setting outside Merged); a
setting that differs only by a larger iteration cap reuses an estimate of
the same parameters that converged within the smaller cap (em_cov's loop
then stops at the same iteration, so the result is the same array).  Both
shortcuts are exact (test_eval_estimator_ablation.py checks them against
fresh fits).  A setting refit with other parameters can still return the
default's R bit for bit, e.g. a PSD floor below the smallest eigenvalue
of every matrix it clamps (MMLU: psd_1e-4 and psd_1e-3 leave the pairwise
R unchanged); est_same_as_default marks every split where the setting's R
equals the default's (np.array_equal; reused or refit), and such a split
is evaluated with the default estimate.  A setting is an ALIAS of the
default on a split (alias_of_default) if its R and its eligible set equal
the default's, so every value it logs equals the default's; a setting
that is an alias on every split is labelled "(= default)" in the figure
and left out of the factor counts and maxima (n_alias_cells).

Correctness of the copy.  estimate_corr takes no parameters, so
estimate_with() is a parameterized copy of estimate_corr_pairwise /
estimate_corr_em that calls em_cov with explicit arguments.  At the
default parameters it reproduces estimate_corr bit-identically:
--check-default all (the default) compares R, means and SDs of every
split with estimate_corr(B_train, O_train) by np.array_equal and aborts
on any difference (default_checked in the log); the test file checks all
four regimes (pairwise, unregularized EM, regularized EM, n_train < N).

Matched design.  The splits are cv_splits(M, folds, repeats, 0.1, SEED)
(default 5 x 10 = 50 splits; repetition 0 = the seed-42 folds).  On a
split every setting and method sees the same training / validation
models and the same common random numbers (random: random_select's
permutation seeded with Split.seed(SEED + RANDOM_SEED_OFFSET) = (49, ...),
restricted to the setting's eligible set).  No new random stream is
introduced.  The selectors are eval_baselines.run_selection (the
eligibility mask, near-tie rule and seeds of every other script), so the
default rows equal the mi / entropy / trace / random / facloc rows of
logs/baselines_{ds}.npz split by split.  Methods (METHODS, the --methods
default): mi, entropy, trace and random in paired_stats' METHOD_ORDER,
then facloc, greedy facility location on the clipped correlation matrix
(baseline_select.facility_location_select), the paper's headline method
since 2026-10-19.  facloc is appended to the method axis (ALL_METHODS
order), so the first entries of every (..., n_methods, ...) array are the
pre-facloc methods at their old positions; the printed summary and the
figure show the methods in the pair orientation (display_methods,
paired_stats.method_rank: facloc between hub and random).  It draws no
random numbers and every per-cell value is computed per method, so
adding it changed no other method's values; the counts and maxima over
several methods (the factor / dataset summaries and the method-pair
totals) keep the methods they had before (all but APART_METHODS), and
facloc's cells and pairs get the same summaries of their own (macro
piece Facloc).  Each
setting's own R drives the selection and the imputation weights; the
metric is always computed in the DEFAULT setting's standardization
(training means and SDs), so every
setting is scored on the same splits, the same validation models and in
the same units.  The scored CELLS are those of impute_and_evaluate (the
convention of every other script): the observed benchmarks of a
validation model outside the selection, for the validation models that
observe at least one selected benchmark.  They depend on the selection,
so two settings with different selections are scored on different cells
(the k selected columns drop out, and on sparse data so do the models
that observe none of the picks); cover logs the scored share of the
validation cells (eval_baselines.evaluated_share).  The paired
differences are over splits, as everywhere.  (The forced pairwise branch
estimates its R from its own standardization, which differs from the EM
branch's only for columns whose SD is below the EM floor, e.g. one MTEB
column; the renormalization to unit diagonal makes R nearly invariant to
it.)

Metric.  Pooled R^2 of eval_entropy_vs_mi.impute_and_evaluate (imported,
not copied), at k = 1..--kmax; reported at --ks.  R^2 is NaN on a split
where no validation model observes both a selected and an unselected
benchmark (cover 0: a selection failure).  Its rate depends on the
setting at small k on sparse data (Merged and BenchPress at k = 1: up to
36 / 40 of 50 splits for MI under shrink_x0 / pairwise), and
paired_stats drops such splits, so a cell's paired difference can rest on
far fewer than n_splits pairs (n_pairs in the tables and macros; a
setting whose picks no validation model observes can look like an
improvement on the remaining splits).  A cell with n_pairs < MIN_PAIR_FRAC
(0.9) x n_splits is LOW-N: it keeps its own numbers but is left out of
the factor maxima, the significance counts and the method-pair counts
(n_low_n_cells, low_n_cells, EstNLowN).  A setting changes R^2
in two ways: through the selection and through the imputation weights
(e.g. shrinkage scales the cross-correlations the conditional mean uses,
which lowers R^2 even where the selected set is unchanged).  r2_selonly
separates them: the setting's selection imputed with the DEFAULT estimate,
so r2_selonly - r2_default is the selection effect alone (0 wherever the
set equals the default's) and r2 - r2_selonly the imputation-weight
effect.  Jaccard |A_s ∩ A_0| / |A_s ∪ A_0| of a setting's first k picks
with the default's (same split, method and k).  NaN where fewer than k
benchmarks are eligible or the estimate failed.

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir())
-------------------------------------------------------------------------
  LOGS/estimator_ablation_{ds}.npz
    r2, rmse, jaccard   (n_splits, n_settings, n_methods, k_max) float
    r2_selonly          (n_splits, n_settings, n_methods, k_max) R^2 of the
                        setting's selection imputed with the default
                        estimate (= r2 where the estimate is the default)
    cover               (n_splits, n_settings, n_methods, k_max) share of
                        the validation cells that r2 scores
                        (eval_baselines.evaluated_share; NaN without a
                        size-k selection)
    selected            (n_splits, n_settings, n_methods, k_max) int, picks
                        in order, -1 padded
    index_ties          (n_splits, n_settings, n_methods, k_max) int
                        (greedy_select.index_tie_count)
    {setting}__{method}_r2   (n_splits, k_max) the same R^2, one array per
                        setting and method (paired_stats.suffix_table
                        layout, methods "{setting}__{method}")
    est_branch_em, est_failed, est_reused, em_converged   (n_splits,
                        n_settings) bool
    est_same_as_default (n_splits, n_settings) bool, R equals the
                        default's bit for bit (reused or refit)
    alias_of_default    (n_splits, n_settings) bool, R and eligible set
                        equal the default's (every logged value equal)
    est_alpha, est_eps, est_tol (n_splits, n_settings) float; est_max_iter,
    em_n_iter, n_eligible_setting   (n_splits, n_settings) int (0: pairwise)
    r_rel_diff, r_max_abs_diff, r_lam_min   (n_splits, n_settings) ||R -
                        R_0||_F / ||R_0||_F, max |R - R_0|, smallest
                        eigenvalue of R
    rep, fold, split_id, n_train, n_val, n_eligible   (n_splits,) int
                        (n_eligible: default m)
    eligible            (n_splits, N) bool, default eligible set
    estimation_failed, default_checked   (n_splits,) bool
    settings, setting_factor, setting_label, setting_macro   (n_settings,)
    methods, ks_report, benchmark_names; k_max, holdout_frac, n_folds,
    n_repeats, seed, min_train_obs, min_target_obs, pred_clip,
    random_seed_offset   scalars
  LOGS/estimator_ablation_{ds}.json   config (settings and their
                        overrides), estimator diagnostics per setting,
                        summary tables (per setting x method x k: mean
                        R^2, NaN rate, paired difference to the default
                        and its n_pairs, Jaccard, cover, alias / low-n
                        flags; per factor and per dataset: largest
                        |difference|, smallest Jaccard, significant,
                        alias and low-n cells; method pairs under every
                        setting with their status against the default),
                        runtime, a flat "macros" dict and "macros_core",
                        the subset without per-cell values (summary
                        counts and maxima, default R^2, diagnostics; the
                        one to ingest if not all are wanted)
  LOGS/estimator_ablation_{ds}_paired.csv   columns PAIRED_FIELDS:
                        comparison "vs_default" (r2 of the setting - r2 of
                        the default; method_a = "{setting}:{method}",
                        method_b = "default:{method}"),
                        "selection_vs_default" (r2_selonly of the setting -
                        r2 of the default, same names) and "methods" (the
                        oriented pairs of METHODS under one setting), at
                        --ks, then paired_stats.STAT_FIELDS
  OUT/estimator_ablation_{ds}.pdf   per method (columns): paired R^2
                        difference to the default (mean, Nadeau-Bengio 95%
                        interval), the selection-only difference, and the
                        mean Jaccard (rows), one line per setting, one
                        marker per k (open markers: low-n cells)

Significance, cells and method pairs.  A (setting, method, k) cell is
significant if the NB 95% interval of its paired difference to the
default excludes 0.  The factor and dataset counts and maxima run over the
non-alias, non-low-n cells (aliases have difference 0 and Jaccard 1
exactly; a factor whose cells are all aliases gets max |difference| 0 and
min Jaccard 1, one whose non-alias cells are all low-n gets NaN and no
EstMaxAbsDiff macro, e.g. BenchPress estimator at k = 1; the minimum
Jaccard runs over the low-n cells too).  Method pairs (paired_stats.orient
order; low-n if either the setting's or the default's pair has fewer than
MIN_PAIR_FRAC x n_splits pairs) are compared under
every setting; against the default's pair, a non-alias, non-low-n setting
is "reversed" (both intervals exclude 0, with opposite signs), "lost" (the
default's interval excludes 0, the setting's does not), "gained" (the
setting's excludes 0, the default's does not) or "same".  Only "reversed"
counts as a reversal.

Macros ("macros" in the JSON; letters only; {Ds} = make_numbers.camel of
the dataset, {S} = the setting's macro piece (Default, Pairwise, Em,
ShrinkZero, ShrinkQuarter, ShrinkFour, AlphaPointZeroOne, AlphaPointOne,
AlphaPointThree, PsdMinusSix, PsdMinusFour, PsdMinusThree, PsdMinusTwo,
TolLoose, TolTight, IterFour, EligOne, EligTwo, EligFive, EligTen), {M} =
camel of the method, {F} = camel of the factor, K{k} = "K" +
make_numbers.int_words(k); [core] = also in "macros_core"):
  EstRsq{Ds}{S}{M}K{k}                 mean R^2 over the splits where it
                                       is finite ([core] for S = Default)
  EstNRsq{Ds}{S}{M}K{k}                number of those splits
  EstDiff{Ds}{S}{M}K{k}, EstDiffLo..., EstDiffHi...   paired difference
                                       setting - default, NB 95% interval
  EstNPairs{Ds}{S}{M}K{k}              number of pairs (splits where both
                                       are finite) behind EstDiff
  EstSelDiff{Ds}{S}{M}K{k}, EstSelDiffLo..., EstSelDiffHi...   the same
                                       for r2_selonly - default (selection
                                       effect alone)
  EstJacc{Ds}{S}{M}K{k}                mean Jaccard with the default's set
  [core] EstMaxAbsDiff{Ds}{F}K{k}, EstMinJacc{Ds}{F}K{k}   over the
                                       factor's cells (see above)
  [core] EstNSig{Ds}{F}K{k}, EstNCells{Ds}{F}K{k}, EstNLowN{Ds}{F}K{k}
                                       significant / counted / low-n
                                       cells; EstNSig{Ds}{F},
                                       EstNCells{Ds}{F}, EstNLowN{Ds}{F}
                                       summed over the reported k
  [core] EstNAliasSettings{Ds}{F}      the factor's settings that are
                                       aliases on every split
  [core] EstMaxAbsDiff{Ds}K{k}, EstNSig{Ds}K{k}, EstNCells{Ds}K{k},
  EstNLowN{Ds}K{k}; EstNSig{Ds}, EstNCells{Ds}, EstNLowN{Ds}   the same
                                       over all factors
  EstPair{Ds}{S}{A}Vs{B}K{k}, EstPairLo..., EstPairHi...   method pairs
                                       (orient order) under each setting
  [core] EstPairDefaultSig{Ds}{A}Vs{B}K{k}   +1 / -1 / 0: the default's
                                       interval lies above / below 0 /
                                       contains it
  [core] EstPairReversals{Ds}{A}Vs{B}K{k}, EstPairLostSig...,
  EstPairGainedSig...                  settings with that status (see
                                       above)
  [core] EstPairNSigDefault{Ds}, EstPairReversals{Ds},
  EstPairLostSig{Ds}, EstPairGainedSig{Ds}   summed over pairs and k
  [core] the counts and maxima above over facloc's cells / the pairs
  with facloc instead (APART_METHODS): EstMaxAbsDiff{Ds}{F}FaclocK{k},
  EstMinJacc{Ds}{F}FaclocK{k}, EstNSig{Ds}{F}FaclocK{k},
  EstNCells{Ds}{F}FaclocK{k}, EstNLowN{Ds}{F}FaclocK{k},
  EstNSig{Ds}{F}Facloc, EstNCells{Ds}{F}Facloc, EstNLowN{Ds}{F}Facloc,
  EstMaxAbsDiff{Ds}FaclocK{k}, EstNSig{Ds}FaclocK{k},
  EstNCells{Ds}FaclocK{k}, EstNLowN{Ds}FaclocK{k}, EstNSig{Ds}Facloc,
  EstNCells{Ds}Facloc, EstNLowN{Ds}Facloc, EstPairNSigDefault{Ds}Facloc,
  EstPairReversals{Ds}Facloc, EstPairLostSig{Ds}Facloc,
  EstPairGainedSig{Ds}Facloc (JSON summary: factors_facloc,
  dataset_facloc); the ones without the suffix run over the other methods
  [core] EstEmIter{Ds}{S}, EstEmNotConv{Ds}{S}, EstRelDiffR{Ds}{S},
  EstAlpha{Ds}{S}, EstNEligible{Ds}{S}, EstSameAsDefault{Ds}{S},
  EstAliasSplits{Ds}{S}                estimator diagnostics (means over
                                       splits; NotConv / SameAsDefault /
                                       AliasSplits: numbers of splits)
  [core] EstNSplits{Ds}, EstNReps{Ds}, EstDefaultChecked{Ds},
  EstDefaultEm{Ds}

Usage:
    python eval_estimator_ablation.py [--datasets mmlu mteb merged
                                       benchpress] [--repeats 5]
                                      [--folds 10] [--kmax 10]
                                      [--ks 1 3 5 10] [--methods mi
                                       entropy trace random facloc]
                                      [--settings NAME ...] [--jobs 1]
                                      [--check-default all|rep0|none]
                                      [--from-logs]

--settings restricts the run to the named settings (default is always
run); --jobs N evaluates the splits in N worker processes (each
single-threaded; results are identical to --jobs 1); --from-logs
recomputes the summaries, macros, paired CSV and figure from an existing
npz log without re-estimating (the JSON keeps its config and runtime).

Runtime (--repeats 5, all settings, AMD EPYC 7502 hosts, BLAS single-
threaded; the JSON records "runtime_sec" (wall) and "split_time_sum_sec"
(sum of the per-split single-threaded times)): MMLU 96-140 s and MTEB
987-1,133 s wall with --jobs 1 (peak RSS 0.41 / 0.17 GB); Merged
19,300-20,600 s and BenchPress 12,800-12,900 s summed over splits, i.e.
1,140-1,250 s and 760-785 s wall with --jobs 20 on a 64-core host shared
with other jobs (ranges over two runs on 2026-10-18; the per-split times
are inflated by the contention; ~170 MB per process).  EM fits dominate,
and tol_5e-5 (cap 4,000 iterations, reached on 43/50 Merged and 48/50
BenchPress splits) is about half of the Merged / BenchPress time; use
--jobs for those two datasets.  The summaries alone (--from-logs, all
four datasets) take ~20 s.
"""

import argparse
import json
import math
import multiprocessing
import pathlib
import sys
import time
import warnings
from typing import NamedTuple, Optional

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from cv_split import (MIN_TRAIN_OBS, N_REPEATS, RANDOM_SEED_OFFSET,
                      cv_splits, eligible_mask, target_mask)
from em_cov import em_cov
from eval_entropy_vs_mi import (estimate_corr, impute_and_evaluate,
                                protocol_config)
from score_bounds import dataset_bounds
from eval_baselines import METHOD_LABEL, evaluated_share, run_selection
from greedy_select import index_tie_count, selection_at_k
from paired_stats import (STAT_FIELDS, method_rank, orient, paired_stats,
                          write_csv)
from make_numbers import camel, int_words

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 10
KS_REPORT = [1, 3, 5, 10]
N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42                       # eval_baselines.SEED (same selections)
# A cell (or method pair) with fewer pairs than this share of the splits
# is low-n: left out of the factor / dataset counts and maxima
MIN_PAIR_FRAC = 0.9

DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
# paired_stats.METHOD_ORDER order, then facloc (the paper's headline
# method since 2026-10-19), appended so that the method axis of the
# arrays keeps the earlier methods' positions; displays (print_summary,
# plot) use display_methods (facloc, unranked, between hub and random)
METHODS = ["mi", "entropy", "trace", "random", "facloc"]
ALL_METHODS = ["mi", "entropy", "trace", "pca", "hub", "random", "facloc"]
# Methods whose cells and method pairs are summarized apart (macro suffix
# camel(m), JSON summary factors_{m} / dataset_{m}): the counts and maxima
# over several methods keep the methods they had before facloc was added
APART_METHODS = ["facloc"]

# The default estimator (eval_entropy_vs_mi.estimate_corr and its
# helpers; test_eval_estimator_ablation.py checks that estimate_with()
# reproduces it bit-identically at these values)
PAIRWISE_MIN_OBS_FRAC = 0.9     # EM if obs_frac < 0.9 or n_train < N
REGULARIZE_OBS_FRAC = 0.5       # regularized EM if n_train < N or < 0.5
EPS_PAIRWISE = 1e-6             # estimate_corr_pairwise eigenvalue floor
EPS_EM = 1e-6                   # em_cov eps_psd, unregularized
EPS_EM_REG = 1e-3               # em_cov eps_psd, regularized
ITER_EM = 500                   # em_cov max_iter, unregularized
ITER_EM_REG = 1000              # em_cov max_iter, regularized
EM_TOL = 5e-4                   # em_cov tol

DISPLAY_NAME = {"mmlu": "MMLU", "mteb": "MTEB", "merged": "Merged",
                "benchpress": "BenchPress"}

FACTORS = ["default", "estimator", "shrinkage", "psd_floor",
           "em_convergence", "eligibility"]
FACTOR_LABEL = {"default": "default", "estimator": "estimator",
                "shrinkage": "shrinkage", "psd_floor": "PSD floor",
                "em_convergence": "EM convergence",
                "eligibility": "eligibility"}

PAIRED_FIELDS = (["dataset", "holdout_frac", "k", "comparison", "factor",
                  "setting", "method_a", "method_b"] + STAT_FIELDS)

# Marker and color per reported k (one sequential hue, light to dark)
K_COLORS = ["#6BAED6", "#3182BD", "#08519C", "#08306B"]
K_MARKERS = ["o", "s", "^", "D"]


def display_name(ds_name):
    return DISPLAY_NAME.get(ds_name, ds_name)


def matrix_path(ds_name):
    return DATA / f"{ds_name}.matrix.npz"


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis)


def _num(x):
    """JSON-safe float (None for NaN / inf)."""
    x = float(x)
    return x if math.isfinite(x) else None


# ── Settings ─────────────────────────────────────────────────────

class Setting(NamedTuple):
    """One estimation choice changed from the default (None = default)."""
    name: str                   # log identifier
    factor: str                 # FACTORS
    label: str                  # figure / table label
    macro: str                  # letters-only macro piece
    estimator: Optional[str] = None     # "pairwise" / "em": forced branch
    shrink_scale: Optional[float] = None  # alpha = min(1, c * alpha_0)
    alpha: Optional[float] = None       # fixed shrinkage intensity
    psd_eps: Optional[float] = None     # eigenvalue floor
    tol: Optional[float] = None         # EM tolerance
    iter_scale: Optional[int] = None    # EM iteration-cap multiplier
    min_obs: Optional[int] = None       # eligibility threshold m

    def overrides(self):
        """The non-default fields (JSON config)."""
        return {f: v for f, v in self._asdict().items()
                if f not in ("name", "factor", "label", "macro")
                and v is not None}


SETTINGS = [
    Setting("default", "default", "default (paper)", "Default"),
    Setting("pairwise", "estimator", "pairwise (forced)", "Pairwise",
            estimator="pairwise"),
    Setting("em", "estimator", "EM (forced)", "Em", estimator="em"),
    Setting("shrink_x0", "shrinkage", r"$\alpha = 0$", "ShrinkZero",
            shrink_scale=0.0),
    Setting("shrink_x0.25", "shrinkage", r"$\alpha = \alpha_0/4$",
            "ShrinkQuarter", shrink_scale=0.25),
    Setting("shrink_x4", "shrinkage", r"$\alpha = 4\alpha_0$", "ShrinkFour",
            shrink_scale=4.0),
    Setting("alpha_0.01", "shrinkage", r"$\alpha = 0.01$ fixed",
            "AlphaPointZeroOne", alpha=0.01),
    Setting("alpha_0.1", "shrinkage", r"$\alpha = 0.1$ fixed",
            "AlphaPointOne", alpha=0.1),
    Setting("alpha_0.3", "shrinkage", r"$\alpha = 0.3$ fixed",
            "AlphaPointThree", alpha=0.3),
    Setting("psd_1e-6", "psd_floor", r"$\varepsilon = 10^{-6}$",
            "PsdMinusSix", psd_eps=1e-6),
    Setting("psd_1e-4", "psd_floor", r"$\varepsilon = 10^{-4}$",
            "PsdMinusFour", psd_eps=1e-4),
    Setting("psd_1e-3", "psd_floor", r"$\varepsilon = 10^{-3}$",
            "PsdMinusThree", psd_eps=1e-3),
    Setting("psd_1e-2", "psd_floor", r"$\varepsilon = 10^{-2}$",
            "PsdMinusTwo", psd_eps=1e-2),
    Setting("tol_5e-3", "em_convergence", r"tol $5\cdot 10^{-3}$",
            "TolLoose", tol=5e-3),
    Setting("tol_5e-5", "em_convergence",
            r"tol $5\cdot 10^{-5}$, $4\times$ iter.", "TolTight", tol=5e-5,
            iter_scale=4),
    Setting("iter_x4", "em_convergence", r"$4\times$ iterations", "IterFour",
            iter_scale=4),
    Setting("elig_1", "eligibility", r"$m = 1$", "EligOne", min_obs=1),
    Setting("elig_2", "eligibility", r"$m = 2$", "EligTwo", min_obs=2),
    Setting("elig_5", "eligibility", r"$m = 5$", "EligFive", min_obs=5),
    Setting("elig_10", "eligibility", r"$m = 10$", "EligTen", min_obs=10),
]
SETTING_BY_NAME = {s.name: s for s in SETTINGS}
DEFAULT = SETTINGS[0]


def select_settings(names=None):
    """SETTINGS restricted to `names` (in SETTINGS order), default first."""
    if not names:
        return list(SETTINGS)
    unknown = sorted(set(names) - set(SETTING_BY_NAME))
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(unknown)}")
    return [s for s in SETTINGS if s.name == "default" or s.name in names]


# ── Parameterized estimator ──────────────────────────────────────

class EstimatorParams(NamedTuple):
    """Concrete estimator parameters of one setting on one split."""
    branch: str                 # "pairwise" or "em"
    eps: float                  # eigenvalue / PSD floor
    alpha: float                # shrinkage intensity applied
    shrinkage: object           # em_cov's argument ("auto", 0.0 or alpha)
    max_iter: int               # EM iteration cap (0: pairwise)
    tol: float                  # EM tolerance (0.0: pairwise)
    regularized: bool           # EM regime of the default rule

    @property
    def key(self):
        """Parameters that determine the estimate (shrinkage "auto" and
        its value alpha give the same em_cov computation)."""
        return (self.branch, self.eps, self.alpha, self.max_iter, self.tol)


def default_branch(M_tr, N, obs_frac):
    """estimate_corr's rule: EM if < 90% observed or n_train < N."""
    return ("em" if (obs_frac < PAIRWISE_MIN_OBS_FRAC or M_tr < N)
            else "pairwise")


def auto_alpha(M_tr, N):
    """em_cov's shrinkage="auto" intensity (N - M)/N if M < N, else 0."""
    return max(0.0, (N - M_tr) / N) if M_tr < N else 0.0


def resolve_params(setting, M_tr, N, obs_frac):
    """EstimatorParams of `setting` for a training matrix of M_tr models,
    N benchmarks and observed fraction obs_frac (estimate_corr's rules for
    every field the setting leaves at None)."""
    branch = setting.estimator or default_branch(M_tr, N, obs_frac)
    if branch not in ("pairwise", "em"):
        raise ValueError(f"unknown estimator {branch!r}")
    if branch == "pairwise":
        eps = EPS_PAIRWISE if setting.psd_eps is None else float(
            setting.psd_eps)
        # the pairwise branch has no shrinkage (alpha_0 = 0), so only a
        # fixed intensity applies
        alpha = 0.0 if setting.alpha is None else float(setting.alpha)
        return EstimatorParams("pairwise", eps, alpha, alpha, 0, 0.0, False)
    regularized = M_tr < N or obs_frac < REGULARIZE_OBS_FRAC
    alpha0 = auto_alpha(M_tr, N)
    if setting.alpha is not None:
        alpha = float(setting.alpha)
        shrinkage = alpha
    elif setting.shrink_scale is not None:
        alpha = min(1.0, float(setting.shrink_scale) * alpha0)
        shrinkage = alpha
    else:
        alpha = alpha0
        shrinkage = "auto" if M_tr < N else 0.0     # estimate_corr_em
    eps = ((EPS_EM_REG if regularized else EPS_EM)
           if setting.psd_eps is None else float(setting.psd_eps))
    max_iter = ((ITER_EM_REG if regularized else ITER_EM)
                * (1 if setting.iter_scale is None
                   else int(setting.iter_scale)))
    tol = EM_TOL if setting.tol is None else float(setting.tol)
    return EstimatorParams("em", eps, alpha, shrinkage, int(max_iter), tol,
                           regularized)


class Estimate(NamedTuple):
    R: np.ndarray
    col_mean: np.ndarray
    col_std: np.ndarray
    params: EstimatorParams
    n_iter: int                 # EM iterations (0: pairwise)
    converged: bool             # EM converged within max_iter


def _pairwise(B_train, O_train, eps, alpha):
    """estimate_corr_pairwise with the eigenvalue floor eps and an
    optional post-hoc shrinkage toward I (alpha = 0: none)."""
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
    eigvals = np.maximum(eigvals, eps)
    R = (eigvecs * eigvals[None, :]) @ eigvecs.T
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(R), 1e-12))
    R = R * np.outer(d_inv, d_inv)
    if alpha > 0:
        R = (1.0 - alpha) * R + alpha * np.eye(R.shape[0])
    return R, col_mean, col_std


def _em(B_train, O_train, p):
    """estimate_corr_em with the EM parameters of p; returns (R,
    col_mean, col_std, n_iter, converged)."""
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std_floor = 0.01 * (np.abs(col_mean) + 1.0)
    col_std = np.sqrt(np.maximum(col_var, col_std_floor ** 2))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]

    result = em_cov(B_std, O_train, max_iter=p.max_iter, tol=p.tol,
                    shrinkage=p.shrinkage, eps_psd=p.eps, verbose=False)
    Sigma = result["Sigma"]
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    return R, col_mean, col_std, int(result["n_iter"]), bool(
        result["converged"])


def estimate_with(params, B_train, O_train):
    """Estimate of the correlation matrix with explicit parameters
    (resolve_params); at the default setting's parameters it equals
    estimate_corr(B_train, O_train) bit for bit."""
    if params.branch == "pairwise":
        R, cm, cs = _pairwise(B_train, O_train, params.eps, params.alpha)
        return Estimate(R, cm, cs, params, 0, True)
    R, cm, cs, n_iter, conv = _em(B_train, O_train, params)
    return Estimate(R, cm, cs, params, n_iter, conv)


def cached_estimate(cache, params):
    """An estimate in `cache` (list of Estimate) that equals the fit with
    `params` exactly, or None: one with the same key, or (EM) one with the
    same eps, alpha and tolerance that converged within params.max_iter
    iterations (em_cov's loop stops at the same iteration for any cap >=
    n_iter)."""
    for est in cache:
        if est.params.key == params.key:
            return est
    if params.branch != "em":
        return None
    for est in cache:
        q = est.params
        if (q.branch == "em" and (q.eps, q.alpha, q.tol)
                == (params.eps, params.alpha, params.tol)
                and est.converged and est.n_iter <= params.max_iter):
            return est
    return None


def jaccard(a, b):
    """|a ∩ b| / |a ∪ b| of two index collections (NaN if either is
    None)."""
    if a is None or b is None:
        return np.nan
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── Per-split evaluation ─────────────────────────────────────────

SPLIT_SET_KEYS = {               # per-setting arrays: key -> (dtype, fill)
    "est_branch_em": (bool, False), "est_failed": (bool, False),
    "est_same_as_default": (bool, False), "est_reused": (bool, False),
    "alias_of_default": (bool, False),
    "em_converged": (bool, False), "est_alpha": (float, np.nan),
    "est_eps": (float, np.nan), "est_tol": (float, np.nan),
    "est_max_iter": (np.int64, 0), "em_n_iter": (np.int64, 0),
    "n_eligible_setting": (np.int64, 0), "r_rel_diff": (float, np.nan),
    "r_max_abs_diff": (float, np.nan), "r_lam_min": (float, np.nan),
}
SPLIT_SMK_KEYS = {               # (settings, methods, k) arrays
    "r2": (float, np.nan), "rmse": (float, np.nan),
    "r2_selonly": (float, np.nan), "cover": (float, np.nan),
    "jaccard": (float, np.nan), "selected": (np.int64, -1),
    "index_ties": (np.int64, 0),
}


def evaluate_split(B, O, benchmark_names, split, settings, methods, k_max,
                   check_default=True, bounds=None):
    """All settings and methods on one split.  Returns a dict of the
    split's arrays (SPLIT_SET_KEYS: (n_settings,), SPLIT_SMK_KEYS:
    (n_settings, n_methods, k_max)) and scalars (n_train, n_val,
    n_eligible, eligible, estimation_failed, default_checked, n_fits,
    seconds).  Only the training rows enter the estimates, the
    eligibility masks and the selections.  Every setting is scored on the
    split's targets (target_mask of the training rows, rule D1; they do
    not depend on the setting) with predictions clipped to `bounds`
    (rule D2; None: no clip)."""
    t0 = time.time()
    n_set, n_m = len(settings), len(methods)
    assert settings[0].name == "default"
    N = B.shape[1]
    B_train, O_train = B[split.train_idx], O[split.train_idx]
    B_test, O_test = B[split.val_idx], O[split.val_idx]
    M_tr = len(split.train_idx)
    obs_frac = O_train.sum() / (M_tr * N)

    out = {key: np.full(n_set, fill, dtype=dt)
           for key, (dt, fill) in SPLIT_SET_KEYS.items()}
    out.update({key: np.full((n_set, n_m, k_max), fill, dtype=dt)
                for key, (dt, fill) in SPLIT_SMK_KEYS.items()})
    eligible0 = eligible_mask(O_train)
    target = target_mask(O_train)            # D1, every setting and method
    out.update(n_train=M_tr, n_val=len(split.val_idx),
               n_eligible=int(eligible0.sum()), eligible=eligible0,
               estimation_failed=False, default_checked=False, n_fits=0)

    params = [resolve_params(s, M_tr, N, obs_frac) for s in settings]
    for i, (s, p) in enumerate(zip(settings, params)):
        if s.min_obs is not None:       # eligibility: default estimate
            p = params[0]
            params[i] = p
        out["est_branch_em"][i] = p.branch == "em"
        out["est_alpha"][i] = p.alpha
        out["est_eps"][i] = p.eps
        out["est_tol"][i] = p.tol
        out["est_max_iter"][i] = p.max_iter

    # Default estimate (and its proof against estimate_corr)
    try:
        est0 = estimate_with(params[0], B_train, O_train)
    except Exception as e:                          # noqa: BLE001
        print(f" — default estimation failed: {e}", flush=True)
        out["estimation_failed"] = True
        out["est_failed"][:] = True
        out["seconds"] = time.time() - t0
        return out
    out["n_fits"] = 1
    if check_default:
        R_ref, cm_ref, cs_ref = estimate_corr(B_train, O_train)
        same = (np.array_equal(est0.R, R_ref)
                and np.array_equal(est0.col_mean, cm_ref)
                and np.array_equal(est0.col_std, cs_ref))
        if not same:
            raise RuntimeError(
                f"split {split.split_id}: estimate_with(default) differs "
                f"from estimate_corr (max |dR| = "
                f"{np.max(np.abs(est0.R - R_ref)):.3e})")
        out["default_checked"] = True
    col_mean0, col_std0 = est0.col_mean, est0.col_std

    # Estimates of the settings (exact reuse where possible)
    cache = [est0]
    ests = [est0]
    for i in range(1, n_set):
        s, p = settings[i], params[i]
        if s.min_obs is not None:
            ests.append(est0)
            out["est_reused"][i] = True
            continue
        est = cached_estimate(cache, p)
        if est is not None:
            out["est_reused"][i] = True
        else:
            try:
                est = estimate_with(p, B_train, O_train)
            except Exception as e:                  # noqa: BLE001
                print(f" — {s.name}: estimation failed: {e}", flush=True)
                out["est_failed"][i] = True
                ests.append(None)
                continue
            out["n_fits"] += 1
            cache.append(est)
        ests.append(est)

    # R diagnostics, once per distinct estimate.  A refit whose R equals
    # the default's bit for bit (e.g. a PSD floor that never binds) is the
    # default estimate for every later use, like a reused one.
    diag = {}
    same_R = {}
    for i, est in enumerate(ests):
        if est is None:
            continue
        if id(est) not in same_R:
            same_R[id(est)] = est is est0 or np.array_equal(est.R, est0.R)
        out["est_same_as_default"][i] = same_R[id(est)]
        out["em_n_iter"][i] = est.n_iter
        out["em_converged"][i] = est.params.branch == "em" and est.converged
        if id(est) not in diag:
            dR = est.R - est0.R
            diag[id(est)] = (
                float(np.linalg.norm(dR) / max(np.linalg.norm(est0.R),
                                               1e-300)),
                float(np.max(np.abs(dR))),
                float(np.linalg.eigvalsh(est.R)[0]))
        (out["r_rel_diff"][i], out["r_max_abs_diff"][i],
         out["r_lam_min"][i]) = diag[id(est)]

    # Selection and imputation (cached per estimate, eligible set, method)
    evals = {}
    for i, (s, est) in enumerate(zip(settings, ests)):
        eligible = (eligible0 if s.min_obs is None
                    else eligible_mask(O_train, s.min_obs))
        out["n_eligible_setting"][i] = int(eligible.sum())
        if est is None:
            continue
        if out["est_same_as_default"][i]:
            est = est0                  # the same R: the same selections
            out["alias_of_default"][i] = bool(np.array_equal(eligible,
                                                             eligible0))
        for j, method in enumerate(methods):
            key = (id(est), eligible.tobytes(), method)
            if key not in evals:
                result = run_selection(method, est.R, k_max, benchmark_names,
                                       split, eligible)
                r2 = np.full(k_max, np.nan)
                rmse = np.full(k_max, np.nan)
                r2_sel = np.full(k_max, np.nan)
                cover = np.full(k_max, np.nan)
                ties = np.zeros(k_max, dtype=np.int64)
                for k in range(1, k_max + 1):
                    sel = selection_at_k(result, k)
                    ties[k - 1] = index_tie_count(result, k)
                    if sel is None:
                        continue
                    imp = impute_and_evaluate(B_test, O_test, sel, est.R,
                                              col_mean0, col_std0, N,
                                              target=target, bounds=bounds)
                    r2[k - 1], rmse[k - 1] = imp["r2"], imp["rmse"]
                    cover[k - 1] = evaluated_share(O_test, sel, target)
                    # the same selection imputed with the default estimate
                    r2_sel[k - 1] = (imp["r2"] if est is est0 else
                                     impute_and_evaluate(
                                         B_test, O_test, sel, est0.R,
                                         col_mean0, col_std0, N,
                                         target=target,
                                         bounds=bounds)["r2"])
                picks = list(result["selected"])[:k_max]
                sel_arr = np.full(k_max, -1, dtype=np.int64)
                sel_arr[:len(picks)] = picks
                evals[key] = (r2, rmse, r2_sel, cover, sel_arr, ties)
            r2, rmse, r2_sel, cover, sel_arr, ties = evals[key]
            out["r2"][i, j] = r2
            out["rmse"][i, j] = rmse
            out["r2_selonly"][i, j] = r2_sel
            out["cover"][i, j] = cover
            out["selected"][i, j] = sel_arr
            out["index_ties"][i, j] = ties

    # Jaccard of each setting's first k picks with the default's
    for i in range(n_set):
        if ests[i] is None:
            continue
        for j in range(n_m):
            for k in range(1, k_max + 1):
                a = out["selected"][i, j, :k]
                b = out["selected"][0, j, :k]
                if np.any(a < 0) or np.any(b < 0):
                    continue
                out["jaccard"][i, j, k - 1] = jaccard(a, b)
    out["seconds"] = time.time() - t0
    return out


# Worker state shared with the processes of --jobs (fork)
_WORK = {}


def _work(split):
    w = _WORK
    return evaluate_split(w["B"], w["O"], w["names"], split, w["settings"],
                          w["methods"], w["k_max"],
                          w["check_all"] or (w["check_rep0"]
                                             and split.rep == 0),
                          w["bounds"])


def run_dataset(ds_name, n_repeats=N_REPEATS, n_folds=N_FOLDS, k_max=K_MAX,
                methods=METHODS, settings=None, jobs=1,
                check_default="all", ks=KS_REPORT):
    """Every setting and method on the matched splits of one dataset.
    Returns the per-split arrays stacked over the splits (rows in
    cv_splits order) plus the run's metadata, i.e. the npz content."""
    settings = list(SETTINGS) if settings is None else list(settings)
    methods = list(methods)
    print(f"\n{'#' * 70}\n# Dataset: {ds_name.upper()}\n{'#' * 70}")
    d = np.load(matrix_path(ds_name), allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    names = d["benchmark_names"]
    M, N = B.shape
    k_max = min(k_max, N)
    print(f"{M} models x {N} benchmarks, {O.mean() * 100:.1f}% observed; "
          f"{n_repeats} x {n_folds}-fold CV, {HOLDOUT_FRAC:.0%} holdout; "
          f"{len(settings)} settings x {len(methods)} methods, k <= {k_max}",
          flush=True)

    _WORK.clear()
    _WORK.update(B=B, O=O, names=names, settings=settings, methods=methods,
                 k_max=k_max, check_all=check_default == "all",
                 check_rep0=check_default == "rep0",
                 bounds=dataset_bounds(ds_name, names, DATA))
    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED))
    rows = []
    t0 = time.time()
    if jobs > 1:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(jobs) as pool:
            results = pool.imap(_work, splits)
            for s, res in zip(splits, results):
                rows.append(res)
                _print_split(s, res, settings, methods, k_max)
    else:
        for s in splits:
            res = _work(s)
            rows.append(res)
            _print_split(s, res, settings, methods, k_max)
    wall = time.time() - t0
    _WORK.clear()

    out = {}
    for key in list(SPLIT_SET_KEYS) + list(SPLIT_SMK_KEYS) + [
            "n_train", "n_val", "n_eligible", "eligible",
            "estimation_failed", "default_checked", "n_fits"]:
        out[key] = np.array([r[key] for r in rows])
    seconds = np.array([r["seconds"] for r in rows])
    out.update(rep=np.array([s.rep for s in splits]),
               fold=np.array([s.fold for s in splits]),
               split_id=np.array([s.split_id for s in splits]))
    out.update(settings=np.array([s.name for s in settings]),
               setting_factor=np.array([s.factor for s in settings]),
               setting_label=np.array([s.label for s in settings]),
               setting_macro=np.array([s.macro for s in settings]),
               methods=np.array(methods),
               ks_report=np.array(sorted({int(k) for k in ks
                                          if k <= k_max})),
               benchmark_names=names, k_max=np.int64(k_max),
               holdout_frac=np.float64(HOLDOUT_FRAC),
               n_folds=np.int64(n_folds), n_repeats=np.int64(n_repeats),
               seed=np.int64(SEED),
               **{k: (np.bool_(v) if isinstance(v, bool) else np.int64(v))
                  for k, v in protocol_config().items()},
               random_seed_offset=np.int64(RANDOM_SEED_OFFSET))
    for i, s in enumerate(settings):
        for j, m in enumerate(methods):
            out[f"{s.name}__{m}_r2"] = out["r2"][:, i, j, :]
    timing = {"wall_sec": wall, "split_time_sum_sec": float(seconds.sum()),
              "split_time_max_sec": float(seconds.max()),
              "n_fits": int(out["n_fits"].sum())}
    return out, timing


def _print_split(s, res, settings, methods, k_max):
    at = min(5, k_max) - 1
    n_alias = int(np.sum(res["alias_of_default"][1:]))
    text = "  ".join(f"{m}={res['r2'][0, j, at]:.3f}"
                     for j, m in enumerate(methods))
    print(f"  Rep {s.rep} fold {s.fold}: {res['n_train']} train / "
          f"{res['n_val']} val, {res['n_eligible']} eligible; "
          f"{res['n_fits']} fits, {n_alias} settings = default; default "
          f"R²@{at + 1} {text}  ({res['seconds']:.1f}s)", flush=True)


# ── Summaries, paired rows, macros ───────────────────────────────

def display_methods(out):
    """The methods of a log in display order (print_summary, plot): the
    pair orientation of paired_stats.method_rank, ties in the logged order
    (mi, entropy, trace, [pca, hub,] facloc, random)."""
    methods = [str(m) for m in out["methods"]]
    return sorted(methods, key=lambda m: (method_rank(m), methods.index(m)))


def _ok_rows(out):
    return ~np.asarray(out["estimation_failed"], dtype=bool)


def alias_mask(out):
    """(n_splits, n_settings) bool: on that split the setting is an alias
    of the default (R bit-identical to the default's and the same eligible
    set, so every value it logs equals the default's).  Logs written
    before alias_of_default existed get it from their diagnostics: R
    equal if est_same_as_default or max |R - R_0| = 0 (the two flag the
    same splits in new logs), and the eligible set equal if it has the
    default's size (eligible_mask(O, m) is nested in m)."""
    if "alias_of_default" in out:
        return np.asarray(out["alias_of_default"], dtype=bool)
    with np.errstate(invalid="ignore"):
        same_R = (np.asarray(out["est_same_as_default"], dtype=bool)
                  | (np.asarray(out["r_max_abs_diff"], dtype=float) == 0.0))
    same_el = (np.asarray(out["n_eligible_setting"])
               == np.asarray(out["n_eligible"])[:, None])
    return same_R & same_el & ~np.asarray(out["est_failed"], dtype=bool)


def alias_settings(out):
    """(n_settings,) bool: the setting is an alias of the default on every
    (non-failed) split; the default itself is True."""
    ok = _ok_rows(out)
    a = alias_mask(out)[ok]
    res = a.all(axis=0) if ok.any() else np.zeros(a.shape[1], dtype=bool)
    res[0] = True
    return res


def setting_ticks(out):
    """Figure / table labels of the settings, "(= default)" appended for
    the aliases on every split."""
    alias = alias_settings(out)
    return [str(lab) + (" (= default)" if i > 0 and alias[i] else "")
            for i, lab in enumerate(out["setting_label"])]


def min_pairs(n_splits):
    """Smallest number of pairs of a cell that is not low-n."""
    return MIN_PAIR_FRAC * n_splits


def pair_status(ref_sig, sig, alias=False, low_n=False):
    """Status of a method pair under a setting against the default's pair
    (signs of _sig): "alias", "low_n", "reversed" (both significant with
    opposite signs), "lost" (only the default significant), "gained" (only
    the setting significant) or "same"."""
    if alias:
        return "alias"
    if low_n:
        return "low_n"
    if ref_sig != 0 and sig == -ref_sig:
        return "reversed"
    if ref_sig != 0 and sig == 0:
        return "lost"
    if ref_sig == 0 and sig != 0:
        return "gained"
    return "same"


def _paired(out, a, b):
    ok = _ok_rows(out)
    return paired_stats(a[ok], b[ok], out["n_train"][ok], out["n_val"][ok],
                        out["rep"][ok])


def paired_rows(ds_name, out):
    """Rows over PAIRED_FIELDS: every setting vs the default (per method)
    and the oriented method pairs under every setting, at ks_report."""
    settings = [str(s) for s in out["settings"]]
    factors = [str(f) for f in out["setting_factor"]]
    methods = [str(m) for m in out["methods"]]
    rows = []
    for k in out["ks_report"]:
        k = int(k)
        for i, (s, f) in enumerate(zip(settings, factors)):
            if i > 0:
                for comp, key in (("vs_default", "r2"),
                                  ("selection_vs_default", "r2_selonly")):
                    for j, m in enumerate(methods):
                        st = _paired(out, out[key][:, i, j, k - 1],
                                     out["r2"][:, 0, j, k - 1])
                        rows.append({"dataset": ds_name,
                                     "holdout_frac": HOLDOUT_FRAC, "k": k,
                                     "comparison": comp, "factor": f,
                                     "setting": s, "method_a": f"{s}:{m}",
                                     "method_b": f"default:{m}", **st})
            for a, b in orient(methods):
                ja, jb = methods.index(a), methods.index(b)
                st = _paired(out, out["r2"][:, i, ja, k - 1],
                             out["r2"][:, i, jb, k - 1])
                rows.append({"dataset": ds_name, "holdout_frac": HOLDOUT_FRAC,
                             "k": k, "comparison": "methods", "factor": f,
                             "setting": s, "method_a": a, "method_b": b,
                             **st})
    return rows


def _sig(row):
    lo, hi = row["ci_low"], row["ci_high"]
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return 0
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def _cell_summary(cells):
    """Counts and maxima over vs-default rows `cells` (one factor or all,
    one k): aliases are left out (difference 0, Jaccard 1 exactly), low-n
    cells are counted and listed but left out of the maxima and the
    significance counts.  Without non-alias cells max_abs_diff is 0 and
    min_jaccard 1."""
    live = [r for r in cells if not r["alias"]]
    use = [r for r in live if not r["low_n"]]
    low = [r for r in live if r["low_n"]]
    finite = [r for r in use if math.isfinite(r["mean_diff"])]
    sig = [r for r in use if r["significant"] != 0]
    jacs = [r["jaccard_mean"] for r in live
            if math.isfinite(r["jaccard_mean"])]
    worst = max(finite, key=lambda r: abs(r["mean_diff"])) if finite else None
    worst_sig = max(sig, key=lambda r: abs(r["mean_diff"])) if sig else None
    low_d = [abs(r["mean_diff"]) for r in low if math.isfinite(r["mean_diff"])]
    return {
        "max_abs_diff": (abs(worst["mean_diff"]) if worst
                         else (0.0 if not live else np.nan)),
        "max_abs_diff_setting": worst["setting"] if worst else None,
        "max_abs_diff_method": worst["method"] if worst else None,
        "max_abs_diff_sig": (abs(worst_sig["mean_diff"]) if worst_sig
                             else np.nan),
        "max_abs_diff_sig_setting": (worst_sig["setting"] if worst_sig
                                     else None),
        "max_abs_diff_sig_method": (worst_sig["method"] if worst_sig
                                    else None),
        "min_jaccard": min(jacs) if jacs else (1.0 if not live else np.nan),
        "n_sig_pos": sum(1 for r in sig if r["significant"] > 0),
        "n_sig_neg": sum(1 for r in sig if r["significant"] < 0),
        "n_cells": len(use), "n_alias_cells": len(cells) - len(live),
        "n_low_n_cells": len(low),
        "max_abs_diff_low_n": max(low_d) if low_d else np.nan,
        "low_n_cells": [{"setting": r["setting"], "method": r["method"],
                         "n_pairs": r["n_pairs"], "mean_diff": r["mean_diff"],
                         "significant": r["significant"]} for r in low],
        "significant_cells": [{"setting": r["setting"], "method": r["method"],
                               "mean_diff": r["mean_diff"],
                               "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                               "sel_mean_diff": r["sel_mean_diff"]}
                              for r in sig],
    }


def summarize(ds_name, out, paired):
    """(tables, macros): JSON summary tables and the flat macro dict;
    tables["macros_core"] names the core subset ([core] in the module
    docstring; save() writes it as the JSON's "macros_core")."""
    settings = [str(s) for s in out["settings"]]
    factors = [str(f) for f in out["setting_factor"]]
    macro_of = [str(m) for m in out["setting_macro"]]
    labels = [str(x) for x in out["setting_label"]]
    methods = [str(m) for m in out["methods"]]
    ks = [int(k) for k in out["ks_report"]]
    ok = _ok_rows(out)
    n_ok = int(ok.sum())
    n_min = min_pairs(n_ok)
    alias = alias_mask(out)
    alias_set = alias_settings(out)
    D = camel(ds_name)
    mac = {}
    core = []

    def put(name, value, is_core=False):
        assert name.isalpha(), name
        v = float(value)
        if math.isfinite(v):
            mac[name] = v
            if is_core:
                core.append(name)

    by_key = {(r["comparison"], r["setting"], r["method_a"], r["method_b"],
               r["k"]): r for r in paired}
    vs_rows = []
    for i, s in enumerate(settings):
        S = macro_of[i]
        for j, m in enumerate(methods):
            M_ = camel(m)
            for k in ks:
                r2 = out["r2"][ok, i, j, k - 1]
                jac = out["jaccard"][ok, i, j, k - 1]
                n_fin = int(np.sum(np.isfinite(r2)))
                row = {"setting": s, "factor": factors[i], "label": labels[i],
                       "method": m, "k": k, "alias": bool(alias_set[i]),
                       "mean_r2": _nanmean(r2), "std_r2": _nanstd(r2),
                       "n_splits": n_fin,
                       "nan_rate": 1.0 - n_fin / n_ok if n_ok else np.nan,
                       "jaccard_mean": _nanmean(jac),
                       "jaccard_min": (np.nanmin(jac) if np.isfinite(jac).any()
                                       else np.nan),
                       "same_set_rate": (
                           float(np.mean(jac[np.isfinite(jac)] == 1.0))
                           if np.isfinite(jac).any() else np.nan),
                       "same_as_default_rate": _nanmean(
                           out["est_same_as_default"][ok, i].astype(float)),
                       "alias_rate": _nanmean(alias[ok, i].astype(float))}
                if "cover" in out:
                    row["cover_mean"] = _nanmean(out["cover"][ok, i, j, k - 1])
                Kk = f"K{int_words(k)}"
                put(f"EstRsq{D}{S}{M_}{Kk}", row["mean_r2"], i == 0)
                put(f"EstNRsq{D}{S}{M_}{Kk}", n_fin)
                if i > 0:
                    pr = by_key[("vs_default", s, f"{s}:{m}", f"default:{m}",
                                 k)]
                    row.update({f: pr[f] for f in (
                        "mean_diff", "se_nb", "ci_low", "ci_high", "p_nb",
                        "win_rate", "tie_rate", "loss_rate")})
                    row["n_pairs"] = int(pr["n_splits"])
                    row["low_n"] = bool(pr["n_splits"] < n_min)
                    n_fin0 = int(np.sum(np.isfinite(out["r2"][ok, 0, j,
                                                              k - 1])))
                    row["nan_rate_default"] = (1.0 - n_fin0 / n_ok if n_ok
                                               else np.nan)
                    row["significant"] = _sig(pr)
                    ps = by_key[("selection_vs_default", s, f"{s}:{m}",
                                 f"default:{m}", k)]
                    row.update(sel_mean_diff=ps["mean_diff"],
                               sel_ci_low=ps["ci_low"],
                               sel_ci_high=ps["ci_high"],
                               sel_significant=_sig(ps))
                    put(f"EstDiff{D}{S}{M_}{Kk}", pr["mean_diff"])
                    put(f"EstDiffLo{D}{S}{M_}{Kk}", pr["ci_low"])
                    put(f"EstDiffHi{D}{S}{M_}{Kk}", pr["ci_high"])
                    put(f"EstNPairs{D}{S}{M_}{Kk}", pr["n_splits"])
                    put(f"EstSelDiff{D}{S}{M_}{Kk}", ps["mean_diff"])
                    put(f"EstSelDiffLo{D}{S}{M_}{Kk}", ps["ci_low"])
                    put(f"EstSelDiffHi{D}{S}{M_}{Kk}", ps["ci_high"])
                    put(f"EstJacc{D}{S}{M_}{Kk}", row["jaccard_mean"])
                vs_rows.append(row)

    def put_cells(prefix, suffix, summ):
        put(f"EstMaxAbsDiff{D}{prefix}{suffix}", summ["max_abs_diff"], True)
        put(f"EstNSig{D}{prefix}{suffix}",
            summ["n_sig_pos"] + summ["n_sig_neg"], True)
        put(f"EstNCells{D}{prefix}{suffix}", summ["n_cells"], True)
        put(f"EstNLowN{D}{prefix}{suffix}", summ["n_low_n_cells"], True)

    # Cell groups of the factor / dataset summaries: the methods other
    # than APART_METHODS (macros without a method piece), then each method
    # of APART_METHODS on its own (macro piece camel(m), after the factor)
    apart = [m for m in APART_METHODS if m in methods]
    groups = [("", lambda m: m not in apart)] + [
        (camel(a), lambda m, a=a: m == a) for a in apart]
    group_rows = {}
    for G, in_group in groups:
        factor_rows = []
        all_cells = {k: [] for k in ks}
        for f in FACTORS[1:]:
            idx = [i for i, g in enumerate(factors) if g == f]
            if not idx:
                continue
            F = camel(f)
            n_sig = n_cells = n_low = 0
            for k in ks:
                cells = [r for r in vs_rows if r["factor"] == f
                         and r["k"] == k and in_group(r["method"])]
                all_cells[k].extend(cells)
                row = {"factor": f, "k": k, **_cell_summary(cells)}
                factor_rows.append(row)
                n_sig += row["n_sig_pos"] + row["n_sig_neg"]
                n_cells += row["n_cells"]
                n_low += row["n_low_n_cells"]
                Kk = f"K{int_words(k)}"
                put_cells(F + G, Kk, row)
                put(f"EstMinJacc{D}{F}{G}{Kk}", row["min_jaccard"], True)
            put(f"EstNSig{D}{F}{G}", n_sig, True)
            put(f"EstNCells{D}{F}{G}", n_cells, True)
            put(f"EstNLowN{D}{F}{G}", n_low, True)
            if not G:
                put(f"EstNAliasSettings{D}{F}",
                    int(sum(alias_set[i] for i in idx)), True)

        dataset_rows = []
        for k in ks:
            row = {"k": k, **_cell_summary(all_cells[k])}
            dataset_rows.append(row)
            put_cells(G, f"K{int_words(k)}", row)
        put(f"EstNSig{D}{G}", sum(r["n_sig_pos"] + r["n_sig_neg"]
                                  for r in dataset_rows), True)
        put(f"EstNCells{D}{G}", sum(r["n_cells"] for r in dataset_rows),
            True)
        put(f"EstNLowN{D}{G}", sum(r["n_low_n_cells"] for r in dataset_rows),
            True)
        group_rows[G] = (factor_rows, dataset_rows)

    pair_rows = []
    # Method-pair totals: the pairs without an APART_METHODS method, then
    # per such method the pairs with it (macro suffix camel(m))
    totals = {G: {"n_sig_default": 0, "reversed": 0, "lost": 0, "gained": 0}
              for G, _ in groups}
    for k in ks:
        Kk = f"K{int_words(k)}"
        for a, b in orient(methods):
            AB = f"{camel(a)}Vs{camel(b)}"
            ref = by_key[("methods", "default", a, b, k)]
            ref_sig = _sig(ref)
            ref_low = ref["n_splits"] < n_min
            lists = {key: [] for key in ("reversed", "lost", "gained",
                                         "same", "low_n", "alias")}
            for i, s in enumerate(settings):
                pr = by_key[("methods", s, a, b, k)]
                sig = _sig(pr)
                low = bool(pr["n_splits"] < n_min or ref_low)
                status = ("default" if i == 0 else
                          pair_status(ref_sig, sig, bool(alias_set[i]), low))
                if i > 0:
                    lists[status].append(s)
                pair_rows.append({"setting": s, "factor": factors[i], "k": k,
                                  "method_a": a, "method_b": b,
                                  "mean_diff": pr["mean_diff"],
                                  "ci_low": pr["ci_low"],
                                  "ci_high": pr["ci_high"],
                                  "n_splits": int(pr["n_splits"]),
                                  "significant": sig, "low_n": low,
                                  "status": status,
                                  "reversal": status == "reversed"})
                S = macro_of[i]
                put(f"EstPair{D}{S}{AB}{Kk}", pr["mean_diff"])
                put(f"EstPairLo{D}{S}{AB}{Kk}", pr["ci_low"])
                put(f"EstPairHi{D}{S}{AB}{Kk}", pr["ci_high"])
            put(f"EstPairDefaultSig{D}{AB}{Kk}", ref_sig, True)
            put(f"EstPairReversals{D}{AB}{Kk}", len(lists["reversed"]), True)
            put(f"EstPairLostSig{D}{AB}{Kk}", len(lists["lost"]), True)
            put(f"EstPairGainedSig{D}{AB}{Kk}", len(lists["gained"]), True)
            tot = totals[next((camel(m) for m in apart if m in (a, b)), "")]
            tot["n_sig_default"] += int(ref_sig != 0 and not ref_low)
            for key in ("reversed", "lost", "gained"):
                tot[key] += len(lists[key])
            pair_rows.append({"setting": "*", "k": k, "method_a": a,
                              "method_b": b,
                              "default_mean_diff": ref["mean_diff"],
                              "default_ci_low": ref["ci_low"],
                              "default_ci_high": ref["ci_high"],
                              "default_n_splits": int(ref["n_splits"]),
                              "default_significant": ref_sig,
                              "default_low_n": bool(ref_low),
                              "reversal_settings": lists["reversed"],
                              "lost_settings": lists["lost"],
                              "gained_settings": lists["gained"],
                              "low_n_settings": lists["low_n"],
                              "alias_settings": lists["alias"]})
    for G, tot in totals.items():
        put(f"EstPairNSigDefault{D}{G}", tot["n_sig_default"], True)
        put(f"EstPairReversals{D}{G}", tot["reversed"], True)
        put(f"EstPairLostSig{D}{G}", tot["lost"], True)
        put(f"EstPairGainedSig{D}{G}", tot["gained"], True)

    diag_rows = []
    for i, s in enumerate(settings):
        S = macro_of[i]
        em = out["est_branch_em"][ok, i] & ~out["est_failed"][ok, i]
        fitted = ~out["est_failed"][ok, i]
        row = {"setting": s, "factor": factors[i], "label": labels[i],
               "overrides": SETTING_BY_NAME[s].overrides()
               if s in SETTING_BY_NAME else {},
               "n_splits": n_ok,
               "n_failed": int(np.sum(out["est_failed"][ok, i])),
               "n_em": int(em.sum()),
               "n_same_as_default": int(np.sum(
                   out["est_same_as_default"][ok, i])),
               "n_alias": int(np.sum(alias[ok, i])),
               "alias": bool(alias_set[i]),
               "alpha_mean": _nanmean(out["est_alpha"][ok, i]),
               "alpha_values": sorted({float(a) for a in
                                       out["est_alpha"][ok, i]}),
               "eps_values": sorted({float(a) for a in out["est_eps"][ok, i]}),
               "max_iter_values": sorted({int(a) for a in
                                          out["est_max_iter"][ok, i]}),
               "em_iter_mean": (float(np.mean(out["em_n_iter"][ok, i][em]))
                                if em.any() else np.nan),
               "em_iter_max": (int(np.max(out["em_n_iter"][ok, i][em]))
                               if em.any() else None),
               "em_not_converged": int(np.sum(
                   em & ~out["em_converged"][ok, i])),
               "r_rel_diff_mean": _nanmean(out["r_rel_diff"][ok, i]),
               "r_rel_diff_max": (float(np.nanmax(out["r_rel_diff"][ok, i]))
                                  if fitted.any() else np.nan),
               "r_max_abs_diff_max": (
                   float(np.nanmax(out["r_max_abs_diff"][ok, i]))
                   if fitted.any() else np.nan),
               "r_lam_min_mean": _nanmean(out["r_lam_min"][ok, i]),
               "n_eligible_mean": _nanmean(
                   out["n_eligible_setting"][ok, i].astype(float))}
        diag_rows.append(row)
        if em.any():
            put(f"EstEmIter{D}{S}", row["em_iter_mean"], True)
            put(f"EstEmNotConv{D}{S}", row["em_not_converged"], True)
        put(f"EstRelDiffR{D}{S}", row["r_rel_diff_mean"], True)
        put(f"EstAlpha{D}{S}", row["alpha_mean"], True)
        put(f"EstNEligible{D}{S}", row["n_eligible_mean"], True)
        put(f"EstSameAsDefault{D}{S}", row["n_same_as_default"], True)
        put(f"EstAliasSplits{D}{S}", row["n_alias"], True)

    put(f"EstNSplits{D}", n_ok, True)
    put(f"EstNReps{D}", len(np.unique(out["rep"][ok])), True)
    put(f"EstDefaultChecked{D}", int(np.sum(out["default_checked"])), True)
    put(f"EstDefaultEm{D}", int(np.sum(out["est_branch_em"][ok, 0])), True)
    factor_rows, dataset_rows = group_rows[""]
    tables = {"min_pair_frac": MIN_PAIR_FRAC, "min_pairs": n_min,
              "vs_default": vs_rows, "factors": factor_rows,
              "dataset": dataset_rows, "method_pairs": pair_rows,
              "estimator": diag_rows, "macros_core": core}
    for a in apart:
        tables[f"factors_{a}"], tables[f"dataset_{a}"] = \
            group_rows[camel(a)]
    return tables, mac


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _num(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def _fmt(x, spec):
    return format(x, spec) if x is not None and math.isfinite(x) else "-"


def print_summary(ds_name, out, tables):
    methods = display_methods(out)
    ks = [int(k) for k in out["ks_report"]]
    n = int(_ok_rows(out).sum())
    alias_set = alias_settings(out)
    print(f"\n== {display_name(ds_name)}: R² difference setting − default "
          f"(mean over the paired splits of {n}; * = NB 95% interval "
          f"excludes 0; n = low-n cell, fewer than {tables['min_pairs']:.0f} "
          f"pairs) and Jaccard with the default's set ==")
    k_show = [k for k in ks if k in (1, 5, 10)] or ks
    rows = {(r["setting"], r["method"], r["k"]): r
            for r in tables["vs_default"]}
    for title, pre in (("", ""), ("selection only (the setting's selection "
                                  "imputed with the default estimate):",
                                  "sel_")):
        if title:
            print(f"\n{title}")
        print(f"{'setting':<14}" + "".join(f"{m + '@' + str(k):<16}"
                                           for m in methods for k in k_show))
        for i, s in enumerate(out["settings"]):
            s = str(s)
            if i > 0 and alias_set[i]:
                print(f"{s:<14}(= default on every split)")
                continue
            cells = []
            for m in methods:
                for k in k_show:
                    r = rows[(s, m, k)]
                    if s == "default":
                        cells.append(f"{r['mean_r2']:.3f}")
                        continue
                    star = "*" if r[f"{pre}significant"] else " "
                    low = "n" if r["low_n"] else " "
                    jac = (f" J{r['jaccard_mean']:.2f}"
                           if not pre and math.isfinite(r["jaccard_mean"])
                           else "")
                    d = r[f"{pre}mean_diff"]
                    cells.append(f"{d:+.3f}{star}{low}{jac}")
            print(f"{s:<14}" + "".join(f"{c:<16}" for c in cells))
    print(f"\n{'setting':<14}{'EM':>4}{'=def':>6}{'alias':>6}{'iter':>8}"
          f"{'noconv':>8}{'alpha':>8}{'|dR|/|R|':>10}{'lam_min':>10}"
          f"{'n_elig':>8}")
    for r in tables["estimator"]:
        it = r["em_iter_mean"]
        print(f"{r['setting']:<14}{r['n_em']:>4}{r['n_same_as_default']:>6}"
              f"{r['n_alias']:>6}"
              f"{(f'{it:.0f}' if math.isfinite(it) else '-'):>8}"
              f"{r['em_not_converged']:>8}{r['alpha_mean']:>8.3f}"
              f"{r['r_rel_diff_mean']:>10.2e}{r['r_lam_min_mean']:>10.2e}"
              f"{r['n_eligible_mean']:>8.1f}")
    # the methods other than APART_METHODS, then each of those on its own
    apart = [m for m in APART_METHODS if f"factors_{m}" in tables]
    groups = [(" methods other than " + ", ".join(apart),
               tables["factors"], tables["dataset"])] + [
        (f" {m} only", tables[f"factors_{m}"], tables[f"dataset_{m}"])
        for m in apart]
    rows = []
    for title, factor_rows, dataset_rows in groups:
        rows += [dict(r, group=title) for r in factor_rows]
        rows += [dict(r, group=title, factor="(all factors)")
                 for r in dataset_rows]
    print("\nlargest |mean difference| per factor (non-alias cells with "
          "enough pairs; low-n cells listed separately):")
    for i, r in enumerate(rows):
        if apart and (i == 0 or r["group"] != rows[i - 1]["group"]):
            print(f"{r['group']}:")
        sig_txt = (f"; largest significant "
                   f"{_fmt(r['max_abs_diff_sig'], '.4f')}"
                   f" ({r['max_abs_diff_sig_setting']}, "
                   f"{r['max_abs_diff_sig_method']})"
                   if r["max_abs_diff_sig_setting"] else "")
        print(f"  {r['factor']:<15} k={r['k']:<3} "
              f"{_fmt(r['max_abs_diff'], '.4f')} "
              f"({r['max_abs_diff_setting']}, {r['max_abs_diff_method']}); "
              f"min Jaccard {_fmt(r['min_jaccard'], '.2f')}; "
              f"significant +{r['n_sig_pos']} / -{r['n_sig_neg']} of "
              f"{r['n_cells']}{sig_txt}; {r['n_alias_cells']} alias, "
              f"{r['n_low_n_cells']} low-n cells")
        for c in r["low_n_cells"] if r["factor"] != "(all factors)" else []:
            print(f"      low-n: {c['setting']} {c['method']} "
                  f"{c['n_pairs']} pairs, {_fmt(c['mean_diff'], '+.4f')}")
    stars = [r for r in tables["method_pairs"] if r.get("setting") == "*"]
    n_sig = sum(1 for r in stars if r["default_significant"]
                and not r["default_low_n"])
    n_rev = sum(len(r["reversal_settings"]) for r in stars)
    print(f"\nmethod pairs: {n_sig} of {len(stars)} default differences "
          f"significant; {n_rev} (setting, pair, k) reversals")
    for r in stars:
        if not (r["reversal_settings"] or r["lost_settings"]
                or r["gained_settings"]):
            continue
        print(f"  {r['method_a']} − {r['method_b']} k={r['k']}: default "
              f"{_fmt(r['default_mean_diff'], '+.4f')} "
              f"[{_fmt(r['default_ci_low'], '+.4f')}, "
              f"{_fmt(r['default_ci_high'], '+.4f')}]"
              + (" low-n" if r["default_low_n"] else ""))
        for key in ("reversal", "lost", "gained"):
            if r[f"{key}_settings"]:
                print(f"      {key}: {', '.join(r[f'{key}_settings'])}")
    if not n_rev:
        print("  no setting reverses a significant method difference of "
              "the default")


# ── Plotting ─────────────────────────────────────────────────────

PLOT_ROWS = [  # (summary field prefix or "jaccard", x label)
    ("", r"$R^2$ − default (95% NB interval)"),
    ("sel_", r"selection only: $R^2$ − default"),
    ("jaccard", "Jaccard with default set"),
]


def plot_dataset(ds_name, out, tables, path=None):
    """Per method (columns): paired R² difference to the default (mean and
    NB 95% interval), the same for the setting's selection imputed with
    the default estimate, and the mean Jaccard with the default's set
    (rows); one line per non-default setting (grouped by factor; "(=
    default)" for aliases on every split), one marker per k (open: low-n
    cell)."""
    settings = [str(s) for s in out["settings"]][1:]
    factors = [str(f) for f in out["setting_factor"]][1:]
    labels = [str(x) for x in out["setting_label"]][1:]
    methods = display_methods(out)
    ks = [int(k) for k in out["ks_report"]]
    if not settings:
        return None
    ok = _ok_rows(out)
    rows = {(r["setting"], r["method"], r["k"]): r
            for r in tables["vs_default"]}
    ticks = setting_ticks(out)[1:]
    n_s, n_m = len(settings), len(methods)
    y = np.arange(n_s)[::-1].astype(float)
    off = (np.arange(len(ks)) - (len(ks) - 1) / 2) * 0.16
    n_rows = len(PLOT_ROWS)
    fig, axes = plt.subplots(n_rows, n_m, figsize=(3.1 * n_m + 1.8,
                                                   n_rows * (0.24 * n_s + 1.0)),
                             sharey=True, squeeze=False)
    for j, m in enumerate(methods):
        for row, (pre, xlabel) in enumerate(PLOT_ROWS):
            ax = axes[row, j]
            for t, k in enumerate(ks):
                xs, lo, hi, low = [], [], [], []
                for s in settings:
                    r = rows[(s, m, k)]
                    low.append(bool(r["low_n"]) and not r["alias"])
                    if pre == "jaccard":
                        xs.append(r["jaccard_mean"])
                    else:
                        mean = r[f"{pre}mean_diff"]
                        xs.append(mean)
                        lo.append(mean - r[f"{pre}ci_low"])
                        hi.append(r[f"{pre}ci_high"] - mean)
                xs = np.array(xs, dtype=float)
                low = np.array(low, dtype=bool)
                color = K_COLORS[t % len(K_COLORS)]
                label = f"$k = {k}$"
                for part, fill in ((~low, color), (low, "white")):
                    if not part.any():
                        continue
                    kw = dict(color=color, markerfacecolor=fill,
                              marker=K_MARKERS[t % len(K_MARKERS)],
                              markersize=4, linestyle="none", label=label)
                    label = None
                    if pre == "jaccard":
                        ax.plot(xs[part], (y + off[t])[part], **kw)
                    else:
                        ax.errorbar(xs[part], (y + off[t])[part],
                                    xerr=[np.nan_to_num(np.array(lo)[part]),
                                          np.nan_to_num(np.array(hi)[part])],
                                    elinewidth=1.0, capsize=0, **kw)
            if pre == "jaccard":
                ax.set_xlim(-0.03, 1.03)
            else:
                ax.axvline(0.0, color="#444444", linewidth=0.8)
            ax.set_xlabel(xlabel)
            if row == 0:
                ax.set_title(METHOD_LABEL.get(m, m), fontsize=10)
            for i in range(1, n_s):
                if factors[i] != factors[i - 1]:
                    ax.axhline(y[i] + 0.5, color="#BBBBBB", linewidth=0.6)
            ax.grid(True, axis="x", alpha=0.3)
            ax.set_ylim(-0.6, n_s - 0.4)
    for row in range(n_rows):
        axes[row, 0].set_yticks(y)
        axes[row, 0].set_yticklabels(ticks, fontsize=7)
    axes[0, -1].legend(fontsize=7, loc="best")
    n = int(ok.sum())
    fig.suptitle(f"{display_name(ds_name)}: covariance-estimator ablation "
                 f"({n} matched splits, one factor at a time)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    path = OUT / f"estimator_ablation_{ds_name}.pdf" if path is None else path
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)
    return path


# ── Saving ───────────────────────────────────────────────────────

def load_log(ds_name):
    """(out, config, timing) of an existing LOGS/estimator_ablation_{ds}
    log (--from-logs): the npz arrays, and the config and runtime of its
    JSON (None / NaN where the JSON is missing)."""
    path = LOGS / f"estimator_ablation_{ds_name}.npz"
    with np.load(path, allow_pickle=True) as d:     # written by save()
        out = {key: d[key] for key in d.files}
    json_path = LOGS / f"estimator_ablation_{ds_name}.json"
    doc = (json.loads(json_path.read_text()) if json_path.is_file()
           else {})

    def num(key):
        v = doc.get(key)
        return np.nan if v is None else float(v)

    timing = {"wall_sec": num("runtime_sec"),
              "split_time_sum_sec": num("split_time_sum_sec"),
              "n_fits": int(doc.get("n_fits", int(np.sum(out["n_fits"]))))}
    return out, doc.get("config"), timing


def save(ds_name, out, tables, macros, paired, config, timing,
         write_npz=True):
    path = LOGS / f"estimator_ablation_{ds_name}.npz"
    if write_npz:
        np.savez(path, **out)
        print(f"Results saved to {path}", flush=True)
    csv_path = LOGS / f"estimator_ablation_{ds_name}_paired.csv"
    write_csv(paired, csv_path, PAIRED_FIELDS)
    print(f"Paired rows saved to {csv_path}", flush=True)
    summary = {k: v for k, v in tables.items() if k != "macros_core"}
    core = {name: macros[name] for name in tables.get("macros_core", [])
            if name in macros}
    doc = {"script": "eval_estimator_ablation.py", "dataset": ds_name,
           "config": config, "summary": summary,
           "runtime_sec": round(float(timing["wall_sec"]), 1),
           "split_time_sum_sec": round(float(timing["split_time_sum_sec"]),
                                       1),
           "n_fits": int(timing["n_fits"]), "macros": macros,
           "macros_core": core}
    json_path = LOGS / f"estimator_ablation_{ds_name}.json"
    tmp = json_path.with_name(f".{json_path.name}.tmp")
    tmp.write_text(json.dumps(_jsonable(doc), indent=1, allow_nan=False)
                   + "\n")
    tmp.replace(json_path)
    print(f"Summary saved to {json_path}", flush=True)


def run_config(args, settings, methods, out):
    return {"argv": list(sys.argv), "methods": methods,
            "settings": [{"name": s.name, "factor": s.factor,
                          "label": s.label, "macro": s.macro,
                          "overrides": s.overrides()} for s in settings],
            "k_max": int(out["k_max"]),
            "ks_report": [int(k) for k in out["ks_report"]],
            "n_repeats": args.repeats, "n_folds": args.folds,
            "holdout_frac": HOLDOUT_FRAC, "seed": SEED,
            **protocol_config(),
            "random_seed_tag": SEED + RANDOM_SEED_OFFSET,
            "new_seed_tags": [],
            "default_estimator": {
                "pairwise_min_obs_frac": PAIRWISE_MIN_OBS_FRAC,
                "regularize_obs_frac": REGULARIZE_OBS_FRAC,
                "eps_pairwise": EPS_PAIRWISE, "eps_em": EPS_EM,
                "eps_em_regularized": EPS_EM_REG, "iter_em": ITER_EM,
                "iter_em_regularized": ITER_EM_REG, "em_tol": EM_TOL,
                "shrinkage": "(N - n_train)/N if n_train < N (EM), else 0"},
            "metric": "impute_and_evaluate pooled R^2 in the default "
                      "setting's standardization; imputation weights and "
                      "selection from each setting's own R; scored cells "
                      "are each selection's own (observed unselected "
                      "benchmarks of the validation models observing a "
                      "pick; cover)",
            "min_pair_frac": MIN_PAIR_FRAC,
            "alias": "R bit-identical to the default's and the same "
                     "eligible set (alias_of_default)",
            "reversal": "default and setting intervals both exclude 0 "
                        "with opposite signs",
            "check_default": args.check_default, "jobs": args.jobs}


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Covariance-estimator ablation (estimator rule, "
                    "shrinkage, PSD floor, EM convergence, eligibility) on "
                    "the matched repeated-CV splits.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names, each read from "
                             "{name}.matrix.npz in the data directory "
                             f"(default: {' '.join(DATASETS)})")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions (default %(default)s; 1 = "
                             "repetition 0 only)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--kmax", type=int, default=K_MAX)
    parser.add_argument("--ks", type=int, nargs="+", default=KS_REPORT,
                        help="k of the summaries, macros and paired rows "
                             "(default %(default)s)")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=ALL_METHODS)
    parser.add_argument("--settings", nargs="+", default=None,
                        choices=[s.name for s in SETTINGS],
                        help="run only these settings (default: all; "
                             "'default' is always run)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="worker processes over the splits (default 1; "
                             "results do not depend on it)")
    parser.add_argument("--check-default", choices=["all", "rep0", "none"],
                        default="all",
                        help="verify the default estimate against "
                             "estimate_corr bit for bit on every split "
                             "(all), on repetition 0, or not")
    parser.add_argument("--from-logs", action="store_true",
                        help="recompute summaries, macros, the paired CSV "
                             "and the figure from existing "
                             "logs/estimator_ablation_{ds}.npz (no "
                             "estimation; config and runtime are kept from "
                             "the existing JSON)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.kmax < 1:
        parser.error("--kmax must be at least 1")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if any(k < 1 or k > args.kmax for k in args.ks):
        parser.error("--ks must lie in 1..--kmax")
    if args.from_logs:
        missing = [ds for ds in args.datasets if not (
            LOGS / f"estimator_ablation_{ds}.npz").is_file()]
        if missing:
            parser.error("no log for " + ", ".join(
                str(LOGS / f"estimator_ablation_{ds}.npz") for ds in missing))
    else:
        missing = [ds for ds in args.datasets
                   if not matrix_path(ds).is_file()]
        if missing:
            parser.error("no data matrix for " + ", ".join(
                f"{ds} ({matrix_path(ds)})" for ds in missing))
    methods = [m for m in ALL_METHODS if m in args.methods]
    settings = select_settings(args.settings)

    total_t0 = time.time()
    for ds_name in args.datasets:
        if args.from_logs:
            out, config, timing = load_log(ds_name)
            config = config or {"from_logs": True}
        else:
            out, timing = run_dataset(ds_name, args.repeats, args.folds,
                                      args.kmax, methods, settings, args.jobs,
                                      args.check_default, args.ks)
            config = run_config(args, settings, methods, out)
        paired = paired_rows(ds_name, out)
        tables, macros = summarize(ds_name, out, paired)
        print_summary(ds_name, out, tables)
        plot_dataset(ds_name, out, tables)
        save(ds_name, out, tables, macros, paired, config, timing,
             write_npz=not args.from_logs)
        print(f"\n{ds_name.upper()} completed in {timing['wall_sec']:.0f}s "
              f"(sum over splits {timing['split_time_sum_sec']:.0f}s, "
              f"{timing['n_fits']} estimator fits)", flush=True)
    print(f"\nAll datasets completed in {time.time() - total_t0:.0f}s",
          flush=True)
