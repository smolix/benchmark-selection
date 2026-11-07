r"""Prospective (temporal) evaluation: do benchmarks selected on old models
predict new models?

The CV experiments hold out random models.  Here the held-out models are
the NEW ones: released or evaluated after the models the selection was
fitted on.  Every step that the CV scripts do on the training rows uses the
OLD rows only (column means and standard deviations, the correlation
matrix of eval_entropy_vs_mi.estimate_corr, the eligibility mask
cv_split.eligible_mask, every selection); the new rows are used only as
imputation targets: each new model is conditioned on its observed selected
benchmarks and its other observed benchmarks are predicted, pooled R² =
1 − SS_res / SS_tot over test models × unselected observed targets in
train-standardized units with ridge 0.01, exactly as
eval_entropy_vs_mi.impute_and_evaluate (per_model_ss reproduces it test
model by test model, which the bootstrap needs).

Settings (--datasets; one log per setting, tag in parentheses).  "New"
means new to the leaderboard: on mteb2026, helm and olv2sub some test
models were released (uploaded) before the training side ended; their
number is logged (info.test_release, macro NTestOldProsp):
  (mteb2026)  train = data/mteb.matrix.npz, the paper's MTEB snapshot (263
              models, results repository commit 37dfa3b5, 2026-03-13);
              test = the rows of data/mteb2026.matrix.npz flagged
              recommended_prospective (57 models whose 56-task results were
              added to the repository after the snapshot commit, duplicates
              of snapshot rows removed; mteb2026.meta.json
              prospective_test_set).  21 of the 57 have a release_date on or
              before 2026-03-14 (9 before 2026).  Variant "late": the 36
              with release_date > 2026-03-14 (optional_release_date_filter),
              new in time as well.  Check: the in_snapshot rows of mteb2026
              against mteb.matrix.npz (max |diff|, rows that differ).
  (helm)      helm_lite: train = the 30 models of the HELM Lite v1.0.0
              release (models.in_snapshot, the models of the paper's HELM
              columns), test = the 61 models first evaluated in releases
              v1.1.0 - v1.13.0 (2024-03-01 - 2025-01-10); 2 of them were
              released before the v1.0.0 release date 2023-12-19 (Mistral
              Medium (2312), Phi-2).  Check: the in_snapshot rows against
              data/helm.npz and the helm/* columns of data/merged.matrix.npz.
  (epoch)     Epoch AI Benchmarking Hub, rolling origin, split by time on
              both axes: for every cutoff c of --epoch-cutoffs (default the
              four of meta.prospective_split_cutoffs between 2024-07 and
              2026-01), train = models with release_date < c, test =
              release_date >= c, and only the benchmarks with release_date
              < c are kept (benchmarks without a valid date would be dropped
              too; there are none).  The columns of benchmarks released on or
              after c are removed for both sides: Epoch scored many old
              models on them after c (97 / 179 / 190 / 238 training cells at
              the four default cutoffs), and with them such benchmarks were
              eligible and picked (LMCA, released 2026-08-12, was the first
              pick of entropy, trace and hub at c = 2025-01-01).  Without
              them no score on a benchmark that did not exist at c enters
              the fit, the eligibility, the selection or the targets; the
              new models are predicted on the benchmarks of c.  Counts per
              cutoff: info.time_mask, macros NBenchFutureProsp,
              NTrainCellsFutureProsp, NTestCellsFutureProsp, NEligFutureProsp
              (the post-cutoff benchmarks that would be eligible).  Residual
              look-ahead that data/epoch.* cannot remove: (i) a kept cell of
              an old model may have been evaluated after c (Epoch re-runs
              old models; the per-run "Started at" times are in the raw CSVs
              of Epoch's own runs only and not exported), so the training
              rows are denser than at c, though on benchmarks of c only: of
              the 498 / 737 / 1209 / 1575 kept training cells at the four
              default cutoffs, 56 / 153 / 52 / 110 come from Epoch runs
              started on or after c (GPQA diamond, MATH level 5, OTIS Mock
              AIME, SWE-Bench verified, ...; raw-data count, outside this
              script); the other kept cells are external reports without
              run dates;
              (ii) Epoch deleted the scores of the superseded FrontierMath
              versions of models later scored on the successors (meta
              cells_missing_but_in_raw, reason superseded; per side in
              info.time_mask.superseded_missing_cells), so a few cells that
              existed at c are missing.
  (epoch_bymodel)  sensitivity, not run by default: epoch split by model
              release date only, every benchmark kept (the first design,
              with the look-ahead described above: post-cutoff benchmarks
              are eligible, picked and targets).
  (olv2sub)   Open LLM Leaderboard v2 leaf sub-tasks, rolling origin on
              models.submission_date (--olv2sub-cutoffs, default
              2024-10-01, 2025-01-01, 2025-03-01: the leaderboard ran from
              2024-06 to 2025-03; each cutoff leaves >= MIN_OLV2SUB_TEST =
              300 test models, 3229 / 1888 / 334 after duplicate removal).
              The 9 models without a submission date are left out.  The 7
              math_hard/* columns are dropped: their scorer version is mixed
              (models.math_hard_task_version; v3.0 = the 2025-02-13 bulk
              re-scoring, v2.0 = runs after it that were not re-scored), the
              versions shift the MATH aggregate by up to 0.71 (meta
              counts.math_hard_aggregate_shift_between_scorer_versions) and
              they are confounded with time: test sets of late cutoffs are
              mostly v2.0, training sets mostly v3.0 (per-cutoff version
              counts in the JSON log).  The four ifeval/* columns (four
              metrics of one run) and the three nested gpqa/* subsets are
              kept; they are redundant by construction for prospective and
              control splits alike.  Test models uploaded to the Hub before
              c (models.upload_to_hub_date) are counted as "old releases".
  (olv2sub_math)  sensitivity, not run by default: olv2sub with the
              math_hard/* columns (their v3.0 re-scoring of 2025-02-13 is
              itself a look-ahead for the earlier cutoffs).

Duplicates across the boundary.  make_variant drops every test row that
duplicates a training row: the same observation pattern on the variant's
benchmarks with >= MIN_TRAIN_OBS observed cells and identical values, or
(olv2sub) the same non-empty models.model_sha, i.e. the same weights
submitted again under another name.  The dropped rows are listed in
info.duplicates (macro NTestDupProsp).  On olv2sub this removes 3 / 3 / 6
test rows (e.g. microsoft/phi-4 = NyxKrage/Microsoft_Phi-4,
NikolaSigmoid/AceMath-1.5B-Instruct-1epoch with the model_sha of
nvidia/AceMath-1.5B-Instruct); none on mteb2026 (whose meta already
excludes re-uploads of snapshot rows), helm or epoch.

Methods: mi, entropy, trace, hub, pca, facloc (deterministic given the
training rows) and random (stochastic: on the prospective split the mean
over D = --repeats x 10 = 50 common-random-number draws, sd over the draws
logged; entropy_rf was dropped on 2026-10-18, protocol rule D3, and its
stream SEED + 14 is retired).  facloc is greedy facility location on the
clipped correlation matrix (baseline_select.facility_location_select), the
paper's headline method since 2026-10-19; it draws no random numbers, so
adding it changed no other method's values (random's draws are seeded per
draw, draw_seed).  Near-ties are broken as in every other script
(greedy_select).
k = 1 .. min(--kmax = 15, N); values at k > n_eligible are NaN, and so is
k = N (nothing left to predict).

Targets.  A test cell is a target if its benchmark was observed for at
least max(--target-min-obs, MIN_TARGET_OBS) training models (default
--target-min-obs = cv_split.MIN_TARGET_OBS = 2, protocol rule D1, the
target rule of every script, cv_split.target_mask): a benchmark without
training data has no training mean or standard deviation
(impute_and_evaluate would standardize it with mean 0 and the std floor
0.01, i.e. clip it to +-10).  On mteb2026, helm and olv2sub every test
benchmark has enough training data, so the restriction is void; on epoch
it removes the targets of the pre-cutoff benchmarks with fewer than 2
training models (epoch_bymodel: also those of the post-cutoff benchmarks;
counts in the log).  {m}_r2_all is always logged: R² over the targets of
impute_and_evaluate (MIN_TARGET_OBS; equal to {m}_r2 at the default
--target-min-obs).  Until 2026-10-18 the default was MIN_TRAIN_OBS = 3
and r2_all scored every observed unselected cell; --target-min-obs 3
reproduces that r2.  Every prediction is clipped to its benchmark's valid
score range (score_bounds.dataset_bounds of the setting's data, rule D2)
before it is scored.

Coverage.  A test model contributes to a method's pooled R² at k only if
it observes one of the k picks and a target ({m}_n_models), so each
method's R² covers its own subset of the test models.  The new models of
mteb2026 are mostly partial submissions (median 25 of 56 tasks): at k = 1
only 9 of 57 contribute (1 of 36 in "late").  The random-split control
matches the size of the test set, not its observation density: its test
rows are mostly well-covered old models (prosp_test_fill vs test_fill,
macros FillTestProsp / FillTestCtrl; NModelsProsp vs NModelsCtrl), and
where coverage differs the penalty mixes coverage with temporal drift.
A statistic that rests on fewer than --min-models (default MIN_MODELS =
10) contributing test models stays in the npz / JSON (flag low_coverage)
but gets no macro (the JSON lists the names, macros_suppressed): the
prospective R² / RMSE macros of a method at k need >= min_models
prospective contributors (stochastic methods: the mean over the draws),
the control ones >= min_models on average over the control splits, a
penalty both, and a bootstrap (control) pair both methods' prospective
(control) counts.

Shift-adjusted R².  The pooled R² measures SS_tot about the TRAINING
mean (0 in train-standardized units).  New models that are shifted as a
group (on Epoch they score several training SDs above the old models)
inflate SS_tot, so a regression that merely tracks the shift gets a high
R² while its errors are larger than on a random split.  {m}_r2_c is the
same pooled fit with SS_tot about each target column's mean over the
observed cells of the evaluation (test) set: 1 − SS_res / Σ (z −
z̄_test)², summed over the targets in columns with >= MIN_CENTER_OBS = 2
observed test cells (with one cell a column's centered sum of squares is
0 by construction; on the time-masked epoch splits at k >= 13 such
columns made Σ (z − z̄_test)² vanish and r2_c ~ −1e9), and NaN where that
sum is still <= 1e-12.  The test means enter only this baseline, never
the fit or the selection.  On random splits r2_c ≈ r2; the prospective
penalty in r2_c
measures prediction quality net of the shift.  RMSE (train-std units)
is logged too, but its unit is the training SD, which differs between
the prospective and the control training sets.

Random-split control.  For every setting / variant, C = --repeats x 10 = 50
random splits of the union of its train and test rows with the same train
and test sizes; control split d trains on a uniformly random subset of
n_train rows, permutation seed draw_seed(SEED + CONTROL_SPLIT_SEED_OFFSET,
d), and evaluates exactly like the prospective split (own fit, own
eligible set, one mask shared by all methods), the stochastic methods
with ALL D draws d' = 0 .. D − 1 and the draw seeds of the prospective
split.  Every split's statistic of a stochastic method is therefore the
mean over the same D common random draws, and conditionally on those
draws the C + 1 split statistics are exchangeable under "the prospective
split is a random split".  The prospective penalty of a method at k is
R²_prospective − mean_d R²_control,d, and penalty_pval = (1 + #{d:
R²_control,d <= R²_prospective}) / (1 + C) is the one-sided permutation
p-value of "the prospective split is harder than a random split of the
same size"; for random it ranks a D-draw mean among D-draw means, like
with like.  Over the control splits methods are also compared
with paired_stats.paired_stats (Nadeau-Bengio corrected interval; it was
derived for repeated random subsampling, this design).

Common random numbers (seed tags; new tags start at SEED + 13, after the
random baseline's SEED + RANDOM_SEED_OFFSET = SEED + 7 and the retired
SEED + ENTROPY_RF_SEED_OFFSET = SEED + 11 of the CV scripts):
  draw_seed(tag, d) = (tag, d), d = 0 .. D − 1 (or C − 1)
  SEED + 13  PROSP_RANDOM_SEED_OFFSET     random's permutation (random_select)
  SEED + 14  PROSP_ENTROPY_RF_SEED_OFFSET retired (entropy_rf, dropped)
  SEED + 15  CONTROL_SPLIT_SEED_OFFSET    control split d's row permutation
  SEED + 16  BOOT_SEED_OFFSET             bootstrap resamples (one stream
                                          per variant, default_rng(SEED + 16))
Draw d of random uses the seed (tag, d) on every split, the
prospective one and each control split alike.  Every method sees the same
split, correlation matrix and eligible set.  Seeds depend neither on the
setting nor on the variant: control split d of every variant takes the
same permutation of its union's row POSITIONS.  make_variant orders each
union training rows first (and drops duplicates per variant), so the same
positions hold different models in different variants: control training
sets of different cutoffs are not nested (control_split itself nests in
index space: a larger n_train extends the same permutation prefix).

Paired bootstrap over test models.  For the pairs BOOT_PAIRS (mi − random,
mi − entropy, mi − trace, trace − random, entropy − random, then the
facility-location pairs of paired_stats.MACRO_PAIRS: mi − facloc, entropy
− facloc, trace − facloc, facloc − random, appended so that the first five
rows of the pair axis are unchanged) at k in BOOT_K =
1, 3, 5, 10: B = --boot = 2000 resamples of the test models with
replacement (the same resamples for every method and pair), pooled R²
recomputed on each resample for both methods (stochastic methods: mean
over their D draws of the resampled pooled R²); logged: the full-sample
difference, the bootstrap mean and sd, the 95% percentile interval and
P(diff > 0) over the resamples.

Usage:
    python eval_prospective.py [--datasets mteb2026 helm epoch olv2sub]
                               [--repeats 5] [--kmax 15] [--boot 2000]
                               [--epoch-cutoffs 2024-07-01 ...]
                               [--olv2sub-cutoffs 2024-10-01 ...]
                               [--target-min-obs 2] [--min-models 10]
                               [--workers 1] [--plot-only]

--repeats R gives D = C = 10 R draws and control splits (default
N_REPEATS = 5: 50, the size of the matched CV design).  --plot-only
rebuilds the figures from the logs.  figures/prospective.pdf is written
when one invocation covers all four default settings (or by --plot-only);
after parallel per-setting runs call --plot-only.

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir()),
per setting tag; V variants (cutoffs), C control splits, D draws, N
benchmarks of the setting (a variant uses N_v <= N of them: epoch drops
the post-cutoff ones; every benchmark index below is into the setting's
benchmark_names), K = min(kmax, N) columns (column k − 1 = k benchmarks),
T = largest test set:
  LOGS/prospective_{tag}.npz
    Control split rows (S = V·C, variant-major; the per-split rows that
    paired_stats pairs on, one variant at a time: filter on `variant`):
      variant, draw (S,)             variant index, control draw d
      rep, fold, split_id (S,)       d // 10, d % 10, d (cv_splits layout)
      n_train, n_val, n_eligible (S,), eligible (S, N), estimation_failed (S,)
      n_test_cells (S,), test_fill (S,)   observed test cells and their
                                     fraction of n_val x N_v
      {m}_r2, {m}_r2_c, {m}_r2_all, {m}_rmse (S, K)   control R² (target
                                     rule above), shift-adjusted R², R² over
                                     all targets, RMSE (stochastic methods:
                                     mean over the D draws)
      {m}_r2_draws (S, D, K)         stochastic methods: every draw's R²
      {m}_n_models, {m}_n_cells (S, K)      test models contributing to the
                                     pooled R² (an observed selected
                                     benchmark and a target) and target
                                     cells evaluated: on sparse test rows
                                     (mteb2026, epoch) a selection covers
                                     only part of the test models
      {m}_selected (S, K)            picks in order, -1 padded (pca: its
                                     largest selection; stochastic: draw 0)
    Prospective split (one per variant):
      variant_keys, variant_labels, variant_macros, variant_cutoffs (V,)
      variant_columns (V, N), variant_n_bench (V,)   benchmarks of each
                                     variant
      prosp_n_train, prosp_n_val, prosp_n_eligible, prosp_estimation_failed
      (V,), prosp_eligible (V, N), prosp_test_fill (V,)
      prosp_n_test_dup (V,)          test rows dropped as duplicates of
                                     training rows
      prosp_n_test_old (V,)          test models released (olv2sub:
                                     uploaded) before the training side
                                     ended (-1: no dates, or new in time by
                                     construction as on epoch)
      prosp_n_bench_future, prosp_n_train_cells_future,
      prosp_n_test_cells_future, prosp_n_elig_future (V,)   epoch settings:
                                     post-cutoff benchmarks, their training
                                     and test cells and how many of them
                                     the training rows make eligible
                                     (dropped from the variant on epoch,
                                     kept on epoch_bymodel; -1 elsewhere)
      prosp_impute_check (V,)        max |prosp_{m}_r2_all − impute_and_
                                     evaluate's R²| over the deterministic
                                     methods at BOOT_K (cross-check), facloc
                                     excepted: prosp_facloc_impute_check
                                     (V,) is facloc's (CHECK_APART: the
                                     methods the check had before facloc
                                     keep their value)
      prosp_n_test_cells (V,)        observed test cells (before selection)
      prosp_n_test_cells_no_train, prosp_n_test_cells_few_train (V,)  those
                                     in benchmarks with 0 / with 1 ..
                                     target_min_obs − 1 training observations
                                     (not targets); prosp_n_test_bench,
                                     prosp_n_test_bench_no_train,
                                     prosp_n_test_bench_few_train (V,) the
                                     corresponding benchmark counts
      prosp_{m}_r2, prosp_{m}_r2_c, prosp_{m}_r2_all, prosp_{m}_rmse,
      prosp_{m}_n_models, prosp_{m}_n_cells (V, K)   (stochastic: mean
                                     over the D draws)
      prosp_{m}_r2_sd (V, K)         stochastic: sd over the draws (ddof 0)
      prosp_{m}_r2_draws (V, D, K)   stochastic: every draw
      prosp_{m}_selected (V, K)      deterministic picks (-1 padded);
                                     (V, D, K) for the stochastic methods;
                                     prosp_pca_selected_per_k (V, K, K)
      prosp_{m}_index_ties, prosp_{m}_tie_breaks (V, K)   deterministic
      prosp_{m}_ss_res, prosp_{m}_ss_tot, prosp_{m}_n_eval,
      prosp_{m}_ss_res_c, prosp_{m}_ss_tot_c, prosp_{m}_n_eval_c
      (V, len(BOOT_K), T)            deterministic: per test model sums
                                     behind the pooled R² and the
                                     shift-adjusted R² at BOOT_K (NaN /
                                     -1 padded); test_model_names,
                                     test_model_dates (V, T)
    Comparisons (V, K): ctrl_{m}_mean, ctrl_{m}_sd (nanstd over control
      splits), penalty_{m} = prosp_{m}_r2 − ctrl_{m}_mean, penalty_pval_{m};
      ctrl_{m}_r2_c_mean, penalty_c_{m} = prosp_{m}_r2_c −
      ctrl_{m}_r2_c_mean (the shift-adjusted penalty);
      ctrl_{m}_rmse_mean, rmse_penalty_{m} = prosp_{m}_rmse −
      ctrl_{m}_rmse_mean (> 0: larger prospective error in training-SD
      units); ctrl_{m}_n_models_mean.
    Bootstrap (V, P, len(BOOT_K)): boot_diff, boot_mean, boot_sd, boot_lo,
      boot_hi, boot_prob_pos, boot_prob_zero (identical resampled R²,
      e.g. identical selections), boot_n (finite resamples); boot_pairs
      (P,) 'a-b', boot_k
    Control pairs (V, P, len(BOOT_K)): ctrl_pair_diff, ctrl_pair_se,
      ctrl_pair_lo, ctrl_pair_hi, ctrl_pair_win, ctrl_pair_n (paired_stats)
    benchmark_names (N,), methods; scalars seed, n_draws, n_controls,
    n_boot, k_max, min_train_obs, target_min_obs, min_models and the seed
    offsets
  LOGS/prospective_{tag}.json
    config, checks (mteb2026: snapshot rows; helm: helm.npz and merged),
    per variant: sizes, benchmarks, target counts, info (duplicates,
    test_release; epoch: time_mask; olv2sub: scorer versions per side),
    summary tables (prospective, control, penalty, bootstrap, control
    pairs, with contributing-model counts and low_coverage flags),
    runtime_s, macros_suppressed (names withheld for low coverage) and
    "macros", a flat {name: number} dict:
      RsqProsp{S}{M}K{k}, RsqStdProsp{S}{M}K{k} (stochastic: sd over draws)
      RsqCtrl{S}{M}K{k}, RsqStdCtrl{S}{M}K{k}, PenaltyProsp{S}{M}K{k},
      PenaltyPvalProsp{S}{M}K{k}, RmseProsp{S}{M}K{k}, RmseCtrl{S}{M}K{k},
      RmsePenaltyProsp{S}{M}K{k}, NModelsProsp{S}{M}K{k},
      NModelsCtrl{S}{M}K{k} (contributing test models, control: mean),
      RsqCenProsp{S}{M}K{k}, RsqCenCtrl{S}{M}K{k},
      PenaltyCenProsp{S}{M}K{k} (shift-adjusted)       k in MACRO_K <= K
      BootDiffProsp{S}{A}Vs{B}K{k}, BootLo..., BootHi..., BootProbPos...
      PairDiffCtrl{S}{A}Vs{B}K{k}, PairLoCtrl..., PairHiCtrl...  k in BOOT_K
      NTrainProsp{S}, NTestProsp{S}, NEligProsp{S}, NBenchProsp{S},
      NTestCellsProsp{S}, NTestCellsNoTrainProsp{S},
      NTestCellsFewTrainProsp{S}, NTestBenchNoTrainProsp{S},
      NTestBenchFewTrainProsp{S}, NCtrlProsp{S}, NDrawsProsp{S},
      FillTestProsp{S}, FillTestCtrl{S} (control mean), NTestDupProsp{S},
      NTestOldProsp{S}, NBenchFutureProsp{S}, NTrainCellsFutureProsp{S},
      NTestCellsFutureProsp{S}, NEligFutureProsp{S} (the last five where
      defined), MinModelsProsp{setting}
      ProspCheck...                                     the data checks
    {S} = setting + variant (Mteb, MtebLate, Helm, EpochCutJanTwentyfive,
    OlvTwoSubCutOctTwentyfour, ...), {M} = make_numbers.camel(method),
    k spelled (Kfive); non-finite values are left out, and so are the
    low-coverage statistics (Coverage above).
  OUT/prospective_{tag}.pdf          one column per variant: R² (top) and
                                     shift-adjusted R² (bottom) vs k,
                                     prospective solid, control mean
                                     dashed; hollow markers: fewer than
                                     min_models contributing test models;
                                     the y axis is cut below the 10th
                                     percentile of a panel's values − 0.1
                                     (never above −0.5), so single
                                     outliers can lie off the panel
  OUT/prospective.pdf                the same, one column per setting
                                     (its PRIMARY_VARIANT)
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time
import warnings
from collections import Counter, OrderedDict
from typing import NamedTuple

import numpy as np

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
from baseline_select import (facility_location_select, greedy_trace_select,
                             hub_select, pca_cssp_select)
import cv_split
import score_bounds
from cv_split import (ENTROPY_RF_SEED_OFFSET, MIN_TARGET_OBS, MIN_TRAIN_OBS,
                      N_REPEATS, RANDOM_SEED_OFFSET, eligible_mask,
                      target_mask)
from eval_entropy_vs_mi import estimate_corr, impute_and_evaluate
from score_bounds import clip_z, dataset_bounds, z_limits
from paired_stats import paired_stats
from make_numbers import camel, int_words

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

K_MAX = 15
SEED = 42
DRAWS_PER_REPEAT = 10          # draws / control splits per --repeats
N_BOOT = 2000
BOOT_K = [1, 3, 5, 10]         # k of the bootstrap and control pairs
MACRO_K = [1, 3, 5, 10, 15]    # k of the R² / penalty macros
LEVEL = 0.95
RIDGE = 0.01                   # impute_and_evaluate's ridge
CLIP = 10.0                    # impute_and_evaluate's clip (train-std units)
DIFF_TOL = 1e-9                # |diff| above which data checks count a cell
MIN_MODELS = 10                # contributing test models a macro needs
MIN_CENTER_OBS = 2             # test cells a column needs for its test mean

# Seed tags SEED + offset of the common-random-number streams (draw_seed);
# 7 belongs to the CV scripts' random stream, 11 (entropy_rf) is retired.
PROSP_RANDOM_SEED_OFFSET = 13
# Retired with entropy_rf (protocol rule D3); kept reserved
PROSP_ENTROPY_RF_SEED_OFFSET = 14
CONTROL_SPLIT_SEED_OFFSET = 15
BOOT_SEED_OFFSET = 16
assert len({RANDOM_SEED_OFFSET, ENTROPY_RF_SEED_OFFSET,
            PROSP_RANDOM_SEED_OFFSET, PROSP_ENTROPY_RF_SEED_OFFSET,
            CONTROL_SPLIT_SEED_OFFSET, BOOT_SEED_OFFSET}) == 6

# facloc (facility location, the paper's headline method since
# 2026-10-19) is the last deterministic method: no random numbers, so the
# other methods' values are those of the logs without it
METHODS = ["mi", "entropy", "trace", "hub", "pca", "facloc", "random"]
DETERMINISTIC = ["mi", "entropy", "trace", "hub", "pca", "facloc"]
STOCHASTIC = ["random"]
PLOT_METHODS = ["mi", "entropy", "trace", "hub", "pca", "facloc", "random"]
# The key pairs, then facloc's pairs of paired_stats.MACRO_PAIRS (appended:
# the boot_* / ctrl_pair_* rows of the first five pairs are unchanged)
BOOT_PAIRS = [("mi", "random"), ("mi", "entropy"), ("mi", "trace"),
              ("trace", "random"), ("entropy", "random"),
              ("mi", "facloc"), ("entropy", "facloc"), ("trace", "facloc"),
              ("facloc", "random")]
# Deterministic methods whose impute_and_evaluate cross-check is logged
# apart (prosp_{m}_impute_check), so that prosp_impute_check keeps the
# methods it had before facloc was added
CHECK_APART = ["facloc"]
# Methods drawn with the thicker line
MAIN_METHODS = ("mi", "entropy", "trace", "facloc")

DEFAULT_SETTINGS = ["mteb2026", "helm", "epoch", "olv2sub"]
ALL_SETTINGS = DEFAULT_SETTINGS + ["olv2sub_math", "epoch_bymodel"]
SETTING_ALIASES = {"helm_lite": "helm"}
SETTING_MACRO = {"mteb2026": "Mteb", "helm": "Helm", "epoch": "Epoch",
                 "olv2sub": "OlvTwoSub", "olv2sub_math": "OlvTwoSubMath",
                 "epoch_bymodel": "EpochByModel"}
SETTING_LABEL = {"mteb2026": "MTEB, snapshot → 2026",
                 "helm": "HELM Lite, v1.0 → v1.13",
                 "epoch": "Epoch AI, by release date",
                 "olv2sub": "OLL v2 sub-tasks, by submission",
                 "olv2sub_math": "OLL v2 sub-tasks (+MATH)",
                 "epoch_bymodel": "Epoch AI, by model date only"}
EPOCH_CUTOFFS = ["2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"]
OLV2SUB_CUTOFFS = ["2024-10-01", "2025-01-01", "2025-03-01"]
MIN_OLV2SUB_TEST = 300
MTEB_LATE_AFTER = "2026-03-14"
# Variant of each setting shown in the combined figure
PRIMARY_VARIANT = {"mteb2026": "main", "helm": "main",
                   "epoch": "cut_2025-01-01", "olv2sub": "cut_2025-01-01",
                   "olv2sub_math": "cut_2025-01-01",
                   "epoch_bymodel": "cut_2025-01-01"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

METHOD_LABEL = {
    "entropy": "Entropy (Alg. 1)",
    "mi": "Mutual Info (Alg. 2)",
    "random": "Random",
    "hub": "Hub correlation",
    "pca": "PCA subset (CSSP)",
    "trace": "Greedy trace min.",
    "facloc": "Facility location",
}
METHOD_COLOR = {
    "entropy": "#4477AA",
    "mi": "#CC3311",
    "random": "#888888",
    "hub": "#228833",
    "pca": "#AA3377",
    "trace": "#EE7733",
    "facloc": "#009988",
}
METHOD_MARKER = {"entropy": "o", "mi": "o", "random": "x", "hub": "s",
                 "pca": "d", "trace": "^", "facloc": "v"}


# ── Seeds and splits ─────────────────────────────────────────────

def draw_seed(tag, draw):
    """Common-random-number seed of draw / control split `draw` for the
    stream `tag` (SEED + one of the *_SEED_OFFSET constants): (tag, draw)."""
    return (int(tag), int(draw))


def control_split(M, n_train, draw, seed=SEED):
    """(train_idx, test_idx) of control split `draw`: the first n_train
    entries of the permutation of the M union rows drawn from
    draw_seed(seed + CONTROL_SPLIT_SEED_OFFSET, draw), both sorted."""
    rng = np.random.default_rng(draw_seed(seed + CONTROL_SPLIT_SEED_OFFSET,
                                          draw))
    perm = rng.permutation(M)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


class Variant(NamedTuple):
    """One prospective split of a setting.  B, O hold the union of its
    rows on its benchmarks: rows 0 .. n_train − 1 are the training (old)
    models, the rest the test (new) models, each group in the source
    matrix's order (source row indices `rows`); column j is the setting's
    benchmark cols[j]."""
    key: str
    label: str
    macro: str
    B: np.ndarray
    O: np.ndarray
    n_train: int
    model_names: np.ndarray
    model_dates: np.ndarray
    cutoff: str
    info: dict
    cols: np.ndarray
    rows: np.ndarray

    @property
    def n_test(self):
        return self.B.shape[0] - self.n_train


def _as_index(sel):
    """Row / column selection (bool mask or indices) as an int array."""
    a = np.asarray(sel)
    return np.flatnonzero(a) if a.dtype == bool else a.astype(int)


def _row_key(b, o):
    """Hashable observation pattern and observed values of one row
    (-0.0 normalized to 0.0)."""
    o = o != 0
    return o.tobytes(), (np.where(o, b, 0.0) + 0.0).tobytes()


def boundary_duplicates(B, O, train, test, sha=None, min_obs=MIN_TRAIN_OBS):
    """Test rows that duplicate a training row: [(test row, training row,
    reason)].  reason "identical": the same observation pattern on the
    columns of B with >= min_obs observed cells and exactly the same
    values; "model_sha": the same non-empty sha[i] (the same weights under
    another name).  The first matching training row is reported."""
    seen, sha_of = {}, {}
    for i in train:
        if np.count_nonzero(O[i]) >= min_obs:
            seen.setdefault(_row_key(B[i], O[i]), int(i))
        if sha is not None and str(sha[i] or ""):
            sha_of.setdefault(str(sha[i]), int(i))
    out = []
    for i in test:
        key = _row_key(B[i], O[i])
        if np.count_nonzero(O[i]) >= min_obs and key in seen:
            out.append((int(i), seen[key], "identical"))
        elif sha is not None and str(sha[i] or "") in sha_of:
            out.append((int(i), sha_of[str(sha[i])], "model_sha"))
    return out


def make_variant(key, label, macro, B, O, names, dates, train, test,
                 cutoff="", info=None, cols=None, sha=None):
    """Variant of the rows train and test (bool masks or indices) of B,
    O, names and dates on the columns cols (default all): training rows
    first, each group in its given order.  Test rows that duplicate a
    training row on these columns (boundary_duplicates; with sha, also
    the same model_sha) are dropped and listed in info["duplicates"]."""
    train, test = _as_index(train), _as_index(test)
    if np.intersect1d(train, test).size:
        raise ValueError(f"{key}: training and test rows overlap")
    B = np.asarray(B, dtype=np.float64)
    O = np.asarray(O, dtype=np.float64)
    cols = np.arange(B.shape[1]) if cols is None else _as_index(cols)
    B, O = B[:, cols], O[:, cols]
    names = np.asarray(names)
    dups = boundary_duplicates(B, O, train, test, sha)
    drop = {t for t, _, _ in dups}
    test = np.array([i for i in test if i not in drop], dtype=int)
    info = dict(info or {})
    info["duplicates"] = {
        "rule": "test row with the observation pattern and values of a "
                f"training row (>= {MIN_TRAIN_OBS} observed cells)"
                + (", or its model_sha" if sha is not None else ""),
        "n_dropped": len(dups),
        "rows": [{"test": str(names[t]), "train": str(names[r]),
                  "reason": why} for t, r, why in dups]}
    idx = np.concatenate([train, test]).astype(int)
    return Variant(key, label, macro, B[idx], O[idx], int(len(train)),
                   names[idx], np.asarray(dates)[idx], cutoff, info, cols,
                   idx)


def release_info(v, old, rule):
    """info entry of the test models of v that are new to the leaderboard
    but not new in time: old (bool per test model, in v's test order)
    under `rule`; also stored as info["n_test_old"]."""
    test_names = v.model_names[v.n_train:]
    old = np.asarray(old, dtype=bool)
    v.info["test_release"] = {"rule": rule, "n_test": int(v.n_test),
                              "n_old": int(old.sum()),
                              "old_models": [str(n) for n in
                                             test_names[old]]}
    v.info["n_test_old"] = int(old.sum())


def cutoff_macro(cutoff):
    """Macro slot of a cutoff date: 2024-07-01 -> 'CutJulTwentyfour',
    2025-01-15 -> 'CutJanTwentyfiveDfifteen'."""
    y, m, d = (int(x) for x in cutoff.split("-"))
    s = "Cut" + MONTHS[m - 1] + int_words(y % 100).capitalize()
    return s + ("D" + int_words(d) if d != 1 else "")


def date_split(dates, cutoff):
    """(train, test) bool masks: date < cutoff, date >= cutoff (ISO date
    strings; rows without a valid date are in neither)."""
    valid = np.array([bool(_DATE.match(str(x))) for x in dates])
    s = np.array([str(x) for x in dates])
    return valid & (s < cutoff), valid & (s >= cutoff)


def _load_matrix(name):
    d = np.load(DATA / f"{name}.matrix.npz", allow_pickle=True)
    return (d["B"].astype(np.float64), d["O"].astype(np.float64),
            np.array([str(x) for x in d["model_names"]]),
            np.array([str(x) for x in d["benchmark_names"]]))


def _load_meta(name):
    with open(DATA / f"{name}.meta.json") as f:
        return json.load(f)


# ── Data checks ──────────────────────────────────────────────────

def compare_rows(B_ref, O_ref, B_new, O_new, pairs):
    """Cell-wise comparison of matched rows (i_ref, i_new) over common
    columns.  Returns the totals and the rows with a difference."""
    out = {"n_rows": len(pairs), "n_cells_compared": 0, "n_cells_differ": 0,
           "max_abs_diff": 0.0, "n_obs_only_ref": 0, "n_obs_only_new": 0}
    rows = []
    for i_ref, i_new, name_ref, name_new in pairs:
        o_r, o_n = O_ref[i_ref] != 0, O_new[i_new] != 0
        both = o_r & o_n
        d = np.abs(B_ref[i_ref, both] - B_new[i_new, both])
        n_diff = int(np.sum(d > DIFF_TOL))
        only_r, only_n = int(np.sum(o_r & ~o_n)), int(np.sum(o_n & ~o_r))
        mx = float(d.max()) if d.size else 0.0
        out["n_cells_compared"] += int(both.sum())
        out["n_cells_differ"] += n_diff
        out["max_abs_diff"] = max(out["max_abs_diff"], mx)
        out["n_obs_only_ref"] += only_r
        out["n_obs_only_new"] += only_n
        if n_diff or only_r or only_n:
            rows.append({"name": name_new, "ref_name": name_ref,
                         "n_cells_differ": n_diff, "max_abs_diff": mx,
                         "n_obs_only_ref": only_r, "n_obs_only_new": only_n})
    rows.sort(key=lambda r: -r["max_abs_diff"])
    out["n_rows_differ"] = sum(1 for r in rows if r["n_cells_differ"])
    out["n_rows_obs_mismatch"] = sum(
        1 for r in rows if r["n_obs_only_ref"] or r["n_obs_only_new"])
    out["rows"] = rows
    return out


def check_mteb_snapshot(snap, new, meta):
    """in_snapshot rows of mteb2026 (matched by snapshot_name) against
    data/mteb.matrix.npz; snap and new are (B, O, model_names,
    benchmark_names)."""
    Bs, Os, ms, bs = snap
    Bn, On, mn, bn = new
    if list(bs) != list(bn):
        raise ValueError("mteb2026 columns differ from mteb.matrix.npz")
    s_idx = {n: i for i, n in enumerate(ms)}
    pairs, missing = [], []
    for i_new, m in enumerate(meta["models"]):
        if not m.get("in_snapshot"):
            continue
        ref = m.get("snapshot_name") or m["name"]
        if ref not in s_idx:
            missing.append(ref)
            continue
        pairs.append((s_idx[ref], i_new, ref, m["name"]))
    out = compare_rows(Bs, Os, Bn, On, pairs)
    covered = {p[0] for p in pairs}
    out.update(reference="data/mteb.matrix.npz",
               n_snapshot_rows=len(ms),
               n_renamed=sum(1 for p in pairs if p[2] != p[3]),
               in_snapshot_names_not_in_matrix=missing,
               matrix_rows_not_covered=[ms[i] for i in range(len(ms))
                                        if i not in covered])
    return out


def check_helm_snapshot(B, O, names, bench, meta):
    """helm_lite in_snapshot rows against data/helm.npz (by
    snapshot_display_name) and the helm/* columns of merged (by the helm
    names of data/canonical_mapping.json)."""
    mods = meta["models"]
    snap_name = {n: (mods[n].get("snapshot_display_name") or n)
                 for n in names if mods[n].get("in_snapshot")}
    row_of = {snap_name[n]: i for i, n in enumerate(names) if n in snap_name}
    checks = {}

    h = np.load(DATA / "helm.npz", allow_pickle=True)
    hn = [str(x) for x in h["model_names"]]
    hb = [str(x) for x in h["benchmark_names"]]
    cols = [b for b in hb if b in bench]
    Bh = h["B"][:, [hb.index(b) for b in cols]].astype(np.float64)
    Oh = h["O"][:, [hb.index(b) for b in cols]].astype(np.float64)
    Bl = B[:, [list(bench).index(b) for b in cols]]
    Ol = O[:, [list(bench).index(b) for b in cols]]
    pairs = [(i, row_of[n], n, names[row_of[n]])
             for i, n in enumerate(hn) if n in row_of]
    res = compare_rows(Bh, Oh, Bl, Ol, pairs)
    res.update(reference="data/helm.npz", n_reference_rows=len(hn),
               n_columns=len(cols),
               reference_columns_not_compared=[b for b in hb
                                               if b not in bench],
               reference_rows_unmatched=[n for n in hn if n not in row_of],
               n_in_snapshot=len(snap_name))
    checks["helm_npz"] = res

    d = np.load(DATA / "merged.matrix.npz", allow_pickle=True)
    mB, mO = d["B"].astype(np.float64), d["O"].astype(np.float64)
    mn = [str(x) for x in d["model_names"]]
    mb = [str(x) for x in d["benchmark_names"]]
    with open(DATA / "canonical_mapping.json") as f:
        canon = json.load(f)["canonical_models"]
    helm_to_canon = {g["datasets"]["helm"]: g["canonical_name"]
                     for g in canon if "helm" in g["datasets"]}
    mcols = [b for b in bench if f"helm/{b}" in mb]
    Bm = mB[:, [mb.index(f"helm/{b}") for b in mcols]]
    Om = mO[:, [mb.index(f"helm/{b}") for b in mcols]]
    Bl = B[:, [list(bench).index(b) for b in mcols]]
    Ol = O[:, [list(bench).index(b) for b in mcols]]
    pairs, unmatched = [], []
    for hname, cname in sorted(helm_to_canon.items()):
        if cname not in mn:
            continue
        if hname not in row_of:
            unmatched.append(hname)
            continue
        pairs.append((mn.index(cname), row_of[hname], cname,
                      names[row_of[hname]]))
    res = compare_rows(Bm, Om, Bl, Ol, pairs)
    res.update(reference="data/merged.matrix.npz helm/* columns",
               n_columns=len(mcols),
               reference_columns_not_compared=[
                   b for b in mb if b.startswith("helm/")
                   and b[len("helm/"):] not in bench],
               mapped_helm_models_unmatched=unmatched,
               n_merged_rows_with_helm=int(
                   (mO[:, [i for i, b in enumerate(mb)
                           if b.startswith("helm/")]].sum(1) > 0).sum()))
    checks["merged"] = res
    return checks


# ── Settings ─────────────────────────────────────────────────────

def load_mteb2026(args):
    snap = _load_matrix("mteb")
    new = _load_matrix("mteb2026")
    meta = _load_meta("mteb2026")
    Bn, On, mn, bn = new
    if [m["name"] for m in meta["models"]] != list(mn):
        raise ValueError("mteb2026.meta.json models out of row order")
    checks = {"mteb_snapshot": check_mteb_snapshot(snap, new, meta)}

    test = [i for i, m in enumerate(meta["models"])
            if m.get("recommended_prospective")]
    listed = meta["prospective_test_set"]["models"]
    if sorted(mn[test]) != sorted(listed):
        raise ValueError("recommended_prospective differs from "
                         "prospective_test_set.models")
    late_listed = set(meta["prospective_test_set"]
                      ["optional_release_date_filter"]["models"])
    late = [i for i in test
            if str(meta["models"][i].get("release_date") or "")
            > MTEB_LATE_AFTER]
    if set(mn[late]) != late_listed:
        raise ValueError("release-date filter differs from "
                         "optional_release_date_filter.models")
    Bs, Os, ms, _ = snap
    leak = sorted(set(mn[test]) & set(ms))
    if leak:
        raise ValueError(f"test models in the snapshot matrix: {leak}")

    # Union: snapshot rows (training) then the mteb2026 test rows
    rel = {m.get("snapshot_name") or m["name"]: m.get("release_date") or ""
           for m in meta["models"] if m.get("in_snapshot")}
    B = np.vstack([Bs, Bn])
    O = np.vstack([Os, On])
    names = np.concatenate([ms, mn])
    dates = np.array([rel.get(n, "") for n in ms]
                     + [meta["models"][i].get("release_date") or ""
                        for i in range(len(mn))])
    M_s = len(ms)
    train = np.arange(M_s)
    info = {"train_source": "data/mteb.matrix.npz (snapshot)",
            "test_source": "data/mteb2026.matrix.npz recommended_prospective",
            "test_rule": meta["prospective_test_set"]["rule"],
            "excluded_duplicates":
                meta["prospective_test_set"].get("excluded_duplicates")}
    variants = [
        make_variant("main", "57 models added after the snapshot", "",
                     B, O, names, dates, train, M_s + np.array(test),
                     info=info),
        make_variant("late", f"release_date > {MTEB_LATE_AFTER}", "Late",
                     B, O, names, dates, train, M_s + np.array(late),
                     info=dict(info, test_rule=info["test_rule"]
                               + f" and release_date > {MTEB_LATE_AFTER}")),
    ]
    for v in variants:
        rel = [str(d) for d in v.model_dates[v.n_train:]]
        release_info(v, [bool(_DATE.match(d)) and d <= MTEB_LATE_AFTER
                         for d in rel],
                     f"release_date <= {MTEB_LATE_AFTER} (snapshot commit "
                     "2026-03-13)")
        v.info["test_release"]["by_year"] = dict(sorted(
            Counter(d[:4] if _DATE.match(d) else "none" for d in rel)
            .items()))
    return bn, variants, checks


def load_helm(args):
    B, O, names, bench = _load_matrix("helm_lite")
    meta = _load_meta("helm_lite")
    mods = meta["models"]
    if list(mods) != list(names):
        raise ValueError("helm_lite.meta.json models out of row order")
    checks = check_helm_snapshot(B, O, names, bench, meta)
    train = np.array([bool(mods[n]["in_snapshot"]) for n in names])
    dates = np.array([mods[n].get("first_eval_suite_date") or ""
                      for n in names])
    suites = Counter(mods[n]["first_eval_suite"] for n in names[~train])
    info = {"train_rule": "models.in_snapshot (HELM Lite v1.0.0)",
            "test_rule": "first_eval_suite v1.1.0 - v1.13.0",
            "test_first_eval_suites": dict(sorted(suites.items())),
            "test_first_eval_dates": [min(dates[~train]),
                                      max(dates[~train])]}
    variants = [make_variant("main", "v1.0.0 models → later releases", "",
                             B, O, names, dates, train, ~train, info=info)]
    # The v1.0.0 release date: the first evaluation of every snapshot model
    snap_date = max(mods[n]["first_eval_suite_date"] for n in names[train])
    for v in variants:
        rel = [str(mods[n].get("release_date") or "")
               for n in v.model_names[v.n_train:]]
        release_info(v, [bool(_DATE.match(d)) and d <= snap_date
                         for d in rel],
                     f"release_date <= {snap_date} (HELM Lite v1.0.0)")
    return bench, variants, checks


def load_epoch(args, time_mask=True):
    """Epoch rolling origin.  time_mask (setting epoch): only the
    benchmarks with release_date < c; False (epoch_bymodel): all."""
    B, O, names, bench = _load_matrix("epoch")
    meta = _load_meta("epoch")
    if [m["name"] for m in meta["models"]] != list(names):
        raise ValueError("epoch.meta.json models out of row order")
    if [b["name"] for b in meta["benchmarks"]] != list(bench):
        raise ValueError("epoch.meta.json benchmarks out of column order")
    dates = np.array([m.get("release_date") or "" for m in meta["models"]])
    bench_dates = np.array([str(b.get("release_date") or "")
                            for b in meta["benchmarks"]])
    dated = np.array([bool(_DATE.match(d)) for d in bench_dates])
    listed = meta.get("prospective_split_cutoffs", {})
    row_of = {n: i for i, n in enumerate(names)}
    col_of = {b: j for j, b in enumerate(bench)}
    superseded = [(row_of[c["model"]], col_of[c["benchmark"]])
                  for c in meta.get("cells_missing_but_in_raw", [])
                  if c.get("reason") == "superseded"
                  and c["model"] in row_of and c["benchmark"] in col_of]
    variants = []
    for c in args.epoch_cutoffs:
        train, test = date_split(dates, c)
        past = dated & (bench_dates < c)
        cols = past if time_mask else np.ones(len(bench), dtype=bool)
        info = {"train_rule": f"release_date < {c}",
                "test_rule": f"release_date >= {c}",
                "benchmark_rule": (f"benchmarks with release_date < {c}"
                                   if time_mask else
                                   "every benchmark (split by model only)"),
                "n_benchmarks_released_on_or_after": int(np.sum(~past)),
                "meta_cutoff": listed.get(c),
                "n_models_without_date": int(np.sum(~(train | test)))}
        v = make_variant(f"cut_{c}", f"cutoff {c}", cutoff_macro(c), B, O,
                         names, dates, train, test, cutoff=c, info=info,
                         cols=cols)
        tr, te = v.rows[:v.n_train], v.rows[v.n_train:]
        side_rows = {"train": set(tr.tolist()), "test": set(te.tolist())}
        fut = ~past
        el_fut = eligible_mask(O[tr]) & fut
        v.info["time_mask"] = {
            "applied": bool(time_mask),
            "n_bench_kept": int(len(v.cols)),
            "n_bench_future": int(fut.sum()),
            "future_benchmarks": [str(b) for b in bench[fut]],
            "n_train_cells_future": int(O[tr][:, fut].sum()),
            "n_test_cells_future": int(O[te][:, fut].sum()),
            "n_eligible_future": int(el_fut.sum()),
            "eligible_future": [str(b) for b in bench[el_fut]],
            "superseded_missing_cells": {
                side: int(sum(1 for i, j in superseded
                              if i in rows and cols[j]))
                for side, rows in side_rows.items()}}
        variants.append(v)
    return bench, variants, {}


def load_olv2sub(args, keep_math=False):
    B, O, names, bench = _load_matrix("olv2sub")
    meta = _load_meta("olv2sub")
    if list(meta["model_names"]) != list(names):
        raise ValueError("olv2sub.meta.json models out of row order")
    mods = meta["models"]
    dates = np.array([str(x or "") for x in mods["submission_date"]])
    upload = np.array([str(x or "") for x in mods["upload_to_hub_date"]])
    sha = np.array([str(x or "") for x in mods["model_sha"]])
    version = np.array([np.nan if v is None else float(v)
                        for v in mods["math_hard_task_version"]])
    math_cols = np.array([b.startswith("math_hard/") for b in bench])
    keep = np.ones(len(bench), dtype=bool) if keep_math else ~math_cols
    shift = meta.get("counts", {}).get(
        "math_hard_aggregate_shift_between_scorer_versions")
    variants = []
    for c in args.olv2sub_cutoffs:
        train, test = date_split(dates, c)
        info = {"train_rule": f"submission_date < {c}",
                "test_rule": f"submission_date >= {c}",
                "n_models_without_date": int(np.sum(~(train | test))),
                "math_hard_columns": "kept" if keep_math else "dropped",
                "math_hard_aggregate_shift_between_scorer_versions": shift}
        # the setting's benchmarks are bench[keep]: every variant uses all
        v = make_variant(f"cut_{c}", f"cutoff {c}", cutoff_macro(c),
                         B[:, keep], O[:, keep], names, dates, train, test,
                         cutoff=c, info=info, sha=sha)
        if v.n_test < MIN_OLV2SUB_TEST:
            print(f"  warning: olv2sub cutoff {c} leaves {v.n_test} "
                  f"< {MIN_OLV2SUB_TEST} test models", flush=True)
        sides = (("train", v.rows[:v.n_train]), ("test", v.rows[v.n_train:]))
        v.info["math_hard_task_version_counts"] = {
            side: {f"v{x:.1f}": int(n) for x, n in sorted(
                Counter(version[r][np.isfinite(version[r])]).items())}
            for side, r in sides}
        up = upload[v.rows[v.n_train:]]
        release_info(v, [bool(_DATE.match(u)) and u < c for u in up],
                     f"upload_to_hub_date < {c} (submitted on or after it)")
        v.info["test_release"]["n_without_upload_date"] = int(
            sum(1 for u in up if not _DATE.match(u)))
        variants.append(v)
    checks = {"math_hard_dropped": [] if keep_math
              else [str(b) for b in bench[math_cols]],
              "math_hard_reason": None if keep_math else (
                  "models.math_hard_task_version is mixed (v1.0 original "
                  "scorer, v2.0 2024-10-24 fix or later runs not re-scored, "
                  "v3.0 2025-02-13 bulk re-scoring); the versions are "
                  "confounded with submission date, so MATH_Lvl5 values are "
                  "not comparable across the train/test boundary")}
    return bench[keep], variants, checks


LOADERS = {"mteb2026": load_mteb2026, "helm": load_helm, "epoch": load_epoch,
           "olv2sub": load_olv2sub,
           "olv2sub_math": lambda args: load_olv2sub(args, keep_math=True),
           "epoch_bymodel": lambda args: load_epoch(args, time_mask=False)}


# ── Imputation per test model ────────────────────────────────────

def make_test_set(B_test, O_test, col_mean, col_std, bounds=None):
    """Train-standardized test matrix and its observation-pattern groups.

    Z = clip(O · (B − mean) / std, ±CLIP) · O exactly as
    impute_and_evaluate; rows with identical observation patterns (O ==
    1) share one regression per selection.  "z_lo", "z_hi": the valid
    score range in these units (score_bounds.z_limits of `bounds`, the
    D2 clip of the predictions; None: no clip)."""
    z_lo, z_hi = z_limits(bounds, col_mean, col_std)
    O_test = np.asarray(O_test, dtype=np.float64)
    Z = O_test * (B_test - col_mean[None, :]) / col_std[None, :]
    Z = np.clip(Z, -CLIP, CLIP) * O_test
    obs = O_test == 1
    # Test-set column means of Z (baseline of the shift-adjusted R²); a
    # column needs >= MIN_CENTER_OBS observed test cells for it (with one
    # cell its centered sum of squares is 0 by construction)
    n_obs = obs.sum(axis=0)
    zbar = np.sum(np.where(obs, Z, 0.0), axis=0) / np.maximum(n_obs, 1)
    center_ok = n_obs >= MIN_CENTER_OBS
    if obs.shape[0] == 0:
        return {"Z": Z, "zbar": zbar, "center_ok": center_ok, "groups": [],
                "n": 0, "z_lo": z_lo, "z_hi": z_hi}
    patterns, inverse = np.unique(obs, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    groups = [(patterns[g], np.flatnonzero(inverse == g))
              for g in range(len(patterns))]
    return {"Z": Z, "zbar": zbar, "center_ok": center_ok, "groups": groups,
            "n": obs.shape[0], "z_lo": z_lo, "z_hi": z_hi}


def per_model_ss(test, selected, R, target_ok, target_all=None):
    """Per test model sums of squares of the imputation of its observed
    unselected benchmarks from its observed selected ones.

    Model i is conditioned on obs_A = [j in selected (in order) observed
    for i] and predicts ev = its observed unselected targets (target_all:
    the D1 targets of impute_and_evaluate, cv_split.target_mask of the
    training rows; None: every column) with W = R_ev,A (R_AA + RIDGE·I)⁻¹,
    as impute_and_evaluate, the predictions clipped to the valid range
    (test["z_lo"], test["z_hi"], rule D2); a model with an empty obs_A or
    ev contributes nothing.  Returns a dict of (n,) arrays: ss_res, ss_tot,
    n_eval over the targets (ev ∩ target_ok); ss_res_c, ss_tot_c, n_eval_c
    over the targets whose column has a test-set mean (test["center_ok"]),
    ss_tot_c about those means test["zbar"] (the shift-adjusted R²'s
    baseline); and ss_res_all, ss_tot_all, n_eval_all over all of ev
    (impute_and_evaluate: r2 = 1 − Σ ss_res_all / max(Σ ss_tot_all,
    1e-12))."""
    n = test["n"]
    Z = test["Z"]
    out = {key: np.zeros(n) for key in ("ss_res", "ss_tot", "ss_res_c",
                                        "ss_tot_c", "ss_res_all",
                                        "ss_tot_all")}
    for key in ("n_eval", "n_eval_c", "n_eval_all"):
        out[key] = np.zeros(n, dtype=np.int64)
    selected = [int(j) for j in selected]
    sel_mask = np.zeros(R.shape[0], dtype=bool)
    sel_mask[selected] = True
    cand = ~sel_mask
    if target_all is not None:
        cand &= np.asarray(target_all, dtype=bool)
    for pattern, rows in test["groups"]:
        obs_A = [j for j in selected if pattern[j]]
        ev = np.flatnonzero(pattern & cand)
        if not obs_A or ev.size == 0:
            continue
        R_oo = R[np.ix_(obs_A, obs_A)] + RIDGE * np.eye(len(obs_A))
        R_eo = R[np.ix_(ev, obs_A)]
        W = np.linalg.solve(R_oo, R_eo.T).T
        X_pred, _ = clip_z(Z[np.ix_(rows, obs_A)] @ W.T, test["z_lo"][ev],
                           test["z_hi"][ev])
        X_true = Z[np.ix_(rows, ev)]
        sq = (X_true - X_pred) ** 2
        tot = X_true ** 2
        out["ss_res_all"][rows] = sq.sum(axis=1)
        out["ss_tot_all"][rows] = tot.sum(axis=1)
        out["n_eval_all"][rows] = ev.size
        tk = target_ok[ev]
        out["ss_res"][rows] = sq[:, tk].sum(axis=1)
        out["ss_tot"][rows] = tot[:, tk].sum(axis=1)
        out["n_eval"][rows] = int(tk.sum())
        tc = tk & test["center_ok"][ev]
        out["ss_res_c"][rows] = sq[:, tc].sum(axis=1)
        out["ss_tot_c"][rows] = (
            (X_true - test["zbar"][ev]) ** 2)[:, tc].sum(axis=1)
        out["n_eval_c"][rows] = int(tc.sum())
    return out


def pooled_r2(ss_res, ss_tot, n_eval, flat_nan=False):
    """1 − Σ ss_res / max(Σ ss_tot, 1e-12); NaN without evaluated cells,
    and with flat_nan also where Σ ss_tot <= 1e-12 (no variance about the
    baseline: R² undefined)."""
    if np.sum(n_eval) == 0:
        return np.nan
    tot = np.sum(ss_tot)
    if flat_nan and tot <= 1e-12:
        return np.nan
    return 1.0 - np.sum(ss_res) / max(tot, 1e-12)


# ── Selection and split evaluation ───────────────────────────────

def run_selection(method, R, k_max, benchmark_names, eligible, draw=None):
    """Selection of `method` restricted to `eligible`; random uses draw
    `draw` of its stream (draw_seed)."""
    if method == "entropy":
        return greedy_entropy(R, k_max, benchmark_names=benchmark_names,
                              eligible=eligible)
    if method == "mi":
        return greedy_mi(R, k_max, benchmark_names=benchmark_names,
                         eligible=eligible)
    if method == "trace":
        return greedy_trace_select(R, k_max, benchmark_names=benchmark_names,
                                   eligible=eligible)
    if method == "hub":
        return hub_select(R, k_max, benchmark_names=benchmark_names,
                          eligible=eligible)
    if method == "pca":
        return pca_cssp_select(R, k_max, benchmark_names=benchmark_names,
                               eligible=eligible)
    if method == "facloc":
        return facility_location_select(R, k_max,
                                        benchmark_names=benchmark_names,
                                        eligible=eligible)
    if method == "random":
        return random_select(
            R, k_max, seed=draw_seed(SEED + PROSP_RANDOM_SEED_OFFSET, draw),
            benchmark_names=benchmark_names, eligible=eligible)
    raise ValueError(method)


# Per test model sums kept at BOOT_K (bootstrap; logged per model)
PER_MODEL_SS = ("ss_res", "ss_tot", "n_eval", "ss_res_c", "ss_tot_c",
                "n_eval_c")


def padded(sel, k_max):
    sel = [int(j) for j in list(sel)[:k_max]]
    return sel + [-1] * (k_max - len(sel))


def evaluate_selection(result, k_max, test, R, target_ok, keep_ks=(),
                       target_all=None):
    """Per-k R² (targets), R² over all targets, RMSE, contributing test
    models (an observed selected benchmark and a target) and evaluated
    target cells of one selection (NaN / 0 where fewer than k picks), its
    picks and tie counts, and the per-model sums at keep_ks."""
    out = {key: np.full(k_max, np.nan) for key in ("r2", "r2_c", "r2_all",
                                                   "rmse")}
    out["n_models"] = np.zeros(k_max, dtype=np.int64)
    out["n_cells"] = np.zeros(k_max, dtype=np.int64)
    out["index_ties"] = np.zeros(k_max, dtype=np.int64)
    out["tie_breaks"] = np.zeros(k_max, dtype=np.int64)
    out["ss"] = {}
    for k in range(1, k_max + 1):
        out["index_ties"][k - 1] = index_tie_count(result, k)
        out["tie_breaks"][k - 1] = tie_break_count(result, k)
        selected = selection_at_k(result, k)
        if selected is None:
            continue
        ss = per_model_ss(test, selected, R, target_ok, target_all)
        out["r2"][k - 1] = pooled_r2(ss["ss_res"], ss["ss_tot"], ss["n_eval"])
        out["r2_c"][k - 1] = pooled_r2(ss["ss_res_c"], ss["ss_tot_c"],
                                       ss["n_eval_c"], flat_nan=True)
        out["r2_all"][k - 1] = pooled_r2(ss["ss_res_all"], ss["ss_tot_all"],
                                         ss["n_eval_all"])
        n_ev = int(ss["n_eval"].sum())
        out["rmse"][k - 1] = (np.sqrt(ss["ss_res"].sum() / n_ev) if n_ev
                              else np.nan)
        out["n_models"][k - 1] = int(np.count_nonzero(ss["n_eval"]))
        out["n_cells"][k - 1] = n_ev
        if k in keep_ks:
            out["ss"][k] = {key: ss[key] for key in PER_MODEL_SS}
    out["selected"] = padded(result["selected"], k_max)
    if "selected_per_k" in result:
        out["selected_per_k"] = [padded(s, k_max)
                                 for s in result["selected_per_k"]]
    if "first" in result:
        out["first"] = -1 if result["first"] is None else int(result["first"])
    return out


def evaluate_split(B, O, train_idx, test_idx, k_max, benchmark_names, draws,
                   target_min_obs=MIN_TARGET_OBS, keep_ks=(),
                   crosscheck=False, bounds=None):
    """Every method on one train/test split.

    Correlation, means, standard deviations, eligibility and targets come
    from the training rows only; the test rows enter make_test_set only.
    Targets: the test cells of benchmarks observed for at least
    max(target_min_obs, MIN_TARGET_OBS) training models (r2; r2_all: at
    least MIN_TARGET_OBS, cv_split.target_mask, rule D1); predictions are
    clipped to `bounds` (the variant's valid score ranges, rule D2; None:
    no clip).
    Deterministic methods are run once, the stochastic ones for every draw
    in `draws`.  Returns a dict with the split sizes, the eligible mask,
    the target counts, "failed" and per method a list of
    evaluate_selection outputs (one per draw).  crosscheck: also
    "impute_check", max |r2_all − impute_and_evaluate r2| over the
    deterministic methods at keep_ks except those of CHECK_APART (0 up to
    rounding: both score the same cells now that impute_and_evaluate's
    vectorized path requires identical observation rows), and
    "impute_check_{m}", the same for each method m of CHECK_APART."""
    B_tr, O_tr = B[train_idx], O[train_idx]
    B_te, O_te = B[test_idx], O[test_idx]
    N = B.shape[1]
    eligible = eligible_mask(O_tr)
    n_obs_train = np.count_nonzero(O_tr, axis=0)
    target_all = target_mask(O_tr)                       # D1
    target_ok = target_all & (n_obs_train >= target_min_obs)
    obs_te = O_te == 1
    split = {"n_train": len(train_idx), "n_val": len(test_idx),
             "eligible": eligible, "n_eligible": int(eligible.sum()),
             "n_bench": int(N), "n_test_cells": int(obs_te.sum()),
             "test_fill": (float(obs_te.sum()) / (len(test_idx) * N)
                           if len(test_idx) and N else np.nan),
             "n_test_cells_no_train": int(obs_te[:, n_obs_train == 0].sum()),
             "n_test_cells_few_train": int(
                 obs_te[:, (n_obs_train > 0) & ~target_ok].sum()),
             "n_test_bench": int(np.sum(obs_te.any(axis=0))),
             "n_test_bench_no_train": int(np.sum(obs_te.any(axis=0)
                                                 & (n_obs_train == 0))),
             "n_test_bench_few_train": int(np.sum(
                 obs_te.any(axis=0) & (n_obs_train > 0) & ~target_ok)),
             "failed": False, "methods": {}}
    try:
        R, col_mean, col_std = estimate_corr(B_tr, O_tr)
    except Exception as e:  # noqa: BLE001 - logged, split left NaN
        print(f" — estimation failed: {e}", flush=True)
        split["failed"] = True
        return split
    test = make_test_set(B_te, O_te, col_mean, col_std, bounds)
    check = 0.0
    check_apart = {m: 0.0 for m in CHECK_APART if m in DETERMINISTIC}
    for m in METHODS:
        runs = [None] if m in DETERMINISTIC else list(draws)
        evals = []
        for d in runs:
            result = run_selection(m, R, k_max, benchmark_names, eligible, d)
            evals.append(evaluate_selection(result, k_max, test, R,
                                            target_ok, keep_ks, target_all))
            if crosscheck and m in DETERMINISTIC:
                for k in keep_ks:
                    sel = selection_at_k(result, k)
                    if sel is None:
                        continue
                    ref = impute_and_evaluate(B_te, O_te, sel, R, col_mean,
                                              col_std, N, target=target_all,
                                              bounds=bounds)["r2"]
                    got = evals[-1]["r2_all"][k - 1]
                    if not (np.isfinite(ref) or np.isfinite(got)):
                        continue
                    if m in check_apart:
                        check_apart[m] = max(check_apart[m], abs(ref - got))
                    else:
                        check = max(check, abs(ref - got))
        split["methods"][m] = evals
    if crosscheck:
        split["impute_check"] = check
        for m, c in check_apart.items():
            split[f"impute_check_{m}"] = c
    return split


def to_setting_columns(split, cols, n_setting):
    """Map the column indices of an evaluate_split result on a variant's
    benchmarks (column j = the setting's benchmark cols[j]) to the
    setting's indices, in place: the eligible mask becomes (n_setting,)
    (False off the variant), picks become setting indices (-1 kept)."""
    cols = np.asarray(cols, dtype=int)
    full = np.zeros(n_setting, dtype=bool)
    full[cols] = split["eligible"]
    split["eligible"] = full

    def remap(sel):
        return [int(cols[j]) if j >= 0 else -1 for j in sel]

    for evals in split["methods"].values():
        for e in evals:
            e["selected"] = remap(e["selected"])
            if "selected_per_k" in e:
                e["selected_per_k"] = [remap(s) for s in e["selected_per_k"]]
            if "first" in e:
                e["first"] = remap([e["first"]])[0]
    return split


# ── Bootstrap and comparisons ────────────────────────────────────

def boot_weights(n, n_boot, rng):
    """(n_boot, n) resampling counts of n test models (with replacement)."""
    idx = rng.integers(0, n, size=(n_boot, n))
    W = np.zeros((n_boot, n))
    for b in range(n_boot):
        W[b] = np.bincount(idx[b], minlength=n)
    return W


def boot_r2(W, ss):
    """Resampled pooled R² (n_boot,) from per-model sums ss with arrays of
    shape (D, n): mean over the D draws of 1 − W·ss_res / max(W·ss_tot,
    1e-12) (NaN where a resample has no evaluated cell)."""
    res = W @ np.atleast_2d(ss["ss_res"]).T
    tot = W @ np.atleast_2d(ss["ss_tot"]).T
    nev = W @ np.atleast_2d(ss["n_eval"]).astype(np.float64).T
    r2 = np.where(nev > 0, 1.0 - res / np.maximum(tot, 1e-12), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(r2, axis=1)


def stack_ss(evals, k):
    """Per-model sums of one method at k stacked over its draws: arrays
    (D, n), or None if k was not reached."""
    if any(k not in e["ss"] for e in evals):
        return None
    return {key: np.stack([e["ss"][k][key] for e in evals])
            for key in ("ss_res", "ss_tot", "n_eval")}


def bootstrap_pairs(prosp, pairs, ks, n_boot, seed, level=LEVEL):
    """Paired bootstrap over test models of the prospective split: dict of
    (P, len(ks)) arrays diff (full sample), mean, sd, lo, hi, prob_pos, n."""
    shape = (len(pairs), len(ks))
    out = {key: np.full(shape, np.nan) for key in ("diff", "mean", "sd",
                                                   "lo", "hi", "prob_pos",
                                                   "prob_zero")}
    out["n"] = np.zeros(shape, dtype=np.int64)
    if prosp["failed"]:
        return out
    n = prosp["n_val"]
    W = boot_weights(n, n_boot, np.random.default_rng(seed))
    ones = np.ones((1, n))
    q = [100 * (1 - level) / 2, 100 * (1 + level) / 2]
    for ik, k in enumerate(ks):
        cache = {}
        for m in sorted({m for p in pairs for m in p}):
            ss = stack_ss(prosp["methods"][m], k)
            cache[m] = None if ss is None else (boot_r2(W, ss),
                                                boot_r2(ones, ss)[0])
        for ip, (a, b) in enumerate(pairs):
            if cache[a] is None or cache[b] is None:
                continue
            d = cache[a][0] - cache[b][0]
            d = d[np.isfinite(d)]
            out["diff"][ip, ik] = cache[a][1] - cache[b][1]
            out["n"][ip, ik] = d.size
            if d.size:
                out["mean"][ip, ik] = d.mean()
                out["sd"][ip, ik] = d.std(ddof=1) if d.size > 1 else np.nan
                out["lo"][ip, ik], out["hi"][ip, ik] = np.percentile(d, q)
                out["prob_pos"][ip, ik] = np.mean(d > 0)
                out["prob_zero"][ip, ik] = np.mean(d == 0)
    return out


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(a, axis=axis)


def penalty_pval(prosp, ctrl):
    """(K,) one-sided permutation p-values (1 + #{ctrl_d <= prosp}) / (1 +
    #finite ctrl_d) per column of ctrl (C, K); NaN where prosp is NaN or
    no control value is finite.  prosp and every ctrl_d must be the same
    statistic (for random the mean over the same D draws) for
    the ranks to be exchangeable under the null."""
    out = np.full(prosp.shape, np.nan)
    for k in range(prosp.shape[0]):
        c = ctrl[:, k][np.isfinite(ctrl[:, k])]
        if np.isfinite(prosp[k]) and c.size:
            out[k] = (1 + np.sum(c <= prosp[k])) / (1 + c.size)
    return out


# ── One variant ──────────────────────────────────────────────────

def method_r2(split, m, key, k_max):
    """(k_max,) R² of method m on a split (stochastic: mean over draws)
    and its sd over draws (NaN for deterministic or failed)."""
    if split["failed"]:
        return np.full(k_max, np.nan), np.full(k_max, np.nan)
    a = np.stack([e[key] for e in split["methods"][m]]).astype(np.float64)
    if m in DETERMINISTIC:
        return a[0], np.full(k_max, np.nan)
    return _nanmean(a, axis=0), _nanstd(a, axis=0)


# Per-split quantities of every method (log keys {m}_{key}, prosp_{m}_{key})
SPLIT_KEYS = ("r2", "r2_c", "r2_all", "rmse", "n_models", "n_cells")


def draws_r2(split, m, k_max, n_draws):
    """(n_draws, k_max) R² of every draw of a stochastic method (NaN if
    the split failed)."""
    if split["failed"]:
        return np.full((n_draws, k_max), np.nan)
    return np.stack([e["r2"] for e in split["methods"][m]])


# Worker state of the control-split processes (--workers; spawn)
_CTRL = {}


def _init_ctrl_worker(work):
    _CTRL.clear()
    _CTRL.update(work)


def _ctrl_split(d, work=None):
    """Control split d of one variant (evaluate_split on control_split(M,
    n_train, d), mapped to the setting's columns); `work` (default the
    worker state) holds the variant's data and settings."""
    w = _CTRL if work is None else work
    tr, te = control_split(w["M"], w["n_tr"], d)
    return to_setting_columns(
        evaluate_split(w["B"], w["O"], tr, te, w["k_max"], w["bench"],
                       range(w["n_draws"]), w["target_min_obs"],
                       bounds=w["bounds"]), w["cols"], w["N"])


def control_splits(work, n_controls, workers=1):
    """The n_controls control-split results in order d = 0, 1, ...: in this
    process (workers <= 1), else in `workers` spawned single-threaded
    processes (ProcessPoolExecutor.map keeps the order).  Every control
    split is a pure function of the variant's data and d (its row
    permutation draw_seed(SEED + CONTROL_SPLIT_SEED_OFFSET, d), the draw
    seeds of the stochastic methods), so the results do not depend on
    workers (test_eval_prospective checks it)."""
    if workers <= 1 or n_controls <= 1:
        for d in range(n_controls):
            yield _ctrl_split(d, work)
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, n_controls),
                             mp_context=ctx, initializer=_init_ctrl_worker,
                             initargs=(work,)) as pool:
        yield from pool.map(_ctrl_split, range(n_controls))


def run_variant(v, benchmark_names, k_max, n_draws, n_controls, n_boot,
                target_min_obs=MIN_TARGET_OBS, label="", bounds=None,
                workers=1):
    """Prospective split, C control splits, comparisons and bootstrap of
    one variant; returns a dict of per-variant results (see save_log).
    benchmark_names are the setting's; the variant uses v.cols of them,
    and picks / eligible masks are mapped back to the setting's indices.
    bounds: the setting's valid score ranges (score_bounds.Bounds, rule
    D2; None: no clip), restricted to v.cols.  workers > 1 evaluates the
    control splits in that many processes (control_splits_parallel; the
    results are those of the serial loop).
    Every split, control splits included, runs the stochastic methods
    with all n_draws draws (seeds (tag, d), d = 0 .. n_draws − 1)."""
    M, Nv = v.B.shape
    N = len(benchmark_names)
    bench_v = np.asarray(benchmark_names)[v.cols]
    n_tr, n_te = v.n_train, v.n_test
    boot_ks = [k for k in BOOT_K if k <= k_max]
    vb = (None if bounds is None else
          (np.asarray(bounds[0])[v.cols], np.asarray(bounds[1])[v.cols]))
    t0 = time.time()
    print(f"\n  [{label}] {v.label}: {n_tr} train / {n_te} test, {Nv}/{N} "
          "benchmarks", end="", flush=True)
    prosp = evaluate_split(v.B, v.O, np.arange(n_tr), np.arange(n_tr, M),
                           k_max, bench_v, range(n_draws),
                           target_min_obs, keep_ks=boot_ks, crosscheck=True,
                           bounds=vb)
    to_setting_columns(prosp, v.cols, N)
    print(f", {prosp['n_eligible']}/{Nv} eligible; test cells "
          f"{prosp['n_test_cells']} ({prosp['n_test_cells_no_train']} "
          f"without training data, {prosp['n_test_cells_few_train']} with "
          f"< {target_min_obs} training models); max |R²_all − "
          f"impute_and_evaluate| {prosp.get('impute_check', np.nan):.2g}"
          + "".join(f" ({m} {prosp[f'impute_check_{m}']:.2g})"
                    for m in CHECK_APART if f"impute_check_{m}" in prosp)
          + f"  ({time.time() - t0:.1f}s)", flush=True)

    ctrl = []
    t1 = time.time()
    work = dict(B=v.B, O=v.O, M=M, n_tr=n_tr, k_max=k_max, bench=bench_v,
                n_draws=n_draws, target_min_obs=target_min_obs, bounds=vb,
                cols=v.cols, N=N)
    for d, res_d in enumerate(control_splits(work, n_controls, workers)):
        ctrl.append(res_d)
        if (d + 1) % DRAWS_PER_REPEAT == 0 or d + 1 == n_controls:
            print(f"    control splits {d + 1}/{n_controls} "
                  f"({time.time() - t1:.0f}s)", flush=True)

    res = {"variant": v, "prosp": prosp, "ctrl": ctrl, "k_max": k_max}
    for m in METHODS:
        for key in SPLIT_KEYS:
            mean, sd = method_r2(prosp, m, key, k_max)
            res[f"prosp_{m}_{key}"] = mean
            if key == "r2":
                res[f"prosp_{m}_r2_sd"] = sd
            res[f"ctrl_{m}_{key}"] = np.stack(
                [method_r2(c, m, key, k_max)[0] for c in ctrl]) \
                if ctrl else np.zeros((0, k_max))
        if m in STOCHASTIC:
            res[f"ctrl_{m}_r2_draws"] = (
                np.stack([draws_r2(c, m, k_max, n_draws) for c in ctrl])
                if ctrl else np.zeros((0, n_draws, k_max)))
        cr = res[f"ctrl_{m}_r2"]
        res[f"ctrl_{m}_mean"] = _nanmean(cr, axis=0)
        res[f"ctrl_{m}_sd"] = _nanstd(cr, axis=0)
        res[f"penalty_{m}"] = res[f"prosp_{m}_r2"] - res[f"ctrl_{m}_mean"]
        res[f"penalty_pval_{m}"] = penalty_pval(res[f"prosp_{m}_r2"], cr)
        res[f"ctrl_{m}_rmse_mean"] = _nanmean(res[f"ctrl_{m}_rmse"], axis=0)
        res[f"rmse_penalty_{m}"] = (res[f"prosp_{m}_rmse"]
                                    - res[f"ctrl_{m}_rmse_mean"])
        res[f"ctrl_{m}_n_models_mean"] = _nanmean(res[f"ctrl_{m}_n_models"],
                                                  axis=0)
        res[f"ctrl_{m}_r2_c_mean"] = _nanmean(res[f"ctrl_{m}_r2_c"], axis=0)
        res[f"penalty_c_{m}"] = (res[f"prosp_{m}_r2_c"]
                                 - res[f"ctrl_{m}_r2_c_mean"])

    boot = bootstrap_pairs(prosp, BOOT_PAIRS, boot_ks, n_boot,
                           SEED + BOOT_SEED_OFFSET)
    pad = len(BOOT_K) - len(boot_ks)
    for key, arr in boot.items():
        fill = 0 if arr.dtype.kind == "i" else np.nan
        res[f"boot_{key}"] = np.pad(arr, ((0, 0), (0, pad)),
                                    constant_values=fill)

    n_train = np.array([c["n_train"] for c in ctrl])
    n_val = np.array([c["n_val"] for c in ctrl])
    rep = np.arange(n_controls) // DRAWS_PER_REPEAT
    shape = (len(BOOT_PAIRS), len(BOOT_K))
    cp = {key: np.full(shape, np.nan) for key in ("diff", "se", "lo", "hi",
                                                  "win")}
    cp["n"] = np.zeros(shape, dtype=np.int64)
    for ip, (a, b) in enumerate(BOOT_PAIRS):
        for ik, k in enumerate(BOOT_K):
            if k > k_max or n_controls == 0:
                continue
            st = paired_stats(res[f"ctrl_{a}_r2"][:, k - 1],
                              res[f"ctrl_{b}_r2"][:, k - 1], n_train, n_val,
                              rep)
            cp["diff"][ip, ik] = st["mean_diff"]
            cp["se"][ip, ik] = st["se_nb"]
            cp["lo"][ip, ik] = st["ci_low"]
            cp["hi"][ip, ik] = st["ci_high"]
            cp["win"][ip, ik] = st["win_rate"]
            cp["n"][ip, ik] = st["n_splits"]
    for key, arr in cp.items():
        res[f"ctrl_pair_{key}"] = arr
    res["runtime_s"] = time.time() - t0
    return res


# ── Logs ─────────────────────────────────────────────────────────

def _stack(results, key):
    return np.stack([r[key] for r in results])


def build_log(tag, benchmark_names, results, args):
    """The npz arrays of one setting (module docstring)."""
    V = len(results)
    K = results[0]["k_max"]
    N = len(benchmark_names)
    T = max(r["variant"].n_test for r in results)
    nb = len(BOOT_K)
    log = {"benchmark_names": np.array([str(b) for b in benchmark_names]),
           "methods": np.array(METHODS),
           "variant_keys": np.array([r["variant"].key for r in results]),
           "variant_labels": np.array([r["variant"].label for r in results]),
           "variant_macros": np.array([r["variant"].macro for r in results]),
           "variant_cutoffs": np.array([r["variant"].cutoff
                                        for r in results])}
    cols = np.zeros((V, N), dtype=bool)
    for iv, r in enumerate(results):
        cols[iv, r["variant"].cols] = True
    log["variant_columns"] = cols
    log["variant_n_bench"] = cols.sum(axis=1).astype(np.int64)

    # Control split rows
    rows = [(iv, d, c) for iv, r in enumerate(results)
            for d, c in enumerate(r["ctrl"])]
    S = len(rows)
    log["variant"] = np.array([iv for iv, _, _ in rows], dtype=np.int64)
    log["draw"] = np.array([d for _, d, _ in rows], dtype=np.int64)
    log["rep"] = log["draw"] // DRAWS_PER_REPEAT
    log["fold"] = log["draw"] % DRAWS_PER_REPEAT
    log["split_id"] = log["draw"].copy()
    for key in ("n_train", "n_val", "n_eligible", "n_test_cells",
                "n_test_cells_no_train", "n_test_cells_few_train"):
        log[key] = np.array([c[key] for _, _, c in rows], dtype=np.int64)
    log["test_fill"] = np.array([c["test_fill"] for _, _, c in rows],
                                dtype=np.float64)
    log["eligible"] = (np.stack([c["eligible"] for _, _, c in rows])
                       if S else np.zeros((0, N), dtype=bool))
    log["estimation_failed"] = np.array([c["failed"] for _, _, c in rows],
                                        dtype=bool)
    D = args.n_draws
    for m in METHODS:
        for key in SPLIT_KEYS:
            log[f"{m}_{key}"] = (np.concatenate([r[f"ctrl_{m}_{key}"]
                                                 for r in results])
                                 if S else np.zeros((0, K)))
        if m in STOCHASTIC:
            log[f"{m}_r2_draws"] = (np.concatenate(
                [r[f"ctrl_{m}_r2_draws"] for r in results])
                if S else np.zeros((0, D, K)))
        # picks in order (stochastic methods: draw 0)
        sel = np.full((S, K), -1, dtype=np.int64)
        for i, (_, _, c) in enumerate(rows):
            if not c["failed"]:
                sel[i] = c["methods"][m][0]["selected"]
        log[f"{m}_selected"] = sel

    # Prospective split
    pros = [r["prosp"] for r in results]
    for key in ("n_train", "n_val", "n_eligible", "n_test_cells",
                "n_test_cells_no_train", "n_test_cells_few_train",
                "n_test_bench", "n_test_bench_no_train",
                "n_test_bench_few_train"):
        log[f"prosp_{key}"] = np.array([p[key] for p in pros],
                                       dtype=np.int64)
    log["prosp_eligible"] = np.stack([p["eligible"] for p in pros])
    log["prosp_estimation_failed"] = np.array([p["failed"] for p in pros])
    log["prosp_impute_check"] = np.array([p.get("impute_check", np.nan)
                                          for p in pros], dtype=np.float64)
    for m in CHECK_APART:
        if m in DETERMINISTIC:
            log[f"prosp_{m}_impute_check"] = np.array(
                [p.get(f"impute_check_{m}", np.nan) for p in pros],
                dtype=np.float64)
    log["prosp_test_fill"] = np.array([p["test_fill"] for p in pros],
                                      dtype=np.float64)
    infos = [r["variant"].info for r in results]
    log["prosp_n_test_dup"] = np.array(
        [i.get("duplicates", {}).get("n_dropped", 0) for i in infos],
        dtype=np.int64)
    log["prosp_n_test_old"] = np.array([i.get("n_test_old", -1)
                                        for i in infos], dtype=np.int64)
    for key in ("n_bench_future", "n_train_cells_future",
                "n_test_cells_future", "n_eligible_future"):
        name = key.replace("n_eligible", "n_elig")
        log[f"prosp_{name}"] = np.array(
            [i["time_mask"][key] if "time_mask" in i else -1 for i in infos],
            dtype=np.int64)
    for m in METHODS:
        for key in SPLIT_KEYS:
            log[f"prosp_{m}_{key}"] = _stack(results, f"prosp_{m}_{key}")
        if m in STOCHASTIC:
            log[f"prosp_{m}_r2_sd"] = _stack(results, f"prosp_{m}_r2_sd")
            draws = np.full((V, D, K), np.nan)
            sel = np.full((V, D, K), -1, dtype=np.int64)
            for iv, p in enumerate(pros):
                if not p["failed"]:
                    draws[iv] = np.stack([e["r2"] for e in p["methods"][m]])
                    sel[iv] = np.stack([e["selected"]
                                        for e in p["methods"][m]])
            log[f"prosp_{m}_r2_draws"] = draws
            log[f"prosp_{m}_selected"] = sel
            continue
        sel = np.full((V, K), -1, dtype=np.int64)
        ties = np.zeros((V, K), dtype=np.int64)
        tbs = np.zeros((V, K), dtype=np.int64)
        ss = {key: (np.full((V, nb, T), -1, dtype=np.int64)
                    if key.startswith("n_") else np.full((V, nb, T), np.nan))
              for key in PER_MODEL_SS}
        for iv, p in enumerate(pros):
            if p["failed"]:
                continue
            e = p["methods"][m][0]
            sel[iv], ties[iv], tbs[iv] = (e["selected"], e["index_ties"],
                                          e["tie_breaks"])
            for ik, k in enumerate(BOOT_K):
                if k in e["ss"]:
                    for key in ss:
                        ss[key][iv, ik, :p["n_val"]] = e["ss"][k][key]
        log[f"prosp_{m}_selected"] = sel
        log[f"prosp_{m}_index_ties"] = ties
        log[f"prosp_{m}_tie_breaks"] = tbs
        for key, arr in ss.items():
            log[f"prosp_{m}_{key}"] = arr
    per_k = np.full((V, K, K), -1, dtype=np.int64)
    for iv, p in enumerate(pros):
        if not p["failed"]:
            for kk, s in enumerate(p["methods"]["pca"][0].get(
                    "selected_per_k", [])):
                per_k[iv, kk] = s
    log["prosp_pca_selected_per_k"] = per_k
    names = np.full((V, T), "", dtype=object)
    dates = np.full((V, T), "", dtype=object)
    for iv, r in enumerate(results):
        v = r["variant"]
        names[iv, :v.n_test] = v.model_names[v.n_train:]
        dates[iv, :v.n_test] = v.model_dates[v.n_train:]
    log["test_model_names"] = names.astype(str)
    log["test_model_dates"] = dates.astype(str)

    # Comparisons, bootstrap, control pairs
    for m in METHODS:
        for key in (f"ctrl_{m}_mean", f"ctrl_{m}_sd", f"penalty_{m}",
                    f"penalty_pval_{m}", f"ctrl_{m}_rmse_mean",
                    f"rmse_penalty_{m}", f"ctrl_{m}_n_models_mean",
                    f"ctrl_{m}_r2_c_mean", f"penalty_c_{m}"):
            log[key] = _stack(results, key)
    for key in ("diff", "mean", "sd", "lo", "hi", "prob_pos", "prob_zero",
                "n"):
        log[f"boot_{key}"] = _stack(results, f"boot_{key}")
    for key in ("diff", "se", "lo", "hi", "win", "n"):
        log[f"ctrl_pair_{key}"] = _stack(results, f"ctrl_pair_{key}")
    log["boot_pairs"] = np.array([f"{a}-{b}" for a, b in BOOT_PAIRS])
    log["boot_k"] = np.array(BOOT_K, dtype=np.int64)
    log.update(seed=np.int64(SEED), n_draws=np.int64(D),
               n_controls=np.int64(args.n_controls),
               n_boot=np.int64(args.boot), k_max=np.int64(K),
               min_train_obs=np.int64(MIN_TRAIN_OBS),
               min_target_obs=np.int64(cv_split.MIN_TARGET_OBS),
               pred_clip=np.bool_(score_bounds.PRED_CLIP),
               target_min_obs=np.int64(args.target_min_obs),
               min_models=np.int64(args.min_models),
               prosp_random_seed_offset=np.int64(PROSP_RANDOM_SEED_OFFSET),
               control_split_seed_offset=np.int64(CONTROL_SPLIT_SEED_OFFSET),
               boot_seed_offset=np.int64(BOOT_SEED_OFFSET))
    return log


def _jsonable(obj):
    """numpy -> native Python for strict JSON; non-finite floats -> None."""
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


def coverage(log, iv, m, k):
    """(prospective, control-mean) contributing test models of method m
    at k on variant iv (stochastic: the mean over the draws)."""
    return (float(log[f"prosp_{m}_n_models"][iv, k - 1]),
            float(log[f"ctrl_{m}_n_models_mean"][iv, k - 1]))


def make_macros(tag, log, checks, suppressed=None):
    """Flat {MacroName: number} dict of one setting (module docstring);
    non-finite values are left out, and so are the statistics that rest
    on fewer than log["min_models"] contributing test models (their
    names are appended to `suppressed` if given)."""
    macros = OrderedDict()
    withheld = [] if suppressed is None else suppressed
    min_models = int(log["min_models"]) if "min_models" in log \
        else MIN_MODELS

    def put(name, value, n_models=None):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"bad macro name {name}")
        if name in macros or name in withheld:
            raise ValueError(f"duplicate macro {name}")
        if value is None:
            return
        value = float(value) if not isinstance(value, (int, np.integer)) \
            else int(value)
        if isinstance(value, float) and not np.isfinite(value):
            return
        if n_models is not None and not (np.isfinite(n_models)
                                         and n_models >= min_models):
            withheld.append(name)
            return
        macros[name] = value

    base = SETTING_MACRO[tag]
    K = int(log["k_max"])
    put(f"MinModelsProsp{base}", min_models)
    for iv, vm in enumerate(log["variant_macros"]):
        S = base + str(vm)
        for key, name in (("prosp_n_train", "NTrain"),
                          ("prosp_n_val", "NTest"),
                          ("prosp_n_eligible", "NElig"),
                          ("prosp_n_test_cells", "NTestCells"),
                          ("prosp_n_test_cells_no_train",
                           "NTestCellsNoTrain"),
                          ("prosp_n_test_cells_few_train",
                           "NTestCellsFewTrain"),
                          ("prosp_n_test_bench_no_train",
                           "NTestBenchNoTrain"),
                          ("prosp_n_test_bench_few_train",
                           "NTestBenchFewTrain"),
                          ("variant_n_bench", "NBench"),
                          ("prosp_n_test_dup", "NTestDup")):
            put(f"{name}Prosp{S}", int(log[key][iv]))
        # -1: not defined for this setting
        for key, name in (("prosp_n_test_old", "NTestOld"),
                          ("prosp_n_bench_future", "NBenchFuture"),
                          ("prosp_n_train_cells_future",
                           "NTrainCellsFuture"),
                          ("prosp_n_test_cells_future", "NTestCellsFuture"),
                          ("prosp_n_elig_future", "NEligFuture")):
            if int(log[key][iv]) >= 0:
                put(f"{name}Prosp{S}", int(log[key][iv]))
        put(f"NCtrlProsp{S}", int(np.sum(log["variant"] == iv)))
        put(f"NDrawsProsp{S}", int(log["n_draws"]))
        put(f"FillTestProsp{S}", log["prosp_test_fill"][iv])
        put(f"FillTestCtrl{S}",
            _nanmean(log["test_fill"][log["variant"] == iv]))
        for m in METHODS:
            M = camel(m)
            for k in MACRO_K:
                if k > K:
                    continue
                Kw = "K" + int_words(k)
                n_p, n_c = coverage(log, iv, m, k)
                n_pc = min(n_p, n_c)
                put(f"RsqProsp{S}{M}{Kw}", log[f"prosp_{m}_r2"][iv, k - 1],
                    n_p)
                if m in STOCHASTIC:
                    put(f"RsqStdProsp{S}{M}{Kw}",
                        log[f"prosp_{m}_r2_sd"][iv, k - 1], n_p)
                put(f"RsqCtrl{S}{M}{Kw}", log[f"ctrl_{m}_mean"][iv, k - 1],
                    n_c)
                put(f"RsqStdCtrl{S}{M}{Kw}", log[f"ctrl_{m}_sd"][iv, k - 1],
                    n_c)
                put(f"PenaltyProsp{S}{M}{Kw}", log[f"penalty_{m}"][iv, k - 1],
                    n_pc)
                put(f"PenaltyPvalProsp{S}{M}{Kw}",
                    log[f"penalty_pval_{m}"][iv, k - 1], n_pc)
                put(f"RmseProsp{S}{M}{Kw}", log[f"prosp_{m}_rmse"][iv, k - 1],
                    n_p)
                put(f"RmseCtrl{S}{M}{Kw}",
                    log[f"ctrl_{m}_rmse_mean"][iv, k - 1], n_c)
                put(f"RmsePenaltyProsp{S}{M}{Kw}",
                    log[f"rmse_penalty_{m}"][iv, k - 1], n_pc)
                put(f"NModelsProsp{S}{M}{Kw}", n_p)
                put(f"NModelsCtrl{S}{M}{Kw}", n_c)
                put(f"RsqCenProsp{S}{M}{Kw}",
                    log[f"prosp_{m}_r2_c"][iv, k - 1], n_p)
                put(f"RsqCenCtrl{S}{M}{Kw}",
                    log[f"ctrl_{m}_r2_c_mean"][iv, k - 1], n_c)
                put(f"PenaltyCenProsp{S}{M}{Kw}",
                    log[f"penalty_c_{m}"][iv, k - 1], n_pc)
        for ip, (a, b) in enumerate(BOOT_PAIRS):
            P = f"{camel(a)}Vs{camel(b)}"
            for ik, k in enumerate(BOOT_K):
                if k > K:
                    continue
                Kw = "K" + int_words(k)
                (pa, ca), (pb, cb) = coverage(log, iv, a, k), \
                    coverage(log, iv, b, k)
                for stat, key in (("Diff", "diff"), ("Lo", "lo"),
                                  ("Hi", "hi"), ("ProbPos", "prob_pos")):
                    put(f"Boot{stat}Prosp{S}{P}{Kw}",
                        log[f"boot_{key}"][iv, ip, ik], min(pa, pb))
                for stat, key in (("Diff", "diff"), ("Lo", "lo"),
                                  ("Hi", "hi")):
                    put(f"Pair{stat}Ctrl{S}{P}{Kw}",
                        log[f"ctrl_pair_{key}"][iv, ip, ik], min(ca, cb))

    def put_check(prefix, c):
        put(f"{prefix}MaxAbsDiff", c["max_abs_diff"])
        put(f"{prefix}NRows", c["n_rows"])
        put(f"{prefix}NRowsDiffer", c["n_rows_differ"])
        put(f"{prefix}NRowsObsMismatch", c["n_rows_obs_mismatch"])
        put(f"{prefix}NCells", c["n_cells_compared"])
        put(f"{prefix}NCellsDiffer", c["n_cells_differ"])

    if "mteb_snapshot" in checks:
        put_check("ProspCheckMtebSnap", checks["mteb_snapshot"])
        put("ProspCheckMtebSnapNRenamed", checks["mteb_snapshot"]["n_renamed"])
    if "helm_npz" in checks:
        put_check("ProspCheckHelmSnap", checks["helm_npz"])
        put_check("ProspCheckHelmMerged", checks["merged"])
    if "math_hard_dropped" in checks:
        put(f"NDroppedMathProsp{base}", len(checks["math_hard_dropped"]))
    return macros


def summary_json(tag, log, results, checks, args, runtime):
    """The JSON log of one setting."""
    K = int(log["k_max"])
    ks = [k for k in MACRO_K if k <= K]
    mm = args.min_models
    variants = []
    for iv, r in enumerate(results):
        v = r["variant"]
        table = {}
        for m in METHODS:
            table[m] = {f"k{k}": {
                "low_coverage": coverage(log, iv, m, k)[0] < mm,
                "control_low_coverage": coverage(log, iv, m, k)[1] < mm,
                "prospective": log[f"prosp_{m}_r2"][iv, k - 1],
                "prospective_sd_draws": (log[f"prosp_{m}_r2_sd"][iv, k - 1]
                                         if m in STOCHASTIC else None),
                "prospective_all_targets": log[f"prosp_{m}_r2_all"][iv,
                                                                   k - 1],
                "control_mean": log[f"ctrl_{m}_mean"][iv, k - 1],
                "control_sd": log[f"ctrl_{m}_sd"][iv, k - 1],
                "penalty": log[f"penalty_{m}"][iv, k - 1],
                "penalty_pval": log[f"penalty_pval_{m}"][iv, k - 1],
                "prospective_rmse": log[f"prosp_{m}_rmse"][iv, k - 1],
                "control_rmse_mean": log[f"ctrl_{m}_rmse_mean"][iv, k - 1],
                "rmse_penalty": log[f"rmse_penalty_{m}"][iv, k - 1],
                "prospective_n_models": log[f"prosp_{m}_n_models"][iv,
                                                                  k - 1],
                "prospective_n_cells": log[f"prosp_{m}_n_cells"][iv, k - 1],
                "control_n_models_mean": log[f"ctrl_{m}_n_models_mean"][
                    iv, k - 1],
                "prospective_r2_centered": log[f"prosp_{m}_r2_c"][iv, k - 1],
                "control_r2_centered_mean": log[f"ctrl_{m}_r2_c_mean"][
                    iv, k - 1],
                "penalty_centered": log[f"penalty_c_{m}"][iv, k - 1]}
                for k in ks}
        boot, cpair = [], []
        for ip, (a, b) in enumerate(BOOT_PAIRS):
            for ik, k in enumerate(BOOT_K):
                if k > K:
                    continue
                (pa, ca), (pb, cb) = coverage(log, iv, a, k), \
                    coverage(log, iv, b, k)
                boot.append({"a": a, "b": b, "k": k, **{
                    key: log[f"boot_{key}"][iv, ip, ik]
                    for key in ("diff", "mean", "sd", "lo", "hi",
                                "prob_pos", "prob_zero", "n")},
                    "n_models_a": pa, "n_models_b": pb,
                    "low_coverage": min(pa, pb) < mm})
                cpair.append({"a": a, "b": b, "k": k, **{
                    key: log[f"ctrl_pair_{key}"][iv, ip, ik]
                    for key in ("diff", "se", "lo", "hi", "win", "n")},
                    "n_models_a": ca, "n_models_b": cb,
                    "low_coverage": min(ca, cb) < mm})
        p = r["prosp"]
        in_v = log["variant"] == iv
        variants.append({
            "key": v.key, "label": v.label, "macro": SETTING_MACRO[tag]
            + v.macro, "cutoff": v.cutoff, "n_train": v.n_train,
            "n_test": v.n_test, "n_bench": int(len(v.cols)),
            "benchmarks": [str(log["benchmark_names"][j]) for j in v.cols],
            "n_eligible": p["n_eligible"],
            "test_fill": p["test_fill"],
            "control_test_fill_mean": _nanmean(log["test_fill"][in_v]),
            "estimation_failed": p["failed"],
            "eligible": [str(b) for b, e in zip(log["benchmark_names"],
                                                p["eligible"]) if e],
            "targets": {key: p[key] for key in (
                "n_test_cells", "n_test_cells_no_train",
                "n_test_cells_few_train", "n_test_bench",
                "n_test_bench_no_train", "n_test_bench_few_train")},
            "selected": {m: [str(log["benchmark_names"][j])
                             for j in log[f"prosp_{m}_selected"][iv] if j >= 0]
                         for m in DETERMINISTIC},
            "test_dates": ([min(v.model_dates[v.n_train:]),
                            max(v.model_dates[v.n_train:])]
                           if v.n_test else None),
            "train_dates": ([min(v.model_dates[:v.n_train]),
                             max(v.model_dates[:v.n_train])]
                            if v.n_train else None),
            "control_estimation_failed": int(np.sum(
                [c["failed"] for c in r["ctrl"]])),
            "impute_check_max_abs_diff": p.get("impute_check"),
            **{f"{m}_impute_check_max_abs_diff": p.get(f"impute_check_{m}")
               for m in CHECK_APART if m in DETERMINISTIC},
            "info": v.info, "r2": table, "bootstrap": boot,
            "control_pairs": cpair, "runtime_s": r["runtime_s"]})
    suppressed = []
    macros = make_macros(tag, log, checks, suppressed)
    return {
        "script": "eval_prospective.py", "setting": tag,
        "label": SETTING_LABEL[tag],
        "config": {"seed": SEED, "n_draws": args.n_draws,
                   "n_controls": args.n_controls, "n_boot": args.boot,
                   "control_draws": "every control split runs all n_draws "
                                    "draws of the stochastic methods, the "
                                    "seeds of the prospective split",
                   "k_max": K, "min_train_obs": MIN_TRAIN_OBS,
                   "min_target_obs": int(cv_split.MIN_TARGET_OBS),
                   "pred_clip": bool(score_bounds.PRED_CLIP),
                   "bounds": "score_bounds.dataset_bounds of the setting's "
                             "data (data/score_bounds.json)",
                   "target_min_obs": args.target_min_obs,
                   "min_models": args.min_models,
                   "boot_k": BOOT_K, "boot_pairs": [list(p)
                                                    for p in BOOT_PAIRS],
                   "methods": METHODS, "level": LEVEL, "ridge": RIDGE,
                   "seed_tags": {
                       "random": SEED + PROSP_RANDOM_SEED_OFFSET,
                       "control_split": SEED + CONTROL_SPLIT_SEED_OFFSET,
                       "bootstrap": SEED + BOOT_SEED_OFFSET},
                   "seed_rule": "draw_seed(tag, d) = (tag, d); bootstrap "
                                "default_rng(SEED + 16) per variant",
                   "epoch_cutoffs": args.epoch_cutoffs,
                   "olv2sub_cutoffs": args.olv2sub_cutoffs,
                   "benchmark_names": [str(b)
                                       for b in log["benchmark_names"]]},
        "checks": checks, "variants": variants, "runtime_s": runtime,
        "macros_suppressed": suppressed, "macros": macros}


def save_log(tag, log, summary):
    path = LOGS / f"prospective_{tag}.npz"
    np.savez(path, **log)
    jpath = LOGS / f"prospective_{tag}.json"
    tmp = jpath.with_name(f".{jpath.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(_jsonable(summary), f, indent=1, allow_nan=False)
        f.write("\n")
    os.replace(tmp, jpath)
    print(f"Results saved to {path} and {jpath} "
          f"({len(summary['macros'])} macros, "
          f"{len(summary['macros_suppressed'])} withheld for coverage < "
          f"{summary['config']['min_models']} test models)", flush=True)


# ── Reporting ────────────────────────────────────────────────────

def print_checks(tag, checks):
    def show(name, c):
        print(f"  {name}: {c['n_rows']} rows, {c['n_cells_compared']} cells "
              f"compared, max |diff| {c['max_abs_diff']:.3g}; "
              f"{c['n_cells_differ']} cells in {c['n_rows_differ']} rows "
              f"differ by > {DIFF_TOL:g}; observation pattern differs in "
              f"{c['n_rows_obs_mismatch']} rows ({c['n_obs_only_ref']} "
              f"cells only in the reference, {c['n_obs_only_new']} only in "
              "the new matrix)")
        for r in c["rows"][:20]:
            ref = (f" (= {r['ref_name']})" if r["ref_name"] != r["name"]
                   else "")
            print(f"     {r['name']}{ref}: {r['n_cells_differ']} cells, max "
                  f"{r['max_abs_diff']:.3g}; obs only ref "
                  f"{r['n_obs_only_ref']}, only new {r['n_obs_only_new']}")
    if "mteb_snapshot" in checks:
        c = checks["mteb_snapshot"]
        print(f"\n== Check: mteb2026 in_snapshot rows vs data/mteb.matrix.npz "
              f"({c['n_renamed']} renamed) ==")
        show("mteb2026", c)
        if c["in_snapshot_names_not_in_matrix"] or \
                c["matrix_rows_not_covered"]:
            print("  unmatched: "
                  f"{c['in_snapshot_names_not_in_matrix']} / "
                  f"{c['matrix_rows_not_covered']}")
    if "helm_npz" in checks:
        print("\n== Check: helm_lite in_snapshot rows vs data/helm.npz and "
              "merged helm/* ==")
        show("helm.npz", checks["helm_npz"])
        show("merged", checks["merged"])
        print(f"  not compared: helm.npz "
              f"{checks['helm_npz']['reference_columns_not_compared']}, "
              f"merged {checks['merged']['reference_columns_not_compared']}")
    if checks.get("math_hard_dropped"):
        print(f"\n  olv2sub: dropped {len(checks['math_hard_dropped'])} "
              f"math_hard/* columns ({checks['math_hard_reason']})")


def print_summary(tag, log, results):
    K = int(log["k_max"])
    ks = [k for k in MACRO_K if k <= K]
    mm = int(log["min_models"])
    for iv, r in enumerate(results):
        v = r["variant"]
        print(f"\n== {SETTING_LABEL[tag]} — {v.label}: {v.n_train} train / "
              f"{v.n_test} test, {len(v.cols)} benchmarks, "
              f"{int(log['prosp_n_eligible'][iv])} eligible;"
              f" R² prospective / random-split control mean (penalty); "
              f"* = fewer than {mm} contributing test models ==")
        if v.info.get("math_hard_task_version_counts"):
            print(f"   MATH scorer versions: "
                  f"{v.info['math_hard_task_version_counts']}")
        tm = v.info.get("time_mask")
        if tm:
            what = "dropped" if tm["applied"] else "KEPT: look-ahead"
            print(f"   benchmarks released on or after {v.cutoff}: "
                  f"{tm['n_bench_future']} ({what}); "
                  f"their training cells {tm['n_train_cells_future']}, "
                  f"test cells {tm['n_test_cells_future']}, eligible on the "
                  f"training rows {tm['n_eligible_future']}; superseded "
                  f"cells missing {tm['superseded_missing_cells']}")
        dup = v.info.get("duplicates", {})
        if dup.get("n_dropped"):
            print(f"   dropped {dup['n_dropped']} test rows duplicating "
                  "training rows: " + ", ".join(
                      f"{d['test']} (= {d['train']}, {d['reason']})"
                      for d in dup["rows"]))
        rel = v.info.get("test_release")
        if rel:
            print(f"   new to the leaderboard but not in time: "
                  f"{rel['n_old']} of {rel['n_test']} ({rel['rule']})")
        print(f"   test-cell fill: prospective "
              f"{log['prosp_test_fill'][iv]:.3f}, control mean "
              f"{_nanmean(log['test_fill'][log['variant'] == iv]):.3f}")
        print(f"{'method':<24}" + "".join(f"k={k:<20}" for k in ks))
        for m in METHODS:
            cells = []
            for k in ks:
                p = log[f"prosp_{m}_r2"][iv, k - 1]
                c = log[f"ctrl_{m}_mean"][iv, k - 1]
                star = "*" if coverage(log, iv, m, k)[0] < mm else ""
                cells.append(f"{p:.3f}/{c:.3f} ({p - c:+.3f}){star}")
            print(f"{METHOD_LABEL[m]:<24}" + "".join(f"{c:<22}"
                                                     for c in cells))
        print("   shift-adjusted R² (SS_tot about the test-set means) "
              "prospective / control (penalty):")
        for m in METHODS:
            cells = []
            for k in ks:
                p = log[f"prosp_{m}_r2_c"][iv, k - 1]
                c = log[f"ctrl_{m}_r2_c_mean"][iv, k - 1]
                cells.append(f"{p:.3f}/{c:.3f} ({p - c:+.3f})")
            print(f"   {m:<21}" + "".join(f"{c:<22}" for c in cells))
        print("   RMSE (train-std units) prospective / control mean:")
        for m in METHODS:
            cells = [f"{log[f'prosp_{m}_rmse'][iv, k - 1]:.3f}/"
                     f"{log[f'ctrl_{m}_rmse_mean'][iv, k - 1]:.3f}"
                     for k in ks]
            print(f"   {m:<21}" + "".join(f"{c:<22}" for c in cells))
        print(f"   contributing test models (of {v.n_test}) prospective / "
              "control mean:")
        for m in METHODS:
            cells = [f"{log[f'prosp_{m}_n_models'][iv, k - 1]:.0f}/"
                     f"{log[f'ctrl_{m}_n_models_mean'][iv, k - 1]:.1f}"
                     for k in ks]
            print(f"   {m:<21}" + "".join(f"{c:<22}" for c in cells))
        n_ex = (int(log["prosp_n_test_cells_no_train"][iv])
                + int(log["prosp_n_test_cells_few_train"][iv]))
        if n_ex:
            print(f"   targets: {int(log['prosp_n_test_cells'][iv])} test "
                  f"cells; {int(log['prosp_n_test_cells_no_train'][iv])} in "
                  f"{int(log['prosp_n_test_bench_no_train'][iv])} benchmarks "
                  "without training data and "
                  f"{int(log['prosp_n_test_cells_few_train'][iv])} in "
                  f"{int(log['prosp_n_test_bench_few_train'][iv])} with too "
                  "few are not targets")
        print(f"   paired bootstrap over test models (B = {int(log['n_boot'])}"
              f"): diff [95% CI] P(diff > 0)")
        for ip, (a, b) in enumerate(BOOT_PAIRS):
            cells = []
            for ik, k in enumerate(BOOT_K):
                if k > K:
                    continue
                low = min(coverage(log, iv, a, k)[0],
                          coverage(log, iv, b, k)[0]) < mm
                cells.append(f"k={k} {log['boot_diff'][iv, ip, ik]:+.3f} "
                             f"[{log['boot_lo'][iv, ip, ik]:+.3f}, "
                             f"{log['boot_hi'][iv, ip, ik]:+.3f}] "
                             f"{log['boot_prob_pos'][iv, ip, ik]:.2f}"
                             + ("*" if low else ""))
            print(f"   {a + ' − ' + b:<18}" + "  ".join(cells))


# ── Plots ────────────────────────────────────────────────────────

# Figure rows: (prospective key, control-mean key, y label)
PLOT_ROWS = [("prosp_{m}_r2", "ctrl_{m}_mean",
              "Imputation $R^2$ (new models)"),
             ("prosp_{m}_r2_c", "ctrl_{m}_r2_c_mean",
              "Shift-adjusted $R^2$")]


def plot_panel(ax, log, iv, methods, title=None, row=0):
    """R² (row 0) or shift-adjusted R² (row 1) vs k of variant iv:
    prospective solid with markers (hollow: fewer than min_models
    contributing test models), control mean dashed.  Methods the log does
    not have (a log written before facloc was added) are skipped."""
    K = int(log["k_max"])
    ks = np.arange(1, K + 1)
    p_key, c_key, _ = PLOT_ROWS[row]
    mm = int(log["min_models"]) if "min_models" in log else MIN_MODELS
    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for m in methods:
            if p_key.format(m=m) not in log:
                continue
            c = METHOD_COLOR[m]
            lw = 1.6 if m in MAIN_METHODS else 1.1
            y = log[p_key.format(m=m)][iv]
            low = log[f"prosp_{m}_n_models"][iv] < mm
            ax.plot(ks, y, "-", color=c, linewidth=lw)
            for sel, face in ((~low, c), (low, "white")):
                ax.plot(ks[sel], y[sel], linestyle="none",
                        marker=METHOD_MARKER[m], color=c,
                        markerfacecolor=face, markersize=3.5)
            ax.plot(ks, log[c_key.format(m=m)][iv], "--", color=c,
                    linewidth=0.9, alpha=0.85)
            vals += [y, log[c_key.format(m=m)][iv]]
    if title is not None:
        n_tr = int(log["prosp_n_train"][iv])
        n_te = int(log["prosp_n_val"][iv])
        nb = (f", {int(log['variant_n_bench'][iv])} benchmarks"
              if "variant_n_bench" in log else "")
        ax.set_title(f"{title}\n{n_tr} old → {n_te} new models{nb}",
                     fontsize=9)
    ax.set_xlim(0.5, K + 0.5)
    ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= K])
    ax.grid(True, alpha=0.3)
    # A few outliers (e.g. a k = 1 pick that few new models report) must
    # not flatten the panel: cut the axis below the 10th percentile of its
    # values minus 0.1 (never above -0.5); R² never exceeds 1
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    lo, hi = ax.get_ylim()
    if v.size:
        lo = max(lo, min(-0.5, float(np.percentile(v, 10)) - 0.1))
    ax.set_ylim(lo, min(hi, 1.02))


def plot_grid(panels, methods, path):
    """Two rows (R², shift-adjusted R²) × one column per (log, variant
    index, title) of `panels`; shared legend below."""
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(max(3.8 * n, 5.6), 6.4),
                             squeeze=False, sharex="col")
    for j, (log, iv, title) in enumerate(panels):
        for row in range(2):
            plot_panel(axes[row, j], log, iv, methods,
                       title if row == 0 else None, row)
        axes[1, j].set_xlabel("Selected benchmarks $k$")
    for row in range(2):
        axes[row, 0].set_ylabel(PLOT_ROWS[row][2])
    fig.tight_layout()
    # Legend just below the axes; savefig(bbox_inches="tight") keeps it
    methods = [m for m in methods
               if any(f"prosp_{m}_r2" in log for log, _, _ in panels)]
    fig.legend(handles=legend_handles(methods), loc="upper center",
               ncol=min(len(methods) + 2, max(4, 2 * n)), fontsize=7,
               bbox_to_anchor=(0.5, 0.0))
    _savefig(fig, path)


def legend_handles(methods):
    hs = [Line2D([], [], color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                 markersize=4, label=METHOD_LABEL[m]) for m in methods]
    hs += [Line2D([], [], color="black", linestyle="-",
                  label="prospective (old → new)"),
           Line2D([], [], color="black", linestyle="--",
                  label="random-split control"),
           Line2D([], [], color="black", linestyle="none", marker="o",
                  markerfacecolor="white", markersize=4,
                  label="few contributing new models")]
    return hs


def _savefig(fig, path):
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.pdf")
    fig.savefig(tmp, bbox_inches="tight", dpi=150)
    os.replace(tmp, path)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)


def plot_setting(tag, log, methods=PLOT_METHODS):
    panels = [(log, iv, f"{SETTING_LABEL[tag]}:\n{log['variant_labels'][iv]}")
              for iv in range(len(log["variant_keys"]))]
    plot_grid(panels, methods, OUT / f"prospective_{tag}.pdf")


def primary_index(tag, log):
    keys = list(log["variant_keys"])
    want = PRIMARY_VARIANT.get(tag)
    return keys.index(want) if want in keys else len(keys) // 2


def load_log(tag):
    path = LOGS / f"prospective_{tag}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def plot_combined(tags=DEFAULT_SETTINGS, methods=PLOT_METHODS):
    """figures/prospective.pdf: the primary variant of each setting with a
    log (all of `tags` required)."""
    logs = {t: load_log(t) for t in tags}
    missing = [t for t, lg in logs.items() if lg is None]
    if missing:
        print(f"Combined figure skipped: no log for {', '.join(missing)}",
              flush=True)
        return
    design = {(int(lg["n_draws"]), int(lg["n_controls"]),
               int(lg["target_min_obs"]),
               int(lg.get("min_models", MIN_MODELS)))
              for lg in logs.values()}
    if len(design) > 1:
        print(f"warning: logs of different designs in the combined figure: "
              f"{sorted(design)}", flush=True)
    panels = []
    for t in tags:
        lg = logs[t]
        iv = primary_index(t, lg)
        cut = str(lg["variant_cutoffs"][iv])
        panels.append((lg, iv, SETTING_LABEL[t]
                       + (f"\ncutoff {cut}" if cut else "")))
    plot_grid(panels, methods, OUT / "prospective.pdf")


# ── Main ─────────────────────────────────────────────────────────

def run_setting(tag, args):
    t0 = time.time()
    print(f"\n{'#'*70}\n# Prospective: {tag} ({SETTING_LABEL[tag]})\n"
          f"{'#'*70}", flush=True)
    bench, variants, checks = LOADERS[tag](args)
    print_checks(tag, checks)
    K = min(args.kmax, len(bench))
    bounds = dataset_bounds(tag, bench, DATA)          # D2, the setting's data
    results = [run_variant(v, bench, K, args.n_draws, args.n_controls,
                           args.boot, args.target_min_obs, label=tag,
                           bounds=bounds, workers=args.workers)
               for v in variants]
    log = build_log(tag, bench, results, args)
    runtime = time.time() - t0
    summary = summary_json(tag, log, results, checks, args, runtime)
    print_summary(tag, log, results)
    save_log(tag, log, summary)
    plot_setting(tag, log, args.plot_methods)
    print(f"\n{tag} completed in {runtime:.0f}s", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prospective (temporal) evaluation: selection on old "
                    "models, imputation of new models, with random-split "
                    "controls and a paired bootstrap.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_SETTINGS,
                        help="prospective settings "
                             f"({', '.join(ALL_SETTINGS)}; helm_lite = "
                             f"helm; default: {' '.join(DEFAULT_SETTINGS)})")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="10 x repeats draws of random and "
                             "random-split control splits (default "
                             "%(default)s: 50)")
    parser.add_argument("--kmax", type=int, default=K_MAX)
    parser.add_argument("--boot", type=int, default=N_BOOT,
                        help="bootstrap resamples of the test models "
                             "(default %(default)s)")
    parser.add_argument("--epoch-cutoffs", nargs="+", default=EPOCH_CUTOFFS,
                        help="release-date cutoffs YYYY-MM-DD of the epoch "
                             "setting")
    parser.add_argument("--olv2sub-cutoffs", nargs="+",
                        default=OLV2SUB_CUTOFFS,
                        help="submission-date cutoffs YYYY-MM-DD of olv2sub")
    parser.add_argument("--target-min-obs", type=int, default=MIN_TARGET_OBS,
                        help="targets: test cells of benchmarks observed for "
                             "at least max(this, MIN_TARGET_OBS) training "
                             "models (default %(default)s = cv_split."
                             "MIN_TARGET_OBS, impute_and_evaluate's rule D1; "
                             "3 = MIN_TRAIN_OBS, the default before "
                             "2026-10-18)")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes evaluating the control splits "
                             "(spawn, single-threaded each; results do not "
                             "depend on it; default %(default)s)")
    parser.add_argument("--min-models", type=int, default=MIN_MODELS,
                        help="contributing test models a statistic needs "
                             "for its macro (default %(default)s; the npz "
                             "and JSON keep every value)")
    parser.add_argument("--plot-entropy-rf", action="store_true",
                        help="ignored: entropy_rf was dropped (protocol "
                             "rule D3); kept so old command lines run")
    parser.add_argument("--plot-only", action="store_true",
                        help="rebuild the figures from the logs only")
    args = parser.parse_args(argv)
    args.datasets = [SETTING_ALIASES.get(d, d) for d in args.datasets]
    bad = [d for d in args.datasets if d not in ALL_SETTINGS]
    if bad:
        parser.error(f"unknown setting(s) {bad}; choose from {ALL_SETTINGS}")
    args.datasets = list(OrderedDict.fromkeys(args.datasets))
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.kmax < 1:
        parser.error("--kmax must be at least 1")
    if args.boot < 1:
        parser.error("--boot must be at least 1")
    if args.target_min_obs < 0:
        parser.error("--target-min-obs must be >= 0")
    if args.min_models < 1:
        parser.error("--min-models must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    for c in args.epoch_cutoffs + args.olv2sub_cutoffs:
        if not _DATE.match(c):
            parser.error(f"cutoff {c} is not YYYY-MM-DD")
    args.n_draws = args.n_controls = DRAWS_PER_REPEAT * args.repeats
    if args.plot_entropy_rf:
        print("note: --plot-entropy-rf is ignored (entropy_rf was dropped, "
              "protocol rule D3)", flush=True)
    args.plot_methods = list(PLOT_METHODS)
    return args


if __name__ == "__main__":
    args = parse_args()
    total_t0 = time.time()
    if args.plot_only:
        for tag in args.datasets:
            log = load_log(tag)
            if log is None:
                print(f"no log for {tag}", flush=True)
                continue
            plot_setting(tag, log, args.plot_methods)
        plot_combined(methods=args.plot_methods)
        sys.exit(0)
    for tag in args.datasets:
        run_setting(tag, args)
    if all(t in args.datasets for t in DEFAULT_SETTINGS):
        plot_combined(methods=args.plot_methods)
    print(f"\nAll settings completed in {time.time() - total_t0:.0f}s",
          flush=True)
