r"""Fitted matrix-completion imputers vs the Gaussian conditional at fixed
selections (reviewer request B1: CBK3's matrix-completion alternatives,
QZ19's "does the linear framework hit a ceiling?").

On the matched splits of cv_split.cv_splits (10% holdout, --repeats x
--folds splits) every selection method of METHODS (mi, entropy, trace,
pca, hub, random and, since 2026-10-19, facloc: greedy facility location
on the correlation matrix clipped at 0, the paper's headline method)
picks its benchmarks exactly as in eval_baselines.py (same estimate_corr
correlation, the split's eligible_mask(O_train), the same common random
numbers; run_selection is imported from eval_baselines), and each size-k
selection A is scored under three imputers on the SAME cells:

  gaussian    the paper's imputer: Gaussian conditional mean with ridge
              0.01 given the observed selected benchmarks
              (eval_entropy_vs_mi.impute_and_evaluate, imported, not
              copied)
  softimpute  nuclear-norm-regularized matrix completion (Mazumder,
              Hastie & Tibshirani 2010, "Spectral regularization
              algorithms for learning large incomplete matrices", JMLR 11):
              minimize 1/2 ||P_Omega(X - Z)||_F^2 + lambda ||Z||_* by
              soft-thresholded SVD iterations Z <- S_lambda(P_Omega X +
              P_Omega-perp Z), Nesterov-accelerated (FISTA, Beck & Teboulle
              2009; SoftImpute is proximal gradient with unit step, so the
              accelerated iterates solve the same convex problem) with
              function-value restart (O'Donoghue & Candes 2015), warm
              starts down a geometric lambda path (--solver exact: a
              safeguarded exact row step, see soft_impute; --solver plain:
              Mazumder et al.'s iteration as is)
  hardimpute  rank-r Hard-Impute (Mazumder et al. 2010; iterative SVD
              imputation, Troyanskaya et al. 2001): Z <- P_r(P_Omega X +
              P_Omega-perp Z), i.e. every missing cell is refilled with
              its value in the rank-r truncated SVD of the filled matrix,
              plain iterations (_hard_map_svd).  This is a majorize-
              minimize scheme for min ||P_Omega(X - Z)||^2 over rank(Z)
              <= r: that residual and the rank-r tail of the filled matrix
              decrease in turn.  Every fit starts at the SoftImpute fit of
              the same cells at SoftImpute's chosen lambda (Mazumder et
              al. initialize Hard-Impute with Soft-Impute solutions;
              --hard-init zero: the zero fill), each rank of the path from
              that same start (no warm start up the rank path).  Ranks r
              <= k only (a validation model observes at most k selected
              benchmarks, so its rank-r fit is underdetermined above k;
              --no-rank-cap).  No regularization, and two consequences:
              (a) at a fixed point a validation model is predicted by the
              least-squares rank-r fit to its observed selected values (at
              k = 1 the ratio x_j v_U / v_j), which blows up where V_r
              restricted to those benchmarks is ill-conditioned:
              converged fits with R^2 < -1, 82 of BenchPress's 2,081
              values (34 at k = 1), 13 of 2,093 on Merged (10), none
              on MMLU and MTEB; (b) on the sparse matrices some fits
              drift: the residual keeps decreasing while some imputed
              cells keep growing, the behavior of a rank-r least-squares
              problem whose infimum is not attained (the image of the
              rank-r matrices under P_Omega need not be closed), so these
              fits do not converge and their R^2 depends on the cap
              (counted: *_nonconv, *_converged, ImpHardConv; see
              "HardImpute convergence and the cap" below).  The first
              version of this
              script refilled each row by its exact least-squares rank-r
              fit u V_r^T, u = argmin ||x_O - u V_r[O]^T|| (--hard-solver
              lsq, _hard_map_lsq: Hard-Impute's inner linear iteration at
              fixed V_r run to its limit; same fixed points, F also
              monotone, iterates unbounded), warm-started up the rank
              path and SQUAREM-extrapolated.  That map follows the drift
              directions orders of magnitude faster and runs off even
              where Hard-Impute converges to a finite fixed point
              (test_hard_impute_bounded_lsq_diverges), and most of its
              exploded values (R^2 < -1: 28 / 236 / 377 of about 2,100 on
              MTEB / Merged / BenchPress, down to -3.7e8) were solver
              artifacts: now 10 / 52 / 261, smallest -27 / -12 / -79.
              Means are still affected by (a) and (b): use medians and
              win rates as well (ImpExplode*, RsqMedianImp*,
              PairMedianImp*).

What it says about the Gaussian imputer.  For a fixed SVD of the filled
matrix, a row's SoftImpute fixed point is the Gaussian conditional mean
under the covariance Sigma = V diag(max(s / lambda, 1)) V^T (soft_impute):
nuclear-norm completion imputes like the paper's conditional mean with a
covariance whose eigenvalues grow like the singular values s rather than
s^2 (for lambda below every s, Sigma is proportional to (F^T F)^{1/2});
HardImpute is the conditional mean of a rank-r covariance.

Units and scored cells.  Every imputer works in train-standardized units:
the training column means / standard deviations of estimate_corr, the
same clip of the standardized cells to +-10 as impute_and_evaluate
(applied to training and validation cells alike; for training cells it
is a no-op in practice).  Every imputer's predictions (Gaussian,
SoftImpute, HardImpute) are clipped to the benchmark's valid score range
before they are scored (protocol rule D2, score_bounds.z_limits in the
split's standardized units, the rule of every script; the inner-holdout
R^2 that tunes lambda and the rank, and {m}_hardimpute_max_abs, use the
unclipped fit).  The scored cells are impute_and_evaluate's: validation
models that observe at least one selected benchmark, their observed
unselected target benchmarks (rule D1, cv_split.target_mask: benchmarks
observed by fewer than MIN_TARGET_OBS = 2 training models are never
scored, though they stay in the fitted matrix); pooled R^2 =
1 - SS_res / max(SS_tot, 1e-12) with SS_tot = sum of squared standardized
targets (scored_mask, pooled_r2; no shortcut over observation patterns).
A per-model reimplementation of the Gaussian conditional on scored_mask
(gaussian_predict) is checked against impute_and_evaluate on every split
and (method, k) (gaussian_check_maxdiff; protocol_changes.md H1: the
datasets here have no split at risk from impute_and_evaluate's fast path).

Transductive fit.  The completion imputers are fitted on one matrix: the
training models' observed standardized cells plus the validation models'
observed SELECTED cells only (a validation model observing none of A
carries no information and is not scored).  The validation models'
unselected cells never enter a fit; they are the scored targets.

Hyperparameters by an inner holdout of training cells that imitates the
outer task's cell pattern (held-out models keep only their selected
cells, as the validation models do) but not its share of such query
rows: in the inner fit the held-out training models AND every validation
model have only their selected cells (28% of the fitted rows on MMLU:
0.2 x 0.9 + 0.1; split 0 means 0.26 on MTEB, 0.20 on Merged), in the
refit only the validation models (10%; 0.10, 0.07).  Transductive
SoftImpute shrinks more as that share grows, so the inner R^2 is biased
low: mean inner R^2 at the choice vs mean validation R^2 over all
(split, method, k) 0.580 vs 0.727 (MMLU), 0.512 vs 0.597 (MTEB), 0.306
vs 0.358 (Merged; BenchPress 0.206 vs 0.165).  The choice still tracks a
validation oracle (refits at every grid value, scored on the validation
cells) on split 0: it loses at most 1.7e-4 validation R^2 against the
best grid value on MMLU (15 sets) and 5.8e-4 on MTEB (15 sets; validation
R^2 is flat below lambda*), 0.004 on average and 0.03 at most on Merged
(16 sets, the largest at random k = 1), small against SoftImpute's
deficits to the Gaussian.  Per split, one permutation of the training
models is drawn from Split.seed(SEED + 21) (INNER_SEED_OFFSET; common
random numbers:
the same permutation for every method, k and imputer).  For a selection
A, the training models observing at least one benchmark of A are taken
in that order and the first ceil(--inner-frac * n) of them (default
0.2) form the inner holdout H: their observed unselected cells are held
out of the fit (they keep their selected cells, as validation models do)
and scored with the pooled R^2 above.  lambda runs down the geometric grid
lambda_max * --lambda-min-ratio^(g / (--n-lambda - 1)), g = 0..n-1
(lambda_max = largest singular value of the zero-filled standardized
training matrix of the split; default 16 values from lambda_max to
1e-3 lambda_max), the rank r up --ranks (default 1, 2, 3, 4, 5, 6, 8, 10,
12, 15, capped below N and at k).  A value becomes the choice only if it
improves the best inner R^2 so far by more than --min-improve (default
1e-4; plateaus and ties keep the larger lambda / smaller rank), and a
path stops after --patience (default 2; 0: full path) values without
such an improvement; on the lambda path the count starts at the first
improvement over lambda_max (the zero fit), so a flat start does not end
it.  The chosen value is then refitted with all training cells
(SoftImpute: warm start from its inner-holdout fit; HardImpute: from the
SoftImpute refit, see above; --no-refit reports the inner-holdout fit
itself).  Seed tag SEED + 21 is distinct from the random baseline's
SEED + 7, the retired entropy_rf tag SEED + 11 and every other stream in
code/
(test_eval_imputers checks the tree).

Exact row compression.  Both iterations are linear in the rows of a
missingness pattern (from a zero or warm start the missing block of the
rows with pattern p stays X_p C_p), so the n_p rows of a pattern with
n_p > |O_p| are replaced by the |O_p| rows of R_p in the reduced QR
X_p = Q_p R_p; Gram matrix, objective, singular values and convergence
measure are unchanged and the rows' predictions are Q_p times the
pseudo-rows' (Completion).  MMLU (fully observed) then iterates on about
N + k pseudo-rows instead of 5,452 models.  Convergence: relative squared
change of the missing cells ||Delta Y||^2 / ||P_Omega X||^2 < --tol
(SoftImpute, default 1e-7, at most --max-iter = 1000 iterations per
lambda) or --hard-tol (default 1e-12, at most --hard-max-iter = 3000
steps per fit, the same cap on every dataset); fits that hit the cap are
counted (*_nonconv; *_converged for the reported HardImpute fit).

Settings and runtime.  The defaults were chosen on split 0 of each
dataset (seven selected sets each) and checked on repetition 0 (10
splits, all methods and k): SoftImpute with FISTA at tol 1e-7 differs
from --solver exact by a mean |Delta R^2| of 0.003 (MMLU) / 0.005 (MTEB)
and from tol 1e-9 by 0.002 / 0.004 (largest single values 0.08 on MTEB),
at a third of their cost; a coarser 8-value lambda grid with tol 1e-7
lost 0.14 at k = 1 on MMLU and is not used.  The change criterion
stops SoftImpute slightly early, with a known sign on the dense
matrices: at the chosen lambda the certified optimum (FISTA continued to
a relative duality gap below 1e-9 on the full matrix) has a LOWER
validation R^2 than the reported fit, by 0.0027 on average (0.0007 to
0.0041) over the 15 distinct sets of MMLU split 0 and by 0.0039 (-0.0006
to +0.0099) on MTEB split 0; on Merged split 0 (16 sets) the two agree
within 0.0013 in either direction (mean -0.0002; gap below 1.2e-7 after
200,000 steps).  The reported SoftImpute R^2 are therefore slightly
optimistic on MMLU and MTEB, and its deficits against the Gaussian (0.07
to 0.26 on MMLU and 0.06 to 0.22 on MTEB at k = 5) are if anything
understated by a few thousandths.

HardImpute convergence and the cap.  The cap --hard-max-iter 3000 (tol
1e-12) is the same on every dataset.  Mazumder's iteration converges
linearly, and slowly when many cells are missing (the checked finite
fixed points of BenchPress and Merged needed 1,500 to 4,500 steps from
the SoftImpute start; many Merged fits need more), and the drifting fits
of (b) do not converge at all.  Reported fits that converged, full runs:
MMLU 2,100 of 2,100; MTEB 1,191 of 2,100; BenchPress 1,052 of 2,081;
Merged 424 of 2,093.  On repetition 0 the cap was varied (1,000 / 3,000
/ 10,000 steps; the Gaussian and SoftImpute values are unchanged, and so
are the fits converged at the same rank).  Share of the reported fits
converged: MMLU 0.995 / 1.000 / 1.000, no value moves by more than 0.03;
MTEB 0.51 / 0.55 / 0.58, 4% of the values move by more than 0.05 between
3,000 and 10,000 steps, median R^2 0.701 / 0.699 / 0.697; BenchPress
0.00 / 0.57 / 0.64, 23% move by more than 0.05, the k = 5 median
differences to the Gaussian change by up to 0.15; Merged 0.00 / 0.18 /
0.62, 18% move by more than 0.05, the k = 5 median differences change by
up to 0.23.  HardImpute values are thus properties of rank-r HardImpute
on MMLU (all fits converged, cap-insensitive), aggregate properties on
MTEB (k = 5 median differences to the Gaussian move by at most 0.013
between 3,000 and 10,000 steps; single values more), and cap-dependent on
Merged and BenchPress.

Within a split, a fit depends only on the selected SET, so a set that two
methods (or two k) share is fitted once and reused (a paired tie).

facloc (2026-10-19).  Facility location joined the methods after the
experiment above: it draws no random numbers and every fit depends only
on its set, so each per-method array of the other methods is that of a
run without it, bit for bit.  Statistics that pool over the methods --
n_sets and gaussian_check_maxdiff per split, the exploded and converged
shares, the hyperparameter boundary shares, the nonconverged and
inner-failed counts, the method ranks, their Spearman agreement and the
best-method agreement -- are taken over POOLED_METHODS (the six methods
of the original experiment that the run has; all of its methods if it
has none of them), so they too are unchanged; n_sets_all and
gaussian_check_maxdiff_all cover every method of the run.  facloc has
every per-method array, table entry, macro and paired row, and its rank
ImpRank{I}{Ds}FaclocK{k} is its rank among the pooled methods and itself
(1 = it beats all six).  Its paired rows follow all the others in the
CSV, which are a run's without it, in the same order.

Paired statistics (paired_stats.paired_stats, Nadeau-Bengio corrected):
  imputer      per method and k: softimpute - gaussian, hardimpute -
               gaussian, softimpute - hardimpute
  method       per imputer and k: every method pair (paired_stats.orient)
  interaction  per fitted imputer, method pair and k: the method
               difference under the fitted imputer minus the same
               difference under the Gaussian (does the imputer change a
               method gap?)
plus rank agreement of the method means (Spearman between imputers) and
the share of splits with the same best method.

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir()):
  LOGS/imputers_{ds}.npz
    {m}_{imp}_r2, {m}_{imp}_rmse      (n_splits, n_k) float, imp in
                                      gaussian, softimpute, hardimpute;
                                      column j is k = ks[j]; NaN where the
                                      split has fewer than k eligible
                                      benchmarks or k >= N
    {m}_n_eval                        (n_splits, n_k) int, cells scored
                                      (the same for every imputer)
    {m}_gaussian_ref_r2               (n_splits, n_k) gaussian_predict's
                                      R^2 (check of impute_and_evaluate)
    {m}_softimpute_lam_idx, _lam_ratio, _eff_rank, _iters, _nonconv
                                      (n_splits, n_k): chosen grid index
                                      (-1: none), lambda* / lambda_max,
                                      singular values above lambda* in the
                                      final fit, iterations and fits that
                                      hit --max-iter (path + refit)
    {m}_softimpute_inner_r2           (n_splits, n_k, n_lambda) inner R^2
                                      path (NaN beyond the early stop)
    {m}_hardimpute_rank, _rank_idx, _iters, _nonconv   (n_splits, n_k)
    {m}_hardimpute_converged          (n_splits, n_k) bool, the reported
                                      (refitted) fit converged within
                                      --hard-max-iter
    {m}_hardimpute_max_abs            (n_splits, n_k) largest |prediction|
                                      on the scored cells (standardized
                                      units; the inputs are clipped to +-10)
    {m}_hardimpute_inner_r2           (n_splits, n_k, n_ranks)
    {m}_n_inner, {m}_n_inner_cells    (n_splits, n_k) int, inner holdout
                                      models / cells
    {m}_inner_failed                  (n_splits, n_k) bool, no inner cell
                                      (first path value used)
    {m}_selected                      (n_splits, k_max) int, -1 padded (pca:
                                      its largest set; pca_selected_per_k
                                      (n_splits, k_max, k_max) its sets)
    rep, fold, split_id, n_train, n_val, n_eligible   (n_splits,) int
    eligible (n_splits, N) bool, estimation_failed (n_splits,) bool
    lambda_max (n_splits,), n_sets (n_splits,) int distinct sets fitted
    for the POOLED_METHODS, n_sets_all the same for every method,
    split_seconds (n_splits,), gaussian_check_maxdiff (n_splits,) over
    the POOLED_METHODS, gaussian_check_maxdiff_all over every method
    ks, ranks, lambda_ratios, methods, imputers, benchmark_names
    holdout_frac, n_folds, n_repeats, seed, min_train_obs,
    inner_seed_offset, inner_frac, patience, tol, max_iter, hard_tol,
    hard_max_iter, rank_cap, hard_solver, hard_init, clip, ridge, refit,
    solver, min_improve, k_max                                  scalars
  LOGS/imputers_{ds}.json   config, runtime_s, summary tables and a flat
    "macros" dict (letters only, make_numbers.camel naming; {I} imputer
    Gauss, Soft or Hard, {Ds} dataset, {M} method, K{k} spelled k):
      RsqImp{I}{Ds}{M}K{k}, RsqStdImp..., RsqMedianImp...   R^2 mean /
                                            std / median over splits
      ImpExplode{I}{Ds}, ImpExplodeN{I}{Ds}  share / number of the
                                            (split, method, k) R^2 values
                                            below -1 (HardImpute: rank-1
                                            ratio fits at k = 1 and
                                            drifting fits; the Gaussian on
                                            BenchPress; means and mean
                                            differences are then dominated
                                            by them, use the medians and
                                            win rates as well)
      ImpHardConv{Ds}                       share of the (split, method, k)
                                            HardImpute values whose
                                            reported fit converged
      Pair{S}Imp{Ds}{M}{Ia}Vs{Ib}K{k}       imputer pairs (S in Diff, Se,
                                            Lo, Hi, Win, Wins, N; and
                                            Median, the median per-split
                                            difference)
      Pair{S}Imp{I}{Ds}{A}Vs{B}K{k}         method pairs under imputer I
                                            (k in paired_stats.K_MACRO)
      Pair{S}ImpInter{I}{Ds}{A}Vs{B}K{k}    interaction, I in Soft, Hard
      ImpRank{I}{Ds}{M}K{k}                 rank of M by mean R^2 (1 best)
                                            among the POOLED_METHODS
                                            (facloc: among them and
                                            itself)
      ImpRankCorr{I}{Ds}K{k}, ImpBestAgree{I}{Ds}K{k}   vs Gauss
      ImpLamRatio{Ds}{M}K{k} (median), ImpSoftRank{Ds}{M}K{k},
      ImpHardRank{Ds}{M}K{k} (means), ImpLamAtMin{Ds}, ImpLamAtMax{Ds},
      ImpHardRankAtMax{Ds}, ImpHardRankAtMin{Ds} (shares of the
      (split, method, k) choices at the grid ends: smallest / largest
      lambda; largest allowed rank / rank 1, over the k with at least two
      allowed ranks), ImpInnerFailed{Ds}, ImpNonconvSoft{Ds},
      ImpNonconvHard{Ds} (fits that hit the iteration cap, summed over
      the (split, method, k) rows), ImpGaussCheck{Ds} (max |R^2
      difference| impute_and_evaluate vs gaussian_predict), ImpNSplits{Ds},
      ImpRuntimeMin{Ds} (wall-clock minutes of the run)
  LOGS/imputers_{ds}_paired.csv   every paired row above: dataset,
    holdout_frac, k, comparison, imputer_a, imputer_b, method_a, method_b,
    then paired_stats.STAT_FIELDS
  OUT/imputers_{ds}.pdf    median R^2 over splits vs k (Gaussian solid,
                           softimpute dashed, hardimpute dotted) and the
                           median per-split differences softimpute -
                           Gaussian, hardimpute - Gaussian with their
                           interquartile range (robust statistics: exploding
                           fits dominate the means on the sparse matrices;
                           the mean statistics are in the CSV and macros);
                           axes clipped to [-1, 1] with a count of the
                           points outside
  OUT/imputers_all.pdf     the same, one row per dataset of the run

Usage:
    python eval_imputers.py [--datasets mmlu mteb merged benchpress]
                            [--repeats 5] [--folds 10]
                            [--ks 1 2 3 5 7 10 15]
                            [--methods mi entropy trace pca hub random
                                       facloc]
                            [--imputers gaussian softimpute hardimpute]
                            [--n-lambda 16] [--lambda-min-ratio 1e-3]
                            [--ranks 1 2 3 4 5 6 8 10 12 15]
                            [--inner-frac 0.2] [--patience 2]
                            [--min-improve 1e-4] [--tol 1e-7]
                            [--max-iter 1000] [--hard-tol 1e-12]
                            [--hard-max-iter 3000] [--no-rank-cap]
                            [--solver fista|exact|plain]
                            [--hard-solver svd|lsq] [--hard-init soft|zero]
                            [--no-refit] [--from-logs]
                            [--reps R ...] [--merge] [--workers 1]

Shards: --reps R writes the per-split arrays of repetition R only to
LOGS/imputers_shards/imputers_<ds>_rep<R>.npz (no summary; a
subdirectory, so no glob of LOGS/imputers_*.npz sees a shard); after the
shards of every repetition 0 .. --repeats-1, --merge (same settings,
checked) concatenates them in repetition order into LOGS/imputers_<ds>.npz
and writes the JSON, paired CSV and figure exactly as one process would
(every split depends only on itself, so the arrays are bit-identical;
config.runtime_s and ImpRuntimeMin are then the summed shard runtimes).
run_all.sh runs the shards as separate jobs (imputers_<ds>_rep<R>) and
the merge as imputers_<ds>.

Workers: --workers W evaluates the splits of each dataset, or of each
--reps shard, in W spawned single-threaded processes (evaluate_split,
one split per task; the BLAS thread variables default to 1 for them; a
worker refuses to start if the parent overrode the protocol constants,
test-only module state that a spawned process does not see).  Every split
is a pure function of itself, so its rows are those of the serial run bit
for bit; they are assembled, and their progress lines printed, in split
order (stdout equals the serial run's apart from the per-split seconds).
Only config.workers (for a merge: the shards' W) and the wall-clock
fields (split_seconds, config.runtime_s, ImpRuntimeMin, the shards'
shard_runtime_s) depend on W.  A shard has --folds splits, so W above
--folds adds nothing there; the default W = 1 runs the splits in this
process, as before.

Runtime (single-threaded, --repeats 5, 50 splits, defaults, the same
--hard-max-iter 3000 everywhere; AMD EPYC 7502 hosts shared with other
jobs; CPU = user time):
  mmlu        37 min CPU, 37 min wall
  mteb        95 min CPU, 95 min wall
  benchpress  97 min CPU, 97 min wall
  merged      318 min CPU: the five repetitions ran as five single-
              threaded processes, one per repetition, through a harness
              calling run_dataset with cv_splits(..., reps=[r]) and then
              this script's summary functions; per-split results depend
              only on the split (every array bit-identical to one process
              on MMLU; a single Merged process printed the same per-split
              values for its first 18 splits, at 395 s per split, i.e.
              ~5.5 h wall for 50)
HardImpute's fits that run to the cap dominate (one Hard-Impute step is
an N x N partial eigendecomposition, ~0.5 ms on Merged).  About 39
distinct selected sets per split are fitted (6 methods x 7 k, shared
sets fitted once); the paths stop early on plateaus (--patience).
facloc (2026-10-19) adds its sets that no other method picks: the
runtimes above scale with n_sets_all / n_sets.
"""

import argparse
import json
import math
import multiprocessing
import os
import pathlib
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

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
from cv_split import (MIN_TRAIN_OBS, N_REPEATS, cv_splits, eligible_mask,
                      target_mask)
from eval_entropy_vs_mi import (estimate_corr, impute_and_evaluate,
                                protocol_config)
from score_bounds import clip_z, dataset_bounds, z_limits
from eval_baselines import METHOD_COLOR, METHOD_LABEL, run_selection
from paired_stats import (K_MACRO, STAT_FIELDS, orient, paired_stats,
                          write_csv)
from make_numbers import PAIR_STATS, camel, int_words

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

KS = [1, 2, 3, 5, 7, 10, 15]
N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42                     # eval_baselines.SEED (selections must match)
CLIP = 10.0                   # impute_and_evaluate's clip of the inputs
RIDGE = 0.01                  # impute_and_evaluate's ridge
SS_TOT_FLOOR = 1e-12          # impute_and_evaluate's floor of SS_tot
# Common random numbers of the inner-holdout permutation of the training
# models: Split.seed(SEED + 21).  Distinct from RANDOM_SEED_OFFSET = 7
# (random baseline), ENTROPY_RF_SEED_OFFSET = 11 (retired with entropy_rf,
# rule D3, still reserved) and every other stream of code/.
INNER_SEED_OFFSET = 21
INNER_FRAC = 0.2
N_LAMBDA = 16
LAMBDA_MIN_RATIO = 1e-3
RANKS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
PATIENCE = 2
MIN_IMPROVE = 1e-4            # inner R² gain that counts on a path
TOL = 1e-7                    # SoftImpute convergence (relative squared)
MAX_ITER = 1000               # SoftImpute iterations per lambda
HARD_TOL = 1e-12              # HardImpute convergence (relative squared)
HARD_MAX_ITER = 3000          # HardImpute steps per fit (every dataset)
SOLVERS = ["fista", "exact", "plain"]
SOLVER = "fista"
HARD_SOLVERS = ["svd", "lsq"]
HARD_SOLVER = "svd"           # Mazumder et al.'s Hard-Impute iteration
HARD_INITS = ["soft", "zero"]
HARD_INIT = "soft"            # start every HardImpute fit at SoftImpute's
PINV_RTOL = 1e-10             # lsq refill: eigenvalues of V_O^T V_O kept

DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
# paired_stats.METHOD_ORDER (entropy_rf was dropped, rule D3), then
# facloc (appended on 2026-10-19; module docstring)
METHODS = ["mi", "entropy", "trace", "pca", "hub", "random", "facloc"]
ALL_METHODS = ["mi", "entropy", "trace", "pca", "hub", "random", "facloc"]
# The methods of the original experiment: statistics pooled over methods
# use these (module docstring, "facloc")
POOLED_METHODS = ("mi", "entropy", "trace", "pca", "hub", "random")
IMPUTERS = ["gaussian", "softimpute", "hardimpute"]
FITTED = ["softimpute", "hardimpute"]
IMPUTER_PAIRS = [("softimpute", "gaussian"), ("hardimpute", "gaussian"),
                 ("softimpute", "hardimpute")]
IMPUTER_MACRO = {"gaussian": "Gauss", "softimpute": "Soft",
                 "hardimpute": "Hard"}
IMPUTER_LABEL = {"gaussian": "Gaussian conditional",
                 "softimpute": "SoftImpute (nuclear norm)",
                 "hardimpute": "HardImpute (rank r)"}
IMPUTER_STYLE = {"gaussian": "-", "softimpute": "--", "hardimpute": ":"}
KS_REPORT = [1, 3, 5, 10, 15]
Y_FLOOR = -1.0                # figure y-axes: lower values are off-scale
EXPLODE_R2 = -1.0             # R^2 below this counts as an exploded fit

PAIRED_FIELDS = (["dataset", "holdout_frac", "k", "comparison", "imputer_a",
                  "imputer_b", "method_a", "method_b"] + STAT_FIELDS)

DISPLAY_NAME = {"mmlu": "MMLU", "mteb": "MTEB", "merged": "Merged",
                "benchpress": "BenchPress", "olv2sub": "OLL v2 sub-tasks",
                "olv1items": "OLL v1 tasks", "epoch": "Epoch AI",
                "helm_lite": "HELM Lite", "helm_mmlu": "HELM MMLU",
                "helm_classic": "HELM Classic", "mteb2026": "MTEB 2026"}


def display_name(ds_name):
    """Display name of a dataset (the raw name if none is registered)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def pooled_methods(methods):
    """The methods of a run that statistics pooled over methods use: its
    POOLED_METHODS, in its order (every method if it has none of them;
    module docstring, "facloc")."""
    methods = [str(m) for m in methods]
    return [m for m in methods if m in POOLED_METHODS] or methods


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis)


def _nanmedian(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _num(x):
    """JSON-safe float (None for NaN / inf)."""
    x = float(x)
    return x if math.isfinite(x) else None


# ── Units, scored cells, pooled R² ───────────────────────────────

def standardize(B, O, col_mean, col_std, clip=CLIP):
    """Train-standardized cells clipped to +-clip, 0 where unobserved
    (impute_and_evaluate's B_test_std)."""
    Z = O * (B - col_mean[None, :]) / col_std[None, :]
    return np.clip(Z, -clip, clip) * O


def scored_mask(O_test, selected, target=None):
    """(M_test, N) bool: the cells impute_and_evaluate scores for the
    selection `selected` -- the observed unselected target benchmarks
    (target: the split's cv_split.target_mask, rule D1; None: every
    column) of every validation model that observes at least one selected
    benchmark.  Computed per model; no shortcut over observation
    patterns."""
    O = np.asarray(O_test) > 0
    in_a = np.zeros(O.shape[1], dtype=bool)
    in_a[list(selected)] = True
    rows = O[:, in_a].any(axis=1)
    cells = O & ~in_a[None, :] & rows[:, None]
    if target is not None:
        cells &= np.asarray(target, dtype=bool)[None, :]
    return cells


def pooled_r2(X_true, X_pred, mask, z_lim=None):
    """Pooled R² = 1 - SS_res / max(SS_tot, 1e-12) and RMSE over the cells
    `mask` (standardized units; SS_tot about 0, the training mean), and the
    number of cells (impute_and_evaluate's definition).  z_lim: (z_lo,
    z_hi), (N,) each, the valid score range in these units (rule D2,
    score_bounds.z_limits); the predictions are clipped to it before they
    are scored (None: no clip)."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {"r2": np.nan, "rmse": np.nan, "n_eval": 0}
    t = X_true[mask]
    p = X_pred[mask]
    if z_lim is not None:
        cols = np.nonzero(mask)[1]
        p, _ = clip_z(p, z_lim[0][cols], z_lim[1][cols])
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum(t ** 2))
    return {"r2": 1.0 - ss_res / max(ss_tot, SS_TOT_FLOOR),
            "rmse": math.sqrt(ss_res / n), "n_eval": n}


def gaussian_predict(Z_test, O_test, selected, R, ridge=RIDGE):
    """Gaussian conditional mean of every unselected benchmark given each
    validation model's observed selected cells, ridge `ridge` on R_oo
    (impute_and_evaluate's predictor, model by model, grouped by the
    observed subset of `selected`).  Rows observing none of `selected` are
    0 (unscored)."""
    O = np.asarray(O_test) > 0
    M, N = Z_test.shape
    sel = np.asarray(list(selected), dtype=np.int64)
    rest = np.setdiff1d(np.arange(N), sel)
    pred = np.zeros((M, N))
    pats, inv = np.unique(O[:, sel], axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    for p, pat in enumerate(pats):
        obs = sel[pat]
        if obs.size == 0 or rest.size == 0:
            continue
        rows = np.flatnonzero(inv == p)
        R_oo = R[np.ix_(obs, obs)] + ridge * np.eye(obs.size)
        W = np.linalg.solve(R_oo, R[np.ix_(obs, rest)])
        pred[np.ix_(rows, rest)] = Z_test[np.ix_(rows, obs)] @ W
    return pred


# ── Pattern-compressed completion problem ────────────────────────

class Completion:
    """A completion problem: standardized values X (n, N) observed on
    `mask`, with the rows of each missingness pattern p compressed exactly.

    Rows of a pattern with n_p > |O_p| rows are replaced by the |O_p|
    pseudo-rows R_p of the reduced QR X_p[:, O_p] = Q_p R_p (other rows
    are kept as they are).  SoftImpute and HardImpute iterates are linear
    in the rows of a pattern (missing block X_p C_p from a zero or
    compressed warm start), so Gram matrix, objective, singular values,
    convergence measure and fixed points are those of the full problem,
    and a row's missing values are Q_p @ (pseudo-rows' values).

    P (n_ps, N) pseudo-row values (0 where missing), mask (n_ps, N),
    Q (n, n_ps) the block map, real_mask (n, N).  Rows observing nothing
    have no pseudo-row (their values are 0).  compress=False keeps every
    row (Q = I on the rows observing something; for tests).
    """

    def __init__(self, X, mask, compress=True):
        mask = np.asarray(mask, dtype=bool)
        X = np.where(mask, np.asarray(X, dtype=np.float64), 0.0)
        n, N = X.shape
        self.n, self.N = n, N
        self.real_mask = mask
        pats, inv = np.unique(mask, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        blocks, masks, pids, parts = [], [], [], []
        off = 0
        for p, pat in enumerate(pats):
            cols = np.flatnonzero(pat)
            if cols.size == 0:
                continue
            rows = np.flatnonzero(inv == p)
            Xp = X[np.ix_(rows, cols)]
            if compress and rows.size > cols.size:
                Qp, Rp = np.linalg.qr(Xp)
            else:
                Qp, Rp = np.eye(rows.size), Xp
            m = Rp.shape[0]
            blk = np.zeros((m, N))
            blk[:, cols] = Rp
            blocks.append(blk)
            masks.append(np.repeat(pat[None, :], m, axis=0))
            pids.append(np.full(m, len(parts)))
            parts.append((rows, Qp, off, m, pat))
            off += m
        self.n_ps = off
        self.P = np.vstack(blocks) if blocks else np.zeros((0, N))
        self.mask = np.vstack(masks) if masks else np.zeros((0, N), bool)
        self.Q = np.zeros((n, off))
        for rows, Qp, o, m, _ in parts:
            self.Q[rows, o:o + m] = Qp
        self.obs2 = max(float(np.sum(self.P ** 2)), 1e-300)
        pid = np.concatenate(pids) if pids else np.zeros(0, np.int64)
        pat_of = np.array([pt for *_, pt in parts]).reshape(-1, N)
        incomplete = ~pat_of.all(axis=1)
        self.inc_rows = np.flatnonzero(incomplete[pid])
        self.comp_rows = np.flatnonzero(~incomplete[pid])
        # renumber the incomplete patterns 0..n_ip-1
        new_id = np.cumsum(incomplete) - 1
        self.inc_pid = new_id[pid[self.inc_rows]]
        self.inc_patterns = pat_of[incomplete].astype(np.float64)
        # row_conditional: single-pseudo-row patterns batched by |O|
        # (gather indices precomputed), multi-row patterns one by one
        single, self._multi = {}, []
        for _, _, o, m, pt in parts:
            if pt.all():
                continue
            cols, miss = np.flatnonzero(pt), np.flatnonzero(~pt)
            if m == 1:
                single.setdefault(cols.size, []).append((o, cols, miss))
            else:
                self._multi.append((slice(o, o + m), cols, miss,
                                    self.P[o:o + m][:, cols]))
        self._single = []
        for size, items in sorted(single.items()):
            rows = np.array([o for o, _, _ in items], dtype=np.int64)
            Oi = np.array([c for _, c, _ in items], dtype=np.int64)
            Ui = np.array([u for _, _, u in items], dtype=np.int64)
            self._single.append((rows, Oi, Ui,
                                 self.P[rows[:, None], Oi]))

    def zeros(self):
        return np.zeros_like(self.P)

    def row_conditional(self, Sigma):
        """Missing cells of every pseudo-row set to the Gaussian
        conditional mean x_O Sigma_OO^{-1} Sigma_OU given its observed
        cells (0 on observed cells)."""
        out = np.zeros_like(self.P)
        for rows, Oi, Ui, Xo in self._single:
            A = Sigma[Oi[:, :, None], Oi[:, None, :]]
            C = np.linalg.solve(A, Sigma[Oi[:, :, None], Ui[:, None, :]])
            out[rows[:, None], Ui] = np.matmul(Xo[:, None, :], C)[:, 0, :]
        for sl, o, u, Xo in self._multi:
            C = np.linalg.solve(Sigma[np.ix_(o, o)], Sigma[np.ix_(o, u)])
            block = out[sl]                   # a view of out
            block[:, u] = Xo @ C
        return out

    def to_real(self, S, rows=None):
        """Values of the real rows (all, or `rows`) from pseudo-row values."""
        Q = self.Q if rows is None else self.Q[rows]
        return Q @ S

    def to_pseudo(self, Y):
        """Pseudo-row values of real-row missing values Y (n, N): the
        projection Q^T Y of Y's missing cells (a warm start; exact if Y is
        in the compressed form)."""
        S = self.Q.T @ np.where(self.real_mask, 0.0, Y)
        return np.where(self.mask, 0.0, S)


def _soft_eval(prob, S, lam):
    """Soft-thresholded SVD of the filled matrix F = P + S (eigen-
    decomposition of F^T F): returns (Z = S_lam(F), objective
    1/2 ||P_Omega(X - Z)||^2 + lam ||Z||_*, V, s, keep = s > lam)."""
    F = prob.P + S
    e, V = np.linalg.eigh(F.T @ F)
    s = np.sqrt(np.maximum(e, 0.0))
    keep = s > lam
    Vk = V[:, keep]
    Z = F @ ((Vk * (1.0 - lam / s[keep])) @ Vk.T)
    fval = (0.5 * float(np.sum(np.where(prob.mask, prob.P - Z, 0.0) ** 2))
            + lam * float(np.sum(s[keep] - lam)))
    return Z, fval, V, s, keep


def soft_impute(prob, lam, S0=None, tol=TOL, max_iter=MAX_ITER,
                solver=SOLVER):
    """SoftImpute at one lambda: minimize f(Z) = 1/2 ||P_Omega(X - Z)||^2 +
    lam ||Z||_*, whose minimizers are the fixed points of Z = S_lam(P_Omega
    X + P_Omega-perp Z) (S_lam: soft-thresholding of the singular values).
    S0: warm start (pseudo-row missing values).

    solver "plain": Mazumder et al.'s iteration Z <- S_lam(P_Omega X +
    P_Omega-perp Z); "fista" (default): the same with Nesterov
    extrapolation and function-value restart.  Both stop when the
    relative squared change of the missing cells is below tol; at small
    lam a step moves the missing cells by about lam / s, so from a cold
    start they can stop far from the solution (the lambda path's warm
    starts avoid this in practice).  "exact": for the current fill F =
    U diag(s) V^T the plain map's row fixed point is the Gaussian
    conditional mean x_O Sigma_OO^{-1} Sigma_OU under Sigma = V diag(
    max(s / lam, 1)) V^T = (I - W)^{-1}, W = V diag((1 - lam/s)_+) V^T;
    this exact row step is taken whenever its objective is at most that of
    the plain step, else the plain step (so f decreases at least as fast
    as under plain iterations and the iterates converge to a minimizer),
    and convergence is declared when the exact step changes the missing
    cells by less than tol (relative squared).  It converges in a few
    iterations when observed rows dominate the Gram matrix (MMLU) but is
    slower than fista on the sparse matrices; a sensitivity check.

    Returns (S, info): S the missing cells, info iters, converged,
    eff_rank (singular values > lam), objective, exact_steps."""
    P, obs = prob.P, prob.mask
    miss = ~obs
    S = prob.zeros() if S0 is None else np.where(miss, S0, 0.0)
    converged, it, n_exact = False, 0, 0
    if solver == "exact":
        cur = _soft_eval(prob, S, lam)
        for it in range(1, max_iter + 1):
            Z, fval, V, s, keep = cur
            # Sigma = I + V_k diag(s/lam - 1) V_k^T (exactly I if nothing
            # survives the threshold)
            Vk = V[:, keep]
            Sigma = np.eye(prob.N) + (Vk * (s[keep] / lam - 1.0)) @ Vk.T
            S_ex = prob.row_conditional(Sigma)
            if float(np.sum((S_ex - S) ** 2)) / prob.obs2 < tol:
                S = S_ex
                converged = True
                break
            S_pl = np.where(miss, Z, 0.0)
            c_ex = _soft_eval(prob, S_ex, lam)
            c_pl = _soft_eval(prob, S_pl, lam)
            if c_ex[1] <= c_pl[1]:
                S, cur = S_ex, c_ex
                n_exact += 1
            else:
                S, cur = S_pl, c_pl
        _, fval, _, _, keep = cur
        return S, {"iters": it, "converged": converged,
                   "eff_rank": int(keep.sum()), "objective": fval,
                   "exact_steps": n_exact}
    if solver not in ("plain", "fista"):
        raise ValueError(solver)
    Y = S.copy()
    t = 1.0
    f_prev = np.inf
    rank, fval = 0, np.nan
    for it in range(1, max_iter + 1):
        Z, fval, _, _, keep = _soft_eval(prob, Y, lam)
        rank = int(keep.sum())
        new = np.where(miss, Z, 0.0)
        delta = float(np.sum((new - S) ** 2)) / prob.obs2
        if solver == "fista":
            if fval > f_prev:                 # restart the momentum
                t = 1.0
                Y = new
            else:
                t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
                Y = new + ((t - 1.0) / t_next) * (new - S)
                t = t_next
        else:
            Y = new
        S = new
        f_prev = fval
        if delta < tol:
            converged = True
            break
    return S, {"iters": it, "converged": converged, "eff_rank": rank,
               "objective": fval, "exact_steps": 0}


def _top_eig(F, rank):
    """(V_r, F(S)): the top-`rank` eigenvectors of the Gram matrix F^T F of
    the filled matrix F = P + S (scipy's partial symmetric eigensolver) and
    the sum of its N - rank smallest eigenvalues, ||F||^2 minus the top
    ones (= ||F - F V_r V_r^T||^2, the rank-r tail)."""
    from scipy.linalg import eigh
    N = F.shape[1]
    e, V = eigh(F.T @ F, subset_by_index=[N - rank, N - 1])
    tail = float(np.sum(F * F)) - float(np.sum(np.maximum(e, 0.0)))
    return V, max(tail, 0.0)


def _hard_map_svd(prob, S, rank):
    """One Hard-Impute step of Mazumder et al. (2010) from the missing
    cells S: (T(S), F(S), resid).  Z = P_r(P + S) is the rank-r truncated
    SVD of the filled matrix (Z = (P + S) V_r V_r^T, V_r its top-r right
    singular vectors) and every missing cell <- its value in Z.  F(S) =
    ||P + S - Z||^2 (the rank-r tail of the filled matrix), resid =
    ||P_Omega(X - Z)||^2 / ||P_Omega X||^2.  The step is a majorize-
    minimize step for min ||P_Omega(X - Z)||^2 over rank(Z) <= r, and the
    two decrease in turn: F(T(S)) <= resid(S) ||P_Omega X||^2 <= F(S)."""
    F = prob.P + S
    Vr, f_val = _top_eig(F, rank)
    Z = (F @ Vr) @ Vr.T
    resid = float(np.sum(np.where(prob.mask, prob.P - Z, 0.0) ** 2))
    return np.where(prob.mask, 0.0, Z), f_val, resid / prob.obs2


def _hard_map_lsq(prob, S, rank):
    """The exact least-squares refill (--hard-solver lsq; a diagnostic, not
    used for results): (T(S), F(S), resid).  V_r = top-r eigenvectors of
    the filled Gram (P + S)^T (P + S), then every missing cell <- its
    least-squares rank-r reconstruction u V_r^T, u = argmin ||x_O - u
    V_r[O]^T|| (minimum norm if rank-deficient).  This is Hard-Impute's
    map with its inner linear iteration at fixed V_r run to its limit.
    It has the same fixed points and also decreases F(S) (= min over V_r
    of ||(P + S)(I - V_r V_r^T)||^2), but that does not bound the
    iterates: where min ||P_Omega(X - Z)||^2 over rank(Z) <= r is not
    attained, F keeps decreasing along directions in which the missing
    cells grow without bound, and this map follows them even from starts
    where Hard-Impute converges to a finite fixed point
    (test_hard_impute_bounded_lsq_diverges)."""
    P, obs = prob.P, prob.mask
    miss = ~obs
    rows, pid, pats = prob.inc_rows, prob.inc_pid, prob.inc_patterns
    F = P + S
    Vr, f_val = _top_eig(F, rank)
    new = np.zeros_like(S)
    resid = 0.0
    if rows.size:
        VO = pats[:, :, None] * Vr[None, :, :]           # (n_pat, N, r)
        G = np.matmul(VO.transpose(0, 2, 1), Vr)         # V_O^T V_O
        w, U = np.linalg.eigh(G)
        cut = PINV_RTOL * np.maximum(w[:, -1:], 1e-300)
        inv_w = np.where(w > cut, 1.0 / np.where(w > cut, w, 1.0), 0.0)
        G_inv = np.matmul(U * inv_w[:, None, :], U.transpose(0, 2, 1))
        b = P[rows] @ Vr
        u = np.matmul(G_inv[pid], b[:, :, None])[:, :, 0]
        fit = u @ Vr.T
        new[rows] = np.where(miss[rows], fit, 0.0)
        resid = float(np.sum(np.where(obs[rows], P[rows] - fit, 0.0) ** 2))
    if prob.comp_rows.size:
        Pc = P[prob.comp_rows]
        resid += float(np.sum((Pc - (Pc @ Vr) @ Vr.T) ** 2))
    return new, f_val, resid / prob.obs2


def hard_impute(prob, rank, S0=None, tol=HARD_TOL, max_iter=HARD_MAX_ITER,
                solver=HARD_SOLVER):
    """HardImpute at rank `rank` from the missing cells S0 (default 0)
    until a step changes the missing cells by less than tol (relative
    squared change ||Delta S||^2 / ||P_Omega X||^2) or max_iter steps.
    solver "svd" (default): Mazumder et al.'s Hard-Impute iteration
    Z <- P_r(P_Omega X + P_Omega-perp Z) (_hard_map_svd), plain, without
    extrapolation; "lsq": the exact least-squares refill (_hard_map_lsq,
    a diagnostic: its iterates can run off to infinity).  Returns (S,
    info): iters, converged, residual (||P_Omega(X - Z)||^2 /
    ||P_Omega X||^2 of the last rank-r fit Z), objective F at the last
    step's input, max_abs (largest |S| of the pseudo-rows)."""
    if solver not in HARD_SOLVERS:
        raise ValueError(solver)
    step = _hard_map_svd if solver == "svd" else _hard_map_lsq
    obs2 = prob.obs2
    S = prob.zeros() if S0 is None else np.where(prob.mask, 0.0, S0)
    rank = int(min(rank, prob.N))
    converged, it, resid, f_val = False, 0, np.nan, np.nan
    for it in range(1, max_iter + 1):
        S1, f_val, resid = step(prob, S, rank)
        delta = float(np.sum((S1 - S) ** 2)) / obs2
        S = S1
        if delta < tol:
            converged = True
            break
    return S, {"iters": it, "converged": converged, "residual": resid,
               "objective": f_val,
               "max_abs": float(np.abs(S).max()) if S.size else 0.0}


# ── One selected set: inner holdout, paths, refit ────────────────

def inner_holdout(O_train, selected, perm, inner_frac=INNER_FRAC):
    """Inner-holdout training models (sorted positions in the training
    block) of the selection `selected`: the training models that observe
    at least one selected benchmark, in the order of the split's
    permutation `perm` (Split.seed(SEED + INNER_SEED_OFFSET)), first
    ceil(inner_frac * n) of them."""
    O = np.asarray(O_train) > 0
    sel = np.asarray(list(selected), dtype=np.int64)
    cand = perm[O[perm][:, sel].any(axis=1)]
    n_h = int(math.ceil(inner_frac * cand.size))
    return np.sort(cand[:n_h])


def lambda_ratios(n_lambda=N_LAMBDA, min_ratio=LAMBDA_MIN_RATIO):
    """Geometric grid lambda / lambda_max = min_ratio^(g / (n_lambda - 1)),
    g = 0..n_lambda-1 (1 down to min_ratio)."""
    if n_lambda == 1:
        return np.ones(1)
    return min_ratio ** (np.arange(n_lambda) / (n_lambda - 1))


def lambda_max(Z_train):
    """Largest singular value of the zero-filled standardized training
    matrix (SoftImpute's solution is 0 at and above it)."""
    e = np.linalg.eigvalsh(Z_train.T @ Z_train)
    return float(np.sqrt(max(e[-1], 0.0)))


def _path(fit, values, score, patience, min_improve=None, null_first=False):
    """Warm-started path over `values` (fit(value, S) -> (S, info)):
    inner score after each value; a value becomes the best if it improves
    the best inner R^2 by more than min_improve (default MIN_IMPROVE;
    plateaus keep the earlier, more regularized value); early stop after
    `patience` values without such an improvement (0: none).  null_first
    (the lambda path, whose first value lambda_max is the zero fit):
    the patience count starts only after the first improvement, so a flat
    start near lambda_max does not end the path.  Returns (best index,
    scores (len(values),) NaN beyond the stop, best S, total iterations,
    fits that did not converge, info of the best value's fit)."""
    min_improve = MIN_IMPROVE if min_improve is None else min_improve
    scores = np.full(len(values), np.nan)
    best_g, best_sc, best_S, best_info = -1, -np.inf, None, None
    S, iters, nonconv = None, 0, 0
    for g, v in enumerate(values):
        S, info = fit(v, S)
        iters += info["iters"]
        nonconv += int(not info["converged"])
        sc = score(S)
        scores[g] = sc
        if best_g < 0 or (np.isfinite(sc) and sc > best_sc + min_improve):
            best_g, best_sc, best_S = g, (sc if np.isfinite(sc)
                                          else -np.inf), S.copy()
            best_info = info
        elif (patience and g - best_g >= patience
              and not (null_first and best_g == 0)):
            break
    return best_g, scores, best_S, iters, nonconv, best_info


def fit_set(Z_all, O_all, n_train, selected, perm, lam_max, ratios, ranks,
            imputers=FITTED, inner_frac=INNER_FRAC, patience=PATIENCE,
            tol=TOL, max_iter=MAX_ITER, refit=True, solver=SOLVER,
            min_improve=None, hard_tol=HARD_TOL, hard_max_iter=HARD_MAX_ITER,
            rank_cap=True, hard_solver=HARD_SOLVER, hard_init=HARD_INIT):
    """Fit the completion imputers for one selected set on one split.

    Z_all (n_train + n_val, N): standardized cells of the training models
    then the validation models (0 where unobserved), O_all its bool mask.
    The fit uses the training cells and the validation models' observed
    selected cells only.  rank_cap: HardImpute ranks r <= k = |selected|
    only (a validation model observes at most k selected benchmarks, so
    its rank-r fit is underdetermined for r > k).  hard_init "soft": every
    HardImpute fit (each rank of the inner path, and the refit) starts at
    the SoftImpute fit of the same cells at the SoftImpute choice lambda*
    (the SoftImpute path runs for it even if softimpute is not among
    `imputers`); "zero": at the zero fill.  Returns {imputer: {"pred":
    (n_val, N) standardized predictions (scored on scored_mask), "idx",
    "value", "inner_r2", "iters", "nonconv", "converged" (the reported
    fit), ("eff_rank")}, "n_inner", "n_inner_cells", "inner_failed"}."""
    if hard_init not in HARD_INITS:
        raise ValueError(hard_init)
    O_all = np.asarray(O_all, dtype=bool)
    n, N = Z_all.shape
    in_a = np.zeros(N, dtype=bool)
    in_a[list(selected)] = True
    val_rows = np.arange(n_train, n)
    O_tr = O_all[:n_train]

    inner = inner_holdout(O_tr, selected, perm, inner_frac)
    held = np.zeros_like(O_all)
    held[inner] = O_tr[inner] & ~in_a[None, :]
    fit_mask = O_all.copy()
    fit_mask[n_train:] &= in_a[None, :]
    mask_a = fit_mask & ~held
    prob_a = Completion(Z_all, mask_a)
    prob_b = Completion(Z_all, fit_mask) if refit else prob_a
    n_cells = int(held.sum())
    out = {"n_inner": int(inner.size), "n_inner_cells": n_cells,
           "inner_failed": n_cells == 0}

    def score(S):
        if n_cells == 0:
            return np.nan
        pred = prob_a.to_real(S, inner)
        return pooled_r2(Z_all[inner], pred, held[inner])["r2"]

    def finish(name, values, fit, path, S0=None):
        """Refit at the chosen value (from S0, default: the inner-holdout
        fit) and the result dict; returns (result, S of the reported
        fit)."""
        best_g, scores, best_S, iters, nonconv, best_info = path
        if refit:
            if S0 is None:
                S0 = prob_b.to_pseudo(prob_a.to_real(best_S))
            S, info = fit(values[best_g], S0, prob_b)
            iters += info["iters"]
            nonconv += int(not info["converged"])
        else:
            S, info = best_S, best_info
        res = {"pred": prob_b.to_real(S, val_rows), "idx": best_g,
               "value": values[best_g], "inner_r2": scores,
               "iters": iters, "nonconv": nonconv,
               "converged": bool(info["converged"])}
        if name == "softimpute":
            res["eff_rank"] = info["eff_rank"]
        return res, S

    soft_start = (None, None)        # SoftImpute fits (inner, refit)
    if "softimpute" in imputers or ("hardimpute" in imputers
                                    and hard_init == "soft"):
        lams = lam_max * ratios

        def fit_soft(lam, S, prob=prob_a, iters=max_iter):
            return soft_impute(prob, lam, S, tol, iters, solver)

        path = _path(fit_soft, lams, score, patience, min_improve,
                     null_first=True)
        res, S_b = finish("softimpute", lams, fit_soft, path)
        if "softimpute" in imputers:
            out["softimpute"] = res
        soft_start = (path[2], S_b)

    if "hardimpute" in imputers:
        rk = [int(r) for r in ranks
              if not rank_cap or r <= len(selected)] or [int(min(ranks))]
        if hard_init == "soft":
            start_a, start_b = soft_start
        else:
            start_a, start_b = prob_a.zeros(), prob_b.zeros()

        def fit_hard(r, S, prob=prob_a, iters=hard_max_iter):
            return hard_impute(prob, r, S, hard_tol, iters, hard_solver)

        # every rank from the same start (no warm start up the rank path)
        path = _path(lambda r, S: fit_hard(r, start_a), rk, score, patience,
                     min_improve)
        out["hardimpute"], _ = finish("hardimpute", rk, fit_hard, path,
                                      S0=start_b)
    return out


# ── Per-split evaluation ─────────────────────────────────────────

def _padded(indices, length):
    out = np.full(length, -1, dtype=np.int64)
    out[:len(indices)] = indices
    return out


def _split_arrays(splits, ctx):
    """The per-split arrays of run_dataset (module docstring, "Outputs")
    for `splits`, one row each in their order, at their defaults (NaN, -1,
    0 or False: what a split leaves where k exceeds its eligible
    benchmarks or its estimate failed); ctx: run_dataset's split
    context."""
    S_n = len(splits)
    N = ctx["B"].shape[1]
    nk = len(ctx["ks"])
    k_max, methods, fitted = ctx["k_max"], ctx["methods"], ctx["fitted"]
    nan2 = lambda: np.full((S_n, nk), np.nan)
    neg2 = lambda: np.full((S_n, nk), -1, dtype=np.int64)
    arr = {
        "rep": np.array([s.rep for s in splits], dtype=np.int64),
        "fold": np.array([s.fold for s in splits], dtype=np.int64),
        "split_id": np.array([s.split_id for s in splits], dtype=np.int64),
        "n_train": np.array([len(s.train_idx) for s in splits],
                            dtype=np.int64),
        "n_val": np.array([len(s.val_idx) for s in splits], dtype=np.int64),
        "n_eligible": np.zeros(S_n, dtype=np.int64),
        "eligible": np.zeros((S_n, N), dtype=bool),
        "estimation_failed": np.zeros(S_n, dtype=bool),
        "lambda_max": np.full(S_n, np.nan),
        "n_sets": np.zeros(S_n, dtype=np.int64),
        "split_seconds": np.zeros(S_n),
        "gaussian_check_maxdiff": np.full(S_n, np.nan),
        "n_sets_all": np.zeros(S_n, dtype=np.int64),
        "gaussian_check_maxdiff_all": np.full(S_n, np.nan),
    }
    for m in methods:
        for imp in ["gaussian"] + fitted:
            arr[f"{m}_{imp}_r2"] = nan2()
            arr[f"{m}_{imp}_rmse"] = nan2()
        arr[f"{m}_gaussian_ref_r2"] = nan2()
        arr[f"{m}_n_eval"] = np.zeros((S_n, nk), dtype=np.int64)
        arr[f"{m}_selected"] = np.full((S_n, k_max), -1, dtype=np.int64)
        arr[f"{m}_n_inner"] = np.zeros((S_n, nk), dtype=np.int64)
        arr[f"{m}_n_inner_cells"] = np.zeros((S_n, nk), dtype=np.int64)
        arr[f"{m}_inner_failed"] = np.zeros((S_n, nk), dtype=bool)
        if "softimpute" in fitted:
            arr[f"{m}_softimpute_lam_idx"] = neg2()
            arr[f"{m}_softimpute_lam_ratio"] = nan2()
            arr[f"{m}_softimpute_eff_rank"] = neg2()
            arr[f"{m}_softimpute_iters"] = np.zeros((S_n, nk), np.int64)
            arr[f"{m}_softimpute_nonconv"] = np.zeros((S_n, nk), np.int64)
            arr[f"{m}_softimpute_inner_r2"] = np.full(
                (S_n, nk, len(ctx["ratios"])), np.nan)
        if "hardimpute" in fitted:
            arr[f"{m}_hardimpute_rank"] = neg2()
            arr[f"{m}_hardimpute_rank_idx"] = neg2()
            arr[f"{m}_hardimpute_iters"] = np.zeros((S_n, nk), np.int64)
            arr[f"{m}_hardimpute_nonconv"] = np.zeros((S_n, nk), np.int64)
            arr[f"{m}_hardimpute_converged"] = np.zeros((S_n, nk), bool)
            arr[f"{m}_hardimpute_max_abs"] = nan2()
            arr[f"{m}_hardimpute_inner_r2"] = np.full(
                (S_n, nk, len(ctx["ranks"])), np.nan)
    if "pca" in methods:
        arr["pca_selected_per_k"] = np.full((S_n, k_max, k_max), -1,
                                            dtype=np.int64)
    return arr


def _echo(text, end="\n"):
    """evaluate_split's default report: printed, flushed."""
    print(text, end=end, flush=True)


def evaluate_split(ctx, arr, i, s, log=_echo):
    """Every method, k and imputer on split s: fills row i of the per-split
    arrays arr (_split_arrays) and reports the split's progress line
    through log(text, end).  ctx: run_dataset's split context (data,
    bounds, settings).  A pure function of ctx and the split (its
    estimate, selections, inner permutation Split.seed(SEED + 21) and
    fits), so the rows do not depend on the process that computes them
    (--workers)."""
    B, O = ctx["B"], ctx["O"]
    benchmark_names, bounds = ctx["benchmark_names"], ctx["bounds"]
    ks, k_max, methods = ctx["ks"], ctx["k_max"], ctx["methods"]
    pool, fitted = ctx["pooled"], ctx["fitted"]
    ratios, ranks = ctx["ratios"], ctx["ranks"]
    N = B.shape[1]
    t0 = time.time()
    B_train, O_train = B[s.train_idx], O[s.train_idx]
    B_test, O_test = B[s.val_idx], O[s.val_idx]
    el = eligible_mask(O_train)
    target = target_mask(O_train)        # D1, every method and imputer
    arr["eligible"][i] = el
    arr["n_eligible"][i] = int(el.sum())
    log(f"  Split {s.split_id} (rep {s.rep}, fold {s.fold}): "
        f"{len(s.train_idx)} train / {len(s.val_idx)} test, "
        f"{int(el.sum())}/{N} eligible", end="")
    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:
        log(f" — estimation failed: {e}")
        arr["estimation_failed"][i] = True
        return

    z_lim = z_limits(bounds, col_mean, col_std)          # D2
    Z_train = standardize(B_train, O_train, col_mean, col_std)
    Z_test = standardize(B_test, O_test, col_mean, col_std)
    Z_all = np.vstack([Z_train, Z_test])
    O_all = np.vstack([O_train, O_test]) > 0
    n_tr = len(s.train_idx)
    lam_max = lambda_max(Z_train)
    arr["lambda_max"][i] = lam_max
    perm = np.random.default_rng(
        s.seed(SEED + INNER_SEED_OFFSET)).permutation(n_tr)
    cache = {}
    pooled_sets = set()          # the sets of the POOLED_METHODS
    maxdiff = maxdiff_all = 0.0
    for m in methods:
        pooled = m in pool
        result = run_selection(m, R, k_max, benchmark_names, s, el)
        arr[f"{m}_selected"][i] = _padded(result["selected"][:k_max],
                                          k_max)
        if m == "pca":
            for kk, sel_kk in enumerate(result["selected_per_k"]):
                arr["pca_selected_per_k"][i, kk] = _padded(sel_kk, k_max)
        for j, k in enumerate(ks):
            selected = selection_at_k(result, k) if k < N else None
            if selected is None:
                continue
            mask = scored_mask(O_test, selected, target)
            imp = impute_and_evaluate(B_test, O_test, selected, R,
                                      col_mean, col_std, N,
                                      target=target, bounds=bounds)
            arr[f"{m}_gaussian_r2"][i, j] = imp["r2"]
            arr[f"{m}_gaussian_rmse"][i, j] = imp["rmse"]
            ref = pooled_r2(Z_test, gaussian_predict(
                Z_test, O_test, selected, R), mask, z_lim)
            arr[f"{m}_gaussian_ref_r2"][i, j] = ref["r2"]
            arr[f"{m}_n_eval"][i, j] = ref["n_eval"]
            if np.isfinite(ref["r2"]) or np.isfinite(imp["r2"]):
                diff = (abs(ref["r2"] - imp["r2"])
                        if np.isfinite(ref["r2"] - imp["r2"]) else np.inf)
                maxdiff_all = max(maxdiff_all, diff)
                if pooled:
                    maxdiff = max(maxdiff, diff)
            if not fitted:
                continue
            key = tuple(sorted(int(x) for x in selected))
            if pooled:
                pooled_sets.add(key)
            if key not in cache:
                cache[key] = fit_set(
                    Z_all, O_all, n_tr, key, perm, lam_max, ratios,
                    ranks, fitted, *ctx["fit_args"])
            fs = cache[key]
            arr[f"{m}_n_inner"][i, j] = fs["n_inner"]
            arr[f"{m}_n_inner_cells"][i, j] = fs["n_inner_cells"]
            arr[f"{m}_inner_failed"][i, j] = fs["inner_failed"]
            for name in fitted:
                res = fs[name]
                sc = pooled_r2(Z_test, res["pred"], mask, z_lim)
                arr[f"{m}_{name}_r2"][i, j] = sc["r2"]
                arr[f"{m}_{name}_rmse"][i, j] = sc["rmse"]
                arr[f"{m}_{name}_iters"][i, j] = res["iters"]
                arr[f"{m}_{name}_nonconv"][i, j] = res["nonconv"]
                path = arr[f"{m}_{name}_inner_r2"]
                path[i, j, :len(res["inner_r2"])] = res["inner_r2"]
                if name == "softimpute":
                    arr[f"{m}_softimpute_lam_idx"][i, j] = res["idx"]
                    arr[f"{m}_softimpute_lam_ratio"][i, j] = \
                        ratios[res["idx"]]
                    arr[f"{m}_softimpute_eff_rank"][i, j] = \
                        res["eff_rank"]
                else:
                    arr[f"{m}_hardimpute_rank_idx"][i, j] = res["idx"]
                    arr[f"{m}_hardimpute_rank"][i, j] = res["value"]
                    arr[f"{m}_hardimpute_converged"][i, j] = \
                        res["converged"]
                    arr[f"{m}_hardimpute_max_abs"][i, j] = (
                        float(np.abs(res["pred"][mask]).max())
                        if mask.any() else np.nan)
    arr["gaussian_check_maxdiff"][i] = maxdiff
    arr["gaussian_check_maxdiff_all"][i] = maxdiff_all
    arr["n_sets"][i] = len(pooled_sets)
    arr["n_sets_all"][i] = len(cache)
    arr["split_seconds"][i] = time.time() - t0
    j5 = ks.index(5) if 5 in ks else 0
    log(f"  {len(cache)} sets  k={ks[j5]}: " + "  ".join(
        f"{m}:" + "/".join(f"{arr[f'{m}_{imp}_r2'][i, j5]:.3f}"
                           for imp in ["gaussian"] + fitted)
        for m in methods) + f"  ({arr['split_seconds'][i]:.1f}s)")
    if maxdiff_all > 1e-8:
        log(f"  WARNING: impute_and_evaluate differs from the "
            f"per-model Gaussian by {maxdiff_all:.2e} (scored cells?)")


# Worker state of the split processes (--workers; spawn)
_CTX = {}
# BLAS thread variables of the worker processes: 1 unless set (run_all.sh
# sets them for every job)
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS")


def _init_worker(ctx):
    _CTX.clear()
    _CTX.update(ctx)
    # test-only overrides of the protocol constants (module state of the
    # parent) do not reach a spawned process: refuse rather than differ
    if protocol_config() != ctx["protocol"]:
        raise RuntimeError(f"protocol {ctx['protocol']} overridden in the "
                           f"parent process: run with --workers 1")


def _run_split(s):
    """evaluate_split of split s in a worker process: (its one-row
    arrays, the text it reported)."""
    row = _split_arrays([s], _CTX)
    text = []
    evaluate_split(_CTX, row, 0, s,
                   log=lambda t, end="\n": text.append(t + end))
    return row, "".join(text)


def map_splits(ctx, splits, workers):
    """_run_split of every split, in split order, in `workers` spawned
    single-threaded processes (ProcessPoolExecutor.map keeps the order;
    every split is a pure function of ctx and the split, so the rows do
    not depend on workers)."""
    for var in BLAS_THREAD_VARS:
        os.environ.setdefault(var, "1")
    mp = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(splits)),
                             mp_context=mp, initializer=_init_worker,
                             initargs=(ctx,)) as pool:
        yield from pool.map(_run_split, splits)


def run_dataset(ds_name, ks=KS, methods=METHODS, imputers=IMPUTERS,
                n_folds=N_FOLDS, n_repeats=N_REPEATS, n_lambda=N_LAMBDA,
                lambda_min_ratio=LAMBDA_MIN_RATIO, ranks=RANKS,
                inner_frac=INNER_FRAC, patience=PATIENCE, tol=TOL,
                max_iter=MAX_ITER, refit=True, solver=SOLVER,
                min_improve=None, hard_tol=HARD_TOL,
                hard_max_iter=HARD_MAX_ITER, rank_cap=True,
                hard_solver=HARD_SOLVER, hard_init=HARD_INIT, reps=None,
                workers=1):
    """Every method, k and imputer on the matched splits of one dataset;
    returns the dict of per-split arrays saved as LOGS/imputers_{ds}.npz.
    reps: only the splits of these repetitions (cv_splits(...,
    reps=reps), a shard; split_id and n_repeats are those of the full
    design, see merge_shards).  workers > 1: the splits run in that many
    spawned single-threaded processes (map_splits), their rows and
    progress lines assembled in split order; the arrays are those of the
    serial run bit for bit (split_seconds is wall clock)."""
    print(f"\n{'#'*70}\n# Dataset: {ds_name.upper()}\n{'#'*70}")
    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)     # D2
    print(f"{M} models × {N} tasks, {O.mean()*100:.1f}% observed",
          flush=True)
    ks = [int(k) for k in ks]
    k_max = min(max(ks), N)
    ranks = [int(r) for r in ranks if 0 < int(r) < N]
    fitted = [imp for imp in FITTED if imp in imputers]
    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED,
                            reps=reps))
    S_n = len(splits)
    print(f"{n_repeats} × {n_folds}-fold CV = {S_n} matched splits; "
          f"k in {ks}; imputers {['gaussian'] + fitted}; inner holdout "
          f"{inner_frac:.0%} (seed tag SEED + {INNER_SEED_OFFSET})",
          flush=True)

    ratios = lambda_ratios(n_lambda, lambda_min_ratio)
    ctx = {"B": B, "O": O, "benchmark_names": benchmark_names,
           "bounds": bounds, "ks": ks, "k_max": k_max, "methods": methods,
           "pooled": pooled_methods(methods), "fitted": fitted,
           "ratios": ratios, "ranks": ranks,
           "fit_args": (inner_frac, patience, tol, max_iter, refit, solver,
                        min_improve, hard_tol, hard_max_iter, rank_cap,
                        hard_solver, hard_init),
           "protocol": protocol_config()}
    arr = _split_arrays(splits, ctx)
    if workers <= 1 or S_n <= 1:
        for i, s in enumerate(splits):
            evaluate_split(ctx, arr, i, s)
    else:
        for i, (row, text) in enumerate(map_splits(ctx, splits, workers)):
            for key, val in row.items():
                arr[key][i] = val[0]
            print(text, end="", flush=True)

    arr.update(
        benchmark_names=benchmark_names, ks=np.array(ks, dtype=np.int64),
        ranks=np.array(ranks, dtype=np.int64), lambda_ratios=ratios,
        methods=np.array(methods), imputers=np.array(["gaussian"] + fitted),
        holdout_frac=np.float64(HOLDOUT_FRAC), n_folds=np.int64(n_folds),
        n_repeats=np.int64(n_repeats), seed=np.int64(SEED),
        **{k: (np.bool_(v) if isinstance(v, bool) else np.int64(v))
           for k, v in protocol_config().items()},
        inner_seed_offset=np.int64(INNER_SEED_OFFSET),
        inner_frac=np.float64(inner_frac), patience=np.int64(patience),
        tol=np.float64(tol), max_iter=np.int64(max_iter),
        hard_tol=np.float64(hard_tol), hard_max_iter=np.int64(hard_max_iter),
        rank_cap=np.bool_(rank_cap), hard_solver=np.array(hard_solver),
        hard_init=np.array(hard_init),
        clip=np.float64(CLIP), ridge=np.float64(RIDGE),
        refit=np.bool_(refit), solver=np.array(solver),
        min_improve=np.float64(MIN_IMPROVE if min_improve is None
                               else min_improve),
        k_max=np.int64(k_max))
    return arr


# ── Paired statistics ────────────────────────────────────────────

def _strs(a):
    return [str(x) for x in np.asarray(a).ravel()]


def paired_rows(ds_name, arr):
    """Paired R² rows (PAIRED_FIELDS) of the imputer, method and
    interaction comparisons (module docstring) at every k: first those of
    the pooled methods (pooled_methods), then the rows of every other
    method (facloc), which follow in the order of a run with all methods.
    So the first rows are those of a run without facloc."""
    methods = _strs(arr["methods"])
    pool = pooled_methods(methods)
    rows = _paired_rows(ds_name, arr, pool)
    if len(pool) < len(methods):
        rows += [r for r in _paired_rows(ds_name, arr, methods)
                 if r["method_a"] not in pool or r["method_b"] not in pool]
    return rows


def _paired_rows(ds_name, arr, methods):
    """paired_rows of the methods `methods` of the log arr."""
    imputers = _strs(arr["imputers"])
    ks = [int(k) for k in arr["ks"]]
    meta = (arr["n_train"], arr["n_val"], arr["rep"])
    rows = []

    def add(k, comp, ia, ib, ma, mb, a, b):
        rows.append({"dataset": ds_name, "holdout_frac": HOLDOUT_FRAC,
                     "k": k, "comparison": comp, "imputer_a": ia,
                     "imputer_b": ib, "method_a": ma, "method_b": mb,
                     **paired_stats(a, b, *meta)})

    for j, k in enumerate(ks):
        for ia, ib in IMPUTER_PAIRS:
            if ia not in imputers or ib not in imputers:
                continue
            for m in methods:
                add(k, "imputer", ia, ib, m, m, arr[f"{m}_{ia}_r2"][:, j],
                    arr[f"{m}_{ib}_r2"][:, j])
        for imp in imputers:
            for ma, mb in orient(methods):
                add(k, "method", imp, imp, ma, mb,
                    arr[f"{ma}_{imp}_r2"][:, j], arr[f"{mb}_{imp}_r2"][:, j])
        for imp in imputers:
            if imp == "gaussian":
                continue
            for ma, mb in orient(methods):
                add(k, "interaction", imp, "gaussian", ma, mb,
                    arr[f"{ma}_{imp}_r2"][:, j] - arr[f"{mb}_{imp}_r2"][:, j],
                    arr[f"{ma}_gaussian_r2"][:, j]
                    - arr[f"{mb}_gaussian_r2"][:, j])
    return rows


# ── Summaries and macros ─────────────────────────────────────────

def _ranks_desc(values):
    """Ranks of `values` (1 = largest; ties share their mean rank; NaN
    stays NaN)."""
    from scipy.stats import rankdata
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = rankdata(-v[ok])
    return out


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx, ry = _ranks_desc(x[ok]), _ranks_desc(y[ok])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _best_index(r2_stack):
    """(S,) index of the best method per split (ties: first in method
    order), -1 where any method is NaN."""
    ok = np.all(np.isfinite(r2_stack), axis=1)
    best = np.full(r2_stack.shape[0], -1, dtype=np.int64)
    if ok.any():
        best[ok] = np.argmax(r2_stack[ok], axis=1)
    return best


def summarize(ds_name, arr, paired):
    """Summary tables and the flat macros dict of one dataset.  Per-method
    entries cover every method; statistics pooled over methods (exploded
    and converged shares, boundary shares, counts, ranks and their
    agreement) the pooled methods only (pooled_methods; module docstring,
    "facloc"), and each other method is ranked among them and itself."""
    D = camel(ds_name)
    methods, imputers = _strs(arr["methods"]), _strs(arr["imputers"])
    pool = pooled_methods(methods)
    extra = [m for m in methods if m not in pool]
    fitted = [imp for imp in imputers if imp != "gaussian"]
    ks = [int(k) for k in arr["ks"]]
    macros = {}

    def mac(name, value):
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} is not letters-only")
        v = None if value is None else _num(value)
        if v is not None:
            macros[name] = v

    def K(k):
        return "K" + int_words(k)

    tables = {"r2": {}, "imputer_pairs": {}, "method_ranks": {},
              "rank_agreement": {}, "trace_vs_mi": {}, "hyperparameters": {},
              "convergence": {}}
    # R² means
    for imp in imputers:
        I = IMPUTER_MACRO[imp]
        t_imp = {}
        for m in methods:
            row = {}
            for j, k in enumerate(ks):
                x = arr[f"{m}_{imp}_r2"][:, j]
                row[str(k)] = {"mean": _num(_nanmean(x)),
                               "std": _num(_nanstd(x)),
                               "median": _num(_nanmedian(x)),
                               "n": int(np.isfinite(x).sum())}
                mac(f"RsqImp{I}{D}{camel(m)}{K(k)}", _nanmean(x))
                mac(f"RsqStdImp{I}{D}{camel(m)}{K(k)}", _nanstd(x))
                mac(f"RsqMedianImp{I}{D}{camel(m)}{K(k)}", _nanmedian(x))
            t_imp[m] = row
        tables["r2"][imp] = t_imp
        # exploding fits: R^2 < EXPLODE_R2 (HardImpute fits drifting
        # without a finite limit, rank-1 ratio fits at k = 1; the Gaussian
        # on BenchPress)
        vals = np.concatenate([arr[f"{m}_{imp}_r2"].ravel()
                               for m in pool])
        vals = vals[np.isfinite(vals)]
        n_exp = int(np.sum(vals < EXPLODE_R2))
        tables.setdefault("exploded", {})[imp] = {
            "n": n_exp, "of": int(vals.size),
            "share": _num(n_exp / vals.size) if vals.size else None}
        mac(f"ImpExplode{I}{D}", n_exp / vals.size if vals.size else np.nan)
        mac(f"ImpExplodeN{I}{D}", n_exp)

    # paired macros
    for r in paired:
        k, comp = int(r["k"]), r["comparison"]
        if comp == "imputer":
            name = (f"Imp{D}{camel(r['method_a'])}"
                    f"{IMPUTER_MACRO[r['imputer_a']]}Vs"
                    f"{IMPUTER_MACRO[r['imputer_b']]}{K(k)}")
            stats_ = PAIR_STATS
            key = f"{r['imputer_a']}-{r['imputer_b']}"
            med, q25, q75 = paired_quantiles(
                arr[f"{r['method_a']}_{r['imputer_a']}_r2"][:, ks.index(k)],
                arr[f"{r['method_a']}_{r['imputer_b']}_r2"][:, ks.index(k)])
            tables["imputer_pairs"].setdefault(key, {}).setdefault(
                r["method_a"], {})[str(k)] = dict(
                {f: _num(r[f]) for f in ("mean_diff", "se_nb", "ci_low",
                                         "ci_high", "win_rate", "p_nb")},
                median_diff=_num(med), q25_diff=_num(q25),
                q75_diff=_num(q75))
            mac(f"PairMedian{name}", med)
        elif k in K_MACRO and comp == "method":
            name = (f"Imp{IMPUTER_MACRO[r['imputer_a']]}{D}"
                    f"{camel(r['method_a'])}Vs{camel(r['method_b'])}{K(k)}")
            stats_ = PAIR_STATS
        elif k in K_MACRO and comp == "interaction":
            name = (f"ImpInter{IMPUTER_MACRO[r['imputer_a']]}{D}"
                    f"{camel(r['method_a'])}Vs{camel(r['method_b'])}{K(k)}")
            stats_ = {s: PAIR_STATS[s] for s in ("Diff", "Se", "Lo", "Hi",
                                                  "Win", "N")}
        else:
            continue
        for s, (field, _) in stats_.items():
            mac(f"Pair{s}{name}", r[field])
        if (comp == "method" and r["method_a"] == "mi"
                and r["method_b"] == "trace"):
            tables["trace_vs_mi"].setdefault(r["imputer_a"], {})[str(k)] = {
                "mi_minus_trace": _num(r["mean_diff"]),
                "ci": [_num(r["ci_low"]), _num(r["ci_high"])],
                "win_rate_mi": _num(r["win_rate"])}

    # method ranks and agreement between imputers (pooled methods; every
    # other method ranked among them and itself)
    for j, k in enumerate(ks):
        means = {imp: [_nanmean(arr[f"{m}_{imp}_r2"][:, j]) for m in pool]
                 for imp in imputers}
        stacks = {imp: np.stack([arr[f"{m}_{imp}_r2"][:, j]
                                 for m in pool], axis=1)
                  for imp in imputers}
        for imp in imputers:
            rk = _ranks_desc(means[imp])
            ranks_k = {m: _num(rk[q]) for q, m in enumerate(pool)}
            for q, m in enumerate(pool):
                mac(f"ImpRank{IMPUTER_MACRO[imp]}{D}{camel(m)}{K(k)}", rk[q])
            for m in extra:
                r_m = _ranks_desc(means[imp] + [
                    _nanmean(arr[f"{m}_{imp}_r2"][:, j])])[-1]
                ranks_k[m] = _num(r_m)
                mac(f"ImpRank{IMPUTER_MACRO[imp]}{D}{camel(m)}{K(k)}", r_m)
            tables["method_ranks"].setdefault(imp, {})[str(k)] = ranks_k
            if imp == "gaussian":
                continue
            rho = _spearman(means[imp], means["gaussian"])
            bg, bf = _best_index(stacks["gaussian"]), _best_index(stacks[imp])
            ok = (bg >= 0) & (bf >= 0)
            agree = float(np.mean(bg[ok] == bf[ok])) if ok.any() else np.nan
            fin_g = [x for x in means["gaussian"] if np.isfinite(x)]
            fin_f = [x for x in means[imp] if np.isfinite(x)]
            tables["rank_agreement"].setdefault(imp, {})[str(k)] = {
                "spearman_vs_gaussian": _num(rho),
                "best_split_agreement": _num(agree),
                "best_gaussian": (pool[int(np.nanargmax(means["gaussian"]))]
                                  if fin_g else None),
                "best": (pool[int(np.nanargmax(means[imp]))]
                         if fin_f else None)}
            I = IMPUTER_MACRO[imp]
            mac(f"ImpRankCorr{I}{D}{K(k)}", rho)
            mac(f"ImpBestAgree{I}{D}{K(k)}", agree)

    # hyperparameters and convergence
    if "softimpute" in fitted:
        n_lam = len(arr["lambda_ratios"])
        idx = np.concatenate([arr[f"{m}_softimpute_lam_idx"].ravel()
                              for m in pool])
        idx = idx[idx >= 0]
        hp = {"lam_ratio_median": {}, "eff_rank_mean": {}}
        for m in methods:
            for j, k in enumerate(ks):
                lr = arr[f"{m}_softimpute_lam_ratio"][:, j]
                er = arr[f"{m}_softimpute_eff_rank"][:, j].astype(float)
                er[er < 0] = np.nan
                hp["lam_ratio_median"].setdefault(m, {})[str(k)] = \
                    _num(_nanmedian(lr))
                hp["eff_rank_mean"].setdefault(m, {})[str(k)] = \
                    _num(_nanmean(er))
                mac(f"ImpLamRatio{D}{camel(m)}{K(k)}", _nanmedian(lr))
                mac(f"ImpSoftRank{D}{camel(m)}{K(k)}", _nanmean(er))
        at_min = float(np.mean(idx == n_lam - 1)) if idx.size else np.nan
        at_max = float(np.mean(idx == 0)) if idx.size else np.nan
        hp["share_at_smallest_lambda"] = _num(at_min)
        hp["share_at_lambda_max"] = _num(at_max)
        mac(f"ImpLamAtMin{D}", at_min)
        mac(f"ImpLamAtMax{D}", at_max)
        tables["hyperparameters"]["softimpute"] = hp
        nc = sum(int(arr[f"{m}_softimpute_nonconv"].sum()) for m in pool)
        it = np.concatenate([arr[f"{m}_softimpute_iters"].ravel()
                             for m in pool])
        tables["convergence"]["softimpute"] = {
            "nonconverged_fits": nc,
            "mean_iters_per_set": _num(np.mean(it[it > 0])) if (it > 0).any()
            else None}
        mac(f"ImpNonconvSoft{D}", nc)
    if "hardimpute" in fitted:
        # boundary shares over the k with a choice (>= 2 allowed ranks)
        ranks = [int(r) for r in arr["ranks"]]
        cap = bool(arr["rank_cap"]) if "rank_cap" in arr else True
        n_fit = n_top = n_one = 0
        for j, k in enumerate(ks):
            allowed = [r for r in ranks if not cap or r <= k]
            if len(allowed) < 2:
                continue
            for m in pool:
                r = arr[f"{m}_hardimpute_rank"][:, j]
                r = r[r > 0]
                n_fit += r.size
                n_top += int(np.sum(r == allowed[-1]))
                n_one += int(np.sum(r == allowed[0]))
        hp = {"rank_mean": {}, "rank_median": {}}
        for m in methods:
            for j, k in enumerate(ks):
                r = arr[f"{m}_hardimpute_rank"][:, j].astype(float)
                r[r < 0] = np.nan
                hp["rank_mean"].setdefault(m, {})[str(k)] = _num(_nanmean(r))
                hp["rank_median"].setdefault(m, {})[str(k)] = \
                    _num(_nanmedian(r))
                mac(f"ImpHardRank{D}{camel(m)}{K(k)}", _nanmean(r))
        at_max = n_top / n_fit if n_fit else np.nan
        at_min = n_one / n_fit if n_fit else np.nan
        hp["share_at_max_rank"] = _num(at_max)    # largest allowed rank
        hp["share_at_min_rank"] = _num(at_min)
        hp["boundary_fits"] = n_fit
        mac(f"ImpHardRankAtMax{D}", at_max)
        mac(f"ImpHardRankAtMin{D}", at_min)
        tables["hyperparameters"]["hardimpute"] = hp
        nc = sum(int(arr[f"{m}_hardimpute_nonconv"].sum()) for m in pool)
        it = np.concatenate([arr[f"{m}_hardimpute_iters"].ravel()
                             for m in pool])
        conv = {"nonconverged_fits": nc,
                "mean_iters_per_set": _num(np.mean(it[it > 0]))
                if (it > 0).any() else None}
        if all(f"{m}_hardimpute_converged" in arr for m in pool):
            # the reported (refitted) fits that converged, over the
            # (split, method, k) values; their R^2 and the others'
            ok = np.concatenate([np.isfinite(arr[f"{m}_hardimpute_r2"]).ravel()
                                 for m in pool])
            cv = np.concatenate([arr[f"{m}_hardimpute_converged"].ravel()
                                 for m in pool])[ok]
            r2 = np.concatenate([arr[f"{m}_hardimpute_r2"].ravel()
                                 for m in pool])[ok]
            share = float(np.mean(cv)) if cv.size else np.nan
            conv.update(
                reported_converged=int(cv.sum()), reported=int(cv.size),
                reported_converged_share=_num(share),
                median_r2_converged=_num(_nanmedian(r2[cv]))
                if cv.any() else None,
                median_r2_nonconverged=_num(_nanmedian(r2[~cv]))
                if (~cv).any() else None,
                exploded_converged=int(np.sum(r2[cv] < EXPLODE_R2)),
                exploded_nonconverged=int(np.sum(r2[~cv] < EXPLODE_R2)))
            mac(f"ImpHardConv{D}", share)
        tables["convergence"]["hardimpute"] = conv
        mac(f"ImpNonconvHard{D}", nc)
    if fitted:
        failed = sum(int(arr[f"{m}_inner_failed"].sum()) for m in pool)
        tables["hyperparameters"]["inner_failed"] = failed
        mac(f"ImpInnerFailed{D}", failed)
    chk = arr["gaussian_check_maxdiff"]
    gmax = float(np.nanmax(chk)) if np.isfinite(chk).any() else np.nan
    tables["gaussian_check_maxdiff"] = _num(gmax)
    mac(f"ImpGaussCheck{D}", gmax)
    if "gaussian_check_maxdiff_all" in arr:       # every method (facloc)
        chk = arr["gaussian_check_maxdiff_all"]
        tables["gaussian_check_maxdiff_all"] = _num(
            np.nanmax(chk) if np.isfinite(chk).any() else np.nan)
    mac(f"ImpNSplits{D}", len(arr["split_id"]))
    return tables, macros


def print_summary(ds_name, arr, tables, paired):
    methods, imputers = _strs(arr["methods"]), _strs(arr["imputers"])
    ks = [int(k) for k in arr["ks"]]
    ks_rep = [k for k in KS_REPORT if k in ks]
    S_n = len(arr["split_id"])
    print(f"\n== {display_name(ds_name)}: imputation R² (mean ± std over "
          f"{S_n} splits) ==")
    print(f"{'method':<10}{'imputer':<12}" + "".join(f"k={k:<11}"
                                                   for k in ks_rep))
    def fmt(x, w=16):
        return ("n/a" if x is None else f"{x:.3g}" if abs(x) >= 1000
                else f"{x:.3f}").ljust(w // 2)

    for m in methods:
        for imp in imputers:
            cells = []
            for k in ks_rep:
                c = tables["r2"][imp][m][str(k)]
                cells.append("n/a" if c["mean"] is None else
                             f"{fmt(c['mean'], 0)}±{fmt(c['std'], 0)}")
            print(f"{m:<10}{imp:<12}" + "".join(f"{c:<15} " for c in cells))
    print(f"\n== {display_name(ds_name)}: median R² over splits ==")
    for m in methods:
        print(f"{m:<10}" + "  ".join(
            f"{imp[:4]} " + " ".join(fmt(tables["r2"][imp][m][str(k)][
                "median"], 12) for k in ks_rep) for imp in imputers))
    print("  exploded fits (R² < {:g}): ".format(EXPLODE_R2) + ", ".join(
        f"{imp} {v['n']}/{v['of']}" for imp, v in
        tables.get("exploded", {}).items()))
    print(f"\n== {display_name(ds_name)}: paired R² differences of the "
          f"imputers (Nadeau–Bengio 95% CI, win rate) ==")
    for r in paired:
        if r["comparison"] != "imputer" or r["k"] not in (1, 5, 10) \
                or r["n_splits"] == 0:
            continue
        print(f"  k={r['k']:<3}{r['method_a']:<9}{r['imputer_a']:>11} − "
              f"{r['imputer_b']:<11}{r['mean_diff']:+.3f}  "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  win "
              f"{r['win_rate']:.2f}")
    print(f"\n== {display_name(ds_name)}: method ranks by mean R² "
          f"(1 = best) ==")
    pool = pooled_methods(methods)
    for imp in imputers:
        for k in ks_rep:
            rk = tables["method_ranks"][imp][str(k)]
            order = sorted((v, m) for m, v in rk.items()
                           if v is not None and m in pool)
            others = "".join(
                f"; {m} rank {v:g} of {len(pool) + 1}"
                for m, v in rk.items() if m not in pool and v is not None)
            print(f"  {imp:<11} k={k:<3}" + " > ".join(m for _, m in order)
                  + others)
    for imp, rows in tables["rank_agreement"].items():
        print(f"  {imp} vs gaussian: Spearman " + ", ".join(
            f"k={k} {v['spearman_vs_gaussian']:.2f}" if v[
                "spearman_vs_gaussian"] is not None else f"k={k} n/a"
            for k, v in rows.items()))
    print("  mi − trace: " + "; ".join(
        f"{imp} k={k} {v['mi_minus_trace']:+.3f} [{v['ci'][0]:+.3f}, "
        f"{v['ci'][1]:+.3f}]" for imp, rows in tables["trace_vs_mi"].items()
        for k, v in rows.items() if int(k) in (5, 10)
        and v["mi_minus_trace"] is not None))
    hp = tables["hyperparameters"]
    if "softimpute" in hp:
        print(f"  softimpute: lambda* at lambda_max in "
              f"{hp['softimpute']['share_at_lambda_max']}, at the smallest "
              f"grid value in {hp['softimpute']['share_at_smallest_lambda']}"
              f" of the fits; nonconverged fits "
              f"{tables['convergence']['softimpute']['nonconverged_fits']}")
    if "hardimpute" in hp:
        cv = tables["convergence"]["hardimpute"]
        print(f"  hardimpute: rank at max {hp['hardimpute']['share_at_max_rank']}"
              f", at 1 {hp['hardimpute']['share_at_min_rank']}; nonconverged "
              f"fits {cv['nonconverged_fits']}")
        if "reported" in cv:
            print(f"  hardimpute reported fits converged: "
                  f"{cv['reported_converged']}/{cv['reported']}; median R² "
                  f"converged {cv['median_r2_converged']}, not converged "
                  f"{cv['median_r2_nonconverged']}; R² < {EXPLODE_R2:g}: "
                  f"{cv['exploded_converged']} converged, "
                  f"{cv['exploded_nonconverged']} not")
    if "inner_failed" in hp:
        print(f"  inner holdout without cells: {hp['inner_failed']}")
    print(f"  Gaussian check (impute_and_evaluate vs per-model): max "
          f"|ΔR²| = {tables['gaussian_check_maxdiff']}")


# ── Plotting ─────────────────────────────────────────────────────

def _clip_ylim(ax, values, floor=Y_FLOOR, ceil=1.0):
    """y-limits of the finite `values`, clipped to [floor, ceil]; a note
    counts the points outside (HardImpute's exploding fits on the sparse
    matrices)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return
    lo, hi = max(float(v.min()), floor), min(float(v.max()), ceil)
    pad = 0.05 * max(hi - lo, 1e-3)
    ax.set_ylim(lo - pad, hi + pad)
    n_off = int(np.sum((v < floor) | (v > ceil)))
    if n_off:
        ax.text(0.02, 0.02, f"{n_off} point(s) outside [{floor:g}, "
                f"{ceil:g}] off-scale", transform=ax.transAxes, fontsize=7,
                va="bottom")


def paired_quantiles(a, b):
    """(median, 25th, 75th percentile) of the per-split differences a - b
    over the splits where both are finite (NaN if none)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if not d.size:
        return np.nan, np.nan, np.nan
    q25, med, q75 = np.percentile(d, [25, 50, 75])
    return float(med), float(q25), float(q75)


def _panels(axes, ds_name, arr, paired=None, title=True):
    """R^2 and paired-difference panels of one dataset, in robust
    statistics (HardImpute's and, on BenchPress, the Gaussian's exploding
    fits dominate means): median R^2 over splits; median per-split
    difference fitted - Gaussian with its interquartile range.  The
    Nadeau-Bengio mean statistics are in the paired CSV and the macros."""
    methods, imputers = _strs(arr["methods"]), _strs(arr["imputers"])
    ks = np.array([int(k) for k in arr["ks"]])
    ax = axes[0]
    shown = []
    for m in methods:
        for imp in imputers:
            med = _nanmedian(arr[f"{m}_{imp}_r2"], axis=0)
            shown.extend(med)
            ax.plot(ks, med, IMPUTER_STYLE[imp], color=METHOD_COLOR[m],
                    marker="o" if imp == "gaussian" else None,
                    markersize=3, linewidth=1.3 if imp == "gaussian" else 1.0)
    _clip_ylim(ax, shown)
    ax.set_ylabel("Imputation $R^2$ (median over splits)")
    if title:
        ax.set_title(f"{display_name(ds_name)}: $R^2$ by imputer")
    width = 0.08 * (len(methods) - 1) / 2
    for ax, fit in zip(axes[1:], [imp for imp in FITTED if imp in imputers]):
        shown = []
        for q, m in enumerate(methods):
            x = ks + (q - (len(methods) - 1) / 2) * 0.08
            st = np.array([paired_quantiles(arr[f"{m}_{fit}_r2"][:, j],
                                            arr[f"{m}_gaussian_r2"][:, j])
                           for j in range(len(ks))])
            med, lo, hi = st[:, 0], st[:, 1], st[:, 2]
            shown.extend(np.concatenate([med, lo, hi]))
            ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o-",
                        color=METHOD_COLOR[m], markersize=3, linewidth=1.0,
                        capsize=2)
        ax.axhline(0.0, color="black", linewidth=0.8)
        _clip_ylim(ax, shown + [0.0])
        ax.set_ylabel(f"$R^2$ {fit} − Gaussian (median, IQR)")
        if title:
            ax.set_title(f"{display_name(ds_name)}: "
                         f"{IMPUTER_LABEL[fit].split(' (')[0]} − Gaussian "
                         f"(paired)")
    for ax in axes:
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xticks(ks)
        ax.set_xlim(ks.min() - 0.5 - width, ks.max() + 0.5 + width)
        ax.grid(True, alpha=0.3)


def _legend(fig, methods, imputers):
    handles = [Line2D([], [], color=METHOD_COLOR[m], marker="o",
                      linewidth=1.3, label=METHOD_LABEL[m]) for m in methods]
    handles += [Line2D([], [], color="black", linestyle=IMPUTER_STYLE[imp],
                       label=IMPUTER_LABEL[imp]) for imp in imputers]
    fig.legend(handles=handles, loc="lower center", ncol=min(
        len(handles), 5), fontsize=8, frameon=False)


def plot_dataset(ds_name, arr, paired):
    methods, imputers = _strs(arr["methods"]), _strs(arr["imputers"])
    n_pan = 1 + len([imp for imp in FITTED if imp in imputers])
    fig, axes = plt.subplots(1, n_pan, figsize=(5 * n_pan, 4.2))
    axes = np.atleast_1d(axes)
    _panels(axes, ds_name, arr, paired)
    _legend(fig, methods, imputers)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    path = OUT / f"imputers_{ds_name}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)


def plot_all(results):
    """OUT/imputers_all.pdf: one row per dataset (R² and the paired
    differences)."""
    names = list(results)
    arr0 = results[names[0]][0]
    methods, imputers = _strs(arr0["methods"]), _strs(arr0["imputers"])
    n_pan = 1 + len([imp for imp in FITTED if imp in imputers])
    fig, axes = plt.subplots(len(names), n_pan,
                             figsize=(5 * n_pan, 3.6 * len(names)),
                             squeeze=False)
    for row, ds in enumerate(names):
        arr, paired = results[ds]
        _panels(axes[row], ds, arr, paired)
    _legend(fig, methods, imputers)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = OUT / "imputers_all.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)


# ── Logs ─────────────────────────────────────────────────────────

def save(ds_name, arr, tables, macros, paired, config):
    np.savez(LOGS / f"imputers_{ds_name}.npz", **arr)
    write_csv(paired, LOGS / f"imputers_{ds_name}_paired.csv", PAIRED_FIELDS)
    doc = {"script": "eval_imputers.py", "dataset": ds_name,
           "config": config, "tables": tables, "macros": macros}
    path = LOGS / f"imputers_{ds_name}.json"
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, allow_nan=False)
    print(f"Results saved to {LOGS}/imputers_{ds_name}.{{npz,json}} and "
          f"_paired.csv ({len(macros)} macros)", flush=True)


def load_log(ds_name):
    """(arr, config) of an earlier run (LOGS/imputers_{ds}.npz and the
    config of its JSON; runtime of the original run kept)."""
    with np.load(LOGS / f"imputers_{ds_name}.npz", allow_pickle=True) as d:
        arr = {key: d[key] for key in d.files}
    path = LOGS / f"imputers_{ds_name}.json"
    config = {}
    if path.is_file():
        with open(path) as f:
            config = json.load(f).get("config", {})
    config["from_logs"] = True
    return arr, config


# ── Per-repetition shards ────────────────────────────────────────

# Shards live in a subdirectory, so that no glob of LOGS/imputers_*.npz
# (paired_stats, make_numbers) sees them as a dataset
SHARD_DIR = "imputers_shards"
# Keys of the log that are not per split (the rest have the split axis 0)
NON_SPLIT_KEYS = ("benchmark_names", "ks", "ranks", "lambda_ratios",
                  "methods", "imputers")
SHARD_KEYS = ("shard_rep", "shard_runtime_s", "shard_workers")


def shard_path(ds_name, rep):
    """LOGS/imputers_shards/imputers_{ds}_rep{rep}.npz"""
    return LOGS / SHARD_DIR / f"imputers_{ds_name}_rep{int(rep)}.npz"


def save_shard(ds_name, rep, arr, runtime_s, workers=1):
    """The per-split arrays of repetition `rep` (run_dataset(...,
    reps=[rep])) with its runtime and worker processes (--workers), for
    merge_shards."""
    path = shard_path(ds_name, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arr, shard_rep=np.int64(rep),
             shard_runtime_s=np.float64(runtime_s),
             shard_workers=np.int64(workers))
    print(f"Shard saved to {path}", flush=True)
    return path


def _is_split_key(key, val):
    return key not in NON_SPLIT_KEYS and np.ndim(val) >= 1


def merge_shards(ds_name, n_repeats):
    """(arr, runtime_s, workers): the log of a single run_dataset over all
    n_repeats repetitions, from the shards of repetitions 0 .. n_repeats −
    1 (shard_path).  Per-split arrays are concatenated in repetition order
    (cv_splits is rep-major, so the rows are those of the single-process
    run); every other array and scalar must agree between the shards
    (ValueError otherwise: shards of different settings or data);
    runtime_s is the sum of the shards' wall-clock runtimes (each that of
    one single-threaded process, or of its --workers processes), workers
    the shards' --workers (one number if they agree, else the list in
    repetition order; 1 for shards written before --workers).  Every split
    depends only on itself (its estimate, selections, inner permutation
    Split.seed(SEED + 21) and fits), so the merged arrays equal the
    single-process run's bit for bit, whatever the shards' workers
    (test_eval_imputers checks it)."""
    parts = []
    for rep in range(n_repeats):
        path = shard_path(ds_name, rep)
        if not path.is_file():
            raise ValueError(f"missing shard {path}")
        with np.load(path, allow_pickle=True) as d:
            part = {key: d[key] for key in d.files}
        if int(part["shard_rep"]) != rep or not np.all(part["rep"] == rep):
            raise ValueError(f"{path}: not the shard of repetition {rep}")
        if int(part["n_repeats"]) != n_repeats:
            raise ValueError(f"{path}: run with --repeats "
                             f"{int(part['n_repeats'])}, not {n_repeats}")
        parts.append(part)
    first = parts[0]
    keys = [k for k in first if k not in SHARD_KEYS]
    arr, runtime, workers = {}, 0.0, []
    for part in parts:
        if sorted(k for k in part if k not in SHARD_KEYS) != sorted(keys):
            raise ValueError(f"shards of {ds_name} have different keys")
        runtime += float(part["shard_runtime_s"])
        workers.append(int(part.get("shard_workers", 1)))
    for key in keys:
        vals = [p[key] for p in parts]
        if _is_split_key(key, vals[0]):
            arr[key] = np.concatenate(vals, axis=0)
        else:
            for v in vals[1:]:
                if v.shape != vals[0].shape or not np.array_equal(
                        v, vals[0]):
                    raise ValueError(f"shards of {ds_name} disagree on "
                                     f"{key!r}")
            arr[key] = vals[0]
    return arr, runtime, (workers[0] if len(set(workers)) == 1 else workers)


def shard_config_mismatch(arr, config):
    """Settings of the merging invocation (config, main's) that differ
    from those the shards ran with (their scalars): [] if all agree."""
    pairs = {"ks": "ks", "methods": "methods", "imputers": "imputers",
             "n_folds": "folds", "n_repeats": "repeats",
             "inner_frac": "inner_frac", "patience": "patience",
             "tol": "tol", "max_iter": "max_iter", "hard_tol": "hard_tol",
             "hard_max_iter": "hard_max_iter", "rank_cap": "rank_cap",
             "hard_solver": "hard_solver", "hard_init": "hard_init",
             "refit": "refit", "solver": "solver",
             "min_improve": "min_improve",
             "min_target_obs": "min_target_obs", "pred_clip": "pred_clip"}
    bad = []
    for key, cfg_key in pairs.items():
        if key not in arr:
            bad.append(key)
            continue
        a = np.asarray(arr[key])
        c = np.asarray(config[cfg_key])
        same = (a.shape == c.shape and (np.array_equal(a.astype(str),
                                                       c.astype(str))
                                        if a.dtype.kind in "US" else
                                        np.array_equal(a, c)))
        if not same:
            bad.append(key)
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Gaussian conditional vs fitted matrix-completion "
                    "imputers (SoftImpute, HardImpute) at the selections of "
                    "eval_baselines.py on the matched repeated-CV splits.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names, read from "
                             "data/<name>.matrix.npz (default: %(default)s)")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions (default %(default)s)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--ks", type=int, nargs="+", default=KS,
                        help="selection sizes (default %(default)s)")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=ALL_METHODS,
                        help="selection methods (default %(default)s)")
    parser.add_argument("--imputers", nargs="+", default=IMPUTERS,
                        choices=IMPUTERS,
                        help="imputers; gaussian always runs (default "
                             "%(default)s)")
    parser.add_argument("--n-lambda", type=int, default=N_LAMBDA,
                        help="SoftImpute lambda grid size (default "
                             "%(default)s)")
    parser.add_argument("--lambda-min-ratio", type=float,
                        default=LAMBDA_MIN_RATIO,
                        help="smallest lambda / lambda_max (default "
                             "%(default)s)")
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS,
                        help="HardImpute ranks, increasing (default "
                             "%(default)s; capped below N and, unless "
                             "--no-rank-cap, at k)")
    parser.add_argument("--inner-frac", type=float, default=INNER_FRAC,
                        help="share of the training models observing the "
                             "selection held out for the hyperparameter "
                             "choice (default %(default)s)")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="stop a path after this many values without "
                             "inner improvement; 0 = full path (default "
                             "%(default)s)")
    parser.add_argument("--tol", type=float, default=TOL,
                        help="SoftImpute: relative squared change of the "
                             "missing cells (default %(default)s)")
    parser.add_argument("--max-iter", type=int, default=MAX_ITER,
                        help="SoftImpute iterations per lambda (default "
                             "%(default)s)")
    parser.add_argument("--hard-tol", type=float, default=HARD_TOL,
                        help="HardImpute: relative squared change of the "
                             "missing cells (default %(default)s)")
    parser.add_argument("--hard-max-iter", type=int, default=HARD_MAX_ITER,
                        help="HardImpute iterations per fit, the same cap "
                             "on every dataset (default %(default)s)")
    parser.add_argument("--hard-solver", choices=HARD_SOLVERS,
                        default=HARD_SOLVER,
                        help="HardImpute iteration: svd (Mazumder et al.'s "
                             "Hard-Impute, refill with the rank-r SVD; "
                             "default) or lsq (exact least-squares refill; "
                             "a diagnostic, its iterates can diverge)")
    parser.add_argument("--hard-init", choices=HARD_INITS,
                        default=HARD_INIT,
                        help="HardImpute start: soft (the SoftImpute fit at "
                             "its chosen lambda; default) or zero")
    parser.add_argument("--no-rank-cap", action="store_true",
                        help="HardImpute ranks above k too (default: "
                             "r <= k)")
    parser.add_argument("--no-refit", action="store_true",
                        help="report the inner-holdout fit at the chosen "
                             "value (no refit with all training cells)")
    parser.add_argument("--solver", choices=SOLVERS, default=SOLVER,
                        help="SoftImpute solver: fista (Nesterov-accelerated "
                             "SoftImpute), exact (safeguarded exact row "
                             "steps) or plain (default %(default)s)")
    parser.add_argument("--min-improve", type=float, default=MIN_IMPROVE,
                        help="inner R² gain that counts as an improvement "
                             "on a path (default %(default)s)")
    parser.add_argument("--from-logs", action="store_true",
                        help="recompute summaries, paired CSVs and figures "
                             "from existing LOGS/imputers_<ds>.npz")
    parser.add_argument("--reps", type=int, nargs="+", default=None,
                        help="shard: run only these CV repetitions (of "
                             "--repeats) and write one LOGS/imputers_shards/"
                             "imputers_<ds>_rep<r>.npz per repetition, no "
                             "summary (then --merge)")
    parser.add_argument("--merge", action="store_true",
                        help="merge the shards of repetitions 0 .. "
                             "--repeats-1 into LOGS/imputers_<ds>.npz and "
                             "write the summaries, paired CSV and figures, "
                             "as a single run would")
    parser.add_argument("--workers", type=int, default=1,
                        help="evaluate the splits of a dataset (or of a "
                             "--reps shard) in this many spawned single-"
                             "threaded processes; the logs do not depend "
                             "on it (default %(default)s)")
    args = parser.parse_args(argv)
    if args.folds < 2 or args.repeats < 1 or min(args.ks) < 1:
        parser.error("need --folds >= 2, --repeats >= 1 and --ks >= 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if sum([args.from_logs, args.merge, args.reps is not None]) > 1:
        parser.error("--from-logs, --merge and --reps exclude each other")
    if args.reps is not None and (
            not args.reps or len(set(args.reps)) != len(args.reps)
            or not all(0 <= r < args.repeats for r in args.reps)):
        parser.error(f"--reps must be distinct repetitions in 0.."
                     f"{args.repeats - 1}")
    if not 0.0 < args.inner_frac < 1.0:
        parser.error("--inner-frac must lie in (0, 1)")
    if args.n_lambda < 1 or not 0.0 < args.lambda_min_ratio <= 1.0:
        parser.error("need --n-lambda >= 1 and 0 < --lambda-min-ratio <= 1")
    if sorted(set(args.ranks)) != list(args.ranks) or min(args.ranks) < 1:
        parser.error("--ranks must be increasing positive integers")
    if (args.patience < 0 or min(args.max_iter, args.hard_max_iter) < 1
            or min(args.tol, args.hard_tol) <= 0 or args.min_improve < 0):
        parser.error("need --patience >= 0, --max-iter, --hard-max-iter "
                     ">= 1, --tol, --hard-tol > 0, --min-improve >= 0")
    if sorted(set(args.ks)) != list(args.ks):
        parser.error("--ks must be increasing")
    missing = [ds for ds in args.datasets
               if not (DATA / f"{ds}.matrix.npz").is_file()]
    if missing and not (args.from_logs or args.merge):
        parser.error(f"no {DATA}/<name>.matrix.npz for {missing}")
    imputers = ["gaussian"] + [imp for imp in FITTED if imp in args.imputers]
    config = {"repeats": args.repeats, "folds": args.folds, "ks": args.ks,
              "methods": args.methods, "imputers": imputers,
              "holdout_frac": HOLDOUT_FRAC, "seed": SEED,
              **protocol_config(), "clip": CLIP, "ridge": RIDGE,
              "inner_seed_offset": INNER_SEED_OFFSET,
              "inner_frac": args.inner_frac, "n_lambda": args.n_lambda,
              "lambda_min_ratio": args.lambda_min_ratio,
              "ranks": args.ranks, "patience": args.patience,
              "tol": args.tol, "max_iter": args.max_iter,
              "hard_tol": args.hard_tol, "hard_max_iter": args.hard_max_iter,
              "rank_cap": not args.no_rank_cap,
              "hard_solver": args.hard_solver, "hard_init": args.hard_init,
              "refit": not args.no_refit, "solver": args.solver,
              "min_improve": args.min_improve}

    def compute(ds_name, reps=None):
        return run_dataset(
            ds_name, args.ks, args.methods, imputers, args.folds,
            args.repeats, args.n_lambda, args.lambda_min_ratio,
            args.ranks, args.inner_frac, args.patience, args.tol,
            args.max_iter, not args.no_refit, args.solver,
            args.min_improve, args.hard_tol, args.hard_max_iter,
            not args.no_rank_cap, args.hard_solver, args.hard_init,
            reps=reps, workers=args.workers)

    total_t0 = time.time()
    if args.reps is not None:            # shards only: no summaries
        for ds_name in args.datasets:
            for rep in args.reps:
                t0 = time.time()
                arr = compute(ds_name, [rep])
                save_shard(ds_name, rep, arr, time.time() - t0,
                           args.workers)
        print(f"\nAll shards completed in {time.time() - total_t0:.0f}s",
              flush=True)
        return
    results = {}
    for ds_name in args.datasets:
        ds_t0 = time.time()
        if args.from_logs:
            arr, cfg = load_log(ds_name)
        else:
            if args.merge:
                arr, runtime, workers = merge_shards(ds_name, args.repeats)
                bad = shard_config_mismatch(arr, config)
                if bad:
                    parser.error(f"--merge {ds_name}: the shards were run "
                                 f"with other settings ({', '.join(bad)})")
                print(f"{ds_name}: merged {args.repeats} shards "
                      f"({runtime:.0f} s in the shard processes)",
                      flush=True)
            else:
                arr = compute(ds_name)
                runtime = time.time() - ds_t0
                workers = args.workers
            cfg = dict(config, runtime_s=round(runtime, 1), workers=workers)
            # per-split arrays first, so a failing summary loses nothing
            np.savez(LOGS / f"imputers_{ds_name}.npz", **arr)
        paired = paired_rows(ds_name, arr)
        tables, macros = summarize(ds_name, arr, paired)
        if cfg.get("runtime_s") is not None:
            macros[f"ImpRuntimeMin{camel(ds_name)}"] = round(
                float(cfg["runtime_s"]) / 60.0, 2)
        print_summary(ds_name, arr, tables, paired)
        plot_dataset(ds_name, arr, paired)
        save(ds_name, arr, tables, macros, paired, cfg)
        results[ds_name] = (arr, paired)
        print(f"{ds_name.upper()} completed in {time.time() - ds_t0:.0f}s",
              flush=True)
    if len(results) > 1:
        plot_all(results)
    print(f"\nAll datasets completed in {time.time() - total_t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
