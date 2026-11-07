r"""Mechanism analysis: why the selection objectives rank as they do.

On the matched splits (cv_split.cv_splits, 10% holdout) greedy trace has
the highest mean R^2 at k = 5 and 10 on MMLU, MTEB and merged (BenchPress:
every method negative and noisy), MI's first pick on merged is far worse
than trace's, hub is worst on MMLU and MTEB, and random is competitive on
the sparse matrices (merged, BenchPress).  This script recomputes, split by
split, everything needed to explain that ranking from the objectives
themselves and writes per-split arrays, summary tables, macros and
figures.  It re-runs the selectors of eval_baselines.py (same estimator,
same eligibility mask, same common random numbers, so the train-side
selections and R^2 are those of logs/baselines_{ds}.npz; check with
--check-logs), facility location (facloc, the paper's headline method
since 2026-10-19) included; eval_baselines' similarity variants
facloc_abs / facloc_sq are not re-run here.  The protocol rules of
every script apply to the realized R^2 (impute_and_evaluate: the split's
targets, cv_split.target_mask, rule D1, and predictions clipped to the
valid score range, score_bounds.py, rule D2); the expected R^2 (exp_r2_*)
score the same target cells (D1) but are analytic expectations of the
unclipped linear predictor (D2 has no closed form there), so the "rest"
term of the gap decomposition includes the effect of the clip.
(entropy_rf was dropped on 2026-10-18, rule D3.)

Objectives as functions of principal blocks
-------------------------------------------
Let R be the N x N correlation matrix, A the selected set (|A| = k), C =
V \ A its complement, R2 = R @ R and P = R^{-1}.  Then

  trace    F_tr(A) = tr(R_{.A} R_AA^{-1} R_{A.}) = tr(R_AA^{-1} (R2)_AA)
  entropy  F_H(A)  = log det R_AA
  MI       F_MI(A) = 1/2 [log det R_AA + log det R_CC - log det R]
                   = 1/2 [log det R_AA + log det P_AA]       (Jacobi:
                     det R_CC = det R det P_AA)
  hub      sum_{j in A} mean_{i != j} |R_ij|                  (modular)
  facloc   F_FL(A) = sum_{j=1..N} max_{i in A} s_ij,  s_ij = max(R_ij, 0),
                     s_jj = 1 (baseline_select.facility_location_select)

and, if every cell is observed, the population value of the paper's
metric (pooled R^2 of the best linear predictor over the test models and
the unselected benchmarks, standardized units, ratio of expectations) is

  R2_pop(A) = tr(R_CA R_AA^{-1} R_AC) / (N - k) = (F_tr(A) - k) / (N - k),

an increasing affine function of F_tr at fixed k (only second moments
enter; no Gaussianity is needed).  Greedy trace therefore greedily
maximizes a plug-in version of the metric, R2_pop under the training
estimate R_tr: greedy, so not the exact maximizer (brute_trace_same), and
under R_tr, so neither the population value nor the realized,
mask-dependent R^2.  In the eigenbasis R = V diag(lam) V^T the three
blocks weight direction i by lam_i^2 (R2: trace), lam_i (R: entropy) and
lam_i and 1/lam_i (R and P: MI).  At k = 1, F_tr = (R2)_jj = sum_i R_ij^2
= 1 + sum_{i != j} R_ij^2 (one plus N - 1 times j's mean squared
correlation; the hub score is the mean |r|, so the two rank alike only
roughly) and F_MI = 1/2 log P_jj = -1/2 log(1 - R^2_{j|rest}): MI's first
pick is the benchmark best predicted BY the rest, trace's the one that
best predicts the rest.  Later MI gains are
  Delta_MI(j | A) = 1/2 log sigma^2_{j|A} + 1/2 log [(R_CC)^{-1}]_jj,
C = V \ A (j in C; eligible or not): the entropy novelty plus a bonus that
equals 1/2 log P_jj only at A = {}.

Facility location vs trace.  Both objectives credit every benchmark j of
the ground set (selected ones included) for how well A covers it: trace
with its joint explained variance, F_tr(A) = sum_j R^2(j | A), where
R^2(j | A) = (R_jA R_AA^{-1} R_Aj) / R_jj is the regression of j on all of
A; facility location with the single best clipped correlation, max_{i in
A} max(R_ij, 0): r, not r^2, and no joint regression (two proxies of one
target add nothing beyond the better of them, a negatively correlated one
nothing at all).  F_FL is monotone submodular (greedy within 1 - 1/e of
its optimum); F_tr is not submodular.  With s_ij = R_ij^2 (eval_baselines'
facloc_sq) the k = 1 scores coincide (1 + sum_{i != j} R_ij^2); the
facility-location section ("facloc" tables, fl_* arrays, ObjFl... macros)
measures, per split, how the two differ in realized R^2, in which
benchmarks they pick, and in how well their greedy gains predict the
realized R^2 gains, along their own greedy paths and over all candidates
of a greedy state.

Theory status (theory_*): F_tr is monotone but not submodular (gains can
grow along the greedy path: trace_increase_frac).  Das and Kempe's greedy
guarantee F_tr(greedy_k) >= (1 - e^{-gamma_{U,k}}) F_tr(OPT_k) uses the
submodularity ratio gamma_{U,k} itself; trace_gamma computes it exactly
for U = greedy trace's set of size --gamma-k (theory_trace_gamma_bound =
1 - e^{-gamma}).  Only its spectral corollaries are vacuous here:
gamma >= lam_min(R) (trace_spectral_bound) and gamma >= the 2k-sparse
eigenvalue lam_min(R, 2k) <= 1 - max_{a != b} |R_ab| (trace_sparse_eig_ub,
trace_sparse_bound_ub, over eligible pairs).  F_MI is not monotone: a
candidate's gain Delta_MI(j | A) can be negative (R = I but r_01 = 0.95:
MI({0}) = 1.164 > MI({0, 1}) = 0); mi_chosen_neg_gains counts only greedy
MI's own picks, mi_min_gain / mi_neg_gain_frac every eligible candidate
along greedy MI's path, mi_min_pair_gain every eligible pair {a} -> {a, j}.
Their negative part is a lower bound on the epsilon of Krause et al.'s
approximate monotonicity (gains >= -epsilon for |A| <= 2k), which their
(1 - 1/e) OPT - k epsilon guarantee needs small (mi_eps_lb, mi_keps_rel at
k = EPS_K).  F_H = log det R_AA is monotone non-increasing on a
correlation matrix (each step adds log sigma^2_{j|A} <= 0); the
differential entropy 1/2 log det(2 pi e R_AA) grows by 1/2 log(2 pi e
sigma^2_{j|A}), which is negative iff the conditional variance is below
1/(2 pi e) (entropy_below_floor: unit-dependent, as differential entropy
is).

What is computed (per split s, method m, k = 1..k_max)
-------------------------------------------------------
Selections run on two correlation matrices with the split's eligibility
mask: the training estimate R_tr = estimate_corr(B_train, O_train) (the
paper's selection) and the oracle R_or, the same estimator on ALL models
with its size-dependent rules (EM vs pairwise, regularization, shrinkage)
set as for the splits' training size (estimate_corr_oracle; --oracle full
uses estimate_corr(B, O) as is).  R_or serves H4; it uses the validation
models by construction and is a diagnostic only, never a method.  For
each selected set (train- and
oracle-selected) the script records

  r2, r2_oimp       realized pooled R^2 on the validation models
                    (eval_entropy_vs_mi.impute_and_evaluate, train
                    standardization) with imputation weights from R_tr /
                    from R_or
  pop_r2_tr/_or     R2_pop(A) under R_tr / R_or (H1 identity, all cells)
  exp_r2_tr/_or     expected R^2 of the actual predictor (weights from
                    R_tr, ridge 0.01) on the validation models' actual
                    observation pattern, with the cells and models
                    impute_and_evaluate scores, if the truth were R_tr /
                    R_or (missing-data version of the identity)
  trace_tr, logdet_tr/_or, mi_tr/_or, hub_tr, facloc_tr   objective
                    values (facloc_tr = F_FL(A) under R_tr)
  mean_abs_r_tr/_or, max_abs_r_or, eff_rank_tr/_or   redundancy of A:
                    mean / max |r| over pairs in A, participation ratio
                    k^2 / ||R_AA||_F^2 divided by k (1 = orthogonal set)
  cov_models, cov_cells   fraction of validation models (cells) the metric
                    scores: a model none of whose observed benchmarks is
                    selected is skipped by impute_and_evaluate
  k_eff, k_obs      mean number of selected benchmarks a validation model
                    observed (scored models / all models; observed_picks)
  n_suites          distinct suites ("suite/benchmark" prefixes) in A, NaN
                    if names carry no suite

plus, per split: the k = 1 statistics of every eligible singleton
(single_*: realized R^2, R2_pop, R^2_{j|rest}, MI score, hub score, share
of P_jj (not of the MI score 1/2 log P_jj) carried by the bottom 10% of
the eigenvalues, share of (R2)_jj carried by the top 10%, facility-location
score F_FL({j}) = sum_i s_ij); N_RANDOM_SETS
uniformly random eligible k-sets for k in RANDOM_SET_KS (rand_*: the
quantities above), drawn from the common random numbers Split.seed(SEED +
RANDOM_SETS_SEED_OFFSET), offset 19, distinct from the random baseline's
7 and the retired entropy_rf tag 11; the exact maximizers of F_tr, F_H,
F_MI and F_FL under R_tr over all eligible k-sets for k = 2..--brute-k
(brute force, block formulas) with their realized R^2; theory diagnostics
(above: trace's
diminishing-returns violations and submodularity ratio, MI's negative
gains, entropy steps with conditional variance below 1/(2 pi e),
eigenvalues of R_tr); Spearman correlations between the train and oracle
k = 1 score vectors (score stability); Jaccard overlap of train- and
oracle-selected sets.

Facility-location section (per split, train side; fl_* arrays, needs
facloc and trace among the methods): the overlap of the two greedy
selections at k = 1..k_max; along each greedy path P in (facloc, trace)
the gains of each objective O in (facloc, trace) for P's picks (O = P:
P's own greedy gains), the realized R^2 gains r2(A_t) - r2(A_{t-1})
(r2(empty set) = 0: predicting the training mean, the zero of the
standardized scale, is R^2 = 0 under impute_and_evaluate's SS_tot) and
their Spearman correlation over the steps; and the gain landscape at the
states A_t of the first --fl-landscape-k picks (default FL_LAND_K = 10):
for every eligible candidate j not in A_t the realized gain r2(A_t + j) -
r2(A_t), its Spearman correlation over the candidates with each
objective's gain, and the regret of each objective's pick (the best
candidate's realized gain minus that of the candidate O picks at A_t: for
O = P the pick P made, path[t]; for O != P the pick of O's greedy step
from A_t, argmax_j gain_O(j) with O's selector's near-tie rule
(_greedy_choice): facloc's near-ties broken by explained variance, then
the lowest index, as facility_location_select; trace's near-ties to the
lowest index, as greedy_trace_select).  t = 0 reuses the singletons'
realized R^2; each later state costs one impute_and_evaluate per
candidate.

Gap decomposition (tables "gap", macros ObjGap*): the in-sample-to-
realized gap r2 - R2_pop splits, under one truth S at a time (train R_tr
or oracle R_or), into mask = exp_r2_S - pop_r2_S (the validation models'
observation pattern and ridge; under S = R_or it also contains the cost of
R_tr's weights) and rest = r2 - exp_r2_S (sampling and misfit), on the
splits where all five quantities are finite.  Mixing the truths (e.g.
exp_r2_or - pop_r2_or against r2 - pop_r2_tr) is not a decomposition.

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir())
-------------------------------------------------------------------------
{tag} is {ds} for the default --oracle matched and {ds}_fulloracle for
--oracle full (whose macros start with ObjFull instead of Obj), so both
runs can share one experiment root.
  LOGS/analyze_objectives_{tag}.npz
    rep, fold, split_id, n_train, n_val, n_eligible   (S,) int
    data_sha256      sha256 of the input data/{ds}.matrix.npz (also in the
                     JSON config with its data/SHA256SUMS status)
    eligible (S, N) bool, estimation_failed (S,) bool
    {m}_{q}          (S, k_max) train-selected set of method m, q above
    {m}_osel_{q}     (S, k_max) oracle-selected set (same q)
    {m}_selected, {m}_osel_selected   (S, k_max) int, -1 padded (pca: its
                     largest set; its per-k sets are in pca_selected_per_k,
                     pca_osel_selected_per_k (S, k_max, k_max))
    {m}_jacc_oracle  (S, k_max) Jaccard of train- vs oracle-selected sets
    single_{q}       (S, N) singleton statistics (NaN if ineligible)
    rand_{q}         (S, N_RANDOM_SETS, len(RANDOM_SET_KS)); rand_ks
    brute_{obj}_{field}   (S, brute_k) for obj in trace, entropy, mi,
                     facloc and k = 1..brute_k (column k-1; k = 1 is the
                     singleton argmax): value (optimum), greedy (greedy
                     value), r2 (realized R^2 of the optimum), r2_greedy,
                     same (greedy set == optimum)
    theory_*         (S,) diagnostics (module docstring, theory_diagnostics);
                     stab_{obj} (S,) score stability (obj in trace, mi,
                     hub, facloc)
    path_{key}       (S, k_max) greedy-path terms (path_terms): novelty
                     1/2 log sigma^2_{j|A} of the picks of mi, entropy,
                     trace and facloc, MI's gain and its bonus 1/2 log
                     [(R_CC)^{-1}]_jj (C the complement before the pick),
                     and min_gain_mi, the smallest MI gain of any eligible
                     candidate at that step (mi_gain_landscape)
    fl_shared_trace, fl_jacc_trace   (S, k_max) facility-location section:
                     picks shared by the size-k facloc and trace selections
                     and their Jaccard overlap (NaN without both)
    fl_path_{P}_dr2  (S, k_max) realized R^2 gains along greedy path P
                     (P in facloc, trace); fl_path_{P}_gain_{O} (S, k_max)
                     the gains of objective O (facloc, trace) for P's
                     picks; fl_path_{P}_rho_{O} (S,) their Spearman
                     correlation over the steps
    fl_land_{P}_rho_{O}, fl_land_{P}_regret_{O}, fl_land_{P}_best_dr2,
    fl_land_{P}_n_cand   (S, fl_land_k) gain landscape at the state before
                     pick t + 1 of path P (column t): Spearman over the
                     candidates of O's gain and the realized gain, regret
                     of O's pick (P's own pick for O = P), the best
                     realized gain, the number of candidates
    oracle_corr      (N, N) R_or; oracle_lam_min its smallest eigenvalue
    benchmark_names, methods, k_max, holdout_frac, n_folds, n_repeats,
    seed, min_train_obs, min_target_obs, pred_clip,
    random_sets_seed_offset, n_random_sets, brute_k,
    gamma_k, fl_land_k
  LOGS/analyze_objectives_{tag}.json  config, summary tables, "macros"
  LOGS/analyze_objectives_{tag}_paired.csv  paired R^2 differences
                     (paired_stats.paired_stats) of oracle vs train
                     selection / imputation per method, of method pairs
                     under oracle selection, of exact optimum vs greedy,
                     and (comparison "train") trace - facloc of the
                     train-side selections; columns dataset, holdout_frac,
                     k, comparison, method_a, method_b + STAT_FIELDS
  OUT/analyze_objectives_{tag}.pdf    six panels (singletons, surrogate
                     rank correlations, predicted vs realized, redundancy,
                     oracle selection, coverage)
  OUT/analyze_objectives_facloc_{tag}.pdf  facility location vs trace:
                     paired realized R^2 difference by k, overlap of the
                     selections, gain-landscape correlations by pick
  OUT/analyze_objectives_all.pdf      singletons and oracle-selection
                     panels of all datasets of the run (_all_fulloracle
                     for --oracle full)

Methods and invariance.  METHODS = CORE_METHODS (mi, entropy, trace,
pca, hub, random: paired_stats.METHOD_ORDER) + facloc.  facloc was added
on 2026-10-19; it draws no random numbers (each random stream has its own
generator: the random baseline Split.seed(SEED + 7), the random sets
Split.seed(SEED + 19)), so every per-method key of the other methods is
unchanged.  Every aggregate across methods keeps its key and its value
over CORE_METHODS (_trace_max_*_frac, _split_method_spearman_*,
_method_spearman_*, insample_gap "_all", oracle_effects and their macros
ObjMethodSpearman..., ObjInSampleGapMax{Ds}, ObjInSampleGapWorstK{Ds},
Obj{Cmp}{Stat}{Ds}); its twin over METHODS has the key suffix
_with_facloc (insample_gap "_all_with_facloc", table
"oracle_effects_with_facloc") and the macro infix WithFacloc after the
field name (ObjMethodSpearmanWithFacloc{Ds}{Q}K{k},
ObjInSampleGapMaxWithFacloc{Ds}, Obj{Cmp}WithFacloc{Stat}{Ds}).

Macros ("macros" in the JSON; letters only, make_numbers.camel names,
{Ds} dataset, {M} method, K{k} spelled k): Obj{Field}{Ds}{M}K{k} for the
per-method means (PopRsq, PopRsqOracle, ExpRsqOracle, Optimism, Redund,
EffRank, CoverCells, CoverModels, OracleSelRsq, OracleBothRsq,
OracleImpRsq, JaccOracle, FaclocVal (F_FL of the set), and OracleSelGain
/ OracleSelGainLo / OracleSelGainHi, the paired difference
oracle-selected minus train-selected with its Nadeau-Bengio 95%
interval), ObjSpearman{Ds}{Q}K{k}
(mean within-split Spearman between an in-sample quantity Q and the
realized R^2: singletons at k = 1, random sets at RANDOM_SET_KS),
ObjMethodSpearman{Ds}{Q}K{k} (the same across the CORE_METHODS' sets of
a split; ObjMethodSpearmanWithFacloc... across all METHODS),
ObjFirst{Ds}{M}{Field} (k = 1 picks), ObjSingle{Ds}{Field}
(all eligible singletons: RestRsqMin, RestRsqSplitMinMedian,
RestRsqMedian, PLowShareMedian), ObjBrute{Ds}{Obj}K{k}{Field} (Ratio is
the mean over splits of (F(greedy) - off) / (F(opt) - off), off = k for
trace, 0 for MI and facloc; RatioMin / RatioMedian its worst / median
split;
FRatio, FRatioMin the plain F(greedy) / F(opt) that Das and Kempe's bound
covers), ObjTheory{Ds}{Field} (split means; ObjTheory{Ds}{Field}Min /
Max / Median for the gain and bound diagnostics), ObjStab{Ds}{Obj},
ObjPath{Ds}{Key} (greedy-path terms, mean over picks 1-5),
ObjNbr{Ds}{M}{First|Set}{Tight|NClose|Relevance} (oracle-R
neighbourhoods of the k = 1 pick / size-5 set),
ObjGap{Total|Mask|MaskShare}{Train|Oracle}{Ds}{M}K{k} (gap
decomposition above), ObjInSampleGapMax{Ds}{M} / ObjInSampleGapWorstK{Ds}{M}
(largest |mean(r2 - pop_r2_tr)| over k = 1..k_max and its k; without {M}:
over CORE_METHODS, WithFacloc: over METHODS) and Obj{Cmp}{Stat}{Ds} for
Cmp in OracleSel, OracleImp, OracleBoth (all rows of that paired
comparison, every CORE_METHODS method and k; Obj{Cmp}WithFacloc{Stat}{Ds}
every method): NCompared, NSig (95% interval excludes 0), NSigPos,
NSigNeg, NLarge (|mean difference| > LARGE_DIFF = 0.05), MaxAbsDiff,
MedianHalfWidth, MaxHalfWidth.  Facility-location section (k in
KS_REPORT; P = path, O = objective, each Facloc or Trace):
ObjFlVsTrace{Diff|DiffLo|DiffHi|Win}{Ds}K{k} (realized R^2 facloc -
trace over the splits, Nadeau-Bengio 95% interval, share of splits with
facloc ahead), ObjFlVsTrace{Shared|Jacc|Same}{Ds}K{k} (mean share k^-1 |A_FL
∩ A_tr| of shared picks, mean Jaccard, share of splits with identical
sets), ObjFlPathRho{Ds}{P}Gain{O} (mean over splits of the Spearman of
O's gains and the realized R^2 gains along P's greedy path),
ObjFlLand{Rho|Regret}{Ds}{P}Gain{O} (gain landscape: mean over splits and
over picks 1-5) and ...K{k} (at pick k), ObjFlLandBestGain{Ds}{P}K{k}.

Usage:
    python analyze_objectives.py [--datasets mmlu mteb merged benchpress]
                                 [--repeats 5] [--folds 10] [--kmax 15]
                                 [--n-random-sets 30] [--random-ks 5 10]
                                 [--brute-k 3] [--gamma-k 2]
                                 [--fl-landscape-k 10]
                                 [--check-logs DIR]
                                 [--oracle matched|full] [--from-logs]
                                 [--workers 1]

--workers W evaluates the splits in W spawned single-threaded processes
(map_splits; keep OMP / OpenBLAS / MKL at one thread per process).  Every
split is a pure function of the dataset, the oracle correlation and the
split (objectives_split; its random sets and random baseline draw from
the split's own common random numbers), and the rows and progress lines
are assembled in split order, so the npz, JSON (apart from config.workers
and config.runtime_s), CSV and figures equal the serial run's and the
printed lines differ only in the per-split times
(test_analyze_objectives checks the arrays).  The oracle estimate before
the splits and the summaries after them stay serial.

--check-logs compares with DIR/baselines_{ds}.npz and, if DIR/../
manifest.json (run_all.sh) records the input data, checks that it was made
from the same data/{ds}.matrix.npz (sha256): a match against a log of
other data is no check.

Runtime (single-threaded, --repeats 5, --gamma-k 2; runtime_s in the JSON
config): MMLU ~3 min, MTEB ~6 min, merged and BenchPress ~15 min each on a
4090 host shared with another 40-process job (1.5 / 3 / 12 / 12 min in an
earlier run on a less loaded host); the theory diagnostics take < 0.1 s
per split, --gamma-k 3 about 3 s per merged split.  Facility location
(facloc as a seventh method, F_FL as a set quantity and brute-force
objective, the facility-location section with --fl-landscape-k 10)
added about 10% (2026-10-19, --repeats 1 on 4090-1: 379 -> 411 s for
the four datasets, 144 -> 161 s for --oracle full on merged).
"""

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import sys
import time
import warnings
from collections import Counter

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import (_cholesky_state, _complement_precision_diag,
                           _near_max, _tie_break, selection_at_k)
from cv_split import (MIN_TRAIN_OBS, N_REPEATS, cv_splits, eligible_mask,
                      target_mask)
from eval_entropy_vs_mi import (estimate_corr, impute_and_evaluate,
                                protocol_config)
from score_bounds import dataset_bounds
from baseline_select import facloc_similarity
from eval_baselines import (METHOD_COLOR, METHOD_LABEL, METHOD_STYLE,
                            run_selection)
from paired_stats import (STAT_FIELDS, orient, orient_pair, paired_stats,
                          write_csv)
from make_numbers import camel, int_words

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42                       # eval_baselines.SEED (selections must match)
RIDGE = 0.01                    # impute_and_evaluate's ridge
# Common random numbers of the random k-subsets of the surrogate analysis:
# Split.seed(SEED + 19); distinct from RANDOM_SEED_OFFSET = 7 (random
# baseline) and ENTROPY_RF_SEED_OFFSET = 11 (retired: entropy_rf).
RANDOM_SETS_SEED_OFFSET = 19
N_RANDOM_SETS = 30
RANDOM_SET_KS = [5, 10]
BRUTE_K = 3
GAMMA_K = 2                     # |U| = |S| bound of the submodularity ratio
EPS_K = 5                       # budget k of the Krause et al. epsilon bound
LARGE_DIFF = 0.05               # |paired R^2 difference| counted as large
SPECTRAL_SHARE = 0.1            # bottom / top share of the eigenvalues
CLOSE_R = 0.9                   # |r| of a "close" neighbour (cluster size)
ENTROPY_FLOOR = 1.0 / (2.0 * math.pi * math.e)
KS_REPORT = [1, 5, 10]

DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
# paired_stats.METHOD_ORDER: the methods of every aggregate across methods
# that predates facility location (module docstring, "Methods and
# invariance")
CORE_METHODS = ["mi", "entropy", "trace", "pca", "hub", "random"]
# facloc (facility location, the headline method since 2026-10-19) last:
# it ranks between hub and random in paired_stats' orientation and draws
# no random numbers
METHODS = CORE_METHODS + ["facloc"]
GREEDY_OF = {"trace": "trace", "entropy": "entropy", "mi": "mi",
             "facloc": "facloc"}
BRUTE_OBJECTIVES = ["trace", "entropy", "mi", "facloc"]

SET_QUANTITIES = ["r2", "r2_oimp", "pop_r2_tr", "pop_r2_or", "exp_r2_tr",
                  "exp_r2_or", "trace_tr", "logdet_tr", "logdet_or", "mi_tr",
                  "mi_or", "hub_tr", "mean_abs_r_tr", "mean_abs_r_or",
                  "max_abs_r_or", "eff_rank_tr", "eff_rank_or", "cov_models",
                  "cov_cells", "k_eff", "k_obs", "n_suites", "facloc_tr"]
SINGLE_QUANTITIES = ["r2", "pop_r2_tr", "pop_r2_or", "exp_r2_tr", "exp_r2_or",
                     "rest_r2_tr", "rest_r2_or", "mi_tr", "mi_or", "hub_tr",
                     "p_low_share_tr", "r2_top_share_tr", "cov_models",
                     "n_obs_train", "facloc_tr"]
# In-sample (or oracle) quantities whose rank correlation with the
# realized R^2 the surrogate analysis reports: key -> (array, label)
SURROGATES = {
    "trace": ("pop_r2_tr", "trace $R^2_{pop}$ (train)"),
    "entropy": ("logdet_tr", "entropy log det (train)"),
    "mi": ("mi_tr", "MI (train)"),
    "hub": ("hub_tr", "hub (train)"),
    "exp_train": ("exp_r2_tr", "expected $R^2$, test mask (train)"),
    "exp_oracle": ("exp_r2_or", "expected $R^2$, test mask (oracle)"),
    "pop_oracle": ("pop_r2_or", "trace $R^2_{pop}$ (oracle)"),
    "facloc": ("facloc_tr", "facility location $F_{FL}$ (train)"),
}
SINGLE_SURROGATE_KEY = {"trace": "pop_r2_tr", "entropy": None, "mi": "mi_tr",
                        "hub": "hub_tr", "exp_train": "exp_r2_tr",
                        "exp_oracle": "exp_r2_or", "pop_oracle": "pop_r2_or",
                        "facloc": "facloc_tr"}
# In-sample quantities of the across-method rank agreement (summarize):
# within-split Spearman with the realized R^2 over the methods' sets, and
# Spearman of the method means (facloc_tr, F_FL, added last)
METHOD_SPEARMAN_QS = ["pop_r2_tr", "exp_r2_tr", "exp_r2_or", "logdet_tr",
                      "mi_tr", "hub_tr", "facloc_tr"]
METHOD_MEAN_SPEARMAN_FS = ["pop_r2_tr", "exp_r2_or", "logdet_tr", "mi_tr",
                           "mean_abs_r_or", "eff_rank_or", "cov_cells",
                           "facloc_tr"]
# k = 1 score vectors of the score-stability check (train vs oracle R)
STAB_OBJECTIVES = ["trace", "mi", "hub", "facloc"]
# Facility-location section: greedy paths whose states are examined and
# objectives whose gains are compared (module docstring)
FL_PATHS = ["facloc", "trace"]
FL_GAINS = ["facloc", "trace"]
FL_LAND_K = 10                  # gain landscape before picks 1..10

DISPLAY_NAME = {"mmlu": "MMLU", "mteb": "MTEB", "merged": "Merged",
                "benchpress": "BenchPress"}


def display_name(ds_name):
    return DISPLAY_NAME.get(ds_name, ds_name)


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


def out_tag(ds_name, oracle="matched"):
    """File tag of a run: {ds} (matched oracle) or {ds}_fulloracle."""
    return ds_name if oracle == "matched" else f"{ds_name}_fulloracle"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_provenance(ds_name):
    """{"file", "sha256", "sha256sums"} of DATA/{ds}.matrix.npz; sha256sums
    is its status in DATA/SHA256SUMS: "match", "mismatch", "unlisted" or
    "no SHA256SUMS" (a stale copy of the data shows as a mismatch)."""
    path = DATA / f"{ds_name}.matrix.npz"
    sha = _sha256_file(path)
    sums = DATA / "SHA256SUMS"
    status = "no SHA256SUMS"
    if sums.is_file():
        listed = {}
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                listed[parts[1].lstrip("*")] = parts[0]
        want = listed.get(path.name)
        status = ("unlisted" if want is None
                  else "match" if want == sha else "mismatch")
    return {"file": path.name, "sha256": sha, "sha256sums": status}


# ── Objectives as block functions ────────────────────────────────

class Spectral:
    """R with the derived matrices the objectives need: R2 = R @ R,
    P = R^{-1} (eigendecomposition, eigenvalues floored at 1e-300),
    lam / V (ascending eigh), hub = mean_{i != j} |R_ij|, fl = the
    facility-location similarity s_ij = max(R_ij, 0), s_jj = 1
    (baseline_select.facloc_similarity, as facility_location_select
    builds it), fl_cols[j] its column j (the cover s_.j that j provides)
    and fl_single = F_FL({j}) = sum_i s_ij."""

    def __init__(self, R):
        R = np.asarray(R, dtype=np.float64)
        self.R = R
        self.N = R.shape[0]
        lam, V = np.linalg.eigh(R)
        self.lam, self.V = lam, V
        lam_c = np.maximum(lam, 1e-300)
        self.R2 = R @ R
        self.P = (V / lam_c[None, :]) @ V.T
        self.P = 0.5 * (self.P + self.P.T)
        self.logdet = float(np.sum(np.log(lam_c)))
        absR = np.abs(R)
        np.fill_diagonal(absR, 0.0)
        self.hub = absR.sum(axis=1) / max(self.N - 1, 1)
        self.fl = facloc_similarity(R, "clipped")
        self.fl_cols = np.ascontiguousarray(self.fl.T)
        self.fl_single = np.sum(self.fl, axis=0)


def _blocks(M, sets):
    """(T, k, k) principal blocks M[A, A] of the index sets (T, k)."""
    return M[sets[:, :, None], sets[:, None, :]]


def _batched_logdet(blocks):
    sign, ld = np.linalg.slogdet(blocks)
    return np.where(sign > 0, ld, -np.inf)


def _trace_of_blocks(RA, R2A):
    try:
        return np.trace(np.linalg.solve(RA, R2A), axis1=1, axis2=2)
    except np.linalg.LinAlgError:
        return np.array([np.trace(np.linalg.pinv(a) @ b)
                         for a, b in zip(RA, R2A)])


def trace_objective(sp, sets):
    """F_tr = tr(R_AA^{-1} (R2)_AA) of the index sets `sets` ((T, k) int)
    under the Spectral sp (the trace of set_objectives, alone)."""
    sets = np.atleast_2d(np.asarray(sets, dtype=np.int64))
    return _trace_of_blocks(_blocks(sp.R, sets), _blocks(sp.R2, sets))


def facloc_objective(sp, sets):
    """F_FL(A) = sum_j max_{i in A} s_ij of the index sets `sets` ((T, k)
    int) under the Spectral sp: (T,) array, the facloc_values of
    facility_location_select for its own sets (same cover vector, summed
    alike)."""
    sets = np.atleast_2d(np.asarray(sets, dtype=np.int64))
    cover = sp.fl_cols[sets[:, 0]]
    for c in range(1, sets.shape[1]):
        cover = np.maximum(cover, sp.fl_cols[sets[:, c]])
    return np.sum(cover, axis=1)


def _brute_values(sp, sets):
    """{obj: (T,) values} of the BRUTE_OBJECTIVES for the index sets
    `sets` (set_objectives and facloc_objective)."""
    tr, ld, mi = set_objectives(sp, sets)
    return {"trace": tr, "entropy": ld, "mi": mi,
            "facloc": facloc_objective(sp, sets)}


def set_objectives(sp, sets):
    """F_tr, log det R_AA and F_MI of the index sets `sets` ((T, k) int)
    under the Spectral sp (module docstring): returns (trace, logdet, mi)
    arrays of length T."""
    sets = np.atleast_2d(np.asarray(sets, dtype=np.int64))
    RA = _blocks(sp.R, sets)
    R2A = _blocks(sp.R2, sets)
    PA = _blocks(sp.P, sets)
    ld = _batched_logdet(RA)
    mi = 0.5 * (ld + _batched_logdet(PA))
    tr = _trace_of_blocks(RA, R2A)
    return tr, ld, mi


def pop_r2(sp, A):
    """R2_pop(A) = (F_tr(A) - k) / (N - k): population pooled R^2 of the
    best linear predictor of the unselected benchmarks (every cell
    observed)."""
    k = len(A)
    if k >= sp.N:
        return np.nan
    tr, _, _ = set_objectives(sp, np.array([A]))
    return float((tr[0] - k) / (sp.N - k))


def redundancy(R, A):
    """(mean |r|, max |r|) over the pairs of A (NaN for k = 1) and the
    normalized participation ratio k / ||R_AA||_F^2 (1 = orthogonal)."""
    A = list(A)
    k = len(A)
    RA = R[np.ix_(A, A)]
    fro2 = float(np.sum(RA ** 2))
    eff = k / fro2 if fro2 > 0 else np.nan
    if k < 2:
        return np.nan, np.nan, eff
    off = np.abs(RA[np.triu_indices(k, 1)])
    return float(off.mean()), float(off.max()), eff


def expected_r2(O_test, selected, R_fit, truths, ridge=RIDGE, target=None):
    """Expected pooled R^2 of impute_and_evaluate's predictor on the
    validation models' observation pattern.

    For each validation model i (grouped by observation pattern):
    A_i = selected benchmarks it observed, U_i = its observed unselected
    targets (target: the split's D1 mask, None: every column); skipped if
    either is empty, exactly as impute_and_evaluate (whose D2 clip has no
    closed form here: the expectation is that of the unclipped predictor).
    The predictor is W_i = R_fit[U_i, A_i] (R_fit[A_i, A_i] + ridge I)^{-1};
    if the standardized scores had correlation S (unit diagonal),
        E SS_res_i = tr S_UU - 2 tr(W S_AU) + tr(W S_AA W^T),
        E SS_tot_i = tr S_UU = |U_i|.
    Returns ({name: 1 - sum SS_res / sum SS_tot for S = truths[name]},
    fraction of models scored, fraction of observed unselected cells
    scored).
    """
    sel = [int(j) for j in selected]
    in_A = np.zeros(R_fit.shape[0], dtype=bool)
    in_A[sel] = True
    cand = ~in_A if target is None else ~in_A & np.asarray(target, bool)
    Ob = np.asarray(O_test) > 0
    pats, counts = np.unique(Ob, axis=0, return_counts=True)
    ss_res = dict.fromkeys(truths, 0.0)
    ss_tot = dict.fromkeys(truths, 0.0)
    n_models = n_cells = total_cells = 0
    for pat, c in zip(pats, counts):
        obs_A = [j for j in sel if pat[j]]
        U = np.flatnonzero(pat & cand)
        total_cells += c * len(U)
        if not obs_A or len(U) == 0:
            continue
        n_models += c
        n_cells += c * len(U)
        Roo = R_fit[np.ix_(obs_A, obs_A)] + ridge * np.eye(len(obs_A))
        Reo = R_fit[np.ix_(U, obs_A)]
        W = np.linalg.solve(Roo, Reo.T).T
        for name, S in truths.items():
            t_uu = float(np.trace(S[np.ix_(U, U)]))
            S_AU = S[np.ix_(obs_A, U)]
            S_AA = S[np.ix_(obs_A, obs_A)]
            res = t_uu - 2.0 * np.sum(W * S_AU.T) + np.sum((W @ S_AA) * W)
            ss_res[name] += c * res
            ss_tot[name] += c * t_uu
    M_test = Ob.shape[0]
    r2 = {name: (1.0 - ss_res[name] / ss_tot[name]) if ss_tot[name] > 0
          else np.nan for name in truths}
    return (r2, n_models / M_test if M_test else np.nan,
            n_cells / total_cells if total_cells else np.nan)


def observed_picks(O_test, selected, target=None):
    """(k_eff, k_obs): mean number of selected benchmarks a validation
    model observed, over the models impute_and_evaluate scores (at least
    one selected benchmark and one unselected target observed; target:
    the split's D1 mask, None: every column; NaN if none) and over all
    validation models."""
    Ob = np.asarray(O_test) > 0
    in_A = np.zeros(Ob.shape[1], dtype=bool)
    in_A[[int(j) for j in selected]] = True
    cand = ~in_A if target is None else ~in_A & np.asarray(target, bool)
    n_sel = Ob[:, in_A].sum(axis=1)
    scored = (n_sel > 0) & (Ob[:, cand].sum(axis=1) > 0)
    k_eff = float(n_sel[scored].mean()) if scored.any() else np.nan
    return k_eff, float(n_sel.mean())


def suites_of(names):
    """Suite label per benchmark (prefix before '/'), None if the names
    carry no suite."""
    names = [str(n) for n in names]
    if not any("/" in n for n in names):
        return None
    return np.array([n.split("/", 1)[0] for n in names])


# ── Per-split evaluation ─────────────────────────────────────────

class SplitContext:
    """Everything one split's set evaluations need."""

    def __init__(self, sp_tr, sp_or, col_mean, col_std, B_test, O_test,
                 suites, target=None, bounds=None):
        self.sp_tr, self.sp_or = sp_tr, sp_or
        self.col_mean, self.col_std = col_mean, col_std
        self.B_test, self.O_test = B_test, O_test
        self.suites = suites
        self.N = sp_tr.N
        self.target, self.bounds = target, bounds     # rules D1, D2

    def realized(self, A, R):
        return impute_and_evaluate(self.B_test, self.O_test, list(A), R,
                                   self.col_mean, self.col_std, self.N,
                                   target=self.target,
                                   bounds=self.bounds)["r2"]

    def set_metrics(self, A):
        """All SET_QUANTITIES of the index set A (module docstring)."""
        A = [int(j) for j in A]
        k = len(A)
        tr, sp_or = self.sp_tr, self.sp_or
        out = {"r2": self.realized(A, tr.R),
               "r2_oimp": self.realized(A, sp_or.R)}
        t_tr, ld_tr, mi_tr = set_objectives(tr, np.array([A]))
        t_or, ld_or, mi_or = set_objectives(sp_or, np.array([A]))
        denom = self.N - k
        out["trace_tr"] = float(t_tr[0])
        out["pop_r2_tr"] = float((t_tr[0] - k) / denom) if denom else np.nan
        out["pop_r2_or"] = float((t_or[0] - k) / denom) if denom else np.nan
        out["logdet_tr"], out["logdet_or"] = float(ld_tr[0]), float(ld_or[0])
        out["mi_tr"], out["mi_or"] = float(mi_tr[0]), float(mi_or[0])
        out["hub_tr"] = float(np.sum(tr.hub[A]))
        exp, cov_m, cov_c = expected_r2(self.O_test, A, tr.R,
                                        {"tr": tr.R, "or": sp_or.R},
                                        target=self.target)
        out["exp_r2_tr"], out["exp_r2_or"] = exp["tr"], exp["or"]
        out["cov_models"], out["cov_cells"] = cov_m, cov_c
        out["k_eff"], out["k_obs"] = observed_picks(self.O_test, A,
                                                    self.target)
        m_tr, _, e_tr = redundancy(tr.R, A)
        m_or, x_or, e_or = redundancy(sp_or.R, A)
        out.update(mean_abs_r_tr=m_tr, mean_abs_r_or=m_or, max_abs_r_or=x_or,
                   eff_rank_tr=e_tr, eff_rank_or=e_or)
        out["n_suites"] = (float(len(set(self.suites[A])))
                           if self.suites is not None else np.nan)
        out["facloc_tr"] = float(facloc_objective(tr, [A])[0])
        return out


def singleton_metrics(ctx, el, n_obs_train, share=SPECTRAL_SHARE):
    """k = 1 statistics of every eligible benchmark ({q: (N,) array},
    NaN where ineligible)."""
    tr, sp_or = ctx.sp_tr, ctx.sp_or
    N = ctx.N
    out = {q: np.full(N, np.nan) for q in SINGLE_QUANTITIES}
    q_n = max(1, int(math.ceil(share * N)))
    lam_c = np.maximum(tr.lam, 1e-300)
    P_terms = tr.V ** 2 / lam_c[None, :]         # (N, N): j, eigen i
    R2_terms = tr.V ** 2 * (tr.lam ** 2)[None, :]
    p_low = P_terms[:, :q_n].sum(axis=1) / P_terms.sum(axis=1)
    r2_top = R2_terms[:, -q_n:].sum(axis=1) / R2_terms.sum(axis=1)
    for j in np.flatnonzero(el):
        out["r2"][j] = ctx.realized([j], tr.R)
        exp, cov_m, _ = expected_r2(ctx.O_test, [j], tr.R,
                                    {"tr": tr.R, "or": sp_or.R},
                                    target=ctx.target)
        out["exp_r2_tr"][j], out["exp_r2_or"][j] = exp["tr"], exp["or"]
        out["cov_models"][j] = cov_m
    pj = np.diag(tr.P)
    pj_or = np.diag(sp_or.P)
    out["pop_r2_tr"][el] = ((np.diag(tr.R2) - 1.0) / (N - 1))[el]
    out["pop_r2_or"][el] = ((np.diag(sp_or.R2) - 1.0) / (N - 1))[el]
    out["rest_r2_tr"][el] = (1.0 - 1.0 / pj)[el]
    out["rest_r2_or"][el] = (1.0 - 1.0 / pj_or)[el]
    out["mi_tr"][el] = (0.5 * np.log(pj))[el]
    out["mi_or"][el] = (0.5 * np.log(pj_or))[el]
    out["hub_tr"][el] = tr.hub[el]
    out["p_low_share_tr"][el] = p_low[el]
    out["r2_top_share_tr"][el] = r2_top[el]
    out["n_obs_train"][el] = n_obs_train[el]
    out["facloc_tr"][el] = tr.fl_single[el]
    return out


def brute_force(sp, pool, k, chunk=40000):
    """Exact maximizers of F_tr, F_H, F_MI and F_FL over all k-subsets of
    `pool` under sp: {obj: (value, set)}."""
    best = {obj: (-np.inf, None) for obj in BRUTE_OBJECTIVES}
    combos = itertools.combinations(np.asarray(pool, dtype=np.int64), k)
    while True:
        block = np.fromiter(itertools.chain.from_iterable(
            itertools.islice(combos, chunk)), dtype=np.int64)
        if block.size == 0:
            break
        sets = block.reshape(-1, k)
        values = _brute_values(sp, sets)
        for obj in BRUTE_OBJECTIVES:
            vals = values[obj]
            vals = np.where(np.isfinite(vals), vals, -np.inf)
            i = int(np.argmax(vals))
            if vals[i] > best[obj][0]:
                best[obj] = (float(vals[i]), sets[i].tolist())
    return best


def trace_path_gains(R, selected, eligible):
    """Trace gains Delta_t(j) = sum_i S_ij^2 / S_jj (S the conditional
    covariance after t picks of `selected`) of every eligible unselected j,
    t = 0..len(selected)-1: (len(selected), N), NaN where not a
    candidate."""
    N = R.shape[0]
    S = np.array(R, dtype=np.float64)
    unsel = np.ones(N, dtype=bool)
    G = np.full((len(selected), N), np.nan)
    for t, j_star in enumerate(selected):
        cand = np.flatnonzero(unsel)
        pool = cand[eligible[cand]]
        d = np.diag(S)[pool]
        col = np.sum(S[np.ix_(cand, pool)] ** 2, axis=0)
        G[t, pool] = np.where(d > 1e-12, col / np.maximum(d, 1e-12), np.nan)
        s_col = S[:, j_star].copy()
        S -= np.outer(s_col, s_col) / max(S[j_star, j_star], 1e-300)
        S[j_star, :] = 0.0
        S[:, j_star] = 0.0
        unsel[j_star] = False
    return G


def _mi_gain(sig2, p_c):
    """1/2 [log sigma^2_{j|A} + log [(R_CC)^{-1}]_jj], -inf where either
    factor is not positive (greedy_select.greedy_mi's scores)."""
    sig2 = np.asarray(sig2, dtype=np.float64)
    p_c = np.asarray(p_c, dtype=np.float64)
    ok = (sig2 > 1e-300) & (p_c > 1e-300)
    out = np.full(sig2.shape, -np.inf)
    out[ok] = 0.5 * (np.log(sig2[ok]) + np.log(p_c[ok]))
    return out


def mi_gain_landscape(R, selected, eligible):
    """MI gains Delta_MI(j | A_t) = 1/2 log sigma^2_{j|A_t} + 1/2 log
    [(R_CC)^{-1}]_jj (C = V \\ A_t, eligible or not) of every eligible
    j not in A_t, for the prefixes A_t = selected[:t] of a greedy MI path,
    t = 0..len(selected)-1: (len(selected), N), NaN where not a candidate.
    Row t's maximum is greedy MI's gain of pick t + 1 (same factorization
    as greedy_mi); its minimum shows whether some candidate would make MI
    decrease."""
    R = np.asarray(R, dtype=np.float64)
    N = R.shape[0]
    selected = [int(j) for j in selected]
    el = np.ones(N, dtype=bool) if eligible is None else np.asarray(eligible)
    G = np.full((len(selected), N), np.nan)
    diag = np.diag(R)
    for t in range(len(selected)):
        A = selected[:t]
        in_A = np.zeros(N, dtype=bool)
        in_A[A] = True
        comp = np.flatnonzero(~in_A)
        p_c = np.zeros(N)
        p_c[comp] = _complement_precision_diag(R, comp)
        if A:
            X = np.linalg.solve(R[np.ix_(A, A)], R[A, :])
            sig2 = diag - np.sum(R[A, :] * X, axis=0)
        else:
            sig2 = diag.copy()
        cand = comp[el[comp]]
        G[t, cand] = _mi_gain(sig2[cand], p_c[cand])
    return G


def mi_pair_gains(R, eligible):
    """MI gains of the pairs, Delta_MI(j | {a}) = 1/2 log(R_jj - R_aj^2 /
    R_aa) + 1/2 log [(R_{-a,-a})^{-1}]_jj (= 1/2 log(1 - r_aj^2) + ... on
    a correlation matrix), for eligible a != j: (N, N), row a, column j,
    NaN elsewhere."""
    R = np.asarray(R, dtype=np.float64)
    N = R.shape[0]
    el = np.ones(N, dtype=bool) if eligible is None else np.asarray(eligible)
    G = np.full((N, N), np.nan)
    diag = np.diag(R)
    for a in np.flatnonzero(el):
        comp = np.delete(np.arange(N), a)
        p_c = _complement_precision_diag(R, comp)
        sig2 = diag[comp] - R[a, comp] ** 2 / R[a, a]
        ok = el[comp]
        G[a, comp[ok]] = _mi_gain(sig2[ok], p_c[ok])
    return G


def submodularity_ratio(sp, U, pool, k, tol=1e-9, chunk=40000):
    """Das and Kempe's submodularity ratio of F_tr under sp,

      gamma_{U,k} = min over L subset of U, S subset of pool \\ L with
                    |S| <= k of  sum_{x in S} [F(L + x) - F(L)]
                                 / [F(L u S) - F(L)],

    exactly (all L and S; |S| = 1 gives 1, so gamma <= 1).  (L, S) with
    F(L u S) - F(L) <= tol N are skipped (0/0).  The greedy guarantee is
    F_tr(greedy_k) >= (1 - e^{-gamma_{U,k}}) F_tr(OPT_k) for U the greedy
    set of size k."""
    U = [int(u) for u in U]
    pool = np.asarray(pool, dtype=np.int64)
    gamma = 1.0
    floor = tol * sp.N
    for r in range(len(U) + 1):
        for L in itertools.combinations(U, r):
            L = np.asarray(L, dtype=np.int64)
            cand = pool[~np.isin(pool, L)]
            if len(cand) < 2:
                continue
            f_L = float(trace_objective(sp, L[None, :])[0]) if r else 0.0
            L_rows = np.broadcast_to(L, (len(cand), r))
            single = trace_objective(
                sp, np.concatenate([L_rows, cand[:, None]], axis=1)) - f_L
            for s in range(2, min(k, len(cand)) + 1):
                combos = itertools.combinations(range(len(cand)), s)
                while True:
                    block = np.fromiter(itertools.chain.from_iterable(
                        itertools.islice(combos, chunk)), dtype=np.int64)
                    if block.size == 0:
                        break
                    idx = block.reshape(-1, s)
                    sets = np.concatenate(
                        [np.broadcast_to(L, (len(idx), r)), cand[idx]], axis=1)
                    den = trace_objective(sp, sets) - f_L
                    num = single[idx].sum(axis=1)
                    ok = den > floor
                    if ok.any():
                        gamma = min(gamma, float(np.min(num[ok] / den[ok])))
    return gamma


def _finite_min(a):
    """Minimum over the non-NaN entries (may be -inf), NaN if none."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(a.min()) if a.size else np.nan


THEORY_KEYS = ["trace_increase_frac", "trace_max_ratio",
               "trace_max_step_ratio", "trace_gamma", "trace_gamma_bound",
               "trace_spectral_bound", "trace_sparse_eig_ub",
               "trace_sparse_bound_ub", "mi_chosen_neg_gains", "mi_min_gain",
               "mi_neg_gain_frac", "mi_first_neg_size", "mi_min_pair_gain",
               "mi_nonmonotone", "mi_eps_lb", "mi_keps_rel", "mi_peak_k",
               "entropy_below_floor", "entropy_first_below_k", "lam_min",
               "lam_max", "n_lam_below_pointzeroone"]
# theory keys whose min / max / median over splits also become macros
THEORY_RANGE_KEYS = ["trace_gamma", "trace_gamma_bound",
                     "trace_spectral_bound", "trace_sparse_eig_ub",
                     "trace_sparse_bound_ub", "mi_min_gain",
                     "mi_min_pair_gain", "mi_eps_lb", "mi_keps_rel",
                     "lam_min"]


def theory_diagnostics(R, results, eligible, rtol=1e-9, sp=None,
                       gamma_k=GAMMA_K, eps_k=EPS_K, mi_gains=None):
    """Per-split diagnostics of the theoretical properties (module
    docstring, keys THEORY_KEYS): trace's diminishing-returns violations
    along its greedy path and its exact submodularity ratio gamma_{U,
    gamma_k} with the spectral bounds; MI's negative gains, of its own
    picks (mi_chosen_neg_gains) and of every eligible candidate along its
    path (mi_gains = mi_gain_landscape(...), computed if None) and of every
    eligible pair, and the implied lower bound on Krause et al.'s epsilon
    for budget eps_k (path sets of size <= 2 eps_k); entropy's steps with
    conditional variance below 1/(2 pi e)."""
    out = {}
    el = np.asarray(eligible, dtype=bool)
    pool = np.flatnonzero(el)
    if sp is None:
        sp = Spectral(R)
    sel = list(results["trace"]["selected"])
    G = trace_path_gains(R, sel, eligible)
    if len(sel) > 1:
        prev, cur = G[:-1], G[1:]
        ok = np.isfinite(prev) & np.isfinite(cur)
        inc = cur[ok] > prev[ok] * (1.0 + rtol)
        out["trace_increase_frac"] = float(inc.mean()) if ok.any() else np.nan
        base = np.broadcast_to(G[0], cur.shape)
        ok0 = np.isfinite(base) & np.isfinite(cur)
        out["trace_max_ratio"] = (float(np.max(cur[ok0] / base[ok0]))
                                  if ok0.any() else np.nan)
        out["trace_max_step_ratio"] = (float(np.max(cur[ok] / prev[ok]))
                                       if ok.any() else np.nan)
    else:
        out.update(trace_increase_frac=np.nan, trace_max_ratio=np.nan,
                   trace_max_step_ratio=np.nan)
    # Submodularity ratio of F_tr at U = greedy trace's first gamma_k picks
    lam = np.linalg.eigvalsh(R)
    if gamma_k >= 2 and len(sel) >= gamma_k and len(pool) > gamma_k:
        gam = submodularity_ratio(sp, sel[:gamma_k], pool, gamma_k)
        out["trace_gamma"] = gam
        out["trace_gamma_bound"] = 1.0 - math.exp(-gam)
    else:
        out["trace_gamma"] = out["trace_gamma_bound"] = np.nan
    out["trace_spectral_bound"] = 1.0 - math.exp(-max(float(lam[0]), 0.0))
    if len(pool) >= 2:
        Rp = np.abs(R[np.ix_(pool, pool)])
        np.fill_diagonal(Rp, 0.0)
        ub = 1.0 - float(Rp.max())
        out["trace_sparse_eig_ub"] = ub
        out["trace_sparse_bound_ub"] = 1.0 - math.exp(-max(ub, 0.0))
    else:
        out["trace_sparse_eig_ub"] = out["trace_sparse_bound_ub"] = np.nan

    # MI: chosen gains, every candidate along the path, every pair
    gains = np.asarray(results["mi"]["gains"], dtype=float)
    out["mi_chosen_neg_gains"] = float(np.sum(gains[np.isfinite(gains)] < 0))
    if mi_gains is None:
        mi_gains = mi_gain_landscape(R, results["mi"]["selected"], el)
    cand = mi_gains[~np.isnan(mi_gains)]
    out["mi_min_gain"] = _finite_min(mi_gains)
    out["mi_neg_gain_frac"] = float(np.mean(cand < 0)) if cand.size else np.nan
    neg_rows = np.flatnonzero(np.any(mi_gains < 0, axis=1))
    out["mi_first_neg_size"] = float(neg_rows[0]) if neg_rows.size else np.nan
    out["mi_min_pair_gain"] = _finite_min(mi_pair_gains(R, el))
    lows = [v for v in (out["mi_min_gain"], out["mi_min_pair_gain"])
            if not np.isnan(v)]
    out["mi_nonmonotone"] = float(min(lows) < 0) if lows else np.nan
    low_k = [v for v in (_finite_min(mi_gains[:2 * eps_k + 1]),
                         out["mi_min_pair_gain"]) if not np.isnan(v)]
    out["mi_eps_lb"] = max(0.0, -min(low_k)) if low_k else np.nan
    mv = np.asarray(results["mi"]["mi_values"], dtype=float)
    out["mi_keps_rel"] = (eps_k * out["mi_eps_lb"] / mv[eps_k - 1]
                          if len(mv) >= eps_k and np.isfinite(mv[eps_k - 1])
                          and mv[eps_k - 1] > 0 else np.nan)
    out["mi_peak_k"] = (float(np.nanargmax(mv) + 1) if np.isfinite(mv).any()
                        else np.nan)
    ld = np.asarray(results["entropy"]["log_dets"], dtype=float)
    steps = np.diff(np.concatenate([[0.0], ld]))
    steps = steps[np.isfinite(steps)]
    out["entropy_below_floor"] = float(np.sum(steps < math.log(ENTROPY_FLOOR)))
    out["entropy_first_below_k"] = (
        float(np.argmax(steps < math.log(ENTROPY_FLOOR)) + 1)
        if np.any(steps < math.log(ENTROPY_FLOOR)) else np.nan)
    out["lam_min"] = float(lam[0])
    out["lam_max"] = float(lam[-1])
    out["n_lam_below_pointzeroone"] = float(np.sum(lam < 0.01))
    return out


PATH_KEYS = ["novelty_mi", "bonus_mi", "gain_mi", "min_gain_mi",
             "novelty_entropy", "novelty_trace", "novelty_facloc"]


def path_terms(results, k_max, mi_gains=None):
    """Per-step terms of the greedy paths, (k_max,) arrays (NaN beyond the
    last pick): novelty_{m} = 1/2 log sigma^2_{j|A} of each pick of m (the
    entropy gain; m in mi, entropy, trace and, if results has it, facloc:
    from the log dets of its prefixes), MI's gain 1/2 log
    sigma^2_{j|A} + 1/2 log [(R_CC)^{-1}]_jj split into that novelty and
    the bonus 1/2 log [(R_CC)^{-1}]_jj = -1/2 log sigma^2_{j|C \\ j} for
    being predictable from the unselected rest (C the complement before
    the pick; the bonus is 1/2 log P_jj only at the first pick), and
    min_gain_mi, the smallest gain of any eligible candidate at that step
    (rows of mi_gains = mi_gain_landscape(...); NaN if not given)."""
    out = {}
    for m in ("mi", "entropy", "trace", "facloc"):
        if m not in results:
            continue
        ld = np.asarray(results[m]["log_dets"], dtype=float)[:k_max]
        out[f"novelty_{m}"] = 0.5 * np.diff(np.concatenate([[0.0], ld]))
    gains = np.asarray(results["mi"]["gains"], dtype=float)[:k_max]
    out["gain_mi"] = gains
    out["bonus_mi"] = gains - out["novelty_mi"]
    low = np.full(len(gains), np.nan)
    if mi_gains is not None:
        for t, row in enumerate(np.asarray(mi_gains)[:len(gains)]):
            low[t] = _finite_min(row)
    out["min_gain_mi"] = low
    return out


def _spearman(x, y):
    from scipy import stats
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok])[0])


def _jaccard(a, b):
    if a is None or b is None:
        return np.nan
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def _padded(indices, length):
    out = np.full(length, -1, dtype=np.int64)
    out[:len(indices)] = indices
    return out


# ── Facility location vs trace ───────────────────────────────────

def facloc_path_gains(S, selected, eligible):
    """Facility-location gains sum_i max(s_ij - c_i, 0) (c the cover
    max_{a in A_t} s_.a, zero for A_0) of every eligible j not in A_t, for
    the prefixes A_t = selected[:t] of a path, t = 0..len(selected)-1:
    (len(selected), N), NaN where not a candidate.  S is the similarity
    (Spectral.fl); the arithmetic is facility_location_select's, so on
    its own path row t's value at pick t + 1 is its gains[t]."""
    S = np.asarray(S, dtype=np.float64)
    N = S.shape[0]
    el = (np.ones(N, dtype=bool) if eligible is None
          else np.asarray(eligible, dtype=bool))
    cover = np.zeros(N, dtype=np.float64)
    unsel = np.ones(N, dtype=bool)
    G = np.full((len(selected), N), np.nan)
    for t, j_star in enumerate(selected):
        pool = np.flatnonzero(unsel & el)
        G[t, pool] = np.sum(np.maximum(S[:, pool] - cover[:, None], 0.0),
                            axis=0)
        cover = np.maximum(cover, S[:, int(j_star)])
        unsel[int(j_star)] = False
    return G


def _greedy_choice(objective, R, prefix, cand, gains):
    """Position in cand (ascending) of the candidate that the greedy step
    of `objective` picks at the state A = prefix, given its gains over
    cand (NaN: not a candidate of the objective), with that selector's
    near-tie rule (baseline_select): facloc's gains within TIE_RTOL ·
    max(|best|, 1) of the best are broken by the explained variance of the
    unselected benchmarks in R given A, then by the lowest index
    (greedy_select._tie_break, as facility_location_select); trace's
    within TIE_RTOL · max(|best|, max_i R_ii) go to the lowest index (as
    greedy_trace_select).  None if no gain is finite.  On the objective's
    own path (gains of facloc_path_gains / trace_path_gains) this is the
    pick the selector made."""
    gains = np.asarray(gains, dtype=np.float64)
    ok = np.isfinite(gains)
    if not ok.any():
        return None
    v = np.where(ok, gains, -np.inf)
    if objective != "facloc":
        return int(np.flatnonzero(_near_max(v, float(np.max(np.diag(R)))))[0])
    tied = np.flatnonzero(_near_max(v, 1.0))
    if len(tied) == 1:
        return int(tied[0])
    prefix = [int(j) for j in prefix]
    L, d = _cholesky_state(R, prefix)
    unsel = np.setdiff1d(np.arange(R.shape[0]), prefix)
    j_star, _, _ = _tie_break(R, L, d, len(prefix), cand[tied], unsel)
    return int(np.searchsorted(cand, j_star))


def _pad_nan(values, length):
    out = np.full(length, np.nan)
    v = np.asarray(values, dtype=float)[:length]
    out[:len(v)] = v
    return out


def facloc_trace_split(ctx, R, el, res, r2, single_r2, k_max, land_k):
    """Per-split arrays of the facility-location section (module
    docstring) as {key without the fl_ prefix: array}: overlap of the
    facloc and trace selections (shared_trace, jacc_trace: (k_max,)),
    per greedy path P in FL_PATHS the realized R^2 gains (path_{P}_dr2),
    the gains of each objective O in FL_GAINS for P's picks
    (path_{P}_gain_{O}) and their Spearman over the steps
    (path_{P}_rho_{O}, scalar), and the gain landscape before picks
    1..land_k (land_{P}_rho_{O}, land_{P}_regret_{O}, land_{P}_best_dr2,
    land_{P}_n_cand: (land_k,)).

    res: {m: train-side selector result} with facloc and trace; r2: {m:
    (k_max,) realized R^2 of m's size-k sets (the {m}_r2 row)};
    single_r2: (N,) realized R^2 of the singletons (single_r2 row), the
    realized gains at t = 0 (r2 of the empty set = 0)."""
    out = {}
    shared = np.full(k_max, np.nan)
    jacc = np.full(k_max, np.nan)
    for k in range(1, k_max + 1):
        a = selection_at_k(res["facloc"], k)
        b = selection_at_k(res["trace"], k)
        if a is None or b is None:
            continue
        shared[k - 1] = len(set(a) & set(b))
        jacc[k - 1] = _jaccard(a, b)
    out["shared_trace"], out["jacc_trace"] = shared, jacc

    S = ctx.sp_tr.fl
    for P in FL_PATHS:
        path = [int(j) for j in res[P]["selected"][:k_max]]
        n = len(path)
        G = {"facloc": facloc_path_gains(S, path, el),
             "trace": trace_path_gains(R, path, el)}
        r2_P = np.asarray(r2[P], dtype=float)[:n]
        dr2 = np.diff(np.concatenate([[0.0], r2_P]))
        out[f"path_{P}_dr2"] = _pad_nan(dr2, k_max)
        for O in FL_GAINS:
            g = np.array([G[O][t, j] for t, j in enumerate(path)])
            out[f"path_{P}_gain_{O}"] = _pad_nan(g, k_max)
            out[f"path_{P}_rho_{O}"] = _spearman(g, dr2)

        land = {f"{key}_{O}": np.full(land_k, np.nan)
                for key in ("rho", "regret") for O in FL_GAINS}
        land["best_dr2"] = np.full(land_k, np.nan)
        land["n_cand"] = np.full(land_k, np.nan)
        for t in range(min(land_k, n)):
            cand = np.flatnonzero(np.isfinite(G["facloc"][t]))
            if t == 0:
                d = np.asarray(single_r2, dtype=float)[cand]
            else:
                d = np.array([ctx.realized(path[:t] + [int(j)], R)
                              for j in cand]) - r2_P[t - 1]
            land["n_cand"][t] = len(cand)
            if not np.isfinite(d).any():
                continue
            best = float(np.nanmax(d))
            land["best_dr2"][t] = best
            for O in FL_GAINS:
                g = G[O][t, cand]
                land[f"rho_{O}"][t] = _spearman(g, d)
                # O's pick at A_t: P's own pick for O = P, else O's
                # greedy step from A_t (its selector's near-tie rule)
                top = (int(np.searchsorted(cand, path[t])) if O == P
                       else _greedy_choice(O, R, path[:t], cand, g))
                if top is not None:
                    land[f"regret_{O}"][t] = best - d[top]
        for key, v in land.items():
            out[f"land_{P}_{key}"] = v
    return out


# ── Dataset driver ───────────────────────────────────────────────

def estimate_corr_oracle(B, O, M_ref=None):
    """Oracle correlation R_or from ALL models (B, O).

    M_ref None: eval_entropy_vs_mi.estimate_corr(B, O) itself.  M_ref an
    int (the training size of the splits; the default "matched" oracle):
    the same estimator with every size-dependent choice of estimate_corr /
    estimate_corr_em made as for M_ref training models, so that R_or
    differs from the split estimates R_tr only by the data.  The choices
    are EM vs pairwise (EM if fewer than 90% of the entries are observed
    or M_ref < N), the EM regularization (eps 1e-3 / 1000 iterations if
    M_ref < N or fewer than 50% observed, else 1e-6 / 500) and the
    Ledoit-Wolf shrinkage of em_cov's "auto" rule, alpha = (N - M_ref) / N
    for M_ref < N, else 0.  Without matching, merged (M = 118 > N = 114 >
    M_train = 106) would get an unshrunk oracle next to training
    estimates shrunk by alpha = 0.07.
    """
    from eval_entropy_vs_mi import estimate_corr_pairwise
    from em_cov import em_cov
    if M_ref is None:
        return estimate_corr(B, O)[0]
    M, N = B.shape
    obs_frac = O.sum() / (M * N)
    if not (obs_frac < 0.9 or M_ref < N):
        return estimate_corr_pairwise(B, O)[0]
    # estimate_corr_em with the size rules evaluated at M_ref
    col_count = np.maximum(O.sum(axis=0), 1.0)
    col_mean = (B * O).sum(axis=0) / col_count
    B_c = O * (B - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std_floor = 0.01 * (np.abs(col_mean) + 1.0)
    col_std = np.sqrt(np.maximum(col_var, col_std_floor ** 2))
    B_std = O * (B - col_mean[None, :]) / col_std[None, :]
    regularized = M_ref < N or obs_frac < 0.5
    shrink = (N - M_ref) / N if M_ref < N else 0.0
    result = em_cov(B_std, O, max_iter=1000 if regularized else 500,
                    tol=5e-4, shrinkage=shrink,
                    eps_psd=1e-3 if regularized else 1e-6, verbose=False)
    Sigma = result["Sigma"]
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    return Sigma * np.outer(d_inv, d_inv)


def split_arrays(S, N, k_max, n_random_sets, nk_r, brute_k, land_k,
                 methods=None):
    """The per-split arrays of run_dataset (module docstring), (S, ...)
    each, in the npz's key order: NaN-filled, -1 for the selections, False
    / 0 for the eligibility fields.  methods: default METHODS."""
    methods = METHODS if methods is None else methods
    with_fl = "facloc" in methods and "trace" in methods
    arr = {"n_eligible": np.zeros(S, dtype=np.int64),
           "eligible": np.zeros((S, N), dtype=bool),
           "estimation_failed": np.zeros(S, dtype=bool)}
    for src in ("", "osel_"):
        for m in methods:
            for q in SET_QUANTITIES:
                arr[f"{m}_{src}{q}"] = np.full((S, k_max), np.nan)
            arr[f"{m}_{src}selected"] = np.full((S, k_max), -1, np.int64)
        arr[f"pca_{src}selected_per_k"] = np.full((S, k_max, k_max), -1,
                                                  np.int64)
    for m in methods:
        arr[f"{m}_jacc_oracle"] = np.full((S, k_max), np.nan)
    for q in SINGLE_QUANTITIES:
        arr[f"single_{q}"] = np.full((S, N), np.nan)
    for q in SET_QUANTITIES:
        arr[f"rand_{q}"] = np.full((S, n_random_sets, nk_r), np.nan)
    for obj in BRUTE_OBJECTIVES:
        if GREEDY_OF[obj] not in methods:
            continue
        for field in ("value", "greedy", "r2", "r2_greedy", "same"):
            arr[f"brute_{obj}_{field}"] = np.full((S, brute_k), np.nan)
    for key in THEORY_KEYS:
        arr[f"theory_{key}"] = np.full(S, np.nan)
    for key in PATH_KEYS:
        if key == "novelty_facloc" and "facloc" not in methods:
            continue
        arr[f"path_{key}"] = np.full((S, k_max), np.nan)
    for obj in STAB_OBJECTIVES:
        arr[f"stab_{obj}"] = np.full(S, np.nan)
        arr[f"stab_{obj}_same_pick"] = np.full(S, np.nan)
    if with_fl:
        for key in ("shared_trace", "jacc_trace"):
            arr[f"fl_{key}"] = np.full((S, k_max), np.nan)
        for P in FL_PATHS:
            arr[f"fl_path_{P}_dr2"] = np.full((S, k_max), np.nan)
            for key in ("best_dr2", "n_cand"):
                arr[f"fl_land_{P}_{key}"] = np.full((S, land_k), np.nan)
            for gain in FL_GAINS:       # (O is the observation mask)
                arr[f"fl_path_{P}_gain_{gain}"] = np.full((S, k_max), np.nan)
                arr[f"fl_path_{P}_rho_{gain}"] = np.full(S, np.nan)
                for key in ("rho", "regret"):
                    arr[f"fl_land_{P}_{key}_{gain}"] = np.full((S, land_k),
                                                               np.nan)
    return arr


def _say(text, end="\n"):
    print(text, end=end, flush=True)


def objectives_split(dctx, s, say=_say):
    """Every per-split quantity of the matched split s: {key: row} over
    the keys of split_arrays (the rows stay NaN / -1 if the estimation
    fails).  dctx holds the dataset and the run options (run_dataset); the
    progress line goes to say(text, end), printed at once by default.
    The split's random sets and random baseline draw from its own common
    random numbers (Split.seed), so the rows depend on dctx and s alone."""
    B, O, N = dctx["B"], dctx["O"], dctx["N"]
    names, suites, bounds = dctx["names"], dctx["suites"], dctx["bounds"]
    R_or, sp_or = dctx["R_or"], dctx["sp_or"]
    methods, k_max, land_k = dctx["methods"], dctx["k_max"], dctx["land_k"]
    n_random_sets, random_ks = dctx["n_random_sets"], dctx["random_ks"]
    brute_k, gamma_k = dctx["brute_k"], dctx["gamma_k"]
    with_fl = "facloc" in methods and "trace" in methods
    arr = split_arrays(1, N, k_max, n_random_sets, len(random_ks), brute_k,
                       land_k, methods)
    i = 0

    t0 = time.time()
    B_train, O_train = B[s.train_idx], O[s.train_idx]
    el = eligible_mask(O_train)
    arr["eligible"][i] = el
    arr["n_eligible"][i] = int(el.sum())
    say(f"  Split {s.split_id} (rep {s.rep}, fold {s.fold}): "
        f"{len(s.train_idx)} train / {len(s.val_idx)} test, "
        f"{int(el.sum())}/{N} eligible", end="")
    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:                   # as eval_baselines
        say(f" — estimation failed: {e}")
        arr["estimation_failed"][i] = True
        return {key: v[0] for key, v in arr.items()}
    sp_tr = Spectral(R)
    ctx = SplitContext(sp_tr, sp_or, col_mean, col_std, B[s.val_idx],
                       O[s.val_idx], suites, target_mask(O_train),
                       bounds)

    results = {}
    for src, Sig in (("", R), ("osel_", R_or)):
        for m in methods:
            res = run_selection(m, Sig, k_max, names, s, el)
            results[(src, m)] = res
            arr[f"{m}_{src}selected"][i] = _padded(res["selected"],
                                                   k_max)
            if m == "pca":
                for kk, sel_kk in enumerate(res["selected_per_k"]):
                    arr[f"pca_{src}selected_per_k"][i, kk] = _padded(
                        sel_kk, k_max)
            for k in range(1, k_max + 1):
                A = selection_at_k(res, k)
                if A is None:
                    continue
                for q, v in ctx.set_metrics(A).items():
                    arr[f"{m}_{src}{q}"][i, k - 1] = v
    for m in methods:
        for k in range(1, k_max + 1):
            arr[f"{m}_jacc_oracle"][i, k - 1] = _jaccard(
                selection_at_k(results[("", m)], k),
                selection_at_k(results[("osel_", m)], k))

    # k = 1: every eligible singleton
    single = singleton_metrics(ctx, el, O_train.sum(axis=0))
    for q, v in single.items():
        arr[f"single_{q}"][i] = v

    # Random eligible k-sets (common random numbers SEED + 19)
    rng = np.random.default_rng(s.seed(SEED + RANDOM_SETS_SEED_OFFSET))
    pool = np.flatnonzero(el)
    for ik, k in enumerate(random_ks):
        if k > len(pool):
            continue
        for r in range(n_random_sets):
            A = rng.permutation(pool)[:k].tolist()
            for q, v in ctx.set_metrics(A).items():
                arr[f"rand_{q}"][i, r, ik] = v

    # Exact maximizers at small k (k = 1 from the singletons)
    brute_objs = [obj for obj in BRUTE_OBJECTIVES
                  if GREEDY_OF[obj] in methods]
    greedy_res = {obj: results[("", GREEDY_OF[obj])]
                  for obj in brute_objs}
    for k in range(1, brute_k + 1):
        if k > len(pool):
            break
        best = brute_force(sp_tr, pool, k)
        for obj in brute_objs:
            if obj == "entropy" and k == 1:
                continue        # log det R_jj = 0: every singleton ties
            val, A_opt = best[obj]
            G = selection_at_k(greedy_res[obj], k)
            g_val = float(_brute_values(sp_tr, np.array([G]))[obj][0])
            arr[f"brute_{obj}_value"][i, k - 1] = val
            arr[f"brute_{obj}_greedy"][i, k - 1] = g_val
            arr[f"brute_{obj}_same"][i, k - 1] = float(
                set(A_opt) == set(G))
            arr[f"brute_{obj}_r2"][i, k - 1] = ctx.realized(A_opt, R)
            arr[f"brute_{obj}_r2_greedy"][i, k - 1] = arr[
                f"{GREEDY_OF[obj]}_r2"][i, k - 1]

    train_res = {m: results[("", m)] for m in methods}
    mi_gains = mi_gain_landscape(R, train_res["mi"]["selected"], el)
    th = theory_diagnostics(R, train_res, el, sp=sp_tr, gamma_k=gamma_k,
                            mi_gains=mi_gains)
    for key in THEORY_KEYS:
        arr[f"theory_{key}"][i] = th[key]
    for key, v in path_terms(train_res, k_max, mi_gains).items():
        arr[f"path_{key}"][i] = v

    # Stability of the k = 1 scores, train vs oracle correlation
    pairs = {"trace": (np.diag(sp_tr.R2), np.diag(sp_or.R2)),
             "mi": (np.diag(sp_tr.P), np.diag(sp_or.P)),
             "hub": (sp_tr.hub, sp_or.hub),
             "facloc": (sp_tr.fl_single, sp_or.fl_single)}
    for obj in STAB_OBJECTIVES:
        a, b = pairs[obj]
        arr[f"stab_{obj}"][i] = _spearman(a[el], b[el])
        arr[f"stab_{obj}_same_pick"][i] = float(
            pool[np.argmax(a[el])] == pool[np.argmax(b[el])])

    # Facility location vs trace (train side)
    if with_fl:
        fl = facloc_trace_split(
            ctx, R, el, train_res,
            {m: arr[f"{m}_r2"][i] for m in FL_PATHS},
            arr["single_r2"][i], k_max, land_k)
        for key, v in fl.items():
            arr[f"fl_{key}"][i] = v

    fl5 = (f"facloc {arr['facloc_r2'][i, min(5, k_max) - 1]:.3f} "
           if "facloc" in methods else "")
    say(f"  trace {arr['trace_r2'][i, min(5, k_max) - 1]:.3f} "
        f"mi {arr['mi_r2'][i, min(5, k_max) - 1]:.3f} {fl5}@5, "
        f"oracle-sel "
        f"trace {arr['trace_osel_r2'][i, min(5, k_max) - 1]:.3f} "
        f"mi {arr['mi_osel_r2'][i, min(5, k_max) - 1]:.3f}  "
        f"({time.time() - t0:.1f}s)")
    return {key: v[0] for key, v in arr.items()}


# Worker state of the split processes (--workers; spawn)
_CTX = {}


def _init_worker(ctx):
    _CTX.clear()
    _CTX.update(ctx)


def _run_split(s):
    lines = []
    rows = objectives_split(_CTX, s,
                            lambda text, end="\n": lines.append((text, end)))
    return rows, lines


def map_splits(ctx, splits, workers=1):
    """(rows, progress lines) of objectives_split for every split, in
    split order: in this process (workers <= 1; the progress line is
    printed as the split runs, the list is empty) or in `workers` spawned
    single-threaded processes (ProcessPoolExecutor.map keeps the order;
    the caller prints each split's lines in turn).  Every split is a pure
    function of ctx and the split, so the rows do not depend on workers."""
    if workers <= 1 or len(splits) <= 1:
        for s in splits:
            yield objectives_split(ctx, s), []
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    mp = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(splits)),
                             mp_context=mp, initializer=_init_worker,
                             initargs=(ctx,)) as pool:
        yield from pool.map(_run_split, splits)


def run_dataset(ds_name, k_max=K_MAX, n_folds=N_FOLDS, n_repeats=N_REPEATS,
                n_random_sets=N_RANDOM_SETS, random_ks=RANDOM_SET_KS,
                brute_k=BRUTE_K, oracle="matched", gamma_k=GAMMA_K,
                land_k=FL_LAND_K, workers=1):
    """Every per-split array of one dataset plus the run scalars (module
    docstring); workers > 1 evaluates the splits in that many processes
    (map_splits; the arrays are those of the serial run)."""
    print(f"\n{'#'*70}\n# Dataset: {ds_name.upper()}\n{'#'*70}")
    prov = data_provenance(ds_name)
    print(f"input {prov['file']} sha256 {prov['sha256'][:16]}… "
          f"(SHA256SUMS: {prov['sha256sums']})", flush=True)
    if prov["sha256sums"] == "mismatch":
        print(f"WARNING: {prov['file']} differs from its SHA256SUMS entry "
              f"(stale or modified copy of the data?)", file=sys.stderr,
              flush=True)
    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    names = d["benchmark_names"]
    M, N = B.shape
    bounds = dataset_bounds(ds_name, names, DATA)              # D2
    k_max = min(k_max, N - 1)
    land_k = max(0, min(land_k, k_max))
    random_ks = [k for k in random_ks if k < N]
    suites = suites_of(names)
    print(f"{M} models × {N} tasks, {O.mean()*100:.1f}% observed", flush=True)

    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED))
    t0 = time.time()
    M_ref = int(np.median([len(s.train_idx) for s in splits]))
    R_or = estimate_corr_oracle(B, O, M_ref if oracle == "matched" else None)
    sp_or = Spectral(R_or)
    print(f"oracle correlation (all {M} models, estimator of "
          f"{'M_train = ' + str(M_ref) if oracle == 'matched' else 'M'}): "
          f"{time.time() - t0:.1f}s, lambda_min {sp_or.lam[0]:.2e}",
          flush=True)

    S = len(splits)
    arr = {"rep": np.array([s.rep for s in splits], dtype=np.int64),
           "fold": np.array([s.fold for s in splits], dtype=np.int64),
           "split_id": np.array([s.split_id for s in splits], dtype=np.int64),
           "n_train": np.array([len(s.train_idx) for s in splits],
                               dtype=np.int64),
           "n_val": np.array([len(s.val_idx) for s in splits],
                             dtype=np.int64)}
    arr.update(split_arrays(S, N, k_max, n_random_sets, len(random_ks),
                            brute_k, land_k))
    ctx = dict(B=B, O=O, N=N, names=names, suites=suites, bounds=bounds,
               R_or=R_or, sp_or=sp_or, methods=list(METHODS), k_max=k_max,
               land_k=land_k, n_random_sets=n_random_sets,
               random_ks=random_ks, brute_k=brute_k, gamma_k=gamma_k)
    for i, (rows, lines) in enumerate(map_splits(ctx, splits, workers)):
        for text, end in lines:
            print(text, end=end, flush=True)
        for key, v in rows.items():
            arr[key][i] = v

    arr.update(benchmark_names=names, methods=np.array(METHODS),
               rand_ks=np.array(random_ks, dtype=np.int64),
               k_max=np.int64(k_max), holdout_frac=np.float64(HOLDOUT_FRAC),
               n_folds=np.int64(n_folds), n_repeats=np.int64(n_repeats),
               seed=np.int64(SEED),
               **{k: (np.bool_(v) if isinstance(v, bool) else np.int64(v))
                  for k, v in protocol_config().items()},
               random_sets_seed_offset=np.int64(RANDOM_SETS_SEED_OFFSET),
               n_random_sets=np.int64(n_random_sets),
               brute_k=np.int64(brute_k), gamma_k=np.int64(gamma_k),
               eps_k=np.int64(EPS_K), fl_land_k=np.int64(land_k),
               data_sha256=np.array(prov["sha256"]),
               data_sha256sums=np.array(prov["sha256sums"]),
               oracle_lam_min=np.float64(sp_or.lam[0]),
               oracle_corr=R_or, oracle=np.array(oracle),
               oracle_m_ref=np.int64(M_ref if oracle == "matched" else M))
    return arr


# ── Summaries ────────────────────────────────────────────────────

def _k_word(k):
    return "K" + int_words(k)


def first_picks(arr, m, src=""):
    """(S,) k = 1 picks of method m (-1 if none); PCA's size-1 set, not
    the first pivot of its largest set."""
    if m == "pca":
        return arr[f"pca_{src}selected_per_k"][:, 0, 0]
    return arr[f"{m}_{src}selected"][:, 0]


def methods_in(arr, methods=None):
    """The methods of `methods` (default METHODS) that the per-split
    arrays hold (a log written before facloc joined has none of its
    keys)."""
    return [m for m in (METHODS if methods is None else methods)
            if f"{m}_r2" in arr]


def paired_rows(ds_name, arr, ks):
    """Paired R^2 differences over the matched splits (paired_stats): per
    method oracle vs train selection / imputation, method pairs under
    oracle selection, exact optimum vs greedy, and (comparison "train")
    the train-side trace - facloc (paired_stats orientation) of the
    facility-location section."""
    rows = []
    meta = (arr["n_train"], arr["n_val"], arr["rep"])
    methods = methods_in(arr)

    def add(k, comparison, a_name, b_name, a, b):
        st = paired_stats(a, b, *meta)
        rows.append({"dataset": ds_name, "holdout_frac": HOLDOUT_FRAC, "k": k,
                     "comparison": comparison, "method_a": a_name,
                     "method_b": b_name, **st})

    k_max = int(arr["k_max"])
    for k in ks:
        if k > k_max:
            continue
        c = k - 1
        for m in methods:
            base = arr[f"{m}_r2"][:, c]
            add(k, "oracle_sel - train", m, m, arr[f"{m}_osel_r2"][:, c],
                base)
            add(k, "oracle_imp - train", m, m, arr[f"{m}_r2_oimp"][:, c],
                base)
            add(k, "oracle_both - train", m, m,
                arr[f"{m}_osel_r2_oimp"][:, c], base)
        for ma, mb in orient(methods):
            add(k, "oracle_sel", ma, mb, arr[f"{ma}_osel_r2"][:, c],
                arr[f"{mb}_osel_r2"][:, c])
            add(k, "oracle_both", ma, mb, arr[f"{ma}_osel_r2_oimp"][:, c],
                arr[f"{mb}_osel_r2_oimp"][:, c])
        if "facloc" in methods and "trace" in methods:
            ma, mb = orient_pair("trace", "facloc")
            add(k, "train", ma, mb, arr[f"{ma}_r2"][:, c],
                arr[f"{mb}_r2"][:, c])
    for obj in BRUTE_OBJECTIVES:
        if f"brute_{obj}_r2" not in arr:
            continue
        for k in range(1, int(arr["brute_k"]) + 1):
            add(k, "exact_opt - greedy", f"opt_{obj}", obj,
                arr[f"brute_{obj}_r2"][:, k - 1],
                arr[f"brute_{obj}_r2_greedy"][:, k - 1])
            if obj != "trace":
                add(k, "exact_opt - exact_opt", f"opt_trace", f"opt_{obj}",
                    arr["brute_trace_r2"][:, k - 1],
                    arr[f"brute_{obj}_r2"][:, k - 1])
    return rows


PAIRED_FIELDS = (["dataset", "holdout_frac", "k", "comparison", "method_a",
                  "method_b"] + STAT_FIELDS)


def _method_aggregates(arr, tk, group, k, tsuf, msuf, Ds, mac):
    """Aggregates across the methods `group` at k into tk (the k-table of
    summarize, per-method rows filled): how often trace's (and facloc's,
    if in the group) set has the largest in-sample R2_pop / realized R^2
    of the group (_trace_max_pop_frac{tsuf}, _trace_max_realized_frac
    {tsuf}, _facloc_max_..._frac), the within-split Spearman between an
    in-sample quantity and the realized R^2 across the group's sets, mean
    over splits (_split_method_spearman{tsuf}_{q}, macro
    ObjMethodSpearman{msuf}{Ds}{Q}K{k}; duplicate sets of different
    methods count once per method), and the Spearman of the group's means
    (_method_spearman{tsuf}_{f}).  tsuf = msuf = "" is the original
    computation over CORE_METHODS."""
    c = k - 1
    pop = np.stack([arr[f"{m}_pop_r2_tr"][:, c] for m in group], 1)
    rea = np.stack([arr[f"{m}_r2"][:, c] for m in group], 1)
    ok = np.all(np.isfinite(pop), axis=1)
    okr = np.all(np.isfinite(rea), axis=1)
    for top in ("trace", "facloc"):
        if top not in group:
            continue
        ti = group.index(top)
        sfx = tsuf if top == "trace" else ""
        tk[f"_{top}_max_pop_frac{sfx}"] = _num(np.mean(
            pop[ok, ti] >= pop[ok].max(axis=1) - 1e-12)) if ok.any() else None
        tk[f"_{top}_max_realized_frac{sfx}"] = _num(np.mean(
            rea[okr, ti] >= rea[okr].max(axis=1) - 1e-12)) if okr.any() else None
    for q in METHOD_SPEARMAN_QS:
        if any(f"{m}_{q}" not in arr for m in group):
            continue
        val = np.stack([arr[f"{m}_{q}"][:, c] for m in group], 1)
        rho = np.array([_spearman(val[s], rea[s])
                        for s in range(len(rea))])
        tk[f"_split_method_spearman{tsuf}_{q}"] = _num(_nanmean(rho))
        mac(f"ObjMethodSpearman{msuf}{Ds}{camel(q)}{_k_word(k)}",
            _nanmean(rho))
    mean_r2 = [tk[m]["r2"] for m in group]
    for f in METHOD_MEAN_SPEARMAN_FS:
        if any(f not in tk[m] for m in group):
            continue
        tk[f"_method_spearman{tsuf}_{f}"] = _num(_spearman(
            [np.nan if tk[m][f] is None else tk[m][f] for m in group],
            [np.nan if v is None else v for v in mean_r2]))


def _oracle_effects(rows):
    """Counts and extremes of the finite rows of one paired comparison
    (summarize: significant, large, interval widths), or None."""
    rows = [r for r in rows if np.isfinite(r["mean_diff"])]
    if not rows:
        return None
    md = np.array([r["mean_diff"] for r in rows], dtype=float)
    lo = np.array([r["ci_low"] for r in rows], dtype=float)
    hi = np.array([r["ci_high"] for r in rows], dtype=float)
    half = (hi - lo) / 2.0
    sig_pos, sig_neg = lo > 0, hi < 0
    j = int(np.argmax(np.abs(md)))
    return {"n_compared": len(rows), "n_sig": int(np.sum(sig_pos | sig_neg)),
            "n_sig_pos": int(sig_pos.sum()), "n_sig_neg": int(sig_neg.sum()),
            "n_large": int(np.sum(np.abs(md) > LARGE_DIFF)),
            "max_abs_diff": _num(abs(md[j])),
            "max_abs_at": {"method": rows[j]["method_a"],
                           "k": int(rows[j]["k"]),
                           "mean_diff": _num(md[j]),
                           "ci": [_num(lo[j]), _num(hi[j])]},
            "median_half_width": _num(np.nanmedian(half))
            if np.isfinite(half).any() else None,
            "max_half_width": _num(np.nanmax(half))
            if np.isfinite(half).any() else None,
            "significant": [{"method": r["method_a"], "k": int(r["k"]),
                             "mean_diff": _num(r["mean_diff"]),
                             "ci": [_num(r["ci_low"]), _num(r["ci_high"])]}
                            for r, s in zip(rows, sig_pos | sig_neg) if s],
            "large": [{"method": r["method_a"], "k": int(r["k"]),
                       "mean_diff": _num(r["mean_diff"]),
                       "ci": [_num(r["ci_low"]), _num(r["ci_high"])]}
                      for r in rows if abs(r["mean_diff"]) > LARGE_DIFF]}


def summarize(ds_name, arr, paired, ks=KS_REPORT):
    """Summary tables and the flat macros dict of one dataset."""
    Ds = camel(ds_name)
    names = [str(n) for n in arr["benchmark_names"]]
    k_max = int(arr["k_max"])
    ks = [k for k in ks if k <= k_max]
    macros = {}
    tables = {"methods": {}, "first_pick": {}, "surrogate": {}, "brute": {},
              "theory": {}, "stability": {}, "paths": {},
              "neighbourhood": {}, "gap": {}, "insample_gap": {},
              "oracle_effects": {}}
    full_oracle = "oracle" in arr and str(arr["oracle"]) == "full"
    # The methods of the log; aggregates across methods over CORE_METHODS
    # (original keys) and, with facloc, over all of them (twins)
    methods = methods_in(arr)
    core = methods_in(arr, CORE_METHODS)
    groups = [(core, "", "")]
    if len(methods) > len(core):
        groups.append((methods, "_with_facloc", "WithFacloc"))

    def mac(name, value):
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} is not letters-only")
        if full_oracle:                 # ObjX... -> ObjFullX...
            name = "ObjFull" + name[len("Obj"):]
        v = _num(value) if value is not None else None
        if v is not None:
            macros[name] = v

    pair_idx = {(r["k"], r["comparison"], r["method_a"], r["method_b"]): r
                for r in paired}

    # Per-method means at the reported k
    fields = {"r2": None, "pop_r2_tr": "PopRsq", "pop_r2_or": "PopRsqOracle",
              "exp_r2_tr": "ExpRsqTrain", "exp_r2_or": "ExpRsqOracle",
              "logdet_tr": "LogDet", "mi_tr": "MiVal",
              "mean_abs_r_or": "Redund", "max_abs_r_or": "RedundMax",
              "eff_rank_or": "EffRank", "cov_cells": "CoverCells",
              "cov_models": "CoverModels", "k_eff": "KEff",
              "n_suites": "NSuites",
              "osel_r2": "OracleSelRsq", "osel_r2_oimp": "OracleBothRsq",
              "r2_oimp": "OracleImpRsq", "jacc_oracle": "JaccOracle",
              "facloc_tr": "FaclocVal"}
    for k in ks:
        c = k - 1
        tk = {}
        for m in methods:
            row = {}
            for f, mname in fields.items():
                if f"{m}_{f}" not in arr:       # a log without facloc_tr
                    continue
                v = _nanmean(arr[f"{m}_{f}"][:, c])
                row[f] = _num(v)
                if mname:
                    mac(f"Obj{mname}{Ds}{camel(m)}{_k_word(k)}", v)
            row["r2_sd"] = _num(_nanstd(arr[f"{m}_r2"][:, c]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(arr[f"{m}_r2"][:, c])
            row["r2_median"] = _num(med)
            mac(f"ObjMedianRsq{Ds}{camel(m)}{_k_word(k)}", med)
            opt = arr[f"{m}_pop_r2_tr"][:, c] - arr[f"{m}_pop_r2_or"][:, c]
            row["optimism"] = _num(_nanmean(opt))
            mac(f"ObjOptimism{Ds}{camel(m)}{_k_word(k)}", _nanmean(opt))
            # in-sample R2_pop of trace's set is the largest among methods?
            r = pair_idx.get((k, "oracle_sel - train", m, m))
            if r is not None:
                row["oracle_sel_gain"] = _num(r["mean_diff"])
                row["oracle_sel_gain_ci"] = [_num(r["ci_low"]),
                                             _num(r["ci_high"])]
                for suf, key in (("", "mean_diff"), ("Lo", "ci_low"),
                                 ("Hi", "ci_high")):
                    mac(f"ObjOracleSelGain{suf}{Ds}{camel(m)}{_k_word(k)}",
                        r[key])
            tk[m] = row
        # Aggregates across methods: over CORE_METHODS under the original
        # keys, over every method under the _with_facloc / WithFacloc
        # twins (module docstring, "Methods and invariance")
        for group, tsuf, msuf in groups:
            _method_aggregates(arr, tk, group, k, tsuf, msuf, Ds, mac)
        tables["methods"][str(k)] = tk

    # Gap decomposition r2 - pop = (exp - pop) + (r2 - exp), one truth at a
    # time, on the splits where all five quantities are finite
    for k in ks:
        c = k - 1
        gk = {}
        for m in methods:
            q = {f: arr[f"{m}_{f}"][:, c] for f in
                 ("r2", "pop_r2_tr", "pop_r2_or", "exp_r2_tr", "exp_r2_or")}
            ok = np.all([np.isfinite(v) for v in q.values()], axis=0)
            row = {"n_splits": int(ok.sum())}
            for truth, tw in (("tr", "Train"), ("or", "Oracle")):
                pop, exp = q[f"pop_r2_{truth}"][ok], q[f"exp_r2_{truth}"][ok]
                total = _nanmean(q["r2"][ok] - pop) if ok.any() else np.nan
                mask = _nanmean(exp - pop) if ok.any() else np.nan
                share = (mask / total if np.isfinite(total) and total != 0
                         else np.nan)
                row[truth] = {"total": _num(total), "mask": _num(mask),
                              "rest": _num(total - mask),
                              "mask_share": _num(share)}
                for fname, v in (("Total", total), ("Mask", mask),
                                 ("MaskShare", share)):
                    mac(f"ObjGap{fname}{tw}{Ds}{camel(m)}{_k_word(k)}", v)
            gk[m] = row
        tables["gap"][str(k)] = gk

    # Largest in-sample-to-realized gap |mean(r2 - pop_r2_tr)| over k, per
    # method and over each group of methods
    worst = {}
    for m in methods:
        diff = _nanmean(arr[f"{m}_r2"] - arr[f"{m}_pop_r2_tr"], axis=0)
        if not np.isfinite(diff).any():
            continue
        j = int(np.nanargmax(np.abs(diff)))
        tables["insample_gap"][m] = {"max_abs": _num(abs(diff[j])),
                                     "signed": _num(diff[j]), "k": j + 1}
        mac(f"ObjInSampleGapMax{Ds}{camel(m)}", abs(diff[j]))
        mac(f"ObjInSampleGapWorstK{Ds}{camel(m)}", j + 1)
        worst[m] = (abs(diff[j]), j + 1)
    for group, tsuf, msuf in groups:
        worst_all = (np.nan, None, None)
        for m in group:
            if m in worst and not worst[m][0] <= worst_all[0]:
                worst_all = (worst[m][0], m, worst[m][1])
        if worst_all[1] is not None:
            tables["insample_gap"]["_all" + tsuf] = {
                "max_abs": _num(worst_all[0]), "method": worst_all[1],
                "k": worst_all[2]}
            mac(f"ObjInSampleGapMax{msuf}{Ds}", worst_all[0])
            mac(f"ObjInSampleGapWorstK{msuf}{Ds}", worst_all[2])

    # Oracle effects over every method (of a group) and k of a paired
    # comparison: how many are significant / large, and how wide the
    # intervals are
    for group, tsuf, msuf in groups:
        key = "oracle_effects" + tsuf
        tables.setdefault(key, {})
        for comparison, cname in (("oracle_sel - train", "OracleSel"),
                                  ("oracle_imp - train", "OracleImp"),
                                  ("oracle_both - train", "OracleBoth")):
            eff = _oracle_effects([r for r in paired
                                   if r["comparison"] == comparison
                                   and r["method_a"] in group])
            if eff is None:
                continue
            tables[key][comparison] = eff
            for fname, fkey in (("NCompared", "n_compared"), ("NSig", "n_sig"),
                                ("NSigPos", "n_sig_pos"),
                                ("NSigNeg", "n_sig_neg"),
                                ("NLarge", "n_large"),
                                ("MaxAbsDiff", "max_abs_diff"),
                                ("MedianHalfWidth", "median_half_width"),
                                ("MaxHalfWidth", "max_half_width")):
                mac(f"Obj{cname}{msuf}{fname}{Ds}",
                    eff[fkey] if eff[fkey] is not None else np.nan)

    # k = 1 picks
    for m in methods:
        sel = first_picks(arr, m)
        ok = sel >= 0
        if not ok.any():
            continue
        cnt = Counter(sel[ok].tolist())
        j, n = cnt.most_common(1)[0]
        rows = np.flatnonzero(ok)
        pick = sel[ok]

        def at(q):
            return _nanmean(arr[f"single_{q}"][rows, pick])
        row = {"modal_name": names[j], "modal_splits": int(n),
               "n_distinct": len(cnt), "n_splits": int(ok.sum()),
               "r2": _num(_nanmean(arr[f"{m}_r2"][:, 0]))}
        for q in ("rest_r2_tr", "rest_r2_or", "mi_tr", "pop_r2_tr",
                  "pop_r2_or", "exp_r2_or", "hub_tr", "p_low_share_tr",
                  "r2_top_share_tr", "cov_models", "n_obs_train"):
            row[q] = _num(at(q))
        tables["first_pick"][m] = row
        for q, fname in (("r2", "Rsq"), ("rest_r2_tr", "RestRsq"),
                         ("mi_tr", "MiScore"), ("pop_r2_tr", "PopRsq"),
                         ("pop_r2_or", "PopRsqOracle"),
                         ("exp_r2_or", "ExpRsqOracle"),
                         ("p_low_share_tr", "PLowShare"),
                         ("cov_models", "CoverModels"),
                         ("n_obs_train", "NObsTrain")):
            mac(f"ObjFirst{Ds}{camel(m)}{fname}", row[q] if row[q] is not None
                else np.nan)
        mac(f"ObjFirst{Ds}{camel(m)}ModalSplits", n)
        mac(f"ObjFirst{Ds}{camel(m)}NDistinct", len(cnt))
    # singleton landscape (all eligible (split, benchmark) cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rest = arr["single_rest_r2_tr"]
        split_min = np.nanmin(rest, axis=1)
        single = {
            "rest_r2_tr_median": _num(np.nanmedian(rest)),
            "rest_r2_tr_min": _num(np.nanmin(rest)),
            "rest_r2_tr_split_min_median": _num(np.nanmedian(split_min)),
            "p_low_share_tr_median": _num(np.nanmedian(
                arr["single_p_low_share_tr"])),
            "r2_best_mean": _num(_nanmean(np.nanmax(arr["single_r2"],
                                                    axis=1))),
            "r2_median_mean": _num(_nanmean(np.nanmedian(arr["single_r2"],
                                                         axis=1))),
        }
    tables["first_pick"]["_singletons"] = single
    for key, fname in (("rest_r2_tr_median", "RestRsqMedian"),
                       ("rest_r2_tr_min", "RestRsqMin"),
                       ("rest_r2_tr_split_min_median", "RestRsqSplitMinMedian"),
                       ("p_low_share_tr_median", "PLowShareMedian")):
        mac(f"ObjSingle{Ds}{fname}", single[key] if single[key] is not None
            else np.nan)

    # Surrogate quality: within-split Spearman with realized R^2
    sur = {}
    S = arr["single_r2"].shape[0]
    rho1 = {}
    for key, skey in SINGLE_SURROGATE_KEY.items():
        if skey is None or f"single_{skey}" not in arr:
            continue
        rho = np.array([_spearman(arr[f"single_{skey}"][s],
                                  arr["single_r2"][s]) for s in range(S)])
        rho1[key] = rho
    sur["1"] = {key: {"rho_mean": _num(_nanmean(r)), "rho_sd": _num(_nanstd(r)),
                      "frac_pos": _num(_nanmean(r > 0))}
                for key, r in rho1.items()}
    for key, r in rho1.items():
        mac(f"ObjSpearman{Ds}{camel(key)}Kone", _nanmean(r))
    for ik, k in enumerate(arr["rand_ks"].tolist()):
        sk = {}
        for key, (q, _) in SURROGATES.items():
            if f"rand_{q}" not in arr:
                continue
            r = np.array([_spearman(arr[f"rand_{q}"][s, :, ik],
                                    arr["rand_r2"][s, :, ik])
                          for s in range(S)])
            sk[key] = {"rho_mean": _num(_nanmean(r)),
                       "rho_sd": _num(_nanstd(r)),
                       "frac_pos": _num(_nanmean(r > 0))}
            mac(f"ObjSpearman{Ds}{camel(key)}{_k_word(k)}", _nanmean(r))
        for q in ("mean_abs_r_or", "eff_rank_or", "cov_cells", "k_eff"):
            r = np.array([_spearman(arr[f"rand_{q}"][s, :, ik],
                                    arr["rand_r2"][s, :, ik])
                          for s in range(S)])
            sk[q] = {"rho_mean": _num(_nanmean(r)),
                     "rho_sd": _num(_nanstd(r)),
                     "frac_pos": _num(_nanmean(r > 0))}
            mac(f"ObjSpearman{Ds}{camel(q)}{_k_word(k)}", _nanmean(r))
        sk["_rand_r2_mean"] = _num(_nanmean(arr["rand_r2"][:, :, ik]))
        sur[str(k)] = sk
    tables["surrogate"] = sur

    # Exact optimum vs greedy
    for k in range(1, int(arr["brute_k"]) + 1):
        bk = {}
        for obj in BRUTE_OBJECTIVES:
            if f"brute_{obj}_value" not in arr:
                continue
            v = arr[f"brute_{obj}_value"][:, k - 1]
            g = arr[f"brute_{obj}_greedy"][:, k - 1]
            row = {"opt_value": _num(_nanmean(v)),
                   "greedy_value": _num(_nanmean(g)),
                   "gap": _num(_nanmean(v - g)),
                   "same_frac": _num(_nanmean(arr[f"brute_{obj}_same"][:,
                                                                     k - 1])),
                   "r2_opt": _num(_nanmean(arr[f"brute_{obj}_r2"][:, k - 1])),
                   "r2_greedy": _num(_nanmean(
                       arr[f"brute_{obj}_r2_greedy"][:, k - 1]))}
            if obj in ("trace", "mi", "facloc"):
                # trace: ratio of the explained complement variance
                # F_tr - k (the numerator of R2_pop); MI and facility
                # location (monotone submodular, F(empty) = 0: greedy within
                # 1 - 1/e): ratio of values.  Per split; the mean hides
                # the worst split (RatioMin).
                off = k if obj == "trace" else 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = (g - off) / (v - off)
                    f_ratios = g / v
                ratios = np.where(np.isfinite(ratios), ratios, np.nan)
                f_ratios = np.where(np.isfinite(f_ratios), f_ratios, np.nan)
                has = np.isfinite(ratios).any()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    row["ratio"] = _num(_nanmean(ratios))
                    row["ratio_min"] = _num(np.nanmin(ratios)) if has else None
                    row["ratio_median"] = (_num(np.nanmedian(ratios)) if has
                                           else None)
                    row["ratio_frac_below_ninetyseven_half"] = (
                        _num(np.mean(ratios[np.isfinite(ratios)] < 0.975))
                        if has else None)
                    row["f_ratio"] = _num(_nanmean(f_ratios))
                    row["f_ratio_min"] = (_num(np.nanmin(f_ratios))
                                          if np.isfinite(f_ratios).any()
                                          else None)
                for fname, key in (("Ratio", "ratio"), ("RatioMin", "ratio_min"),
                                   ("RatioMedian", "ratio_median"),
                                   ("FRatio", "f_ratio"),
                                   ("FRatioMin", "f_ratio_min")):
                    mac(f"ObjBrute{Ds}{camel(obj)}{_k_word(k)}{fname}",
                        row[key] if row[key] is not None else np.nan)
            mac(f"ObjBrute{Ds}{camel(obj)}{_k_word(k)}Gap", row["gap"]
                if row["gap"] is not None else np.nan)
            mac(f"ObjBrute{Ds}{camel(obj)}{_k_word(k)}SameFrac",
                row["same_frac"] if row["same_frac"] is not None else np.nan)
            mac(f"ObjBrute{Ds}{camel(obj)}{_k_word(k)}OptRsq",
                row["r2_opt"] if row["r2_opt"] is not None else np.nan)
            bk[obj] = row
        tables["brute"][str(k)] = bk

    # Theory diagnostics
    for key in [k for k in arr if k.startswith("theory_")]:
        v = arr[key]
        name = key[len("theory_"):]
        fin = v[~np.isnan(v)]
        tables["theory"][name] = {"mean": _num(_nanmean(v)),
                                  "median": _num(np.median(fin))
                                  if fin.size else None,
                                  "min": _num(np.min(fin)) if fin.size
                                  else None,
                                  "max": _num(np.max(fin)) if fin.size
                                  else None,
                                  "frac_neg": _num(np.mean(fin < 0))
                                  if fin.size else None}
        mac(f"ObjTheory{Ds}{camel(name)}", _nanmean(v))
        if name in THEORY_RANGE_KEYS and fin.size:
            for fname, val in (("Min", np.min(fin)), ("Max", np.max(fin)),
                               ("Median", np.median(fin))):
                mac(f"ObjTheory{Ds}{camel(name)}{fname}", val)
    tables["theory"]["oracle_lam_min"] = _num(arr["oracle_lam_min"])
    # Neighbourhoods of the picks under the oracle correlation: tightness
    # (largest |r| with another benchmark), cluster size (# benchmarks with
    # |r| >= CLOSE_R) and relevance (mean r^2 with the others), for the
    # k = 1 pick and the size-5 set of every method
    if "oracle_corr" in arr:
        Rabs = np.abs(np.asarray(arr["oracle_corr"], dtype=float))
        np.fill_diagonal(Rabs, 0.0)
        tight = Rabs.max(axis=1)
        n_close = (Rabs >= CLOSE_R).sum(axis=1)
        relevance = (Rabs ** 2).sum(axis=1) / max(Rabs.shape[0] - 1, 1)
        tables["neighbourhood"] = {"close_r": CLOSE_R}
        k5 = min(5, k_max)
        for m in methods:
            sets = {"first": first_picks(arr, m)[:, None],
                    "set": (arr["pca_selected_per_k"][:, k5 - 1, :k5]
                            if m == "pca" else arr[f"{m}_selected"][:, :k5])}
            row = {}
            for which, idx in sets.items():
                ok = idx >= 0
                safe = np.maximum(idx, 0)
                for name, v in (("tight", tight), ("n_close", n_close),
                                ("relevance", relevance)):
                    val = _nanmean(np.where(ok, v[safe], np.nan))
                    row[f"{which}_{name}"] = _num(val)
                    mac(f"ObjNbr{Ds}{camel(m)}{camel(which)}{camel(name)}",
                        val)
                # training models that observed the picks (split-specific)
                n_obs = np.take_along_axis(arr["single_n_obs_train"], safe,
                                           axis=1)
                n_obs = np.where(ok, n_obs, np.nan)
                row[f"{which}_n_obs"] = _num(_nanmean(n_obs))
                row[f"{which}_frac_obs_below_ten"] = _num(_nanmean(
                    np.where(np.isfinite(n_obs), n_obs < 10, np.nan)))
                mac(f"ObjNbr{Ds}{camel(m)}{camel(which)}NObs",
                    _nanmean(n_obs))
                mac(f"ObjNbr{Ds}{camel(m)}{camel(which)}FracObsBelowTen",
                    row[f"{which}_frac_obs_below_ten"]
                    if row[f"{which}_frac_obs_below_ten"] is not None
                    else np.nan)
            tables["neighbourhood"][m] = row
    # Greedy-path terms: mean over splits per step, and over steps 1..5
    tables["paths"] = {}
    for key in PATH_KEYS:
        if f"path_{key}" not in arr:
            continue
        a = arr[f"path_{key}"]
        tables["paths"][key] = {
            "per_step": [_num(v) for v in _nanmean(a, axis=0)],
            "mean_first_five": _num(_nanmean(a[:, :5]))}
        mac(f"ObjPath{Ds}{camel(key)}", _nanmean(a[:, :5]))
    for obj in STAB_OBJECTIVES:
        if f"stab_{obj}" not in arr:
            continue
        tables["stability"][obj] = {
            "spearman_mean": _num(_nanmean(arr[f"stab_{obj}"])),
            "same_pick_frac": _num(_nanmean(arr[f"stab_{obj}_same_pick"]))}
        mac(f"ObjStab{Ds}{camel(obj)}", _nanmean(arr[f"stab_{obj}"]))
        mac(f"ObjStab{Ds}{camel(obj)}SamePick",
            _nanmean(arr[f"stab_{obj}_same_pick"]))
    if "fl_jacc_trace" in arr:
        tables["facloc"] = summarize_facloc(arr, pair_idx, ks, Ds, mac)
    return tables, macros


def summarize_facloc(arr, pair_idx, ks, Ds, mac):
    """Table "facloc" of summarize (facility location vs trace, module
    docstring) and its ObjFl... macros.  pair_idx: the paired rows by
    (k, comparison, method_a, method_b); the "train" row is trace -
    facloc (paired_stats orientation), reported here as facloc - trace."""
    k_max = int(arr["k_max"])
    out = {"vs_trace": {}, "overlap": {}, "path": {}, "landscape": {}}
    ks_all = np.arange(1, k_max + 1)
    shared = arr["fl_shared_trace"] / ks_all[None, :]
    jacc = arr["fl_jacc_trace"]
    same = np.where(np.isfinite(jacc), jacc == 1.0, np.nan)
    out["overlap"] = {"k": ks_all.tolist(),
                      "shared_frac": [_num(v) for v in _nanmean(shared, 0)],
                      "jaccard": [_num(v) for v in _nanmean(jacc, 0)],
                      "same_frac": [_num(v) for v in _nanmean(same, 0)]}
    ks_pair = sorted({r[0] for r in pair_idx if r[1] == "train"})
    a, b = orient_pair("trace", "facloc")
    sign = 1.0 if a == "facloc" else -1.0       # facloc - trace
    for k in sorted(set(ks) | set(ks_pair)):
        if k > k_max:
            continue
        c = k - 1
        row = {"r2_facloc": _num(_nanmean(arr["facloc_r2"][:, c])),
               "r2_trace": _num(_nanmean(arr["trace_r2"][:, c])),
               "shared_frac": _num(_nanmean(shared[:, c])),
               "jaccard": _num(_nanmean(jacc[:, c])),
               "same_frac": _num(_nanmean(same[:, c]))}
        r = pair_idx.get((k, "train", a, b))
        if r is not None:
            lo, hi = sign * r["ci_low"], sign * r["ci_high"]
            row.update(diff=_num(sign * r["mean_diff"]),
                       ci=[_num(min(lo, hi)), _num(max(lo, hi))],
                       win_rate=_num(r["win_rate"] if sign > 0
                                     else r["loss_rate"]),
                       tie_rate=_num(r["tie_rate"]),
                       n_splits=int(r["n_splits"]))
        out["vs_trace"][str(k)] = row
        if k not in ks:
            continue
        kw = _k_word(k)
        if r is not None:
            for fname, key in (("Diff", "diff"), ("Win", "win_rate")):
                mac(f"ObjFlVsTrace{fname}{Ds}{kw}",
                    row[key] if row[key] is not None else np.nan)
            for fname, v in (("DiffLo", row["ci"][0]),
                             ("DiffHi", row["ci"][1])):
                mac(f"ObjFlVsTrace{fname}{Ds}{kw}",
                    v if v is not None else np.nan)
        for fname, key in (("Shared", "shared_frac"), ("Jacc", "jaccard"),
                           ("Same", "same_frac")):
            mac(f"ObjFlVsTrace{fname}{Ds}{kw}",
                row[key] if row[key] is not None else np.nan)

    land_k = int(arr["fl_land_k"]) if "fl_land_k" in arr else 0
    for P in FL_PATHS:
        out["path"][P] = {}
        land = {"best_dr2_per_pick": [_num(v) for v in _nanmean(
                    arr[f"fl_land_{P}_best_dr2"], 0)] if land_k else [],
                "n_cand_per_pick": [_num(v) for v in _nanmean(
                    arr[f"fl_land_{P}_n_cand"], 0)] if land_k else []}
        for k in ks:
            if k <= land_k:
                mac(f"ObjFlLandBestGain{Ds}{camel(P)}{_k_word(k)}",
                    _nanmean(arr[f"fl_land_{P}_best_dr2"][:, k - 1]))
        for O in FL_GAINS:
            rho = arr[f"fl_path_{P}_rho_{O}"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(rho) if np.isfinite(rho).any() else np.nan
            out["path"][P][O] = {
                "rho_mean": _num(_nanmean(rho)), "rho_median": _num(med),
                "frac_pos": _num(_nanmean(np.where(np.isfinite(rho),
                                                   rho > 0, np.nan))),
                "gain_per_step": [_num(v) for v in _nanmean(
                    arr[f"fl_path_{P}_gain_{O}"], 0)]}
            mac(f"ObjFlPathRho{Ds}{camel(P)}Gain{camel(O)}", _nanmean(rho))
            if not land_k:
                continue
            entry = {}
            for key, fname in (("rho", "Rho"), ("regret", "Regret")):
                a_ = arr[f"fl_land_{P}_{key}_{O}"]
                entry[f"{key}_per_pick"] = [_num(v) for v in _nanmean(a_, 0)]
                entry[f"{key}_mean_first_five"] = _num(_nanmean(a_[:, :5]))
                mac(f"ObjFlLand{fname}{Ds}{camel(P)}Gain{camel(O)}",
                    _nanmean(a_[:, :5]))
                for k in ks:
                    if k <= land_k:
                        mac(f"ObjFlLand{fname}{Ds}{camel(P)}Gain{camel(O)}"
                            f"{_k_word(k)}", _nanmean(a_[:, k - 1]))
            land[O] = entry
        out["path"][P]["dr2_per_step"] = [_num(v) for v in _nanmean(
            arr[f"fl_path_{P}_dr2"], 0)]
        out["landscape"][P] = land
    return out


def print_summary(ds_name, tables):
    print(f"\n== {display_name(ds_name)}: mechanism summary ==")
    for k, tk in tables["methods"].items():
        print(f"k = {k}: method   R²     popR²tr popR²or expR²or  "
              f"|r|in  effrk  cover  k_eff  oracle-sel  gain [95% CI]")
        for m in METHODS:
            if m not in tk:
                continue
            r = tk[m]

            def f(x, w=7, p=3):
                return f"{x:{w}.{p}f}" if x is not None else " " * (w - 3) + "n/a"
            ci = r.get("oracle_sel_gain_ci") or [None, None]
            print(f"  {m:<11}{f(r['r2'])}{f(r['pop_r2_tr'])}{f(r['pop_r2_or'])}"
                  f"{f(r['exp_r2_or'])}{f(r['mean_abs_r_or'])}"
                  f"{f(r['eff_rank_or'])}{f(r['cov_cells'])}"
                  f"{f(r.get('k_eff'), 7, 2)}"
                  f"{f(r['osel_r2'], 10)}  {f(r.get('oracle_sel_gain'), 7)} "
                  f"[{f(ci[0], 6)}, {f(ci[1], 6)}]")
        print(f"  trace has the largest in-sample R²_pop in "
              f"{tk['_trace_max_pop_frac']} of splits, the largest realized "
              f"R² in {tk['_trace_max_realized_frac']} (of "
              f"{', '.join(CORE_METHODS)})")
        if "_facloc_max_realized_frac" in tk:
            print(f"  with facloc: trace largest R²_pop / realized R² in "
                  f"{tk.get('_trace_max_pop_frac_with_facloc')} / "
                  f"{tk.get('_trace_max_realized_frac_with_facloc')}, "
                  f"facloc in {tk['_facloc_max_pop_frac']} / "
                  f"{tk['_facloc_max_realized_frac']} of splits")
    print("k = 1 picks (modal pick, splits; R²_j|rest; MI score; R²_pop; "
          "realized R²; models covered):")
    for m, r in tables["first_pick"].items():
        if m.startswith("_"):
            continue
        print(f"  {m:<11}{r['modal_name'][:34]:<36}{r['modal_splits']:>3}  "
              f"{_fmt(r['rest_r2_tr'])}  {_fmt(r['mi_tr'], '.2f')}  "
              f"{_fmt(r['pop_r2_tr'])}  {_fmt(r['r2'])}  "
              f"{_fmt(r['cov_models'], '.2f')}  P_jj low-10% share "
              f"{_fmt(r['p_low_share_tr'], '.3f')}")
    sg = tables["first_pick"].get("_singletons", {})
    print(f"  all eligible singletons: R²_j|rest min "
          f"{_fmt(sg.get('rest_r2_tr_min'), '.5f')}, median of split minima "
          f"{_fmt(sg.get('rest_r2_tr_split_min_median'), '.5f')}, median "
          f"{_fmt(sg.get('rest_r2_tr_median'), '.4f')}; P_jj low-10% share "
          f"median {_fmt(sg.get('p_low_share_tr_median'), '.3f')}")
    print("Within-split Spearman with realized R² (mean over splits):")
    for k, sk in tables["surrogate"].items():
        print(f"  k = {k}: " + ", ".join(
            f"{key} {_fmt(v['rho_mean'], '+.2f')}" for key, v in sk.items()
            if not key.startswith("_")))
    for k, bk in tables["brute"].items():
        print(f"  exact optimum k = {k}: " + ", ".join(
            f"{obj} same {_fmt(r['same_frac'], '.2f')} gap "
            f"{_fmt(r['gap'], '.3g')} R² opt {_fmt(r['r2_opt'])} / greedy "
            f"{_fmt(r['r2_greedy'])}"
            + (f" ratio mean {_fmt(r.get('ratio'), '.4f')} min "
               f"{_fmt(r.get('ratio_min'), '.4f')}" if "ratio" in r else "")
            for obj, r in bk.items()))
    th = tables["theory"]

    def t(key, stat="mean", spec=".3f"):
        return _fmt((th.get(key) or {}).get(stat), spec)
    print(f"  theory: trace gains increasing along its path in "
          f"{t('trace_increase_frac')} of (step, candidate) pairs (max "
          f"step-to-step ratio {t('trace_max_step_ratio', 'max', '.2f')}, "
          f"max ratio to the first gain {t('trace_max_ratio', 'max', '.2f')})"
          f"; submodularity ratio gamma median "
          f"{t('trace_gamma', 'median')} [min {t('trace_gamma', 'min')}], "
          f"bound 1-e^-gamma median {t('trace_gamma_bound', 'median')} vs "
          f"spectral 1-e^-lam_min {t('trace_spectral_bound', 'median', '.4f')}"
          f", sparse-eigenvalue bound <= "
          f"{t('trace_sparse_bound_ub', 'max', '.3f')}")
    mi_chosen = th.get("mi_chosen_neg_gains") or th.get("mi_neg_gains") or {}
    print(f"  theory: MI's own negative gains per split "
          f"{_fmt(mi_chosen.get('mean'), '.2f')}; but some eligible "
          f"candidate has a negative gain in "
          f"{t('mi_nonmonotone', 'mean', '.2f')} of splits (path min "
          f"{t('mi_min_gain', 'min', '.3f')}, share of negative candidate "
          f"gains {t('mi_neg_gain_frac', 'mean', '.3f')}, pairs min "
          f"{t('mi_min_pair_gain', 'min', '.3f')}); Krause epsilon >= "
          f"{t('mi_eps_lb', 'median', '.3f')} (median; max "
          f"{t('mi_eps_lb', 'max', '.3f')}), k·eps / MI(greedy) at k = "
          f"{EPS_K}: median {t('mi_keps_rel', 'median', '.2f')}")
    print(f"  theory: entropy steps with sigma² < 1/(2πe) per split "
          f"{t('entropy_below_floor', 'mean', '.2f')}; λ_min(R_tr) median "
          f"{t('lam_min', 'median', '.2e')}, oracle "
          f"{_fmt(th.get('oracle_lam_min'), '.2e')}")
    if tables.get("gap"):
        print("  gap r2 - R²_pop = mask + rest, per truth (train | oracle): "
              "total, mask, mask share")
        for k, gk in tables["gap"].items():
            print(f"    k = {k}: " + "; ".join(
                f"{m} " + " | ".join(
                    f"{_fmt(g[tr]['total'], '+.3f')} "
                    f"{_fmt(g[tr]['mask'], '+.3f')} "
                    f"{_fmt(g[tr]['mask_share'], '.2f')}"
                    for tr in ("tr", "or"))
                for m, g in gk.items() if m in ("trace", "mi", "random")))
    if tables.get("insample_gap", {}).get("_all"):
        a = tables["insample_gap"]["_all"]
        print(f"  largest |mean(r2 - R²_pop,tr)| over methods and k: "
              f"{_fmt(a['max_abs'], '.4f')} ({a['method']}, k = {a['k']}); "
              f"trace {_fmt(tables['insample_gap'].get('trace', {}).get('max_abs'), '.4f')}")
    for comparison, eff in tables.get("oracle_effects", {}).items():
        at = eff["max_abs_at"]
        print(f"  {comparison}: {eff['n_sig']}/{eff['n_compared']} "
              f"significant ({eff['n_sig_pos']} +, {eff['n_sig_neg']} -), "
              f"{eff['n_large']} with |diff| > {LARGE_DIFF}; largest "
              f"{_fmt(at['mean_diff'], '+.3f')} ({at['method']} k = {at['k']}"
              f", CI [{_fmt(at['ci'][0], '+.3f')}, {_fmt(at['ci'][1], '+.3f')}"
              f"]); CI half-width median "
              f"{_fmt(eff['median_half_width'], '.3f')}, max "
              f"{_fmt(eff['max_half_width'], '.3f')}")
    print("  k = 1 score stability (Spearman train vs oracle; same pick): "
          + ", ".join(f"{o} {_fmt(v['spearman_mean'])}/"
                      f"{_fmt(v['same_pick_frac'], '.2f')}"
                      for o, v in tables["stability"].items()))
    if tables.get("paths"):
        print("  greedy-path terms, mean over picks 1-5 (nats): " + ", ".join(
            f"{key} {_fmt(v['mean_first_five'], '+.2f')}"
            for key, v in tables["paths"].items()))
    nb = tables.get("neighbourhood", {})
    if nb:
        print(f"  picks' neighbourhoods (oracle R; first pick | size-5 set): "
              f"max |r|, # with |r| >= {nb['close_r']}, mean r², "
              f"# training models observing, share observed by < 10")
        for m in METHODS:
            r = nb.get(m)
            if r:
                print(f"    {m:<11}" + " | ".join(
                    f"{_fmt(r[w + '_tight'])} {_fmt(r[w + '_n_close'], '5.1f')}"
                    f" {_fmt(r[w + '_relevance'])} "
                    f"{_fmt(r[w + '_n_obs'], '7.1f')} "
                    f"{_fmt(r[w + '_frac_obs_below_ten'], '.2f')}"
                    for w in ("first", "set")))
    if tables.get("facloc"):
        print_facloc(tables["facloc"])


def print_facloc(fl):
    """The facility-location section of print_summary."""
    print("  facility location vs trace (train side): realized R² facloc "
          "- trace [95% NB CI], facloc ahead; shared picks / k, Jaccard, "
          "identical sets:")
    for k, r in fl["vs_trace"].items():
        ci = r.get("ci") or [None, None]
        print(f"    k = {k:>2}: R² {_fmt(r['r2_facloc'])} vs "
              f"{_fmt(r['r2_trace'])}, diff {_fmt(r.get('diff'), '+.3f')} "
              f"[{_fmt(ci[0], '+.3f')}, {_fmt(ci[1], '+.3f')}], ahead in "
              f"{_fmt(r.get('win_rate'), '.2f')}; shared "
              f"{_fmt(r['shared_frac'], '.2f')}, Jaccard "
              f"{_fmt(r['jaccard'], '.2f')}, same {_fmt(r['same_frac'], '.2f')}")
    print("    Spearman of objective gains and realized R² gains along each "
          "greedy path (mean over splits):")
    for P, byO in fl["path"].items():
        print(f"      path {P:<7}" + ", ".join(
            f"{O} gains {_fmt(v['rho_mean'], '+.2f')}"
            for O, v in byO.items() if isinstance(v, dict)))
    if any(fl["landscape"][P].get(O) for P in fl["landscape"]
           for O in FL_GAINS):
        print("    gain landscape (all candidates of a greedy state; picks "
              "1-5): Spearman with the realized gain | regret of the "
              "objective's pick:")
        for P, land in fl["landscape"].items():
            print(f"      states of {P:<7}" + ", ".join(
                f"{O} {_fmt(land[O]['rho_mean_first_five'], '+.2f')} | "
                f"{_fmt(land[O]['regret_mean_first_five'], '.4f')}"
                for O in FL_GAINS if land.get(O)))


def _fmt(x, spec=".3f"):
    """format(x, spec), 'n/a' for None / NaN."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return format(x, spec)


# ── Plotting ─────────────────────────────────────────────────────

PLOT_METHODS = ["facloc", "trace", "entropy", "pca", "mi", "hub", "random"]
# Marker per method: secondary encoding next to the shared METHOD_COLOR
PLOT_MARKER = {"trace": "^", "entropy": "o", "pca": "d",
               "mi": "v", "hub": "s", "random": "x", "facloc": "P"}
CURVE_METHODS = ["facloc", "trace", "entropy", "mi", "random"]
# Facility-location figure: line style per greedy path (own path solid)
FL_PATH_STYLE = {"facloc": "-", "trace": "--"}


def _plot_methods(arr, methods):
    """The methods of `methods` that the log holds (methods_in)."""
    return methods_in(arr, methods)


def _panel_singletons(ax, arr, title=True, ds_name=""):
    x = _nanmean(arr["single_mi_tr"], axis=0)
    y = _nanmean(arr["single_r2"], axis=0)
    c = _nanmean(arr["single_pop_r2_tr"], axis=0)
    ok = np.isfinite(x) & np.isfinite(y)
    sc = ax.scatter(x[ok], y[ok], c=c[ok], cmap="Blues", s=14,
                    edgecolors="#555555", linewidths=0.3)
    for m, mk in (("mi", "o"), ("trace", "s"), ("facloc", "D")):
        if f"{m}_selected" not in arr:
            continue
        sel = first_picks(arr, m)
        j = Counter(sel[sel >= 0].tolist()).most_common(1)[0][0]
        ax.scatter([x[j]], [y[j]], s=70, facecolors="none",
                   edgecolors=METHOD_COLOR[m], linewidths=1.6, marker=mk,
                   label=f"{m} pick")
    ax.set_xlabel(r"MI score $-\frac{1}{2}\log(1-R^2_{j|\mathrm{rest}})$")
    ax.set_ylabel("realized $R^2$ of $\\{j\\}$")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(f"{display_name(ds_name)}: k = 1 singletons", fontsize=9)
    return sc


def _panel_surrogates(ax, tables):
    keys = ["trace", "entropy", "mi", "hub", "facloc", "exp_oracle"]
    labels = ["trace", "entropy", "MI", "hub", "FL", "exp. (oracle)"]
    ks = list(tables["surrogate"].keys())
    shades = ["#9ECAE1", "#4292C6", "#08519C"][:len(ks)]
    w = 0.8 / max(len(ks), 1)
    for ik, k in enumerate(ks):
        vals = [tables["surrogate"][k].get(key, {}).get("rho_mean")
                for key in keys]
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(np.arange(len(keys)) + (ik - (len(ks) - 1) / 2) * w, vals,
               w * 0.92, color=shades[ik], label=f"k = {k}")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Spearman with realized $R^2$")
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("in-sample objective vs test $R^2$", fontsize=9)


def _panel_predicted(ax, arr, k):
    c = k - 1
    for m in _plot_methods(arr, PLOT_METHODS):
        ax.scatter(arr[f"{m}_pop_r2_tr"][:, c], arr[f"{m}_r2"][:, c], s=12,
                   color=METHOD_COLOR[m], alpha=0.7, label=m,
                   marker=PLOT_MARKER[m], linewidths=0.8)
    lims = [np.nanmin([ax.get_xlim()[0], ax.get_ylim()[0]]),
            np.nanmax([ax.get_xlim()[1], ax.get_ylim()[1]])]
    ax.plot(lims, lims, color="#333333", linewidth=0.8, linestyle=":")
    ax.set_xlabel(f"in-sample $R^2_{{pop}}$ (train), k = {k}")
    ax.set_ylabel("realized $R^2$")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)


def _panel_redundancy(ax, arr, ks):
    for m in _plot_methods(arr, PLOT_METHODS):
        xs = [_nanmean(arr[f"{m}_mean_abs_r_or"][:, k - 1]) for k in ks]
        ys = [_nanmean(arr[f"{m}_r2"][:, k - 1]) for k in ks]
        ax.plot(xs, ys, "-", color=METHOD_COLOR[m], linewidth=1.0)
        for x, y, fill in zip(xs, ys, [METHOD_COLOR[m], "none", "none"]):
            ax.plot([x], [y], PLOT_MARKER[m], color=METHOD_COLOR[m],
                    markerfacecolor=fill, markersize=6)
        ax.annotate(m, (xs[-1], ys[-1]), fontsize=6, xytext=(3, 0),
                    textcoords="offset points")
    ax.set_xlabel("mean $|r|$ within the selected set (oracle $R$)")
    ax.set_ylabel("realized $R^2$")
    ax.set_title("redundancy vs $R^2$ (" + ", ".join(
        f"k={k} {'filled' if i == 0 else 'open'}"
        for i, k in enumerate(ks)) + ")", fontsize=9)
    ax.grid(True, alpha=0.3)


def _panel_oracle(ax, arr, title=True, ds_name=""):
    k_max = int(arr["k_max"])
    ks = np.arange(1, k_max + 1)
    for m in _plot_methods(arr, CURVE_METHODS):
        ax.plot(ks, _nanmean(arr[f"{m}_r2"], axis=0), PLOT_MARKER[m] + "-",
                color=METHOD_COLOR[m], markersize=3, linewidth=1.2,
                label=f"{m} (train $R$)")
        ax.plot(ks, _nanmean(arr[f"{m}_osel_r2"], axis=0), ":",
                color=METHOD_COLOR[m], linewidth=1.2,
                label=f"{m} (oracle selection)")
    ax.set_xlabel("selected benchmarks $k$")
    ax.set_ylabel("realized $R^2$")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(f"{display_name(ds_name)}: train vs oracle selection",
                     fontsize=9)


def _panel_coverage(ax, arr):
    k_max = int(arr["k_max"])
    ks = np.arange(1, k_max + 1)
    for m in _plot_methods(arr, PLOT_METHODS):
        ax.plot(ks, _nanmean(arr[f"{m}_cov_cells"], axis=0),
                PLOT_MARKER[m] + "-",
                color=METHOD_COLOR[m], markersize=3, linewidth=1.0, label=m)
    ax.set_xlabel("selected benchmarks $k$")
    ax.set_ylabel("share of test cells scored")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_dataset(ds_name, arr, tables, tag=None):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    sc = _panel_singletons(axes[0, 0], arr, ds_name=ds_name)
    fig.colorbar(sc, ax=axes[0, 0], label="trace score $R^2_{pop}(\\{j\\})$")
    _panel_surrogates(axes[0, 1], tables)
    _panel_predicted(axes[0, 2], arr, min(5, int(arr["k_max"])))
    _panel_redundancy(axes[1, 0], arr,
                      [k for k in (5, 10) if k <= int(arr["k_max"])]
                      or [int(arr["k_max"])])
    _panel_oracle(axes[1, 1], arr, ds_name=ds_name)
    _panel_coverage(axes[1, 2], arr)
    plt.tight_layout()
    path = OUT / f"analyze_objectives_{tag or ds_name}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)


def plot_facloc(ds_name, arr, tag=None):
    """OUT/analyze_objectives_facloc_{tag}.pdf: facility location vs trace
    (module docstring): (a) realized R^2 facloc - trace by k with the
    Nadeau-Bengio 95% interval, (b) overlap of the two selections, (c)
    gain landscape: Spearman of each objective's gain (color) with the
    realized gain over the candidates, at the states of each greedy path
    (line style).  Nothing without the fl_* arrays."""
    if "fl_jacc_trace" not in arr:
        return
    k_max = int(arr["k_max"])
    ks = np.arange(1, k_max + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    st = [paired_stats(arr["facloc_r2"][:, k - 1], arr["trace_r2"][:, k - 1],
                       arr["n_train"], arr["n_val"], arr["rep"]) for k in ks]
    mean = np.array([s["mean_diff"] for s in st], dtype=float)
    lo = np.array([s["ci_low"] for s in st], dtype=float)
    hi = np.array([s["ci_high"] for s in st], dtype=float)
    ax.fill_between(ks, lo, hi, color=METHOD_COLOR["facloc"], alpha=0.2,
                    linewidth=0)
    ax.plot(ks, mean, PLOT_MARKER["facloc"] + "-",
            color=METHOD_COLOR["facloc"], markersize=4, linewidth=1.2)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("selected benchmarks $k$")
    ax.set_ylabel("realized $R^2$: facility location $-$ trace")
    ax.set_title(f"{display_name(ds_name)}: paired difference (95% CI)",
                 fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ks, _nanmean(arr["fl_shared_trace"] / ks[None, :], axis=0),
            "o-", color="#4477AA", markersize=3, linewidth=1.2,
            label="shared picks / $k$")
    ax.plot(ks, _nanmean(arr["fl_jacc_trace"], axis=0), "s--",
            color="#BBBBBB", markersize=3, linewidth=1.2, label="Jaccard")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("selected benchmarks $k$")
    ax.set_ylabel("overlap of the facloc and trace sets")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    land_k = int(arr["fl_land_k"]) if "fl_land_k" in arr else 0
    picks = np.arange(1, land_k + 1)
    for P in FL_PATHS:
        for O in FL_GAINS:
            if not land_k:
                continue
            ax.plot(picks, _nanmean(arr[f"fl_land_{P}_rho_{O}"], axis=0),
                    PLOT_MARKER[O] + FL_PATH_STYLE[P],
                    color=METHOD_COLOR[O], markersize=3, linewidth=1.2,
                    label=f"{O} gain, {P} states")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("pick $t$ (state $A_{t-1}$ of the greedy path)")
    ax.set_ylabel("Spearman: gain vs realized $R^2$ gain")
    if land_k:
        ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT / f"analyze_objectives_facloc_{tag or ds_name}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)


def plot_all(all_arr, suffix=""):
    n = len(all_arr)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.5), squeeze=False)
    for j, (ds_name, arr) in enumerate(all_arr.items()):
        _panel_singletons(axes[0, j], arr, ds_name=ds_name)
        _panel_oracle(axes[1, j], arr, ds_name=ds_name)
    plt.tight_layout()
    path = OUT / f"analyze_objectives_all{suffix}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)


# ── Consistency with the baseline logs ───────────────────────────

def manifest_data_sha256(log_dir, ds_name):
    """sha256 of data/{ds}.matrix.npz as recorded in log_dir/../manifest.json
    (run_all.sh's "inputs"), or None if there is no such record."""
    path = pathlib.Path(log_dir).resolve().parent / "manifest.json"
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            inputs = json.load(f).get("inputs") or {}
    except (OSError, ValueError):
        return None
    name = f"{ds_name}.matrix.npz"
    for key, rec in inputs.items():
        if pathlib.PurePath(key).name == name and isinstance(rec, dict):
            return rec.get("sha256")
    return None


def check_logs(ds_name, arr, log_dir):
    """Compare train-side selections and R^2 with an eval_baselines.py log
    (same splits, estimator and selectors): returns a dict of max
    differences (None if the log is missing or has other splits), with
    "_data" recording whether the log's run manifest names the same input
    data (sha256; None if unknown).  A log of other input data is
    compared anyway but flagged: agreement with it is no check."""
    path = pathlib.Path(log_dir) / f"baselines_{ds_name}.npz"
    if not path.is_file():
        print(f"  check: {path} not found; skipped")
        return None
    d = np.load(path, allow_pickle=True)
    if not np.array_equal(d["split_id"], arr["split_id"]):
        print(f"  check: {path} has other splits; skipped")
        return None
    ours = str(arr["data_sha256"]) if "data_sha256" in arr else None
    theirs = manifest_data_sha256(log_dir, ds_name)
    same_data = (None if ours is None or theirs is None
                 else bool(ours == theirs))
    if same_data is False:
        print(f"WARNING: {path} was made from other input data "
              f"({theirs[:16]}… vs {ours[:16]}…): the comparison below is "
              f"no check", file=sys.stderr, flush=True)
    elif same_data is None:
        print(f"  check: no manifest record of the input data of {path}; "
              f"data identity unchecked")
    k = min(int(d["k_max"]), int(arr["k_max"]))
    out = {"_data": {"ours_sha256": ours, "log_sha256": theirs,
                     "same_data": same_data}}
    for m in methods_in(arr):
        if f"{m}_selected" not in d.files or f"{m}_r2" not in d.files:
            print(f"  check: {path} has no {m}; skipped for {m}")
            continue
        same = np.array_equal(d[f"{m}_selected"][:, :k],
                              arr[f"{m}_selected"][:, :k])
        diff = np.nanmax(np.abs(d[f"{m}_r2"][:, :k] - arr[f"{m}_r2"][:, :k]))
        out[m] = {"same_selected": bool(same), "max_abs_r2_diff": _num(diff)}
    print("  check vs " + str(path) + ": " + ", ".join(
        f"{m} {'=' if v['same_selected'] else '≠'}sel Δ{v['max_abs_r2_diff']:.1e}"
        for m, v in out.items() if not m.startswith("_")))
    return out


def load_log(ds_name, oracle="matched"):
    """(arr, config) of an earlier run: LOGS/analyze_objectives_{tag}.npz
    and the config of its JSON (runtime_s of the original run kept)."""
    tag = out_tag(ds_name, oracle)
    with np.load(LOGS / f"analyze_objectives_{tag}.npz",
                 allow_pickle=True) as d:
        arr = {key: d[key] for key in d.files}
    path = LOGS / f"analyze_objectives_{tag}.json"
    config = {}
    if path.is_file():
        with open(path) as f:
            config = json.load(f).get("config", {})
    config["from_logs"] = True
    return arr, config


def save(ds_name, arr, tables, macros, paired, config, check, tag=None):
    tag = tag or ds_name
    np.savez(LOGS / f"analyze_objectives_{tag}.npz", **arr)
    write_csv(paired, LOGS / f"analyze_objectives_{tag}_paired.csv",
              PAIRED_FIELDS)
    doc = {"script": "analyze_objectives.py", "dataset": ds_name,
           "config": config, "tables": tables, "check_logs": check,
           "macros": macros}
    path = LOGS / f"analyze_objectives_{tag}.json"
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, allow_nan=False)
    print(f"Results saved to {LOGS}/analyze_objectives_{tag}.{{npz,json}}"
          f" and _paired.csv ({len(macros)} macros)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mechanism analysis of the selection objectives on the "
                    "matched repeated-CV splits.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names, read from "
                             "data/<name>.matrix.npz (default: %(default)s)")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions (default %(default)s)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--kmax", type=int, default=K_MAX)
    parser.add_argument("--n-random-sets", type=int, default=N_RANDOM_SETS,
                        help="random eligible k-sets per split and k "
                             "(default %(default)s)")
    parser.add_argument("--random-ks", type=int, nargs="+",
                        default=RANDOM_SET_KS,
                        help="k of the random sets (default %(default)s)")
    parser.add_argument("--brute-k", type=int, default=BRUTE_K,
                        help="exact maximizers for k <= this "
                             "(default %(default)s)")
    parser.add_argument("--gamma-k", type=int, default=GAMMA_K,
                        help="size k of the exact submodularity ratio "
                             "gamma_{U,k} of trace (U = greedy trace's first "
                             "k picks; 0 or 1 skips it; cost grows as "
                             "N^k; default %(default)s)")
    parser.add_argument("--fl-landscape-k", type=int, default=FL_LAND_K,
                        help="facility-location section: gain landscape "
                             "over all candidates before picks 1..this of "
                             "the facloc and trace paths (one imputation "
                             "per candidate and state after the first; 0 "
                             "skips it; default %(default)s)")
    parser.add_argument("--check-logs", default=None,
                        help="directory with baselines_<ds>.npz to compare "
                             "the train-side selections and R² with")
    parser.add_argument("--from-logs", action="store_true",
                        help="recompute the summaries, paired CSVs and "
                             "figures from existing LOGS/"
                             "analyze_objectives_<ds>.npz (no selection runs;"
                             " the run options are taken from the logs)")
    parser.add_argument("--oracle", choices=["matched", "full"],
                        default="matched",
                        help="oracle correlation of all models: 'matched' "
                             "(estimator size rules of the training splits; "
                             "default) or 'full' (estimate_corr(B, O) as is)"
                             " — see estimate_corr_oracle")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes evaluating the splits (spawn, "
                             "single-threaded each; the results do not "
                             "depend on it; default %(default)s)")
    args = parser.parse_args()
    if args.folds < 2 or args.repeats < 1 or args.kmax < 1:
        parser.error("need --folds >= 2, --repeats >= 1 and --kmax >= 1")
    if args.n_random_sets < 3 or args.brute_k < 1 or args.gamma_k < 0:
        parser.error("need --n-random-sets >= 3, --brute-k >= 1 and "
                     "--gamma-k >= 0")
    if args.fl_landscape_k < 0:
        parser.error("need --fl-landscape-k >= 0")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    missing = [ds for ds in args.datasets
               if not (DATA / f"{ds}.matrix.npz").is_file()]
    if missing:
        parser.error(f"no {DATA}/<name>.matrix.npz for {missing}")

    config = {"repeats": args.repeats, "folds": args.folds,
              "kmax": args.kmax, "holdout_frac": HOLDOUT_FRAC, "seed": SEED,
              **protocol_config(), "ridge": RIDGE,
              "n_random_sets": args.n_random_sets,
              "random_ks": args.random_ks, "brute_k": args.brute_k,
              "gamma_k": args.gamma_k, "eps_k": EPS_K,
              "large_diff": LARGE_DIFF,
              "random_sets_seed_offset": RANDOM_SETS_SEED_OFFSET,
              "spectral_share": SPECTRAL_SHARE, "close_r": CLOSE_R,
              "oracle": args.oracle, "methods": METHODS,
              "fl_landscape_k": args.fl_landscape_k,
              "workers": args.workers}
    total_t0 = time.time()
    all_arr = {}
    for ds_name in args.datasets:
        ds_t0 = time.time()
        tag = out_tag(ds_name, args.oracle)
        if args.from_logs:
            arr, cfg = load_log(ds_name, args.oracle)
        else:
            arr = run_dataset(ds_name, args.kmax, args.folds, args.repeats,
                              args.n_random_sets, args.random_ks,
                              args.brute_k, args.oracle, args.gamma_k,
                              args.fl_landscape_k, args.workers)
            cfg = dict(config, runtime_s=round(time.time() - ds_t0, 1),
                       data={"file": f"{ds_name}.matrix.npz",
                             "sha256": str(arr["data_sha256"]),
                             "sha256sums": str(arr["data_sha256sums"])})
            # per-split arrays first, so a failing summary loses nothing
            # (rerun the rest with --from-logs)
            np.savez(LOGS / f"analyze_objectives_{tag}.npz", **arr)
        paired = paired_rows(ds_name, arr, sorted(set(KS_REPORT + [3, 15])))
        tables, macros = summarize(ds_name, arr, paired)
        print_summary(ds_name, tables)
        check = (check_logs(ds_name, arr, args.check_logs)
                 if args.check_logs else None)
        plot_dataset(ds_name, arr, tables, tag)
        plot_facloc(ds_name, arr, tag)
        save(ds_name, arr, tables, macros, paired, cfg, check, tag)
        all_arr[ds_name] = arr
        print(f"{ds_name.upper()} completed in {time.time() - ds_t0:.0f}s",
              flush=True)
    if len(all_arr) > 1:
        plot_all(all_arr, "" if args.oracle == "matched" else "_fulloracle")
    print(f"\nAll datasets completed in {time.time() - total_t0:.0f}s",
          flush=True)
