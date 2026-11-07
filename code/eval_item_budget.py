r"""Whole benchmarks vs random items under a matched question budget.

Question.  A practitioner can afford Q questions per new model.  Is it
better to run a few WHOLE benchmarks (tasks) chosen by a selector and
impute the others from the training models' correlations, or to run Q
uniformly random ITEMS spread over all tasks?  Data: olv1items, the Open
LLM Leaderboard v1 item-level correctness of tinyBenchmarks' lb.pickle:
395 models x 62 tasks (57 MMLU subjects, ARC-Challenge, HellaSwag,
TruthfulQA mc2, Winogrande, GSM8K) with all 28,659 item scores in
data/olv1items.items.npz (decoded as meta.items.decode says; the mean of a
task's item scores reproduces its matrix entry exactly, which load_items
checks).  Task sizes range from 100 items (ten MMLU subjects) to 10,042
(HellaSwag).

Scope (user decision).  The item-level baseline is uniform random items,
with the budget matched on the total number of questions.  The anchor
points and IRT estimators of tinyBenchmarks (Maia Polo et al., 2024) are
deliberately not reimplemented; items_center (and its uncentered variant
items_gauss) is the Gaussian analogue built from the same train
correlation as the benchmark selectors, with the training models' answers
on the sampled items as the only item-level information.  The budget
counts questions although questions have unequal costs: GSM8K is
generative (a sampled chain of thought per item, scored by exact match),
the other tasks score log-likelihoods of 2-5 fixed choices (TruthfulQA
mc2: of all its reference answers), and prompts differ (25-shot ARC,
10-shot HellaSwag, 5-shot MMLU / Winogrande / GSM8K, 0-shot TruthfulQA);
in tokens or GPU time a GSM8K item costs far more than an MMLU item.

Matched design.  The splits are cv_splits(M, folds, repeats, 0.1, SEED)
over the models (default 5 x 10 = 50 splits; repetition 0 = the seed-42
folds).  On every split all methods and budgets share
  * the training / validation models and the train estimates of the 62
    task scores' means, standard deviations and correlation matrix R
    (eval_entropy_vs_mi.estimate_corr, the paper's estimator; olv1items
    is complete with more training models than tasks, so it takes the
    pairwise branch, i.e. the sample correlation),
  * one eligibility mask, eligible_mask(O_train) (tasks with at least
    MIN_TRAIN_OBS training models; on olv1items every task), which
    restricts the whole-task selection; item sampling ignores it,
  * common random numbers: bench_random's task order is random_select's
    permutation seeded with Split.seed(SEED + RANDOM_SEED_OFFSET), the
    random baseline of every other script on that split; the item sample
    is the first Q entries of ONE permutation of all items per split,
    rng = default_rng(Split.seed(SEED + ITEM_SEED_OFFSET)),
    ITEM_SEED_OFFSET = 18, so the samples are nested across budgets and
    shared by items_center, items_gauss, items_only and all validation
    models of the split.  Tag 18 is distinct from every other tag in the
    code tree: 7 (random) and 11 (entropy_rf, retired) of the CV scripts,
    13-16 of eval_prospective.py and 19 of analyze_objectives.py
    (test_eval_item_budget.py scans code/*.py for collisions).
The budgets Q (questions per model, default 50, 100, ..., 12,800) are the
same on every split.

Methods.  Task costs c_t = number of items of task t.
  bench_mi       eval_cost_budget.budgeted_select(R, c, Q, "mi"): cost-
                 effective greedy on MI gains per question + best-singleton
                 fallback (gains clipped at 0, so it may stop before the
                 budget is spent)
  bench_entropy  the same with the shifted entropy gains (D_FLOOR) of
                 Appendix B.3
  bench_trace    budgeted_select(R, c, Q, "trace"): the same rule for the
                 trace objective of baseline_select.greedy_trace_select
                 (variance of all unselected tasks explained by j, per
                 question) + best singleton under that objective
  bench_facloc   budgeted_select(R, c, Q, "facloc"): the same rule for
                 facility location on the clipped correlation matrix (the
                 headline method, user decision 2026-10-19; S as
                 baseline_select.facility_location_select builds it, gain
                 sum_t max(s_jt − c_t, 0) over all tasks given the cover
                 c_t, per question) + best singleton F({j}); monotone
                 submodular, so the (1 − 1/e)/2 knapsack guarantee of
                 budgeted_select applies.  Inserted before bench_random
                 (pairs read bench_facloc − bench_random); it draws no
                 random numbers, so no other method's rows change
  bench_random   the split's random_select order of the eligible tasks
                 (common random numbers above), each added while it fits
                 (eval_cost_budget.order_fill, skip-and-continue)
  items_center   the Q sampled items, centered on the TRAINING models'
                 answers to the same items (a difference estimator):
                     y_t = (ybar_t - abar_t) / sd_t,
                 ybar_t the validation model's mean over its n_t sampled
                 items of task t, abar_t the training models' mean over
                 the same items, read as a noisy observation y_t = z_t +
                 e_t of the standardized task score z_t = (s_t - mu_t) /
                 sd_t (mu_t = the training models' mean of the whole
                 task), with Var e_t = u_t / n_t * (N_t - n_t) / (N_t - 1)
                 / sd_t^2 (u_t below), and the posterior mean of ALL 62 z
                 given y:
                     z_hat = R[:, o] (R_oo + D + r I)^{-1} y_o,
                 o the sampled tasks, D = diag(Var e_o), which also
                 denoises the sampled tasks themselves
  items_gauss    the same sample and posterior without the centering:
                 y_t = (ybar_t - mu_t) / sd_t, Var e_t = v_t / n_t * (N_t -
                 n_t) / (N_t - 1) / sd_t^2 (sample mean of n_t of the N_t
                 items drawn without replacement, finite-population
                 correction; v_t below)
  items_only     the same sample; a task with n_t >= 1 sampled items is
                 estimated by the mean of its sampled items, a task
                 without any by the train mean
Ties and fallbacks of the bench methods are budgeted_select's (gain/cost
ratios within a relative TIE_RTOL tie and go to explained variance per
question, then the lowest index; the entropy singleton must beat the
greedy set by TIE_RTOL; MI keeps the strict comparison).  For the trace
objective the gain IS the explained variance, so the explained-variance
tie-break re-ties the tied ratios and the lowest index decides
(tie_breaks == index_ties, as in greedy_trace_select), and the singleton
replaces the greedy set only if its explained variance exceeds the greedy
set's by more than TIE_RTOL * max(s, value), s = max_i R_ii.  The
facility-location singleton (near-ties within TIE_RTOL * max(|best|, 1),
then explained variance, then the lowest index) must beat the greedy set
by TIE_RTOL * max(1, value).  (The first
version of this script had its own copy of the trace knapsack;
test_eval_item_budget.py keeps it as an oracle that budgeted_select's
trace objective must reproduce.)

Noise model of items_center.  For a fixed sample, the error of the plain
sample mean ybar_t - s_t has a part that is the same for every model: the
sampled items are easier or harder than the task's average item.  This
part is about a third of the squared error of items_gauss on olv1items
(the log keeps it as {m}_noise_shared), and the training models' answers
to the same items estimate it: write x_mi = s_mt + (xbar_i - mu_t) + r_mi
with xbar_i the training models' mean of item i.  The residuals r_mi =
x_mi - xbar_i - s_mt + mu_t of a model sum to 0 over the task's items, so
ybar_t - abar_t - (s_t - mu_t) is the mean of n_t of the model's N_t
residuals drawn without replacement, with variance sigma_m^2 / n_t * (N_t
- n_t) / (N_t - 1), sigma_m^2 = mean_i r_mi^2.  u_t is its average over
the training models (double-centered residual variance, = mean_i Var_m
x_mi - Var_m s_mt, ddof 0), one formula for binary and continuous
(TruthfulQA mc2) items.  It is an in-sample estimate (xbar_i is fitted to
the same models), smaller than a held-out model's by a factor of about
1 - 2 / n_train (0.6% here), and it ignores that sigma_m^2 varies with
the model's accuracy; the log keeps the calibration ratio
{m}_noise_calib = sum e^2 / sum Var e over the validation models and
sampled tasks (1 = calibrated).  Only training data enter u, abar and
D, so z_hat is again one linear map per split.

Noise model of items_gauss.  v_t is the per-item variance of task t:
p_t (1 - p_t) for binary tasks with p_t = the TRAIN mean of task t, and
for TruthfulQA mc2 (continuous items in [0, 1]) the average over the
training models of the within-model variance (ddof 0) of its item scores.
The train mean, not the test model's own sample mean, because (i) at the
small n_t of most budgets (about Q / 287 items per 100-item MMLU subject)
the own sample mean is often exactly 0 or 1, which would make a single
question an exact, zero-variance observation that the posterior copies
into the task and its correlated neighbours, and (ii) with v_t from
training data only, D depends on the sampling design (n_t, N_t) and not
on the test model's answers, so z_hat is one linear map per split, shared
by all validation models (an exact Gaussian conditioning, the same form as
the benchmark imputation) and the test responses enter only through y.
For binary tasks the training models' average within-model variance would
be p_t (1 - p_t) - Var_m(p_mt), smaller by the between-model variance, so
the binomial noise is slightly conservative; the log keeps both
(noise_unit_var, noise_within_var).  r = IMPUTE_RIDGE = 0.01 is the ridge
of eval_entropy_vs_mi.impute_and_evaluate, so the benchmark imputation is
the same formula with D = 0 on the selected tasks.

Metric.  Pooled R^2 = 1 - SS_res / SS_tot over the validation models x
ALL 62 tasks (the targets of the split, cv_split.target_mask, rule D1;
on olv1items every task; every prediction clipped to the task's valid
range [0, 1], score_bounds.py, rule D2) in train-standardized units (z =
(s - mu) / sd clipped to
+-10, as impute_and_evaluate standardizes; SS_tot = sum z^2, so the train
mean scores 0).  Whole tasks a bench method selected count with their
exact values (zero error); sampled tasks count with their estimates.
r2_unobs restricts the pooled R^2 to the tasks the method did not observe
exactly: the unselected tasks of a bench method (equal to
impute_and_evaluate's R^2 of that selection) and, for the item methods,
the tasks not sampled completely (n_t < N_t: in practice all 62).  The
task set (and the denominator) of r2_unobs therefore depends on the
method; only r2 compares every method on the same cells.  Below the
cheapest task (Q < 100) no whole task fits: the bench methods predict the
train mean and score R^2 = 0 at cost 0 (logged as 0, not NaN as in
eval_cost_budget.py, because the train mean is a valid prediction of all
62 tasks; r2_unobs is then the same 0).

Paired statistics.  paired_stats.paired_stats (Nadeau-Bengio corrected
t-interval, win/tie/loss over the matched splits) of R^2 a - b, for every
budget and pair of METHODS (a before b in METHODS order: bench_mi,
bench_entropy, bench_trace, bench_facloc, bench_random, items_center,
items_gauss, items_only).  r2 rows are written for every pair.  r2_unobs
rows only for
pairs of the same family (paired_comparable): two bench methods (each
over its own unselected tasks, the convention of the paper's R^2) or two
item methods (the same tasks, as they share the item sample).  A bench
and an item method would compare R^2 over different task sets with
different denominators, so those rows are not written.  KEY_PAIRS
(bench_mi - items_gauss, bench_mi - items_only, items_gauss - items_only,
bench_trace - items_gauss, bench_mi - bench_random, and against the
strongest item baseline bench_mi - items_center, bench_entropy -
items_center, bench_trace - items_center, items_center - items_gauss;
then the facility-location pairs bench_facloc - items_gauss, bench_facloc
- bench_random, bench_facloc - items_center, bench_mi - bench_facloc,
bench_trace - bench_facloc) are printed and get macros, of r2 only.

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir()):
  LOGS/item_budget_{ds}.npz
    budgets                         (n_budgets,) int, Q
    costs, benchmark_names, task_binary   (N,) items per task, names, bool
    methods                         (n_methods,)
    {m}_r2, {m}_r2_unobs, {m}_rmse  (n_splits, n_budgets) pooled R^2 over
                                    all tasks / over the tasks m did not
                                    observe exactly / RMSE over all cells
    {m}_n_exact                     (n_splits, n_budgets) int, tasks
                                    observed exactly (selected; items: all
                                    items sampled)
    {m}_n_obs                       (n_splits, n_budgets) int, tasks with
                                    any observation (selected; items: >= 1
                                    sampled item)
    {m}_cost                        (n_splits, n_budgets) questions used
                                    (bench: sum of selected task sizes;
                                    items: Q)
    bench_*_selected                (n_splits, n_budgets, N) int, picks in
                                    selection order, -1 padded
    bench_*_index_ties, _tie_breaks (n_splits, n_budgets) int (bench_random:
                                    0)
    bench_{mi,entropy,trace,facloc}_singleton   (n_splits, n_budgets)
                                    bool, the best-singleton fallback won
    items_n_sampled                 (n_splits, n_budgets, N) int, sampled
                                    items per task (all item methods)
    noise_unit_var, noise_within_var    (n_splits, N) v_t used by
                                    items_gauss / training models' average
                                    within-model item variance
    noise_center_var                (n_splits, N) u_t used by items_center
    {m}_noise_calib, {m}_noise_shared   (n_splits, n_budgets), m in
                                    NOISE_METHODS (items_center,
                                    items_gauss): sum e^2 / sum Var e
                                    (e = y_t - z_t over the validation
                                    models and the tasks with 0 < n_t <
                                    N_t) / the part n_val sum_t ebar_t^2 /
                                    sum e^2 of the squared error common to
                                    all validation models (ebar_t: mean of
                                    e_t over them)
    n_clipped                       (n_splits,) int, validation cells whose
                                    standardized score was clipped at +-10
    rep, fold, split_id, n_train, n_val, n_eligible   (n_splits,) int
    eligible                        (n_splits, N) bool
    estimation_failed               (n_splits,) bool (rows NaN)
    holdout_frac, n_folds, n_repeats, min_train_obs, min_target_obs,
    pred_clip, seed,
    item_seed_offset, random_seed_offset, impute_ridge, z_clip,
    n_items_total, n_models         scalars
  LOGS/item_budget_{ds}.json        config, summary tables (means / stds
                                    per method, metric and budget; paired
                                    rows of KEY_PAIRS; most frequently
                                    selected tasks), runtime and a flat
                                    "macros" dict (below)
  LOGS/item_budget_{ds}_paired.csv  every pair and budget for r2, the
                                    same-family pairs for r2_unobs
                                    (PAIRED_FIELDS: dataset, holdout_frac,
                                    budget, metric, method_a, method_b,
                                    then paired_stats.STAT_FIELDS)
  OUT/item_budget_{ds}.pdf          mean R^2 (+- std over splits) vs Q (log
                                    axis): all tasks / tasks not observed
                                    exactly

Macros (make_numbers.py style: letters only, numbers spelled with
make_numbers.num_words, identifiers with make_numbers.camel; {Ds} = camel
of the dataset, olv1items -> OlvOneItems; {M} = camel of the method,
BenchMi, BenchEntropy, BenchTrace, BenchFacloc, BenchRandom, ItemsCenter,
ItemsGauss, ItemsOnly; {q} = num_words(Q): Qfifty, Qonehundred, ...,
Qtwelvethousandeighthundred):
  ItemBudgetRsq{Ds}{M}Q{q}, ItemBudgetRsqStd...   R^2 split mean / std
  ItemBudgetRsqUnobs{Ds}{M}Q{q}       r2_unobs split mean
  ItemBudgetSize{Ds}{M}Q{q}           mean n_obs (tasks selected / sampled)
  ItemBudgetCost{Ds}{M}Q{q}           mean questions used (bench methods)
  ItemBudgetNoiseCalib{Ds}{M}Q{q}, ItemBudgetNoiseShared...   split means
                                      of {m}_noise_calib / _noise_shared
                                      (NOISE_METHODS)
  ItemBudgetPair{S}{Ds}{A}Vs{B}Q{q}   KEY_PAIRS (r2), S in Diff, Se, Lo, Hi,
                                      Win, Wins, N (make_numbers.PAIR_STATS)
  ItemBudgetSigPos{Ds}{A}Vs{B}, ItemBudgetSigNeg...   number of budgets
                                      whose 95% interval lies above / below 0
  ItemBudget{NSplits,NReps,NModels,NTasks,NItems,NBudgets}{Ds},
  ItemBudgetMinTaskItems{Ds}, ItemBudgetMaxTaskItems{Ds}, ItemBudgetNClipped{Ds}

Fixed per-task overhead (opt-in, --fixed-cost C0 ...; without the flag
nothing above changes).  Integrating and running a benchmark costs more
than its questions: harness setup, prompt format, scoring code, one
evaluation job per task.  With an overhead of C0 item-equivalents per
task (one level per C0, default 0 10 50 100 200; C0 = 0 is the run above,
bit for bit: test_eval_item_budget.py), the budget Q (now in
item-equivalents) pays
  whole tasks   sum over the selected tasks of (N_t + C0): the bench
                methods' costs become c_t = N_t + C0 (budgeted_select's
                gain / cost ratio and bench_random's order_fill see C0);
  random items  one per sampled item plus C0 for every task with at least
                one sampled item.  The sample walks the split's item
                permutation (the same common random numbers,
                Split.seed(SEED + ITEM_SEED_OFFSET)) skip-and-continue,
                like order_fill: an item of a task already touched costs 1,
                the first item of a new task 1 + C0, and each is bought
                while it fits (fixed_item_sample).  Once a new task no
                longer fits no later one does, so the sample is the
                longest affordable prefix of the permutation plus the next
                items of the tasks it touched while budget remains.  Within
                every task the sampled items are the first n_t of that
                task's items in permutation order, a simple random sample
                without replacement given n_t, so the finite-population
                noise models of items_center / items_gauss stay exact; the
                sampling design (n_t) still depends on the permutation
                only, never on scores.  At C0 = 0 it is the first Q items.
Every method's true cost (questions + C0 x tasks touched) is logged and
held to Q ({m}_cost <= Q; violations, expected all False).  Human
evaluation (annotators, review of open-ended outputs) would add a further
per-task burden for which there are no data; it is not modelled.  Default
budget grid of the fixed runs (fixed_budget_grid): FIXED_BUDGETS, 50 x
2^(k/2) rounded, k = 0 .. 18 (50 to 25,600; the plain grid at half-octave
steps), plus the cheapest whole task min_t N_t + C0 of every level (on
olv1items 100, 110, 150, 200, 300 for the default levels; 100 and 200 are
grid points already), so that the crossover is resolved to a factor
sqrt(2) and exactly at the budget where the whole-task methods start
buying.  An explicit --budgets grid is used as given.

The uniform item baseline under overhead (interpretation).  As specified
(user decision: uniform random items, budget matched on questions), the
item sample ignores the overhead: it walks a uniform permutation of all
items and so touches a new task with almost every early item, paying C0
each time.  Under overhead most of its budget therefore goes to overhead
(OLL v1, mean over the splits: C0 = 10, Q = 141: 11.6 tasks touched for
25 questions, 83% overhead; C0 = 200, Q = 12,800: 60 tasks, 756
questions, 94%; logged as the overhead_share summary and the
OverheadShare / Questions macros).  Because the walk buys a new task
whenever 1 + C0 still fits, the number of questions it buys, and with it
the item methods' R², is not monotone in Q (C0 = 100: 100 questions at
Q = 800, 33 at Q = 1,131; items_center R² 0.582 and 0.490), and the
samples of different budgets need not be nested as they are at C0 = 0.
The crossover budgets below say where whole tasks beat THIS item design;
an overhead-aware item design (e.g. two-stage sampling: a few random
tasks, then random items within them) is not run, and the results say
nothing about it.  "Whole tasks win as soon as the cheapest task fits"
is a property of the uniform item baseline, to be worded that way.

Crossover (crossover()): for a whole-task method A and an item method B
at overhead C0, d(Q) = mean paired R² difference A − B over all tasks
(r2: one target set for every method) on the budget grid.  d is not
interpolated between grid points: A's R² is a step function of Q (its
selection changes only where a further task, or set of tasks, becomes
affordable, and it is exactly 0 below the cheapest whole task min_t N_t
+ C0), so a crossing between two grid points can be anywhere between
them (the first version interpolated log-linearly and reported crossings
below the cheapest task, where A buys nothing).
  state       -1 A is ahead already at the smallest Q, 0 A overtakes B
              inside the grid, 1 A is never ahead
  q_first     the first grid Q with d > 0 (NaN if state 1): the closed
              upper end of the grid bracket of the first crossing
  q_first_lo  the grid Q before it (d <= 0 there; NaN unless state 0):
              the first crossing lies in (q_first_lo, q_first].  When the
              bracket contains the cheapest whole task, A scores exactly 0
              below that cost, so A can lead there only where B's R² is
              negative; with the cheapest task on the grid (the default
              grid) q_first is then that cost unless B's R² is < 0 at a
              grid budget below it
  q_stay      the smallest grid Q from which d > 0 at every larger grid Q
              (A stays ahead; NaN if A is not ahead at the largest Q)
  q_sig       the smallest grid Q whose 95% Nadeau-Bengio interval of d
              lies above 0; q_sig_stay: from which every larger grid Q is
              significant
  n_ahead, n_sig   grid budgets with d > 0 / significantly > 0
Outputs (instead of the item_budget_* files):
  LOGS/item_fixed_{ds}.npz   (savez_compressed) the per-split arrays of the
                             plain run with a level axis after the split
                             axis, (n_splits, n_levels, n_budgets[, N]),
                             plus {m}_questions (questions used; cost =
                             questions + C0 x tasks touched); the
                             level-independent eligible, n_eligible,
                             noise_*_var, n_clipped (n_splits[, N]);
                             fixed_costs, total_cost, min_task_cost
                             (n_levels,; min_task_cost = min_t N_t + C0),
                             budgets (n_budgets,), costs (N,) items per
                             task and the plain run's metadata and scalars
  LOGS/item_fixed_{ds}_paired.csv   every comparable pair (as above), level
                             and budget (columns dataset, holdout_frac,
                             fixed_cost, budget, metric, method_a,
                             method_b, then STAT_FIELDS)
  LOGS/item_fixed_{ds}.json  config, notes, summary per level (r2, r2_unobs,
                             n_obs, n_exact, questions, cost and
                             overhead_share = (cost - questions) / cost,
                             NaN where nothing is bought), FIXED_KEY_PAIRS
                             rows, the crossover table of every bench x
                             item pair and level, checks, runtime, macros
                             (finite values only; "macros_undefined"
                             lists the rest):
      ItemBudgetFixedRsq{Ds}{M}C{c}Q{q}, ItemBudgetFixedSize...   R² split
                             mean (all tasks) / tasks observed; {c} =
                             num_words(C0): Czero, Cten, Cfifty,
                             Conehundred, Ctwohundred
      ItemBudgetFixedOverheadShare{Ds}{M}C{c}Q{q}   split mean of
                             overhead_share for the bench methods and
                             for the item sample ({M} = Items: one sample
                             shared by the three item methods);
                             ItemBudgetFixedQuestions{Ds}ItemsC{c}Q{q}
                             its mean number of questions
      ItemBudgetFixedPair{S}{Ds}{A}Vs{B}C{c}Q{q}   FIXED_KEY_PAIRS (each
                             bench method vs items_center), S in
                             make_numbers.PAIR_STATS;
                             ItemBudgetFixedSigPos / SigNeg{Ds}{A}Vs{B}C{c}
      ItemBudgetFixedCross{K}{Ds}{A}Vs{B}C{c}   crossover of every bench x
                             item pair, K in State, Q (q_first), QLo
                             (q_first_lo), QStay, QSig, QSigStay, NAhead,
                             NSig
      ItemBudgetFixedOverhead{Ds}C{c}, ItemBudgetFixedTotal{Ds}C{c} (cost
      of everything), ItemBudgetFixedMinTaskCost{Ds}C{c} (the cheapest
      whole task, min_t N_t + C0: no whole task fits below it),
      ItemBudgetFixed{NSplits,NBudgets,NLevels,MinQ,MaxQ,Violations}{Ds}
  OUT/item_fixed_{ds}.pdf    R² vs Q per level (items_only, far below the
                             others at small Q, left out), and the
                             crossover budgets q_stay / q_sig of each
                             whole-task method vs items_center against C0,
                             with the cheapest whole task for reference

Usage:
    python eval_item_budget.py [--datasets olv1items] [--budgets Q ...]
                               [--repeats 5] [--folds 10]
                               [--fixed-cost [C0 ...]]

Datasets need data/<name>.matrix.npz (fully observed) and
data/<name>.items.npz (build_olv1items.py layout).
"""

import argparse
import json
import math
import pathlib
import re
import sys
import time
import warnings
from collections import Counter
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

from greedy_select import TIE_RTOL, random_select
import cv_split
import score_bounds
from cv_split import (cv_splits, eligible_mask, target_mask,
                      ENTROPY_RF_SEED_OFFSET, MIN_TRAIN_OBS, N_REPEATS,
                      RANDOM_SEED_OFFSET)
from score_bounds import clip_z, dataset_bounds, z_limits
from eval_entropy_vs_mi import estimate_corr
from eval_cost_budget import budgeted_select, order_fill
from paired_stats import STAT_FIELDS, paired_stats, write_csv
from make_numbers import PAIR_STATS, camel, num_words

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42
# Common-random-number stream of the item sample: Split.seed(SEED + 18).
# Distinct from RANDOM_SEED_OFFSET (7, random task order),
# ENTROPY_RF_SEED_OFFSET (11, the retired entropy_rf stream),
# eval_prospective's 13-16 and analyze_objectives' 19
# (test_eval_item_budget.py scans the
# code tree for other *_SEED_OFFSET constants and SEED + n tags).
ITEM_SEED_OFFSET = 18
assert ITEM_SEED_OFFSET not in (RANDOM_SEED_OFFSET, ENTROPY_RF_SEED_OFFSET)
IMPUTE_RIDGE = 0.01   # = eval_entropy_vs_mi.impute_and_evaluate's RIDGE
Z_CLIP = 10.0         # = impute_and_evaluate's clip of standardized scores

DEFAULT_BUDGETS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
DATASETS = ["olv1items"]

# bench_facloc (facility location, the headline method) sits before
# bench_random so that its pairs read bench_facloc − bench_random; it draws
# no random numbers, so no other method's rows depend on its position
BENCH_METHODS = ["bench_mi", "bench_entropy", "bench_trace", "bench_facloc",
                 "bench_random"]
ITEM_METHODS = ["items_center", "items_gauss", "items_only"]
METHODS = BENCH_METHODS + ITEM_METHODS
# Item methods with a noise model (calibration diagnostics)
NOISE_METHODS = ["items_center", "items_gauss"]
KNAP_OBJECTIVE = {"bench_mi": "mi", "bench_entropy": "entropy",
                  "bench_trace": "trace", "bench_facloc": "facloc"}
SINGLETON_METHODS = ["bench_mi", "bench_entropy", "bench_trace",
                     "bench_facloc"]
METRICS = ["r2", "r2_unobs"]
METRIC_LABEL = {"r2": "all tasks",
                "r2_unobs": "tasks not observed exactly"}
# Pairs printed and turned into macros (r2), a - b as listed (METHODS
# order).  The first five are the original key pairs; the next four
# compare with the strongest item baseline, items_center; the last five
# are the facility-location pairs (bench_trace's pattern, plus facloc vs
# random and vs the MI and trace knapsacks).
KEY_PAIRS = [("bench_mi", "items_gauss"), ("bench_mi", "items_only"),
             ("items_gauss", "items_only"), ("bench_trace", "items_gauss"),
             ("bench_mi", "bench_random"),
             ("bench_mi", "items_center"), ("bench_entropy", "items_center"),
             ("bench_trace", "items_center"), ("items_center", "items_gauss"),
             ("bench_facloc", "items_gauss"), ("bench_facloc", "bench_random"),
             ("bench_facloc", "items_center"), ("bench_mi", "bench_facloc"),
             ("bench_trace", "bench_facloc")]

PAIRED_FIELDS = (["dataset", "holdout_frac", "budget", "metric",
                  "method_a", "method_b"] + STAT_FIELDS)
MACRO_PREFIX = "ItemBudget"
TOP_TASKS = 5         # most frequently selected tasks per method and budget

# Fixed per-task overhead (--fixed-cost; module docstring)
DEFAULT_FIXED_COSTS = [0, 10, 50, 100, 200]       # item-equivalents per task
FIXED_BUDGETS = [int(round(50 * 2 ** (k / 2))) for k in range(19)]
FIXED_STEM = "item_fixed"
FIXED_MACRO_PREFIX = "ItemBudgetFixed"
# Pairs with per-budget macros (every pair is in the CSV, every bench x
# item pair in the crossover table): each whole-task method against the
# strongest item baseline
FIXED_KEY_PAIRS = [(m, "items_center") for m in BENCH_METHODS]
CROSS_PAIRS = [(a, b) for a in BENCH_METHODS for b in ITEM_METHODS]
CROSS_FIELDS = [("State", "state"), ("Q", "q_first"),
                ("QLo", "q_first_lo"), ("QStay", "q_stay"),
                ("QSig", "q_sig"), ("QSigStay", "q_sig_stay"),
                ("NAhead", "n_ahead"), ("NSig", "n_sig")]
FIXED_PAIRED_FIELDS = (["dataset", "holdout_frac", "fixed_cost", "budget",
                        "metric", "method_a", "method_b"] + STAT_FIELDS)
# Per-split keys of evaluate_split that do not depend on the overhead
LEVEL_FREE_KEYS = ("eligible", "n_eligible", "noise_unit_var",
                   "noise_within_var", "noise_center_var", "n_clipped")
FIXED_PLOT_METHODS = [m for m in METHODS if m != "items_only"]

DISPLAY_NAME = {"olv1items": "OLL v1 items"}

METHOD_LABEL = {
    "bench_mi": "Whole tasks, knapsack MI",
    "bench_entropy": "Whole tasks, knapsack entropy",
    "bench_trace": "Whole tasks, knapsack trace",
    "bench_facloc": "Whole tasks, knapsack facility location",
    "bench_random": "Whole tasks, random",
    "items_center": "Random items, centered + Gaussian posterior",
    "items_gauss": "Random items, Gaussian posterior",
    "items_only": "Random items, sample means",
}
METHOD_SHORT = {"bench_mi": "B-MI", "bench_entropy": "B-Ent",
                "bench_trace": "B-Tr", "bench_facloc": "B-FL",
                "bench_random": "B-Rand",
                "items_center": "I-Cent", "items_gauss": "I-Gauss",
                "items_only": "I-Mean"}
# bench_facloc: Paul Tol's purple (#AA3377); the teal of the other
# scripts' facility location is taken here by the item methods
METHOD_COLOR = {
    "bench_mi": "#CC3311",
    "bench_entropy": "#4477AA",
    "bench_trace": "#EE7733",
    "bench_facloc": "#AA3377",
    "bench_random": "#888888",
    "items_center": "#117733",
    "items_gauss": "#009988",
    "items_only": "#33BBEE",
}
METHOD_STYLE = {
    "bench_mi": "o-",
    "bench_entropy": "o-",
    "bench_trace": "^-",
    "bench_facloc": "v-",
    "bench_random": "x--",
    "items_center": "D-",
    "items_gauss": "s-",
    "items_only": "s--",
}
# Drawn with thicker lines: the strongest selectors, the headline method
# and the strongest item baseline
PLOT_EMPHASIS = ("bench_mi", "bench_trace", "bench_facloc", "items_center")


def display_name(ds_name):
    """Display name of a dataset (the raw name if none is registered)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def matrix_path(ds_name):
    return DATA / f"{ds_name}.matrix.npz"


def items_path(ds_name):
    return DATA / f"{ds_name}.items.npz"


# ── Data ─────────────────────────────────────────────────────────

class ItemData(NamedTuple):
    """Task matrix and item scores of one dataset (models in matrix order).

    scores[i, m] is model m's score on item i (0/1, or the continuous mc2
    value of a TruthfulQA item); items of task t are rows
    offsets[t]:offsets[t + 1], item_task[i] their task.
    """
    B: np.ndarray                # (M, N) task scores
    O: np.ndarray                # (M, N) observation mask (all ones)
    benchmark_names: np.ndarray  # (N,)
    model_names: np.ndarray      # (M,)
    scores: np.ndarray           # (n_items, M) float64
    item_task: np.ndarray        # (n_items,) int
    offsets: np.ndarray          # (N + 1,) int
    n_items: np.ndarray          # (N,) int, items per task
    binary: np.ndarray           # (N,) bool, 0/1 items


def decode_item_scores(z):
    """(n_items, n_models) float64 item scores of an items.npz (meta.items.
    decode / build_olv1items.load_item_scores): binary items unpacked from
    correct_bits (np.packbits along the model axis, bitorder 'big'), the
    non-binary rows float_item_index replaced by float_scores."""
    n_models = int(z["n_models"])
    s = np.unpackbits(z["correct_bits"], axis=1, count=n_models,
                      bitorder="big").astype(np.float64)
    s[np.asarray(z["float_item_index"], dtype=np.int64)] = z["float_scores"]
    return s


def load_items(mat_path, it_path, atol=1e-12):
    """ItemData of a matrix / items file pair, with consistency checks:
    same models and tasks in the same order, contiguous task item ranges,
    a fully observed matrix and task scores equal to the means of their
    item scores (within atol)."""
    d = np.load(mat_path, allow_pickle=False)
    z = np.load(it_path, allow_pickle=False)
    B = np.asarray(d["B"], dtype=np.float64)
    O = np.asarray(d["O"], dtype=np.float64)
    names = np.asarray(d["benchmark_names"])
    models = np.asarray(d["model_names"])
    M, N = B.shape
    if not np.all(O == 1):
        raise ValueError(f"{mat_path}: {int(np.sum(O != 1))} unobserved "
                         "cells; item-level evaluation needs every model's "
                         "items for every task")
    if not (np.array_equal(np.asarray(z["model_names"]), models)
            and np.array_equal(np.asarray(z["task_names"]), names)):
        raise ValueError(f"{it_path}: models or tasks differ from "
                         f"{mat_path}")
    offsets = np.asarray(z["task_offsets"], dtype=np.int64)
    n_items = np.asarray(z["task_n_items"], dtype=np.int64)
    item_task = np.asarray(z["item_task"], dtype=np.int64)
    binary = np.asarray(z["task_binary"], dtype=bool)
    scores = decode_item_scores(z)
    if (len(offsets) != N + 1 or offsets[0] != 0
            or not np.array_equal(np.diff(offsets), n_items)
            or offsets[-1] != len(item_task)
            or not np.array_equal(item_task,
                                  np.repeat(np.arange(N), n_items))
            or scores.shape != (len(item_task), M)):
        raise ValueError(f"{it_path}: inconsistent item layout")
    if np.any(n_items < 2):
        raise ValueError(f"{it_path}: every task needs >= 2 items")
    rec = np.add.reduceat(scores, offsets[:-1], axis=0) / n_items[:, None]
    err = float(np.max(np.abs(rec.T - B)))
    if not err <= atol:
        raise ValueError(f"{it_path}: item means differ from the task "
                         f"matrix by up to {err:.3g}")
    return ItemData(B, O, names, models, scores, item_task, offsets,
                    n_items, binary)


def load_dataset(ds_name):
    return load_items(matrix_path(ds_name), items_path(ds_name))


# ── Selection ────────────────────────────────────────────────────

def bench_select(method, R, costs, budget, eligible, random_order):
    """Whole-task selection of a bench method under `budget`: {selected,
    singleton, tie_breaks, index_ties (counts)}.  bench_mi / bench_entropy
    / bench_trace / bench_facloc are eval_cost_budget.budgeted_select with
    the objective KNAP_OBJECTIVE[method]; random_order is the split's
    random_select order (bench_random)."""
    if method in KNAP_OBJECTIVE:
        res = budgeted_select(R, costs, budget, KNAP_OBJECTIVE[method],
                              eligible=eligible)
    elif method == "bench_random":
        return {"selected": order_fill(random_order, costs, budget),
                "singleton": False, "tie_breaks": 0, "index_ties": 0}
    else:
        raise ValueError(method)
    return {"selected": [int(j) for j in res["selected"]],
            "singleton": bool(res["singleton"]),
            "tie_breaks": len(res["tie_breaks"]),
            "index_ties": len(res["index_ties"])}


# ── Prediction ───────────────────────────────────────────────────

def standardize(B_test, col_mean, col_std):
    """Train-standardized validation scores, clipped like
    impute_and_evaluate: (z, number of clipped cells)."""
    z = (B_test - col_mean[None, :]) / col_std[None, :]
    return np.clip(z, -Z_CLIP, Z_CLIP), int(np.sum(np.abs(z) > Z_CLIP))


def gauss_weights(R, obs, noise=None, ridge=IMPUTE_RIDGE):
    """(N, |obs|) weights W of the Gaussian posterior mean E[z | y] = W y_obs
    for y_obs = z_obs + e, e ~ N(0, diag(noise) + ridge·I):
    W = R[:, obs] (R_oo + diag(noise) + ridge·I)^{-1}.  With noise = None
    (0) the rows of the unobserved tasks are impute_and_evaluate's
    weights."""
    obs = np.asarray(obs, dtype=np.int64)
    A = R[np.ix_(obs, obs)] + ridge * np.eye(len(obs))
    if noise is not None:
        A = A + np.diag(np.asarray(noise, dtype=float))
    return np.linalg.solve(A, R[obs, :]).T


def bench_predict(R, selected, Z):
    """Predicted standardized scores (M_val, N) given whole tasks
    `selected`: exact values on them, the Gaussian conditional mean
    (ridge IMPUTE_RIDGE) elsewhere; the train mean (0) if none."""
    Z_hat = np.zeros_like(Z)
    if len(selected):
        sel = np.asarray(selected, dtype=np.int64)
        Z_hat = Z[:, sel] @ gauss_weights(R, sel).T
        Z_hat[:, sel] = Z[:, sel]
    return Z_hat


def item_permutation(n_items, split):
    """The split's permutation of all items (common random numbers,
    Split.seed(SEED + ITEM_SEED_OFFSET)); the sample of budget Q is its
    first Q entries."""
    rng = np.random.default_rng(split.seed(SEED + ITEM_SEED_OFFSET))
    return rng.permutation(n_items)


def fixed_item_sample(perm, item_task, N, budget, fixed_cost):
    """The items bought under `budget` item-equivalents when every task
    touched costs `fixed_cost` (C0) on top of its items (module docstring,
    fixed overhead): the permutation `perm` walked skip-and-continue, an
    item of a task already touched costing 1 and the first item of a new
    task 1 + C0.  Returns the bought items in permutation order (C0 = 0:
    perm[:budget]).

    Vectorized: while 1 + C0 still fits, every item fits, so a prefix
    perm[:p] (p = number of positions whose cumulative cost is within the
    budget) is bought; at position p a new task no longer fits and, as the
    remaining budget r < 1 + C0 only shrinks, no later new task does
    either, so the walk ends with the next r items of the tasks the prefix
    touched.
    """
    perm = np.asarray(perm, dtype=np.int64)
    budget, fixed_cost = int(budget), int(fixed_cost)
    if fixed_cost < 0:
        raise ValueError("fixed_cost must be >= 0")
    if fixed_cost == 0:
        return perm[:budget]
    tasks = np.asarray(item_task, dtype=np.int64)[perm]
    first = np.zeros(len(perm), dtype=np.int64)
    first[np.unique(tasks, return_index=True)[1]] = 1
    cum = np.cumsum(1 + fixed_cost * first)
    p = int(np.searchsorted(cum, budget, side="right"))
    rest = budget - (int(cum[p - 1]) if p else 0)
    if p == len(perm) or rest <= 0:
        return perm[:p]
    touched = np.zeros(N, dtype=bool)
    touched[tasks[:p]] = True
    tail = perm[p:][touched[tasks[p:]]][:rest]
    return np.concatenate([perm[:p], tail])


def sampled_counts(sample, item_task, N):
    """(N,) number of sampled items per task."""
    return np.bincount(item_task[sample], minlength=N).astype(np.int64)


def sample_task_means(scores_val, sample, item_task, N):
    """(M_val, N) mean over the sampled items of each task for every
    validation model (scores_val: (n_items, M_val)); NaN where a task has
    no sampled item.  Sums in sample order (np.add.at: deterministic)."""
    sums = np.zeros((N, scores_val.shape[1]))
    np.add.at(sums, item_task[sample], scores_val[sample])
    n = sampled_counts(sample, item_task, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / n[:, None]
    means[n == 0] = np.nan
    return means.T


def item_unit_variance(data, train_idx, col_mean):
    """Per-item variances of the tasks from the TRAINING models: (v, w).

    v (items_gauss's noise): p_t (1 − p_t) with p_t = col_mean[t] (train
    mean task score) for binary tasks; for non-binary tasks (TruthfulQA
    mc2) the training models' mean within-model variance (ddof 0) of the
    task's item scores.  w: that mean within-model variance for every task
    (binary: mean_m p_mt (1 − p_mt) = p_t (1 − p_t) − Var_m(p_mt)), a
    diagnostic of how conservative the binomial v is.
    """
    B_train = data.B[train_idx]
    w = np.mean(B_train * (1.0 - B_train), axis=0)
    v = col_mean * (1.0 - col_mean)
    for t in np.flatnonzero(~data.binary):
        X = data.scores[data.offsets[t]:data.offsets[t + 1]][:, train_idx]
        w[t] = float(np.mean(np.var(X, axis=0)))
        v[t] = w[t]
    return np.maximum(v, 0.0), np.maximum(w, 0.0)


def train_item_stats(data, train_idx):
    """Item statistics of the TRAINING models for items_center: (xbar, u).

    xbar (n_items,): the training models' mean score of every item.
    u (N,): the training models' average double-centered residual variance
    of each task, u_t = mean_{m, i in t} r_mi², r_mi = x_mi − xbar_i −
    s_mt + mu_t (s_mt = mean_{i in t} x_mi, mu_t = mean_m s_mt), which
    equals mean_{i in t} Var_m x_mi − Var_m s_mt (ddof 0).  r_mi sums to
    0 over the task's items for every model, so u_t / n_t · (N_t − n_t) /
    (N_t − 1) is the variance of the mean of n_t residuals drawn without
    replacement, averaged over the models.  Computed by the identity (no
    (n_items, n_train) residual array); test_eval_item_budget.py checks it
    against the explicit double centering.
    """
    X = data.scores[:, train_idx]
    xbar = np.mean(X, axis=1)
    item_var = np.mean((X - xbar[:, None]) ** 2, axis=1)  # Var_m x_mi
    s = (np.add.reduceat(X, data.offsets[:-1], axis=0)
         / data.n_items[:, None])                         # (N, n_train)
    u = (np.add.reduceat(item_var, data.offsets[:-1]) / data.n_items
         - np.var(s, axis=1))
    return xbar, np.maximum(u, 0.0)


def centered_sample_means(means, xbar, sample, item_task, N):
    """(M_val, N) sample means `means` (sample_task_means) minus the
    training models' mean over the same sampled items (xbar: the training
    models' item means, train_item_stats); NaN where a task has no sampled
    item.  The offset depends on the sample and on training data only."""
    offset = sample_task_means(xbar[:, None], sample, item_task, N)[0]
    return means - offset[None, :]


def sample_noise_var(unit_var, n_sampled, n_total, col_std):
    """Variance of the standardized mean of n_t of N_t items drawn without
    replacement: v_t / n_t · (N_t − n_t) / (N_t − 1) / sd_t²; NaN where
    n_t = 0 (0 where the task is sampled completely)."""
    n = np.asarray(n_sampled, dtype=float)
    Nt = np.asarray(n_total, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (unit_var / n * (Nt - n) / (Nt - 1.0)
               / np.asarray(col_std, dtype=float) ** 2)
    return np.where(n > 0, var, np.nan)


def items_posterior(R, Y, n_sampled, n_total, unit_var, col_std):
    """Gaussian posterior mean (M_val, N) of all standardized task scores
    from the standardized observations Y (NaN where a task has no sampled
    item) with noise sample_noise_var(unit_var, ...) on the sampled tasks
    (items_gauss: uncentered Y and v; items_center: centered Y and u)."""
    obs = np.flatnonzero(n_sampled > 0)
    if len(obs) == 0:
        return np.zeros(Y.shape)
    noise = sample_noise_var(unit_var, n_sampled, n_total, col_std)[obs]
    W = gauss_weights(R, obs, noise)
    return Y[:, obs] @ W.T


def items_predict(R, Y, n_sampled, n_total, unit_var, col_std):
    """(items_only, items_gauss) predictions (M_val, N) from the
    standardized sample means Y (NaN where a task has no sampled item)."""
    Z_only = np.where(np.isnan(Y), 0.0, Y)
    return Z_only, items_posterior(R, Y, n_sampled, n_total, unit_var,
                                   col_std)


def noise_diagnostics(Z, Y, n_sampled, n_total, unit_var, col_std):
    """(calibration, shared fraction) of a noise model on the tasks with
    0 < n_t < N_t: e = Y − Z over the validation models, calibration =
    sum e² / (n_val · sum_t Var e_t), shared = n_val · sum_t ebar_t² /
    sum e² (ebar_t the mean of e_t over the validation models: the part
    of the squared error common to all of them).  NaN without such a
    task."""
    cols = np.flatnonzero((n_sampled > 0) & (n_sampled < n_total))
    if len(cols) == 0 or Z.shape[0] == 0:
        return np.nan, np.nan
    e = Y[:, cols] - Z[:, cols]
    var = sample_noise_var(unit_var, n_sampled, n_total, col_std)[cols]
    sse = float(np.sum(e ** 2))
    shared = float(Z.shape[0] * np.sum(np.mean(e, axis=0) ** 2))
    return (sse / max(Z.shape[0] * float(np.sum(var)), 1e-300),
            shared / max(sse, 1e-300))


def pooled_r2(Z, Z_hat, cols=None):
    """(R², RMSE) pooled over all rows and the task columns `cols` (bool
    mask, None: all); NaN for no column."""
    if cols is not None:
        Z, Z_hat = Z[:, cols], Z_hat[:, cols]
    if Z.size == 0:
        return np.nan, np.nan
    ss_res = float(np.sum((Z - Z_hat) ** 2))
    ss_tot = float(np.sum(Z ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12), float(np.sqrt(ss_res / Z.size))


# ── Main experiment ──────────────────────────────────────────────

def evaluate_split(split, data, budgets, return_predictions=False,
                   fixed_cost=None, bounds=None):
    """Every method and budget on one matched split.

    Returns a dict of this split's rows (see the npz keys of the module
    docstring without the leading split axis), plus "predictions"
    {(method, budget index): Z_hat} and "Z" (the standardized validation
    scores) with return_predictions.  Raises if the correlation estimate
    fails.

    fixed_cost: None (the plain run) or the per-task overhead C0 in
    item-equivalents (module docstring): task costs N_t + C0, the item
    sample fixed_item_sample(..., C0), {m}_cost the true cost (questions +
    C0 x tasks touched) and the extra rows {m}_questions.  C0 = 0 gives
    the plain rows exactly.

    Protocol rules: the R² of every method score the targets of
    target_mask(O_train) only (rule D1; on olv1items every task), and the
    predictions are clipped to the valid score range `bounds` (rule D2,
    score_bounds.dataset_bounds: [0, 1] on olv1items; None: no clip) in
    train-standardized units before they are scored.
    """
    B, O = data.B, data.O
    M, N = B.shape
    c0 = 0 if fixed_cost is None else int(fixed_cost)
    if c0 < 0:
        raise ValueError("fixed_cost must be >= 0")
    costs = data.n_items.astype(float)
    if c0:
        costs = costs + float(c0)
    nb = len(budgets)
    B_train, O_train = B[split.train_idx], O[split.train_idx]
    el = eligible_mask(O_train)
    target = target_mask(O_train)            # D1, shared by every method
    R, col_mean, col_std = estimate_corr(B_train, O_train)
    z_lo, z_hi = z_limits(bounds, col_mean, col_std)      # D2
    Z, n_clipped = standardize(B[split.val_idx], col_mean, col_std)
    unit_var, within_var = item_unit_variance(data, split.train_idx,
                                              col_mean)
    xbar_train, center_var = train_item_stats(data, split.train_idx)
    random_order = random_select(
        R, N, seed=split.seed(SEED + RANDOM_SEED_OFFSET),
        eligible=el)["selected"]
    perm = item_permutation(len(data.item_task), split)
    scores_val = data.scores[:, split.val_idx]

    row = {"eligible": el, "n_eligible": int(el.sum()),
           "noise_unit_var": unit_var, "noise_within_var": within_var,
           "noise_center_var": center_var,
           "n_clipped": n_clipped,
           "items_n_sampled": np.zeros((nb, N), dtype=np.int64)}
    for m in METHODS:
        for key in ("r2", "r2_unobs", "rmse", "cost"):
            row[f"{m}_{key}"] = np.full(nb, np.nan)
        for key in ("n_exact", "n_obs"):
            row[f"{m}_{key}"] = np.zeros(nb, dtype=np.int64)
    for m in NOISE_METHODS:
        for key in ("noise_calib", "noise_shared"):
            row[f"{m}_{key}"] = np.full(nb, np.nan)
    for m in BENCH_METHODS:
        row[f"{m}_selected"] = np.full((nb, N), -1, dtype=np.int64)
        row[f"{m}_index_ties"] = np.zeros(nb, dtype=np.int64)
        row[f"{m}_tie_breaks"] = np.zeros(nb, dtype=np.int64)
    for m in SINGLETON_METHODS:
        row[f"{m}_singleton"] = np.zeros(nb, dtype=bool)
    if fixed_cost is not None:
        for m in METHODS:
            row[f"{m}_questions"] = np.zeros(nb, dtype=np.int64)
    preds = {}

    all_targets = bool(target.all())

    def record(m, b, Z_hat, exact):
        Z_hat, _ = clip_z(Z_hat, z_lo, z_hi)
        if all_targets:
            row[f"{m}_r2"][b], row[f"{m}_rmse"][b] = pooled_r2(Z, Z_hat)
            row[f"{m}_r2_unobs"][b], _ = pooled_r2(Z, Z_hat, ~exact)
        else:
            row[f"{m}_r2"][b], row[f"{m}_rmse"][b] = pooled_r2(Z, Z_hat,
                                                              target)
            row[f"{m}_r2_unobs"][b], _ = pooled_r2(Z, Z_hat,
                                                  ~exact & target)
        if return_predictions:
            preds[(m, b)] = Z_hat

    cache = {}  # identical selections recur across methods and budgets
    for b, Q in enumerate(budgets):
        for m in BENCH_METHODS:
            res = bench_select(m, R, costs, Q, el, random_order)
            sel = res["selected"]
            key = tuple(sel)
            if key not in cache:
                cache[key] = bench_predict(R, sel, Z)
            exact = np.zeros(N, dtype=bool)
            exact[sel] = True
            record(m, b, cache[key], exact)
            row[f"{m}_n_exact"][b] = row[f"{m}_n_obs"][b] = len(sel)
            row[f"{m}_cost"][b] = float(costs[sel].sum())
            if fixed_cost is not None:
                row[f"{m}_questions"][b] = int(data.n_items[sel].sum())
            row[f"{m}_selected"][b, :len(sel)] = sel
            row[f"{m}_index_ties"][b] = res["index_ties"]
            row[f"{m}_tie_breaks"][b] = res["tie_breaks"]
            if m in SINGLETON_METHODS:
                row[f"{m}_singleton"][b] = res["singleton"]

        sample = (perm[:Q] if c0 == 0 else
                  fixed_item_sample(perm, data.item_task, N, Q, c0))
        n_t = sampled_counts(sample, data.item_task, N)
        row["items_n_sampled"][b] = n_t
        means = sample_task_means(scores_val, sample, data.item_task, N)
        Y = (means - col_mean[None, :]) / col_std[None, :]
        Y_center = (centered_sample_means(means, xbar_train, sample,
                                          data.item_task, N)
                    / col_std[None, :])
        Z_only, Z_gauss = items_predict(R, Y, n_t, data.n_items, unit_var,
                                        col_std)
        Z_center = items_posterior(R, Y_center, n_t, data.n_items,
                                   center_var, col_std)
        for m, Y_m, v_m in (("items_center", Y_center, center_var),
                            ("items_gauss", Y, unit_var)):
            (row[f"{m}_noise_calib"][b],
             row[f"{m}_noise_shared"][b]) = noise_diagnostics(
                Z, Y_m, n_t, data.n_items, v_m, col_std)
        exact = n_t == data.n_items
        for m, Z_hat in (("items_center", Z_center), ("items_gauss", Z_gauss),
                         ("items_only", Z_only)):
            record(m, b, Z_hat, exact)
            row[f"{m}_n_exact"][b] = int(exact.sum())
            row[f"{m}_n_obs"][b] = int(np.sum(n_t > 0))
            if fixed_cost is None:
                row[f"{m}_cost"][b] = float(Q)
            else:
                row[f"{m}_cost"][b] = float(len(sample)
                                            + c0 * int(np.sum(n_t > 0)))
                row[f"{m}_questions"][b] = len(sample)
    if return_predictions:
        row["predictions"] = preds
        row["Z"] = Z
    return row


def run(ds_name, budgets, n_repeats=N_REPEATS, n_folds=N_FOLDS, data=None,
        bounds=None):
    """Every method and budget on the matched splits of one dataset
    (data: an ItemData, default load_dataset(ds_name); bounds: the valid
    score ranges of rule D2, default score_bounds.dataset_bounds(ds_name)
    when data is loaded here, None (no clip) for given data).  Returns the
    save dict (npz keys of the module docstring)."""
    if data is None:
        data = load_dataset(ds_name)
        if bounds is None:
            bounds = dataset_bounds(ds_name, data.benchmark_names, DATA)
    M, N = data.B.shape
    n_items_total = len(data.item_task)
    budgets = [int(q) for q in budgets]
    if any(q < 1 or q > n_items_total for q in budgets):
        raise ValueError(f"budgets must lie in 1..{n_items_total}")
    names = [str(b) for b in data.benchmark_names]
    print(f"{display_name(ds_name)}: {M} models × {N} tasks, "
          f"{n_items_total} items, task sizes {int(data.n_items.min())} "
          f"({names[int(np.argmin(data.n_items))]}) – "
          f"{int(data.n_items.max())} "
          f"({names[int(np.argmax(data.n_items))]}); "
          f"{int(np.sum(~data.binary))} non-binary task(s): "
          f"{', '.join(n for n, bi in zip(names, data.binary) if not bi)}",
          flush=True)
    print(f"budgets Q: {budgets}", flush=True)

    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED))
    S, nb = len(splits), len(budgets)
    print(f"{n_repeats} × {n_folds}-fold CV = {S} matched splits, "
          f"{HOLDOUT_FRAC:.0%} holdout; item samples: "
          f"Split.seed(SEED + {ITEM_SEED_OFFSET})", flush=True)

    out = {"budgets": np.array(budgets, dtype=np.int64),
           "costs": data.n_items.astype(np.int64),
           "benchmark_names": np.array(names),
           "task_binary": data.binary.copy(),
           "methods": np.array(METHODS)}
    rows = []
    for s in splits:
        t0 = time.time()
        try:
            r = evaluate_split(s, data, budgets, bounds=bounds)
        except Exception as e:  # noqa: BLE001 (logged, row left NaN)
            print(f"  rep {s.rep} fold {s.fold}: estimation failed: {e}",
                  flush=True)
            r = None
        rows.append(r)
        if r is not None:
            print(f"  rep {s.rep} fold {s.fold} ({time.time() - t0:.1f}s): "
                  + "  ".join(f"{METHOD_SHORT[m]} {r[f'{m}_r2'][-1]:.3f}"
                              for m in METHODS)
                  + f"  (R² at Q={budgets[-1]})", flush=True)

    template = next((r for r in rows if r is not None), None)
    if template is None:
        raise RuntimeError(f"{ds_name}: every split failed")
    for key, val in template.items():
        val = np.asarray(val)
        fill = (np.nan if val.dtype.kind == "f"
                else -1 if key.endswith("_selected") else 0)
        arr = np.full((S,) + val.shape, fill, dtype=val.dtype)
        for i, r in enumerate(rows):
            if r is not None:
                arr[i] = r[key]
        out[key] = arr
    out.update(
        rep=np.array([s.rep for s in splits], dtype=np.int64),
        fold=np.array([s.fold for s in splits], dtype=np.int64),
        split_id=np.array([s.split_id for s in splits], dtype=np.int64),
        n_train=np.array([len(s.train_idx) for s in splits], dtype=np.int64),
        n_val=np.array([len(s.val_idx) for s in splits], dtype=np.int64),
        estimation_failed=np.array([r is None for r in rows]),
        holdout_frac=np.float64(HOLDOUT_FRAC), n_folds=np.int64(n_folds),
        n_repeats=np.int64(n_repeats), min_train_obs=np.int64(MIN_TRAIN_OBS),
        min_target_obs=np.int64(cv_split.MIN_TARGET_OBS),
        pred_clip=np.bool_(score_bounds.PRED_CLIP),
        seed=np.int64(SEED), item_seed_offset=np.int64(ITEM_SEED_OFFSET),
        random_seed_offset=np.int64(RANDOM_SEED_OFFSET),
        impute_ridge=np.float64(IMPUTE_RIDGE), z_clip=np.float64(Z_CLIP),
        n_items_total=np.int64(n_items_total), n_models=np.int64(M))
    return out


# ── Paired statistics, summaries and macros ──────────────────────

def method_pairs():
    """Every pair (a, b), a before b in METHODS order."""
    return [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]


def method_family(m):
    """'bench' or 'items'."""
    if m in BENCH_METHODS:
        return "bench"
    if m in ITEM_METHODS:
        return "items"
    raise ValueError(m)


def paired_comparable(metric, a, b):
    """Whether a paired a − b row of `metric` compares like with like: r2
    always (all tasks, one denominator); r2_unobs only within a family
    (a bench method's unselected tasks vs an item method's all tasks
    would differ in cells and denominator)."""
    return metric == "r2" or method_family(a) == method_family(b)


def paired_rows(ds_name, out):
    """paired_stats of a − b for every budget, metric and comparable pair
    (paired_comparable), as CSV rows (PAIRED_FIELDS)."""
    rows = []
    for b, Q in enumerate(out["budgets"]):
        for metric in METRICS:
            for ma, mb in method_pairs():
                if not paired_comparable(metric, ma, mb):
                    continue
                st = paired_stats(out[f"{ma}_{metric}"][:, b],
                                  out[f"{mb}_{metric}"][:, b],
                                  out["n_train"], out["n_val"], out["rep"])
                rows.append({"dataset": ds_name, "holdout_frac": HOLDOUT_FRAC,
                             "budget": int(Q), "metric": metric,
                             "method_a": ma, "method_b": mb, **st})
    return rows


def _nan_mean_std(a, axis=None):
    """np.nanmean / np.nanstd (ddof=0) without all-NaN-slice warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)


def _significant(r):
    return bool(np.isfinite(r["ci_low"])
                and (r["ci_low"] > 0 or r["ci_high"] < 0))


def _json_num(x):
    """JSON number: int for integers (counts), float otherwise, None for
    NaN / inf."""
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    return x if np.isfinite(x) else None


def summary_tables(out):
    """{metric: {method: {Q: {mean, std, n}}}} plus mean set sizes, costs
    and the noise diagnostics of NOISE_METHODS, for the JSON."""
    budgets = [int(q) for q in out["budgets"]]
    tables = {}
    for metric in (METRICS + ["n_obs", "n_exact", "cost", "noise_calib",
                              "noise_shared"]):
        tables[metric] = {}
        for m in (NOISE_METHODS if metric.startswith("noise_") else METHODS):
            arr = np.asarray(out[f"{m}_{metric}"], dtype=float)
            arr = np.where(out["estimation_failed"][:, None], np.nan, arr)
            mu, sd = _nan_mean_std(arr, axis=0)
            n = np.sum(np.isfinite(arr), axis=0)
            tables[metric][m] = {
                str(q): {"mean": _json_num(mu[b]), "std": _json_num(sd[b]),
                         "n": int(n[b])} for b, q in enumerate(budgets)}
    return tables


def top_tasks(out, k=TOP_TASKS):
    """{method: {Q: [[task, n_splits], ...]}} the k tasks selected in the
    most splits (ties by first occurrence in the task order)."""
    names = [str(n) for n in out["benchmark_names"]]
    res = {}
    for m in BENCH_METHODS:
        res[m] = {}
        sel = out[f"{m}_selected"]
        for b, q in enumerate(out["budgets"]):
            cnt = Counter(int(j) for row in sel[:, b] for j in row if j >= 0)
            order = sorted(cnt, key=lambda j: (-cnt[j], j))[:k]
            res[m][str(int(q))] = [[names[j], cnt[j]] for j in order]
    return res


def macro_dict(ds_name, out, paired):
    """Flat {MacroName: number} (module docstring, 'Macros')."""
    ds = camel(ds_name)
    P = MACRO_PREFIX
    budgets = [int(q) for q in out["budgets"]]
    ok = ~out["estimation_failed"]
    mac = {}

    def put(name, value):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name!r} is not letters only")
        if name in mac:
            raise ValueError(f"duplicate macro {name}")
        mac[name] = _json_num(value)

    for m in METHODS:
        M_ = camel(m)
        for b, q in enumerate(budgets):
            Q_ = "Q" + num_words(q)
            mu, sd = _nan_mean_std(out[f"{m}_r2"][ok, b])
            put(f"{P}Rsq{ds}{M_}{Q_}", mu)
            put(f"{P}RsqStd{ds}{M_}{Q_}", sd)
            put(f"{P}RsqUnobs{ds}{M_}{Q_}",
                _nan_mean_std(out[f"{m}_r2_unobs"][ok, b])[0])
            put(f"{P}Size{ds}{M_}{Q_}",
                float(np.mean(out[f"{m}_n_obs"][ok, b])))
            if m in BENCH_METHODS:
                put(f"{P}Cost{ds}{M_}{Q_}",
                    float(np.mean(out[f"{m}_cost"][ok, b])))
            if m in NOISE_METHODS:
                put(f"{P}NoiseCalib{ds}{M_}{Q_}",
                    _nan_mean_std(out[f"{m}_noise_calib"][ok, b])[0])
                put(f"{P}NoiseShared{ds}{M_}{Q_}",
                    _nan_mean_std(out[f"{m}_noise_shared"][ok, b])[0])
    lookup = {(r["budget"], r["metric"], r["method_a"], r["method_b"]): r
              for r in paired}
    for ma, mb in KEY_PAIRS:
        pair = f"{camel(ma)}Vs{camel(mb)}"
        n_pos = n_neg = 0
        for q in budgets:
            r = lookup[(q, "r2", ma, mb)]
            for suffix, (field, _) in PAIR_STATS.items():
                put(f"{P}Pair{suffix}{ds}{pair}Q{num_words(q)}", r[field])
            if _significant(r):
                n_pos += int(r["ci_low"] > 0)
                n_neg += int(r["ci_high"] < 0)
        put(f"{P}SigPos{ds}{pair}", n_pos)
        put(f"{P}SigNeg{ds}{pair}", n_neg)
    put(f"{P}NSplits{ds}", int(ok.sum()))
    put(f"{P}NReps{ds}", len(np.unique(out["rep"][ok])))
    put(f"{P}NModels{ds}", int(out["n_models"]))
    put(f"{P}NTasks{ds}", len(out["costs"]))
    put(f"{P}NItems{ds}", int(out["n_items_total"]))
    put(f"{P}NBudgets{ds}", len(budgets))
    put(f"{P}MinTaskItems{ds}", int(np.min(out["costs"])))
    put(f"{P}MaxTaskItems{ds}", int(np.max(out["costs"])))
    put(f"{P}NClipped{ds}", int(np.sum(out["n_clipped"])))
    return mac


def write_json(path, ds_name, out, paired, runtime_sec, argv):
    """Config, summary tables, key-pair rows, top tasks and macros."""
    ratio = (out["noise_within_var"]
             / np.maximum(out["noise_unit_var"], 1e-300))[
                 ~out["estimation_failed"]]
    binary = out["task_binary"]
    key_rows = [r for r in paired
                if (r["method_a"], r["method_b"]) in KEY_PAIRS]
    doc = {
        "script": "eval_item_budget.py",
        "dataset": ds_name,
        "argv": list(argv),
        "config": {
            "budgets": [int(q) for q in out["budgets"]],
            "methods": METHODS,
            "key_pairs": [list(p) for p in KEY_PAIRS],
            "n_repeats": int(out["n_repeats"]), "n_folds": int(out["n_folds"]),
            "holdout_frac": HOLDOUT_FRAC, "seed": SEED,
            "min_train_obs": MIN_TRAIN_OBS,
            "min_target_obs": int(cv_split.MIN_TARGET_OBS),
            "pred_clip": bool(score_bounds.PRED_CLIP),
            "random_seed_tag": SEED + RANDOM_SEED_OFFSET,
            "item_seed_tag": SEED + ITEM_SEED_OFFSET,
            "impute_ridge": IMPUTE_RIDGE, "z_clip": Z_CLIP,
            "tie_rtol": TIE_RTOL,
            "n_items_total": int(out["n_items_total"]),
            "task_n_items": {str(n): int(c) for n, c in
                             zip(out["benchmark_names"], out["costs"])},
            "non_binary_tasks": [str(n) for n, bi in
                                 zip(out["benchmark_names"], binary) if not bi],
            "noise_model": {
                "items_gauss": "binary: p(1-p), p = train mean task score; "
                               "non-binary: train models' mean within-model "
                               "item variance; times (N_t-n_t)/(N_t-1)/n_t, "
                               "divided by the train sd^2",
                "items_center": "observation (model mean on the sampled "
                                "items - training models' mean on the same "
                                "items) / train sd; noise: training models' "
                                "mean double-centered residual variance "
                                "times (N_t-n_t)/(N_t-1)/n_t, divided by "
                                "the train sd^2",
            },
            "r2_unobs_pairs": "same-family pairs only (bench-bench, "
                              "items-items); bench-items r2_unobs rows "
                              "would compare different task sets",
        },
        "diagnostics": {
            "noise_within_over_unit_binary_median":
                _json_num(np.median(ratio[:, binary])) if binary.any()
                else None,
            "noise_within_over_unit_binary_min":
                _json_num(np.min(ratio[:, binary])) if binary.any() else None,
            "n_clipped_cells": int(np.sum(out["n_clipped"])),
            "estimation_failed": int(np.sum(out["estimation_failed"])),
            "singleton_wins": {m: int(np.sum(out[f"{m}_singleton"]))
                               for m in SINGLETON_METHODS},
            "index_ties": {m: int(np.sum(out[f"{m}_index_ties"]))
                           for m in BENCH_METHODS},
        },
        "summary": summary_tables(out),
        "paired_key_pairs": [{k: (_json_num(v) if isinstance(v, float) else v)
                              for k, v in r.items()} for r in key_rows],
        "top_tasks": top_tasks(out),
        "runtime_sec": round(float(runtime_sec), 1),
        "macros": macro_dict(ds_name, out, paired),
    }
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)
    print(f"Summary saved to {path}", flush=True)


def print_summary(ds_name, out, paired):
    budgets = [int(q) for q in out["budgets"]]
    ok = ~out["estimation_failed"]
    S = int(ok.sum())
    for metric in METRICS:
        print(f"\n== {display_name(ds_name)}: pooled R² over "
              f"{METRIC_LABEL[metric]} vs question budget Q "
              f"(mean over {S} splits) ==")
        print(f"{'method':<44}" + "".join(f"{q:>12}" for q in budgets))
        for m in METHODS:
            cells = []
            for b in range(len(budgets)):
                mu, _ = _nan_mean_std(out[f"{m}_{metric}"][ok, b])
                cells.append(f"{mu:.3f}/{np.mean(out[f'{m}_n_obs'][ok, b]):.0f}")
            print(f"{METHOD_LABEL[m]:<44}" + "".join(f"{c:>12}"
                                                     for c in cells))
    print("(cell = mean R² / mean #tasks observed: selected whole tasks, or "
          "tasks with >= 1 sampled item; the r2_unobs task set is each "
          "method's own: unselected tasks for whole tasks, all tasks for "
          "items)")
    print("noise model, sum e² / sum Var e (1 = calibrated) | part of the "
          "squared error common to all validation models:")
    for m in NOISE_METHODS:
        print(f"  {METHOD_LABEL[m]:<44}" + "".join(
            f"{_nan_mean_std(out[f'{m}_noise_calib'][ok, b])[0]:>6.2f}|"
            f"{_nan_mean_std(out[f'{m}_noise_shared'][ok, b])[0]:<5.2f}"
            for b in range(len(budgets))))
    print("mean questions used by the whole-task methods: " + "; ".join(
        f"{m} " + "/".join(f"{np.mean(out[f'{m}_cost'][ok, b]):.0f}"
                           for b in range(len(budgets)))
        for m in BENCH_METHODS))
    print("best-singleton fallback won (budget × split cells): " + ", ".join(
        f"{m} {int(out[f'{m}_singleton'].sum())}" for m in SINGLETON_METHODS)
        + "; index ties: " + ", ".join(
            f"{m} {int(out[f'{m}_index_ties'].sum())}"
            for m in BENCH_METHODS)
        + f"; clipped validation cells: {int(out['n_clipped'].sum())}")

    lookup = {(r["budget"], r["metric"], r["method_a"], r["method_b"]): r
              for r in paired}
    print(f"\n== {display_name(ds_name)}: paired R² differences a − b over "
          f"all tasks (mean; * = Nadeau–Bengio 95% CI excludes 0) ==")
    print(f"{'pair':<32}" + "".join(f"{q:>10}" for q in budgets))
    for ma, mb in KEY_PAIRS:
        cells = []
        for q in budgets:
            r = lookup[(q, "r2", ma, mb)]
            cells.append("n/a" if r["n_splits"] == 0 else
                         f"{r['mean_diff']:+.3f}"
                         f"{'*' if _significant(r) else ' '}")
        print(f"{ma + ' − ' + mb:<32}" + "".join(f"{c:>10}" for c in cells))


def plot_budget(ds_name, out, ymin=-0.5):
    """Mean R² (± std over splits) vs Q, all tasks / tasks not observed
    exactly; means below ymin are drawn at the lower edge (open markers)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    x = np.asarray(out["budgets"], dtype=float)
    ok = ~out["estimation_failed"]
    for ax, metric in zip(axes, METRICS):
        for m in METHODS:
            mus, sds = _nan_mean_std(out[f"{m}_{metric}"][ok], axis=0)
            lw = 1.5 if m in PLOT_EMPHASIS else 1.0
            low = mus < ymin
            shown = np.where(low, ymin, mus)
            ax.plot(x, shown, METHOD_STYLE[m], color=METHOD_COLOR[m],
                    label=METHOD_LABEL[m]
                    + (" (below axis at small Q)" if low.any() else ""),
                    markersize=4, linewidth=lw)
            if low.any():
                ax.plot(x[low], shown[low], "v", color=METHOD_COLOR[m],
                        markersize=6, markerfacecolor="white")
            ax.fill_between(x, np.maximum(mus - sds, ymin),
                            np.maximum(mus + sds, ymin),
                            color=METHOD_COLOR[m], alpha=0.10)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(q):,}" for q in x], rotation=45,
                           fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("Question budget $Q$ per model")
        ax.set_title(f"$R^2$ over {METRIC_LABEL[metric]}"
                     + (" (each method's own tasks)" if metric == "r2_unobs"
                        else ""), fontsize=9)
        ax.set_ylim(ymin, 1.0)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Imputation $R^2$ (pooled, standardized)")
    handles, labels = axes[0].get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out_path = OUT / f"item_budget_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}", flush=True)


# ── Fixed per-task overhead (--fixed-cost) ───────────────────────

def fixed_budget_grid(budgets, n_items, fixed_costs):
    """The default budget grid of a --fixed-cost run: `budgets` plus the
    cheapest whole task min_t N_t + C0 of every level C0 (sorted, unique
    ints), the budget from which the whole-task methods buy anything."""
    cheapest = int(np.min(n_items))
    return sorted({int(q) for q in budgets}
                  | {cheapest + int(c0) for c0 in fixed_costs})


def run_fixed(ds_name, budgets, fixed_costs, n_repeats=N_REPEATS,
              n_folds=N_FOLDS, data=None, bounds=None):
    """Every method, overhead level C0 and budget on the matched splits of
    one dataset (data: an ItemData, default load_dataset(ds_name)).  All
    levels share each split's training estimates, random task order and
    item permutation.  Returns the save dict of LOGS/item_fixed_{ds}.npz
    (module docstring); bounds as in run."""
    if data is None:
        data = load_dataset(ds_name)
        if bounds is None:
            bounds = dataset_bounds(ds_name, data.benchmark_names, DATA)
    M, N = data.B.shape
    n_items_total = len(data.item_task)
    budgets = [int(q) for q in budgets]
    fixed_costs = [int(c) for c in fixed_costs]
    if any(q < 1 for q in budgets) or any(c < 0 for c in fixed_costs):
        raise ValueError("budgets must be >= 1 and fixed costs >= 0")
    print(f"{display_name(ds_name)}: {M} models × {N} tasks, "
          f"{n_items_total} items; fixed overhead per task C0 = "
          f"{fixed_costs} item-equivalents; budgets Q: {budgets}",
          flush=True)
    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED))
    S, nf = len(splits), len(fixed_costs)
    print(f"{n_repeats} × {n_folds}-fold CV = {S} matched splits, "
          f"{HOLDOUT_FRAC:.0%} holdout; item samples: "
          f"Split.seed(SEED + {ITEM_SEED_OFFSET}), walked skip-and-continue "
          f"under the overhead", flush=True)
    rows = []
    for s in splits:
        t0 = time.time()
        try:
            levels = [evaluate_split(s, data, budgets, fixed_cost=c0,
                                     bounds=bounds)
                      for c0 in fixed_costs]
        except Exception as e:  # noqa: BLE001 (logged, row left NaN)
            print(f"  rep {s.rep} fold {s.fold}: estimation failed: {e}",
                  flush=True)
            levels = None
        rows.append(levels)
        if levels is not None:
            print(f"  rep {s.rep} fold {s.fold} ({time.time() - t0:.1f}s): "
                  "R² at Q=" + f"{budgets[-1]}: " + "  ".join(
                      f"C0={c0}: B-Tr {lv['bench_trace_r2'][-1]:.3f} "
                      f"B-FL {lv['bench_facloc_r2'][-1]:.3f} "
                      f"I-Cent {lv['items_center_r2'][-1]:.3f}"
                      for c0, lv in zip(fixed_costs, levels)), flush=True)
    template = next((r for r in rows if r is not None), None)
    if template is None:
        raise RuntimeError(f"{ds_name}: every split failed")
    out = {"budgets": np.array(budgets, dtype=np.int64),
           "fixed_costs": np.array(fixed_costs, dtype=np.int64),
           "total_cost": np.array([n_items_total + N * c0
                                   for c0 in fixed_costs], dtype=np.int64),
           "min_task_cost": np.array([int(data.n_items.min()) + c0
                                      for c0 in fixed_costs], dtype=np.int64),
           "costs": data.n_items.astype(np.int64),
           "benchmark_names": np.array([str(b) for b in
                                        data.benchmark_names]),
           "task_binary": data.binary.copy(),
           "methods": np.array(METHODS)}
    for key in template[0]:
        if key in LEVEL_FREE_KEYS:
            val = np.asarray(template[0][key])
            for lv in template[1:]:
                if not np.array_equal(np.asarray(lv[key]), val):
                    raise RuntimeError(f"{key} differs across levels")
            shape, get = val.shape, (lambda r, k=key: r[0][k])
        else:
            val = np.stack([np.asarray(lv[key]) for lv in template])
            shape = val.shape
            get = (lambda r, k=key: np.stack([np.asarray(lv[k])
                                              for lv in r]))
        fill = (np.nan if val.dtype.kind == "f"
                else -1 if key.endswith("_selected") else 0)
        arr = np.full((S,) + shape, fill, dtype=val.dtype)
        for i, r in enumerate(rows):
            if r is not None:
                arr[i] = get(r)
        out[key] = arr
    out.update(
        rep=np.array([s.rep for s in splits], dtype=np.int64),
        fold=np.array([s.fold for s in splits], dtype=np.int64),
        split_id=np.array([s.split_id for s in splits], dtype=np.int64),
        n_train=np.array([len(s.train_idx) for s in splits], dtype=np.int64),
        n_val=np.array([len(s.val_idx) for s in splits], dtype=np.int64),
        estimation_failed=np.array([r is None for r in rows]),
        holdout_frac=np.float64(HOLDOUT_FRAC), n_folds=np.int64(n_folds),
        n_repeats=np.int64(n_repeats), min_train_obs=np.int64(MIN_TRAIN_OBS),
        min_target_obs=np.int64(cv_split.MIN_TARGET_OBS),
        pred_clip=np.bool_(score_bounds.PRED_CLIP),
        seed=np.int64(SEED), item_seed_offset=np.int64(ITEM_SEED_OFFSET),
        random_seed_offset=np.int64(RANDOM_SEED_OFFSET),
        impute_ridge=np.float64(IMPUTE_RIDGE), z_clip=np.float64(Z_CLIP),
        n_items_total=np.int64(n_items_total), n_models=np.int64(M))
    return out


def fixed_paired_rows(ds_name, out):
    """paired_stats of a − b for every level, budget, metric and
    comparable pair (paired_comparable), as CSV rows (FIXED_PAIRED_FIELDS)."""
    rows = []
    for f, c0 in enumerate(out["fixed_costs"]):
        for b, Q in enumerate(out["budgets"]):
            for metric in METRICS:
                for ma, mb in method_pairs():
                    if not paired_comparable(metric, ma, mb):
                        continue
                    st = paired_stats(out[f"{ma}_{metric}"][:, f, b],
                                      out[f"{mb}_{metric}"][:, f, b],
                                      out["n_train"], out["n_val"],
                                      out["rep"])
                    rows.append({"dataset": ds_name,
                                 "holdout_frac": HOLDOUT_FRAC,
                                 "fixed_cost": int(c0), "budget": int(Q),
                                 "metric": metric, "method_a": ma,
                                 "method_b": mb, **st})
    return rows


def _fixed_lookup(paired):
    return {(r["fixed_cost"], r["budget"], r["metric"], r["method_a"],
             r["method_b"]): r for r in paired}


def crossover(budgets, diffs, ci_low=None):
    """Where a whole-task method overtakes an item method along an
    increasing budget grid (module docstring, Crossover).

    diffs: mean paired R² differences A − B at the budgets (NaN counts as
    not ahead); ci_low: their 95% lower bounds (NaN: not significant).
    Returns {state, q_first, q_first_lo, q_stay, q_sig, q_sig_stay,
    n_ahead, n_sig} (NaN where undefined).  Grid values only: A's R² is a
    step function of Q, so d is not interpolated; the first crossing lies
    in (q_first_lo, q_first]."""
    q = np.asarray(budgets, dtype=float)
    d = np.asarray(diffs, dtype=float)
    lo = (np.full(len(d), np.nan) if ci_low is None
          else np.asarray(ci_low, dtype=float))
    if len(q) == 0 or len(d) != len(q) or len(lo) != len(q):
        raise ValueError("budgets, diffs and ci_low must match")
    if np.any(np.diff(q) <= 0):
        raise ValueError("budgets must increase")
    ahead = np.isfinite(d) & (d > 0)
    sig = np.isfinite(lo) & (lo > 0)

    def first_of_tail(mask):
        """Smallest grid Q from which `mask` holds to the end (NaN)."""
        if not mask[-1]:
            return math.nan
        i = len(mask) - 1
        while i > 0 and mask[i - 1]:
            i -= 1
        return float(q[i])

    res = {"state": 1, "q_first": math.nan, "q_first_lo": math.nan,
           "q_stay": first_of_tail(ahead), "q_sig": math.nan,
           "q_sig_stay": first_of_tail(sig), "n_ahead": int(ahead.sum()),
           "n_sig": int(sig.sum())}
    if sig.any():
        res["q_sig"] = float(q[np.argmax(sig)])
    if ahead.any():
        i = int(np.argmax(ahead))
        res["q_first"] = float(q[i])
        if i == 0:
            res["state"] = -1
        else:
            res.update(state=0, q_first_lo=float(q[i - 1]))
    return res


def crossover_table(out, paired):
    """[{fixed_cost, method_a, method_b, **crossover(...)}] for every level
    and CROSS_PAIRS pair (r2, all tasks)."""
    lookup = _fixed_lookup(paired)
    budgets = [int(q) for q in out["budgets"]]
    rows = []
    for c0 in out["fixed_costs"]:
        for ma, mb in CROSS_PAIRS:
            rs = [lookup[(int(c0), q, "r2", ma, mb)] for q in budgets]
            rows.append({"fixed_cost": int(c0), "method_a": ma,
                         "method_b": mb, **crossover(
                             budgets, [r["mean_diff"] for r in rs],
                             [r["ci_low"] for r in rs])})
    return rows


def overhead_share(out, m):
    """(n_splits, n_levels, n_budgets) share of method m's true cost spent
    on the overhead, (cost − questions) / cost = C0 × tasks touched / cost;
    NaN where m buys nothing (cost 0).  The item methods share one item
    sample, so theirs are equal."""
    cost = np.asarray(out[f"{m}_cost"], dtype=float)
    over = cost - np.asarray(out[f"{m}_questions"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cost > 0, over / cost, np.nan)


def fixed_summary(out):
    """{C0: {metric: {method: {Q: {mean, std, n}}}}} for r2, r2_unobs,
    n_obs, n_exact, questions, cost and overhead_share (over the
    estimated splits)."""
    budgets = [int(q) for q in out["budgets"]]
    ok = ~out["estimation_failed"]
    shares = {m: overhead_share(out, m) for m in METHODS}
    tab = {}
    for f, c0 in enumerate(out["fixed_costs"]):
        level = {}
        for metric in METRICS + ["n_obs", "n_exact", "questions", "cost",
                                 "overhead_share"]:
            level[metric] = {}
            for m in METHODS:
                arr = np.asarray(shares[m] if metric == "overhead_share"
                                 else out[f"{m}_{metric}"],
                                 dtype=float)[ok, f]
                mu, sd = _nan_mean_std(arr, axis=0)
                n = np.sum(np.isfinite(arr), axis=0)
                level[metric][m] = {
                    str(q): {"mean": _json_num(mu[b]), "std": _json_num(sd[b]),
                             "n": int(n[b])} for b, q in enumerate(budgets)}
        tab[str(int(c0))] = level
    return tab


def fixed_macro_dict(ds_name, out, paired, cross, undefined=None):
    """Flat {MacroName: finite number} of a --fixed-cost run (module
    docstring); undefined statistics are left out and their names
    appended to `undefined` if given."""
    ds, P = camel(ds_name), FIXED_MACRO_PREFIX
    budgets = [int(q) for q in out["budgets"]]
    ok = ~out["estimation_failed"]
    mac = {}

    def put(name, value):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name!r} is not letters only")
        if name in mac:
            raise ValueError(f"duplicate macro {name}")
        v = _json_num(value)
        if v is None:
            if undefined is not None:
                undefined.append(name)
        else:
            mac[name] = v

    viol = sum(int(np.sum(out[f"{m}_cost"][ok]
                          > np.asarray(out["budgets"])[None, None, :]))
               for m in METHODS)
    put(f"{P}NSplits{ds}", int(ok.sum()))
    put(f"{P}NBudgets{ds}", len(budgets))
    put(f"{P}NLevels{ds}", len(out["fixed_costs"]))
    put(f"{P}MinQ{ds}", min(budgets))
    put(f"{P}MaxQ{ds}", max(budgets))
    put(f"{P}Violations{ds}", viol)
    lookup = _fixed_lookup(paired)
    # the item methods share one sample: one overhead share, tag "Items"
    share_methods = [(m, camel(m)) for m in BENCH_METHODS] + [
        (ITEM_METHODS[0], "Items")]
    shares = {m: overhead_share(out, m) for m, _ in share_methods}
    for f, c0 in enumerate(out["fixed_costs"]):
        C_ = "C" + num_words(int(c0))
        put(f"{P}Overhead{ds}{C_}", int(c0))
        put(f"{P}Total{ds}{C_}", int(out["total_cost"][f]))
        put(f"{P}MinTaskCost{ds}{C_}", int(np.min(out["costs"])) + int(c0))
        for m in METHODS:
            M_ = camel(m)
            for b, q in enumerate(budgets):
                Q_ = "Q" + num_words(q)
                put(f"{P}Rsq{ds}{M_}{C_}{Q_}",
                    _nan_mean_std(out[f"{m}_r2"][ok, f, b])[0])
                put(f"{P}Size{ds}{M_}{C_}{Q_}",
                    float(np.mean(out[f"{m}_n_obs"][ok, f, b])))
        for m, M_ in share_methods:
            for b, q in enumerate(budgets):
                Q_ = "Q" + num_words(q)
                put(f"{P}OverheadShare{ds}{M_}{C_}{Q_}",
                    _nan_mean_std(shares[m][ok, f, b])[0])
        for b, q in enumerate(budgets):
            put(f"{P}Questions{ds}Items{C_}Q{num_words(q)}", float(np.mean(
                out[f"{ITEM_METHODS[0]}_questions"][ok, f, b])))
        for ma, mb in FIXED_KEY_PAIRS:
            pair = f"{camel(ma)}Vs{camel(mb)}"
            n_pos = n_neg = 0
            for q in budgets:
                r = lookup[(int(c0), q, "r2", ma, mb)]
                for suffix, (field, _) in PAIR_STATS.items():
                    put(f"{P}Pair{suffix}{ds}{pair}{C_}Q{num_words(q)}",
                        r[field])
                if _significant(r):
                    n_pos += int(r["ci_low"] > 0)
                    n_neg += int(r["ci_high"] < 0)
            put(f"{P}SigPos{ds}{pair}{C_}", n_pos)
            put(f"{P}SigNeg{ds}{pair}{C_}", n_neg)
    for row in cross:
        pair = f"{camel(row['method_a'])}Vs{camel(row['method_b'])}"
        C_ = "C" + num_words(int(row["fixed_cost"]))
        for suffix, field in CROSS_FIELDS:
            put(f"{P}Cross{suffix}{ds}{pair}{C_}", row[field])
    return mac


FIXED_NOTES = {
    "cost_model": "budget Q in item-equivalents; whole tasks cost N_t + C0 "
                  "each, random items 1 each plus C0 per task touched",
    "item_sample": "the split's item permutation (seed tag SEED + 18) walked "
                   "skip-and-continue: an item of a touched task costs 1, "
                   "the first item of a new task 1 + C0; within each task "
                   "the sample is a simple random sample given n_t; C0 = 0 "
                   "is the first Q items (the plain run)",
    "bench": "budgeted_select / order_fill on the costs N_t + C0 (gain / "
             "cost ratio, tie rule and best-singleton fallback see C0); "
             "random task order = Split.seed(SEED + 7)",
    "metric": "r2 over all tasks in train-standardized units (one target "
              "set, so whole tasks and items compare directly); bought "
              "tasks count with their exact values",
    "crossover": "per bench x item pair and C0: state (-1 ahead at the "
                 "smallest Q, 0 crosses inside the grid, 1 never ahead), "
                 "q_first (first grid Q with mean paired difference > 0; "
                 "grid values only, not interpolated: the whole-task R^2 "
                 "is a step function of Q and exactly 0 below the "
                 "cheapest task), q_first_lo (the grid Q before it; the "
                 "first crossing lies in (q_first_lo, q_first]), q_stay "
                 "(ahead at every larger grid Q), q_sig / q_sig_stay (95% "
                 "Nadeau-Bengio interval above 0)",
    "budget_grid": "default: half-octave grid 50..25,600 plus the cheapest "
                   "whole task min_t N_t + C0 of every level; an explicit "
                   "--budgets grid is used as given",
    "item_baseline": "the uniform item sample ignores the overhead (user "
                     "decision: uniform random items): it touches a new task "
                     "with almost every early item and pays C0 for each, so "
                     "under overhead most of its budget goes to overhead "
                     "(summary overhead_share), and the questions it buys "
                     "(and the item R^2) are not monotone in Q and need not "
                     "be nested across budgets for C0 > 0; the crossovers "
                     "say where whole tasks beat this item design, not an "
                     "overhead-aware one (e.g. a few random tasks, random "
                     "items within them), which is not run",
    "human_evaluation": "the burden of human evaluation is a further "
                        "per-task cost; no data, not modelled",
}


def write_fixed_json(path, ds_name, out, paired, cross, runtime_sec, argv,
                     added_budgets=()):
    """Config, notes, summary per level, key pairs, crossovers, macros.
    added_budgets: the budgets fixed_budget_grid added to the grid (the
    cheapest whole task of a level that was not a grid point)."""
    undefined = []
    macros = fixed_macro_dict(ds_name, out, paired, cross, undefined)
    ok = ~out["estimation_failed"]
    viol = {m: int(np.sum(out[f"{m}_cost"][ok]
                          > np.asarray(out["budgets"])[None, None, :]))
            for m in METHODS}
    key_rows = [r for r in paired if r["metric"] == "r2"
                and (r["method_a"], r["method_b"]) in FIXED_KEY_PAIRS]
    doc = {
        "script": "eval_item_budget.py --fixed-cost",
        "dataset": ds_name,
        "argv": list(argv),
        "config": {
            "budgets": [int(q) for q in out["budgets"]],
            "budgets_added_min_task": [int(q) for q in added_budgets],
            "fixed_costs": [int(c) for c in out["fixed_costs"]],
            "total_cost": [int(c) for c in out["total_cost"]],
            "min_task_cost": [int(np.min(out["costs"])) + int(c)
                              for c in out["fixed_costs"]],
            "methods": METHODS,
            "key_pairs": [list(p) for p in FIXED_KEY_PAIRS],
            "cross_pairs": [list(p) for p in CROSS_PAIRS],
            "n_repeats": int(out["n_repeats"]), "n_folds": int(out["n_folds"]),
            "holdout_frac": HOLDOUT_FRAC, "seed": SEED,
            "min_train_obs": MIN_TRAIN_OBS,
            "min_target_obs": int(cv_split.MIN_TARGET_OBS),
            "pred_clip": bool(score_bounds.PRED_CLIP),
            "random_seed_tag": SEED + RANDOM_SEED_OFFSET,
            "item_seed_tag": SEED + ITEM_SEED_OFFSET,
            "impute_ridge": IMPUTE_RIDGE, "z_clip": Z_CLIP,
            "n_items_total": int(out["n_items_total"]),
        },
        "notes": FIXED_NOTES,
        "checks": {"violations": viol,
                   "violations_total": sum(viol.values()),
                   "estimation_failed": int(np.sum(out["estimation_failed"]))},
        "summary": fixed_summary(out),
        "paired_key_pairs": [{k: (_json_num(v) if isinstance(v, float) else v)
                              for k, v in r.items()} for r in key_rows],
        "crossover": [{k: (_json_num(v) if isinstance(v, float) else v)
                       for k, v in r.items()} for r in cross],
        "runtime_sec": round(float(runtime_sec), 1),
        "n_macros": len(macros),
        "macros": macros,
        "macros_undefined": sorted(undefined),
    }
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)
    print(f"Summary saved to {path} ({len(macros)} macros, {len(undefined)} "
          f"undefined statistics left out)", flush=True)
    return doc


def _q_text(v):
    return "-" if not np.isfinite(v) else f"{v:,.0f}"


def _bracket_text(r):
    """The first crossing's grid bracket: '(lo,first]' inside the grid,
    '<=first' if ahead at the smallest Q, '-' if never ahead."""
    if r["state"] == 1:
        return "-"
    if r["state"] == -1:
        return f"<={_q_text(r['q_first'])}"
    return f"({_q_text(r['q_first_lo'])},{_q_text(r['q_first'])}]"


def print_fixed_summary(ds_name, out, paired, cross):
    budgets = [int(q) for q in out["budgets"]]
    ok = ~out["estimation_failed"]
    S = int(ok.sum())
    lookup = _fixed_lookup(paired)
    item_share = overhead_share(out, ITEM_METHODS[0])
    for f, c0 in enumerate(out["fixed_costs"]):
        c0 = int(c0)
        print(f"\n== {display_name(ds_name)}, overhead C0 = {c0} "
              f"item-equivalents per task: pooled R² over all tasks / "
              f"tasks observed vs budget Q (mean over {S} splits) ==")
        print(f"{'method':<16}" + "".join(f"{q:>11}" for q in budgets))
        for m in METHODS:
            cells = []
            for b in range(len(budgets)):
                mu = _nan_mean_std(out[f"{m}_r2"][ok, f, b])[0]
                cells.append(f"{mu:.3f}/"
                             f"{np.mean(out[f'{m}_n_obs'][ok, f, b]):.0f}")
            print(f"{METHOD_SHORT[m]:<16}" + "".join(f"{c:>11}"
                                                     for c in cells))
        cells = []
        for b in range(len(budgets)):
            n_q = np.mean(out[f"{ITEM_METHODS[0]}_questions"][ok, f, b])
            sh = _nan_mean_std(item_share[ok, f, b])[0]
            cells.append(f"{n_q:.0f}/" + ("-" if not np.isfinite(sh)
                                          else f"{100 * sh:.0f}%"))
        print(f"{'items: q/ovh':<16}" + "".join(f"{c:>11}" for c in cells))
        print("(items: q/ovh = questions bought by the uniform item sample "
              "shared by the item methods / share of its cost spent on the "
              "overhead, C0 × tasks touched / cost)")
        print("paired R² differences a − b (mean; * = Nadeau–Bengio 95% CI "
              "excludes 0):")
        for ma, mb in CROSS_PAIRS:
            if mb == "items_only":
                continue
            cells = []
            for q in budgets:
                r = lookup[(c0, q, "r2", ma, mb)]
                cells.append("n/a" if r["n_splits"] == 0 else
                             f"{r['mean_diff']:+.3f}"
                             f"{'*' if _significant(r) else ' '}")
            print(f"  {METHOD_SHORT[ma] + ' − ' + METHOD_SHORT[mb]:<14}"
                  + "".join(f"{c:>11}" for c in cells))
    levels = [int(c) for c in out["fixed_costs"]]
    print(f"\n== {display_name(ds_name)}: budget Q at which whole tasks "
          f"overtake items (q_stay: ahead from there on / (q_first_lo, "
          f"q_first]: grid bracket of the first crossing, not interpolated, "
          f"'<=Q' = ahead at the smallest Q / q_sig: first significant win; "
          f"'-' = never) ==")
    w = 28
    print(f"{'pair':<16}" + "".join(f"{'C0=' + str(c):>{w}}" for c in levels))
    print(f"{'cheapest task':<16}" + "".join(
        f"{int(np.min(out['costs'])) + c:>{w},}" for c in levels))
    for ma, mb in CROSS_PAIRS:
        cells = []
        for c0 in levels:
            r = next(x for x in cross if x["fixed_cost"] == c0
                     and x["method_a"] == ma and x["method_b"] == mb)
            cells.append(f"{_q_text(r['q_stay'])}/{_bracket_text(r)}/"
                         f"{_q_text(r['q_sig'])}")
        print(f"{METHOD_SHORT[ma] + ' vs ' + METHOD_SHORT[mb]:<16}"
              + "".join(f"{c:>{w}}" for c in cells))
    print("(whole tasks score exactly R² = 0 below the cheapest task, so a "
          "bracket ending at the cheapest task puts the crossing at that "
          "cost unless the item method's R² is negative somewhere below it; "
          "the item baseline ignores the overhead, see 'items: q/ovh')")
    viol = sum(int(np.sum(out[f"{m}_cost"][ok]
                          > np.asarray(out["budgets"])[None, None, :]))
               for m in METHODS)
    print(f"checks: budget violations {viol} (every method, level, budget "
          f"and split); estimation failed on "
          f"{int(out['estimation_failed'].sum())} splits")


def plot_fixed(ds_name, out, cross, ymin=0.0, path=None):
    """R² over all tasks vs Q per overhead level (FIXED_PLOT_METHODS), and
    the crossover budgets of every whole-task method vs items_center
    against C0 (q_stay solid, q_sig dotted)."""
    levels = [int(c) for c in out["fixed_costs"]]
    n_panels = len(levels) + 1
    n_cols = 3
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols,
                                                      3.2 * n_rows),
                             squeeze=False)
    axes = axes.ravel()
    x = np.asarray(out["budgets"], dtype=float)
    ok = ~out["estimation_failed"]
    for f, c0 in enumerate(levels):
        ax = axes[f]
        for m in FIXED_PLOT_METHODS:
            mus = _nan_mean_std(out[f"{m}_r2"][ok, f], axis=0)[0]
            ax.plot(x, np.maximum(mus, ymin), METHOD_STYLE[m],
                    color=METHOD_COLOR[m], label=METHOD_LABEL[m],
                    markersize=3, linewidth=1.5 if m in PLOT_EMPHASIS
                    else 1.0)
        ax.set_xscale("log")
        ax.set_ylim(ymin, 1.0)
        ax.set_title(f"overhead $C_0$ = {c0} items per task", fontsize=9)
        ax.set_xlabel("Budget $Q$ (item-equivalents per model)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Imputation $R^2$ (all tasks)", fontsize=8)
    ax = axes[len(levels)]
    for m in BENCH_METHODS:
        for field, style in (("q_stay", "-"), ("q_sig", ":")):
            ys = [next(r for r in cross if r["fixed_cost"] == c0
                       and r["method_a"] == m
                       and r["method_b"] == "items_center")[field]
                  for c0 in levels]
            ax.plot(levels, ys, METHOD_STYLE[m][0] + style,
                    color=METHOD_COLOR[m], markersize=4,
                    label=(f"{METHOD_SHORT[m]} vs I-Cent"
                           if field == "q_stay" else None))
    first = [int(np.min(out["costs"])) + c0 for c0 in levels]
    ax.plot(levels, first, ":", color="black", linewidth=0.8,
            label="cheapest whole task ($\\min_t N_t + C_0$)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Overhead $C_0$ (item-equivalents per task)", fontsize=8)
    ax.set_ylabel("Crossover budget $Q$", fontsize=8)
    ax.set_title("whole tasks overtake items_center\n(solid: stays ahead, "
                 "dotted: first significant)", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)
    for ax in axes[n_panels:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    path = OUT / f"{FIXED_STEM}_{ds_name}.pdf" if path is None else path
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)
    return path


def main_fixed(datasets, budgets, fixed_costs, n_repeats, n_folds, argv,
               add_min_task=False):
    """--fixed-cost: run_fixed per dataset, then its CSV, JSON and figure.
    add_min_task (the default grid): the budgets of each dataset are
    fixed_budget_grid(budgets, its task sizes, fixed_costs)."""
    for ds_name in datasets:
        t0 = time.time()
        data = load_dataset(ds_name)
        grid = (fixed_budget_grid(budgets, data.n_items, fixed_costs)
                if add_min_task else [int(q) for q in budgets])
        added = sorted(set(grid) - {int(q) for q in budgets})
        if added:
            print(f"budgets added to the grid (the cheapest whole task "
                  f"min_t N_t + C0 of a level): {added}", flush=True)
        out = run_fixed(ds_name, grid, fixed_costs, n_repeats, n_folds,
                        data=data)
        paired = fixed_paired_rows(ds_name, out)
        cross = crossover_table(out, paired)
        print_fixed_summary(ds_name, out, paired, cross)
        plot_fixed(ds_name, out, cross)
        stem = f"{FIXED_STEM}_{ds_name}"
        np.savez_compressed(LOGS / f"{stem}.npz", **out)
        print(f"Results saved to {LOGS / f'{stem}.npz'}", flush=True)
        write_csv(paired, LOGS / f"{stem}_paired.csv", FIXED_PAIRED_FIELDS)
        print(f"Paired statistics saved to {LOGS / f'{stem}_paired.csv'}",
              flush=True)
        write_fixed_json(LOGS / f"{stem}.json", ds_name, out, paired, cross,
                         time.time() - t0, argv, added_budgets=added)
        print(f"{display_name(ds_name)} completed in {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Whole benchmarks chosen by knapsack selection vs "
                    "uniformly random items under a matched question "
                    "budget, on the matched repeated-CV splits.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names, each with data/<name>.matrix.npz"
                             " and data/<name>.items.npz (default: "
                             "%(default)s)")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=None,
                        help="question budgets Q per model (default: "
                             f"{DEFAULT_BUDGETS}; with --fixed-cost: "
                             f"{FIXED_BUDGETS} plus the cheapest whole task "
                             "min_t N_t + C0 of every level, in "
                             "item-equivalents; an explicit grid is used "
                             "as given)")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions; splits = repeats × folds, "
                             "rep 0 = the original seed-42 folds "
                             "(default: %(default)s)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument(
        "--fixed-cost", nargs="*", type=int, default=None, metavar="C0",
        help="opt-in fixed overhead of C0 item-equivalents per task "
             "touched (whole tasks: N_t + C0 each; random items: 1 each "
             "plus C0 per task with a sampled item), one level per C0 "
             "(flag without values: "
             + " ".join(str(c) for c in DEFAULT_FIXED_COSTS)
             + "); writes LOGS/item_fixed_<ds>.{npz,json}, "
             "LOGS/item_fixed_<ds>_paired.csv and OUT/item_fixed_<ds>.pdf "
             "with the crossover budgets, instead of the item_budget_* "
             "outputs (default: off)")
    args = parser.parse_args()
    if args.repeats < 1 or args.folds < 2:
        parser.error("need --repeats >= 1 and --folds >= 2")
    default_grid = args.budgets is None
    if args.budgets is None:
        args.budgets = (DEFAULT_BUDGETS if args.fixed_cost is None
                        else FIXED_BUDGETS)
    if any(q < 1 for q in args.budgets):
        parser.error("--budgets must be positive integers")
    if len(set(args.budgets)) != len(args.budgets):
        parser.error("--budgets must be distinct")
    if args.fixed_cost is not None:
        if any(c < 0 for c in args.fixed_cost):
            parser.error("--fixed-cost overheads must be >= 0")
        if len(set(args.fixed_cost)) != len(args.fixed_cost):
            parser.error("--fixed-cost overheads must be distinct")
    missing = [ds for ds in args.datasets
               if not (matrix_path(ds).is_file() and items_path(ds).is_file())]
    if missing:
        avail = sorted(p.name[:-len(".items.npz")]
                       for p in DATA.glob("*.items.npz"))
        parser.error(f"no {DATA}/<name>.matrix.npz + <name>.items.npz for "
                     f"{missing}; available: {' '.join(avail)}")
    budgets = sorted(args.budgets)
    if args.fixed_cost is not None:
        main_fixed(args.datasets, budgets,
                   sorted(args.fixed_cost or DEFAULT_FIXED_COSTS),
                   args.repeats, args.folds, sys.argv,
                   add_min_task=default_grid)
        sys.exit(0)

    for ds_name in args.datasets:
        t0 = time.time()
        out = run(ds_name, budgets, args.repeats, args.folds)
        paired = paired_rows(ds_name, out)
        print_summary(ds_name, out, paired)
        plot_budget(ds_name, out)
        np.savez(LOGS / f"item_budget_{ds_name}.npz", **out)
        print(f"Results saved to {LOGS / f'item_budget_{ds_name}.npz'}",
              flush=True)
        write_csv(paired, LOGS / f"item_budget_{ds_name}_paired.csv",
                  PAIRED_FIELDS)
        print(f"Paired statistics saved to "
              f"{LOGS / f'item_budget_{ds_name}_paired.csv'}", flush=True)
        write_json(LOGS / f"item_budget_{ds_name}.json", ds_name, out, paired,
                   time.time() - t0, sys.argv)
        print(f"{display_name(ds_name)} completed in {time.time() - t0:.0f}s",
              flush=True)
