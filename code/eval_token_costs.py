"""Token-cost budgets on HELM: does the selector's cost model matter?

When the true cost of running a benchmark is the number of tokens it
processes, how much does it matter which cost model the budgeted selector
uses: the true token cost, the instance count (the proxy of the paper's
cost experiment, eval_cost_budget.py) or a uniform cost (cardinality,
cost-blind)?  Every final selection is held to the TRUE token budget.

Datasets (data/<ds>.matrix.npz + data/<ds>.meta.json, data/build_helm.py):
helm_lite (10 core scenarios), helm_mmlu (57 MMLU subjects), helm_classic
(16 scenarios, 82% observed).  Costs are per benchmark, from
meta.benchmark_cost (means over the models with an observed cell):
  tokens     the tokens one run of the benchmark processes on a model:
             cost_tokens_requests_mean = cost_tokens_mean (sum over the
             scenario's sub-runs of '# eval' x ('# prompt tokens' + '#
             output tokens'), one in-context trial, each model's own
             tokenizer: ONE request per instance) x requests_per_instance.
             HELM Classic's multiple_choice_separate and ranking scenarios
             send several requests per instance (HellaSwag 4, OpenbookQA 8,
             MS MARCO (regular) 30.5, MS MARCO (TREC) 224.6); on helm_lite
             and helm_mmlu every run sends one and the two columns are
             equal.  After the cost caveats below.  The TRUE cost.
  instances  instances_mean: number of evaluation instances (the paper's
             proxy: test questions per subject)
  uniform    1 per benchmark (cardinality)
Budgets: B_f = f x T, T = sum_j tokens_j over all priced benchmarks
(eligible or not; every benchmark unless --cost-caveats exclude), f in
--budgets (default 0.01 0.02 0.05 0.1 0.2 0.3 0.5).

Cost caveats (COST_CAVEATS, --cost-caveats; disclosed in the JSON
"cost_caveats" and the printed summary).  Benchmarks whose published token
cost the data release itself marks as wrong:
  helm_classic OpenbookQA.  meta caveat: "OpenbookQA's reported '# prompt
  tokens' (~5) appears to cover only the calibration requests, so its
  token cost is underestimated even after the request correction".  As
  published it costs 2,598 tokens one request per instance (0.02% of the
  one-request suite) and 20,784 with the request correction (0.06%), so
  it is nearly free: before this fix every method, random included, chose
  it on every split at every budget.  Its core run
  (multiple_choice_separate_calibrated, 0-shot) sends, for each of the 4
  answer choices, one original request (question + choice) and one
  calibration request (the choice alone): 8 per instance, and HELM's '#
  prompt tokens' p = 5.10 is the length of a calibration request.  HELM's
  own multiple-choice ablation of the same 500 questions (v0.4.0
  groups/ablation_multiple_choice.json, table 'General information', 6
  models, 0-shot) reports 15.02-15.42 prompt tokens for
  multiple_choice_separate_original (the original requests alone) and
  5.27-5.44 for _calibrated, a ratio r = 2.85 (median over the models).
  --cost-caveats modes:
    impute (default)  the original requests priced at r x p: per instance
                      4 (r p + o) + 4 (p + o) tokens (o = '# output
                      tokens'), cost_tokens_requests_mean x 1.91 = 39,656
                      tokens, 79 per instance (HELM Lite's 5-shot
                      multiple_choice_joint OpenbookQA: 260); cell costs
                      scale by the same factor; scenario single: one
                      original request per instance, cost_tokens_mean x
                      (r p + o) / (p + o) = 7,316 tokens.
    exclude           flagged benchmarks are not purchasable by any method
                      (random: the shared random order without them); they
                      stay in the correlation matrix and remain imputation
                      targets of both metrics; T sums the priced benchmarks.
    keep              the published costs (the runs before this fix).
  Outputs of a non-default mode get the file suffix _{mode} and the macro
  prefix Tok{Mode} (TokExclude..., TokKeep...).

Selectors (objective o in {mi, trace, entropy, facloc}, cost model c):
  {o}_{c}   design "knap" (primary): the cost-effective greedy of
            eval_cost_budget.budgeted_select (Appendix B.3), with the
            gain/cost ratio gain_o(j | A) / c_j ranking the candidates and
            ratio ties broken as there (explained variance per unit of c,
            then lowest index), while feasibility is the TRUE budget:
            candidates are the selectable (eligible and priced) benchmarks
            whose token cost still fits (spent + tokens_j <= B_f), and the
            best-singleton fallback ranges over the selectable benchmarks
            with tokens_j <= B_f (singleton values do not depend on
            costs).  proxy_budgeted_select
            implements it; for c = tokens it IS the direct knapsack
            budgeted_select(R, tokens, B_f, o), which is checked on every
            split and budget ({scen}_knap_direct_mismatch, expected all
            False).  Objectives as in budgeted_select: shifted entropy, MI
            (heuristic), trace (explained variance of the unselected
            benchmarks) and facloc (the headline method, user decision
            2026-10-19: facility location on the clipped correlation
            matrix of baseline_select.facility_location_select, gain
            sum_j max(s_ij - c_j, 0) over all N targets given the cover
            c_j; normalized monotone submodular, so the knapsack of the
            true cost model carries the (1 - 1/e)/2 guarantee of
            budgeted_select).  The facloc methods are appended after the
            three others (OBJECTIVES) and draw no random numbers, so every
            array of the other methods is unchanged by them.
            design "fill" (robustness): the full cost-effective greedy
            order under c (no budget, no singleton fallback; it stops where
            no positive gain is left, e.g. MI gains turn negative), filled
            under the true budget: order_fill(order, tokens, B_f), i.e.
            "rank under the cost model, then buy what fits".  It differs
            from knap where the order skips a benchmark (later gains are
            conditioned on the skipped one).  The order stops where the
            objective's gains turn non-positive, as budgeted_select does
            under any budget (knap MI selections stop there too); MI gains
            do, so a fill MI order can end before any affordable benchmark
            and select nothing although something fits (helm_lite
            mi_uniform at f = 2%: empty on 25 of 50 splits).
  random    the split's random order (random_select, seed
            Split.seed(SEED + RANDOM_SEED_OFFSET): the random baseline of
            every other script, common random numbers) over the eligible
            priced benchmarks (random_select restricts one permutation of
            all N, so excluding a benchmark leaves the order of the others
            unchanged), filled under the true budget; logged under both
            designs (identical arrays).
No new common-random-number stream is used.

Scenarios (true cost and the "tokens" cost model):
  main      true cost = tokens model = tokens above (the selector knows
            the cost table it is held to).
  cell      sensitivity: the true cost of benchmark j on a split is the mean
            of meta.cell_costs.cost_tokens x m_j over the split's
            VALIDATION models observing j (the cost of running j on the
            held-out models; m_j = requests_per_instance x caveat factor,
            the main multiplier; fallback: training-model mean, then the
            main cost, cell_fallback); budgets B_f,s = f x sum_j true_s,j
            (priced).  No cost model sees the held-out models: tokens is
            the same mean over the TRAINING models observing j (fallback:
            main cost), instances the mean of meta.cell_costs.instances
            over the training models observing j (fallback instances_mean;
            it differs from instances_mean only for helm_lite's LegalBench
            and WMT 2014, whose instance counts differ by model), uniform 1.
  single    sensitivity, only where it differs from main (helm_classic):
            one request per instance as HELM's tables count it, true cost
            = tokens model = cost_tokens_mean (OpenbookQA under impute:
            one original request per instance).  The main scenario of the
            runs before the request correction became primary, in which MS
            MARCO (TREC) (20,913 tokens instead of 4.7M) and HellaSwag
            (82,738 instead of 330,951) looked nearly free.
Feasibility always uses the true cost (the budget is enforced on the
realized spend; only the ranking uses the cost model), so in scenario cell
the selections depend on the validation models through their token counts,
never through their scores: no score leakage.

Matched design: cv_splits(M, --folds, --repeats, 0.1, SEED), default 5 x 10
= 50 splits, rep 0 = the historical seed-42 folds.  On every split all
scenarios, designs, methods and budgets share the train/validation models,
the estimated correlation matrix (eval_entropy_vs_mi.estimate_corr, EM
where the data are incomplete), one eligibility mask eligible_mask(O_train)
(>= MIN_TRAIN_OBS training models; ineligible benchmarks stay in the
correlation matrix and remain imputation targets), one target mask
target_mask(O_train) (protocol rule D1: benchmarks observed by fewer
than cv_split.MIN_TARGET_OBS = 2 training models are never scored, by
either metric) and the random order; every prediction is clipped to its
benchmark's valid score range (score_bounds.py, rule D2: [0, 1] on
HELM).

Metrics (PRIMARY_METRIC = r2_suite; both logged, paired and given macros).
  r2_suite  (primary here, suite_r2): impute_and_evaluate's predictions
            pooled over ALL observed validation cells, a selected cell
            counting as known (residual 0): 1 − SS_res(unselected) /
            SS_tot(all observed), in train-standardized units.  One target
            set for every method and budget, so selections of different
            size compare directly; the empty selection scores 0 (training-
            mean prediction); finite unless the estimate failed.  Because a
            bought benchmark counts as known, it rewards covering many
            cheap benchmarks: each one bought adds its share of the
            standardized suite variance, however well the others would
            have predicted it.
  r2        (secondary: the paper's metric of the other experiments):
            impute_and_evaluate's pooled R² over validation models x
            UNSELECTED observed benchmarks.  Under a budget the selected
            sets differ in size, so r2 scores methods on different target
            sets, and at large budgets only 1-3 targets are left (helm_lite
            at f = 50%: 8 of 10 selected; helm_classic: 14 of 16, where
            every 14-benchmark selection had the same r2), which makes it
            noisy and hard to compare across methods.  NaN where nothing
            was selected (nothing affordable, e.g. helm_lite at f = 1%,
            below its cheapest scenario), where no validation model
            observes a selected benchmark (impute_and_evaluate skips such
            models, e.g. on helm_classic for selections of scenarios run
            for half of the models), or where the estimate failed.
Both are evaluated on the sorted index set of a selection (cached), so
identical sets tie exactly across methods.

Paired statistics (paired_stats.paired_stats, Nadeau-Bengio corrected
t-interval over the matched splits, win / tie / loss) of both metrics
for, at every f, scenario and design: tokens - instances, tokens - uniform
and instances - uniform within each objective, every {o}_{c} - random, and
objective pairs at equal c (mi before entropy before trace before
facloc, paired_stats.method_rank: facloc is unranked there, so it comes
after the other objectives and before random).  A pair uses the splits
where BOTH values are
finite (same splits, same eligibility, so the pairing stays valid), so
for r2 its n can be below the number of splits and can differ between
neighbouring rows; n is the CSV's n_splits column and the Pair...N macro,
and the printed tables show it as [n] (method means) and (n) (paired
differences) wherever it is below the number of estimated splits.

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir();
{sfx} = '' for --cost-caveats impute, else _{mode}, e.g. _exclude):
  LOGS/token_costs_{ds}{sfx}.npz   (savez_compressed; per-split arrays
                              first axis = split, pair on split_id)
    {scen}_{design}_{m}_r2, _r2_suite, _rmse   (n_splits, n_budgets) float
    {scen}_{design}_{m}_size           (n_splits, n_budgets) int, benchmarks
                                       selected
    {scen}_{design}_{m}_cost           (n_splits, n_budgets) true token cost
                                       of the selection
    {scen}_{design}_{m}_violations     (n_splits, n_budgets) bool, cost >
                                       budget (expected all False)
    {scen}_{design}_{m}_selected       (n_splits, n_budgets, N) int, picks in
                                       order, -1 padded
    {scen}_knap_{o}_{c}_singleton      (n_splits, n_budgets) bool fallback won
    {scen}_knap_{o}_{c}_index_ties, _tie_breaks   (n_splits, n_budgets) int
    {scen}_knap_direct_mismatch        (n_splits, n_objectives, n_budgets)
                                       bool, {o}_tokens != budgeted_select
                                       (main, single; expected all False)
    {scen}_budgets                     (n_splits, n_budgets) absolute budgets
    {scen}_true_costs, {scen}_rank_tokens, {scen}_rank_instances
                                       (n_splits, N) true costs / tokens and
                                       instances cost models of each split
    cell_fallback                      (n_splits, N) bool, no validation model
                                       observed the benchmark
    costs_tokens, costs_requests       (N,) published cost_tokens_mean /
                                       cost_tokens_requests_mean
    costs_true_main, costs_true_single, costs_mult_req   (N,) true costs of
                                       main / single and the cell multiplier
                                       m_j, after the cost caveats
    costs_instances, benchmark_names   (N,)
    priced, caveat_flag                (N,) bool: purchasable; flagged by
                                       COST_CAVEATS
    cost_caveat_mode                   'impute' / 'exclude' / 'keep'
    budgets_frac (n_budgets,); scenarios, designs, objectives,
    cost_models, methods (names)
    rep, fold, split_id, n_train, n_val, n_eligible   (n_splits,) int
    eligible (n_splits, N) bool (eligible_mask; selectable = eligible &
    priced); estimation_failed (n_splits,) bool
    holdout_frac, n_folds, n_repeats, min_train_obs, min_target_obs,
    pred_clip, seed, total_tokens (main T), matrix_sha256, meta_sha256,
    runtime_s                         scalars
  LOGS/token_costs_{ds}{sfx}.json   config (with primary_metric and
                              metric_notes), cost table, cost_caveats
                              (mode, flagged benchmarks with published and
                              imputed costs, the ablation data, and every
                              caveat of the meta), correlations (Spearman /
                              Pearson / Pearson of logs, true cost vs
                              instances), summary tables, key paired
                              statistics, the main-knap selection share of
                              every benchmark, checks, runtime, a flat
                              "macros" dict of finite numbers (names below)
                              and "macros_undefined": the names whose value
                              is undefined (NaN: nothing selected or no
                              finite pair at that budget, e.g. every r2
                              macro of helm_lite at Bone), never emitted
                              as macros
  LOGS/token_costs_{ds}{sfx}_paired.csv   every pair above (columns
                              dataset, holdout_frac, scenario, design,
                              metric (r2 / r2_suite), budget_frac, budget
                              (mean over splits), method_a, method_b, then
                              paired_stats.STAT_FIELDS)
  OUT/token_costs{sfx}.pdf     datasets of the invocation (rows); columns:
                              R² vs f per objective (tokens / instances /
                              uniform / random, main knap), then the paired
                              differences tokens - instances and tokens -
                              uniform (95% Nadeau-Bengio band)
  OUT/token_costs_suite{sfx}.pdf   the same for the full-suite R² (the
                              primary metric)

Macros (make_numbers' naming: letters only, {P} = Tok, or Tok{Mode} for a
non-default --cost-caveats mode; {Ds} = camel(ds), e.g. HelmLite; {O} in
Mi, Trace, Entropy, Facloc; {C} in Tokens, Instances, Uniform; B{b} the
budget in percent, spelled: Bone, Btwo, ..., Bfifty; {T} the
scenario/design tag: ''
main knap, Fill main fill, Cell cell knap, Single single knap, each
followed by Suite for the full-suite R², e.g. TokSuiteRsq...,
TokCellSuitePairDiff...; Size and Spent have no Suite; {Bn} =
camel(benchmark name), e.g. OpenbookQa):
  {P}{T}Rsq{Ds}{O}{C}B{b}, {P}{T}Rsq{Ds}RandomB{b}    R² split mean
  {P}RsqStd{Ds}..., {P}SuiteRsqStd{Ds}...   its std (main knap)
  {P}{T}Size{Ds}{O}{C}B{b}          mean number of benchmarks selected
  {P}Spent{Ds}{O}{C}B{b}            mean true cost / budget (main knap)
  {P}MaxCount{Ds}B{b}               most selectable benchmarks the budget
                                    affords (cheapest first, mean over splits)
  {P}{T}Pair{S}{Ds}{A}Vs{B}B{b}     paired R² difference A - B, statistic {S}
                                    in Diff, Se, Lo, Hi, Win, N (as
                                    make_numbers' Pair macros); A, B as
                                    {O}{C} or Random; main: within-objective
                                    cost pairs, vs random and objective pairs
                                    at Tokens; Fill/Cell/Single:
                                    TokensVsInstances and TokensVsUniform
  {P}Spearman{Ds}, {P}Pearson{Ds}, {P}PearsonLog{Ds}   main true cost vs
                                    instances across the priced benchmarks
                                    ({P}Single...: the single-scenario cost)
  {P}NBench{Ds}, {P}NPriced{Ds}, {P}TotalTokens{Ds} (main T),
  {P}SingleTotalTokens{Ds}, {P}TotalInstances{Ds}, {P}CostRatio{Ds} (max /
  min true cost), {P}PerInstanceRatio{Ds} (max / min true cost per
  instance), {P}NSplits{Ds}, {P}Violations{Ds} (all scenarios),
  {P}DirectMismatch{Ds}, {P}CellFallback{Ds}, {P}Empty{Ds}B{b} (splits
  where random, hence every method, selects nothing, main)
  {P}NCaveats{Ds}; per flagged benchmark {P}Caveat{X}{Ds}{Bn} with {X} in
  PublishedOne (cost_tokens_mean), Published (cost_tokens_requests_mean),
  TrueMain, TrueSingle, TrueMainPerInstance, and under impute Ratio (r)
  and FactorMain, FactorSingle (true / published)

Fixed per-benchmark overhead (opt-in, --fixed-cost RATIO ...; without the
flag nothing above changes).  Each benchmark also costs a fixed amount
that does not scale with its tokens (integration, harness setup, one model
load per suite): the true cost of benchmark j is c0 + tokens_j, c0 = RATIO
x the median main true cost over the priced benchmarks (after the cost
caveats), one level per RATIO (default 0 0.1 0.5 1 2; RATIO 0 is the main
scenario above, bit for bit: test_eval_token_costs.py).  Budgets B_f = f x
sum_j (c0 + tokens_j) over the priced benchmarks, so f = 1 still buys
everything.  Cost models: tokens = the true cost c0 + tokens_j (the
selector knows the overhead), instances = instances_j + RATIO x the median
instance count (the same overhead in the proxy's own units), uniform = 1
(cardinality, which the true cost approaches as RATIO grows).  Feasibility
always uses the true cost, so every method, random included (the shared
permutation: no new random-number stream), is held to B_f
(violations, expected all False; {o}_tokens is checked against the direct
budgeted_select as above).  Main scenario only (cell and single are not
run), both designs.  The burden of human evaluation is a further
per-benchmark cost for which there are no data; it is not modelled.
Outputs ({sfx} as above; token_fixed_* is not matched by token_costs_*):
  LOGS/token_fixed_{ds}{sfx}.npz   (savez_compressed) run_arrays' main_*
                              arrays and max_count with a level axis after
                              the split axis, (n_splits, n_levels,
                              n_budgets[, ...]); costs_true_main and
                              costs_instances (the instances cost model)
                              (n_levels, N); total_tokens, fixed_ratios,
                              fixed_costs (c0, tokens), fixed_costs_instances
                              (n_levels,); costs_instances_base (N,),
                              median_tokens, median_instances; split
                              metadata and the level-independent arrays of
                              run_arrays
  LOGS/token_fixed_{ds}{sfx}_paired.csv   paired_rows' pairs at every
                              level (columns dataset, holdout_frac,
                              fixed_ratio, fixed_cost, scenario, design,
                              metric, budget_frac, budget, method_a,
                              method_b, then STAT_FIELDS)
  LOGS/token_fixed_{ds}{sfx}.json  config, notes, cost levels, summary per
                              level, the FIXED_PAIRS rows (main knap),
                              checks, runtime and macros ({P} = TokFixed,
                              or TokFixed{Mode} for a non-default
                              --cost-caveats mode; {r} = num_words(RATIO):
                              Fzero, Fzeropointone, ..., Ftwo; the other
                              parts as above; finite values only, the
                              undefined ones listed in macros_undefined):
      {P}{T}Rsq{Ds}{M}F{r}B{b}, {P}{T}RsqStd...   main knap R² split mean /
                              std, {M} = {O}{C} or Random, {T} = '' (r2) or
                              Suite (r2_suite, the primary metric)
      {P}Size{Ds}{M}F{r}B{b}, {P}Spent...   benchmarks bought / true cost
                              over budget
      {P}{T}Pair{S}{Ds}{A}Vs{B}F{r}B{b}   FIXED_PAIRS (per objective:
                              tokens − instances, tokens − uniform, tokens
                              − random), S in Diff, Se, Lo, Hi, Win, N;
                              {P}{T}SigPos / SigNeg{Ds}{A}Vs{B}F{r}: budgets
                              whose 95% interval lies above / below 0
      {P}Overhead{Ds}F{r} (c0 in tokens), {P}OverheadInstances{Ds}F{r},
      {P}TotalTokens{Ds}F{r}, {P}MaxCount{Ds}F{r}B{b}, {P}Empty{Ds}F{r}B{b},
      {P}MedianTokens{Ds}, {P}MedianInstances{Ds}, {P}NBench{Ds},
      {P}NPriced{Ds}, {P}NSplits{Ds}, {P}Violations{Ds},
      {P}DirectMismatch{Ds}
  OUT/token_fixed{sfx}.pdf     per dataset two rows, the paired full-suite
                              R² difference tokens − random and tokens −
                              uniform per objective (main knap, 95%
                              Nadeau-Bengio band) vs f; one column per level

Usage:
    python eval_token_costs.py [--datasets helm_lite helm_mmlu helm_classic]
                               [--repeats 5] [--folds 10] [--budgets F ...]
                               [--scenarios main cell single]
                               [--cost-caveats impute|exclude|keep]
                               [--fixed-cost [RATIO ...]] [--plot-only]
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

from cv_split import (cv_splits, eligible_mask, target_mask, MIN_TRAIN_OBS,
                      N_REPEATS, RANDOM_SEED_OFFSET)
from greedy_select import random_select, _eligible, _near_max
from eval_entropy_vs_mi import (estimate_corr, impute_and_evaluate,
                                protocol_config)
from score_bounds import dataset_bounds
from eval_cost_budget import (OBJECTIVES as BUDGET_OBJECTIVES, TIE_RTOL,
                              D_FLOOR, budgeted_select, order_fill,
                              sequential_cost, suite_r2,
                              _chol_state_add, _chol_state_init,
                              _complement_precision_full, _entropy_gain,
                              _facloc_gains, _facloc_similarity,
                              _mi_gain, _tiebreak_by_explained, _trace_gains)
from paired_stats import STAT_FIELDS, method_rank, paired_stats, write_csv
from make_numbers import camel, num_words

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42

DATASETS = ["helm_lite", "helm_mmlu", "helm_classic"]
DEFAULT_BUDGETS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
# facloc (facility location, the headline method) appended last: no other
# objective's arrays, pairs or random streams change
OBJECTIVES = ["mi", "trace", "entropy", "facloc"]
COST_MODELS = ["tokens", "instances", "uniform"]
SCENARIOS = ["main", "cell", "single"]
DESIGNS = ["knap", "fill"]
METHODS = [f"{o}_{c}" for o in OBJECTIVES for c in COST_MODELS] + ["random"]
# Scenarios whose tokens cost model is the true cost (direct-knapsack check)
DIRECT_SCENARIOS = ("main", "single")
# Macro tag of each (scenario, design) reported in macros
MACRO_TAGS = {("main", "knap"): "", ("main", "fill"): "Fill",
              ("cell", "knap"): "Cell", ("single", "knap"): "Single"}

# Cost caveats (module docstring): benchmarks whose published token cost
# the data release marks as wrong, per dataset, with the data of the
# imputation.  "meta_caveat" is the start of the caveat in
# data/<ds>.meta.json "caveats" (checked at run time).  OpenbookQA: HELM
# Classic v0.4.0 groups/ablation_multiple_choice.json, table 'General
# information', column 'OpenbookQA - # prompt tokens' of the 6 ablation
# models (same 500 test questions, 0 in-context examples, one trial):
# multiple_choice_separate_original = the original requests (question +
# choice) alone, multiple_choice_separate_calibrated = what HELM tabulates
# for the calibrated method, a calibration request (the choice alone).
COST_CAVEATS = {
    "helm_classic": {
        "OpenbookQA": {
            "meta_caveat": "OpenbookQA's reported '# prompt tokens'",
            "method": "multiple_choice_separate_calibrated",
            "n_choices": 4,
            "ablation_models": ["Anthropic-LM v4-s3 (52B)", "BLOOM (176B)",
                                "GPT-J (6B)", "GPT-NeoX (20B)",
                                "OPT (175B)", "OPT (66B)"],
            "ablation_prompt_tokens_original": [15.024, 15.424, 15.024,
                                                15.19, 15.024, 15.024],
            "ablation_prompt_tokens_calibrated": [5.27, 5.444, 5.27, 5.346,
                                                  5.27, 5.27],
            "source": ("HELM Classic v0.4.0 "
                       "groups/ablation_multiple_choice.json, 'General "
                       "information', 'OpenbookQA - # prompt tokens'"),
        },
    },
}
CAVEAT_MODES = ("impute", "exclude", "keep")
DEFAULT_CAVEAT_MODE = "impute"
# Metric of the main text of the token-budget results (module docstring)
PRIMARY_METRIC = "r2_suite"
METRIC_NOTES = {
    "r2_suite": "primary: pooled over ALL observed validation cells, a "
                "selected cell counted as known (residual 0); one target "
                "set for every method and budget; empty selection = 0; "
                "rewards covering many cheap benchmarks (each bought "
                "benchmark adds its share of the standardized suite "
                "variance however predictable it was)",
    "r2": "secondary (the paper's metric elsewhere): pooled over the "
          "UNSELECTED observed benchmarks only, so methods are scored on "
          "different target sets under a budget and on 1-3 targets at "
          "large budgets; NaN for an empty selection or when no "
          "validation model observes a selected benchmark; pairs use the "
          "splits where both values are finite (n_splits)"}

DISPLAY_NAME = {"helm_lite": "HELM Lite", "helm_mmlu": "HELM MMLU",
                "helm_classic": "HELM Classic"}
OBJECTIVE_LABEL = {"mi": "MI", "trace": "Trace", "entropy": "Entropy",
                   "facloc": "Facility location"}
COST_LABEL = {"tokens": "tokens (true)", "instances": "instances",
              "uniform": "uniform"}
# Objectives: the repo's MI red and entropy blue (eval_entropy_vs_mi), trace
# olive (CVD-separable from both; markers as secondary encoding), facility
# location eval_baselines' teal.  Cost models: Paul Tol colors validated
# for CVD / normal-vision separation; random is the repo's gray baseline
# (dashed, x markers).
OBJECTIVE_COLOR = {"mi": "#CC3311", "trace": "#999933", "entropy": "#4477AA",
                   "facloc": "#009988"}
OBJECTIVE_MARKER = {"mi": "o", "trace": "D", "entropy": "s", "facloc": "v"}
COST_COLOR = {"tokens": "#0077BB", "instances": "#CC6677",
              "uniform": "#117733", "random": "#888888"}
COST_STYLE = {"tokens": "o-", "instances": "s--", "uniform": "^:",
              "random": "x--"}
PAIR_STATS = [("Diff", "mean_diff"), ("Se", "se_nb"), ("Lo", "ci_low"),
              ("Hi", "ci_high"), ("Win", "win_rate"), ("N", "n_splits")]
# Metrics: the paper's pooled R² over unselected benchmarks, and the
# full-suite R² (suite_r2); key suffix -> macro tag
METRICS = {"r2": "", "r2_suite": "Suite"}

PAIRED_FIELDS = (["dataset", "holdout_frac", "scenario", "design", "metric",
                  "budget_frac", "budget", "method_a", "method_b"]
                 + STAT_FIELDS)

# Fixed per-benchmark overhead (--fixed-cost; module docstring)
DEFAULT_FIXED_RATIOS = [0.0, 0.1, 0.5, 1.0, 2.0]   # c0 / median true cost
FIXED_STEM = "token_fixed"
# Pairs printed, kept in the JSON and given macros (main knap; the CSV has
# every paired_rows pair): per objective, the true-cost knapsack vs the
# instance proxy, vs cardinality and vs random
FIXED_PAIRS = [p for o in OBJECTIVES for p in (
    (f"{o}_tokens", f"{o}_instances"), (f"{o}_tokens", f"{o}_uniform"),
    (f"{o}_tokens", "random"))]
FIXED_PAIRED_FIELDS = (["dataset", "holdout_frac", "fixed_ratio",
                        "fixed_cost"] + PAIRED_FIELDS[2:])
# Level-dependent vectors of run_arrays' save dict (stacked on axis 0);
# its main_* arrays and max_count get a level axis after the split axis
FIXED_LEVEL_VECTORS = ("costs_true_main", "costs_instances", "total_tokens")


def display_name(ds_name):
    """Display name of a dataset (the raw name if none is registered)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def matrix_path(ds_name):
    return DATA / f"{ds_name}.matrix.npz"


def meta_path(ds_name):
    return DATA / f"{ds_name}.meta.json"


def mode_suffix(mode):
    """Output-file suffix of a --cost-caveats mode ('' for the default)."""
    return "" if mode == DEFAULT_CAVEAT_MODE else f"_{mode}"


def macro_prefix(mode):
    """Macro prefix of a --cost-caveats mode (Tok for the default)."""
    return "Tok" if mode == DEFAULT_CAVEAT_MODE else f"Tok{camel(mode)}"


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ── Proxy-aware budgeted greedy ──────────────────────────────────

def proxy_budgeted_select(Sigma, rank_costs, true_costs, budget,
                          objective="entropy", eligible=None,
                          singleton=True):
    """eval_cost_budget.budgeted_select with separate ranking and
    feasibility costs.

    The cost-effective greedy ranks the feasible candidates by gain /
    rank_costs[j] (ratio near-ties within a relative TIE_RTOL broken by
    explained variance x 1 / rank_costs, then by the lowest index), while a
    candidate is feasible only if its TRUE cost still fits:
    spent + true_costs[j] <= budget, spent = true cost of the picks so far.
    The best-singleton fallback (singleton=True) ranges over the eligible
    benchmarks with true_costs[j] <= budget, with budgeted_select's values,
    tie handling and margins (they do not depend on costs).  Line for line
    the arithmetic of budgeted_select, so
    rank_costs == true_costs gives exactly budgeted_select(Sigma,
    true_costs, budget, objective, eligible) (test_eval_token_costs.py and
    the {scen}_knap_direct_mismatch check of every run).

    budget = np.inf with singleton=False returns the full cost-effective
    order under rank_costs (design "fill"), which stops where no candidate
    has a positive gain.

    Returns budgeted_select's dict (selected, singleton, value, tie_breaks,
    index_ties) plus spent (true cost of `selected`, accumulated in pick
    order).
    """
    if objective not in BUDGET_OBJECTIVES:
        raise ValueError(f"objective {objective!r} not in "
                         f"{BUDGET_OBJECTIVES}")
    rank_costs = np.asarray(rank_costs, dtype=float)
    true_costs = np.asarray(true_costs, dtype=float)
    N = Sigma.shape[0]
    if rank_costs.shape != (N,) or true_costs.shape != (N,):
        raise ValueError("rank_costs and true_costs must have shape (N,)")
    el = _eligible(eligible, N)
    state = _chol_state_init(Sigma)
    spent = 0.0
    greedy_value = 0.0  # accumulated (shifted) objective
    tie_breaks, index_ties = [], []
    if objective == "facloc":
        S_fl = _facloc_similarity(Sigma)
        cover = np.zeros(N)

    while True:
        best_j, best_ratio = -1, -np.inf
        if objective == "mi":
            P_full = _complement_precision_full(Sigma, state["selected"])
        elif objective == "trace":
            sel_set = set(state["selected"])
            unsel = np.array([j for j in range(N) if j not in sel_set],
                             dtype=int)
            trace_gain = np.zeros(N)
            if len(unsel):
                trace_gain[unsel] = _trace_gains(state, Sigma, unsel)
        elif objective == "facloc":
            sel_set = set(state["selected"])
            unsel = np.array([j for j in range(N) if j not in sel_set],
                             dtype=int)
            fl_gain = np.zeros(N)
            if len(unsel):
                fl_gain[unsel] = _facloc_gains(S_fl, cover, unsel)
        cand_j, cand_ratio = [], []  # all feasible (j, ratio)
        for j in range(N):
            if (j in state["selected"] or spent + true_costs[j] > budget
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
            ratio = gain / rank_costs[j]
            cand_j.append(j)
            cand_ratio.append(ratio)
            if ratio > best_ratio:
                best_ratio = ratio
                best_j = j
        if best_j < 0 or best_ratio <= 0:
            break
        cand_j, cand_ratio = np.array(cand_j), np.array(cand_ratio)
        tied = cand_j[_near_max(cand_ratio, 0.0) & (cand_ratio > 0)]
        step = state["t"]
        best_j, n_cand, n_index = _tiebreak_by_explained(
            state, Sigma, tied, 1.0 / rank_costs[tied])
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
        spent += true_costs[best_j]
        greedy_value += best_gain
        _chol_state_add(state, Sigma, best_j)
        if objective == "facloc":
            cover = np.maximum(cover, S_fl[:, best_j])

    result = {"selected": list(state["selected"]), "singleton": False,
              "value": greedy_value, "tie_breaks": tie_breaks,
              "index_ties": index_ties, "spent": spent}
    if not singleton:
        return result

    # Best-singleton fallback (budgeted_select's), feasibility in true cost
    afford = [j for j in range(N)
              if true_costs[j] <= budget and (el is None or el[j])]
    if afford:
        n_cand = n_index = 1
        if objective == "entropy":
            def single_value(j):
                return max(0.5 * np.log(max(Sigma[j, j], 1e-300) / D_FLOOR),
                           0.0)
            vals = np.array([single_value(j) for j in afford])
            tied = np.array(afford)[_near_max(vals, 1.0)]
            best_single, n_cand, n_index = _tiebreak_by_explained(
                _chol_state_init(Sigma), Sigma, tied, np.ones(len(tied)))
            margin = TIE_RTOL * max(1.0, abs(greedy_value))
        elif objective == "trace":
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
                index_ties=[(0, n_index)] if n_index > 1 else [],
                spent=float(true_costs[best_single]))
    return result


# suite_r2 (the full-suite R², primary metric here) and sequential_cost are
# eval_cost_budget's (imported above; its --fixed-cost runs use them too).


def max_affordable(costs, budget, eligible=None):
    """Largest number of (eligible) benchmarks whose costs fit the budget:
    cheapest first."""
    c = np.sort(np.asarray(costs, dtype=float)[
        slice(None) if eligible is None else np.asarray(eligible, bool)])
    spent, n = 0.0, 0
    for v in c:
        if spent + v > budget:
            break
        spent += v
        n += 1
    return n


# ── Costs ────────────────────────────────────────────────────────

def _float_array(values):
    """Float array of a (nested) JSON list, null as NaN."""
    def conv(v):
        if isinstance(v, list):
            return [conv(x) for x in v]
        return np.nan if v is None else float(v)
    return np.array(conv(values), dtype=float)


def load_cost_tables(meta, benchmark_names, M):
    """Per-benchmark cost vectors and the per-cell costs of a HELM
    meta.json: "tokens" (cost_tokens_mean, one request per instance),
    "requests" (cost_tokens_requests_mean), "instances" (instances_mean),
    "rpi" (requests_per_instance; 1 if absent) (N,); "prompt",
    "completion" ('# prompt / # output tokens' means, NaN if absent) (N,);
    "cell" (cell_costs.cost_tokens) and "cell_instances"
    (cell_costs.instances, None if absent) (M, N), NaN where unobserved."""
    bc = meta["benchmark_cost"]
    names = [str(b) for b in bc["benchmark_names"]]
    if names != list(benchmark_names):
        raise ValueError("meta.benchmark_cost.benchmark_names do not match "
                         "the matrix columns")
    N = len(names)
    tables = {"tokens": np.array(bc["cost_tokens_mean"], dtype=float),
              "instances": np.array(bc["instances_mean"], dtype=float),
              "requests": np.array(bc.get("cost_tokens_requests_mean",
                                          bc["cost_tokens_mean"]),
                                   dtype=float),
              "rpi": np.array(bc.get("requests_per_instance", [1.0] * N),
                              dtype=float)}
    for key, v in tables.items():
        if v.shape != (N,) or not np.all(np.isfinite(v) & (v > 0)):
            raise ValueError(f"benchmark_cost {key}: need {N} positive "
                             f"finite costs")
    for key, src in (("prompt", "prompt_tokens_mean"),
                     ("completion", "completion_tokens_mean")):
        v = _float_array(bc.get(src, [None] * N))
        if v.shape != (N,):
            raise ValueError(f"benchmark_cost.{src}: need {N} values")
        tables[key] = v
    for key, src in (("cell", "cost_tokens"), ("cell_instances", "instances")):
        if src not in meta["cell_costs"]:
            if key == "cell":
                raise ValueError("meta.cell_costs.cost_tokens missing")
            tables[key] = None
            continue
        cell = _float_array(meta["cell_costs"][src])
        if cell.shape != (M, N):
            raise ValueError(f"cell_costs.{src} has shape {cell.shape}, "
                             f"expected ({M}, {N})")
        tables[key] = cell
    return tables


def caveat_imputation(entry, prompt, completion, rpi):
    """Imputation factors of a flagged multiple_choice_separate_calibrated
    benchmark (module docstring, Cost caveats).

    r = median over the ablation models of original / calibrated '#
    prompt tokens'.  With HELM's tabulated p (a calibration request) and o
    ('# output tokens'), one original request costs r p + o and an instance
    of the calibrated method k (r p + o) + k (p + o) for k answer choices
    (2 k requests).  Returns (r, factor_single, factor_main): the true
    costs relative to the published one-request cost (instances x (p + o))
    and request-corrected cost (rpi x instances x (p + o)).
    """
    orig = np.asarray(entry["ablation_prompt_tokens_original"], dtype=float)
    cal = np.asarray(entry["ablation_prompt_tokens_calibrated"], dtype=float)
    if orig.shape != cal.shape or orig.size == 0 or np.any(cal <= 0):
        raise ValueError("caveat ablation: need matching positive values")
    r = float(np.median(orig / cal))
    k = int(entry["n_choices"])
    p, o, rpi = float(prompt), float(completion), float(rpi)
    if not (np.isfinite(p) and np.isfinite(o) and p > 0 and o >= 0):
        raise ValueError("caveat imputation needs the benchmark's prompt "
                         "and completion token means")
    if rpi != 2 * k:
        raise ValueError(f"caveat imputation assumes {2 * k} requests per "
                         f"instance ({entry['method']}), meta has {rpi:g}")
    one = p + o
    f_single = (r * p + o) / one
    f_main = (k * (r * p + o) + k * (p + o)) / (rpi * one)
    return r, f_single, f_main


def prepare_costs(tables, benchmark_names, caveats=None,
                  mode=DEFAULT_CAVEAT_MODE, meta_caveats=()):
    """Scenario cost vectors of a dataset after its cost caveats.

    caveats: {benchmark: COST_CAVEATS entry} of the dataset (None: none);
    mode in CAVEAT_MODES; meta_caveats: the meta's "caveats" list (each
    flagged benchmark's caveat is looked up there for the disclosure).
    Returns a copy of `tables` (load_cost_tables) with
      rpi (N,)          requests_per_instance (requests / tokens if absent)
      true_main (N,)    cost_tokens_requests_mean, x factor_main (impute)
      true_single (N,)  cost_tokens_mean, x factor_single (impute)
      mult_req (N,)     cell-cost multiplier of scenario cell:
                        requests_per_instance (x factor_main under impute)
      priced (N,) bool  purchasable (False: flagged, mode exclude)
      caveat_flag (N,) bool, caveat_mode, caveats (list of disclosure
                        dicts, imputation computed in every mode)
    Unflagged benchmarks keep the published values exactly (factor 1.0).
    """
    if mode not in CAVEAT_MODES:
        raise ValueError(f"cost caveat mode {mode!r} not in {CAVEAT_MODES}")
    names = [str(b) for b in benchmark_names]
    N = len(names)
    caveats = caveats or {}
    unknown = [b for b in caveats if b not in names]
    if unknown:
        raise ValueError(f"cost caveats for unknown benchmarks {unknown}")
    out = dict(tables)
    tokens = np.asarray(tables["tokens"], dtype=float)
    requests = np.asarray(tables["requests"], dtype=float)
    rpi = tables.get("rpi")
    rpi = (requests / tokens if rpi is None
           else np.asarray(rpi, dtype=float))
    f_main, f_single = np.ones(N), np.ones(N)
    priced = np.ones(N, dtype=bool)
    flag = np.zeros(N, dtype=bool)
    records = []
    for b, entry in caveats.items():
        j = names.index(b)
        flag[j] = True
        found = [str(c) for c in meta_caveats
                 if str(c).startswith(entry["meta_caveat"])]
        prompt = np.asarray(tables.get("prompt", np.full(N, np.nan)))[j]
        compl = np.asarray(tables.get("completion", np.full(N, np.nan)))[j]
        r, fs, fm = caveat_imputation(entry, prompt, compl, rpi[j])
        if mode == "impute":
            f_main[j], f_single[j] = fm, fs
        elif mode == "exclude":
            priced[j] = False
        n_inst = float(tables["instances"][j])
        records.append({
            "benchmark": b, "index": j, "mode": mode,
            "caveat": found[0] if found else None,
            "caveat_in_meta": bool(found), "method": entry["method"],
            "n_choices": int(entry["n_choices"]),
            "requests_per_instance": float(rpi[j]), "instances": n_inst,
            "prompt_tokens_mean": float(prompt),
            "completion_tokens_mean": float(compl),
            "published_tokens_one_request": float(tokens[j]),
            "published_tokens_requests": float(requests[j]),
            "ablation_models": list(entry["ablation_models"]),
            "ablation_prompt_tokens_original":
                list(entry["ablation_prompt_tokens_original"]),
            "ablation_prompt_tokens_calibrated":
                list(entry["ablation_prompt_tokens_calibrated"]),
            "ablation_source": entry["source"], "ratio": r,
            "imputed_factor_single": fs, "imputed_factor_main": fm,
            "imputed_tokens_single": float(tokens[j] * fs),
            "imputed_tokens_main": float(requests[j] * fm),
            "imputed_tokens_main_per_instance":
                float(requests[j] * fm / n_inst),
            "applied": {"impute": "imputed costs in every scenario",
                        "exclude": "not purchasable (still a target)",
                        "keep": "published costs"}[mode]})
    out["rpi"] = rpi
    out["true_main"] = requests * f_main
    out["true_single"] = tokens * f_single
    out["mult_req"] = rpi * f_main
    out["priced"] = priced
    out["caveat_flag"] = flag
    out["caveat_mode"] = mode
    out["caveats"] = records
    return out


def _col_mean(C, rows):
    """Column means of C[rows] over finite entries and their counts."""
    sub = C[rows]
    ok = np.isfinite(sub)
    cnt = ok.sum(axis=0)
    tot = np.where(ok, sub, 0.0).sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), cnt


def scenario_costs(scenario, tables, split):
    """(true_costs, tokens_model, instances_model, fallback) of one split
    and scenario (tables of prepare_costs).

    main / single: true = tokens model = true_main / true_single,
    instances model = instances_mean, fallback None.  cell: true = mean of
    cell cost x mult_req over the split's validation models observing the
    benchmark (fallback: training mean, then true_main; `fallback` marks
    benchmarks without a validation observation); the tokens model is the
    same mean over the training models (fallback true_main) and the
    instances model the mean of cell_instances over the training models
    (fallback instances_mean): no cost model uses the validation models.
    """
    if scenario == "main":
        return (tables["true_main"], tables["true_main"],
                tables["instances"], None)
    if scenario == "single":
        return (tables["true_single"], tables["true_single"],
                tables["instances"], None)
    if scenario != "cell":
        raise ValueError(scenario)
    C = tables["cell"] * np.asarray(tables["mult_req"], dtype=float)[None, :]
    val, n_val = _col_mean(C, split.val_idx)
    tr, n_tr = _col_mean(C, split.train_idx)
    model = np.where(n_tr > 0, tr, tables["true_main"])
    true = np.where(n_val > 0, val, model)
    CI = tables.get("cell_instances")
    if CI is None:
        inst = np.asarray(tables["instances"], dtype=float)
    else:
        ci, n_ci = _col_mean(CI, split.train_idx)
        inst = np.where(n_ci > 0, ci, tables["instances"])
    return true, model, inst, n_val == 0


def active_scenarios(scenarios, tables):
    """The requested scenarios that apply (tables of prepare_costs):
    single only where its costs differ from main's."""
    out = []
    for scen in scenarios:
        if scen == "single" and np.array_equal(tables["true_single"],
                                               tables["true_main"]):
            continue
        out.append(scen)
    return out


def cost_correlations(a, b):
    """Spearman, Pearson and Pearson-of-logs correlation of two positive
    cost vectors across benchmarks."""
    from scipy import stats
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"spearman": math.nan, "pearson": math.nan,
                "pearson_log": math.nan}
    return {"spearman": float(stats.spearmanr(a, b)[0]),
            "pearson": float(stats.pearsonr(a, b)[0]),
            "pearson_log": float(stats.pearsonr(np.log(a), np.log(b))[0])}


# ── Per-split evaluation ─────────────────────────────────────────

def evaluate_split(B, O, split, tables, budgets_frac, scenarios,
                   bounds=None):
    """Every scenario, design, method and budget on one matched split.

    Returns {"eligible", "failed", "scen": {scen: {...}}} where each
    scenario holds "budgets" (n_budgets,), "true", "rank_tokens",
    "rank_instances" (N,), "fallback" ((N,) bool or None),
    "results"[(design, method)] = list over budgets of dicts (selected,
    r2, r2_suite, rmse, cost, violation; knap: singleton, index_ties,
    tie_breaks) and "direct_mismatch" (n_objectives, n_budgets) bool
    (DIRECT_SCENARIOS) or None.  tables: prepare_costs' (raw
    load_cost_tables tables are prepared without caveats).  Selectable:
    eligible_mask(O_train) & tables["priced"], shared by every method.
    Selections use only the training models' scores (and, in scenario
    cell, the validation models' token counts for feasibility).  Both
    metrics score the targets of target_mask(O_train) only (rule D1) and
    clip the predictions to `bounds` (rule D2; None: no clip).
    """
    N = B.shape[1]
    if "true_main" not in tables:
        tables = prepare_costs(tables, [f"b{j}" for j in range(N)])
    priced = np.asarray(tables["priced"], dtype=bool)
    B_train, O_train = B[split.train_idx], O[split.train_idx]
    B_test, O_test = B[split.val_idx], O[split.val_idx]
    el = eligible_mask(O_train)
    target = target_mask(O_train)            # D1, shared by every method
    sel_el = el & priced
    out = {"eligible": el, "failed": False, "scen": {}}
    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:  # pragma: no cover - logged and skipped
        out["failed"] = True
        out["error"] = str(e)
        return out

    # The split's shared random permutation restricted to the selectable
    # benchmarks (an excluded one drops out, the others keep their order)
    rand_order = random_select(
        R, N, seed=split.seed(SEED + RANDOM_SEED_OFFSET),
        eligible=sel_el)["selected"]
    imputed = {}

    def evaluate(sel):
        key = tuple(sorted(int(j) for j in sel))
        if key not in imputed:
            if key:
                imp = impute_and_evaluate(B_test, O_test, list(key), R,
                                          col_mean, col_std, N,
                                          target=target, bounds=bounds)
                r2, rmse = float(imp["r2"]), float(imp["rmse"])
            else:
                r2 = rmse = math.nan
            imputed[key] = (r2, rmse, suite_r2(B_test, O_test, key, R,
                                               col_mean, col_std,
                                               target=target,
                                               bounds=bounds))
        return imputed[key]

    def record(sel, true, budget, res=None):
        r2, rmse, r2s = evaluate(sel)
        cost = sequential_cost(sel, true)
        rec = {"selected": [int(j) for j in sel], "r2": r2, "rmse": rmse,
               "r2_suite": r2s, "cost": cost,
               "violation": bool(cost > budget)}
        if res is not None:
            rec.update(singleton=bool(res["singleton"]),
                       index_ties=len(res["index_ties"]),
                       tie_breaks=len(res["tie_breaks"]))
        return rec

    for scen in scenarios:
        true, tok_model, inst_model, fallback = scenario_costs(scen, tables,
                                                               split)
        total = float(np.sum(true[priced]))
        budgets = np.array([f * total for f in budgets_frac])
        rank = {"tokens": tok_model, "instances": inst_model,
                "uniform": np.ones(N)}
        orders = {(o, c): proxy_budgeted_select(
                      R, rank[c], rank[c], np.inf, o, sel_el,
                      singleton=False)["selected"]
                  for o in OBJECTIVES for c in COST_MODELS}
        results = {(d, m): [] for d in DESIGNS for m in METHODS}
        direct = (np.zeros((len(OBJECTIVES), len(budgets)), dtype=bool)
                  if scen in DIRECT_SCENARIOS else None)
        for b, budget in enumerate(budgets):
            for oi, o in enumerate(OBJECTIVES):
                for c in COST_MODELS:
                    res = proxy_budgeted_select(R, rank[c], true, budget, o,
                                                sel_el)
                    results[("knap", f"{o}_{c}")].append(
                        record(res["selected"], true, budget, res))
                    if direct is not None and c == "tokens":
                        ref = budgeted_select(R, true, budget, o,
                                              eligible=sel_el)
                        direct[oi, b] = (
                            ref["selected"] != res["selected"]
                            or ref["singleton"] != res["singleton"])
                    sel = order_fill(orders[(o, c)], true, budget)
                    results[("fill", f"{o}_{c}")].append(
                        record(sel, true, budget))
            sel = order_fill(rand_order, true, budget)
            rec = record(sel, true, budget)
            for d in DESIGNS:
                results[(d, "random")].append(dict(rec))
        out["scen"][scen] = {"budgets": budgets, "true": np.asarray(true),
                             "rank_tokens": np.asarray(tok_model),
                             "rank_instances": np.asarray(inst_model),
                             "fallback": fallback, "results": results,
                             "direct_mismatch": direct,
                             "max_count": [max_affordable(true, bb, sel_el)
                                           for bb in budgets]}
    return out


# ── Main experiment ──────────────────────────────────────────────

def _key(scen, design, method):
    return f"{scen}_{design}_{method}"


def run_arrays(B, O, benchmark_names, tables, budgets_frac,
               n_repeats=N_REPEATS, n_folds=N_FOLDS, scenarios=SCENARIOS,
               verbose=True, bounds=None):
    """The experiment on one dataset given as arrays (B with missing = 0,
    O mask, cost tables of prepare_costs; tables of load_cost_tables are
    prepared without caveats; bounds: the valid score ranges of rule D2,
    None: no clip).  Returns the save dict (see the module
    docstring for its keys)."""
    B = np.asarray(B, dtype=np.float64)
    O = np.asarray(O, dtype=np.float64)
    M, N = B.shape
    if "true_main" not in tables:
        tables = prepare_costs(tables, benchmark_names)
    priced = np.asarray(tables["priced"], dtype=bool)
    scenarios = active_scenarios(scenarios, tables)
    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED))
    S, nb = len(splits), len(budgets_frac)
    if verbose:
        print(f"{n_repeats} x {n_folds}-fold CV = {S} matched splits, "
              f"{HOLDOUT_FRAC:.0%} holdout, selectable: benchmarks with "
              f">= {MIN_TRAIN_OBS} training models; scenarios: "
              f"{' '.join(scenarios)}", flush=True)

    out = {"benchmark_names": np.array(benchmark_names),
           "budgets_frac": np.array(budgets_frac, dtype=float),
           "costs_tokens": np.asarray(tables["tokens"], dtype=float),
           "costs_instances": np.asarray(tables["instances"], dtype=float),
           "costs_requests": np.asarray(tables["requests"], dtype=float),
           "costs_true_main": np.asarray(tables["true_main"], dtype=float),
           "costs_true_single": np.asarray(tables["true_single"],
                                           dtype=float),
           "costs_mult_req": np.asarray(tables["mult_req"], dtype=float),
           "priced": priced.copy(),
           "caveat_flag": np.asarray(tables["caveat_flag"], dtype=bool),
           "cost_caveat_mode": np.array(str(tables["caveat_mode"]))}
    for scen in scenarios:
        out[f"{scen}_budgets"] = np.full((S, nb), np.nan)
        out[f"{scen}_true_costs"] = np.full((S, N), np.nan)
        out[f"{scen}_rank_tokens"] = np.full((S, N), np.nan)
        out[f"{scen}_rank_instances"] = np.full((S, N), np.nan)
        if scen in DIRECT_SCENARIOS:
            out[f"{scen}_knap_direct_mismatch"] = np.zeros(
                (S, len(OBJECTIVES), nb), dtype=bool)
        for d in DESIGNS:
            for m in METHODS:
                k = _key(scen, d, m)
                out[f"{k}_r2"] = np.full((S, nb), np.nan)
                out[f"{k}_r2_suite"] = np.full((S, nb), np.nan)
                out[f"{k}_rmse"] = np.full((S, nb), np.nan)
                out[f"{k}_size"] = np.zeros((S, nb), dtype=np.int64)
                out[f"{k}_cost"] = np.full((S, nb), np.nan)
                out[f"{k}_violations"] = np.zeros((S, nb), dtype=bool)
                out[f"{k}_selected"] = np.full((S, nb, N), -1, dtype=np.int64)
                if d == "knap" and m != "random":
                    out[f"{k}_singleton"] = np.zeros((S, nb), dtype=bool)
                    out[f"{k}_index_ties"] = np.zeros((S, nb), dtype=np.int64)
                    out[f"{k}_tie_breaks"] = np.zeros((S, nb), dtype=np.int64)
    if "cell" in scenarios:
        out["cell_fallback"] = np.zeros((S, N), dtype=bool)
    out["max_count"] = np.zeros((S, nb), dtype=np.int64)
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
        res = evaluate_split(B, O, s, tables, budgets_frac, scenarios,
                             bounds)
        out["eligible"][i] = res["eligible"]
        out["n_eligible"][i] = int(res["eligible"].sum())
        if res["failed"]:
            out["estimation_failed"][i] = True
            if verbose:
                print(f"  rep {s.rep} fold {s.fold}: estimation failed: "
                      f"{res.get('error')}", flush=True)
            continue
        for scen, sr in res["scen"].items():
            out[f"{scen}_budgets"][i] = sr["budgets"]
            out[f"{scen}_true_costs"][i] = sr["true"]
            out[f"{scen}_rank_tokens"][i] = sr["rank_tokens"]
            out[f"{scen}_rank_instances"][i] = sr["rank_instances"]
            if sr["direct_mismatch"] is not None:
                out[f"{scen}_knap_direct_mismatch"][i] = sr["direct_mismatch"]
            if scen == "cell":
                out["cell_fallback"][i] = sr["fallback"]
            if scen == "main":
                out["max_count"][i] = sr["max_count"]
            for (d, m), recs in sr["results"].items():
                k = _key(scen, d, m)
                for b, rec in enumerate(recs):
                    sel = rec["selected"]
                    out[f"{k}_r2"][i, b] = rec["r2"]
                    out[f"{k}_r2_suite"][i, b] = rec["r2_suite"]
                    out[f"{k}_rmse"][i, b] = rec["rmse"]
                    out[f"{k}_size"][i, b] = len(sel)
                    out[f"{k}_cost"][i, b] = rec["cost"]
                    out[f"{k}_violations"][i, b] = rec["violation"]
                    out[f"{k}_selected"][i, b, :len(sel)] = sel
                    if "singleton" in rec:
                        out[f"{k}_singleton"][i, b] = rec["singleton"]
                        out[f"{k}_index_ties"][i, b] = rec["index_ties"]
                        out[f"{k}_tie_breaks"][i, b] = rec["tie_breaks"]
        if verbose:
            r2 = out["main_knap_trace_tokens_r2_suite"][i]
            r2_fl = out["main_knap_facloc_tokens_r2_suite"][i]
            mid = min(3, nb - 1)
            print(f"  rep {s.rep} fold {s.fold} done ({time.time()-t0:.1f}s),"
                  f" {out['n_eligible'][i]}/{N} eligible, suite "
                  f"R²@{100 * budgets_frac[mid]:g}%: trace_tokens "
                  f"{r2[mid]:.3f}, facloc_tokens {r2_fl[mid]:.3f}",
                  flush=True)

    out.update(scenarios=np.array(scenarios), designs=np.array(DESIGNS),
               objectives=np.array(OBJECTIVES),
               cost_models=np.array(COST_MODELS), methods=np.array(METHODS),
               holdout_frac=np.float64(HOLDOUT_FRAC),
               n_folds=np.int64(n_folds), n_repeats=np.int64(n_repeats),
               **{k: (np.bool_(v) if isinstance(v, bool) else np.int64(v))
                  for k, v in protocol_config().items()},
               seed=np.int64(SEED),
               total_tokens=np.float64(np.sum(
                   np.asarray(tables["true_main"])[priced])))
    return out


def run(ds_name, budgets_frac, n_repeats=N_REPEATS, n_folds=N_FOLDS,
        scenarios=SCENARIOS, caveat_mode=DEFAULT_CAVEAT_MODE):
    """run_arrays on data/<ds>.matrix.npz with data/<ds>.meta.json, after
    the dataset's COST_CAVEATS under caveat_mode.  Returns (out, tables)
    with prepare_costs' tables."""
    B, O, names, tables = load_dataset(ds_name, caveat_mode)
    out = run_arrays(B, O, names, tables, budgets_frac, n_repeats, n_folds,
                     scenarios, bounds=dataset_bounds(ds_name, names, DATA))
    out.update(matrix_sha256=np.array(_sha256(matrix_path(ds_name))),
               meta_sha256=np.array(_sha256(meta_path(ds_name))))
    return out, tables


def load_dataset(ds_name, caveat_mode=DEFAULT_CAVEAT_MODE):
    """(B, O, benchmark names, prepare_costs' tables) of data/<ds>, after
    the dataset's COST_CAVEATS under caveat_mode, with the consistency
    warnings and the cost-caveat disclosure printed."""
    d = np.load(matrix_path(ds_name), allow_pickle=False)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    names = [str(b) for b in d["benchmark_names"]]
    M, N = B.shape
    meta = json.loads(meta_path(ds_name).read_text())
    raw = load_cost_tables(meta, names, M)
    cell_obs = np.isfinite(raw["cell"])
    if not np.array_equal(cell_obs, O > 0):
        print(f"WARNING: {ds_name}: cell_costs observed pattern differs from "
              f"O in {int(np.sum(cell_obs != (O > 0)))} cells", flush=True)
    rel = (np.abs(raw["requests"] - raw["tokens"] * raw["rpi"])
           / raw["requests"])
    if np.max(rel) > 1e-6:
        print(f"WARNING: {ds_name}: cost_tokens_requests_mean != "
              f"cost_tokens_mean x requests_per_instance (max relative "
              f"difference {np.max(rel):.2g}); scenario cell scales cell "
              f"costs by requests_per_instance", flush=True)
    tables = prepare_costs(raw, names, COST_CAVEATS.get(ds_name), caveat_mode,
                           meta.get("caveats", []))
    tables["meta_caveats"] = [str(c) for c in meta.get("caveats", [])]
    for rec in tables["caveats"]:
        if not rec["caveat_in_meta"]:
            print(f"WARNING: {ds_name}: the cost caveat of {rec['benchmark']} "
                  f"is no longer in the meta's caveats; check COST_CAVEATS",
                  flush=True)
        print(f"COST CAVEAT {ds_name} {rec['benchmark']} (mode "
              f"{caveat_mode}: {rec['applied']}): published "
              f"{rec['published_tokens_one_request']:,.0f} tokens one "
              f"request per instance, {rec['published_tokens_requests']:,.0f}"
              f" with {rec['requests_per_instance']:g} requests per "
              f"instance; HELM's '# prompt tokens' "
              f"{rec['prompt_tokens_mean']:.2f} is a calibration "
              f"request's (meta caveat); ablation ratio "
              f"original / calibration r = {rec['ratio']:.3f} -> imputed "
              f"{rec['imputed_tokens_main']:,.0f} tokens "
              f"({rec['imputed_tokens_main_per_instance']:.1f} per instance; "
              f"x{rec['imputed_factor_main']:.3f}), single "
              f"{rec['imputed_tokens_single']:,.0f}", flush=True)
    t = tables["true_main"][tables["priced"]]
    pn = [nm for nm, p in zip(names, tables["priced"]) if p]
    print(f"{display_name(ds_name)}: {M} models x {N} benchmarks "
          f"({len(pn)} priced), {100 * O.mean():.1f}% observed; total "
          f"{t.sum():.4g} tokens (all requests), "
          f"{tables['instances'].sum():.0f} instances", flush=True)
    print(f"token cost range: {t.min():.4g} ({pn[int(np.argmin(t))]}) – "
          f"{t.max():.4g} ({pn[int(np.argmax(t))]})", flush=True)
    return B, O, names, tables


# ── Summaries and paired statistics ──────────────────────────────

def _nan_mean_std(a, axis=None):
    """np.nanmean / np.nanstd (ddof=0) without all-NaN-slice warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)


def _scenarios(out):
    return [str(s) for s in np.asarray(out["scenarios"]).ravel()]


def method_label(m):
    if m == "random":
        return "random"
    o, c = m.split("_", 1)
    return f"{o} / {c}"


def summary_tables(out):
    """{scen: {design: {method: {r2_mean, r2_std, r2_suite_mean,
    r2_suite_std, size_mean, spent_mean, n_empty, n_finite,
    n_finite_suite}}}} (lists over budgets; spent = true cost / budget;
    n_finite / n_finite_suite count the splits with a finite r2 /
    r2_suite, the splits each mean and std is over)."""
    tab = {}
    ok = ~out["estimation_failed"]
    for scen in _scenarios(out):
        budgets = out[f"{scen}_budgets"]
        tab[scen] = {}
        for d in DESIGNS:
            tab[scen][d] = {}
            for m in METHODS:
                k = _key(scen, d, m)
                entry = {}
                for metric in METRICS:
                    mu, sd = _nan_mean_std(out[f"{k}_{metric}"], axis=0)
                    entry[f"{metric}_mean"] = np.asarray(mu).tolist()
                    entry[f"{metric}_std"] = np.asarray(sd).tolist()
                size = out[f"{k}_size"][ok].astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    spent = out[f"{k}_cost"] / budgets
                sp, _ = _nan_mean_std(spent[ok], axis=0)
                entry.update(
                    size_mean=size.mean(axis=0).tolist(),
                    spent_mean=np.asarray(sp).tolist(),
                    n_empty=(size == 0).sum(axis=0).astype(int).tolist(),
                    n_finite=np.isfinite(out[f"{k}_r2"]).sum(axis=0)
                    .astype(int).tolist(),
                    n_finite_suite=np.isfinite(out[f"{k}_r2_suite"])
                    .sum(axis=0).astype(int).tolist())
                tab[scen][d][m] = entry
    return tab


def oriented(a, b):
    """(a, b) of two methods '{o}_{c}' / 'random' in the reporting
    orientation: objectives by paired_stats.method_rank (mi, entropy,
    trace, then facloc, which METHOD_ORDER does not rank: between hub and
    random), then cost models tokens, instances, uniform; random last."""
    def rank(m):
        if m == "random":
            return (method_rank("random"), 0)
        o, c = m.split("_", 1)
        return (method_rank(o), COST_MODELS.index(c))
    return (b, a) if rank(b) < rank(a) else (a, b)


def method_pairs():
    """Every reported pair, oriented: within-objective cost pairs, each
    method vs random, objective pairs at equal cost model."""
    pairs = []
    for o in OBJECTIVES:
        for i, ca in enumerate(COST_MODELS):
            for cb in COST_MODELS[i + 1:]:
                pairs.append(oriented(f"{o}_{ca}", f"{o}_{cb}"))
    for m in METHODS[:-1]:
        pairs.append(oriented(m, "random"))
    for c in COST_MODELS:
        for i, oa in enumerate(OBJECTIVES):
            for ob in OBJECTIVES[i + 1:]:
                pairs.append(oriented(f"{oa}_{c}", f"{ob}_{c}"))
    return pairs


def paired_rows(ds_name, out):
    """paired_stats of both METRICS for method_pairs() at every budget,
    scenario and design, as CSV rows (PAIRED_FIELDS)."""
    rows = []
    for scen in _scenarios(out):
        budgets = out[f"{scen}_budgets"]
        for d in DESIGNS:
            for metric in METRICS:
                for b, frac in enumerate(out["budgets_frac"]):
                    mean_budget, _ = _nan_mean_std(budgets[:, b])
                    for ma, mb in method_pairs():
                        st = paired_stats(
                            out[f"{_key(scen, d, ma)}_{metric}"][:, b],
                            out[f"{_key(scen, d, mb)}_{metric}"][:, b],
                            out["n_train"], out["n_val"], out["rep"])
                        rows.append({"dataset": ds_name,
                                     "holdout_frac": HOLDOUT_FRAC,
                                     "scenario": scen, "design": d,
                                     "metric": metric,
                                     "budget_frac": float(frac),
                                     "budget": float(mean_budget),
                                     "method_a": ma, "method_b": mb, **st})
    return rows


def _lookup(paired):
    return {(r["scenario"], r["design"], r["metric"], r["budget_frac"],
             r["method_a"], r["method_b"]): r for r in paired}


def _macro_method(m):
    return "Random" if m == "random" else "".join(
        camel(p) for p in m.split("_", 1))


def _out_mode(out):
    """--cost-caveats mode of a save dict (logs before the mode: keep)."""
    return (str(out["cost_caveat_mode"]) if "cost_caveat_mode" in out
            else "keep")


def macros_of(ds_name, out, tables, paired, undefined=None):
    """Flat {MacroName: finite number} dict (module docstring, 'Macros').

    A statistic that is undefined (NaN / inf: nothing selected, no finite
    pair) is not emitted; its name is appended to `undefined` (a list) if
    given, so a LaTeX consumer never sees null.  The prefix is
    macro_prefix of the save dict's cost-caveat mode."""
    P = macro_prefix(_out_mode(out))
    D = camel(ds_name)
    fracs = np.asarray(out["budgets_frac"], dtype=float)
    words = [num_words(100.0 * f) for f in fracs]
    mac = {}

    def put(name, v):
        v = int(v) if isinstance(v, (bool, np.bool_, np.integer)) else v
        if isinstance(v, int) or (isinstance(v, float) and math.isfinite(v)):
            mac[name] = v
        elif undefined is not None:
            undefined.append(name)

    names = [str(b) for b in out["benchmark_names"]]
    priced = np.asarray(out.get("priced", np.ones(len(names), bool)), bool)
    t_all = np.asarray(out.get("costs_true_main", tables["tokens"]), float)
    t, n = t_all[priced], np.asarray(tables["instances"], float)[priced]
    corr = cost_correlations(t, n)
    put(f"{P}Spearman{D}", corr["spearman"])
    put(f"{P}Pearson{D}", corr["pearson"])
    put(f"{P}PearsonLog{D}", corr["pearson_log"])
    if "single" in _scenarios(out):
        ts = np.asarray(out["costs_true_single"], float)[priced]
        rc = cost_correlations(ts, n)
        put(f"{P}SingleSpearman{D}", rc["spearman"])
        put(f"{P}SinglePearson{D}", rc["pearson"])
        put(f"{P}SinglePearsonLog{D}", rc["pearson_log"])
        put(f"{P}SingleTotalTokens{D}", float(np.sum(ts)))
    put(f"{P}NBench{D}", int(len(names)))
    put(f"{P}NPriced{D}", int(np.sum(priced)))
    put(f"{P}TotalTokens{D}", float(np.sum(t)))
    put(f"{P}TotalInstances{D}", float(np.sum(tables["instances"])))
    put(f"{P}CostRatio{D}", float(np.max(t) / np.min(t)))
    per = t / n
    put(f"{P}PerInstanceRatio{D}", float(np.max(per) / np.min(per)))
    put(f"{P}NSplits{D}", int(len(out["split_id"])))
    put(f"{P}Violations{D}", int(sum(
        int(np.sum(out[k])) for k in out if k.endswith("_violations"))))
    put(f"{P}DirectMismatch{D}", int(sum(
        int(np.sum(out[k])) for k in out
        if k.endswith("_knap_direct_mismatch"))))
    if "cell_fallback" in out:
        put(f"{P}CellFallback{D}", int(np.sum(out["cell_fallback"])))
    recs = tables.get("caveats", [])
    put(f"{P}NCaveats{D}", int(len(recs)))
    for rec in recs:
        Bn = camel(rec["benchmark"])
        j = int(rec["index"])
        put(f"{P}CaveatPublishedOne{D}{Bn}",
            rec["published_tokens_one_request"])
        put(f"{P}CaveatPublished{D}{Bn}", rec["published_tokens_requests"])
        put(f"{P}CaveatTrueMain{D}{Bn}", float(t_all[j]))
        put(f"{P}CaveatTrueSingle{D}{Bn}",
            float(np.asarray(out["costs_true_single"], float)[j]))
        put(f"{P}CaveatTrueMainPerInstance{D}{Bn}",
            float(t_all[j]) / rec["instances"])
        if rec["mode"] == "impute":
            put(f"{P}CaveatRatio{D}{Bn}", rec["ratio"])
            put(f"{P}CaveatFactorMain{D}{Bn}", rec["imputed_factor_main"])
            put(f"{P}CaveatFactorSingle{D}{Bn}",
                rec["imputed_factor_single"])
    ok = ~out["estimation_failed"]
    for b, w in enumerate(words):
        put(f"{P}MaxCount{D}B{w}", float(np.mean(out["max_count"][ok, b])))
        put(f"{P}Empty{D}B{w}", int(np.sum(
            out["main_knap_random_size"][ok, b] == 0)))

    lookup = _lookup(paired)
    for (scen, d), stag in MACRO_TAGS.items():
        if scen not in _scenarios(out):
            continue
        budgets = out[f"{scen}_budgets"]
        for m in METHODS:
            k = _key(scen, d, m)
            M_ = _macro_method(m)
            size = out[f"{k}_size"][ok].astype(float).mean(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                spent, _ = _nan_mean_std((out[f"{k}_cost"] / budgets)[ok],
                                         axis=0)
            for b, w in enumerate(words):
                put(f"{P}{stag}Size{D}{M_}B{w}", float(size[b]))
                if stag == "":
                    put(f"{P}Spent{D}{M_}B{w}", float(np.asarray(spent)[b]))
            for metric, mtag in METRICS.items():
                tag = stag + mtag
                mu, sd = _nan_mean_std(out[f"{k}_{metric}"], axis=0)
                for b, w in enumerate(words):
                    put(f"{P}{tag}Rsq{D}{M_}B{w}", float(mu[b]))
                    if stag == "":
                        put(f"{P}{tag}RsqStd{D}{M_}B{w}", float(sd[b]))
        if stag == "":
            pairs = [p for p in method_pairs()
                     if p[1] == "random" or p[0].split("_")[0]
                     == p[1].split("_")[0] or p[0].endswith("_tokens")
                     and p[1].endswith("_tokens")]
        else:
            pairs = [(f"{o}_tokens", f"{o}_{c}") for o in OBJECTIVES
                     for c in ("instances", "uniform")]
        for metric, mtag in METRICS.items():
            tag = stag + mtag
            for ma, mb in pairs:
                for b, w in enumerate(words):
                    r = lookup[(scen, d, metric, float(fracs[b]), ma, mb)]
                    for sname, field in PAIR_STATS:
                        v = r[field]
                        put(f"{P}{tag}Pair{sname}{D}{_macro_method(ma)}Vs"
                            f"{_macro_method(mb)}B{w}",
                            int(v) if field == "n_splits" else float(v))
    return mac


def json_safe(x):
    """Floats/ints/lists for json.dumps (NaN and inf as null)."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def selection_share(out, scen="main", design="knap"):
    """{method: (n_budgets, N) list}: fraction of the estimated splits on
    which each benchmark is selected."""
    ok = ~out["estimation_failed"]
    N = len(out["benchmark_names"])
    share = {}
    for m in METHODS:
        sel = out[f"{_key(scen, design, m)}_selected"][ok]   # (S, nb, N)
        hit = np.zeros(sel.shape[:2] + (N,), dtype=bool)
        s_idx, b_idx, p_idx = np.nonzero(sel >= 0)
        hit[s_idx, b_idx, sel[s_idx, b_idx, p_idx]] = True
        share[m] = (hit.mean(axis=0) if len(hit) else
                    np.full(hit.shape[1:], np.nan)).tolist()
    return share


def json_report(ds_name, out, tables, paired, macros, config,
                undefined=None):
    """The LOGS/token_costs_{ds}{sfx}.json payload (undefined: the macro
    names macros_of left out)."""
    names = [str(b) for b in out["benchmark_names"]]
    P, D = macro_prefix(_out_mode(out)), camel(ds_name)
    priced = np.asarray(out["priced"], dtype=bool)
    key_pairs = [r for r in paired
                 if r["method_a"].endswith("_tokens")
                 and r["method_a"].split("_")[0] == r["method_b"].split("_")[0]]
    checks = {
        "violations": {k[:-len("_violations")]: int(np.sum(out[k]))
                       for k in out if k.endswith("_violations")},
        "violations_total": macros[f"{P}Violations{D}"],
        "direct_mismatch_total": macros[f"{P}DirectMismatch{D}"],
        "estimation_failed": int(np.sum(out["estimation_failed"])),
        "n_eligible_min": int(np.min(out["n_eligible"])),
        "n_eligible_max": int(np.max(out["n_eligible"])),
        "cell_fallback_cells": (int(np.sum(out["cell_fallback"]))
                                if "cell_fallback" in out else None),
        "unpriced_selected": int(sum(
            int(np.sum(np.isin(out[k], np.flatnonzero(~priced))))
            for k in out if k.endswith("_selected")))}
    t_main, t_single = out["costs_true_main"], out["costs_true_single"]
    inst = np.asarray(tables["instances"], dtype=float)
    return json_safe({
        "dataset": ds_name, "config": config,
        "primary_metric": PRIMARY_METRIC, "metric_notes": METRIC_NOTES,
        "cost_table": [{"benchmark": nm,
                        "tokens_true_main": t_main[j],
                        "tokens_true_single": t_single[j],
                        "tokens_published_one_request": tables["tokens"][j],
                        "tokens_published_requests": tables["requests"][j],
                        "requests_per_instance": tables["rpi"][j],
                        "instances": inst[j],
                        "tokens_per_instance": t_main[j] / inst[j],
                        "priced": bool(priced[j]),
                        "caveat": bool(out["caveat_flag"][j])}
                       for j, nm in enumerate(names)],
        "cost_caveats": {"mode": _out_mode(out),
                         "modes": list(CAVEAT_MODES),
                         "flagged": tables.get("caveats", []),
                         "meta_caveats": tables.get("meta_caveats", [])},
        "correlations": {
            "true_main_vs_instances": cost_correlations(t_main[priced],
                                                        inst[priced]),
            "true_single_vs_instances": cost_correlations(t_single[priced],
                                                          inst[priced]),
            "true_main_vs_single": cost_correlations(t_main[priced],
                                                     t_single[priced])},
        "budgets_frac": out["budgets_frac"],
        "budgets_main": (np.asarray(out["budgets_frac"], dtype=float)
                         * float(out["total_tokens"])),
        "max_count_mean": np.mean(out["max_count"][~out["estimation_failed"]],
                                  axis=0),
        "summary": summary_tables(out),
        "selection_share_main_knap": selection_share(out),
        "paired_cost_model": key_pairs,
        "checks": checks,
        "runtime_s": float(out["runtime_s"]),
        "n_macros": len(macros),
        "macros": macros,
        "macros_undefined": sorted(undefined or []),
        "macros_undefined_note": ("statistics without a finite value (no "
                                  "selection, or no split where both "
                                  "values of a pair are finite); not "
                                  "emitted as macros")})


# ── Reporting ────────────────────────────────────────────────────

def _sig(r):
    return (np.isfinite(r["ci_low"]) and (r["ci_low"] > 0
                                          or r["ci_high"] < 0))


def metric_order():
    """METRICS keys, PRIMARY_METRIC first."""
    return [PRIMARY_METRIC] + [m for m in METRICS if m != PRIMARY_METRIC]


def print_summary(ds_name, out, tables, paired):
    names = [str(b) for b in out["benchmark_names"]]
    fracs = np.asarray(out["budgets_frac"], dtype=float)
    ok = ~out["estimation_failed"]
    S, S_ok = len(out["split_id"]), int(ok.sum())
    mode = _out_mode(out)
    priced = np.asarray(out["priced"], dtype=bool)
    flag = np.asarray(out["caveat_flag"], dtype=bool)
    t, n = out["costs_true_main"], np.asarray(tables["instances"], float)
    print(f"\n== {display_name(ds_name)}: cost table (true cost: all "
          f"requests; --cost-caveats {mode}) ==")
    print(f"{'benchmark':<32}{'tokens':>12}{'instances':>10}"
          f"{'tok/inst':>9}{'req/inst':>9}{'1 request':>12}  note")
    note = {"impute": "imputed (caveat)", "exclude": "NOT PURCHASABLE",
            "keep": "published (caveat)"}[mode]
    for j in np.argsort(-t):
        print(f"{names[j][:31]:<32}{t[j]:>12.4g}{n[j]:>10.0f}"
              f"{t[j] / n[j]:>9.1f}{tables['rpi'][j]:>9.4g}"
              f"{tables['tokens'][j]:>12.4g}  {note if flag[j] else ''}")
    c = cost_correlations(t[priced], n[priced])
    print(f"true tokens vs instances across {int(priced.sum())} priced "
          f"benchmarks: Spearman {c['spearman']:.3f}, Pearson "
          f"{c['pearson']:.3f}, Pearson(log) {c['pearson_log']:.3f}")
    for rec in tables.get("caveats", []):
        print(f"cost caveat {rec['benchmark']}: {rec['caveat']}\n  "
              f"published {rec['published_tokens_one_request']:,.0f} (one "
              f"request) / {rec['published_tokens_requests']:,.0f} (all "
              f"requests); ablation r = {rec['ratio']:.3f}; imputed "
              f"{rec['imputed_tokens_main']:,.0f} / single "
              f"{rec['imputed_tokens_single']:,.0f}; applied: "
              f"{rec['applied']}")

    tab = summary_tables(out)
    lookup = _lookup(paired)
    for scen in _scenarios(out):
        for d in DESIGNS:
            if (scen, d) not in MACRO_TAGS:
                continue
            print(f"\n== {display_name(ds_name)} [{scen}, {d}]: full-suite "
                  f"R² (primary) / R² / #selected vs true token budget "
                  f"(means over {S_ok} of {S} splits; [n] = splits with a "
                  f"finite R² where fewer) ==")
            print(f"{'method':<20}" + "".join(f"{100 * f:>21g}%"
                                              for f in fracs))
            for m in METHODS:
                st = tab[scen][d][m]
                cells = []
                for ms, mu, sz, nf in zip(st["r2_suite_mean"], st["r2_mean"],
                                          st["size_mean"], st["n_finite"]):
                    cells.append(f"{ms:.3f}/{mu:.3f}/{sz:.1f}"
                                 + (f"[{nf}]" if nf < S_ok else ""))
                print(f"{method_label(m):<20}"
                      + "".join(f"{x:>22}" for x in cells))
            for metric in metric_order():
                print(f"paired {metric} differences (mean; * = "
                      f"Nadeau–Bengio 95% CI excludes 0; (n) = paired "
                      f"splits where fewer than {S_ok}):")
                for o in OBJECTIVES:
                    for cb in ("instances", "uniform"):
                        ma, mb = f"{o}_tokens", f"{o}_{cb}"
                        cells = []
                        for f in fracs:
                            r = lookup[(scen, d, metric, float(f), ma, mb)]
                            ns = int(r["n_splits"])
                            cells.append("n/a" if ns == 0 else
                                         f"{r['mean_diff']:+.3f}"
                                         f"{'*' if _sig(r) else ' '}"
                                         + (f"({ns})" if ns < S_ok else ""))
                        print(f"  {ma + ' − ' + mb:<36}"
                              + "".join(f"{x:>13}" for x in cells))
    share = selection_share(out)
    for j in np.flatnonzero(flag):
        print(f"\nselection share of {names[j]} (main knap, flagged cost; "
              f"{'not purchasable' if not priced[j] else mode}):")
        for m in METHODS:
            print(f"  {method_label(m):<20}" + "".join(
                f"{100 * f:>6g}%: {np.asarray(share[m])[b, j]:.2f}"
                for b, f in enumerate(fracs)))
    mc = np.mean(out["max_count"][ok], axis=0)
    print("\nmost benchmarks the main budgets afford (cheapest first): "
          + ", ".join(f"{100 * f:g}%: {v:.1f}" for f, v in zip(fracs, mc)))
    viol = sum(int(np.sum(out[k])) for k in out if k.endswith("_violations"))
    mism = sum(int(np.sum(out[k])) for k in out
               if k.endswith("_knap_direct_mismatch"))
    n_el = out["n_eligible"]
    print(f"checks: budget violations {viol} (all selections, scenarios, "
          f"designs); {{o}}_tokens vs direct knapsack mismatches {mism}; "
          f"eligible {n_el.min()}–{n_el.max()} of {len(names)} "
          f"({int(priced.sum())} priced); estimation failed on "
          f"{S - S_ok} splits"
          + (f"; cell-cost fallbacks {int(out['cell_fallback'].sum())} "
             f"(split x benchmark)" if "cell_fallback" in out else ""))


PLOT_FILES = {"r2": "token_costs.pdf", "r2_suite": "token_costs_suite.pdf"}
METRIC_LABEL = {"r2": "Imputation $R^2$", "r2_suite": "Full-suite $R^2$"}


def plot_token_costs(results, metric="r2", path=None, suffix=""):
    """results: [(ds_name, out, paired)].  Rows = datasets; columns = the
    metric (r2: the paper's R², OUT/token_costs{suffix}.pdf; r2_suite:
    full-suite R², the primary metric, OUT/token_costs_suite{suffix}.pdf)
    vs f per objective (main knap: tokens, instances, uniform, random),
    then its paired differences tokens − instances and tokens − uniform
    per objective (95% Nadeau–Bengio band).
    """
    if not results:
        return None
    n_rows, n_cols = len(results), len(OBJECTIVES) + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols,
                                                      2.3 * n_rows),
                             squeeze=False)
    for r, (ds, out, paired) in enumerate(results):
        x = 100 * np.asarray(out["budgets_frac"], dtype=float)
        lookup = _lookup(paired)
        for ci, o in enumerate(OBJECTIVES):
            ax = axes[r, ci]
            for c in COST_MODELS + ["random"]:
                m = "random" if c == "random" else f"{o}_{c}"
                mu, _ = _nan_mean_std(out[f"main_knap_{m}_{metric}"], axis=0)
                ax.plot(x, mu, COST_STYLE[c], color=COST_COLOR[c],
                        markersize=3.5, linewidth=1.5 if c == "tokens"
                        else 1.0, label=("random" if c == "random"
                                         else COST_LABEL[c]))
            ax.set_title(f"{display_name(ds)}: {OBJECTIVE_LABEL[o]}",
                         fontsize=8)
            if ci == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=8)
        for k, cb in enumerate(("instances", "uniform")):
            ax = axes[r, len(OBJECTIVES) + k]
            ax.axhline(0.0, color="#888888", linewidth=0.8)
            for o in OBJECTIVES:
                rows = [lookup[("main", "knap", metric, float(f),
                                f"{o}_tokens", f"{o}_{cb}")]
                        for f in out["budgets_frac"]]
                mu = np.array([p["mean_diff"] for p in rows], dtype=float)
                lo = np.array([p["ci_low"] for p in rows], dtype=float)
                hi = np.array([p["ci_high"] for p in rows], dtype=float)
                ax.plot(x, mu, OBJECTIVE_MARKER[o] + "-",
                        color=OBJECTIVE_COLOR[o], markersize=3,
                        linewidth=1.2, label=OBJECTIVE_LABEL[o])
                ax.fill_between(x, lo, hi, color=OBJECTIVE_COLOR[o],
                                alpha=0.12)
            ax.set_title(f"$\\Delta$ tokens − {cb}", fontsize=8)
        for ax in axes[r]:
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}" for v in x], fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            ax.grid(True, alpha=0.3)
            if r == n_rows - 1:
                ax.set_xlabel("Token budget (% of suite)", fontsize=7)
    axes[0, 0].legend(fontsize=6, loc="lower right")
    axes[0, len(OBJECTIVES)].legend(fontsize=6, loc="lower left")
    plt.tight_layout()
    if path is None:
        stem = PLOT_FILES[metric][:-len(".pdf")]
        path = OUT / f"{stem}{suffix}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)
    return path


def log_stem(ds_name, mode=DEFAULT_CAVEAT_MODE):
    """File stem of a dataset's logs under a --cost-caveats mode."""
    return f"token_costs_{ds_name}{mode_suffix(mode)}"


def load_log(ds_name, mode=DEFAULT_CAVEAT_MODE):
    """(out, paired) of an existing LOGS/token_costs_{ds}{sfx}.npz (paired
    statistics recomputed)."""
    with np.load(LOGS / f"{log_stem(ds_name, mode)}.npz",
                 allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    return out, paired_rows(ds_name, out)


# ── Fixed per-benchmark overhead (--fixed-cost) ──────────────────

def fixed_tables(tables, ratio):
    """(tables, c0, c0_instances) of overhead level `ratio` (prepare_costs'
    tables): main true cost c0 + true_main_j with c0 = ratio x the median
    of true_main over the priced benchmarks, and the instances cost model
    instances_j + ratio x the median priced instance count.  Ratio 0
    returns the tables themselves (the plain main scenario)."""
    ratio = float(ratio)
    if not (math.isfinite(ratio) and ratio >= 0.0):
        raise ValueError(f"fixed-cost ratio must be finite and >= 0, got "
                         f"{ratio!r}")
    if ratio == 0.0:
        return tables, 0.0, 0.0
    priced = np.asarray(tables["priced"], dtype=bool)
    true = np.asarray(tables["true_main"], dtype=float)
    inst = np.asarray(tables["instances"], dtype=float)
    c0 = ratio * float(np.median(true[priced]))
    i0 = ratio * float(np.median(inst[priced]))
    return dict(tables, true_main=true + c0, instances=inst + i0), c0, i0


def _fixed_level_key(key):
    """Whether a run_arrays key is a per-split array that gets a level axis."""
    return key.startswith("main_") or key == "max_count"


def run_fixed_arrays(B, O, benchmark_names, tables, budgets_frac, ratios,
                     n_repeats=N_REPEATS, n_folds=N_FOLDS, verbose=True,
                     bounds=None):
    """run_arrays' main scenario at every overhead level (fixed_tables) on
    the same matched splits; returns the save dict of
    LOGS/token_fixed_{ds}{sfx}.npz (module docstring).  Level-independent
    arrays (split metadata, eligibility, published costs) are checked to
    agree across levels."""
    if "true_main" not in tables:
        tables = prepare_costs(tables, benchmark_names)
    ratios = [float(r) for r in ratios]
    levels = [fixed_tables(tables, r) for r in ratios]
    outs = []
    for r, (tab, c0, i0) in zip(ratios, levels):
        t0 = time.time()
        o = run_arrays(B, O, benchmark_names, tab, budgets_frac, n_repeats,
                       n_folds, ["main"], verbose=False, bounds=bounds)
        outs.append(o)
        if verbose:
            print(f"  overhead {r:g} x median: c0 = {c0:.6g} tokens "
                  f"({i0:.6g} instances) per benchmark, suite "
                  f"{float(o['total_tokens']):.6g} tokens; "
                  f"{len(o['split_id'])} splits done "
                  f"({time.time() - t0:.1f}s)", flush=True)
    out = {}
    for k, v in outs[0].items():
        if _fixed_level_key(k):
            out[k] = np.stack([o[k] for o in outs], axis=1)
        elif k in FIXED_LEVEL_VECTORS:
            out[k] = np.stack([np.asarray(o[k]) for o in outs], axis=0)
        else:
            for o in outs[1:]:
                if not np.array_equal(np.asarray(o[k]), np.asarray(v)):
                    raise RuntimeError(f"{k} differs across overhead levels")
            out[k] = v
    priced = np.asarray(tables["priced"], dtype=bool)
    out.update(
        fixed_ratios=np.array(ratios, dtype=np.float64),
        fixed_costs=np.array([c0 for _, c0, _ in levels], dtype=np.float64),
        fixed_costs_instances=np.array([i0 for _, _, i0 in levels],
                                       dtype=np.float64),
        costs_instances_base=np.asarray(tables["instances"], dtype=float),
        median_tokens=np.float64(np.median(
            np.asarray(tables["true_main"], dtype=float)[priced])),
        median_instances=np.float64(np.median(
            np.asarray(tables["instances"], dtype=float)[priced])))
    return out


def fixed_level_view(out, f):
    """Level f of a token_fixed save dict in run_arrays' layout (scenario
    main), for summary_tables / paired_rows."""
    view = {}
    for k, v in out.items():
        if _fixed_level_key(k):
            view[k] = v[:, f]
        elif k in FIXED_LEVEL_VECTORS:
            view[k] = v[f]
        else:
            view[k] = v
    return view


def fixed_paired_rows(ds_name, out):
    """paired_rows of every overhead level, with the level's fixed_ratio and
    fixed_cost (c0 in tokens) as columns (FIXED_PAIRED_FIELDS)."""
    rows = []
    for f, (ratio, c0) in enumerate(zip(out["fixed_ratios"],
                                        out["fixed_costs"])):
        for r in paired_rows(ds_name, fixed_level_view(out, f)):
            rows.append({"fixed_ratio": float(ratio), "fixed_cost": float(c0),
                         **r})
    return rows


def _fixed_lookup(paired):
    return {(r["fixed_ratio"], r["scenario"], r["design"], r["metric"],
             r["budget_frac"], r["method_a"], r["method_b"]): r
            for r in paired}


def fixed_macro_prefix(mode):
    """Macro prefix of a --fixed-cost run (TokFixed for the default
    --cost-caveats mode, else TokFixed{Mode})."""
    return "TokFixed" + ("" if mode == DEFAULT_CAVEAT_MODE else camel(mode))


def fixed_macros(ds_name, out, paired, undefined=None):
    """Flat {MacroName: finite number} of a --fixed-cost run (module
    docstring); undefined statistics are left out and their names appended
    to `undefined` if given."""
    P, D = fixed_macro_prefix(_out_mode(out)), camel(ds_name)
    ok = ~out["estimation_failed"]
    F = ["F" + num_words(float(r)) for r in out["fixed_ratios"]]
    W = ["B" + num_words(100.0 * float(f)) for f in out["budgets_frac"]]
    mac = {}

    def put(name, v):
        if not (name.isascii() and name.isalpha()):
            raise ValueError(f"macro name {name!r} is not letters only")
        if name in mac:
            raise ValueError(f"duplicate macro {name}")
        v = int(v) if isinstance(v, (bool, np.bool_, np.integer)) else v
        if isinstance(v, int) or (isinstance(v, float) and math.isfinite(v)):
            mac[name] = v
        elif undefined is not None:
            undefined.append(name)

    priced = np.asarray(out["priced"], dtype=bool)
    put(f"{P}MedianTokens{D}", float(out["median_tokens"]))
    put(f"{P}MedianInstances{D}", float(out["median_instances"]))
    put(f"{P}NBench{D}", int(len(priced)))
    put(f"{P}NPriced{D}", int(priced.sum()))
    put(f"{P}NSplits{D}", int(ok.sum()))
    put(f"{P}Violations{D}", int(sum(
        int(np.sum(out[k])) for k in out if k.endswith("_violations"))))
    put(f"{P}DirectMismatch{D}", int(np.sum(out["main_knap_direct_mismatch"])))
    lookup = _fixed_lookup(paired)
    for f, Fw in enumerate(F):
        ratio = float(out["fixed_ratios"][f])
        budgets = out["main_budgets"][:, f]
        put(f"{P}Overhead{D}{Fw}", float(out["fixed_costs"][f]))
        put(f"{P}OverheadInstances{D}{Fw}",
            float(out["fixed_costs_instances"][f]))
        put(f"{P}TotalTokens{D}{Fw}", float(out["total_tokens"][f]))
        for b, Bw in enumerate(W):
            put(f"{P}MaxCount{D}{Fw}{Bw}",
                float(np.mean(out["max_count"][ok, f, b])))
            put(f"{P}Empty{D}{Fw}{Bw}", int(np.sum(
                out["main_knap_random_size"][ok, f, b] == 0)))
        for m in METHODS:
            k = _key("main", "knap", m)
            M_ = _macro_method(m)
            size = out[f"{k}_size"][ok, f].astype(float).mean(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                spent = _nan_mean_std((out[f"{k}_cost"][:, f]
                                       / budgets)[ok], axis=0)[0]
            for metric, T in METRICS.items():
                mu, sd = _nan_mean_std(out[f"{k}_{metric}"][ok, f], axis=0)
                for b, Bw in enumerate(W):
                    put(f"{P}{T}Rsq{D}{M_}{Fw}{Bw}", float(mu[b]))
                    put(f"{P}{T}RsqStd{D}{M_}{Fw}{Bw}", float(sd[b]))
            for b, Bw in enumerate(W):
                put(f"{P}Size{D}{M_}{Fw}{Bw}", float(size[b]))
                put(f"{P}Spent{D}{M_}{Fw}{Bw}", float(spent[b]))
        for ma, mb in FIXED_PAIRS:
            pair = f"{_macro_method(ma)}Vs{_macro_method(mb)}"
            for metric, T in METRICS.items():
                n_pos = n_neg = 0
                for b, Bw in enumerate(W):
                    r = lookup[(ratio, "main", "knap", metric,
                                float(out["budgets_frac"][b]), ma, mb)]
                    for sname, field in PAIR_STATS:
                        v = r[field]
                        put(f"{P}{T}Pair{sname}{D}{pair}{Fw}{Bw}",
                            int(v) if field == "n_splits" else float(v))
                    if _sig(r):
                        n_pos += int(r["ci_low"] > 0)
                        n_neg += int(r["ci_high"] < 0)
                put(f"{P}{T}SigPos{D}{pair}{Fw}", n_pos)
                put(f"{P}{T}SigNeg{D}{pair}{Fw}", n_neg)
    return mac


FIXED_NOTES = {
    "cost_model": "true cost of benchmark j = c0 + tokens_j (main scenario, "
                  "after the cost caveats), c0 = fixed_ratio x the median "
                  "true cost of the priced benchmarks",
    "budgets": "B_f = f x sum_j (c0 + tokens_j) over the priced benchmarks; "
               "fixed_ratio 0 = the plain main scenario (same budgets, "
               "selections and R²)",
    "cost_models": "tokens = c0 + tokens_j (true); instances = instances_j "
                   "+ fixed_ratio x median instances (the overhead in the "
                   "proxy's units); uniform = 1 (cardinality)",
    "methods": "feasibility always in true cost (knap: proxy_budgeted_"
               "select; fill: order_fill; random: the shared permutation, "
               "seed tag SEED + 7, filled skip-and-continue); no new "
               "random-number stream",
    "scope": "main scenario only (cell and single are not run)",
    "human_evaluation": "the burden of human evaluation (annotators, review "
                        "of open-ended outputs) is a further per-benchmark "
                        "cost; no data, not modelled",
}


def fixed_json(ds_name, out, tables, paired, macros, config, undefined=None):
    """The LOGS/token_fixed_{ds}{sfx}.json payload."""
    names = [str(b) for b in out["benchmark_names"]]
    priced = np.asarray(out["priced"], dtype=bool)
    key_rows = [r for r in paired if r["design"] == "knap"
                and (r["method_a"], r["method_b"]) in FIXED_PAIRS]
    viol = {k[:-len("_violations")]: int(np.sum(out[k]))
            for k in out if k.endswith("_violations")}
    unpriced = np.flatnonzero(~priced)
    checks = {"violations": viol, "violations_total": sum(viol.values()),
              "direct_mismatch_total": int(np.sum(
                  out["main_knap_direct_mismatch"])),
              "unpriced_selected": int(sum(
                  int(np.sum(np.isin(out[k], unpriced)))
                  for k in out if k.endswith("_selected"))),
              "estimation_failed": int(np.sum(out["estimation_failed"])),
              "n_eligible_min": int(np.min(out["n_eligible"])),
              "n_eligible_max": int(np.max(out["n_eligible"]))}
    levels = []
    for f, ratio in enumerate(out["fixed_ratios"]):
        view = fixed_level_view(out, f)
        tab = summary_tables(view)["main"]
        levels.append({"fixed_ratio": float(ratio),
                       "fixed_cost_tokens": float(out["fixed_costs"][f]),
                       "fixed_cost_instances":
                           float(out["fixed_costs_instances"][f]),
                       "total_tokens": float(out["total_tokens"][f]),
                       "budgets_mean": np.mean(out["main_budgets"][:, f],
                                               axis=0),
                       "max_count_mean": np.mean(
                           out["max_count"][~out["estimation_failed"], f],
                           axis=0),
                       "summary": tab})
    base = np.asarray(out["costs_true_main"][0], dtype=float) - float(
        out["fixed_costs"][0])
    return json_safe({
        "script": "eval_token_costs.py --fixed-cost", "dataset": ds_name,
        "config": config, "notes": FIXED_NOTES,
        "primary_metric": PRIMARY_METRIC, "metric_notes": METRIC_NOTES,
        "cost_table": [{"benchmark": nm, "tokens_true_main": base[j],
                        "instances": out["costs_instances_base"][j],
                        "priced": bool(priced[j]),
                        "caveat": bool(out["caveat_flag"][j])}
                       for j, nm in enumerate(names)],
        "cost_caveats": {"mode": _out_mode(out),
                         "flagged": tables.get("caveats", [])},
        "median_tokens": float(out["median_tokens"]),
        "median_instances": float(out["median_instances"]),
        "budgets_frac": out["budgets_frac"], "levels": levels,
        "paired_key_pairs": key_rows, "checks": checks,
        "runtime_s": config.get("runtime_s"),
        "n_macros": len(macros), "macros": macros,
        "macros_undefined": sorted(undefined or []),
        "macros_undefined_note": ("statistics without a finite value (no "
                                  "selection, or no split where both "
                                  "values of a pair are finite); not "
                                  "emitted as macros")})


def print_fixed_summary(ds_name, out, paired):
    fracs = np.asarray(out["budgets_frac"], dtype=float)
    ok = ~out["estimation_failed"]
    S, S_ok = len(out["split_id"]), int(ok.sum())
    lookup = _fixed_lookup(paired)
    for f, ratio in enumerate(out["fixed_ratios"]):
        ratio = float(ratio)
        print(f"\n== {display_name(ds_name)} [main, knap], fixed overhead "
              f"c0 = {ratio:g} x median = {out['fixed_costs'][f]:.4g} "
              f"tokens per benchmark (suite {out['total_tokens'][f]:.4g}): "
              f"full-suite R² (primary) / R² / #selected vs true budget "
              f"(means over {S_ok} of {S} splits) ==")
        print(f"{'method':<20}" + "".join(f"{100 * x:>21g}%" for x in fracs))
        for m in METHODS:
            k = _key("main", "knap", m)
            cells = []
            for b in range(len(fracs)):
                ms = _nan_mean_std(out[f"{k}_r2_suite"][ok, f, b])[0]
                mu = _nan_mean_std(out[f"{k}_r2"][ok, f, b])[0]
                sz = out[f"{k}_size"][ok, f, b].mean()
                cells.append(f"{ms:.3f}/{mu:.3f}/{sz:.1f}")
            print(f"{method_label(m):<20}" + "".join(f"{x:>22}"
                                                     for x in cells))
        for metric in metric_order():
            print(f"paired {metric} differences (mean; * = Nadeau–Bengio 95% "
                  f"CI excludes 0; (n) = paired splits where fewer than "
                  f"{S_ok}):")
            for ma, mb in FIXED_PAIRS:
                cells = []
                for x in fracs:
                    r = lookup[(ratio, "main", "knap", metric, float(x), ma,
                                mb)]
                    ns = int(r["n_splits"])
                    cells.append("n/a" if ns == 0 else
                                 f"{r['mean_diff']:+.3f}"
                                 f"{'*' if _sig(r) else ' '}"
                                 + (f"({ns})" if ns < S_ok else ""))
                print(f"  {ma + ' − ' + mb:<36}"
                      + "".join(f"{x:>13}" for x in cells))
    viol = sum(int(np.sum(out[k])) for k in out if k.endswith("_violations"))
    mism = int(np.sum(out["main_knap_direct_mismatch"]))
    print(f"\nchecks: budget violations {viol} (every level, design, method "
          f"and budget); {{o}}_tokens vs direct knapsack mismatches {mism}; "
          f"estimation failed on {S - S_ok} splits")


def plot_fixed(results, path=None, suffix="", metric=PRIMARY_METRIC):
    """results: [(ds_name, out, paired)] of --fixed-cost runs.  Per dataset
    two rows: the paired `metric` difference {o}_tokens − random and
    {o}_tokens − {o}_uniform per objective (main knap, 95% Nadeau–Bengio
    band) vs f; one column per overhead level."""
    if not results:
        return None
    ratios = np.asarray(results[0][1]["fixed_ratios"], dtype=float)
    n_rows, n_cols = 2 * len(results), len(ratios)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols,
                                                      2.0 * n_rows),
                             squeeze=False, sharey="row")
    for d, (ds, out, paired) in enumerate(results):
        x = 100 * np.asarray(out["budgets_frac"], dtype=float)
        lookup = _fixed_lookup(paired)
        for f, ratio in enumerate(np.asarray(out["fixed_ratios"], float)):
            for r0, cb in enumerate(("random", "uniform")):
                ax = axes[2 * d + r0, f]
                ax.axhline(0.0, color="#888888", linewidth=0.8)
                for o in OBJECTIVES:
                    mb = "random" if cb == "random" else f"{o}_{cb}"
                    rows = [lookup[(float(ratio), "main", "knap", metric,
                                    float(fr), f"{o}_tokens", mb)]
                            for fr in out["budgets_frac"]]
                    mu = np.array([p["mean_diff"] for p in rows], float)
                    lo = np.array([p["ci_low"] for p in rows], float)
                    hi = np.array([p["ci_high"] for p in rows], float)
                    ax.plot(x, mu, OBJECTIVE_MARKER[o] + "-",
                            color=OBJECTIVE_COLOR[o], markersize=3,
                            linewidth=1.2, label=OBJECTIVE_LABEL[o])
                    ax.fill_between(x, lo, hi, color=OBJECTIVE_COLOR[o],
                                    alpha=0.12)
                if d == 0 and r0 == 0:
                    ax.set_title(f"$c_0$ = {ratio:g} × median", fontsize=8)
                if f == 0:
                    ax.set_ylabel(f"{display_name(ds)}\n$\\Delta$ tokens − "
                                  f"{cb}", fontsize=7)
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Token budget (% of suite incl. $c_0$)", fontsize=7)
    axes[0, 0].legend(fontsize=6, loc="best")
    fig.suptitle(f"Fixed per-benchmark overhead: paired "
                 f"{METRIC_LABEL[metric]} difference (true-cost knapsack)",
                 fontsize=9)
    plt.tight_layout()
    path = OUT / f"{FIXED_STEM}{suffix}.pdf" if path is None else path
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)
    return path


def fixed_log_stem(ds_name, mode=DEFAULT_CAVEAT_MODE):
    """File stem of a dataset's --fixed-cost logs under a caveat mode."""
    return f"{FIXED_STEM}_{ds_name}{mode_suffix(mode)}"


def load_fixed_log(ds_name, mode=DEFAULT_CAVEAT_MODE):
    """(out, paired) of an existing LOGS/token_fixed_{ds}{sfx}.npz."""
    with np.load(LOGS / f"{fixed_log_stem(ds_name, mode)}.npz",
                 allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    return out, fixed_paired_rows(ds_name, out)


def main_fixed(args, ratios, mode):
    """--fixed-cost: run_fixed_arrays per dataset, its CSV, JSON, figure."""
    sfx = mode_suffix(mode)
    total_t0 = time.time()
    results = []
    for ds_name in args.datasets:
        t0 = time.time()
        B, O, names, tables = load_dataset(ds_name, mode)
        out = run_fixed_arrays(B, O, names, tables, args.budgets, ratios,
                               args.repeats, args.folds,
                               bounds=dataset_bounds(ds_name, names, DATA))
        out.update(matrix_sha256=np.array(_sha256(matrix_path(ds_name))),
                   meta_sha256=np.array(_sha256(meta_path(ds_name))))
        paired = fixed_paired_rows(ds_name, out)
        print_fixed_summary(ds_name, out, paired)
        undefined = []
        macros = fixed_macros(ds_name, out, paired, undefined)
        stem = fixed_log_stem(ds_name, mode)
        np.savez_compressed(LOGS / f"{stem}.npz", **out)
        print(f"Results saved to {LOGS / f'{stem}.npz'}", flush=True)
        write_csv(paired, LOGS / f"{stem}_paired.csv", FIXED_PAIRED_FIELDS)
        print(f"Paired statistics saved to {LOGS / f'{stem}_paired.csv'}",
              flush=True)
        config = {"dataset": ds_name, "repeats": args.repeats,
                  "folds": args.folds, "holdout_frac": HOLDOUT_FRAC,
                  "seed": SEED, **protocol_config(),
                  "random_seed_tag": SEED + RANDOM_SEED_OFFSET,
                  "budgets_frac": args.budgets, "fixed_ratios": ratios,
                  "scenarios": ["main"], "designs": DESIGNS,
                  "objectives": OBJECTIVES, "cost_models": COST_MODELS,
                  "methods": METHODS, "fixed_pairs": FIXED_PAIRS,
                  "cost_caveats": mode,
                  "macro_prefix": fixed_macro_prefix(mode),
                  "primary_metric": PRIMARY_METRIC,
                  "matrix": matrix_path(ds_name).name,
                  "matrix_sha256": str(out["matrix_sha256"]),
                  "meta": meta_path(ds_name).name,
                  "meta_sha256": str(out["meta_sha256"]),
                  "runtime_s": round(time.time() - t0, 1)}
        report = fixed_json(ds_name, out, tables, paired, macros, config,
                            undefined)
        jpath = LOGS / f"{stem}.json"
        jpath.write_text(json.dumps(report, indent=1, allow_nan=False))
        print(f"Summary saved to {jpath} ({len(macros)} macros, "
              f"{len(undefined)} undefined statistics left out)", flush=True)
        results.append((ds_name, out, paired))
        print(f"{display_name(ds_name)} completed in {time.time() - t0:.0f}s",
              flush=True)
    plot_fixed(results, suffix=sfx)
    print(f"\nAll datasets completed in {time.time() - total_t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Budgeted selection under true token costs (HELM): "
                    "selector cost model tokens vs instances vs uniform, on "
                    "the matched repeated-CV splits.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names with data/<name>.matrix.npz and "
                             "a HELM data/<name>.meta.json (benchmark_cost, "
                             "cell_costs); default: %(default)s")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions; splits = repeats x folds, rep "
                             "0 = the original seed-42 folds (default: "
                             "%(default)s)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=DEFAULT_BUDGETS,
                        help="true token budgets as fractions of the total "
                             "token cost (default: %(default)s)")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                        choices=SCENARIOS + ["requests"],
                        help="main (cost_tokens_requests_mean: all requests "
                             "per instance), cell (validation-model cell "
                             "costs; cost models from training models), "
                             "single (cost_tokens_mean, one request per "
                             "instance, only where it differs); main is "
                             "always run; 'requests' (the request-corrected "
                             "scenario of earlier runs) is now main and is "
                             "accepted as an alias (default: all)")
    parser.add_argument("--cost-caveats", choices=CAVEAT_MODES,
                        default=DEFAULT_CAVEAT_MODE,
                        help="benchmarks whose published token cost the "
                             "data release marks as wrong (COST_CAVEATS: "
                             "helm_classic OpenbookQA): impute (from HELM's "
                             "multiple-choice ablation), exclude (not "
                             "purchasable, still a target) or keep (as "
                             "published); a non-default mode writes "
                             "token_costs_<ds>_<mode>.* and macros "
                             "Tok<Mode>... (default: %(default)s)")
    parser.add_argument("--plot-only", action="store_true",
                        help="only redraw OUT/token_costs{,_suite}.pdf from "
                             "existing LOGS/token_costs_<ds>.npz (of "
                             "--cost-caveats; with --fixed-cost: "
                             "OUT/token_fixed.pdf from "
                             "LOGS/token_fixed_<ds>.npz)")
    parser.add_argument(
        "--fixed-cost", nargs="*", type=float, default=None,
        metavar="RATIO",
        help="opt-in fixed per-benchmark overhead: true cost c0 + tokens_j "
             "with c0 = RATIO x the median main true cost of the priced "
             "benchmarks, one level per RATIO (flag without values: "
             + " ".join(f"{r:g}" for r in DEFAULT_FIXED_RATIOS)
             + "); budgets are fractions of sum_j (c0 + tokens_j), the "
             "instances cost model gets RATIO x the median instance count, "
             "main scenario only; writes LOGS/token_fixed_<ds>.{npz,json}, "
             "LOGS/token_fixed_<ds>_paired.csv and OUT/token_fixed.pdf "
             "instead of the token_costs_* outputs (default: off)")
    args = parser.parse_args()
    if args.repeats < 1 or args.folds < 2:
        parser.error("need --repeats >= 1 and --folds >= 2")
    if any(not (np.isfinite(f) and 0 < f <= 1) for f in args.budgets):
        parser.error("--budgets must be fractions in (0, 1]")
    if args.fixed_cost is not None:
        if any(not (np.isfinite(r) and r >= 0) for r in args.fixed_cost):
            parser.error("--fixed-cost ratios must be finite and >= 0")
        if len(set(args.fixed_cost)) != len(args.fixed_cost):
            parser.error("--fixed-cost ratios must be distinct")
        fixed_ratios = sorted(args.fixed_cost or DEFAULT_FIXED_RATIOS)
        if args.plot_only:
            found = [ds for ds in args.datasets if (
                LOGS / f"{fixed_log_stem(ds, args.cost_caveats)}.npz"
            ).is_file()]
            if not found:
                parser.error(f"no {LOGS}/{FIXED_STEM}_<ds>"
                             f"{mode_suffix(args.cost_caveats)}.npz for "
                             f"{args.datasets}")
            plot_fixed([(ds, *load_fixed_log(ds, args.cost_caveats))
                        for ds in found],
                       suffix=mode_suffix(args.cost_caveats))
            sys.exit(0)
        missing = [ds for ds in args.datasets
                   if not (matrix_path(ds).is_file()
                           and meta_path(ds).is_file())]
        if missing:
            parser.error(f"no {DATA}/<name>.matrix.npz + .meta.json for "
                         f"{missing}")
        main_fixed(args, fixed_ratios, args.cost_caveats)
        sys.exit(0)
    if "requests" in args.scenarios:
        print("note: --scenarios requests is the main scenario now "
              "(true cost = all requests per instance)", flush=True)
    scenarios = ["main"] + [s for s in SCENARIOS
                            if s != "main" and s in args.scenarios]
    mode = args.cost_caveats
    sfx = mode_suffix(mode)
    if args.plot_only:
        found = [ds for ds in args.datasets
                 if (LOGS / f"{log_stem(ds, mode)}.npz").is_file()]
        if not found:
            parser.error(f"no {LOGS}/token_costs_<ds>{sfx}.npz for "
                         f"{args.datasets}")
        logged = [(ds, *load_log(ds, mode)) for ds in found]
        for metric in METRICS:
            plot_token_costs(logged, metric, suffix=sfx)
        sys.exit(0)
    missing = [ds for ds in args.datasets
               if not (matrix_path(ds).is_file() and meta_path(ds).is_file())]
    if missing:
        parser.error(f"no {DATA}/<name>.matrix.npz + .meta.json for "
                     f"{missing}")

    total_t0 = time.time()
    results = []
    for ds_name in args.datasets:
        t0 = time.time()
        out, tables = run(ds_name, args.budgets, args.repeats, args.folds,
                          scenarios, mode)
        out["runtime_s"] = np.float64(time.time() - t0)
        paired = paired_rows(ds_name, out)
        print_summary(ds_name, out, tables, paired)
        undefined = []
        macros = macros_of(ds_name, out, tables, paired, undefined)
        config = {"dataset": ds_name, "repeats": args.repeats,
                  "folds": args.folds, "holdout_frac": HOLDOUT_FRAC,
                  "seed": SEED, **protocol_config(),
                  "random_seed_tag": SEED + RANDOM_SEED_OFFSET,
                  "budgets_frac": args.budgets,
                  "scenarios": _scenarios(out), "designs": DESIGNS,
                  "objectives": OBJECTIVES, "cost_models": COST_MODELS,
                  "methods": METHODS, "cost_caveats": mode,
                  "macro_prefix": macro_prefix(mode),
                  "primary_metric": PRIMARY_METRIC,
                  "true_cost": {
                      "main": "meta.benchmark_cost.cost_tokens_requests_mean"
                              " (= cost_tokens_mean x requests_per_instance)"
                              ", flagged benchmarks per cost_caveats",
                      "cell": "mean over the validation models of "
                              "meta.cell_costs.cost_tokens x costs_mult_req",
                      "single": "meta.benchmark_cost.cost_tokens_mean (one "
                                "request per instance), flagged benchmarks "
                                "per cost_caveats"},
                  "matrix": matrix_path(ds_name).name,
                  "matrix_sha256": str(out["matrix_sha256"]),
                  "meta": meta_path(ds_name).name,
                  "meta_sha256": str(out["meta_sha256"])}
        stem = log_stem(ds_name, mode)
        np.savez_compressed(LOGS / f"{stem}.npz", **out)
        print(f"Results saved to {LOGS / f'{stem}.npz'}", flush=True)
        write_csv(paired, LOGS / f"{stem}_paired.csv", PAIRED_FIELDS)
        print(f"Paired statistics saved to {LOGS / f'{stem}_paired.csv'}",
              flush=True)
        report = json_report(ds_name, out, tables, paired, macros, config,
                             undefined)
        jpath = LOGS / f"{stem}.json"
        jpath.write_text(json.dumps(report, indent=1, allow_nan=False))
        print(f"Summary saved to {jpath} ({len(macros)} macros, "
              f"{len(undefined)} undefined statistics left out)", flush=True)
        results.append((ds_name, out, paired))
        print(f"{display_name(ds_name)} completed in {time.time() - t0:.0f}s",
              flush=True)
    for metric in METRICS:
        plot_token_costs(results, metric, suffix=sfx)
    print(f"\nAll datasets completed in {time.time() - total_t0:.0f}s",
          flush=True)
