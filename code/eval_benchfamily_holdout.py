r"""Benchmark-family holdout: does a selection made without a family still
predict it?

Reviewer request B3 (QSWq W2, "benchmark-family holdouts").
eval_family_holdout.py holds out MODEL families; this script holds out
BENCHMARK families.  A family F is a group of benchmarks that belong
together (an MMLU supercategory, an MTEB task type, a source leaderboard
of Merged, a parent benchmark of the OLL v2 sub-tasks).  The scenario is
a new family that the selection did not anticipate: the training models
do have F's scores (so imputation can use them), but the selection must
predict F from benchmarks outside F.

Design (matched splits, cv_split.cv_splits: --repeats x --folds splits,
10% holdout; repetition 0 = the historical seed-42 folds).  On a split
the correlation matrix R, the training standardization, the eligible
set (benchmarks observed for >= MIN_TRAIN_OBS training models,
cv_split.eligible_mask) and the imputation targets (benchmarks observed
for >= MIN_TARGET_OBS training models, cv_split.target_mask, protocol
rule D1: F's other cells are not scored) are computed once and shared by
every family, method and arm; every prediction is clipped to its
benchmark's valid score range (score_bounds.py, rule D2).  For each
held-out family F the candidates are the eligible benchmarks outside F
(n_candidates), and three arms select from them:

  blind   the selection never sees F: the selector runs on a correlation
          matrix re-estimated from the training models' non-F columns
          alone, R_blind = estimate_corr(B_train[:, keep], O_train[:,
          keep]) with keep = V \ F (--blind-cov refit, the default since
          2026-10-18).  Neither the objective nor the estimate involves F's
          training scores, so changing them cannot change a blind pick.
          --blind-cov sub (the default until 2026-10-18; outputs tagged
          _sub) runs the selector on R[keep, keep], the non-F block of the
          split's full estimate.  On a fully observed matrix (MMLU, OLL v2
          sub-tasks, HELM MMLU, OLL v1), where estimate_corr is the
          pairwise estimator, R[keep, keep] equals the refit up to its
          eigenvalue floor (identical picks and R^2 in the 2026-10-18 runs).
          On a sparse matrix (MTEB 77% observed, Merged 31%) estimate_corr
          is the EM fit, which imputes the non-F block from all columns,
          so under sub F's training scores shape R[keep, keep] and the
          blind arm is only blind to F in its objective;
          --compare-blind-cov measures the difference.
  aware   candidates outside F, objective over ALL benchmarks (F's
          benchmarks are targets, not candidates): the selector runs on
          the full R with eligible & ~F as its eligibility mask
  random  uniformly random order of the eligible non-F benchmarks, one
          arm shared by both modes: random_select on the full R with
          eligible & ~F and the random baseline's common random numbers
          Split.seed(SEED + RANDOM_SEED_OFFSET) (SEED + 7), i.e. the CV
          random baseline's permutation of all N benchmarks
          (eval_baselines) restricted to the family's candidates.  No new
          seed stream is used.
The aware and random arms use the split's full R and do not depend on
--blind-cov.  "aware - blind" is therefore the gain from letting the
selection know F, both in its objective and (under refit) in its estimate
of R.

Methods (eval_baselines.run_selection, the selectors of the main protocol
with their tie rules): mi (Alg. 2), entropy (Alg. 1), trace (greedy
total-variance), pca (CSSP), hub (mean |correlation|), facloc (greedy
facility location on the clipped correlation matrix,
baseline_select.facility_location_select; the paper's headline method
since 2026-10-19), each in a blind and an aware arm ({m}_blind,
{m}_aware), plus random.  What "aware" changes per method: mi's
complement, trace's residual trace, hub's mean |r|, PCA's eigenvectors
and facloc's coverage sum (its targets run over all benchmarks) include
F; entropy's objective log det R_AA
does not involve F at all, so entropy_blind and entropy_aware differ only
where a near-tie is broken by explained variance (always at k = 1 on a
correlation matrix: the hub of the non-F block vs of all benchmarks).
facloc draws no random numbers, so adding it changed no other arm's
values; the arms of APART_METHODS (facloc) stay out of the other arms'
common support (below), so their tables, paired rows and macros are those
of the logs without facloc.
k runs over 1 .. min(--kmax, N - 1); where a family has fewer than k
candidates every arm is NaN at that k.

Metric: pooled R^2 of eval_entropy_vs_mi.impute_and_evaluate (Gaussian
conditional mean from the selected benchmarks, full training R, ridge
0.01, train-standardized units, R^2 = 1 - SS_res / SS_tot pooled over
cells) restricted to F's cells: the observed F cells of the validation
models that observe at least one selected benchmark (the cell rules of
impute_and_evaluate; other unselected benchmarks are not scored).
family_r2 passes impute_and_evaluate an observation mask that keeps only
the selected and F columns, one call per group of validation models with
identical observation patterns on those columns, and pools the groups'
sums of squares.  Within a group every row has the same pattern, so the
result is exact whether or not the imputed function's vectorized fast
path is the flawed one (camera_ready/drafts/protocol_changes.md H1: it
scores only validation model 0's columns when all models agree on the
selected columns and observe model 0's targets); the imputation itself is
always impute_and_evaluate's.

Families (benchmark_families; FAMILY_SCHEME):
  mmlu_category  mmlu, helm_mmlu (and olv1items): the four MMLU
                 supercategories (STEM 18, humanities 13, social sciences
                 12, other 14 subjects) of hendrycks/test categories.py,
                 hard-coded below (MMLU_SUBCATEGORY, MMLU_CATEGORIES).
                 On mmlu and helm_mmlu every column must be one of the 57
                 subjects; on olv1items each non-MMLU task is its own
                 single-benchmark family.
  mteb_type      mteb, mteb2026: the MTEB task type of each task (mteb's
                 TaskMetadata.type, as recorded in benchmarks[].type of
                 data/mteb2026.meta.json, whose columns are those of
                 data/mteb.matrix.npz): Classification 12, Clustering 11,
                 PairClassification 3, Reranking 4, Retrieval 15, STS 10,
                 Summarization 1 (checked against MTEB_TYPE_COUNTS, the
                 English MTEB of Muennighoff et al. 2023).
  source         merged (default): the source leaderboard of each column
                 (the text before the first "/"), except that the copies
                 of one benchmark in two sources form one family
                 (MERGED_SHARED): "mmlu" = the 57 mmlu/* subjects, helm/MMLU
                 (HELM Lite's MMLU) and open_llm/MMLU-PRO (MMLU-Pro, built
                 in part from filtered MMLU questions; Wang et al. 2024);
                 "math" = helm/MATH and open_llm/MATH_Lvl5 (the level-5
                 subset of MATH).  So holding out MMLU removes every MMLU
                 copy from the candidates and from the blind objective.  10
                 families: mmlu 59, open_llm 4 (IFEval, BBH, GPQA, MUSR),
                 math 2, helm 9, alpaca_eval 3, arena_hard 1, livebench 3,
                 wildbench 11, mt_bench 8, bigcode 14.  Benchmarks of one
                 topic but different questions (helm/GSM8K vs MATH,
                 wildbench/Math, mt_bench/math, the coding columns) stay
                 in their source families, and so does helm/Mean win rate,
                 HELM's aggregate over its scenarios (MMLU and MATH among
                 them).
  prefix         olv2sub (parent benchmark: ifeval, bbh, math_hard, gpqa,
                 musr, mmlu_pro; checked against the meta's
                 benchmarks.group); merged with --merged-families prefix
                 (the scheme until 2026-10-18; outputs tagged _prefix: the
                 9 source leaderboards, so helm/MMLU and open_llm/MMLU-PRO
                 stay candidates when mmlu is held out).  The text before
                 the first "/" (the whole name if there is none).
Families observed for fewer than --min-family-models models of the whole
matrix (Merged's mt_bench: 3) are evaluated and logged but left out of
"all" (family_in_mean).

Common support (common_support): every summary compares the arms on the
same cells.  At each k, a (split, family) cell enters the tables, the
paired statistics, "all", the macros and the figure only if every arm has
a finite R^2 there (on sparse data an arm is NaN where no validation model
observes any of its picks, e.g. a random k = 1 pick on Merged).  So the
table columns can be subtracted, and their difference is the paired mean
difference of the same cells.  The npz keeps every arm's own values.
"Every arm" means the arms other than those of APART_METHODS (facloc);
an entry with a facloc arm (its tables, the paired rows with it, its
macros) uses the common support of those arms and both facloc arms
(support_for), i.e. the other arms' support minus the cells where a
facloc arm is NaN (diagnostics: cells_common_facloc and nan_cells; in the
--repeats 1 validation runs of 2026-10-19 on mmlu, mteb, merged, olv2sub,
helm_mmlu and olv1items the two supports coincided at every reported k).
Within a cell the scored models still depend on the selection (the cell
rules of impute_and_evaluate, as in the main protocol).

All families ("all", pooled_r2): per split, the R^2 pooled over the
held-out cells of the families in the common support, 1 - sum_f SS_res,f
/ max(sum_f SS_tot,f, 1e-12), each family's cells imputed from the
selection made with that family held out.  It is the paper's pooled R^2
on the union of the family cells.  An unweighted mean of per-family R^2 is
not used: a family with few target cells and z-scores near 0 (Merged's
arena_hard, one benchmark seen by a few validation models) has SS_tot near
0 and per-family R^2 of order -100, which would dominate it; pooled, such
a family weighs by its SS_tot.

Paired statistics (paired_stats.paired_stats: Nadeau-Bengio corrected t
interval over the splits, win / tie / loss) per family (and "all") and k
for
  blind - random   {m}_blind vs random
  aware - blind    {m}_aware vs {m}_blind
  aware - random   {m}_aware vs random
  blind, aware     method pairs within a mode (paired_stats.orient order)

Usage:
    python eval_benchfamily_holdout.py [--datasets mmlu mteb merged olv2sub
                                        helm_mmlu] [--repeats 5]
                                       [--folds 10] [--kmax 10]
                                       [--blind-cov refit|sub]
                                       [--merged-families source|prefix]
                                       [--min-family-models 10]
                                       [--from-logs] [--compare-blind-cov]

--from-logs re-reads LOGS/benchfamily_holdout_{tag}.npz and rewrites the
summary, paired CSV, JSON (runtime of the original run) and figure; it
refuses a log whose blind covariance or family scheme differs from the
options.  --compare-blind-cov (no recomputation) reads the refit log
{ds} and the sub log {ds}_sub of each dataset (run both first) and writes
LOGS/benchfamily_holdout_{ds}[_prefix]_blindcov.json: per method and k the
share of (split, family) cells whose size-k blind sets coincide (as sets,
and as pick sequences) and the paired refit - sub R^2 differences per
family and "all" (support: both runs' common supports), with the
significant ones listed.

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir();
tag = {ds}, + _sub with --blind-cov sub, + _prefix with
--merged-families prefix on Merged; until 2026-10-18 the default was sub
and --blind-cov refit outputs were tagged _refit):
  LOGS/benchfamily_holdout_{tag}.npz
    rep, fold, split_id, n_train, n_val, n_eligible   (S,) int, S splits
                     (rows in cv_splits order, rep-major)
    eligible (S, N) bool, estimation_failed (S,) bool
    {arm}_r2         (S, F, K) pooled R^2 on family f's cells at k = 1..K
                     (arm in ARMS: {m}_blind, {m}_aware, random)
    {arm}_ss_res, {arm}_ss_tot   (S, F, K) its sums of squares (NaN
                     where the R^2 is NaN; "all" pools them)
    {arm}_n_cells, {arm}_n_models   (S, F, K) int, scored F cells /
                     validation models (0 where the R^2 is NaN)
    {arm}_selected   (S, F, K) int, picks in order, -1 padded (pca: its
                     size-K set; pca_{mode}_selected_per_k (S, F, K, K)
                     row kk-1 = the size-kk set)
    {arm}_index_ties, {arm}_tie_breaks   (S, F, K) int, picks of the
                     size-k selection decided by the lowest index / made
                     among near-tied scores
    {m}_jacc_blind_aware   (S, F, K) Jaccard of the blind and aware sets
    n_candidates, n_target_cells, n_target_models   (S, F) int (eligible
                     non-F benchmarks; observed F cells / models among the
                     validation models)
    blind_refit_failed (S, F) bool (refit estimate failed: blind arms NaN)
    family_names, family_size, family_models (F,), family_in_mean (F,)
    family_of (N,) int (family index of each benchmark), benchmark_names,
    arms, methods, scheme, blind_cov, k_max, holdout_frac, n_folds,
    n_repeats, seed, min_train_obs, random_seed_offset, min_family_models
  LOGS/benchfamily_holdout_{tag}_paired.csv   PAIRED_FIELDS, every family
                     (and "all"), k = 1..K, comparison and pair
  LOGS/benchfamily_holdout_{tag}.json   config (families and members,
                     seeds), summary tables (mean / sd / n of every arm per
                     family and k in KS_REPORT, plus "all", on the common
                     support), the key paired rows at KS_REPORT,
                     diagnostics (common_support: cells with some finite
                     arm, cells in the support, NaN cells per arm, splits
                     of "all"), runtime, "macros"
  OUT/benchfamily_holdout_{tag}.pdf     R^2 vs k per family (blind solid,
                     aware dashed, random grey; common support) and the
                     paired differences blind - random and aware - blind of
                     "all" (95% NB interval bands)
  LOGS/benchfamily_holdout_{ds}[_prefix]_blindcov.json  (--compare-blind-cov)
                     agreement (set_equal, seq_equal per method and k),
                     paired refit - sub rows of every family and "all",
                     the significant ones, "macros"

Macros ("macros" in the JSON; letters only, make_numbers.camel names; {Ds}
dataset, {Fam} camel(family) or All, {M} method, K{k} spelled k in
KS_REPORT; prefix {P} = BenchFam, + Prefix with --merged-families prefix
on Merged, + Sub with --blind-cov sub):
  {P}Rsq{Ds}{Fam}{M}{Blind|Aware}K{k}, {P}Rsq{Ds}{Fam}RandomK{k}
      mean R^2 over the splits of the common support (All: of the
      per-split pooled R^2 of "all"); differences of two of them are the
      paired mean differences
  {P}Pair{S}{Ds}{Fam}{M}{BlindVsRandom|AwareVsBlind|AwareVsRandom}K{k}
      paired difference, S in Diff, Lo, Hi (Nadeau-Bengio 95% interval);
      for All also Se, Win, Wins, N
  {P}Jacc{Ds}All{M}K{k}   mean blind-aware Jaccard over splits and
      families; {P}NFam{Ds}, {P}NFamMean{Ds}, {P}Size{Ds}{Fam},
      {P}Models{Ds}{Fam}, {P}NCand{Ds}{Fam} (mean candidates),
      {P}NSplits{Ds}, {P}NFailed{Ds}, {P}NRefitFailed{Ds}
  --compare-blind-cov ({Q} = BenchFam[Prefix]Cov): {Q}SetEq{Ds}{M}K{k},
      {Q}SeqEq{Ds}{M}K{k} (greedy methods; not pca, whose size-k sets are
      not nested), {Q}Pair{Diff|Lo|Hi}{Ds}All{M}K{k} (refit - sub),
      {Q}MaxAbsDiff{Ds} (largest |paired mean difference| of "all" over
      methods and k = 1..K), {Q}NSig{Ds} / {Q}NPairs{Ds} (of those
      (method, k) rows, the ones whose interval excludes 0 / all),
      {Q}NSplits{Ds}; the methods there are those other than
      APART_METHODS, and {Q}MaxAbsDiff{Ds}Facloc, {Q}NSig{Ds}Facloc,
      {Q}NPairs{Ds}Facloc are the same over facloc's rows

Runtime (single-threaded, --repeats 5, 4090-1 shared with other jobs,
load 4-45, 2026-10-18; the refit adds one estimate_corr per family and
split, cheap on the pairwise path, one EM fit each on MTEB and Merged):
MMLU 55 s, MTEB 642 s (sub 113 s), Merged 5819 s (sub 725 s), OLL v2
sub-tasks 48 s, HELM MMLU 14 s, OLL v1 (olv1items) 31 s.  --workers W
runs the splits in W spawned processes (benchfam_split is a pure function
of the split; the arrays equal the serial run's, test_eval_benchfamily_
holdout checks it).
"""

import argparse
import json
import math
import pathlib
import sys
import time
import warnings
from collections import OrderedDict
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

from greedy_select import TIE_RTOL, index_tie_count, selection_at_k, \
    tie_break_count
from cv_split import (MIN_TRAIN_OBS, N_REPEATS, RANDOM_SEED_OFFSET,
                      cv_splits, eligible_mask, target_mask)
from eval_entropy_vs_mi import display_name, estimate_corr, \
    impute_and_evaluate, protocol_config
from score_bounds import dataset_bounds
import eval_baselines
from eval_baselines import METHOD_COLOR, METHOD_LABEL, run_selection
from paired_stats import STAT_FIELDS, orient, paired_stats, write_csv
from make_numbers import camel, int_words

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

K_MAX = 10
N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42                     # eval_baselines.SEED (run_selection's seeds)
assert SEED == eval_baselines.SEED
KS_REPORT = [1, 3, 5, 10]
RIDGE = 0.01                  # impute_and_evaluate's ridge (config only)
Z_CLIP = 10.0                 # impute_and_evaluate's clip of z-scores
SS_TOT_FLOOR = 1e-12          # impute_and_evaluate's floor of SS_tot
MIN_FAMILY_MODELS = 10
BLIND_COV = ["refit", "sub"]  # --blind-cov choices, default first

DATASETS = ["mmlu", "mteb", "merged", "olv2sub", "helm_mmlu"]
# paired_stats.METHOD_ORDER order (mi, entropy, trace, pca, hub; facloc,
# not in it, ranks after hub); facloc is the paper's headline method since
# 2026-10-19
METHODS = ["mi", "entropy", "trace", "pca", "hub", "facloc"]
MODES = ["blind", "aware"]
ARMS = [f"{m}_{mode}" for m in METHODS for mode in MODES] + ["random"]
# Methods whose arms stay out of the other arms' common support (and whose
# blind-cov counts are kept apart): the summaries of the other arms keep
# the cells they had before facloc was added (module docstring)
APART_METHODS = ["facloc"]
COMPARISONS = OrderedDict([          # name -> (arm a, arm b) of method m
    ("blind - random", lambda m: (f"{m}_blind", "random")),
    ("aware - blind", lambda m: (f"{m}_aware", f"{m}_blind")),
    ("aware - random", lambda m: (f"{m}_aware", "random")),
])
COMPARISON_MACRO = {"blind - random": "BlindVsRandom",
                    "aware - blind": "AwareVsBlind",
                    "aware - random": "AwareVsRandom"}
ALL = "all"                          # all families pooled (pooled_r2)

# ── Benchmark families ───────────────────────────────────────────

# MMLU subject -> subcategory and subcategory lists of the four
# supercategories, exactly as in hendrycks/test categories.py
# (https://github.com/hendrycks/test/blob/master/categories.py, read
# 2026-10-18; Hendrycks et al., "Measuring Massive Multitask Language
# Understanding", ICLR 2021).
MMLU_SUBCATEGORY = {
    "abstract_algebra": "math",
    "anatomy": "health",
    "astronomy": "physics",
    "business_ethics": "business",
    "clinical_knowledge": "health",
    "college_biology": "biology",
    "college_chemistry": "chemistry",
    "college_computer_science": "computer science",
    "college_mathematics": "math",
    "college_medicine": "health",
    "college_physics": "physics",
    "computer_security": "computer science",
    "conceptual_physics": "physics",
    "econometrics": "economics",
    "electrical_engineering": "engineering",
    "elementary_mathematics": "math",
    "formal_logic": "philosophy",
    "global_facts": "other",
    "high_school_biology": "biology",
    "high_school_chemistry": "chemistry",
    "high_school_computer_science": "computer science",
    "high_school_european_history": "history",
    "high_school_geography": "geography",
    "high_school_government_and_politics": "politics",
    "high_school_macroeconomics": "economics",
    "high_school_mathematics": "math",
    "high_school_microeconomics": "economics",
    "high_school_physics": "physics",
    "high_school_psychology": "psychology",
    "high_school_statistics": "math",
    "high_school_us_history": "history",
    "high_school_world_history": "history",
    "human_aging": "health",
    "human_sexuality": "culture",
    "international_law": "law",
    "jurisprudence": "law",
    "logical_fallacies": "philosophy",
    "machine_learning": "computer science",
    "management": "business",
    "marketing": "business",
    "medical_genetics": "health",
    "miscellaneous": "other",
    "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy",
    "nutrition": "health",
    "philosophy": "philosophy",
    "prehistory": "history",
    "professional_accounting": "other",
    "professional_law": "law",
    "professional_medicine": "health",
    "professional_psychology": "psychology",
    "public_relations": "politics",
    "security_studies": "politics",
    "sociology": "culture",
    "us_foreign_policy": "politics",
    "virology": "health",
    "world_religions": "philosophy",
}
MMLU_CATEGORIES = OrderedDict([
    ("STEM", ["physics", "chemistry", "biology", "computer science", "math",
              "engineering"]),
    ("humanities", ["history", "philosophy", "law"]),
    ("social sciences", ["politics", "culture", "economics", "geography",
                         "psychology"]),
    ("other (business, health, misc.)", ["other", "business", "health"]),
])
# Family identifiers of the supercategories (macro names camel(id))
MMLU_FAMILY_ID = {"STEM": "stem", "humanities": "humanities",
                  "social sciences": "social_sciences",
                  "other (business, health, misc.)": "other"}

# Task-type composition of the English MTEB (Muennighoff et al., "MTEB:
# Massive Text Embedding Benchmark", EACL 2023), the 56 columns of
# data/mteb.matrix.npz; mteb_types() warns if the meta file disagrees.
MTEB_TYPE_COUNTS = {"Classification": 12, "Clustering": 11,
                    "PairClassification": 3, "Reranking": 4, "Retrieval": 15,
                    "STS": 10, "Summarization": 1}
MTEB_META = "mteb2026.meta.json"     # benchmarks[].name / .type

# Merged ("source" scheme): columns that are copies of a benchmark of
# another source, with the family they join.  helm/MMLU is HELM Lite's
# MMLU (MMLU questions); open_llm/MMLU-PRO is MMLU-Pro (Wang et al.,
# NeurIPS 2024 D&B), whose questions are in part filtered MMLU questions
# with 10 options; open_llm/MATH_Lvl5 is the level-5 subset of MATH
# (Hendrycks et al. 2021), which helm/MATH also evaluates.
MERGED_SHARED = OrderedDict([("helm/MMLU", "mmlu"),
                             ("open_llm/MMLU-PRO", "mmlu"),
                             ("helm/MATH", "math"),
                             ("open_llm/MATH_Lvl5", "math")])

FAMILY_SCHEME = {"mmlu": "mmlu_category", "helm_mmlu": "mmlu_category",
                 "olv1items": "mmlu_category", "mteb": "mteb_type",
                 "mteb2026": "mteb_type", "merged": "source",
                 "olv2sub": "prefix"}
# --merged-families: the scheme of the datasets whose default is "source"
MERGED_FAMILIES = {"source": "source", "prefix": "prefix"}
# Datasets whose every column must be an MMLU subject
MMLU_ONLY = {"mmlu", "helm_mmlu"}
SCHEME_LABEL = {"mmlu_category": "MMLU supercategory (hendrycks/test)",
                "mteb_type": "MTEB task type",
                "source": "source leaderboard, cross-source copies of a "
                          "benchmark merged",
                "prefix": "column prefix (source / parent benchmark)"}


def matrix_path(ds_name):
    return DATA / f"{ds_name}.matrix.npz"


def mmlu_category(subject):
    """Family id of an MMLU subject (stem, humanities, social_sciences,
    other), None if `subject` is not one of the 57 subjects."""
    sub = MMLU_SUBCATEGORY.get(subject)
    if sub is None:
        return None
    for cat, subs in MMLU_CATEGORIES.items():
        if sub in subs:
            return MMLU_FAMILY_ID[cat]
    raise ValueError(f"subcategory {sub!r} of {subject!r} in no category")


def mteb_types(names, meta_path=None):
    """{task: MTEB task type} for every name, from benchmarks[].type of
    data/mteb2026.meta.json (or `meta_path`); raises if a name has none.
    Warns (stderr) if the type counts differ from MTEB_TYPE_COUNTS."""
    path = pathlib.Path(meta_path) if meta_path else DATA / MTEB_META
    with open(path) as f:
        meta = json.load(f)
    types = {str(b["name"]): str(b["type"]) for b in meta["benchmarks"]}
    missing = [n for n in names if n not in types]
    if missing:
        raise ValueError(f"no MTEB task type in {path} for {missing}")
    out = {n: types[n] for n in names}
    counts = {}
    for t in out.values():
        counts[t] = counts.get(t, 0) + 1
    if len(names) == sum(MTEB_TYPE_COUNTS.values()) \
            and counts != MTEB_TYPE_COUNTS:
        print(f"warning: MTEB task-type counts {counts} differ from "
              f"{MTEB_TYPE_COUNTS}", file=sys.stderr)
    return out


def _check_groups(ds_name, names, fam_of):
    """For olv2sub: the prefix families must coincide with the meta file's
    benchmarks.group partition (one group per prefix and vice versa)."""
    path = DATA / f"{ds_name}.meta.json"
    if not path.is_file():
        return
    with open(path) as f:
        meta = json.load(f)
    groups = (meta.get("benchmarks") or {}).get("group")
    cols = meta.get("benchmark_names")
    if not isinstance(groups, list) or not isinstance(cols, list):
        return
    if [str(c) for c in cols] != list(names) or len(groups) != len(names):
        raise ValueError(f"{path}: benchmark_names / group do not match "
                         f"the matrix columns")
    pairs = set(zip(fam_of, map(str, groups)))
    if len(pairs) != len(set(fam_of)) or len(pairs) != len(set(groups)):
        raise ValueError(f"{path}: prefix families {sorted(set(fam_of))} "
                         f"do not match the groups {sorted(set(groups))}")


def dataset_scheme(ds_name, merged_families="source"):
    """Family scheme of a dataset: FAMILY_SCHEME, with the "source"
    datasets (Merged) switched to MERGED_FAMILIES[merged_families]."""
    scheme = FAMILY_SCHEME.get(ds_name)
    if scheme is None:
        raise ValueError(f"no benchmark-family scheme for {ds_name!r}; "
                         f"known: {sorted(FAMILY_SCHEME)}")
    if scheme == "source":
        scheme = MERGED_FAMILIES[merged_families]
    return scheme


def benchmark_families(ds_name, names, mteb_meta=None, scheme=None):
    """(scheme, families) of a dataset: families is a list of (family id,
    ascending column indices) partitioning range(N), in FAMILY_SCHEME
    order (MMLU: STEM, humanities, social sciences, other, then any
    non-MMLU column as its own family; otherwise order of first column).
    `scheme` (default FAMILY_SCHEME[ds_name]) is "prefix" for Merged with
    --merged-families prefix."""
    names = [str(n) for n in names]
    if scheme is None:
        scheme = dataset_scheme(ds_name)
    if scheme not in SCHEME_LABEL:
        raise ValueError(f"unknown family scheme {scheme!r}")
    if scheme == "mmlu_category":
        fam_of = [mmlu_category(n) or n for n in names]
        extra = [n for n in names if mmlu_category(n) is None]
        if ds_name in MMLU_ONLY and extra:
            raise ValueError(f"{ds_name}: columns {extra} are not MMLU "
                             f"subjects")
        order = [MMLU_FAMILY_ID[c] for c in MMLU_CATEGORIES]
        order = [f for f in order if f in fam_of] + extra
    elif scheme == "mteb_type":
        types = mteb_types(names, mteb_meta)
        fam_of = [types[n] for n in names]
        order = list(OrderedDict.fromkeys(fam_of))
    elif scheme == "source":
        # Source prefix; a cross-source copy joins its benchmark's family
        fam_of = [MERGED_SHARED.get(n, n.split("/", 1)[0]) for n in names]
        order = list(OrderedDict.fromkeys(fam_of))
    else:
        fam_of = [n.split("/", 1)[0] for n in names]
        order = list(OrderedDict.fromkeys(fam_of))
        if ds_name == "olv2sub":
            _check_groups(ds_name, names, fam_of)
    if ALL in order:
        raise ValueError(f"family id {ALL!r} is reserved")
    camels = [camel(f) for f in order]
    if len(set(camels)) != len(camels) or not all(camels):
        raise ValueError(f"family ids {order} do not give distinct macro "
                         f"names {camels}")
    fam_of = np.array(fam_of, dtype=object)
    return scheme, [(f, np.flatnonzero(fam_of == f)) for f in order]


# ── Selection ────────────────────────────────────────────────────

def _map_result(result, keep):
    """A selector result on the submatrix R[keep, keep] with its picks
    mapped back to full-matrix indices (selected, selected_per_k)."""
    out = dict(result)
    out["selected"] = [int(keep[j]) for j in result["selected"]]
    if result.get("selected_per_k") is not None:
        out["selected_per_k"] = [[int(keep[j]) for j in sel]
                                 for sel in result["selected_per_k"]]
    return out


def select_arm(arm, R, eligible, in_family, k_max, names, split,
               R_blind=None):
    """Selector result of one arm on one split and held-out family (picks
    as full-matrix indices, never in the family).

    blind: run_selection on R_blind (run_dataset passes the refit of the
    non-family columns under --blind-cov refit; None: R[keep, keep], keep
    = the non-family benchmarks, --blind-cov sub) with eligible[keep];
    aware: on R with
    eligible & ~in_family; random: random_select on R with eligible &
    ~in_family and the split's random-baseline seed (SEED + 7)."""
    names = np.asarray(names)
    cand = eligible & ~in_family
    if arm == "random":
        return run_selection("random", R, k_max, names, split, cand)
    method, mode = arm.rsplit("_", 1)
    if mode == "aware":
        return run_selection(method, R, k_max, names, split, cand)
    if mode != "blind":
        raise ValueError(arm)
    keep = np.flatnonzero(~in_family)
    R_sub = R[np.ix_(keep, keep)] if R_blind is None else R_blind
    if R_sub.shape != (len(keep), len(keep)):
        raise ValueError(f"R_blind has shape {R_sub.shape}, expected "
                         f"({len(keep)}, {len(keep)})")
    res = run_selection(method, R_sub, k_max, names[keep], split,
                        eligible[keep])
    return _map_result(res, keep)


# ── Metric ───────────────────────────────────────────────────────

def _zscores(B, col_mean, col_std):
    """Train-standardized, clipped values of observed cells, as in
    impute_and_evaluate."""
    return np.clip((B - col_mean[None, :]) / col_std[None, :],
                   -Z_CLIP, Z_CLIP)


class FamilyScore(NamedTuple):
    """family_r2's result: pooled R^2, scored cells and models, and the
    sums of squares (NaN when nothing is scored)."""
    r2: float
    n_cells: int
    n_models: int
    ss_res: float
    ss_tot: float


def family_r2(B_test, O_test, selected, R, col_mean, col_std, fam_idx,
              target=None, bounds=None):
    """Pooled R^2 of impute_and_evaluate on the family's target cells.

    Protocol rules: only the family's targets count (target: the split's
    cv_split.target_mask, rule D1; None: every column), and the
    predictions are clipped to the valid score range (bounds:
    score_bounds.dataset_bounds, rule D2; None: no clip).

    The observation mask handed to impute_and_evaluate keeps only the
    selected and the family columns, so its targets are the observed
    family cells of the validation models that observe at least one
    selected benchmark (its cell rules).  Validation models are grouped by
    their observation pattern on those columns and each group is imputed
    by one impute_and_evaluate call; within a group every row has the same
    pattern, so the vectorized fast path scores every cell
    (protocol_changes H1).  SS_res of a group is rmse^2 * n_cells, SS_tot
    the sum of the squared z-scores of its target cells, and R^2 = 1 -
    sum SS_res / max(sum SS_tot, 1e-12).

    Returns a FamilyScore (r2, n_cells, n_models, ss_res, ss_tot);
    (nan, 0, 0, nan, nan) if nothing is scored.
    """
    selected = [int(j) for j in selected]
    fam_idx = np.asarray(fam_idx, dtype=int)
    if set(selected) & set(fam_idx.tolist()):
        raise ValueError("a selected benchmark is in the held-out family")
    nothing = FamilyScore(math.nan, 0, 0, math.nan, math.nan)
    if target is not None:
        fam_idx = fam_idx[np.asarray(target, dtype=bool)[fam_idx]]
    if not selected or fam_idx.size == 0:
        return nothing
    N = R.shape[0]
    n_sel = len(selected)
    cols = np.array(selected + fam_idx.tolist(), dtype=int)
    obs = np.asarray(O_test)[:, cols] != 0
    scored = obs[:, :n_sel].any(axis=1) & obs[:, n_sel:].any(axis=1)
    rows_scored = np.flatnonzero(scored)
    if len(rows_scored) == 0:
        return nothing
    O_eval = np.zeros(np.shape(O_test), dtype=np.float64)
    O_eval[:, cols] = obs
    patterns, inverse = np.unique(obs[rows_scored], axis=0,
                                  return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    ss_res = ss_tot = 0.0
    n_cells = 0
    for g, pat in enumerate(patterns):
        rows = rows_scored[inverse == g]
        targets = fam_idx[pat[n_sel:]]
        imp = impute_and_evaluate(B_test[rows], O_eval[rows], selected, R,
                                  col_mean, col_std, N, target=target,
                                  bounds=bounds)
        n_g = len(rows) * len(targets)
        ss_res += float(imp["rmse"]) ** 2 * n_g
        z = _zscores(B_test[np.ix_(rows, targets)], col_mean[targets],
                     col_std[targets])
        ss_tot += float(np.sum(z ** 2))
        n_cells += n_g
    return FamilyScore(1.0 - ss_res / max(ss_tot, SS_TOT_FLOOR), n_cells,
                       len(rows_scored), ss_res, ss_tot)


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else math.nan


# ── Per-dataset run ──────────────────────────────────────────────

def _empty(S, F, K):
    out = {}
    for arm in ARMS:
        for key in ("r2", "ss_res", "ss_tot"):
            out[f"{arm}_{key}"] = np.full((S, F, K), np.nan)
        for key in ("n_cells", "n_models", "index_ties", "tie_breaks"):
            out[f"{arm}_{key}"] = np.zeros((S, F, K), dtype=np.int64)
        out[f"{arm}_selected"] = np.full((S, F, K), -1, dtype=np.int64)
    for mode in MODES:
        out[f"pca_{mode}_selected_per_k"] = np.full((S, F, K, K), -1,
                                                    dtype=np.int64)
    for m in METHODS:
        out[f"{m}_jacc_blind_aware"] = np.full((S, F, K), np.nan)
    for key in ("n_candidates", "n_target_cells", "n_target_models"):
        out[key] = np.zeros((S, F), dtype=np.int64)
    out["blind_refit_failed"] = np.zeros((S, F), dtype=bool)
    return out


def evaluate_family(out, s_row, f, results, k_max, B_test, O_test, R,
                    col_mean, col_std, fam_idx, target=None, bounds=None):
    """Per-k R^2, coverage, picks and ties of every arm's result on one
    (split, family) cell of `out`; results maps arm -> selector result
    (None: arm not run, left NaN)."""
    for arm, res in results.items():
        if res is None:
            continue
        sel = list(res["selected"])[:k_max]
        out[f"{arm}_selected"][s_row, f, :len(sel)] = sel
        if arm.startswith("pca_"):
            for kk, sk in enumerate(res["selected_per_k"][:k_max]):
                out[f"{arm}_selected_per_k"][s_row, f, kk, :len(sk)] = sk
        for k in range(1, k_max + 1):
            out[f"{arm}_index_ties"][s_row, f, k - 1] = index_tie_count(res,
                                                                        k)
            out[f"{arm}_tie_breaks"][s_row, f, k - 1] = tie_break_count(res,
                                                                        k)
            sel_k = selection_at_k(res, k)
            if sel_k is None:
                continue
            score = family_r2(B_test, O_test, sel_k, R, col_mean, col_std,
                              fam_idx, target, bounds)
            for key, val in score._asdict().items():
                out[f"{arm}_{key}"][s_row, f, k - 1] = val
    for m in METHODS:
        rb, ra = results.get(f"{m}_blind"), results.get(f"{m}_aware")
        if rb is None or ra is None:
            continue
        for k in range(1, k_max + 1):
            sb, sa = selection_at_k(rb, k), selection_at_k(ra, k)
            if sb is not None and sa is not None:
                out[f"{m}_jacc_blind_aware"][s_row, f, k - 1] = _jaccard(sb,
                                                                         sa)


def benchfam_split(ctx, s):
    """Every family and arm on one matched split s: {"meta": the split's
    metadata, "out": its row of every _empty array (shape without the
    split axis), "lines": the progress lines}.  ctx holds the dataset
    (B, O, names, families, fam_masks, k_max, blind_cov, bounds, in_mean,
    verbose).  A pure function of ctx and s, so run_dataset's result does
    not depend on the number of worker processes."""
    t0 = time.time()
    B, O, names = ctx["B"], ctx["O"], ctx["names"]
    families, fam_masks = ctx["families"], ctx["fam_masks"]
    k_max, blind_cov, verbose = ctx["k_max"], ctx["blind_cov"], ctx["verbose"]
    bounds, in_mean = ctx["bounds"], ctx["in_mean"]
    N = B.shape[1]
    F = len(families)
    out = _empty(1, F, k_max)
    s_row = 0
    lines = []
    B_train, O_train = B[s.train_idx], O[s.train_idx]
    B_test, O_test = B[s.val_idx], O[s.val_idx]
    # One eligibility mask and one target mask (rule D1) per split, shared
    # by every family and arm
    eligible = eligible_mask(O_train)
    target = target_mask(O_train)
    meta = {"rep": s.rep, "fold": s.fold, "split_id": s.split_id,
            "n_train": len(s.train_idx), "n_val": len(s.val_idx),
            "n_eligible": int(eligible.sum()), "eligible": eligible}
    for f, (_, idx) in enumerate(families):
        obs_f = O_test[:, np.asarray(idx)[target[idx]]] != 0
        out["n_candidates"][s_row, f] = int(np.sum(eligible
                                                   & ~fam_masks[f]))
        out["n_target_cells"][s_row, f] = int(obs_f.sum())
        out["n_target_models"][s_row, f] = int(np.sum(obs_f.any(axis=1)))
    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:  # noqa: BLE001 (logged as a failed split)
        meta["estimation_failed"] = True
        if verbose:
            lines.append(f"  Rep {s.rep} fold {s.fold}: estimation failed: "
                         f"{e}")
        return {"meta": meta, "out": {k: v[0] for k, v in out.items()},
                "lines": lines}
    meta["estimation_failed"] = False

    for f, (fam, idx) in enumerate(families):
        in_family = fam_masks[f]
        R_blind = None
        blind_ok = True
        if blind_cov == "refit":
            keep = np.flatnonzero(~in_family)
            try:
                R_blind = estimate_corr(B_train[:, keep],
                                        O_train[:, keep])[0]
            except Exception as e:  # noqa: BLE001
                blind_ok = False
                out["blind_refit_failed"][s_row, f] = True
                if verbose:
                    lines.append(f"  family {fam}: refit failed: {e}")
        results = {}
        for arm in ARMS:
            if arm.endswith("_blind") and not blind_ok:
                results[arm] = None
                continue
            results[arm] = select_arm(arm, R, eligible, in_family, k_max,
                                      names, s, R_blind)
        evaluate_family(out, s_row, f, results, k_max, B_test, O_test, R,
                        col_mean, col_std, idx, target, bounds)

    if verbose:
        k5 = min(5, k_max) - 1
        fm = {}
        for arm in ("mi_blind", "mi_aware", "trace_blind", "trace_aware",
                    "facloc_blind", "facloc_aware", "random"):
            v = out[f"{arm}_r2"][s_row, in_mean, k5]
            ok = np.isfinite(v)
            fm[arm] = (1.0 - np.sum(out[f"{arm}_ss_res"][s_row, in_mean,
                                                         k5][ok])
                       / max(np.sum(out[f"{arm}_ss_tot"][s_row, in_mean,
                                                         k5][ok]),
                             SS_TOT_FLOOR)) if ok.any() else math.nan
        lines.append(f"  Rep {s.rep} fold {s.fold}: {len(s.train_idx)} "
                     f"train / {len(s.val_idx)} val, {int(eligible.sum())}/"
                     f"{N} eligible; all-family R²@{k5 + 1}: " + "  ".join(
                         f"{a} {v:.3f}" for a, v in fm.items())
                     + f"  ({time.time() - t0:.1f}s)")
    return {"meta": meta, "out": {k: v[0] for k, v in out.items()},
            "lines": lines}


# Worker state of the split processes (--workers; spawn)
_CTX = {}


def _init_worker(ctx):
    _CTX.clear()
    _CTX.update(ctx)


def _run_split(s):
    return benchfam_split(_CTX, s)


def map_splits(ctx, splits, workers=1):
    """benchfam_split of every split, in order: in this process (workers
    <= 1) or in `workers` spawned single-threaded processes
    (ProcessPoolExecutor.map keeps the order; every split is a pure
    function of ctx and the split, so the arrays do not depend on
    workers)."""
    if workers <= 1 or len(splits) <= 1:
        for s in splits:
            yield benchfam_split(ctx, s)
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    mp = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(splits)),
                             mp_context=mp, initializer=_init_worker,
                             initargs=(ctx,)) as pool:
        yield from pool.map(_run_split, splits)


def run_dataset(ds_name, n_repeats=N_REPEATS, n_folds=N_FOLDS, k_max=K_MAX,
                blind_cov=BLIND_COV[0], min_family_models=MIN_FAMILY_MODELS,
                verbose=True, merged_families="source", workers=1):
    """Every arm on every matched split and held-out family of one
    dataset; returns the dict of arrays saved to the npz (module
    docstring).  blind_cov: "refit" (default) or "sub"; merged_families:
    "source" (default) or "prefix" (dataset_scheme); workers > 1
    evaluates the splits in that many processes (map_splits; the arrays
    are those of the serial run)."""
    if blind_cov not in BLIND_COV:
        raise ValueError(f"blind_cov {blind_cov!r} not in {BLIND_COV}")
    d = np.load(matrix_path(ds_name), allow_pickle=False)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    names = np.asarray([str(n) for n in d["benchmark_names"]])
    M, N = B.shape
    scheme, families = benchmark_families(
        ds_name, names, scheme=dataset_scheme(ds_name, merged_families))
    F = len(families)
    k_max = min(k_max, N - 1)
    S = n_repeats * n_folds
    fam_masks = np.zeros((F, N), dtype=bool)
    for f, (_, idx) in enumerate(families):
        fam_masks[f, idx] = True
    family_of = np.argmax(fam_masks, axis=0)
    family_models = np.array([int(np.sum(np.any(O[:, idx] != 0, axis=1)))
                              for _, idx in families])
    in_mean = family_models >= min_family_models

    if verbose:
        print(f"\n{'#' * 70}\n# Dataset: {ds_name.upper()}\n{'#' * 70}")
        print(f"{M} models × {N} benchmarks, {O.mean() * 100:.1f}% observed; "
              f"{n_repeats} × {n_folds}-fold CV, {HOLDOUT_FRAC:.0%} holdout, "
              f"k ≤ {k_max}, blind covariance: {blind_cov}")
        print(f"families ({SCHEME_LABEL[scheme]}): " + ", ".join(
            f"{fam} {len(idx)}" + ("" if in_mean[f] else
                                   f" [{family_models[f]} models: not in "
                                   f"'{ALL}']")
            for f, (fam, idx) in enumerate(families)), flush=True)

    out = _empty(S, F, k_max)
    meta = {key: [] for key in ("rep", "fold", "split_id", "n_train",
                                "n_val", "n_eligible", "eligible",
                                "estimation_failed")}
    ctx = dict(B=B, O=O, names=names, families=families,
               fam_masks=fam_masks, k_max=k_max, blind_cov=blind_cov,
               bounds=dataset_bounds(ds_name, names, DATA), in_mean=in_mean,
               verbose=verbose)
    splits = list(cv_splits(M, n_folds, n_repeats, HOLDOUT_FRAC, SEED))
    for s_row, part in enumerate(map_splits(ctx, splits, workers)):
        for key in meta:
            meta[key].append(part["meta"][key])
        for key, val in part["out"].items():
            out[key][s_row] = val
        for line in part["lines"]:
            print(line, flush=True)

    for key, vals in meta.items():
        out[key] = np.array(vals)
    out.update(
        benchmark_names=names,
        family_names=np.array([fam for fam, _ in families]),
        family_size=np.array([len(idx) for _, idx in families]),
        family_models=family_models, family_in_mean=in_mean,
        family_of=family_of.astype(np.int64),
        arms=np.array(ARMS), methods=np.array(METHODS),
        scheme=np.array(scheme), blind_cov=np.array(blind_cov),
        k_max=np.int64(k_max), holdout_frac=np.float64(HOLDOUT_FRAC),
        n_folds=np.int64(n_folds), n_repeats=np.int64(n_repeats),
        seed=np.int64(SEED),
        **{k: (np.bool_(v) if isinstance(v, bool) else np.int64(v))
           for k, v in protocol_config().items()},
        random_seed_offset=np.int64(RANDOM_SEED_OFFSET),
        min_family_models=np.int64(min_family_models))
    return out


# ── Paired statistics and summaries ──────────────────────────────

PAIRED_FIELDS = (["dataset", "holdout_frac", "blind_cov", "family", "k",
                  "comparison", "method_a", "method_b"] + STAT_FIELDS)


def log_arms(out):
    """The arms of a log (its "arms" array)."""
    return [str(a) for a in np.atleast_1d(out["arms"])]


def is_apart(arm):
    """True for an arm of a method of APART_METHODS ({m}_blind,
    {m}_aware)."""
    return any(str(arm) in (f"{m}_{mode}" for mode in MODES)
               for m in APART_METHODS)


def core_arms(out):
    """The arms of a log other than those of APART_METHODS."""
    return [a for a in log_arms(out) if not is_apart(a)]


def common_support(out, k, arms=None):
    """(S, F) bool: the (split, family) cells where every arm of `arms`
    (default: every arm of the log except those of APART_METHODS,
    core_arms) has a finite R^2 at k (module docstring, Common support)."""
    arms = core_arms(out) if arms is None else list(arms)
    ok = np.ones(np.shape(out[f"{arms[0]}_r2"])[:2], dtype=bool)
    for a in arms:
        ok &= np.isfinite(out[f"{a}_r2"][:, :, k - 1])
    return ok


def apart_support(out, k, methods):
    """(S, F) bool: the cells where every core arm and both arms of each
    APART_METHODS method of `methods` are finite at k."""
    return common_support(out, k, core_arms(out) + [
        f"{m}_{mode}" for m in methods for mode in MODES])


def support_for(out, k, arms, core=None, cache=None):
    """Common support of an entry with the arms `arms`: that of the core
    arms (common_support; `core` if given), or, if `arms` include arms of
    APART_METHODS methods, that of the core arms and both arms of those
    methods (apart_support, kept in the dict `cache` if given)."""
    apart = tuple(m for m in APART_METHODS
                  if any(a in (f"{m}_{mode}" for mode in MODES)
                         for a in arms))
    if not apart:
        return common_support(out, k) if core is None else core
    if cache is None:
        return apart_support(out, k, apart)
    if (k, apart) not in cache:
        cache[(k, apart)] = apart_support(out, k, apart)
    return cache[(k, apart)]


def supported_r2(out, arm, k, support=None):
    """(S, F) R^2 of `arm` at k, NaN outside `support` (default:
    support_for(out, k, [arm]), i.e. common_support(out, k) for the arms
    other than those of APART_METHODS)."""
    support = support_for(out, k, [arm]) if support is None else support
    return np.where(support, out[f"{arm}_r2"][:, :, k - 1], np.nan)


def pooled_r2(out, arms, k, support=None):
    """[(S,) per arm]: R^2 pooled over the held-out cells of all families
    (ALL), 1 - sum_f SS_res / max(sum_f SS_tot, 1e-12) over the families
    with family_in_mean in `support` (default: support_for(out, k, arms),
    the same families for every arm); NaN where there is none."""
    mask = np.asarray(out["family_in_mean"], dtype=bool)
    support = support_for(out, k, arms) if support is None else support
    ok = np.asarray(support, dtype=bool)[:, mask]
    res = []
    for a in arms:
        if not np.all(np.isfinite(out[f"{a}_r2"][:, mask, k - 1])[ok]):
            raise ValueError(f"arm {a} is NaN inside the support at k={k}")
        sr = np.where(ok, out[f"{a}_ss_res"][:, mask, k - 1], 0.0).sum(1)
        st = np.where(ok, out[f"{a}_ss_tot"][:, mask, k - 1], 0.0).sum(1)
        r2 = 1.0 - sr / np.maximum(st, SS_TOT_FLOOR)
        res.append(np.where(ok.any(axis=1), r2, np.nan))
    return res


def family_mean(out, arm, k, support=None):
    """(S,) per-split ALL value of one arm (pooled_r2, common support)."""
    return pooled_r2(out, [arm], k, support)[0]


def family_mean_pair(out, a, b, k, support=None):
    """(S,), (S,) per-split ALL values of arms a and b on the same
    families (pooled_r2, common support)."""
    return tuple(pooled_r2(out, [a, b], k, support))


def support_diagnostics(out, ks=KS_REPORT):
    """{k: cells_any, cells_common, splits_all, nan_cells}: (split,
    family_in_mean) cells where some core arm (core_arms) / every core arm
    is finite, splits with a finite "all", and per arm the cells of
    cells_any where it is NaN (for a core arm: dropped from every arm by
    the common support); for each APART_METHODS method m with arms in the
    log also cells_common_{m}, the cells where every core arm and both of
    m's arms are finite (the support of the entries with m's arms)."""
    mask = np.asarray(out["family_in_mean"], dtype=bool)
    K = int(out["k_max"])
    core = core_arms(out)
    diag = OrderedDict()
    for k in [k for k in ks if k <= K]:
        fin = {a: np.isfinite(out[f"{a}_r2"][:, mask, k - 1])
               for a in log_arms(out)}
        anyf = np.any([fin[a] for a in core], axis=0)
        common = common_support(out, k)[:, mask]
        diag[str(k)] = {
            "cells_any": int(anyf.sum()), "cells_common": int(common.sum()),
            "splits_all": int(np.sum(common.any(axis=1))),
            "nan_cells": {a: int(np.sum(anyf & ~f)) for a, f in fin.items()
                          if np.any(anyf & ~f)}}
        for m in APART_METHODS:
            if f"{m}_blind" in fin and f"{m}_aware" in fin:
                diag[str(k)][f"cells_common_{m}"] = int(np.sum(
                    apart_support(out, k, [m])[:, mask]))
    return diag


def _pairs():
    """(comparison, arm a, arm b) of every paired row of one family / k."""
    pairs = [(c, *fn(m)) for c, fn in COMPARISONS.items() for m in METHODS]
    for mode in MODES:
        pairs += [(mode, f"{a}_{mode}", f"{b}_{mode}")
                  for a, b in orient(METHODS)]
    return pairs


def paired_rows(ds_name, out):
    """Paired R^2 differences over the matched splits for every family
    (and "all", pooled_r2), k = 1..K and pair of _pairs(), on the common
    support of all arms (support_for: of the core arms, and of those and
    both arms of an APART_METHODS method for a pair with one of them)."""
    rows = []
    meta = (out["n_train"], out["n_val"], out["rep"])
    fams = [str(x) for x in out["family_names"]]
    K = int(out["k_max"])
    supports = {k: common_support(out, k) for k in range(1, K + 1)}
    cache = {}
    for f, fam in enumerate(fams + [ALL]):
        for k in range(1, K + 1):
            for comparison, a, b in _pairs():
                sup = support_for(out, k, (a, b), supports[k], cache)
                if fam == ALL:
                    va, vb = family_mean_pair(out, a, b, k, sup)
                else:
                    va = supported_r2(out, a, k, sup)[:, f]
                    vb = supported_r2(out, b, k, sup)[:, f]
                rows.append({"dataset": ds_name,
                             "holdout_frac": HOLDOUT_FRAC,
                             "blind_cov": str(out["blind_cov"]),
                             "family": fam, "k": k,
                             "comparison": comparison, "method_a": a,
                             "method_b": b,
                             **paired_stats(va, vb, *meta)})
    return rows


def _num(x):
    """JSON-safe float (None for NaN / inf)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _mean_sd_n(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None, None, 0
    return float(np.mean(v)), float(np.std(v)), int(len(v))


def summary_tables(out, ks=KS_REPORT):
    """{family: {k: {arm: {mean, sd, n, cells}}}} on the common support
    (ALL: the per-split pooled R^2 of all families, pooled_r2), k in ks <=
    K; every core arm of a (family, k) has the same n (the APART_METHODS
    arms: the support of the core arms and theirs, support_for)."""
    K = int(out["k_max"])
    fams = [str(x) for x in out["family_names"]]
    tables = OrderedDict()
    supports = {k: common_support(out, k) for k in ks if k <= K}
    cache = {}
    for f, fam in enumerate(fams + [ALL]):
        tf = OrderedDict()
        for k in [k for k in ks if k <= K]:
            tk = OrderedDict()
            for arm in ARMS:
                sup = support_for(out, k, [arm], supports[k], cache)
                if fam == ALL:
                    v = family_mean(out, arm, k, sup)
                    cells = None
                else:
                    v = supported_r2(out, arm, k, sup)[:, f]
                    c = out[f"{arm}_n_cells"][:, f, k - 1]
                    cells = _num(np.mean(c[np.isfinite(v)])) \
                        if np.isfinite(v).any() else None
                mu, sd, n = _mean_sd_n(v)
                tk[arm] = {"mean": mu, "sd": sd, "n": n, "cells": cells}
            tf[str(k)] = tk
        tables[fam] = tf
    return tables


def is_prefix_override(ds_name, scheme):
    """True if `scheme` is --merged-families prefix on a "source" dataset."""
    return FAMILY_SCHEME.get(ds_name) == "source" and str(scheme) == "prefix"


def macro_prefix(ds_name, out):
    """BenchFam, + Prefix (--merged-families prefix on Merged), + Sub
    (--blind-cov sub)."""
    return ("BenchFam"
            + ("Prefix" if is_prefix_override(ds_name, out["scheme"]) else "")
            + ("Sub" if str(out["blind_cov"]) == "sub" else ""))


def macro_dict(ds_name, out, paired, tables, ks=KS_REPORT):
    """Flat {MacroName: number} (module docstring)."""
    Ds = camel(ds_name)
    pre = macro_prefix(ds_name, out)
    K = int(out["k_max"])
    ks = [k for k in ks if k <= K]
    macros = OrderedDict()

    def mac(name, value):
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} is not letters-only")
        if name in macros:
            raise ValueError(f"duplicate macro {name!r}")
        if value is not None and math.isfinite(float(value)):
            macros[name] = float(value) if not isinstance(value, int) \
                else int(value)

    fams = [str(x) for x in out["family_names"]]
    fam_macro = {fam: camel(fam) for fam in fams}
    fam_macro[ALL] = "All"
    for fam in fams + [ALL]:
        Fm = fam_macro[fam]
        for k in ks:
            kw = "K" + int_words(k)
            for arm in ARMS:
                t = tables[fam][str(k)][arm]
                mac(f"{pre}Rsq{Ds}{Fm}{camel(arm)}{kw}", t["mean"])
    lookup = {(r["family"], r["k"], r["comparison"], r["method_a"],
               r["method_b"]): r for r in paired}
    for fam in fams + [ALL]:
        Fm = fam_macro[fam]
        stats = (("Diff", "mean_diff"), ("Lo", "ci_low"), ("Hi", "ci_high"))
        if fam == ALL:
            stats += (("Se", "se_nb"), ("Win", "win_rate"),
                      ("Wins", "n_wins"), ("N", "n_splits"))
        for k in ks:
            kw = "K" + int_words(k)
            for comparison, fn in COMPARISONS.items():
                for m in METHODS:
                    a, b = fn(m)
                    r = lookup[(fam, k, comparison, a, b)]
                    for sname, field in stats:
                        mac(f"{pre}Pair{sname}{Ds}{Fm}{camel(m)}"
                            f"{COMPARISON_MACRO[comparison]}{kw}", r[field])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in ks:
            for m in METHODS:
                j = out[f"{m}_jacc_blind_aware"][:, out["family_in_mean"],
                                                 k - 1]
                mac(f"{pre}Jacc{Ds}All{camel(m)}K{int_words(k)}",
                    np.nanmean(j) if np.isfinite(j).any() else None)
    mac(f"{pre}NFam{Ds}", len(fams))
    mac(f"{pre}NFamMean{Ds}", int(np.sum(out["family_in_mean"])))
    mac(f"{pre}NSplits{Ds}", len(out["split_id"]))
    mac(f"{pre}NFailed{Ds}", int(np.sum(out["estimation_failed"])))
    mac(f"{pre}NRefitFailed{Ds}", int(np.sum(out["blind_refit_failed"])))
    ok = ~out["estimation_failed"]
    for f, fam in enumerate(fams):
        Fm = fam_macro[fam]
        mac(f"{pre}Size{Ds}{Fm}", int(out["family_size"][f]))
        mac(f"{pre}Models{Ds}{Fm}", int(out["family_models"][f]))
        mac(f"{pre}NCand{Ds}{Fm}",
            float(np.mean(out["n_candidates"][ok, f])) if ok.any() else None)
    return dict(macros)


def _json_row(r):
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in r.items()}


def write_json(path, ds_name, out, paired, tables, runtime_sec, argv):
    """Config, summary tables, key paired rows, diagnostics, macros."""
    fams = [str(x) for x in out["family_names"]]
    names = [str(x) for x in out["benchmark_names"]]
    K = int(out["k_max"])
    key_rows = [_json_row(r) for r in paired
                if r["comparison"] in COMPARISONS and r["k"] in KS_REPORT]
    ok = ~out["estimation_failed"]

    def same_frac(m, k):
        """Share of (split, family) cells whose blind and aware size-k
        sets coincide."""
        j = out[f"{m}_jacc_blind_aware"][:, :, k - 1]
        j = j[np.isfinite(j)]
        return _num(np.mean(j == 1.0)) if len(j) else None
    same = {m: {str(k): same_frac(m, k) for k in KS_REPORT if k <= K}
            for m in METHODS}
    doc = {
        "script": "eval_benchfamily_holdout.py",
        "dataset": ds_name,
        "argv": list(argv),
        "config": {
            "scheme": str(out["scheme"]),
            "blind_cov": str(out["blind_cov"]),
            "families": OrderedDict(
                (fam, {"size": int(out["family_size"][f]),
                       "models": int(out["family_models"][f]),
                       "in_family_mean": bool(out["family_in_mean"][f]),
                       "members": [names[j] for j in
                                   np.flatnonzero(out["family_of"] == f)]})
                for f, fam in enumerate(fams)),
            "methods": METHODS, "arms": ARMS, "modes": MODES,
            "k_max": K, "ks_report": KS_REPORT,
            "n_repeats": int(out["n_repeats"]), "n_folds": int(out["n_folds"]),
            "holdout_frac": HOLDOUT_FRAC, "seed": SEED,
            **protocol_config(),
            "min_family_models": int(out["min_family_models"]),
            "random_seed_tag": SEED + RANDOM_SEED_OFFSET,
            "new_seed_tags": [],
            "impute_ridge": RIDGE, "z_clip": Z_CLIP, "tie_rtol": TIE_RTOL,
            "merged_shared": (dict(MERGED_SHARED) if str(out["scheme"])
                              == "source" else None),
            "blind": ("selector on estimate_corr of the training models' "
                      "non-family columns" if str(out["blind_cov"])
                      == "refit" else "selector on the non-family block "
                      "R[keep, keep] of the split's full estimate"),
            "metric": "pooled R^2 of impute_and_evaluate on the held-out "
                      "family's observed cells of validation models that "
                      "observe >= 1 selected benchmark (train "
                      "standardization, full training R)",
            "support": "common support: a (split, family) cell at k "
                       "enters tables, paired rows, 'all' and macros only "
                       "if every arm is finite there",
            "all": "per split: R^2 pooled over the held-out cells of the "
                   "families with in_family_mean in the common support, "
                   "1 - sum SS_res / sum SS_tot; then mean / paired over "
                   "splits",
        },
        "diagnostics": {
            "estimation_failed": int(np.sum(out["estimation_failed"])),
            "blind_refit_failed": int(np.sum(out["blind_refit_failed"])),
            "common_support": support_diagnostics(out),
            "n_candidates_mean": {fam: _num(np.mean(out["n_candidates"][ok, f]))
                                  if ok.any() else None
                                  for f, fam in enumerate(fams)},
            "splits_with_targets": {fam: int(np.sum(
                out["n_target_cells"][:, f] > 0)) for f, fam in
                enumerate(fams)},
            "target_cells_mean": {fam: _num(np.mean(
                out["n_target_cells"][:, f])) for f, fam in enumerate(fams)},
            "index_ties_at_kmax": {arm: int(np.sum(
                out[f"{arm}_index_ties"][:, :, K - 1])) for arm in ARMS},
            "blind_equals_aware_frac": same,
        },
        "summary": tables,
        "paired_key_rows": key_rows,
        "runtime_sec": _num(round(float(runtime_sec), 1)),
        "macros": macro_dict(ds_name, out, paired, tables),
    }
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)
    print(f"Summary saved to {path} ({len(doc['macros'])} macros)",
          flush=True)


def _significant(r):
    return (r["n_splits"] > 1 and np.isfinite(r["ci_low"])
            and (r["ci_low"] > 0 or r["ci_high"] < 0))


def print_summary(ds_name, out, paired, tables):
    K = int(out["k_max"])
    ks = [k for k in KS_REPORT if k <= K]
    fams = [str(x) for x in out["family_names"]]
    lookup = {(r["family"], r["k"], r["comparison"], r["method_a"],
               r["method_b"]): r for r in paired}
    n_splits = len(out["split_id"])
    print(f"\n== {display_name(ds_name)}: R² on the held-out family's cells, "
          f"blind / aware (mean over the {n_splits} splits' common support; "
          f"blind covariance {out['blind_cov']}, families "
          f"{out['scheme']}) ==")
    for f, fam in enumerate(fams + [ALL]):
        n_k = [tables[fam][str(k)][ARMS[0]]["n"] for k in ks]
        if fam == ALL:
            head = (f"all: R² pooled over the cells of "
                    f"{int(out['family_in_mean'].sum())} families")
        else:
            head = (f"{fam} ({int(out['family_size'][f])} benchmarks, "
                    f"{int(out['family_models'][f])} models"
                    + ("" if out["family_in_mean"][f] else
                       f", not in '{ALL}'") + ")")
        if any(n != n_splits for n in n_k):
            head += "; splits in the support: " + ", ".join(
                f"k={k} {n}" for k, n in zip(ks, n_k))
        print(f"\n-- {head}")
        print(f"{'method':<22}" + "".join(f"{'k=' + str(k):>16}" for k in ks))

        def cell(arm, k):
            v = tables[fam][str(k)][arm]["mean"]
            return "n/a" if v is None else f"{v:.3f}"
        for m in METHODS:
            print(f"{METHOD_LABEL[m]:<22}" + "".join(
                f"{cell(m + '_blind', k) + '/' + cell(m + '_aware', k):>16}"
                for k in ks))
        print(f"{METHOD_LABEL['random']:<22}" + "".join(
            f"{cell('random', k):>16}" for k in ks))
        for comparison, fn in COMPARISONS.items():
            if comparison == "aware - random":
                continue
            cells = []
            for k in ks:
                parts = []
                for m in ("mi", "trace", "entropy", "facloc"):
                    a, b = fn(m)
                    r = lookup[(fam, k, comparison, a, b)]
                    parts.append("n/a" if r["n_splits"] == 0 else
                                 f"{r['mean_diff']:+.2f}"
                                 f"{'*' if _significant(r) else ''}")
                cells.append(" ".join(parts))
            print(f"  {comparison + ' (mi trace ent fl)':<37}"
                  + "  ".join(f"k={k}: {c}" for k, c in zip(ks, cells)))
    print("(* = Nadeau–Bengio 95% interval excludes 0)")


# ── Plotting ─────────────────────────────────────────────────────

def plot_dataset(ds_name, out, paired, path, ymin=-1.0):
    """R^2 vs k per family (blind solid, aware dashed, random grey) plus
    two panels of paired differences of "all" with 95% NB bands; means
    below ymin are drawn at the lower edge as open markers."""
    K = int(out["k_max"])
    ks = np.arange(1, K + 1)
    fams = [str(x) for x in out["family_names"]]
    n_panels = len(fams) + 2
    ncols = min(4, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols,
                                                    2.7 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    # Common support of every arm at each k: r2[arm] is (S, F, K)
    r2 = {arm: np.stack([supported_r2(out, arm, int(k)) for k in ks],
                        axis=2) for arm in ARMS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Shared lower limit: 0 if every mean is >= 0, else the next tenth
        # below the lowest mean, but not below ymin
        lowest = np.nanmin([np.nanmin(np.nanmean(r2[arm], axis=0))
                            for arm in ARMS] + [0.0])
        ymin = max(ymin, math.floor(10 * lowest) / 10)
        for f, fam in enumerate(fams):
            ax = flat[f]
            series = [(f"{m}_{mode}", METHOD_COLOR[m],
                       "-" if mode == "blind" else "--",
                       "o" if mode == "blind" else None,
                       1.4 if mode == "blind" else 1.0)
                      for m in METHODS for mode in MODES]
            series.append(("random", METHOD_COLOR["random"], "--", "x", 1.0))
            for arm, color, ls, marker, lw in series:
                mu = np.nanmean(r2[arm][:, f, :], axis=0)
                low = mu < ymin
                ax.plot(ks, np.where(low, ymin, mu), linestyle=ls,
                        marker=marker, markersize=3, color=color,
                        linewidth=lw)
                if low.any():
                    ax.plot(ks[low], np.full(low.sum(), ymin), "v",
                            color=color, markersize=5,
                            markerfacecolor="white")
            note = "" if out["family_in_mean"][f] else f", not in '{ALL}'"
            ax.set_title(f"{fam} ({int(out['family_size'][f])} bench."
                         f"{note})", fontsize=8)
            if not np.isfinite(r2["random"][:, f, :]).any():
                ax.text(0.5, 0.5, "no split where\nevery arm is scored",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color="grey")
            ax.set_ylim(ymin, 1.0)
            ax.set_ylabel("$R^2$ on held-out family", fontsize=7)
        lookup = {(r["family"], r["k"], r["comparison"], r["method_a"],
                   r["method_b"]): r for r in paired}
        for j, comparison in enumerate(("blind - random", "aware - blind")):
            ax = flat[len(fams) + j]
            for m in METHODS:
                a, b = COMPARISONS[comparison](m)
                rr = [lookup[(ALL, int(k), comparison, a, b)] for k in ks]
                mu = np.array([r["mean_diff"] for r in rr], dtype=float)
                lo = np.array([r["ci_low"] for r in rr], dtype=float)
                hi = np.array([r["ci_high"] for r in rr], dtype=float)
                ax.plot(ks, mu, "-o", markersize=3, color=METHOD_COLOR[m],
                        linewidth=1.4)
                ax.fill_between(ks, lo, hi, color=METHOD_COLOR[m],
                                alpha=0.12, linewidth=0)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
            ax.set_title(f"all families pooled: "
                         f"{comparison.replace('-', '−')}", fontsize=8)
            ax.set_ylabel("paired $\\Delta R^2$", fontsize=7)
        for ax in flat[:n_panels]:
            ax.set_xlabel("selected benchmarks $k$", fontsize=7)
            ax.set_xlim(0.5, K + 0.5)
            ax.set_xticks([k for k in KS_REPORT if k <= K])
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
        for ax in flat[n_panels:]:
            ax.set_visible(False)
    handles = [Line2D([], [], color=METHOD_COLOR[m], linewidth=1.4,
                      label=METHOD_LABEL[m]) for m in METHODS]
    handles += [Line2D([], [], color=METHOD_COLOR["random"], linestyle="--",
                       marker="x", markersize=4, label=METHOD_LABEL["random"]),
                Line2D([], [], color="black", linestyle="-", marker="o",
                       markersize=3, label="blind (selection without F)"),
                Line2D([], [], color="black", linestyle="--",
                       label="aware (objective includes F)")]
    fig.suptitle(f"{display_name(ds_name)}: benchmark-family holdout "
                 f"({len(out['split_id'])} splits, blind covariance "
                 f"{out['blind_cov']})", fontsize=9)
    plt.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────

def output_stem(ds_name, blind_cov=BLIND_COV[0], merged_families="source"):
    """benchfamily_holdout_{ds}, + _prefix (--merged-families prefix on a
    "source" dataset), + _sub (--blind-cov sub)."""
    scheme = dataset_scheme(ds_name, merged_families)
    return (f"benchfamily_holdout_{ds_name}"
            + ("_prefix" if is_prefix_override(ds_name, scheme) else "")
            + ("_sub" if blind_cov == "sub" else ""))


def load_log(stem, blind_cov=None, scheme=None):
    """(arrays, runtime_sec) of a saved LOGS/{stem}.npz and its JSON's
    runtime (NaN if the JSON is missing).  Raises ValueError if the log's
    blind_cov / scheme differ from the given ones (e.g. a log written
    before 2026-10-18, when the untagged default was sub and Merged's
    scheme prefix)."""
    path = LOGS / f"{stem}.npz"
    if not path.is_file():
        raise ValueError(f"no log {path}")
    with np.load(path, allow_pickle=False) as d:
        out = {key: d[key] for key in d.files}
    for key, want in (("blind_cov", blind_cov), ("scheme", scheme)):
        if want is not None and str(out[key]) != want:
            raise ValueError(f"{path} has {key} {str(out[key])!r}, the "
                             f"options ask for {want!r}")
    runtime = math.nan
    jpath = LOGS / f"{stem}.json"
    if jpath.is_file():
        runtime = json.loads(jpath.read_text()).get("runtime_sec") or math.nan
    return out, runtime


# ── Blind covariance: refit vs sub ───────────────────────────────

def size_k_sets(out, arm, k):
    """(S, F) object array of the size-k selection of `arm` as a frozenset
    (pca: its size-k set selected_per_k), None where it has none."""
    S, F = np.shape(out[f"{arm}_selected"])[:2]
    res = np.full((S, F), None, dtype=object)
    for s in range(S):
        for f in range(F):
            if arm.startswith("pca_"):
                sel = out[f"{arm}_selected_per_k"][s, f, k - 1, :k]
            else:
                sel = out[f"{arm}_selected"][s, f, :k]
            if len(sel) == k and np.all(sel >= 0):
                res[s, f] = frozenset(int(j) for j in sel)
    return res


def blind_cov_comparison(ds_name, refit, sub, ks=None):
    """Refit vs sub blind arms of one dataset (two logs of the same splits,
    families and options except blind_cov).  Returns {agreement, paired,
    significant, macros}: agreement[m][k] = {set_equal, seq_equal, n}
    shares of the (split, family) cells with a size-k blind selection in
    both runs; paired = paired_stats rows of refit - sub per family and
    "all" for k = 1..K (support: both runs' common supports); significant
    = the "all" rows whose interval excludes 0."""
    for key in ("split_id", "family_names", "benchmark_names", "family_of",
                "family_in_mean"):
        if not np.array_equal(refit[key], sub[key]):
            raise ValueError(f"refit and sub logs differ in {key}")
    if str(refit["blind_cov"]) != "refit" or str(sub["blind_cov"]) != "sub":
        raise ValueError("expected a refit and a sub log")
    if str(refit["scheme"]) != str(sub["scheme"]):
        raise ValueError("refit and sub logs use different family schemes")
    for arm in ARMS:
        if not arm.endswith("_blind") and not np.array_equal(
                refit[f"{arm}_r2"], sub[f"{arm}_r2"], equal_nan=True):
            raise ValueError(f"{arm} differs between the refit and sub "
                             f"logs (it must not depend on --blind-cov)")
    K = int(refit["k_max"])
    ks = list(range(1, K + 1)) if ks is None else ks
    fams = [str(x) for x in refit["family_names"]]
    meta = (refit["n_train"], refit["n_val"], refit["rep"])
    agreement = OrderedDict()
    for m in METHODS:
        arm = f"{m}_blind"
        agreement[m] = OrderedDict()
        for k in ks:
            a, b = size_k_sets(refit, arm, k), size_k_sets(sub, arm, k)
            both = np.array([[x is not None and y is not None
                              for x, y in zip(ra, rb)]
                             for ra, rb in zip(a, b)], dtype=bool)
            n = int(both.sum())
            set_eq = np.mean([a[i] == b[i] for i in zip(*np.nonzero(both))]) \
                if n else math.nan
            seq_eq = math.nan
            if n and m != "pca":
                seq = np.all(refit[f"{arm}_selected"][:, :, :k]
                             == sub[f"{arm}_selected"][:, :, :k], axis=2)
                seq_eq = float(np.mean(seq[both]))
            agreement[m][str(k)] = {"set_equal": _num(set_eq),
                                    "seq_equal": _num(seq_eq), "n": n}
    paired, significant = [], []
    for k in range(1, K + 1):
        support = common_support(refit, k) & common_support(sub, k)
        apart_sup = {m: apart_support(refit, k, [m])
                     & apart_support(sub, k, [m])
                     for m in APART_METHODS if m in METHODS}
        for m in METHODS:
            arm = f"{m}_blind"
            sup = apart_sup.get(m, support)
            for f, fam in enumerate(fams + [ALL]):
                if fam == ALL:
                    va = family_mean(refit, arm, k, sup)
                    vb = family_mean(sub, arm, k, sup)
                else:
                    va = supported_r2(refit, arm, k, sup)[:, f]
                    vb = supported_r2(sub, arm, k, sup)[:, f]
                r = {"dataset": ds_name, "holdout_frac": HOLDOUT_FRAC,
                     "blind_cov": "refit - sub", "family": fam, "k": k,
                     "comparison": "refit - sub", "method_a": f"{arm}_refit",
                     "method_b": f"{arm}_sub",
                     **paired_stats(va, vb, *meta)}
                paired.append(r)
                if fam == ALL and _significant(r):
                    significant.append(_json_row(r))
    Ds = camel(ds_name)
    pre = ("BenchFam" + ("Prefix" if is_prefix_override(ds_name,
                                                        refit["scheme"])
                         else "") + "Cov")
    macros = OrderedDict()
    for m in METHODS:
        for k in [k for k in KS_REPORT if k <= K and k in ks]:
            kw = "K" + int_words(k)
            g = agreement[m][str(k)]
            for name, val in ((f"{pre}SetEq{Ds}{camel(m)}{kw}",
                               g["set_equal"]),
                              (f"{pre}SeqEq{Ds}{camel(m)}{kw}",
                               g["seq_equal"])):
                if val is not None:
                    macros[name] = val
            r = next(r for r in paired if r["family"] == ALL
                     and r["k"] == k and r["method_a"] == f"{m}_blind_refit")
            for sname, field in (("Diff", "mean_diff"), ("Lo", "ci_low"),
                                 ("Hi", "ci_high")):
                if math.isfinite(r[field]):
                    macros[f"{pre}Pair{sname}{Ds}All{camel(m)}{kw}"] = \
                        float(r[field])
    # Counts and maximum over the "all" rows of the methods other than
    # APART_METHODS (the macros without a method piece), then over each
    # APART_METHODS method's rows (suffix camel(m))
    apart = [m for m in APART_METHODS if m in METHODS]
    groups = [("", lambda m: m not in apart)] + [
        (camel(a), lambda m, a=a: m == a) for a in apart]
    for G, in_group in groups:
        rows = [r for r in paired if r["family"] == ALL
                and in_group(r["method_a"][:-len("_blind_refit")])]
        all_rows = [r for r in rows if math.isfinite(r["mean_diff"])]
        if all_rows:
            macros[f"{pre}MaxAbsDiff{Ds}{G}"] = float(max(
                abs(r["mean_diff"]) for r in all_rows))
        macros[f"{pre}NSig{Ds}{G}"] = sum(1 for r in rows
                                          if _significant(r))
        macros[f"{pre}NPairs{Ds}{G}"] = len(all_rows)
    macros[f"{pre}NSplits{Ds}"] = len(refit["split_id"])
    assert all(name.isalpha() for name in macros)
    return {"agreement": agreement, "paired": paired,
            "significant": significant, "macros": dict(macros)}


def compare_blind_cov(ds_name, merged_families="source"):
    """--compare-blind-cov for one dataset: reads the refit and sub logs,
    prints and writes LOGS/benchfamily_holdout_{ds}[_prefix]_blindcov.json
    and its _paired.csv."""
    scheme = dataset_scheme(ds_name, merged_families)
    refit, _ = load_log(output_stem(ds_name, "refit", merged_families),
                        "refit", scheme)
    sub, _ = load_log(output_stem(ds_name, "sub", merged_families), "sub",
                      scheme)
    res = blind_cov_comparison(ds_name, refit, sub)
    K = int(refit["k_max"])
    ks = [k for k in KS_REPORT if k <= K]
    lookup = {(r["family"], r["k"], r["method_a"]): r for r in res["paired"]}
    print(f"\n== {display_name(ds_name)}: blind arms, refit - sub "
          f"({len(refit['split_id'])} splits) ==")
    print(f"{'method':<10}" + "".join(f"{'k=' + str(k):>34}" for k in ks))
    for m in METHODS:
        cells = []
        for k in ks:
            g = res["agreement"][m][str(k)]
            r = lookup[(ALL, k, f"{m}_blind_refit")]
            seq = "  -" if g["seq_equal"] is None else \
                f"{g['seq_equal']:.2f}"
            cells.append(f"set {g['set_equal']:.2f} seq {seq} "
                         f"{r['mean_diff']:+.4f}"
                         f"{'*' if _significant(r) else ' '}")
        print(f"{m:<10}" + "".join(f"{c:>34}" for c in cells))
    diffs = [abs(r["mean_diff"]) for r in res["paired"]
             if r["family"] == ALL and math.isfinite(r["mean_diff"])]
    print(f"set / seq: share of (split, family) cells with identical size-k "
          f"blind sets / pick sequences; then the paired refit - sub R² of "
          f"'all' (* = interval excludes 0: {len(res['significant'])} of "
          f"{len(diffs)} (method, k) rows, k = 1..{K}; max |diff| "
          + (f"{max(diffs):.4f})" if diffs else "n/a)"))
    for r in res["significant"]:
        print(f"  {r['method_a']} k={r['k']}: {r['mean_diff']:+.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
              f"{r['n_splits']} splits")
    stem = (f"benchfamily_holdout_{ds_name}"
            + ("_prefix" if is_prefix_override(ds_name, scheme) else "")
            + "_blindcov")
    write_csv(res["paired"], LOGS / f"{stem}_paired.csv", PAIRED_FIELDS)
    doc = {"script": "eval_benchfamily_holdout.py", "dataset": ds_name,
           "mode": "compare-blind-cov", "scheme": scheme,
           "n_splits": len(refit["split_id"]), "k_max": K,
           "agreement": res["agreement"],
           "paired_all": [_json_row(r) for r in res["paired"]
                          if r["family"] == ALL],
           "significant": res["significant"], "macros": res["macros"]}
    path = LOGS / f"{stem}.json"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)
    print(f"Comparison saved to {path} ({len(res['macros'])} macros)",
          flush=True)
    return res


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark-family holdout on the matched repeated-CV "
                    "splits: selection blind to / aware of a held-out "
                    "benchmark family vs random, R² on the family's cells.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names with a {name}.matrix.npz in the "
                             "data directory and a family scheme "
                             f"({', '.join(sorted(FAMILY_SCHEME))}; default: "
                             f"{' '.join(DATASETS)})")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions; splits = repeats × folds, rep 0 "
                             "= the original seed-42 folds (default "
                             "%(default)s)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--kmax", type=int, default=K_MAX,
                        help="largest k (capped at N - 1; default "
                             "%(default)s)")
    parser.add_argument("--blind-cov", choices=BLIND_COV,
                        default=BLIND_COV[0],
                        help="blind arm's correlation matrix: re-estimated "
                             "from the training models' non-family columns "
                             "(refit, default) or the non-family block of "
                             "the split's full estimate (sub; outputs "
                             "tagged _sub)")
    parser.add_argument("--merged-families", choices=list(MERGED_FAMILIES),
                        default="source",
                        help="Merged's families: source leaderboards with "
                             "the cross-source copies of MMLU and MATH "
                             "merged (source, default) or plain column "
                             "prefixes (prefix; outputs tagged _prefix)")
    parser.add_argument("--min-family-models", type=int,
                        default=MIN_FAMILY_MODELS,
                        help="families observed for fewer models are left "
                             "out of 'all' (default %(default)s)")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes evaluating the splits (spawn, "
                             "single-threaded each; the results do not "
                             "depend on it; default %(default)s)")
    parser.add_argument("--from-logs", action="store_true",
                        help="no recomputation: re-read LOGS/"
                             "benchfamily_holdout_{tag}.npz and rewrite the "
                             "summary, paired CSV, JSON and figure")
    parser.add_argument("--compare-blind-cov", action="store_true",
                        help="no recomputation: compare the refit and sub "
                             "logs of each dataset (both must exist) and "
                             "write benchfamily_holdout_{ds}_blindcov.json")
    args = parser.parse_args(argv)
    if args.from_logs and args.compare_blind_cov:
        parser.error("--from-logs and --compare-blind-cov exclude each other")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.kmax < 1:
        parser.error("--kmax must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    unknown = [ds for ds in args.datasets if ds not in FAMILY_SCHEME]
    if unknown:
        parser.error(f"no benchmark-family scheme for {unknown}; known: "
                     f"{', '.join(sorted(FAMILY_SCHEME))}")
    if args.compare_blind_cov:
        for ds_name in args.datasets:
            try:
                compare_blind_cov(ds_name, args.merged_families)
            except ValueError as e:
                parser.error(f"--compare-blind-cov {ds_name}: {e}")
        return
    missing = [ds for ds in args.datasets if not matrix_path(ds).is_file()]
    if missing and not args.from_logs:
        parser.error("no data matrix for " + ", ".join(
            f"{ds} ({matrix_path(ds)})" for ds in missing))

    total_t0 = time.time()
    for ds_name in args.datasets:
        t0 = time.time()
        stem = output_stem(ds_name, args.blind_cov, args.merged_families)
        if args.from_logs:
            try:
                out, runtime = load_log(
                    stem, args.blind_cov,
                    dataset_scheme(ds_name, args.merged_families))
            except ValueError as e:
                parser.error(f"--from-logs {ds_name}: {e}")
        else:
            out = run_dataset(ds_name, args.repeats, args.folds, args.kmax,
                              args.blind_cov, args.min_family_models,
                              merged_families=args.merged_families,
                              workers=args.workers)
            runtime = time.time() - t0
            # saved first, so a failure below loses no computation
            # (--from-logs redoes everything after this point)
            np.savez(LOGS / f"{stem}.npz", **out)
            print(f"Results saved to {LOGS / f'{stem}.npz'}", flush=True)
        paired = paired_rows(ds_name, out)
        tables = summary_tables(out)
        print_summary(ds_name, out, paired, tables)
        plot_dataset(ds_name, out, paired, OUT / f"{stem}.pdf")
        write_csv(paired, LOGS / f"{stem}_paired.csv", PAIRED_FIELDS)
        print(f"Paired statistics saved to {LOGS / f'{stem}_paired.csv'}",
              flush=True)
        write_json(LOGS / f"{stem}.json", ds_name, out, paired, tables,
                   runtime, sys.argv if argv is None else argv)
        print(f"{display_name(ds_name)} completed in {time.time() - t0:.0f}s",
              flush=True)
    print(f"\nAll datasets completed in {time.time() - total_t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
