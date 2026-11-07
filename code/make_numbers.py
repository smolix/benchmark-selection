r"""Extract every paper-facing number from an experiment's result logs.

Reads the result files under an experiment root (the layout the eval_*
scripts write when BENCHSELECT_EXPERIMENT_ROOT is set; the repository
root itself is also a valid root) and emits

  (a) a LaTeX file of \newcommand macros, one per quantity, rounded for
      the text (2 decimals by default), and
  (b) a flat CSV with full precision and provenance, columns
      key,value,source_file,source_key,index,aggregation,kind,scope
      (key = macro name without the backslash; kind = float, std, int,
      num0, num1 or text, which fixes both the parsing of value and the
      macro format; scope = run if the source file lies under the root,
      shared if it was read from outside it, e.g. data/*.matrix.npz),

so paper text can be generated from the logs and two runs can be diffed
(compare_runs.py).  Missing or unreadable files are skipped with a
warning on stderr; a root without any results of its own is an error.

Inputs (ROOT = --root; {ds} is any dataset name, the stem of its
data/{ds}.matrix.npz):

  ROOT/logs/entropy_vs_mi_{ds}.npz      eval_entropy_vs_mi.py
  ROOT/logs/{ds}_results.npz            eval_benchpress.py (emi_*, cv_p*_*;
                                        benchpress_results.npz by default)
  ROOT/logs/baselines_{ds}.npz          eval_baselines.py
  ROOT/logs/greedy_cv_{ds}.npz          eval_greedy_all.py
  ROOT/logs/greedy_cv_splits_{ds}.json  eval_greedy_all.py (per split)
  ROOT/logs/greedy_mi_cv_splits_{ds}.json   plot_mi_selection_all.py (splits)
  ROOT/logs/greedy_cv_selections.json   eval_greedy_all.py (rep 0; used for
                                        datasets without a per-split log)
  ROOT/logs/greedy_mi_cv_selections_all.json   plot_mi_selection_all.py (idem)
  ROOT/logs/tabimpute_{ds}.npz          eval_tabimpute.py
  ROOT/logs/tabimpute_extra_{ds}.npz    eval_tabimpute.py (BENCHSELECT_METHODS)
  ROOT/logs/family_holdout_{ds}.npz     eval_family_holdout.py
  ROOT/logs/cost_budget_{ds}.npz        eval_cost_budget.py
  ROOT/logs/normality_summary.json      run_normality.py
  ROOT/logs/reviewer_checks*.json       eval_reviewer_checks.py
  ROOT/logs/baselines_{ds}_redundancy.json  eval_baselines.py ("macros")
  ROOT/logs/{prefix}*.json              the flat "macros" dict of the
                                        reviewer-request experiments
                                        (JSON_MACRO_LOGS): prospective_,
                                        item_budget_, item_fixed_,
                                        token_costs_, token_fixed_,
                                        cost_fixed_, analyze_objectives_,
                                        benchfamily_holdout_,
                                        selection_stability_,
                                        estimator_ablation_, imputers_
  ROOT/logs/{estimator_ablation,prospective,imputers}_*.npz   per-split
                                        rows for the paired CSVs only
                                        (paired_stats.py)
  ROOT/logs_logit/{entropy_vs_mi,greedy_cv}_{ds}_logit.npz   eval_logit.py
  ROOT/stdout/logit.txt, logit_extra.txt   eval_logit.py stdout (run_all.sh
                                        jobs logit and logit_extra)
  {ds}.cov.npz in --cov-dir, else ROOT/cov, ROOT/data, ROOT and then
      $BENCHSELECT_COV_DIR if set           estimate_covariance.py
  {ds}.matrix.npz in --data-dir (default: path_config.data_dir())

logs/greedy_mi_cv_selections.json (plot_mi_selection_mmlu.py) is the MMLU
subset of greedy_mi_cv_selections_all.json and is not read; nor are the
eval scripts' own *_paired.csv files and greedy_mi_cv_{ds}.npz (no R²).

Matched splits.  The CV logs have one row per matched split
(cv_split.cv_splits: n_repeats x n_folds splits per holdout fraction; the
first n_folds rows are repetition 0, the historical folds) with the
split's rep, fold, split_id, n_train, n_val and n_eligible.  Logs written
before the matched design have no split metadata; their rows are rep-0
folds.  --reps R ... restricts every per-split quantity (means, stds,
paired statistics, selection statistics, the random family rows, which
are recomputed from the {m}_r2_splits rows) to the given
repetitions; --reps 0 gives the numbers of the single-repetition protocol,
to compare with logs that predate it (compare_runs.py --new-reps 0).

Aggregation: R^2, residual variance, MI, log-det and Jaccard values are
np.nanmean / np.nanstd (ddof=0) over splits (or folds of a pre-matched-
design log, or held-out families), exactly as the scripts' summaries and
plots compute them.  NaN entries (failed estimate; k beyond the eligible
benchmarks of a split) are left out.  The CSV aggregation column says how
many were finite ("nanmean over 47/50 splits", "nanmean over 7/9
families"; ", reps 0" under --reps), and the .tex comment of every mean or
std that covers only part of its splits, folds or families says "partial".

Macro naming scheme
-------------------
Names are letters only (TeX control words) and depend only on the
file/key names (plus, where noted, benchmark names and parameter values),
never on the results.  Numbers in the fixed K/P/B/Var/Worst slots are
spelled out in lower case after the capital slot letter (5 -> Kfive, 15 ->
Kfifteen, 95 -> Varninetyfive, 0.5% -> Bzeropointfive); identifiers are
CamelCased, with any digits in them spelled and capitalized (knap_mi ->
KnapMi, alibaba-nlp -> AlibabaNlp, mardia_beta1 -> MardiaBetaOne, r2_k5 ->
RTwoKFive).  Below, {Ds} is the dataset (Mmlu, Mteb, Merged, Benchpress,
or the CamelCased name of any other: olv2sub -> OlvTwoSub, helm_lite ->
HelmLite), {M} the method (Entropy, Mi, Random, Hub, Pca, Trace,
Facloc = facility location, the paper's headline method since
2026-10-19, and eval_baselines.py's similarity ablation FaclocAbs,
FaclocSq; EntropyRf macros are gone with entropy_rf, protocol rule D3), {k}
a spelled-out k in {1, 2, 3, 5, 7, 10, 15}, {p} a spelled-out holdout
percentage (ten, twenty, fifty, ninety) and {B} a CamelCased
benchmark name.  "Splits" are the matched splits of the log (folds of a
pre-matched-design log).

  \Rsq{Ds}{M}K{k}             R^2 split mean, entropy_vs_mi_{ds}.npz
                              ({ds}_results.npz emi_*, e.g. BenchPress)
  \Rsq{Ds}{M}K{k}P{p}         greedy_cv_{ds}.npz at holdout p; {M} is Entropy
                              (greedy), Random (rand_*), Trace (trace_*) or
                              Facloc (facloc_*; the file's methods /
                              method_prefixes); {ds}_results.npz cv_p*
  \RsqBase{Ds}{M}K{k}         baselines_{ds}.npz
  \RsqTab{Ds}{M}K{k}          tabimpute_{ds}.npz and tabimpute_extra_{ds}.npz
  \RsqLogit{Ds}{M}K{k}[P{p}]  logs_logit/ entropy_vs_mi (no P) / greedy_cv (P)
  \RsqFam{Ds}{M}K{k}          family_holdout_{ds}.npz, mean over held-out families
  \RsqFam{Ds}{M}Kfive{Fam}    per-family R^2 at k = 5 (e.g. RsqFamMmluMiKfiveLlama)
  \RsqStd...                  split std of each of the means above ("Rsq" ->
                              "RsqStd"; e.g. RsqStdMmluMiKfive), not per-family
  \ResidVar..., \MiVal...     split means (no std) of the *_resid_var and
                              *_mi_val arrays, named like the R^2 means
                              ("Rsq" -> "ResidVar"/"MiVal") for entropy_vs_mi,
                              {ds}_results emi_* and logs_logit/ entropy_vs_mi
  \ResidVar...P{p}, \LogDet...P{p}   the same for greedy_cv / {ds}_results cv_*
                              resid_var and entropy (= log det R_AA) arrays
  \Pair{S}{T}{Ds}{A}Vs{B}K{k}[P{p}]  paired R^2 difference A - B over the
                              matched splits (paired_stats.py), for the method
                              pairs paired_stats.MACRO_PAIRS (MiVsRandom,
                              MiVsEntropy, MiVsTrace, EntropyVsRandom,
                              TraceVsRandom, MiVsFacloc, EntropyVsFacloc,
                              TraceVsFacloc, FaclocVsRandom and the
                              similarity ablation FaclocVsFaclocAbs,
                              FaclocVsFaclocSq) present in a
                              log, k in
                              {1, 3, 5, 10, 15}; {T} the
                              source tag of the R^2 means ('' entropy_vs_mi /
                              greedy_cv (P) / {ds}_results, Base, Tab, Logit);
                              statistic {S}: Diff (mean difference), Se
                              (Nadeau-Bengio corrected standard error), Lo /
                              Hi (95% t-interval), Win (fraction of splits
                              with A > B), Wins (their number), N (splits
                              where both are finite).  E.g.
                              PairDiffBaseMergedMiVsTraceKfive
  \Split{T}{Ds}[P{p}]{F}      split design of a matched-design log: NSplits,
                              NReps, EligMin / EligMax / EligMean (eligible
                              benchmarks per split), NRestricted (splits with
                              fewer than all benchmarks eligible), NFailed
                              (failed covariance estimates)
  \Ties{T}{Ds}{M}[P{p}]       picks of the size-15 selections decided by the
                              lowest index ({m}_index_ties at k = 15, summed
                              over splits); \TiesSplits... the number of
                              splits with at least one
  \JaccFam{Ds}{M}K{k}         selection-stability Jaccard (k = 5, 10, 15),
                              mean over families; \JaccStdFam... its std
  \NFam{Ds}, \FamList{Ds}     number / comma-separated list of held-out families
  \Sel{Ds}{M}P{p}...          selection order per split, as in the selection-
                              order figures (plot_selection_order.py), from
                              greedy_cv_splits_{ds}.json ({M} = Entropy;
                              Trace and Facloc at the figures' 10% holdout
                              only, P ten: SEL_SPLIT_LOGS, SEL_HOLDOUTS)
                              and greedy_mi_cv_splits_{ds}.json
                              ({M} = Mi), else from the rep-0 logs
                              greedy_cv_selections.json /
                              greedy_mi_cv_selections_all.json; splits
                              without picks are left out (NEmpty of them):
                                NFolds (= NSplits, splits shown; the
                                historical name), NReps, NDistinct
                                (benchmarks picked in any split), NAllFolds
                                (picked in every split), NLeadAllFolds
                                (leading rows of the figure picked in every
                                split), CommonPrefix (picks identical, in
                                order, in all splits), FirstName / FirstFolds
                                (most frequent first pick, ties to the
                                earliest split, and its split count),
                                FirstDistinct, RowName{r} (r-th row of the
                                figure: picked in >= max(2, 20% of the)
                                splits, by mean rank; r <= 10), Fold{f} /
                                Rep{r}Fold{f} (picks of fold f of rep 0 / of
                                rep r > 0 in order, text), BenchFolds{B}
                                (splits picking B), BenchMeanRank{B} (the
                                figure's mean position, 16 where not picked),
                                BenchFirst{B} (splits starting with B; only
                                if > 0), IndexTies (picks among the first 15
                                decided by the lowest index, summed)
  \CostRsq{Ds}{M}B{b}         cost_budget_{ds}.npz R^2 split mean at budget b%
                              of the suite (Bone = 1%, Bzeropointtwo = 0.2%);
                              {M} in KnapEntropy, KnapMi, CardEntropy, CardMi,
                              CheapFirst, Random, KnapTrace, CardTrace,
                              KnapFacloc, CardFacloc;
                              \CostRsqStd... its std
  \CostSize{Ds}{M}B{b}        mean selected-set size at that budget
  \CostBudget{Ds}B{b}         the budget in examples; \CostTotal{Ds} full suite
  \CostMin{Ds}, \CostMax{Ds}  cheapest / most expensive benchmark (examples),
                              \CostMinName{Ds}, \CostMaxName{Ds} their names,
                              \CostRatio{Ds} max / min, \CostItems{Ds}{B} each
  \Norm{Ds}{Field}            normality_summary.json scalars (NormMmluWMedian,
                              NormMergedNRejectBh, NormMtebMardiaBetaOne, ...)
  \Norm{Ds}WorstW{r}          W of the r-th worst benchmark (r = one, two, ...);
                              \Norm{Ds}WorstName{r} its name
  \EigK{Ds}Var{v}             number of correlation eigencomponents explaining
                              v% of variance, v in ninety, ninetyfive, ninetynine
                              (cov file if found, else recomputed; CSV says which)
  \EigKLogit{Ds}Var{v}        the same in logit space, as printed by eval_logit.py
                              (stdout/logit.txt; v in ninety, ninetyfive)
  \EmIter{Ds}, \EmConverged{Ds}   EM iterations / converged flag of the cov file
  \Data{Ds}M, \Data{Ds}N      matrix shape; \Data{Ds}ObsPct, \Data{Ds}MissPct
  \Rev{Tag}{Path}             reviewer_checks{_tag}.json flattened generically:
                              nested keys CamelCased and concatenated; list items
                              named by their dataset/name/method/... field when
                              unique, else by their leading parameter fields
                              (up to 3 string, integer or <= 3-decimal values
                              that name every item uniquely, e.g.
                              FracZeropointoneDeltaZeropointone), else Item{i}
                              (0-based, ItemZero); integer lists (indices,
                              folds, k grids) and lists of them become one
                              exact text macro ('41, 50, 15', nested lists
                              joined by '; '); other numeric lists of up to 16
                              values also get one macro per item and add
                              {Path}ListMean (and ListStd/ListMin/ListMax/
                              ListCount when longer than 16 or nested, ragged
                              nests pooled); string lists get one comma-joined
                              text macro {Path} plus per-item macros when
                              short; strings are text macros; booleans 0/1;
                              null is n/a; a ...Frac/...Fraction value also
                              gets {Path}Pct = 100 x value.  Name clashes get
                              AltOne, AltTwo, ... (with a warning)

  {Name}                      every entry of the flat "macros" dict of a
                              JSON log above, under its own name (the
                              scripts build them with camel / int_words /
                              num_words; prefixes Prosp/Rsq...Prosp,
                              ItemBudget, ItemBudgetFixed, Tok, TokFixed,
                              CostFixed, Obj, BenchFam, Stab, Est, Imp /
                              RsqImp / Pair...Imp, Redund; see each
                              script's docstring); JSON integers are
                              counts, null is n/a, a float named with a
                              Std or Se component gets std digits

Macro groups.  Every record belongs to one group (TEX_GROUPS): 'main' (the
paper's DATASETS mmlu, mteb, merged, benchpress: their logs, paired and
design macros, selection orders, eigenspectra, normality, data shapes, and
the reviewer checks), 'datasets' (the same sources of every other dataset,
e.g. the data-release datasets and merged_v1) and one group per
JSON_MACRO_LOGS experiment.  --tex-dir DIR writes 'main' to --out-tex
(default DIR/numbers.tex) and each other group to DIR/numbers_{group}.tex
(every file, also an empty one), so the paper \input's numbers.tex plus
the experiment files it cites; a full run has well over 100k macros, more
than TeX's string pool holds.  Without --tex-dir, --out-tex gets every
macro (warning above TEX_MACRO_WARN).  The CSV always has every record.

--datasets also filters the JSON logs: a log is read if its file-name stem
after the prefix is a wanted dataset or starts with one and '_'
(token_costs_helm_classic_exclude, benchfamily_holdout_mteb_sub,
analyze_objectives_merged_fulloracle); a prospective tag counts as the
data of its setting (helm -> helm_lite, epoch_bymodel -> epoch,
olv2sub_math -> olv2sub).

If two sources map to the same name (e.g. a future entropy_vs_mi_benchpress.npz
next to benchpress_results.npz), the first one processed wins and the other is
reported as a warning.

Formatting: real values use --digits decimals (default 2) and standard
deviations --std-digits (default digits + 1); counts are integers, mean set
sizes, mean ranks and percentages 1 decimal; integer parts of 4 or more
digits are grouped with --thousands (default {,}: 5{,}452); minus signs are
\ensuremath{-}, NaN is \textrm{n/a}, strings are LaTeX-escaped.

Paired statistics: --paired-dir DIR also writes DIR/paired_{ds}.csv, the
per-dataset CSV of every method pair, k and source of paired_stats.py
(same tables, same --reps; including the CSV-only tables of the estimator
ablation, the prospective control splits and the imputers, whose scripts
write their own paired macros).

Usage:
    python make_numbers.py --root DIR --out-tex numbers.tex --out-csv numbers.csv
                           [--tex-dir DIR] [--cov-dir DIR] [--data-dir DIR]
                           [--eigen auto|cov|recompute|skip] [--digits 2]
                           [--std-digits 3] [--thousands '{,}']
                           [--reps 0 ...] [--datasets NAME ...]
                           [--paired-dir DIR]
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import sys
from collections import Counter, OrderedDict

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir as default_data_dir
import paired_stats

K_REPORT = [1, 2, 3, 5, 7, 10, 15]
K_FAMILY = 5                          # k of the per-family R² macros
K_PAIRED = paired_stats.K_MACRO       # k of the paired-difference macros
EIG_LEVELS = [0.90, 0.95, 0.99]
EIG_DATASETS = ["mmlu", "mteb", "merged"]   # as in plot_eigenspectrum.py
SEL_K_MAX = 15                        # K_MAX of the selection-order plots
SEL_ROWS = 10                         # RowName macros: first figure rows
SEL_MIN_ROW_SHARE = 0.2               # plot_selection_order.MIN_ROW_SHARE
# Per-split selection logs (prefix of logs/{prefix}{ds}.json) and the
# methods whose Sel macros they give (trace and facloc: the entries that
# eval_greedy_all.py appends since 2026-10-19; a log without them gives
# none); the rep-0 logs serve datasets without a per-split log.
SEL_SPLIT_LOGS = [("greedy_cv_splits_",
                   OrderedDict([("entropy", "Entropy"), ("trace", "Trace"),
                                ("facloc", "Facloc")])),
                  ("greedy_mi_cv_splits_", OrderedDict([("mi", "Mi")]))]
# Holdout fractions of the Sel macros of a method (default: every holdout
# of its log): trace and facloc only at the holdout of their selection-
# order figures (plot_selection_order.HOLDOUT_FRAC, run_all.sh's
# selection_order jobs), which keeps numbers.tex and numbers_datasets.tex
# within TEX_MACRO_WARN at 5 x 10 splits
SEL_FIGURE_HOLDOUT = 0.1
SEL_HOLDOUTS = {"Trace": (SEL_FIGURE_HOLDOUT,),
                "Facloc": (SEL_FIGURE_HOLDOUT,)}
SEL_FILES = [("greedy_cv_selections.json", "Entropy"),
             ("greedy_mi_cv_selections_all.json", "Mi")]
# Paired macros: statistic suffix -> (paired_stats field, kind)
PAIR_STATS = OrderedDict([("Diff", ("mean_diff", "float")),
                          ("Se", ("se_nb", "std")),
                          ("Lo", ("ci_low", "float")),
                          ("Hi", ("ci_high", "float")),
                          ("Win", ("win_rate", "float")),
                          ("Wins", ("n_wins", "int")),
                          ("N", ("n_splits", "int"))])
AUX_METRICS = OrderedDict([("resid_var", "ResidVar"), ("mi_val", "MiVal"),
                           ("entropy", "LogDet")])
MAX_LIST_ITEMS = 16                   # longer numeric JSON lists: summary only
MAX_TEXT_LEN = 4000                   # longer JSON strings are skipped
MAX_NAME_FIELDS = 3                   # leading parameter fields naming items
ID_FIELDS = ("dataset", "name", "method", "check", "label", "id", "key",
             "benchmark")
CSV_FIELDS = ["key", "value", "source_file", "source_key", "index",
              "aggregation", "kind", "scope"]
KINDS = ("float", "std", "int", "num0", "num1", "text")
THOUSANDS = "{,}"
RESULT_DIRS = ("logs", "logs_logit", "stdout")
LOGIT_STDOUT = ("logit.txt", "logit_extra.txt")   # eval_logit eigenspectra

DATASETS = ["mmlu", "mteb", "merged", "benchpress"]

# Macro groups (--tex-dir): 'main' = numbers.tex, the paper's numbers:
# every per-dataset source of DATASETS, the reviewer checks and the
# paired / design macros of their tables; 'datasets' = the same sources of
# any other dataset (data release: olv2sub, epoch, helm_*, mteb2026,
# merged_v1, ...); then one group per experiment whose JSON logs carry a
# flat "macros" dict (JSON_MACRO_LOGS).  Each non-main group goes to
# DIR/numbers_{group}.tex (written even when empty), so the paper inputs
# only the files it cites: all groups together are well over 100k macros,
# beyond what TeX's string pool holds in one document.
JSON_MACRO_LOGS = [
    # (file prefix, group, script)
    ("prospective_", "prospective", "eval_prospective.py"),
    ("item_budget_", "item_budget", "eval_item_budget.py"),
    ("item_fixed_", "item_fixed", "eval_item_budget.py --fixed-cost"),
    ("token_costs_", "token_costs", "eval_token_costs.py"),
    ("token_fixed_", "token_fixed", "eval_token_costs.py --fixed-cost"),
    ("cost_fixed_", "cost_fixed", "eval_cost_budget.py --fixed-cost"),
    ("analyze_objectives_", "analyze_objectives", "analyze_objectives.py"),
    ("benchfamily_holdout_", "benchfamily_holdout",
     "eval_benchfamily_holdout.py"),
    ("selection_stability_", "selection_stability",
     "eval_selection_stability.py"),
    ("estimator_ablation_", "estimator_ablation",
     "eval_estimator_ablation.py"),
    ("imputers_", "imputers", "eval_imputers.py"),
]
TEX_GROUPS = (["main", "datasets"]
              + [group for _, group, _ in JSON_MACRO_LOGS])
# eval_baselines.py's redundancy summaries (per dataset, group by dataset)
REDUNDANCY_SUFFIX = "_redundancy.json"
# Above this many macros in one .tex file, warn (TeX capacity; --tex-dir)
TEX_MACRO_WARN = 40000
# Experiment logs of fixed settings (not of DATASETS) that a full
# run_all.sh run writes: {data set: [log stems]}; prospective tags name the
# setting, whose data paired_stats.PROSPECTIVE_DATA gives
FIXED_EXPERIMENT_LOGS = {
    "mteb2026": ["prospective_mteb2026"],
    "helm_lite": ["prospective_helm", "token_costs_helm_lite",
                  "token_fixed_helm_lite"],
    "epoch": ["prospective_epoch", "prospective_epoch_bymodel"],
    "olv2sub": ["prospective_olv2sub", "prospective_olv2sub_math"],
    "olv1items": ["item_budget_olv1items", "item_fixed_olv1items"],
    "helm_mmlu": ["token_costs_helm_mmlu", "token_fixed_helm_mmlu"],
    "helm_classic": ["token_costs_helm_classic",
                     "token_costs_helm_classic_exclude",
                     "token_fixed_helm_classic"],
}


def expected_files(want=None):
    """{subdir: [file names]} that a full run_all.sh + run_tabimpute.sh run
    writes for the paper's DATASETS and the fixed experiments (those whose
    data set `want(ds)` accepts), whose absence is reported; any other
    dataset's logs are read if present.  The eval_imputers.py logs
    (run_all.sh imputers_<ds>, the merge of the per-repetition shards, in
    the default run) count, like the TabImpute ones."""
    want = want or (lambda ds: True)
    d3 = [ds for ds in DATASETS[:3] if want(ds)]
    d4 = [ds for ds in DATASETS if want(ds)]
    fam = [ds for ds in ("mmlu", "mteb", "merged") if want(ds)]
    fixed = [f"{stem}.json" for ds, stems in FIXED_EXPERIMENT_LOGS.items()
             if want(ds) for stem in stems]
    return {
        "logs": ([f"entropy_vs_mi_{ds}.npz" for ds in d3]
                 + (["benchpress_results.npz"] if want("benchpress") else [])
                 + [f"baselines_{ds}.npz" for ds in d4]
                 + [f"greedy_cv_{ds}.npz" for ds in d3]
                 + ([name for name, _ in SEL_FILES] if d3 else [])
                 + [f"tabimpute_{ds}.npz" for ds in d4]
                 + [f"tabimpute_extra_{ds}.npz" for ds in d4]
                 + [f"family_holdout_{ds}.npz" for ds in d4]
                 + [f"cost_budget_{ds}.npz" for ds in ("mmlu", "mteb")
                    if want(ds)]
                 + [f"cost_fixed_{ds}.json" for ds in ("mmlu", "mteb")
                    if want(ds)]
                 + (["normality_summary.json", "reviewer_checks.json"]
                    if d4 else [])
                 + [f"{stem}_{ds}.json" for ds in d4
                    for stem in ("analyze_objectives", "selection_stability",
                                 "estimator_ablation", "imputers")]
                 + (["analyze_objectives_merged_fulloracle.json"]
                    if want("merged") else [])
                 + [f"benchfamily_holdout_{ds}.json" for ds in fam]
                 + [f"benchfamily_holdout_{ds}{tag}.json"
                    for ds in fam if ds != "mmlu"
                    for tag in ("_sub", "_blindcov")]
                 + fixed),
        "logs_logit": ([f"entropy_vs_mi_{ds}_logit.npz" for ds in d4]
                       + [f"greedy_cv_{ds}_logit.npz" for ds in d4]),
        "stdout": ["logit.txt"] if d4 else [],
    }


EXPECTED = expected_files()


# ── Names ────────────────────────────────────────────────────────

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_TOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+(?:\.\d+)?")


def int_words(n):
    """Spell an integer in lower-case letters: 95 -> 'ninetyfive'."""
    n = int(n)
    if n < 0:
        return "minus" + int_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (_ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + "hundred" + (int_words(rest) if rest else "")
    for base, name in ((10 ** 12, "trillion"), (10 ** 9, "billion"),
                       (10 ** 6, "million"), (10 ** 3, "thousand")):
        if n >= base:
            rest = n % base
            return (int_words(n // base) + name
                    + (int_words(rest) if rest else ""))


def num_words(x):
    """Spell a number: 5 -> 'five', 0.5 -> 'zeropointfive' (6 decimals max)."""
    if isinstance(x, str):
        s = x
    else:
        s = np.format_float_positional(round(float(x), 6), trim="-")
    neg = s.startswith("-")
    whole, _, frac = s.lstrip("-").partition(".")
    out = ("minus" if neg else "") + int_words(int(whole or 0))
    if frac:
        out += "point" + "".join(_ONES[int(c)] for c in frac)
    return out


def camel(s):
    """CamelCase an identifier into letters only ('knap_mi' -> 'KnapMi')."""
    out = []
    for tok in _TOKEN.findall(str(s).replace("²", "2")):
        if tok[0].isdigit():
            tok = num_words(tok)
        out.append(tok[0].upper() + tok[1:].lower())
    return "".join(out)


# ── Formatting ───────────────────────────────────────────────────

_TEX_ESCAPE = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
               "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
               "\n": " ", "\r": " "}


def latex_escape(s):
    return "".join(_TEX_ESCAPE.get(c, c) for c in str(s))


def format_value(value, kind, digits=2, std_digits=None, thousands=THOUSANDS):
    """Macro body for one value (see 'Formatting' in the module docstring)."""
    if kind == "text":
        return latex_escape(value)
    v = float(value)
    if math.isnan(v):
        return r"\textrm{n/a}"
    if math.isinf(v):
        return r"\ensuremath{-\infty}" if v < 0 else r"\ensuremath{\infty}"
    if kind == "int":
        s = str(int(value))
    elif kind == "num0":
        s = f"{v:.0f}"
    elif kind == "num1":
        s = f"{v:.1f}"
    elif kind == "std":
        s = f"{v:.{digits + 1 if std_digits is None else std_digits}f}"
    else:
        s = f"{v:.{digits}f}"
    sign = ""
    if s.startswith("-"):
        s = s[1:]
        if float(s) != 0.0:          # no sign on a value that rounds to 0
            sign = r"\ensuremath{-}"
    whole, dot, frac = s.partition(".")
    if thousands and len(whole) > 3:
        head = len(whole) % 3 or 3
        whole = whole[:head] + "".join(thousands + whole[i:i + 3]
                                       for i in range(head, len(whole), 3))
    return sign + whole + dot + frac


def csv_value(value, kind):
    """Full-precision CSV cell (shortest round-trip repr for floats)."""
    if kind == "text":
        return str(value)
    if kind == "int":
        return str(int(value))
    return repr(float(value))


_COVERAGE = re.compile(r"over (\d+)/(\d+) ")


def coverage(aggregation):
    """(finite, total) of an aggregation such as 'nanmean over 7/9
    families', or None if it is not a mean/std over folds or families."""
    m = _COVERAGE.search(str(aggregation))
    return (int(m.group(1)), int(m.group(2))) if m else None


def is_partial(record):
    """True for a mean/std over only some of its folds or families."""
    cov = coverage(record["aggregation"])
    return cov is not None and cov[0] < cov[1]


def from_results(record):
    """True if a record was read from the run's own logs/, logs_logit/
    or stdout/ (not from cov files, data matrices or outside the root)."""
    return (record.get("scope") == "run"
            and str(record["source_file"]).split("/")[0] in RESULT_DIRS)


# ── Collection ───────────────────────────────────────────────────

class Numbers:
    """Ordered collection of extracted quantities, one per macro.

    reps: CV repetitions that per-split quantities are restricted to (None:
    all rows); datasets: dataset names to extract (None: all found)."""

    def __init__(self, root, reps=None, datasets=None):
        self.root_arg = str(root)
        self.root = pathlib.Path(root).expanduser().resolve()
        self.records = OrderedDict()
        self.warnings = []
        self.section = ""
        self.group = "main"         # TEX_GROUPS entry of the records added
        self.reps = None if reps is None else sorted({int(r) for r in reps})
        self.datasets = None if datasets is None else list(datasets)
        self.tables = []            # paired_stats split tables (extract)

    def want(self, ds):
        """True if dataset ds is extracted (--datasets)."""
        return self.datasets is None or str(ds) in self.datasets

    def want_stem(self, rest):
        """True if a log stem's dataset part (e.g. 'helm_classic_exclude',
        'mteb_sub', 'merged_fulloracle') names a wanted dataset: rest is
        the dataset or starts with it and '_'."""
        if self.datasets is None:
            return True
        return any(rest == ds or rest.startswith(ds + "_")
                   for ds in self.datasets)

    @staticmethod
    def ds_group(ds):
        """'main' for the paper's DATASETS, 'datasets' for any other."""
        return "main" if str(ds) in DATASETS else "datasets"

    def use_dataset(self, ds):
        """Records added next belong to dataset ds's group."""
        self.group = self.ds_group(ds)

    def rows(self, d, pre, n_rows, holdout_frac=paired_stats.DEFAULT_HOLDOUT):
        """(row indices, unit, index text) of the n_rows split rows under
        key prefix `pre` that per-split means cover: all rows, or those of
        self.reps.  unit is 'splits' for a matched-design log, 'folds' for
        a log without split metadata (its rows are rep-0 folds)."""
        meta, legacy = paired_stats.split_meta(d, pre, n_rows, holdout_frac)
        unit = "folds" if legacy else "splits"
        if self.reps is None:
            return np.arange(n_rows), unit, ":"
        idx = np.flatnonzero(np.isin(meta["rep"], self.reps))
        return idx, unit, f"rep in ({', '.join(map(str, self.reps))})"

    def reps_note(self):
        """', reps 0 1' under --reps, else ''."""
        if self.reps is None:
            return ""
        return f", reps {' '.join(map(str, self.reps))}"

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)

    def rel(self, path):
        """Path relative to the root (or the repository), else absolute."""
        path = pathlib.Path(path).resolve()
        for base in (self.root, CODE.parent):
            try:
                return str(path.relative_to(base))
            except ValueError:
                pass
        return str(path)

    def add(self, key, value, kind, source_file, source_key, index="",
            aggregation="", on_clash="skip"):
        """Record one quantity.  on_clash='skip' drops a duplicate name
        (first source wins); 'rename' appends AltOne, AltTwo, ..."""
        if not re.fullmatch(r"[A-Za-z]+", key):
            raise ValueError(f"macro name {key!r} is not letters-only")
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r} for {key}")
        src = self.rel(source_file)
        if key in self.records:
            old = self.records[key]
            if on_clash == "skip":
                self.warn(f"\\{key} from {src}:{source_key} duplicates "
                          f"{old['source_file']}:{old['source_key']}; ignored")
                return
            n = 1
            while f"{key}Alt{int_words(n).capitalize()}" in self.records:
                n += 1
            new_key = f"{key}Alt{int_words(n).capitalize()}"
            self.warn(f"\\{key} from {src}:{source_key} clashes with "
                      f"{old['source_file']}:{old['source_key']}; "
                      f"renamed to \\{new_key}")
            key = new_key
        if kind == "int":
            value = int(value)
        elif kind != "text":
            value = float(value)
        scope = ("run" if pathlib.Path(source_file).resolve()
                 .is_relative_to(self.root) else "shared")
        if self.group not in TEX_GROUPS:
            raise ValueError(f"unknown macro group {self.group!r}")
        self.records[key] = {"key": key, "value": value, "kind": kind,
                             "source_file": src, "source_key": source_key,
                             "index": index, "aggregation": aggregation,
                             "scope": scope, "section": self.section,
                             "group": self.group}


def _read_npz(nums, path):
    try:
        with np.load(path, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:
        nums.warn(f"cannot read {path}: {e}; skipped")
        return None


def _read_json(nums, path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except Exception as e:
        nums.warn(f"cannot read {path}: {e}; skipped")
        return None


def _guarded(nums, path, func, *args, **kwargs):
    """Run one file's extractor all-or-nothing: if the file is malformed,
    drop whatever it added so far and report it instead of crashing."""
    before = set(nums.records)
    try:
        func(nums, *args, **kwargs)
    except Exception as e:
        for key in [k for k in nums.records if k not in before]:
            del nums.records[key]
        nums.warn(f"{nums.rel(path)}: extraction failed "
                  f"({type(e).__name__}: {e}); file skipped")


def _glob(directory, pattern):
    """Sorted matches, ignoring dotfiles (e.g. macOS '._' resource forks)."""
    return [p for p in sorted(directory.glob(pattern))
            if not p.name.startswith(".")]


def _stem_ds(path, prefix, suffix=".npz"):
    return path.name[len(prefix):-len(suffix)]


def _nan_stats(x):
    """(nanmean, nanstd ddof=0, #non-NaN) of a 1-D array; NaN if empty."""
    x = np.asarray(x, dtype=float)
    n = int((~np.isnan(x)).sum())
    if n == 0:
        return np.nan, np.nan, 0
    return float(np.nanmean(x)), float(np.nanstd(x)), n


def _method_keys(d, prefix="", suffix="_r2"):
    """(method, key) for every '<prefix><method><suffix>' array, in file
    order."""
    out = []
    for key in d:
        if key.startswith(prefix) and key.endswith(suffix):
            method = key[len(prefix):-len(suffix)]
            if method:
                out.append((method, key))
    return out


# ── R² blocks ────────────────────────────────────────────────────

def _add_r2(nums, path, src_key, arr, head, tail="", unit="folds",
            prefix="Rsq", with_std=True, rows=None, where=":"):
    """{prefix}{head}K{k}{tail} (split mean) and, with_std,
    {prefix}Std{head}K{k}{tail} (split std) for every reported k, from a
    (splits, k_max) array (R² unless another prefix is given), over the
    rows `rows` (default all; `where` describes them in the CSV index)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        nums.warn(f"{nums.rel(path)}:{src_key} has shape {arr.shape}, "
                  f"expected ({unit}, k); skipped")
        return
    if rows is not None:
        arr = arr[rows]
    note = nums.reps_note() if rows is not None else ""
    all_nan = []
    for k in K_REPORT:
        if k > arr.shape[1]:
            continue
        mean, std, n = _nan_stats(arr[:, k - 1])
        if n == 0:
            all_nan.append(k)
        name = f"{head}K{int_words(k)}{tail}"
        idx = f"[{where}, {k - 1}]"
        nums.add(prefix + name, mean, "float", path, src_key, idx,
                 f"nanmean over {n}/{arr.shape[0]} {unit}{note}")
        if with_std:
            nums.add(prefix + "Std" + name, std, "std", path, src_key, idx,
                     f"nanstd(ddof=0) over {n}/{arr.shape[0]} {unit}{note}")
    if all_nan:
        # e.g. k beyond the eligible benchmarks, or nothing left to impute
        nums.warn(f"{nums.rel(path)}:{src_key}: all {unit} NaN at k = "
                  f"{', '.join(map(str, all_nan))}")


_CV_KEY = re.compile(r"^(cv_)?p(\d+)_(.*?)(r2|resid_var|entropy)$")


def _add_greedy_cv(nums, path, d, ds, tag=""):
    """Greedy CV arrays {cv}p{pct}_{pre}{metric} (cv = '' or 'cv_', as in
    {ds}_results.npz; method key prefix pre as in the file's methods /
    method_prefixes, by default '' entropy, 'rand_' random): R² with
    std first, then split means of
    resid_var and entropy (log det)."""
    prefixes = {"": paired_stats._cv_prefixes(d, ""),
                "cv_": paired_stats._cv_prefixes(d, "cv_")}
    matches = [(key, _CV_KEY.match(key)) for key in d]
    rows_of = {}
    for want_r2 in (True, False):
        for key, m in matches:
            if not m or (m.group(4) == "r2") != want_r2:
                continue
            cv, pct, pre, metric = (m.group(1) or "", int(m.group(2)),
                                    m.group(3), m.group(4))
            method = prefixes[cv].get(pre)
            if method is None or np.asarray(d[key]).ndim != 2:
                continue
            hp = f"{cv}p{pct}_"
            if hp not in rows_of:
                rows_of[hp] = nums.rows(d, hp, np.asarray(d[key]).shape[0],
                                        pct / 100.0)
            rows, unit, where = rows_of[hp]
            head = f"{tag}{camel(ds)}{camel(method)}"
            tail = "P" + int_words(pct)
            if want_r2:
                _add_r2(nums, path, key, d[key], head, tail, unit,
                        rows=rows, where=where)
            else:
                _add_r2(nums, path, key, d[key], head, tail, unit,
                        prefix=AUX_METRICS[metric], with_std=False,
                        rows=rows, where=where)


def _add_methods(nums, path, d, ds, tag="", prefix="", aux=False):
    """R² of every '<prefix><method>_r2' array; with aux, also the split
    means of the matching _resid_var and _mi_val arrays."""
    found = _method_keys(d, prefix)
    if not found:
        return
    rows, unit, where = nums.rows(d, prefix, np.asarray(d[found[0][1]])
                                  .shape[0])
    for method, key in found:
        _add_r2(nums, path, key, d[key], f"{tag}{camel(ds)}{camel(method)}",
                unit=unit, rows=rows, where=where)
    if not aux:
        return
    for metric in ("resid_var", "mi_val"):
        for method, key in _method_keys(d, prefix, "_" + metric):
            _add_r2(nums, path, key, d[key],
                    f"{tag}{camel(ds)}{camel(method)}",
                    prefix=AUX_METRICS[metric], with_std=False, unit=unit,
                    rows=rows, where=where)


# ── Family holdout and cost budget ───────────────────────────────

def _family_rows(nums, d, method, arr):
    """(F, k) family rows of a method under --reps: the nan-mean over the
    selected repetitions of its {m}_r2_splits rows (rep, fold = family
    index), as eval_family_holdout.rep_mean averages all of them; the
    stored rows without --reps or in a log without split rows (whose rows
    are rep 0).  The flag is False if --reps could not be applied."""
    key = f"{method}_r2_splits"
    if nums.reps is None:
        return arr, True
    if "rep" not in d or "fold" not in d:          # pre-matched-design log
        return (arr if 0 in nums.reps
                else np.full(arr.shape, np.nan)), True
    if key not in d:
        return arr, False
    splits = np.asarray(d[key], dtype=float)
    rep, fold = np.asarray(d["rep"]), np.asarray(d["fold"])
    out = np.full(arr.shape, np.nan)
    for f in range(arr.shape[0]):
        sel = splits[(fold == f) & np.isin(rep, nums.reps)]
        finite = ~np.isnan(sel)
        count = finite.sum(axis=0)
        total = np.where(finite, sel, 0.0).sum(axis=0)
        out[f] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out, True


def _add_family(nums, path, d, ds):
    D = camel(ds)
    fams = [str(f) for f in d.get("families", [])]
    for method, key in _method_keys(d):
        arr = np.asarray(d[key], dtype=float)
        if arr.ndim == 2:
            arr, applied = _family_rows(nums, d, method, arr)
            if not applied:
                nums.warn(f"{nums.rel(path)}:{key}: no {method}_r2_splits; "
                          f"--reps ignored (rows are the stored means)")
        _add_r2(nums, path, key, arr, f"Fam{D}{camel(method)}",
                unit="families")
        if arr.ndim == 2 and fams and arr.shape[0] == len(fams) \
                and arr.shape[1] >= K_FAMILY:
            for i, fam in enumerate(fams):
                nums.add(f"RsqFam{D}{camel(method)}K{int_words(K_FAMILY)}"
                         f"{camel(fam)}", arr[i, K_FAMILY - 1], "float",
                         path, key, f"[{i}, {K_FAMILY - 1}]",
                         f"held-out family {fam}")
        elif arr.ndim == 2:
            nums.warn(f"{nums.rel(path)}:{key}: families missing or "
                      f"mismatched; per-family R² skipped")
    for key in d:
        m = re.match(r"^stability_(.+)_k(\d+)$", key)
        if not m:
            continue
        mean, std, n = _nan_stats(d[key])
        size = np.asarray(d[key]).size
        name = f"Fam{D}{camel(m.group(1))}K{int_words(int(m.group(2)))}"
        nums.add("Jacc" + name, mean, "float", path, key, "[:]",
                 f"nanmean over {n}/{size} families")
        nums.add("JaccStd" + name, std, "std", path, key, "[:]",
                 f"nanstd(ddof=0) over {n}/{size} families")
    if "families" in d:
        nums.add(f"NFam{D}", len(fams), "int", path, "families", "", "len")
        nums.add(f"FamList{D}", ", ".join(fams), "text", path, "families",
                 "", "comma-joined")


def _add_costs(nums, path, d, D):
    """Suite total, cheapest / most expensive benchmark and per-benchmark
    costs (examples) from the 'costs' and 'benchmark_names' arrays."""
    costs = np.asarray(d["costs"], dtype=float)
    nums.add(f"CostTotal{D}", np.sum(costs), "num0", path, "costs", "",
             "sum over benchmarks")
    if not costs.size:
        return
    lo, hi = int(np.argmin(costs)), int(np.argmax(costs))
    nums.add(f"CostMin{D}", costs[lo], "num0", path, "costs", f"[{lo}]",
             "min over benchmarks")
    nums.add(f"CostMax{D}", costs[hi], "num0", path, "costs", f"[{hi}]",
             "max over benchmarks")
    nums.add(f"CostRatio{D}", costs[hi] / costs[lo] if costs[lo] > 0
             else np.nan, "num0", path, "costs", "", "max / min")
    names = [str(n) for n in d.get("benchmark_names", [])]
    if len(names) != costs.size:
        nums.warn(f"{nums.rel(path)}: benchmark_names missing or "
                  f"mismatched; per-benchmark costs skipped")
        return
    nums.add(f"CostMinName{D}", names[lo], "text", path, "benchmark_names",
             f"[{lo}]", "argmin of costs (first if tied)")
    nums.add(f"CostMaxName{D}", names[hi], "text", path, "benchmark_names",
             f"[{hi}]", "argmax of costs (first if tied)")
    for i, (name, c) in enumerate(zip(names, costs)):
        nums.add(f"CostItems{D}{camel(name) or 'Unnamed'}", c, "num0", path,
                 "costs", f"[{i}]", f"cost of {name}", "rename")


def _add_cost(nums, path, d, ds):
    D = camel(ds)
    if "budgets_frac" not in d:
        nums.warn(f"{nums.rel(path)}: no budgets_frac; skipped")
        return
    fracs = np.asarray(d["budgets_frac"], dtype=float)
    words = [num_words(100.0 * f) for f in fracs]
    if "budgets" in d:
        for b, w in enumerate(words):
            nums.add(f"CostBudget{D}B{w}", d["budgets"][b], "num0", path,
                     "budgets", f"[{b}]", f"{100.0 * fracs[b]:g}% of total")
    cols = None
    note = nums.reps_note()
    for method, key in _method_keys(d):
        r2 = np.asarray(d[key], dtype=float)          # (budgets, splits)
        size_key = f"{method}_size"
        if r2.ndim != 2 or r2.shape[0] != len(fracs):
            nums.warn(f"{nums.rel(path)}:{key} has shape {r2.shape}, "
                      f"expected ({len(fracs)}, splits); skipped")
            continue
        if cols is None:            # split columns of the --reps filter
            cols, unit, where = nums.rows(d, "", r2.shape[1])
        for b, w in enumerate(words):
            mean, std, n = _nan_stats(r2[b, cols])
            name = f"{D}{camel(method)}B{w}"
            idx = f"[{b}, {where}]"
            nums.add("CostRsq" + name, mean, "float", path, key, idx,
                     f"nanmean over {n}/{cols.size} {unit}{note}")
            nums.add("CostRsqStd" + name, std, "std", path, key, idx,
                     f"nanstd(ddof=0) over {n}/{cols.size} {unit}{note}")
            if size_key in d:
                sizes = np.asarray(d[size_key], dtype=float)[b, cols]
                s_mean, _, s_n = _nan_stats(sizes)
                nums.add("CostSize" + name, s_mean, "num1", path, size_key,
                         idx, f"nanmean over {s_n}/{sizes.size} {unit}{note}")
    if "costs" in d:
        _add_costs(nums, path, d, D)


# ── Selection order ──────────────────────────────────────────────

def _sel_entries(entries, method=None, default_ds=None):
    """[(i, entry)] of a selection log (a list of per-split dicts), keeping
    those of `method` (per-split logs; None: all, the rep-0 logs), with
    rep / split_id defaulted for entries that predate the matched design
    (rep 0, split_id = fold)."""
    if not isinstance(entries, list):
        raise ValueError("expected a list of per-split entries")
    out = []
    for i, e in enumerate(entries):
        if method is not None and e.get("method") != method:
            continue
        fold = int(e.get("fold", i))
        out.append((i, dict(e, dataset=str(e.get("dataset", default_ds)),
                            fold=fold, rep=int(e.get("rep", 0)),
                            split_id=int(e.get("split_id", fold)))))
    return out


def _add_selections(nums, path, entries, method):
    """Per (dataset, holdout) group of selection-log entries ([(i, entry)],
    _sel_entries) the statistics the selection-order figures show
    (plot_selection_order.py): split counts and mean positions, with
    SEL_K_MAX + 1 for splits that did not pick a benchmark, over the splits
    of --reps (default all) that picked anything, in split_id order; only
    the holdouts SEL_HOLDOUTS gives the method, if it lists any."""
    groups = OrderedDict()
    holdouts = SEL_HOLDOUTS.get(method)
    for i, e in entries:
        if not nums.want(e["dataset"]):
            continue
        if nums.reps is not None and e["rep"] not in nums.reps:
            continue
        if holdouts is not None and not any(
                abs(float(e["holdout_frac"]) - h) < 1e-9 for h in holdouts):
            continue
        key = (e["dataset"], float(e["holdout_frac"]))
        groups.setdefault(key, []).append((i, e))
    note = nums.reps_note()
    for (ds, p), items in groups.items():
        nums.use_dataset(ds)
        head = f"Sel{camel(ds)}{method}P{num_words(100 * p)}"
        where = f"[dataset={ds}, holdout_frac={p:g}]"
        items = sorted(items, key=lambda it: it[1]["split_id"])
        n_empty = sum(1 for _, e in items if not e["selected_names"])
        items = [(i, e) for i, e in items if e["selected_names"]]
        n_folds = len(items)
        orders = [[str(n) for n in e["selected_names"]][:SEL_K_MAX]
                  for _, e in items]
        unit = "splits"

        def add(name, value, kind, how, src_key=where, on_clash="skip"):
            nums.add(head + name, value, kind, path, src_key, "", how + note,
                     on_clash)

        add("NFolds", n_folds, "int", f"number of {unit} with picks")
        add("NSplits", n_folds, "int", f"number of {unit} with picks")
        add("NReps", len({e["rep"] for _, e in items}), "int",
            "CV repetitions among them")
        add("NEmpty", n_empty, "int", f"{unit} without picks, left out")
        for (i, e), order in zip(items, orders):
            name = f"Fold{int_words(e['fold']).capitalize()}"
            if e["rep"]:
                name = f"Rep{int_words(e['rep']).capitalize()}" + name
            add(name, ", ".join(order), "text",
                "selection order, comma-joined", f"[{i}].selected_names")
        ties = [sum(1 for t in e["index_ties"] if t[0] < SEL_K_MAX)
                for _, e in items if e.get("index_ties") is not None]
        if ties:
            add("IndexTies", sum(ties), "int",
                f"picks among the first {SEL_K_MAX} decided by the lowest "
                f"index, summed over {len(ties)} {unit}")

        firsts = Counter(order[0] for order in orders if order)
        if firsts:
            name, count = firsts.most_common(1)[0]
            add("FirstName", name, "text",
                "most frequent first pick (ties: earliest split)")
            add("FirstFolds", count, "int",
                f"{unit} (of {n_folds}) starting with FirstName")
        add("FirstDistinct", len(firsts), "int", "distinct first picks")
        common = 0
        if orders and all(orders):
            for picks in zip(*orders):
                if len(set(picks)) > 1:
                    break
                common += 1
        add("CommonPrefix", common, "int",
            "leading picks identical (same order) in every split")

        positions = OrderedDict()                 # name -> 1-based positions
        for order in orders:
            for pos, name in enumerate(order, 1):
                positions.setdefault(name, []).append(pos)
        stats = []
        for name, ranks in positions.items():
            all_ranks = ranks + [SEL_K_MAX + 1] * (n_folds - len(ranks))
            stats.append((name, len(ranks), float(np.mean(all_ranks))))
        stats.sort(key=lambda s: s[2])            # stable, as in the plots
        add("NDistinct", len(stats), "int", "benchmarks picked in any split")
        add("NAllFolds", sum(s[1] == n_folds for s in stats), "int",
            "benchmarks picked in every split")
        lead = 0
        while lead < len(stats) and stats[lead][1] == n_folds:
            lead += 1
        add("NLeadAllFolds", lead, "int",
            "leading figure rows (by mean rank) picked in every split")
        # plot_selection_order.min_row_count
        min_rows = max(2, math.ceil(SEL_MIN_ROW_SHARE * n_folds - 1e-9))
        rows = [s for s in stats if s[1] >= min_rows][:SEL_ROWS]
        for r, (name, _, _) in enumerate(rows, 1):
            add(f"RowName{int_words(r).capitalize()}", name, "text",
                f"row {r} of the figure (picked in >= {min_rows} splits, "
                f"by mean rank)")
        for name, count, mean_rank in stats:
            B = camel(name) or "Unnamed"
            add(f"BenchFolds{B}", count, "int",
                f"{unit} (of {n_folds}) picking {name}", on_clash="rename")
            add(f"BenchMeanRank{B}", mean_rank, "num1",
                f"mean position of {name} over {n_folds} {unit} "
                f"({SEL_K_MAX + 1} where not picked)", on_clash="rename")
            if firsts.get(name):
                add(f"BenchFirst{B}", firsts[name], "int",
                    f"{unit} (of {n_folds}) starting with {name}",
                    on_clash="rename")


# ── Paired differences and split design ──────────────────────────

def _table_names(table):
    """(tag, {Ds}, tail) of a paired_stats.SplitTable's macro names."""
    tail = (f"P{int_words(round(100 * table.holdout_frac))}"
            if table.per_holdout else "")
    return table.tag, camel(table.dataset), tail


def _table_unit(table):
    return "folds" if table.legacy else "splits"


def _add_paired(nums, tables):
    """\\Pair{S}{T}{Ds}{A}Vs{B}K{k}[P{p}] for the MACRO_PAIRS of every
    table at K_PAIRED (paired_stats.compare over the table's rows); tables
    marked CSV-only (t.macros False: their scripts write their own paired
    macros) are skipped."""
    note = nums.reps_note()
    for t in tables:
        if not t.macros:
            continue
        nums.use_dataset(t.dataset)
        tag, D, tail = _table_names(t)
        unit = _table_unit(t)
        for a, b in paired_stats.MACRO_PAIRS:
            if a not in t.r2 or b not in t.r2:
                continue
            (file_a, key_a), (file_b, key_b) = t.keys[a], t.keys[b]
            path = t.path.parent / file_a
            src_key = f"{key_a} - {key_b}"
            if file_b != file_a:
                src_key = f"{key_a} - {file_b}:{key_b}"
            for k in K_PAIRED:
                if (k > min(t.r2[a].shape[1], t.r2[b].shape[1])
                        if t.ks is None else t.column(k) is None):
                    continue
                st = paired_stats.compare(t, a, b, k)
                col = k - 1 if t.ks is None else t.column(k)
                name = (f"{tag}{D}{camel(a)}Vs{camel(b)}K{int_words(k)}"
                        f"{tail}")
                how = (f"paired over {st['n_splits']}/{t.n_rows} {unit}"
                       f"{note}")
                for suffix, (field, kind) in PAIR_STATS.items():
                    desc = {"Diff": "mean of a - b",
                            "Se": "Nadeau-Bengio corrected se",
                            "Lo": "95% t-interval low",
                            "Hi": "95% t-interval high",
                            "Win": "fraction a > b",
                            "Wins": "splits a > b",
                            "N": "splits both finite"}[suffix]
                    nums.add(f"Pair{suffix}{name}", st[field], kind, path,
                             src_key, f"[:, {col}]", f"{how}; {desc}")


def _add_design(nums, tables):
    """\\Split{T}{Ds}[P{p}]{F} (split counts, eligibility) and
    \\Ties{T}{Ds}{M}[P{p}] / \\TiesSplits... (index-decided picks at the
    largest logged k) of every matched-design table."""
    note = nums.reps_note()
    for t in tables:
        if t.legacy or not t.macros:
            continue
        nums.use_dataset(t.dataset)
        tag, D, tail = _table_names(t)
        path = t.path
        head = f"Split{tag}{D}{tail}"
        meta = t.meta
        nums.add(head + "NSplits", t.n_rows, "int", path, "split_id", "",
                 f"number of split rows{note}")
        nums.add(head + "NReps", int(np.unique(meta["rep"]).size), "int",
                 path, "rep", "", f"distinct reps{note}")
        n_el = meta["n_eligible"]
        if t.n_rows and np.isfinite(n_el).all():
            nums.add(head + "EligMin", int(n_el.min()), "int", path,
                     "n_eligible", "", f"min over splits{note}")
            nums.add(head + "EligMax", int(n_el.max()), "int", path,
                     "n_eligible", "", f"max over splits{note}")
            nums.add(head + "EligMean", float(n_el.mean()), "num1", path,
                     "n_eligible", "", f"mean over splits{note}")
            if t.n_benchmarks:
                nums.add(head + "NRestricted",
                         int(np.sum(n_el < t.n_benchmarks)), "int", path,
                         "n_eligible", "", f"splits with n_eligible < "
                         f"{t.n_benchmarks} benchmarks{note}")
        if t.failed is not None:
            nums.add(head + "NFailed", int(np.sum(t.failed)), "int", path,
                     "failed", "", f"splits whose estimation failed{note}")
        for m, ties in t.ties.items():
            if ties is None or ties.ndim != 2 or not ties.shape[1]:
                continue
            last = np.maximum(ties[:, -1], 0)        # -1: failed split
            file_m, key_m = t.keys[m]
            src = key_m[:-len("r2")] + "index_ties" if key_m.endswith("r2") \
                else key_m
            name = f"{tag}{D}{camel(m)}{tail}"
            nums.add("Ties" + name, int(last.sum()), "int",
                     t.path.parent / file_m, src, f"[:, {ties.shape[1] - 1}]",
                     f"sum over {t.n_rows} splits{note}")
            nums.add("TiesSplits" + name, int(np.count_nonzero(last)), "int",
                     t.path.parent / file_m, src, f"[:, {ties.shape[1] - 1}]",
                     f"splits with any (of {t.n_rows}){note}")


# ── Generic JSON flattening ──────────────────────────────────────

def _is_num(v):
    return v is None or isinstance(v, (bool, int, float))


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _list_id_field(items):
    """A field whose string values name every dict in the list uniquely."""
    if not all(isinstance(it, dict) for it in items):
        return None
    for field in ID_FIELDS:
        vals = [it.get(field) for it in items]
        if all(isinstance(v, str) and camel(v) for v in vals) \
                and len({camel(v) for v in vals}) == len(vals):
            return field
    return None


def _is_param(v):
    """Parameter-like value: a string, an integer or a float with at most
    3 decimals (0.1, 0.25), as opposed to a computed result."""
    if isinstance(v, str):
        return bool(camel(v))
    if _is_int(v):
        return True
    return isinstance(v, float) and math.isfinite(v) and round(v, 3) == v


def _param_name(item, fields):
    """'FracZeropointoneDeltaZeropointtwo' for frac=0.1, delta=0.2."""
    return "".join(camel(f) + (camel(item[f]) if isinstance(item[f], str)
                               else num_words(item[f]).capitalize())
                   for f in fields)


def _list_param_fields(items):
    """The fewest leading fields (at most MAX_NAME_FIELDS, same order in
    every dict) whose parameter-like values name every dict uniquely,
    e.g. ['frac', 'delta'] for a contamination grid; None if none do."""
    if not all(isinstance(it, dict) and it for it in items):
        return None
    lead = list(items[0])[:MAX_NAME_FIELDS]
    for j in range(1, len(lead) + 1):
        fields = lead[:j]
        if not all(list(it)[:j] == fields and _is_param(it[fields[-1]])
                   for it in items):
            return None
        if len({_param_name(it, fields) for it in items}) == len(items):
            return fields
    return None


def _add_int_text(nums, lists, name, path, jpath, nested):
    """One exact text macro for integer lists ('41, 50, 15', nested lists
    joined by '; '); a sha256 digest if that exceeds MAX_TEXT_LEN."""
    text = "; ".join(", ".join(str(u) for u in v) for v in lists)
    how = "exact values" + ("; lists joined by '; '" if nested else "")
    if len(text) > MAX_TEXT_LEN:
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        text = f"sha256 {digest} of {sum(len(v) for v in lists)} values"
        how = f"sha256 of the exact values ({how})"
    nums.add(name, text, "text", path, jpath, "", how, "rename")


def _add_summary(nums, x, name, path, jpath, full, shape=None):
    """{name}ListMean (+ ListStd/ListMin/ListMax/ListCount if full) of a
    numeric JSON list; the 'List' infix keeps these from clashing with
    explicit fields such as r2 next to r2_mean."""
    mean, std, n = _nan_stats(x)
    desc = f"{n}/{x.size} values" + (f", shape {shape}" if shape else "")
    nums.add(name + "ListMean", mean, "float", path, jpath, "[...]",
             f"nanmean over {desc}", "rename")
    if full:
        finite = x[~np.isnan(x)]
        nums.add(name + "ListStd", std, "std", path, jpath, "[...]",
                 f"nanstd(ddof=0) over {desc}", "rename")
        nums.add(name + "ListMin", finite.min() if n else np.nan, "float",
                 path, jpath, "[...]", f"nanmin over {desc}", "rename")
        nums.add(name + "ListMax", finite.max() if n else np.nan, "float",
                 path, jpath, "[...]", f"nanmax over {desc}", "rename")
        nums.add(name + "ListCount", x.size, "int", path, jpath, "[...]",
                 "number of values", "rename")


def _flatten(nums, obj, name, path, jpath):
    """Emit every leaf of a JSON value; `name` is the macro built so far.
    Robust to unknown schemas: name clashes are renamed, never dropped."""
    if isinstance(obj, bool):
        nums.add(name, int(obj), "int", path, jpath, "", "bool", "rename")
    elif isinstance(obj, int):
        nums.add(name, obj, "int", path, jpath, "", "value", "rename")
    elif obj is None or isinstance(obj, float):
        v = np.nan if obj is None else obj
        nums.add(name, v, "std" if name.endswith("Std") else "float", path,
                 jpath, "", "value", "rename")
        if re.search(r"Frac(?:tion)?$", name):
            nums.add(name + "Pct", 100.0 * v, "float", path, jpath, "",
                     "100 * value", "rename")
    elif isinstance(obj, str):
        if len(obj) <= MAX_TEXT_LEN:
            nums.add(name, obj, "text", path, jpath, "", "text", "rename")
        else:
            nums.warn(f"{nums.rel(path)}:{jpath}: string longer than "
                      f"{MAX_TEXT_LEN} chars skipped")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{jpath}.{k}" if jpath else str(k)
            _flatten(nums, v, name + (camel(k) or "Unnamed"), path, sub)
    elif isinstance(obj, list):
        _flatten_list(nums, obj, name, path, jpath)
    else:
        nums.warn(f"{nums.rel(path)}:{jpath}: unsupported {type(obj)}")


def _flatten_list(nums, items, name, path, jpath):
    if not items:
        return
    short = len(items) <= MAX_LIST_ITEMS
    if all(_is_int(v) for v in items):
        # indices, folds, seeds, k grids: exact values, a mean is meaningless
        _add_int_text(nums, [items], name, path, jpath, nested=False)
        return
    if all(_is_num(v) for v in items):
        x = np.array([np.nan if v is None else float(v) for v in items])
        if short:
            for i, v in enumerate(items):
                _flatten(nums, v, f"{name}Item{int_words(i).capitalize()}",
                         path, f"{jpath}[{i}]")
        _add_summary(nums, x, name, path, jpath, full=not short)
        return
    if all(isinstance(v, str) for v in items):
        # e.g. benchmark-name lists: one comma-joined text macro
        _flatten(nums, ", ".join(items), name, path, jpath)
        if short:
            for i, v in enumerate(items):
                _flatten(nums, v, f"{name}Item{int_words(i).capitalize()}",
                         path, f"{jpath}[{i}]")
        return
    if all(isinstance(v, list) and v and all(_is_int(u) for u in v)
           for v in items):
        # per-run selections, (step, n_candidates) pairs, ...: exact values
        _add_int_text(nums, items, name, path, jpath, nested=True)
        return
    if all(isinstance(v, list) and v and all(_is_num(u) for u in v)
           for v in items):
        # numeric list of lists, possibly ragged (e.g. per-fold gains)
        x = np.array([np.nan if u is None else float(u)
                      for v in items for u in v])
        lengths = {len(v) for v in items}
        shape = ((len(items), lengths.pop()) if len(lengths) == 1
                 else f"ragged, {len(items)} lists")
        _add_summary(nums, x, name, path, jpath, full=True, shape=shape)
        return
    try:
        x = np.array(items, dtype=float)       # rectangular numeric nest
    except (TypeError, ValueError):
        x = None
    if x is not None and x.ndim >= 2:
        _add_summary(nums, x.ravel(), name, path, jpath, full=True,
                     shape=x.shape)
        return
    field = _list_id_field(items)
    params = None if field else _list_param_fields(items)
    for i, item in enumerate(items):
        if field:
            rest = {k: v for k, v in item.items() if k != field}
            _flatten(nums, rest, name + camel(item[field]), path,
                     f"{jpath}[{i}]")
        elif params:
            rest = {k: v for k, v in item.items() if k not in params}
            _flatten(nums, rest, name + _param_name(item, params), path,
                     f"{jpath}[{i}]")
        else:
            _flatten(nums, item, f"{name}Item{int_words(i).capitalize()}",
                     path, f"{jpath}[{i}]")


# ── Flat "macros" dicts of the experiment JSON logs ──────────────

_STD_NAME = re.compile(r"(?:Std|Se)(?=[A-Z]|$)")


def _macro_kind(name, value):
    """(kind, value) of one entry of a flat JSON "macros" dict, or None if
    the value is not a scalar: JSON integers and booleans are counts (int),
    null is n/a, a float whose name has a Std or Se component (RsqStd...,
    PairSe...) is a spread (std digits), any other float a real value."""
    if isinstance(value, bool):
        return "int", int(value)
    if isinstance(value, int):
        return "int", value
    if value is None:
        return "float", math.nan
    if isinstance(value, float):
        return ("std" if _STD_NAME.search(name) else "float"), value
    if isinstance(value, str):
        return "text", value
    return None


def _add_json_macros(nums, path, obj, script):
    """Every entry of a JSON log's flat "macros" dict under its own name:
    the scripts build letters-only names in this module's scheme (camel,
    int_words, num_words).  A name already taken (first source wins), not
    letters-only or with a non-scalar value is reported and skipped."""
    macros = obj.get("macros") if isinstance(obj, dict) else None
    if not isinstance(macros, dict):
        nums.warn(f"{nums.rel(path)}: no 'macros' dict; skipped")
        return
    how = f"macros dict of {script}"
    bad = []
    for name, value in macros.items():
        kv = _macro_kind(str(name), value)
        if not re.fullmatch(r"[A-Za-z]+", str(name)) or kv is None \
                or (kv[0] == "text" and len(kv[1]) > MAX_TEXT_LEN):
            bad.append(str(name))
            continue
        nums.add(str(name), kv[1], kv[0], path, f"macros.{name}", "", how)
    if bad:
        nums.warn(f"{nums.rel(path)}: {len(bad)} macros entries skipped "
                  f"(name not letters-only or value not a scalar), e.g. "
                  f"{', '.join(bad[:3])}")


def _add_normality(nums, path, summary):
    if not isinstance(summary, dict):
        nums.warn(f"{nums.rel(path)}: expected a dict of datasets; skipped")
        return
    for ds, stats in summary.items():
        nums.use_dataset(ds)
        D = camel(ds)
        if not isinstance(stats, dict):
            _flatten(nums, stats, f"Norm{D}", path, str(ds))
            continue
        for field, v in stats.items():
            if field == "worst_benchmarks" and isinstance(v, list):
                for r, item in enumerate(v, 1):
                    if not isinstance(item, dict):
                        continue
                    jp = f"{ds}.worst_benchmarks[{r - 1}]"
                    if "W" in item:
                        nums.add(f"Norm{D}WorstW{int_words(r)}", item["W"],
                                 "float", path, jp + ".W", "",
                                 f"rank {r} smallest Shapiro-Wilk W")
                    if "name" in item:
                        nums.add(f"Norm{D}WorstName{int_words(r)}",
                                 item["name"], "text", path, jp + ".name",
                                 "", f"rank {r} smallest Shapiro-Wilk W")
                continue
            _flatten(nums, v, f"Norm{D}{camel(field)}", path, f"{ds}.{field}")


# ── Eigenspectrum, EM and data shapes ────────────────────────────

def _corr_eig_k(Sigma):
    """k reaching each EIG_LEVELS fraction of variance, computed exactly
    as plot_eigenspectrum.py does (correlation of Sigma, eigvalsh)."""
    diag_std = np.maximum(np.sqrt(np.diag(Sigma)), 1e-12)
    D_inv = 1.0 / diag_std
    R = Sigma * np.outer(D_inv, D_inv)
    eigvals = np.linalg.eigvalsh(R)[::-1]
    cum_var = np.cumsum(eigvals) / eigvals.sum()
    return [int(np.searchsorted(cum_var, t)) + 1 for t in EIG_LEVELS]


def _em_sigma(matrix_path):
    """Sigma of a matrix by the estimate_covariance.py recipe (EM)."""
    from em_cov import em_cov
    with np.load(matrix_path, allow_pickle=True) as d:
        B = d["B"].astype(np.float64)
        O = d["O"].astype(np.float64)
    M, N = B.shape
    col_count = np.maximum(O.sum(axis=0), 1.0)
    col_mean = (B * O).sum(axis=0) / col_count
    B_c = O * (B - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))
    B_std = O * (B - col_mean[None, :]) / col_std[None, :]
    obs_frac = int(O.sum()) / B.size
    regularized = M < N or obs_frac < 0.5
    result = em_cov(B_std, O, verbose=False,
                    max_iter=1000 if regularized else 500,
                    tol=5e-4 if regularized else 1e-6,
                    shrinkage="auto" if M < N else 0.0,
                    eps_psd=1e-3 if regularized else 1e-6)
    return result["Sigma"]


def _cov_files(cov_dirs):
    """{ds: path} of the first {ds}.cov.npz in the search order."""
    found = OrderedDict()
    for d in cov_dirs:
        for p in _glob(d, "*.cov.npz"):
            found.setdefault(p.name[:-len(".cov.npz")], p)
    return found


def _add_eigen(nums, cov_files, cov_dirs, data_root, mode):
    if mode == "skip":
        return
    datasets = list(EIG_DATASETS)
    datasets += [ds for ds in cov_files if ds not in datasets]
    for ds in filter(nums.want, datasets):
        nums.use_dataset(ds)
        cov = cov_files.get(ds)
        if mode in ("auto", "cov") and cov is not None:
            sig = _read_npz(nums, cov)
            if sig is None or "Sigma" not in sig:
                continue
            Sigma, src, skey = sig["Sigma"], cov, "Sigma"
            how = "from cov file"
        elif mode in ("auto", "recompute"):
            mat = data_root / f"{ds}.matrix.npz"
            if not mat.is_file():
                nums.warn(f"eigenspectrum {ds}: no cov file and no {mat}; "
                          f"skipped")
                continue
            if mode == "auto":
                nums.warn(f"eigenspectrum {ds}: no {ds}.cov.npz in "
                          f"{[str(d) for d in cov_dirs]}; recomputing from "
                          f"{mat} (pass --cov-dir to use the run's own)")
            Sigma, src, skey = _em_sigma(mat), mat, "B/O"
            how = "recomputed by em_cov (estimate_covariance.py recipe)"
        else:
            nums.warn(f"eigenspectrum {ds}: no {ds}.cov.npz; skipped")
            continue
        for t, k in zip(EIG_LEVELS, _corr_eig_k(Sigma)):
            nums.add(f"EigK{camel(ds)}Var{int_words(round(100 * t))}", k,
                     "int", src, skey, "",
                     f"{how}; corr(Sigma) eigvalsh; smallest k with "
                     f"cumvar >= {t:.2f}")


def _add_em(nums, path, ds):
    """EM iteration count and convergence flag stored in a cov file."""
    with np.load(path, allow_pickle=True) as d:
        if "n_iter" in d.files:
            nums.add(f"EmIter{camel(ds)}", int(d["n_iter"]), "int", path,
                     "n_iter", "", "EM iterations (estimate_covariance.py)")
        if "converged" in d.files:
            nums.add(f"EmConverged{camel(ds)}", int(bool(d["converged"])),
                     "int", path, "converged", "", "bool")


_EIG_LINE = re.compile(r"^\s*([A-Za-z][\w-]*): (\d+% at k=\d+.*)$")


def _add_logit_eigen(nums, path):
    """EigKLogit{Ds}Var{v} from the 'Eigenspectrum (logit space)' block
    that eval_logit.py prints ('  MMLU: 90% at k=2, 95% at k=8')."""
    lines = pathlib.Path(path).read_text().splitlines()
    start = next((i for i, line in enumerate(lines)
                  if "Eigenspectrum (logit space)" in line), None)
    if start is None:
        nums.warn(f"{nums.rel(path)}: no 'Eigenspectrum (logit space)' "
                  f"block; logit eigenspectrum skipped")
        return
    found = False
    for i in range(start + 1, len(lines)):
        m = _EIG_LINE.match(lines[i])
        if not m:
            if found:
                break                             # end of the block
            if lines[i].startswith("#") or not lines[i].strip():
                continue                          # banner before the block
            break
        found = True
        nums.use_dataset(m.group(1).lower())
        D = camel(m.group(1).lower())
        for pct, k in re.findall(r"(\d+)% at k=(\d+)", m.group(2)):
            nums.add(f"EigKLogit{D}Var{int_words(int(pct))}", int(k), "int",
                     path, f"line {i + 1}", "",
                     f"printed by eval_logit.py; smallest k with cumvar >= "
                     f"{int(pct) / 100:.2f}, logit space")
    if not found:
        nums.warn(f"{nums.rel(path)}: empty logit eigenspectrum block")


def _add_data(nums, data_root):
    paths = _glob(data_root, "*.matrix.npz")
    if not paths:
        nums.warn(f"no *.matrix.npz in {data_root}; dataset shapes skipped")
    for path in paths:
        ds = path.name[:-len(".matrix.npz")]
        if not nums.want(ds):
            continue
        nums.use_dataset(ds)
        try:
            with np.load(path, allow_pickle=True) as d:
                O = d["O"].astype(np.float64)
        except Exception as e:
            nums.warn(f"cannot read {path}: {e}; skipped")
            continue
        D = camel(ds)
        obs = O.sum() / O.size
        nums.add(f"Data{D}M", O.shape[0], "int", path, "O", ".shape[0]",
                 "models")
        nums.add(f"Data{D}N", O.shape[1], "int", path, "O", ".shape[1]",
                 "benchmarks")
        nums.add(f"Data{D}ObsPct", 100.0 * obs, "num1", path, "O", "",
                 "100 * mean(O)")
        nums.add(f"Data{D}MissPct", 100.0 * (1.0 - obs), "num1", path, "O",
                 "", "100 * (1 - mean(O))")


# ── Extraction driver ────────────────────────────────────────────

def _add_selection_log(nums, path, entries, method_camel, method=None,
                       default_ds=None, skip_ds=()):
    """Sel macros of one selection log: the entries of `method` (None: all,
    a rep-0 log), leaving out datasets in skip_ds."""
    sel = [(i, e) for i, e in _sel_entries(entries, method, default_ds)
           if e["dataset"] not in skip_ds]
    _add_selections(nums, path, sel, method_camel)


def extract(root, cov_dir=None, data_dir=None, eigen="auto", reps=None,
            datasets=None):
    """Extract all quantities under `root`; returns a Numbers instance
    whose .records maps macro name -> record dict (CSV_FIELDS + section)
    and whose .tables holds the paired_stats split tables.  reps restricts
    per-split quantities to those CV repetitions, datasets the extraction
    to those dataset names (None: all)."""
    nums = Numbers(root, reps, datasets)
    logs = nums.root / "logs"
    logit = nums.root / "logs_logit"
    data_root = (pathlib.Path(data_dir).expanduser().resolve()
                 if data_dir else default_data_dir())
    if not nums.root.is_dir():
        nums.warn(f"root {nums.root} does not exist")
    for sub, names in expected_files(nums.want).items():
        for name in names:
            if not (nums.root / sub / name).is_file():
                nums.warn(f"missing {sub}/{name}; skipped")

    def files(directory, pattern, prefix, suffix=".npz", exclude=()):
        """[(path, ds)] of the matches of the wanted datasets."""
        out = []
        for p in _glob(directory, pattern):
            if any(p.name.startswith(e) for e in exclude):
                continue
            if directory != logit and p.stem.endswith("_logit"):
                continue
            ds = _stem_ds(p, prefix, suffix)
            if ds and nums.want(ds):
                out.append((p, ds))
        return out

    # {ds}_results.npz of eval_benchpress.py (any dataset name)
    results = []
    for p, ds in files(logs, "*_results.npz", "", "_results.npz"):
        d = _read_npz(nums, p)
        if d is not None and any(k.startswith(("emi_", "cv_")) for k in d):
            results.append((p, ds, d))

    nums.section = "R^2, entropy vs MI vs random (entropy_vs_mi_*, " \
                   "{ds}_results emi_*)"
    for p, ds in files(logs, "entropy_vs_mi_*.npz", "entropy_vs_mi_"):
        d = _read_npz(nums, p)
        if d is not None:
            nums.use_dataset(ds)
            _guarded(nums, p, _add_methods, p, d, ds, aux=True)
    for p, ds, d in results:
        nums.use_dataset(ds)
        _guarded(nums, p, _add_methods, p, d, ds, prefix="emi_", aux=True)

    nums.section = "R^2, greedy entropy CV by holdout fraction " \
                   "(greedy_cv_*, {ds}_results cv_*)"
    for p, ds in files(logs, "greedy_cv_*.npz", "greedy_cv_"):
        d = _read_npz(nums, p)
        if d is not None:
            nums.use_dataset(ds)
            _guarded(nums, p, _add_greedy_cv, p, d, ds)
    for p, ds, d in results:
        nums.use_dataset(ds)
        _guarded(nums, p, _add_greedy_cv, p, d, ds)

    nums.section = "R^2, baselines (baselines_*)"
    for p, ds in files(logs, "baselines_*.npz", "baselines_"):
        d = _read_npz(nums, p)
        if d is not None:
            nums.use_dataset(ds)
            _guarded(nums, p, _add_methods, p, d, ds, tag="Base")

    nums.section = "R^2, TabImpute imputation (tabimpute_*, tabimpute_extra_*)"
    for p, ds in (files(logs, "tabimpute_*.npz", "tabimpute_",
                        exclude=("tabimpute_extra_",))
                  + files(logs, "tabimpute_extra_*.npz", "tabimpute_extra_")):
        d = _read_npz(nums, p)
        if d is not None:
            nums.use_dataset(ds)
            _guarded(nums, p, _add_methods, p, d, ds, tag="Tab")

    nums.section = "R^2, logit score space (logs_logit/)"
    for p, ds in files(logit, "entropy_vs_mi_*_logit.npz", "entropy_vs_mi_",
                       "_logit.npz"):
        d = _read_npz(nums, p)
        if d is not None:
            nums.use_dataset(ds)
            _guarded(nums, p, _add_methods, p, d, ds, tag="Logit", aux=True)
    for p, ds in files(logit, "greedy_cv_*_logit.npz", "greedy_cv_",
                       "_logit.npz"):
        d = _read_npz(nums, p)
        if d is not None:
            nums.use_dataset(ds)
            _guarded(nums, p, _add_greedy_cv, p, d, ds, tag="Logit")

    tables = paired_stats.load_tables(nums.root, nums.datasets, nums.reps,
                                      data_root, warn=nums.warn)
    nums.tables = tables
    nums.section = "Paired R^2 differences over the matched splits " \
                   "(paired_stats.py)"
    for t in tables:
        _guarded(nums, t.path, _add_paired, [t])

    nums.section = "Matched design: splits, eligibility, index ties"
    for t in tables:
        _guarded(nums, t.path, _add_design, [t])

    nums.section = "Selection order per split (greedy_cv_splits_*, " \
                   "greedy_mi_cv_splits_*; else greedy_cv_selections.json, " \
                   "greedy_mi_cv_selections_all.json)"
    covered = {}          # method camel -> datasets with a per-split log
    for prefix, methods in SEL_SPLIT_LOGS:
        for p, ds in files(logs, f"{prefix}*.json", prefix, ".json"):
            entries = _read_json(nums, p)
            if not isinstance(entries, list):
                if entries is not None:
                    nums.warn(f"{nums.rel(p)}: expected a list; skipped")
                continue
            for method, M in methods.items():
                if any(isinstance(e, dict) and e.get("method") == method
                       for e in entries):
                    covered.setdefault(M, set()).add(ds)
                    _guarded(nums, p, _add_selection_log, p, entries, M,
                             method, ds)
    for name, M in SEL_FILES:
        p = logs / name
        if p.is_file():
            entries = _read_json(nums, p)
            if entries is not None:
                _guarded(nums, p, _add_selection_log, p, entries, M,
                         skip_ds=covered.get(M, set()))

    nums.section = "Model-family holdout (family_holdout_*)"
    for p, ds in files(logs, "family_holdout_*.npz", "family_holdout_"):
        d = _read_npz(nums, p)
        if d is not None:
            nums.use_dataset(ds)
            _guarded(nums, p, _add_family, p, d, ds)

    nums.section = "Cost-budgeted selection (cost_budget_*)"
    for p, ds in files(logs, "cost_budget_*.npz", "cost_budget_"):
        d = _read_npz(nums, p)
        if d is not None:
            nums.use_dataset(ds)
            _guarded(nums, p, _add_cost, p, d, ds)

    nums.section = "Normality diagnostics (normality_summary.json)"
    p = logs / "normality_summary.json"
    if p.is_file():
        summary = _read_json(nums, p)
        if isinstance(summary, dict) and nums.datasets is not None:
            summary = {ds: v for ds, v in summary.items() if nums.want(ds)}
        if summary is not None:
            _guarded(nums, p, _add_normality, p, summary)

    nums.section = "Reviewer checks (reviewer_checks*.json)"
    nums.group = "main"
    for p in _glob(logs, "reviewer_checks*.json"):
        obj = _read_json(nums, p)
        if obj is not None:
            tag = camel(p.stem[len("reviewer_checks"):])
            _guarded(nums, p, _flatten, obj, "Rev" + tag, p, "")

    nums.section = ("Columns redundant by construction "
                    "(baselines_*_redundancy.json macros)")
    for p in _glob(logs, f"baselines_*{REDUNDANCY_SUFFIX}"):
        ds = p.name[len("baselines_"):-len(REDUNDANCY_SUFFIX)]
        if not ds or not nums.want(ds):
            continue
        obj = _read_json(nums, p)
        if obj is not None:
            nums.use_dataset(ds)
            _guarded(nums, p, _add_json_macros, p, obj, "eval_baselines.py")

    # The reviewer-request experiments: their own macro groups
    for prefix, group, script in JSON_MACRO_LOGS:
        nums.section = f"{script} (logs/{prefix}*.json macros)"
        for p in _glob(logs, f"{prefix}*.json"):
            rest = p.stem[len(prefix):]
            if prefix == "prospective_":       # setting tag -> its data
                rest = paired_stats.PROSPECTIVE_DATA.get(rest, rest)
            if not rest or not nums.want_stem(rest):
                continue
            obj = _read_json(nums, p)
            if obj is not None:
                nums.group = group
                _guarded(nums, p, _add_json_macros, p, obj, script)

    nums.section = "Eigenspectrum (variance explained by top-k components)"
    if cov_dir:
        cov_dirs = [pathlib.Path(cov_dir).expanduser().resolve()]
        if not cov_dirs[0].is_dir():
            nums.warn(f"--cov-dir {cov_dirs[0]} does not exist")
    else:
        cov_dirs = [nums.root / "cov", nums.root / "data", nums.root]
        if os.environ.get("BENCHSELECT_COV_DIR"):
            cov_dirs.append(pathlib.Path(os.environ["BENCHSELECT_COV_DIR"])
                            .expanduser().resolve())
    cov_dirs = [d for d in cov_dirs if d.is_dir()]
    cov_files = _cov_files(cov_dirs)
    _guarded(nums, cov_dirs[0] if cov_dirs else nums.root, _add_eigen,
             cov_files, cov_dirs, data_root, eigen)
    # logit.txt: the paper's DATASETS; logit_extra.txt: run_all.sh's
    # logit_extra job over DATASETS_EXTRA
    for name in LOGIT_STDOUT:
        p = nums.root / "stdout" / name
        if p.is_file():
            _guarded(nums, p, _add_logit_eigen, p)

    nums.section = "EM convergence (cov files)"
    for ds, p in cov_files.items():
        if nums.want(ds):
            nums.use_dataset(ds)
            _guarded(nums, p, _add_em, p, ds)

    nums.section = "Dataset shapes (data/*.matrix.npz)"
    _guarded(nums, data_root, _add_data, data_root)
    return nums


# ── Output ───────────────────────────────────────────────────────

def write_tex(nums, path, digits=2, std_digits=None, thousands=THOUSANDS,
              groups=None):
    """\\newcommand file of the records in `groups` (None: every record);
    returns the number of macros written."""
    records = [r for r in nums.records.values()
               if groups is None or r.get("group", "main") in groups]
    what = ("" if groups is None
            else f" of group{'s' if len(groups) > 1 else ''} "
                 f"{', '.join(groups)}")
    lines = ["% Generated by code/make_numbers.py; do not edit by hand.",
             f"% root: {nums.root_arg}",
             f"% {len(records)} macros{what}; full precision and provenance "
             f"in the companion CSV.",
             "% Naming scheme: see the make_numbers.py module docstring.",
             "% 'partial': a mean or std over only some of its folds or "
             "families (the others are NaN)."]
    section = None
    for r in records:
        if r["section"] != section:
            section = r["section"]
            lines += ["", f"% --- {section} ---"]
        body = format_value(r["value"], r["kind"], digits, std_digits,
                            thousands)
        where = f"{r['source_file']}:{r['source_key']}{r['index']}"
        if is_partial(r):
            where += f"; partial: {r['aggregation']}"
        lines.append(f"\\newcommand{{\\{r['key']}}}{{{body}}}% {where}")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(records)


def group_tex_path(tex_dir, group):
    """DIR/numbers_{group}.tex of a non-main macro group."""
    return pathlib.Path(tex_dir) / f"numbers_{group}.tex"


def write_group_tex(nums, tex_dir, digits=2, std_digits=None,
                    thousands=THOUSANDS):
    """DIR/numbers_{group}.tex for every TEX_GROUPS entry but 'main', also
    when a group has no records (a fixed set of files); returns {group:
    (path, n_macros)}."""
    out = {}
    for group in TEX_GROUPS[1:]:
        path = group_tex_path(tex_dir, group)
        out[group] = (path, write_tex(nums, path, digits, std_digits,
                                      thousands, groups=[group]))
    return out


def write_csv(nums, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in nums.records.values():
            w.writerow([r["key"], csv_value(r["value"], r["kind"]),
                        r["source_file"], r["source_key"], r["index"],
                        r["aggregation"], r["kind"], r["scope"]])


def _infer_kind(raw):
    """Kind of a CSV value without a kind column: numeric only if raw is
    exactly what csv_value writes for that number (so text such as a hash
    '0123...' or '12e4...' stays text)."""
    try:
        v = float(raw)
    except ValueError:
        return "text"
    if raw == repr(v):
        return "float"
    return "int" if re.fullmatch(r"-?[1-9]\d*|0", raw) else "text"


def read_csv(path):
    """Records from a CSV written by write_csv.  The kind column decides
    how each value is parsed; CSVs written before it existed fall back to
    _infer_kind and are marked legacy (their std/num formats are unknown)."""
    records = OrderedDict()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row["value"]
            legacy = row.get("kind") not in KINDS
            kind = _infer_kind(raw) if legacy else row["kind"]
            if kind == "text":
                value = raw
            elif kind == "int":
                value = int(raw)
            else:
                value = float(raw)
            records[row["key"]] = dict(row, value=value, kind=kind, raw=raw,
                                       scope=row.get("scope") or "",
                                       legacy=legacy)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="experiment root containing logs/ and logs_logit/")
    parser.add_argument("--out-tex", help="LaTeX macro file to write")
    parser.add_argument("--out-csv", help="full-precision CSV to write")
    parser.add_argument("--cov-dir", help="directory with {ds}.cov.npz "
                        "(default: ROOT/cov, ROOT/data, ROOT, "
                        "$BENCHSELECT_COV_DIR)")
    parser.add_argument("--data-dir", help="directory with {ds}.matrix.npz "
                        "(default: path_config.data_dir())")
    parser.add_argument("--eigen", default="auto",
                        choices=["auto", "cov", "recompute", "skip"],
                        help="eigenspectrum source: cov file, else recompute "
                        "(auto); cov only; always recompute; skip")
    parser.add_argument("--digits", type=int, default=2,
                        help="decimals for real-valued macros")
    parser.add_argument("--std-digits", type=int, default=None,
                        help="decimals for standard deviations "
                        "(default: digits + 1)")
    parser.add_argument("--thousands", default=THOUSANDS,
                        help="separator grouping integer parts of 4+ digits "
                        "(default '{,}'; '' for none)")
    parser.add_argument("--reps", nargs="+", type=int,
                        help="restrict per-split quantities to these CV "
                        "repetitions (default: all rows; 0 = the "
                        "single-repetition protocol)")
    parser.add_argument("--datasets", nargs="+",
                        help="extract only these dataset names (default: "
                        "every dataset with logs)")
    parser.add_argument("--paired-dir", help="also write DIR/paired_{ds}.csv "
                        "(paired_stats.py, every method pair and k)")
    parser.add_argument("--tex-dir", help="split the macros by group: "
                        "--out-tex (default DIR/numbers.tex) gets group "
                        "'main' (the paper's datasets) only, "
                        "DIR/numbers_{group}.tex every other group "
                        f"({', '.join(TEX_GROUPS[1:])}; written even when "
                        "empty).  Without it --out-tex gets every macro")
    args = parser.parse_args(argv)
    if not (args.out_tex or args.out_csv or args.paired_dir or args.tex_dir):
        parser.error("give --out-tex, --out-csv, --paired-dir and/or "
                     "--tex-dir")
    if args.reps is not None and min(args.reps) < 0:
        parser.error("--reps must be non-negative")
    if args.tex_dir and not args.out_tex:
        args.out_tex = str(pathlib.Path(args.tex_dir) / "numbers.tex")

    nums = extract(args.root, cov_dir=args.cov_dir, data_dir=args.data_dir,
                   eigen=args.eigen, reps=args.reps, datasets=args.datasets)
    if not any(from_results(r) for r in nums.records.values()):
        print(f"make_numbers.py: error: nothing extracted from {nums.root}'s "
              f"own {', '.join(d + '/' for d in RESULT_DIRS)}; wrong --root?",
              file=sys.stderr)
        sys.exit(2)
    tex_counts = OrderedDict()
    if args.out_tex:
        tex_counts[args.out_tex] = write_tex(
            nums, args.out_tex, args.digits, args.std_digits, args.thousands,
            groups=["main"] if args.tex_dir else None)
    if args.tex_dir:
        for group, (path, n) in write_group_tex(
                nums, args.tex_dir, args.digits, args.std_digits,
                args.thousands).items():
            tex_counts[str(path)] = n
    for path, n in tex_counts.items():
        if n > TEX_MACRO_WARN:
            nums.warn(f"{path} defines {n} macros (more than "
                      f"{TEX_MACRO_WARN}): TeX may run out of string pool "
                      f"or hash space"
                      + ("" if args.tex_dir else "; split with --tex-dir"))
    if args.out_csv:
        write_csv(nums, args.out_csv)
    paired = {}
    if args.paired_dir:
        paired = paired_stats.write_dataset_csvs(nums.tables, args.paired_dir)

    counts = OrderedDict()
    for r in nums.records.values():
        counts[r["section"]] = counts.get(r["section"], 0) + 1
    print(f"{len(nums.records)} quantities from {nums.root}:")
    for section, n in counts.items():
        print(f"  {n:5d}  {section}")
    by_group = Counter(r.get("group", "main") for r in nums.records.values())
    print("by macro group: " + ", ".join(f"{g} {by_group[g]}"
                                         for g in TEX_GROUPS if by_group[g]))
    partial = [r["key"] for r in nums.records.values() if is_partial(r)]
    if partial:
        print(f"{len(partial)} means/stds cover only some folds or families "
              f"(marked 'partial' in the .tex), e.g. {', '.join(partial[:3])}")
    shared = [r["key"] for r in nums.records.values()
              if r["scope"] == "shared"]
    if shared:
        print(f"{len(shared)} quantities read from outside the root "
              f"(scope 'shared'), e.g. {', '.join(shared[:3])}")
    for path, n in tex_counts.items():
        print(f"wrote {path} ({n} macros)")
    if args.out_csv:
        print(f"wrote {args.out_csv} ({len(nums.records)} rows)")
    for ds, (path, n) in paired.items():
        print(f"wrote {path} ({n} paired rows)")
    if nums.warnings:
        print(f"{len(nums.warnings)} warning(s), see stderr")


if __name__ == "__main__":
    main()
