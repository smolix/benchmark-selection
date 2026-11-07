#!/usr/bin/env bash
# Regenerate the CPU experiments of the paper and of its data release: every
# log, every CPU figure (including figures/benchmark_overlap.pdf) and the
# paper-number macros (make_numbers.py), including the matrix-completion
# imputers (eval_imputers.py, as per-repetition shards).  Not covered: the
# TabImpute appendix, which needs a GPU (see run_tabimpute.sh).
#
# Usage (from any directory; paths are resolved relative to this script):
#
#   BENCHSELECT_EXPERIMENT_ROOT=/scratch/benchselect-run \
#   PYTHON=~/venvs/benchselect/bin/python \
#       bash code/run_all.sh [--repeats N] [--dry-run] [--list]
#
#   --repeats N, -r N   CV repetitions of the matched design (default 5 =
#                   cv_split.N_REPEATS: 5 x 10 = 50 splits per dataset and
#                   holdout); passed as --repeats to every split-based
#                   script.  Repetition 0 is the original seed-42 10-fold
#                   split, so --repeats 1 reproduces the single-repetition
#                   runs and the rep-0 rows of any N equal them.
#   --dry-run, -n   print the environment and every command; run nothing
#   --list, -l      list jobs, groups, dependencies and expected outputs
#
#   DATASETS="mmlu merged"   datasets of the per-dataset jobs (space or comma
#                   separated; default "mmlu mteb merged benchpress", the
#                   paper's): any NAME with a NAME.matrix.npz in the data
#                   directory.  See "Datasets".
#   DATASETS_EXTRA="olv2sub epoch"   datasets that get the standard battery
#                   only (default "olv2sub olv1items epoch helm_lite
#                   helm_mmlu helm_classic mteb2026", the data release;
#                   empty or "none": no extended battery).  See "Datasets".
#   RUN_ONLY="baselines family"   run only these jobs or job groups (space or
#                   comma separated, see --list; a job also answers to the
#                   tags in parentheses there: extra = the extended battery,
#                   datafix = baselines on merged_v1).  A selected job whose
#                   producer is not selected reads that producer's outputs
#                   from an earlier run in the same root, provided the
#                   producer's last run there succeeded with the same
#                   command (same --repeats, DATASETS, DATASETS_EXTRA,
#                   WORKERS and interpreter; a mismatch skips the job
#                   instead of mixing designs).  RUN_ONLY=tests runs the
#                   unit tests, which are not in the default set;
#                   RUN_ONLY=imputers the matrix-completion imputers alone.
#   MAX_JOBS=N      at most N cores busy (default: the online CPUs, counted
#                   without OMP_NUM_THREADS / OMP_THREAD_LIMIT, which GNU
#                   nproc obeys); a job with worker processes counts as
#                   that many
#   WORKERS=N       worker processes of the jobs that parallelize over
#                   splits (--workers; estimator_<ds>: --jobs), each
#                   single-threaded: the slow jobs of the standard battery
#                   (greedy_cv, entropy_vs_mi, mi_selection and baselines
#                   on BS_PARALLEL_BATTERY = epoch merged merged_v1
#                   benchpress mteb2026 mteb helm_classic, benchpress,
#                   logit_<ds> on BS_PARALLEL_LOGIT = merged benchpress
#                   mteb2026 mteb helm_classic), reviewer_checks,
#                   analyze_objectives[_fulloracle], stability_<ds> and
#                   estimator_<ds> (BS_PARALLEL_DATASETS = merged
#                   benchpress mteb), the imputer shards
#                   imputers_<ds>_rep<r> (at most 10 processes: a shard
#                   holds the 10 splits of one repetition),
#                   benchfam_{merged,mteb}[_sub] and, over control splits,
#                   prospective_{epoch,epoch_bymodel,mteb2026}; the short
#                   jobs keep one process (a pool takes a few seconds to
#                   spawn).  --list shows the cores of each job (CPU).
#                   Default min(16, MAX_JOBS / 4), at least 1.  Results
#                   do not depend on it.
#   PYTHON          interpreter (default python3); checked against the lock
#                   requirements.txt, override with BENCHSELECT_ALLOW_UNPINNED=1
#   BENCHSELECT_GIT_COMMIT, BENCHSELECT_GIT_DIRTY, BENCHSELECT_GIT_DIFF_SHA256
#                   provenance to record when the tree is a copy, not a git
#                   work tree (recorded unchecked; code_sha256 in the
#                   manifest is the authoritative record of the code)
#
# Running on a server from a copy of the work tree: leave out macOS
# AppleDouble files and xattrs (COPYFILE_DISABLE=1, --no-xattrs; the hashes
# skip dotfiles anyway) and include the requirements files, which the
# interpreter check needs:
#
#   COPYFILE_DISABLE=1 tar --no-xattrs -cf - requirements*.txt code/*.py \
#       code/*.sh data/*.npz data/*.json data/SHA256SUMS [data/*.csv] |
#       ssh HOST 'mkdir -p DIR && tar -xf - -C DIR'
#   (data/*.npz, not only data/*.matrix.npz: data/*.items.npz holds the OLL
#   v1 item scores of item_budget and data/helm.npz the HELM Lite v1.0.0
#   snapshot that prospective_helm reads; data/*.json includes the meta
#   files and data/score_bounds.json; data/SHA256SUMS: analyze_objectives.py
#   checks the matrices against it; data/*.csv, the raw collection tables,
#   only feed the cross-check that plot_benchmark_overlap.py prints and
#   writes to its summary JSON.  data/.env is matched by none of these.)
#   BENCHSELECT_GIT_COMMIT=$(git rev-parse HEAD)
#   BENCHSELECT_GIT_DIFF_SHA256=$(git diff HEAD --binary | shasum -a 256 | cut -c1-64)
#   BENCHSELECT_GIT_DIRTY=1 if `git status --porcelain -- code 'requirements*.txt'`
#       prints anything (untracked scripts count: they are not in the diff)
#
# Outputs, all under $BENCHSELECT_EXPERIMENT_ROOT (required: the driver refuses
# to run without it, and refuses roots inside the repository's data/, logs*/,
# paper/ or code/, also when reached through '..' or a symlink):
#   logs/ logs_logit/ figures/ figures_logit/   same file names as the paper
#   cov/              EM covariance estimates (BENCHSELECT_COV_DIR)
#   numbers/          numbers.tex (LaTeX macros of the paper's DATASETS),
#                     numbers_<group>.tex (the data-release datasets and one
#                     file per experiment: prospective, item_budget, ...;
#                     make_numbers.py TEX_GROUPS) and numbers.csv (every
#                     macro, full precision) from make_numbers.py, and
#                     paired_<ds>.csv (paired method differences over the
#                     matched splits); RUN_ONLY=numbers after
#                     run_tabimpute.sh adds those numbers
#   stdout/<job>.txt  stdout + stderr of each job
#   manifest.json     provenance of the latest invocation: git state, host,
#                     Python and package versions, sha256 of the input data
#                     (data/*.npz, *.json, *.csv, SHA256SUMS) and code, the
#                     design
#                     parameters (repeats, datasets), timestamps, per-job
#                     exit codes and durations,
#                     and the sha256 and origin of every artifact: the job of
#                     this invocation that wrote it, or the earlier run (with
#                     its code tree hash) whose archived manifest records it,
#                     or "unknown" (archived in manifests/)
#
# Every job runs single-threaded (OMP / OpenBLAS / MKL / vecLib = 1) with
# MPLBACKEND=Agg, PYTHONHASHSEED=0 and a fixed SOURCE_DATE_EPOCH (PDF
# metadata), so independent jobs run concurrently without oversubscription;
# the caller's other PYTHON* variables (PYTHONOPTIMIZE strips asserts) are
# cleared.  A job's expected outputs are deleted before it starts; it fails
# if it exits non-zero, leaves an expected output missing, or its stdout
# reports a failure that the script caught and survived (status log_errors,
# e.g. "estimation failed" for one fold).
#
# Exit status: 0 every job ok; 1 a job failed or was skipped, the code or
# input data changed during the run, or something in the repository changed
# (its top-level files, data/, logs/, logs_logit/, paper/ or code/); 2 usage
# or setup error, nothing run; 3 the interpreter does not match the lock
# file, nothing run.
#
# Matched design.  Every split-based job runs with --repeats N on the splits
# of cv_split.cv_splits: N x 10-fold CV per (dataset, holdout), repetition 0
# = cv_folds(M, 10, 42) with the seed-42 training subsamples, split_id =
# rep * 10 + fold.  Within a split every method of every script sees the same
# train / validation models, the same eligible set (benchmarks observed for
# >= MIN_TRAIN_OBS = 3 training models), the same imputation targets
# (protocol rule D1: benchmarks observed for >= MIN_TARGET_OBS = 2 training
# models; the cells of the others are never scored) and the same common
# random numbers (Split.seed), so the per-split rows of all logs pair on
# split_id.  Every imputer's predictions are clipped to the benchmark's
# valid score range before scoring (rule D2, code/score_bounds.py, audit
# data/score_bounds.json); entropy_rf was dropped (rule D3).  The paired
# R² differences (mean, Nadeau-Bengio corrected SE, 95% t-interval, win
# rate) of the methods of one log go to
# the scripts' own *_paired.csv files and, for every log (paired_stats.py),
# to numbers/paired_<ds>.csv and the \Pair... macros of numbers.tex.
# run_tabimpute.sh runs rep 0 only by default (split_id 0-9 of these runs).
#
# Datasets.  Jobs named <job>_<ds> run once per dataset ds in DATASETS
# (datasets share no state: folds and seeds are per dataset, so a per-dataset
# job reproduces a joint run exactly).  benchpress goes through
# eval_benchpress.py (logs/benchpress_results.npz, from which make_numbers.py
# reads the BenchPress numbers) instead of eval_greedy_all.py /
# eval_entropy_vs_mi.py / plot_mi_selection_all.py; every other name,
# including new ones, through the latter.  cost_<ds> and cost_fixed_<ds> run
# for the datasets in DATASETS with a cost table (mmlu, mteb).  The
# combining jobs (greedy_cv_selections, mi_cv_selections, selection_order,
# logit, reviewer_checks, analyze_objectives, stability_all, imputers_all)
# cover exactly the DATASETS of the invocation that ran them, so add a
# dataset to a root by rerunning with the full list.  Per dataset of
# DATASETS also: stability_<ds>, estimator_<ds>, imputers_<ds>_rep<r> (one
# shard per repetition r) and imputers_<ds> (their merge);
# analyze_objectives_fulloracle and baselines_merged_v1 (pre-fix Merged)
# with merged; benchfam_<ds> for the datasets of DATASETS and
# DATASETS_EXTRA with a family scheme (mmlu mteb merged olv2sub helm_mmlu
# olv1items), benchfam_<ds>_sub and benchfam_blindcov for mteb, merged.
#
# DATASETS_EXTRA (the data release) get the standard battery: greedy_cv,
# entropy_vs_mi, mi_selection, baselines, family and logit (not on epoch:
# the logit transform does not apply there, see eval_logit.py) per
# dataset, plus the combining jobs selection_order_extra, logit_extra
# (figures_logit/extra/eigenspectrum_logit.pdf, stdout/logit_extra.txt) and
# eigenspectrum_release (figures/eigenspectrum_<names>.pdf), all also
# selected by RUN_ONLY=extra; the paper's combining files stay those of
# DATASETS.  numbers.tex holds the macros of DATASETS (make_numbers.py
# group main); those of DATASETS_EXTRA and merged_v1 go to
# numbers/numbers_datasets.tex, each experiment's to
# numbers/numbers_<experiment>.tex.
#
# Fixed jobs, independent of DATASETS: covariance and eigenspectrum (mmlu
# mteb merged, the paper figure), normality (mmlu mteb merged),
# benchmark_overlap (data/canonical_mapping.json), and the experiments on
# their own data: prospective_<setting> (mteb2026 helm epoch olv2sub, the
# sensitivity settings olv2sub_math epoch_bymodel) and prospective_plot,
# item_budget and item_fixed (olv1items), token_costs, token_costs_exclude
# and token_fixed (helm_lite helm_mmlu helm_classic).
#
# Commands.  The canonical full run (paper + data release + imputers, 5 x
# 10 matched splits, 138 jobs; --list shows them, their cores and BS_EST):
#
#   BENCHSELECT_EXPERIMENT_ROOT=ROOT PYTHON=~/venvs/benchselect/bin/python \
#       bash code/run_all.sh --repeats 5
#
# On a 64-thread 4090 host (EPYC 7502) at MAX_JOBS 60, WORKERS 16 it takes
# 66 min wall (4090-0, 2026-10-20), the jobs holding 61 of the 66
# core-hours of the 60 cores (cores x wall time); the longest are
# estimator_merged (29 min), estimator_benchpress (18 min), token_fixed
# (15 min, one process), stability_benchpress (15 min) and the five
# imputers_merged shards (11-13 min each, 10 processes).  With the
# battery, the logit runs, reviewer_checks, analyze_objectives and the
# imputer shards in one process each it took 1 h 37 min (4090-1,
# 2026-10-20), bounded by the imputers_merged shards (~95 min each); the
# logs, figures and macros of the two runs are bit-identical apart from
# runtimes, worker counts and source hashes.
#
# The paper alone (no extended battery): DATASETS_EXTRA=none in front of
# the same command.  RUN_ONLY="imputers numbers" reruns only the imputers
# (shards, merges, imputers_all) and the numbers into an existing ROOT.
#
# Job graph (A -> B: B reads A's outputs; all other jobs are independent):
#   covariance -> eigenspectrum     cov/*.cov.npz -> figures/eigenspectrum.pdf
#   greedy_cv_<ds> -> greedy_cv_selections, selection_order
#                   logs/greedy_cv_splits_<ds>.json -> logs/greedy_cv_selections.json
#                   (rep-0 entropy, historical format) and
#                   figures/selection_order_<ds>.pdf (entropy),
#                   figures/selection_order_{facloc,trace}_<ds>.pdf (all splits)
#   mi_selection_<ds> -> mi_cv_selections
#                   logs/greedy_mi_cv_splits_<ds>.json
#                   -> logs/greedy_mi_cv_selections_all.json
#   greedy_cv_<extra ds> -> selection_order_extra
#   stability_<ds> -> stability_all        figures/selection_stability_all.pdf
#   prospective_{mteb2026,helm,epoch,olv2sub} -> prospective_plot
#                                          figures/prospective.pdf
#   benchfam_{mteb,merged}{,_sub} -> benchfam_blindcov
#                   logs/benchfamily_holdout_<ds>_blindcov.{json,_paired.csv}
#   imputers_<ds>_rep<r> -> imputers_<ds> -> imputers_all
#                   logs/imputers_shards/imputers_<ds>_rep<r>.npz ->
#                   logs/imputers_<ds>.{npz,json} -> figures/imputers_all.pdf
#   entropy_vs_mi_<ds>, benchpress, logit_<ds>, logit, baselines_<ds>,
#   family_<ds>, cost_<ds>, reviewer_checks, normality, benchmark_overlap,
#   the experiment jobs
#   every job above -> numbers      numbers/numbers{,_<group>}.tex,
#                                   numbers/numbers.csv,
#                                   numbers/paired_<ds>.csv
# eval_logit.py runs per dataset (logit_<ds>, --no-eigenspectrum) plus once
# for its eigenspectrum figure, which combines all DATASETS (job logit,
# --eigenspectrum-only; make_numbers.py reads the logit-space k90/k95 from
# its stdout/logit.txt);
# eval_greedy_all.py and plot_mi_selection_all.py run per dataset with
# --selections-json none, then once with --selections-from-logs, which
# writes the joint run's selection file.  Both splits reproduce a joint run
# exactly.  Jobs launch longest first (BS_EST, the measured wall time at
# --repeats 5; see --list), as their dependencies and the free cores
# allow; a multi-core job that does not fit yet is held, and the jobs
# after it start only where they do not delay it (EASY backfilling on
# BS_EST).
#
# Scripts deliberately not run:
#   eval_greedy_mmlu.py, eval_greedy_merged.py   single-dataset predecessors
#       of eval_greedy_all.py: same greedy_cv_{mmlu,merged}.pdf (they would
#       race with it), no logs (their imputation is the shared one).
#   plot_mi_selection_mmlu.py   MMLU-only predecessor of plot_mi_selection_all.py:
#       same selection_order_mi_mmlu.pdf; its greedy_mi_cv_selections.json is
#       the MMLU subset of greedy_mi_cv_selections_all.json and unused.
#   run_greedy_mmlu.py          figures/greedy_mmlu.pdf, not in paper.tex.
#   compare_runs.py             diffs the numbers of two roots; run it by hand
#       (e.g. the repository against a new root).
#   eval_tabimpute.py, test_tabimpute_sanity.py   GPU, see run_tabimpute.sh.
#   em_cov.py, normality.py, model_families.py    __main__ is a self-test/demo.
#   data/*.py   download_*.py fetch the scores, build_matrices.py (with
#       load_data.py) and build_<name>.py build *.matrix.npz; build_cost_tables.py
#       builds mmlu_test_counts.json and
#       mteb_task_sizes.json in its own venv (requirements-data.txt).  Their
#       outputs are this driver's inputs (hashed in the manifest).
# Not regenerated by anything, and used by neither paper.tex nor the
# rebuttal: logs/greedy_mi_cv_selections.json (plot_mi_selection_mmlu.py) and
# paper/figures/greedy_mmlu.pdf (run_greedy_mmlu.py).  Delete them when the
# repository's logs/ and paper/figures/ are replaced with a root's outputs.

set -euo pipefail

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
    echo "run_all.sh: needs bash >= 4.4 (this is $BASH_VERSION)" >&2
    exit 2
fi

# ══════════════════════════════════════════════════════════════════
# Shared helpers (run_tabimpute.sh sources this file for them)
# ══════════════════════════════════════════════════════════════════

BS_CODE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
BS_REPO=$(dirname "$BS_CODE_DIR")
# The paper's datasets (default of DATASETS); bs_init_datasets sets
# BS_DATASETS from DATASETS.
BS_DEFAULT_DATASETS=(mmlu mteb merged benchpress)
BS_DATASETS=("${BS_DEFAULT_DATASETS[@]}")
# The data-release datasets (default of DATASETS_EXTRA, run_all.sh only):
# the standard battery runs on them too (bs_init_extra_datasets).
BS_DEFAULT_EXTRA=(olv2sub olv1items epoch helm_lite helm_mmlu helm_classic
                  mteb2026)
BS_EXTRA=()
BS_DRIVER=${BS_DRIVER:-run_all}

declare -a BS_JOBS=() BS_SELECTED=() BS_RUNNING=()
declare -a BS_ENV_SET=() BS_ENV_UNSET=() BS_MANIFEST_FLAGS=()
declare -A BS_GROUP=() BS_DEPS=() BS_OUTS=() BS_CMD=() BS_OPTIN=()
declare -A BS_INPUTS=() BS_NOSCAN=() BS_TAGS=() BS_WEIGHT=() BS_EST=()
declare -A BS_IS_SEL=() BS_PID=() BS_RESULT=() BS_T0=()
# EASY backfilling state of bs_run_selected (bs_shadow)
BS_SHADOW=0
BS_SPARE=0
BS_DRY_RUN=0
BS_LIST=0
BS_INTERRUPTED=0
BS_ARGS=""
# --repeats: accepted only by drivers that set BS_ACCEPT_REPEATS=1 before
# bs_parse_args (run_all.sh; run_tabimpute.sh takes
# BENCHSELECT_TABIMPUTE_REPEATS).  Default = cv_split.N_REPEATS.
BS_ACCEPT_REPEATS=0
BS_REPEATS=5

# How the eval scripts report a failure they catch and survive (one fold's
# EM estimate, the Mardia test, a malformed log in make_numbers.py): a job
# whose stdout matches is not ok even though it exited 0.  Extended regex,
# also used by the manifest helper (Python re) to quote the lines.  The
# experiment scripts' summary "estimation failed on 0 splits" is not a
# failure (any other count, "on every split" and "estimation failed: ..."
# are).
BS_LOG_ERROR_RE='estimation failed($|[^ ]| [^o]| o[^n]| on [^0]| on 0[0-9])|extraction failed|EM failed|test failed:| (—|--) failed: |Fold [0-9]+: failed \(|cannot read .*; skipped|Traceback \(most recent call last\)'

bs_die() {
    printf '%s: error: %s\n' "$BS_DRIVER" "$*" >&2
    exit 2
}

bs_log() {
    printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

bs_now() {
    # Epoch seconds with sub-second resolution where bash provides it.
    local t=${EPOCHREALTIME:-}
    if [[ -n $t ]]; then printf '%s' "${t/,/.}"; else date +%s; fi
}

bs_usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

bs_parse_args() {
    local arg val
    BS_ARGS="$*"
    while (( $# )); do
        arg=$1
        shift
        case $arg in
            -n|--dry-run) BS_DRY_RUN=1 ;;
            -l|--list) BS_LIST=1 ;;
            -h|--help) bs_usage; exit 0 ;;
            -r|--repeats|--repeats=*)
                (( BS_ACCEPT_REPEATS )) || bs_die "unknown argument '$arg' (see --help)"
                if [[ $arg == --repeats=* ]]; then
                    val=${arg#--repeats=}
                else
                    (( $# )) || bs_die "$arg needs a value"
                    val=$1
                    shift
                fi
                [[ $val =~ ^[1-9][0-9]*$ ]] || bs_die "--repeats must be a positive integer, got '$val'"
                BS_REPEATS=$val ;;
            *) bs_die "unknown argument '$arg' (see --help)" ;;
        esac
    done
}

# DATASETS (space or comma separated) -> BS_DATASETS: names with a
# NAME.matrix.npz in the data directory, each at most once.
bs_init_datasets() {
    local spec=${DATASETS:-${BS_DEFAULT_DATASETS[*]}} ds avail
    local -a toks=()
    local -A seen=()
    read -r -a toks <<<"${spec//,/ }" || true
    (( ${#toks[@]} )) || bs_die "DATASETS is empty"
    BS_DATASETS=()
    for ds in "${toks[@]}"; do
        [[ $ds =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
            bs_die "DATASETS: '$ds' is not a dataset name (letters, digits, '_', '.', '-')"
        [[ -z ${seen[$ds]:-} ]] || bs_die "DATASETS lists '$ds' twice"
        seen[$ds]=1
        if [[ ! -f $BS_DATA_DIR/$ds.matrix.npz ]]; then
            avail=$(cd "$BS_DATA_DIR" 2>/dev/null && for f in *.matrix.npz; do
                        [[ -f $f ]] && printf '%s ' "${f%.matrix.npz}"; done) || true
            bs_die "DATASETS: no $BS_DATA_DIR/$ds.matrix.npz (available: ${avail% })"
        fi
        BS_DATASETS+=("$ds")
    done
}

# DATASETS_EXTRA (unset: BS_DEFAULT_EXTRA; empty or 'none': no extended
# battery) -> BS_EXTRA, without the names already in BS_DATASETS.
bs_init_extra_datasets() {
    local spec ds avail main=" ${BS_DATASETS[*]} "
    local -a toks=()
    local -A seen=()
    spec=${DATASETS_EXTRA-${BS_DEFAULT_EXTRA[*]}}
    [[ ${spec//[ ,]/} == none ]] && spec=""
    read -r -a toks <<<"${spec//,/ }" || true
    BS_EXTRA=()
    for ds in "${toks[@]}"; do
        [[ $ds =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
            bs_die "DATASETS_EXTRA: '$ds' is not a dataset name (letters, digits, '_', '.', '-')"
        [[ -z ${seen[$ds]:-} ]] || bs_die "DATASETS_EXTRA lists '$ds' twice"
        seen[$ds]=1
        [[ $ds != benchpress ]] ||
            bs_die "DATASETS_EXTRA: benchpress runs through eval_benchpress.py; list it in DATASETS"
        if [[ $main == *" $ds "* ]]; then
            bs_log "note: $ds is in DATASETS; left out of DATASETS_EXTRA"
            continue
        fi
        if [[ ! -f $BS_DATA_DIR/$ds.matrix.npz ]]; then
            avail=$(cd "$BS_DATA_DIR" 2>/dev/null && for f in *.matrix.npz; do
                        [[ -f $f ]] && printf '%s ' "${f%.matrix.npz}"; done) || true
            bs_die "DATASETS_EXTRA: no $BS_DATA_DIR/$ds.matrix.npz (available: ${avail% }; DATASETS_EXTRA=none skips the extended battery)"
        fi
        BS_EXTRA+=("$ds")
    done
}

bs_expand_home() {
    local p=$1
    if [[ $p == "~" ]]; then p=$HOME
    elif [[ $p == "~/"* ]]; then p=$HOME/${p#"~/"}
    fi
    printf '%s' "$p"
}

# Interpreter and data directory (needed to define jobs, even for --list).
bs_init_paths() {
    local py
    py=$(bs_expand_home "${PYTHON:-python3}")
    if [[ $py != */* ]]; then
        py=$(command -v "$py" 2>/dev/null || printf '%s' "$py")
    elif [[ $py != /* ]]; then
        py=$PWD/$py
    fi
    BS_PYTHON=$py
    # Same default as path_config.data_dir()
    BS_DATA_DIR=$(bs_expand_home "${BENCHSELECT_DATA_DIR:-$BS_REPO/data}")
    [[ $BS_DATA_DIR == /* ]] || BS_DATA_DIR=$PWD/$BS_DATA_DIR
    BS_MAX_JOBS=${MAX_JOBS:-$(bs_cpu_count)}
    [[ $BS_MAX_JOBS =~ ^[1-9][0-9]*$ ]] || bs_die "MAX_JOBS must be a positive integer, got '$BS_MAX_JOBS'"
}

# Online CPUs.  GNU nproc honours OMP_NUM_THREADS and OMP_THREAD_LIMIT
# (OMP_NUM_THREADS=1 nproc prints 1), and the caller of a single-threaded
# run usually exports exactly that, so count without them; getconf where
# nproc is missing (macOS).
bs_cpu_count() {
    local n=""
    n=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null) ||
        n=$(getconf _NPROCESSORS_ONLN 2>/dev/null) || n=""
    [[ $n =~ ^[1-9][0-9]*$ ]] || n=4
    printf '%s' "$n"
}

# Cores a job occupies (BS_WEIGHT: jobs running worker processes of their
# own), capped at BS_MAX_JOBS so that every job fits when nothing else
# runs; the result is in BS_W.
bs_weight() {
    BS_W=${BS_WEIGHT[$1]:-1}
    (( BS_W <= BS_MAX_JOBS )) || BS_W=$BS_MAX_JOBS
}

bs_check_root_location() {
    local root=$1 repo=$BS_REPO
    case "$root/" in
        "$repo/" | "$repo/data/"* | "$repo/logs/"* | "$repo/logs_logit/"* | \
        "$repo/paper/"* | "$repo/code/"*)
            bs_die "BENCHSELECT_EXPERIMENT_ROOT=$root would write into the repository's checked-in data, logs, figures or code; choose a scratch directory" ;;
    esac
}

# Physical form of the absolute path $1, which need not exist: '..' and
# symlinks are resolved component by component, as the kernel would, so
# that the location check sees where mkdir would create the directory.
bs_physical_path() {
    local cur=/ comp next phys
    local -a comps=()
    IFS=/ read -r -a comps <<<"$1" || true
    for comp in "${comps[@]}"; do
        case $comp in
            "" | .) continue ;;
            ..) cur=$(dirname -- "$cur") ;;   # cur is physical or new
            *)
                next=${cur%/}/$comp
                if [[ -d $next ]] && phys=$(cd -- "$next" 2>/dev/null && pwd -P); then
                    cur=$phys
                else
                    cur=$next
                fi ;;
        esac
    done
    printf '%s' "$cur"
}

# Validate BENCHSELECT_EXPERIMENT_ROOT; create it unless this is a dry run.
bs_init_root() {
    local root=${BENCHSELECT_EXPERIMENT_ROOT:-}
    [[ -n $root ]] || bs_die "BENCHSELECT_EXPERIMENT_ROOT is not set.  Every log and figure is written under it; refusing to fall back to the repository's logs/ and paper/figures/."
    root=$(bs_expand_home "$root")
    [[ $root == /* ]] || root=$PWD/$root
    while [[ $root == */ && $root != / ]]; do root=${root%/}; done
    bs_check_root_location "$root"
    # Check the resolved location before creating anything.
    root=$(bs_physical_path "$root")
    bs_check_root_location "$root"
    if (( ! BS_DRY_RUN )); then
        mkdir -p -- "$root"
    fi
    BS_ROOT=$root
    BS_RUN_ID="$BS_DRIVER-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    BS_STATE_DIR=$BS_ROOT/.state/$BS_RUN_ID
}

bs_setenv() {
    export "$1=$2"
    BS_ENV_SET+=("$1")
}

bs_unsetenv() {
    unset "$1"
    BS_ENV_UNSET+=("$1")
}

# Environment shared by every job: outputs redirected under the root,
# single-threaded BLAS, deterministic hashing and PDF metadata.
bs_export_env() {
    local v
    for v in BENCHSELECT_LOGS_DIR BENCHSELECT_LOGIT_LOGS_DIR BENCHSELECT_FIGURES_DIR \
             BENCHSELECT_LOGIT_FIGURES_DIR BENCHSELECT_COV_DIR; do
        if [[ -n ${!v:-} ]]; then
            bs_log "note: ignoring $v=${!v}; all outputs go under $BS_ROOT"
        fi
    done
    bs_setenv BENCHSELECT_EXPERIMENT_ROOT "$BS_ROOT"
    bs_setenv BENCHSELECT_DATA_DIR "$BS_DATA_DIR"
    bs_setenv BENCHSELECT_LOGS_DIR "$BS_ROOT/logs"
    bs_setenv BENCHSELECT_LOGIT_LOGS_DIR "$BS_ROOT/logs_logit"
    bs_setenv BENCHSELECT_FIGURES_DIR "$BS_ROOT/figures"
    bs_setenv BENCHSELECT_LOGIT_FIGURES_DIR "$BS_ROOT/figures_logit"
    bs_setenv BENCHSELECT_COV_DIR "$BS_ROOT/cov"
    # Per-job overrides only (eval_logit / eval_tabimpute read them)
    bs_unsetenv BENCHSELECT_DATASETS
    bs_unsetenv BENCHSELECT_METHODS
    bs_unsetenv PYTHONPATH
    # The caller's other interpreter settings (PYTHONOPTIMIZE strips the
    # asserts, PYTHONWARNINGS, PYTHONSAFEPATH, ...); the driver's own are
    # set below.  PYTHON itself only names the interpreter.
    for v in $(compgen -e PYTHON); do
        if [[ $v != PYTHON && $v != PYTHONPATH ]]; then bs_unsetenv "$v"; fi
    done
    for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
             VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS; do
        bs_setenv "$v" 1
    done
    bs_setenv MPLBACKEND Agg
    bs_setenv PYTHONHASHSEED 0
    bs_setenv PYTHONDONTWRITEBYTECODE 1
    bs_setenv PYTHONNOUSERSITE 1
    bs_setenv PYTHONUNBUFFERED 1
    bs_setenv SOURCE_DATE_EPOCH "${SOURCE_DATE_EPOCH:-0}"
}

# bs_job NAME GROUP "DEPS" "OUTPUTS" ARGV...
#   DEPS: jobs whose outputs this job reads.  OUTPUTS: files relative to the
#   root that must exist (non-empty) afterwards.  ARGV runs in code/.
#   Optional: BS_INPUTS[NAME] = the files it actually reads (default: all
#   of its dependencies' outputs), BS_OPTIN[NAME] = not in the default set
#   (and, when not selected, not waited for by the jobs that list it as a
#   dependency: they read its outputs if present), BS_NOSCAN[NAME] = do not
#   scan its stdout for caught failures, BS_TAGS[NAME] = further group
#   names RUN_ONLY selects it by (space separated), BS_WEIGHT[NAME] = cores
#   it occupies (its own worker processes; default 1), BS_EST[NAME] =
#   estimated seconds at --repeats 5 (launch order: longest first; EASY
#   backfilling, bs_shadow).
bs_job() {
    local name=$1 group=$2 deps=$3 outs=$4 q
    shift 4
    # e.g. a dataset whose per-dataset job name equals a combining job's
    [[ -z ${BS_GROUP[$name]+x} ]] || bs_die "job '$name' is defined twice (dataset name clash?)"
    printf -v q '%q ' "$@"
    BS_JOBS+=("$name")
    BS_GROUP[$name]=$group
    BS_DEPS[$name]=$deps
    BS_OUTS[$name]=$outs
    BS_CMD[$name]=${q% }
}

# bs_each PATTERN DS...  ->  PATTERN formatted with each DS, space separated
bs_each() {
    local pat=$1 ds item
    local -a out=()
    shift
    for ds in "$@"; do
        printf -v item "$pat" "$ds"
        out+=("$item")
    done
    printf '%s' "${out[*]}"
}

# Resolve RUN_ONLY into BS_SELECTED (definition order = launch order).
bs_select() {
    local filter=${RUN_ONLY:-} tok name found
    local -a toks=()
    local -A want=()
    BS_SELECTED=()
    read -r -a toks <<<"${filter//,/ }" || true
    if (( ${#toks[@]} == 0 )); then
        for name in "${BS_JOBS[@]}"; do
            if [[ -z ${BS_OPTIN[$name]:-} ]]; then want[$name]=1; fi
        done
    else
        for tok in "${toks[@]}"; do
            found=0
            for name in "${BS_JOBS[@]}"; do
                if [[ $tok == "$name" || $tok == "${BS_GROUP[$name]}" ||
                      " ${BS_TAGS[$name]:-} " == *" $tok "* ]]; then
                    want[$name]=1
                    found=1
                fi
            done
            (( found )) || bs_die "RUN_ONLY: unknown job or group '$tok' (see --list)"
        done
    fi
    for name in "${BS_JOBS[@]}"; do
        if [[ -n ${want[$name]:-} ]]; then
            BS_SELECTED+=("$name")
            BS_IS_SEL[$name]=1
        fi
    done
    (( ${#BS_SELECTED[@]} )) || bs_die "no jobs selected"
    bs_sort_selected
}

# Launch order: longest estimated jobs first (BS_EST), definition order
# among equal estimates (jobs without one keep their place after them).
bs_sort_selected() {
    local name i=0
    local -a keyed=()
    for name in "${BS_SELECTED[@]}"; do
        printf -v name '%012d %06d %s' $(( 999999999999 - ${BS_EST[$name]:-0} )) \
            "$i" "$name"
        keyed+=("$name")
        i=$(( i + 1 ))
    done
    mapfile -t BS_SELECTED < <(printf '%s\n' "${keyed[@]}" | LC_ALL=C sort |
                               awk '{ print $3 }')
}

bs_list_jobs() {
    local name after group est
    local -a deps=()
    printf '%-30s %-26s %-7s %6s %-3s %-15s %s\n' JOB GROUP DEFAULT EST_S \
        CPU AFTER OUTPUTS
    for name in "${BS_JOBS[@]}"; do
        read -r -a deps <<<"${BS_DEPS[$name]}" || true
        after=${BS_DEPS[$name]:--}
        if (( ${#deps[@]} > 1 )); then after="${#deps[@]} jobs above"; fi
        group=${BS_GROUP[$name]}${BS_TAGS[$name]:+ (${BS_TAGS[$name]})}
        est=${BS_EST[$name]:--}
        bs_weight "$name"
        printf '%-30s %-26s %-7s %6s %-3s %-15s %s\n' "$name" "$group" \
            "$([[ -n ${BS_OPTIN[$name]:-} ]] && echo no || echo yes)" \
            "$est" "$BS_W" "$after" "${BS_OUTS[$name]:--}"
    done
}

bs_print_plan() {
    local v name
    echo "# $BS_DRIVER --dry-run: nothing is executed or written"
    echo "# root:   $BS_ROOT"
    echo "# python: $BS_PYTHON"
    echo "# data:   $BS_DATA_DIR"
    echo "# at most $BS_MAX_JOBS cores busy (jobs with worker processes count as several); ${#BS_SELECTED[@]} selected, launched in this order"
    if (( BS_ACCEPT_REPEATS )); then
        echo "# design: --repeats $BS_REPEATS; DATASETS: ${BS_DATASETS[*]}"
    fi
    echo "# environment of every job:"
    for v in "${BS_ENV_SET[@]}"; do
        printf 'export %s=%q\n' "$v" "${!v}"
    done
    for v in "${BS_ENV_UNSET[@]}"; do
        printf 'unset %s\n' "$v"
    done
    for name in "${BS_SELECTED[@]}"; do
        printf '\n# [%s]  after: %s\n#   outputs: %s\n' "$name" \
            "${BS_DEPS[$name]:--}" "${BS_OUTS[$name]:--}"
        printf '(cd %q && %s) > %q 2>&1\n' "$BS_CODE_DIR" "${BS_CMD[$name]}" \
            "$BS_ROOT/stdout/$name.txt"
    done
}

# Status record: one tab-separated line per job in the run's state dir.
bs_write_status() {  # NAME STATUS EXIT T0 T1 MISSING NOTE
    local name=$1 file=$BS_STATE_DIR/$1.status
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "${BS_GROUP[$name]}" "$2" "$3" "$4" "$5" \
        "${BS_DEPS[$name]}" "${BS_OUTS[$name]}" "$6" "$7" "${BS_CMD[$name]}" \
        >"$file.tmp"
    mv -f -- "$file.tmp" "$file"
}

# Last run of a job in this root, across invocations: .state/last/NAME holds
# "NAME<tab>STATUS<tab>RUN_ID<tab>COMMAND".  STATUS is "running" while the
# job runs, so a run killed outright never leaves an earlier "ok" behind.
# COMMAND (the job's %q-quoted argv, which has no tabs) tells a consumer
# whether the outputs come from the design it expects (--repeats, datasets);
# records of older drivers have no COMMAND.
bs_write_last() {  # NAME STATUS
    local file=$BS_ROOT/.state/last/$1
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$BS_RUN_ID" "${BS_CMD[$1]}" >"$file.tmp.$$"
    mv -f -- "$file.tmp.$$" "$file"
}

bs_last_field() {  # NAME FIELD(status|command) -> value, or nothing
    local file=$BS_ROOT/.state/last/$1 n st id cmd=""
    [[ -r $file ]] || return 0
    IFS=$'\t' read -r n st id cmd <"$file" || true
    case $2 in
        status) printf '%s' "$st" ;;
        command) printf '%s' "$cmd" ;;
    esac
}

bs_last_status() {  # NAME -> STATUS, or nothing without a record
    bs_last_field "$1" status
}

# Runs in the background: one job, its stdout file and its status record.
bs_job_wrapper() {
    local name=$1 out=$BS_ROOT/stdout/$1.txt child="" rc=0 t0 t1 f status lockfd
    local nerr=0 note=""
    local -a missing=()
    set +e
    trap 'if [[ -n $child ]]; then kill -TERM "$child" 2>/dev/null || true; fi' TERM INT
    t0=$(bs_now)
    if command -v flock >/dev/null 2>&1; then
        exec {lockfd}>"$BS_ROOT/.state/locks/$name.lock"
        if ! flock -n "$lockfd"; then
            bs_write_status "$name" failed "" "$t0" "$t0" "" \
                "job is already running in this root (lock held by another invocation)"
            return 0
        fi
    fi
    bs_write_last "$name" running
    for f in ${BS_OUTS[$name]}; do rm -f -- "${BS_ROOT:?}/$f"; done
    {
        printf '# job      %s\n' "$name"
        printf '# command  cd %q && %s\n' "$BS_CODE_DIR" "${BS_CMD[$name]}"
        printf '# started  %s on %s\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)"
    } >"$out"
    ( cd "$BS_CODE_DIR" && eval "exec ${BS_CMD[$name]}" ) >>"$out" 2>&1 </dev/null &
    child=$!
    wait "$child"
    rc=$?
    # A trapped signal interrupts wait; keep waiting for the real exit status.
    while kill -0 "$child" 2>/dev/null; do
        wait "$child"
        rc=$?
    done
    t1=$(bs_now)
    for f in ${BS_OUTS[$name]}; do
        [[ -s $BS_ROOT/$f ]] || missing+=("$f")
    done
    if [[ -z ${BS_NOSCAN[$name]:-} ]]; then
        nerr=$(LC_ALL=C grep -a -c -E -e "$BS_LOG_ERROR_RE" -- "$out")
        [[ $nerr =~ ^[0-9]+$ ]] || nerr=0
    fi
    if (( rc != 0 )); then status=failed
    elif (( ${#missing[@]} )); then status=missing_outputs
    elif (( nerr )); then
        status=log_errors
        note="$nerr line(s) of stdout/$name.txt report a caught failure (quoted in the manifest)"
    else status=ok
    fi
    printf '\n# finished %s  exit=%d  status=%s%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$rc" "$status" "${missing[*]:+  missing: ${missing[*]}}" >>"$out"
    bs_write_last "$name" "$status"
    bs_write_status "$name" "$status" "$rc" "$t0" "$t1" "${missing[*]}" "$note"
}

# ready | wait | fail:<reason>
#   A selected dependency must finish ok in this invocation.  An unselected
#   one must have succeeded in its last run in this root (any invocation)
#   with the command it has now (so its outputs come from the same design:
#   --repeats, datasets, interpreter), and the files this job reads
#   (BS_INPUTS, else all of the dependency's outputs) must be present.
#   An opt-in dependency (BS_OPTIN) that is not selected imposes nothing:
#   the job reads its outputs if an earlier run left them (numbers and the
#   opt-in tests job).
bs_dep_state() {
    local name=$1 dep f last cmd
    local -a need=()
    for dep in ${BS_DEPS[$name]}; do
        if [[ -n ${BS_IS_SEL[$dep]:-} ]]; then
            case ${BS_RESULT[$dep]:-} in
                ok) ;;
                "") echo wait; return 0 ;;
                *) echo "fail:dependency $dep ${BS_RESULT[$dep]}"; return 0 ;;
            esac
            continue
        fi
        [[ -z ${BS_OPTIN[$dep]:-} ]] || continue
        last=$(bs_last_status "$dep")
        if [[ $last != ok ]]; then
            [[ -n $last ]] && last="ended $last" || last="is not recorded"
            echo "fail:job $dep is not selected and its last run in this root $last; run it first"
            return 0
        fi
        cmd=$(bs_last_field "$dep" command)
        if [[ -n $cmd && $cmd != "${BS_CMD[$dep]}" ]]; then
            echo "fail:job $dep is not selected and its last run in this root ran a different command (other --repeats, DATASETS or PYTHON?): '$cmd'; run it first"
            return 0
        fi
        if [[ -z ${BS_INPUTS[$name]:-} ]]; then need+=(${BS_OUTS[$dep]}); fi
    done
    if [[ -n ${BS_INPUTS[$name]:-} ]]; then need=(${BS_INPUTS[$name]}); fi
    for f in "${need[@]}"; do
        if [[ ! -s $BS_ROOT/$f ]]; then
            echo "fail:input $f missing (run its producer first)"
            return 0
        fi
    done
    echo ready
}

bs_collect() {
    local name=$1 status rc t0 t1 note
    local -a f=()
    # One field per line keeps empty fields (IFS=tab would merge them).
    mapfile -t f <<<"$(tr '\t' '\n' <"$BS_STATE_DIR/$name.status")"
    status=${f[2]:-} rc=${f[3]:-} t0=${f[4]:-0} t1=${f[5]:-0} note=${f[9]:-}
    BS_RESULT[$name]=$status
    bs_log "$(printf 'done   %-26s %-16s exit=%-4s %8ss  %s' "$name" "$status" \
        "${rc:--}" "$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')" "$note")"
}

bs_on_signal() {
    local n
    BS_INTERRUPTED=1
    bs_log "signal received: stopping ${#BS_RUNNING[@]} running job(s), skipping the rest"
    for n in "${BS_RUNNING[@]}"; do
        kill -TERM "${BS_PID[$n]}" 2>/dev/null || true
    done
}

# EASY backfilling (bs_run_selected): when the running jobs end as
# estimated (start + BS_EST; a job without an estimate, or past it, may end
# any moment), the time BS_SHADOW at which NEED cores are free, and the
# cores BS_SPARE that a job of NEED cores leaves free then.  A job without
# an estimate (the short ones) counts as ending at once when backfilled.
bs_shadow() {  # NEED BUSY NOW
    local need=$1 free=$(( BS_MAX_JOBS - $2 )) end w n
    BS_SHADOW=$3
    while read -r end w; do
        if (( free >= need )); then break; fi
        free=$(( free + w ))
        if (( end > BS_SHADOW )); then BS_SHADOW=$end; fi
    done < <(for n in "${BS_RUNNING[@]}"; do
                 bs_weight "$n"
                 printf '%d %d\n' $(( ${BS_T0[$n]:-0} + ${BS_EST[$n]:-0} )) "$BS_W"
             done | LC_ALL=C sort -n)
    BS_SPARE=$(( free - need ))
    if (( BS_SPARE < 0 )); then BS_SPARE=0; fi
}

# Launch selected jobs as their dependencies complete, in BS_SELECTED
# order, while the cores they occupy (bs_weight; 1 per job unless it runs
# worker processes) fit into BS_MAX_JOBS; returns when every selected job
# has a status record.  A ready multi-core job that does not fit yet holds
# back the jobs after it, so that freed cores accumulate for it instead of
# going to single-core jobs one by one; jobs after it that fit still start
# if, by the estimates (BS_EST), they end before it can start or fit into
# the cores it leaves free then (EASY backfilling, bs_shadow), so that it
# is not delayed and no core idles meanwhile.
bs_run_selected() {
    local name st now blocked launch
    local -i busy=0 t=0
    local -a pending=("${BS_SELECTED[@]}") next=() alive=()
    trap bs_on_signal INT TERM
    while (( ${#pending[@]} + ${#BS_RUNNING[@]} > 0 )); do
        next=()
        blocked=0
        busy=0
        now=$(bs_now)
        t=${now%.*}
        for name in "${BS_RUNNING[@]}"; do
            bs_weight "$name"
            busy+=BS_W
        done
        for name in "${pending[@]}"; do
            if (( BS_INTERRUPTED )); then
                now=$(bs_now)
                bs_write_status "$name" skipped "" "$now" "$now" "" "driver interrupted"
                bs_collect "$name"
                continue
            fi
            st=$(bs_dep_state "$name")
            case $st in
                ready)
                    bs_weight "$name"
                    launch=0
                    if (( busy + BS_W <= BS_MAX_JOBS )); then
                        if (( ! blocked )); then
                            launch=1
                        elif (( t + ${BS_EST[$name]:-0} <= BS_SHADOW )); then
                            launch=1
                        elif (( BS_W <= BS_SPARE )); then
                            launch=1
                            BS_SPARE=$(( BS_SPARE - BS_W ))
                        fi
                    elif (( BS_W > 1 && ! blocked )); then
                        blocked=1
                        bs_shadow "$BS_W" "$busy" "$t"
                    fi
                    if (( launch )); then
                        bs_job_wrapper "$name" &
                        BS_PID[$name]=$!
                        BS_T0[$name]=$t
                        BS_RUNNING+=("$name")
                        busy+=BS_W
                        if (( BS_W > 1 )); then
                            bs_log "start  $name ($BS_W cores)"
                        else
                            bs_log "start  $name"
                        fi
                    else
                        next+=("$name")
                    fi ;;
                wait)
                    next+=("$name") ;;
                fail:*)
                    now=$(bs_now)
                    bs_write_status "$name" skipped "" "$now" "$now" "" "${st#fail:}"
                    bs_collect "$name" ;;
            esac
        done
        pending=("${next[@]}")
        (( ${#BS_RUNNING[@]} )) || continue
        wait -n 2>/dev/null || true
        alive=()
        for name in "${BS_RUNNING[@]}"; do
            if [[ -e $BS_STATE_DIR/$name.status ]]; then
                bs_collect "$name"
            elif ! kill -0 "${BS_PID[$name]}" 2>/dev/null; then
                now=$(bs_now)
                bs_write_status "$name" failed "" "$now" "$now" "" "job wrapper exited without a status record"
                bs_collect "$name"
            else
                alive+=("$name")
            fi
        done
        BS_RUNNING=("${alive[@]}")
    done
    trap - INT TERM
}

# Provenance helper (Python standard library + numpy's build info).
#   preflight: environment record, pin check, snapshot of guarded repo dirs
#   manifest:  per-job records, artifact hashes, repo-write check, summary
read -r -d '' BS_HELPER_PY <<'PY' || true
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# Repository dirs no job may touch (recursively; the top-level files are
# guarded too), names skipped there, and the output dirs under the root.
GUARDED = ["data", "logs", "logs_logit", "paper", "code"]
GUARD_SKIP = {"__pycache__", ".pytest_cache", ".DS_Store", ".env"}
OUTPUT_DIRS = ["logs", "logs_logit", "figures", "figures_logit", "cov",
               "numbers"]
# Whitelist: never record arbitrary environment (tokens, credentials) ...
ENV_KEYS = ["VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "MPLBACKEND",
            "SOURCE_DATE_EPOCH", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
            "RUN_ONLY", "MAX_JOBS", "DATASETS", "DATASETS_EXTRA", "WORKERS",
            "BENCHSELECT_ALLOW_UNPINNED"]
# ... except the interpreter, BLAS and CUDA settings, which can change
# results (PYTHONOPTIMIZE strips asserts, OPENBLAS_CORETYPE picks kernels).
ENV_PREFIXES = ("PYTHON", "OMP_", "OPENBLAS_", "MKL_", "KMP_", "NPY_",
                "CUBLAS_", "CUDA_")
MAX_QUOTED = 20     # log lines quoted per job


def visible(paths):
    """Sorted paths without dotfiles: a tar copy made on macOS without
    COPYFILE_DISABLE=1 adds AppleDouble '._*' files, which must not enter
    the code and input hashes."""
    return sorted(p for p in paths if not p.name.startswith("."))


def code_files(repo):
    return visible([*repo.glob("code/*.py"), *repo.glob("code/*.sh"),
                    *repo.glob("requirements*.txt")])


def input_files(data):
    """Every npz of data/ (score matrices, item scores of
    eval_item_budget.py, data/helm.npz of eval_prospective.py's HELM
    setting, the snapshot and covariance files), JSON inputs (cost tables,
    canonical mapping, meta, score_bounds.json), the raw collection tables
    (plot_benchmark_overlap.py's cross-check) and SHA256SUMS
    (analyze_objectives.py's data check)."""
    return visible([*data.glob("*.npz"), *data.glob("*.json"),
                    *data.glob("*.csv"), *data.glob("SHA256SUMS")])


def rel(p, base):
    return str(p.relative_to(base)) if p.is_relative_to(base) else str(p)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_content_sha256(path):
    """sha256 over the arrays of an .npz, independent of zip timestamps."""
    h = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            data = z.read(name)
            h.update(f"{name}\0{len(data)}\0".encode())
            h.update(data)
    return h.hexdigest()


def file_record(path):
    rec = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.suffix == ".npz":
        try:
            rec["content_sha256"] = npz_content_sha256(path)
        except (OSError, zipfile.BadZipFile) as exc:
            rec["content_sha256"] = f"error: {exc}"
    return rec


def iso(t):
    if t in ("", None):
        return None
    dt = datetime.fromtimestamp(float(t), timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def norm_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirements(path):
    """name==version pins (following -r includes) and '# python==X.Y'."""
    pins, python = {}, None
    for raw in Path(path).read_text().splitlines():
        m = re.match(r"#\s*python\s*==\s*(\S+)", raw.strip())
        if m and python is None:
            python = m.group(1)
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r"):
            sub_pins, sub_python = read_requirements(
                Path(path).parent / line[2:].strip())
            pins = {**sub_pins, **pins}
            python = python or sub_python
            continue
        name, sep, version = line.partition("==")
        if not sep or not name.strip():
            raise ValueError(f"{path}: expected name==version, got {raw!r}")
        pins[norm_name(name.strip())] = version.strip()
    return pins, python


def installed_packages():
    pkgs = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            pkgs[norm_name(name)] = dist.version
    return dict(sorted(pkgs.items()))


def check_requirements(req, pkgs):
    rec = {"file": str(req), "python": None, "pins": {}, "mismatches": [],
           "unpinned_installed": []}
    if not req.is_file():
        rec["mismatches"].append(f"requirements file not found: {req}")
        return rec
    pins, python = read_requirements(req)
    rec["python"], rec["pins"] = python, pins
    have_py = platform.python_version()
    if python and have_py != python and not have_py.startswith(python + "."):
        rec["mismatches"].append(f"python {have_py} (requires {python})")
    for name, want in pins.items():
        have = pkgs.get(name)
        if have != want:
            rec["mismatches"].append(
                f"{name} {have or 'not installed'} (requires {want})")
    # Informational: installed but not in the lock (pip, setuptools, ...).
    rec["unpinned_installed"] = [f"{n}=={v}" for n, v in pkgs.items()
                                 if n not in pins]
    return rec


def git_info(repo, hashed):
    """Commit and dirty flag.  `hashed` are the hashed code files: an
    untracked one (a new script) is missing from `git diff HEAD`, so it
    makes the tree dirty too."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True).stdout
    try:
        top = Path(git("rev-parse", "--show-toplevel").decode().strip())
        if top.resolve() == repo.resolve():
            diff = git("diff", "HEAD", "--binary")
            untracked = git("ls-files", "--others",
                            "--exclude-standard").decode().splitlines()
            new_code = sorted(set(untracked) & set(hashed))
            return {"source": "git",
                    "commit": git("rev-parse", "HEAD").decode().strip(),
                    "dirty": bool(diff) or bool(new_code),
                    "diff_sha256": hashlib.sha256(diff).hexdigest(),
                    "untracked_files": len(untracked),
                    "untracked_code_files": new_code}
    except (OSError, subprocess.CalledProcessError):
        pass
    env = os.environ
    if env.get("BENCHSELECT_GIT_COMMIT"):
        dirty = env.get("BENCHSELECT_GIT_DIRTY", "").strip().lower()
        return {"source": "env (BENCHSELECT_GIT_*, unchecked; the tree is "
                          "not a git work tree)",
                "commit": env["BENCHSELECT_GIT_COMMIT"],
                "dirty": {"1": True, "true": True, "yes": True,
                          "0": False, "false": False, "no": False}.get(dirty),
                "diff_sha256": env.get("BENCHSELECT_GIT_DIFF_SHA256") or None}
    return {"source": "unavailable", "commit": None, "dirty": None,
            "diff_sha256": None}


def host_info():
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_model": cpu,
            "cpu_count": os.cpu_count()}


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
             "--format=csv,noheader"], check=True, capture_output=True,
            text=True, timeout=30).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def numpy_build():
    try:
        import numpy
        deps = numpy.show_config(mode="dicts").get("Build Dependencies", {})
        keep = ("name", "version", "openblas configuration")
        return {k: {f: v for f, v in (deps.get(k) or {}).items() if f in keep}
                for k in ("blas", "lapack")}
    except Exception as exc:  # recorded, never fatal
        return {"error": repr(exc)}


def hashes(paths, base):
    return {rel(p, base): file_record(p) for p in paths}


def code_hashes(repo):
    return {rel(p, repo): sha256_file(p) for p in code_files(repo)}


def tree_sha256(code):
    return hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in code.items())
                          .encode()).hexdigest()


def hf_cache_dir():
    env = os.environ
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    xdg = env.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(xdg) / "huggingface" / "hub"


def hf_files(specs):
    """Revision and sha256 of Hugging Face Hub files in the local cache, as
    hf_hub_download(repo_id, filename) without a revision resolved them
    (refs/main).  spec = REPO_ID:FILENAME[@EXPECTED_SHA256]."""
    out, cache = [], hf_cache_dir()
    for spec in specs:
        spec, _, want = spec.partition("@")
        repo_id, _, filename = spec.partition(":")
        base = cache / ("models--" + repo_id.replace("/", "--"))
        rec = {"repo_id": repo_id, "filename": filename,
               "cache": str(base), "revision": None, "bytes": None,
               "sha256": None, "expected_sha256": want or None}
        try:
            rec["revision"] = (base / "refs" / "main").read_text().strip()
            p = base / "snapshots" / rec["revision"] / filename
            rec["bytes"], rec["sha256"] = p.stat().st_size, sha256_file(p)
        except OSError as exc:
            rec["error"] = f"not in the cache: {exc}"
        rec["ok"] = bool(rec["sha256"]) and (not want or rec["sha256"] == want)
        out.append(rec)
    return out


def snapshot(repo):
    """(size, mtime) of the repository's top-level entries and of everything
    under its GUARDED dirs."""
    snap = {}
    for p in sorted(repo.iterdir()):
        if p.name in GUARD_SKIP:
            continue
        if p.is_dir():
            snap[p.name + "/"] = "dir"
        else:
            st = p.lstat()
            snap[p.name] = [st.st_size, st.st_mtime_ns]
    for sub in GUARDED:
        base = repo / sub
        if not base.is_dir():
            continue
        snap[sub + "/"] = "dir"
        for p in sorted(base.rglob("*")):
            if GUARD_SKIP & set(p.relative_to(base).parts):
                continue
            key = str(p.relative_to(repo))
            if p.is_dir():
                snap[key + "/"] = "dir"
            else:
                st = p.lstat()
                snap[key] = [st.st_size, st.st_mtime_ns]
    return snap


def preflight(a):
    repo, data, state = Path(a.repo), Path(a.data_dir), Path(a.state)
    pkgs = installed_packages()
    req = check_requirements(Path(a.requirements), pkgs)
    allow = os.environ.get("BENCHSELECT_ALLOW_UNPINNED") == "1"
    code = code_hashes(repo)
    tree = tree_sha256(code)
    inputs = input_files(data)
    params = dict(p.partition("=")[::2] for p in a.param)
    pre = {
        "start_epoch": float(a.start),
        "driver": a.driver,
        "run_id": a.run_id,
        "command_line": a.cmdline,
        **({"parameters": params} if params else {}),
        "repo": str(repo),
        "experiment_root": a.root,
        "data_dir": str(data),
        "git": git_info(repo, code),
        "code_tree_sha256": tree,
        "code_sha256": code,
        "inputs": hashes(inputs, repo),
        "host": host_info(),
        "gpus": gpu_info() if a.gpu_info else None,
        "python": {"executable": sys.executable,
                   "version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "requirements": req,
        "allow_unpinned": allow,
        "numpy_build": numpy_build(),
        "packages": pkgs,
        "env": {k: v for k, v in sorted(os.environ.items())
                if k in ENV_KEYS or k.startswith(ENV_PREFIXES)},
        "benchselect_env": {k: v for k, v in sorted(os.environ.items())
                            if k.startswith("BENCHSELECT_")
                            and not k.startswith("BENCHSELECT_GIT_")},
        "repo_snapshot": snapshot(repo),
    }
    (state / "preflight.json").write_text(json.dumps(pre, indent=1))
    g = pre["git"]
    print(f"python   {sys.executable} ({platform.python_version()})")
    print(f"git      {g['commit'] or '-'} dirty={g['dirty']} [{g['source']}]")
    print(f"code     tree sha256 {tree[:16]}… ({len(code)} files)")
    print(f"inputs   {len(inputs)} data files hashed")
    if params:
        print("design   " + "  ".join(f"{k}={v}" for k, v in params.items()))
    if g["source"] == "unavailable":
        print("warning: no git provenance (not a git work tree, and "
              "BENCHSELECT_GIT_COMMIT is not set); the manifest identifies "
              "the code by code_sha256 only", file=sys.stderr)
    if g.get("untracked_code_files"):
        print(f"note: {len(g['untracked_code_files'])} hashed code file(s) "
              "are untracked, so dirty=True: "
              + " ".join(g["untracked_code_files"]))
    if req["mismatches"]:
        print(f"environment does not match {a.requirements}:", file=sys.stderr)
        for m in req["mismatches"]:
            print(f"  {m}", file=sys.stderr)
        if not allow:
            print("set BENCHSELECT_ALLOW_UNPINNED=1 to run anyway "
                  "(recorded in the manifest)", file=sys.stderr)
            return 3
        print("BENCHSELECT_ALLOW_UNPINNED=1: continuing", file=sys.stderr)
    return 0


def diff_maps(before, after):
    """'created/removed/modified: key' for every key whose value differs."""
    out = []
    for k in sorted(set(before) | set(after)):
        if before.get(k) != after.get(k):
            what = ("created" if k not in before else
                    "removed" if k not in after else "modified")
            out.append(f"{what}: {k}")
    return out


def scan_log(path, pattern):
    """Lines of a job's stdout that report a caught failure: their count and
    the first MAX_QUOTED of them as 'LINENO: text'."""
    rx, quoted, n = re.compile(pattern), [], 0
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return 0, []
    for i, line in enumerate(lines, 1):
        if rx.search(line):
            n += 1
            if len(quoted) < MAX_QUOTED:
                quoted.append(f"{i}: {line.strip()[:300]}")
    return n, quoted


def earlier_outputs(root, run_id):
    """Root-relative path -> the latest earlier run that wrote it, from the
    archived manifests (either driver): a job that ran (has an exit code)
    wrote its declared outputs, whose sha256 that manifest lists."""
    latest = {}
    for p in sorted((root / "manifests").glob("*.json")):
        if p.stem == run_id:
            continue
        try:
            m = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        arts = m.get("artifacts") or {}
        for j in m.get("jobs") or []:
            if j.get("exit_code") is None:
                continue
            for out in j.get("outputs") or []:
                if out not in arts:
                    continue
                cand = {"run_id": m.get("run_id"), "driver": m.get("driver"),
                        "job": j.get("name"), "job_status": j.get("status"),
                        "end_utc": j.get("end_utc") or m.get("end_utc") or "",
                        "code_tree_sha256": m.get("code_tree_sha256"),
                        "git_commit": (m.get("git") or {}).get("commit"),
                        "sha256": arts[out].get("sha256")}
                if out not in latest or cand["end_utc"] >= latest[out]["end_utc"]:
                    latest[out] = cand
    return latest


def manifest(a):
    repo, root, state = Path(a.repo), Path(a.root), Path(a.state)
    pre = json.loads((state / "preflight.json").read_text())
    before = pre.pop("repo_snapshot")
    end = time.time()
    noscan = set(a.noscan.split())
    jobs = []
    for name in (state / "jobs.order").read_text().split():
        p = state / f"{name}.status"
        if not p.exists():
            jobs.append({"name": name, "status": "no_status_record"})
            continue
        (name, group, status, rc, t0, t1, deps, outs, missing, note,
         cmd) = p.read_text().rstrip("\n").split("\t", 10)
        ran = (root / "stdout" / f"{name}.txt").exists() and rc != ""
        n_err, quoted = (scan_log(root / "stdout" / f"{name}.txt",
                                  a.log_error_re)
                         if ran and name not in noscan else (0, []))
        jobs.append({
            "name": name, "group": group, "status": status,
            "exit_code": int(rc) if rc else None,
            "start_utc": iso(t0), "end_utc": iso(t1),
            "duration_s": round(float(t1) - float(t0), 3) if t0 and t1 else None,
            "depends_on": deps.split(),
            "command": f"cd {a.code_dir} && {cmd}",
            "stdout": f"stdout/{name}.txt" if ran else None,
            "outputs": outs.split(),
            "missing_outputs": missing.split(),
            "log_error_lines": n_err,
            "log_errors": quoted,
            "note": note or None,
        })

    # Every file under the output dirs, with its origin: a job of this
    # invocation, the latest earlier run whose archived manifest records
    # the same content, or unknown.
    produced = {o: j for j in jobs if j.get("exit_code") is not None
                for o in j["outputs"]}
    earlier = earlier_outputs(root, a.run_id)
    artifacts, provenance_warnings = {}, []
    origins = {"this_run": 0, "earlier_run": 0, "unknown": 0}
    for d in OUTPUT_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            key = str(p.relative_to(root))
            rec = file_record(p)
            e = earlier.get(key)
            if key in produced:
                j = produced[key]
                rec.update(origin="this_run", job=j["name"],
                           job_status=j["status"], run_id=a.run_id)
            elif e and e["sha256"] == rec["sha256"]:
                rec.update(origin="earlier_run",
                           **{k: v for k, v in e.items() if k != "sha256"})
                if e["job_status"] != "ok":
                    provenance_warnings.append(
                        f"{key}: written by job {e['job']} of run "
                        f"{e['run_id']}, which ended {e['job_status']}")
                elif e["code_tree_sha256"] != pre["code_tree_sha256"]:
                    provenance_warnings.append(
                        f"{key}: from run {e['run_id']} (job {e['job']}), code "
                        f"tree {str(e['code_tree_sha256'])[:12]}… (this run: "
                        f"{pre['code_tree_sha256'][:12]}…)")
            else:
                why = (f"changed since run {e['run_id']} (job {e['job']}) "
                       "recorded it" if e else
                       "no archived manifest records a job writing it")
                rec.update(origin="unknown", note=why)
                provenance_warnings.append(f"{key}: origin unknown ({why})")
            origins[rec["origin"]] += 1
            artifacts[key] = rec

    # The code and the inputs must be the ones hashed at the start.
    code_now = code_hashes(repo)
    inputs_now = {rel(p, repo): sha256_file(p)
                  for p in input_files(Path(pre["data_dir"]))}
    changed = ([f"code {c}" for c in diff_maps(pre["code_sha256"], code_now)]
               + [f"input {c}" for c in diff_maps(
                   {k: r["sha256"] for k, r in pre["inputs"].items()},
                   inputs_now)])
    repo_writes = diff_maps(before, snapshot(repo))
    hf = hf_files(a.hf_file) if a.hf_file else None
    hf_bad = [r for r in hf or [] if not r["ok"]]
    bad = [j for j in jobs if j["status"] != "ok"]
    ok = (not bad and not repo_writes and not changed and not hf_bad
          and not a.interrupted)
    man = {
        "schema": 2,
        **{k: pre[k] for k in ("driver", "run_id", "command_line")},
        "ok": ok,
        "interrupted": a.interrupted,
        "start_utc": iso(pre["start_epoch"]),
        "end_utc": iso(end),
        "wall_s": round(end - pre["start_epoch"], 1),
        **{k: v for k, v in pre.items()
           if k not in ("driver", "run_id", "command_line", "start_epoch")},
        "changed_during_run": changed,
        "guarded_repo_dirs": GUARDED + ["(top-level files)"],
        "repo_writes": repo_writes,
        "log_error_pattern": a.log_error_re,
        **({"hf_files": hf} if hf is not None else {}),
        "jobs": jobs,
        "artifact_origins": origins,
        "provenance_warnings": provenance_warnings,
        "artifacts": artifacts,
    }
    out = Path(a.manifest)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(man, indent=1) + "\n")
    tmp.replace(out)
    (root / "manifests").mkdir(exist_ok=True)
    shutil.copyfile(out, root / "manifests" / f"{a.run_id}.json")

    w = max([len(j["name"]) for j in jobs] + [4])
    print(f"\n== {a.driver}: {len(jobs)} job(s), wall {man['wall_s']:.0f}s, "
          f"root {root} ==")
    print(f"{'job':<{w}}  {'status':<16} {'exit':>4} {'duration':>10}")
    for j in jobs:
        dur = j.get("duration_s")
        ec = j.get("exit_code")
        print(f"{j['name']:<{w}}  {j['status']:<16} "
              f"{'-' if ec is None else ec:>4} "
              f"{'-' if dur is None else f'{dur:.1f}s':>10}"
              + (f"  {j['note']}" if j.get("note") else "")
              + (f"  missing: {' '.join(j['missing_outputs'])}"
                 if j.get("missing_outputs") else ""))
        for line in j.get("log_errors", [])[:3]:
            print(f"{'':<{w}}    stdout/{j['name']}.txt:{line}")
    print(f"artifacts: {len(artifacts)} files ({origins['this_run']} from "
          f"this run, {origins['earlier_run']} from earlier runs, "
          f"{origins['unknown']} unknown); manifest: {out}")
    for r in provenance_warnings[:10]:
        print(f"  note: {r}")
    if len(provenance_warnings) > 10:
        print(f"  ... {len(provenance_warnings) - 10} more "
              "(provenance_warnings in the manifest)")
    for r in hf or []:
        print(f"hf file  {r['repo_id']}/{r['filename']} revision "
              f"{r['revision'] or '-'} sha256 {(r['sha256'] or '-')[:16]}…")
    if changed:
        print(f"ERROR: {len(changed)} code or input file(s) changed during "
              "the run; the results may not match code_sha256 / inputs:",
              file=sys.stderr)
        for r in changed[:50]:
            print(f"  {r}", file=sys.stderr)
    if repo_writes:
        print(f"ERROR: {len(repo_writes)} path(s) in the repository changed "
              "during the run (by a job, or by someone editing the tree):",
              file=sys.stderr)
        for r in repo_writes[:50]:
            print(f"  {r}", file=sys.stderr)
    for r in hf_bad:
        print(f"ERROR: {r['repo_id']}/{r['filename']}: "
              + (r.get("error") or f"sha256 {r['sha256']} is not the pinned "
                 f"{r['expected_sha256']}"), file=sys.stderr)
    if bad:
        print(f"FAILED: {len(bad)} of {len(jobs)} job(s) not ok: "
              + ", ".join(f"{j['name']} ({j['status']})" for j in bad),
              file=sys.stderr)
    if a.interrupted:
        print("FAILED: the driver was interrupted", file=sys.stderr)
    if ok:
        print(f"all {len(jobs)} job(s) ok")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(prog="benchselect-provenance")
    ap.add_argument("mode", choices=["preflight", "manifest"])
    for opt in ("driver", "run-id", "repo", "root", "data-dir", "state",
                "code-dir", "requirements", "manifest", "cmdline", "start",
                "log-error-re", "noscan"):
        ap.add_argument(f"--{opt}", default="")
    ap.add_argument("--gpu-info", action="store_true")
    ap.add_argument("--interrupted", action="store_true")
    ap.add_argument("--hf-file", action="append", default=[],
                    help="REPO_ID:FILENAME[@SHA256] to record (and check)")
    ap.add_argument("--param", action="append", default=[],
                    help="NAME=VALUE design parameter to record (preflight)")
    a = ap.parse_args()
    return preflight(a) if a.mode == "preflight" else manifest(a)


sys.exit(main())
PY

bs_helper() {  # MODE [extra flags]
    local mode=$1
    shift
    "$BS_PYTHON" -I -c "$BS_HELPER_PY" "$mode" \
        --driver "$BS_DRIVER" --run-id "$BS_RUN_ID" --repo "$BS_REPO" \
        --root "$BS_ROOT" --data-dir "$BS_DATA_DIR" --state "$BS_STATE_DIR" \
        --code-dir "$BS_CODE_DIR" --requirements "$BS_REQUIREMENTS" \
        --manifest "$BS_MANIFEST" --cmdline "$0${BS_ARGS:+ $BS_ARGS}" \
        --start "$BS_START" --log-error-re "$BS_LOG_ERROR_RE" \
        --noscan "${!BS_NOSCAN[*]}" "$@"
}

# Create the output tree, record the environment, check the pins (exit 3
# on a mismatch, 2 on any other preflight failure).
bs_prepare_root() {
    local extra=("$@") rc=0
    [[ -x $BS_PYTHON ]] || bs_die "PYTHON=$BS_PYTHON is not an executable interpreter"
    [[ -d $BS_DATA_DIR ]] || bs_die "data directory $BS_DATA_DIR does not exist"
    mkdir -p -- "$BS_ROOT"/{logs,logs_logit,figures,figures_logit,cov,numbers,stdout,manifests} \
        "$BS_ROOT/.state/locks" "$BS_ROOT/.state/last" "$BS_STATE_DIR"
    printf '%s\n' "${BS_SELECTED[@]}" >"$BS_STATE_DIR/jobs.order"
    bs_log "$BS_DRIVER: ${#BS_SELECTED[@]} job(s) into $BS_ROOT (run id $BS_RUN_ID)"
    bs_helper preflight "${extra[@]}" || rc=$?
    case $rc in
        0) ;;
        3) printf '%s: error: %s does not match %s; nothing was run\n' \
               "$BS_DRIVER" "$BS_PYTHON" "$BS_REQUIREMENTS" >&2
           exit 3 ;;
        *) bs_die "preflight failed (see above); nothing was run" ;;
    esac
}

# Write the manifest, print the summary; returns the driver's exit status.
bs_finish() {
    local rc=0
    local -a flags=("${BS_MANIFEST_FLAGS[@]}")
    (( BS_INTERRUPTED )) && flags+=(--interrupted)
    bs_helper manifest "${flags[@]}" || rc=$?
    if (( rc > 1 )); then
        echo "$BS_DRIVER: manifest helper failed (exit $rc); raw job records:" >&2
        cat "$BS_STATE_DIR"/*.status >&2 || true
        rc=2
    fi
    return "$rc"
}

# Sourced (run_tabimpute.sh): helpers only.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

# ══════════════════════════════════════════════════════════════════
# CPU jobs
# ══════════════════════════════════════════════════════════════════

# Cost tables (data/build_cost_tables.py) and budget grids (fractions of the
# dataset's total cost) of the cost_<ds> jobs: MMLU = test questions per
# subject, the paper's grid (eval_cost_budget.py's default); MTEB = items per
# task, the grid of the rebuttal's MTEB run (budgets_frac of its log).
declare -A BS_COST_TABLE=([mmlu]=mmlu_test_counts.json [mteb]=mteb_task_sizes.json)
declare -A BS_COST_BUDGETS=([mmlu]="0.01 0.02 0.03 0.05 0.08 0.12 0.20"
                            [mteb]="0.002 0.005 0.01 0.02 0.05 0.1 0.2")
# Datasets without a logit job (eval_logit.py docstring: 3.4% of Epoch's
# observed cells, 38.5% of a FrontierMath column, sit at the logit floor,
# and the logit R² about halves; the job alone takes 70 min)
declare -A BS_LOGIT_SKIP=([epoch]="logit transform not applicable")
# Benchmark-family holdout (eval_benchfamily_holdout.py FAMILY_SCHEME): the
# datasets run when in DATASETS or DATASETS_EXTRA, and those that also get
# the --blind-cov sub sensitivity run and its --compare-blind-cov summary
# (EM-estimated: refit and sub differ there only)
BS_BENCHFAM_DATASETS=(mmlu mteb merged olv2sub helm_mmlu olv1items)
BS_BENCHFAM_SUB=(mteb merged)
# eval_prospective.py settings: the four of the paper (combined figure
# figures/prospective.pdf) and the two sensitivity settings
BS_PROSPECTIVE=(mteb2026 helm epoch olv2sub)
BS_PROSPECTIVE_SENS=(olv2sub_math epoch_bymodel)
# Data of the fixed experiments (numbers --datasets)
BS_FIXED_DATA=(mteb2026 helm_lite epoch olv2sub olv1items helm_mmlu helm_classic)
# Datasets whose evaluation runs in worker processes (selection stability
# --workers, estimator ablation --jobs; results are identical for any
# count): the slow ones (EM on merged and benchpress)
BS_PARALLEL_DATASETS=(merged benchpress mteb)
# The same for the benchmark-family holdout (eval_benchfamily_holdout.py
# --workers, over splits) and the prospective settings (eval_prospective.py
# --workers, over control splits): the slow ones
BS_PARALLEL_BENCHFAM=(merged mteb)
BS_PARALLEL_PROSPECTIVE=(epoch epoch_bymodel mteb2026)
# The same for the standard battery (eval_greedy_all.py,
# eval_entropy_vs_mi.py, plot_mi_selection_all.py, eval_baselines.py and,
# for benchpress, eval_benchpress.py: --workers over the splits of every
# holdout) and for logit_<ds> (eval_logit.py --workers): the datasets
# whose jobs take minutes in one process.  The others keep one process:
# spawning a pool costs a few seconds, more than they take.
BS_PARALLEL_BATTERY=(epoch merged merged_v1 benchpress mteb2026 mteb
                     helm_classic)
BS_PARALLEL_LOGIT=(merged benchpress mteb2026 mteb helm_classic)
# Splits of one imputer shard (imputers_<ds>_rep<r>: one repetition of
# cv_split's 10 folds); its --workers is WORKERS, at most this
BS_SHARD_SPLITS=10
# make_numbers.py TEX_GROUPS after 'main': numbers/numbers_<group>.tex
BS_TEX_GROUPS=(datasets prospective item_budget item_fixed token_costs
               token_fixed cost_fixed analyze_objectives benchfamily_holdout
               selection_stability estimator_ablation imputers)
# plot_selection_order.py --methods of the selection_order jobs: entropy
# (figures/selection_order_<ds>.pdf, the paper's figure), facility
# location (the headline method) and greedy trace
# (figures/selection_order_<method>_<ds>.pdf), all read from
# logs/greedy_cv_splits_<ds>.json
BS_SEL_ORDER_METHODS=(entropy facloc trace)

# Estimated wall time (s) of each job at --repeats 5 with every core busy:
# its core-seconds (below: wall time x cores, every core busy) over the
# cores it occupies (BS_WEIGHT: 1, or its worker processes).  Measured in
# the canonical run of 2026-10-20 on 4090-1 (AMD EPYC 7502, 32 cores / 64
# threads, MAX_JOBS 60, WORKERS 16, facility location in every
# experiment; wall 1 h 37 min, bounded by the imputers_merged shards in
# one process each): the jobs that ran in WORKERS processes there
# (benchfam_{merged,mteb}[_sub], prospective_{epoch,epoch_bymodel,
# mteb2026}, stability_ and estimator_{merged,benchpress}) at wall x 16,
# all others in one process at their wall time.  Only the order matters:
# bs_select launches the longest first.  Jobs not listed rank after the
# listed ones, in definition order.
bs_estimates() {
    local name cpu
    local -A core_s=(
        [estimator_merged]=31400 [estimator_benchpress]=18100
        [stability_benchpress]=12400 [prospective_epoch_bymodel]=11400
        [benchfam_merged]=11300 [stability_merged]=8070
        [prospective_epoch]=7860 [greedy_cv_epoch]=4570
        [reviewer_checks]=4250 [logit_merged]=3770
        [analyze_objectives]=3380 [benchpress]=2940
        [greedy_cv_merged]=2330 [entropy_vs_mi_epoch]=2110
        [baselines_epoch]=2000 [mi_selection_epoch]=1990
        [stability_mteb]=1980 [estimator_mteb]=1840
        [benchfam_merged_sub]=1690 [analyze_objectives_fulloracle]=1480
        [baselines_merged_v1]=1230 [prospective_mteb2026]=1200
        [baselines_benchpress]=1190 [benchfam_mteb]=1160
        [logit_mteb2026]=1100 [baselines_merged]=1050
        [logit_benchpress]=1040 [entropy_vs_mi_merged]=820
        [token_fixed]=820 [mi_selection_merged]=770
        [greedy_cv_mteb2026]=660 [logit_mteb]=620
        [benchfam_mteb_sub]=500 [logit_helm_classic]=440
        [family_epoch]=420 [greedy_cv_mteb]=350
        [greedy_cv_helm_classic]=340 [prospective_olv2sub_math]=280
        [cost_fixed_mteb]=250 [prospective_olv2sub]=245
        [cost_fixed_mmlu]=220 [baselines_mteb2026]=200
        [family_merged]=195 [token_costs]=190 [mi_selection_mteb2026]=185
        [cost_mteb]=175 [item_fixed]=160 [baselines_mteb]=155
        [entropy_vs_mi_mteb2026]=155 [estimator_mmlu]=155
        [family_benchpress]=135 [entropy_vs_mi_mteb]=130
        [mi_selection_mteb]=110 [token_costs_exclude]=105
        [benchfam_mmlu]=100 [benchfam_olv2sub]=90
        [baselines_helm_classic]=86 [mi_selection_helm_classic]=86
        [entropy_vs_mi_helm_classic]=83 [stability_mmlu]=78
        [eigenspectrum_release]=69 [benchfam_olv1items]=67
        [family_mteb2026]=55 [logit]=46 [covariance]=35 [family_mteb]=32
        [logit_mmlu]=29 [benchfam_helm_mmlu]=27 [logit_olv2sub]=21
        [item_budget]=20 [logit_olv1items]=18 [prospective_helm]=16
        [greedy_cv_olv2sub]=16 [greedy_cv_mmlu]=15 [baselines_mmlu]=14
        [greedy_cv_olv1items]=13 [cost_mmlu]=11 [logit_helm_mmlu]=11
        [baselines_olv1items]=10 [greedy_cv_helm_mmlu]=10 [logit_extra]=10
        [logit_helm_lite]=10 [family_helm_classic]=9
        [entropy_vs_mi_mmlu]=9 [baselines_helm_mmlu]=9
        [baselines_olv2sub]=8 [entropy_vs_mi_olv2sub]=7
        # the imputer merges take seconds, but imputers_all and numbers
        # wait for them: they rank above the short jobs
        [imputers_merged]=60 [imputers_benchpress]=40
        [imputers_mteb]=40 [imputers_mmlu]=30
    )
    # imputer shards (one CV repetition each), in one process each there:
    # merged 5580-5720 s, benchpress 2420-2600, mteb 2380-2540, mmlu 790-840
    local -A shard=([merged]=5660 [benchpress]=2500 [mteb]=2470 [mmlu]=810)
    for name in "${BS_JOBS[@]}"; do
        cpu=${core_s[$name]:-}
        if [[ $name =~ ^imputers_(.+)_rep[0-9]+$ ]]; then
            cpu=${shard[${BASH_REMATCH[1]}]:-}
        fi
        [[ -n $cpu ]] || continue
        BS_EST[$name]=$(( cpu / ${BS_WEIGHT[$name]:-1} ))
    done
}

bs_tag() {  # TAG JOB...: JOB is also selected by RUN_ONLY=TAG
    local tag=$1 name
    shift
    for name in "$@"; do
        BS_TAGS[$name]="${BS_TAGS[$name]:+${BS_TAGS[$name]} }$tag"
    done
}

bs_in() {  # WORD LIST...: true if WORD is one of LIST
    local w=$1
    shift
    [[ " $* " == *" $w "* ]]
}

# DS... -> the selection-order figures of BS_SEL_ORDER_METHODS (method by
# method), space separated
bs_sel_order_outs() {
    local m ds
    local -a out=()
    for m in "${BS_SEL_ORDER_METHODS[@]}"; do
        for ds in "$@"; do
            if [[ $m == entropy ]]; then
                out+=("figures/selection_order_$ds.pdf")
            else
                out+=("figures/selection_order_${m}_$ds.pdf")
            fi
        done
    done
    printf '%s' "${out[*]}"
}

# WORKERS (default min(16, MAX_JOBS / 4), at least 1): worker processes of
# each job that parallelizes over splits (bs_par); the driver counts that
# many cores for it.  16 on a 64-core host: four such jobs fill it.
bs_init_workers() {
    local w=${WORKERS:-}
    if [[ -z $w ]]; then
        w=$(( BS_MAX_JOBS / 4 ))
        (( w <= 16 )) || w=16
        (( w >= 1 )) || w=1
    fi
    [[ $w =~ ^[1-9][0-9]*$ ]] || bs_die "WORKERS must be a positive integer, got '$w'"
    BS_WORKERS=$w
}

# bs_par JOB [MAX]: JOB runs its splits in WORKERS processes, at most MAX
# (its split count, when it is smaller); sets BS_WEIGHT[JOB] and leaves
# the --workers value in BS_PW
bs_par() {
    BS_PW=$BS_WORKERS
    if [[ -n ${2:-} ]] && (( $2 < BS_PW )); then BS_PW=$2; fi
    BS_WEIGHT[$1]=$BS_PW
}

# bs_par_if JOB DS LIST...: if DS is one of LIST, JOB runs in WORKERS
# processes (bs_par) and par=(--workers WORKERS), else par=(); par is the
# array of the caller (bs_define_cpu_jobs) that goes on JOB's command line
bs_par_if() {
    local job=$1 ds=$2
    shift 2
    par=()
    if bs_in "$ds" "$@"; then
        bs_par "$job"
        par=(--workers "$BS_PW")
    fi
}

bs_define_cpu_jobs() {
    local py=$BS_PYTHON ds tag outs s v1="" all_fig=""
    local -a rep=(--repeats "$BS_REPEATS") cv=() budgets=() logit_extra=()
    local -a famsub=() ndata=() paired=() par=() shards=()
    local r
    local bp=0
    # benchpress: eval_benchpress.py; every other dataset: the CV scripts
    for ds in "${BS_DATASETS[@]}"; do
        if [[ $ds == benchpress ]]; then bp=1; else cv+=("$ds"); fi
    done

    # ── The standard battery, per dataset: DATASETS (the paper) and
    # DATASETS_EXTRA (the data release; also selected by RUN_ONLY=extra).
    # Launch order comes from BS_EST (bs_select), not from this order.
    # The datasets of BS_PARALLEL_BATTERY run their splits in WORKERS
    # processes (bs_par_if), logit_<ds> those of BS_PARALLEL_LOGIT.
    if (( bp )); then
        bs_par_if benchpress benchpress "${BS_PARALLEL_BATTERY[@]}"
        bs_job benchpress benchpress "" \
            "logs/benchpress_results.npz logs/benchpress_results_paired.csv figures/greedy_cv_benchpress.pdf figures/entropy_vs_mi_benchpress.pdf" \
            "$py" eval_benchpress.py --datasets benchpress "${rep[@]}" "${par[@]}"
    fi
    for ds in "${BS_DATASETS[@]}" "${BS_EXTRA[@]}"; do
        tag=""
        if bs_in "$ds" "${BS_EXTRA[@]}"; then tag=extra; fi
        if [[ $ds != benchpress ]]; then
            bs_par_if "greedy_cv_$ds" "$ds" "${BS_PARALLEL_BATTERY[@]}"
            bs_job "greedy_cv_$ds" greedy_cv "" \
                "logs/greedy_cv_$ds.npz logs/greedy_cv_splits_$ds.json logs/greedy_cv_paired_$ds.csv figures/greedy_cv_$ds.pdf" \
                "$py" eval_greedy_all.py --datasets "$ds" "${rep[@]}" --selections-json none "${par[@]}"
            bs_par_if "entropy_vs_mi_$ds" "$ds" "${BS_PARALLEL_BATTERY[@]}"
            bs_job "entropy_vs_mi_$ds" entropy_vs_mi "" \
                "logs/entropy_vs_mi_$ds.npz figures/entropy_vs_mi_$ds.pdf" \
                "$py" eval_entropy_vs_mi.py --datasets "$ds" "${rep[@]}" "${par[@]}"
            bs_par_if "mi_selection_$ds" "$ds" "${BS_PARALLEL_BATTERY[@]}"
            bs_job "mi_selection_$ds" mi_selection "" \
                "logs/greedy_mi_cv_$ds.npz logs/greedy_mi_cv_splits_$ds.json figures/selection_order_mi_$ds.pdf" \
                "$py" plot_mi_selection_all.py --datasets "$ds" "${rep[@]}" --selections-json none "${par[@]}"
            [[ -z $tag ]] || bs_tag "$tag" "greedy_cv_$ds" "entropy_vs_mi_$ds" "mi_selection_$ds"
        fi
        # baselines: the seven selectors plus facility location (facloc,
        # the headline method) and its similarity ablation (facloc_abs,
        # facloc_sq); datasets with columns redundant by construction (an item_group
        # in data/<ds>.meta.json: olv2sub) also get the redundancy summary
        outs="logs/baselines_$ds.npz logs/baselines_${ds}_paired.csv figures/baselines_$ds.pdf"
        if [[ -f $BS_DATA_DIR/$ds.meta.json ]] &&
           grep -q '"item_group"' "$BS_DATA_DIR/$ds.meta.json"; then
            outs+=" logs/baselines_${ds}_redundancy.json"
        fi
        bs_par_if "baselines_$ds" "$ds" "${BS_PARALLEL_BATTERY[@]}"
        bs_job "baselines_$ds" baselines "" "$outs" \
            "$py" eval_baselines.py --datasets "$ds" "${rep[@]}" "${par[@]}"
        bs_job "family_$ds" family "" \
            "logs/family_holdout_$ds.npz figures/family_holdout_$ds.pdf" \
            "$py" eval_family_holdout.py --datasets "$ds" "${rep[@]}"
        [[ -z $tag ]] || bs_tag "$tag" "baselines_$ds" "family_$ds"
        if [[ -n ${BS_LOGIT_SKIP[$ds]:-} ]]; then
            continue
        fi
        bs_par_if "logit_$ds" "$ds" "${BS_PARALLEL_LOGIT[@]}"
        bs_job "logit_$ds" logit "" \
            "logs_logit/greedy_cv_${ds}_logit.npz logs_logit/entropy_vs_mi_${ds}_logit.npz figures_logit/greedy_cv_${ds}_logit.pdf figures_logit/entropy_vs_mi_${ds}_logit.pdf" \
            "$py" eval_logit.py --datasets "$ds" "${rep[@]}" --no-eigenspectrum "${par[@]}"
        [[ -z $tag ]] || { bs_tag "$tag" "logit_$ds"; logit_extra+=("$ds"); }
    done
    # Pre-fix Merged (data/merged_v1.*, data/README.md): the baselines once
    # more, for the effect of the Open LLM data fix (RsqBaseMergedVOne...)
    if bs_in merged "${BS_DATASETS[@]}" && [[ -f $BS_DATA_DIR/merged_v1.matrix.npz ]]; then
        bs_par_if baselines_merged_v1 merged_v1 "${BS_PARALLEL_BATTERY[@]}"
        bs_job baselines_merged_v1 baselines "" \
            "logs/baselines_merged_v1.npz logs/baselines_merged_v1_paired.csv figures/baselines_merged_v1.pdf" \
            "$py" eval_baselines.py --datasets merged_v1 "${rep[@]}" "${par[@]}"
        bs_tag datafix baselines_merged_v1
        v1=merged_v1
    fi
    # One reviewer_checks job: the checks merge their results into one JSON
    # file, which the driver deletes before a job starts.  clip and mono run
    # on the matched splits of every dataset (one paired CSV each), contam
    # on the full MMLU matrix and, with greedy trace, hub and facility
    # location ranked too and
    # 20 random contaminated subsets, on MTEB and Merged when they are in
    # DATASETS (contam_mteb, contam_merged; eval_reviewer_checks.py's
    # default --contam-dataset).  The splits of clip and mono and the
    # contaminated estimates of contam run in WORKERS processes.
    bs_par reviewer_checks
    bs_job reviewer_checks reviewer_checks "" \
        "logs/reviewer_checks.json $(bs_each 'logs/reviewer_checks_clip_%s_paired.csv' "${BS_DATASETS[@]}") $(bs_each 'logs/reviewer_checks_clip_%s.npz' "${BS_DATASETS[@]}") $(bs_each 'logs/reviewer_checks_mono_%s_paired.csv' "${BS_DATASETS[@]}")" \
        "$py" eval_reviewer_checks.py --check all --datasets "${BS_DATASETS[@]}" "${rep[@]}" \
        --workers "$BS_PW"
    for ds in "${BS_DATASETS[@]}"; do
        [[ -n ${BS_COST_TABLE[$ds]:-} ]] || continue
        read -r -a budgets <<<"${BS_COST_BUDGETS[$ds]}"
        # the facility-location and trace knapsacks (knap_facloc,
        # card_facloc, knap_trace, card_trace) are always logged; the
        # figure draws the facloc pair, --plot-trace adds the trace pair
        bs_job "cost_$ds" cost "" \
            "logs/cost_budget_$ds.npz logs/cost_budget_${ds}_paired.csv figures/cost_budget_$ds.pdf" \
            "$py" eval_cost_budget.py --dataset "$ds" \
            --counts "$BS_DATA_DIR/${BS_COST_TABLE[$ds]}" \
            --budgets "${budgets[@]}" "${rep[@]}" --plot-trace
        # fixed per-benchmark overhead c0 = {0, 0.1, 0.5, 1, 2} x median cost
        bs_job "cost_fixed_$ds" fixed_cost "" \
            "logs/cost_fixed_$ds.npz logs/cost_fixed_$ds.json logs/cost_fixed_${ds}_paired.csv figures/cost_fixed_$ds.pdf" \
            "$py" eval_cost_budget.py --dataset "$ds" \
            --counts "$BS_DATA_DIR/${BS_COST_TABLE[$ds]}" \
            --budgets "${budgets[@]}" "${rep[@]}" --fixed-cost
    done
    bs_job covariance covariance "" \
        "$(bs_each 'cov/%s.cov.npz' mmlu mteb merged)" \
        "$py" estimate_covariance.py
    bs_job eigenspectrum eigenspectrum covariance \
        "figures/eigenspectrum.pdf" \
        "$py" plot_eigenspectrum.py
    BS_INPUTS[eigenspectrum]=$(bs_each 'cov/%s.cov.npz' mmlu mteb merged)
    if (( ${#cv[@]} )); then
        # The joint run's rep-0 selection files from the per-dataset logs
        bs_job greedy_cv_selections greedy_cv "$(bs_each 'greedy_cv_%s' "${cv[@]}")" \
            "logs/greedy_cv_selections.json" \
            "$py" eval_greedy_all.py --datasets "${cv[@]}" --selections-from-logs
        BS_INPUTS[greedy_cv_selections]=$(bs_each 'logs/greedy_cv_splits_%s.json' "${cv[@]}")
        bs_job mi_cv_selections mi_selection "$(bs_each 'mi_selection_%s' "${cv[@]}")" \
            "logs/greedy_mi_cv_selections_all.json" \
            "$py" plot_mi_selection_all.py --datasets "${cv[@]}" --selections-from-logs
        BS_INPUTS[mi_cv_selections]=$(bs_each 'logs/greedy_mi_cv_splits_%s.json' "${cv[@]}")
        # Selection order over all splits: entropy (the paper's figure; MI:
        # mi_selection_<ds>; entropy_rf was dropped, protocol rule D3),
        # facility location (the headline method) and greedy trace, all
        # from the greedy_cv_<ds> selection logs (BS_SEL_ORDER_METHODS)
        bs_job selection_order selection_order "$(bs_each 'greedy_cv_%s' "${cv[@]}")" \
            "$(bs_sel_order_outs "${cv[@]}")" \
            "$py" plot_selection_order.py --datasets "${cv[@]}" "${rep[@]}" \
            --methods "${BS_SEL_ORDER_METHODS[@]}"
        BS_INPUTS[selection_order]=$(bs_each 'logs/greedy_cv_splits_%s.json' "${cv[@]}")
    fi
    # Named logit: make_numbers.py reads the printed logit-space k90/k95
    # from stdout/logit.txt, as when one job ran the whole of eval_logit.py
    bs_job logit logit "" \
        "figures_logit/eigenspectrum_logit.pdf" \
        "$py" eval_logit.py --datasets "${BS_DATASETS[@]}" --eigenspectrum-only
    # The combining jobs of the extended battery, under their own names so
    # that the paper's files stay those of DATASETS: the selection-order
    # figures (per-dataset file names), the logit eigenspectrum (its fixed
    # file name goes to figures_logit/extra/; make_numbers.py reads
    # stdout/logit_extra.txt) and the raw eigenspectrum (EM recomputed for
    # datasets without a cov file; figures/eigenspectrum_<names>.pdf).  The
    # rep-0 selection files (greedy_cv_selections, mi_cv_selections) are
    # the paper's historical format and cover DATASETS only.
    if (( ${#BS_EXTRA[@]} )); then
        bs_job selection_order_extra selection_order "$(bs_each 'greedy_cv_%s' "${BS_EXTRA[@]}")" \
            "$(bs_sel_order_outs "${BS_EXTRA[@]}")" \
            "$py" plot_selection_order.py --datasets "${BS_EXTRA[@]}" "${rep[@]}" \
            --methods "${BS_SEL_ORDER_METHODS[@]}"
        BS_INPUTS[selection_order_extra]=$(bs_each 'logs/greedy_cv_splits_%s.json' "${BS_EXTRA[@]}")
        bs_job eigenspectrum_release eigenspectrum "" \
            "figures/eigenspectrum_$(IFS=_; printf '%s' "${BS_EXTRA[*]}").pdf" \
            "$py" plot_eigenspectrum.py --datasets "${BS_EXTRA[@]}" --recompute-missing
        bs_tag extra selection_order_extra eigenspectrum_release
        if (( ${#logit_extra[@]} )); then
            bs_job logit_extra logit "" \
                "figures_logit/extra/eigenspectrum_logit.pdf" \
                env BENCHSELECT_LOGIT_FIGURES_DIR="$BS_ROOT/figures_logit/extra" \
                "$py" eval_logit.py --datasets "${logit_extra[@]}" --eigenspectrum-only
            bs_tag extra logit_extra
        fi
    fi
    bs_job normality normality "" \
        "logs/normality_summary.json figures/normality_diagnostics.pdf" \
        "$py" run_normality.py
    # Appendix E figure; node and edge counts (and the cross-check against
    # the raw collection tables data/<collection>.csv, if present) in the
    # summary JSON
    bs_job benchmark_overlap overlap "" \
        "figures/benchmark_overlap.pdf logs/benchmark_overlap.json" \
        "$py" plot_benchmark_overlap.py --summary "$BS_ROOT/logs/benchmark_overlap.json"

    # ── Reviewer-request experiments on DATASETS ────────────────────
    # Why MI, entropy, trace and facility location rank as they do (one
    # job: the combined figure needs every dataset; facility location vs
    # trace in figures/analyze_objectives_facloc_<ds>.pdf); the unshrunk
    # full-data oracle differs from the matched one only on Merged (fewer
    # models than benchmarks); the splits run in WORKERS processes
    if (( ${#BS_DATASETS[@]} > 1 )); then
        all_fig=" figures/analyze_objectives_all.pdf"
    fi
    bs_par analyze_objectives
    bs_job analyze_objectives objectives "" \
        "$(bs_each 'logs/analyze_objectives_%s.npz' "${BS_DATASETS[@]}") $(bs_each 'logs/analyze_objectives_%s.json' "${BS_DATASETS[@]}") $(bs_each 'logs/analyze_objectives_%s_paired.csv' "${BS_DATASETS[@]}") $(bs_each 'figures/analyze_objectives_%s.pdf' "${BS_DATASETS[@]}") $(bs_each 'figures/analyze_objectives_facloc_%s.pdf' "${BS_DATASETS[@]}")$all_fig" \
        "$py" analyze_objectives.py --datasets "${BS_DATASETS[@]}" "${rep[@]}" \
        --workers "$BS_PW"
    if bs_in merged "${BS_DATASETS[@]}"; then
        bs_par analyze_objectives_fulloracle
        bs_job analyze_objectives_fulloracle objectives "" \
            "logs/analyze_objectives_merged_fulloracle.npz logs/analyze_objectives_merged_fulloracle.json logs/analyze_objectives_merged_fulloracle_paired.csv figures/analyze_objectives_merged_fulloracle.pdf figures/analyze_objectives_facloc_merged_fulloracle.pdf" \
            "$py" analyze_objectives.py --datasets merged "${rep[@]}" --oracle full \
            --workers "$BS_PW"
    fi
    # Selection stability under training subsamples (B4) and the
    # covariance-estimator ablation (B5); the slow datasets
    # (BS_PARALLEL_DATASETS) run their splits in WORKERS processes
    for ds in "${BS_DATASETS[@]}"; do
        par=()
        if bs_in "$ds" "${BS_PARALLEL_DATASETS[@]}"; then par=(--workers "$BS_WORKERS"); fi
        bs_job "stability_$ds" stability "" \
            "logs/selection_stability_$ds.npz logs/selection_stability_$ds.json logs/selection_stability_${ds}_paired.csv figures/selection_stability_$ds.pdf" \
            "$py" eval_selection_stability.py --datasets "$ds" "${rep[@]}" "${par[@]}"
        (( ${#par[@]} == 0 )) || BS_WEIGHT[stability_$ds]=$BS_WORKERS
        par=()
        if bs_in "$ds" "${BS_PARALLEL_DATASETS[@]}"; then par=(--jobs "$BS_WORKERS"); fi
        bs_job "estimator_$ds" estimator "" \
            "logs/estimator_ablation_$ds.npz logs/estimator_ablation_$ds.json logs/estimator_ablation_${ds}_paired.csv figures/estimator_ablation_$ds.pdf" \
            "$py" eval_estimator_ablation.py --datasets "$ds" "${rep[@]}" "${par[@]}"
        (( ${#par[@]} == 0 )) || BS_WEIGHT[estimator_$ds]=$BS_WORKERS
    done
    if (( ${#BS_DATASETS[@]} > 1 )); then
        bs_job stability_all stability "$(bs_each 'stability_%s' "${BS_DATASETS[@]}")" \
            "figures/selection_stability_all.pdf" \
            "$py" eval_selection_stability.py --datasets "${BS_DATASETS[@]}" --all-figure
        BS_INPUTS[stability_all]=$(bs_each 'logs/selection_stability_%s.npz' "${BS_DATASETS[@]}")
    fi
    # Fitted matrix-completion imputers (B1), in the default run: one shard
    # job per CV repetition (imputers_<ds>_rep<r>, --reps r: the per-split
    # arrays of repetition r in logs/imputers_shards/), then the merge
    # (imputers_<ds>, --merge: concatenates the shards in repetition order
    # into the single-process log, bit for bit, and writes its JSON, CSV
    # and figure).  Merged's 50 splits take ~7.9 CPU-hours with every core
    # busy, i.e. ~1.6 h per shard in one process; each shard runs its 10
    # splits in WORKERS processes (at most BS_SHARD_SPLITS), so that the
    # slowest split bounds it.
    for ds in "${BS_DATASETS[@]}"; do
        shards=()
        for (( r = 0; r < BS_REPEATS; r++ )); do
            bs_par "imputers_${ds}_rep$r" "$BS_SHARD_SPLITS"
            bs_job "imputers_${ds}_rep$r" imputers "" \
                "logs/imputers_shards/imputers_${ds}_rep$r.npz" \
                "$py" eval_imputers.py --datasets "$ds" "${rep[@]}" --reps "$r" \
                --workers "$BS_PW"
            shards+=("imputers_${ds}_rep$r")
        done
        bs_job "imputers_$ds" imputers "${shards[*]}" \
            "logs/imputers_$ds.npz logs/imputers_$ds.json logs/imputers_${ds}_paired.csv figures/imputers_$ds.pdf" \
            "$py" eval_imputers.py --datasets "$ds" "${rep[@]}" --merge
        BS_INPUTS[imputers_$ds]=$(for (( r = 0; r < BS_REPEATS; r++ )); do
                                     printf 'logs/imputers_shards/imputers_%s_rep%d.npz ' "$ds" "$r"; done)
        BS_INPUTS[imputers_$ds]=${BS_INPUTS[imputers_$ds]% }
    done
    if (( ${#BS_DATASETS[@]} > 1 )); then
        # --from-logs recomputes the summaries of every dataset from its npz
        # (same JSON, CSV and figures) and adds the combined figure
        bs_job imputers_all imputers "$(bs_each 'imputers_%s' "${BS_DATASETS[@]}")" \
            "figures/imputers_all.pdf" \
            "$py" eval_imputers.py --datasets "${BS_DATASETS[@]}" --from-logs
        BS_INPUTS[imputers_all]=$(bs_each 'logs/imputers_%s.npz' "${BS_DATASETS[@]}")
    fi
    # Benchmark-family holdout (B3): DATASETS and DATASETS_EXTRA that have
    # a family scheme; blind arm re-estimated without the family (refit,
    # the default), the sub sensitivity on the EM datasets, then refit -
    # sub
    for ds in "${BS_BENCHFAM_DATASETS[@]}"; do
        bs_in "$ds" "${BS_DATASETS[@]}" "${BS_EXTRA[@]}" || continue
        par=()
        if bs_in "$ds" "${BS_PARALLEL_BENCHFAM[@]}"; then par=(--workers "$BS_WORKERS"); fi
        bs_job "benchfam_$ds" benchfam "" \
            "logs/benchfamily_holdout_$ds.npz logs/benchfamily_holdout_$ds.json logs/benchfamily_holdout_${ds}_paired.csv figures/benchfamily_holdout_$ds.pdf" \
            "$py" eval_benchfamily_holdout.py --datasets "$ds" "${rep[@]}" "${par[@]}"
        (( ${#par[@]} == 0 )) || BS_WEIGHT[benchfam_$ds]=$BS_WORKERS
        if bs_in "$ds" "${BS_EXTRA[@]}"; then bs_tag extra "benchfam_$ds"; fi
        bs_in "$ds" "${BS_BENCHFAM_SUB[@]}" || continue
        famsub+=("$ds")
        bs_job "benchfam_${ds}_sub" benchfam "" \
            "logs/benchfamily_holdout_${ds}_sub.npz logs/benchfamily_holdout_${ds}_sub.json logs/benchfamily_holdout_${ds}_sub_paired.csv figures/benchfamily_holdout_${ds}_sub.pdf" \
            "$py" eval_benchfamily_holdout.py --datasets "$ds" "${rep[@]}" --blind-cov sub "${par[@]}"
        (( ${#par[@]} == 0 )) || BS_WEIGHT[benchfam_${ds}_sub]=$BS_WORKERS
    done
    if (( ${#famsub[@]} )); then
        bs_job benchfam_blindcov benchfam \
            "$(bs_each 'benchfam_%s' "${famsub[@]}") $(bs_each 'benchfam_%s_sub' "${famsub[@]}")" \
            "$(bs_each 'logs/benchfamily_holdout_%s_blindcov.json' "${famsub[@]}") $(bs_each 'logs/benchfamily_holdout_%s_blindcov_paired.csv' "${famsub[@]}")" \
            "$py" eval_benchfamily_holdout.py --datasets "${famsub[@]}" --compare-blind-cov
        BS_INPUTS[benchfam_blindcov]="$(bs_each 'logs/benchfamily_holdout_%s.npz' "${famsub[@]}") $(bs_each 'logs/benchfamily_holdout_%s_sub.npz' "${famsub[@]}")"
    fi

    # ── Fixed experiments (their own data, independent of DATASETS) ──
    # Prospective evaluation: selection on old models, new models as
    # targets (MTEB snapshot -> 2026, HELM v1.0 -> v1.13, Epoch and OLL v2
    # by date) and the two sensitivity settings; then the combined figure
    # (--plot-only re-renders the four settings' figures from their logs)
    for s in "${BS_PROSPECTIVE[@]}" "${BS_PROSPECTIVE_SENS[@]}"; do
        par=()
        if bs_in "$s" "${BS_PARALLEL_PROSPECTIVE[@]}"; then par=(--workers "$BS_WORKERS"); fi
        bs_job "prospective_$s" prospective "" \
            "logs/prospective_$s.npz logs/prospective_$s.json figures/prospective_$s.pdf" \
            "$py" eval_prospective.py --datasets "$s" "${rep[@]}" "${par[@]}"
        (( ${#par[@]} == 0 )) || BS_WEIGHT[prospective_$s]=$BS_WORKERS
    done
    bs_job prospective_plot prospective "$(bs_each 'prospective_%s' "${BS_PROSPECTIVE[@]}")" \
        "figures/prospective.pdf" \
        "$py" eval_prospective.py --datasets "${BS_PROSPECTIVE[@]}" --plot-only
    BS_INPUTS[prospective_plot]=$(bs_each 'logs/prospective_%s.npz' "${BS_PROSPECTIVE[@]}")
    # Whole tasks vs uniformly random items at matched question budgets
    # (OLL v1 items), and with a fixed per-task overhead C0
    bs_job item_budget item_budget "" \
        "logs/item_budget_olv1items.npz logs/item_budget_olv1items.json logs/item_budget_olv1items_paired.csv figures/item_budget_olv1items.pdf" \
        "$py" eval_item_budget.py "${rep[@]}"
    bs_job item_fixed fixed_cost "" \
        "logs/item_fixed_olv1items.npz logs/item_fixed_olv1items.json logs/item_fixed_olv1items_paired.csv figures/item_fixed_olv1items.pdf" \
        "$py" eval_item_budget.py "${rep[@]}" --fixed-cost
    # True token budgets on HELM Lite / MMLU / Classic (OpenbookQA's cost
    # imputed, the default), the exclude sensitivity on HELM Classic, and
    # the fixed per-benchmark overhead
    bs_job token_costs token_costs "" \
        "$(bs_each 'logs/token_costs_%s.npz' helm_lite helm_mmlu helm_classic) $(bs_each 'logs/token_costs_%s.json' helm_lite helm_mmlu helm_classic) $(bs_each 'logs/token_costs_%s_paired.csv' helm_lite helm_mmlu helm_classic) figures/token_costs.pdf figures/token_costs_suite.pdf" \
        "$py" eval_token_costs.py "${rep[@]}"
    bs_job token_costs_exclude token_costs "" \
        "logs/token_costs_helm_classic_exclude.npz logs/token_costs_helm_classic_exclude.json logs/token_costs_helm_classic_exclude_paired.csv figures/token_costs_exclude.pdf figures/token_costs_suite_exclude.pdf" \
        "$py" eval_token_costs.py --datasets helm_classic "${rep[@]}" --cost-caveats exclude
    bs_job token_fixed fixed_cost "" \
        "$(bs_each 'logs/token_fixed_%s.npz' helm_lite helm_mmlu helm_classic) $(bs_each 'logs/token_fixed_%s.json' helm_lite helm_mmlu helm_classic) $(bs_each 'logs/token_fixed_%s_paired.csv' helm_lite helm_mmlu helm_classic) figures/token_fixed.pdf" \
        "$py" eval_token_costs.py "${rep[@]}" --fixed-cost

    # Paper-number macros and paired statistics (numbers/paired_<ds>.csv,
    # paired_stats.py) from the logs of every job above, the imputers
    # included (plus logs/tabimpute_*.npz if run_tabimpute.sh has written
    # them into this root: then run RUN_ONLY=numbers again).  Missing
    # files are skipped with a warning.  numbers.tex holds the paper's
    # DATASETS (group main), numbers_<group>.tex the data-release datasets
    # and each experiment (make_numbers.py --tex-dir).
    for ds in "${BS_DATASETS[@]}" "${BS_EXTRA[@]}" \
              ${v1} \
              "${BS_FIXED_DATA[@]}"; do
        bs_in "$ds" "${ndata[@]}" || ndata+=("$ds")
    done
    for ds in "${BS_DATASETS[@]}" "${BS_EXTRA[@]}" \
              ${v1} \
              mteb2026 helm_lite epoch olv2sub; do
        bs_in "$ds" "${paired[@]}" || paired+=("$ds")
    done
    bs_job numbers numbers "${BS_JOBS[*]}" \
        "numbers/numbers.tex numbers/numbers.csv $(bs_each 'numbers/numbers_%s.tex' "${BS_TEX_GROUPS[@]}") $(bs_each 'numbers/paired_%s.csv' "${paired[@]}")" \
        "$py" make_numbers.py --root "$BS_ROOT" --cov-dir "$BS_ROOT/cov" \
        --data-dir "$BS_DATA_DIR" --datasets "${ndata[@]}" \
        --out-tex "$BS_ROOT/numbers/numbers.tex" \
        --out-csv "$BS_ROOT/numbers/numbers.csv" \
        --tex-dir "$BS_ROOT/numbers" \
        --paired-dir "$BS_ROOT/numbers"
    # Opt-in (RUN_ONLY=tests).  pytest reports its own failures, so its
    # output is not scanned.  test_tabimpute_sanity.py is a GPU script, not
    # a pytest module.
    bs_job tests tests "" "" \
        "$py" -m pytest -q -p no:cacheprovider --ignore=test_tabimpute_sanity.py .
    BS_OPTIN[tests]=1
    BS_NOSCAN[tests]=1
    bs_estimates
}

BS_START=$(bs_now)
BS_ACCEPT_REPEATS=1
bs_parse_args "$@"
bs_init_paths
bs_init_datasets
bs_init_extra_datasets
bs_init_workers
if (( BS_LIST )); then
    BS_ROOT='$BENCHSELECT_EXPERIMENT_ROOT'   # job commands are not listed
    bs_define_cpu_jobs
    printf '# --repeats %s; DATASETS: %s; DATASETS_EXTRA: %s; MAX_JOBS %s; WORKERS %s\n' \
        "$BS_REPEATS" "${BS_DATASETS[*]}" "${BS_EXTRA[*]:-none}" "$BS_MAX_JOBS" \
        "$BS_WORKERS"
    bs_list_jobs
    exit 0
fi
bs_init_root
bs_define_cpu_jobs
bs_select
if [[ -n ${BENCHSELECT_DATASETS:-} ]]; then
    bs_log "note: ignoring BENCHSELECT_DATASETS=$BENCHSELECT_DATASETS (run_tabimpute.sh's list); run_all.sh reads DATASETS"
fi
bs_export_env
bs_setenv CUDA_VISIBLE_DEVICES ""   # CPU only: no job may touch the GPUs
BS_REQUIREMENTS=$BS_REPO/requirements.txt
BS_MANIFEST=$BS_ROOT/manifest.json
if (( BS_DRY_RUN )); then
    bs_print_plan
    exit 0
fi
bs_prepare_root --param "repeats=$BS_REPEATS" --param "datasets=${BS_DATASETS[*]}" \
    --param "datasets_extra=${BS_EXTRA[*]:-}" --param "workers=$BS_WORKERS" \
    --param "max_jobs=$BS_MAX_JOBS"
bs_run_selected
rc=0
bs_finish || rc=$?
exit "$rc"
