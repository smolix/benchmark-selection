#!/usr/bin/env python3
"""Build the per-benchmark evaluation-cost tables used by code/eval_cost_budget.py.

Produces:
    mmlu_test_counts.json  — {subject: test questions} for the 57 MMLU subjects
    mteb_task_sizes.json   — {task: mteb num_samples} for the 56 MTEB tasks

The benchmark lists are the ``benchmark_names`` of mmlu.matrix.npz and
mteb.matrix.npz, so every column of the score matrices gets a cost.  Both
files are written as ``json.dumps(table, indent=1, sort_keys=True)`` with
integer values and no trailing newline.  With the default ``--mteb-scope
benchmark`` the output is byte-identical to the checked-in tables;
``--mteb-scope task`` reproduces the all-language MTEB table used by the
rebuttal-era (July 2026) cost-budget runs.  All requested tables are computed before any
file is written, so a failed run leaves the existing files untouched.

Requirements: a separate venv built from requirements-data.txt at the
repository root (see the install commands there), not ~/venvs/benchselect.

Network: the MMLU table reads cais/mmlu metadata from the HuggingFace Hub
and fails without Hub access; the MTEB table needs no network, so
``--only mteb`` runs offline.

Provenance
----------
MMLU: ``num_examples`` of the ``test`` split of each subject config of
    cais/mmlu on the HuggingFace Hub, pinned to MMLU_REVISION.
    ``datasets.load_dataset_builder`` reads it from the repository's
    metadata files, the ``dataset_info`` block of the dataset card
    (README.md) taking precedence over the legacy dataset_infos.json; at
    the pinned revision both agree for all 57 subjects.  Only these two
    files (~200 KB) are fetched, no questions.  The 57 subjects sum to the
    14,042 questions of the ``all`` test split.

MTEB: ``num_samples`` from the descriptive-statistics JSON shipped inside
    the mteb package (mteb/descriptive_stats/<type>/<task>.json), so the
    table depends on the installed mteb version (pinned: MTEB_VERSION; any
    other version prints a warning), not on the Hub.  What one sample is
    depends on the task type, so the unit is neither "test examples" nor
    "texts embedded":
        Retrieval, Reranking      corpus documents + queries
                                  (MSMARCO dev: 8,841,823 + 6,980 = 8,848,803)
        Clustering                texts, summed over all clustering sets
        Classification            evaluation texts; the train-split sample
                                  embedded to fit the classifier is not
                                  counted
        STS, PairClassification   sentence pairs (two texts each)
        Summarization             source texts (SummEval: 100), whereas
                                  MTEB embeds the 1,100 human and 1,548
                                  machine summaries
    CQADupstackRetrieval is an aggregate task with no statistics of its
    own; its size is the sum over its 12 CQADupstack*Retrieval subtasks.

    ``--mteb-scope`` selects which part of each task is counted:
        benchmark  (default) only the eval splits and hf_subsets selected by
                   the MTEB(eng, classic) benchmark (MTEB_BENCHMARK), which
                   are the scores in mteb.matrix.npz: data/download_mteb.py
                   loads them with ``mteb.load_results(tasks=
                   benchmark.tasks)``, which keeps only these splits and
                   subsets.  This reproduces the checked-in
                   mteb_task_sizes.json (total 40,232,830).
        task       every language subset and every default eval split of
                   ``mteb.get_task(name)``, i.e. ``TaskMetadata.n_samples``
                   summed over ``TaskMetadata.eval_splits`` (total
                   40,858,920).  This was the table of the rebuttal-era
                   runs; it overstates the cost of the multilingual tasks
                   relative to what the scores measure.
    The scopes differ only for eight multilingual tasks, of which the
    benchmark scores the English test subset alone:
                                              task  benchmark
        AmazonCounterfactualClassification   5,805        670
        AmazonReviewsClassification         60,000      5,000
        MTOPDomainClassification            30,517      4,386
        MTOPIntentClassification            30,517      4,386
        MassiveIntentClassification        255,357      2,974
        MassiveScenarioClassification      255,357      2,974
        STS17 (en-en)                        5,346        250
        STS22 (en)                           4,028        197
    The cost-budget results for MTEB depend on this choice.

Usage:
    python build_cost_tables.py [--input-dir DIR] [--output-dir DIR]
                                [--only {mmlu,mteb}] [--mmlu-revision REV]
                                [--mteb-scope {task,benchmark}]
"""

import json
import logging
import pathlib
import warnings

import numpy as np

DEFAULT_DATA = pathlib.Path(__file__).resolve().parent

# Optional HF_TOKEN (higher Hub rate limits); cais/mmlu itself is public.
try:
    from dotenv import load_dotenv
    load_dotenv(DEFAULT_DATA / ".env")
except ImportError:
    pass

MMLU_REPO = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"  # Hub commit, 2024-03-08
MMLU_SPLIT = "test"

MTEB_VERSION = "2.18.7"                 # as pinned in requirements-data.txt
MTEB_BENCHMARK = "MTEB(eng, classic)"   # as in download_mteb.py
MTEB_SCOPES = ("task", "benchmark")

MMLU_FILE = "mmlu_test_counts.json"
MTEB_FILE = "mteb_task_sizes.json"


def benchmark_names(input_dir, name):
    """Column names of {name}.matrix.npz, in matrix order."""
    with np.load(input_dir / f"{name}.matrix.npz") as z:
        return [str(b) for b in z["benchmark_names"]]


def mmlu_test_counts(subjects, revision=MMLU_REVISION):
    """Test-split question count per MMLU subject, from Hub dataset-info metadata."""
    from datasets import load_dataset_builder
    from tqdm import tqdm

    counts = {}
    for subject in tqdm(subjects, desc="MMLU subjects", unit="subject"):
        splits = load_dataset_builder(MMLU_REPO, subject, revision=revision).info.splits
        if not splits or MMLU_SPLIT not in splits:
            raise RuntimeError(f"{MMLU_REPO}/{subject}@{revision}: "
                               f"no split info for '{MMLU_SPLIT}'")
        counts[subject] = int(splits[MMLU_SPLIT].num_examples)
    return counts


def _mteb_task_size(task, scope="task"):
    """num_samples of one task in the given scope; aggregate tasks sum their subtasks.

    scope "task" sums the split totals over TaskMetadata.eval_splits (all
    hf_subsets); scope "benchmark" sums the per-subset counts over the
    task object's own eval_splits and hf_subsets, which a benchmark narrows.
    """
    if getattr(task, "is_aggregate", False):
        return sum(_mteb_task_size(sub, scope) for sub in task.metadata.tasks)
    md = task.metadata
    stats = md.descriptive_stats
    if not stats:
        raise RuntimeError(f"mteb has no descriptive statistics for {md.name}")
    if scope == "task":
        return sum(int(md.n_samples[split]) for split in md.eval_splits)

    size = 0
    for split in task.eval_splits:
        subset_stats = stats[split].get("hf_subset_descriptive_stats")
        if subset_stats is None:
            # Single-subset task: the split total is the subset count.
            if list(task.hf_subsets) != ["default"]:
                raise RuntimeError(f"{md.name}/{split}: no per-subset statistics "
                                   f"for hf_subsets {task.hf_subsets}")
            size += int(stats[split]["num_samples"])
        else:
            size += sum(int(subset_stats[s]["num_samples"]) for s in task.hf_subsets)
    return size


def _report_warnings(caught, task_names=None):
    """Print the warnings raised by mteb, or only those naming one of task_names.

    mteb.get_benchmark builds every registered benchmark and warns about
    unrelated beta tasks; with task_names given those are only counted.
    """
    relevant = {str(w.message) for w in caught
                if task_names is None or any(n in str(w.message) for n in task_names)}
    for msg in sorted(relevant):
        print(f"  WARNING (mteb): {msg}")
    n_other = sum(1 for w in caught if str(w.message) not in relevant)
    if n_other:
        print(f"  ({n_other} mteb warnings about tasks outside the table ignored)")


def mteb_task_sizes(task_names, scope="task"):
    """mteb num_samples per task (see the module docstring for both scopes)."""
    import mteb

    if scope not in MTEB_SCOPES:
        raise ValueError(f"scope must be one of {MTEB_SCOPES}, got {scope!r}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if scope == "task":
            tasks = {name: mteb.get_task(name) for name in task_names}
        else:
            mteb_logger = logging.getLogger("mteb")
            level = mteb_logger.level
            mteb_logger.setLevel(logging.ERROR)  # beta-task notices of other benchmarks
            try:
                benchmark = mteb.get_benchmark(MTEB_BENCHMARK)
            finally:
                mteb_logger.setLevel(level)
            tasks = {t.metadata.name: t for t in benchmark.tasks}
            missing = [name for name in task_names if name not in tasks]
            if missing:
                raise RuntimeError(f"{MTEB_BENCHMARK} has no task {missing[:5]}")
    _report_warnings(caught, task_names if scope == "benchmark" else None)

    sizes = {}
    for name in task_names:
        task = tasks[name]
        if task.metadata.name != name:
            raise RuntimeError(f"mteb.get_task({name!r}) resolved to "
                               f"{task.metadata.name!r}")
        sizes[name] = _mteb_task_size(task, scope)
    return sizes


def write_table(table, path):
    """Write {benchmark: int} as indent-1, key-sorted JSON (no trailing newline)."""
    path.write_text(json.dumps(table, indent=1, sort_keys=True))
    lo = min(table, key=table.get)
    hi = max(table, key=table.get)
    print(f"  {path.name}: {len(table)} entries, total {sum(table.values()):,}, "
          f"range {table[lo]:,} ({lo}) – {table[hi]:,} ({hi})")
    print(f"  Saved {path}")


def build_cost_tables(input_dir, output_dir, only=None, mmlu_revision=MMLU_REVISION,
                      mteb_scope="task"):
    """Build both cost tables (or only one of them) into output_dir."""
    input_dir = pathlib.Path(input_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading benchmark lists from {input_dir}/")
    print(f"Writing cost tables to {output_dir}/")

    tables = {}
    if only in (None, "mmlu"):
        subjects = benchmark_names(input_dir, "mmlu")
        print(f"MMLU: {len(subjects)} subjects, {MMLU_REPO}@{mmlu_revision[:12]} "
              f"'{MMLU_SPLIT}' split (needs Hub access)")
        tables[MMLU_FILE] = mmlu_test_counts(subjects, mmlu_revision)

    if only in (None, "mteb"):
        import mteb
        tasks = benchmark_names(input_dir, "mteb")
        detail = f" ({MTEB_BENCHMARK} splits and subsets)" if mteb_scope == "benchmark" else ""
        print(f"MTEB: {len(tasks)} tasks, mteb {mteb.__version__} task metadata, "
              f"scope '{mteb_scope}'{detail}")
        if mteb.__version__ != MTEB_VERSION:
            print(f"  WARNING: mteb {mteb.__version__} is not the pinned {MTEB_VERSION} "
                  f"(requirements-data.txt); its task statistics may differ.")
        tables[MTEB_FILE] = mteb_task_sizes(tasks, mteb_scope)

    for filename, table in tables.items():
        write_table(table, output_dir / filename)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        default=DEFAULT_DATA,
        help="Directory containing mmlu.matrix.npz and mteb.matrix.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Directory for the JSON cost tables. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--only",
        choices=["mmlu", "mteb"],
        help="Build only one of the two tables (mteb needs no network).",
    )
    parser.add_argument(
        "--mmlu-revision",
        default=MMLU_REVISION,
        help=f"Hub revision of {MMLU_REPO} (default: the pinned commit).",
    )
    parser.add_argument(
        "--mteb-scope",
        choices=MTEB_SCOPES,
        default="benchmark",
        help=f"MTEB samples counted: 'benchmark' = only the splits and subsets "
             f"{MTEB_BENCHMARK} scores (default; the checked-in table), 'task' = "
             "all subsets and default eval splits (rebuttal-era table).",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    return args


if __name__ == "__main__":
    args = parse_args()
    build_cost_tables(args.input_dir, args.output_dir, args.only, args.mmlu_revision,
                      args.mteb_scope)
