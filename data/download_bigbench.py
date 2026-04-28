#!/usr/bin/env python3
"""Download and assemble the BIG-Bench score matrix from published results.

Produces:
    bigbench.csv   — tidy CSV (rows = models, columns = tasks)
    bigbench.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install pandas numpy requests

BIG-Bench publishes model evaluation results in its GitHub repository.
We download the aggregated results JSON/CSV from the BIG-Bench repo.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent

# BIG-Bench results are stored in the GitHub repo under benchmark_tasks/
# The most accessible format is the "results" directory with per-model JSONs.
# We use the BIG-Bench Lite subset (24 tasks) for a manageable matrix.

BIGBENCH_RESULTS_BASE = (
    "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks"
)

# BIG-Bench Lite task list (the curated 24-task subset)
BIGBENCH_LITE_TASKS = [
    "auto_debugging",
    "bbq_lite_json",
    "code_line_description",
    "conceptual_combinations",
    "conlang_translation",
    "emoji_movie",
    "formal_fallacies_syllogisms_negation",
    "hindu_knowledge",
    "known_unknowns",
    "language_identification",
    "linguistics_puzzles",
    "logic_grid_puzzle",
    "logical_deduction",
    "misconceptions_russian",
    "novel_concepts",
    "operators",
    "parsinlu_reading_comprehension",
    "play_dialog_same_or_different",
    "repeat_copy_logic",
    "strange_stories",
    "strategyqa",
    "symbol_interpretation",
    "vitaminc_fact_verification",
    "winowhy",
]


def _load_env() -> None:
    """Load variables from a .env file next to this script (if present)."""
    import os
    env_path = HERE / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle both "KEY=val" and "export KEY=val"
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            if key and value:
                os.environ.setdefault(key.strip(), value.strip())


def _github_headers() -> dict:
    """Return GitHub API headers, using GITHUB_TOKEN env var if available."""
    import os
    _load_env()
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _build_score_file_index() -> dict[str, list[str]]:
    """Use a SINGLE GitHub Trees API call to find all score files.

    Returns a dict mapping task_name -> list of raw download URLs.
    This replaces the old approach of 24 separate Contents API calls,
    avoiding GitHub's 60-req/hr unauthenticated rate limit.
    """
    tree_url = (
        "https://api.github.com/repos/google/BIG-bench/git/trees/main?recursive=1"
    )
    headers = _github_headers()
    resp = requests.get(tree_url, headers=headers, timeout=60)
    if resp.status_code != 200:
        tqdm.write(f"  Git Trees API returned {resp.status_code} "
                   f"(rate-limit remaining: {resp.headers.get('X-RateLimit-Remaining', '?')})")
        return {}

    tree_data = resp.json()
    if tree_data.get("truncated", False):
        tqdm.write("  Warning: tree response was truncated; some files may be missing.")

    task_set = set(BIGBENCH_LITE_TASKS)
    index: dict[str, list[str]] = {t: [] for t in BIGBENCH_LITE_TASKS}

    for item in tree_data.get("tree", []):
        path = item.get("path", "")
        # Match:  bigbench/benchmark_tasks/{task}/results/scores_*.json
        if not path.startswith("bigbench/benchmark_tasks/"):
            continue
        parts = path.split("/")
        if (len(parts) == 5
                and parts[2] in task_set
                and parts[3] == "results"
                and parts[4].startswith("scores")
                and parts[4].endswith(".json")):
            task = parts[2]
            raw_url = f"{BIGBENCH_RESULTS_BASE}/{task}/results/{parts[4]}"
            index[task].append(raw_url)

    return index


def _extract_score(data: dict | list, task: str, url: str) -> dict | None:
    """Extract a single (model, task, score) record from a BIG-Bench JSON file.

    The actual schema is::

        {
          "model": {"model_family": "BIG-G sparse", "model_name": "125m", ...},
          "scores": [
            {
              "number_of_shots": 3,
              "preferred_score": "exact_str_match",   # <-- metric NAME
              "score_dict": {"exact_str_match": 0.42, ...},
              ...
            },
            ...
          ],
          "task": { ... }
        }

    We pick the entry with the highest number_of_shots and read
    score_dict[preferred_score] as the numeric value.
    """
    # Derive model name from the filename in the URL
    fname = url.rsplit("/", 1)[-1]
    model_name = fname.replace("scores_", "").replace(".json", "")

    if isinstance(data, dict) and "scores" in data:
        # Use model metadata if available for a nicer name
        model_info = data.get("model", {})
        if isinstance(model_info, dict):
            family = model_info.get("model_family", "")
            name = model_info.get("model_name", "")
            if family and name:
                model_name = f"{family} {name}"

        scores_list = data["scores"]
        if not isinstance(scores_list, list) or not scores_list:
            return None

        # Pick the entry with the most shots (typically the best result)
        best = max(scores_list, key=lambda e: e.get("number_of_shots", 0))
        metric_name = best.get("preferred_score", "")
        score_dict = best.get("score_dict", {})
        value = score_dict.get(metric_name)

        # Fall back to normalized_aggregate_score if preferred metric missing
        if value is None:
            value = score_dict.get("normalized_aggregate_score")

        if value is not None and isinstance(value, (int, float)):
            return {"model": model_name, "task": task, "score": float(value)}

    return None


def main():
    # ── Skip if REAL data already exists (not empty placeholders) ──
    csv_path = HERE / "bigbench.csv"
    npz_path = HERE / "bigbench.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"BIG-Bench data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} tasks).  Skipping download.")
            return
        else:
            print("Found empty placeholder files from a previous failed run. Re-downloading …")

    print("Downloading BIG-Bench Lite results …")

    # ------------------------------------------------------------------
    # Strategy 1: Single Trees API call + raw.githubusercontent downloads
    # ------------------------------------------------------------------
    all_records = []

    print("  Building file index via Git Trees API (single API call) …")
    score_index = _build_score_file_index()
    total_files = sum(len(v) for v in score_index.values())

    if total_files > 0:
        print(f"  Found {total_files} score files across "
              f"{sum(1 for v in score_index.values() if v)} tasks.")

        for task in tqdm(BIGBENCH_LITE_TASKS, desc="Tasks", unit="task"):
            urls = score_index.get(task, [])
            for url in urls:
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    record = _extract_score(data, task, url)
                    if record:
                        all_records.append(record)
                except Exception:
                    continue

            n_models = len([r for r in all_records if r["task"] == task])
            tqdm.write(f"  {task}: {n_models} models")

    if not all_records:
        # ------------------------------------------------------------------
        # Strategy 2: Fall back to per-task Contents API (old approach)
        # ------------------------------------------------------------------
        print("  Trees API yielded no results. Falling back to per-task API …")
        headers = _github_headers()
        for task in tqdm(BIGBENCH_LITE_TASKS, desc="Tasks (fallback)", unit="task"):
            api_url = (
                f"https://api.github.com/repos/google/BIG-bench/contents/"
                f"bigbench/benchmark_tasks/{task}/results"
            )
            try:
                resp = requests.get(api_url, headers=headers, timeout=30)
                if resp.status_code == 403:
                    remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                    tqdm.write(f"  {task}: rate-limited (remaining={remaining})")
                    continue
                if resp.status_code != 200:
                    tqdm.write(f"  {task}: HTTP {resp.status_code}")
                    continue
                files = resp.json()
                if not isinstance(files, list):
                    continue
                json_files = [f for f in files
                              if f["name"].endswith(".json")
                              and f["name"].startswith("scores")]
                for f in json_files:
                    try:
                        raw_url = f["download_url"]
                        data = requests.get(raw_url, timeout=30).json()
                        record = _extract_score(data, task, raw_url)
                        if record:
                            all_records.append(record)
                    except Exception:
                        continue
                n_models = len([r for r in all_records if r["task"] == task])
                tqdm.write(f"  {task}: {n_models} models")
            except Exception as e:
                tqdm.write(f"  {task}: error ({e})")

    if not all_records:
        print("WARNING: Could not download BIG-Bench results.")
        print("  The BIG-Bench results are distributed across many files in")
        print("  the GitHub repo.  To download manually:")
        print("    git clone https://github.com/google/BIG-bench.git")
        print("    # then run this script's assembly logic on the local clone")
        print()
        print("  Saving empty placeholder files.")

        pd.DataFrame().to_csv(HERE / "bigbench.csv")
        np.savez(
            HERE / "bigbench.npz",
            B=np.empty((0, 0)),
            O=np.empty((0, 0)),
            model_names=np.array([]),
            benchmark_names=np.array([]),
        )
        return

    # Pivot to matrix form
    df_long = pd.DataFrame(all_records)
    df = df_long.pivot_table(
        index="model", columns="task", values="score", aggfunc="first"
    )

    # Drop tasks with very few models
    min_obs = max(3, len(df) * 0.1)
    df = df.loc[:, df.notna().sum() >= min_obs]
    df = df.dropna(how="all")

    print(f"\n  Score matrix: {df.shape[0]} models × {df.shape[1]} tasks")
    n_missing = df.isna().sum().sum()
    n_total = df.shape[0] * df.shape[1]
    if n_total > 0:
        print(f"  Missing entries: {n_missing}/{n_total} ({n_missing/n_total:.1%})")

    # Save
    csv_path = HERE / "bigbench.csv"
    df.to_csv(csv_path)
    print(f"  Saved {csv_path}")

    B = df.values.astype(np.float64)
    O = (~np.isnan(B)).astype(np.float64)
    B = np.nan_to_num(B, nan=0.0)
    npz_path = HERE / "bigbench.npz"
    np.savez(
        npz_path,
        B=B,
        O=O,
        model_names=np.array(df.index.tolist()),
        benchmark_names=np.array(df.columns.tolist()),
    )
    print(f"  Saved {npz_path}")


if __name__ == "__main__":
    main()
