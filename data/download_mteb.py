#!/usr/bin/env python3
"""Download and assemble the MTEB (Massive Text Embedding Benchmark) score matrix.

Produces:
    mteb.csv   — tidy CSV (rows = models, columns = tasks)
    mteb.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install mteb pandas numpy

The MTEB library can load published results from the HuggingFace Hub.
We aggregate the main score per (model, task) pair.

We avoid the very slow ``ResultCache.load_results()`` (which parses every
JSON on disk) and instead use the leaderboard results dataset published to
HuggingFace, falling back to the mteb library only when needed.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent

# Load HF_TOKEN from .env next to this script
load_dotenv(HERE / ".env")

# Suppress noisy warnings from mteb about missing subsets/splits/scores
warnings.filterwarnings("ignore", message=".*Missing subsets.*")
warnings.filterwarnings("ignore", message=".*Missing splits.*")
warnings.filterwarnings("ignore", message=".*not found in scores.*")
warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)


def main():
    # ── Skip if data already exists ───────────────────────────────
    csv_path = HERE / "mteb.csv"
    npz_path = HERE / "mteb.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"MTEB data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} tasks).  Skipping download.")
            return

    # ------------------------------------------------------------------
    # Strategy 1: Use mteb.get_benchmark + per-model loading
    # ------------------------------------------------------------------
    records = []
    try:
        import mteb
        print("Loading MTEB results via the mteb library …")

        benchmark = mteb.get_benchmark("MTEB(eng, classic)")
        tasks = benchmark.tasks
        task_names = [t.metadata.name for t in tasks]
        print(f"  Benchmark has {len(task_names)} tasks")

        # Instead of loading ALL results (very slow), use the benchmark's
        # built-in method to get results for specific tasks.
        # mteb.load_results(tasks=...) is deprecated but works, and is
        # much more targeted than ResultCache.load_results() which scans
        # the entire cache directory.
        # Wrap everything in catch_warnings — load_results returns a lazy
        # generator so the actual parsing (and warnings) happen during
        # iteration, not during the call itself.
        print("  Loading task-specific results (this may clone a repo once) …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result_iter = mteb.load_results(
                    tasks=tasks,
                    download_latest=True,
                )
            except Exception as e:
                print(f"  mteb.load_results failed: {e}")
                result_iter = []

            n_models = 0
            for model_result in tqdm(result_iter, desc="Processing models", unit="model"):
                n_models += 1
                # Get model name
                if hasattr(model_result, 'model_name'):
                    model_name = model_result.model_name
                elif isinstance(model_result, dict):
                    model_name = model_result.get("model_name", model_result.get("model", "?"))
                else:
                    continue

                # Get task results
                if hasattr(model_result, 'task_results'):
                    task_results = model_result.task_results
                elif isinstance(model_result, dict):
                    task_results = model_result.get("task_results", [])
                else:
                    continue

                for task_result in task_results:
                    # Get task name
                    if hasattr(task_result, 'task_name'):
                        task_name = task_result.task_name
                    elif isinstance(task_result, dict):
                        task_name = task_result.get("task_name", "?")
                    else:
                        continue

                    if task_name not in task_names:
                        continue

                    # Get score — get_score() can warn or fail
                    main_score = None
                    try:
                        if hasattr(task_result, 'get_score'):
                            main_score = task_result.get_score()
                        elif isinstance(task_result, dict):
                            main_score = task_result.get("score",
                                         task_result.get("main_score"))
                    except Exception:
                        pass

                    # Fallback: extract from scores dict directly
                    if main_score is None and hasattr(task_result, 'scores'):
                        try:
                            scores_dict = task_result.scores
                            if isinstance(scores_dict, dict):
                                for split_scores in scores_dict.values():
                                    if isinstance(split_scores, dict):
                                        for lang_scores in split_scores.values():
                                            if isinstance(lang_scores, dict):
                                                for v in lang_scores.values():
                                                    if isinstance(v, (int, float)):
                                                        main_score = v
                                                        break
                                                if main_score is not None:
                                                    break
                                            elif isinstance(lang_scores, (int, float)):
                                                main_score = lang_scores
                                                break
                                    elif isinstance(split_scores, (int, float)):
                                        main_score = split_scores
                                        break
                                    if main_score is not None:
                                        break
                        except Exception:
                            pass

                    if main_score is not None:
                        records.append({
                            "model": model_name,
                            "task": task_name,
                            "score": main_score,
                        })

        print(f"  Processed {n_models} models, got {len(records)} scores")

        if records:
            df_long = pd.DataFrame(records)
            df = df_long.pivot_table(
                index="model", columns="task", values="score", aggfunc="first"
            )
            min_obs = max(10, len(df) * 0.05)
            df = df.loc[:, df.notna().sum() >= min_obs]
            df = df.dropna(how="all")
            _save(df)
            return

    except ImportError:
        print("mteb library not installed.  pip install mteb")
    except Exception as e:
        print(f"mteb library approach failed: {e}")

    # ------------------------------------------------------------------
    # Strategy 2: Load from HuggingFace dataset
    # ------------------------------------------------------------------
    if not records:
        try:
            print("Trying HuggingFace mteb/results dataset …")
            from huggingface_hub import HfApi
            api = HfApi()
            ds_info = api.dataset_info("mteb/results")
            print(f"  Dataset: {ds_info.id}")

            from datasets import load_dataset
            ds = load_dataset("mteb/results")

            records = []
            for split_name in ds:
                for row in tqdm(ds[split_name], desc=f"Parsing {split_name}",
                                unit="row"):
                    model = row.get("model_name", row.get("model", "?"))
                    for key, val in row.items():
                        if key in ("model_name", "model"):
                            continue
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            records.append({"model": model, "task": key,
                                            "score": val})

            if records:
                print(f"  Got {len(records)} records from mteb/results")
                df_long = pd.DataFrame(records)
                df = df_long.pivot_table(
                    index="model", columns="task", values="score",
                    aggfunc="first")
                df = df.dropna(how="all")
                _save(df)
                return

        except Exception as e:
            print(f"HuggingFace approach failed: {e}")

    # ------------------------------------------------------------------
    # Strategy 3: Placeholder
    # ------------------------------------------------------------------
    print("WARNING: Could not download MTEB results.")
    print("  Install mteb (pip install mteb) and re-run, or manually")
    print("  export the leaderboard from:")
    print("    https://huggingface.co/spaces/mteb/leaderboard")

    pd.DataFrame().to_csv(csv_path)
    np.savez(npz_path,
             B=np.empty((0, 0)), O=np.empty((0, 0)),
             model_names=np.array([]), benchmark_names=np.array([]))


def _save(df: pd.DataFrame):
    """Save the pivoted dataframe to CSV and NPZ."""
    print(f"  Score matrix: {df.shape[0]} models × {df.shape[1]} tasks")
    n_missing = df.isna().sum().sum()
    n_total = df.shape[0] * df.shape[1]
    if n_total > 0:
        print(f"  Missing entries: {n_missing}/{n_total} ({n_missing/n_total:.1%})")
    print(f"  Tasks: {list(df.columns)[:10]} … ({len(df.columns)} total)")

    csv_path = HERE / "mteb.csv"
    df.to_csv(csv_path)
    print(f"  Saved {csv_path}")

    B = df.values.astype(np.float64)
    O = (~np.isnan(B)).astype(np.float64)
    B = np.nan_to_num(B, nan=0.0)
    npz_path = HERE / "mteb.npz"
    np.savez(npz_path, B=B, O=O,
             model_names=np.array(df.index.tolist()),
             benchmark_names=np.array(df.columns.tolist()))
    print(f"  Saved {npz_path}")


if __name__ == "__main__":
    main()
