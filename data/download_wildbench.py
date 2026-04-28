#!/usr/bin/env python3
"""Download and assemble the WildBench score matrix.

Produces:
    wildbench.csv   — tidy CSV (rows = models, columns = tasks/categories)
    wildbench.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install datasets huggingface_hub pandas numpy requests python-dotenv tqdm

WildBench evaluates LLMs on challenging real-world user queries grouped
into multiple task categories.  Results are published on HuggingFace at
allenai/WildBench-V2 (and earlier at allenai/WildBench).

Scores include per-category breakdowns and overall WB scores (WB-Score,
WB-Reward, WB-Elo).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


def _hf_headers() -> dict:
    import os
    headers = {}
    token = os.environ.get("HF_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def main():
    csv_path = HERE / "wildbench.csv"
    npz_path = HERE / "wildbench.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"WildBench data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} categories).  Skipping download.")
            return

    print("Downloading WildBench results …")
    df = None

    # ------------------------------------------------------------------
    # Strategy 1: Download per-category WB_scores from HF Spaces data
    # The all_stat_wildbench.-1.json has 63 models with per-category WB_score.
    # ------------------------------------------------------------------
    print("  Trying WildBench HF Spaces data …")
    hf_space_url = (
        "https://huggingface.co/spaces/allenai/WildBench/resolve/main/"
        "data_dir/all_stat_wildbench.-1.json"
    )
    try:
        resp = requests.get(hf_space_url, timeout=30, allow_redirects=True)
        if resp.status_code == 200:
            data = resp.json()
            records = []
            for model_key, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                # Extract WB_score.original_task=* keys (11 task categories)
                for key, val in entry.items():
                    if (key.startswith("WB_score.original_task=")
                            and isinstance(val, (int, float))):
                        task = key.replace("WB_score.original_task=", "")
                        records.append({
                            "model": model_key, "task": task, "score": val
                        })
            if records:
                print(f"  Got {len(records)} records from HF Spaces")
                df_long = pd.DataFrame(records)
                df = df_long.pivot_table(
                    index="model", columns="task", values="score",
                    aggfunc="first"
                )
    except Exception as e:
        print(f"  HF Spaces: {e}")

    # ------------------------------------------------------------------
    # Strategy 1b: Fallback to GitHub score.json (5 categories)
    # ------------------------------------------------------------------
    if df is None:
        print("  Trying WildBench score.json from GitHub …")
        score_url = ("https://raw.githubusercontent.com/allenai/WildBench/main/"
                     "leaderboard/data_dir/score.json")
        try:
            resp = requests.get(score_url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                records = []
                for model_key, entry in data.items():
                    if not isinstance(entry, dict):
                        continue
                    model_name = entry.get("model", model_key)
                    cat_scores = entry.get("task_categorized_scores", {})
                    if isinstance(cat_scores, dict):
                        for cat, score in cat_scores.items():
                            if isinstance(score, (int, float)):
                                records.append({
                                    "model": model_name, "task": cat,
                                    "score": score
                                })
                if records:
                    print(f"  Got {len(records)} records from score.json")
                    df_long = pd.DataFrame(records)
                    df = df_long.pivot_table(
                        index="model", columns="task", values="score",
                        aggfunc="first"
                    )
        except Exception as e:
            print(f"  score.json: {e}")

    # ------------------------------------------------------------------
    # Strategy 2: HuggingFace Space API
    # ------------------------------------------------------------------
    if df is None:
        print("  Trying WildBench HF Space API …")
        space_urls = [
            "https://allenai-wildbench.hf.space/api/leaderboard",
            "https://allenai-wildbench-leaderboard.hf.space/api/leaderboard",
        ]
        for url in space_urls:
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                records = []
                if isinstance(data, list):
                    for entry in data:
                        model = entry.get("model", entry.get("model_name", "?"))
                        for k, v in entry.items():
                            if k in ("model", "model_name"):
                                continue
                            if isinstance(v, (int, float)):
                                records.append({
                                    "model": model, "task": k, "score": v
                                })
                if records:
                    print(f"  Got {len(records)} from HF Space")
                    df_long = pd.DataFrame(records)
                    df = df_long.pivot_table(
                        index="model", columns="task", values="score",
                        aggfunc="first"
                    )
                    break
            except Exception as e:
                tqdm.write(f"  {url}: {e}")

    # ------------------------------------------------------------------
    # Strategy 3: GitHub raw files
    # ------------------------------------------------------------------
    if df is None:
        print("  Trying GitHub raw data …")
        github_urls = [
            ("https://raw.githubusercontent.com/allenai/WildBench/main/"
             "leaderboard/data_dir/all_stat.json"),
            ("https://raw.githubusercontent.com/allenai/WildBench/main/"
             "leaderboard/data_dir/pairwise_data.json"),
        ]
        for url in github_urls:
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                records = []

                if isinstance(data, dict):
                    # Could be {model: {metric: val, ...}, ...}
                    for model, metrics in data.items():
                        if isinstance(metrics, dict):
                            for k, v in metrics.items():
                                if isinstance(v, (int, float)):
                                    records.append({
                                        "model": model, "task": k, "score": v
                                    })
                        elif isinstance(metrics, list):
                            for item in metrics:
                                if isinstance(item, dict):
                                    for k, v in item.items():
                                        if isinstance(v, (int, float)):
                                            records.append({
                                                "model": model, "task": k,
                                                "score": v
                                            })
                elif isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            model = entry.get("model", entry.get("model_name", "?"))
                            for k, v in entry.items():
                                if k not in ("model", "model_name") and \
                                   isinstance(v, (int, float)):
                                    records.append({
                                        "model": model, "task": k, "score": v
                                    })

                if records:
                    print(f"  Got {len(records)} records from {url.split('/')[-1]}")
                    df_long = pd.DataFrame(records)
                    df = df_long.pivot_table(
                        index="model", columns="task", values="score",
                        aggfunc="first"
                    )
                    break
            except Exception as e:
                tqdm.write(f"  {url}: {e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if df is None or df.shape[0] == 0:
        print("WARNING: Could not download WildBench results.")
        print("  Saving empty placeholder files.")
        pd.DataFrame().to_csv(csv_path)
        np.savez(npz_path, B=np.empty((0, 0)), O=np.empty((0, 0)),
                 model_names=np.array([]), benchmark_names=np.array([]))
        return

    min_obs = max(3, len(df) * 0.1)
    df = df.loc[:, df.notna().sum() >= min_obs]
    df = df.dropna(how="all")

    print(f"  Score matrix: {df.shape[0]} models × {df.shape[1]} categories")
    n_missing = df.isna().sum().sum()
    n_total = df.shape[0] * df.shape[1]
    if n_total > 0:
        print(f"  Missing entries: {n_missing}/{n_total} ({n_missing/n_total:.1%})")

    df.to_csv(csv_path)
    print(f"  Saved {csv_path}")

    B = df.values.astype(np.float64)
    O = (~np.isnan(B)).astype(np.float64)
    B = np.nan_to_num(B, nan=0.0)
    np.savez(npz_path, B=B, O=O,
             model_names=np.array(df.index.tolist()),
             benchmark_names=np.array(df.columns.tolist()))
    print(f"  Saved {npz_path}")


if __name__ == "__main__":
    main()
