#!/usr/bin/env python3
"""Download and assemble the LiveBench score matrix.

Produces:
    livebench.csv   — tidy CSV (rows = models, columns = tasks)
    livebench.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install pandas numpy requests python-dotenv tqdm

LiveBench is a continuously-updated benchmark with 6 categories:
    math, coding, reasoning, language, data_analysis, instruction_following
Each category has multiple sub-tasks (18+ total).

Results are published at https://github.com/LiveBench/LiveBench and
on the HuggingFace leaderboard.
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


def _github_headers() -> dict:
    import os
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def main():
    csv_path = HERE / "livebench.csv"
    npz_path = HERE / "livebench.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"LiveBench data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} tasks).  Skipping download.")
            return

    print("Downloading LiveBench results …")

    # ------------------------------------------------------------------
    # Strategy 1: Download from livebench/model_judgment on HuggingFace
    # This dataset has per-question scores with model, task, category, score.
    # ------------------------------------------------------------------
    df = None
    try:
        from datasets import load_dataset
        print("  Trying livebench/model_judgment from HuggingFace …")
        try:
            ds = load_dataset("livebench/model_judgment")
            split_name = "leaderboard" if "leaderboard" in ds else list(ds.keys())[0]
            lb = ds[split_name].to_pandas()
            print(f"  Loaded {len(lb)} rows, categories: {sorted(lb['category'].unique())}")

            # Aggregate: mean score per (model, category)
            agg = lb.groupby(["model", "category"])["score"].mean().reset_index()
            df = agg.pivot_table(index="model", columns="category",
                                 values="score", aggfunc="first")
            print(f"  Pivoted: {df.shape[0]} models × {df.shape[1]} categories")
        except Exception as e:
            print(f"  livebench/model_judgment: {e}")
    except ImportError:
        print("  datasets library not installed")

    # ------------------------------------------------------------------
    # Strategy 2: Download from GitHub
    # ------------------------------------------------------------------
    if df is None:
        print("  Trying LiveBench GitHub repo …")

        # Try known leaderboard CSV/JSON paths
        github_urls = [
            ("https://raw.githubusercontent.com/LiveBench/LiveBench/main/"
             "docs/leaderboard.csv"),
            ("https://raw.githubusercontent.com/LiveBench/LiveBench/main/"
             "data/leaderboard.csv"),
            ("https://raw.githubusercontent.com/LiveBench/LiveBench/main/"
             "leaderboard.csv"),
            ("https://raw.githubusercontent.com/LiveBench/LiveBench/main/"
             "docs/leaderboard.json"),
        ]

        for url in github_urls:
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    continue

                if url.endswith(".csv"):
                    df_raw = pd.read_csv(pd.io.common.StringIO(resp.text))
                else:
                    data = resp.json()
                    if isinstance(data, list):
                        df_raw = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        df_raw = pd.DataFrame(data.get("data", data.get("rows", [data])))
                    else:
                        continue

                if df_raw.shape[0] > 0:
                    print(f"  Found: {url.split('/')[-1]} ({df_raw.shape[0]} rows)")
                    print(f"  Columns: {list(df_raw.columns)[:10]}")
                    # Identify model column
                    model_col = None
                    for c in df_raw.columns:
                        if c.lower().strip() in ("model", "model_name", "name"):
                            model_col = c
                            break
                    if model_col:
                        df_raw = df_raw.set_index(model_col)
                    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        df = df_raw[numeric_cols]
                        break
            except Exception as e:
                tqdm.write(f"  {url}: {e}")

    # ------------------------------------------------------------------
    # Strategy 3: Parse per-model result files from GitHub
    # ------------------------------------------------------------------
    if df is None:
        print("  Trying per-model results from GitHub …")
        headers = _github_headers()
        records = []

        # Enumerate model result directories
        for results_path in [
            "data/results",
            "results",
            "output",
        ]:
            api_url = (f"https://api.github.com/repos/LiveBench/LiveBench/"
                       f"contents/{results_path}")
            try:
                resp = requests.get(api_url, headers=headers, timeout=30)
                if resp.status_code != 200:
                    continue
                items = resp.json()
                if not isinstance(items, list):
                    continue
                print(f"  Found {len(items)} items in {results_path}/")

                for item in tqdm(items[:200], desc="Parsing results"):
                    if item.get("type") == "file" and item["name"].endswith(".json"):
                        try:
                            raw_url = item.get("download_url", "")
                            r = requests.get(raw_url, timeout=15)
                            if r.status_code != 200:
                                continue
                            data = r.json()
                            model = (data.get("model", "")
                                     or item["name"].replace(".json", ""))
                            for k, v in data.items():
                                if k in ("model", "model_name"):
                                    continue
                                if isinstance(v, (int, float)):
                                    records.append({
                                        "model": model, "task": k, "score": v
                                    })
                        except Exception:
                            continue

                if records:
                    break
            except Exception:
                continue

        if records:
            df_long = pd.DataFrame(records)
            df = df_long.pivot_table(
                index="model", columns="task", values="score", aggfunc="first"
            )

    # ------------------------------------------------------------------
    # Strategy 4: Fetch from the LiveBench HuggingFace Space API
    # ------------------------------------------------------------------
    if df is None:
        print("  Trying LiveBench HF Space API …")
        space_urls = [
            "https://livebench-livebench.hf.space/api/leaderboard",
            "https://livebench-livebench-leaderboard.hf.space/api/leaderboard",
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
                                records.append({"model": model, "task": k, "score": v})
                if records:
                    df_long = pd.DataFrame(records)
                    df = df_long.pivot_table(
                        index="model", columns="task", values="score",
                        aggfunc="first"
                    )
                    print(f"  Got data from HF Space API")
                    break
            except Exception as e:
                tqdm.write(f"  {url}: {e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if df is None or df.shape[0] == 0:
        print("WARNING: Could not download LiveBench results.")
        print("  Saving empty placeholder files.")
        pd.DataFrame().to_csv(csv_path)
        np.savez(npz_path, B=np.empty((0, 0)), O=np.empty((0, 0)),
                 model_names=np.array([]), benchmark_names=np.array([]))
        return

    # Drop tasks with very few models
    min_obs = max(3, len(df) * 0.1)
    df = df.loc[:, df.notna().sum() >= min_obs]
    df = df.dropna(how="all")

    print(f"  Score matrix: {df.shape[0]} models × {df.shape[1]} tasks")
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
