#!/usr/bin/env python3
"""Download and assemble the Arena-Hard-Auto score matrix.

Produces:
    arena_hard.csv   — tidy CSV (rows = models, columns = metrics/categories)
    arena_hard.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install pandas numpy requests python-dotenv tqdm

Arena-Hard-Auto is a benchmark using GPT-4-Turbo as an automatic judge
to evaluate LLMs on 500 challenging prompts derived from Chatbot Arena.
Results are published at https://github.com/lmarena/arena-hard-auto.

The primary score is a win-rate against a baseline model, but per-category
breakdowns are also available.
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
    csv_path = HERE / "arena_hard.csv"
    npz_path = HERE / "arena_hard.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"Arena-Hard data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} metrics).  Skipping download.")
            return

    print("Downloading Arena-Hard-Auto results …")

    # ------------------------------------------------------------------
    # Strategy 1: Download leaderboard table from GitHub repo
    # ------------------------------------------------------------------
    df = None

    # Try known paths for the leaderboard data
    urls = [
        ("https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
         "leaderboard/arena_hard_leaderboard_20240731.csv"),
        ("https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
         "data/arena_hard_leaderboard.csv"),
        ("https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
         "leaderboard_table.csv"),
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            df_raw = pd.read_csv(pd.io.common.StringIO(resp.text))
            if df_raw.shape[0] > 0:
                print(f"  Found CSV at {url.split('/')[-1]} "
                      f"({df_raw.shape[0]} models)")
                print(f"  Columns: {list(df_raw.columns)}")
                # Set model name as index
                model_col = None
                for c in df_raw.columns:
                    cl = c.lower().strip()
                    if cl in ("model", "model_name", "name"):
                        model_col = c
                        break
                if model_col:
                    df_raw = df_raw.set_index(model_col)
                elif df_raw.columns[0] == "" or "Unnamed" in str(df_raw.columns[0]):
                    df_raw = df_raw.set_index(df_raw.columns[0])

                numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df = df_raw[numeric_cols]
                    break
        except Exception as e:
            tqdm.write(f"  {url}: {e}")

    # ------------------------------------------------------------------
    # Strategy 2: Parse the leaderboard JSON
    # ------------------------------------------------------------------
    if df is None:
        json_urls = [
            ("https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
             "data/arena_hard_leaderboard.json"),
            ("https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
             "leaderboard_table.json"),
        ]
        for url in json_urls:
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                records = []
                if isinstance(data, dict):
                    # Could be {model_name: {metric: value, ...}, ...}
                    for model, metrics in data.items():
                        if isinstance(metrics, dict):
                            row = {"model": model}
                            row.update({k: v for k, v in metrics.items()
                                        if isinstance(v, (int, float))})
                            if len(row) > 1:
                                records.append(row)
                elif isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            model = (entry.get("model") or entry.get("model_name")
                                     or entry.get("name", "?"))
                            row = {"model": model}
                            for k, v in entry.items():
                                if k not in ("model", "model_name", "name") and \
                                   isinstance(v, (int, float)):
                                    row[k] = v
                            if len(row) > 1:
                                records.append(row)
                if records:
                    df = pd.DataFrame(records).set_index("model")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df = df[numeric_cols]
                    print(f"  Found JSON with {df.shape[0]} models")
                    break
            except Exception as e:
                tqdm.write(f"  {url}: {e}")

    # ------------------------------------------------------------------
    # Strategy 3: Parse per-model result directories
    # ------------------------------------------------------------------
    if df is None:
        print("  Trying per-model result directories …")
        headers = _github_headers()
        api_url = ("https://api.github.com/repos/lmarena/arena-hard-auto/"
                   "contents/data/arena-hard-v0.1")
        records = []
        try:
            resp = requests.get(api_url, headers=headers, timeout=30)
            if resp.status_code == 200:
                items = resp.json()
                dirs = [item for item in items if item.get("type") == "dir"]
                print(f"  Found {len(dirs)} model result directories")

                for d in tqdm(dirs, desc="Models", unit="model"):
                    model_name = d["name"]
                    # Try to get a summary or stats file
                    for fname in ["stats.json", "result.json", "summary.json"]:
                        raw_url = (
                            f"https://raw.githubusercontent.com/lmarena/"
                            f"arena-hard-auto/main/data/arena-hard-v0.1/"
                            f"{model_name}/{fname}"
                        )
                        try:
                            r = requests.get(raw_url, timeout=10)
                            if r.status_code != 200:
                                continue
                            data = r.json()
                            row = {"model": model_name}
                            if isinstance(data, dict):
                                for k, v in data.items():
                                    if isinstance(v, (int, float)):
                                        row[k] = v
                            if len(row) > 1:
                                records.append(row)
                                break
                        except Exception:
                            continue
        except Exception as e:
            print(f"  {e}")

        if records:
            df = pd.DataFrame(records).set_index("model")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_cols]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if df is None or df.shape[0] == 0:
        print("WARNING: Could not download Arena-Hard results.")
        print("  Saving empty placeholder files.")
        pd.DataFrame().to_csv(csv_path)
        np.savez(npz_path, B=np.empty((0, 0)), O=np.empty((0, 0)),
                 model_names=np.array([]), benchmark_names=np.array([]))
        return

    df = df.dropna(how="all")
    df = df.loc[:, df.notna().any()]

    print(f"  Score matrix: {df.shape[0]} models × {df.shape[1]} metrics")
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
