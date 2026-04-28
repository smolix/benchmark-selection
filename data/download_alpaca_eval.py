#!/usr/bin/env python3
"""Download and assemble the AlpacaEval 2 score matrix.

Produces:
    alpaca_eval.csv   — tidy CSV (rows = models, columns = metrics)
    alpaca_eval.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install pandas numpy requests python-dotenv tqdm

AlpacaEval 2 publishes its leaderboard at
    https://github.com/tatsu-lab/alpaca_eval
with per-model results in the results/ directory.

The leaderboard CSV is at a known path in the repo.  Each model also
has per-instruction annotations that could provide finer-grained scores,
but the primary output here is the summary metrics (win_rate,
length_controlled_winrate, avg_length, etc.).
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
    csv_path = HERE / "alpaca_eval.csv"
    npz_path = HERE / "alpaca_eval.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"AlpacaEval data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} metrics).  Skipping download.")
            return

    print("Downloading AlpacaEval 2 results …")
    records = []

    # ------------------------------------------------------------------
    # Strategy 1: Download the leaderboard CSV directly from GitHub
    # ------------------------------------------------------------------
    leaderboard_urls = [
        # The canonical location in the repo (data_AlpacaEval_2 directory)
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/"
        "src/alpaca_eval/leaderboards/data_AlpacaEval_2/"
        "weighted_alpaca_eval_gpt4_turbo_leaderboard.csv",
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/"
        "src/alpaca_eval/leaderboards/data_AlpacaEval_2/"
        "alpaca_eval_gpt4_turbo_fn_leaderboard.csv",
        "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/"
        "src/alpaca_eval/leaderboards/data_AlpacaEval_2/"
        "alpaca_eval_cot_gpt4_turbo_fn_leaderboard.csv",
    ]

    df = None
    for url in leaderboard_urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            df_raw = pd.read_csv(pd.io.common.StringIO(resp.text))
            if df_raw.shape[0] > 0 and df_raw.shape[1] > 1:
                print(f"  Found leaderboard CSV ({df_raw.shape[0]} models)")
                print(f"  Columns: {list(df_raw.columns)}")
                # Identify the model name column
                model_col = None
                for c in df_raw.columns:
                    if c.lower() in ("model", "model_name", "name", ""):
                        model_col = c
                        break
                if model_col is None:
                    # First column is usually the model name or it's the index
                    if df_raw.columns[0] == "" or "Unnamed" in str(df_raw.columns[0]):
                        df_raw = df_raw.set_index(df_raw.columns[0])
                    else:
                        model_col = df_raw.columns[0]
                        df_raw = df_raw.set_index(model_col)

                if model_col and model_col in df_raw.columns:
                    df_raw = df_raw.set_index(model_col)

                # Keep only numeric columns
                numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
                df = df_raw[numeric_cols]
                if df.shape[1] > 0:
                    break
                df = None
        except Exception as e:
            tqdm.write(f"  {url}: {e}")

    # ------------------------------------------------------------------
    # Strategy 2: Enumerate per-model result dirs via GitHub API
    # ------------------------------------------------------------------
    if df is None:
        print("  Leaderboard CSV not found. Trying per-model results …")
        headers = _github_headers()
        api_url = ("https://api.github.com/repos/tatsu-lab/alpaca_eval/"
                   "contents/results/latest")
        try:
            resp = requests.get(api_url, headers=headers, timeout=30)
            if resp.status_code == 200:
                dirs = [item for item in resp.json()
                        if item.get("type") == "dir"]
                print(f"  Found {len(dirs)} model directories")

                for d in tqdm(dirs, desc="Models", unit="model"):
                    model_name = d["name"]
                    # Try to get the model's metrics.json
                    metrics_url = (
                        f"https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/"
                        f"main/results/latest/{model_name}/metrics.json"
                    )
                    try:
                        r = requests.get(metrics_url, timeout=15)
                        if r.status_code != 200:
                            continue
                        data = r.json()
                        row = {"model": model_name}
                        if isinstance(data, dict):
                            for k, v in data.items():
                                if isinstance(v, (int, float)):
                                    row[k] = v
                                elif isinstance(v, dict):
                                    for kk, vv in v.items():
                                        if isinstance(vv, (int, float)):
                                            row[f"{k}_{kk}"] = vv
                        if len(row) > 1:
                            records.append(row)
                    except Exception:
                        continue

            if records:
                df = pd.DataFrame(records).set_index("model")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_cols]
        except Exception as e:
            print(f"  GitHub API failed: {e}")

    # ------------------------------------------------------------------
    # Strategy 3: Try the alpaca_eval Python package
    # ------------------------------------------------------------------
    if df is None and not records:
        try:
            print("  Trying alpaca_eval package …")
            import alpaca_eval
            leaderboard = alpaca_eval.get_leaderboard()
            if leaderboard is not None and len(leaderboard) > 0:
                df = leaderboard.select_dtypes(include=[np.number])
                print(f"  Got {df.shape[0]} models from alpaca_eval package")
        except (ImportError, Exception) as e:
            print(f"  alpaca_eval package: {e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if df is None or df.shape[0] == 0:
        print("WARNING: Could not download AlpacaEval results.")
        print("  Saving empty placeholder files.")
        pd.DataFrame().to_csv(csv_path)
        np.savez(npz_path, B=np.empty((0, 0)), O=np.empty((0, 0)),
                 model_names=np.array([]), benchmark_names=np.array([]))
        return

    df = df.dropna(how="all")
    # Drop columns that are all NaN
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
