#!/usr/bin/env python3
"""Download and assemble the BigCode / EvalPlus score matrix.

Produces:
    bigcode.csv   — tidy CSV (rows = models, columns = coding benchmarks)
    bigcode.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install datasets huggingface_hub pandas numpy requests python-dotenv tqdm

The BigCode leaderboard evaluates code generation models on:
    HumanEval, HumanEval+, MBPP, MBPP+, MultiPL-E (multiple languages)

Results are published at:
    https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard
    https://huggingface.co/datasets/bigcode/bigcode-models-leaderboard
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
    csv_path = HERE / "bigcode.csv"
    npz_path = HERE / "bigcode.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"BigCode data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} benchmarks).  Skipping download.")
            return

    print("Downloading BigCode leaderboard results …")
    df = None

    # ------------------------------------------------------------------
    # Strategy 1: HuggingFace bigcode/bigcodebench-results dataset
    # Contains model, complete, instruct solve-rate columns.
    # ------------------------------------------------------------------
    try:
        from datasets import load_dataset
        print("  Trying bigcode/bigcodebench-results from HuggingFace …")
        try:
            ds = load_dataset("bigcode/bigcodebench-results")
            split_name = list(ds.keys())[0]
            records = []
            for row in tqdm(ds[split_name], desc="Parsing bigcodebench-results"):
                model = row.get("model", row.get("Model", "?"))
                for key, val in row.items():
                    if key.lower() in ("model", "model_name", "link", "moe",
                                       "type", "date", "prefill",
                                       "size", "act_param"):
                        continue
                    if isinstance(val, (int, float)) and not (
                        isinstance(val, float) and np.isnan(val)
                    ):
                        records.append({
                            "model": model,
                            "task": key,
                            "score": val,
                        })
            if records:
                print(f"  Got {len(records)} records from bigcodebench-results")
                df_long = pd.DataFrame(records)
                df = df_long.pivot_table(
                    index="model", columns="task", values="score",
                    aggfunc="first"
                )
        except Exception as e:
            print(f"  bigcode/bigcodebench-results: {e}")
    except ImportError:
        print("  datasets library not installed")

    # ------------------------------------------------------------------
    # Strategy 1b: Per-domain breakdown from bigcode/bigcodebench-domain
    # ------------------------------------------------------------------
    if df is None or df.shape[1] <= 4:
        try:
            from datasets import load_dataset
            print("  Trying bigcode/bigcodebench-domain for per-domain scores …")
            ds_domain = load_dataset("bigcode/bigcodebench-domain")
            records = []
            for split in ds_domain:
                for row in ds_domain[split]:
                    model = row.get("Model", row.get("model", "?"))
                    for key, val in row.items():
                        if key.lower() in ("model",):
                            continue
                        if isinstance(val, (int, float)) and not (
                            isinstance(val, float) and np.isnan(val)
                        ):
                            records.append({
                                "model": model,
                                "task": f"{key}_{split}",
                                "score": val,
                            })
            if records:
                print(f"  Got {len(records)} records from bigcodebench-domain")
                df_long = pd.DataFrame(records)
                df = df_long.pivot_table(
                    index="model", columns="task", values="score",
                    aggfunc="first"
                )
        except Exception as e:
            print(f"  bigcode/bigcodebench-domain: {e}")

    # ------------------------------------------------------------------
    # Strategy 2: EvalPlus leaderboard JSON from GitHub
    # ------------------------------------------------------------------
    if df is None:
        print("  Trying EvalPlus GitHub data …")
        github_urls = [
            ("https://raw.githubusercontent.com/evalplus/evalplus/master/"
             "results/evalplus_results.json"),
            ("https://raw.githubusercontent.com/evalplus/evalplus/master/"
             "results/leaderboard.json"),
            ("https://raw.githubusercontent.com/evalplus/evalplus/main/"
             "results/evalplus_results.json"),
        ]
        for url in github_urls:
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                records = []
                if isinstance(data, dict):
                    for model, metrics in data.items():
                        if isinstance(metrics, dict):
                            for k, v in metrics.items():
                                if isinstance(v, (int, float)):
                                    records.append({
                                        "model": model, "task": k, "score": v
                                    })
                elif isinstance(data, list):
                    for entry in data:
                        if not isinstance(entry, dict):
                            continue
                        model = (entry.get("model") or entry.get("Model")
                                 or entry.get("model_name", "?"))
                        for k, v in entry.items():
                            if k.lower() in ("model", "model_name"):
                                continue
                            if isinstance(v, (int, float)):
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
    # Strategy 3: HF Space API for BigCode leaderboard
    # ------------------------------------------------------------------
    if df is None:
        print("  Trying BigCode HF Space API …")
        space_urls = [
            "https://bigcode-bigcode-models-leaderboard.hf.space/api/leaderboard",
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
                        model = entry.get("model", entry.get("Model", "?"))
                        for k, v in entry.items():
                            if k.lower() in ("model", "model_name"):
                                continue
                            if isinstance(v, (int, float)):
                                records.append({
                                    "model": model, "task": k, "score": v
                                })
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
        print("WARNING: Could not download BigCode results.")
        print("  Saving empty placeholder files.")
        pd.DataFrame().to_csv(csv_path)
        np.savez(npz_path, B=np.empty((0, 0)), O=np.empty((0, 0)),
                 model_names=np.array([]), benchmark_names=np.array([]))
        return

    min_obs = max(3, len(df) * 0.1)
    df = df.loc[:, df.notna().sum() >= min_obs]
    df = df.dropna(how="all")

    print(f"  Score matrix: {df.shape[0]} models × {df.shape[1]} benchmarks")
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
