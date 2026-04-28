#!/usr/bin/env python3
"""Download and assemble the Open LLM Leaderboard v2 score matrix.

Produces:
    open_llm.csv   — tidy CSV (rows = models, columns = benchmarks)
    open_llm.npz   — B (score matrix), O (observation mask), model_names,
                      benchmark_names

Requirements:
    pip install huggingface_hub pandas numpy

The leaderboard v2 benchmarks are:
    IFEval, BBH, MATH (Lvl 5), GPQA, MUSR, MMLU-PRO

Strategy:
    Use the huggingface_hub API to list and download individual JSON result
    files from the open-llm-leaderboard/results repo.  This avoids the
    datasets library which chokes on malformed JSON in that repo.
"""

from __future__ import annotations

import json
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# Load HF_TOKEN (and any other vars) from .env next to this script
load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")

HERE = pathlib.Path(__file__).resolve().parent

REPO_ID = "open-llm-leaderboard/results"

# Mapping: benchmark display name → (group key in "results" dict, metric key)
BENCHMARK_MAP = {
    "IFEval": ("leaderboard_ifeval", "prompt_level_strict_acc,none"),
    "BBH": ("leaderboard_bbh", "acc_norm,none"),
    "MATH_Lvl5": ("leaderboard_math_hard", "exact_match,none"),
    "GPQA": ("leaderboard_gpqa", "acc_norm,none"),
    "MUSR": ("leaderboard_musr", "acc_norm,none"),
    "MMLU-PRO": ("leaderboard_mmlu_pro", "acc,none"),
}


def _extract_scores(data: dict) -> dict | None:
    """Extract benchmark scores from a single result JSON."""
    results = data.get("results") or {}
    if not results:
        return None

    scores = {}
    for bench_name, (group_key, metric_key) in BENCHMARK_MAP.items():
        val = None
        if group_key in results:
            val = results[group_key].get(metric_key)
        if val is None:
            for k, v in results.items():
                if group_key in k and isinstance(v, dict):
                    val = v.get(metric_key)
                    if val is not None:
                        break
        scores[bench_name] = val

    if all(v is None for v in scores.values()):
        return None
    return scores


def main():
    # ── Skip if data already exists ───────────────────────────────
    csv_path = HERE / "open_llm.csv"
    npz_path = HERE / "open_llm.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"Open LLM data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} benchmarks).  Skipping download.")
            return

    api = HfApi()

    # ── Strategy 1: list all JSON files in the repo ──────────────
    print("Listing files in open-llm-leaderboard/results …")
    try:
        all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
        json_files = [f for f in all_files if f.endswith(".json")]
        print(f"  Found {len(json_files)} JSON files")
    except Exception as e:
        print(f"  Could not list repo files: {e}")
        json_files = []

    if not json_files:
        # ── Strategy 2: use the leaderboard contents endpoint ────
        print("Trying Open LLM Leaderboard contents API …")
        try:
            import requests
            url = "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            print(f"  Got {len(data)} entries from leaderboard API")
            records = []
            for entry in data:
                model = entry.get("model", {}).get("name", "")
                if not model:
                    model = entry.get("model_name", "?")
                scores = {}
                for bench in BENCHMARK_MAP:
                    val = entry.get(bench) or entry.get(bench.lower())
                    scores[bench] = val
                if not all(v is None for v in scores.values()):
                    scores["model"] = model
                    records.append(scores)
            if records:
                _save(pd.DataFrame(records).set_index("model"))
                return
        except Exception as e2:
            print(f"  Leaderboard API also failed: {e2}")
        print("ERROR: Could not download Open LLM Leaderboard data.")
        sys.exit(1)

    # ── Parse each JSON file ─────────────────────────────────────
    records = []
    bad = 0
    for fpath in tqdm(json_files, desc="Downloading results", unit="file"):
        try:
            local = hf_hub_download(
                REPO_ID, fpath, repo_type="dataset",
            )
            with open(local) as f:
                data = json.load(f)
        except Exception:
            bad += 1
            continue

        # Model name: from JSON or from file path
        model = (
            data.get("model_name")
            or data.get("model_name_sanitized")
            or fpath.rsplit("/", 1)[-1].replace(".json", "")
        )

        scores = _extract_scores(data)
        if scores is not None:
            scores["model"] = model
            records.append(scores)

    print(f"  Parsed {len(records)} result files ({bad} skipped due to errors)")

    if not records:
        print("ERROR: no valid result files found.")
        sys.exit(1)

    df = pd.DataFrame(records).set_index("model")
    _save(df)


def _save(df: pd.DataFrame):
    benchmarks = list(BENCHMARK_MAP.keys())
    # Keep only expected columns (add missing ones as NaN)
    for b in benchmarks:
        if b not in df.columns:
            df[b] = np.nan
    df = df[benchmarks]

    # De-duplicate: keep the latest entry per model
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(how="all")

    print(f"  Score matrix: {df.shape[0]} models × {df.shape[1]} benchmarks")
    n_missing = df.isna().sum().sum()
    n_total = df.shape[0] * df.shape[1]
    print(f"  Missing entries: {n_missing}/{n_total} ({n_missing/n_total:.1%})")
    print(f"  Benchmarks: {list(df.columns)}")

    csv_path = HERE / "open_llm.csv"
    df.to_csv(csv_path)
    print(f"  Saved {csv_path}")

    B = df.values.astype(np.float64)
    O = (~np.isnan(B)).astype(np.float64)
    B = np.nan_to_num(B, nan=0.0)
    npz_path = HERE / "open_llm.npz"
    np.savez(
        npz_path,
        B=B,
        O=O,
        model_names=np.array(df.index.tolist()),
        benchmark_names=np.array(benchmarks),
    )
    print(f"  Saved {npz_path}")


if __name__ == "__main__":
    main()
