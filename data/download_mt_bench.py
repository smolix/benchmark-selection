#!/usr/bin/env python3
"""Download and assemble the MT-Bench score matrix.

Produces:
    mt_bench.csv   — tidy CSV (rows = models, columns = 8 categories)
    mt_bench.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install pandas numpy requests python-dotenv tqdm

MT-Bench evaluates LLMs on 80 multi-turn questions across 8 categories
(writing, roleplay, reasoning, math, coding, extraction, stem, humanities).
Each answer is judged by GPT-4 on a 1-10 scale.

Results are published in the lmsys/FastChat GitHub repo under
fastchat/llm_judge/data/mt_bench/.
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

MT_BENCH_CATEGORIES = [
    "writing", "roleplay", "reasoning", "math",
    "coding", "extraction", "stem", "humanities",
]

# Question ID → category mapping (MT-Bench has 80 questions, 10 per category)
# Questions 81-160 are turn-2 follow-ups
QUESTION_CATEGORY = {}
for i, cat in enumerate(MT_BENCH_CATEGORIES):
    for q in range(i * 10 + 81, i * 10 + 91):
        QUESTION_CATEGORY[q] = cat
    for q in range(i * 10 + 1, i * 10 + 11):
        QUESTION_CATEGORY[q] = cat


def _github_headers() -> dict:
    import os
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def main():
    csv_path = HERE / "mt_bench.csv"
    npz_path = HERE / "mt_bench.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"MT-Bench data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} categories).  Skipping download.")
            return

    print("Downloading MT-Bench results …")
    records = []

    # ------------------------------------------------------------------
    # Strategy 1: Load GPT-4 judgments from SUSTech/mt_bench_judge on HF
    # (The original JSONL files were removed from the FastChat repo.)
    # ------------------------------------------------------------------
    try:
        from datasets import load_dataset
        print("  Loading SUSTech/mt_bench_judge from HuggingFace …")
        ds = load_dataset("SUSTech/mt_bench_judge")
        scores_by_model = {}  # model → {category: [scores]}
        for row in tqdm(ds["train"], desc="Parsing judgments", unit="row"):
            model = row.get("model", "")
            cat = row.get("category", "").lower().strip()
            score = row.get("score")
            if not model or not cat or score is None:
                continue
            if cat not in MT_BENCH_CATEGORIES:
                continue
            if model not in scores_by_model:
                scores_by_model[model] = {c: [] for c in MT_BENCH_CATEGORIES}
            scores_by_model[model][cat].append(float(score))

        for model, cats in scores_by_model.items():
            for cat, scores_list in cats.items():
                if scores_list:
                    records.append({
                        "model": model,
                        "task": cat,
                        "score": np.mean(scores_list),
                    })
        print(f"  Got {len(records)} records from {len(scores_by_model)} models")
    except Exception as e:
        print(f"  SUSTech/mt_bench_judge: {e}")

    # ------------------------------------------------------------------
    # Strategy 2: Try the MT-Bench leaderboard on HuggingFace
    # ------------------------------------------------------------------
    if not records:
        print("  Trying HuggingFace datasets …")
        try:
            from datasets import load_dataset
            for ds_name in [
                "lmsys/mt_bench_human_judgments",
            ]:
                try:
                    ds = load_dataset(ds_name)
                    print(f"  Loaded {ds_name}")
                    scores_by_model = {}
                    for split in ds:
                        for row in tqdm(ds[split], desc=f"Parsing {split}"):
                            model = row.get("model", row.get("model_a", "?"))
                            cat = row.get("category", "")
                            score = row.get("score", row.get("judge_score"))
                            if not model or not cat or score is None:
                                continue
                            cat = cat.lower().strip()
                            if cat not in MT_BENCH_CATEGORIES:
                                continue
                            if model not in scores_by_model:
                                scores_by_model[model] = {c: [] for c in MT_BENCH_CATEGORIES}
                            scores_by_model[model][cat].append(float(score))

                    for model, cats in scores_by_model.items():
                        for cat, scores in cats.items():
                            if scores:
                                records.append({
                                    "model": model,
                                    "task": cat,
                                    "score": np.mean(scores),
                                })
                    if records:
                        break
                except Exception as e:
                    tqdm.write(f"  {ds_name}: {e}")
        except ImportError:
            print("  datasets library not installed")

    # ------------------------------------------------------------------
    # Strategy 3: Try scraping the known MT-Bench results table
    # ------------------------------------------------------------------
    if not records:
        print("  Trying MT-Bench results table from GitHub …")
        # The FastChat repo sometimes has a summary table
        table_url = ("https://raw.githubusercontent.com/lm-sys/FastChat/main/"
                     "fastchat/llm_judge/data/mt_bench/mt_bench_results.json")
        try:
            resp = requests.get(table_url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for entry in data:
                        model = entry.get("model", "?")
                        for cat in MT_BENCH_CATEGORIES:
                            score = entry.get(cat)
                            if score is not None:
                                records.append({
                                    "model": model,
                                    "task": cat,
                                    "score": float(score),
                                })
        except Exception as e:
            print(f"  {e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if not records:
        print("WARNING: Could not download MT-Bench results.")
        print("  Saving empty placeholder files.")
        pd.DataFrame().to_csv(csv_path)
        np.savez(npz_path, B=np.empty((0, 0)), O=np.empty((0, 0)),
                 model_names=np.array([]), benchmark_names=np.array([]))
        return

    df_long = pd.DataFrame(records)
    df = df_long.pivot_table(
        index="model", columns="task", values="score", aggfunc="first"
    )

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
