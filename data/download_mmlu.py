#!/usr/bin/env python3
"""Download and assemble MMLU per-subject score matrix.

Produces:
    mmlu.csv   — tidy CSV (rows = models, columns = 57 MMLU subjects)
    mmlu.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install huggingface_hub pandas numpy requests python-dotenv tqdm

The Open LLM Leaderboard v1 stored per-subject MMLU accuracy in
individual model "details" repos on HuggingFace.  We can also parse
results from the `open-llm-leaderboard/details_*` repos or from the
HF dataset snapshot.

Strategy 1: Download from the HF dataset `cais/hails/mmlu-pro-results`
            or a known aggregated leaderboard dataset.
Strategy 2: Scrape the `open-llm-leaderboard/details` repos which
            contain per-task breakdowns including MMLU subjects.
Strategy 3: Use the official MMLU leaderboard CSV from the
            hendrycks/test GitHub repo (limited models only).
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import traceback

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

# The 57 MMLU subjects
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


def _hf_headers() -> dict:
    """Return HuggingFace API headers, using HF_TOKEN env var if available."""
    import os
    headers = {}
    token = os.environ.get("HF_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def main():
    csv_path = HERE / "mmlu.csv"
    npz_path = HERE / "mmlu.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"MMLU data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} subjects).  Skipping download.")
            return

    print("Downloading MMLU per-subject results …")

    # ------------------------------------------------------------------
    # Strategy 1: Download from open-llm-leaderboard-old/results repo
    # This is a single HF dataset repo containing ~10K JSON files with
    # per-subject MMLU scores in keys like harness|hendrycksTest-{subject}|5.
    # ------------------------------------------------------------------
    records = []
    try:
        from huggingface_hub import snapshot_download
        from collections import defaultdict
        import glob as _glob

        REPO_ID = "open-llm-leaderboard-old/results"
        print(f"  Downloading snapshot of {REPO_ID} (parallel) …")
        local_dir = snapshot_download(
            REPO_ID, repo_type="dataset",
            allow_patterns="*.json",
        )
        print(f"  Snapshot at: {local_dir}")

        # Find all JSON files in the snapshot
        all_json = sorted(_glob.glob(f"{local_dir}/**/*.json", recursive=True))
        print(f"  Found {len(all_json)} JSON files locally")

        # Group by model directory, keep only the latest file per model
        model_files = defaultdict(list)
        for fpath in all_json:
            rel = pathlib.Path(fpath).relative_to(local_dir)
            parts = str(rel).rsplit("/", 1)
            if len(parts) == 2:
                model_dir, fname = parts
                model_files[model_dir].append(fpath)

        print(f"  {len(model_files)} unique models")

        bad = 0
        for model_dir in tqdm(model_files, desc="Parsing results",
                              unit="model"):
            try:
                fpath = sorted(model_files[model_dir])[-1]
                with open(fpath) as f:
                    data = json.load(f)

                results = data.get("results", {})
                model_name = (
                    data.get("model_name")
                    or model_dir.replace("/", "/", 1)  # org/model
                )

                found_any = False
                for subject in MMLU_SUBJECTS:
                    score = None
                    for key_pattern in [
                        f"harness|hendrycksTest-{subject}|5",
                        f"hendrycksTest-{subject}",
                        f"mmlu:{subject}",
                        subject,
                    ]:
                        if key_pattern in results:
                            entry = results[key_pattern]
                            if isinstance(entry, dict):
                                score = (entry.get("acc_norm")
                                         or entry.get("acc")
                                         or entry.get("acc_norm,none")
                                         or entry.get("acc,none"))
                            elif isinstance(entry, (int, float)):
                                score = entry
                            if score is not None:
                                break
                    if score is not None:
                        records.append({
                            "model": model_name,
                            "task": subject,
                            "score": float(score),
                        })
                        found_any = True
                if not found_any:
                    bad += 1
            except Exception:
                bad += 1
                continue

        print(f"  Parsed {len(model_files)} models, got {len(records)} scores "
              f"({bad} skipped)")

    except ImportError:
        print("  huggingface_hub not installed")
    except Exception as e:
        print(f"  Strategy 1 failed: {e}")

    if records:
        print(f"  Got {len(records)} records from old leaderboard")

    # ------------------------------------------------------------------
    # Strategy 2: Use the aggregated leaderboard datasets on HF
    # ------------------------------------------------------------------
    if not records:
        try:
            print("  Trying HF datasets approach …")
            from datasets import load_dataset

            # Try the v1 leaderboard dataset (without trust_remote_code)
            for ds_name in [
                "open-llm-leaderboard/results",
                "open-llm-leaderboard/contents",
            ]:
                try:
                    ds = load_dataset(ds_name)
                    for split in ds:
                        for row in tqdm(ds[split], desc=f"Parsing {ds_name}:{split}"):
                            model = row.get("model_name", row.get("Model",
                                      row.get("fullname", row.get("model", "?"))))
                            for key, val in row.items():
                                # Match MMLU subject keys
                                for subject in MMLU_SUBJECTS:
                                    if subject in key and isinstance(val, (int, float)):
                                        if not np.isnan(val):
                                            records.append({
                                                "model": model,
                                                "task": subject,
                                                "score": val,
                                            })
                    if records:
                        print(f"  Got {len(records)} from {ds_name}")
                        break
                except Exception as e:
                    tqdm.write(f"  {ds_name}: {e}")
                    continue
        except ImportError:
            print("  datasets library not installed")

    # ------------------------------------------------------------------
    # Strategy 3: Download from known MMLU leaderboard CSV (GitHub)
    # ------------------------------------------------------------------
    if not records:
        print("  Trying GitHub raw results …")
        # Some community-maintained MMLU result collections
        urls = [
            "https://raw.githubusercontent.com/hendrycks/test/master/results.csv",
        ]
        headers = _hf_headers()
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code != 200:
                    continue
                df_raw = pd.read_csv(pd.io.common.StringIO(resp.text))
                if df_raw.shape[0] > 0:
                    # Reshape to long format
                    model_col = df_raw.columns[0]
                    for col in df_raw.columns[1:]:
                        col_clean = col.strip().lower().replace(" ", "_")
                        if col_clean in MMLU_SUBJECTS:
                            for _, row in df_raw.iterrows():
                                val = row[col]
                                if pd.notna(val):
                                    records.append({
                                        "model": str(row[model_col]),
                                        "task": col_clean,
                                        "score": float(val),
                                    })
                    if records:
                        print(f"  Got {len(records)} from {url}")
                        break
            except Exception as e:
                print(f"  {url}: {e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if not records:
        print("WARNING: Could not download MMLU per-subject results.")
        print("  Saving empty placeholder files.")
        pd.DataFrame().to_csv(csv_path)
        np.savez(npz_path, B=np.empty((0, 0)), O=np.empty((0, 0)),
                 model_names=np.array([]), benchmark_names=np.array([]))
        return

    df_long = pd.DataFrame(records)
    df = df_long.pivot_table(
        index="model", columns="task", values="score", aggfunc="first"
    )

    # Drop subjects with very few models
    min_obs = max(3, len(df) * 0.1)
    df = df.loc[:, df.notna().sum() >= min_obs]
    df = df.dropna(how="all")

    _save(df, csv_path, npz_path)


def _save(df: pd.DataFrame, csv_path: pathlib.Path, npz_path: pathlib.Path):
    print(f"  Score matrix: {df.shape[0]} models × {df.shape[1]} subjects")
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
