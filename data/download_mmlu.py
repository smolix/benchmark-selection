#!/usr/bin/env python3
"""Download and assemble MMLU per-subject score matrix.

Produces (in --output-dir, default: this directory):
    mmlu.csv   — tidy CSV (rows = models, columns = 57 MMLU subjects)
    mmlu.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install huggingface_hub pandas numpy requests python-dotenv tqdm

The Open LLM Leaderboard v1 stored per-subject MMLU accuracy in
individual model "details" repos on HuggingFace.  We can also parse
results from the `open-llm-leaderboard/details_*` repos or from the
HF dataset snapshot.

Strategy 1: Snapshot of the HF dataset `open-llm-leaderboard-old/results`
            (``<org>/<model>/results_<timestamp>.json``, keys
            ``harness|hendrycksTest-<subject>|5``), or a local copy of it
            given with --results-dir (e.g. a git clone; the copy used for
            the checks in README.md is at revision RESULTS_REV_CHECKED).
Strategy 2: Scrape the `open-llm-leaderboard/details` repos which
            contain per-task breakdowns including MMLU subjects.
Strategy 3: Use the official MMLU leaderboard CSV from the
            hendrycks/test GitHub repo (limited models only).

Result files per model directory (Strategy 1):
    default                  only the LAST file in path order is read (the
                             rule that produced the frozen mmlu.csv: 5,454
                             rows -> 5,452 in mmlu.matrix.npz).  When that
                             file has no MMLU (a later run of other tasks
                             only), the model is dropped even if an earlier
                             file has all 57 subjects.
    --recover-earlier-files  every subject is taken from the latest file
                             (timestamps compared with ':' read as '-')
                             that contains it.  At RESULTS_REV_CHECKED this
                             adds 1,640 models and leaves every value of the
                             default rows unchanged (frozen 5,452 + 1,640 =
                             7,092 matrix rows).  Opt-in: the paper's matrix
                             keeps the default rule.

The frozen mmlu.csv was downloaded to a case-insensitive (macOS) filesystem,
which merged 18 model directories that differ only by case.  On a
case-sensitive filesystem the default rule gives 5,470 rows (5,468 in the
matrix); README.md has the reconciliation.

Usage:
    python download_mmlu.py                    # online; skips if outputs exist
    cd /tmp && python -I .../data/download_mmlu.py --results-dir CLONE \\
        --output-dir DIR [--recover-earlier-files]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import traceback

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent

RESULTS_REPO = "open-llm-leaderboard-old/results"
# HEAD of the git clone used for the checks in README.md (last commit
# 2024-07-18; also the revision pinned in raw_manifests/olv1items.json)
RESULTS_REV_CHECKED = "23474373f8874f9057d23b97e5a41e911d2721c5"

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


def _subject_scores(results) -> dict:
    """{subject: score} found in one results dict (same keys and metric
    order as the per-subject loop in main())."""
    scores = {}
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
            scores[subject] = float(score)
    return scores


def _latest_per_subject(model_dir: str, files: list[str]):
    """--recover-earlier-files: every subject from the latest result file
    that contains it.  File timestamps are compared with ':' read as '-'
    (both spellings occur); unreadable files are skipped.

    Returns (model_name, {subject: score}, used_earlier_file).  model_name
    is taken as in the default rule, from the last file in path order.
    """
    last = sorted(files)[-1]
    scores, source = {}, {}
    for fpath in sorted(files, key=lambda p: os.path.basename(p).replace(":", "-")):
        try:
            with open(fpath) as f:
                found = _subject_scores(json.load(f).get("results", {}))
        except Exception:
            continue
        for subject, score in found.items():
            scores[subject] = score
            source[subject] = fpath
    model_name = model_dir
    try:
        with open(last) as f:
            model_name = json.load(f).get("model_name") or model_dir
    except Exception:
        pass
    return model_name, scores, any(p != last for p in source.values())


def _git_head(repo: pathlib.Path) -> str | None:
    """HEAD commit of a local git clone (read from .git, no subprocess)."""
    git = repo / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    except OSError:
        pass
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=pathlib.Path,
                        help=f"local copy of {RESULTS_REPO} (e.g. a git clone); "
                             "skips the download and never falls back to the "
                             "online strategies")
    parser.add_argument("--revision",
                        help="online: revision of the snapshot (default: HEAD, as before)")
    parser.add_argument("--output-dir", type=pathlib.Path, default=HERE,
                        help="where mmlu.{csv,npz} are written (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the outputs already exist")
    parser.add_argument("--recover-earlier-files", action="store_true",
                        help="take each subject from the latest result file that "
                             "contains it instead of reading only the last file per "
                             "model (recovers models whose last run had no MMLU)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()
    csv_path = out_dir / "mmlu.csv"
    npz_path = out_dir / "mmlu.npz"
    if not args.force and csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"MMLU data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} subjects).  Skipping download.")
            return
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.results_dir is None:
        # Load HF_TOKEN (and any other vars) from .env next to this script;
        # offline runs (--results-dir) never read it.
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")

    print("Downloading MMLU per-subject results …")
    if args.recover_earlier_files:
        print("  --recover-earlier-files: each subject from the latest file "
              "that contains it")

    # ------------------------------------------------------------------
    # Strategy 1: Download from open-llm-leaderboard-old/results repo
    # This is a single HF dataset repo containing ~10K JSON files with
    # per-subject MMLU scores in keys like harness|hendrycksTest-{subject}|5.
    # ------------------------------------------------------------------
    records = []
    try:
        from collections import defaultdict
        import glob as _glob

        REPO_ID = RESULTS_REPO
        if args.results_dir is not None:
            local_dir = str(args.results_dir.resolve())
            head = _git_head(args.results_dir.resolve())
            print(f"  Reading local copy of {REPO_ID}: {local_dir}"
                  + (f" (git HEAD {head})" if head else ""))
            if head and head != RESULTS_REV_CHECKED:
                print(f"  note: README.md checks refer to {RESULTS_REV_CHECKED}")
        else:
            from huggingface_hub import snapshot_download
            print(f"  Downloading snapshot of {REPO_ID} (parallel) …")
            local_dir = snapshot_download(
                REPO_ID, repo_type="dataset",
                allow_patterns="*.json",
                revision=args.revision,
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
        n_from_earlier = 0
        for model_dir in tqdm(model_files, desc="Parsing results",
                              unit="model"):
            if args.recover_earlier_files:
                model_name, scores, earlier = _latest_per_subject(
                    model_dir, model_files[model_dir])
                for subject in MMLU_SUBJECTS:
                    if subject in scores:
                        records.append({
                            "model": model_name,
                            "task": subject,
                            "score": scores[subject],
                        })
                bad += not scores
                n_from_earlier += earlier
                continue
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
        if args.recover_earlier_files:
            print(f"  {n_from_earlier} models use a file other than their last "
                  f"one for some subject")

    except ImportError:
        print("  huggingface_hub not installed")
    except Exception as e:
        print(f"  Strategy 1 failed: {e}")

    if records:
        print(f"  Got {len(records)} records from old leaderboard")
    elif args.results_dir is not None:
        print(f"ERROR: no MMLU scores found under {args.results_dir}")
        sys.exit(1)

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
