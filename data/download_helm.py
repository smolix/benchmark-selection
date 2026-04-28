#!/usr/bin/env python3
"""Download and assemble the HELM Lite score matrix.

Produces:
    helm.csv   — tidy CSV (rows = models, columns = scenarios)
    helm.npz   — B, O, model_names, benchmark_names

Requirements:
    pip install pandas numpy requests

HELM publishes benchmark results as JSON via GCS.  The directory layout is:

    benchmark_output/
      releases/{version}/
        summary.json          ← minimal: {release, suites: ["lite"], date}
      run_suites/{suite}/
        groups/
          {group_name}.json   ← table with header + rows
        runs/
          {run_spec}/
            run_spec.json
            stats.json        ← per-metric scores

We try several strategies to extract the model × scenario score matrix.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent

# HELM benchmark output base URL (GCS public bucket)
HELM_BASE = "https://storage.googleapis.com/crfm-helm-public"


def _fetch_json(url: str, label: str = "") -> dict | list | None:
    """Fetch JSON from a URL, return None on failure."""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        if label:
            tqdm.write(f"    {label}: {e}")
        return None


# ---------------------------------------------------------------------------
# Group-table parsers
# ---------------------------------------------------------------------------

def _parse_header_rows_table(data: dict | list) -> list[dict]:
    """Parse HELM group-table format:
        { "title": "...", "header": [...], "rows": [[cell, ...], ...] }
    or a list of such tables.
    """
    records = []
    tables = data if isinstance(data, list) else [data]
    for table in tables:
        if not isinstance(table, dict):
            continue
        header = table.get("header", [])
        rows = table.get("rows", [])
        if not header or not rows:
            continue

        # header is a list of cells; first column is typically the model
        col_names = []
        for h in header:
            if isinstance(h, dict):
                col_names.append(h.get("value", h.get("name", str(h))))
            else:
                col_names.append(str(h))

        for row in rows:
            if not isinstance(row, list) or len(row) != len(col_names):
                continue
            # First cell = model
            cell0 = row[0]
            if isinstance(cell0, dict):
                model = cell0.get("value", cell0.get("name", str(cell0)))
            else:
                model = str(cell0)

            # Remaining cells = scenario scores
            for col_name, cell in zip(col_names[1:], row[1:]):
                score = None
                if isinstance(cell, (int, float)):
                    score = cell
                elif isinstance(cell, dict):
                    score = cell.get("value", cell.get("mean"))
                    if isinstance(score, str):
                        try:
                            score = float(score)
                        except ValueError:
                            score = None
                elif isinstance(cell, str):
                    try:
                        score = float(cell)
                    except ValueError:
                        pass
                if score is not None:
                    records.append({"model": model, "scenario": col_name, "score": score})

    return records


def _parse_flat_table(data) -> list[dict]:
    """Parse flat list-of-dicts or dict-with-rows table."""
    records = []
    rows = data
    if isinstance(data, dict):
        rows = data.get("data", data.get("rows", data.get("items", [])))
    if not isinstance(rows, list):
        return records
    for row in rows:
        if not isinstance(row, dict):
            continue
        model = row.get("model", row.get("model_name", "?"))
        for key, val in row.items():
            if key in ("model", "model_name", "model_deployment"):
                continue
            score = None
            if isinstance(val, (int, float)):
                score = val
            elif isinstance(val, dict):
                score = val.get("mean", val.get("value"))
            if score is not None:
                records.append({"model": model, "scenario": key, "score": score})
    return records


# ---------------------------------------------------------------------------
# Per-run stat parser
# ---------------------------------------------------------------------------

def _parse_run_stats(base_url: str) -> list[dict]:
    """Download individual run stats from runs/ directory."""
    records = []
    run_specs = _fetch_json(f"{base_url}/run_specs.json")
    if not run_specs:
        return records

    specs = run_specs if isinstance(run_specs, list) else \
            run_specs.get("run_specs", run_specs.get("items", []))

    for spec in tqdm(specs, desc="Fetching per-run stats", unit="run"):
        if not isinstance(spec, dict):
            continue
        run_name = spec.get("name", spec.get("run_spec_key", ""))
        if not run_name:
            continue

        model = (spec.get("adapter_spec", {}).get("model", "") or
                 spec.get("model", "") or
                 spec.get("model_deployment", ""))
        sc_spec = spec.get("scenario_spec", {})
        scenario = (sc_spec.get("class_name", "").split(".")[-1] or
                    sc_spec.get("name", "") or
                    spec.get("scenario", ""))

        if not model or not scenario:
            parts = run_name.split(":")
            if len(parts) >= 2:
                scenario = scenario or parts[0]
                for p in parts[1:]:
                    if p.startswith("model="):
                        model = model or p.split("=", 1)[1].split(",")[0]
        if not model or not scenario:
            continue

        stats = _fetch_json(f"{base_url}/runs/{run_name}/stats.json")
        if not stats:
            continue
        stat_list = stats if isinstance(stats, list) else stats.get("stats", [])
        for stat in stat_list:
            if not isinstance(stat, dict):
                continue
            name_obj = stat.get("name", {})
            metric = name_obj.get("name", "") if isinstance(name_obj, dict) else str(name_obj)
            if metric in ("exact_match", "quasi_exact_match", "accuracy", "f1_score"):
                score = stat.get("mean", stat.get("sum"))
                if score is not None:
                    records.append({"model": model, "scenario": scenario, "score": score})
                    break

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check for existing output
    csv_path = HERE / "helm.csv"
    npz_path = HERE / "helm.npz"
    if csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"HELM data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} scenarios).  Skipping download.")
            return

    print("Fetching HELM Lite results …")

    # ------------------------------------------------------------------
    # Step 1: Discover suite names from summary.json
    # ------------------------------------------------------------------
    suite_names = []
    versions = ["latest", "v1.0.0", "v0.5.0", "v0.4.0", "v0.3.0"]
    found_version = None

    for ver in tqdm(versions, desc="Trying HELM versions", unit="ver"):
        url = f"{HELM_BASE}/lite/benchmark_output/releases/{ver}/summary.json"
        summary = _fetch_json(url)
        if summary is None:
            continue
        found_version = ver
        tqdm.write(f"  Found summary.json at {ver}")
        if isinstance(summary, dict):
            tqdm.write(f"  Keys: {list(summary.keys())}")
            raw_suites = summary.get("suites", [])
            # suites can be a list of strings or list of dicts
            for s in raw_suites:
                if isinstance(s, str):
                    suite_names.append(s)
                elif isinstance(s, dict):
                    suite_names.append(s.get("name", s.get("suite", "")))
            tqdm.write(f"  Suites: {suite_names}")
        break

    if not suite_names:
        suite_names = ["lite"]  # default for HELM Lite
        print(f"  Using default suite name: {suite_names}")

    # ------------------------------------------------------------------
    # Step 2: Fetch group tables from run_suites/{suite}/groups/
    # ------------------------------------------------------------------
    records = []

    # The HELM GCS layout puts group tables under:
    #   benchmark_output/run_suites/{suite}/groups/{group_name}.json
    # Common group names for HELM Lite:
    group_names = [
        "core_scenarios",
        "core_scenarios_accuracy",
        "targeted_evaluations",
        "targeted_evaluations_accuracy",
    ]

    # Also try the releases/{version}/groups/ path
    url_templates = []
    for suite in suite_names:
        for gname in group_names:
            url_templates.append(
                f"{HELM_BASE}/lite/benchmark_output/run_suites/{suite}/groups/{gname}.json"
            )
    if found_version:
        for gname in group_names:
            url_templates.append(
                f"{HELM_BASE}/lite/benchmark_output/releases/{found_version}/groups/{gname}.json"
            )

    for url in tqdm(url_templates, desc="Trying group tables", unit="url"):
        data = _fetch_json(url)
        if data is None:
            continue
        tqdm.write(f"  Found: {url.split('benchmark_output/')[-1]}")

        # Debug: show structure
        if isinstance(data, dict):
            tqdm.write(f"    Type: dict, keys: {list(data.keys())[:10]}")
        elif isinstance(data, list):
            tqdm.write(f"    Type: list of {len(data)} items")
            if data and isinstance(data[0], dict):
                tqdm.write(f"    First item keys: {list(data[0].keys())[:10]}")

        # Try header+rows format first, then flat format
        recs = _parse_header_rows_table(data)
        if not recs:
            recs = _parse_flat_table(data)
        if recs:
            tqdm.write(f"    → {len(recs)} records")
            records.extend(recs)

    if records:
        print(f"  Total: {len(records)} records from group tables")

    # ------------------------------------------------------------------
    # Step 3: Try per-run stats if group tables failed
    # ------------------------------------------------------------------
    if not records:
        print("  Group tables yielded no data. Trying per-run stats …")
        for suite in suite_names:
            base = f"{HELM_BASE}/lite/benchmark_output/run_suites/{suite}"
            records = _parse_run_stats(base)
            if records:
                print(f"  Extracted {len(records)} records from run stats ({suite})")
                break

    # ------------------------------------------------------------------
    # Step 4: Try HuggingFace datasets as fallback
    # ------------------------------------------------------------------
    if not records:
        try:
            print("  Trying HuggingFace datasets …")
            from datasets import load_dataset
            for ds_name in ["open-llm-leaderboard/helm-results",
                            "HuggingFaceH4/open_llm_leaderboard_helm"]:
                try:
                    ds = load_dataset(ds_name)
                    for split in ds:
                        for row in ds[split]:
                            model = row.get("model", row.get("model_name", "?"))
                            for k, v in row.items():
                                if k in ("model", "model_name"):
                                    continue
                                if isinstance(v, (int, float)) and not np.isnan(v):
                                    records.append({"model": model, "scenario": k, "score": v})
                    if records:
                        print(f"  Found {len(records)} records in {ds_name}")
                        break
                except Exception:
                    continue
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if not records:
        print("WARNING: Could not extract structured scores from HELM.")
        print("  Saving empty placeholder files.")
        print("  To populate, try one of:")
        print("    1. pip install crfm-helm && helm-summarize --suite lite")
        print(f"    2. gsutil rsync -r gs://crfm-helm-public/lite/benchmark_output ./helm_raw")
        print("    3. Manually export from https://crfm.stanford.edu/helm/lite/latest/")

        pd.DataFrame().to_csv(csv_path)
        np.savez(npz_path,
                 B=np.empty((0, 0)), O=np.empty((0, 0)),
                 model_names=np.array([]), benchmark_names=np.array([]))
        return

    # Pivot to matrix form
    df_long = pd.DataFrame(records)
    df = df_long.pivot_table(index="model", columns="scenario", values="score",
                             aggfunc="first")

    # Filter out metadata columns — keep only actual benchmark scores.
    # HELM group tables include columns like "# eval", "# output tokens",
    # "# prompt tokens", "# train", "Observed inference time (s)", "truncated".
    # We keep columns whose names contain known score-metric suffixes.
    _SCORE_PATTERNS = re.compile(
        r"(- EM$|- F1$|- BLEU|- Equivalent|- acc|- accuracy|"
        r"win rate|^Mean |^Overall )", re.IGNORECASE,
    )
    score_cols = [c for c in df.columns if _SCORE_PATTERNS.search(c)]
    if score_cols:
        # Strip the common "Scenario - Metric" prefix down to just Scenario
        # e.g. "GSM8K - EM" → "GSM8K", but keep "Mean win rate" as-is.
        rename = {}
        for c in score_cols:
            parts = c.rsplit(" - ", 1)
            if len(parts) == 2 and parts[0] not in rename.values():
                rename[c] = parts[0]
            else:
                rename[c] = c
        df = df[score_cols].rename(columns=rename)
    else:
        # Fallback: drop columns that look like metadata
        _META_PATTERNS = re.compile(
            r"(# eval|# output|# prompt|# train|inference time|truncated)",
            re.IGNORECASE,
        )
        df = df[[c for c in df.columns if not _META_PATTERNS.search(c)]]

    # Drop scenarios with very few observations
    min_obs = max(3, len(df) * 0.1)
    df = df.loc[:, df.notna().sum() >= min_obs]
    df = df.dropna(how="all")

    print(f"  Score matrix: {df.shape[0]} models × {df.shape[1]} scenarios")
    n_missing = df.isna().sum().sum()
    n_total = df.shape[0] * df.shape[1]
    if n_total > 0:
        print(f"  Missing entries: {n_missing}/{n_total} ({n_missing/n_total:.1%})")
    print(f"  Scenarios: {list(df.columns)}")

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
