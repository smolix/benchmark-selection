"""Build three score matrices for experiments.

1. MMLU: 5454 models x 57 subject-level tasks (already exists, just standardize format)
2. MTEB: 268 models x 56 embedding tasks (already exists, just standardize format)
3. Merged: canonical models (appearing in >= 2 collections) x all tasks from those
   collections, with collection-prefixed column names.

Each matrix is saved as:
  - {name}.matrix.csv  (models x tasks, NaN for missing)
  - {name}.matrix.npz  (B=scores, O=observation mask, model_names, benchmark_names)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(__file__).resolve().parent

MERGED_AUXILIARY_COLUMNS = {
    "alpaca_eval": [
        "standard_error",
        "n_wins",
        "n_wins_base",
        "n_draws",
        "n_total",
        "avg_length",
        "lc_standard_error",
    ],
    "arena_hard": [
        "rating_q025",
        "rating_q975",
        "avg_tokens",
    ],
}


def save_matrix(name, df, output_dir):
    """Save a DataFrame as both CSV and NPZ."""
    csv_path = output_dir / f"{name}.matrix.csv"
    npz_path = output_dir / f"{name}.matrix.npz"

    df.to_csv(csv_path)

    B = df.values.astype(np.float64)
    O = (~np.isnan(B)).astype(np.float64)
    np.savez(npz_path,
             B=np.nan_to_num(B, nan=0.0),
             O=O,
             model_names=np.array(df.index.tolist()),
             benchmark_names=np.array(df.columns.tolist()))

    n_obs = int(O.sum())
    n_total = B.size
    print(f"  {name}: {df.shape[0]} models x {df.shape[1]} tasks, "
          f"observed={n_obs}/{n_total} ({100*n_obs/n_total:.1f}%), "
          f"missing={n_total - n_obs}")


def build_matrices(input_dir, output_dir, include_auxiliary_metrics=False):
    """Build all matrix files from source CSVs."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading source files from {input_dir}/")
    print(f"Writing matrix files to {output_dir}/")

    # --- 1. MMLU ---
    print("Building MMLU matrix...")
    mmlu = pd.read_csv(input_dir / "mmlu.csv", index_col=0)
    # Drop models with mostly-missing rows (< 50% observed)
    obs_frac = mmlu.notna().mean(axis=1)
    drop_models = obs_frac[obs_frac < 0.5].index.tolist()
    if drop_models:
        print(f"  Dropping {len(drop_models)} near-empty models: {drop_models}")
        mmlu = mmlu.drop(index=drop_models)
    save_matrix("mmlu", mmlu, output_dir)

    # --- 2. MTEB ---
    print("Building MTEB matrix...")
    mteb = pd.read_csv(input_dir / "mteb.csv", index_col=0)
    # Drop models with only 1 observed benchmark (no pairwise covariance signal)
    obs_count = mteb.notna().sum(axis=1)
    drop_models_mteb = obs_count[obs_count <= 1].index.tolist()
    if drop_models_mteb:
        print(f"  Dropping {len(drop_models_mteb)} single-benchmark models: {drop_models_mteb}")
        mteb = mteb.drop(index=drop_models_mteb)
    save_matrix("mteb", mteb, output_dir)

    # --- 3. Merged ---
    print("Building merged matrix...")

    with open(input_dir / "canonical_mapping.json") as f:
        canon = json.load(f)

    # Only keep models in >= 2 collections (all 118 satisfy this)
    groups = [g for g in canon["canonical_models"] if len(g["datasets"]) >= 2]
    groups.sort(key=lambda g: g["canonical_name"].lower())

    # Load all per-collection DataFrames
    collections = ["mmlu", "open_llm", "helm", "alpaca_eval", "arena_hard",
                   "livebench", "wildbench", "mt_bench", "bigcode", "mteb"]

    collection_dfs = {}
    for coll in collections:
        path = input_dir / f"{coll}.csv"
        if path.exists():
            collection_dfs[coll] = pd.read_csv(path, index_col=0)

    if not include_auxiliary_metrics:
        for coll, columns in MERGED_AUXILIARY_COLUMNS.items():
            if coll not in collection_dfs:
                continue
            drop_cols = [c for c in columns if c in collection_dfs[coll].columns]
            if drop_cols:
                print(f"  Dropping auxiliary {coll} columns: {drop_cols}")
                collection_dfs[coll] = collection_dfs[coll].drop(columns=drop_cols)

    # Build column list: prefix each task with collection name to avoid collisions
    # e.g., "mmlu/abstract_algebra", "helm/GSM8K", etc.
    all_columns = []
    for coll in collections:
        if coll not in collection_dfs:
            continue
        for col in collection_dfs[coll].columns:
            all_columns.append(f"{coll}/{col}")

    # Build the merged matrix
    model_names = [g["canonical_name"] for g in groups]
    merged = pd.DataFrame(np.nan, index=model_names, columns=all_columns)

    for g in groups:
        canon_name = g["canonical_name"]
        for coll, orig_name in g["datasets"].items():
            if coll not in collection_dfs:
                continue
            df = collection_dfs[coll]
            if orig_name not in df.index:
                continue
            row = df.loc[orig_name]
            for col in df.columns:
                prefixed = f"{coll}/{col}"
                val = row[col]
                if pd.notna(val):
                    merged.loc[canon_name, prefixed] = val

    # Drop columns that are entirely NaN (no canonical model has data there)
    merged = merged.dropna(axis=1, how="all")

    save_matrix("merged", merged, output_dir)

    # Print summary by collection
    print("\n  Per-collection coverage in merged matrix:")
    for coll in collections:
        cols = [c for c in merged.columns if c.startswith(f"{coll}/")]
        if not cols:
            continue
        sub = merged[cols]
        n_models_with_data = (sub.notna().any(axis=1)).sum()
        n_obs = sub.notna().sum().sum()
        n_total = sub.size
        print(f"    {coll:>12}: {len(cols):>3} tasks, "
              f"{n_models_with_data:>3} models with data, "
              f"observed={n_obs}/{n_total} ({100*n_obs/n_total:.1f}%)")

    print(f"\nDone. Files saved to {output_dir}/")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATA,
        help="Directory containing source CSV and mapping files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated matrix files. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--include-auxiliary-metrics",
        action="store_true",
        help="Keep auxiliary merged columns such as errors, counts, and token lengths.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    return args


if __name__ == "__main__":
    args = parse_args()
    build_matrices(args.input_dir, args.output_dir, args.include_auxiliary_metrics)
