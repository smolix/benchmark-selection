"""Build score matrix from BenchPress llm_benchmark_data.json.

Produces benchpress.matrix.npz with:
  B : (M, N) score matrix (NaN for missing)
  O : (M, N) observation mask
  model_names : (M,) array of model names
  benchmark_names : (N,) array of benchmark names
"""

import json
import pathlib
import numpy as np


DEFAULT_DATA = pathlib.Path(__file__).resolve().parent


def build_benchpress(input_dir, output_dir):
    input_dir = pathlib.Path(input_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    src = input_dir / "benchpress.json"
    with open(src) as f:
        d = json.load(f)

    models = d["models"]
    benchmarks = d["benchmarks"]
    scores = d["scores"]

    model_ids = [m["id"] for m in models]
    bench_ids = [b["id"] for b in benchmarks]
    model_names = np.array([m["name"] for m in models])
    benchmark_names = np.array([b["name"] for b in benchmarks])

    model_idx = {mid: i for i, mid in enumerate(model_ids)}
    bench_idx = {bid: i for i, bid in enumerate(bench_ids)}

    M, N = len(models), len(benchmarks)
    B = np.full((M, N), np.nan, dtype=np.float64)
    O = np.zeros((M, N), dtype=np.float64)

    for s in scores:
        i = model_idx[s["model_id"]]
        j = bench_idx[s["benchmark_id"]]
        B[i, j] = s["score"]
        O[i, j] = 1.0

    # Replace NaN with 0 in B (our convention: unobserved = 0, mask in O)
    B_out = np.where(O == 1, B, 0.0)

    obs_frac = O.sum() / (M * N)
    print(f"BenchPress: {M} models × {N} benchmarks, {obs_frac:.1%} observed")
    print(f"Total scores: {int(O.sum())}")

    # Check for models or benchmarks with very few observations
    model_obs = O.sum(axis=1)
    bench_obs = O.sum(axis=0)
    print(f"Models: min {int(model_obs.min())} obs, max {int(model_obs.max())} obs")
    print(f"Benchmarks: min {int(bench_obs.min())} obs, max {int(bench_obs.max())} obs")

    # Drop models with fewer than 2 observations (can't contribute to covariance)
    keep = model_obs >= 2
    if not keep.all():
        print(f"Dropping {(~keep).sum()} models with <2 observations")
        B_out = B_out[keep]
        O = O[keep]
        model_names = model_names[keep]
        M = len(model_names)

    out = output_dir / "benchpress.matrix.npz"
    np.savez(out, B=B_out, O=O, model_names=model_names, benchmark_names=benchmark_names)
    print(f"Saved {out} ({M} x {N})")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        default=DEFAULT_DATA,
        help="Directory containing benchpress.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Directory for benchpress.matrix.npz. Defaults to --input-dir.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    return args


if __name__ == "__main__":
    args = parse_args()
    build_benchpress(args.input_dir, args.output_dir)
