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

DATA = pathlib.Path(__file__).resolve().parent
src = DATA / "benchpress.json"

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

out = DATA / "benchpress.matrix.npz"
np.savez(out, B=B_out, O=O, model_names=model_names, benchmark_names=benchmark_names)
print(f"Saved {out} ({M} × {N})")
