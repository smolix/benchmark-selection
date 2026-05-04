# `data/` — score matrices and per-leaderboard sources

## Primary release matrices (used in the paper)

| File | Shape (M × N) | % observed | Description |
|------|---------------|-----------:|-------------|
| `mmlu.matrix.npz`       | 5452 × 57   | 100.0 | Per-subject MMLU scores; full grid. |
| `mteb.matrix.npz`       | 263 × 56    |  77.3 | English MTEB tasks across embedding models. |
| `merged.matrix.npz`     | 118 × 114   |  31.1 | Models that appear in ≥ 2 leaderboards (Appendix E), with score-like collection-prefixed task columns. |
| `benchpress.matrix.npz` | 83 × 49     |  33.8 | BenchPress release matrix; appendix experiments only. |

CSV mirrors of the same matrices (`*.matrix.csv`) are provided for inspection.

### File format (`*.matrix.npz`)

```
B               (M, N) float64   scores; entries with O=0 are 0.0 placeholders
O               (M, N) float64   observation mask, 1 = observed, 0 = missing
model_names     (M,)   <U…       model identifiers (rows of B)
benchmark_names (N,)   <U…       benchmark/task identifiers (columns of B)
```

The matrices are loaded directly via `np.load(path, allow_pickle=True)`; the
helper `code/em_cov.py` consumes `(B, O)` directly.

## Per-leaderboard sources

These are the inputs to `build_matrices.py` — the script that assembles the
three primary matrices.

| File | Source | Used to build |
|------|--------|---------------|
| `mmlu.npz` / `mmlu.csv`             | per-subject MMLU leaderboard       | `mmlu.matrix.npz` |
| `mteb.npz` / `mteb.csv`             | MTEB English leaderboard            | `mteb.matrix.npz` |
| `open_llm.npz` / `open_llm.csv`     | Open LLM Leaderboard v2             | `merged.matrix.npz` |
| `helm.npz` / `helm.csv`             | HELM Lite                           | `merged.matrix.npz` |
| `alpaca_eval.npz` / `alpaca_eval.csv` | AlpacaEval 2                      | `merged.matrix.npz` |
| `arena_hard.npz` / `arena_hard.csv` | Arena-Hard-Auto                     | `merged.matrix.npz` |
| `livebench.npz` / `livebench.csv`   | LiveBench                           | `merged.matrix.npz` |
| `wildbench.npz` / `wildbench.csv`   | WildBench                           | `merged.matrix.npz` |
| `mt_bench.npz` / `mt_bench.csv`     | MT-Bench                            | `merged.matrix.npz` |
| `bigcode.npz` / `bigcode.csv`       | BigCodeBench                        | `merged.matrix.npz` |
| `bigbench.npz` / `bigbench.csv`     | BIG-Bench Lite (omitted from paper) | (not used) |
| `benchpress.json`                   | BenchPress release JSON             | `benchpress.matrix.npz` |

`canonical_mapping.json` lists the model-name canonicalization used to merge
across leaderboards (Appendix E).  The merged matrix excludes auxiliary count,
uncertainty, and length columns from AlpacaEval and Arena-Hard; pass
`--include-auxiliary-metrics` to `build_matrices.py` to retain them.

## Building from scratch

```bash
# 1. download every source leaderboard (writes npz + csv per source)
bash data/download_all.sh

# 2. assemble the three primary matrices (mmlu, mteb, merged)
python data/build_matrices.py
```

Each download script is a small standalone fetcher; rerun individually if a
single source needs refreshing. `download_benchpress.py` produces both
`benchpress.json` and `benchpress.matrix.npz`.

## Loading helper

`load_data.py` exposes the per-leaderboard sources via a single call:

```python
from load_data import load_dataset
B, O, model_names, bench_names = load_dataset("open_llm")
```

For the primary release matrices, prefer `np.load` directly:

```python
import numpy as np
d = np.load("data/mmlu.matrix.npz", allow_pickle=True)
B, O = d["B"], d["O"]
```
