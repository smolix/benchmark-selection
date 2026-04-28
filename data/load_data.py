"""Unified loader for benchmark score matrices.

Usage:
    from load_data import load_dataset

    B, O, model_names, bench_names = load_dataset("open_llm")
    B, O, model_names, bench_names = load_dataset("helm")
    B, O, model_names, bench_names = load_dataset("mteb")
    B, O, model_names, bench_names = load_dataset("mmlu")
    B, O, model_names, bench_names = load_dataset("alpaca_eval")
    B, O, model_names, bench_names = load_dataset("mt_bench")
    B, O, model_names, bench_names = load_dataset("arena_hard")
    B, O, model_names, bench_names = load_dataset("livebench")
    B, O, model_names, bench_names = load_dataset("wildbench")
    B, O, model_names, bench_names = load_dataset("bigcode")

Each call returns:
    B : (M, N) float64 score matrix (0 where unobserved)
    O : (M, N) float64 observation mask (1 = observed, 0 = missing)
    model_names  : (M,) array of strings
    bench_names  : (N,) array of strings
"""

from __future__ import annotations

import pathlib

import numpy as np
from numpy.typing import NDArray

HERE = pathlib.Path(__file__).resolve().parent

AVAILABLE = [
    "open_llm",
    "helm",
    "mteb",
    "mmlu",
    "alpaca_eval",
    "mt_bench",
    "arena_hard",
    "livebench",
    "wildbench",
    "bigcode",
]


def load_dataset(
    name: str,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load a pre-downloaded score matrix by name.

    Parameters
    ----------
    name : one of the AVAILABLE dataset names.

    Returns
    -------
    B, O, model_names, benchmark_names
    """
    if name not in AVAILABLE:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {AVAILABLE}.")

    npz_path = HERE / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found.  Run download_all.sh first:\n"
            f"  cd {HERE} && bash download_all.sh"
        )

    data = np.load(npz_path, allow_pickle=True)
    B = data["B"].astype(np.float64)
    O = data["O"].astype(np.float64)
    model_names = data["model_names"]
    benchmark_names = data["benchmark_names"]

    if B.size == 0:
        raise RuntimeError(
            f"Dataset {name!r} is empty (download may have failed).  "
            f"Re-run the download script or check {HERE / f'download_{name}.py'}."
        )

    return B, O, model_names, benchmark_names


def summary(name: str) -> str:
    """Print a summary of a dataset."""
    B, O, models, benches = load_dataset(name)
    M, N = B.shape
    n_missing = int((O == 0).sum())
    n_total = M * N
    frac = n_missing / n_total if n_total > 0 else 0.0
    lines = [
        f"Dataset: {name}",
        f"  Models (M):     {M}",
        f"  Benchmarks (N): {N}",
        f"  Missing:        {n_missing}/{n_total} ({frac:.1%})",
        f"  Benchmarks:     {list(benches)}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    for name in AVAILABLE:
        try:
            print(summary(name))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Dataset: {name}\n  {e}")
        print()
