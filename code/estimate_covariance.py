"""Estimate covariance matrices for the three experiment matrices via EM.

Loads each .matrix.npz file, standardizes columns to zero mean / unit variance,
runs the EM algorithm to handle missing entries, and saves the results.

Output per matrix (saved to data/{name}.cov.npz):
    mu       : (N,) mean vector (in standardized space)
    Sigma    : (N, N) covariance matrix (= correlation matrix, since standardized)
    B_imp    : (M, N) imputed score matrix (original scale)
    col_mean : (N,) per-column mean used for standardization
    col_std  : (N,) per-column std used for standardization
    model_names    : (M,) model name strings
    benchmark_names: (N,) benchmark name strings
    n_iter   : number of EM iterations
    converged: whether EM converged
"""

import sys
import pathlib
import numpy as np

# Add code/ and data/ to path
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import covariance_dir, data_dir

DATA = data_dir()
COV_OUT = covariance_dir()
COV_OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(DATA))

from em_cov import em_cov

MATRICES = ["mmlu", "mteb", "merged"]


def load_matrix(name: str) -> dict:
    """Load a .matrix.npz file."""
    path = DATA / f"{name}.matrix.npz"
    d = np.load(path, allow_pickle=True)
    return {
        "B": d["B"].astype(np.float64),
        "O": d["O"].astype(np.float64),
        "model_names": d["model_names"],
        "benchmark_names": d["benchmark_names"],
    }


def standardize(B, O):
    """Standardize observed entries to zero mean, unit variance per column.

    Returns standardized B, column means, column stds.
    Unobserved entries (O=0) remain zero.
    """
    col_sum = (B * O).sum(axis=0)
    col_count = O.sum(axis=0)
    col_count = np.maximum(col_count, 1.0)
    col_mean = col_sum / col_count

    # Variance from observed entries
    B_centered = O * (B - col_mean[None, :])
    col_var = (B_centered ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))

    B_std = O * (B - col_mean[None, :]) / col_std[None, :]
    return B_std, col_mean, col_std


def main():
    for name in MATRICES:
        print(f"{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        mat = load_matrix(name)
        B, O = mat["B"], mat["O"]
        M, N = B.shape
        n_obs = int(O.sum())
        n_total = B.size
        pct_miss = 100 * (1 - n_obs / n_total)
        print(f"  Shape: {M} models x {N} tasks")
        print(f"  Missing: {n_total - n_obs}/{n_total} ({pct_miss:.1f}%)")

        # Standardize
        B_std, col_mean, col_std = standardize(B, O)

        # Run EM with stronger PSD regularization when covariance is
        # rank-deficient or very sparse. Ledoit-Wolf shrinkage itself is
        # only used in the rank-deficient case.
        obs_frac = n_obs / n_total
        regularized = M < N or obs_frac < 0.5
        use_shrink = "auto" if M < N else 0.0
        if regularized:
            reason = f"M={M} < N={N}" if M < N else f"observed={obs_frac:.1%}"
            print(f"  {reason}: using regularized EM settings")
        n_iter = 1000 if regularized else 500
        em_tol = 5e-4 if regularized else 1e-6
        em_eps = 1e-3 if regularized else 1e-6
        result = em_cov(B_std, O, verbose=True, max_iter=n_iter, tol=em_tol,
                        shrinkage=use_shrink, eps_psd=em_eps)

        mu_std = result["mu"]
        Sigma = result["Sigma"]
        B_imp_std = result["B_imp"]

        # Undo standardization on imputed matrix
        B_imp = B_imp_std * col_std[None, :] + col_mean[None, :]
        # Keep original observed values
        B_imp = O * B + (1 - O) * B_imp

        # Diagnostics
        eigvals = np.linalg.eigvalsh(Sigma)[::-1]
        cum_var = np.cumsum(eigvals) / eigvals.sum()
        print(f"  EM: {result['n_iter']} iterations, "
              f"converged={result['converged']}")
        print(f"  Eigenvalue spectrum (top 10):")
        for i in range(min(10, N)):
            print(f"    λ_{i+1:2d} = {eigvals[i]:.4f}  "
                  f"(cumulative: {cum_var[i]:.3f})")
        # How many for 90%, 95%?
        for threshold in [0.90, 0.95, 0.99]:
            k = int(np.searchsorted(cum_var, threshold)) + 1
            print(f"  {threshold:.0%} variance explained by {k} components")

        # Save
        out_path = COV_OUT / f"{name}.cov.npz"
        np.savez(
            out_path,
            mu=mu_std,
            Sigma=Sigma,
            B_imp=B_imp,
            col_mean=col_mean,
            col_std=col_std,
            model_names=mat["model_names"],
            benchmark_names=mat["benchmark_names"],
            n_iter=result["n_iter"],
            converged=result["converged"],
            eigvals=eigvals,
            loglik=np.array(result["loglik"]),
        )
        print(f"  Saved to {out_path}")
        print()


if __name__ == "__main__":
    main()
