"""Plot eigenvalue spectrum of correlation matrices for MMLU, MTEB, Merged.

Reproduces Figure 1 of the paper.  Loads each .matrix.npz file and estimates
the correlation matrix via EM (using Ledoit-Wolf shrinkage when M < N).  If a
precomputed {name}.cov.npz exists in data/ it is reused; otherwise it is
generated and cached.
"""

import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = pathlib.Path(__file__).resolve().parent
ROOT = CODE.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))

from em_cov import em_cov

MATRICES = {
    "mmlu":   {"label": "MMLU (57 tasks)",    "color": "#4477AA"},
    "mteb":   {"label": "MTEB (56 tasks)",    "color": "#EE7733"},
    "merged": {"label": "Merged (124 tasks)", "color": "#228833"},
}


def estimate_correlation(name: str) -> np.ndarray:
    """Return the (N x N) correlation matrix for `name`, caching to data/."""
    cov_path = DATA / f"{name}.cov.npz"
    if cov_path.exists():
        Sigma = np.load(cov_path, allow_pickle=True)["Sigma"]
    else:
        mat = np.load(DATA / f"{name}.matrix.npz", allow_pickle=True)
        B = mat["B"].astype(np.float64)
        O = mat["O"].astype(np.float64)
        M, N = B.shape

        # Standardize observed entries to mean 0, var 1 per column
        col_count = np.maximum(O.sum(axis=0), 1.0)
        col_mean = (B * O).sum(axis=0) / col_count
        col_var = ((O * (B - col_mean[None, :])) ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
        col_std = np.sqrt(np.maximum(col_var, 1e-12))
        B_std = O * (B - col_mean[None, :]) / col_std[None, :]

        # Run EM (Ledoit-Wolf shrinkage when M < N)
        use_shrink = "auto" if M < N else 0.0
        n_iter = 1000 if M < N else 500
        em_tol = 5e-4 if M < N else 1e-6
        em_eps = 1e-3 if M < N else 1e-6
        result = em_cov(B_std, O, verbose=False, max_iter=n_iter, tol=em_tol,
                        shrinkage=use_shrink, eps_psd=em_eps)
        Sigma = result["Sigma"]
        np.savez(cov_path, Sigma=Sigma, mu=result["mu"])
        print(f"  cached {cov_path.name}")
    return Sigma


fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for name, style in MATRICES.items():
    print(f"[{name}] estimating correlation...")
    Sigma = estimate_correlation(name)
    N = Sigma.shape[0]

    # Convert covariance to correlation matrix: R = D^{-1/2} Sigma D^{-1/2}
    diag_std = np.sqrt(np.diag(Sigma))
    diag_std = np.maximum(diag_std, 1e-12)  # avoid division by zero
    D_inv = 1.0 / diag_std
    R = Sigma * np.outer(D_inv, D_inv)

    eigvals = np.linalg.eigvalsh(R)[::-1]
    cum_var = np.cumsum(eigvals) / eigvals.sum()
    residual = 1.0 - cum_var

    # Truncate data where residual drops below visible range
    k_all = np.arange(1, N + 1)
    mask = residual >= 1e-3
    if mask.any():
        last = np.max(np.where(mask)[0]) + 1  # include last point above threshold
    else:
        last = N
    ax.semilogy(k_all[:last], residual[:last],
                label=style["label"], color=style["color"], linewidth=2)

    # Print summary
    for t in [0.90, 0.95, 0.99]:
        k = int(np.searchsorted(cum_var, t)) + 1
        print(f"{name:8s}: {t:.0%} variance at k={k}")

ax.set_xlabel("Number of components $k$")
ax.set_ylabel("Residual variance fraction")
ax.set_xlim(1, 42)
ax.set_ylim(1e-3, 1.0)
for level, pct in [(0.10, "90%"), (0.05, "95%"), (0.01, "99%")]:
    ax.axhline(level, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(42.5, level, pct, fontsize=8, va="center", color="gray")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT / "eigenspectrum.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"\nSaved to {out_path}")
plt.close()
