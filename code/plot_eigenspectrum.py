"""Plot eigenvalue spectrum of correlation matrices for MMLU, MTEB, Merged."""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = pathlib.Path(__file__).resolve().parent

from path_config import covariance_dir, figures_dir

DATA = covariance_dir()
OUT = figures_dir()
OUT.mkdir(parents=True, exist_ok=True)

MATRICES = {
    "mmlu":   {"label": "MMLU (57 tasks)",    "color": "#4477AA"},
    "mteb":   {"label": "MTEB (56 tasks)",    "color": "#EE7733"},
    "merged": {"label": "Merged (114 tasks)", "color": "#228833"},
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for name, style in MATRICES.items():
    d = np.load(DATA / f"{name}.cov.npz", allow_pickle=True)
    Sigma = d["Sigma"]
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
