"""Run greedy entropy selection on MMLU and plot residual variance decay."""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import covariance_dir, figures_dir
from greedy_select import greedy_entropy

DATA = covariance_dir()
OUT = figures_dir()
OUT.mkdir(parents=True, exist_ok=True)

K_MAX = 15

# Load MMLU covariance and convert to correlation matrix
d = np.load(DATA / "mmlu.cov.npz", allow_pickle=True)
Sigma = d["Sigma"]
names = d["benchmark_names"]
N = Sigma.shape[0]

diag_std = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
D_inv = 1.0 / diag_std
R = Sigma * np.outer(D_inv, D_inv)

# Eigenvalue lower bound on residual variance
eigvals = np.linalg.eigvalsh(R)[::-1]
total_var = eigvals.sum()
eig_residual = np.array([1.0 - np.sum(eigvals[:k]) / total_var
                         for k in range(1, K_MAX + 1)])

# Run greedy
result = greedy_entropy(R, K_MAX, benchmark_names=names)

print("Greedy selection order (MMLU):")
for i, (idx, name) in enumerate(zip(result["selected"], result["names"])):
    resid = result["residual_variance"][i]
    print(f"  {i+1:2d}. {name:40s}  residual = {resid:.4f}")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ks = np.arange(1, K_MAX + 1)

ax.semilogy(ks, result["residual_variance"], "o-", color="#4477AA",
            linewidth=2, markersize=5, label="Greedy entropy")
ax.semilogy(ks, eig_residual, "--", color="gray",
            linewidth=1.5, label="Eigenvalue lower bound")

ax.set_xlabel("Number of selected benchmarks $k$")
ax.set_ylabel("Residual variance fraction")
ax.set_xlim(0.5, K_MAX + 0.5)
ax.set_xticks(ks)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT / "greedy_mmlu.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"\nSaved to {out_path}")
plt.close()
