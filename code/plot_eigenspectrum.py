"""Plot the eigenvalue spectrum of the correlation matrices of the datasets.

Usage:
    python plot_eigenspectrum.py                       # paper figure
    python plot_eigenspectrum.py --datasets mmlu mteb merged benchpress \\
        [--xmax 60] [--out PATH] [--recompute-missing]

For each dataset NAME the EM covariance estimate NAME.cov.npz (written by
estimate_covariance.py) is read from covariance_dir() (BENCHSELECT_COV_DIR,
else data/), converted to the correlation matrix R = D^{-1/2} Sigma D^{-1/2},
and the residual variance fraction 1 - sum_{j<=k} lambda_j / tr(R) is
plotted against k on a log scale.  The k reaching 90/95/99% of the variance
are printed (make_numbers.py recomputes them as the EigK* macros).

Datasets are addressed by name: any NAME with a NAME.cov.npz works.  The
legend label "<display name> (<N> tasks)" takes N from the data (the order
of Sigma, checked against its benchmark_names), so it follows the matrices
instead of being hard-coded.  Names without an entry in DISPLAY_NAME or
COLOR are shown raw and get the next unused palette color.
--recompute-missing estimates Sigma of a dataset that has no cov file from
data/NAME.matrix.npz by the estimate_covariance.py recipe (EM, slow for large
matrices) and does not save it.

The spectrum is a full-data descriptive statistic: no CV splits, no selection
and hence no eligibility mask are involved.

The x axis ends at X_MAX = 42 for the paper figure and, for any other
--datasets, at the largest k plotted if that is beyond 42 (OLL v1 tasks
has 62 columns, HELM MMLU 57); --xmax overrides both.  With more than six
datasets the legend has two columns.

Output: figures/eigenspectrum.pdf for the default datasets (mmlu, mteb,
merged: the paper figure), figures/eigenspectrum_<name>_<name>....pdf for any
other --datasets, so a custom selection never overwrites the paper figure;
--out overrides both.
"""

import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import covariance_dir, data_dir, figures_dir

DATASETS = ["mmlu", "mteb", "merged"]
LEVELS = [0.90, 0.95, 0.99]          # explained-variance levels: printed k
REF_LINES = [(0.10, "90%"), (0.05, "95%"), (0.01, "99%")]   # residual lines
X_MAX = 42                          # x-axis limit of the paper figure
RESID_MIN = 1e-3                     # y-axis floor; curves stop below it

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "mteb": "MTEB",
    "merged": "Merged",
    "benchpress": "BenchPress",
    # data release (data/build_*.py)
    "olv2sub": "OLL v2 sub-tasks",
    "olv1items": "OLL v1 tasks",
    "epoch": "Epoch AI",
    "helm_lite": "HELM Lite",
    "helm_mmlu": "HELM MMLU",
    "helm_classic": "HELM Classic",
    "mteb2026": "MTEB 2026",
}

COLOR = {
    "mmlu": "#4477AA",
    "mteb": "#EE7733",
    "merged": "#228833",
}

# Paul Tol's bright/vibrant colors; datasets without a COLOR entry take the
# first one not used by another curve of the figure.
PALETTE = ["#4477AA", "#EE7733", "#228833", "#CC3311", "#AA3377",
           "#66CCEE", "#CCBB44", "#009988", "#BBBBBB", "#000000"]


def display_name(ds_name):
    """Display name of a dataset (the raw name if none is registered)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def assign_colors(datasets):
    """{dataset: color}: COLOR entries, then unused PALETTE colors."""
    colors = {ds: COLOR[ds] for ds in datasets if ds in COLOR}
    free = [c for c in PALETTE if c not in colors.values()]
    for ds in datasets:
        if ds not in colors:
            colors[ds] = free.pop(0) if free else None   # None: mpl cycle
    return colors


def em_sigma(name):
    """Sigma of data/NAME.matrix.npz by the estimate_covariance.py recipe
    (column standardization, EM with its regularization settings).  Keep in
    sync with estimate_covariance.main(); nothing is written."""
    from em_cov import em_cov
    from estimate_covariance import load_matrix, standardize
    mat = load_matrix(name)
    B, O = mat["B"], mat["O"]
    M, N = B.shape
    B_std, _, _ = standardize(B, O)
    obs_frac = O.sum() / B.size
    regularized = M < N or obs_frac < 0.5
    result = em_cov(B_std, O, verbose=False,
                    max_iter=1000 if regularized else 500,
                    tol=5e-4 if regularized else 1e-6,
                    shrinkage="auto" if M < N else 0.0,
                    eps_psd=1e-3 if regularized else 1e-6)
    return result["Sigma"], mat["benchmark_names"]


def load_sigma(name, cov_dir, recompute_missing=False):
    """(Sigma, benchmark_names or None, source) of dataset `name`."""
    path = cov_dir / f"{name}.cov.npz"
    if path.is_file():
        with np.load(path, allow_pickle=True) as d:
            names = d["benchmark_names"] if "benchmark_names" in d.files \
                else None
            return d["Sigma"], names, str(path)
    if not recompute_missing:
        raise FileNotFoundError(
            f"{path} not found: run estimate_covariance.py for {name!r} "
            f"(or pass --recompute-missing to estimate it here by EM)")
    matrix = data_dir() / f"{name}.matrix.npz"
    if not matrix.is_file():
        raise FileNotFoundError(f"{path} and {matrix} not found")
    print(f"{name}: no {path.name}; estimating Sigma from {matrix} by EM",
          flush=True)
    Sigma, names = em_sigma(name)
    return Sigma, names, f"{matrix} (EM, not saved)"


def warn_if_stale(name, names, src):
    """Warn when the cov file's benchmarks differ from data/NAME.matrix.npz
    (a cov file left over from an earlier build of the matrix)."""
    matrix = data_dir() / f"{name}.matrix.npz"
    if names is None or not matrix.is_file():
        return
    with np.load(matrix, allow_pickle=True) as d:
        current = d["benchmark_names"]
    if not np.array_equal(np.asarray(names, dtype=str),
                          np.asarray(current, dtype=str)):
        print(f"WARNING {name}: the {len(names)} benchmarks of {src} differ "
              f"from the {len(current)} of {matrix}; rerun "
              f"estimate_covariance.py")


def correlation_spectrum(Sigma):
    """Cumulative explained-variance fraction of corr(Sigma), descending
    eigenvalues (as make_numbers.py's EigK* macros)."""
    # Convert covariance to correlation matrix: R = D^{-1/2} Sigma D^{-1/2}
    diag_std = np.sqrt(np.diag(Sigma))
    diag_std = np.maximum(diag_std, 1e-12)  # avoid division by zero
    D_inv = 1.0 / diag_std
    R = Sigma * np.outer(D_inv, D_inv)

    eigvals = np.linalg.eigvalsh(R)[::-1]
    return np.cumsum(eigvals) / eigvals.sum()


def default_out(datasets):
    if list(datasets) == DATASETS:
        return figures_dir() / "eigenspectrum.pdf"
    return figures_dir() / f"eigenspectrum_{'_'.join(datasets)}.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names, each with a NAME.cov.npz "
                             f"(default: {' '.join(DATASETS)})")
    parser.add_argument("--xmax", type=float, default=None,
                        help=f"x-axis limit (default {X_MAX} for the paper "
                             f"figure, else max({X_MAX}, the largest k "
                             f"plotted))")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="output PDF (default: figures/eigenspectrum.pdf "
                             "for the default datasets, else "
                             "figures/eigenspectrum_<names>.pdf)")
    parser.add_argument("--recompute-missing", action="store_true",
                        help="estimate Sigma by EM from data/NAME.matrix.npz "
                             "when NAME.cov.npz is missing")
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets lists a name twice")

    cov_dir = covariance_dir()
    out_path = args.out if args.out is not None else default_out(args.datasets)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = assign_colors(args.datasets)

    spectra = []
    for name in args.datasets:
        try:
            Sigma, names, src = load_sigma(name, cov_dir,
                                           args.recompute_missing)
        except FileNotFoundError as e:
            sys.exit(f"plot_eigenspectrum.py: {e}")
        N = Sigma.shape[0]
        if Sigma.shape != (N, N):
            raise ValueError(f"{src}: Sigma has shape {Sigma.shape}")
        if names is not None and len(names) != N:
            raise ValueError(f"{src}: {len(names)} benchmark_names but "
                             f"Sigma is {N}x{N}")
        warn_if_stale(name, names, src)

        cum_var = correlation_spectrum(Sigma)
        residual = 1.0 - cum_var

        # Truncate data where residual drops below visible range
        mask = residual >= RESID_MIN
        if mask.any():
            last = np.max(np.where(mask)[0]) + 1  # include last point above threshold
        else:
            last = N
        spectra.append((name, N, cum_var, residual, last))

    xmax = args.xmax
    if xmax is None:
        xmax = (X_MAX if list(args.datasets) == DATASETS
                else max([X_MAX] + [last for *_, last in spectra]))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for name, N, cum_var, residual, last in spectra:
        k_all = np.arange(1, N + 1)
        ax.semilogy(k_all[:last], residual[:last],
                    label=f"{display_name(name)} ({N} tasks)",
                    color=colors[name], linewidth=2)

        # Print summary
        for t in LEVELS:
            k = int(np.searchsorted(cum_var, t)) + 1
            print(f"{name:8s}: {t:.0%} variance at k={k}")
            if k > xmax:
                print(f"{name:8s}: k={k} is beyond --xmax {xmax:g}")

    ax.set_xlabel("Number of components $k$")
    ax.set_ylabel("Residual variance fraction")
    ax.set_xlim(1, xmax)
    ax.set_ylim(RESID_MIN, 1.0)
    for level, pct in REF_LINES:
        ax.axhline(level, color="gray", linestyle="--", linewidth=0.8,
                   alpha=0.5)
        ax.text(xmax + 0.5, level, pct, fontsize=8, va="center",
                color="gray")
    many = len(spectra) > 6                  # e.g. the data-release datasets
    ax.legend(loc="upper right", fontsize=7 if many else 9,
              ncol=2 if many else 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
