# `code/` — modules and experiment scripts

All scripts resolve their paths through `path_config.py`, so they can be
invoked from any working directory:

```bash
python /path/to/release/code/eval_greedy_all.py
```

Outputs are written to `release/figures/` (and `release/figures_logit/` for
the logit-space variants); intermediate JSON/NPZ logs go to `release/logs/`.
Use `BENCHSELECT_DATA_DIR`, `BENCHSELECT_EXPERIMENT_ROOT`, or the more specific
`BENCHSELECT_*_DIR` variables in `path_config.py` to redirect a run.

## Core modules

| File | Description |
|------|-------------|
| `greedy_select.py` | Algorithms 1 (`greedy_entropy`) and 2 (`greedy_mi`) plus a `random_select` baseline. Pivoted-Cholesky entropy in `O(k²N)`; numerically stable MI via fresh complement Cholesky each step. |
| `em_cov.py`        | EM for Gaussian covariance under MAR missingness (Section 4.3 / Appendix G). Supports Ledoit–Wolf shrinkage, PSD projection, and rank-deficient initialization for `M < N`. |
| `cv_split.py`      | Balanced k-fold splits with optional training-set subsampling — used to vary the holdout fraction in the CV protocol. |
| `normality.py`     | Mardia's multivariate skewness/kurtosis test, per-column Shapiro–Wilk with Bonferroni / Benjamini–Hochberg correction (Appendix H). |
| `path_config.py`   | Central path configuration for data, figures, covariance caches, and logs. |

## Experiment drivers

| Script | Paper output | Notes |
|--------|--------------|-------|
| `estimate_covariance.py`  | caches `data/{name}.cov.npz` | Standardize columns, run EM, save (`mu`, `Sigma`, `B_imp`). Recommended one-time precomputation; downstream scripts auto-cache as needed. |
| `eval_greedy_all.py`      | Figures 2–4, `logs/greedy_cv_*.npz`, `logs/greedy_cv_selections.json` | 10-fold CV across all four holdout fractions on MMLU/MTEB/Merged. Pairwise covariance for fully observed (MMLU); EM for the rest. Dominant runtime on Merged. |
| `eval_entropy_vs_mi.py`   | Figures 5–7 | Same protocol at 10% holdout, compares `greedy_entropy` vs `greedy_mi` vs random. |
| `eval_benchpress.py`      | Figures 13–14 (Appendix I) | Greedy CV + entropy-vs-MI on the BenchPress 83 × 49 matrix. |
| `eval_tabimpute.py`       | Figure 15, Table 4 (Appendix J) | Same selection protocol, swaps Gaussian conditional mean for TabImpute V2 imputation. Requires the `tabimpute` package and a GPU. |
| `eval_logit.py`           | Figures 16–18, Table 5 (Appendix K) | Repeats greedy CV and entropy-vs-MI in logit-transformed score space. Outputs to `figures_logit/` and `logs_logit/`. |
| `run_normality.py`        | Figure 12, `logs/normality_summary.json` (Appendix H) | Per-column Shapiro–Wilk plus Mardia's test on fully-observed sub-blocks. |

## Plotting helpers

| Script | Paper output | Inputs |
|--------|--------------|--------|
| `plot_eigenspectrum.py`     | Figure 1 | Matrices in `data/`; auto-runs EM if `*.cov.npz` is missing and caches the result. |
| `plot_selection_order.py`   | Figures 9a, 10a, 11a (entropy selection order) | Reads `logs/greedy_cv_selections.json` produced by `eval_greedy_all.py`. |
| `plot_mi_selection_all.py`  | Figures 9b, 10b, 11b (MI selection order on MMLU/MTEB/Merged) | Runs MI selection itself; writes selection logs to `logs/`. |

## Tests

| File | Coverage |
|------|----------|
| `test_em_cov.py`     | E-step / M-step identities, PSD projection, pairwise-complete covariance, observed log-likelihood. |
| `test_normality.py`  | Mardia and Shapiro–Wilk consistency on synthetic Gaussian / non-Gaussian samples; multiple-testing correction. |
| `test_tabimpute_sanity.py` | Basic TabImpute integration check when the optional package is installed. |

Run with `pytest code/` from the release root.

## Selection-algorithm interface

```python
from greedy_select import greedy_entropy, greedy_mi, random_select

# All three return the indices of the selected benchmarks (length k).
S = greedy_entropy(Sigma, k)             # Algorithm 1
S = greedy_mi(Sigma, k)                  # Algorithm 2
S = random_select(Sigma, k, seed=0)      # baseline
```

`Sigma` is the `N × N` correlation (or covariance) matrix of benchmark scores;
no other state is required.

## EM-covariance interface

```python
from em_cov import em_cov

result = em_cov(B_std, O,                 # standardized scores, observation mask
                max_iter=500, tol=1e-6,
                shrinkage="auto",          # "auto" applies Ledoit-Wolf when M < N
                eps_psd=1e-6, verbose=True)
result["mu"]        # (N,)  estimated mean
result["Sigma"]     # (N,N) estimated covariance
result["B_imp"]     # (M,N) imputed standardized matrix
result["loglik"]    # convergence trace
```
