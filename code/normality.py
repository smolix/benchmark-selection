"""Normality diagnostics for multivariate score matrices.

Implements the tests described in Section 6.3 of the paper:

  - Mardia's multivariate skewness and kurtosis (Mardia, 1970)
  - Shapiro-Wilk univariate normality per column (Shapiro & Wilk, 1965)
  - Multiple-testing correction (Bonferroni and Benjamini-Hochberg)

References
----------
Mardia, K. V. (1970).  Measures of multivariate skewness and kurtosis with
    applications.  Biometrika 57(3), 519--530.
Shapiro, S. S. & Wilk, M. B. (1965).  An analysis of variance test for
    normality (complete samples).  Biometrika 52(3/4), 591--611.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------------------
# Mardia's multivariate skewness and kurtosis
# ---------------------------------------------------------------------------

@dataclass
class MardiaResult:
    """Results of Mardia's multivariate normality test."""
    beta_1: float       # multivariate skewness  β̂_{1,N}
    beta_2: float       # multivariate kurtosis  β̂_{2,N}
    skew_stat: float    # test statistic  M β̂_{1,N} / 6
    skew_df: int        # chi-squared degrees of freedom
    skew_pvalue: float  # p-value for skewness
    kurt_stat: float    # standardised kurtosis z-score
    kurt_pvalue: float  # p-value for kurtosis (two-sided)


def mardia_test(
    X: NDArray[np.floating],
    Sigma_inv: NDArray[np.floating] | None = None,
) -> MardiaResult:
    r"""Mardia's multivariate skewness and kurtosis test.

    For M observations x_1, …, x_M in R^N, define the squared Mahalanobis
    distances d_{ij} = (x_i - x̄)^T Σ̂^{-1} (x_j - x̄).  Then:

        β̂_{1,N} = (1/M²) Σ_{i,j} d_{ij}³       (multivariate skewness)
        β̂_{2,N} = (1/M)  Σ_i     d_{ii}²         (multivariate kurtosis)

    Under normality:
        M β̂_{1,N} / 6  ~  χ²(N(N+1)(N+2)/6)
        β̂_{2,N}        ~  N(N(N+2), 8N(N+2)/M)

    Parameters
    ----------
    X : (M, N) data matrix (rows = observations).
    Sigma_inv : (N, N) inverse covariance.  If None, computed from X.

    Returns
    -------
    MardiaResult with test statistics and p-values.
    """
    X = np.asarray(X, dtype=np.float64)
    M, N = X.shape

    X_centered = X - X.mean(axis=0)

    if Sigma_inv is None:
        Sigma_hat = (X_centered.T @ X_centered) / M
        L, low = cho_factor(Sigma_hat)
        # D = X_centered @ Sigma_hat^{-1} @ X_centered^T  → (M, M)
        Z = cho_solve((L, low), X_centered.T)  # (N, M)
        D = X_centered @ Z  # (M, M) Mahalanobis distance matrix d_{ij}
    else:
        D = X_centered @ Sigma_inv @ X_centered.T

    # --- Skewness ---
    # β̂_{1,N} = (1/M²) Σ_{ij} d_{ij}³
    D3 = D ** 3
    beta_1 = D3.sum() / (M * M)

    # Test: M β̂_{1,N}/6 ~ χ²(df)  where df = N(N+1)(N+2)/6
    skew_stat = M * beta_1 / 6.0
    skew_df = N * (N + 1) * (N + 2) // 6
    skew_pvalue = 1.0 - stats.chi2.cdf(skew_stat, df=skew_df)

    # --- Kurtosis ---
    # β̂_{2,N} = (1/M) Σ_i d_{ii}²
    diag_D = np.diag(D)
    beta_2 = (diag_D ** 2).sum() / M

    # Under normality: E[β̂_{2,N}] = N(N+2), Var = 8N(N+2)/M
    kurt_mean = N * (N + 2)
    kurt_var = 8.0 * N * (N + 2) / M
    kurt_stat = (beta_2 - kurt_mean) / np.sqrt(kurt_var)
    kurt_pvalue = 2.0 * (1.0 - stats.norm.cdf(abs(kurt_stat)))

    return MardiaResult(
        beta_1=beta_1,
        beta_2=beta_2,
        skew_stat=skew_stat,
        skew_df=skew_df,
        skew_pvalue=skew_pvalue,
        kurt_stat=kurt_stat,
        kurt_pvalue=kurt_pvalue,
    )


# ---------------------------------------------------------------------------
# Per-column Shapiro-Wilk with multiple-testing correction
# ---------------------------------------------------------------------------

@dataclass
class ShapiroWilkResult:
    """Results of per-column Shapiro-Wilk normality tests."""
    W: NDArray[np.floating]           # (N,) W statistics per column
    pvalues_raw: NDArray[np.floating] # (N,) unadjusted p-values
    pvalues_bonf: NDArray[np.floating]  # (N,) Bonferroni-adjusted
    pvalues_bh: NDArray[np.floating]    # (N,) Benjamini-Hochberg-adjusted
    reject_bonf: NDArray[np.bool_]    # (N,) rejected at α after Bonferroni
    reject_bh: NDArray[np.bool_]      # (N,) rejected at α after BH
    alpha: float                      # significance level used


def _benjamini_hochberg(pvalues: NDArray[np.floating]) -> NDArray[np.floating]:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)

    adjusted = pvalues * n / ranked
    # Enforce monotonicity: working from largest rank down
    adjusted = np.minimum(adjusted, 1.0)
    rev_order = np.argsort(-ranked)
    cummin = np.minimum.accumulate(adjusted[rev_order])
    adjusted[rev_order] = cummin
    return adjusted


def shapiro_wilk_columns(
    X: NDArray[np.floating],
    alpha: float = 0.05,
) -> ShapiroWilkResult:
    r"""Shapiro-Wilk test applied to each column of X.

    For each column j (benchmark), computes:

        W_j = (Σ a_i x_{(i)})² / Σ(x_i - x̄)²

    with Bonferroni and Benjamini-Hochberg corrections for N tests.

    Parameters
    ----------
    X : (M, N) data matrix.
    alpha : significance level for rejection decisions.

    Returns
    -------
    ShapiroWilkResult with per-column statistics and adjusted p-values.
    """
    X = np.asarray(X, dtype=np.float64)
    M, N = X.shape

    W_stats = np.empty(N)
    pvals = np.empty(N)

    for j in range(N):
        col = X[:, j]
        # scipy requires 3 ≤ M ≤ 5000 for Shapiro-Wilk
        if M < 3:
            W_stats[j] = np.nan
            pvals[j] = np.nan
        else:
            w, p = stats.shapiro(col)
            W_stats[j] = w
            pvals[j] = p

    # Multiple testing corrections
    pvals_bonf = np.minimum(pvals * N, 1.0)
    pvals_bh = _benjamini_hochberg(pvals)

    return ShapiroWilkResult(
        W=W_stats,
        pvalues_raw=pvals,
        pvalues_bonf=pvals_bonf,
        pvalues_bh=pvals_bh,
        reject_bonf=pvals_bonf < alpha,
        reject_bh=pvals_bh < alpha,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Combined diagnostics report
# ---------------------------------------------------------------------------

@dataclass
class NormalityReport:
    """Complete normality diagnostic report."""
    mardia: MardiaResult
    shapiro_wilk: ShapiroWilkResult
    n_obs: int
    n_dim: int


def normality_diagnostics(
    X: NDArray[np.floating],
    alpha: float = 0.05,
    residual: bool = False,
    mu: NDArray[np.floating] | None = None,
    Sigma: NDArray[np.floating] | None = None,
) -> NormalityReport:
    r"""Run full normality diagnostics on a score matrix.

    Parameters
    ----------
    X : (M, N) score matrix.
    alpha : significance level.
    residual : if True, apply Shapiro-Wilk to Mahalanobis residuals
               rather than raw columns (more powerful for detecting
               departures from the joint Gaussian model).
    mu : (N,) mean vector.  If None, estimated from X.
    Sigma : (N, N) covariance.  If None, estimated from X.

    Returns
    -------
    NormalityReport combining Mardia and Shapiro-Wilk results.
    """
    X = np.asarray(X, dtype=np.float64)
    M, N = X.shape

    if mu is None:
        mu = X.mean(axis=0)
    if Sigma is None:
        X_c = X - mu
        Sigma = (X_c.T @ X_c) / M

    Sigma_inv = np.linalg.inv(Sigma)

    # Mardia test on raw data
    mardia = mardia_test(X, Sigma_inv=Sigma_inv)

    # Shapiro-Wilk on columns (or residuals)
    if residual:
        # Transform to standardised residuals via Sigma^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-12)
        Sigma_inv_half = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
        Z = (X - mu) @ Sigma_inv_half
        sw = shapiro_wilk_columns(Z, alpha=alpha)
    else:
        sw = shapiro_wilk_columns(X, alpha=alpha)

    return NormalityReport(
        mardia=mardia,
        shapiro_wilk=sw,
        n_obs=M,
        n_dim=N,
    )


def print_report(report: NormalityReport, col_names: list[str] | None = None):
    """Pretty-print a normality diagnostics report."""
    M, N = report.n_obs, report.n_dim
    m = report.mardia
    sw = report.shapiro_wilk

    print(f"Normality Diagnostics  (M={M} observations, N={N} dimensions)")
    print("=" * 64)

    print(f"\nMardia's Multivariate Skewness")
    print(f"  β̂₁ = {m.beta_1:.4f}")
    print(f"  Test statistic (Mβ̂₁/6) = {m.skew_stat:.4f}")
    print(f"  χ²({m.skew_df}) p-value = {m.skew_pvalue:.4e}")
    print(f"  {'REJECT' if m.skew_pvalue < sw.alpha else 'PASS'} "
          f"at α = {sw.alpha}")

    print(f"\nMardia's Multivariate Kurtosis")
    print(f"  β̂₂ = {m.beta_2:.4f}  (expected under H₀: {N*(N+2):.1f})")
    print(f"  z-score = {m.kurt_stat:.4f}")
    print(f"  p-value = {m.kurt_pvalue:.4e}")
    print(f"  {'REJECT' if m.kurt_pvalue < sw.alpha else 'PASS'} "
          f"at α = {sw.alpha}")

    print(f"\nShapiro-Wilk Per-Column Tests (α = {sw.alpha})")
    print(f"  {'Column':<20s} {'W':>8s} {'p-raw':>10s} "
          f"{'p-Bonf':>10s} {'p-BH':>10s} {'Bonf':>6s} {'BH':>6s}")
    print("  " + "-" * 72)
    for j in range(N):
        name = col_names[j] if col_names else f"col_{j}"
        print(f"  {name:<20s} {sw.W[j]:8.4f} {sw.pvalues_raw[j]:10.4e} "
              f"{sw.pvalues_bonf[j]:10.4e} {sw.pvalues_bh[j]:10.4e} "
              f"{'FAIL' if sw.reject_bonf[j] else 'ok':>6s} "
              f"{'FAIL' if sw.reject_bh[j] else 'ok':>6s}")

    n_bonf = sw.reject_bonf.sum()
    n_bh = sw.reject_bh.sum()
    print(f"\n  Rejections: {n_bonf}/{N} (Bonferroni), {n_bh}/{N} (BH)")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    N = 6
    M = 200

    # --- Test 1: Gaussian data (should pass) ---
    print("Test 1: Multivariate Gaussian data")
    print("-" * 64)
    A = np.random.randn(N, N)
    Sigma_true = A @ A.T + 0.5 * np.eye(N)
    mu_true = np.random.randn(N)
    X_gauss = np.random.multivariate_normal(mu_true, Sigma_true, size=M)

    report = normality_diagnostics(X_gauss, alpha=0.05)
    print_report(report, col_names=[f"bench_{j}" for j in range(N)])

    # --- Test 2: Heavy-tailed (Student-t) data (should fail kurtosis) ---
    print("\n\nTest 2: Student-t (df=3) data — expect kurtosis rejection")
    print("-" * 64)
    X_t = np.random.standard_t(df=3, size=(M, N))
    # Add correlation structure
    L = np.linalg.cholesky(Sigma_true)
    X_t = X_t @ L.T + mu_true

    report_t = normality_diagnostics(X_t, alpha=0.05)
    print_report(report_t, col_names=[f"bench_{j}" for j in range(N)])

    # --- Test 3: Skewed data (should fail skewness) ---
    print("\n\nTest 3: Log-normal marginals — expect skewness rejection")
    print("-" * 64)
    X_ln = np.exp(X_gauss * 0.5)  # log-normal transform
    report_ln = normality_diagnostics(X_ln, alpha=0.05)
    print_report(report_ln, col_names=[f"bench_{j}" for j in range(N)])
