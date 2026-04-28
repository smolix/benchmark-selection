"""Tests for normality diagnostics module."""

import numpy as np
import pytest
from scipy import stats

from normality import (
    MardiaResult,
    ShapiroWilkResult,
    _benjamini_hochberg,
    mardia_test,
    normality_diagnostics,
    shapiro_wilk_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gaussian(M, N, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    Sigma = A @ A.T + 0.3 * np.eye(N)
    mu = rng.standard_normal(N)
    X = rng.multivariate_normal(mu, Sigma, size=M)
    return X, mu, Sigma


def _make_heavy_tailed(M, N, df=3, seed=0):
    """Student-t with correlation structure."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    Sigma = A @ A.T + 0.3 * np.eye(N)
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_t(df=df, size=(M, N))
    return Z @ L.T, Sigma


def _make_skewed(M, N, seed=0):
    """Log-normal marginals."""
    X, _, _ = _make_gaussian(M, N, seed=seed)
    return np.exp(0.5 * X)


# ---------------------------------------------------------------------------
# Mardia test: return types and basic properties
# ---------------------------------------------------------------------------

class TestMardiaBasic:
    def test_returns_dataclass(self):
        X, _, _ = _make_gaussian(100, 4)
        result = mardia_test(X)
        assert isinstance(result, MardiaResult)

    def test_skewness_nonnegative(self):
        X, _, _ = _make_gaussian(100, 4)
        result = mardia_test(X)
        assert result.beta_1 >= 0.0

    def test_pvalues_in_01(self):
        X, _, _ = _make_gaussian(100, 4)
        result = mardia_test(X)
        assert 0.0 <= result.skew_pvalue <= 1.0
        assert 0.0 <= result.kurt_pvalue <= 1.0

    def test_skew_df_formula(self):
        """df = N(N+1)(N+2)/6."""
        N = 5
        X, _, _ = _make_gaussian(100, N)
        result = mardia_test(X)
        assert result.skew_df == N * (N + 1) * (N + 2) // 6

    def test_kurtosis_expected_value(self):
        """Under normality, E[β₂] = N(N+2)."""
        N = 4
        X, _, _ = _make_gaussian(5000, N, seed=7)
        result = mardia_test(X)
        expected = N * (N + 2)
        # With 5000 samples this should be close
        assert abs(result.beta_2 - expected) < 3.0

    def test_external_sigma_inv(self):
        """Passing Sigma_inv should give same result as auto-computed."""
        X, _, _ = _make_gaussian(100, 4)
        r1 = mardia_test(X)
        X_c = X - X.mean(axis=0)
        Sigma_hat = (X_c.T @ X_c) / len(X)
        Sigma_inv = np.linalg.inv(Sigma_hat)
        r2 = mardia_test(X, Sigma_inv=Sigma_inv)
        np.testing.assert_allclose(r1.beta_1, r2.beta_1, rtol=1e-10)
        np.testing.assert_allclose(r1.beta_2, r2.beta_2, rtol=1e-10)


# ---------------------------------------------------------------------------
# Mardia test: statistical power
# ---------------------------------------------------------------------------

class TestMardiaPower:
    def test_gaussian_passes(self):
        """Gaussian data should not be rejected (at α=0.05)."""
        # Run multiple seeds and check that rejection rate is reasonable
        n_reject_skew = 0
        n_reject_kurt = 0
        n_trials = 20
        for seed in range(n_trials):
            X, _, _ = _make_gaussian(200, 4, seed=seed)
            r = mardia_test(X)
            if r.skew_pvalue < 0.05:
                n_reject_skew += 1
            if r.kurt_pvalue < 0.05:
                n_reject_kurt += 1
        # Under H0, expect ~5% rejection.  Allow up to 25% for randomness.
        assert n_reject_skew <= 0.25 * n_trials
        assert n_reject_kurt <= 0.25 * n_trials

    def test_heavy_tail_rejects_kurtosis(self):
        """Student-t(3) should be detected by kurtosis test."""
        X, _ = _make_heavy_tailed(300, 4, df=3, seed=42)
        r = mardia_test(X)
        assert r.kurt_pvalue < 0.01

    def test_skewed_rejects_skewness(self):
        """Log-normal data should be detected by skewness test."""
        X = _make_skewed(300, 4, seed=42)
        r = mardia_test(X)
        assert r.skew_pvalue < 0.01


# ---------------------------------------------------------------------------
# Mardia test: known small example (sanity check computation)
# ---------------------------------------------------------------------------

class TestMardiaComputation:
    def test_identity_case(self):
        """With identity covariance and zero mean, d_{ij} = (x_i-x̄)·(x_j-x̄)."""
        rng = np.random.default_rng(99)
        M, N = 50, 3
        X = rng.standard_normal((M, N))
        r = mardia_test(X)

        # Manually compute β₁ and β₂
        X_c = X - X.mean(axis=0)
        S = (X_c.T @ X_c) / M
        S_inv = np.linalg.inv(S)
        D = X_c @ S_inv @ X_c.T
        beta_1_manual = (D ** 3).sum() / M ** 2
        beta_2_manual = (np.diag(D) ** 2).sum() / M

        np.testing.assert_allclose(r.beta_1, beta_1_manual, rtol=1e-10)
        np.testing.assert_allclose(r.beta_2, beta_2_manual, rtol=1e-10)


# ---------------------------------------------------------------------------
# Shapiro-Wilk: return types and properties
# ---------------------------------------------------------------------------

class TestShapiroWilkBasic:
    def test_returns_dataclass(self):
        X, _, _ = _make_gaussian(100, 5)
        result = shapiro_wilk_columns(X)
        assert isinstance(result, ShapiroWilkResult)

    def test_shapes(self):
        M, N = 100, 7
        X, _, _ = _make_gaussian(M, N)
        r = shapiro_wilk_columns(X)
        assert r.W.shape == (N,)
        assert r.pvalues_raw.shape == (N,)
        assert r.pvalues_bonf.shape == (N,)
        assert r.pvalues_bh.shape == (N,)
        assert r.reject_bonf.shape == (N,)
        assert r.reject_bh.shape == (N,)

    def test_W_in_01(self):
        X, _, _ = _make_gaussian(100, 5)
        r = shapiro_wilk_columns(X)
        assert np.all(r.W > 0)
        assert np.all(r.W <= 1)

    def test_bonferroni_ge_raw(self):
        X, _, _ = _make_gaussian(100, 5)
        r = shapiro_wilk_columns(X)
        assert np.all(r.pvalues_bonf >= r.pvalues_raw - 1e-15)

    def test_bonferroni_le_1(self):
        X, _, _ = _make_gaussian(100, 5)
        r = shapiro_wilk_columns(X)
        assert np.all(r.pvalues_bonf <= 1.0 + 1e-15)

    def test_bh_ge_raw(self):
        X, _, _ = _make_gaussian(100, 5)
        r = shapiro_wilk_columns(X)
        assert np.all(r.pvalues_bh >= r.pvalues_raw - 1e-15)

    def test_agrees_with_scipy(self):
        """Our W should match scipy.stats.shapiro exactly."""
        X, _, _ = _make_gaussian(100, 3)
        r = shapiro_wilk_columns(X)
        for j in range(3):
            w_ref, p_ref = stats.shapiro(X[:, j])
            np.testing.assert_allclose(r.W[j], w_ref, rtol=1e-10)
            np.testing.assert_allclose(r.pvalues_raw[j], p_ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# Shapiro-Wilk: statistical power
# ---------------------------------------------------------------------------

class TestShapiroWilkPower:
    def test_gaussian_passes(self):
        X, _, _ = _make_gaussian(200, 6, seed=42)
        r = shapiro_wilk_columns(X, alpha=0.05)
        # Expect most columns to pass
        assert r.reject_bonf.sum() <= 2
        assert r.reject_bh.sum() <= 2

    def test_heavy_tail_rejects(self):
        X, _ = _make_heavy_tailed(200, 6, df=3, seed=42)
        r = shapiro_wilk_columns(X, alpha=0.05)
        # Most columns should fail
        assert r.reject_bh.sum() >= 3

    def test_skewed_rejects(self):
        X = _make_skewed(200, 6, seed=42)
        r = shapiro_wilk_columns(X, alpha=0.05)
        assert r.reject_bh.sum() >= 4


# ---------------------------------------------------------------------------
# Benjamini-Hochberg correction
# ---------------------------------------------------------------------------

class TestBH:
    def test_single_pvalue(self):
        adj = _benjamini_hochberg(np.array([0.03]))
        np.testing.assert_allclose(adj, [0.03])

    def test_all_significant(self):
        pvals = np.array([0.001, 0.002, 0.003])
        adj = _benjamini_hochberg(pvals)
        assert np.all(adj <= 0.01)

    def test_monotonicity(self):
        """Adjusted p-values for sorted input should be non-decreasing
        when re-sorted by original rank."""
        rng = np.random.default_rng(0)
        pvals = rng.uniform(0, 1, size=20)
        adj = _benjamini_hochberg(pvals)
        # After sorting by original p-value, adjusted should be non-decreasing
        order = np.argsort(pvals)
        adj_sorted = adj[order]
        assert np.all(np.diff(adj_sorted) >= -1e-15)

    def test_bounded_by_1(self):
        pvals = np.array([0.8, 0.9, 0.95])
        adj = _benjamini_hochberg(pvals)
        assert np.all(adj <= 1.0 + 1e-15)

    def test_ge_raw(self):
        rng = np.random.default_rng(1)
        pvals = rng.uniform(0, 1, size=50)
        adj = _benjamini_hochberg(pvals)
        assert np.all(adj >= pvals - 1e-15)


# ---------------------------------------------------------------------------
# Combined normality_diagnostics
# ---------------------------------------------------------------------------

class TestNormalityDiagnostics:
    def test_returns_report(self):
        X, _, _ = _make_gaussian(100, 4)
        report = normality_diagnostics(X)
        assert hasattr(report, "mardia")
        assert hasattr(report, "shapiro_wilk")
        assert report.n_obs == 100
        assert report.n_dim == 4

    def test_custom_mu_sigma(self):
        """Passing external mu/Sigma should not crash."""
        X, mu, Sigma = _make_gaussian(100, 4)
        report = normality_diagnostics(X, mu=mu, Sigma=Sigma)
        assert report.mardia.skew_pvalue >= 0

    def test_residual_mode(self):
        """Residual mode should also work."""
        X, _, _ = _make_gaussian(100, 4)
        report = normality_diagnostics(X, residual=True)
        # Whitened Gaussian should still pass
        assert report.shapiro_wilk.reject_bh.sum() <= 2

    def test_custom_alpha(self):
        X, _, _ = _make_gaussian(100, 4)
        r1 = normality_diagnostics(X, alpha=0.01)
        r2 = normality_diagnostics(X, alpha=0.50)
        assert r1.shapiro_wilk.alpha == 0.01
        assert r2.shapiro_wilk.alpha == 0.50
        # More permissive alpha → at least as many rejections
        assert r2.shapiro_wilk.reject_bh.sum() >= r1.shapiro_wilk.reject_bh.sum()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_univariate(self):
        """N=1 should work."""
        X = np.random.default_rng(0).standard_normal((100, 1))
        r = mardia_test(X)
        assert r.skew_df == 1  # 1*2*3/6 = 1
        sw = shapiro_wilk_columns(X)
        assert sw.W.shape == (1,)

    def test_two_obs(self):
        """M=2 is degenerate for covariance; should not crash."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        # Mardia with a provided non-singular Sigma_inv
        S_inv = np.eye(2)
        r = mardia_test(X, Sigma_inv=S_inv)
        assert np.isfinite(r.beta_1)

    def test_large_N_small_M(self):
        """More dimensions than observations — still computes."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 20))
        # Sigma is singular, but we pass our own regularised inverse
        S = (X - X.mean(0)).T @ (X - X.mean(0)) / 10 + 0.1 * np.eye(20)
        r = mardia_test(X, Sigma_inv=np.linalg.inv(S))
        assert np.isfinite(r.beta_1)
        assert np.isfinite(r.beta_2)

    def test_constant_column(self):
        """A constant column has W=1 or NaN; should not crash."""
        X = np.random.default_rng(0).standard_normal((50, 3))
        X[:, 1] = 5.0  # constant
        sw = shapiro_wilk_columns(X)
        assert sw.W.shape == (3,)
        # scipy.stats.shapiro returns W≈1 for constant input


# ---------------------------------------------------------------------------
# Calibration test: Type-I error under H0
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_mardia_skewness_calibration(self):
        """Under H0, the fraction of rejections at α should be ≈ α."""
        alpha = 0.05
        n_trials = 200
        rejections = 0
        for seed in range(n_trials):
            X, _, _ = _make_gaussian(100, 3, seed=seed + 1000)
            r = mardia_test(X)
            if r.skew_pvalue < alpha:
                rejections += 1
        rate = rejections / n_trials
        # Should be near 0.05; allow [0.01, 0.12] range
        assert 0.01 <= rate <= 0.12, f"Rejection rate {rate:.3f} outside [0.01, 0.12]"

    def test_mardia_kurtosis_calibration(self):
        alpha = 0.05
        n_trials = 200
        rejections = 0
        for seed in range(n_trials):
            X, _, _ = _make_gaussian(100, 3, seed=seed + 2000)
            r = mardia_test(X)
            if r.kurt_pvalue < alpha:
                rejections += 1
        rate = rejections / n_trials
        assert 0.01 <= rate <= 0.12, f"Rejection rate {rate:.3f} outside [0.01, 0.12]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
