"""Tests for the EM covariance estimation module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from em_cov import (
    _e_step,
    _m_step,
    _mean_from_observed,
    _observed_loglik,
    _pairwise_complete_cov,
    _project_psd,
    em_cov,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_problem(M=80, N=5, missing_frac=0.3, seed=0):
    """Generate a synthetic EM problem and return ground truth."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    Sigma_true = A @ A.T + 0.5 * np.eye(N)
    mu_true = rng.standard_normal(N)
    B_full = rng.multivariate_normal(mu_true, Sigma_true, size=M)

    O = (rng.random((M, N)) > missing_frac).astype(np.float64)
    # guarantee every row has ≥1 obs, every col has ≥2
    for i in range(M):
        if O[i].sum() == 0:
            O[i, rng.integers(N)] = 1.0
    for j in range(N):
        while O[:, j].sum() < 2:
            idx = rng.choice(np.where(O[:, j] == 0)[0])
            O[idx, j] = 1.0

    B_obs = B_full * O
    return B_full, B_obs, O, mu_true, Sigma_true


# ===================================================================
# Initialisation helpers
# ===================================================================

class TestMeanFromObserved:
    def test_full_observation(self):
        """With O=1 everywhere, should equal column means of B."""
        rng = np.random.default_rng(1)
        B = rng.standard_normal((30, 4))
        O = np.ones_like(B)
        mu = _mean_from_observed(B, O)
        np.testing.assert_allclose(mu, B.mean(axis=0), atol=1e-14)

    def test_masked_entries_ignored(self):
        """Masked entries (even if nonzero in B) should not affect the mean."""
        B = np.array([[1.0, 2.0],
                      [3.0, 999.0],  # 999 is unobserved
                      [5.0, 6.0]])
        O = np.array([[1, 1],
                      [1, 0],
                      [1, 1]], dtype=float)
        mu = _mean_from_observed(B * O, O)  # B*O zeros out unobserved
        np.testing.assert_allclose(mu[0], 3.0)        # (1+3+5)/3
        np.testing.assert_allclose(mu[1], 4.0)        # (2+6)/2

    def test_single_observation(self):
        """Column with one observed entry → mean = that entry."""
        B = np.array([[10.0, 0.0],
                      [0.0, 7.0]])
        O = np.array([[1, 0],
                      [0, 1]], dtype=float)
        mu = _mean_from_observed(B, O)
        np.testing.assert_allclose(mu, [10.0, 7.0])

    def test_all_unobserved_column(self):
        """Column with no observations → mean should not be NaN/inf."""
        B = np.zeros((3, 2))
        O = np.array([[1, 0], [1, 0], [1, 0]], dtype=float)
        mu = _mean_from_observed(B, O)
        assert np.all(np.isfinite(mu))


class TestPairwiseCompleteCov:
    def test_full_observation(self):
        """Should match standard sample covariance when fully observed."""
        rng = np.random.default_rng(2)
        M, N = 50, 3
        B = rng.standard_normal((M, N))
        O = np.ones((M, N))
        mu = B.mean(axis=0)
        Sigma_pw = _pairwise_complete_cov(B, O, mu)
        Sigma_ref = np.cov(B, rowvar=False)  # uses ddof=1 by default
        np.testing.assert_allclose(Sigma_pw, Sigma_ref, atol=1e-12)

    def test_symmetric(self):
        B_full, B_obs, O, _, _ = _make_problem()
        mu = _mean_from_observed(B_obs, O)
        Sigma_pw = _pairwise_complete_cov(B_obs, O, mu)
        np.testing.assert_allclose(Sigma_pw, Sigma_pw.T, atol=1e-14)


class TestProjectPSD:
    def test_already_psd(self):
        """PSD matrix should be unchanged (up to eps floor)."""
        S = np.array([[2.0, 0.5], [0.5, 1.0]])
        S_proj = _project_psd(S, eps=1e-10)
        np.testing.assert_allclose(S_proj, S, atol=1e-10)

    def test_negative_eigenvalue_clamped(self):
        """Matrix with a negative eigenvalue should become PSD."""
        S = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
        S_proj = _project_psd(S, eps=1e-6)
        eigvals = np.linalg.eigvalsh(S_proj)
        assert np.all(eigvals >= 1e-6 - 1e-12)

    def test_symmetric_output(self):
        rng = np.random.default_rng(3)
        S = rng.standard_normal((5, 5))
        S = S + S.T  # symmetric but not PSD
        S_proj = _project_psd(S)
        np.testing.assert_allclose(S_proj, S_proj.T, atol=1e-14)


# ===================================================================
# E-step
# ===================================================================

class TestEStep:
    def test_fully_observed_no_change(self):
        """If O=1 everywhere, B_tilde = B and all C_i = 0."""
        rng = np.random.default_rng(4)
        M, N = 10, 3
        mu = np.zeros(N)
        Sigma = np.eye(N)
        B = rng.standard_normal((M, N))
        O = np.ones((M, N))

        B_tilde, C_list = _e_step(B, O, mu, Sigma)
        np.testing.assert_allclose(B_tilde, B, atol=1e-14)
        for C_i in C_list:
            np.testing.assert_allclose(C_i, 0.0, atol=1e-14)

    def test_fully_missing_row(self):
        """Fully missing row → imputed to mean, C_i = Sigma."""
        N = 3
        mu = np.array([1.0, 2.0, 3.0])
        Sigma = np.array([[2, 0.5, 0], [0.5, 1, 0.3], [0, 0.3, 1.5]])
        B = np.zeros((2, N))
        O = np.array([[0, 0, 0],   # fully missing
                      [1, 1, 1]], dtype=float)  # fully observed
        B[1] = np.array([0.5, 1.5, 2.5])

        B_tilde, C_list = _e_step(B, O, mu, Sigma)
        np.testing.assert_allclose(B_tilde[0], mu, atol=1e-14)
        np.testing.assert_allclose(C_list[0], Sigma, atol=1e-14)
        np.testing.assert_allclose(C_list[1], 0.0, atol=1e-14)

    def test_conditional_mean_formula(self):
        """Check E-step imputation matches the conditional Gaussian formula."""
        N = 3
        mu = np.array([0.0, 0.0, 0.0])
        Sigma = np.array([[1.0, 0.8, 0.3],
                          [0.8, 1.0, 0.5],
                          [0.3, 0.5, 1.0]])
        # Model with benchmarks 0,1 observed, benchmark 2 missing
        B = np.array([[1.0, 0.5, 0.0]])  # entry [0,2] is unobserved
        O = np.array([[1, 1, 0]], dtype=float)

        B_tilde, C_list = _e_step(B, O, mu, Sigma)

        # Manual: E[X_2 | X_0=1, X_1=0.5]
        obs = [0, 1]
        mis = [2]
        Sigma_oo = Sigma[np.ix_(obs, obs)]
        Sigma_mo = Sigma[np.ix_(mis, obs)]
        Sigma_mm = Sigma[np.ix_(mis, mis)]
        cond_mean = mu[mis] + Sigma_mo @ np.linalg.solve(Sigma_oo, B[0, obs] - mu[obs])
        cond_cov = Sigma_mm - Sigma_mo @ np.linalg.solve(Sigma_oo, Sigma_mo.T)

        np.testing.assert_allclose(B_tilde[0, 2], cond_mean[0], atol=1e-12)
        np.testing.assert_allclose(C_list[0][2, 2], cond_cov[0, 0], atol=1e-12)
        # Observed entries unchanged
        np.testing.assert_allclose(B_tilde[0, :2], B[0, :2], atol=1e-14)

    def test_C_i_only_nonzero_on_missing(self):
        """C_i[j,k] should be zero unless both j,k are missing for row i."""
        _, B_obs, O, mu_true, Sigma_true = _make_problem(M=20, N=4, seed=5)
        mu = _mean_from_observed(B_obs, O)
        Sigma = _project_psd(_pairwise_complete_cov(B_obs, O, mu))
        _, C_list = _e_step(B_obs, O, mu, Sigma)

        for i in range(len(C_list)):
            obs = np.where(O[i] == 1)[0]
            C_i = C_list[i]
            # rows/cols corresponding to observed entries should be zero
            for j in obs:
                np.testing.assert_allclose(C_i[j, :], 0.0, atol=1e-14)
                np.testing.assert_allclose(C_i[:, j], 0.0, atol=1e-14)

    def test_C_i_positive_semidefinite(self):
        """Each C_i should be PSD (it's a conditional covariance)."""
        _, B_obs, O, _, _ = _make_problem(M=20, N=4, seed=6)
        mu = _mean_from_observed(B_obs, O)
        Sigma = _project_psd(_pairwise_complete_cov(B_obs, O, mu))
        _, C_list = _e_step(B_obs, O, mu, Sigma)
        for C_i in C_list:
            eigvals = np.linalg.eigvalsh(C_i)
            assert np.all(eigvals >= -1e-10), f"Negative eigenvalue in C_i: {eigvals.min()}"


# ===================================================================
# M-step
# ===================================================================

class TestMStep:
    def test_mu_is_mean_of_completed(self):
        rng = np.random.default_rng(7)
        M, N = 20, 3
        B_tilde = rng.standard_normal((M, N))
        C_list = [np.zeros((N, N)) for _ in range(M)]
        mu, _ = _m_step(B_tilde, C_list)
        np.testing.assert_allclose(mu, B_tilde.mean(axis=0), atol=1e-14)

    def test_sigma_without_correction(self):
        """When all C_i=0, Sigma should equal the ML covariance of B_tilde."""
        rng = np.random.default_rng(8)
        M, N = 30, 4
        B_tilde = rng.standard_normal((M, N))
        C_list = [np.zeros((N, N)) for _ in range(M)]
        _, Sigma = _m_step(B_tilde, C_list)
        B_c = B_tilde - B_tilde.mean(axis=0)
        Sigma_ref = (B_c.T @ B_c) / M
        np.testing.assert_allclose(Sigma, Sigma_ref, atol=1e-14)

    def test_correction_term_adds(self):
        """Non-zero C_i should increase diagonal of Sigma."""
        rng = np.random.default_rng(9)
        M, N = 20, 3
        B_tilde = rng.standard_normal((M, N))
        C_zero = [np.zeros((N, N)) for _ in range(M)]
        _, Sigma_no_corr = _m_step(B_tilde, C_zero)

        C_nonzero = [0.1 * np.eye(N) for _ in range(M)]
        _, Sigma_with_corr = _m_step(B_tilde, C_nonzero)

        # Diagonal should be larger with correction
        assert np.all(np.diag(Sigma_with_corr) > np.diag(Sigma_no_corr))

    def test_sigma_symmetric(self):
        rng = np.random.default_rng(10)
        M, N = 20, 4
        B_tilde = rng.standard_normal((M, N))
        C_list = [rng.standard_normal((N, N)) for _ in range(M)]
        # Make C_i symmetric
        C_list = [(C + C.T) / 2 for C in C_list]
        _, Sigma = _m_step(B_tilde, C_list)
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-14)


# ===================================================================
# Observed-data log-likelihood
# ===================================================================

class TestLoglik:
    def test_full_obs_matches_scipy(self):
        """With no missing data, should equal the multivariate normal logpdf sum."""
        from scipy.stats import multivariate_normal
        rng = np.random.default_rng(11)
        M, N = 30, 3
        mu = rng.standard_normal(N)
        A = rng.standard_normal((N, N))
        Sigma = A @ A.T + 0.2 * np.eye(N)
        B = rng.multivariate_normal(mu, Sigma, size=M)
        O = np.ones((M, N))

        ll_ours = _observed_loglik(B, O, mu, Sigma)
        ll_ref = multivariate_normal.logpdf(B, mean=mu, cov=Sigma).sum()
        np.testing.assert_allclose(ll_ours, ll_ref, rtol=1e-10)

    def test_missing_reduces_contribution(self):
        """Rows with fewer observations should contribute less to the loglik."""
        rng = np.random.default_rng(12)
        N = 3
        mu = np.zeros(N)
        Sigma = np.eye(N)
        B = rng.standard_normal((2, N))

        O_full = np.ones((2, N))
        ll_full = _observed_loglik(B, O_full, mu, Sigma)

        O_partial = np.array([[1, 1, 0], [1, 0, 0]], dtype=float)
        ll_partial = _observed_loglik(B * O_partial, O_partial, mu, Sigma)

        # Fewer observations → less extreme loglik magnitude
        assert abs(ll_partial) < abs(ll_full)

    def test_fully_missing_row_contributes_zero(self):
        """A row with O_i = 0 should contribute 0 to the loglik."""
        N = 2
        mu = np.zeros(N)
        Sigma = np.eye(N)
        B = np.array([[1.0, 2.0], [0.0, 0.0]])
        O = np.array([[1, 1], [0, 0]], dtype=float)
        ll = _observed_loglik(B, O, mu, Sigma)

        # Should equal loglik of just the first row
        O_one = np.ones((1, N))
        ll_one = _observed_loglik(B[:1], O_one, mu, Sigma)
        np.testing.assert_allclose(ll, ll_one, atol=1e-14)


# ===================================================================
# Full EM: convergence properties
# ===================================================================

class TestEMConvergence:
    def test_loglik_monotonically_increases(self):
        """EM must never decrease the observed-data log-likelihood."""
        _, B_obs, O, _, _ = _make_problem(M=60, N=5, missing_frac=0.3, seed=20)
        result = em_cov(B_obs, O, max_iter=50, tol=1e-10)
        ll = result["loglik"]
        diffs = np.diff(ll)
        # Allow tiny numerical noise
        assert np.all(diffs >= -1e-8), (
            f"Log-likelihood decreased: min diff = {diffs.min():.2e}"
        )

    def test_converges_flag(self):
        """Should converge within reasonable iterations for a small problem."""
        _, B_obs, O, _, _ = _make_problem(M=80, N=4, missing_frac=0.2, seed=21)
        result = em_cov(B_obs, O, max_iter=200, tol=1e-6)
        assert result["converged"]

    def test_more_iterations_if_needed(self):
        """With max_iter=1, should not converge."""
        _, B_obs, O, _, _ = _make_problem(M=60, N=5, seed=22)
        result = em_cov(B_obs, O, max_iter=1, tol=1e-12)
        assert result["n_iter"] == 1
        assert not result["converged"]


# ===================================================================
# Full EM: recovery quality
# ===================================================================

class TestEMRecovery:
    def test_recovers_mean(self):
        """Estimated mean should be close to the true mean (large M)."""
        B_full, B_obs, O, mu_true, _ = _make_problem(
            M=500, N=4, missing_frac=0.2, seed=30
        )
        result = em_cov(B_obs, O, max_iter=200)
        mu_err = np.linalg.norm(result["mu"] - mu_true)
        # With 500 samples, 4 dims, 20% missing, error should be modest
        assert mu_err < 1.0, f"Mean error too large: {mu_err:.4f}"

    def test_recovers_covariance(self):
        """Relative Frobenius error on Sigma should be reasonable."""
        B_full, B_obs, O, _, Sigma_true = _make_problem(
            M=500, N=4, missing_frac=0.2, seed=31
        )
        result = em_cov(B_obs, O, max_iter=200)
        rel_err = (np.linalg.norm(result["Sigma"] - Sigma_true, "fro")
                   / np.linalg.norm(Sigma_true, "fro"))
        assert rel_err < 0.5, f"Covariance relative error too large: {rel_err:.4f}"

    def test_no_missing_recovers_mle(self):
        """With 0% missing, EM should match the full-data MLE exactly."""
        rng = np.random.default_rng(32)
        M, N = 100, 3
        mu_true = rng.standard_normal(N)
        A = rng.standard_normal((N, N))
        Sigma_true = A @ A.T + 0.3 * np.eye(N)
        B = rng.multivariate_normal(mu_true, Sigma_true, size=M)
        O = np.ones((M, N))

        result = em_cov(B, O, max_iter=200, tol=1e-10)

        # MLE with no missing data
        mu_mle = B.mean(axis=0)
        B_c = B - mu_mle
        Sigma_mle = (B_c.T @ B_c) / M

        np.testing.assert_allclose(result["mu"], mu_mle, atol=1e-6)
        np.testing.assert_allclose(result["Sigma"], Sigma_mle, atol=1e-5)

    def test_imputed_matrix_shape_and_observed_unchanged(self):
        """B_imp should have correct shape and keep observed entries intact."""
        B_full, B_obs, O, _, _ = _make_problem(M=40, N=5, seed=33)
        result = em_cov(B_obs, O)
        B_imp = result["B_imp"]

        assert B_imp.shape == B_obs.shape
        # Observed entries must be preserved exactly
        mask = O.astype(bool)
        np.testing.assert_allclose(B_imp[mask], B_obs[mask], atol=1e-14)

    def test_imputation_closer_than_zero(self):
        """Imputed values should be closer to truth than just leaving zeros."""
        B_full, B_obs, O, _, _ = _make_problem(
            M=200, N=5, missing_frac=0.3, seed=34
        )
        result = em_cov(B_obs, O)
        B_imp = result["B_imp"]

        missing = ~O.astype(bool)
        err_zero = np.sqrt(np.mean((0.0 - B_full[missing]) ** 2))
        err_imp = np.sqrt(np.mean((B_imp[missing] - B_full[missing]) ** 2))
        assert err_imp < err_zero, (
            f"Imputation RMSE ({err_imp:.4f}) not better than zero ({err_zero:.4f})"
        )

    def test_imputation_closer_than_col_mean(self):
        """Imputed values should beat naive column-mean imputation."""
        B_full, B_obs, O, _, _ = _make_problem(
            M=200, N=5, missing_frac=0.3, seed=35
        )
        result = em_cov(B_obs, O)
        B_imp = result["B_imp"]

        # Column-mean baseline
        col_means = _mean_from_observed(B_obs, O)
        B_mean_imp = B_obs.copy()
        for j in range(B_obs.shape[1]):
            missing_j = O[:, j] == 0
            B_mean_imp[missing_j, j] = col_means[j]

        missing = ~O.astype(bool)
        err_mean = np.sqrt(np.mean((B_mean_imp[missing] - B_full[missing]) ** 2))
        err_em = np.sqrt(np.mean((B_imp[missing] - B_full[missing]) ** 2))
        assert err_em < err_mean, (
            f"EM RMSE ({err_em:.4f}) not better than column mean ({err_mean:.4f})"
        )


# ===================================================================
# Full EM: output structure
# ===================================================================

class TestEMOutput:
    def test_return_keys(self):
        _, B_obs, O, _, _ = _make_problem(M=30, N=3, seed=40)
        result = em_cov(B_obs, O, max_iter=10)
        for key in ("mu", "Sigma", "B_imp", "n_iter", "loglik", "converged"):
            assert key in result, f"Missing key: {key}"

    def test_sigma_symmetric(self):
        _, B_obs, O, _, _ = _make_problem(M=50, N=4, seed=41)
        result = em_cov(B_obs, O)
        np.testing.assert_allclose(result["Sigma"], result["Sigma"].T, atol=1e-14)

    def test_sigma_psd(self):
        _, B_obs, O, _, _ = _make_problem(M=50, N=4, seed=42)
        result = em_cov(B_obs, O)
        eigvals = np.linalg.eigvalsh(result["Sigma"])
        assert np.all(eigvals >= -1e-10)

    def test_shapes(self):
        M, N = 40, 6
        _, B_obs, O, _, _ = _make_problem(M=M, N=N, seed=43)
        result = em_cov(B_obs, O)
        assert result["mu"].shape == (N,)
        assert result["Sigma"].shape == (N, N)
        assert result["B_imp"].shape == (M, N)


# ===================================================================
# NaN interface
# ===================================================================

class TestNaNInterface:
    def test_nan_mask_equivalent(self):
        """Passing NaNs in B (without O) should give same result as explicit O."""
        B_full, B_obs, O, _, _ = _make_problem(M=60, N=4, seed=50)

        result_explicit = em_cov(B_obs, O, max_iter=50, tol=1e-8)

        B_nan = B_obs.copy()
        B_nan[O == 0] = np.nan
        result_nan = em_cov(B_nan, None, max_iter=50, tol=1e-8)

        np.testing.assert_allclose(result_explicit["mu"], result_nan["mu"], atol=1e-10)
        np.testing.assert_allclose(
            result_explicit["Sigma"], result_nan["Sigma"], atol=1e-8
        )

    def test_nan_no_crash(self):
        """NaN input shouldn't crash."""
        rng = np.random.default_rng(51)
        B = rng.standard_normal((30, 3))
        B[0, 1] = np.nan
        B[5, :] = np.nan
        B[10, 2] = np.nan
        result = em_cov(B, max_iter=20)
        assert np.all(np.isfinite(result["mu"]))
        assert np.all(np.isfinite(result["Sigma"]))


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_single_column(self):
        """N=1: scalar variance estimation."""
        rng = np.random.default_rng(60)
        B = rng.normal(5.0, 2.0, size=(100, 1))
        O = (rng.random((100, 1)) > 0.2).astype(float)
        for i in range(100):
            if O[i, 0] == 0:
                O[i, 0] = 1  # ensure at least one per row (trivial for N=1)
                break
        result = em_cov(B * O, O, max_iter=50)
        assert result["mu"].shape == (1,)
        assert result["Sigma"].shape == (1, 1)

    def test_two_models(self):
        """M=2 should not crash (very few observations)."""
        rng = np.random.default_rng(61)
        B = rng.standard_normal((2, 3))
        O = np.ones((2, 3))
        O[0, 2] = 0
        result = em_cov(B * O, O, max_iter=20)
        assert np.all(np.isfinite(result["mu"]))

    def test_high_missing_fraction(self):
        """60% missing: should still converge and give finite output."""
        _, B_obs, O, _, _ = _make_problem(M=100, N=4, missing_frac=0.6, seed=62)
        result = em_cov(B_obs, O, max_iter=200)
        assert np.all(np.isfinite(result["mu"]))
        assert np.all(np.isfinite(result["Sigma"]))
        assert np.all(np.isfinite(result["B_imp"]))

    def test_low_missing_fraction(self):
        """5% missing: should converge quickly."""
        _, B_obs, O, _, _ = _make_problem(M=100, N=4, missing_frac=0.05, seed=63)
        result = em_cov(B_obs, O, max_iter=200, tol=1e-8)
        assert result["converged"]
        assert result["n_iter"] <= 30

    def test_more_dims_than_obs(self):
        """N > M: rank-deficient regime.  Should not crash."""
        rng = np.random.default_rng(64)
        M, N = 5, 10
        B = rng.standard_normal((M, N))
        O = (rng.random((M, N)) > 0.2).astype(float)
        for i in range(M):
            if O[i].sum() == 0:
                O[i, rng.integers(N)] = 1.0
        for j in range(N):
            while O[:, j].sum() < 2:
                idx = rng.choice(np.where(O[:, j] == 0)[0])
                O[idx, j] = 1.0
        result = em_cov(B * O, O, max_iter=50)
        assert np.all(np.isfinite(result["mu"]))


# ===================================================================
# Degradation with missing rate
# ===================================================================

class TestDegradation:
    def test_error_increases_with_missing_rate(self):
        """More missing data → larger covariance estimation error."""
        errors = []
        for frac in [0.1, 0.3, 0.5]:
            B_full, B_obs, O, _, Sigma_true = _make_problem(
                M=300, N=4, missing_frac=frac, seed=70
            )
            result = em_cov(B_obs, O, max_iter=200)
            rel_err = (np.linalg.norm(result["Sigma"] - Sigma_true, "fro")
                       / np.linalg.norm(Sigma_true, "fro"))
            errors.append(rel_err)
        # Errors should be non-decreasing (with some tolerance for randomness)
        assert errors[-1] > errors[0] * 0.8, (
            f"Expected error to increase with missing rate, got {errors}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
