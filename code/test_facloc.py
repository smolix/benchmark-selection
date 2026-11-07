"""Tests for baseline_select.facility_location_select (facility location).

Facility location of Yao et al. (2026, arXiv 2607.09739, Sec. 4.4) on the
correlation matrix clipped at 0, F(A) = sum_j max_{i in A} max(R_ij, 0)
over all N benchmarks (s_jj = 1), maximized greedily over the eligible
candidates.  Checks the greedy against a brute-force objective, its
submodular structure (non-increasing gains, nesting, the (1 - 1/e) bound),
the treatment of eligibility (candidates only; ineligible benchmarks stay
targets), clipping, the near-tie rule (explained variance, then lowest
index), invariances, and that a CV split's selection depends on its
training rows only.  The similarity ablation (2026-10-19): "sq" (R_ij²)
against the brute-force greedy and greedy trace's first step, and
"clipped" / "abs" bit for bit against the implementation before "sq"
existed.  Integration: facloc and its variants keep every existing pair's
orientation in paired_stats (and MACRO_PAIRS reads as given); the columns
of eval_baselines' facloc, facloc_abs and facloc_sq are this function
(with their similarity) on the split's training estimate and shared
eligibility mask, scored by impute_and_evaluate, and the six other
methods' rows (and facloc's) are bit-identical to a run without the
appended methods (skipped while eval_baselines has no facloc).
Synthetic data, a few seconds.
"""

from __future__ import annotations

import atexit
import json
import itertools
import os
import shutil
import tempfile

# Outputs of the imported eval scripts go to a scratch root unless the
# caller (run_all.sh's tests job) set one
if not os.environ.get("BENCHSELECT_EXPERIMENT_ROOT"):
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-facloc-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

from baseline_select import (FACLOC_SIMILARITIES, facility_location_select,
                             facloc_similarity, hub_select)
from cv_split import cv_splits, eligible_mask, target_mask
from greedy_select import (TIE_RTOL, _cholesky_state, _eligible, _near_max,
                           _tie_break)

SEED = 42


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _factor_corr(N=20, n_factors=3, seed=0, positive=False):
    """Random factor-model correlation matrix (exact unit diagonal); mixed
    signs unless `positive`."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, n_factors))
    if positive:
        W = np.abs(W)
    S = W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))
    s = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return R


def _loading_corr(W):
    """Correlation matrix W Wᵀ + diag(1 − row norms²) of loadings W."""
    W = np.asarray(W, dtype=float)
    R = W @ W.T
    np.fill_diagonal(R, 1.0)
    return R


def _sim(R, similarity="clipped"):
    S = {"abs": np.abs, "sq": np.square}.get(
        similarity, lambda x: np.maximum(x, 0.0))(R)
    S = S.copy()
    np.fill_diagonal(S, 1.0)
    return S


def _F(R, A, similarity="clipped"):
    """Brute-force facility-location value of the set A (F(∅) = 0)."""
    if len(A) == 0:
        return 0.0
    S = _sim(R, similarity)
    return float(np.sum(np.max(S[list(A)], axis=0)))


def _mask(N, seed, p=0.7):
    rng = np.random.default_rng(seed)
    el = rng.random(N) < p
    el[rng.integers(N)] = True
    return el


def _cond_cov(Sigma, A):
    A = list(A)
    if not A:
        return Sigma.copy()
    S_AA = Sigma[np.ix_(A, A)]
    S_VA = Sigma[:, A]
    return Sigma - S_VA @ np.linalg.solve(S_AA, S_VA.T)


def _explained(R, A, j):
    """Brute-force drop of the conditional-variance trace of the unselected
    benchmarks (j included) when j joins A."""
    N = R.shape[0]
    U = [i for i in range(N) if i not in A]
    U2 = [i for i in U if i != j]
    return (np.trace(_cond_cov(R, A)[np.ix_(U, U)])
            - np.trace(_cond_cov(R, A + [j])[np.ix_(U2, U2)]))


def _naive_greedy(R, k, eligible=None, similarity="clipped", tol=1e-9):
    """Greedy on the brute-force objective with the project's near-tie rule
    (gains within tol·max(best, 1), then explained variance within
    tol·max(ev, 1), then the lowest index), written independently."""
    N = R.shape[0]
    pool = [j for j in range(N) if eligible is None or eligible[j]]
    A = []
    for _ in range(min(k, len(pool))):
        base = _F(R, A, similarity)
        gains = {j: _F(R, A + [j], similarity) - base
                 for j in pool if j not in A}
        best = max(gains.values())
        tied = [j for j, g in gains.items()
                if g >= best - tol * max(abs(best), 1.0)]
        if len(tied) > 1:
            ev = {j: _explained(R, A, j) for j in tied}
            top = max(ev.values())
            tied = [j for j in tied if ev[j] >= top - tol * max(top, 1.0)]
        A.append(min(tied))
    return A


def _reference_facloc(Sigma, k, eligible=None, similarity="clipped"):
    """facility_location_select as it was before similarity="sq" (and
    baseline_select.facloc_similarity) existed, for "clipped" and "abs":
    selected, gains, facloc_values, tie_breaks, index_ties."""
    N = Sigma.shape[0]
    el = _eligible(eligible, N)
    n_steps = min(k, N if el is None else int(el.sum()))
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    S = np.abs(R) if similarity == "abs" else np.maximum(R, 0.0)
    np.fill_diagonal(S, 1.0)
    cover = np.zeros(N, dtype=np.float64)
    unselected = np.ones(N, dtype=bool)
    selected, values, gains, tie_breaks, index_ties = [], [], [], [], []
    for t in range(n_steps):
        pool = np.flatnonzero(unselected if el is None
                              else unselected & el)
        gain = np.sum(np.maximum(S[:, pool] - cover[:, None], 0.0), axis=0)
        cand = pool[_near_max(gain, 1.0)]
        if len(cand) > 1:
            L, d = _cholesky_state(Sigma, selected)
            j_star, n_cand, n_index = _tie_break(
                Sigma, L, d, t, cand, np.flatnonzero(unselected))
            tie_breaks.append((t, n_cand))
            if n_index > 1:
                index_ties.append((t, n_index))
        else:
            j_star = int(cand[0])
        gains.append(float(gain[np.searchsorted(pool, j_star)]))
        selected.append(j_star)
        unselected[j_star] = False
        cover = np.maximum(cover, S[:, j_star])
        values.append(float(np.sum(cover)))
    pad = np.full(k - len(gains), np.nan)
    return {"selected": selected, "gains": np.concatenate([gains, pad]),
            "facloc_values": np.concatenate([values, pad]),
            "tie_breaks": tie_breaks, "index_ties": index_ties}


GENERIC = [(N, nf, s) for N, nf, s in
           [(12, 2, 0), (20, 3, 1), (25, 4, 2), (30, 3, 3), (18, 5, 4)]]


# ---------------------------------------------------------------------------
# Objective and greedy
# ---------------------------------------------------------------------------

class TestObjective:
    @pytest.mark.parametrize("N,nf,seed", GENERIC)
    @pytest.mark.parametrize("masked", [False, True])
    def test_picks_maximize_brute_force_gain(self, N, nf, seed, masked):
        R = _factor_corr(N, nf, seed)
        el = _mask(N, 100 + seed) if masked else None
        res = facility_location_select(R, 10, eligible=el)
        sel = res["selected"]
        for t, j in enumerate(sel):
            base = _F(R, sel[:t])
            gain = {v: _F(R, sel[:t] + [v]) - base for v in range(N)
                    if v not in sel[:t] and (el is None or el[v])}
            assert j in gain
            assert gain[j] >= max(gain.values()) - 1e-12
            assert res["gains"][t] == pytest.approx(gain[j], abs=1e-12)
            assert res["facloc_values"][t] == pytest.approx(
                _F(R, sel[:t + 1]), abs=1e-12)

    @pytest.mark.parametrize("N,nf,seed", GENERIC)
    def test_matches_naive_greedy(self, N, nf, seed):
        """The brute-force greedy with the same tie rule.  Generic matrices
        still have exact FL ties (two candidates that only cover themselves
        and each other gain (1 − c_i) + (s_ij − c_j) both); explained
        variance decides them, never the index."""
        R = _factor_corr(N, nf, seed)
        el = _mask(N, 200 + seed)
        for e in (None, el):
            res = facility_location_select(R, 12, eligible=e)
            assert res["selected"] == _naive_greedy(R, 12, e)
            assert res["index_ties"] == []

    def test_mutual_pair_tie(self):
        """i and j covering only themselves and each other tie exactly;
        the one explaining more residual variance wins."""
        R = np.eye(5)
        for i, j, r in [(0, 1, 0.6), (2, 3, 0.5), (2, 4, 0.3)]:
            R[i, j] = R[j, i] = r
        res = facility_location_select(R, 5)
        # step 1: 2 gains 1.8; then 0 and 1 tie at 1.6
        assert res["selected"][:2] == [2, 0]
        assert res["tie_breaks"][0] == (1, 2)
        assert res["index_ties"] == [(1, 2)]   # 0 and 1 exchangeable
        # break the symmetry in Σ only through a third benchmark that
        # neither covers (clipped at 0) but which 1 explains
        R2 = R.copy()
        R2[1, 4] = R2[4, 1] = -0.4
        assert np.linalg.eigvalsh(R2).min() > 0
        res2 = facility_location_select(R2, 5)
        assert res2["selected"][:2] == [2, 1]
        assert res2["tie_breaks"][0] == (1, 2)
        assert res2["index_ties"] == []

    @pytest.mark.parametrize("N,nf,seed", GENERIC)
    def test_submodular_structure(self, N, nf, seed):
        """Gains are non-increasing (F monotone submodular), F(A_t) is
        their running sum, each pick's own term makes its gain > 0, and
        the selections are nested in k."""
        R = _factor_corr(N, nf, seed)
        res = facility_location_select(R, N)
        g = res["gains"]
        assert np.all(g > 0)
        assert np.all(np.diff(g) <= 1e-12)
        np.testing.assert_allclose(np.cumsum(g), res["facloc_values"],
                                   atol=1e-10)
        assert res["facloc_values"][-1] == pytest.approx(N)
        for k in (1, 3, 7):
            assert facility_location_select(R, k)["selected"] == \
                res["selected"][:k]

    @pytest.mark.parametrize("seed", range(4))
    def test_greedy_bound_exhaustive(self, seed):
        R = _factor_corr(9, 2, 10 + seed)
        res = facility_location_select(R, 4)
        for k in range(1, 5):
            opt = max(_F(R, A) for A in itertools.combinations(range(9), k))
            val = res["facloc_values"][k - 1]
            assert val >= (1 - 1 / np.e) * opt - 1e-12
            assert val <= opt + 1e-12
        opt1 = max(_F(R, [j]) for j in range(9))
        assert res["facloc_values"][0] == pytest.approx(opt1)

    @pytest.mark.parametrize("N,nf,seed", GENERIC[:3])
    def test_residual_variance_trajectory(self, N, nf, seed):
        R = _factor_corr(N, nf, seed)
        res = facility_location_select(R, 6)
        sel = res["selected"]
        for t in range(1, 7):
            C = _cond_cov(R, sel[:t])
            unsel = [i for i in range(N) if i not in sel[:t]]
            assert res["residual_variance"][t - 1] == pytest.approx(
                np.trace(C[np.ix_(unsel, unsel)]) / N, rel=1e-6)
            assert res["log_dets"][t - 1] == pytest.approx(
                np.linalg.slogdet(R[np.ix_(sel[:t], sel[:t])])[1], abs=1e-8)


# ---------------------------------------------------------------------------
# Similarity: clipping, hub agreement
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_negative_correlations_clipped(self):
        """A benchmark anti-correlated with all others covers only itself
        under the clipped similarity; |R| (and hub) pick it first."""
        lam = np.array([-0.95, 0.6, 0.55, 0.5, 0.45, 0.6, 0.5, 0.4])
        R = _loading_corr(lam[:, None])
        res = facility_location_select(R, 8)
        assert res["selected"][0] != 0
        assert facility_location_select(R, 1, similarity="abs")[
            "selected"] == [0]
        assert hub_select(R, 1)["selected"] == [0]
        # whenever it is picked, benchmark 0 gains exactly its own term
        t0 = res["selected"].index(0)
        assert res["gains"][t0] == pytest.approx(1.0)
        assert _F(R, [0]) == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", range(5))
    def test_first_pick_is_hub_without_negative_correlations(self, seed):
        """With R >= 0 the step-1 gain is 1 + (N − 1) · hub score."""
        R = _factor_corr(20, 2, seed, positive=True)
        assert np.all(R >= 0)
        res = facility_location_select(R, 1)
        hub = hub_select(R, 1)
        assert res["selected"] == hub["selected"]
        assert res["gains"][0] == pytest.approx(
            1 + 19 * hub["hub_scores"][res["selected"][0]])

    def test_abs_equals_clipped_on_nonnegative(self):
        R = _factor_corr(15, 3, 7, positive=True)
        a = facility_location_select(R, 10)
        b = facility_location_select(R, 10, similarity="abs")
        assert a["selected"] == b["selected"]
        np.testing.assert_array_equal(a["facloc_values"], b["facloc_values"])

    def test_unknown_similarity(self):
        assert FACLOC_SIMILARITIES == ("clipped", "abs", "sq")
        with pytest.raises(ValueError):
            facility_location_select(np.eye(3), 1, similarity="cos")
        with pytest.raises(ValueError):
            facloc_similarity(np.eye(3), "cos")

    @pytest.mark.parametrize("similarity", ["clipped", "abs", "sq"])
    def test_similarity_matrix(self, similarity):
        """facloc_similarity rescales Σ to a correlation and applies the
        similarity, diagonal exactly 1 (also for a covariance)."""
        R = _factor_corr(12, 3, 21)
        d = np.linspace(0.5, 3.0, 12)
        Sigma = R * np.outer(d, d)
        S = facloc_similarity(Sigma, similarity)
        np.testing.assert_allclose(S, _sim(R, similarity), atol=1e-14)
        assert np.all(np.diag(S) == 1.0)

    @pytest.mark.parametrize("similarity", ["clipped", "abs"])
    @pytest.mark.parametrize("N,nf,seed", GENERIC)
    def test_clipped_abs_unchanged(self, similarity, N, nf, seed):
        """"clipped" and "abs" are bit for bit the implementation before
        "sq" was added (inlined here), with and without eligibility."""
        R = _factor_corr(N, nf, seed)
        el = _mask(N, 300 + seed)
        for e in (None, el):
            ref = _reference_facloc(R, 10, e, similarity)
            res = facility_location_select(R, 10, eligible=e,
                                           similarity=similarity)
            assert res["selected"] == ref["selected"]
            assert res["tie_breaks"] == ref["tie_breaks"]
            assert res["index_ties"] == ref["index_ties"]
            for key in ("gains", "facloc_values"):
                np.testing.assert_array_equal(res[key], ref[key])

    @pytest.mark.parametrize("N,nf,seed", GENERIC)
    def test_sq_matches_naive_greedy(self, N, nf, seed):
        R = _factor_corr(N, nf, seed)
        el = _mask(N, 400 + seed)
        for e in (None, el):
            res = facility_location_select(R, 8, eligible=e, similarity="sq")
            assert res["selected"] == _naive_greedy(R, 8, e, "sq")
            assert res["similarity"] == "sq"

    @pytest.mark.parametrize("N,nf,seed", GENERIC)
    def test_sq_first_step_is_trace_first_step(self, N, nf, seed):
        """On a correlation matrix the "sq" step-1 gain of j is
        1 + sum_{i != j} R_ij^2 = sum_i R_ij^2 / R_jj, greedy trace's
        first-step reduction, so the first picks agree (generic Σ: no
        near-ties); later picks credit the best single proxy only."""
        from baseline_select import greedy_trace_select
        R = _factor_corr(N, nf, seed)
        res = facility_location_select(R, 1, similarity="sq")
        tr = greedy_trace_select(R, 1)
        assert res["selected"] == tr["selected"]
        j = res["selected"][0]
        assert res["gains"][0] == pytest.approx(np.sum(R[:, j] ** 2),
                                                rel=1e-12)
        assert 1.0 - tr["residual_variance"][0] == pytest.approx(
            res["gains"][0] / N, rel=1e-10)

    def test_diagonal_is_one(self):
        """Self-similarity is 1 whatever the rounding of the diagonal: on the
        identity every pick gains exactly 1."""
        R = np.eye(6)
        R[np.diag_indices(6)] += np.linspace(-1e-15, 1e-15, 6)
        res = facility_location_select(R, 6)
        np.testing.assert_array_equal(res["gains"], np.ones(6))


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

class TestEligibility:
    @staticmethod
    def _ineligible_targets_corr():
        """0 ~ 2 (factor A); 1 ~ 3..6 (factor B); 3..6 ineligible."""
        W = np.zeros((7, 2))
        W[[0, 2], 0] = 0.9
        W[1, 1] = 0.9
        W[3:, 1] = 0.8
        return _loading_corr(W), np.array([1, 1, 1, 0, 0, 0, 0], bool)

    def test_ineligible_benchmarks_count_as_targets(self):
        R, el = self._ineligible_targets_corr()
        res = facility_location_select(R, 3, eligible=el)
        # 1 covers the four ineligible benchmarks: gain 1 + 4·0.72
        assert res["selected"][0] == 1
        assert res["gains"][0] == pytest.approx(1 + 4 * 0.72)
        assert set(res["selected"]) == {0, 1, 2}
        # dropping them from the ground set would pick 0 (tied with 2) first
        sub = [0, 1, 2]
        assert facility_location_select(R[np.ix_(sub, sub)], 1)[
            "selected"] == [0]

    @pytest.mark.parametrize("N,nf,seed", GENERIC)
    def test_only_eligible_selected(self, N, nf, seed):
        R = _factor_corr(N, nf, seed)
        el = _mask(N, 300 + seed, p=0.5)
        res = facility_location_select(R, N, eligible=el)
        assert sorted(res["selected"]) == np.flatnonzero(el).tolist()
        n = int(el.sum())
        for key in ("gains", "facloc_values", "residual_variance",
                    "log_dets"):
            assert res[key].shape == (N,)
            assert np.all(np.isfinite(res[key][:n]))
            assert np.all(np.isnan(res[key][n:]))

    def test_all_true_mask_is_unrestricted(self):
        R = _factor_corr(20, 3, 5)
        a = facility_location_select(R, 10)
        b = facility_location_select(R, 10, eligible=np.ones(20, bool))
        assert a["selected"] == b["selected"]
        np.testing.assert_array_equal(a["gains"], b["gains"])

    def test_no_eligible(self):
        R = _factor_corr(8, 2, 0)
        res = facility_location_select(R, 3, eligible=np.zeros(8, bool))
        assert res["selected"] == []
        assert np.all(np.isnan(res["gains"]))
        assert np.all(np.isnan(res["residual_variance"]))

    def test_mask_shape_checked(self):
        with pytest.raises(ValueError):
            facility_location_select(np.eye(4), 2, eligible=np.ones(3, bool))


# ---------------------------------------------------------------------------
# Near-ties and invariances
# ---------------------------------------------------------------------------

class TestTies:
    def test_tie_broken_by_explained_variance(self):
        """Gains of 0 and 1 tie exactly (1 + 0.5 + 0.5 = 1 + 0.8 + 0.2);
        1 explains more variance (1.68 vs 1.5) and wins over the lower
        index."""
        R = np.eye(6)
        for i, j, r in [(0, 2, 0.5), (0, 3, 0.5), (1, 4, 0.8), (1, 5, 0.2)]:
            R[i, j] = R[j, i] = r
        assert np.linalg.eigvalsh(R).min() > 0
        res = facility_location_select(R, 2)
        assert res["selected"][0] == 1
        assert res["tie_breaks"][0] == (0, 2)
        assert res["index_ties"] == []
        # after 1, the tie is gone: 0 gains 2
        assert res["selected"][1] == 0

    def test_exchangeable_block_lowest_index(self):
        """Exchangeable columns tie on gain and explained variance: the
        lowest index decides (recorded in index_ties), also under
        rounding-level noise."""
        lam = np.array([0.4, 0.8, 0.5, 0.8, 0.3, 0.8, 0.45])
        R = _loading_corr(lam[:, None])
        res = facility_location_select(R, 3)
        assert res["selected"][0] == 1
        assert res["index_ties"][0] == (0, 3)
        rng = np.random.default_rng(0)
        E = rng.uniform(-1e-15, 1e-15, R.shape)
        Rp = R + (E + E.T) / 2
        assert facility_location_select(Rp, 3)["selected"] == \
            res["selected"]

    @pytest.mark.parametrize("N,nf,seed", GENERIC)
    def test_permutation_equivariance(self, N, nf, seed):
        R = _factor_corr(N, nf, seed)
        el = _mask(N, 400 + seed)
        perm = np.random.default_rng(seed).permutation(N)
        Rp = R[np.ix_(perm, perm)]
        a = facility_location_select(R, 10, eligible=el)["selected"]
        b = facility_location_select(Rp, 10, eligible=el[perm])["selected"]
        assert [int(perm[j]) for j in b] == a

    @pytest.mark.parametrize("c", [1e-3, 7.5])
    def test_covariance_scale_invariance(self, c):
        """Sigma is rescaled to a correlation: a covariance (per-column
        scales) selects as its correlation matrix."""
        R = _factor_corr(20, 3, 9)
        scales = c * np.linspace(0.5, 2.0, 20)
        Sigma = R * np.outer(scales, scales)
        assert facility_location_select(Sigma, 10)["selected"] == \
            facility_location_select(R, 10)["selected"]

    def test_tolerance_is_relative_to_gain(self):
        """Gains closer than TIE_RTOL · max(best, 1) tie; farther apart they
        do not."""
        lam = np.array([0.7, 0.7, 0.2, 0.3])
        for delta, tied in [(0.1 * TIE_RTOL, True), (1e3 * TIE_RTOL, False)]:
            R = _loading_corr(lam[:, None])
            R[1, 2] = R[2, 1] = R[1, 2] + delta
            res = facility_location_select(R, 1)
            assert (res["tie_breaks"] != []) == tied
            if not tied:
                assert res["selected"] == [1]

    def test_deterministic_and_no_global_rng(self):
        R = _factor_corr(25, 4, 11)
        state = np.random.get_state()
        a = facility_location_select(R, 15)
        b = facility_location_select(R, 15)
        assert np.random.get_state()[1].tolist() == state[1].tolist()
        assert a["selected"] == b["selected"]
        for key in ("gains", "facloc_values", "residual_variance"):
            np.testing.assert_array_equal(a[key], b[key])

    def test_names(self):
        R = _factor_corr(6, 2, 0)
        names = np.array([f"b{j}" for j in range(6)])
        res = facility_location_select(R, 3, benchmark_names=names)
        assert res["names"] == [f"b{j}" for j in res["selected"]]


# ---------------------------------------------------------------------------
# CV harness: no leakage from validation rows
# ---------------------------------------------------------------------------

def _synthetic_scores(M=60, N=10, seed=0, p_obs=0.85):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((M, 2))
    W = rng.uniform(0.3, 0.9, (2, N))
    B = Z @ W + 0.4 * rng.standard_normal((M, N))
    O = (rng.random((M, N)) < p_obs).astype(float)
    return B * O, O


class TestNoLeakage:
    def test_selection_ignores_validation_rows(self):
        from eval_entropy_vs_mi import estimate_corr
        B, O = _synthetic_scores()
        M, N = B.shape
        rng = np.random.default_rng(1)
        for s in cv_splits(M, 5, 2, 0.1, SEED):
            el = eligible_mask(O[s.train_idx])
            R, _, _ = estimate_corr(B[s.train_idx], O[s.train_idx])
            sel = facility_location_select(R, 6, eligible=el)["selected"]
            # corrupt the validation rows (values and observation pattern)
            B2, O2 = B.copy(), O.copy()
            B2[s.val_idx] = 100 * rng.standard_normal((len(s.val_idx), N))
            O2[s.val_idx] = 1.0
            el2 = eligible_mask(O2[s.train_idx])
            R2, _, _ = estimate_corr(B2[s.train_idx], O2[s.train_idx])
            assert np.array_equal(el, el2)
            np.testing.assert_array_equal(R, R2)
            assert facility_location_select(R2, 6, eligible=el2)[
                "selected"] == sel

    def test_benchmark_seen_mostly_in_validation_is_not_selectable(self):
        """A benchmark observed by fewer than MIN_TRAIN_OBS training models
        (but by every validation model, and built to be a strong facility)
        is never picked: eligibility comes from the training rows only."""
        from eval_entropy_vs_mi import estimate_corr
        B, O = _synthetic_scores(seed=3)
        M, N = B.shape
        s = next(iter(cv_splits(M, 5, 1, 0.1, SEED)))
        hub_col = 0
        # make column 0 the sum of all others (best facility) ...
        B[:, hub_col] = B[:, 1:].sum(axis=1)
        O[:, hub_col] = 0.0
        # ... observed only in the validation rows and two training rows
        O[s.val_idx, hub_col] = 1.0
        O[s.train_idx[:2], hub_col] = 1.0
        B = B * O
        el = eligible_mask(O[s.train_idx])
        assert not el[hub_col]
        R, _, _ = estimate_corr(B[s.train_idx], O[s.train_idx])
        sel = facility_location_select(R, N, eligible=el)["selected"]
        assert hub_col not in sel
        assert len(sel) == int(el.sum())


# ---------------------------------------------------------------------------
# Integration: paired_stats orientation, eval_baselines wiring
# ---------------------------------------------------------------------------

# eval_baselines.METHODS before facloc
BASE_METHODS = ["entropy", "mi", "random", "hub", "pca", "trace"]


class TestIntegration:
    def test_paired_orientation(self):
        """facloc is not in paired_stats.METHOD_ORDER, so it ranks between
        hub and random: appending it to eval_baselines' methods keeps every
        existing pair and its orientation (the paired CSV only gains
        rows), and every MACRO_PAIRS pair (facloc ones included) reads as
        given."""
        from paired_stats import MACRO_PAIRS, orient, orient_pair
        old = orient(BASE_METHODS)
        new = orient(BASE_METHODS + ["facloc"])
        assert [p for p in new if "facloc" not in p] == old
        assert {p for p in new if "facloc" in p} == {
            ("mi", "facloc"), ("entropy", "facloc"), ("trace", "facloc"),
            ("pca", "facloc"), ("hub", "facloc"), ("facloc", "random")}
        for a, b in MACRO_PAIRS:
            assert orient_pair(a, b) == (a, b), (a, b)
        # the similarity variants (eval_baselines' facloc_abs, facloc_sq)
        # rank with facloc, after it in the given order: every earlier
        # pair keeps its orientation
        var = orient(BASE_METHODS + ["facloc", "facloc_abs", "facloc_sq"])
        # MACRO_PAIRS' facloc pairs: against the core methods (pairs of
        # `new`) and against the variants (facloc first)
        assert set(p for p in MACRO_PAIRS if "facloc" in p
                   and not {"facloc_abs", "facloc_sq"} & set(p)) <= set(new)
        assert set(p for p in MACRO_PAIRS if "facloc" in p) <= set(var)
        assert [p for p in var if not {"facloc_abs", "facloc_sq"} & set(p)] \
            == new
        assert ("facloc", "facloc_abs") in var and \
            ("facloc", "facloc_sq") in var and \
            ("facloc_abs", "facloc_sq") in var
        assert ("trace", "facloc_sq") in var and \
            ("facloc_sq", "random") in var

    def test_eval_baselines_wiring(self, tmp_path, monkeypatch):
        """End to end on a synthetic matrix: eval_baselines' facloc,
        facloc_abs and facloc_sq rows are facility_location_select (with
        their similarity) on the split's training estimate with the shared
        eligibility mask, scored by impute_and_evaluate, and the other
        methods' rows equal a run without them (they draw no random
        numbers, so no common-random-number stream shifts)."""
        import eval_baselines as eb
        if "facloc" not in eb.METHODS:
            pytest.skip("eval_baselines.METHODS has no facloc yet")
        from eval_entropy_vs_mi import estimate_corr, impute_and_evaluate
        M, N, k_max, folds, reps = 60, 10, 6, 5, 2
        B, O = _synthetic_scores(M, N, seed=5, p_obs=0.8)
        # column N-1: the best facility, observed by three models only, so
        # it is ineligible in the splits that hold one of them out
        B[:, -1] = B[:, :-1].sum(axis=1)
        O[:, -1] = 0.0
        O[:3, -1] = 1.0
        B = B * O
        np.savez(tmp_path / "synth.matrix.npz", B=B, O=O,
                 benchmark_names=np.array([f"b{j}" for j in range(N)]))
        (tmp_path / "synth.meta.json").write_text(json.dumps(
            {"score_bounds": {"default": [None, None]}}))
        monkeypatch.setattr(eb, "DATA", tmp_path)
        metrics, out, _, kk, _ = eb.run_dataset("synth", k_max, folds, reps)
        assert kk == k_max
        splits = list(cv_splits(M, folds, reps, eb.HOLDOUT_FRAC, eb.SEED))
        assert out["split_id"].tolist() == [s.split_id for s in splits]
        assert 0 < int((~out["eligible"][:, -1]).sum()) < len(splits)
        # facloc and its similarity ablation (eval_baselines.FACLOC_VARIANTS)
        assert eb.FACLOC_VARIANTS == {"facloc": "clipped",
                                      "facloc_abs": "abs", "facloc_sq": "sq"}
        assert eb.METHODS[-3:] == ["facloc", "facloc_abs", "facloc_sq"]
        for i, s in enumerate(splits):
            el = eligible_mask(O[s.train_idx])
            np.testing.assert_array_equal(out["eligible"][i], el)
            R, mu, sd = estimate_corr(B[s.train_idx], O[s.train_idx])
            for m, sim in eb.FACLOC_VARIANTS.items():
                res = facility_location_select(R, k_max, eligible=el,
                                               similarity=sim)
                sel = res["selected"]
                assert out[f"{m}_selected"][i].tolist() == \
                    sel + [-1] * (k_max - len(sel))
                for k in range(1, k_max + 1):
                    if k > len(sel):
                        assert np.isnan(metrics[m]["r2"][i, k - 1])
                        continue
                    # the split's targets (rule D1; the column of three
                    # observations is no target where two of them are held
                    # out) and no valid-range clip (unbounded synthetic
                    # data)
                    r2 = impute_and_evaluate(
                        B[s.val_idx], O[s.val_idx], sel[:k], R, mu, sd, N,
                        target=target_mask(O[s.train_idx]))["r2"]
                    assert metrics[m]["r2"][i, k - 1] == r2
                    assert metrics[m]["resid_var"][i, k - 1] == \
                        res["residual_variance"][k - 1]
        # appending facloc and its variants leaves the other methods bit
        # for bit, and appending the variants leaves facloc bit for bit
        others = [m for m in eb.METHODS if m not in eb.FACLOC_VARIANTS]
        assert others == BASE_METHODS
        for subset in (others, others + ["facloc"]):
            monkeypatch.setattr(eb, "METHODS", subset)
            metrics7, out7, _, _, _ = eb.run_dataset("synth", k_max, folds,
                                                     reps)
            for m in subset:
                for key in eb.METRICS:
                    np.testing.assert_array_equal(metrics[m][key],
                                                  metrics7[m][key])
                for key in ("selected", "index_ties", "tie_breaks"):
                    np.testing.assert_array_equal(out[f"{m}_{key}"],
                                                  out7[f"{m}_{key}"])
        monkeypatch.setattr(eb, "METHODS", others + list(eb.FACLOC_VARIANTS))
        rows = eb.paired_rows("synth", metrics, out, k_max)
        pairs = {(r["method_a"], r["method_b"]) for r in rows}
        assert {("mi", "facloc"), ("trace", "facloc"), ("facloc", "random"),
                ("facloc", "facloc_abs"), ("facloc", "facloc_sq"),
                ("facloc_abs", "random")} <= pairs
