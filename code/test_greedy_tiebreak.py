"""Tests for the deterministic entropy tie-break (greedy_select, budgeted greedy)."""

from __future__ import annotations

import numpy as np
import pytest

from greedy_select import (
    TIE_RTOL,
    _explained_variance,
    greedy_entropy,
    greedy_mi,
)
from eval_cost_budget import (
    D_FLOOR,
    _chol_state_add,
    _chol_state_init,
    _complement_precision_full,
    _mi_gain,
    budgeted_greedy,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _factor_corr(N=30, n_factors=3, seed=0):
    """Random factor-model correlation matrix with an exactly unit diagonal."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, n_factors))
    S = W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))
    s = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return R


def _hub_corr(N=30, hub=17, seed=0):
    """One-factor correlation matrix whose hub (largest loading) is `hub`."""
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0.3, 0.7, N)
    lam[hub] = 0.95
    R = np.outer(lam, lam)
    np.fill_diagonal(R, 1.0)
    return R


def _perturb_diag(R, rng, scale=1e-15):
    """R with its diagonal perturbed by U(-scale, scale) rounding-level noise."""
    Rp = R.copy()
    Rp[np.diag_indices_from(Rp)] += rng.uniform(-scale, scale, R.shape[0])
    return Rp


def _hub_score(R):
    return np.sum(R ** 2, axis=0)


def _loading_corr(W):
    """Correlation W Wᵀ + diag(1 − |w_i|²) from factor loadings (rows of W)."""
    R = W @ W.T
    np.fill_diagonal(R, 1.0)
    return R


def _later_tie_corr():
    """Hub 0; after it, 1 and 2 tie exactly on d but 2 explains more.

    Both load 0.3 on the hub's factor (equal residual variance after the
    hub); only 2 shares the second factor with the group 3..6.
    """
    W = np.array([[0.95, 0.0],
                  [0.3, 0.0], [0.3, 0.5],
                  [0.5, 0.4], [0.55, 0.35], [0.6, 0.3], [0.5, 0.45]])
    return _loading_corr(W)


def _block_corr():
    """Hub 0, distinct 1..3, exchangeable block 4..7 picked at steps 1-3."""
    W = np.array([[0.95, 0.0],
                  [0.6, 0.1], [0.55, 0.2], [0.5, 0.3],
                  [0.2, 0.6], [0.2, 0.6], [0.2, 0.6], [0.2, 0.6]])
    return _loading_corr(W)


# Frozen copies of the pre-tie-break algorithms (reference implementations)

def _argmax_greedy_entropy(Sigma, k):
    """Plain argmax pivoting: greedy_entropy before the tie-break."""
    N = Sigma.shape[0]
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, k), dtype=np.float64)
    selected = []
    for t in range(k):
        d_masked = d.copy()
        for j in selected:
            d_masked[j] = -np.inf
        j_star = int(np.argmax(d_masked))
        selected.append(j_star)
        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2
    return selected, d


def _frozen_complement_precision_diag(Sigma, comp):
    from scipy.linalg import solve_triangular
    m = len(comp)
    Sigma_sub = Sigma[np.ix_(comp, comp)]
    try:
        L = np.linalg.cholesky(Sigma_sub)
        L_inv = solve_triangular(L, np.eye(m), lower=True)
        return np.sum(L_inv ** 2, axis=0)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma_sub)
        eigvals = np.maximum(eigvals, 1e-10)
        return np.sum(eigvecs ** 2 / eigvals[None, :], axis=1)


def _frozen_greedy_mi(Sigma, k):
    """greedy_mi selection loop as of HEAD before the tie-break change."""
    N = Sigma.shape[0]
    total_var = np.trace(Sigma)
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, k), dtype=np.float64)
    selected, selected_set = [], set()
    log_dets, residual_variance, gains = [], [], []
    cum_logdet = 0.0
    for t in range(k):
        comp = np.array(sorted(set(range(N)) - selected_set))
        P_diag_comp = _frozen_complement_precision_diag(Sigma, comp)
        P_full = np.zeros(N)
        for i, idx in enumerate(comp):
            P_full[idx] = P_diag_comp[i]
        best_j, best_score = int(comp[0]), -np.inf
        for j in comp:
            dj, pj = d[j], P_full[j]
            if dj > 1e-300 and pj > 1e-300:
                score = np.log(dj) + np.log(pj)
            else:
                score = -np.inf
            if score > best_score:
                best_score, best_j = score, int(j)
        j_star = best_j
        selected.append(j_star)
        selected_set.add(j_star)
        gains.append(0.5 * best_score)
        cum_logdet += np.log(max(d[j_star], 1e-300))
        log_dets.append(cum_logdet)
        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected_set and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2
        resid = sum(d[j] for j in range(N) if j not in selected_set)
        residual_variance.append(resid / total_var)
    return {"selected": selected, "d": d, "log_dets": np.array(log_dets),
            "residual_variance": np.array(residual_variance),
            "gains": np.array(gains)}


def _frozen_budgeted_mi(Sigma, costs, budget):
    """budgeted_greedy(objective="mi") as of HEAD before the change."""
    N = Sigma.shape[0]
    state = _chol_state_init(Sigma)
    spent, greedy_value = 0.0, 0.0
    while True:
        best_j, best_ratio, best_gain = -1, -np.inf, 0.0
        P_full = _complement_precision_full(Sigma, state["selected"])
        for j in range(N):
            if j in state["selected"] or spent + costs[j] > budget:
                continue
            gain = _mi_gain(state, Sigma, P_full, j)
            ratio = gain / costs[j]
            if ratio > best_ratio:
                best_ratio, best_j, best_gain = ratio, j, gain
        if best_j < 0 or best_ratio <= 0:
            break
        spent += costs[best_j]
        greedy_value += best_gain
        _chol_state_add(state, Sigma, best_j)
    greedy_set = list(state["selected"])
    afford = [j for j in range(N) if costs[j] <= budget]
    if afford:
        P0 = _complement_precision_full(Sigma, [])
        def single_value(j):
            dj = max(Sigma[j, j], 1e-300)
            pj = max(P0[j], 1e-300)
            return max(0.5 * (np.log(dj) + np.log(pj)), 0.0)
        best_single = max(afford, key=single_value)
        if single_value(best_single) > greedy_value:
            return [best_single]
    return greedy_set


# ===================================================================
# Explained-variance helper
# ===================================================================

class TestExplainedVariance:
    @pytest.mark.parametrize("pivots", [[], [4], [4, 11], [0, 7, 19, 3]])
    def test_matches_brute_force(self, pivots):
        """Cholesky-state helper == trace drop of explicit Schur complement."""
        R = _factor_corr(N=24, seed=3)
        N = R.shape[0]
        state = _chol_state_init(R)
        for j in pivots:
            _chol_state_add(state, R, j)
        unsel = np.array([j for j in range(N) if j not in pivots])
        ev = _explained_variance(R, state["L"], state["d"], state["t"],
                                 unsel, unsel)

        if pivots:
            S = np.array(pivots)
            R_res = R - R[:, S] @ np.linalg.solve(R[np.ix_(S, S)], R[S, :])
        else:
            R_res = R
        R_uu = R_res[np.ix_(unsel, unsel)]
        expected = np.sum(R_uu ** 2, axis=0) / np.diag(R_uu)
        np.testing.assert_allclose(ev, expected, rtol=1e-10)

    def test_step1_is_hub_score(self):
        R = _factor_corr(N=20, seed=4)
        N = R.shape[0]
        all_idx = np.arange(N)
        ev = _explained_variance(R, np.zeros((N, 0)), np.diag(R).copy(), 0,
                                 all_idx, all_idx)
        np.testing.assert_allclose(ev, _hub_score(R), rtol=1e-14)


# ===================================================================
# greedy_entropy tie-break
# ===================================================================

class TestGreedyEntropyTieBreak:
    def test_first_pick_is_hub(self):
        """(a) Unit diagonal: the first pick is argmax_j sum_i R_ij²."""
        R = _hub_corr(N=30, hub=17)
        res = greedy_entropy(R, 5)
        assert res["selected"][0] == int(np.argmax(_hub_score(R))) == 17
        assert res["tie_breaks"][0] == (0, 30)
        # Control: plain argmax resolves the exact tie to index 0
        assert _argmax_greedy_entropy(R, 1)[0] == [0]

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_first_pick_is_hub_factor_model(self, seed):
        R = _factor_corr(N=30, seed=seed)
        res = greedy_entropy(R, 3)
        assert res["selected"][0] == int(np.argmax(_hub_score(R)))

    @pytest.mark.parametrize("seed", [0, 1])
    def test_invariant_to_diagonal_noise(self, seed):
        """(b) Selection unchanged under 50 ±1e-15 diagonal perturbations."""
        R = _factor_corr(N=30, seed=seed)
        N = R.shape[0]
        ref = greedy_entropy(R, N)["selected"]
        rng = np.random.default_rng(100 + seed)
        old_first = set()
        for _ in range(50):
            Rp = _perturb_diag(R, rng)
            assert greedy_entropy(Rp, N)["selected"] == ref
            old_first.add(_argmax_greedy_entropy(Rp, 1)[0][0])
        # Control: the same noise does move the plain-argmax first pick
        assert len(old_first) > 1

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_permutation_equivariance(self, seed):
        """(c) Permuting rows/cols permutes the selection accordingly."""
        R = _factor_corr(N=30, seed=seed)
        N = R.shape[0]
        ref = greedy_entropy(R, N)["selected"]
        rng = np.random.default_rng(200 + seed)
        for _ in range(5):
            p = rng.permutation(N)
            sel_p = greedy_entropy(R[np.ix_(p, p)], N)["selected"]
            assert [int(p[j]) for j in sel_p] == ref

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_no_ties_equals_argmax_pivoting(self, seed):
        """(d) Distinct conditional variances: identical to plain argmax."""
        R = _factor_corr(N=30, seed=seed)
        N = R.shape[0]
        scale = np.random.default_rng(300 + seed).uniform(0.5, 3.0, N)
        Sigma = R * np.outer(scale, scale)
        res = greedy_entropy(Sigma, N)
        sel_old, d_old = _argmax_greedy_entropy(Sigma, N)
        assert res["selected"] == sel_old
        assert np.array_equal(res["d"], d_old)
        assert res["tie_breaks"] == []
        assert res["index_ties"] == []

    def test_tie_breaks_diagnostic(self):
        R = _factor_corr(N=20, seed=5)
        res = greedy_entropy(R, 20)
        assert res["tie_breaks"][0] == (0, 20)
        assert all(n > 1 for _, n in res["tie_breaks"])
        # Generic matrix: explained variance always decides, never the index
        assert res["index_ties"] == []

    def test_later_step_tie_uses_explained_variance(self):
        """A d-tie at step 2 is broken by explained variance, not by index."""
        R = _later_tie_corr()
        res = greedy_entropy(R, 2)
        assert res["selected"] == [0, 2]
        assert res["tie_breaks"] == [(0, 7), (1, 2)]
        assert res["index_ties"] == []
        # Control: the tie is exact, so plain argmax takes the lower index
        sel_old, _ = _argmax_greedy_entropy(R, 2)
        assert sel_old == [0, 1]

    def test_exchangeable_block_lowest_index(self):
        """Exchangeable columns: lowest index decides, robust to noise."""
        R = _block_corr()
        N = R.shape[0]
        ref = greedy_entropy(R, N)
        assert ref["selected"][:4] == [0, 4, 5, 6]
        assert ref["index_ties"][:3] == [(1, 4), (2, 3), (3, 2)]
        rng = np.random.default_rng(500)
        ev_argmax = set()
        block = np.arange(4, 8)
        for _ in range(50):
            Rp = _perturb_diag(R, rng)
            assert greedy_entropy(Rp, N)["selected"] == ref["selected"]
            # Control: an exact argmax on explained variance follows noise
            state = _chol_state_init(Rp)
            _chol_state_add(state, Rp, 0)
            ev = _explained_variance(Rp, state["L"], state["d"], 1, block,
                                     np.arange(1, N))
            ev_argmax.add(int(block[np.argmax(ev)]))
        assert len(ev_argmax) > 1

    @pytest.mark.parametrize("c", [1e-10, 1e-8, 1e-3, 1e3])
    def test_global_scale_invariance(self, c):
        """Tolerances follow the variance scale of Σ: c·Σ selects as Σ."""
        # Distinct diagonal: no ties at any scale, plain argmax order
        lam = np.array([0.8, 0.8, 0.8, 0.99])
        v = np.array([1.0, 0.97, 0.94, 0.91])
        S = _loading_corr(lam[:, None]) * np.sqrt(np.outer(v, v))
        res = greedy_entropy(c * S, 4)
        assert res["selected"] == _argmax_greedy_entropy(c * S, 4)[0]
        assert res["selected"] == greedy_entropy(S, 4)["selected"]
        assert res["tie_breaks"] == []
        # Unit-diagonal correlation scaled by c: same step-1 tie and order
        R = _factor_corr(N=20, seed=5)
        res = greedy_entropy(c * R, 20)
        assert res["selected"] == greedy_entropy(R, 20)["selected"]
        assert res["tie_breaks"][0] == (0, 20)

    def test_tolerance_scale(self):
        """Diagonal gaps well above TIE_RTOL are not treated as ties."""
        R = _hub_corr(N=10, hub=6)
        R[2, 2] += 100 * TIE_RTOL
        res = greedy_entropy(R, 1)
        assert res["selected"] == [2]
        assert res["tie_breaks"] == []


# ===================================================================
# Budgeted greedy (eval_cost_budget)
# ===================================================================

class TestBudgetedGreedyTieBreak:
    @pytest.mark.parametrize("unit_cost", [100.0, 1e6])
    @pytest.mark.parametrize("n_picks", [1, 5])
    def test_equal_costs_deterministic(self, n_picks, unit_cost):
        """(e) Equal costs: deterministic under diagonal perturbation."""
        R = _factor_corr(N=30, seed=6)
        N = R.shape[0]
        costs = np.full(N, unit_cost)
        budget = unit_cost * n_picks
        ref = budgeted_greedy(R, costs, budget, "entropy")
        assert len(ref) == n_picks
        assert ref[0] == int(np.argmax(_hub_score(R)))
        # Equal costs: gain/cost order == conditional-variance order
        assert ref == greedy_entropy(R, n_picks)["selected"]
        rng = np.random.default_rng(400 + n_picks)
        for _ in range(50):
            Rp = _perturb_diag(R, rng)
            assert budgeted_greedy(Rp, costs, budget, "entropy") == ref

    def test_tied_cost_groups_deterministic(self):
        """Mixed costs with many exact cost ties (as on MMLU)."""
        R = _factor_corr(N=30, seed=7)
        N = R.shape[0]
        rng = np.random.default_rng(8)
        costs = rng.choice([100.0, 100.0, 100.0, 150.0, 300.0], N)
        budget = 0.2 * costs.sum()
        ref = budgeted_greedy(R, costs, budget, "entropy")
        # Step 1: ratios tie exactly within the cheapest cost group
        cheapest = np.flatnonzero(costs == costs.min())
        assert ref[0] == int(cheapest[np.argmax(_hub_score(R)[cheapest])])
        for _ in range(50):
            Rp = _perturb_diag(R, rng)
            assert budgeted_greedy(Rp, costs, budget, "entropy") == ref

    def test_budget_below_min_cost(self):
        R = _factor_corr(N=10, seed=9)
        assert budgeted_greedy(R, np.full(10, 100.0), 50.0, "entropy") == []

    @pytest.mark.parametrize("unit_cost", [1.0, 1e4, 1e6, 1e9])
    def test_equal_costs_match_greedy_entropy_any_units(self, unit_cost):
        """Equal costs reproduce greedy_entropy whatever the cost units."""
        for seed in range(20):
            R = _factor_corr(N=30, seed=seed)
            costs = np.full(30, unit_cost)
            assert (budgeted_greedy(R, costs, 10 * unit_cost, "entropy")
                    == greedy_entropy(R, 10)["selected"])

    @pytest.mark.parametrize("unit", [1.0, 1e-3])
    def test_ratio_ties_independent_of_cost_units(self, unit):
        """A 2e-4 relative ratio gap is not a tie, in any cost units."""
        R = np.array([[1.0, 0.1, 0.0, 0.0],
                      [0.1, 1.0, 0.6, 0.6],
                      [0.0, 0.6, 1.0, 0.3],
                      [0.0, 0.6, 0.3, 1.0]])
        costs = np.array([1.0e6, 1.0002e6, 9e6, 9e6]) * unit
        # 1 is the hub, but 0 has the strictly larger gain/cost ratio
        assert int(np.argmax(_hub_score(R))) == 1
        assert budgeted_greedy(R, costs, 1.5e6 * unit, "entropy") == [0]

    def test_costs_as_list(self):
        R = _factor_corr(N=30, seed=7)
        costs = np.random.default_rng(8).choice([100.0, 150.0, 300.0], 30)
        budget = 0.2 * costs.sum()
        for objective in ["entropy", "mi"]:
            assert (budgeted_greedy(R, list(costs), budget, objective)
                    == budgeted_greedy(R, costs, budget, objective))

    def test_mixed_cost_ratio_tie_weighted_by_cost(self):
        """Ratios tied across different costs: explained variance per cost.

        Gains ½log(Σ_jj/D_FLOOR) are 2.30 (Σ_00 = 0.01) and 4.61 (Σ_11 = 1)
        at costs 1 and 2, so both ratios equal ½log 100.  Benchmark 1
        explains more variance (1.98 vs 1.54) but less per unit cost.
        """
        C = np.array([[1.0, 0.5, 0.8, 0.8],
                      [0.5, 1.0, 0.7, 0.7],
                      [0.8, 0.7, 1.0, 0.7],
                      [0.8, 0.7, 0.7, 1.0]])
        assert np.linalg.eigvalsh(C).min() > 0
        sd = np.sqrt([0.01, 1.0, 1.0, 1.0])
        Sigma = C * np.outer(sd, sd)
        assert budgeted_greedy(Sigma, [1.0, 2.0, 4.0, 4.0], 3.0,
                               "entropy") == [0, 1]

    def test_singleton_needs_margin_to_replace_greedy(self):
        """A value-tied singleton must not replace the greedy set by noise."""
        R = _factor_corr(N=12, seed=12)
        hub_score = _hub_score(R)
        hub = int(np.argmax(hub_score))
        h2 = int(np.argsort(-hub_score)[1])
        costs = np.full(12, 100.0)
        costs[hub] = 150.0  # greedy takes the best cheaper benchmark h2
        rng = np.random.default_rng(13)
        flips = 0
        for _ in range(50):
            Rp = _perturb_diag(R, rng, scale=1e-13)
            assert budgeted_greedy(Rp, costs, 150.0, "entropy") == [h2]
            # Control: the singleton (hub) often wins by rounding alone
            flips += (np.log(Rp[hub, hub] / D_FLOOR)
                      > np.log(Rp[h2, h2] / D_FLOOR))
        assert flips > 0

    def test_singleton_tie_broken_by_explained_variance(self):
        """Fallback singleton among value-tied benchmarks is the hub."""
        cheap = 0.01 * np.array([[1.0, 0.99], [0.99, 1.0]])
        R = _hub_corr(N=8, hub=5)
        Sigma = np.zeros((10, 10))
        Sigma[:2, :2], Sigma[2:, 2:] = cheap, R
        costs = np.array([1.0, 1.0] + [10.0] * 8)
        # Greedy takes the two cheap, nearly redundant benchmarks (value
        # 2.30 + 0.34); a single unit-variance one (4.61) beats it.
        assert budgeted_greedy(Sigma, costs, 10.0, "entropy") == [7]
        rng = np.random.default_rng(14)
        for _ in range(20):
            Sp = _perturb_diag(Sigma, rng)
            assert budgeted_greedy(Sp, costs, 10.0, "entropy") == [7]


# ===================================================================
# MI selection must be numerically unchanged
# ===================================================================

class TestMIUnchanged:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_greedy_mi_matches_frozen(self, seed):
        """(f) greedy_mi output identical to the pre-change algorithm."""
        R = _factor_corr(N=25, seed=seed)
        N = R.shape[0]
        new = greedy_mi(R, N)
        old = _frozen_greedy_mi(R, N)
        assert new["selected"] == old["selected"]
        for key in ["d", "log_dets", "residual_variance", "gains"]:
            assert np.array_equal(new[key], old[key]), key

    @pytest.mark.parametrize("frac", [0.02, 0.1, 0.3])
    def test_budgeted_mi_matches_frozen(self, frac):
        R = _factor_corr(N=25, seed=10)
        N = R.shape[0]
        costs = np.random.default_rng(11).choice([100.0, 150.0, 400.0], N)
        budget = frac * costs.sum()
        assert (budgeted_greedy(R, costs, budget, "mi")
                == _frozen_budgeted_mi(R, costs, budget))

    def test_budgeted_mi_singleton_strict(self):
        """MI fallback keeps the strict comparison (no tie margin).

        Greedy takes 0 (cheaper); singleton 1 is better by ~7e-12 only.
        """
        R = np.array([[1.0, 0.0, 0.5],
                      [0.0, 1.0, 0.5 + 1e-11],
                      [0.5, 0.5 + 1e-11, 1.0]])
        costs = np.array([1.0, 1.0 + 1e-6, 100.0])
        assert _frozen_budgeted_mi(R, costs, 1.5) == [1]
        assert budgeted_greedy(R, costs, 1.5, "mi") == [1]


# ===================================================================
# Cost budget: historical random draws (knap_entropy_rf, whose draw pool
# was tested here, was dropped: protocol rule D3)
# ===================================================================

class TestCostBudgetRandom:
    def test_legacy_random_reproduces_historical_draws(self):
        """--legacy-random: the b-th rng.permutation(N) of
        default_rng((SEED + 7, fold)) for the b-th budget, as the script
        drew them before the matched design (rep 0)."""
        from cv_split import cv_splits
        from eval_cost_budget import legacy_random_orders
        N, n_budgets = 57, 7
        for s in cv_splits(120, 10, 2, 0.1, 42):
            orders = legacy_random_orders(N, s, n_budgets, None)
            seed = (49, s.fold) if s.rep == 0 else (49, s.rep, s.fold)
            rng = np.random.default_rng(seed)
            assert orders == [rng.permutation(N).tolist()
                              for _ in range(n_budgets)]
        el = np.random.default_rng(5).random(N) < 0.7
        s = next(iter(cv_splits(120, 10, 1, 0.1, 42)))
        rng = np.random.default_rng((49, 0))
        assert legacy_random_orders(N, s, 3, el) == [
            [int(j) for j in rng.permutation(N) if el[j]] for _ in range(3)]

    def test_legacy_random_fills_per_budget(self):
        from cv_split import cv_splits
        from eval_cost_budget import (legacy_random_orders, order_fill,
                                      select_for_budget, split_orders)
        R = _factor_corr(N=25, seed=4)
        N = R.shape[0]
        costs = np.random.default_rng(6).choice([100.0, 150.0, 400.0], N)
        budgets = [0.05 * costs.sum(), 0.2 * costs.sum()]
        s = next(iter(cv_splits(100, 10, 1, 0.1, 42)))
        orders = split_orders(R, costs, s, None, len(budgets), True)
        legacy = legacy_random_orders(N, s, len(budgets), None)
        for b, budget in enumerate(budgets):
            res = select_for_budget("random", R, costs, budget, orders, s,
                                    None, b)
            assert res["selected"] == order_fill(legacy[b], costs, budget)
        # default: one shared random_select permutation for every budget
        default = split_orders(R, costs, s, None)
        shared = default["random"]["selected"]
        for b, budget in enumerate(budgets):
            res = select_for_budget("random", R, costs, budget, default, s,
                                    None, b)
            assert res["selected"] == order_fill(shared, costs, budget)

    def test_budgeted_select_has_no_random_first_pick(self):
        """The first_seed option went with knap_entropy_rf (D3)."""
        from eval_cost_budget import METHODS, budgeted_select
        R = _factor_corr(N=10, seed=1)
        res = budgeted_select(R, np.ones(10), 3.0, "entropy")
        assert "first" not in res and "first_pool" not in res
        with pytest.raises(TypeError):
            budgeted_select(R, np.ones(10), 3.0, "entropy", first_seed=1)
        assert not any("entropy_rf" in m for m in METHODS)
