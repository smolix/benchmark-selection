"""Tests for analyze_objectives.py (synthetic data, < 60 s).

Covers the block formulas of the objectives (trace, entropy, MI via
Jacobi's identity), the population-R^2 identity R2_pop(A) = (F_tr(A) - k) /
(N - k) against its definition and a Monte Carlo estimate, the masked
expected R^2 against impute_and_evaluate (cells, skipped models, value in
expectation), the k = 1 characterization of MI and trace, the brute-force
maximizers, the trace-gain path, MI's non-monotonicity (toy R = I but
r_01 = 0.95; candidate vs chosen gains; the gain landscape against
greedy_mi and the block formula), the exact submodularity ratio against
its definition, Das and Kempe's guarantee and the modular case, the
spectral bounds, and an end-to-end run on a small synthetic dataset: no
leakage of the validation rows into the train-side quantities, common
random numbers of the random sets, letters-only macros, the new summary
statistics (ratio ranges, gap decomposition, oracle effect counts)
against the per-split arrays, input provenance and the --check-logs data
check, the full-oracle file tag and macro prefix, and the output files.
Facility location (2026-10-19): F_FL and its path gains against
facility_location_select, its brute-force optimum and the 1 - 1/e bound,
facloc's selections against facility_location_select on the split's
training estimate / the oracle R, the facility-location section's arrays
(overlap, path gains, gain landscape and the regret of each objective's
pick) against recomputation, _greedy_choice against the selectors' picks
(including an exact facloc gain tie broken by explained variance), and the
invariance of the extension (a run without facloc agrees with the full
run on every key, macro, paired row and table entry it has).  --workers:
a run whose splits are evaluated in two spawned processes has the serial
run's arrays, paired rows, macros and printed lines (apart from times).
"""

from __future__ import annotations

import itertools
import json
import re

import numpy as np
import pytest

import analyze_objectives as ao
from baseline_select import facility_location_select, greedy_trace_select
from eval_entropy_vs_mi import impute_and_evaluate
from greedy_select import greedy_entropy, greedy_mi


def _fl_value(R, A):
    """Brute-force F_FL(A) = sum_j max_{i in A} max(R_ij, 0), s_jj = 1."""
    S = np.maximum(R, 0.0)
    np.fill_diagonal(S, 1.0)
    return float(np.sum(np.max(S[list(A)], axis=0)))


def _factor_corr(N=20, n_factors=3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, n_factors))
    S = W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))
    s = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return R


def _logdet(M):
    sign, ld = np.linalg.slogdet(M)
    assert sign > 0
    return ld


# ---------------------------------------------------------------------------
# Block formulas and the population identity
# ---------------------------------------------------------------------------

def test_block_formulas_match_definitions():
    R = _factor_corr(20, seed=1)
    sp = ao.Spectral(R)
    N = R.shape[0]
    rng = np.random.default_rng(0)
    for k in (1, 2, 4, 7):
        for _ in range(5):
            A = sorted(rng.choice(N, k, replace=False).tolist())
            C = [j for j in range(N) if j not in A]
            tr, ld, mi = ao.set_objectives(sp, np.array([A]))
            RA = R[np.ix_(A, A)]
            direct_tr = np.trace(R[:, A] @ np.linalg.solve(RA, R[A, :]))
            assert tr[0] == pytest.approx(direct_tr, rel=1e-10)
            assert ld[0] == pytest.approx(_logdet(RA), abs=1e-10)
            direct_mi = 0.5 * (_logdet(RA) + _logdet(R[np.ix_(C, C)])
                               - _logdet(R))
            assert mi[0] == pytest.approx(direct_mi, abs=1e-8)
            # R2_pop = explained variance of the complement / (N - k)
            expl = np.trace(R[np.ix_(C, A)] @ np.linalg.solve(RA, R[np.ix_(A, C)]))
            assert ao.pop_r2(sp, A) == pytest.approx(expl / (N - k), rel=1e-10)


def test_greedy_mi_value_equals_block_mi():
    """greedy_mi's I(X_A; X_rest) equals the Jacobi block formula."""
    R = _factor_corr(16, seed=2)
    sp = ao.Spectral(R)
    res = greedy_mi(R, 5)
    for k in range(1, 6):
        _, _, mi = ao.set_objectives(sp, np.array([res["selected"][:k]]))
        assert mi[0] == pytest.approx(res["mi_values"][k - 1], abs=1e-8)


def test_pop_r2_matches_monte_carlo():
    R = _factor_corr(12, seed=3)
    sp = ao.Spectral(R)
    rng = np.random.default_rng(1)
    X = rng.multivariate_normal(np.zeros(12), R, size=60000)
    A = [0, 5, 9]
    C = [j for j in range(12) if j not in A]
    W = R[np.ix_(C, A)] @ np.linalg.inv(R[np.ix_(A, A)])
    res = X[:, C] - X[:, A] @ W.T
    mc = 1.0 - np.sum(res ** 2) / np.sum(X[:, C] ** 2)
    assert mc == pytest.approx(ao.pop_r2(sp, A), abs=0.01)


def test_expected_r2_full_mask_is_pop_r2():
    R = _factor_corr(15, seed=4)
    sp = ao.Spectral(R)
    O = np.ones((7, 15))
    A = [2, 3, 11]
    r2, cov_m, cov_c = ao.expected_r2(O, A, R, {"t": R}, ridge=0.0)
    assert r2["t"] == pytest.approx(ao.pop_r2(sp, A), rel=1e-10)
    assert cov_m == 1.0 and cov_c == 1.0


def test_expected_r2_matches_impute_and_evaluate():
    """Same scored cells and skipped models as impute_and_evaluate, and the
    same value in expectation (large synthetic validation set)."""
    N = 10
    R = _factor_corr(N, seed=5)
    rng = np.random.default_rng(2)
    M = 40000
    X = rng.multivariate_normal(np.zeros(N), R, size=M)
    O = (rng.uniform(size=(M, N)) < 0.7).astype(float)
    A = [1, 4]
    B = X * O
    realized = impute_and_evaluate(B, O, A, R, np.zeros(N), np.ones(N),
                                   N)["r2"]
    exp, cov_m, cov_c = ao.expected_r2(O, A, R, {"t": R})
    assert exp["t"] == pytest.approx(realized, abs=0.01)
    # models without any selected benchmark observed are skipped
    has_A = O[:, A].sum(axis=1) > 0
    unsel = [j for j in range(N) if j not in A]
    has_U = O[:, unsel].sum(axis=1) > 0
    assert cov_m == pytest.approx(np.mean(has_A & has_U))
    scored = np.sum(O[has_A & has_U][:, unsel])
    assert cov_c == pytest.approx(scored / O[:, unsel].sum())


# ---------------------------------------------------------------------------
# k = 1: MI picks the best-predicted benchmark, trace the best predictor
# ---------------------------------------------------------------------------

def _duplicate_pair_corr():
    """A one-factor block of 10 benchmarks (loadings 0.75) plus a pair of
    near-duplicates (r = 0.98) uncorrelated with the block."""
    N = 12
    lam = np.full(10, 0.75)
    R = np.eye(N)
    R[:10, :10] = np.outer(lam, lam)
    np.fill_diagonal(R, 1.0)
    R[10, 11] = R[11, 10] = 0.98
    return R


def test_first_picks_mi_vs_trace():
    R = _duplicate_pair_corr()
    sp = ao.Spectral(R)
    mi_pick = greedy_mi(R, 1)["selected"][0]
    tr_pick = greedy_trace_select(R, 1)["selected"][0]
    assert mi_pick in (10, 11)                      # near-duplicate
    assert tr_pick < 10                             # factor block
    # MI's k = 1 score is -1/2 log(1 - R^2_{j|rest}) = 1/2 log P_jj
    P = np.linalg.inv(R)
    assert mi_pick == int(np.argmax(np.diag(P)))
    rest_r2 = 1 - 1 / np.diag(P)
    assert rest_r2[mi_pick] > rest_r2[tr_pick]
    # ...but it predicts the rest far worse
    assert ao.pop_r2(sp, [mi_pick]) < 0.5 * ao.pop_r2(sp, [tr_pick])
    # trace's first pick maximizes (R^2)_jj = sum_i R_ij^2
    assert tr_pick == int(np.argmax(np.diag(R @ R)))


# ---------------------------------------------------------------------------
# Brute force, trace path, redundancy
# ---------------------------------------------------------------------------

def test_brute_force_is_exact_and_dominates_greedy():
    R = _factor_corr(11, seed=6)
    sp = ao.Spectral(R)
    pool = np.arange(11)
    assert ao.BRUTE_OBJECTIVES == ["trace", "entropy", "mi", "facloc"]
    for k in (1, 2, 3):
        best = ao.brute_force(sp, pool, k, chunk=17)
        sets = np.array(list(itertools.combinations(range(11), k)))
        vals = ao._brute_values(sp, sets)
        tr, ld, mi = ao.set_objectives(sp, sets)
        for key, v in (("trace", tr), ("entropy", ld), ("mi", mi)):
            np.testing.assert_array_equal(vals[key], v)
        np.testing.assert_allclose(
            vals["facloc"], [_fl_value(R, A) for A in sets], atol=1e-12)
        for obj in ao.BRUTE_OBJECTIVES:
            assert best[obj][0] == pytest.approx(np.max(vals[obj]))
        g = {"trace": greedy_trace_select(R, k)["selected"],
             "entropy": greedy_entropy(R, k)["selected"],
             "mi": greedy_mi(R, k)["selected"],
             "facloc": facility_location_select(R, k)["selected"]}
        for obj in ao.BRUTE_OBJECTIVES:
            gv = ao._brute_values(sp, np.array([g[obj]]))[obj][0]
            assert gv <= best[obj][0] + 1e-10
        # facility location is monotone submodular: greedy within 1 - 1/e
        gv = ao._brute_values(sp, np.array([g["facloc"]]))["facloc"][0]
        assert gv >= (1 - 1 / np.e) * best["facloc"][0] - 1e-12
    # restricted pool: the optimum uses pool members only
    best = ao.brute_force(sp, np.array([0, 3, 5, 7]), 2)
    assert set(best["trace"][1]) <= {0, 3, 5, 7}


def test_trace_path_gains_reproduce_greedy_trace():
    R = _factor_corr(14, seed=7)
    el = np.ones(14, dtype=bool)
    el[[2, 9]] = False
    sel = greedy_trace_select(R, 6, eligible=el)["selected"]
    G = ao.trace_path_gains(R, sel, el)
    assert np.allclose(G[0][el], np.sum(R ** 2, axis=0)[el])
    assert np.all(np.isnan(G[:, ~el]))
    for t, j in enumerate(sel):
        assert int(np.nanargmax(G[t])) == j
        assert np.all(np.isnan(G[t + 1:, j]))


def test_facloc_objective_and_path_gains_reproduce_the_selector():
    """facloc_objective of facility_location_select's prefixes is its
    facloc_values and facloc_path_gains at its picks its gains, bit for
    bit; at t = 0 the gains are the singleton scores F_FL({j})."""
    R = _factor_corr(14, seed=9)
    assert R.min() < -0.1               # some correlations are clipped
    sp = ao.Spectral(R)
    el = np.ones(14, dtype=bool)
    el[[3, 10]] = False
    res = facility_location_select(R, 8, eligible=el)
    sel = res["selected"]
    vals = np.array([ao.facloc_objective(sp, [sel[:t]])[0]
                     for t in range(1, 9)])
    np.testing.assert_array_equal(vals, res["facloc_values"])
    G = ao.facloc_path_gains(sp.fl, sel, el)
    np.testing.assert_array_equal([G[t, j] for t, j in enumerate(sel)],
                                  res["gains"])
    assert np.all(np.isnan(G[:, ~el]))
    np.testing.assert_allclose(G[0][el], sp.fl_single[el], rtol=0,
                               atol=1e-12)
    np.testing.assert_allclose(sp.fl_single,
                               [_fl_value(R, [j]) for j in range(14)],
                               atol=1e-12)
    for t, j in enumerate(sel):
        assert np.all(np.isnan(G[t + 1:, j]))


def _fl_tie_corr():
    """R = I but r_01 = r_23 = 0.3, r_34 = -0.5: candidates 0-3 tie
    exactly on the first facility-location gain 1.3 (r_34 is clipped), and
    3 explains the most variance (1 + 0.09 + 0.25), so facloc's first pick
    is 3, not the lowest tied index 0."""
    R = np.eye(6)
    for i, j, r in ((0, 1, 0.3), (2, 3, 0.3), (3, 4, -0.5)):
        R[i, j] = R[j, i] = r
    return R


@pytest.mark.parametrize("make_R", [lambda: _factor_corr(14, seed=9),
                                    _fl_tie_corr])
def test_greedy_choice_is_the_selectors_pick(make_R):
    """_greedy_choice (the pick the gain-landscape regret scores for O !=
    P) applies each selector's near-tie rule: on its own greedy path it
    returns facility_location_select's / greedy_trace_select's pick at
    every state, also where facloc breaks an exact gain tie by explained
    variance (the lowest tied index would be another candidate)."""
    R = make_R()
    N = R.shape[0]
    el = np.ones(N, dtype=bool)
    if N > 6:
        el[[3, 10]] = False
    res = facility_location_select(R, N - 4, eligible=el)
    tres = greedy_trace_select(R, N - 4, eligible=el)
    paths = {"facloc": (res["selected"],
                        ao.facloc_path_gains(ao.Spectral(R).fl,
                                             res["selected"], el)),
             "trace": (tres["selected"],
                       ao.trace_path_gains(R, tres["selected"], el))}
    for O, (path, G) in paths.items():
        for t, j in enumerate(path):
            cand = np.flatnonzero(el & ~np.isin(np.arange(N), path[:t]))
            top = ao._greedy_choice(O, R, path[:t], cand, G[t, cand])
            assert cand[top] == j, (O, t)
    if N == 6:
        assert res["selected"][0] == 3 and res["tie_breaks"][0] == (0, 4)
        g0 = paths["facloc"][1][0]
        assert int(np.flatnonzero(g0 == np.nanmax(g0))[0]) == 0
    assert ao._greedy_choice("facloc", R, [], np.arange(2),
                             np.full(2, np.nan)) is None


def test_path_terms_split_mi_gain():
    """MI gain = entropy novelty 1/2 log sigma^2_{j|A} + bonus 1/2 log P_jj
    (complement precision); step 1 on a correlation matrix: novelty 0,
    bonus -1/2 log(1 - R^2_{j|rest})."""
    R = _factor_corr(14, seed=8)
    res = {"mi": greedy_mi(R, 4), "entropy": greedy_entropy(R, 4),
           "trace": greedy_trace_select(R, 4)}
    pt = ao.path_terms(res, 4)
    j = res["mi"]["selected"][0]
    assert pt["novelty_mi"][0] == pytest.approx(0.0, abs=1e-12)
    assert pt["bonus_mi"][0] == pytest.approx(
        0.5 * np.log(np.linalg.inv(R)[j, j]), rel=1e-8)
    np.testing.assert_allclose(pt["novelty_mi"] + pt["bonus_mi"],
                               pt["gain_mi"])
    # step t: bonus = 1/2 log [ (R_CC)^{-1} ]_jj over the complement C
    A = res["mi"]["selected"][:2]
    j = res["mi"]["selected"][2]
    C = [i for i in range(14) if i not in A]
    P_C = np.linalg.inv(R[np.ix_(C, C)])
    assert pt["bonus_mi"][2] == pytest.approx(
        0.5 * np.log(P_C[C.index(j), C.index(j)]), rel=1e-8)


# ---------------------------------------------------------------------------
# MI monotonicity: chosen gains vs every candidate's gain
# ---------------------------------------------------------------------------

def _toy_nonmonotone_corr(N=4, r=0.95):
    R = np.eye(N)
    R[0, 1] = R[1, 0] = r
    return R


def test_mi_is_not_monotone_toy():
    """R = I but r_01 = 0.95: MI({0}) = 1.164 > MI({0, 1}) = 0."""
    R = _toy_nonmonotone_corr()
    sp = ao.Spectral(R)
    _, _, mi = ao.set_objectives(sp, np.array([[0]]))
    _, _, mi2 = ao.set_objectives(sp, np.array([[0, 1]]))
    assert mi[0] == pytest.approx(-0.5 * np.log(1 - 0.95 ** 2), abs=1e-12)
    assert mi[0] == pytest.approx(1.164, abs=1e-3)
    assert mi2[0] == pytest.approx(0.0, abs=1e-12)
    G = ao.mi_pair_gains(R, np.ones(4, dtype=bool))
    assert G[0, 1] == pytest.approx(mi2[0] - mi[0], abs=1e-10)
    assert np.all(np.isnan(np.diag(G)))


def test_mi_gain_landscape_reproduces_greedy_mi():
    """Row t of the landscape: max = greedy MI's gain at its pick t + 1,
    every entry = the block-formula difference MI(A + j) - MI(A); row 1 =
    the pair gains of the first pick; ineligible columns NaN."""
    R = _factor_corr(13, n_factors=3, seed=11)
    sp = ao.Spectral(R)
    el = np.ones(13, dtype=bool)
    el[[3, 8]] = False
    res = greedy_mi(R, 5, eligible=el)
    G = ao.mi_gain_landscape(R, res["selected"], el)
    for t, j in enumerate(res["selected"]):
        assert int(np.nanargmax(G[t])) == j
        assert G[t, j] == pytest.approx(res["gains"][t], abs=1e-9)
        A = res["selected"][:t]
        mi_A = (ao.set_objectives(sp, np.array([A]))[2][0] if A else 0.0)
        for c in np.flatnonzero(~np.isnan(G[t]))[:4]:
            mi_Ac = ao.set_objectives(sp, np.array([A + [int(c)]]))[2][0]
            assert G[t, c] == pytest.approx(mi_Ac - mi_A, abs=1e-8)
        assert np.all(np.isnan(G[t, ~el]))
        assert np.all(np.isnan(G[t, A]))
    pair = ao.mi_pair_gains(R, el)
    a = res["selected"][0]
    np.testing.assert_allclose(G[1], pair[a], atol=1e-9, equal_nan=True)


def _all_results(R, k, el=None):
    return {"trace": greedy_trace_select(R, k, eligible=el),
            "entropy": greedy_entropy(R, k, eligible=el),
            "mi": greedy_mi(R, k, eligible=el)}


def test_theory_counts_candidate_not_only_chosen_negative_gains():
    """Greedy MI's own gains are >= 0 on the toy, yet adding 1 to {0}
    lowers MI: mi_chosen_neg_gains = 0 but mi_min_gain < 0."""
    R = _toy_nonmonotone_corr()
    el = np.ones(4, dtype=bool)
    res = _all_results(R, 3, el)
    assert res["mi"]["selected"][0] in (0, 1)
    assert np.all(np.asarray(res["mi"]["gains"]) >= -1e-12)
    th = ao.theory_diagnostics(R, res, el, gamma_k=2)
    assert th["mi_chosen_neg_gains"] == 0.0
    loss = 0.5 * np.log(1 - 0.95 ** 2)                  # -1.164
    assert th["mi_min_gain"] == pytest.approx(loss, abs=1e-9)
    assert th["mi_min_pair_gain"] == pytest.approx(loss, abs=1e-9)
    assert th["mi_first_neg_size"] == 1.0
    assert th["mi_nonmonotone"] == 1.0
    assert th["mi_eps_lb"] == pytest.approx(-loss, abs=1e-9)
    assert 0 < th["mi_neg_gain_frac"] < 1
    # path term: smallest candidate gain per step <= the chosen gain
    pt = ao.path_terms(res, 3, ao.mi_gain_landscape(R, res["mi"]["selected"],
                                                    el))
    assert np.all(pt["min_gain_mi"] <= pt["gain_mi"] + 1e-12)
    assert pt["min_gain_mi"][1] == pytest.approx(loss, abs=1e-9)


# ---------------------------------------------------------------------------
# Submodularity ratio of the trace objective (Das and Kempe)
# ---------------------------------------------------------------------------

def _gamma_by_definition(R, U, pool, k):
    def F(A):
        A = list(A)
        if not A:
            return 0.0
        return float(np.trace(R[:, A] @ np.linalg.solve(R[np.ix_(A, A)],
                                                         R[A, :])))
    gamma = 1.0
    for r in range(len(U) + 1):
        for L in itertools.combinations(U, r):
            rest = [j for j in pool if j not in L]
            fL = F(L)
            for s in range(2, k + 1):
                for S in itertools.combinations(rest, s):
                    den = F(list(L) + list(S)) - fL
                    if den <= 1e-9 * R.shape[0]:
                        continue
                    num = sum(F(list(L) + [x]) - fL for x in S)
                    gamma = min(gamma, num / den)
    return gamma


def test_submodularity_ratio_matches_definition_and_bounds_greedy():
    for seed in (12, 13):
        R = _factor_corr(8, n_factors=2, seed=seed)
        sp = ao.Spectral(R)
        pool = [0, 1, 2, 4, 5, 6, 7]
        for k in (2, 3):
            U = greedy_trace_select(R, k, eligible=np.isin(np.arange(8),
                                                           pool))["selected"]
            g = ao.submodularity_ratio(sp, U, np.array(pool), k, chunk=5)
            assert g == pytest.approx(_gamma_by_definition(R, U, pool, k),
                                      abs=1e-10)
            assert 0 < g <= 1.0
            # Das-Kempe: F(greedy_k) >= (1 - e^{-gamma_{U,k}}) F(OPT_k)
            opt = ao.brute_force(sp, np.array(pool), k)["trace"][0]
            f_g = ao.trace_objective(sp, np.array([U]))[0]
            assert f_g >= (1 - np.exp(-g)) * opt - 1e-10
    # modular F (R = I): gamma = 1
    sp = ao.Spectral(np.eye(6))
    assert ao.submodularity_ratio(sp, [0, 1], np.arange(6), 2) == \
        pytest.approx(1.0)


def test_theory_spectral_bounds():
    R = _factor_corr(10, seed=14)
    el = np.ones(10, dtype=bool)
    el[4] = False
    th = ao.theory_diagnostics(R, _all_results(R, 4, el), el, gamma_k=2)
    lam = np.linalg.eigvalsh(R)
    assert th["trace_spectral_bound"] == pytest.approx(1 - np.exp(-lam[0]))
    off = np.abs(R[np.ix_(el, el)])
    np.fill_diagonal(off, 0)
    assert th["trace_sparse_eig_ub"] == pytest.approx(1 - off.max())
    # the 2-sparse eigenvalue is at most 1 - max |r| (the pair's 2x2 block)
    assert lam[0] <= th["trace_sparse_eig_ub"] + 1e-12
    assert th["trace_gamma_bound"] == pytest.approx(
        1 - np.exp(-th["trace_gamma"]))
    assert th["trace_gamma"] <= 1.0


def test_observed_picks():
    O = np.array([[1, 1, 0, 1],      # 2 picks seen, scored
                  [0, 0, 1, 1],      # none seen: not scored
                  [1, 0, 0, 0],      # 1 pick seen, nothing to score
                  [0, 1, 1, 0]])     # 1 pick seen, scored
    k_eff, k_obs = ao.observed_picks(O, [0, 1])
    assert k_eff == pytest.approx(1.5)
    assert k_obs == pytest.approx((2 + 0 + 1 + 1) / 4)


def test_redundancy_measures():
    R = np.eye(4)
    assert ao.redundancy(R, [0, 1, 2])[2] == pytest.approx(1.0)
    R = np.full((4, 4), 0.5)
    np.fill_diagonal(R, 1.0)
    m, x, eff = ao.redundancy(R, [0, 1, 2])
    assert m == pytest.approx(0.5) and x == pytest.approx(0.5)
    assert eff == pytest.approx(3 / (3 + 6 * 0.25))
    assert np.isnan(ao.redundancy(R, [1])[0])


def test_matched_oracle_estimator():
    """M_ref with the same size rules as M reproduces estimate_corr(B, O);
    M_ref < N adds em_cov's auto shrinkage alpha = (N - M_ref) / N."""
    from eval_entropy_vs_mi import estimate_corr
    rng = np.random.default_rng(3)
    M, N = 60, 8
    R = _factor_corr(N, n_factors=2, seed=9)
    B = rng.multivariate_normal(np.zeros(N), R, size=M)
    O = (rng.uniform(size=(M, N)) < 0.8).astype(float)
    B = B * O
    full = estimate_corr(B, O)[0]
    np.testing.assert_array_equal(ao.estimate_corr_oracle(B, O, None), full)
    np.testing.assert_array_equal(ao.estimate_corr_oracle(B, O, 40), full)
    shrunk = ao.estimate_corr_oracle(B, O, 6)          # alpha = 2 / 8
    off = ~np.eye(N, dtype=bool)
    assert np.mean(np.abs(shrunk[off])) < np.mean(np.abs(full[off]))
    assert np.allclose(np.diag(shrunk), 1.0)


# ---------------------------------------------------------------------------
# End to end on a small synthetic dataset
# ---------------------------------------------------------------------------

def _write_dataset(path, seed=0, M=70, N=9, perturb_rows=None):
    rng = np.random.default_rng(seed)
    R = _factor_corr(N, n_factors=2, seed=seed)
    X = rng.multivariate_normal(np.zeros(N), R, size=M) * 0.1 + 0.5
    O = (rng.uniform(size=(M, N)) < 0.85).astype(np.float64)
    O[:, 0] = 1.0
    if perturb_rows is not None:
        X[perturb_rows] = rng.uniform(0, 1, size=(len(perturb_rows), N))
    B = X * O
    np.savez(path / "synth.matrix.npz", B=B, O=O,
             model_names=np.array([f"m{i}" for i in range(M)]),
             benchmark_names=np.array([f"s{j % 3}/b{j}" for j in range(N)]))
    (path / "synth.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [None, None]}}))


@pytest.fixture
def synth_dirs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    logs = tmp_path / "logs"
    figs = tmp_path / "figures"
    logs.mkdir()
    figs.mkdir()
    monkeypatch.setattr(ao, "DATA", data)
    monkeypatch.setattr(ao, "LOGS", logs)
    monkeypatch.setattr(ao, "OUT", figs)
    return data, logs, figs


KW = dict(k_max=4, n_folds=3, n_repeats=1, n_random_sets=4, random_ks=[3],
          brute_k=2)


def test_end_to_end_no_leakage_and_outputs(synth_dirs):
    data, logs, figs = synth_dirs
    _write_dataset(data)
    arr = ao.run_dataset("synth", **KW)
    assert arr["rep"].tolist() == [0, 0, 0]
    assert arr["split_id"].tolist() == [0, 1, 2]
    for key in ("fold", "n_eligible", "n_train", "n_val"):
        assert arr[key].shape == (3,)

    # Perturb the validation models of split 0: every train-side
    # (in-sample) quantity of split 0 is unchanged
    from cv_split import cv_splits
    s0 = next(cv_splits(70, 3, 1, ao.HOLDOUT_FRAC, ao.SEED))
    _write_dataset(data, perturb_rows=s0.val_idx)
    arr2 = ao.run_dataset("synth", **KW)
    train_side = [f"{m}_{q}" for m in ao.METHODS
                  for q in ("selected", "pop_r2_tr", "logdet_tr", "mi_tr",
                            "hub_tr", "mean_abs_r_tr", "eff_rank_tr")]
    train_side += ["single_pop_r2_tr", "single_mi_tr", "single_rest_r2_tr",
                   "single_p_low_share_tr", "rand_pop_r2_tr",
                   "rand_logdet_tr", "brute_trace_value", "brute_mi_value",
                   "path_min_gain_mi", "path_bonus_mi",
                   # facility location: objective values, selections and
                   # greedy gains are train-side too
                   "single_facloc_tr", "rand_facloc_tr", "facloc_facloc_tr",
                   "brute_facloc_value", "path_novelty_facloc",
                   "fl_shared_trace", "fl_jacc_trace",
                   "fl_path_facloc_gain_facloc", "fl_path_facloc_gain_trace",
                   "fl_path_trace_gain_facloc", "fl_path_trace_gain_trace"]
    train_side += [f"theory_{key}" for key in ao.THEORY_KEYS]
    for key in train_side:
        np.testing.assert_array_equal(arr[key][0], arr2[key][0], err_msg=key)
    assert not np.allclose(arr["trace_r2"][0], arr2["trace_r2"][0])
    # the input data are recorded (here: no SHA256SUMS next to them)
    assert str(arr2["data_sha256"]) == ao._sha256_file(
        data / "synth.matrix.npz")
    assert str(arr2["data_sha256sums"]) == "no SHA256SUMS"
    assert np.all(np.isfinite(arr["theory_trace_gamma"]))
    assert np.all(arr["theory_trace_gamma"] <= 1.0)

    # Rerun: bit-identical (common random numbers, deterministic selectors)
    arr3 = ao.run_dataset("synth", **KW)
    for key in ("rand_r2", "random_selected", "entropy_selected",
                "trace_osel_r2"):
        np.testing.assert_array_equal(arr2[key], arr3[key], err_msg=key)

    # Random sets are eligible k-subsets drawn from Split.seed(SEED + 19)
    assert ao.RANDOM_SETS_SEED_OFFSET not in (7, 11)

    paired = ao.paired_rows("synth", arr, [1, 3])
    tables, macros = ao.summarize("synth", arr, paired)
    assert macros and all(name.isalpha() for name in macros)
    assert all(np.isfinite(v) for v in macros.values())
    assert "ObjPopRsqSynthTraceKone" in macros
    _check_new_summaries(arr, paired, tables, macros)
    ao.print_summary("synth", tables)
    ao.plot_dataset("synth", arr, tables)
    ao.save("synth", arr, tables, macros, paired, {"test": True}, None)
    doc = json.loads((logs / "analyze_objectives_synth.json").read_text())
    assert doc["macros"] == macros
    assert (logs / "analyze_objectives_synth.npz").is_file()
    assert (logs / "analyze_objectives_synth_paired.csv").is_file()
    assert (figs / "analyze_objectives_synth.pdf").is_file()
    # trace's realized and in-sample k = 1 values equal entropy's (same pick)
    np.testing.assert_array_equal(arr["trace_selected"][:, 0],
                                  arr["entropy_selected"][:, 0])


def test_facloc_selection_and_section(synth_dirs):
    """facloc's train- and oracle-side selections are
    facility_location_select on the split's training estimate / on R_or
    with the split's eligibility mask; the facility-location section's
    arrays agree with the selections and realized R^2 they summarize."""
    from cv_split import cv_splits, eligible_mask, target_mask
    from eval_entropy_vs_mi import estimate_corr
    data, logs, figs = synth_dirs
    _write_dataset(data)
    kw = dict(KW, k_max=5, land_k=3)
    arr = ao.run_dataset("synth", **kw)
    d = np.load(data / "synth.matrix.npz")
    B, O = d["B"], d["O"]
    k_max = int(arr["k_max"])
    assert int(arr["fl_land_k"]) == 3
    for i, s in enumerate(cv_splits(70, 3, 1, ao.HOLDOUT_FRAC, ao.SEED)):
        el = eligible_mask(O[s.train_idx])
        R, mu, sd = estimate_corr(B[s.train_idx], O[s.train_idx])
        res = facility_location_select(R, k_max, eligible=el)
        ores = facility_location_select(arr["oracle_corr"], k_max,
                                        eligible=el)
        sel, osel = res["selected"], ores["selected"]
        assert arr["facloc_selected"][i].tolist() == \
            sel + [-1] * (k_max - len(sel))
        assert arr["facloc_osel_selected"][i].tolist() == \
            osel + [-1] * (k_max - len(osel))
        np.testing.assert_array_equal(arr["facloc_facloc_tr"][i],
                                      res["facloc_values"])
        tsel = greedy_trace_select(R, k_max, eligible=el)["selected"]
        for k in range(1, k_max + 1):
            assert arr["fl_shared_trace"][i, k - 1] == len(
                set(sel[:k]) & set(tsel[:k]))
            assert arr["fl_jacc_trace"][i, k - 1] == pytest.approx(
                len(set(sel[:k]) & set(tsel[:k]))
                / len(set(sel[:k]) | set(tsel[:k])))
        # greedy gains along the paths: facloc's own are its gains
        np.testing.assert_array_equal(arr["fl_path_facloc_gain_facloc"][i],
                                      res["gains"])
        for P, path in (("facloc", sel), ("trace", tsel)):
            r2 = arr[f"{P}_r2"][i]
            np.testing.assert_allclose(
                arr[f"fl_path_{P}_dr2"][i],
                np.diff(np.concatenate([[0.0], r2])), rtol=0, atol=1e-15)
            G = ao.trace_path_gains(R, path, el)
            np.testing.assert_array_equal(
                arr[f"fl_path_{P}_gain_trace"][i],
                [G[t, j] for t, j in enumerate(path)])
            # landscape before pick 2: every eligible candidate re-scored
            cand = [j for j in range(9) if el[j] and j != path[0]]
            dr2 = np.array([impute_and_evaluate(
                B[s.val_idx], O[s.val_idx], [path[0], j], R, mu, sd, 9,
                target=target_mask(O[s.train_idx]))["r2"] for j in cand])
            dr2 -= r2[0]
            assert arr[f"fl_land_{P}_n_cand"][i, 1] == len(cand)
            assert arr[f"fl_land_{P}_best_dr2"][i, 1] == pytest.approx(
                np.nanmax(dr2), abs=1e-12)
            assert arr[f"fl_land_{P}_rho_trace"][i, 1] == pytest.approx(
                ao._spearman(G[1, cand], dr2), abs=1e-12, nan_ok=True)
            # regret of each objective's pick: P's own pick for O = P,
            # else O's greedy step from A_1 (_greedy_choice)
            Gf = ao.facloc_path_gains(ao.Spectral(R).fl, path, el)
            ca = np.array(cand)
            for obj, Gobj in (("facloc", Gf), ("trace", G)):
                pick = (cand.index(path[1]) if obj == P else
                        ao._greedy_choice(obj, R, path[:1], ca, Gobj[1, ca]))
                assert arr[f"fl_land_{P}_regret_{obj}"][i, 1] == pytest.approx(
                    np.nanmax(dr2) - dr2[pick], abs=1e-12)
            # before pick 1: the singletons
            single = arr["single_r2"][i]
            assert arr[f"fl_land_{P}_rho_facloc"][i, 0] == pytest.approx(
                ao._spearman(np.where(el, ao.Spectral(R).fl_single,
                                      np.nan), single), nan_ok=True)
    paired = ao.paired_rows("synth", arr, [1, 3])
    tr_rows = [r for r in paired if r["comparison"] == "train"]
    assert {(r["method_a"], r["method_b"]) for r in tr_rows} == \
        {("trace", "facloc")}
    tables, macros = ao.summarize("synth", arr, paired)
    fl = tables["facloc"]
    r1 = fl["vs_trace"]["1"]
    assert r1["diff"] == pytest.approx(np.nanmean(
        arr["facloc_r2"][:, 0] - arr["trace_r2"][:, 0]))
    assert macros["ObjFlVsTraceDiffSynthKone"] == pytest.approx(r1["diff"])
    assert macros["ObjFlVsTraceSharedSynthKone"] == pytest.approx(
        np.mean(arr["fl_shared_trace"][:, 0]))
    assert "ObjFlPathRhoSynthFaclocGainFacloc" in macros
    assert "ObjFlLandRhoSynthTraceGainFaclocKone" in macros
    assert "ObjFaclocValSynthTraceKone" in macros
    assert "ObjBruteSynthFaclocKtwoRatio" in macros
    assert all(name.isalpha() for name in macros)
    ao.print_summary("synth", tables)
    ao.plot_facloc("synth", arr)
    assert (figs / "analyze_objectives_facloc_synth.pdf").is_file()


def test_adding_facloc_changes_no_other_output(synth_dirs, monkeypatch):
    """The invariance of the facloc extension, on synthetic data: a run
    with CORE_METHODS only (as before facloc joined) has a subset of the
    keys of the full run, every one bit-identical (except the method
    list), and every macro, paired row and table entry of its summary
    equals the full run's."""
    data, logs, figs = synth_dirs
    _write_dataset(data, M=70, N=9)
    full = ao.run_dataset("synth", **KW)
    monkeypatch.setattr(ao, "METHODS", list(ao.CORE_METHODS))
    core = ao.run_dataset("synth", **KW)
    monkeypatch.undo()
    assert not any(k.startswith("fl_") for k in core if k != "fl_land_k")
    for key, v in core.items():
        if key == "methods":
            continue
        assert key in full, key
        np.testing.assert_array_equal(np.asarray(v), np.asarray(full[key]),
                                      err_msg=key)
    ks = sorted(set(ao.KS_REPORT + [3, 15]))
    p_full = ao.paired_rows("synth", full, ks)
    p_core = ao.paired_rows("synth", core, ks)
    idx = {(r["k"], r["comparison"], r["method_a"], r["method_b"]): r
           for r in p_full}
    for r in p_core:
        r2 = idx[(r["k"], r["comparison"], r["method_a"], r["method_b"])]
        for f in r:
            assert r[f] == r2[f] or (r[f] != r[f] and r2[f] != r2[f]), f
    t_full, m_full = ao.summarize("synth", full, p_full)
    t_core, m_core = ao.summarize("synth", core, p_core)
    for name, v in m_core.items():
        assert m_full[name] == v, name

    def leaves(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from leaves(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from leaves(v, f"{path}[{i}]")
        else:
            yield path, obj
    lf = dict(leaves(t_full))
    for path, v in leaves(t_core):
        assert path in lf and lf[path] == v, path


def _check_new_summaries(arr, paired, tables, macros):
    """Range statistics, gap decomposition, in-sample gap and the oracle
    effect counts agree with the per-split arrays and paired rows."""
    # brute ratios: the mean hides the worst split
    for k, word in ((1, "Kone"), (2, "Ktwo")):
        g = arr["brute_trace_greedy"][:, k - 1]
        v = arr["brute_trace_value"][:, k - 1]
        ratios = (g - k) / (v - k)
        assert macros[f"ObjBruteSynthTrace{word}RatioMin"] == pytest.approx(
            np.nanmin(ratios))
        assert macros[f"ObjBruteSynthTrace{word}Ratio"] == pytest.approx(
            np.nanmean(ratios))
        assert macros[f"ObjBruteSynthTrace{word}FRatioMin"] == pytest.approx(
            np.nanmin(g / v))
    # singletons: the minimum over all eligible cells
    assert macros["ObjSingleSynthRestRsqMin"] == pytest.approx(
        np.nanmin(arr["single_rest_r2_tr"]))
    assert (macros["ObjSingleSynthRestRsqMin"]
            <= macros["ObjSingleSynthRestRsqSplitMinMedian"]
            <= macros["ObjSingleSynthRestRsqMedian"] + 1e-12)
    # theory ranges
    assert macros["ObjTheorySynthTraceGammaMin"] == pytest.approx(
        np.min(arr["theory_trace_gamma"]))
    assert macros["ObjTheorySynthMiMinGainMin"] == pytest.approx(
        np.min(arr["theory_mi_min_gain"]))
    # gap decomposition: one truth at a time, total = mask + rest
    for m in ("trace", "mi"):
        q = {f: arr[f"{m}_{f}"][:, 0] for f in
             ("r2", "pop_r2_tr", "pop_r2_or", "exp_r2_tr", "exp_r2_or")}
        ok = np.all([np.isfinite(v) for v in q.values()], axis=0)
        assert ok.any()
        q = {f: v[ok] for f, v in q.items()}
        row = tables["gap"]["1"][m]
        for tr in ("tr", "or"):
            assert row[tr]["total"] == pytest.approx(
                np.mean(q["r2"] - q[f"pop_r2_{tr}"]))
            assert row[tr]["mask"] == pytest.approx(
                np.mean(q[f"exp_r2_{tr}"] - q[f"pop_r2_{tr}"]))
            assert row[tr]["rest"] == pytest.approx(
                row[tr]["total"] - row[tr]["mask"])
        assert macros[f"ObjGapMaskTrainSynth{ao.camel(m)}Kone"] == \
            pytest.approx(row["tr"]["mask"])
    # largest in-sample-to-realized gap over k
    d = np.abs(np.mean(arr["hub_r2"] - arr["hub_pop_r2_tr"], axis=0))
    assert macros["ObjInSampleGapMaxSynthHub"] == pytest.approx(d.max())
    assert macros["ObjInSampleGapWorstKSynthHub"] == int(np.argmax(d)) + 1
    assert macros["ObjInSampleGapMaxSynth"] >= d.max() - 1e-15
    # oracle effects: counts over every method of CORE_METHODS (twin with
    # facloc: every method) and k of the comparison
    for methods, table, name in (
            (ao.CORE_METHODS, "oracle_effects", "ObjOracleSelNComparedSynth"),
            (ao.METHODS, "oracle_effects_with_facloc",
             "ObjOracleSelWithFaclocNComparedSynth")):
        rows = [r for r in paired if r["comparison"] == "oracle_sel - train"
                and r["method_a"] in methods
                and np.isfinite(r["mean_diff"])]
        eff = tables[table]["oracle_sel - train"]
        assert eff["n_compared"] == len(rows) == macros[name]
        n_sig = sum(1 for r in rows if r["ci_low"] > 0 or r["ci_high"] < 0)
        assert eff["n_sig"] == n_sig
        assert eff["max_abs_diff"] == pytest.approx(
            max(abs(r["mean_diff"]) for r in rows))
        assert eff["n_large"] == sum(1 for r in rows
                                     if abs(r["mean_diff"]) > ao.LARGE_DIFF)
    assert len(ao.METHODS) == len(ao.CORE_METHODS) + 1
    # the largest in-sample gap over all methods is at least the core one
    assert macros["ObjInSampleGapMaxWithFaclocSynth"] >= \
        macros["ObjInSampleGapMaxSynth"]


def test_provenance_manifest_and_full_oracle_tag(synth_dirs, tmp_path):
    data, logs, figs = synth_dirs
    _write_dataset(data)
    sha = ao._sha256_file(data / "synth.matrix.npz")
    (data / "SHA256SUMS").write_text(f"{'0' * 64}  synth.matrix.npz\n")
    assert ao.data_provenance("synth")["sha256sums"] == "mismatch"
    (data / "SHA256SUMS").write_text(f"{sha}  synth.matrix.npz\n")
    assert ao.data_provenance("synth") == {
        "file": "synth.matrix.npz", "sha256": sha, "sha256sums": "match"}

    kw = dict(KW, n_repeats=1)
    arr = ao.run_dataset("synth", oracle="full", **kw)
    assert str(arr["oracle"]) == "full"
    paired = ao.paired_rows("synth", arr, [1])
    tables, macros = ao.summarize("synth", arr, paired)
    assert macros and all(n.startswith("ObjFull") and n.isalpha()
                          for n in macros)
    tag = ao.out_tag("synth", "full")
    assert tag == "synth_fulloracle" and ao.out_tag("synth") == "synth"
    ao.save("synth", arr, tables, macros, paired, {}, None, tag)
    assert (logs / "analyze_objectives_synth_fulloracle.json").is_file()
    assert not (logs / "analyze_objectives_synth.json").exists()
    back, _ = ao.load_log("synth", "full")
    np.testing.assert_array_equal(back["trace_r2"], arr["trace_r2"])

    # --check-logs: a baselines log of other input data is flagged
    base = tmp_path / "base"
    (base / "logs").mkdir(parents=True)
    log = {"split_id": arr["split_id"], "k_max": arr["k_max"]}
    for m in ao.METHODS:
        log[f"{m}_selected"] = arr[f"{m}_selected"]
        log[f"{m}_r2"] = arr[f"{m}_r2"]
    np.savez(base / "logs" / "baselines_synth.npz", **log)
    assert ao.check_logs("synth", arr, base / "logs")["_data"][
        "same_data"] is None
    for want, same in (("f" * 64, False), (sha, True)):
        (base / "manifest.json").write_text(json.dumps(
            {"inputs": {"data/synth.matrix.npz": {"sha256": want}}}))
        out = ao.check_logs("synth", arr, base / "logs")
        assert out["_data"]["same_data"] is same
        assert all(out[m]["same_selected"] for m in ao.METHODS)


def test_workers_give_the_serial_arrays(synth_dirs, capsys):
    """--workers W: the splits run in W spawned processes (map_splits);
    every array equals the serial run's bit for bit (same keys in the same
    order, dtypes and shapes; NaN where NaN), so do the paired rows and
    macros, and the printed lines differ only in their times."""
    data, logs, figs = synth_dirs
    _write_dataset(data)
    kw = dict(KW, n_repeats=2)
    a = ao.run_dataset("synth", **kw)
    out_a = capsys.readouterr().out
    b = ao.run_dataset("synth", workers=2, **kw)
    out_b = capsys.readouterr().out
    assert list(a) == list(b)
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert x.dtype == y.dtype and x.shape == y.shape, key
        assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), key
    ks = sorted(set(ao.KS_REPORT + [3, 15]))
    p_a, p_b = ao.paired_rows("synth", a, ks), ao.paired_rows("synth", b, ks)
    assert len(p_a) == len(p_b)
    for r, s in zip(p_a, p_b):
        assert list(r) == list(s)
        for f in r:
            assert r[f] == s[f] or (r[f] != r[f] and s[f] != s[f]), f
    assert ao.summarize("synth", a, p_a)[1] == ao.summarize("synth", b,
                                                            p_b)[1]
    times = re.compile(r"\d+\.\d+s\b")
    assert out_a.count("  Split ") == 6
    assert times.sub("T", out_a) == times.sub("T", out_b)
