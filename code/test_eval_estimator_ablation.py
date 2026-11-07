"""Tests for eval_estimator_ablation.py (covariance-estimator ablation).

Synthetic data only (well under 60 s): the parameterized estimator
reproduces eval_entropy_vs_mi.estimate_corr bit for bit in all four
regimes of its rules (pairwise; unregularized EM; regularized EM;
n_train < N with "auto" shrinkage), the resolution of every setting
(branch, eps, alpha, iteration cap, tolerance, aliases of the default),
the knobs act as documented (forced EM on complete data equals pairwise,
shrinkage, PSD floor), the exact-reuse shortcuts equal fresh fits, the
Jaccard overlap, eligibility thresholds, the default rows equal the
pipeline of eval_baselines (same selections, common random numbers,
impute_and_evaluate), no leakage of validation rows into estimates,
eligibility or selections, aliases flagged by bit-identical R (a PSD
floor that never binds) and recovered from old logs, cover =
evaluated_share, the summaries (reversal = both intervals exclude 0 with
opposite signs; lost / gained; low-n cells out of the maxima and counts;
n-pairs macros; the core macro subset), and an end-to-end run (outputs,
letters-only macros, --jobs independence); facloc (a method since
2026-10-19): its picks are facility_location_select on the same estimate
and eligible set, and adding it leaves every array, paired row, summary
table and macro of the other methods unchanged (its own cross-method
summaries carry the suffix Facloc).
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile

# Outputs of the imported eval scripts go to a scratch root unless the
# caller (run_all.sh's tests job) set one
if not os.environ.get("BENCHSELECT_EXPERIMENT_ROOT"):
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-estimator-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

import eval_estimator_ablation as E
from cv_split import RANDOM_SEED_OFFSET, cv_splits, eligible_mask
from em_cov import em_cov
from baseline_select import facility_location_select
from eval_baselines import run_selection
from eval_entropy_vs_mi import estimate_corr, impute_and_evaluate
from greedy_select import random_select, selection_at_k


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _factor_corr(N=8, n_factors=2, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, n_factors))
    S = W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))
    s = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return R


def _data(M=60, N=8, obs=1.0, seed=0):
    """(B, O): factor-model scores (mean 0.5, SD 0.1), observed with
    probability obs (column 0 always)."""
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(N), _factor_corr(N, seed=seed),
                                size=M) * 0.1 + 0.5
    O = (rng.uniform(size=(M, N)) < obs).astype(np.float64)
    O[:, 0] = 1.0
    return X * O, O


# regime -> (M, N, observed fraction, data seed, expected default params);
# the seeds keep the EM fits short (em_reg converges in ~100 iterations)
REGIMES = {
    "pairwise": (60, 8, 1.0, 0, ("pairwise", 1e-6, 0.0, 0, 0.0)),
    "em": (60, 8, 0.8, 0, ("em", 1e-6, 0.0, 500, 5e-4)),
    "em_reg": (60, 8, 0.4, 2, ("em", 1e-3, 0.0, 1000, 5e-4)),
    "em_shrink": (6, 8, 0.75, 0, ("em", 1e-3, 0.25, 1000, 5e-4)),
}


def _regime(name):
    M, N, obs, seed, expected = REGIMES[name]
    B, O = _data(M, N, obs, seed)
    return B, O, expected


def _resolve(setting, O):
    M, N = O.shape
    return E.resolve_params(setting, M, N, O.sum() / (M * N))


def _names(N):
    return np.array([f"b{j}" for j in range(N)])


# ---------------------------------------------------------------------------
# Parameterized estimator: default == estimate_corr, bit for bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime", list(REGIMES))
def test_default_reproduces_estimate_corr(regime):
    B, O, expected = _regime(regime)
    p = _resolve(E.DEFAULT, O)
    assert p.key == expected
    est = E.estimate_with(p, B, O)
    R, cm, cs = estimate_corr(B, O)
    np.testing.assert_array_equal(est.R, R)
    np.testing.assert_array_equal(est.col_mean, cm)
    np.testing.assert_array_equal(est.col_std, cs)
    if p.branch == "em":
        assert p.shrinkage == ("auto" if O.shape[0] < O.shape[1] else 0.0)
        assert 1 <= est.n_iter <= p.max_iter


def test_default_constants_match_estimate_corr_source():
    """The rule constants are those of eval_entropy_vs_mi's source."""
    import inspect
    import eval_entropy_vs_mi as V
    src = inspect.getsource(V.estimate_corr) + inspect.getsource(
        V.estimate_corr_em) + inspect.getsource(V.estimate_corr_pairwise)
    for snippet in ("obs_frac < 0.9 or M_tr < N_tr",
                    "M_tr < N_tr or obs_frac < 0.5",
                    "1e-3 if regularized else 1e-6",
                    "1000 if regularized else 500", "tol=5e-4",
                    '"auto" if M_tr < N_tr else 0.0',
                    "np.maximum(eigvals, 1e-6)"):
        assert snippet in src, snippet
    assert (E.PAIRWISE_MIN_OBS_FRAC, E.REGULARIZE_OBS_FRAC) == (0.9, 0.5)
    assert (E.EPS_EM, E.EPS_EM_REG, E.ITER_EM, E.ITER_EM_REG, E.EM_TOL,
            E.EPS_PAIRWISE) == (1e-6, 1e-3, 500, 1000, 5e-4, 1e-6)


# ---------------------------------------------------------------------------
# Resolution of the settings
# ---------------------------------------------------------------------------

def _by(name):
    return E.SETTING_BY_NAME[name]


def test_setting_table():
    names = [s.name for s in E.SETTINGS]
    assert names[0] == "default" and len(set(names)) == len(names)
    macros = [s.macro for s in E.SETTINGS]
    assert len(set(macros)) == len(macros)
    assert all(m.isalpha() and m[0].isupper() for m in macros)
    assert {s.factor for s in E.SETTINGS} == set(E.FACTORS)
    # one factor at a time: each setting overrides one field (the tight
    # tolerance also raises the iteration cap)
    for s in E.SETTINGS[1:]:
        ov = s.overrides()
        assert len(ov) == (2 if s.name == "tol_5e-5" else 1), s.name
    assert [s.min_obs for s in E.SETTINGS if s.factor == "eligibility"] \
        == [1, 2, 5, 10]


def test_resolve_em_settings():
    M, N = 60, 8                       # n_train >= N, 80% observed: EM
    r = lambda s: E.resolve_params(_by(s), M, N, 0.8)   # noqa: E731
    assert r("default").key == ("em", 1e-6, 0.0, 500, 5e-4)
    assert r("pairwise").key == ("pairwise", 1e-6, 0.0, 0, 0.0)
    assert r("em").key == r("default").key
    for s in ("shrink_x0", "shrink_x0.25", "shrink_x4"):
        assert r(s).key == r("default").key          # alpha_0 = 0
    assert r("alpha_0.1").key == ("em", 1e-6, 0.1, 500, 5e-4)
    assert r("alpha_0.1").shrinkage == 0.1
    assert r("psd_1e-2").key == ("em", 1e-2, 0.0, 500, 5e-4)
    assert r("psd_1e-6").key == r("default").key
    assert r("tol_5e-3").key == ("em", 1e-6, 0.0, 500, 5e-3)
    assert r("tol_5e-5").key == ("em", 1e-6, 0.0, 2000, 5e-5)
    assert r("iter_x4").key == ("em", 1e-6, 0.0, 2000, 5e-4)
    # regularized (sparse) and rank-deficient regimes
    reg = E.resolve_params(_by("default"), 60, 8, 0.3)
    assert reg.key == ("em", 1e-3, 0.0, 1000, 5e-4) and reg.regularized
    p = E.resolve_params(_by("default"), 106, 114, 0.31)    # Merged
    assert p.alpha == pytest.approx(8 / 114) and p.shrinkage == "auto"
    q = E.resolve_params(_by("shrink_x0.25"), 106, 114, 0.31)
    assert q.alpha == pytest.approx(2 / 114) and q.shrinkage == q.alpha
    assert E.resolve_params(_by("shrink_x4"), 106, 114, 0.31).alpha \
        == pytest.approx(32 / 114)
    assert E.resolve_params(_by("shrink_x4"), 6, 8, 0.75).alpha == 1.0
    assert E.resolve_params(_by("shrink_x0"), 106, 114, 0.31).alpha == 0.0
    assert E.resolve_params(_by("iter_x4"), 106, 114, 0.31).max_iter == 4000


def test_resolve_pairwise_settings():
    M, N = 100, 8                      # complete, n_train >= N: pairwise
    r = lambda s: E.resolve_params(_by(s), M, N, 1.0)   # noqa: E731
    base = r("default").key
    assert base == ("pairwise", 1e-6, 0.0, 0, 0.0)
    for s in ("pairwise", "shrink_x0", "shrink_x0.25", "shrink_x4",
              "tol_5e-3", "tol_5e-5", "iter_x4", "psd_1e-6"):
        assert r(s).key == base, s
    assert r("em").key == ("em", 1e-6, 0.0, 500, 5e-4)
    assert r("alpha_0.3").key == ("pairwise", 1e-6, 0.3, 0, 0.0)
    assert r("psd_1e-3").key == ("pairwise", 1e-3, 0.0, 0, 0.0)


# ---------------------------------------------------------------------------
# The knobs
# ---------------------------------------------------------------------------

def test_forced_em_on_complete_data_equals_pairwise():
    B, O, _ = _regime("pairwise")
    pw = E.estimate_with(_resolve(_by("default"), O), B, O)
    em = E.estimate_with(_resolve(_by("em"), O), B, O)
    assert em.params.branch == "em" and em.converged
    np.testing.assert_allclose(em.R, pw.R, atol=1e-10, rtol=0)


def test_pairwise_shrinkage_and_floor():
    B, O, _ = _regime("pairwise")
    R0 = E.estimate_with(_resolve(_by("default"), O), B, O).R
    R3 = E.estimate_with(_resolve(_by("alpha_0.3"), O), B, O).R
    np.testing.assert_array_equal(R3, (1.0 - 0.3) * R0
                                  + 0.3 * np.eye(len(R0)))
    lam0 = np.linalg.eigvalsh(R0)[0]
    assert lam0 > 1e-3
    # a non-binding floor changes nothing, a binding one raises lambda_min
    R4 = E.estimate_with(_resolve(_by("psd_1e-4"), O), B, O).R
    np.testing.assert_array_equal(R4, R0)
    p = _resolve(_by("default"), O)._replace(eps=2.0 * lam0)
    R_hi = E.estimate_with(p, B, O).R
    assert np.linalg.eigvalsh(R_hi)[0] > lam0
    np.testing.assert_allclose(np.diag(R_hi), 1.0, atol=1e-12)


def test_em_shrinkage_and_floor():
    B, O, _ = _regime("em")
    p0 = _resolve(_by("default"), O)
    est0 = E.estimate_with(p0, B, O)
    est1 = E.estimate_with(p0._replace(alpha=1.0, shrinkage=1.0), B, O)
    np.testing.assert_allclose(est1.R, np.eye(len(est0.R)), atol=1e-12)
    est_a = E.estimate_with(_resolve(_by("alpha_0.3"), O), B, O)
    off = ~np.eye(len(est0.R), dtype=bool)
    assert np.mean(np.abs(est_a.R[off])) < np.mean(np.abs(est0.R[off]))
    # the EM call is em_cov with the resolved arguments
    cm, cs = est0.col_mean, est0.col_std
    res = em_cov(O * (B - cm) / cs, O, max_iter=500, tol=5e-4,
                 shrinkage=0.3, eps_psd=1e-6)
    S = res["Sigma"]
    d = 1.0 / np.sqrt(np.diag(S))
    np.testing.assert_array_equal(est_a.R, S * np.outer(d, d))
    est_f = E.estimate_with(_resolve(_by("psd_1e-2"), O), B, O)
    assert np.linalg.eigvalsh(est_f.R)[0] >= np.linalg.eigvalsh(est0.R)[0]


def test_reuse_shortcuts_equal_fresh_fits():
    B, O, _ = _regime("em_reg")
    p0 = _resolve(_by("default"), O)
    est0 = E.estimate_with(p0, B, O)
    assert est0.converged and est0.n_iter < p0.max_iter
    cache = [est0]
    p4 = _resolve(_by("iter_x4"), O)
    assert E.cached_estimate(cache, p4) is est0
    fresh = E.estimate_with(p4, B, O)
    np.testing.assert_array_equal(fresh.R, est0.R)
    assert fresh.n_iter == est0.n_iter
    # a different tolerance is never reused
    assert E.cached_estimate(cache, _resolve(_by("tol_5e-5"), O)) is None
    # a fit that hit its cap is not reused for a larger cap
    short = E.estimate_with(p0._replace(max_iter=2), B, O)
    assert not short.converged
    assert E.cached_estimate([short], p0._replace(max_iter=4)) is None
    assert E.cached_estimate([short], p0._replace(max_iter=2)) is short


def test_jaccard():
    assert E.jaccard([1, 2, 3], [3, 2, 1]) == 1.0
    assert E.jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert E.jaccard([1], [2]) == 0.0
    assert np.isnan(E.jaccard(None, [1]))


# ---------------------------------------------------------------------------
# One split: default pipeline, aliases, eligibility, no leakage
# ---------------------------------------------------------------------------

KMAX = 4


@pytest.fixture(scope="module")
def em_split():
    B, O = _data(60, 8, 0.8, seed=1)
    split = next(cv_splits(60, 3, 1, 0.1, E.SEED))
    res = E.evaluate_split(B, O, _names(8), split, E.SETTINGS, E.METHODS,
                           KMAX)
    return B, O, split, res


def test_default_rows_equal_pipeline(em_split):
    B, O, split, res = em_split
    tr, va = split.train_idx, split.val_idx
    R, cm, cs = estimate_corr(B[tr], O[tr])
    el = eligible_mask(O[tr])
    assert res["default_checked"] and not res["estimation_failed"]
    for j, m in enumerate(E.METHODS):
        result = run_selection(m, R, KMAX, _names(8), split, el)
        np.testing.assert_array_equal(res["selected"][0, j],
                                      result["selected"][:KMAX])
        for k in range(1, KMAX + 1):
            r2 = impute_and_evaluate(B[va], O[va], selection_at_k(result, k),
                                     R, cm, cs, 8)["r2"]
            assert res["r2"][0, j, k - 1] == r2
        np.testing.assert_array_equal(res["jaccard"][0, j], 1.0)
    # selection-only R^2: the setting's selection, default imputation
    np.testing.assert_array_equal(res["r2_selonly"][0], res["r2"][0])
    i = [s.name for s in E.SETTINGS].index("alpha_0.3")
    for j, m in enumerate(E.METHODS):
        for k in range(1, KMAX + 1):
            sel = res["selected"][i, j, :k]
            r2 = impute_and_evaluate(B[va], O[va], list(sel), R, cm, cs,
                                     8)["r2"]
            assert res["r2_selonly"][i, j, k - 1] == r2
            if res["jaccard"][i, j, k - 1] == 1.0:
                assert set(sel) == set(res["selected"][0, j, :k])
    # random: the split's common random numbers (SEED + 7), no new stream
    rnd = random_select(R, KMAX, seed=split.seed(E.SEED + RANDOM_SEED_OFFSET),
                        eligible=el)
    np.testing.assert_array_equal(res["selected"][0, E.METHODS.index(
        "random")], rnd["selected"])
    assert not [a for a in dir(E) if a.endswith("SEED_OFFSET")
                and a != "RANDOM_SEED_OFFSET"]


def test_aliases_are_exact(em_split):
    B, O, split, res = em_split
    names = [s.name for s in E.SETTINGS]
    same = dict(zip(names, res["est_same_as_default"]))
    # parameter aliases (reused default estimate)
    for s in ("em", "shrink_x0", "shrink_x0.25", "shrink_x4", "psd_1e-6",
              "iter_x4", "elig_1", "elig_2", "elig_5", "elig_10"):
        assert same[s], s
    for s in ("pairwise", "alpha_0.01", "alpha_0.1", "alpha_0.3",
              "psd_1e-2", "tol_5e-3", "tol_5e-5"):
        assert not same[s], s
    # est_same_as_default is exactly "R equals the default's bit for bit"
    # (refits included: psd_1e-4 / psd_1e-3 here iff their floor never binds)
    np.testing.assert_array_equal(res["est_same_as_default"],
                                  res["r_max_abs_diff"] == 0.0)
    tr = split.train_idx
    R0 = estimate_corr(B[tr], O[tr])[0]
    for s in ("psd_1e-4", "psd_1e-3"):
        fit = E.estimate_with(_resolve(_by(s), O[tr]), B[tr], O[tr])
        assert same[s] == bool(np.array_equal(fit.R, R0)), s
    for i, s in enumerate(names):
        if same[s] and E.SETTINGS[i].min_obs is None:
            np.testing.assert_array_equal(res["r2"][i], res["r2"][0])
            np.testing.assert_array_equal(res["selected"][i],
                                          res["selected"][0])
            assert res["r_rel_diff"][i] == 0.0
            assert res["alias_of_default"][i]
    # an eligibility setting is an alias iff its eligible set is the default's
    el = eligible_mask(O[tr])
    for i, s in enumerate(E.SETTINGS):
        if s.min_obs is not None:
            assert res["alias_of_default"][i] == bool(np.array_equal(
                eligible_mask(O[tr], s.min_obs), el)), s.name
    # one fit per setting that does not reuse an estimate (default + 9)
    assert res["n_fits"] == 10
    assert res["est_branch_em"].tolist() == [n != "pairwise" for n in names]


def test_eligibility_thresholds():
    B, O = _data(60, 8, 1.0, seed=2)
    split = next(cv_splits(60, 3, 1, 0.1, E.SEED))
    tr = split.train_idx
    O = O.copy()
    rows3 = tr[:2]                     # column 3: 2 training observations
    O[:, 3] = 0.0
    O[rows3, 3] = 1.0
    O[:, 5] = 0.0                      # column 5: 4 training observations
    O[tr[:4], 5] = 1.0
    B = B * O
    settings = [_by(n) for n in ("default", "elig_1", "elig_2", "elig_5",
                                 "elig_10")]
    res = E.evaluate_split(B, O, _names(8), split, settings, E.METHODS, 6)
    n_el = res["n_eligible_setting"].tolist()
    assert n_el == [7, 8, 8, 6, 6]
    for i, s in enumerate(settings):
        el = eligible_mask(O[tr], s.min_obs or 3)
        sel = res["selected"][i]
        assert np.all(el[sel[sel >= 0]]), s.name
    assert res["est_same_as_default"].all()


def test_no_leakage_of_validation_rows(em_split):
    B, O, split, res = em_split
    va = split.val_idx
    rng = np.random.default_rng(7)
    B2, O2 = B.copy(), O.copy()
    B2[va] = rng.uniform(0, 1, size=B2[va].shape)
    O2[va[0], 1:] = 0.0
    B2 = B2 * O2
    res2 = E.evaluate_split(B2, O2, _names(8), split, E.SETTINGS, E.METHODS,
                            KMAX)
    for key in list(E.SPLIT_SET_KEYS) + ["selected", "index_ties",
                                         "jaccard", "eligible", "n_fits"]:
        np.testing.assert_array_equal(res[key], res2[key], err_msg=key)
    assert not np.allclose(res["r2"], res2["r2"], equal_nan=True)


def test_nonbinding_psd_floor_is_an_alias():
    """A refit whose R equals the default's bit for bit (a PSD floor below
    lambda_min of the pairwise R, as psd_1e-4 / psd_1e-3 on MMLU) is
    flagged and evaluated like a reused estimate; a binding floor is
    not."""
    B, O = _data(60, 8, 1.0, seed=0)
    split = next(cv_splits(60, 3, 1, 0.1, E.SEED))
    tr = split.train_idx
    R0 = estimate_corr(B[tr], O[tr])[0]
    lam0 = float(np.linalg.eigvalsh(R0)[0])
    assert 1e-3 < lam0 < 0.5
    psd = [s for s in E.SETTINGS if s.factor == "psd_floor"]
    settings = [E.DEFAULT] + psd
    res = E.evaluate_split(B, O, _names(8), split, settings, E.METHODS,
                           KMAX)
    for i, s in enumerate(settings[1:], start=1):
        binds = s.psd_eps > lam0
        assert res["est_same_as_default"][i] == (not binds), s.name
        assert res["alias_of_default"][i] == (not binds), s.name
        assert not res["est_reused"][i] or s.name == "psd_1e-6", s.name
        if not binds:
            np.testing.assert_array_equal(res["r2"][i], res["r2"][0])
            np.testing.assert_array_equal(res["r2_selonly"][i], res["r2"][i])
            assert res["r_max_abs_diff"][i] == 0.0
    # psd_1e-6 is the parameter alias; 1e-4 and 1e-3 are refit
    assert res["n_fits"] == 1 + sum(1 for s in psd if s.name != "psd_1e-6")


def test_cover_is_evaluated_share(em_split):
    """cover logs which share of the validation cells r2 scores: each
    selection's own observed unselected cells (eval_baselines)."""
    from eval_baselines import evaluated_share
    B, O, split, res = em_split
    O_va = O[split.val_idx]
    for i in range(len(E.SETTINGS)):
        for j in range(len(E.METHODS)):
            for k in range(1, KMAX + 1):
                sel = res["selected"][i, j, :k]
                assert res["cover"][i, j, k - 1] == evaluated_share(
                    O_va, list(sel))
    # fully observed data: every cell outside the selection is scored
    B1, O1 = _data(60, 8, 1.0, seed=0)
    res1 = E.evaluate_split(B1, O1, _names(8), split, [E.DEFAULT], E.METHODS,
                            KMAX)
    np.testing.assert_array_equal(res1["cover"], 1.0)


# ---------------------------------------------------------------------------
# Summaries: aliases, low-n cells, method-pair status
# ---------------------------------------------------------------------------

def test_pair_status():
    assert E.pair_status(1, -1) == "reversed"
    assert E.pair_status(-1, 1) == "reversed"
    assert E.pair_status(1, 0) == "lost"
    assert E.pair_status(0, -1) == "gained"
    assert E.pair_status(0, 0) == "same" and E.pair_status(1, 1) == "same"
    assert E.pair_status(1, -1, alias=True) == "alias"
    assert E.pair_status(1, -1, low_n=True) == "low_n"


def _fake_out(n=20):
    """A synthetic log: methods mi, random; k = 1, 2; settings default,
    pairwise (mi - random significantly negative at both k), alpha_0.01
    (alias of the default), alpha_0.1 (mi - random = the default's k = 1
    noise at both k), alpha_0.3 (as the default, mi R^2 NaN on half the
    splits and +0.5 on the others).  Default mi - random: k = 1 mean
    +0.001, not significant; k = 2 +0.1, significant."""
    names = ["default", "pairwise", "alpha_0.01", "alpha_0.1", "alpha_0.3"]
    settings = [_by(s) for s in names]
    n_set, n_m, K = len(settings), 2, 2
    rng = np.random.default_rng(0)
    base = 0.5 + 0.01 * rng.standard_normal(n)
    tiny = 1e-3 * rng.standard_normal(n)
    alt = np.where(np.arange(n) % 2 == 0, 0.011, -0.009)
    r2 = np.empty((n, n_set, n_m, K))

    def fill(i, d1, d2):
        r2[:, i, 1, :] = base[:, None]
        r2[:, i, 0, 0] = base + d1
        r2[:, i, 0, 1] = base + d2

    fill(0, alt, 0.1 + tiny)
    fill(1, -0.1 + tiny, -0.1 + tiny)
    fill(2, alt, 0.1 + tiny)
    fill(3, alt, alt)
    fill(4, alt + 0.5, 0.6 + tiny)
    r2[::2, 4, 0, :] = np.nan
    alias = np.zeros((n, n_set), dtype=bool)
    alias[:, [0, 2]] = True
    z = np.zeros((n, n_set))
    return {
        "settings": np.array(names),
        "setting_factor": np.array([s.factor for s in settings]),
        "setting_label": np.array([s.label for s in settings]),
        "setting_macro": np.array([s.macro for s in settings]),
        "methods": np.array(["mi", "random"]), "ks_report": np.array([1, 2]),
        "r2": r2, "r2_selonly": r2.copy(), "jaccard": np.ones_like(r2),
        "estimation_failed": np.zeros(n, dtype=bool),
        "n_train": np.full(n, 80), "n_val": np.full(n, 10),
        "rep": np.arange(n) // 10, "default_checked": np.ones(n, dtype=bool),
        "est_same_as_default": alias.copy(), "alias_of_default": alias,
        "est_failed": np.zeros((n, n_set), dtype=bool),
        "est_branch_em": np.zeros((n, n_set), dtype=bool),
        "em_converged": np.zeros((n, n_set), dtype=bool),
        "est_alpha": z.copy(), "est_eps": z + 1e-6,
        "est_max_iter": np.zeros((n, n_set), dtype=np.int64),
        "em_n_iter": np.zeros((n, n_set), dtype=np.int64),
        "r_rel_diff": np.where(alias, 0.0, 0.1),
        "r_max_abs_diff": np.where(alias, 0.0, 0.1), "r_lam_min": z + 0.5,
        "n_eligible_setting": np.full((n, n_set), 8),
        "n_eligible": np.full(n, 8),
    }


def test_summary_reversals_low_n_and_aliases():
    out = _fake_out()
    paired = E.paired_rows("fake", out)
    tables, mac = E.summarize("fake", out, paired)
    star = {r["k"]: r for r in tables["method_pairs"] if r["setting"] == "*"}
    # k = 1: the default difference is not significant, so the setting
    # whose interval excludes 0 with the opposite sign of the default's
    # mean is "gained", not a reversal
    assert star[1]["default_significant"] == 0
    assert star[1]["default_mean_diff"] > 0
    pw = [r for r in tables["method_pairs"] if r["setting"] == "pairwise"
          and r["k"] == 1][0]
    assert pw["significant"] == -1 and not pw["reversal"]
    assert star[1]["reversal_settings"] == []
    assert star[1]["gained_settings"] == ["pairwise"]
    assert star[1]["alias_settings"] == ["alpha_0.01"]
    assert star[1]["low_n_settings"] == ["alpha_0.3"]
    # k = 2: significant default difference, reversed / lost
    assert star[2]["default_significant"] == 1
    assert star[2]["reversal_settings"] == ["pairwise"]
    assert star[2]["lost_settings"] == ["alpha_0.1"]
    assert mac["EstPairReversalsFakeMiVsRandomKone"] == 0
    assert mac["EstPairGainedSigFakeMiVsRandomKone"] == 1
    assert mac["EstPairReversalsFakeMiVsRandomKtwo"] == 1
    assert mac["EstPairLostSigFakeMiVsRandomKtwo"] == 1
    assert mac["EstPairDefaultSigFakeMiVsRandomKtwo"] == 1
    assert (mac["EstPairReversalsFake"], mac["EstPairNSigDefault" + "Fake"]) \
        == (1, 1)
    # low-n: alpha_0.3 mi has 10 of 20 pairs and a +0.5 difference, which
    # is listed but kept out of the factor maximum and counts
    rows = {(r["setting"], r["method"], r["k"]): r
            for r in tables["vs_default"]}
    cell = rows[("alpha_0.3", "mi", 1)]
    assert (cell["n_pairs"], cell["low_n"], cell["nan_rate"]) == (10, True,
                                                                  0.5)
    assert cell["mean_diff"] == pytest.approx(0.5)
    assert mac["EstNPairsFakeAlphaPointThreeMiKone"] == 10
    assert mac["EstNRsqFakeAlphaPointThreeMiKone"] == 10
    assert mac["EstNRsqFakeDefaultMiKone"] == 20
    fac = {(r["factor"], r["k"]): r for r in tables["factors"]}
    sh = fac[("shrinkage", 1)]
    assert sh["n_alias_cells"] == 2 and sh["n_low_n_cells"] == 1
    assert sh["n_cells"] == 3                 # alpha_0.1 x 2, alpha_0.3 rnd
    assert sh["max_abs_diff"] < 0.1
    assert sh["max_abs_diff_low_n"] == pytest.approx(0.5)
    assert [c["setting"] for c in sh["low_n_cells"]] == ["alpha_0.3"]
    assert fac[("shrinkage", 2)]["max_abs_diff"] == pytest.approx(0.1,
                                                                  abs=2e-3)
    assert mac["EstNLowNFakeShrinkageKone"] == 1
    assert mac["EstNCellsFakeShrinkageKone"] == 3
    assert mac["EstNAliasSettingsFakeShrinkage"] == 1
    assert mac["EstNCellsFakeKone"] == 3 + 2      # + pairwise x 2
    # aliases: "(= default)" in the figure, est diagnostics
    ticks = E.setting_ticks(out)
    assert ticks[2].endswith("(= default)")
    assert not any(t.endswith("(= default)") for i, t in enumerate(ticks)
                   if i != 2)
    assert mac["EstAliasSplitsFakeAlphaPointZeroOne"] == 20
    # core subset: summary numbers only
    core = tables["macros_core"]
    assert set(core) <= set(mac)
    assert "EstPairReversalsFake" in core and "EstNSigFakeKtwo" in core
    assert "EstRsqFakeDefaultMiKone" in core
    assert not [c for c in core if c.startswith(("EstDiff", "EstNPairs",
                                                  "EstJacc", "EstSelDiff"))]
    assert "EstRsqFakePairwiseMiKone" not in core


def test_alias_mask_of_logs_without_alias_array(em_split):
    """--from-logs on a log written before alias_of_default existed: the
    mask is recovered from est_same_as_default / r_max_abs_diff and the
    eligible-set sizes."""
    _, _, _, res = em_split
    out = {key: res[key][None] for key in ("alias_of_default",
                                           "est_same_as_default",
                                           "r_max_abs_diff",
                                           "n_eligible_setting", "est_failed")}
    out["n_eligible"] = np.array([res["n_eligible"]])
    out["estimation_failed"] = np.array([False])
    logged = E.alias_mask(out)
    old = dict(out)
    del old["alias_of_default"]
    np.testing.assert_array_equal(E.alias_mask(old), logged)
    # max |R - R_0| = 0 alone recovers it (older logs set the flag only
    # for reused estimates, not for bit-identical refits)
    old["est_same_as_default"] = np.zeros_like(old["est_same_as_default"])
    np.testing.assert_array_equal(E.alias_mask(old), logged)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_dirs(tmp_path, monkeypatch):
    data, logs, figs = (tmp_path / d for d in ("data", "logs", "figures"))
    for d in (data, logs, figs):
        d.mkdir()
    monkeypatch.setattr(E, "DATA", data)
    monkeypatch.setattr(E, "LOGS", logs)
    monkeypatch.setattr(E, "OUT", figs)
    B, O = _data(45, 7, 0.75, seed=3)
    np.savez(data / "synth.matrix.npz", B=B, O=O,
             model_names=np.array([f"m{i}" for i in range(45)]),
             benchmark_names=_names(7))
    (data / "synth.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [None, None]}}))
    return data, logs, figs


def test_end_to_end(synth_dirs):
    _, logs, figs = synth_dirs
    settings = E.select_settings(["pairwise", "shrink_x4", "alpha_0.1",
                                  "psd_1e-2", "tol_5e-3", "elig_5"])
    kw = dict(n_repeats=1, n_folds=3, k_max=4, methods=E.METHODS,
              settings=settings, ks=[1, 3])
    out, timing = E.run_dataset("synth", **kw)
    S, n_set, n_m = 3, len(settings), len(E.METHODS)
    assert out["r2"].shape == (S, n_set, n_m, 4)
    assert out["selected"].dtype == np.int64
    assert out["split_id"].tolist() == [0, 1, 2]
    for key in ("rep", "fold", "n_train", "n_val", "n_eligible"):
        assert out[key].shape == (S,)
    assert out["default_checked"].all()
    np.testing.assert_array_equal(out["pairwise__mi_r2"], out["r2"][:, 1, 0])
    assert out["ks_report"].tolist() == [1, 3]

    paired = E.paired_rows("synth", out)
    tables, macros = E.summarize("synth", out, paired)
    assert macros and all(name.isalpha() for name in macros)
    assert all(np.isfinite(v) for v in macros.values())
    for name in ("EstRsqSynthDefaultMiKone", "EstDiffSynthPairwiseTraceKthree",
                 "EstJaccSynthEligFiveRandomKone", "EstNSplitsSynth",
                 "EstPairReversalsSynthMiVsTraceKone"):
        assert name in macros, name
    assert macros["EstDefaultCheckedSynth"] == 3
    for comp in ("vs_default", "selection_vs_default"):
        assert len([r for r in paired if r["comparison"] == comp]) \
            == 2 * (n_set - 1) * n_m
    assert "EstSelDiffSynthAlphaPointOneMiKthree" in macros
    E.print_summary("synth", out, tables)
    E.plot_dataset("synth", out, tables)
    config = {"test": True}
    E.save("synth", out, tables, macros, paired, config, timing)
    doc = json.loads((logs / "estimator_ablation_synth.json").read_text())
    assert doc["macros"] == macros
    assert doc["macros_core"] and set(doc["macros_core"]) <= set(macros)
    assert doc["macros_core"] == {n: macros[n] for n in tables["macros_core"]}
    assert "macros_core" not in doc["summary"]
    for name in ("EstNSigSynthKthree", "EstNCellsSynthShrinkage",
                 "EstPairNSigDefaultSynth", "EstEmIterSynthDefault"):
        assert name in doc["macros_core"], name
    assert out["cover"].shape == out["r2"].shape
    assert out["alias_of_default"][:, 0].all()
    with np.load(logs / "estimator_ablation_synth.npz") as d:
        np.testing.assert_array_equal(d["r2"], out["r2"])
    assert (logs / "estimator_ablation_synth_paired.csv").is_file()
    assert (figs / "estimator_ablation_synth.pdf").is_file()

    # --from-logs: the saved log reproduces the summaries and macros
    out_l, config_l, timing_l = E.load_log("synth")
    assert config_l == config
    assert timing_l["n_fits"] == timing["n_fits"]
    _, macros_l = E.summarize("synth", out_l, E.paired_rows("synth", out_l))
    assert macros_l == macros

    # --jobs does not change any value
    out2, _ = E.run_dataset("synth", jobs=2, **kw)
    for key in ("r2", "selected", "jaccard", "r_rel_diff", "em_n_iter"):
        np.testing.assert_array_equal(out[key], out2[key], err_msg=key)


# ---------------------------------------------------------------------------
# facloc (facility location, a method since 2026-10-19)
# ---------------------------------------------------------------------------

PRE_FACLOC_METHODS = ["mi", "entropy", "trace", "random"]


def test_facloc_is_facility_location_select(em_split):
    """facloc's picks are facility_location_select (clipped) on the
    split's training estimate and eligible set (default), and on the same
    estimate with eligible_mask(O_train, m) (eligibility settings)."""
    B, O, split, res = em_split
    tr = split.train_idx
    R, _, _ = estimate_corr(B[tr], O[tr])
    j = E.METHODS.index("facloc")
    # appended to the method axis (the earlier methods keep their
    # positions), displayed in the pair orientation (between hub and random)
    assert E.METHODS == PRE_FACLOC_METHODS + ["facloc"]
    assert E.ALL_METHODS[-1] == "facloc" and E.APART_METHODS == ["facloc"]
    assert E.display_methods({"methods": np.array(E.METHODS)}) == \
        ["mi", "entropy", "trace", "facloc", "random"]
    assert E.display_methods({"methods": np.array(E.ALL_METHODS)}) == \
        ["mi", "entropy", "trace", "pca", "hub", "facloc", "random"]
    names = [s.name for s in E.SETTINGS]
    for name, m in (("default", None), ("elig_1", 1), ("elig_10", 10)):
        el = eligible_mask(O[tr]) if m is None else eligible_mask(O[tr], m)
        ref = facility_location_select(R, KMAX, eligible=el,
                                       similarity="clipped")["selected"]
        sel = res["selected"][names.index(name), j]
        assert sel[sel >= 0].tolist() == ref[:KMAX], name


def test_facloc_changes_no_other_value(synth_dirs):
    """A run with facloc equals the pre-facloc run on every array (on the
    method axis: the leading entries, facloc being appended), paired row,
    summary table and macro of the latter; facloc adds only its own cells,
    rows and Facloc macros."""
    settings = E.select_settings(["pairwise", "shrink_x4", "psd_1e-2",
                                  "elig_5"])
    kw = dict(n_repeats=1, n_folds=3, k_max=4, settings=settings, ks=[1, 3])
    new, _ = E.run_dataset("synth", methods=E.METHODS, **kw)
    old, _ = E.run_dataset("synth", methods=PRE_FACLOC_METHODS, **kw)
    idx = [E.METHODS.index(m) for m in PRE_FACLOC_METHODS]
    assert idx == list(range(len(PRE_FACLOC_METHODS)))
    assert set(new) - set(old) == {k for k in new if "facloc" in k}
    for key, a in old.items():
        b = new[key]
        if key == "methods":
            assert list(b[:len(a)]) == list(a) and list(b[len(a):]) == [
                "facloc"]
        elif key in E.SPLIT_SMK_KEYS:
            np.testing.assert_array_equal(a, b[:, :, idx], err_msg=key)
        else:
            np.testing.assert_array_equal(a, b, err_msg=key)
    p_old, p_new = E.paired_rows("synth", old), E.paired_rows("synth", new)
    keep = [r for r in p_new
            if "facloc" not in (r["method_a"] + r["method_b"])]
    assert json.dumps(keep, default=str) == json.dumps(p_old, default=str)
    t_old, m_old = E.summarize("synth", old, p_old)
    t_new, m_new = E.summarize("synth", new, p_new)
    assert {k: m_new[k] for k in m_old} == m_old
    assert set(m_new) > set(m_old)
    assert all("Facloc" in k for k in set(m_new) - set(m_old))
    for key in ("factors", "dataset", "estimator"):
        assert json.dumps(t_new[key], default=str) == \
            json.dumps(t_old[key], default=str), key
    assert {"factors_facloc", "dataset_facloc"} <= set(t_new)
    assert all(c["method"] == "facloc" for r in t_new["factors_facloc"]
               for c in r["low_n_cells"] + r["significant_cells"])
    for name in ("EstMaxAbsDiffSynthShrinkageFaclocKone",
                 "EstNCellsSynthFaclocKthree", "EstNCellsSynthFacloc",
                 "EstPairNSigDefaultSynthFacloc",
                 "EstRsqSynthDefaultFaclocKone",
                 "EstPairSynthPairwiseMiVsFaclocKone",
                 "EstPairSynthDefaultFaclocVsRandomKthree"):
        assert name in m_new, name
    # facloc's dataset cells: one per non-default setting
    assert m_new["EstNCellsSynthFaclocKone"] + \
        t_new["dataset_facloc"][0]["n_alias_cells"] + \
        t_new["dataset_facloc"][0]["n_low_n_cells"] == len(settings) - 1
