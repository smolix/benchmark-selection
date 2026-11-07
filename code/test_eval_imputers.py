"""Tests for eval_imputers.py (fitted matrix-completion imputers vs the
Gaussian conditional at fixed selections).

Synthetic data only (< 60 s): the SoftImpute solver (fixed point of the
soft-thresholded SVD map, accelerated = plain iterations, zero above
lambda_max), the HardImpute solver (Mazumder et al.'s rank-r SVD refill:
its fixed point, exact recovery of low-rank data, the interlaced
decrease of its objectives, bounded iterates on a case where the exact
least-squares refill runs off, every fit started at SoftImpute's), the
exactness of the pattern compression, the scored cells and pooled R^2
against
impute_and_evaluate (and every cell of every model where its fast path
would drop some, protocol_changes.md H1), the inner holdout (training
models only, common random numbers of Split.seed(SEED + 21), the seed
tag unique in the code tree), no leakage of the validation models'
unselected cells into any fit or hyperparameter, and an end-to-end run
(bit-identical rerun, selections of eval_baselines, paired rows,
letters-only macros, output files, --from-logs); facloc (2026-10-19): its
selections are facility_location_select's, and appending it changes no
array, paired row or macro of the pooled methods; --workers (2026-10-20):
the splits in spawned processes reproduce the serial run (every array,
the progress lines, and main's logs as a single run and as shards plus
--merge, config.workers aside).
"""

from __future__ import annotations

import atexit
import json
import math
import os
import pathlib
import re
import shutil
import tempfile

# Outputs of the imported eval scripts go to a scratch root unless the
# caller (run_all.sh's tests job) set one
if not os.environ.get("BENCHSELECT_EXPERIMENT_ROOT"):
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-imputers-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

import eval_imputers as E
from baseline_select import facility_location_select
from cv_split import (ENTROPY_RF_SEED_OFFSET, RANDOM_SEED_OFFSET, cv_splits,
                      eligible_mask)
from eval_baselines import run_selection
from eval_entropy_vs_mi import estimate_corr, impute_and_evaluate
from greedy_select import selection_at_k
from paired_stats import STAT_FIELDS, orient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _low_rank(n=60, N=12, r=3, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, r)) @ rng.standard_normal((r, N))
    return X + noise * rng.standard_normal((n, N))


def _mixed_mask(n=60, N=12, seed=0, sel=(0, 3, 7)):
    """20 complete rows, 25 rows observing exactly `sel`, the rest random
    (each observing >= 4 cells)."""
    rng = np.random.default_rng(seed + 1)
    mask = np.ones((n, N), dtype=bool)
    mask[20:45] = False
    mask[20:45, list(sel)] = True
    rest = rng.uniform(size=(n - 45, N)) < 0.6
    rest[:, :4] = True
    mask[45:] = rest
    return mask


def _svd_soft(F, lam):
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    return (U * np.maximum(s - lam, 0.0)) @ Vt


def _svd_rank(F, r):
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def _factor_corr(N, n_factors=2, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, n_factors))
    S = W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))
    s = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------------------
# Solvers and compression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", E.SOLVERS)
def test_soft_impute_fixed_point_and_compression(solver):
    X, mask = _low_rank(), _mixed_mask()
    lam = 0.3 * E.lambda_max(np.where(mask, X, 0.0))
    comp = E.Completion(X, mask)
    full = E.Completion(X, mask, compress=False)
    assert comp.n_ps < full.n_ps == X.shape[0]
    S1, i1 = E.soft_impute(comp, lam, tol=1e-16, max_iter=20000,
                           solver=solver)
    S2, i2 = E.soft_impute(full, lam, tol=1e-16, max_iter=20000,
                           solver=solver)
    assert i1["converged"] and i2["converged"]
    Y1, Y2 = comp.to_real(S1), full.to_real(S2)
    np.testing.assert_allclose(Y1, Y2, atol=1e-7)
    assert np.all(Y1[mask] == 0.0)
    # Fixed point of Z = S_lam(P_Omega X + P_Omega-perp Z)
    Z = _svd_soft(np.where(mask, X, 0.0) + Y1, lam)
    np.testing.assert_allclose(np.where(mask, 0.0, Z), Y1, atol=1e-6)
    assert abs(i1["objective"] - i2["objective"]) < 1e-8 * i2["objective"]


def test_soft_impute_solvers_agree():
    X, mask = _low_rank(seed=3), _mixed_mask(seed=3)
    prob = E.Completion(X, mask)
    lam = 0.1 * E.lambda_max(np.where(mask, X, 0.0))
    out = {s: E.soft_impute(prob, lam, tol=1e-16, max_iter=50000, solver=s)
           for s in E.SOLVERS}
    assert all(info["converged"] for _, info in out.values())
    ref = prob.to_real(out["exact"][0])
    for s in ("fista", "plain"):
        np.testing.assert_allclose(prob.to_real(out[s][0]), ref, atol=1e-6)
    assert (out["exact"][1]["iters"] < out["fista"][1]["iters"]
            < out["plain"][1]["iters"])
    with pytest.raises(ValueError):
        E.soft_impute(prob, lam, solver="newton")


def test_soft_impute_small_lambda_exact_reaches_fixed_point():
    """At small lambda a plain step moves the missing cells by about
    lambda / s: the change criterion stops plain iterations far from the
    solution, the exact row steps reach the fixed point."""
    rng = np.random.default_rng(11)
    n, N = 300, 8
    X = rng.standard_normal((n, 3)) @ rng.standard_normal((3, N)) \
        + 0.5 * rng.standard_normal((n, N))
    mask = np.ones((n, N), dtype=bool)
    mask[250:] = False
    mask[250:, [0, 1]] = True               # query rows: 2 observed cells
    prob = E.Completion(X, mask)
    lam = 1e-3 * E.lambda_max(np.where(mask, X, 0.0))
    Se, ie = E.soft_impute(prob, lam, tol=1e-12, max_iter=500,
                           solver="exact")
    Sp, ip = E.soft_impute(prob, lam, tol=1e-12, max_iter=500,
                           solver="plain")
    Ye, Yp = prob.to_real(Se), prob.to_real(Sp)
    F = np.where(mask, X, 0.0)
    res_e = np.abs(np.where(mask, 0.0, _svd_soft(F + Ye, lam)) - Ye).max()
    assert ie["converged"] and res_e < 1e-5
    assert ie["objective"] <= ip["objective"] + 1e-9
    assert np.abs(Ye - Yp).max() > 0.1      # plain stopped far away
    # monotone objective of the safeguarded exact iterations
    S, prev = None, np.inf
    for _ in range(15):
        S, info = E.soft_impute(prob, lam, S, tol=1e-30, max_iter=1,
                                solver="exact")
        assert info["objective"] <= prev + 1e-9
        prev = info["objective"]


def test_soft_impute_zero_above_lambda_max():
    X, mask = _low_rank(), _mixed_mask()
    lam_max = E.lambda_max(np.where(mask, X, 0.0))
    prob = E.Completion(X, mask)
    S, info = E.soft_impute(prob, lam_max * 1.0001)
    assert info["converged"] and info["eff_rank"] == 0
    assert np.all(S == 0.0)
    S, info = E.soft_impute(prob, 0.5 * lam_max)
    assert info["eff_rank"] >= 1 and np.abs(S).max() > 0


@pytest.mark.parametrize("rank", [1, 3])
def test_hard_impute_fixed_point_and_compression(rank):
    X, mask = _low_rank(), _mixed_mask()
    comp = E.Completion(X, mask)
    full = E.Completion(X, mask, compress=False)
    S1, i1 = E.hard_impute(comp, rank, tol=1e-24, max_iter=50000)
    S2, i2 = E.hard_impute(full, rank, tol=1e-24, max_iter=50000)
    assert i1["converged"] and i2["converged"]
    Y1, Y2 = comp.to_real(S1), full.to_real(S2)
    np.testing.assert_allclose(Y1, Y2, atol=1e-8)
    assert np.all(Y1[mask] == 0.0)
    # Fixed point of Mazumder et al.'s "fill with the rank-r SVD" iteration
    Z = _svd_rank(np.where(mask, X, 0.0) + Y1, rank)
    np.testing.assert_allclose(np.where(mask, 0.0, Z), Y1, atol=1e-6)
    assert abs(i1["residual"] - i2["residual"]) < 1e-10
    # one svd step is the reference rank-r refill, from any start
    S0 = np.where(comp.mask, 0.0, np.random.default_rng(1).standard_normal(
        comp.P.shape))
    S_step = E.hard_impute(comp, rank, S0, tol=1e-300, max_iter=1)[0]
    Zr = _svd_rank(np.where(mask, X, 0.0) + comp.to_real(S0), rank)
    np.testing.assert_allclose(comp.to_real(S_step), np.where(mask, 0.0, Zr),
                               atol=1e-10)


def test_hard_impute_recovers_exact_low_rank():
    X = _low_rank(n=50, N=10, r=2, noise=0.0, seed=5)
    rng = np.random.default_rng(6)
    mask = rng.uniform(size=X.shape) < 0.6
    mask[:, :3] = True                      # >= 3 > r cells per row
    prob = E.Completion(X, mask)
    for solver in E.HARD_SOLVERS:
        S, info = E.hard_impute(prob, 2, tol=1e-26, max_iter=50000,
                                solver=solver)
        assert info["converged"], solver
        Y = prob.to_real(S)
        np.testing.assert_allclose(Y[~mask], X[~mask], atol=1e-6)


def test_hard_impute_objective_decreases():
    """Both maps decrease the rank-r tail F of the filled matrix; for the
    svd map (a majorize-minimize step) F and the observed-cell residual
    of the rank-r fit decrease in turn."""
    X, mask = _low_rank(seed=8), _mixed_mask(seed=8)
    prob = E.Completion(X, mask)
    for solver in E.HARD_SOLVERS:
        S, prev = None, np.inf
        for it in range(15):
            S, info = E.hard_impute(prob, 2, S, tol=1e-30, max_iter=3,
                                    solver=solver)
            f = E._hard_map_svd(prob, S, 2)[1]   # F at the returned point
            assert f <= prev * (1 + 1e-12) + 1e-12
            prev = f
    S = prob.zeros()
    for it in range(30):
        S1, f, resid = E._hard_map_svd(prob, S, 2)
        f1 = E._hard_map_svd(prob, S1, 2)[1]
        assert f1 <= resid * prob.obs2 * (1 + 1e-12) + 1e-12
        assert resid * prob.obs2 <= f * (1 + 1e-12) + 1e-12
        S = S1
    # the tail objective at a point is the sum of the smallest eigenvalues
    S = prob.zeros()
    F = prob.P + S
    e = np.linalg.eigvalsh(F.T @ F)
    for step in (E._hard_map_svd, E._hard_map_lsq):
        assert abs(step(prob, S, 3)[1] - e[:-3].sum()) < 1e-8
    with pytest.raises(ValueError):
        E.hard_impute(prob, 2, solver="squarem")


def test_hard_impute_bounded_lsq_diverges():
    """Noisy rank-3 100 x 12 matrix, 40% of the cells missing, rank 1.
    Hard-Impute (svd) converges to a finite fixed point and its iterates
    stay bounded; that point is also a fixed point of the exact
    least-squares refill, yet the refill started from the same zero fill
    lowers F below it while its missing cells run off (the defect of the
    earlier solver: a monotone F does not bound the iterates)."""
    g = np.random.default_rng(11)
    X = g.standard_normal((100, 3)) @ g.standard_normal((3, 12)) / math.sqrt(3)
    X = X + 0.3 * g.standard_normal((100, 12))
    mask = np.random.default_rng(12).uniform(size=(100, 12)) < 0.6
    prob = E.Completion(X, mask)
    x_max = np.abs(X[mask]).max()
    S, peak = prob.zeros(), 0.0
    for _ in range(40):                       # 40 x 100 steps, bounded
        S, info = E.hard_impute(prob, 1, S, tol=1e-24, max_iter=100)
        peak = max(peak, np.abs(prob.to_real(S)).max())
        if info["converged"]:
            break
    assert info["converged"] and peak < 2.0 * x_max
    for step in (E._hard_map_svd, E._hard_map_lsq):
        S1 = step(prob, S, 1)[0]
        assert np.linalg.norm(S1 - S) < 1e-6 * np.linalg.norm(S)
    f_fix = E._hard_map_svd(prob, S, 1)[1]
    S_l, info_l = E.hard_impute(prob, 1, tol=1e-24, max_iter=300,
                                solver="lsq")
    assert not info_l["converged"]
    assert np.abs(prob.to_real(S_l)).max() > 20.0 * x_max
    assert E._hard_map_svd(prob, S_l, 1)[1] < f_fix


def test_row_conditional():
    X, mask = _low_rank(), _mixed_mask()
    prob = E.Completion(X, mask)
    Sig = _factor_corr(X.shape[1], seed=2)
    Y = prob.to_real(prob.row_conditional(Sig))
    Xo = np.where(mask, X, 0.0)
    for i in (0, 25, 50):
        o, u = np.flatnonzero(mask[i]), np.flatnonzero(~mask[i])
        ref = Xo[i, o] @ np.linalg.solve(Sig[np.ix_(o, o)], Sig[np.ix_(o, u)])
        np.testing.assert_allclose(Y[i, u], ref, atol=1e-10)
        assert np.all(Y[i, o] == 0.0)


def test_compression_maps():
    X, mask = _low_rank(), _mixed_mask()
    prob = E.Completion(X, mask)
    # Q has orthonormal column blocks; to_pseudo inverts to_real on the
    # compressed form
    np.testing.assert_allclose(prob.Q.T @ prob.Q, np.eye(prob.n_ps),
                               atol=1e-12)
    S = np.where(prob.mask, 0.0, np.random.default_rng(0).standard_normal(
        prob.P.shape))
    np.testing.assert_allclose(prob.to_pseudo(prob.to_real(S)), S,
                               atol=1e-12)
    # Gram matrix of the pseudo-rows = Gram of the observed real cells
    Xo = np.where(mask, X, 0.0)
    np.testing.assert_allclose(prob.P.T @ prob.P, Xo.T @ Xo, atol=1e-9)
    # A row observing nothing has no pseudo-row
    mask2 = mask.copy()
    mask2[0] = False
    prob2 = E.Completion(X, mask2)
    assert np.all(prob2.Q[0] == 0.0)


# ---------------------------------------------------------------------------
# Scored cells and the Gaussian reference
# ---------------------------------------------------------------------------

def _val_data(M=40, N=10, seed=0, obs=0.7):
    rng = np.random.default_rng(seed)
    R = _factor_corr(N, seed=seed)
    X = rng.multivariate_normal(np.zeros(N), R, size=M) * 0.1 + 0.5
    O = (rng.uniform(size=(M, N)) < obs).astype(np.float64)
    return R, X * O, O


@pytest.mark.parametrize("obs", [0.6, 1.0])
def test_gaussian_reference_matches_impute_and_evaluate(obs):
    R, B, O = _val_data(obs=obs, seed=int(obs * 10))
    N = B.shape[1]
    col_mean, col_std = np.full(N, 0.5), np.full(N, 0.1)
    Z = E.standardize(B, O, col_mean, col_std)
    for sel in ([2], [0, 5], [1, 4, 7, 9]):
        imp = impute_and_evaluate(B, O, sel, R, col_mean, col_std, N)
        ref = E.pooled_r2(Z, E.gaussian_predict(Z, O, sel, R),
                          E.scored_mask(O, sel))
        assert abs(imp["r2"] - ref["r2"]) < 1e-12
        assert abs(imp["rmse"] - ref["rmse"]) < 1e-12


def test_scored_mask_every_cell_of_every_model():
    """Validation model 0 observes a subset of the others' columns: the
    configuration where impute_and_evaluate's fast path (H1) scores only
    model 0's columns.  scored_mask keeps every observed unselected cell
    of every model that observes a selected benchmark."""
    O = np.ones((4, 6))
    O[0, 5] = 0.0                    # model 0 misses column 5
    O[3, [0, 1]] = 0.0               # model 3 observes no selected column
    sel = [0, 1]
    mask = E.scored_mask(O, sel)
    assert mask[:3, 2:].sum() == 3 * 4 - 1
    assert mask[1, 5] and mask[2, 5] and not mask[0, 5]
    assert not mask[3].any() and not mask[:, sel].any()
    # the clip of impute_and_evaluate
    Z = E.standardize(np.array([[100.0, 0.5]]), np.ones((1, 2)),
                      np.array([0.5, 0.5]), np.array([0.1, 0.1]))
    assert Z[0, 0] == 10.0 and Z[0, 1] == 0.0


def test_pooled_r2_definition():
    t = np.array([[1.0, -2.0], [0.5, 3.0]])
    p = np.array([[0.5, -1.0], [9.0, 2.0]])
    mask = np.array([[True, True], [False, True]])
    out = E.pooled_r2(t, p, mask)
    ss_res = 0.25 + 1.0 + 1.0
    ss_tot = 1.0 + 4.0 + 9.0
    assert out["n_eval"] == 3
    assert abs(out["r2"] - (1 - ss_res / ss_tot)) < 1e-15
    assert abs(out["rmse"] - math.sqrt(ss_res / 3)) < 1e-15
    assert np.isnan(E.pooled_r2(t, p, np.zeros((2, 2), bool))["r2"])


# ---------------------------------------------------------------------------
# Inner holdout, seed tag, leakage
# ---------------------------------------------------------------------------

def _seed_tags_of_other_scripts():
    """{tag offset: {file, ...}} of the other scripts in code/: the values
    of *_SEED_OFFSET constants and the n of 'SEED + n' (code and docs)."""
    code = pathlib.Path(E.__file__).resolve().parent
    used = {}
    for path in sorted(code.glob("*.py")):
        if path.name == "eval_imputers.py" or path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        hits = [int(m.group(1)) for m in re.finditer(
            r"^\s*[A-Z_]*SEED_OFFSET\s*=\s*(\d+)\b", text, re.M)]
        hits += [int(m.group(1)) for m in re.finditer(
            r"\bSEED\s*\+\s*(\d+)\b", text)]
        for tag in hits:
            used.setdefault(tag, set()).add(path.name)
    return used


def test_inner_seed_tag_unique_in_code_tree():
    used = _seed_tags_of_other_scripts()
    assert {RANDOM_SEED_OFFSET, ENTROPY_RF_SEED_OFFSET} <= set(used)  # scan
    assert E.INNER_SEED_OFFSET == 21
    assert E.INNER_SEED_OFFSET not in used, (
        f"seed tag SEED + {E.INNER_SEED_OFFSET} also used by "
        f"{sorted(used.get(E.INNER_SEED_OFFSET, ()))}")


def test_inner_holdout():
    rng = np.random.default_rng(0)
    O = (rng.uniform(size=(30, 8)) < 0.5).astype(float)
    perm = rng.permutation(30)
    for sel in ([0], [1, 2], [3, 4, 5]):
        h = E.inner_holdout(O, sel, perm, 0.2)
        cand = np.flatnonzero(O[:, sel].any(axis=1))
        assert h.size == math.ceil(0.2 * cand.size)
        assert set(h) <= set(cand) and np.all(np.diff(h) > 0)
        # the first candidates in permutation order
        order = [p for p in perm if p in set(cand)]
        assert set(h) == set(order[:h.size])
    # fully observed: the same models for every selection
    Of = np.ones((30, 8))
    assert np.array_equal(E.inner_holdout(Of, [0], perm),
                          E.inner_holdout(Of, [5, 6], perm))


def _split_problem(seed=0, M_tr=40, M_val=10, N=9, obs=0.8):
    rng = np.random.default_rng(seed)
    R = _factor_corr(N, seed=seed)
    Z = rng.multivariate_normal(np.zeros(N), R, size=M_tr + M_val)
    O = rng.uniform(size=Z.shape) < obs
    O[:, 0] = True
    return np.where(O, Z, 0.0), O, M_tr


FIT_KW = dict(imputers=("softimpute", "hardimpute"), inner_frac=0.25,
              patience=2, tol=1e-10, max_iter=500, min_improve=0.0)


def test_path_patience_and_null_start():
    seq = [0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.9]

    def fit(v, S):
        return np.array([float(v)]), {"iters": 1, "converged": True}

    def score(S):
        return seq[int(S[0])]

    vals = list(range(len(seq)))
    # flat start: the null-first path keeps going until it improves
    g, sc, best, it, nc, info = E._path(fit, vals, score, 2, 1e-4,
                                        null_first=True)
    assert g == 3 and best[0] == 3 and np.isnan(sc[6]) and it == 6
    assert info == {"iters": 1, "converged": True} and nc == 0
    g, sc, *_ = E._path(fit, vals, score, 2, 1e-4)
    assert g == 0 and np.isnan(sc[3])
    g, sc, *_ = E._path(fit, vals, score, 0, 1e-4)          # full path
    assert g == 6 and np.isfinite(sc).all()
    # improvements below min_improve keep the earlier value
    g, *_ = E._path(fit, [0, 3, 4], lambda S: [0.0, 0.5, 0.50005][
        [0, 3, 4].index(int(S[0]))], 0, 1e-4)
    assert g == 1


def test_hard_rank_cap():
    Z, O, n_tr = _split_problem(seed=4)
    perm = np.random.default_rng(5).permutation(n_tr)
    kw = dict(FIT_KW, imputers=("hardimpute",))
    out = E.fit_set(Z, O, n_tr, (0, 2), perm, E.lambda_max(Z[:n_tr]),
                    E.lambda_ratios(4, 1e-2), [1, 2, 3, 4], **kw)
    sc = out["hardimpute"]["inner_r2"]
    assert len(sc) == 2 and out["hardimpute"]["value"] <= 2
    out = E.fit_set(Z, O, n_tr, (0, 2), perm, E.lambda_max(Z[:n_tr]),
                    E.lambda_ratios(4, 1e-2), [1, 2, 3, 4], rank_cap=False,
                    **dict(kw, patience=0))
    assert len(out["hardimpute"]["inner_r2"]) == 4


def test_hard_starts_at_softimpute_fits(monkeypatch):
    """hard_init "soft": every rank of the inner HardImpute path starts at
    the inner SoftImpute fit at lambda*, the refit at the SoftImpute refit
    (no warm start up the rank path); "zero": the zero fill.  The
    SoftImpute path runs for the start even when it is not reported."""
    Z, O, n_tr = _split_problem(seed=6, obs=0.7)
    perm = np.random.default_rng(7).permutation(n_tr)
    args = (Z, O, n_tr, (0, 2, 5), perm, E.lambda_max(Z[:n_tr]),
            E.lambda_ratios(6, 1e-2), [1, 2, 3])
    soft_out, hard_in = [], []
    soft0, hard0 = E.soft_impute, E.hard_impute

    def soft_rec(prob, lam, S0=None, *a, **kw):
        S, info = soft0(prob, lam, S0, *a, **kw)
        soft_out.append(S.copy())
        return S, info

    def hard_rec(prob, rank, S0=None, *a, **kw):
        hard_in.append(None if S0 is None else S0.copy())
        return hard0(prob, rank, S0, *a, **kw)

    monkeypatch.setattr(E, "soft_impute", soft_rec)
    monkeypatch.setattr(E, "hard_impute", hard_rec)
    out = E.fit_set(*args, **dict(FIT_KW, patience=0))
    g = out["softimpute"]["idx"]
    assert len(hard_in) == 3 + 1                       # 3 ranks + refit
    for S0 in hard_in[:-1]:
        np.testing.assert_array_equal(S0, soft_out[g])
    np.testing.assert_array_equal(hard_in[-1], soft_out[-1])
    assert np.abs(soft_out[g]).max() > 0                # lambda* < lambda_max
    # HardImpute alone: the same fits, SoftImpute not reported
    soft_out.clear(), hard_in.clear()
    out_h = E.fit_set(*args, **dict(FIT_KW, patience=0,
                                    imputers=("hardimpute",)))
    assert "softimpute" not in out_h and soft_out
    np.testing.assert_array_equal(out_h["hardimpute"]["pred"],
                                  out["hardimpute"]["pred"])
    hard_in.clear()
    E.fit_set(*args, **dict(FIT_KW, patience=0, imputers=("hardimpute",),
                            hard_init="zero"))
    assert all(np.all(S0 == 0.0) for S0 in hard_in)
    with pytest.raises(ValueError):
        E.fit_set(*args, **dict(FIT_KW, hard_init="warm"))


def test_fit_set_hardimpute_bounded():
    """Every HardImpute prediction of fit_set stays within a small multiple
    of the inputs' range, and the reported fit's convergence is reported."""
    for seed in range(3):
        Z, O, n_tr = _split_problem(seed=seed, obs=0.6)
        perm = np.random.default_rng(seed).permutation(n_tr)
        for sel in [(0,), (0, 3), (0, 2, 4, 6)]:
            out = E.fit_set(Z, O, n_tr, sel, perm, E.lambda_max(Z[:n_tr]),
                            E.lambda_ratios(6, 1e-2), [1, 2, 3, 4],
                            **dict(FIT_KW, hard_max_iter=3000,
                                   hard_tol=1e-12))
            h = out["hardimpute"]
            assert isinstance(h["converged"], bool)
            assert np.abs(h["pred"]).max() < 5.0 * np.abs(Z).max()


def test_fit_set_no_leakage_of_unselected_validation_cells():
    Z, O, n_tr = _split_problem()
    sel = (0, 4)
    perm = np.random.default_rng(1).permutation(n_tr)
    lam_max = E.lambda_max(Z[:n_tr])
    ratios = E.lambda_ratios(6, 1e-2)
    out1 = E.fit_set(Z, O, n_tr, sel, perm, lam_max, ratios, [1, 2, 3],
                     **FIT_KW)
    # Change every unselected validation cell: no fit, no inner score and
    # no hyperparameter may change; the predictions are identical
    Z2 = Z.copy()
    rest = np.setdiff1d(np.arange(Z.shape[1]), sel)
    Z2[np.ix_(np.arange(n_tr, len(Z)), rest)] += 7.0
    out2 = E.fit_set(Z2, O, n_tr, sel, perm, lam_max, ratios, [1, 2, 3],
                     **FIT_KW)
    for name in ("softimpute", "hardimpute"):
        a, b = out1[name], out2[name]
        assert a["idx"] == b["idx"]
        np.testing.assert_array_equal(a["inner_r2"], b["inner_r2"])
        np.testing.assert_array_equal(a["pred"], b["pred"])
    # ... while the selected validation cells do enter the fit
    Z3 = Z.copy()
    Z3[n_tr:, sel[1]] += 3.0
    out3 = E.fit_set(Z3, O, n_tr, sel, perm, lam_max, ratios, [1, 2, 3],
                     **FIT_KW)
    assert not np.allclose(out1["softimpute"]["pred"],
                           out3["softimpute"]["pred"])


def test_fit_set_inner_cells_are_training_cells():
    Z, O, n_tr = _split_problem(seed=2)
    sel = (0, 3)
    perm = np.random.default_rng(3).permutation(n_tr)
    out = E.fit_set(Z, O, n_tr, sel, perm, E.lambda_max(Z[:n_tr]),
                    E.lambda_ratios(5, 1e-2), [1, 2], **FIT_KW)
    inner = E.inner_holdout(O[:n_tr], sel, perm, FIT_KW["inner_frac"])
    assert out["n_inner"] == inner.size and np.all(inner < n_tr)
    in_a = np.zeros(Z.shape[1], bool)
    in_a[list(sel)] = True
    assert out["n_inner_cells"] == int((O[inner] & ~in_a).sum())
    # paths: computed values finite, early stop leaves NaN, choice = argmax
    for name in ("softimpute", "hardimpute"):
        sc = out[name]["inner_r2"]
        fin = np.isfinite(sc)
        assert fin[0] and np.all(fin[:fin.sum()])
        assert out[name]["idx"] == int(np.argmax(np.where(fin, sc, -np.inf)))


# ---------------------------------------------------------------------------
# End to end on a small synthetic dataset
# ---------------------------------------------------------------------------

def _write_dataset(path, seed=0, M=60, N=10, perturb_rows=None):
    rng = np.random.default_rng(seed)
    R = _factor_corr(N, seed=seed)
    X = rng.multivariate_normal(np.zeros(N), R, size=M) * 0.1 + 0.5
    O = (rng.uniform(size=(M, N)) < 0.8).astype(np.float64)
    O[:, 0] = 1.0
    if perturb_rows is not None:
        X[perturb_rows] = rng.uniform(0, 1, size=(len(perturb_rows), N))
    np.savez(path / "synth.matrix.npz", B=X * O, O=O,
             model_names=np.array([f"m{i}" for i in range(M)]),
             benchmark_names=np.array([f"b{j}" for j in range(N)]))
    (path / "synth.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [None, None]}}))


@pytest.fixture
def synth_dirs(tmp_path, monkeypatch):
    data, logs, figs = (tmp_path / d for d in ("data", "logs", "figures"))
    for d in (data, logs, figs):
        d.mkdir()
    monkeypatch.setattr(E, "DATA", data)
    monkeypatch.setattr(E, "LOGS", logs)
    monkeypatch.setattr(E, "OUT", figs)
    return data, logs, figs


RUN_KW = dict(ks=[1, 2, 3, 5], n_folds=3, n_repeats=1, n_lambda=6,
              lambda_min_ratio=1e-2, ranks=[1, 2, 3], inner_frac=0.25,
              patience=2, tol=1e-9, max_iter=300, hard_max_iter=200)


def test_end_to_end(synth_dirs):
    data, logs, figs = synth_dirs
    _write_dataset(data)
    arr = E.run_dataset("synth", **RUN_KW)
    S_n = 3
    assert arr["split_id"].tolist() == [0, 1, 2]
    for m in E.METHODS:
        for imp in E.IMPUTERS:
            assert arr[f"{m}_{imp}_r2"].shape == (S_n, 4)
            assert np.isfinite(arr[f"{m}_{imp}_r2"]).all()
        np.testing.assert_allclose(arr[f"{m}_gaussian_r2"],
                                   arr[f"{m}_gaussian_ref_r2"], atol=1e-12)
    assert np.nanmax(arr["gaussian_check_maxdiff"]) < 1e-12

    # Selections and Gaussian R² are eval_baselines' (same estimator,
    # eligibility, common random numbers)
    d = np.load(data / "synth.matrix.npz")
    B, O = d["B"], d["O"]
    names = d["benchmark_names"]
    for i, s in enumerate(cv_splits(B.shape[0], 3, 1, E.HOLDOUT_FRAC,
                                    E.SEED)):
        R, mu, sd = estimate_corr(B[s.train_idx], O[s.train_idx])
        el = eligible_mask(O[s.train_idx])
        for m in E.METHODS:
            res = run_selection(m, R, 5, names, s, el)
            for j, k in enumerate(RUN_KW["ks"]):
                sel = selection_at_k(res, k)
                imp = impute_and_evaluate(B[s.val_idx], O[s.val_idx], sel, R,
                                          mu, sd, B.shape[1])
                assert arr[f"{m}_gaussian_r2"][i, j] == imp["r2"]
                assert arr[f"{m}_n_eval"][i, j] == int(E.scored_mask(
                    O[s.val_idx], sel).sum())

    # Rerun: bit-identical (common random numbers, deterministic solvers)
    arr2 = E.run_dataset("synth", **RUN_KW)
    for key, val in arr.items():
        if key == "split_seconds":
            continue
        assert np.array_equal(val, arr2[key], equal_nan=val.dtype.kind == "f"
                              ), key

    # Perturbing the validation models of split 0 leaves every train-side
    # quantity of split 0 unchanged (selections, inner holdout sizes)
    s0 = next(cv_splits(60, 3, 1, E.HOLDOUT_FRAC, E.SEED))
    _write_dataset(data, perturb_rows=s0.val_idx)
    arr3 = E.run_dataset("synth", **RUN_KW)
    for m in E.METHODS:
        for key in ("selected", "n_inner"):
            np.testing.assert_array_equal(arr[f"{m}_{key}"][0],
                                          arr3[f"{m}_{key}"][0])
    assert arr["lambda_max"][0] == arr3["lambda_max"][0]
    assert not np.allclose(arr["trace_softimpute_r2"][0],
                           arr3["trace_softimpute_r2"][0])

    # Paired rows, summaries, macros, outputs (the pairs of the pooled
    # methods first, then those with facloc)
    paired = E.paired_rows("synth", arr)
    comps = {r["comparison"] for r in paired}
    assert comps == {"imputer", "method", "interaction"}
    assert all(set(STAT_FIELDS) <= set(r) for r in paired)
    pairs = (orient(list(E.POOLED_METHODS))
             + [p for p in orient(E.METHODS) if "facloc" in p])
    assert sorted(pairs) == sorted(orient(E.METHODS))
    meth = [(r["method_a"], r["method_b"]) for r in paired
            if r["comparison"] == "method" and r["k"] == 1
            and r["imputer_a"] == "gaussian"]
    assert meth == pairs
    j = 1
    for r in paired:
        if r["comparison"] == "imputer" and r["k"] == 2:
            a = arr[f"{r['method_a']}_{r['imputer_a']}_r2"][:, j]
            b = arr[f"{r['method_a']}_{r['imputer_b']}_r2"][:, j]
            assert r["method_a"] == r["method_b"]
            assert abs(r["mean_diff"] - np.mean(a - b)) < 1e-12
        if (r["comparison"] == "interaction" and r["k"] == 2
                and r["imputer_a"] == "softimpute"):
            ma, mb = r["method_a"], r["method_b"]
            did = ((arr[f"{ma}_softimpute_r2"][:, j]
                    - arr[f"{mb}_softimpute_r2"][:, j])
                   - (arr[f"{ma}_gaussian_r2"][:, j]
                      - arr[f"{mb}_gaussian_r2"][:, j]))
            assert abs(r["mean_diff"] - np.mean(did)) < 1e-12
    tables, macros = E.summarize("synth", arr, paired)
    assert macros and all(name.isalpha() for name in macros)
    assert all(np.isfinite(v) for v in macros.values())
    assert "RsqImpSoftSynthTraceKfive" in macros
    assert "PairDiffImpSynthMiSoftVsGaussKtwo" in macros
    assert macros["PairMedianImpSynthMiSoftVsGaussKtwo"] == pytest.approx(
        np.median(arr["mi_softimpute_r2"][:, 1] - arr["mi_gaussian_r2"][:, 1]))
    assert "ImpExplodeHardSynth" in macros
    assert "RsqMedianImpSoftSynthTraceKfive" in macros
    assert "PairDiffImpHardSynthMiVsTraceKone" in macros
    assert "PairDiffImpInterSoftSynthMiVsTraceKthree" in macros
    assert macros["RsqImpGaussSynthMiKone"] == pytest.approx(
        np.mean(arr["mi_gaussian_r2"][:, 0]))
    for name in ("RsqImpSoftSynthFaclocKfive",
                 "PairDiffImpHardSynthMiVsFaclocKone",
                 "PairDiffImpSynthFaclocSoftVsGaussKtwo",
                 "PairDiffImpInterSoftSynthFaclocVsRandomKthree",
                 "ImpHardRankSynthFaclocKfive",
                 "ImpRankGaussSynthFaclocKone"):
        assert name in macros, name
    rank = macros["ImpRankGaussSynthFaclocKone"]
    assert 1 <= rank <= len(E.POOLED_METHODS) + 1
    E.print_summary("synth", arr, tables, paired)
    E.plot_dataset("synth", arr, paired)
    E.save("synth", arr, tables, macros, paired, {"test": True})
    doc = json.loads((logs / "imputers_synth.json").read_text())
    assert doc["macros"] == macros
    for f in ("imputers_synth.npz", "imputers_synth_paired.csv"):
        assert (logs / f).is_file()
    assert (figs / "imputers_synth.pdf").is_file()


def _rows_equal(a, b):
    """Paired rows equal field by field (NaN == NaN)."""
    if a.keys() != b.keys():
        return False
    return all(a[f] == b[f] or (isinstance(a[f], float)
                                and isinstance(b[f], float)
                                and math.isnan(a[f]) and math.isnan(b[f]))
               for f in a)


def test_facloc_is_facility_location_and_changes_nothing_else(synth_dirs):
    """facloc's selections are baseline_select.facility_location_select on
    the split's correlation estimate and eligible mask; appending facloc
    leaves every array of a run of the pooled methods bit-identical
    (n_sets and gaussian_check_maxdiff included), its paired rows a prefix
    and every macro unchanged."""
    data, logs, figs = synth_dirs
    _write_dataset(data, seed=6)
    assert E.METHODS == list(E.POOLED_METHODS) + ["facloc"]
    arr = E.run_dataset("synth", **RUN_KW)
    old = E.run_dataset("synth", methods=list(E.POOLED_METHODS), **RUN_KW)
    d = np.load(data / "synth.matrix.npz")
    B, O, names = d["B"], d["O"], d["benchmark_names"]
    k_max = int(arr["k_max"])
    for i, s in enumerate(cv_splits(B.shape[0], 3, 1, E.HOLDOUT_FRAC,
                                    E.SEED)):
        R, _, _ = estimate_corr(B[s.train_idx], O[s.train_idx])
        el = eligible_mask(O[s.train_idx])
        ref = facility_location_select(R, k_max, benchmark_names=names,
                                       eligible=el)["selected"]
        assert arr["facloc_selected"][i].tolist() == (
            list(ref) + [-1] * (k_max - len(ref)))
    for key, val in old.items():
        if key in ("split_seconds", "methods", "n_sets_all",
                   "gaussian_check_maxdiff_all"):   # every method's: new
            continue
        assert np.array_equal(val, arr[key], equal_nan=val.dtype.kind == "f"
                              ), key
    assert _strs(arr["methods"]) == _strs(old["methods"]) + ["facloc"]
    assert np.array_equal(old["n_sets_all"], old["n_sets"])
    assert (arr["n_sets_all"] >= arr["n_sets"]).all()
    p_old, p_new = E.paired_rows("synth", old), E.paired_rows("synth", arr)
    assert len(p_new) > len(p_old)
    assert all(_rows_equal(a, b) for a, b in zip(p_old, p_new))
    assert all("facloc" in (r["method_a"], r["method_b"])
               for r in p_new[len(p_old):])
    t_old, m_old = E.summarize("synth", old, p_old)
    t_new, m_new = E.summarize("synth", arr, p_new)
    assert {k: m_new[k] for k in m_old} == m_old
    assert any("Facloc" in k for k in set(m_new) - set(m_old))
    for imp, by_k in t_old["method_ranks"].items():
        for k, ranks in by_k.items():
            new = dict(t_new["method_ranks"][imp][k])
            r_fl = new.pop("facloc")
            assert r_fl is None or 1 <= r_fl <= len(ranks) + 1
            assert new == ranks


def _strs(a):
    return [str(x) for x in np.asarray(a).ravel()]


def test_main_and_from_logs(synth_dirs):
    data, logs, figs = synth_dirs
    _write_dataset(data, seed=4)
    argv = ["--datasets", "synth", "--repeats", "1", "--folds", "3",
            "--ks", "1", "3", "--methods", "mi", "trace", "random",
            "--n-lambda", "4", "--lambda-min-ratio", "0.01", "--ranks", "1",
            "2", "--patience", "1", "--max-iter", "200"]
    E.main(argv)
    doc = json.loads((logs / "imputers_synth.json").read_text())
    assert doc["config"]["methods"] == ["mi", "trace", "random"]
    assert doc["config"]["inner_seed_offset"] == 21
    E.main(argv + ["--from-logs"])
    doc2 = json.loads((logs / "imputers_synth.json").read_text())
    assert doc2["macros"] == doc["macros"]
    assert doc2["config"]["from_logs"] is True


def test_shards_merge_to_the_single_process_log(synth_dirs):
    """--reps r shards (one per repetition, in LOGS/imputers_shards/, out
    of every LOGS/imputers_*.npz glob) and --merge reproduce the single
    process run: every array bit for bit (split_seconds is wall clock),
    the paired CSV and the macros (ImpRuntimeMin aside)."""
    data, logs, figs = synth_dirs
    _write_dataset(data, seed=5)
    argv = ["--datasets", "synth", "--repeats", "2", "--folds", "3",
            "--ks", "1", "3", "--methods", "mi", "trace", "random",
            "--n-lambda", "4", "--lambda-min-ratio", "0.01", "--ranks", "1",
            "2", "--patience", "1", "--max-iter", "200"]
    E.main(argv)
    single = dict(np.load(logs / "imputers_synth.npz", allow_pickle=True))
    doc = json.loads((logs / "imputers_synth.json").read_text())
    csv = (logs / "imputers_synth_paired.csv").read_text()
    for f in ("imputers_synth.npz", "imputers_synth.json",
              "imputers_synth_paired.csv"):
        (logs / f).unlink()
    E.main(argv + ["--reps", "1"])
    E.main(argv + ["--reps", "0"])
    assert not (logs / "imputers_synth.npz").exists()
    assert sorted(p.name for p in (logs / E.SHARD_DIR).iterdir()) == [
        "imputers_synth_rep0.npz", "imputers_synth_rep1.npz"]
    assert sorted(p.name for p in logs.glob("imputers_*.npz")) == []
    E.main(argv + ["--merge"])
    merged = dict(np.load(logs / "imputers_synth.npz", allow_pickle=True))
    assert sorted(merged) == sorted(single)
    for key in single:
        if key == "split_seconds":
            continue
        a, b = single[key], merged[key]
        assert a.shape == b.shape and a.dtype == b.dtype, key
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), key
    doc2 = json.loads((logs / "imputers_synth.json").read_text())
    m1 = {k: v for k, v in doc["macros"].items() if "RuntimeMin" not in k}
    m2 = {k: v for k, v in doc2["macros"].items() if "RuntimeMin" not in k}
    assert m1 == m2
    c1 = dict(doc["config"], runtime_s=None)
    c2 = dict(doc2["config"], runtime_s=None)
    assert c1 == c2
    assert (logs / "imputers_synth_paired.csv").read_text() == csv
    # a merge with other settings than the shards' is refused
    with pytest.raises(SystemExit):
        E.main([a if a != "200" else "300" for a in argv] + ["--merge"])
    (logs / E.SHARD_DIR / "imputers_synth_rep1.npz").unlink()
    with pytest.raises(ValueError):
        E.merge_shards("synth", 2)


# ---------------------------------------------------------------------------
# Split-level worker processes (--workers)
# ---------------------------------------------------------------------------

def _without_seconds(text):
    """stdout without its wall-clock fields (per-split and total
    seconds)."""
    return re.sub(r"\(\d+\.\d+s\)|in \d+ ?s\b", "", text)


def _assert_same_arrays(ref, got):
    """Every array of ref in got, bit for bit (split_seconds is wall
    clock), same keys."""
    assert sorted(got) == sorted(ref)
    for key, a in ref.items():
        if key == "split_seconds":
            continue
        b = got[key]
        assert a.shape == b.shape and a.dtype == b.dtype, key
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), key


def test_workers_match_the_serial_run(synth_dirs, capsys):
    """run_dataset with workers = 2 (the splits in spawned processes)
    reproduces workers = 1: every array bit for bit, in the same key
    order, and the same progress lines in split order (seconds aside)."""
    data, logs, figs = synth_dirs
    _write_dataset(data, seed=7)
    kw = dict(RUN_KW, n_repeats=2)
    capsys.readouterr()
    serial = E.run_dataset("synth", **kw)
    out1 = capsys.readouterr().out
    par = E.run_dataset("synth", workers=2, **kw)
    out2 = capsys.readouterr().out
    assert list(par) == list(serial)
    _assert_same_arrays(serial, par)
    assert out1.count("  Split ") == 6
    assert _without_seconds(out2) == _without_seconds(out1)


def test_workers_main_shards_and_merge(synth_dirs):
    """--workers 2 through main, as a single run and as --reps shards
    (then --merge without --workers): the npz bit for bit, the paired CSV,
    the tables and macros of the serial run (ImpRuntimeMin aside); the
    config differs in workers only (a merge's: its shards')."""
    data, logs, figs = synth_dirs
    _write_dataset(data, seed=8)
    argv = ["--datasets", "synth", "--repeats", "2", "--folds", "3",
            "--ks", "1", "3", "--methods", "mi", "trace", "random",
            "facloc", "--n-lambda", "4", "--lambda-min-ratio", "0.01",
            "--ranks", "1", "2", "--patience", "1", "--max-iter", "200"]

    def logs_of_run():
        npz = dict(np.load(logs / "imputers_synth.npz", allow_pickle=True))
        doc = json.loads((logs / "imputers_synth.json").read_text())
        csv = (logs / "imputers_synth_paired.csv").read_text()
        return npz, doc, csv

    def stable(doc):
        return ({k: v for k, v in doc["macros"].items()
                 if "RuntimeMin" not in k},
                dict(doc["config"], runtime_s=None, workers=None),
                doc["tables"])

    E.main(argv)
    ref = logs_of_run()
    assert ref[1]["config"]["workers"] == 1
    E.main(argv + ["--workers", "2"])
    direct = logs_of_run()
    E.main(argv + ["--reps", "0", "1", "--workers", "2"])
    for rep in (0, 1):
        with np.load(E.shard_path("synth", rep)) as d:
            assert int(d["shard_workers"]) == 2
    E.main(argv + ["--merge"])
    merged = logs_of_run()
    for (npz, doc, csv), workers in ((direct, 2), (merged, 2)):
        _assert_same_arrays(ref[0], npz)
        assert csv == ref[2]
        assert stable(doc) == stable(ref[1])
        assert doc["config"]["workers"] == workers


def test_workers_refuse_a_protocol_override(synth_dirs, monkeypatch):
    """A test-only protocol override is module state of this process, which
    a spawned worker does not see: the workers refuse to start instead of
    computing other rows than the serial run."""
    import score_bounds
    data, logs, figs = synth_dirs
    _write_dataset(data, seed=9)
    monkeypatch.setattr(score_bounds, "PRED_CLIP", False)
    with pytest.raises(RuntimeError):
        E.run_dataset("synth", workers=2, **RUN_KW)
