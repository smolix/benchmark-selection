"""Tests for eval_item_budget.py (whole tasks vs random items under a
matched question budget).

Synthetic data only (a few seconds): item layout and decoding, the trace
knapsack (eval_cost_budget.budgeted_select(..., "trace"), which the
script uses: budget, eligibility, equal-cost agreement with
greedy_trace_select, best-singleton fallback, and agreement with the
script's former own implementation, kept here as an oracle), the
facility-location knapsack bench_facloc (equal task sizes reproduce
facility_location_select; adding it leaves every row of the other methods
bit-identical), the Gaussian
imputation (agreement with impute_and_evaluate, the information form of
the posterior), the finite-population noise variance (exact enumeration),
the centered item estimator of items_center (explicit double centering,
exactness for additive item scores, the shared-error diagnostic), the
uniqueness of the item sample's seed tag in the code tree, common random
numbers of the item sample, like-for-like paired rows, and no leakage of
validation data into selections, item samples, noise models or
predictions.  Fixed per-task overhead (--fixed-cost): the skip-and-
continue item walk (step-by-step oracle), level 0 = the plain run, true
costs within budget, the overhead share, the crossover bracket (grid
values only; no crossing below the cheapest affordable whole task, where
the whole-task R² is exactly 0, unless the item R² is negative there) and
the default grid with each level's cheapest whole task.
"""

from __future__ import annotations

import atexit
import itertools
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
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-item-budget-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

import eval_item_budget as E
from baseline_select import facility_location_select, greedy_trace_select
from cv_split import (ENTROPY_RF_SEED_OFFSET, RANDOM_SEED_OFFSET, cv_splits,
                      split_seed)
from eval_cost_budget import budgeted_select
from eval_entropy_vs_mi import estimate_corr, impute_and_evaluate
from greedy_select import TIE_RTOL, _eligible, _near_max


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SIZES = (10, 12, 15, 20, 30, 8, 40)
NONBINARY = (5,)


def _factor_corr(N=20, n_factors=3, seed=0):
    """Random factor-model correlation matrix with an exactly unit diagonal."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, n_factors))
    S = W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))
    s = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return R


def _item_data(scores, sizes, binary):
    """ItemData of item scores (n_items, M) in task blocks of `sizes`."""
    N = len(sizes)
    M = scores.shape[1]
    n_items = np.array(sizes, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_items)])
    item_task = np.repeat(np.arange(N), n_items)
    B = (np.add.reduceat(scores, offsets[:-1], axis=0) / n_items[:, None]).T
    return E.ItemData(B, np.ones_like(B), np.array([f"t{t}" for t in range(N)]),
                      np.array([f"m{m:03d}" for m in range(M)]), scores,
                      item_task, offsets, n_items, np.asarray(binary))


def _synthetic(M=80, sizes=SIZES, nonbinary=NONBINARY, seed=0):
    """ItemData of a logistic one-factor item model: binary items, plus
    continuous items (probabilities) for the tasks in `nonbinary`."""
    rng = np.random.default_rng(seed)
    N = len(sizes)
    ability = rng.standard_normal(M)
    specific = 0.6 * rng.standard_normal((M, N))
    blocks = []
    for t, n in enumerate(sizes):
        diff = rng.normal(0.0, 1.0, n)
        p = 1.0 / (1.0 + np.exp(-(ability[:, None] + specific[:, [t]]
                                  - diff[None, :])))
        x = p if t in nonbinary else (rng.random((M, n)) < p).astype(float)
        blocks.append(x.T)
    binary = np.array([t not in nonbinary for t in range(N)])
    return _item_data(np.vstack(blocks), sizes, binary)


def _additive(M=60, sizes=SIZES, seed=4):
    """ItemData with exactly additive continuous item scores x_mi = a_mt +
    b_i (model-task effect plus item effect, all in (0, 1)): the
    double-centered residuals vanish."""
    rng = np.random.default_rng(seed)
    N = len(sizes)
    F = rng.standard_normal((M, 2))
    W = rng.standard_normal((2, N))
    a = np.clip(0.5 + 0.08 * (F @ W) + 0.04 * rng.standard_normal((M, N)),
                0.15, 0.85)
    blocks = [(a[:, [t]] + rng.uniform(-0.1, 0.1, n)[None, :]).T
              for t, n in enumerate(sizes)]
    return _item_data(np.vstack(blocks), sizes, np.zeros(N, dtype=bool))


def _write_files(data, tmp_path, B=None, O=None):
    """data written in the matrix / items.npz layout of build_olv1items."""
    M = data.B.shape[0]
    bits = np.zeros(data.scores.shape, dtype=np.uint8)
    rows = np.flatnonzero(data.binary[data.item_task])
    bits[rows] = data.scores[rows].astype(np.uint8)
    fidx = np.flatnonzero(~data.binary[data.item_task])
    mat, items = tmp_path / "syn.matrix.npz", tmp_path / "syn.items.npz"
    np.savez(mat, B=data.B if B is None else B,
             O=data.O if O is None else O,
             model_names=data.model_names,
             benchmark_names=data.benchmark_names)
    np.savez(items, model_names=data.model_names,
             task_names=data.benchmark_names, task_offsets=data.offsets,
             task_n_items=data.n_items, task_binary=data.binary,
             item_task=data.item_task.astype(np.int16),
             correct_bits=np.packbits(bits, axis=1, bitorder="big"),
             float_item_index=fidx, float_scores=data.scores[fidx],
             n_models=np.int64(M))
    return mat, items


@pytest.fixture(scope="module")
def data():
    return _synthetic()


@pytest.fixture(scope="module")
def split(data):
    return next(iter(cv_splits(data.B.shape[0], 5, 1, 0.1, E.SEED)))


def _trace_value(R, sel):
    """tr R − tr R_{rest|A}: variance of all tasks explained by A."""
    N = R.shape[0]
    if not sel:
        return 0.0
    rest = np.setdiff1d(np.arange(N), sel)
    A = R[np.ix_(sel, sel)]
    C = R[np.ix_(rest, sel)]
    cond = R[np.ix_(rest, rest)] - C @ np.linalg.solve(A, C.T)
    return float(np.trace(R) - np.trace(cond))


ORACLE_DIAG_FLOOR = 1e-12


def _trace_knapsack_oracle(Sigma, costs, budget, eligible=None):
    """The script's former own trace knapsack (budgeted_trace_select),
    kept as an oracle for eval_cost_budget.budgeted_select(..., "trace").

    Cost-effective greedy on gain_j / cost_j, gain_j = sum_{i unselected}
    S_ij² / S_jj with S the conditional covariance given the picks
    (Schur-complement updates, greedy_trace_select's arithmetic), until no
    affordable eligible task with a positive gain is left; ratios within a
    relative TIE_RTOL tie and go to the lowest index.  Best singleton: the
    affordable eligible task of largest unconditional explained variance
    (near-ties within TIE_RTOL · max(best, s) to the lowest index) replaces
    the greedy set if it exceeds the greedy value by more than TIE_RTOL ·
    max(s, value), s = max_i Σ_ii.
    """
    costs = np.asarray(costs, dtype=float)
    N = Sigma.shape[0]
    el = _eligible(eligible, N)
    scale = float(np.max(np.diag(Sigma)))
    S = np.array(Sigma, dtype=np.float64)
    unselected = np.ones(N, dtype=bool)
    selected, tie_breaks = [], []
    spent = value = 0.0
    while True:
        unsel = np.flatnonzero(unselected)
        pool = unsel[spent + costs[unsel] <= budget]
        if el is not None:
            pool = pool[el[pool]]
        if len(pool) == 0:
            break
        d = np.diag(S)[pool]
        gain = (np.sum(S[np.ix_(unsel, pool)] ** 2, axis=0)
                / np.maximum(d, ORACLE_DIAG_FLOOR))
        gain[d <= ORACLE_DIAG_FLOOR] = 0.0
        ratio = gain / costs[pool]
        if np.max(ratio) <= 0:
            break
        tied = pool[_near_max(ratio, 0.0) & (ratio > 0)]
        j = int(tied[0])
        if len(tied) > 1:
            tie_breaks.append((len(selected), len(tied)))
        value += float(gain[np.searchsorted(pool, j)])
        spent += costs[j]
        selected.append(j)
        unselected[j] = False
        s_col = S[:, j].copy()
        S -= np.outer(s_col, s_col) / max(S[j, j], 1e-300)
        S[j, :] = 0.0
        S[:, j] = 0.0
    result = {"selected": selected, "singleton": False, "value": value,
              "tie_breaks": tie_breaks}
    afford = np.flatnonzero(costs <= budget)
    if el is not None:
        afford = afford[el[afford]]
    if len(afford):
        diag = np.diag(Sigma)[afford]
        g0 = (np.sum(np.asarray(Sigma)[:, afford] ** 2, axis=0)
              / np.maximum(diag, ORACLE_DIAG_FLOOR))
        g0[diag <= ORACLE_DIAG_FLOOR] = 0.0
        tied = afford[_near_max(g0, scale)]
        best = int(tied[0])
        best_val = float(g0[np.searchsorted(afford, best)])
        if best_val > value + TIE_RTOL * max(scale, value):
            result.update(selected=[best], singleton=True, value=best_val,
                          tie_breaks=[(0, len(tied))] if len(tied) > 1
                          else [])
    return result


def _trace(R, costs, budget, eligible=None):
    return budgeted_select(R, costs, budget, "trace", eligible=eligible)


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

def test_load_items_roundtrip(data, tmp_path):
    mat, items = _write_files(data, tmp_path)
    got = E.load_items(mat, items)
    assert np.array_equal(got.scores, data.scores)
    assert np.array_equal(got.B, data.B)
    assert np.array_equal(got.item_task, data.item_task)
    assert np.array_equal(got.binary, data.binary)


def test_load_items_rejects_inconsistent(data, tmp_path):
    B = data.B.copy()
    B[3, 2] += 1e-3
    mat, items = _write_files(data, tmp_path, B=B)
    with pytest.raises(ValueError, match="item means"):
        E.load_items(mat, items)
    O = data.O.copy()
    O[0, 0] = 0
    mat, items = _write_files(data, tmp_path, O=O)
    with pytest.raises(ValueError, match="unobserved"):
        E.load_items(mat, items)


# ---------------------------------------------------------------------------
# Trace knapsack (the shared budgeted_select; the oracle is the script's
# former own copy)
# ---------------------------------------------------------------------------

def test_bench_trace_uses_shared_budgeted_select():
    assert E.KNAP_OBJECTIVE["bench_trace"] == "trace"
    assert not hasattr(E, "budgeted_trace_select")   # no duplicate rule
    R = _factor_corr(18, seed=9)
    costs = np.random.default_rng(2).integers(1, 9, 18).astype(float)
    el = np.ones(18, dtype=bool)
    el[[1, 7]] = False
    for budget in (3.0, 11.0, 25.0, 200.0):
        got = E.bench_select("bench_trace", R, costs, budget, el, [])
        ref = _trace(R, costs, budget, eligible=el)
        assert got["selected"] == [int(j) for j in ref["selected"]]
        assert got["singleton"] == ref["singleton"]


def test_trace_knapsack_equal_costs_is_greedy_trace():
    R = _factor_corr(20, seed=3)
    costs = np.full(20, 7.0)
    ref = greedy_trace_select(R, 20)["selected"]
    assert _trace(R, costs, 7.0 * 20)["selected"] == ref
    assert _trace_knapsack_oracle(R, costs, 7.0 * 20)["selected"] == ref
    el = np.ones(20, dtype=bool)
    el[[0, 4, 9]] = False
    ref = greedy_trace_select(R, 12, eligible=el)["selected"]
    assert _trace(R, costs, 7.0 * 12, eligible=el)["selected"] == ref
    assert _trace_knapsack_oracle(R, costs, 7.0 * 12,
                                  eligible=el)["selected"] == ref


@pytest.mark.parametrize("seed", range(6))
def test_trace_knapsack_matches_oracle(seed):
    """budgeted_select's trace objective reproduces the oracle on random
    instances (unequal and MMLU-like repeated costs, eligibility masks,
    budgets from below the cheapest task to above the total)."""
    rng = np.random.default_rng(100 + seed)
    N = 24
    R = _factor_corr(N, n_factors=2 + seed % 3, seed=200 + seed)
    if seed % 2:
        costs = rng.choice([100.0, 100.0, 100.0, 150.0, 1000.0], N)
    else:
        costs = rng.integers(1, 20, N).astype(float)
    el = rng.random(N) > 0.2
    for budget in np.quantile(np.cumsum(costs), [0.0, 0.02, 0.1, 0.3, 0.7,
                                                 1.0]) * [0.5, 1, 1, 1, 1, 1]:
        for mask in (None, el):
            ref = _trace_knapsack_oracle(R, costs, budget, eligible=mask)
            got = _trace(R, costs, budget, eligible=mask)
            assert [int(j) for j in got["selected"]] == ref["selected"]
            assert got["singleton"] == ref["singleton"]
            assert got["value"] == pytest.approx(ref["value"], rel=1e-9,
                                                 abs=1e-12)


def test_trace_knapsack_ties_go_to_lowest_index():
    # Equicorrelated tasks of equal cost: every step is an exact tie
    N = 8
    R = np.full((N, N), 0.4)
    np.fill_diagonal(R, 1.0)
    costs = np.full(N, 5.0)
    got = _trace(R, costs, 15.0)
    assert got["selected"] == [0, 1, 2] == \
        _trace_knapsack_oracle(R, costs, 15.0)["selected"]
    assert got["index_ties"] and got["tie_breaks"]
    assert [n for _, n in got["tie_breaks"]] == [8, 7, 6]


@pytest.mark.parametrize("budget", [1.0, 5.0, 13.0, 40.0, 1000.0])
def test_trace_knapsack_budget_eligibility_value(budget):
    R = _factor_corr(25, seed=5)
    costs = np.random.default_rng(1).integers(1, 10, 25).astype(float)
    el = np.ones(25, dtype=bool)
    el[[2, 3, 11]] = False
    res = _trace(R, costs, budget, eligible=el)
    sel = [int(j) for j in res["selected"]]
    assert costs[sel].sum() <= budget
    assert len(set(sel)) == len(sel) and all(el[j] for j in sel)
    assert res["value"] == pytest.approx(_trace_value(R, sel), rel=1e-9)
    if budget >= costs[el].min():
        assert sel
        # maximal: no affordable eligible task left
        left = [j for j in range(25)
                if el[j] and j not in sel
                and costs[sel].sum() + costs[j] <= budget]
        assert res["singleton"] or not left


def test_trace_knapsack_singleton_fallback():
    # Task 0 is a hub (loading 0.6 on tasks 1..20, cost 10); task 21 is
    # independent and costs 1 (ratio 1 > hub's 8.2 / 10).  Greedy takes
    # task 21 and then cannot afford the hub: value 1 < 8.2.
    N = 22
    R = np.eye(N)
    R[0, 1:21] = R[1:21, 0] = 0.6
    R[np.ix_(range(1, 21), range(1, 21))] = 0.36
    np.fill_diagonal(R, 1.0)
    costs = np.full(N, 100.0)
    costs[0], costs[21] = 10.0, 1.0
    for fn in (_trace, _trace_knapsack_oracle):
        res = fn(R, costs, 10.0)
        assert res["singleton"] and list(res["selected"]) == [0]
        assert res["value"] == pytest.approx(1 + 20 * 0.36)
        # with room for both, the greedy set wins
        res = fn(R, costs, 11.0)
        assert not res["singleton"] and sorted(res["selected"]) == [0, 21]


def test_bench_facloc_unit_costs_is_facility_location_select():
    """bench_facloc is budgeted_select's facility-location knapsack: with
    equal task sizes and Q = k sizes it picks facility_location_select's
    first k tasks (the cardinality greedy), eligibility masks included; it
    sits before bench_random and has a best-singleton fallback."""
    assert E.KNAP_OBJECTIVE["bench_facloc"] == "facloc"
    assert "bench_facloc" in E.SINGLETON_METHODS
    assert E.BENCH_METHODS[-2:] == ["bench_facloc", "bench_random"]
    N = 20
    R = _factor_corr(N, seed=12)
    el = np.ones(N, dtype=bool)
    el[[1, 6, 13]] = False
    for mask in (None, el):
        ref = facility_location_select(R, N, eligible=mask)["selected"]
        for k in (1, 2, 5, 9, 17):
            got = E.bench_select("bench_facloc", R, np.full(N, 7.0), 7.0 * k,
                                 mask, [])
            assert got["selected"] == ref[:k], (k, mask is None)
            assert not got["singleton"]
    costs = np.random.default_rng(4).integers(1, 9, N).astype(float)
    for budget in (3.0, 11.0, 40.0):
        got = E.bench_select("bench_facloc", R, costs, budget, el, [])
        res = budgeted_select(R, costs, budget, "facloc", eligible=el)
        assert got["selected"] == [int(j) for j in res["selected"]]
        assert costs[got["selected"]].sum() <= budget


def test_bench_facloc_leaves_other_methods_bit_identical(data, split,
                                                         monkeypatch):
    """Without bench_facloc, every row of the other methods is the same,
    bit for bit (it draws no random numbers), with and without overhead."""
    budgets = [5, 9, 30, 70, 130]
    full = {c0: E.evaluate_split(split, data, budgets, fixed_cost=c0)
            for c0 in (None, 3)}
    bench = [m for m in E.BENCH_METHODS if m != "bench_facloc"]
    monkeypatch.setattr(E, "BENCH_METHODS", bench)
    monkeypatch.setattr(E, "METHODS", bench + E.ITEM_METHODS)
    monkeypatch.setattr(E, "SINGLETON_METHODS", [
        m for m in E.SINGLETON_METHODS if m != "bench_facloc"])
    for c0, row in full.items():
        base = E.evaluate_split(split, data, budgets, fixed_cost=c0)
        assert set(row) - set(base) and all(
            k.startswith("bench_facloc_") for k in set(row) - set(base))
        for key, val in base.items():
            val = np.asarray(val)
            assert np.array_equal(val, row[key],
                                  equal_nan=val.dtype.kind == "f"), (c0, key)


def test_bench_select_nothing_affordable():
    R = _factor_corr(10)
    costs = np.full(10, 100.0)
    order = list(range(10))
    for m in E.BENCH_METHODS:
        assert E.bench_select(m, R, costs, 50, None, order)["selected"] == []
        assert E.bench_select(m, R, costs, 100, None, order)["selected"]


# ---------------------------------------------------------------------------
# Imputation and noise model
# ---------------------------------------------------------------------------

def test_bench_predict_matches_impute_and_evaluate(data, split):
    B_tr, O_tr = data.B[split.train_idx], data.O[split.train_idx]
    B_te, O_te = data.B[split.val_idx], data.O[split.val_idx]
    R, mu, sd = estimate_corr(B_tr, O_tr)
    Z, _ = E.standardize(B_te, mu, sd)
    N = R.shape[0]
    for sel in ([2], [0, 4], [6, 1, 3], [5, 0, 2, 6]):
        Z_hat = E.bench_predict(R, sel, Z)
        assert np.array_equal(Z_hat[:, sel], Z[:, sel])     # exact values
        exact = np.isin(np.arange(N), sel)
        r2_unobs, _ = E.pooled_r2(Z, Z_hat, ~exact)
        ref = impute_and_evaluate(B_te, O_te, sel, R, mu, sd, N)["r2"]
        assert r2_unobs == pytest.approx(ref, abs=1e-10)
        # all-task R²: same residuals, all-task denominator
        r2_all, _ = E.pooled_r2(Z, Z_hat)
        ss_res = (1 - r2_unobs) * np.sum(Z[:, ~exact] ** 2)
        assert r2_all == pytest.approx(1 - ss_res / np.sum(Z ** 2))
    assert np.array_equal(E.bench_predict(R, [], Z), np.zeros_like(Z))
    assert E.pooled_r2(Z, np.zeros_like(Z))[0] == pytest.approx(0.0)


def test_gauss_weights_information_form():
    R = _factor_corr(12, seed=7)
    obs = np.array([1, 4, 5, 9])
    noise = np.array([0.5, 3.0, 0.1, 10.0])
    r = E.IMPUTE_RIDGE
    W = E.gauss_weights(R, obs, noise)
    # posterior mean (R^-1 + H' S^-1 H)^-1 H' S^-1 with H the selector
    H = np.zeros((len(obs), 12))
    H[np.arange(len(obs)), obs] = 1.0
    Sinv = np.diag(1.0 / (noise + r))
    ref = np.linalg.solve(np.linalg.inv(R) + H.T @ Sinv @ H, H.T @ Sinv)
    assert np.allclose(W, ref, atol=1e-10)
    # huge noise: no information; no noise and no ridge: exact copy
    assert np.allclose(E.gauss_weights(R, obs, np.full(4, 1e12)), 0.0,
                       atol=1e-9)
    W0 = E.gauss_weights(R, obs, None, ridge=0.0)
    assert np.allclose(W0[obs], np.eye(4), atol=1e-10)


@pytest.mark.parametrize("values", [
    np.array([0, 1, 1, 0, 1, 1, 0], dtype=float),
    np.array([0.1, 0.9, 0.35, 0.6, 0.05, 0.8], dtype=float)])
def test_sample_noise_var_exact_enumeration(values):
    """Variance of the mean of a uniform n-subset (without replacement)
    equals sigma² / n · (N − n) / (N − 1), sigma² the population variance;
    for 0/1 items sigma² = p (1 − p)."""
    Nt = len(values)
    sigma2 = float(np.var(values))
    if set(values) <= {0.0, 1.0}:
        p = values.mean()
        assert sigma2 == pytest.approx(p * (1 - p))
    sd = 0.37
    for n in range(1, Nt + 1):
        means = [values[list(c)].mean()
                 for c in itertools.combinations(range(Nt), n)]
        got = E.sample_noise_var(np.array([sigma2]), np.array([n]),
                                 np.array([Nt]), np.array([sd]))[0]
        assert got == pytest.approx(np.var(means) / sd ** 2, abs=1e-12)
    assert E.sample_noise_var(np.array([sigma2]), np.array([Nt]),
                              np.array([Nt]), np.array([sd]))[0] == 0.0
    assert np.isnan(E.sample_noise_var(np.array([sigma2]), np.array([0]),
                                       np.array([Nt]), np.array([sd]))[0])


def test_item_unit_variance(data, split):
    _, mu, _ = estimate_corr(data.B[split.train_idx], data.O[split.train_idx])
    v, w = E.item_unit_variance(data, split.train_idx, mu)
    for t in range(len(SIZES)):
        X = data.scores[data.offsets[t]:data.offsets[t + 1]][:, split.train_idx]
        within = np.mean(np.var(X, axis=0))
        assert w[t] == pytest.approx(within)
        if data.binary[t]:
            assert v[t] == pytest.approx(mu[t] * (1 - mu[t]))
            assert w[t] <= v[t] + 1e-12   # binomial is conservative
        else:
            assert v[t] == pytest.approx(within)


def test_train_item_stats_double_centering(data, split):
    xbar, u = E.train_item_stats(data, split.train_idx)
    tr = split.train_idx
    assert np.allclose(xbar, data.scores[:, tr].mean(axis=1))
    for t in range(len(SIZES)):
        X = data.scores[data.offsets[t]:data.offsets[t + 1]][:, tr]
        r = (X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True)
             + X.mean())
        assert np.allclose(r.sum(axis=0), 0.0, atol=1e-10)  # per model
        assert u[t] == pytest.approx(np.mean(r ** 2), rel=1e-10)
        # = mean item variance − variance of the task scores (ddof 0)
        assert u[t] == pytest.approx(np.mean(np.var(X, axis=1))
                                     - np.var(X.mean(axis=0)), rel=1e-9)
    # the residual variance is below the within-model variance of
    # items_gauss (it also removes the item-difficulty spread)
    _, mu, _ = estimate_corr(data.B[tr], data.O[tr])
    _, w = E.item_unit_variance(data, tr, mu)
    assert np.all(u <= w + 1e-12)


def test_centered_sample_means(data):
    sample = np.array([0, 3, 11, 12, 40, 41, 42, 120])
    val = np.arange(5)
    xbar = data.scores[:, 5:].mean(axis=1)
    means = E.sample_task_means(data.scores[:, val], sample, data.item_task,
                                len(SIZES))
    got = E.centered_sample_means(means, xbar, sample, data.item_task,
                                  len(SIZES))
    for t in range(len(SIZES)):
        rows = sample[data.item_task[sample] == t]
        if len(rows):
            assert np.allclose(got[:, t], means[:, t] - xbar[rows].mean())
        else:
            assert np.all(np.isnan(got[:, t]))


def test_centered_estimator_exact_for_additive_items():
    """x_mi = a_mt + b_i: the centered observation equals the standardized
    task score whatever the sample, u = 0, and items_center is the
    noise-free (ridge-only) Gaussian conditioning on the sampled tasks;
    the uncentered error (b̄_S − b̄_t) / sd_t is the same for every model,
    so items_gauss's shared fraction is 1."""
    data = _additive()
    split = next(iter(cv_splits(data.B.shape[0], 5, 1, 0.1, E.SEED)))
    budgets = [3, 12, 40]
    row = E.evaluate_split(split, data, budgets, return_predictions=True)
    assert row["n_clipped"] == 0
    assert np.allclose(row["noise_center_var"], 0.0, atol=1e-14)
    tr = split.train_idx
    R, mu, sd = estimate_corr(data.B[tr], data.O[tr])
    Z = row["Z"]
    xbar, _ = E.train_item_stats(data, tr)
    perm = E.item_permutation(len(data.item_task), split)
    N = len(SIZES)
    for b, Q in enumerate(budgets):
        sample = perm[:Q]
        n_t = E.sampled_counts(sample, data.item_task, N)
        obs = np.flatnonzero(n_t > 0)
        means = E.sample_task_means(data.scores[:, split.val_idx], sample,
                                    data.item_task, N)
        Yc = E.centered_sample_means(means, xbar, sample, data.item_task,
                                     N) / sd[None, :]
        assert np.allclose(Yc[:, obs], Z[:, obs], atol=1e-12)
        ref = Z[:, obs] @ E.gauss_weights(R, obs, np.zeros(len(obs))).T
        assert np.allclose(row["predictions"][("items_center", b)], ref,
                           atol=1e-8)
        assert row["items_gauss_noise_shared"][b] == pytest.approx(1.0,
                                                                   rel=1e-9)
        assert row["items_center_r2"][b] > row["items_gauss_r2"][b]


def test_noise_diagnostics():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((6, 4))
    n_t = np.array([0, 2, 5, 3])
    N_t = np.array([5, 5, 5, 9])
    v = np.array([0.2, 0.25, 0.1, 0.15])
    sd = np.array([0.1, 0.2, 0.3, 0.4])
    common = np.array([0.0, 0.3, 0.0, -0.2])
    e = rng.standard_normal((6, 4)) * 0.1 + common[None, :]
    calib, shared = E.noise_diagnostics(Z, Z + e, n_t, N_t, v, sd)
    cols = [1, 3]                       # sampled, not completely
    var = E.sample_noise_var(v, n_t, N_t, sd)[cols]
    assert calib == pytest.approx(np.sum(e[:, cols] ** 2) / (6 * var.sum()))
    ebar = e[:, cols].mean(axis=0)
    assert shared == pytest.approx(6 * np.sum(ebar ** 2)
                                   / np.sum(e[:, cols] ** 2))
    assert 0.0 <= shared <= 1.0
    assert all(np.isnan(E.noise_diagnostics(Z, Z + e, np.array([0, 5, 5, 9]),
                                            N_t, v, sd)))


def test_sample_task_means(data):
    sample = np.array([0, 3, 11, 12, 40, 41, 42, 120])
    val = np.arange(5)
    means = E.sample_task_means(data.scores[:, val], sample, data.item_task,
                                len(SIZES))
    n_t = E.sampled_counts(sample, data.item_task, len(SIZES))
    for t in range(len(SIZES)):
        rows = sample[data.item_task[sample] == t]
        assert n_t[t] == len(rows)
        if len(rows):
            assert np.allclose(means[:, t], data.scores[rows][:, val].mean(0))
        else:
            assert np.all(np.isnan(means[:, t]))


# ---------------------------------------------------------------------------
# Common random numbers and leakage
# ---------------------------------------------------------------------------

def _seed_tags_of_other_scripts():
    """{tag offset: [file, ...]} of the other scripts in code/: the values
    of *_SEED_OFFSET constants and the n of 'SEED + n' (code and docs)."""
    code = pathlib.Path(E.__file__).resolve().parent
    used = {}
    for path in sorted(code.glob("*.py")):
        if path.name == "eval_item_budget.py" or path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        hits = [int(m.group(1)) for m in re.finditer(
            r"^\s*[A-Z_]*SEED_OFFSET\s*=\s*(\d+)\b", text, re.M)]
        hits += [int(m.group(1)) for m in re.finditer(
            r"\bSEED\s*\+\s*(\d+)\b", text)]
        for tag in hits:
            used.setdefault(tag, set()).add(path.name)
    return used


def test_item_seed_tag_unique_in_code_tree():
    used = _seed_tags_of_other_scripts()
    assert {RANDOM_SEED_OFFSET, ENTROPY_RF_SEED_OFFSET} <= set(used)  # scan
    assert E.ITEM_SEED_OFFSET not in used, (
        f"seed tag SEED + {E.ITEM_SEED_OFFSET} also used by "
        f"{sorted(used.get(E.ITEM_SEED_OFFSET, ()))}")


def test_item_sample_seed_stream(split):
    assert E.ITEM_SEED_OFFSET not in (RANDOM_SEED_OFFSET,
                                      ENTROPY_RF_SEED_OFFSET)
    perm = E.item_permutation(500, split)
    assert np.array_equal(perm, E.item_permutation(500, split))
    seed = split_seed(E.SEED + E.ITEM_SEED_OFFSET, split.rep, split.fold,
                      split.holdout_frac)
    assert np.array_equal(perm, np.random.default_rng(seed).permutation(500))
    other = list(cv_splits(80, 5, 2, 0.1, E.SEED))[7]
    assert not np.array_equal(perm, E.item_permutation(500, other))


def test_full_item_budget_recovers_scores(data, split):
    n_all = len(data.item_task)
    row = E.evaluate_split(split, data, [n_all], return_predictions=True)
    assert row["items_only_r2"][0] == pytest.approx(1.0)
    assert row["items_only_n_exact"][0] == len(SIZES)
    assert np.isnan(row["items_only_r2_unobs"][0])   # nothing left
    assert row["items_gauss_r2"][0] > 0.99           # ridge only
    assert row["items_center_r2"][0] > 0.99          # centered = exact here
    for m in E.NOISE_METHODS:                        # no partly sampled task
        assert np.isnan(row[f"{m}_noise_calib"][0])


def _with_val_items(data, split, rows, rng):
    """data with the validation models' scores on item rows `rows`
    redrawn (task scores recomputed)."""
    scores = data.scores.copy()
    cols = split.val_idx
    sub = scores[np.ix_(rows, cols)]
    scores[np.ix_(rows, cols)] = np.where(
        data.binary[data.item_task[rows]][:, None],
        (rng.random(sub.shape) < 0.5).astype(float), rng.random(sub.shape))
    B = (np.add.reduceat(scores, data.offsets[:-1], axis=0)
         / data.n_items[:, None]).T
    return data._replace(scores=scores, B=B)


def test_no_leakage_of_validation_data(data, split):
    budgets = [8, 20, 45, 90]
    base = E.evaluate_split(split, data, budgets, return_predictions=True)
    rng = np.random.default_rng(11)
    perm = E.item_permutation(len(data.item_task), split)

    # (a) every validation response redrawn: selections, item samples,
    # eligibility and noise models are functions of training data only
    changed = _with_val_items(data, split, np.arange(len(data.item_task)),
                              rng)
    alt = E.evaluate_split(split, changed, budgets, return_predictions=True)
    for key in (["items_n_sampled", "noise_unit_var", "noise_within_var",
                 "noise_center_var", "eligible"]
                + [f"{m}_selected" for m in E.BENCH_METHODS]
                + [f"{m}_cost" for m in E.METHODS]):
        assert np.array_equal(base[key], alt[key]), key
    assert not np.allclose(base["Z"], alt["Z"])

    # (b) validation responses redrawn on items outside the largest
    # sample: the item methods' predictions cannot change (items_center's
    # offset uses the training models' answers only)
    unsampled = np.setdiff1d(np.arange(len(data.item_task)),
                             perm[:max(budgets)])
    changed = _with_val_items(data, split, unsampled, rng)
    alt = E.evaluate_split(split, changed, budgets, return_predictions=True)
    for m in E.ITEM_METHODS:
        for b in range(len(budgets)):
            assert np.array_equal(base["predictions"][(m, b)],
                                  alt["predictions"][(m, b)]), (m, b)

    # (c) bench predictions use only the selected tasks of the validation
    # models: redraw the items of one never-selected task
    sel = set()
    for m in E.BENCH_METHODS:
        sel |= set(base[f"{m}_selected"][base[f"{m}_selected"] >= 0].ravel())
    free = [t for t in range(len(SIZES)) if t not in sel]
    if free:
        t = free[0]
        rows = np.arange(data.offsets[t], data.offsets[t + 1])
        changed = _with_val_items(data, split, rows, rng)
        alt = E.evaluate_split(split, changed, budgets,
                               return_predictions=True)
        for m in E.BENCH_METHODS:
            for b in range(len(budgets)):
                assert np.array_equal(base["predictions"][(m, b)],
                                      alt["predictions"][(m, b)]), (m, b)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_paired_comparable():
    assert E.paired_comparable("r2", "bench_mi", "items_center")
    assert E.paired_comparable("r2_unobs", "bench_mi", "bench_random")
    assert E.paired_comparable("r2_unobs", "items_center", "items_only")
    assert not E.paired_comparable("r2_unobs", "bench_trace", "items_gauss")
    with pytest.raises(ValueError):
        E.method_family("hub")


def test_run_end_to_end(data, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "OUT", tmp_path)
    budgets = [5, 9, 30, 100]
    out = E.run("syn", budgets, n_repeats=2, n_folds=4, data=data)
    S, nb, N = 8, len(budgets), len(SIZES)
    for m in E.METHODS:
        for key in ("r2", "r2_unobs", "rmse", "cost", "n_obs", "n_exact"):
            assert out[f"{m}_{key}"].shape == (S, nb)
    for m in E.NOISE_METHODS:
        for key in ("noise_calib", "noise_shared"):
            assert out[f"{m}_{key}"].shape == (S, nb)
    assert out["noise_center_var"].shape == (S, N)
    assert out["bench_mi_selected"].shape == (S, nb, N)
    assert out["items_n_sampled"].shape == (S, nb, N)
    assert np.array_equal(out["items_n_sampled"].sum(axis=2),
                          np.tile(budgets, (S, 1)))
    assert np.array_equal(out["split_id"], np.arange(S))
    assert not out["estimation_failed"].any()
    assert int(out["item_seed_offset"]) == E.ITEM_SEED_OFFSET
    # below the cheapest task (8 items) no whole task fits: train mean
    for m in E.BENCH_METHODS:
        assert np.all(out[f"{m}_r2"][:, 0] == 0.0)
        assert np.all(out[f"{m}_cost"][:, 0] == 0.0)
        assert np.all(out[f"{m}_cost"] <= np.array(budgets)[None, :])
    # nested item samples
    assert np.all(np.diff(out["items_n_sampled"], axis=1) >= 0)

    paired = E.paired_rows("syn", out)
    n_bench, n_items = len(E.BENCH_METHODS), len(E.ITEM_METHODS)
    same_family = (n_bench * (n_bench - 1) + n_items * (n_items - 1)) // 2
    assert len(paired) == nb * (len(E.method_pairs()) + same_family)
    # r2_unobs rows only within a family (like-for-like task sets)
    for p in paired:
        if p["metric"] == "r2_unobs":
            assert (E.method_family(p["method_a"])
                    == E.method_family(p["method_b"])), p
    r = next(p for p in paired if p["budget"] == 30 and p["metric"] == "r2"
             and p["method_a"] == "bench_mi"
             and p["method_b"] == "items_gauss")
    d = out["bench_mi_r2"][:, 2] - out["items_gauss_r2"][:, 2]
    assert r["mean_diff"] == pytest.approx(d.mean())
    assert r["n_splits"] == S and r["n_reps"] == 2
    r = next(p for p in paired if p["budget"] == 100 and p["metric"] == "r2"
             and p["method_a"] == "bench_trace"
             and p["method_b"] == "items_center")
    d2 = out["bench_trace_r2"][:, 3] - out["items_center_r2"][:, 3]
    assert r["mean_diff"] == pytest.approx(d2.mean())

    macros = E.macro_dict("syn", out, paired)
    assert all(name.isalpha() and name.isascii() for name in macros)
    assert macros["ItemBudgetRsqSynBenchMiQthirty"] == pytest.approx(
        out["bench_mi_r2"][:, 2].mean())
    assert macros["ItemBudgetPairDiffSynBenchMiVsItemsGaussQthirty"] == \
        pytest.approx(d.mean())
    assert macros["ItemBudgetPairDiffSynBenchTraceVsItemsCenterQonehundred"] \
        == pytest.approx(d2.mean())
    assert macros["ItemBudgetRsqSynItemsCenterQnine"] == pytest.approx(
        out["items_center_r2"][:, 1].mean())
    assert macros["ItemBudgetNoiseSharedSynItemsGaussQthirty"] == \
        pytest.approx(np.nanmean(out["items_gauss_noise_shared"][:, 2]))
    assert macros["ItemBudgetNSplitsSyn"] == S
    assert macros["ItemBudgetNItemsSyn"] == len(data.item_task)
    # the facility-location knapsack follows bench_trace's pattern
    assert macros["ItemBudgetRsqSynBenchFaclocQonehundred"] == \
        pytest.approx(out["bench_facloc_r2"][:, 3].mean())
    d3 = out["bench_facloc_r2"][:, 2] - out["bench_random_r2"][:, 2]
    assert macros["ItemBudgetPairDiffSynBenchFaclocVsBenchRandomQthirty"] \
        == pytest.approx(d3.mean())
    assert "ItemBudgetPairDiffSynBenchTraceVsBenchFaclocQthirty" in macros

    path = tmp_path / "item_budget_syn.json"
    E.write_json(path, "syn", out, paired, 1.0, ["test"])
    doc = json.loads(path.read_text())
    assert doc["macros"] == macros
    assert isinstance(macros["ItemBudgetNSplitsSyn"], int)
    assert doc["config"]["item_seed_tag"] == E.SEED + E.ITEM_SEED_OFFSET
    assert set(doc["summary"]["noise_calib"]) == set(E.NOISE_METHODS)
    E.plot_budget("syn", out)
    assert (tmp_path / "item_budget_syn.pdf").is_file()

    # reruns are identical (common random numbers, no hidden state)
    again = E.run("syn", budgets, n_repeats=2, n_folds=4, data=data)
    for key, val in out.items():
        val = np.asarray(val)
        assert np.array_equal(val, again[key],
                              equal_nan=val.dtype.kind in "fc"), key


# ---------------------------------------------------------------------------
# Fixed per-task overhead (--fixed-cost)
# ---------------------------------------------------------------------------

def _walk_oracle(perm, item_task, budget, c0):
    """The skip-and-continue walk of fixed_item_sample, written out."""
    touched, sample, spent = set(), [], 0
    for i in perm:
        t = int(item_task[i])
        cost = 1 if t in touched else 1 + c0
        if spent + cost <= budget:
            sample.append(int(i))
            spent += cost
            touched.add(t)
    return sample


class TestFixedItemSample:
    @pytest.mark.parametrize("c0", [0, 1, 3, 10, 50])
    def test_matches_walk_oracle(self, data, c0):
        rng = np.random.default_rng(100 + c0)
        n = len(data.item_task)
        for _ in range(5):
            perm = rng.permutation(n)
            for budget in (1, 2, 5, 11, 30, 64, 140, 400, n + 7 * c0 + 5):
                got = E.fixed_item_sample(perm, data.item_task, len(SIZES),
                                          budget, c0)
                ref = _walk_oracle(perm, data.item_task, budget, c0)
                assert got.tolist() == ref, (c0, budget)
                n_t = E.sampled_counts(got, data.item_task, len(SIZES))
                assert len(got) + c0 * int(np.sum(n_t > 0)) <= budget

    def test_zero_overhead_is_prefix(self, data):
        perm = np.random.default_rng(1).permutation(len(data.item_task))
        for budget in (1, 17, 100):
            assert np.array_equal(
                E.fixed_item_sample(perm, data.item_task, len(SIZES),
                                    budget, 0), perm[:budget])

    @pytest.mark.parametrize("c0", [2, 10, 40])
    def test_within_task_simple_random_sample(self, data, c0):
        """The sampled items of every task are the first n_t of its items
        in permutation order (a simple random sample given n_t)."""
        perm = np.random.default_rng(7).permutation(len(data.item_task))
        pos = np.empty(len(perm), dtype=np.int64)
        pos[perm] = np.arange(len(perm))
        for budget in (25, 90, 260):
            got = E.fixed_item_sample(perm, data.item_task, len(SIZES),
                                      budget, c0)
            assert np.all(np.diff(pos[got]) > 0)       # permutation order
            for t in range(len(SIZES)):
                items_t = perm[data.item_task[perm] == t]
                n_t = int(np.sum(data.item_task[got] == t))
                assert set(got[data.item_task[got] == t]) == set(
                    items_t[:n_t])


class TestFixedEvaluate:
    BUDGETS = [5, 9, 30, 70, 130]      # the synthetic data have 135 items

    def test_zero_overhead_matches_plain(self, data, split):
        plain = E.evaluate_split(split, data, self.BUDGETS)
        fixed = E.evaluate_split(split, data, self.BUDGETS, fixed_cost=0)
        assert set(fixed) == set(plain) | {f"{m}_questions"
                                           for m in E.METHODS}
        for key, val in plain.items():
            val = np.asarray(val)
            assert np.array_equal(val, fixed[key],
                                  equal_nan=val.dtype.kind == "f"), key
        for m in E.METHODS:
            assert np.array_equal(fixed[f"{m}_questions"],
                                  fixed[f"{m}_cost"].astype(np.int64))

    @pytest.mark.parametrize("c0", [3, 12])
    def test_true_budget_and_shared_randomness(self, data, split, c0):
        row = E.evaluate_split(split, data, self.BUDGETS, fixed_cost=c0)
        Q = np.array(self.BUDGETS, dtype=float)
        costs = data.n_items + c0
        R, _, _ = estimate_corr(data.B[split.train_idx],
                                data.O[split.train_idx])
        order = random_select_order(R, split, row["eligible"])
        perm = E.item_permutation(len(data.item_task), split)
        for b, q in enumerate(self.BUDGETS):
            for m in E.BENCH_METHODS:
                sel = row[f"{m}_selected"][b]
                sel = sel[sel >= 0]
                assert row[f"{m}_cost"][b] == costs[sel].sum() <= q
                assert row[f"{m}_questions"][b] == data.n_items[sel].sum()
            sel = row["bench_random_selected"][b]
            assert sel[sel >= 0].tolist() == E.order_fill(
                order, costs.astype(float), q)
            assert row["bench_trace_selected"][b][
                row["bench_trace_selected"][b] >= 0].tolist() == \
                budgeted_select(R, costs.astype(float), q, "trace",
                                eligible=row["eligible"])["selected"]
            sample = E.fixed_item_sample(perm, data.item_task, len(SIZES),
                                         q, c0)
            n_t = E.sampled_counts(sample, data.item_task, len(SIZES))
            assert np.array_equal(row["items_n_sampled"][b], n_t)
            for m in E.ITEM_METHODS:
                assert row[f"{m}_questions"][b] == len(sample)
                assert row[f"{m}_cost"][b] == len(sample) + c0 * np.sum(
                    n_t > 0) <= q
        assert np.all(row["bench_mi_cost"] <= Q)

    def test_no_leakage_with_overhead(self, data, split):
        budgets = [8, 20, 45, 90]
        base = E.evaluate_split(split, data, budgets, return_predictions=True,
                                fixed_cost=5)
        rng = np.random.default_rng(13)
        changed = _with_val_items(data, split, np.arange(len(data.item_task)),
                                  rng)
        alt = E.evaluate_split(split, changed, budgets,
                               return_predictions=True, fixed_cost=5)
        for key in (["items_n_sampled", "noise_unit_var", "noise_center_var",
                     "eligible"]
                    + [f"{m}_selected" for m in E.BENCH_METHODS]
                    + [f"{m}_cost" for m in E.METHODS]
                    + [f"{m}_questions" for m in E.METHODS]):
            assert np.array_equal(base[key], alt[key]), key
        assert not np.allclose(base["Z"], alt["Z"])


def random_select_order(R, split, eligible):
    """bench_random's task order on a split (the shared random baseline)."""
    from greedy_select import random_select
    return random_select(R, R.shape[0],
                         seed=split.seed(E.SEED + RANDOM_SEED_OFFSET),
                         eligible=eligible)["selected"]


def _same(a, b):
    """Equal dicts, NaN equal to NaN."""
    return set(a) == set(b) and all(
        a[k] == b[k] or (isinstance(a[k], float) and isinstance(b[k], float)
                         and math.isnan(a[k]) and math.isnan(b[k]))
        for k in a)


class TestCrossover:
    Q = [50, 100, 200, 400, 800]

    def test_inside_grid_not_interpolated(self):
        """Grid values only: the bracket (q_first_lo, q_first] of the
        first crossing, no interpolation of the step function."""
        r = E.crossover(self.Q, [-0.5, -0.1, 0.3, 0.2, 0.1])
        assert r["state"] == 0
        assert r["q_first"] == 200 and r["q_first_lo"] == 100
        assert r["q_stay"] == 200 and r["n_ahead"] == 3
        assert math.isnan(r["q_sig"]) and r["n_sig"] == 0
        assert set(r) == {f for _, f in E.CROSS_FIELDS}

    def test_non_monotone(self):
        r = E.crossover(self.Q, [-1.0, 0.1, -0.1, 0.2, 0.3],
                        [-2, -0.1, -0.3, 0.05, 0.1])
        assert r["state"] == 0
        assert r["q_first"] == 100 and r["q_first_lo"] == 50
        assert r["q_stay"] == 400
        assert r["q_sig"] == 400 and r["q_sig_stay"] == 400
        assert r["n_ahead"] == 3 and r["n_sig"] == 2

    def test_ahead_everywhere_and_never(self):
        r = E.crossover(self.Q, [0.1] * 5, [0.01] * 5)
        assert r["state"] == -1 and r["q_first"] == 50
        assert math.isnan(r["q_first_lo"])
        assert r["q_stay"] == 50 and r["q_sig_stay"] == 50
        r = E.crossover(self.Q, [-0.1, 0.0, -0.2, -0.01, np.nan])
        assert r["state"] == 1 and math.isnan(r["q_first"])
        assert math.isnan(r["q_first_lo"])
        assert math.isnan(r["q_stay"]) and r["n_ahead"] == 0
        # ahead, then behind at the largest budget: no q_stay
        r = E.crossover(self.Q, [-0.1, 0.2, 0.3, 0.1, -0.01])
        assert r["state"] == 0 and math.isnan(r["q_stay"])
        with pytest.raises(ValueError):
            E.crossover([100, 50], [0.1, 0.2])

    def test_step_function_below_cheapest_task(self):
        """The reviewed cases: the whole-task method scores exactly 0
        below the cheapest task (200 here), the item method R² > 0, so the
        crossing is the cheapest task, never a budget below it (the former
        log-linear interpolation gave 179); and d = 0 exactly where
        neither buys anything is not 'ahead' (it gave the crossing 50)."""
        q = [50, 71, 100, 141, 200, 283]
        item = np.array([0.0, 0.0, 0.0, 0.444, 0.507, 0.476])
        bench = np.array([0.0, 0.0, 0.0, 0.0, 0.709, 0.766])
        r = E.crossover(q, bench - item)
        assert r["q_first"] == 200 and r["q_first_lo"] == 141
        assert r["q_stay"] == 200
        # an item method worse than the train mean (R² < 0) is behind as
        # soon as it buys anything: d(50) = 0, d(71) > 0
        r = E.crossover(q, [0.0, 0.008, -0.009, 0.139, 0.5, 0.9])
        assert r["state"] == 0
        assert r["q_first"] == 71 and r["q_first_lo"] == 50


def test_fixed_budget_grid():
    n_items = np.array([30, 8, 12])
    assert E.fixed_budget_grid([50, 10, 20], n_items, [0, 2, 12, 42]) == [
        8, 10, 20, 50]
    assert E.fixed_budget_grid([50, 10], n_items, [0, 7]) == [8, 10, 15, 50]
    g = E.fixed_budget_grid(E.FIXED_BUDGETS, np.array([100, 1000]),
                            E.DEFAULT_FIXED_COSTS)
    assert g == sorted(set(E.FIXED_BUDGETS) | {110, 150, 300})
    assert all(isinstance(v, int) for v in g)
    assert E.fixed_budget_grid(g, np.array([100, 1000]),
                               E.DEFAULT_FIXED_COSTS) == g


def test_run_fixed_end_to_end(data, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "OUT", tmp_path)
    budgets = [5, 9, 30, 100, 130]     # <= 135 items, as the plain run needs
    levels = [0, 3, 10]
    out = E.run_fixed("syn", budgets, levels, n_repeats=2, n_folds=4,
                      data=data)
    S, nf, nb, N = 8, len(levels), len(budgets), len(SIZES)
    for m in E.METHODS:
        for key in ("r2", "r2_unobs", "cost", "questions", "n_obs"):
            assert out[f"{m}_{key}"].shape == (S, nf, nb)
        assert np.all(out[f"{m}_cost"] <= np.array(budgets)[None, None, :])
    assert out["items_n_sampled"].shape == (S, nf, nb, N)
    assert out["noise_center_var"].shape == (S, N)
    assert np.array_equal(out["total_cost"],
                          len(data.item_task) + N * np.array(levels))
    assert np.array_equal(out["min_task_cost"],
                          min(SIZES) + np.array(levels))
    # overhead share = C0 x tasks touched / true cost; 0 at C0 = 0, NaN
    # where nothing is bought; one item sample for the three item methods
    for m in E.METHODS:
        sh = E.overhead_share(out, m)
        cost = out[f"{m}_cost"]
        over = np.array(levels)[None, :, None] * out[f"{m}_n_obs"]
        pos = cost > 0
        np.testing.assert_allclose(sh[pos], over[pos] / cost[pos])
        assert np.all(np.isnan(sh[cost == 0]))
        assert np.all(sh[:, 0][cost[:, 0] > 0] == 0)
    for m in E.ITEM_METHODS[1:]:
        assert np.array_equal(E.overhead_share(out, m),
                              E.overhead_share(out, E.ITEM_METHODS[0]),
                              equal_nan=True)
    # level 0 = the plain run, bit for bit
    plain = E.run("syn", budgets, n_repeats=2, n_folds=4, data=data)
    for key, val in plain.items():
        val = np.asarray(val)
        if key in out and out[key].ndim >= 2 and key not in \
                E.LEVEL_FREE_KEYS and key != "eligible":
            got = out[key][:, 0]
        else:
            got = out[key]
        assert np.array_equal(val, got, equal_nan=val.dtype.kind == "f"), key
    # the overhead costs the item methods tasks: fewer tasks touched
    assert np.all(out["items_center_n_obs"][:, 2] <= out["items_center_n_obs"]
                  [:, 0])

    paired = E.fixed_paired_rows("syn", out)
    n_bench, n_items = len(E.BENCH_METHODS), len(E.ITEM_METHODS)
    same_family = (n_bench * (n_bench - 1) + n_items * (n_items - 1)) // 2
    assert len(paired) == nf * nb * (len(E.method_pairs()) + same_family)
    cross = E.crossover_table(out, paired)
    assert len(cross) == nf * len(E.CROSS_PAIRS)
    r = next(p for p in paired if p["fixed_cost"] == 10 and p["budget"] == 100
             and p["metric"] == "r2" and p["method_a"] == "bench_trace"
             and p["method_b"] == "items_center")
    d = out["bench_trace_r2"][:, 2, 3] - out["items_center_r2"][:, 2, 3]
    assert r["mean_diff"] == pytest.approx(d.mean())
    c = next(x for x in cross if x["fixed_cost"] == 10
             and x["method_a"] == "bench_trace"
             and x["method_b"] == "items_center")
    rows = [next(p for p in paired if p["fixed_cost"] == 10
                 and p["budget"] == q and p["metric"] == "r2"
                 and p["method_a"] == "bench_trace"
                 and p["method_b"] == "items_center") for q in budgets]
    assert _same(c, {"fixed_cost": 10, "method_a": "bench_trace",
                     "method_b": "items_center",
                     **E.crossover(budgets, [p["mean_diff"] for p in rows],
                                   [p["ci_low"] for p in rows])})
    # no crossing below the cheapest affordable whole task unless the item
    # method's mean R² is negative there (the whole-task R² is exactly 0)
    ok = ~out["estimation_failed"]
    for x in cross:
        f = levels.index(x["fixed_cost"])
        cheapest = int(out["min_task_cost"][f])
        below = np.array(budgets) < cheapest
        assert np.all(out[f"{x['method_a']}_r2"][ok, f][:, below] == 0)
        item_mean = out[f"{x['method_b']}_r2"][ok, f].mean(axis=0)
        if np.isfinite(x["q_first"]) and np.all(item_mean[below] >= 0):
            assert x["q_first"] >= cheapest, x
        if x["state"] == 0:
            assert x["q_first_lo"] < x["q_first"]
            assert x["q_first_lo"] in budgets and x["q_first"] in budgets

    undefined = []
    mac = E.fixed_macro_dict("syn", out, paired, cross, undefined)
    assert all(k.isascii() and k.isalpha() for k in list(mac) + undefined)
    assert all(v is not None and math.isfinite(v) for v in mac.values())
    assert mac["ItemBudgetFixedViolationsSyn"] == 0
    assert mac["ItemBudgetFixedRsqSynBenchTraceCtenQonehundred"] == \
        pytest.approx(out["bench_trace_r2"][:, 2, 3].mean())
    assert mac["ItemBudgetFixedPairDiffSynBenchTraceVsItemsCenterCten"
               "Qonehundred"] == pytest.approx(d.mean())
    assert mac["ItemBudgetFixedCrossStateSynBenchTraceVsItemsCenterCten"] \
        == c["state"]
    assert "ItemBudgetFixedCrossQGridSynBenchTraceVsItemsCenterCten" not in \
        mac
    for suffix, field in E.CROSS_FIELDS:
        name = f"ItemBudgetFixedCross{suffix}SynBenchTraceVsItemsCenterCten"
        assert (name in mac) == math.isfinite(c[field]), name
        if name in mac:
            assert mac[name] == pytest.approx(c[field])
    assert mac["ItemBudgetFixedTotalSynCthree"] == len(data.item_task) + 3 * N
    share = E.overhead_share(out, "items_center")[ok, 2, 3]
    assert mac["ItemBudgetFixedOverheadShareSynItemsCtenQonehundred"] == \
        pytest.approx(share.mean())
    assert 0 < share.mean() < 1
    assert mac["ItemBudgetFixedOverheadShareSynItemsCzeroQonehundred"] == 0
    assert mac["ItemBudgetFixedQuestionsSynItemsCtenQonehundred"] == \
        pytest.approx(out["items_center_questions"][ok, 2, 3].mean())
    assert "ItemBudgetFixedOverheadShareSynBenchTraceCtenQonehundred" in mac
    assert "ItemBudgetFixedCrossStateSynBenchFaclocVsItemsCenterCten" in mac
    assert mac["ItemBudgetFixedRsqSynBenchFaclocCtenQonehundred"] == \
        pytest.approx(out["bench_facloc_r2"][:, 2, 3].mean())
    # bench methods buy nothing at Q = 5 (cheapest task 8): undefined share
    assert "ItemBudgetFixedOverheadShareSynBenchTraceCzeroQfive" in undefined
    path = tmp_path / "item_fixed_syn.json"
    doc = E.write_fixed_json(path, "syn", out, paired, cross, 1.0, ["test"],
                             added_budgets=[8])
    assert json.loads(path.read_text()) == json.loads(json.dumps(doc))
    assert doc["checks"]["violations_total"] == 0
    assert set(doc["summary"]) == {"0", "3", "10"}
    assert "overhead_share" in doc["summary"]["10"]
    assert doc["config"]["min_task_cost"] == [min(SIZES) + c0
                                              for c0 in levels]
    assert doc["config"]["budgets_added_min_task"] == [8]
    for key in ("human_evaluation", "item_baseline", "budget_grid"):
        assert key in doc["notes"]
    assert "interpolat" not in doc["notes"]["crossover"].replace(
        "not interpolated", "")
    p = E.plot_fixed("syn", out, cross)
    assert p.is_file() and p.name == "item_fixed_syn.pdf"
    again = E.run_fixed("syn", budgets, levels, n_repeats=2, n_folds=4,
                        data=data)
    for key, val in out.items():
        val = np.asarray(val)
        assert np.array_equal(val, again[key],
                              equal_nan=val.dtype.kind in "fc"), key


def test_main_fixed_budget_grid(data, tmp_path, monkeypatch):
    """The default grid of a fixed run gains each level's cheapest whole
    task (min_t N_t + C0 = 8, 11 here); an explicit grid is used as
    given."""
    monkeypatch.setattr(E, "OUT", tmp_path)
    monkeypatch.setattr(E, "LOGS", tmp_path)
    monkeypatch.setattr(E, "load_dataset", lambda name: data)
    for add, grid, added in ((True, [5, 8, 9, 11, 30], [8, 11]),
                             (False, [5, 9, 30], [])):
        E.main_fixed(["syn"], [5, 9, 30], [0, 3], 1, 4, ["test"],
                     add_min_task=add)
        with np.load(tmp_path / "item_fixed_syn.npz") as z:
            assert z["budgets"].tolist() == grid
            assert z["min_task_cost"].tolist() == [8, 11]
        doc = json.loads((tmp_path / "item_fixed_syn.json").read_text())
        assert doc["config"]["budgets"] == grid
        assert doc["config"]["budgets_added_min_task"] == added
        assert (tmp_path / "item_fixed_syn_paired.csv").is_file()
        assert (tmp_path / "item_fixed_syn.pdf").is_file()
