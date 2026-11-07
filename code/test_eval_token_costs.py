"""Tests for eval_token_costs.py and the trace objective of
eval_cost_budget.budgeted_select (synthetic data only, < 60 s).

Covers: proxy_budgeted_select reduces to budgeted_select when the ranking
costs are the true costs (every objective, eligibility masks, cost ties);
the trace objective (uniform costs reproduce greedy_trace_select, ratio
ties go to the lowest index, best-singleton fallback); the facility-
location objective (uniform costs reproduce facility_location_select, the
fill design is its order, adding facloc leaves every array of the other
objectives bit-identical); the true token
budget holds for every final selection; the ranking uses the proxy costs;
no leakage (selections ignore validation scores; the tokens and instances
cost models of scenario cell ignore validation models); the cost caveats
(imputation arithmetic, impute / exclude / keep, an excluded benchmark is
never bought but stays a target, common random numbers); the end-to-end
log layout, paired-pair orientation, finite-only macros and names; the
fixed per-benchmark overhead (--fixed-cost: fixed_tables, level 0 = the
plain main scenario bit for bit, the true budget with overhead for every
method, random as the fill of the shared permutation, the large-overhead
limit, outputs).
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from baseline_select import facility_location_select, greedy_trace_select
from cv_split import RANDOM_SEED_OFFSET, cv_splits
from eval_cost_budget import (OBJECTIVES as BUDGET_OBJECTIVES,
                              budgeted_greedy, budgeted_select, order_fill,
                              select_for_budget, split_orders)
from greedy_select import random_select
import eval_token_costs as etc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corr(N, rank=3, seed=0, noise=0.3):
    """Random factor-model correlation matrix (generic: no exact ties)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N, rank))
    S = W @ W.T + noise * np.diag(rng.uniform(0.5, 1.5, size=N))
    d = 1.0 / np.sqrt(np.diag(S))
    return S * np.outer(d, d)


def _synthetic(M=40, N=8, seed=0, miss=0.1):
    """(B, O, names, tables) of a synthetic HELM-like dataset.  Benchmarks
    0, 3, 6 send 4 requests per instance; benchmark 1 is a calibrated
    multiple-choice benchmark (8 requests, p = 5.1, o = 0.1) for the cost
    caveat tests (CAVEAT); its one-request cost is instances x (p + o)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(M, 2))
    W = rng.normal(size=(2, N))
    X = 0.5 + 0.1 * (Z @ W) + 0.03 * rng.normal(size=(M, N))
    O = (rng.uniform(size=(M, N)) > miss).astype(float)
    O[:, 0] = 1.0                       # one fully observed column
    B = X * O
    tokens = rng.uniform(1e4, 1e6, size=N)
    instances = np.round(rng.uniform(50, 2000, size=N))
    prompt = rng.uniform(50, 1000, size=N)
    completion = rng.uniform(0, 20, size=N)
    rpi = np.where(np.arange(N) % 3 == 0, 4.0, 1.0)
    prompt[1], completion[1], rpi[1] = 5.1, 0.1, 8.0
    tokens[1] = instances[1] * (prompt[1] + completion[1])
    cell = tokens[None, :] * rng.uniform(0.8, 1.2, size=(M, N))
    cell[O == 0] = np.nan
    cell_inst = np.tile(instances, (M, 1))
    cell_inst[: M // 2, 2] -= 7.0       # model-dependent instance count
    cell_inst[O == 0] = np.nan
    tables = {"tokens": tokens, "instances": instances,
              "requests": tokens * rpi, "rpi": rpi, "prompt": prompt,
              "completion": completion, "cell": cell,
              "cell_instances": cell_inst}
    names = [f"bench_{j}" for j in range(N)]
    return B, O, names, tables


# Cost caveat of synthetic benchmark 1 (the format of etc.COST_CAVEATS)
CAVEAT = {"bench_1": {"meta_caveat": "bench_1's reported",
                      "method": "multiple_choice_separate_calibrated",
                      "n_choices": 4,
                      "ablation_models": ["a", "b", "c"],
                      "ablation_prompt_tokens_original": [15.0, 16.0, 30.0],
                      "ablation_prompt_tokens_calibrated": [5.0, 5.0, 5.0],
                      "source": "synthetic"}}


def _same_result(a, b):
    return (a["selected"] == b["selected"]
            and a["singleton"] == b["singleton"]
            and a["value"] == b["value"]
            and a["tie_breaks"] == b["tie_breaks"]
            and a["index_ties"] == b["index_ties"])


# ---------------------------------------------------------------------------
# proxy_budgeted_select == budgeted_select for rank == true costs
# ---------------------------------------------------------------------------

class TestProxyReducesToDirect:
    @pytest.mark.parametrize("objective", BUDGET_OBJECTIVES)
    def test_random_instances(self, objective):
        n_checked = 0
        for seed in range(6):
            rng = np.random.default_rng(100 + seed)
            N = 9
            R = _corr(N, rank=3, seed=seed)
            # integer costs with ties, as MMLU's 100-question subjects
            costs = rng.integers(1, 5, size=N).astype(float) * 100.0
            masks = [None, rng.uniform(size=N) > 0.25]
            for el in masks:
                for frac in (0.05, 0.15, 0.3, 0.6, 1.0):
                    budget = frac * costs.sum()
                    ref = budgeted_select(R, costs, budget, objective,
                                          eligible=el)
                    got = etc.proxy_budgeted_select(R, costs, costs, budget,
                                                    objective, el)
                    assert _same_result(ref, got), (seed, frac, objective)
                    n_checked += 1
        assert n_checked == 60

    @pytest.mark.parametrize("objective", BUDGET_OBJECTIVES)
    def test_exchangeable_ties(self, objective):
        """Identity matrix and equal costs: every ratio ties."""
        R = np.eye(6)
        costs = np.full(6, 10.0)
        for budget in (5.0, 10.0, 35.0, 60.0):
            ref = budgeted_select(R, costs, budget, objective)
            got = etc.proxy_budgeted_select(R, costs, costs, budget,
                                            objective)
            assert _same_result(ref, got)

    def test_unknown_objective(self):
        with pytest.raises(ValueError):
            budgeted_select(np.eye(3), np.ones(3), 2.0, "logdet")
        with pytest.raises(ValueError):
            etc.proxy_budgeted_select(np.eye(3), np.ones(3), np.ones(3),
                                      2.0, "logdet")


# ---------------------------------------------------------------------------
# Trace objective of budgeted_select
# ---------------------------------------------------------------------------

class TestTraceObjective:
    @pytest.mark.parametrize("seed", range(4))
    def test_uniform_costs_match_greedy_trace(self, seed):
        R = _corr(12, rank=4, seed=seed)
        ref = greedy_trace_select(R, 12)["selected"]
        got = budgeted_greedy(R, np.ones(12), np.inf, "trace")
        assert got == ref
        el = np.ones(12, dtype=bool)
        el[[2, 7]] = False
        ref = greedy_trace_select(R, 12, eligible=el)["selected"]
        got = budgeted_greedy(R, np.ones(12), np.inf, "trace", eligible=el)
        assert got == ref

    def test_value_is_explained_variance(self):
        R = _corr(8, rank=3, seed=5)
        res = budgeted_select(R, np.ones(8), 3.0, "trace")
        A = res["selected"]
        comp = [j for j in range(8) if j not in A]
        cond = (R[np.ix_(comp, comp)] - R[np.ix_(comp, A)]
                @ np.linalg.solve(R[np.ix_(A, A)], R[np.ix_(A, comp)]))
        assert res["value"] == pytest.approx(8.0 - np.trace(cond), rel=1e-10)

    def test_ratio_ties_lowest_index(self):
        res = budgeted_select(np.eye(5), np.full(5, 2.0), 6.0, "trace")
        assert res["selected"] == [0, 1, 2]
        assert res["tie_breaks"] == res["index_ties"]
        assert [n for _, n in res["index_ties"]] == [5, 4, 3]

    def test_singleton_fallback(self):
        """A cheap independent benchmark has the best ratio but leaves no
        room for the expensive cluster; the cluster's hub alone explains
        far more variance."""
        R = np.full((7, 7), 0.8)
        np.fill_diagonal(R, 1.0)
        R = np.block([[R, np.zeros((7, 1))], [np.zeros((1, 7)), np.eye(1)]])
        R[0, 1:7] = R[1:7, 0] = 0.9           # benchmark 0 is the hub
        costs = np.array([10.0] * 7 + [1.0])
        res = budgeted_select(R, costs, 10.0, "trace")
        assert res["singleton"] and res["selected"] == [0]
        greedy = etc.proxy_budgeted_select(R, costs, costs, 10.0, "trace",
                                           singleton=False)
        assert greedy["selected"] == [7]
        assert res["value"] > greedy["value"]
        assert np.linalg.eigvalsh(R).min() > 0

    def test_cost_budget_methods(self):
        """knap_trace / card_trace of eval_cost_budget on one split."""
        B, O, names, tables = _synthetic(M=30, N=8, seed=3, miss=0.0)
        R = _corr(8, seed=3)
        s = next(iter(cv_splits(30, 5, 1, 0.1, 42)))
        el = np.ones(8, dtype=bool)
        costs = tables["instances"]
        orders = split_orders(R, costs, s, el)
        assert orders["card_trace"]["selected"] == \
            greedy_trace_select(R, 8)["selected"]
        budget = 0.3 * costs.sum()
        knap = select_for_budget("knap_trace", R, costs, budget, orders, s,
                                 el)
        assert knap["selected"] == budgeted_greedy(R, costs, budget, "trace")
        card = select_for_budget("card_trace", R, costs, budget, orders, s,
                                 el)
        assert card["selected"] == order_fill(
            orders["card_trace"]["selected"], costs, budget)
        assert costs[card["selected"]].sum() <= budget


# ---------------------------------------------------------------------------
# Facility-location objective (facloc, the headline method)
# ---------------------------------------------------------------------------

class TestFacilityLocation:
    @pytest.mark.parametrize("seed", range(4))
    def test_uniform_costs_match_facility_location_select(self, seed):
        """Uniform cost model: the knapsack under k x cost and the full
        fill order are facility_location_select's (cardinality) greedy,
        with and without an eligibility mask."""
        N = 12
        R = _corr(N, rank=4, seed=seed)
        el = np.ones(N, dtype=bool)
        el[[2, 7]] = False
        for mask in (None, el):
            ref = facility_location_select(R, N, eligible=mask)["selected"]
            order = etc.proxy_budgeted_select(
                R, np.ones(N), np.ones(N), np.inf, "facloc", mask,
                singleton=False)["selected"]
            assert order == ref
            for k in (1, 3, 6):
                got = etc.proxy_budgeted_select(R, np.ones(N), np.ones(N),
                                                float(k), "facloc", mask)
                assert got["selected"] == ref[:k]
                assert not got["singleton"]

    def test_fill_design_is_facility_location_order(self, run):
        """End to end: main fill facloc_uniform buys facility_location_
        select's order of the split skip-and-continue under the true
        token budget."""
        from cv_split import eligible_mask
        from eval_entropy_vs_mi import estimate_corr
        out, tables, paired, fracs = run
        B, O, names, _ = _synthetic(M=40, N=8, seed=9)
        for i, s in enumerate(cv_splits(40, 4, 1, 0.1, etc.SEED)):
            el = eligible_mask(O[s.train_idx]) & out["priced"]
            R, _, _ = estimate_corr(B[s.train_idx], O[s.train_idx])
            order = facility_location_select(R, 8, eligible=el)["selected"]
            true = out["main_true_costs"][i]
            for b in range(len(fracs)):
                sel = out["main_fill_facloc_uniform_selected"][i, b]
                assert sel[sel >= 0].tolist() == order_fill(
                    order, true, out["main_budgets"][i, b])

    def test_facloc_appended_leaves_other_methods_bit_identical(
            self, run, monkeypatch):
        """Without facloc (the three objectives before it), every array of
        the other methods is the same, bit for bit; the direct-knapsack
        check gains one objective row at the end."""
        out, tables, paired, fracs = run
        B, O, names, _ = _synthetic(M=40, N=8, seed=9)
        objectives = ["mi", "trace", "entropy"]
        monkeypatch.setattr(etc, "OBJECTIVES", objectives)
        monkeypatch.setattr(etc, "METHODS", [
            f"{o}_{c}" for o in objectives for c in etc.COST_MODELS]
            + ["random"])
        base = etc.run_arrays(B, O, names, tables, fracs, n_repeats=1,
                              n_folds=4, verbose=False)
        new = set(out) - set(base) - {"runtime_s"}
        assert new and all("_facloc_" in k for k in new)
        for key, val in base.items():
            val = np.asarray(val)
            got = np.asarray(out[key])
            if key in ("objectives", "methods"):
                assert [m for m in got if "facloc" not in m] == list(val)
                continue
            if key.endswith("_knap_direct_mismatch"):
                got = got[:, :len(objectives)]
            assert np.array_equal(val, got,
                                  equal_nan=val.dtype.kind == "f"), key


# ---------------------------------------------------------------------------
# Proxy ranking, true feasibility
# ---------------------------------------------------------------------------

class TestProxySelection:
    def test_rank_costs_decide_the_order(self):
        """Equal gains; ranking costs favor 0, true costs favor 1."""
        R = np.eye(2)
        rank, true = np.array([1.0, 10.0]), np.array([10.0, 1.0])
        got = etc.proxy_budgeted_select(R, rank, true, 10.0, "entropy")
        assert got["selected"] == [0] and got["spent"] == 10.0
        direct = budgeted_select(R, true, 10.0, "entropy")
        assert direct["selected"] == [1]

    @pytest.mark.parametrize("objective", BUDGET_OBJECTIVES)
    def test_true_budget_always_holds(self, objective):
        rng = np.random.default_rng(11)
        N = 10
        for seed in range(5):
            R = _corr(N, seed=20 + seed)
            true = rng.uniform(1, 100, size=N)
            el = rng.uniform(size=N) > 0.2
            for rank in (rng.uniform(1, 100, size=N), np.ones(N), true):
                order = etc.proxy_budgeted_select(
                    R, rank, rank, np.inf, objective, el,
                    singleton=False)["selected"]
                for frac in (0.01, 0.1, 0.3, 0.7):
                    budget = frac * true.sum()
                    res = etc.proxy_budgeted_select(R, rank, true, budget,
                                                    objective, el)
                    for sel in (res["selected"],
                                order_fill(order, true, budget)):
                        assert etc.sequential_cost(sel, true) <= budget
                        assert all(el[j] for j in sel)
                        assert len(set(sel)) == len(sel)
                    if true[el].min() <= budget:
                        assert res["selected"], "something affordable"
                    else:
                        assert res["selected"] == []

    def test_suite_r2(self):
        """Complete data: the unselected residual equals
        impute_and_evaluate's; selected cells count as known; empty = 0;
        a model without observed picks is predicted by the mean."""
        from eval_entropy_vs_mi import impute_and_evaluate
        rng = np.random.default_rng(5)
        N, M = 6, 9
        R = _corr(N, seed=8)
        B = rng.normal(size=(M, N))
        O = np.ones((M, N))
        mean, std = np.zeros(N), np.ones(N)
        X = np.clip(B, -10, 10)
        sel = [1, 4]
        unsel = [j for j in range(N) if j not in sel]
        r2 = impute_and_evaluate(B, O, sel, R, mean, std, N)["r2"]
        r2s = etc.suite_r2(B, O, sel, R, mean, std)
        tot_all, tot_un = np.sum(X ** 2), np.sum(X[:, unsel] ** 2)
        assert (1 - r2s) * tot_all == pytest.approx((1 - r2) * tot_un,
                                                    rel=1e-12)
        assert etc.suite_r2(B, O, [], R, mean, std) == 0.0
        assert etc.suite_r2(B, O, list(range(N)), R, mean, std) == 1.0
        O2 = O.copy()
        O2[0, sel] = 0                       # model 0: no observed pick
        B2 = B * O2
        r2s2 = etc.suite_r2(B2, O2, sel, R, mean, std)
        ref = etc.suite_r2(B2[1:], O2[1:], sel, R, mean, std)
        X2 = np.clip(B2, -10, 10) * O2
        res1 = (1 - ref) * np.sum(X2[1:] ** 2)
        assert (1 - r2s2) * np.sum(X2 ** 2) == pytest.approx(
            res1 + np.sum(X2[0, unsel] ** 2), rel=1e-12)

    def test_max_affordable(self):
        c = np.array([5.0, 1.0, 3.0, 2.0])
        assert etc.max_affordable(c, 6.0) == 3
        assert etc.max_affordable(c, 0.5) == 0
        assert etc.max_affordable(c, 6.0, [True, False, True, True]) == 2


# ---------------------------------------------------------------------------
# Costs and no-leakage
# ---------------------------------------------------------------------------

def _prep(tables, names, caveats=None, mode="impute"):
    return etc.prepare_costs(tables, names, caveats, mode)


class TestCostsAndLeakage:
    def test_scenario_costs(self):
        B, O, names, tables = _synthetic(M=30, N=6, seed=1)
        tab = _prep(tables, names)
        s = next(iter(cv_splits(30, 5, 1, 0.1, 42)))
        t, r, inst, fb = etc.scenario_costs("main", tab, s)
        assert t is r and fb is None
        assert np.array_equal(t, tables["requests"])   # no caveat: exact
        assert np.array_equal(inst, tables["instances"])
        t, r, _, _ = etc.scenario_costs("single", tab, s)
        assert np.array_equal(t, tables["tokens"]) and t is r
        true, model, inst, fallback = etc.scenario_costs("cell", tab, s)
        C = tables["cell"] * tables["rpi"][None, :]     # all requests
        CI = tables["cell_instances"]
        for j in range(6):
            assert model[j] == pytest.approx(np.nanmean(C[s.train_idx, j]))
            assert inst[j] == pytest.approx(np.nanmean(CI[s.train_idx, j]))
            va = C[s.val_idx, j]
            if np.isfinite(va).any():
                assert true[j] == pytest.approx(np.nanmean(va))
                assert not fallback[j]
            else:
                assert true[j] == model[j] and fallback[j]

    def test_cell_cost_models_ignore_validation_models(self):
        B, O, names, tables = _synthetic(M=30, N=6, seed=2)
        s = next(iter(cv_splits(30, 5, 1, 0.1, 42)))
        tab = _prep(tables, names)
        true0, model0, inst0, _ = etc.scenario_costs("cell", tab, s)
        t2 = dict(tab, cell=tab["cell"].copy(),
                  cell_instances=tab["cell_instances"].copy())
        t2["cell"][s.val_idx] *= 7.0
        t2["cell_instances"][s.val_idx] *= 5.0
        true1, model1, inst1, _ = etc.scenario_costs("cell", t2, s)
        assert np.array_equal(model0, model1)
        assert np.array_equal(inst0, inst1)
        assert not np.allclose(true0, true1)
        t3 = dict(tab, cell=tab["cell"].copy(),
                  cell_instances=tab["cell_instances"].copy())
        t3["cell"][s.train_idx] *= 3.0
        t3["cell_instances"][s.train_idx] *= 2.0
        _, model3, inst3, _ = etc.scenario_costs("cell", t3, s)
        assert np.allclose(model3, 3.0 * model0)
        assert np.allclose(inst3, 2.0 * inst0)

    def test_cell_without_instance_table(self):
        B, O, names, tables = _synthetic(M=30, N=6, seed=2)
        s = next(iter(cv_splits(30, 5, 1, 0.1, 42)))
        tab = _prep(dict(tables, cell_instances=None), names)
        _, _, inst, _ = etc.scenario_costs("cell", tab, s)
        assert np.array_equal(inst, tables["instances"])

    def test_fallback_without_validation_observation(self):
        B, O, names, tables = _synthetic(M=30, N=6, seed=4, miss=0.0)
        s = next(iter(cv_splits(30, 5, 1, 0.1, 42)))
        t2 = _prep(dict(tables, cell=tables["cell"].copy()), names)
        t2["cell"][s.val_idx, 3] = np.nan
        true, model, _, fallback = etc.scenario_costs("cell", t2, s)
        assert fallback[3] and fallback.sum() == 1
        assert true[3] == model[3]

    def test_selections_ignore_validation_scores(self):
        B, O, names, tables = _synthetic(M=36, N=7, seed=6)
        s = list(cv_splits(36, 6, 1, 0.1, 42))[2]
        fracs = [0.1, 0.3, 0.6]
        a = etc.evaluate_split(B, O, s, tables, fracs, etc.SCENARIOS)
        B2 = B.copy()
        rng = np.random.default_rng(0)
        B2[s.val_idx] = (B2[s.val_idx]
                         + rng.normal(size=B2[s.val_idx].shape)) * O[s.val_idx]
        b = etc.evaluate_split(B2, O, s, tables, fracs, etc.SCENARIOS)
        changed = 0
        for scen in a["scen"]:
            for key, recs in a["scen"][scen]["results"].items():
                for ra, rb in zip(recs, b["scen"][scen]["results"][key]):
                    assert ra["selected"] == rb["selected"], (scen, key)
                    changed += ra["r2"] != rb["r2"]
        assert changed > 0


# ---------------------------------------------------------------------------
# Cost caveats
# ---------------------------------------------------------------------------

class TestCostCaveats:
    def test_imputation_arithmetic(self):
        """r = median ratio; 4 original + 4 calibration requests."""
        r, fs, fm = etc.caveat_imputation(CAVEAT["bench_1"], 5.1, 0.1, 8.0)
        assert r == pytest.approx(3.2)
        assert fs == pytest.approx((3.2 * 5.1 + 0.1) / 5.2)
        assert fm == pytest.approx((4 * (3.2 * 5.1 + 0.1) + 4 * 5.2)
                                   / (8 * 5.2))
        with pytest.raises(ValueError):            # not 2 x n_choices
            etc.caveat_imputation(CAVEAT["bench_1"], 5.1, 0.1, 4.0)
        with pytest.raises(ValueError):            # no prompt length
            etc.caveat_imputation(CAVEAT["bench_1"], np.nan, 0.1, 8.0)

    def test_helm_classic_openbookqa_numbers(self):
        """The registered OpenbookQA entry with the meta's p, o and
        published costs gives the docstring's numbers."""
        entry = etc.COST_CAVEATS["helm_classic"]["OpenbookQA"]
        p, o = 5.098125, 0.097875                   # benchmark_cost means
        r, fs, fm = etc.caveat_imputation(entry, p, o, 8.0)
        assert r == pytest.approx(15.024 / 5.27)
        assert 20784.0 * fm == pytest.approx(39656, abs=1)
        assert 2598.0 * fs == pytest.approx(7316, abs=1)
        assert 20784.0 * fm / 500 == pytest.approx(79.3, abs=0.05)

    @pytest.mark.parametrize("mode", etc.CAVEAT_MODES)
    def test_modes(self, mode):
        B, O, names, tables = _synthetic(M=30, N=6, seed=3)
        tab = etc.prepare_costs(tables, names, CAVEAT, mode,
                                ["bench_1's reported cost is wrong"])
        other = np.arange(6) != 1
        # unflagged benchmarks keep the published values exactly
        assert np.array_equal(tab["true_main"][other],
                              tables["requests"][other])
        assert np.array_equal(tab["true_single"][other],
                              tables["tokens"][other])
        assert np.array_equal(tab["mult_req"][other], tables["rpi"][other])
        assert list(tab["caveat_flag"]) == [False, True] + [False] * 4
        (rec,) = tab["caveats"]
        assert rec["caveat_in_meta"] and rec["benchmark"] == "bench_1"
        fm, fs = rec["imputed_factor_main"], rec["imputed_factor_single"]
        if mode == "impute":
            assert tab["priced"].all()
            assert tab["true_main"][1] == pytest.approx(
                tables["requests"][1] * fm)
            assert tab["true_single"][1] == pytest.approx(
                tables["tokens"][1] * fs)
            assert tab["mult_req"][1] == pytest.approx(8.0 * fm)
        else:
            assert tab["true_main"][1] == tables["requests"][1]
            assert tab["true_single"][1] == tables["tokens"][1]
            assert tab["priced"].sum() == (5 if mode == "exclude" else 6)
            assert tab["priced"][1] == (mode != "exclude")
        assert etc.mode_suffix(mode) == ("" if mode == "impute"
                                         else f"_{mode}")
        assert etc.macro_prefix(mode) == {"impute": "Tok",
                                          "exclude": "TokExclude",
                                          "keep": "TokKeep"}[mode]

    def test_errors(self):
        B, O, names, tables = _synthetic(M=30, N=6, seed=3)
        with pytest.raises(ValueError):
            etc.prepare_costs(tables, names, CAVEAT, "drop")
        with pytest.raises(ValueError):
            etc.prepare_costs(tables, names, {"nope": CAVEAT["bench_1"]})
        tab = etc.prepare_costs(tables, names, CAVEAT, "keep", [])
        assert not tab["caveats"][0]["caveat_in_meta"]

    def test_random_order_common_random_numbers(self):
        """Excluding a benchmark removes it from the shared random order
        and leaves the order of the others unchanged."""
        R = np.eye(9)
        el = np.ones(9, dtype=bool)
        el[4] = False
        priced = np.ones(9, dtype=bool)
        priced[[1, 6]] = False
        for s in list(cv_splits(40, 4, 2, 0.1, 42)):
            seed = s.seed(etc.SEED + RANDOM_SEED_OFFSET)
            full = random_select(R, 9, seed=seed, eligible=el)["selected"]
            sub = random_select(R, 9, seed=seed,
                                eligible=el & priced)["selected"]
            assert sub == [j for j in full if priced[j]]

    def test_exclude_end_to_end(self):
        """Excluded: never bought by any method (random included), budgets
        over the priced benchmarks, still a target of both metrics."""
        B, O, names, tables = _synthetic(M=40, N=8, seed=9, miss=0.0)
        tab = etc.prepare_costs(tables, names, CAVEAT, "exclude")
        fracs = [0.05, 0.3, 1.0]
        out = etc.run_arrays(B, O, names, tab, fracs, n_repeats=1,
                             n_folds=4, verbose=False)
        assert list(out["priced"]) == [True, False] + [True] * 6
        for k in out:
            if k.endswith("_selected"):
                assert not np.any(out[k] == 1), k
        tot = np.sum(tab["true_main"][tab["priced"]])
        np.testing.assert_allclose(out["main_budgets"][0],
                                   np.array(fracs) * tot)
        assert out["total_tokens"] == pytest.approx(tot)
        # f = 100%: random buys (up to summation rounding) all 7 priced
        # benchmarks; benchmark 1 stays an imputation target, so R² is
        # defined
        size = out["main_knap_random_size"][:, -1]
        assert np.all(size >= 6) and np.all(size <= 7)
        assert np.all(np.isfinite(out["main_knap_random_r2"][:, -1]))
        assert np.all(out["main_knap_random_r2_suite"][:, -1] < 1.0)

    def test_impute_end_to_end(self):
        B, O, names, tables = _synthetic(M=40, N=8, seed=9)
        tab = etc.prepare_costs(tables, names, CAVEAT, "impute")
        out = etc.run_arrays(B, O, names, tab, [0.1], n_repeats=1,
                             n_folds=4, verbose=False)
        assert np.array_equal(out["costs_true_main"], tab["true_main"])
        assert out["costs_true_main"][1] > tables["requests"][1]
        np.testing.assert_array_equal(out["main_true_costs"],
                                      np.tile(tab["true_main"], (4, 1)))
        assert str(out["cost_caveat_mode"]) == "impute"


# ---------------------------------------------------------------------------
# End to end on a synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run():
    """run_arrays on a synthetic dataset (1 x 4-fold CV, 3 budgets)."""
    B, O, names, tables = _synthetic(M=40, N=8, seed=9)
    fracs = [0.02, 0.1, 0.4]
    out = etc.run_arrays(B, O, names, tables, fracs, n_repeats=1,
                         n_folds=4, verbose=False)
    out["runtime_s"] = np.float64(0.0)
    paired = etc.paired_rows("helm_synth", out)
    return out, tables, paired, fracs


class TestEndToEnd:

    def test_layout_and_checks(self, run):
        out, tables, paired, fracs = run
        S, nb, N = 4, len(fracs), 8
        assert list(out["scenarios"]) == ["main", "cell", "single"]
        assert out["priced"].all() and not out["caveat_flag"].any()
        assert np.array_equal(out["costs_true_main"], tables["requests"])
        assert np.array_equal(out["costs_true_single"], tables["tokens"])
        assert out["total_tokens"] == np.sum(tables["requests"])
        for scen in out["scenarios"]:
            for d in etc.DESIGNS:
                for m in etc.METHODS:
                    k = f"{scen}_{d}_{m}"
                    assert out[f"{k}_r2"].shape == (S, nb)
                    assert out[f"{k}_selected"].shape == (S, nb, N)
                    assert not out[f"{k}_violations"].any()
                    size = out[f"{k}_size"]
                    assert np.array_equal(np.isnan(out[f"{k}_r2"]),
                                          size == 0)
                    suite = out[f"{k}_r2_suite"]
                    assert np.isfinite(suite).all()
                    assert np.all(suite[size == 0] == 0.0)
                    assert np.all(out[f"{k}_cost"]
                                  <= out[f"{scen}_budgets"])
                np.testing.assert_array_equal(
                    out[f"{scen}_knap_random_r2"],
                    out[f"{scen}_fill_random_r2"])
        assert not out["main_knap_direct_mismatch"].any()
        assert not out["single_knap_direct_mismatch"].any()
        assert np.array_equal(out["split_id"], np.arange(S))
        for key in ("rep", "fold", "n_eligible", "n_train", "n_val"):
            assert out[key].shape == (S,)
        tab = etc.summary_tables(out)
        st = tab["main"]["knap"]["mi_tokens"]
        assert st["n_finite_suite"] == [S] * nb
        assert all(0 <= v <= S for v in st["n_finite"])

    def test_single_skipped_when_identical(self):
        B, O, names, tables = _synthetic(M=30, N=5, seed=12)
        tables = dict(tables, requests=tables["tokens"].copy(),
                      rpi=np.ones(5))
        tab = _prep(tables, names)
        assert etc.active_scenarios(etc.SCENARIOS, tab) == ["main", "cell"]
        # an imputed caveat alone also makes single differ from main
        tab = _prep(_synthetic(M=30, N=5, seed=12)[3], names, CAVEAT)
        assert "single" in etc.active_scenarios(etc.SCENARIOS, tab)

    def test_pairs_and_orientation(self, run):
        out, tables, paired, fracs = run
        assert etc.oriented("mi_instances", "mi_tokens") == ("mi_tokens",
                                                             "mi_instances")
        assert etc.oriented("trace_tokens", "mi_tokens") == ("mi_tokens",
                                                             "trace_tokens")
        assert etc.oriented("trace_uniform", "entropy_uniform") == (
            "entropy_uniform", "trace_uniform")
        assert etc.oriented("random", "trace_uniform")[1] == "random"
        # facloc (unranked by paired_stats.METHOD_ORDER) after the other
        # objectives, before random
        assert etc.oriented("facloc_tokens", "mi_tokens") == (
            "mi_tokens", "facloc_tokens")
        assert etc.oriented("facloc_uniform", "trace_uniform") == (
            "trace_uniform", "facloc_uniform")
        assert etc.oriented("random", "facloc_instances") == (
            "facloc_instances", "random")
        pairs = etc.method_pairs()
        # 4 objectives x 3 cost-model pairs + 12 methods vs random + 3 cost
        # models x 6 objective pairs
        assert len(pairs) == len(set(pairs)) == 42
        assert ("trace_tokens", "trace_instances") in pairs
        assert ("facloc_tokens", "facloc_uniform") in pairs
        assert ("entropy_tokens", "facloc_tokens") in pairs
        n_cells = (len(out["scenarios"]) * len(etc.DESIGNS) * len(fracs)
                   * len(etc.METRICS))
        assert len(paired) == n_cells * len(pairs)

    def test_macros_and_json(self, run):
        out, tables, paired, fracs = run
        undefined = []
        mac = etc.macros_of("helm_synth", out, tables, paired, undefined)
        assert mac and all(re.fullmatch(r"[A-Za-z]+", k) for k in mac)
        assert mac["TokViolationsHelmSynth"] == 0
        assert mac["TokDirectMismatchHelmSynth"] == 0
        assert "TokPairDiffHelmSynthMiTokensVsMiInstancesBten" in mac
        assert "TokCellPairDiffHelmSynthTraceTokensVsTraceUniformBforty" \
            in mac
        assert "TokSinglePairDiffHelmSynthMiTokensVsMiUniformBten" in mac
        assert "TokSuitePairWinHelmSynthEntropyTokensVsRandomBforty" in mac
        assert "TokFillSuiteRsqHelmSynthTraceUniformBten" in mac
        # facloc follows the trace pattern
        assert "TokSuiteRsqHelmSynthFaclocTokensBten" in mac
        assert "TokSuitePairDiffHelmSynthFaclocTokensVsRandomBten" in mac
        assert "TokSuitePairDiffHelmSynthTraceTokensVsFaclocTokensBten" \
            in mac
        assert "TokCellPairDiffHelmSynthFaclocTokensVsFaclocUniformBforty" \
            in mac
        assert mac["TokNCaveatsHelmSynth"] == 0
        assert all(isinstance(v, (int, float)) and math.isfinite(v)
                   for v in mac.values())
        assert not set(mac) & set(undefined)
        rep = etc.json_report("helm_synth", out, tables, paired, mac,
                              {"repeats": 1}, undefined)
        text = json.dumps(rep, allow_nan=False)
        back = json.loads(text)
        assert back["checks"]["violations_total"] == 0
        assert back["checks"]["unpriced_selected"] == 0
        assert back["primary_metric"] == "r2_suite"
        assert set(back["summary"]) == set(out["scenarios"])
        assert math.isclose(back["correlations"]["true_main_vs_instances"]
                            ["spearman"], mac["TokSpearmanHelmSynth"])
        assert back["macros_undefined"] == sorted(undefined)
        share = back["selection_share_main_knap"]["random"]
        assert np.asarray(share).shape == (len(fracs), 8)

    def test_undefined_macros_left_out(self):
        """A budget below the cheapest benchmark: nothing is selected, so
        every R² mean and every paired R² statistic there is undefined and
        listed instead of emitted; full-suite R² is 0, n is 0."""
        B, O, names, tables = _synthetic(M=40, N=8, seed=9)
        tab = etc.prepare_costs(tables, names, CAVEAT, "exclude")
        fracs = [1e-6, 0.3]
        out = etc.run_arrays(B, O, names, tab, fracs, n_repeats=1,
                             n_folds=4, verbose=False)
        out["runtime_s"] = np.float64(0.0)
        paired = etc.paired_rows("helm_synth", out)
        undefined = []
        mac = etc.macros_of("helm_synth", out, tab, paired, undefined)
        w = "B" + etc.num_words(100 * 1e-6)
        assert f"TokExcludeRsqHelmSynthMiTokens{w}" in undefined
        assert f"TokExcludePairDiffHelmSynthMiTokensVsMiInstances{w}" \
            in undefined
        assert mac[f"TokExcludeSuiteRsqHelmSynthMiTokens{w}"] == 0.0
        assert mac[f"TokExcludePairNHelmSynthMiTokensVsMiInstances{w}"] == 0
        assert mac[f"TokExcludeEmptyHelmSynth{w}"] == 4
        assert mac["TokExcludeNPricedHelmSynth"] == 7
        assert mac["TokExcludeNCaveatsHelmSynth"] == 1
        assert "TokExcludeCaveatPublishedHelmSynthBenchOne" in mac
        assert all(math.isfinite(v) for v in mac.values())
        assert all(re.fullmatch(r"[A-Za-z]+", k) for k in undefined)


# ---------------------------------------------------------------------------
# Fixed per-benchmark overhead (--fixed-cost)
# ---------------------------------------------------------------------------

FIXED_RATIOS = [0.0, 0.5, 2.0]


@pytest.fixture(scope="module")
def fixed_run():
    """run_fixed_arrays on a synthetic dataset with the cost caveat imputed
    (1 x 4-fold CV, 3 budgets, 3 overhead levels)."""
    B, O, names, tables = _synthetic(M=40, N=8, seed=9)
    tab = etc.prepare_costs(tables, names, CAVEAT, "impute")
    fracs = [0.02, 0.1, 0.4]
    out = etc.run_fixed_arrays(B, O, names, tab, fracs, FIXED_RATIOS,
                               n_repeats=1, n_folds=4, verbose=False)
    paired = etc.fixed_paired_rows("helm_synth", out)
    return B, O, names, tab, fracs, out, paired


class TestFixedCost:
    def test_fixed_tables(self):
        B, O, names, tables = _synthetic(M=30, N=8, seed=2)
        for mode in ("impute", "exclude"):
            tab = etc.prepare_costs(tables, names, CAVEAT, mode)
            same, c0, i0 = etc.fixed_tables(tab, 0)
            assert same is tab and c0 == 0.0 and i0 == 0.0
            new, c0, i0 = etc.fixed_tables(tab, 0.5)
            pr = tab["priced"]
            assert c0 == pytest.approx(0.5 * np.median(tab["true_main"][pr]))
            assert i0 == pytest.approx(0.5 * np.median(tab["instances"][pr]))
            np.testing.assert_allclose(new["true_main"] - tab["true_main"],
                                       c0)
            np.testing.assert_allclose(new["instances"] - tab["instances"],
                                       i0)
            for k in ("true_single", "cell", "priced", "tokens", "mult_req"):
                assert np.array_equal(np.asarray(new[k]),
                                      np.asarray(tab[k]), equal_nan=True)
        with pytest.raises(ValueError):
            etc.fixed_tables(tab, -1.0)

    def test_level_zero_is_main_scenario(self, fixed_run):
        B, O, names, tab, fracs, out, paired = fixed_run
        plain = etc.run_arrays(B, O, names, tab, fracs, n_repeats=1,
                               n_folds=4, scenarios=["main"], verbose=False)
        n_level = 0
        for k, v in plain.items():
            if k.startswith("main_") or k == "max_count":
                a = np.asarray(v)
                assert np.array_equal(a, out[k][:, 0],
                                      equal_nan=a.dtype.kind == "f"), k
                n_level += 1
            elif k in etc.FIXED_LEVEL_VECTORS:
                assert np.array_equal(np.asarray(v), out[k][0]), k
            else:
                assert np.array_equal(np.asarray(v), np.asarray(out[k])), k
        assert n_level > 100

    def test_true_budget_with_overhead(self, fixed_run):
        B, O, names, tab, fracs, out, paired = fixed_run
        pr = tab["priced"]
        for f, (ratio, c0) in enumerate(zip(out["fixed_ratios"],
                                            out["fixed_costs"])):
            true = out["costs_true_main"][f]
            np.testing.assert_allclose(true, tab["true_main"] + c0)
            np.testing.assert_allclose(
                out["main_budgets"][:, f],
                np.tile(np.array(fracs) * (tab["true_main"][pr].sum()
                                           + pr.sum() * c0), (4, 1)))
            assert out["total_tokens"][f] == pytest.approx(true[pr].sum())
        for k in out:
            if k.endswith("_violations"):
                assert not out[k].any(), k
        assert not out["main_knap_direct_mismatch"].any()
        for d in etc.DESIGNS:
            for m in etc.METHODS:
                k = f"main_{d}_{m}"
                assert np.all(out[f"{k}_cost"] <= out["main_budgets"]), k
                assert out[f"{k}_r2_suite"].shape == (4, 3, len(fracs))

    def test_random_fills_shared_permutation(self, fixed_run):
        """Common random numbers: at every level random is the fill of the
        split's one random_select permutation under that level's costs."""
        from cv_split import eligible_mask
        from eval_entropy_vs_mi import estimate_corr
        B, O, names, tab, fracs, out, paired = fixed_run
        for i, s in enumerate(cv_splits(40, 4, 1, 0.1, etc.SEED)):
            el = eligible_mask(O[s.train_idx]) & tab["priced"]
            R, _, _ = estimate_corr(B[s.train_idx], O[s.train_idx])
            order = random_select(R, 8, seed=s.seed(etc.SEED
                                                    + RANDOM_SEED_OFFSET),
                                  eligible=el)["selected"]
            for f in range(len(FIXED_RATIOS)):
                true = out["costs_true_main"][f]
                for b in range(len(fracs)):
                    sel = out["main_knap_random_selected"][i, f, b]
                    assert sel[sel >= 0].tolist() == order_fill(
                        order, true, out["main_budgets"][i, f, b])

    def test_large_overhead_tokens_like_uniform(self):
        """c0 >> tokens_j: ranking by c0 + tokens_j is ranking by count, so
        the trace and MI knapsacks under true costs pick what the uniform
        cost model picks."""
        B, O, names, tables = _synthetic(M=40, N=8, seed=9)
        tab = etc.prepare_costs(tables, names)
        fracs = [0.2, 0.45]
        out = etc.run_fixed_arrays(B, O, names, tab, fracs, [1e6],
                                   n_repeats=1, n_folds=4, verbose=False)
        for o in ("trace", "mi"):
            assert np.array_equal(out[f"main_knap_{o}_tokens_selected"],
                                  out[f"main_knap_{o}_uniform_selected"]), o
        n = out["main_knap_trace_tokens_size"]
        assert np.all(n[:, 0, 0] == 1) and np.all(n[:, 0, 1] == 3)

    def test_paired_macros_json_plot(self, fixed_run, tmp_path,
                                     monkeypatch):
        B, O, names, tab, fracs, out, paired = fixed_run
        n_pairs = len(etc.method_pairs())
        assert len(paired) == (len(FIXED_RATIOS) * len(etc.DESIGNS)
                               * len(etc.METRICS) * len(fracs) * n_pairs)
        assert all(r["scenario"] == "main" for r in paired)
        for p in etc.FIXED_PAIRS:
            assert p in etc.method_pairs()
        r = next(p for p in paired if p["fixed_ratio"] == 2.0
                 and p["design"] == "knap" and p["metric"] == "r2_suite"
                 and p["budget_frac"] == 0.1
                 and p["method_a"] == "trace_tokens"
                 and p["method_b"] == "random")
        d = (out["main_knap_trace_tokens_r2_suite"][:, 2, 1]
             - out["main_knap_random_r2_suite"][:, 2, 1])
        assert r["mean_diff"] == pytest.approx(d.mean())
        assert r["fixed_cost"] == pytest.approx(out["fixed_costs"][2])
        undefined = []
        mac = etc.fixed_macros("helm_synth", out, paired, undefined)
        assert mac and all(k.isascii() and k.isalpha() for k in mac)
        assert all(k.isascii() and k.isalpha() for k in undefined)
        assert all(isinstance(v, (int, float)) and math.isfinite(v)
                   for v in mac.values())
        assert mac["TokFixedViolationsHelmSynth"] == 0
        assert mac["TokFixedDirectMismatchHelmSynth"] == 0
        assert mac["TokFixedOverheadHelmSynthFtwo"] == pytest.approx(
            out["fixed_costs"][2])
        assert mac["TokFixedSuitePairDiffHelmSynthTraceTokensVsRandom"
                   "FtwoBten"] == pytest.approx(d.mean())
        assert mac["TokFixedSuiteRsqHelmSynthMiUniformFzeropointfiveBforty"] \
            == pytest.approx(out["main_knap_mi_uniform_r2_suite"][:, 1, 2]
                             .mean())
        assert "TokFixedSuiteSigPosHelmSynthEntropyTokensVsEntropyUniformFzero" \
            in mac
        rep = etc.fixed_json("helm_synth", out, tab, paired, mac,
                             {"runtime_s": 0.0}, undefined)
        back = json.loads(json.dumps(rep, allow_nan=False))
        assert back["checks"]["violations_total"] == 0
        assert back["checks"]["unpriced_selected"] == 0
        assert [lv["fixed_ratio"] for lv in back["levels"]] == FIXED_RATIOS
        assert back["macros"] == mac
        assert "human_evaluation" in back["notes"]
        monkeypatch.setattr(etc, "OUT", tmp_path)
        path = etc.plot_fixed([("helm_synth", out, paired)])
        assert path.is_file() and path.name == "token_fixed.pdf"

    def test_fixed_log_roundtrip(self, fixed_run, tmp_path, monkeypatch):
        B, O, names, tab, fracs, out, paired = fixed_run
        monkeypatch.setattr(etc, "LOGS", tmp_path)
        np.savez_compressed(tmp_path / "token_fixed_helm_synth.npz", **out)
        back, paired2 = etc.load_fixed_log("helm_synth")
        assert len(paired2) == len(paired)
        for a, b in zip(paired, paired2):
            assert a["method_a"] == b["method_a"]
            assert (a["mean_diff"] == b["mean_diff"]
                    or (math.isnan(a["mean_diff"])
                        and math.isnan(b["mean_diff"])))
