"""Tests for the fixed per-benchmark overhead of eval_cost_budget.py
(--fixed-cost; synthetic data only, a few seconds).

Covers: fixed_overhead (ratio 0 leaves the cost table untouched, c0 =
ratio x median) and the budgets as fractions of the true suite cost; every
method (knapsack, cost-blind orders, cheapest first, random) held to the
true budget for random cost tables, overheads and eligibility masks;
random, cheapest first and the cost-blind orders as skip-and-continue
fills of their split orders under the true costs (random: one shared
permutation for every level, common random numbers); the large-overhead
limit (the trace and MI knapsacks become the cardinality greedy); ratio 0
of the fixed run bit-identical to the plain run(); no leakage of
validation scores into selections; the paired-row layout and orientation,
letters-only finite macros, the JSON payload, the figure and
deterministic reruns.  Facility location (knap_facloc / card_facloc, the
budgeted_select objective "facloc"): unit costs reproduce
baseline_select.facility_location_select (any cost units, eligibility
masks, covariance input), the value is the facility-location objective
of the selection, card_facloc is facility_location_select's order filled
skip-and-continue, the best-singleton fallback, the (1 − 1/e)/2 knapsack
guarantee against the brute-force optimum, and adding the facloc pair
leaves every array of the other methods bit-identical.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import tempfile

# Outputs of the imported eval scripts go to a scratch root unless the
# caller (run_all.sh's tests job) set one
if not os.environ.get("BENCHSELECT_EXPERIMENT_ROOT"):
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-cost-budget-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

import eval_cost_budget as ecb
import eval_token_costs as etc
from baseline_select import facility_location_select, greedy_trace_select
from cv_split import RANDOM_SEED_OFFSET, cv_splits, eligible_mask
from eval_entropy_vs_mi import estimate_corr
from greedy_select import greedy_mi, random_select
from paired_stats import method_rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corr(N, rank=3, seed=0, noise=0.3):
    """Random factor-model correlation matrix (generic: no exact ties)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N, rank))
    S = W @ W.T + noise * np.diag(rng.uniform(0.5, 1.5, size=N))
    d = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def _costs(N, seed):
    """Integer example counts with ties, as MMLU's 100-question subjects."""
    rng = np.random.default_rng(seed)
    return rng.choice([100.0, 100.0, 150.0, 173.0, 400.0, 1534.0], N)


def _synthetic(M=60, N=10, seed=0, miss=0.1):
    """(B, O, names, costs): two-factor scores with missing cells; the last
    benchmark is observed for 3 models only, so it is ineligible on every
    split that holds one of them out (fewer than 3 training models)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(M, 2))
    W = rng.normal(size=(2, N))
    X = 0.5 + 0.1 * (Z @ W) + 0.03 * rng.normal(size=(M, N))
    O = (rng.uniform(size=(M, N)) > miss).astype(float)
    O[:, 0] = 1.0
    O[:, -1] = 0.0
    O[rng.choice(M, 3, replace=False), -1] = 1.0
    return X * O, O, [f"bench_{j}" for j in range(N)], _costs(N, seed + 1)


def _split():
    return next(iter(cv_splits(60, 5, 1, 0.1, ecb.SEED)))


def _fill_oracle(order, costs, budget):
    """Skip-and-continue fill, written out."""
    sel, spent = [], 0.0
    for j in order:
        if spent + costs[j] <= budget:
            sel.append(int(j))
            spent = spent + costs[j]
    return sel


def _facloc_value(Sigma, A):
    """Brute-force facility-location objective F(A) = sum_j max_{i in A}
    max(R_ij, 0) (R_jj = 1) of the correlation matrix of Sigma."""
    if len(A) == 0:
        return 0.0
    d = 1.0 / np.sqrt(np.diag(Sigma))
    S = np.maximum(Sigma * np.outer(d, d), 0.0)
    np.fill_diagonal(S, 1.0)
    return float(np.sum(np.max(S[list(A)], axis=0)))


# Methods of the log before the facility-location pair was appended
BASE_METHODS = ["knap_entropy", "knap_mi", "card_entropy", "card_mi",
                "cheap_first", "random", "knap_trace", "card_trace"]


# ---------------------------------------------------------------------------
# Overhead levels and budgets
# ---------------------------------------------------------------------------

class TestOverhead:
    def test_ratio_zero_leaves_costs_untouched(self):
        costs = _costs(12, 3)
        c0, true = ecb.fixed_overhead(costs, 0)
        assert c0 == 0.0
        assert true is costs or np.shares_memory(true, costs)
        assert true.tobytes() == costs.tobytes()

    @pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 2.0])
    def test_c0_is_ratio_times_median(self, ratio):
        costs = _costs(13, 4)
        c0, true = ecb.fixed_overhead(costs, ratio)
        assert c0 == pytest.approx(ratio * np.median(costs))
        np.testing.assert_allclose(true - costs, c0)
        assert np.all(np.argsort(true, kind="stable")
                      == np.argsort(costs, kind="stable"))

    @pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
    def test_bad_ratio(self, bad):
        with pytest.raises(ValueError):
            ecb.fixed_overhead(np.ones(3), bad)

    def test_budgets_are_fractions_of_true_suite(self):
        costs = _costs(11, 5)
        fracs = [0.01, 0.05, 0.2, 1.0]
        levels, budgets = ecb.fixed_levels(costs, [0, 0.5, 2], fracs)
        assert budgets.shape == (3, 4)
        # ratio 0: the plain run's arithmetic, bit for bit
        assert budgets[0].tolist() == [f * costs.sum() for f in fracs]
        for (c0, _), row in zip(levels, budgets):
            np.testing.assert_allclose(
                row, np.array(fracs) * (costs.sum() + len(costs) * c0))
        # frac 1 buys everything at every level
        for (c0, true), row in zip(levels, budgets):
            assert ecb.sequential_cost(range(len(costs)), true) <= row[-1] * (
                1 + 1e-12)


# ---------------------------------------------------------------------------
# Every method under the true budget
# ---------------------------------------------------------------------------

class TestTrueBudget:
    @pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 2.0])
    def test_every_method_feasible(self, ratio):
        split = _split()
        n_checked = 0
        for seed in range(4):
            N = 14
            R = _corr(N, seed=seed)
            costs = _costs(N, 10 + seed)
            rng = np.random.default_rng(20 + seed)
            for el in (None, rng.uniform(size=N) > 0.2):
                c0, true = ecb.fixed_overhead(costs, ratio)
                fracs = [0.005, 0.02, 0.05, 0.12, 0.3, 0.7]
                budgets = [f * true.sum() for f in fracs]
                orders = ecb.split_orders(R, true, split, el)
                for m in ecb.METHODS:
                    for b, budget in enumerate(budgets):
                        sel = ecb.select_for_budget(m, R, true, budget,
                                                    orders, split, el,
                                                    b)["selected"]
                        assert len(set(sel)) == len(sel)
                        assert ecb.sequential_cost(sel, true) <= budget, (
                            m, ratio, budget)
                        if el is not None:
                            assert all(el[j] for j in sel)
                        if m in ("random", "cheap_first") or m.startswith(
                                "card_"):
                            order = orders[m]["selected"]
                            assert sel == _fill_oracle(order, true, budget)
                        n_checked += 1
        assert n_checked == 4 * 2 * len(ecb.METHODS) * 6

    def test_random_shares_one_permutation(self):
        """Common random numbers: random's order is random_select's
        permutation of the split, whatever the overhead."""
        split = _split()
        R = _corr(12, seed=7)
        costs = _costs(12, 8)
        el = np.ones(12, dtype=bool)
        el[[2, 5]] = False
        ref = random_select(R, 12, seed=split.seed(ecb.SEED
                                                   + RANDOM_SEED_OFFSET),
                            eligible=el)["selected"]
        for ratio in (0, 0.5, 2):
            _, true = ecb.fixed_overhead(costs, ratio)
            orders = ecb.split_orders(R, true, split, el)
            assert orders["random"]["selected"] == ref
            for frac in (0.05, 0.3):
                sel = ecb.select_for_budget("random", R, true,
                                            frac * true.sum(), orders,
                                            split, el)["selected"]
                assert sel == [j for j in ref if j in set(sel)]

    def test_cheap_first_order_unchanged_by_constant(self):
        R = _corr(15, seed=9)
        costs = _costs(15, 9)
        ref = ecb.cheap_first_order(R, costs)["selected"]
        for ratio in (0.1, 1.0, 2.0):
            _, true = ecb.fixed_overhead(costs, ratio)
            assert ecb.cheap_first_order(R, true)["selected"] == ref

    @pytest.mark.parametrize("seed", range(3))
    def test_large_overhead_is_cardinality(self, seed):
        """c0 >> c_j: gain / (c0 + c_j) ranks like the gain, so the trace
        and MI knapsacks pick the cardinality greedy orders (entropy gains
        tie exactly at step 1 and still go to the cheapest)."""
        N = 16
        R = _corr(N, seed=30 + seed)
        costs = _costs(N, 40 + seed)
        _, true = ecb.fixed_overhead(costs, 1e6)
        for k in (1, 3, 6):
            # half a benchmark of slack: k always fit, k + 1 never do
            budget = (k + 0.5) / N * true.sum()
            sel = ecb.budgeted_greedy(R, true, budget, "trace")
            assert sel == greedy_trace_select(R, k)["selected"]
            mi = ecb.budgeted_greedy(R, true, budget, "mi")
            assert 1 <= len(mi) <= k
            assert mi == greedy_mi(R, len(mi))["selected"]


# ---------------------------------------------------------------------------
# The fixed run: ratio 0 = plain run, leakage, outputs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth():
    return _synthetic()


@pytest.fixture(scope="module")
def fixed_run(synth):
    B, O, names, costs = synth
    fracs = [0.02, 0.1, 0.3]
    out = ecb.run_fixed_arrays(B, O, names, costs, fracs, [0, 0.5, 2],
                               n_repeats=2, n_folds=4, verbose=False)
    paired = ecb.fixed_paired_rows("synth", out)
    return out, paired, fracs


def test_ratio_zero_matches_plain_run(synth, fixed_run, tmp_path,
                                      monkeypatch):
    B, O, names, costs = synth
    out, _, fracs = fixed_run
    np.savez(tmp_path / "synth.matrix.npz", B=B, O=O,
             benchmark_names=np.array(names),
             model_names=np.array([f"m{i}" for i in range(B.shape[0])]))
    (tmp_path / "synth.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [None, None]}}))
    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps(dict(zip(names, costs.tolist()))))
    monkeypatch.setattr(ecb, "DATA", tmp_path)
    plain = ecb.run("synth", counts, fracs, n_repeats=2, n_folds=4)
    assert plain["eligible"].shape == out["eligible"].shape
    assert (~plain["eligible"]).any()                 # eligibility matters
    for key in ("eligible", "n_eligible", "rep", "fold", "split_id",
                "n_train", "n_val", "estimation_failed"):
        assert np.array_equal(plain[key], out[key]), key
    assert np.array_equal(plain["budgets"], out["budgets"][0])
    for m in ecb.METHODS:
        for key in ("r2", "rmse", "cost", "index_ties", "tie_breaks"):
            a = np.asarray(plain[f"{m}_{key}"]).T        # (splits, budgets)
            b = out[f"{m}_{key}"][:, 0]
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), (
                m, key)
        assert np.array_equal(plain[f"{m}_size"].T,
                              out[f"{m}_size"][:, 0].astype(float))
        assert np.array_equal(plain[f"{m}_selected"].transpose(1, 0, 2),
                              out[f"{m}_selected"][:, 0])
    for m in ecb.KNAP_OBJECTIVE:
        assert np.array_equal(plain[f"{m}_singleton"].T,
                              out[f"{m}_singleton"][:, 0])
    # the loader wrapper gives the same arrays
    again = ecb.run_fixed("synth", counts, fracs, [0, 0.5, 2], 2, 4)
    for key, val in out.items():
        val = np.asarray(val)
        assert np.array_equal(val, again[key],
                              equal_nan=val.dtype.kind == "f"), key


def test_layout_and_checks(fixed_run, synth):
    out, _, fracs = fixed_run
    costs = synth[3]
    S, nf, nb, N = 8, 3, len(fracs), len(costs)
    assert np.array_equal(out["fixed_ratios"], [0.0, 0.5, 2.0])
    np.testing.assert_allclose(out["fixed_costs"],
                               [0, 0.5 * np.median(costs),
                                2 * np.median(costs)])
    np.testing.assert_allclose(out["total_cost"],
                               costs.sum() + N * out["fixed_costs"])
    for m in ecb.METHODS:
        assert out[f"{m}_r2"].shape == (S, nf, nb)
        assert out[f"{m}_selected"].shape == (S, nf, nb, N)
        assert not out[f"{m}_violations"].any()
        assert np.all(out[f"{m}_cost"] <= out["budgets"][None])
        size = out[f"{m}_size"]
        assert np.array_equal(np.isnan(out[f"{m}_r2"]), size == 0)
        suite = out[f"{m}_r2_suite"]
        assert np.isfinite(suite).all() and np.all(suite[size == 0] == 0)
        # every pick eligible on its split
        sel = out[f"{m}_selected"]
        for i in range(S):
            picks = sel[i][sel[i] >= 0]
            assert out["eligible"][i, picks].all()
    # cheapest first buys the most benchmarks at every level and budget
    for m in ecb.METHODS:
        assert np.all(out[f"{m}_size"] <= out["cheap_first_size"]), m


def test_no_leakage_of_validation_scores(synth):
    B, O, names, costs = synth
    split = _split()
    levels, budgets = ecb.fixed_levels(costs, [0, 0.5, 2], [0.02, 0.1, 0.3])
    base = ecb.evaluate_fixed_split(B, O, split, levels, budgets)
    rng = np.random.default_rng(3)
    B2 = B.copy()
    B2[split.val_idx] = O[split.val_idx] * rng.uniform(
        size=(len(split.val_idx), B.shape[1]))
    alt = ecb.evaluate_fixed_split(B2, O, split, levels, budgets)
    assert not base["failed"] and not alt["failed"]
    for m in ecb.METHODS:
        for key in ("selected", "size", "cost", "violations", "index_ties",
                    "tie_breaks"):
            assert np.array_equal(base[f"{m}_{key}"], alt[f"{m}_{key}"]), (
                m, key)
    assert not np.allclose(base["knap_trace_r2_suite"],
                           alt["knap_trace_r2_suite"])
    # the selections also equal those on the training data alone
    el = eligible_mask(O[split.train_idx])
    R, _, _ = estimate_corr(B[split.train_idx], O[split.train_idx])
    for f, (_, true) in enumerate(levels):
        for b in range(budgets.shape[1]):
            ref = ecb.budgeted_greedy(R, true, budgets[f, b], "trace",
                                      eligible=el)
            got = base["knap_trace_selected"][f, b]
            assert got[got >= 0].tolist() == ref


def test_paired_rows_and_orientation(fixed_run):
    out, paired, fracs = fixed_run
    pairs = ecb.orient(ecb.METHODS)
    assert len(paired) == 3 * len(fracs) * len(ecb.FIXED_METRICS) * len(
        pairs)
    for r in paired:
        assert method_rank(r["method_a"]) <= method_rank(r["method_b"])
    for p in ecb.FIXED_PAIRS:
        assert p in pairs
    r = next(p for p in paired if p["fixed_ratio"] == 0.5
             and p["budget_frac"] == 0.1 and p["metric"] == "r2_suite"
             and p["method_a"] == "knap_trace" and p["method_b"] == "random")
    d = out["knap_trace_r2_suite"][:, 1, 1] - out["random_r2_suite"][:, 1, 1]
    assert r["mean_diff"] == pytest.approx(d.mean())
    assert r["n_splits"] == 8 and r["n_reps"] == 2
    assert r["budget"] == pytest.approx(out["budgets"][1, 1])


def test_macros_json_and_plot(fixed_run, tmp_path, monkeypatch):
    out, paired, fracs = fixed_run
    undefined = []
    mac = ecb.fixed_macros("synth", out, paired, undefined)
    assert mac and all(k.isascii() and k.isalpha() for k in mac)
    assert all(k.isascii() and k.isalpha() for k in undefined)
    assert all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in mac.values())
    assert not set(mac) & set(undefined)
    assert mac["CostFixedViolationsSynth"] == 0
    assert mac["CostFixedNSplitsSynth"] == 8
    assert mac["CostFixedOverheadSynthFzeropointfive"] == pytest.approx(
        out["fixed_costs"][1])
    assert mac["CostFixedSuiteRsqSynthKnapTraceFtwoBten"] == pytest.approx(
        out["knap_trace_r2_suite"][:, 2, 1].mean())
    assert mac["CostFixedSizeSynthCheapFirstFzeroBthirty"] == pytest.approx(
        out["cheap_first_size"][:, 0, 2].mean())
    d = out["knap_mi_r2_suite"][:, 1, 0] - out["random_r2_suite"][:, 1, 0]
    assert mac["CostFixedSuitePairDiffSynthKnapMiVsRandomFzeropointfiveBtwo"] \
        == pytest.approx(d.mean())
    assert "CostFixedPairNSynthKnapTraceVsCheapFirstFtwoBtwo" in mac
    assert "CostFixedSigPosSynthKnapEntropyVsCardEntropyFzero" in mac
    config = {"dataset": "synth", "runtime_s": 0.0}
    rep = ecb.fixed_json("synth", out, paired, mac, config, undefined)
    back = json.loads(json.dumps(rep, allow_nan=False))
    assert back["checks"]["violations_total"] == 0
    assert back["macros"] == mac
    assert set(back["summary"]) == {"0", "0.5", "2"}
    assert len(back["paired_key_pairs"]) == (3 * len(fracs) * 2
                                             * len(ecb.FIXED_PAIRS))
    assert "human_evaluation" in back["notes"]
    monkeypatch.setattr(ecb, "OUT", tmp_path)
    path = ecb.plot_fixed("synth", out, paired)
    assert path.is_file() and path.name == "cost_fixed_synth.pdf"


def test_rerun_identical(synth, fixed_run):
    B, O, names, costs = synth
    out, _, fracs = fixed_run
    again = ecb.run_fixed_arrays(B, O, names, costs, fracs, [0, 0.5, 2],
                                 n_repeats=2, n_folds=4, verbose=False)
    assert set(again) == set(out)
    for key, val in out.items():
        val = np.asarray(val)
        assert np.array_equal(val, again[key],
                              equal_nan=val.dtype.kind == "f"), key


def test_suite_r2_and_sequential_cost_shared():
    """eval_token_costs uses eval_cost_budget's helpers (one definition)."""
    assert etc.suite_r2 is ecb.suite_r2
    assert etc.sequential_cost is ecb.sequential_cost


# ---------------------------------------------------------------------------
# Facility location (knap_facloc / card_facloc)
# ---------------------------------------------------------------------------

class TestFacilityLocation:
    @pytest.mark.parametrize("seed", range(4))
    def test_unit_costs_reproduce_facility_location_select(self, seed):
        """Equal costs and budget k x cost: the knapsack is the cardinality
        greedy, i.e. facility_location_select's selection (any cost unit,
        with and without an eligibility mask, on a correlation and on a
        covariance matrix, whose similarity is its correlation)."""
        N = 14
        R = _corr(N, seed=50 + seed)
        rng = np.random.default_rng(60 + seed)
        D = np.diag(rng.uniform(0.5, 2.0, size=N))
        el = rng.uniform(size=N) > 0.25
        n_checked = 0
        for Sigma in (R, D @ R @ D):
            for mask in (None, el):
                n_el = N if mask is None else int(mask.sum())
                for unit in (1.0, 100.0):
                    costs = np.full(N, unit)
                    for k in sorted({1, 2, 3, 5, 8, n_el}):
                        ref = facility_location_select(Sigma, k,
                                                       eligible=mask)
                        res = ecb.budgeted_select(Sigma, costs, k * unit,
                                                  "facloc", eligible=mask)
                        assert res["selected"] == ref["selected"], (
                            seed, k, unit, mask is None)
                        assert not res["singleton"]
                        assert res["value"] == pytest.approx(
                            _facloc_value(Sigma, res["selected"]),
                            rel=1e-12)
                        assert res["value"] == pytest.approx(
                            ref["facloc_values"][len(res["selected"]) - 1],
                            rel=1e-12)
                        n_checked += 1
        assert n_checked >= 4 * 5

    def test_similarity_is_facility_location_selects(self):
        """_facloc_similarity: unit-diagonal rescaling, clipped at 0, s_jj
        = 1 (facility_location_select's step-1 gains are its column
        sums)."""
        rng = np.random.default_rng(3)
        R = _corr(9, seed=3)
        D = np.diag(rng.uniform(0.5, 2.0, size=9))
        S = ecb._facloc_similarity(D @ R @ D)
        R1 = R.copy()
        np.fill_diagonal(R1, 1.0)
        np.testing.assert_allclose(S, np.maximum(R1, 0.0), rtol=1e-12,
                                   atol=1e-15)
        assert np.all(np.diag(S) == 1.0) and (R < 0).any()
        gains = ecb._facloc_gains(S, np.zeros(9), np.arange(9))
        ref = facility_location_select(D @ R @ D, 1)
        assert int(np.argmax(gains)) == ref["selected"][0]
        assert gains.max() == pytest.approx(ref["gains"][0], rel=1e-12)

    def test_card_and_knap_facloc_on_a_split(self):
        """card_facloc is facility_location_select's full order filled
        skip-and-continue; knap_facloc is budgeted_select("facloc")."""
        split = _split()
        N = 12
        R = _corr(N, seed=11)
        costs = _costs(N, 12)
        el = np.ones(N, dtype=bool)
        el[[3, 8]] = False
        orders = ecb.split_orders(R, costs, split, el)
        ref = facility_location_select(R, N, eligible=el)
        assert orders["card_facloc"]["selected"] == ref["selected"]
        for frac in (0.02, 0.1, 0.3, 0.8):
            budget = frac * costs.sum()
            card = ecb.select_for_budget("card_facloc", R, costs, budget,
                                         orders, split, el)
            assert card["selected"] == _fill_oracle(ref["selected"], costs,
                                                    budget)
            knap = ecb.select_for_budget("knap_facloc", R, costs, budget,
                                         orders, split, el)
            res = ecb.budgeted_select(R, costs, budget, "facloc",
                                      eligible=el)
            assert knap["selected"] == res["selected"]
            assert knap["singleton"] == res["singleton"]
            assert all(el[j] for j in knap["selected"] + card["selected"])

    def test_singleton_fallback(self):
        """A cheap independent benchmark has the best gain / cost but leaves
        no room for the expensive cluster, whose hub alone covers far more
        (F({hub}) = 1 + 6 x 0.9 against 1)."""
        R = np.full((7, 7), 0.8)
        np.fill_diagonal(R, 1.0)
        R = np.block([[R, np.zeros((7, 1))], [np.zeros((1, 7)), np.eye(1)]])
        R[0, 1:7] = R[1:7, 0] = 0.9           # benchmark 0 is the hub
        assert np.linalg.eigvalsh(R).min() > 0
        costs = np.array([10.0] * 7 + [1.0])
        res = ecb.budgeted_select(R, costs, 10.0, "facloc")
        assert res["singleton"] and res["selected"] == [0]
        assert res["value"] == pytest.approx(1 + 6 * 0.9)
        greedy = etc.proxy_budgeted_select(R, costs, costs, 10.0, "facloc",
                                           singleton=False)
        assert greedy["selected"] == [7] and greedy["value"] == 1.0
        # with room for both, the greedy set wins
        res = ecb.budgeted_select(R, costs, 11.0, "facloc")
        assert not res["singleton"] and sorted(res["selected"]) == [0, 7]

    @pytest.mark.parametrize("seed", range(5))
    def test_knapsack_guarantee(self, seed):
        """F >= 0 is monotone submodular, so the better of the ratio greedy
        and the best affordable singleton is within (1 − 1/e)/2 of the
        best affordable set (Krause & Guestrin 2005), here against brute
        force over all subsets of eligible benchmarks."""
        N = 10
        rng = np.random.default_rng(70 + seed)
        R = _corr(N, rank=2 + seed % 3, seed=80 + seed)
        costs = rng.integers(1, 12, size=N).astype(float)
        el = rng.uniform(size=N) > 0.2
        subsets = [[j for j in range(N) if (m >> j) & 1]
                   for m in range(1, 2 ** N)]
        ratio = (1 - 1 / math.e) / 2
        for frac in (0.05, 0.15, 0.3, 0.6):
            budget = frac * costs.sum()
            for mask in (None, el):
                ok = np.ones(N, bool) if mask is None else mask
                opt = max([_facloc_value(R, A) for A in subsets
                           if ok[A].all() and costs[A].sum() <= budget],
                          default=0.0)
                res = ecb.budgeted_select(R, costs, budget, "facloc",
                                          eligible=mask)
                got = _facloc_value(R, res["selected"])
                assert costs[res["selected"]].sum() <= budget
                assert got >= ratio * opt - 1e-9, (frac, got, opt)
                assert res["value"] == pytest.approx(got, rel=1e-12)

    def test_facloc_appended_leaves_other_methods_bit_identical(
            self, synth, fixed_run, monkeypatch):
        """Without the facility-location pair (METHODS of the log before
        it), every array of the other methods is the same, bit for bit:
        facloc draws no random numbers and is appended after them."""
        B, O, names, costs = synth
        out, _, fracs = fixed_run
        monkeypatch.setattr(ecb, "METHODS", BASE_METHODS)
        monkeypatch.setattr(ecb, "KNAP_OBJECTIVE", {
            m: o for m, o in ecb.KNAP_OBJECTIVE.items()
            if m in BASE_METHODS})
        base = ecb.run_fixed_arrays(B, O, names, costs, fracs, [0, 0.5, 2],
                                    n_repeats=2, n_folds=4, verbose=False)
        new = set(out) - set(base)
        assert new and all(k.startswith(("knap_facloc_", "card_facloc_"))
                           for k in new)
        for key, val in base.items():
            if key == "methods":
                assert list(out[key]) == BASE_METHODS + ["knap_facloc",
                                                         "card_facloc"]
                continue
            val = np.asarray(val)
            assert np.array_equal(val, out[key],
                                  equal_nan=val.dtype.kind == "f"), key
