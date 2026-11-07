"""Tests for eval_selection_stability.py (synthetic data, < 60 s).

Covers the set measures (Jaccard, common members, mean pairwise Jaccard),
the population R^2 against the trace objective and greedy_trace_select's
residual variance, the training subsamples (training positions only,
sorted, deterministic, prefix-stable, split-specific), the random subsets
(eligible, nested, random_select's rule) and the random s-swaps of the
matched null (exact overlap, eligible, nested in s, uniform over the sets
with that overlap), the seed tags against every *_SEED_OFFSET in code/,
the Fieller interval of ratio_ci (paired_stats' interval for a constant
denominator, scale equivariance, unbounded intervals), the
estimator-branch diagnostic against estimate_corr, and an end-to-end run
on a small synthetic dataset: the full selections and R^2 equal
eval_baselines.py's on the same splits (check_logs), the random swaps and
cell shares recomputed by hand, no leakage of the validation rows into
anything but the realized R^2, alternatives that depend only on their own
subsample, harm and swap_loss recomputed by hand with the thresholds and
the gap rule, sub_frac = 1 reproducing the full selection, determinism and
independence of --workers, letters-only macros, the output files,
--from-logs, --all-figure and the refusal of first-version logs; and
facloc (a method since 2026-10-19): its full and alternative selections
are facility_location_select on the split / subsample estimates, and
adding it leaves every array, paired row and macro of the other methods
unchanged.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pytest

import eval_selection_stability as ess
from baseline_select import facility_location_select, greedy_trace_select
from cv_split import cv_splits, eligible_mask
from eval_baselines import evaluated_share
from eval_entropy_vs_mi import (estimate_corr, estimate_corr_em,
                                estimate_corr_pairwise, impute_and_evaluate)
from greedy_select import random_select
from paired_stats import paired_stats


def _factor_corr(N=12, n_factors=3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, n_factors))
    S = W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))
    s = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------------------
# Set measures and the population R^2
# ---------------------------------------------------------------------------

def test_jaccard_and_common():
    assert ess.jaccard([1, 2, 3], [3, 2, 1]) == 1.0
    assert ess.jaccard([1, 2, 3], [3, 4, 5]) == pytest.approx(1 / 5)
    assert ess.jaccard([0, 1], [2, 3]) == 0.0
    assert np.isnan(ess.jaccard(None, [1]))
    assert ess.n_common([1, 2, 3], [3, 2, 9]) == 2
    assert ess.n_common(None, [1]) == -1
    sets = [[0, 1], [0, 2], None, [0, 1]]
    # pairs among the three valid sets: 1/3, 1, 1/3
    assert ess.mean_pairwise_jaccard(sets) == pytest.approx(5 / 9)
    assert np.isnan(ess.mean_pairwise_jaccard([[0], None]))


def test_pop_r2_is_trace_objective():
    R = _factor_corr(14, seed=1)
    N = R.shape[0]
    res = greedy_trace_select(R, 6)
    for k in range(1, 7):
        A = res["selected"][:k]
        C = [j for j in range(N) if j not in A]
        F_tr = np.trace(R[:, A] @ np.linalg.solve(R[np.ix_(A, A)], R[A, :]))
        assert ess.pop_r2(R, A) == pytest.approx((F_tr - k) / (N - k),
                                                 rel=1e-6)
        # residual_variance = residual trace / N on a correlation matrix
        rv = res["residual_variance"][k - 1]
        assert ess.pop_r2(R, A) == pytest.approx(1 - rv * N / (N - k),
                                                 rel=1e-6)
        assert ess.pop_r2(R, A) == pytest.approx(
            np.trace(R[np.ix_(C, A)] @ np.linalg.solve(
                R[np.ix_(A, A)], R[np.ix_(A, C)])) / len(C), rel=1e-6)
    assert np.isnan(ess.pop_r2(R, None))
    assert np.isnan(ess.pop_r2(R, list(range(N))))


# ---------------------------------------------------------------------------
# Common random numbers
# ---------------------------------------------------------------------------

def _splits(M=50, n_folds=5, n_repeats=2):
    return list(cv_splits(M, n_folds, n_repeats, ess.HOLDOUT_FRAC, ess.SEED))


def test_subsample_positions():
    splits = _splits()
    s = splits[3]
    n_train = len(s.train_idx)
    pos = ess.subsample_positions(s, n_train, 10, 0.8)
    assert pos.shape == (10, round(0.8 * n_train))
    assert np.all(np.diff(pos, axis=1) > 0)             # sorted, unique
    assert pos.min() >= 0 and pos.max() < n_train       # training rows only
    assert len({tuple(r) for r in pos}) == 10           # distinct draws
    np.testing.assert_array_equal(pos, ess.subsample_positions(
        s, n_train, 10, 0.8))
    # the first draws do not depend on the number of subsamples
    np.testing.assert_array_equal(pos[:3], ess.subsample_positions(
        s, n_train, 3, 0.8))
    # split-specific (another split's draws differ)
    other = ess.subsample_positions(splits[4], n_train, 10, 0.8)
    assert not np.array_equal(pos, other)
    # sub_frac = 1: every training row
    full = ess.subsample_positions(s, n_train, 2, 1.0)
    np.testing.assert_array_equal(full, np.tile(np.arange(n_train), (2, 1)))
    # the stream is SEED + SUB_SEED_OFFSET
    rng = np.random.default_rng(s.seed(ess.SEED + ess.SUB_SEED_OFFSET))
    np.testing.assert_array_equal(
        pos[0], np.sort(rng.permutation(n_train)[:pos.shape[1]]))


def test_random_subsets_follow_random_select():
    s = _splits()[1]
    N = 12
    R = _factor_corr(N, seed=2)
    el = np.ones(N, dtype=bool)
    el[[0, 5, 7]] = False
    sub = ess.random_subsets(s, el, 6, 4)
    assert sub.shape == (4, 6)
    assert np.all(el[sub])                              # eligible only
    assert all(len(set(r)) == 6 for r in sub)
    # draw 0 is random_select's order with the SEED + 24 stream
    ref = random_select(R, 6, seed=s.seed(ess.SEED
                                          + ess.RANDOM_SUBSETS_SEED_OFFSET),
                        eligible=el)["selected"]
    assert sub[0].tolist() == ref
    np.testing.assert_array_equal(sub, ess.random_subsets(s, el, 6, 4))
    # fewer eligible benchmarks than k_max: -1 padding
    few = np.zeros(N, dtype=bool)
    few[[2, 3]] = True
    short = ess.random_subsets(s, few, 4, 3)
    assert np.all(short[:, 2:] == -1)
    assert all(set(r[:2]) == {2, 3} for r in short)


def test_random_swap():
    s0, s1 = _splits()[:2]
    N = 12
    el = np.ones(N, dtype=bool)
    el[[9, 10]] = False
    A = [4, 1, 7, 0, 3]
    orders = ess.swap_orders(s0, N, 6)
    assert orders.shape == (6, N)
    assert all(sorted(r) == list(range(N)) for r in orders)
    np.testing.assert_array_equal(orders, ess.swap_orders(s0, N, 6))
    np.testing.assert_array_equal(orders[:2], ess.swap_orders(s0, N, 2))
    assert not np.array_equal(orders, ess.swap_orders(s1, N, 6))
    rng = np.random.default_rng(s0.seed(ess.SEED + ess.SWAP_SEED_OFFSET))
    np.testing.assert_array_equal(orders[0], rng.permutation(N))
    for order in orders:
        prev = set(A)
        assert ess.random_swap(A, el, order, 0) == A
        for s in range(1, len(A) + 1):
            sw = ess.random_swap(A, el, order, s)
            assert len(sw) == len(set(sw)) == len(A)
            assert len(set(sw) & set(A)) == len(A) - s     # exact overlap
            assert np.all(el[sw])                          # eligible only
            # nested: the kept members shrink, the additions grow
            kept, added = set(sw) & set(A), set(sw) - set(A)
            assert kept <= prev & set(A)
            prev = set(sw)
            if s > 1:
                assert added >= set(last) - set(A)
            last = sw
    assert ess.random_swap(None, el, orders[0], 1) is None
    assert ess.random_swap(A, el, orders[0], 6) is None
    few = np.zeros(N, dtype=bool)
    few[A + [2]] = True                 # one eligible non-member
    assert ess.random_swap(A, few, orders[0], 2) is None
    assert set(ess.random_swap(A, few, orders[0], 1)) - set(A) == {2}

    # Uniform over the sets with exactly k - s members of A: k = 3, s = 1,
    # 3 eligible non-members -> 9 sets of probability 1/9 each
    N, A, n = 7, [0, 1, 2], 9000
    el = np.ones(N, dtype=bool)
    el[6] = False
    rng = np.random.default_rng(0)
    counts = Counter(frozenset(ess.random_swap(A, el, rng.permutation(N), 1))
                     for _ in range(n))
    expect = {frozenset(set(A) - {d} | {a}) for d in A for a in (3, 4, 5)}
    assert set(counts) == expect
    assert max(abs(c - n / 9) for c in counts.values()) < 150   # 5 sd


def test_seed_tags_unique_in_code():
    """The three streams (22 s-swaps, 23 subsamples, 24 random subsets)
    are defined only here and collide with no other seed tag of code/:
    every *_SEED_OFFSET = n assignment and every 'SEED + n' seed of
    another (non-test) module (eval_reviewer_checks owns 25, eval_imputers
    21); this module itself names no SEED-plus-number seed, which
    test_eval_reviewer_checks scans for."""
    assert ess.SWAP_SEED_OFFSET == 22
    assert ess.SUB_SEED_OFFSET == 23
    assert ess.RANDOM_SUBSETS_SEED_OFFSET == 24
    ours = (ess.SWAP_SEED_OFFSET, ess.SUB_SEED_OFFSET,
            ess.RANDOM_SUBSETS_SEED_OFFSET)
    assert len(set(ours)) == 3
    code = pathlib.Path(ess.__file__).resolve().parent
    pats = (re.compile(r"^\s*(\w*SEED_OFFSET)\s*=\s*(\d+)\s*$", re.M),
            re.compile(r"\b(SEED)\s*\+\s*(\d+)\b"))
    others = {}
    for path in code.glob("*.py"):
        if path.name == "eval_selection_stability.py" or \
                path.name.startswith("test_"):
            continue
        text = path.read_text()
        for pat in pats:
            for name, val in pat.findall(text):
                others.setdefault(int(val), []).append(
                    f"{path.name}:{name}")
    for tag in ours:
        assert tag not in others, others.get(tag)
    assert 7 in others and 11 in others          # the scan sees cv_split
    assert 21 in others and 25 in others         # and the newer streams
    assert 20 not in ours                        # TabImpute's reserved tag
    own = pathlib.Path(ess.__file__).read_text()
    assert not pats[1].search(own), pats[1].search(own)


def test_ratio_ci():
    rng = np.random.default_rng(5)
    n = 30
    n_train = np.full(n, 90)
    n_val = np.full(n, 10)
    num = rng.normal(0.3, 0.2, n)
    # constant denominator: paired_stats' interval of the mean
    est, lo, hi = ess.ratio_ci(num, np.ones(n), n_train, n_val)
    ref = paired_stats(num, np.zeros(n), n_train, n_val)
    assert est == pytest.approx(ref["mean_diff"], rel=1e-12)
    assert lo == pytest.approx(ref["ci_low"], rel=1e-10)
    assert hi == pytest.approx(ref["ci_high"], rel=1e-10)
    # ratio of sums, inside its interval, equivariant under scaling
    den = rng.uniform(0.5, 1.5, n)
    num2 = 0.4 * den + rng.normal(0, 0.05, n)
    est, lo, hi = ess.ratio_ci(num2, den, n_train, n_val)
    assert est == pytest.approx(num2.sum() / den.sum(), rel=1e-12)
    assert lo < est < hi and lo < 0.4 < hi
    e2, l2, h2 = ess.ratio_ci(3 * num2, 2 * den, n_train, n_val)
    assert (e2, l2, h2) == pytest.approx((1.5 * est, 1.5 * lo, 1.5 * hi),
                                         rel=1e-10)
    # the bounds solve the corrected t test of num - H den at its quantile
    from scipy import stats
    t = stats.t.ppf(0.975, n - 1)
    for H in (lo, hi):
        d = num2 - H * den
        se = np.sqrt(np.var(d, ddof=1) * (1 / n + 10 / 90))
        assert abs(np.mean(d)) / se == pytest.approx(t, rel=1e-8)
    # NaN rows are left out
    num3, den3 = num2.copy(), den.copy()
    num3[0], den3[1] = np.nan, np.nan
    assert ess.ratio_ci(num3, den3, n_train, n_val)[0] == pytest.approx(
        num2[2:].sum() / den[2:].sum())
    # a denominator not significantly nonzero: unbounded, NaN bounds
    est, lo, hi = ess.ratio_ci(rng.normal(0.1, 1, n), rng.normal(0.02, 1, n),
                               n_train, n_val)
    assert np.isfinite(est) and np.isnan(lo) and np.isnan(hi)
    assert np.isnan(ess.ratio_ci([1.0], [2.0], [9], [1])[1])
    assert np.isnan(ess.ratio_ci([], [], [], [])[0])


def test_estimator_regime_matches_estimate_corr():
    rng = np.random.default_rng(4)
    R = _factor_corr(6, n_factors=2, seed=3)
    cases = [(40, 1.0, 0), (40, 0.75, 1), (40, 0.4, 2), (5, 0.95, 3)]
    for M, frac, regime in cases:
        X = rng.multivariate_normal(np.zeros(6), R, size=M)
        O = (rng.uniform(size=(M, 6)) < frac).astype(float)
        if frac == 1.0:
            O[:] = 1.0
        O[:3] = 1.0                                   # every column seen
        B = X * O
        assert ess.estimator_regime(O) == regime, (M, frac)
        ref = (estimate_corr_pairwise if regime == 0 else estimate_corr_em)
        np.testing.assert_array_equal(estimate_corr(B, O)[0], ref(B, O)[0])


# ---------------------------------------------------------------------------
# End to end on a small synthetic dataset
# ---------------------------------------------------------------------------

M_SYN, N_SYN = 64, 10


def _write_dataset(path, seed=0, perturb_rows=None, perturb_seed=99):
    rng = np.random.default_rng(seed)
    R = _factor_corr(N_SYN, n_factors=2, seed=seed)
    X = rng.multivariate_normal(np.zeros(N_SYN), R, size=M_SYN) * 0.1 + 0.5
    O = (rng.uniform(size=(M_SYN, N_SYN)) < 0.85).astype(np.float64)
    O[:, 0] = 1.0
    if perturb_rows is not None:
        prng = np.random.default_rng(perturb_seed)
        X[perturb_rows] = prng.uniform(0, 1, size=(len(perturb_rows), N_SYN))
    B = X * O
    np.savez(path / "synth.matrix.npz", B=B, O=O,
             model_names=np.array([f"m{i}" for i in range(M_SYN)]),
             benchmark_names=np.array([f"b{j}" for j in range(N_SYN)]))
    (path / "synth.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [None, None]}}))
    return B, O


@pytest.fixture
def synth_dirs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    logs = tmp_path / "logs"
    figs = tmp_path / "figures"
    for p in (data, logs, figs):
        p.mkdir()
    monkeypatch.setattr(ess, "DATA", data)
    monkeypatch.setattr(ess, "LOGS", logs)
    monkeypatch.setattr(ess, "OUT", figs)
    return data, logs, figs


KW = dict(ks=[2, 3], n_folds=3, n_repeats=1, n_sub=3, n_random=4, n_swap=3)
SIDE = ["selected", "alt_selected", "pop_full", "pop_alt", "jacc_full",
        "jacc_pair", "n_common", "rand_jacc_full", "rand_n_common",
        "swap_pop", "cells_full", "cells_alt"]


def _first_split():
    return next(cv_splits(M_SYN, KW["n_folds"], KW["n_repeats"],
                          ess.HOLDOUT_FRAC, ess.SEED))


def test_end_to_end(synth_dirs, monkeypatch):
    data, logs, figs = synth_dirs
    B, O = _write_dataset(data)
    arr = ess.run_dataset("synth", **KW)
    S, K = 3, 2
    assert arr["split_id"].tolist() == [0, 1, 2]
    for key in ("rep", "fold", "n_eligible", "n_train", "n_val"):
        assert arr[key].shape == (S,)
    assert arr["swap_order"].shape == (S, 3, N_SYN)
    assert arr["rand_cells"].shape == (S, K, 4)
    for m in ess.METHODS:
        assert arr[f"{m}_r2_alt"].shape == (S, K, 3)
        assert arr[f"{m}_alt_selected"].shape == (S, K, 3, 3)
        assert arr[f"{m}_rand_jacc_full"].shape == (S, K, 4)
        assert arr[f"{m}_swap_r2"].shape == (S, K, 3, 3)
        assert arr[f"{m}_cells_alt"].shape == (S, K, 3)
        # s = 3 > k = 2: no swap
        assert np.all(np.isnan(arr[f"{m}_swap_r2"][:, 0, :, 2]))
        assert np.all(np.isfinite(arr[f"{m}_swap_pop"][:, 1]))
    assert not arr["estimation_failed"].any()

    # Split 0 by hand: full selection and its R^2 with the full estimate
    s0 = _first_split()
    R, mu, sd = estimate_corr(B[s0.train_idx], O[s0.train_idx])
    el = eligible_mask(O[s0.train_idx])
    Bv, Ov = B[s0.val_idx], O[s0.val_idx]
    for m in ess.METHODS:
        for i, k in enumerate(KW["ks"]):
            A = ess.method_sets(m, R, el, KW["ks"], None, s0)[i]
            assert arr[f"{m}_selected"][0, i, :k].tolist() == list(A)
            r2 = impute_and_evaluate(Bv, Ov, A, R, mu, sd, N_SYN)["r2"]
            assert arr[f"{m}_r2_full"][0, i] == r2
            assert arr[f"{m}_pop_full"][0, i] == ess.pop_r2(R, A)
            assert arr[f"{m}_cells_full"][0, i] == evaluated_share(Ov, A)
            # the random s-swaps of A with the split's permutations
            orders = ess.swap_orders(s0, N_SYN, 3)
            np.testing.assert_array_equal(arr["swap_order"][0], orders)
            for d in range(3):
                for s in range(1, k + 1):
                    sw = ess.random_swap(A, el, orders[d], s)
                    assert arr[f"{m}_swap_r2"][0, i, d, s - 1] == \
                        impute_and_evaluate(Bv, Ov, sw, R, mu, sd,
                                            N_SYN)["r2"]
                    assert arr[f"{m}_swap_pop"][0, i, d, s - 1] == \
                        ess.pop_r2(R, sw)
    # Alternative 0 of split 0: the same method on its subsample only
    pos = ess.subsample_positions(s0, len(s0.train_idx), 3, ess.SUB_FRAC)
    tr = s0.train_idx[pos[0]]
    R0 = estimate_corr(B[tr], O[tr])[0]
    A0 = ess.method_sets("trace", R0, eligible_mask(O[tr]), KW["ks"], None,
                         s0)[1]
    assert arr["trace_alt_selected"][0, 1, 0].tolist() == list(A0)
    assert arr["trace_r2_alt"][0, 1, 0] == impute_and_evaluate(
        Bv, Ov, A0, R, mu, sd, N_SYN)["r2"]
    assert arr["trace_cells_alt"][0, 1, 0] == evaluated_share(Ov, A0)

    # Consistency with eval_baselines.py on the same splits (check_logs)
    import eval_baselines as eb
    monkeypatch.setattr(eb, "DATA", data)
    metrics, out, names, k_max, _ = eb.run_dataset("synth", 3, 3, 1)
    base = {"benchmark_names": names, "k_max": k_max, **out}
    for m in eb.METHODS:
        for key, val in metrics[m].items():
            base[f"{m}_{key}"] = val
    np.savez(logs / "baselines_synth.npz", **base)
    check = ess.check_logs("synth", arr, logs)
    for m in ess.METHODS + ["random"]:
        assert check[m]["same_selected"], m
        assert check[m]["max_abs_r2_diff"] == 0.0, m

    # Summaries, macros and files; --from-logs reproduces them
    paired = ess.paired_rows("synth", arr)
    assert len(paired) == K * len(ess.METHODS) * len(ess.COMPARISONS)
    tables, macros = ess.summarize("synth", arr, paired)
    assert macros and all(re.fullmatch(r"[A-Za-z]+", n) for n in macros)
    for name in ("StabJaccPairSynthTraceKtwo", "StabHarmSynthMiKthree",
                 "StabHarmLoSynthMiKthree", "StabNSwappedSynthMiKthree",
                 "StabCrossJaccSynthPcaKtwo", "StabCellsShiftSynthHubKtwo",
                 "StabWorstCostOverGapSynthTraceKthree",
                 "StabPairDiffSynthPcaAltVsFullKtwo",
                 "StabRsqSynthRandomKthree", "StabNSplitsSynth",
                 "StabNSwapSynth"):
        assert name in macros, name
    assert macros["StabNSplitsSynth"] == 3
    assert macros["StabNSwapSynth"] == 3
    assert macros["StabSubModelsSynth"] == pytest.approx(
        np.mean([round(0.8 * n) for n in arr["n_train"]]))
    t = tables["3"]["mi"]
    assert 0.0 <= t["jacc_full"] <= 1.0 and 0.0 <= t["jacc_pair"] <= 1.0
    assert sum(v["n"] for v in t["by_swap"].values()) == t["n_alternatives"]
    # 3 splits x 3 subsamples: below MIN_N_ALT, every loss and harm is NaN
    for m in ess.METHODS:
        tm = tables["3"][m]
        assert tm["n_swapped"] < ess.MIN_N_ALT
        for key in ("swap_loss", "swap_loss_lo", "null_swap_loss", "harm",
                    "harm_hi", "harm_pop", "low_cost"):
            assert tm[key] is None, (m, key)
        assert tm["status"] in ("uninformative", "stable")
    # cross-split Jaccard by hand
    sets = [arr["trace_selected"][j, 1, :3].tolist() for j in range(S)]
    assert tables["3"]["trace"]["cross_jacc"] == pytest.approx(np.mean(
        [ess.jaccard(a, b) for a, b in combinations(sets, 2)]))
    ess.print_summary("synth", tables, paired)
    ess.plot_dataset("synth", arr, tables)
    ess.plot_all({"synth": arr, "synth2": arr})
    ess.save("synth", arr, tables, macros, paired, {"test": True}, check)
    doc = json.loads((logs / "selection_stability_synth.json").read_text())
    assert doc["macros"] == json.loads(json.dumps(macros))
    for name in ("selection_stability_synth.npz",
                 "selection_stability_synth_paired.csv"):
        assert (logs / name).is_file()
    assert (figs / "selection_stability_synth.pdf").is_file()
    assert (figs / "selection_stability_all.pdf").is_file()
    arr_l, cfg = ess.load_log("synth")
    assert cfg["from_logs"] and cfg["test"]
    _, macros_l = ess.summarize("synth", arr_l, ess.paired_rows("synth",
                                                                arr_l))
    assert json.dumps(macros_l, sort_keys=True) == json.dumps(
        macros, sort_keys=True)

    # --all-figure: the cross-dataset figure from the logs alone
    (figs / "selection_stability_all.pdf").unlink()
    ess.all_figure(["synth"])
    assert (figs / "selection_stability_all.pdf").is_file()
    # first-version logs (no s-swaps, no swap tag) and logs of other seed
    # tags are refused
    old = {key: val for key, val in arr.items()
           if key not in ("swap_order", "swap_seed_offset")}
    np.savez(logs / "selection_stability_old.npz", **old)
    with pytest.raises(ValueError, match="earlier version"):
        ess.load_log("old")
    other = dict(arr, sub_seed_offset=np.int64(25))
    np.savez(logs / "selection_stability_other.npz", **other)
    with pytest.raises(ValueError, match="earlier version"):
        ess.load_log("other")


def test_harm_and_thresholds(synth_dirs, monkeypatch):
    """harm, swap_loss, null_swap_loss and low_cost recomputed by hand
    from the per-split arrays; the gap rule and the sample-size
    thresholds."""
    data, _, _ = synth_dirs
    _write_dataset(data)
    arr = ess.run_dataset("synth", ks=[3], n_folds=4, n_repeats=1, n_sub=5,
                          sub_frac=0.5, n_random=3, n_swap=4)
    monkeypatch.setattr(ess, "MIN_N_ALT", 1)
    monkeypatch.setattr(ess, "MIN_N_SPLITS", 1)
    k, S = 3, 4
    for m in ("trace", "mi"):
        full = arr[f"{m}_r2_full"][:, 0]
        pfull = arr[f"{m}_pop_full"][:, 0]
        num = den = pnum = pden = swaps = 0.0
        n_sw, low_sum, n_low = 0, 0.0, 0
        for j in range(S):
            for b in range(5):
                s = k - arr[f"{m}_n_common"][j, 0, b]
                cost = full[j] - arr[f"{m}_r2_alt"][j, 0, b]
                if arr[f"{m}_jacc_full"][j, 0, b] < ess.LOW_JACCARD:
                    low_sum += cost
                    n_low += 1
                if s < 1:
                    continue
                n_sw += 1
                num += cost
                swaps += s
                den += full[j] - np.mean(arr[f"{m}_swap_r2"][j, 0, :, s - 1])
                pnum += pfull[j] - arr[f"{m}_pop_alt"][j, 0, b]
                pden += pfull[j] - np.mean(
                    arr[f"{m}_swap_pop"][j, 0, :, s - 1])
        assert n_sw > 0, m
        t = ess.method_table(arr, m, 0, True, -0.01)
        assert t["n_swapped"] == n_sw
        assert t["swap_loss"] == pytest.approx(num / swaps, rel=1e-10)
        assert t["null_swap_loss"] == pytest.approx(den / swaps, rel=1e-10)
        if t["rand_swap_loss"] > 0 and t["harm"] is not None:
            assert t["harm_subsets"] == pytest.approx(
                t["swap_loss"] / t["rand_swap_loss"], rel=1e-10)
        if t["harm"] is None:
            assert t["harm_subsets"] is None
        a = ess.alt_arrays(arr, m, 0)
        est = ess.ratio_ci(*ess._split_sums(a["cost"], a["null"],
                                            a["swapped"])[:2],
                           arr["n_train"], arr["n_val"])
        assert est[0] == pytest.approx(num / den, rel=1e-10)
        got = (t["harm"], t["harm_lo"], t["harm_hi"])
        if np.isfinite(est[1]):
            assert got == pytest.approx(est, rel=1e-12)
            assert est[1] < est[0] < est[2]
        else:                   # unbounded interval: no harm
            assert got == (None, None, None)
        if t["harm_pop"] is not None:
            assert t["harm_pop"] == pytest.approx(pnum / pden, rel=1e-10)
        if n_low:
            assert t["low_cost"] == pytest.approx(low_sum / n_low)
        if t["gap"] > 0:
            assert t["worst_cost_over_gap"] == pytest.approx(0.01 / t["gap"])
        else:
            assert t["worst_cost_over_gap"] is None
        assert t["status"] == "tested"
        # no significant gap: no ratio to the gap and no harm
        t0 = ess.method_table(arr, m, 0, False, -0.01)
        assert t0["status"] == "uninformative"
        for key in ("harm", "harm_lo", "cost_over_gap",
                    "worst_cost_over_gap", "harm_subsets"):
            assert t0[key] is None, key
        assert t0["swap_loss"] == pytest.approx(t["swap_loss"])
    # below the thresholds: "stable", losses and harm NaN
    monkeypatch.setattr(ess, "MIN_N_ALT", 10 ** 6)
    t = ess.method_table(arr, "trace", 0, True, -0.01)
    assert t["status"] == "stable"
    assert t["harm"] is None and t["swap_loss"] is None
    assert t["n_swapped"] > 0
    # the direct-test curves: bin means of cost and matched null
    xs, cost, null, n = ess.direct_curves(arr, "trace", 3)
    a = ess.alt_arrays(arr, "trace", 0)
    for x in xs:
        sel = a["valid"] & (a["s"] == x)
        assert n[x] == sel.sum()
        if sel.sum() >= ess.FIG_MIN_BIN:
            assert cost[x] == pytest.approx(np.mean(a["cost"][sel]))
            assert null[x] == pytest.approx(np.mean(a["null"][sel]))
        else:
            assert np.isnan(cost[x])


SPLIT_KW = dict(ks=KW["ks"], n_sub=KW["n_sub"], n_random=KW["n_random"],
                n_swap=KW["n_swap"])
NAMES = np.array([f"b{j}" for j in range(N_SYN)])


def test_no_leakage_and_subsample_locality(tmp_path):
    B, O = _write_dataset(tmp_path)
    s0 = _first_split()
    res = ess.evaluate_split(s0, B, O, NAMES, **SPLIT_KW)

    # Perturbed validation models of split 0: only its realized R^2 moves
    B2, _ = _write_dataset(tmp_path, perturb_rows=s0.val_idx)
    res2 = ess.evaluate_split(s0, B2, O, NAMES, **SPLIT_KW)
    for m in ess.METHODS:
        for q in SIDE:
            np.testing.assert_array_equal(res[f"{m}_{q}"], res2[f"{m}_{q}"],
                                          err_msg=f"{m}_{q}")
    for q in ("rand_order", "rand_pop", "rand_cells", "swap_order",
              "random_baseline_selected"):
        np.testing.assert_array_equal(res[q], res2[q], err_msg=q)
    assert not np.allclose(res["trace_r2_full"], res2["trace_r2_full"])
    assert not np.allclose(res["trace_swap_r2"][1], res2["trace_swap_r2"][1])

    # Perturbed training models outside subsample 0 of split 0: the
    # alternatives of subsample 0 are unchanged (they see only its rows),
    # while the full estimate (and with it pop_full) moves
    pos = ess.subsample_positions(s0, len(s0.train_idx), KW["n_sub"],
                                  ess.SUB_FRAC)
    outside = np.setdiff1d(s0.train_idx, s0.train_idx[pos[0]])
    B3, _ = _write_dataset(tmp_path, perturb_rows=outside)
    res3 = ess.evaluate_split(s0, B3, O, NAMES, **SPLIT_KW)
    for m in ess.METHODS:
        np.testing.assert_array_equal(res[f"{m}_alt_selected"][:, 0],
                                      res3[f"{m}_alt_selected"][:, 0],
                                      err_msg=m)
    assert not np.allclose(res["trace_pop_full"], res3["trace_pop_full"])


KW_FAST = dict(ks=[2, 3], n_folds=2, n_repeats=1, n_sub=2, n_random=3,
               n_swap=2)


def test_full_subsample_and_workers(synth_dirs):
    data, _, _ = synth_dirs
    _write_dataset(data)
    # sub_frac = 1: every alternative is the full selection
    arr1 = ess.run_dataset("synth", **dict(KW_FAST, sub_frac=1.0))
    for m in ess.METHODS:
        full = arr1[f"{m}_selected"][:, :, None, :]
        np.testing.assert_array_equal(
            arr1[f"{m}_alt_selected"],
            np.broadcast_to(full, arr1[f"{m}_alt_selected"].shape))
        np.testing.assert_array_equal(
            arr1[f"{m}_r2_alt"],
            np.broadcast_to(arr1[f"{m}_r2_full"][:, :, None],
                            arr1[f"{m}_r2_alt"].shape))
        assert np.all(arr1[f"{m}_jacc_full"] == 1.0)
    tables, _ = ess.summarize("synth", arr1, ess.paired_rows("synth", arr1))
    assert tables["2"]["mi"]["cost"] == pytest.approx(0.0, abs=1e-12)
    assert tables["2"]["mi"]["ident"] == 1.0
    assert tables["2"]["mi"]["n_swapped"] == 0

    # Deterministic, and independent of the number of worker processes
    arr = ess.run_dataset("synth", **KW_FAST)
    arr_w = ess.run_dataset("synth", **dict(KW_FAST, workers=2))
    for key, val in arr.items():
        if key == "time_s":
            continue
        np.testing.assert_array_equal(val, arr_w[key], err_msg=key)


# ---------------------------------------------------------------------------
# facloc (facility location, a method since 2026-10-19)
# ---------------------------------------------------------------------------

PRE_FACLOC_METHODS = ["mi", "trace", "entropy", "hub", "pca"]


def test_facloc_selections_are_facility_location_select(synth_dirs):
    """facloc's full selections and its alternatives are
    facility_location_select (clipped) on the split's training estimate /
    the subsample's estimate with the matching eligible mask."""
    data, _, _ = synth_dirs
    B, O = _write_dataset(data)
    assert ess.METHODS[:len(PRE_FACLOC_METHODS)] == PRE_FACLOC_METHODS
    assert ess.METHODS[-1] == "facloc"
    arr = ess.run_dataset("synth", **KW_FAST)
    k_max = max(KW_FAST["ks"])
    splits = list(cv_splits(M_SYN, KW_FAST["n_folds"],
                            KW_FAST["n_repeats"], ess.HOLDOUT_FRAC,
                            ess.SEED))
    for j, s in enumerate(splits):
        R = estimate_corr(B[s.train_idx], O[s.train_idx])[0]
        ref = facility_location_select(R, k_max,
                                       eligible=eligible_mask(
                                           O[s.train_idx]),
                                       similarity="clipped")["selected"]
        for i, k in enumerate(KW_FAST["ks"]):
            assert arr["facloc_selected"][j, i, :k].tolist() == ref[:k]
        pos = ess.subsample_positions(s, len(s.train_idx),
                                      KW_FAST["n_sub"], ess.SUB_FRAC)
        for b, p in enumerate(pos):
            tr = s.train_idx[p]
            R_b = estimate_corr(B[tr], O[tr])[0]
            ref_b = facility_location_select(
                R_b, k_max, eligible=eligible_mask(O[tr]))["selected"]
            for i, k in enumerate(KW_FAST["ks"]):
                assert arr["facloc_alt_selected"][j, i, b, :k].tolist() \
                    == ref_b[:k]


def test_facloc_changes_no_other_value(synth_dirs, monkeypatch):
    """Every array, paired row and macro of a run with the pre-facloc
    METHODS equals the run with facloc (time_s and the methods list
    aside); facloc only adds keys, rows and macros of its own."""
    data, _, _ = synth_dirs
    _write_dataset(data)
    new = ess.run_dataset("synth", **KW_FAST)
    p_new = ess.paired_rows("synth", new)
    _, mac_new = ess.summarize("synth", new, p_new)
    monkeypatch.setattr(ess, "METHODS", list(PRE_FACLOC_METHODS))
    old = ess.run_dataset("synth", **KW_FAST)
    p_old = ess.paired_rows("synth", old)
    _, mac_old = ess.summarize("synth", old, p_old)
    assert {k for k in set(new) - set(old)} == {
        k for k in new if k.startswith("facloc_")}
    for key, val in old.items():
        if key == "time_s":
            continue
        if key == "methods":
            assert [m for m in new[key] if m != "facloc"] == list(val)
            continue
        np.testing.assert_array_equal(val, new[key], err_msg=key)
    old_rows = [json.dumps(r, sort_keys=True, default=str) for r in p_old]
    new_rows = [json.dumps(r, sort_keys=True, default=str) for r in p_new
                if r["method"] != "facloc"]
    assert old_rows == new_rows
    assert {k: mac_new[k] for k in mac_old} == mac_old
    assert set(mac_new) > set(mac_old)
    assert all("Facloc" in k for k in set(mac_new) - set(mac_old))
