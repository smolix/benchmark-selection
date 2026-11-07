"""Tests for eval_prospective.py (prospective / temporal evaluation).

Synthetic data throughout; the last tests check the data checks and
loaders on the real matrices and are skipped where those are missing.
facloc (facility location, a method since 2026-10-19): its selection is
baseline_select.facility_location_select on the split's training estimate
and eligible set, and adding it (with its bootstrap pairs and its own
impute cross-check) leaves every value of the other methods unchanged.
Run from code/: python -m pytest -q test_eval_prospective.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

import eval_prospective as ep
from baseline_select import facility_location_select
from cv_split import ENTROPY_RF_SEED_OFFSET, RANDOM_SEED_OFFSET
from eval_entropy_vs_mi import estimate_corr, impute_and_evaluate
from greedy_select import random_select


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _factor_data(M=80, N=10, n_factors=2, miss=0.0, seed=0, shift=None):
    """Scores in [0, 1]-ish from a factor model; O drops a fraction `miss`
    (every row keeps >= 3 cells).  shift: (row mask, offset) added to
    those rows (a population shift of the 'new' models)."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((M, n_factors))
    L = rng.uniform(0.3, 1.0, (N, n_factors))
    B = 0.5 + 0.08 * (F @ L.T + 0.6 * rng.standard_normal((M, N)))
    if shift is not None:
        B[shift[0]] += shift[1]
    O = (rng.random((M, N)) >= miss).astype(np.float64)
    for i in range(M):
        if O[i].sum() < 3:
            O[i, rng.choice(N, 3, replace=False)] = 1.0
    return B * O, O


def _slow_reference(B_test, O_test, selected, R, mean, std, target_ok=None,
                    center=False):
    """Independent per-model loop: pooled R² over observed unselected
    (target) cells, impute_and_evaluate's slow path; center: SS_tot about
    the test-set column means (shift-adjusted R²) over the targets in
    columns with >= 2 observed test cells, NaN if that SS_tot vanishes."""
    Z = np.clip(O_test * (B_test - mean) / std, -10, 10) * O_test
    N = R.shape[1]
    n_obs = (O_test == 1).sum(axis=0)
    zbar = np.array([Z[O_test[:, j] == 1, j].mean()
                     if n_obs[j] else 0.0
                     for j in range(N)]) if center else np.zeros(N)
    res = tot = 0.0
    n = 0
    for i in range(B_test.shape[0]):
        a = [j for j in selected if O_test[i, j] == 1]
        e = [j for j in range(N) if j not in selected and O_test[i, j] == 1]
        if not a or not e:
            continue
        W = np.linalg.solve(R[np.ix_(a, a)] + 0.01 * np.eye(len(a)),
                            R[np.ix_(e, a)].T).T
        pred = W @ Z[i, a]
        keep = [t for t, j in enumerate(e)
                if (target_ok is None or target_ok[j])
                and (not center or n_obs[j] >= 2)]
        res += np.sum((Z[i, e] - pred)[keep] ** 2)
        tot += np.sum((Z[i, e] - zbar[e])[keep] ** 2)
        n += len(keep)
    if n == 0 or (center and tot <= 1e-12):
        return np.nan
    return 1.0 - res / max(tot, 1e-12)


def _args(**kw):
    a = argparse.Namespace(n_draws=3, n_controls=4, boot=50,
                           target_min_obs=ep.MIN_TARGET_OBS,
                           min_models=ep.MIN_MODELS,
                           epoch_cutoffs=ep.EPOCH_CUTOFFS,
                           olv2sub_cutoffs=ep.OLV2SUB_CUTOFFS)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _variant(M=70, N=8, n_train=40, miss=0.15, seed=1, shift=0.0,
             cols=None):
    rows = np.arange(M) >= n_train
    B, O = _factor_data(M, N, miss=miss, seed=seed, shift=(rows, shift))
    names = np.array([f"m{i:03d}" for i in range(M)])
    dates = np.array([f"2025-01-{1 + i % 28:02d}" for i in range(M)])
    return ep.make_variant("main", "synthetic", "", B, O, names, dates,
                           ~rows, rows, cols=cols), \
        np.array([f"b{j}" for j in range(N)])


# ---------------------------------------------------------------------------
# Imputation per test model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("miss", [0.0, 0.25])
def test_per_model_ss_matches_impute_and_evaluate(miss):
    B, O = _factor_data(90, 10, miss=miss, seed=3)
    tr, te = np.arange(60), np.arange(60, 90)
    R, mean, std = estimate_corr(B[tr], O[tr])
    test = ep.make_test_set(B[te], O[te], mean, std)
    ok = np.ones(10, dtype=bool)
    for sel in ([4], [1, 7, 3], [0, 2, 5, 8, 9]):
        ss = ep.per_model_ss(test, sel, R, ok)
        got = ep.pooled_r2(ss["ss_res_all"], ss["ss_tot_all"],
                           ss["n_eval_all"])
        ref = impute_and_evaluate(B[te], O[te], sel, R, mean, std, 10)["r2"]
        assert got == pytest.approx(ref, abs=1e-10)
        assert got == pytest.approx(
            _slow_reference(B[te], O[te], sel, R, mean, std), abs=1e-10)
        # all targets eligible: both R² agree
        assert ep.pooled_r2(ss["ss_res"], ss["ss_tot"], ss["n_eval"]) == got


def test_per_model_ss_counts_every_observed_target():
    """Test model 0 lacks benchmark 9, the others observe it; every model
    shares model 0's pattern on the rest.  Each observed unselected cell
    is a target (slow-path definition), including the extra ones."""
    B, O = _factor_data(50, 10, miss=0.0, seed=4)
    O[40, 9] = 0.0
    B[40, 9] = 0.0
    tr, te = np.arange(40), np.arange(40, 50)
    R, mean, std = estimate_corr(B[tr], O[tr])
    test = ep.make_test_set(B[te], O[te], mean, std)
    ss = ep.per_model_ss(test, [0, 3], R, np.ones(10, dtype=bool))
    assert ss["n_eval_all"][0] == 7 and np.all(ss["n_eval_all"][1:] == 8)
    got = ep.pooled_r2(ss["ss_res_all"], ss["ss_tot_all"], ss["n_eval_all"])
    assert got == pytest.approx(
        _slow_reference(B[te], O[te], [0, 3], R, mean, std), abs=1e-12)


def test_target_restriction_equals_masked_columns():
    B, O = _factor_data(90, 10, miss=0.2, seed=5)
    tr, te = np.arange(60), np.arange(60, 90)
    R, mean, std = estimate_corr(B[tr], O[tr])
    test = ep.make_test_set(B[te], O[te], mean, std)
    ok = np.ones(10, dtype=bool)
    ok[[6, 8]] = False
    sel = [0, 2, 4]
    ss = ep.per_model_ss(test, sel, R, ok)
    O_mask = O[te].copy()
    O_mask[:, [6, 8]] = 0.0
    ref = impute_and_evaluate(B[te] * O_mask, O_mask, sel, R, mean, std,
                              10)["r2"]
    got = ep.pooled_r2(ss["ss_res"], ss["ss_tot"], ss["n_eval"])
    assert got == pytest.approx(ref, abs=1e-10)
    assert got == pytest.approx(
        _slow_reference(B[te], O[te], sel, R, mean, std, ok), abs=1e-10)
    # the all-target R² is unaffected by the restriction
    full = ep.per_model_ss(test, sel, R, np.ones(10, dtype=bool))
    np.testing.assert_array_equal(ss["ss_res_all"], full["ss_res_all"])


def test_shift_adjusted_r2():
    """A uniform shift of the test rows inflates the training-mean R²
    but not the shift-adjusted one (SS_tot about the test means)."""
    M = 90
    rows = np.arange(M) >= 60
    tr, te = np.arange(60), np.arange(60, M)
    sel = [1, 4, 6]
    out = {}
    for shift in (0.0, 0.15):
        B, O = _factor_data(M, 10, miss=0.2, seed=6, shift=(rows, shift))
        R, mean, std = estimate_corr(B[tr], O[tr])
        test = ep.make_test_set(B[te], O[te], mean, std)
        ss = ep.per_model_ss(test, sel, R, np.ones(10, dtype=bool))
        r2 = ep.pooled_r2(ss["ss_res"], ss["ss_tot"], ss["n_eval"])
        r2_c = ep.pooled_r2(ss["ss_res"], ss["ss_tot_c"], ss["n_eval"])
        assert r2_c == pytest.approx(_slow_reference(
            B[te], O[te], sel, R, mean, std, center=True), abs=1e-10)
        out[shift] = (r2, r2_c)
    # without a shift the two baselines nearly agree; with it R² rises
    # above the shift-adjusted R²
    assert abs(out[0.0][0] - out[0.0][1]) < 0.1
    assert out[0.15][0] > out[0.15][1] + 0.1


def test_shift_adjusted_r2_needs_two_test_cells_per_column():
    """A target column observed by one test model has no test-set
    variance (its centered SS is 0): it is left out of r2_c (not of r2),
    and r2_c is NaN when no target column has two test cells."""
    B, O = _factor_data(70, 6, miss=0.0, seed=21)
    tr, te = np.arange(50), np.arange(50, 70)
    O[te[1:], 5] = 0.0                       # column 5: one test cell
    B = B * O
    R, mean, std = estimate_corr(B[tr], O[tr])
    test = ep.make_test_set(B[te], O[te], mean, std)
    assert test["center_ok"].tolist() == [True] * 5 + [False]
    ok = np.ones(6, dtype=bool)
    ss = ep.per_model_ss(test, [0, 1], R, ok)
    assert ss["n_eval"].sum() == ss["n_eval_c"].sum() + 1
    r2_c = ep.pooled_r2(ss["ss_res_c"], ss["ss_tot_c"], ss["n_eval_c"],
                        flat_nan=True)
    assert r2_c == pytest.approx(_slow_reference(
        B[te], O[te], [0, 1], R, mean, std, center=True), abs=1e-10)
    assert np.isfinite(r2_c) and r2_c > -10
    # only columns 0..4 selected: the one target left has one test cell
    ss = ep.per_model_ss(test, [0, 1, 2, 3, 4], R, ok)
    assert ss["n_eval"].sum() == 1 and ss["n_eval_c"].sum() == 0
    assert np.isnan(ep.pooled_r2(ss["ss_res_c"], ss["ss_tot_c"],
                                 ss["n_eval_c"], flat_nan=True))
    assert np.isfinite(ep.pooled_r2(ss["ss_res"], ss["ss_tot"],
                                    ss["n_eval"]))
    # identical test values in a column: no variance -> NaN, not -1e9
    B2 = B.copy()
    B2[te, 2] = 0.5
    test = ep.make_test_set(B2[te], O[te], mean, std)
    ss = ep.per_model_ss(test, [0, 1, 3, 4], R, ok)
    assert ss["n_eval_c"].sum() == len(te)
    assert np.isnan(ep.pooled_r2(ss["ss_res_c"], ss["ss_tot_c"],
                                 ss["n_eval_c"], flat_nan=True))


# ---------------------------------------------------------------------------
# No leakage from the test rows
# ---------------------------------------------------------------------------

def test_estimation_and_selection_use_training_rows_only(monkeypatch):
    v, bench = _variant(miss=0.1, seed=7)
    tr = np.arange(v.n_train)
    te = np.arange(v.n_train, v.B.shape[0])
    seen = []
    real = ep.estimate_corr

    def spy(B_train, O_train):
        seen.append((B_train.copy(), O_train.copy()))
        return real(B_train, O_train)

    monkeypatch.setattr(ep, "estimate_corr", spy)
    a = ep.evaluate_split(v.B, v.O, tr, te, 6, bench, range(3))
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][0], v.B[tr])
    np.testing.assert_array_equal(seen[0][1], v.O[tr])

    # Scramble the test rows: values, observation pattern, a shift
    rng = np.random.default_rng(0)
    B2, O2 = v.B.copy(), v.O.copy()
    B2[te] = rng.uniform(0, 1, B2[te].shape) + 0.3
    O2[te] = (rng.random(O2[te].shape) > 0.3).astype(float)
    B2[te] *= O2[te]
    b = ep.evaluate_split(B2, O2, tr, te, 6, bench, range(3))
    np.testing.assert_array_equal(a["eligible"], b["eligible"])
    for m in ep.METHODS:
        for ea, eb in zip(a["methods"][m], b["methods"][m]):
            assert ea["selected"] == eb["selected"], m
    assert not np.allclose(a["methods"]["mi"][0]["r2"][:4],
                           b["methods"]["mi"][0]["r2"][:4])


def test_eligibility_and_targets_from_training_counts():
    v, bench = _variant(miss=0.0, seed=8)
    O = v.O.copy()
    O[:v.n_train, 2] = 0.0           # no training data for benchmark 2
    O[:2, 5] = 1.0
    O[2:v.n_train, 5] = 0.0          # two training observations of 5
    B = v.B * O
    tr = np.arange(v.n_train)
    te = np.arange(v.n_train, v.B.shape[0])
    # target_min_obs 3 (the default before 2026-10-18): benchmark 5 is
    # left out of r2 but, with 2 >= MIN_TARGET_OBS training models, kept
    # in r2_all; benchmark 2 is no target of either (rule D1)
    s = ep.evaluate_split(B, O, tr, te, 6, bench, range(2),
                          target_min_obs=3)
    assert not s["eligible"][2] and not s["eligible"][5]
    n_te = len(te)
    assert s["n_test_cells_no_train"] == n_te
    assert s["n_test_cells_few_train"] == n_te
    assert s["n_test_bench_no_train"] == 1
    assert s["n_test_bench_few_train"] == 1
    for m in ep.METHODS:
        for e in s["methods"][m]:
            assert 2 not in e["selected"] and 5 not in e["selected"]
    # all-target R² still includes them, the target R² does not
    e = s["methods"]["mi"][0]
    assert np.all(np.isfinite(e["r2"][:5]))
    assert not np.allclose(e["r2"][:5], e["r2_all"][:5])
    s0 = ep.evaluate_split(B, O, tr, te, 6, bench, range(2),
                           target_min_obs=0)
    np.testing.assert_allclose(s0["methods"]["mi"][0]["r2"],
                               s0["methods"]["mi"][0]["r2_all"])
    # the default (MIN_TARGET_OBS = 2): r2 == r2_all, benchmark 5 scored
    sd = ep.evaluate_split(B, O, tr, te, 6, bench, range(2))
    assert sd["n_test_cells_few_train"] == 0
    np.testing.assert_allclose(sd["methods"]["mi"][0]["r2"],
                               sd["methods"]["mi"][0]["r2_all"])
    np.testing.assert_allclose(sd["methods"]["mi"][0]["r2_all"],
                               s["methods"]["mi"][0]["r2_all"])


def test_variant_columns_no_future_leak():
    """A variant on a column subset (epoch's time mask): the dropped
    columns influence nothing (changing them, training cells included,
    leaves every pick and R² unchanged); picks and eligible masks come
    back in setting indices and never hit a dropped column; k_max may
    exceed the number of kept columns."""
    N, K = 8, 7
    keep = np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
    v, bench = _variant(N=N, miss=0.1, seed=12, cols=keep)
    assert v.B.shape[1] == keep.sum()
    np.testing.assert_array_equal(v.cols, np.flatnonzero(keep))
    res = ep.run_variant(v, bench, K, 2, 3, 20, label="test")

    # same data with other values / observations in the dropped columns
    rows = np.arange(70) >= 40
    B, O = _factor_data(70, N, miss=0.1, seed=12, shift=(rows, 0.0))
    rng = np.random.default_rng(5)
    O[:, ~keep] = (rng.random((70, int((~keep).sum()))) > 0.5)
    B[:, ~keep] = rng.uniform(0, 1, (70, int((~keep).sum()))) * O[:, ~keep]
    w = ep.make_variant("main", "synthetic", "", B, O, v.model_names,
                        v.model_dates, ~rows, rows, cols=keep)
    res2 = ep.run_variant(w, bench, K, 2, 3, 20, label="test")
    for m in ep.METHODS:
        np.testing.assert_array_equal(res[f"prosp_{m}_r2"],
                                      res2[f"prosp_{m}_r2"])
        np.testing.assert_array_equal(res[f"ctrl_{m}_r2"],
                                      res2[f"ctrl_{m}_r2"])
    for split in [res["prosp"]] + res["ctrl"]:
        assert split["eligible"].shape == (N,)
        assert not split["eligible"][~keep].any()
        for m in ep.METHODS:
            for e in split["methods"][m]:
                picks = [j for j in e["selected"] if j >= 0]
                assert picks and all(keep[j] for j in picks), m
    args = _args(n_draws=2, n_controls=3, boot=20)
    log = ep.build_log("helm", bench, [res], args)
    np.testing.assert_array_equal(log["variant_columns"][0], keep)
    assert log["variant_n_bench"][0] == keep.sum()
    assert log["prosp_eligible"].shape == (1, N)
    assert not log["eligible"][:, ~keep].any()
    allowed = np.r_[-1, np.flatnonzero(keep)]
    assert np.all(np.isin(log["prosp_mi_selected"][0], allowed))
    # k = 6 picks every kept benchmark (nothing left), k = 7 impossible
    assert np.isnan(log["prosp_mi_r2"][0, 5:]).all()
    s = ep.summary_json("helm", log, [res], {}, args, 1.0)
    assert s["variants"][0]["benchmarks"] == list(bench[keep])
    assert set(s["variants"][0]["selected"]["mi"]) <= set(bench[keep])


# ---------------------------------------------------------------------------
# Splits and common random numbers
# ---------------------------------------------------------------------------

def test_seed_tags_are_new_and_distinct():
    new = {ep.PROSP_RANDOM_SEED_OFFSET, ep.PROSP_ENTROPY_RF_SEED_OFFSET,
           ep.CONTROL_SPLIT_SEED_OFFSET, ep.BOOT_SEED_OFFSET}
    assert len(new) == 4 and min(new) >= 13
    assert not new & {RANDOM_SEED_OFFSET, ENTROPY_RF_SEED_OFFSET}
    assert ep.draw_seed(55, 3) == (55, 3)


def test_control_split_sizes_and_determinism():
    M, n_tr = 57, 40
    seen = set()
    for d in range(5):
        tr, te = ep.control_split(M, n_tr, d)
        assert len(tr) == n_tr and len(te) == M - n_tr
        assert np.intersect1d(tr, te).size == 0
        np.testing.assert_array_equal(np.union1d(tr, te), np.arange(M))
        tr2, _ = ep.control_split(M, n_tr, d)
        np.testing.assert_array_equal(tr, tr2)
        seen.add(tuple(tr))
    assert len(seen) == 5
    # same draw, larger training set: nested in INDEX space (shared
    # permutation); different variants put different models at the same
    # positions, so this does not nest the control training sets of
    # different cutoffs
    a, _ = ep.control_split(M, 20, 2)
    b, _ = ep.control_split(M, 30, 2)
    assert set(a) <= set(b)


def test_stochastic_methods_use_draw_seeds():
    R = np.eye(6) * 0.5 + 0.5
    el = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    for d in range(3):
        r = ep.run_selection("random", R, 4, None, el, d)
        ref = random_select(R, 4, seed=(ep.SEED + 13, d), eligible=el)
        assert r["selected"] == ref["selected"]
    assert ep.STOCHASTIC == ["random"] and "entropy_rf" not in ep.METHODS
    with pytest.raises(ValueError):
        ep.run_selection("entropy_rf", R, 4, None, el, 0)


# METHODS, DETERMINISTIC, BOOT_PAIRS and CHECK_APART before facloc
PRE_FACLOC = {"METHODS": ["mi", "entropy", "trace", "hub", "pca", "random"],
              "DETERMINISTIC": ["mi", "entropy", "trace", "hub", "pca"],
              "BOOT_PAIRS": [("mi", "random"), ("mi", "entropy"),
                             ("mi", "trace"), ("trace", "random"),
                             ("entropy", "random")],
              "CHECK_APART": []}


def test_facloc_is_facility_location_select():
    """run_selection("facloc") is facility_location_select (clipped) on the
    same Sigma and eligible mask, and evaluate_split's facloc picks are
    those on the training estimate and the split's eligible set."""
    B, O = _factor_data(60, 9, miss=0.2, seed=21)
    R, _, _ = estimate_corr(B[:40], O[:40])
    names = np.array([f"b{j}" for j in range(9)])
    el = np.ones(9, dtype=bool)
    el[[2, 5]] = False
    for eligible in (None, el):
        got = ep.run_selection("facloc", R, 6, names, eligible)
        ref = facility_location_select(R, 6, benchmark_names=names,
                                       eligible=eligible,
                                       similarity="clipped")
        assert got["selected"] == ref["selected"]
        assert got["tie_breaks"] == ref["tie_breaks"]
        assert got["index_ties"] == ref["index_ties"]
    split = ep.evaluate_split(B, O, np.arange(40), np.arange(40, 60), 6,
                              names, range(2))
    ref = facility_location_select(R, 6, eligible=ep.eligible_mask(O[:40]))
    e = split["methods"]["facloc"]
    assert len(e) == 1                                  # deterministic
    assert e[0]["selected"] == ep.padded(ref["selected"], 6)
    assert "facloc" in ep.DETERMINISTIC and "facloc" in ep.PLOT_METHODS
    assert ep.METHODS[-2:] == ["facloc", "random"]
    for table in (ep.METHOD_LABEL, ep.METHOD_COLOR, ep.METHOD_MARKER):
        assert "facloc" in table
    # the key pairs keep their rows; facloc's pairs are appended
    assert ep.BOOT_PAIRS[:5] == PRE_FACLOC["BOOT_PAIRS"]
    assert ep.BOOT_PAIRS[5:] == [("mi", "facloc"), ("entropy", "facloc"),
                                 ("trace", "facloc"), ("facloc", "random")]


def test_facloc_changes_no_other_value(monkeypatch):
    """run_variant and build_log with facloc equal the pre-facloc run on
    every key of the latter (the boot_* / ctrl_pair_* rows of the first
    five pairs; prosp_impute_check without facloc's cross-check)."""
    v, bench = _variant(miss=0.15, seed=23, shift=0.05)
    args = _args(n_draws=2, n_controls=3, boot=40)
    new = ep.run_variant(v, bench, 5, 2, 3, 40, label="new")
    log_new = ep.build_log("helm", bench, [new], args)
    mac_new = ep.make_macros("helm", log_new, {})
    for name, val in PRE_FACLOC.items():
        monkeypatch.setattr(ep, name, val)
    old = ep.run_variant(v, bench, 5, 2, 3, 40, label="old")
    log_old = ep.build_log("helm", bench, [old], args)
    mac_old = ep.make_macros("helm", log_old, {})
    P = len(PRE_FACLOC["BOOT_PAIRS"])
    for key in [k for k in old if k.startswith(("prosp_", "ctrl_", "boot_",
                                                "penalty", "rmse_pen"))]:
        a, b = np.asarray(old[key]), np.asarray(new[key])
        if key.startswith(("boot_", "ctrl_pair_")):
            b = b[:P]
        np.testing.assert_array_equal(a, b, err_msg=key)
    assert old["prosp"]["impute_check"] == new["prosp"]["impute_check"]
    assert "impute_check_facloc" in new["prosp"]
    assert set(log_new) - set(log_old) == {
        k for k in log_new if "facloc" in k}
    for key, a in log_old.items():
        b = log_new[key]
        if key == "methods":
            assert [m for m in b if m != "facloc"] == list(a)
            continue
        if key.startswith(("boot_", "ctrl_pair_")) and np.ndim(b) == 3:
            b = b[:, :P]                         # (V, P, len(BOOT_K))
        elif key == "boot_pairs":
            b = b[:P]
        np.testing.assert_array_equal(a, b, err_msg=key)
    assert {k: mac_new[k] for k in mac_old} == dict(mac_old)
    assert set(mac_new) > set(mac_old)
    assert all("Facloc" in k for k in set(mac_new) - set(mac_old))


def test_date_split_and_make_variant():
    dates = np.array(["2024-06-30", "2024-07-01", "None", "", "2025-01-02"])
    tr, te = ep.date_split(dates, "2024-07-01")
    np.testing.assert_array_equal(tr, [True, False, False, False, False])
    np.testing.assert_array_equal(te, [False, True, False, False, True])
    B = np.arange(10.0).reshape(5, 2)
    v = ep.make_variant("k", "l", "", B, np.ones_like(B), np.arange(5),
                        dates, tr, te)
    assert v.n_train == 1 and v.n_test == 2
    np.testing.assert_array_equal(v.B[:, 0], [0, 2, 8])
    np.testing.assert_array_equal(v.rows, [0, 1, 4])
    np.testing.assert_array_equal(v.cols, [0, 1])
    with pytest.raises(ValueError):
        ep.make_variant("k", "l", "", B, np.ones_like(B), np.arange(5),
                        dates, [0, 1], [1, 2])
    # column subset
    v = ep.make_variant("k", "l", "", B, np.ones_like(B), np.arange(5),
                        dates, tr, te, cols=np.array([False, True]))
    assert v.B.shape == (3, 1)
    np.testing.assert_array_equal(v.B[:, 0], [1, 3, 9])


def test_boundary_duplicates_dropped():
    """A test row with a training row's pattern and values (>= 3 observed
    cells) or its model_sha is dropped; near-duplicates and rows with
    fewer than 3 observed cells are kept."""
    rng = np.random.default_rng(0)
    M, N = 12, 6
    B = rng.uniform(0, 1, (M, N))
    O = np.ones((M, N))
    train, test = np.arange(6), np.arange(6, M)
    B[7] = B[2]                                  # exact duplicate
    B[8] = B[3]
    B[8, 0] += 1e-6                              # near duplicate: kept
    O[9, 3:] = 0.0
    O[1, 3:] = 0.0
    B[9] = B[1]                                  # duplicate on 3 cells
    O[10, 2:] = 0.0
    O[4, 2:] = 0.0
    B[10] = B[4]                                 # only 2 cells: kept
    B, O = B * O, O
    names = np.array([f"m{i}" for i in range(M)])
    sha = np.array([f"h{i}" for i in range(M)])
    sha[11] = sha[0]                             # same weights
    sha[6] = ""                                  # empty sha never matches
    sha[5] = ""
    dups = ep.boundary_duplicates(B, O, train, test, sha)
    assert sorted(dups) == [(7, 2, "identical"), (9, 1, "identical"),
                            (11, 0, "model_sha")]
    v = ep.make_variant("k", "l", "", B, O, names, names, train, test,
                        sha=sha)
    assert v.n_test == 3
    assert set(v.model_names[v.n_train:]) == {"m6", "m8", "m10"}
    assert v.info["duplicates"]["n_dropped"] == 3
    assert {d["test"] for d in v.info["duplicates"]["rows"]} == \
        {"m7", "m9", "m11"}
    # without sha only the identical rows go
    v = ep.make_variant("k", "l", "", B, O, names, names, train, test)
    assert v.n_test == 4
    # identical on the kept columns only: duplicates of the variant
    B2 = B.copy()
    B2[7, 5] = 0.123
    v = ep.make_variant("k", "l", "", B2, O, names, names, train, test,
                        cols=np.arange(5))
    assert "m7" not in set(v.model_names)


# ---------------------------------------------------------------------------
# Bootstrap and comparisons
# ---------------------------------------------------------------------------

def _fake_split(ss_by_method, n):
    meth = {}
    for m, draws in ss_by_method.items():
        meth[m] = [{"ss": {1: {"ss_res": r, "ss_tot": t,
                               "n_eval": np.ones(n, dtype=np.int64)}}}
                   for r, t in draws]
    return {"failed": False, "n_val": n, "methods": meth}


def test_bootstrap_pairs():
    rng = np.random.default_rng(0)
    n = 40
    tot = rng.uniform(1, 2, n)
    good = 0.2 * tot
    bad = 0.5 * tot
    split = _fake_split({"mi": [(good, tot)], "trace": [(good, tot)],
                         "random": [(bad, tot), (0.6 * tot, tot)]}, n)
    out = ep.bootstrap_pairs(split, [("mi", "random"), ("mi", "trace")],
                             [1], 300, 99)
    # mean over the two random draws: R² 0.5 and 0.4 -> 0.45
    assert out["diff"][0, 0] == pytest.approx(0.8 - 0.45)
    assert out["prob_pos"][0, 0] == 1.0 and out["lo"][0, 0] > 0
    assert out["diff"][1, 0] == 0.0 and out["prob_zero"][1, 0] == 1.0
    assert out["lo"][1, 0] == out["hi"][1, 0] == 0.0
    again = ep.bootstrap_pairs(split, [("mi", "random"), ("mi", "trace")],
                               [1], 300, 99)
    np.testing.assert_array_equal(out["lo"], again["lo"])
    W = ep.boot_weights(n, 5, np.random.default_rng(1))
    assert np.all(W.sum(axis=1) == n)
    ss = {"ss_res": good[None], "ss_tot": tot[None],
          "n_eval": np.ones((1, n))}
    assert ep.boot_r2(np.ones((1, n)), ss)[0] == pytest.approx(0.8)


def test_penalty_pval():
    ctrl = np.array([[0.5, np.nan], [0.6, np.nan], [0.7, np.nan],
                     [0.8, np.nan]])
    p = ep.penalty_pval(np.array([0.55, 0.5]), ctrl)
    assert p[0] == pytest.approx((1 + 1) / 5)
    assert np.isnan(p[1])
    assert ep.penalty_pval(np.array([0.1, 0.0]), ctrl)[0] == \
        pytest.approx(1 / 5)


# ---------------------------------------------------------------------------
# End to end on a synthetic variant
# ---------------------------------------------------------------------------

def test_run_variant_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, "LOGS", tmp_path)
    monkeypatch.setattr(ep, "OUT", tmp_path)
    v, bench = _variant(miss=0.1, seed=11, shift=0.05)
    args = _args()
    res = ep.run_variant(v, bench, 6, args.n_draws, args.n_controls,
                         args.boot, label="test")
    for m in ep.METHODS:
        np.testing.assert_allclose(
            res[f"penalty_{m}"],
            res[f"prosp_{m}_r2"] - np.nanmean(res[f"ctrl_{m}_r2"], axis=0))
        assert res[f"ctrl_{m}_r2"].shape == (args.n_controls, 6)
    # stochastic prospective value = mean over the draws
    draws = np.stack([e["r2"] for e in res["prosp"]["methods"]["random"]])
    np.testing.assert_allclose(res["prosp_random_r2"], draws.mean(axis=0))
    assert res["prosp"]["impute_check"] < 1e-9
    assert res["prosp"]["impute_check_facloc"] < 1e-9

    log = ep.build_log("helm", bench, [res], args)
    assert log["mi_r2"].shape == (args.n_controls, 6)
    assert log["facloc_r2"].shape == (args.n_controls, 6)
    assert log["prosp_facloc_selected"].shape == (1, 6)
    assert log["prosp_facloc_impute_check"].shape == (1,)
    assert log["boot_diff"].shape == (1, len(ep.BOOT_PAIRS), len(ep.BOOT_K))
    assert list(log["boot_pairs"])[-1] == "facloc-random"
    assert log["prosp_random_r2_draws"].shape == (1, args.n_draws, 6)
    np.testing.assert_array_equal(log["split_id"], np.arange(4))
    assert log["prosp_mi_ss_res"].shape == (1, len(ep.BOOT_K), v.n_test)
    summary = ep.summary_json("helm", log, [res], {}, args, 1.0)
    json.dumps(ep._jsonable(summary), allow_nan=False)
    macros = summary["macros"]
    assert all(name.isalpha() for name in macros)
    assert "RsqProspHelmMiKfive" in macros
    assert "BootDiffProspHelmMiVsRandomKfive" in macros
    assert "PenaltyProspHelmTraceKthree" in macros
    assert "RsqProspHelmFaclocKfive" in macros
    assert "PenaltyProspHelmFaclocKthree" in macros
    assert "BootDiffProspHelmFaclocVsRandomKfive" in macros
    assert "PairDiffCtrlHelmMiVsFaclocKfive" in macros
    assert macros["NTestProspHelm"] == v.n_test
    assert macros["NBenchProspHelm"] == len(bench)
    assert macros["NTestDupProspHelm"] == 0
    assert "NTestOldProspHelm" not in macros          # no release info
    assert "NBenchFutureProspHelm" not in macros      # epoch only
    assert macros["NDrawsProspHelm"] == args.n_draws
    ep.save_log("helm", log, summary)
    ep.plot_setting("helm", log)
    assert (tmp_path / "prospective_helm.pdf").is_file()
    back = ep.load_log("helm")
    np.testing.assert_array_equal(back["prosp_mi_r2"], log["prosp_mi_r2"])


def test_control_splits_use_all_draws_with_prospective_seeds():
    """Every control split runs all D draws of the stochastic methods
    with the prospective split's seeds (tag, d); the control statistic is
    the mean over them, the same statistic as the prospective one, and
    penalty_pval ranks those means."""
    v, bench = _variant(miss=0.1, seed=13)
    D, C, K = 3, 4, 5
    res = ep.run_variant(v, bench, K, D, C, 20, label="test")
    for m in ep.STOCHASTIC:
        dr = res[f"ctrl_{m}_r2_draws"]
        assert dr.shape == (C, D, K)
        mean = ep._nanmean(dr, axis=1)
        np.testing.assert_allclose(res[f"ctrl_{m}_r2"], mean)
        np.testing.assert_allclose(
            res[f"penalty_pval_{m}"],
            ep.penalty_pval(res[f"prosp_{m}_r2"], mean))
        prosp = np.stack([e["r2"] for e in res["prosp"]["methods"][m]])
        np.testing.assert_allclose(res[f"prosp_{m}_r2"],
                                   ep._nanmean(prosp, axis=0))
    M = v.B.shape[0]
    for c in range(C):
        tr, _ = ep.control_split(M, v.n_train, c)
        R, _, _ = estimate_corr(v.B[tr], v.O[tr])
        el = ep.eligible_mask(v.O[tr])
        ctrl = res["ctrl"][c]["methods"]
        assert len(ctrl["random"]) == D and "entropy_rf" not in ctrl
        assert len(ctrl["mi"]) == 1
        for d in range(D):
            ref = random_select(R, K, seed=(ep.SEED + 13, d), eligible=el)
            assert ctrl["random"][d]["selected"] == ep.padded(
                ref["selected"], K)
    log = ep.build_log("helm", bench, [res], _args(n_draws=D, n_controls=C))
    assert log["random_r2_draws"].shape == (C, D, K)
    np.testing.assert_array_equal(log["random_r2"], res["ctrl_random_r2"])


def test_min_models_suppresses_low_coverage_macros():
    v, bench = _variant(miss=0.1, seed=14)
    res = ep.run_variant(v, bench, 5, 2, 3, 20, label="test")
    for mm, emit in ((1, True), (v.n_test + 1, False)):
        args = _args(n_draws=2, n_controls=3, boot=20, min_models=mm)
        log = ep.build_log("helm", bench, [res], args)
        s = ep.summary_json("helm", log, [res], {}, args, 1.0)
        mac, wh = s["macros"], set(s["macros_suppressed"])
        for name in ("RsqProspHelmMiKthree", "RsqCenProspHelmMiKthree",
                     "PenaltyProspHelmTraceKthree",
                     "PenaltyPvalProspHelmRandomKthree",
                     "RsqCtrlHelmMiKthree",
                     "BootDiffProspHelmMiVsRandomKthree",
                     "PairDiffCtrlHelmMiVsRandomKthree"):
            assert (name in mac) == emit, name
            assert (name in wh) != emit, name
        # counts are always emitted
        assert mac["NModelsProspHelmMiKthree"] >= 1
        assert "NModelsCtrlHelmMiKthree" in mac
        assert mac["MinModelsProspHelm"] == mm
        assert 0 < mac["FillTestProspHelm"] <= 1
        assert 0 < mac["FillTestCtrlHelm"] <= 1
        k3 = s["variants"][0]["r2"]["mi"]["k3"]
        assert k3["low_coverage"] != emit
        assert k3["prospective"] is not None    # the value stays logged
        assert not set(mac) & wh


def test_cutoff_macro():
    assert ep.cutoff_macro("2024-07-01") == "CutJulTwentyfour"
    assert ep.cutoff_macro("2025-01-15") == "CutJanTwentyfiveDfifteen"
    assert ep.cutoff_macro("2026-01-01").isalpha()


# ---------------------------------------------------------------------------
# Real data (skipped without the matrices)
# ---------------------------------------------------------------------------

def _have(*names):
    return all((ep.DATA / n).is_file() for n in names)


def _check_variants(bench, variants):
    """Every variant indexes the benchmark list its loader returns (the
    names run_variant passes on) and holds its own columns."""
    for v in variants:
        assert len(v.cols) == v.B.shape[1] == v.O.shape[1]
        assert len(v.cols) and v.cols.min() >= 0
        assert v.cols.max() < len(bench)
        assert len(v.rows) == v.B.shape[0] == len(v.model_names)


@pytest.mark.skipif(not _have("mteb.matrix.npz", "mteb2026.matrix.npz",
                              "mteb2026.meta.json"),
                    reason="MTEB matrices missing")
def test_mteb2026_setting():
    bench, variants, checks = ep.load_mteb2026(_args())
    _check_variants(bench, variants)
    c = checks["mteb_snapshot"]
    assert c["n_rows"] == c["n_snapshot_rows"]
    assert not c["matrix_rows_not_covered"]
    assert c["max_abs_diff"] < 0.05
    main, late = variants
    assert main.n_train == c["n_snapshot_rows"] and main.n_test == 57
    assert late.n_test == 36
    assert not set(main.model_names[main.n_train:]) & \
        set(main.model_names[:main.n_train])
    assert main.info["duplicates"]["n_dropped"] == 0
    assert late.info["duplicates"]["n_dropped"] == 0
    # new to the leaderboard, not in time: released by 2026-03-14
    assert main.info["n_test_old"] == 21 and late.info["n_test_old"] == 0


@pytest.mark.skipif(not _have("helm_lite.matrix.npz", "helm_lite.meta.json",
                              "helm.npz", "merged.matrix.npz",
                              "canonical_mapping.json"),
                    reason="HELM matrices missing")
def test_helm_setting():
    bench, variants, checks = ep.load_helm(_args())
    _check_variants(bench, variants)
    assert checks["helm_npz"]["max_abs_diff"] < 1e-12
    assert checks["merged"]["max_abs_diff"] < 1e-12
    assert checks["helm_npz"]["n_rows"] == 30
    assert variants[0].n_train == 30 and variants[0].n_test == 61
    assert variants[0].info["duplicates"]["n_dropped"] == 0
    assert sorted(variants[0].info["test_release"]["old_models"]) == \
        ["Mistral Medium (2312)", "Phi-2"]


@pytest.mark.skipif(not _have("epoch.matrix.npz", "epoch.meta.json"),
                    reason="epoch matrix missing")
def test_epoch_time_mask():
    """Default epoch: only benchmarks released before the cutoff, so no
    training cell of a post-cutoff benchmark is used (the look-ahead of
    the split by model alone, kept in epoch_bymodel)."""
    args = _args()
    bdate = {b["name"]: b["release_date"]
             for b in ep._load_meta("epoch")["benchmarks"]}
    bench, variants, _ = ep.load_epoch(args)
    _, by_model, _ = ep.LOADERS["epoch_bymodel"](args)
    _check_variants(bench, variants)
    _check_variants(bench, by_model)
    train_future = {"2024-07-01": 97, "2025-01-01": 179,
                    "2025-07-01": 190, "2026-01-01": 238}
    elig_future = {"2024-07-01": 8, "2025-01-01": 15, "2025-07-01": 9,
                   "2026-01-01": 9}
    for v, w in zip(variants, by_model):
        c = v.cutoff
        kept = set(bench[v.cols])
        assert kept and all(bdate[b] < c for b in kept)
        assert all(bdate[b] >= c for b in set(bench) - kept)
        assert max(v.model_dates[:v.n_train]) < c
        assert min(v.model_dates[v.n_train:]) >= c
        tm = v.info["time_mask"]
        assert tm["applied"] and tm["n_bench_kept"] == len(kept)
        assert tm["n_train_cells_future"] == train_future[c]
        assert tm["n_eligible_future"] == elig_future[c]
        assert v.info["duplicates"]["n_dropped"] == 0
        # the split by model only keeps every benchmark, same rows
        assert len(w.cols) == len(bench) and not w.info["time_mask"][
            "applied"]
        np.testing.assert_array_equal(v.rows, w.rows)
        np.testing.assert_array_equal(v.B, w.B[:, v.cols])
    lmca = list(bench).index("LMCA")
    assert lmca not in variants[1].cols and lmca in by_model[1].cols


@pytest.mark.skipif(not _have("olv2sub.matrix.npz", "olv2sub.meta.json"),
                    reason="olv2sub matrix missing")
def test_olv2sub_setting():
    bench, variants, checks = ep.load_olv2sub(_args())
    _check_variants(bench, variants)
    assert len(checks["math_hard_dropped"]) == 7
    assert not any(b.startswith("math_hard/") for b in bench)
    assert all(v.n_test >= ep.MIN_OLV2SUB_TEST for v in variants)
    for v in variants:
        assert max(v.model_dates[:v.n_train]) < v.cutoff
        assert min(v.model_dates[v.n_train:]) >= v.cutoff
    # duplicates across the boundary are gone
    meta = ep._load_meta("olv2sub")
    sha_of = dict(zip(meta["model_names"], meta["models"]["model_sha"]))
    dropped = {"cut_2024-10-01": 3, "cut_2025-01-01": 3,
               "cut_2025-03-01": 6}
    for v in variants:
        assert v.info["duplicates"]["n_dropped"] == dropped[v.key]
        tr_sha = {sha_of[n] for n in v.model_names[:v.n_train]} - {None, ""}
        assert not any(sha_of[n] in tr_sha
                       for n in v.model_names[v.n_train:])
        M = v.B.shape[0]
        assert not ep.boundary_duplicates(v.B, v.O, np.arange(v.n_train),
                                          np.arange(v.n_train, M))
    names = {r["test"] for v in variants for r in v.info["duplicates"]["rows"]}
    assert {"microsoft/phi-4",
            "NikolaSigmoid/AceMath-1.5B-Instruct-1epoch"} <= names


# ---------------------------------------------------------------------------
# Protocol rule D2 and --workers
# ---------------------------------------------------------------------------

def test_prediction_clip_matches_impute_and_evaluate():
    """Predictions are clipped to the valid range (rule D2) exactly as
    impute_and_evaluate clips them: the crosscheck of r2_all against it
    holds with tight bounds, wide bounds change nothing, tight ones do."""
    v, bench = _variant(miss=0.15, seed=31)
    tr = np.arange(v.n_train)
    te = np.arange(v.n_train, v.B.shape[0])
    obs = v.B[v.O == 1]
    N = v.B.shape[1]
    tight = (np.full(N, np.quantile(obs, 0.3)),
             np.full(N, np.quantile(obs, 0.7)))
    wide = (np.full(N, -1e6), np.full(N, 1e6))
    base = ep.evaluate_split(v.B, v.O, tr, te, 4, bench, range(2))
    w = ep.evaluate_split(v.B, v.O, tr, te, 4, bench, range(2), bounds=wide)
    t = ep.evaluate_split(v.B, v.O, tr, te, 4, bench, range(2),
                          keep_ks=[1, 3], crosscheck=True, bounds=tight)
    for m in ep.DETERMINISTIC:
        np.testing.assert_array_equal(base["methods"][m][0]["r2"],
                                      w["methods"][m][0]["r2"])
    assert t["impute_check"] < 1e-12
    assert t["impute_check_facloc"] < 1e-12
    assert not np.allclose(base["methods"]["mi"][0]["r2"],
                           t["methods"]["mi"][0]["r2"])


def test_control_splits_do_not_depend_on_workers():
    v, bench = _variant(miss=0.1, seed=33)
    a = ep.run_variant(v, bench, 4, 2, 3, 10, label="w1", workers=1)
    b = ep.run_variant(v, bench, 4, 2, 3, 10, label="w2", workers=2)
    keys = [k for k in a if k.startswith(("ctrl_", "penalty"))]
    assert keys
    for k in keys:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
    for c in range(3):
        for m in ep.METHODS:
            for ea, eb in zip(a["ctrl"][c]["methods"][m],
                              b["ctrl"][c]["methods"][m]):
                assert ea["selected"] == eb["selected"]
                np.testing.assert_array_equal(ea["r2"], eb["r2"])
