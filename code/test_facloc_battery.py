"""Facility location in the paper's original comparisons (synthetic, < 60 s).

eval_entropy_vs_mi, eval_greedy_all, eval_benchpress and eval_logit
compare greedy entropy / MI / random; since 2026-10-19 they also run
greedy trace minimization and greedy facility location (facloc, the
paper's headline method).  Checked here:

  * every script's facloc selection is baseline_select.
    facility_location_select on the same correlation matrix and eligible
    mask (selection order, residual variances, tie counts), and its trace
    selection greedy_trace_select's;
  * appending trace and facloc changes nothing of a run of the original
    methods: every per-split array of entropy, mi and random is
    bit-identical, the greedy_cv per-split selection log of the original
    methods is a prefix of the new one, and so are the paired rows (the
    rows of the new pairs follow all the original ones).

The end-to-end runs of these scripts on a generic dataset are in
test_battery_generic.py; the shared imputation in
test_impute_and_evaluate.py.
"""

import atexit
import json
import math
import os
import shutil
import tempfile

# Outputs of the imported eval scripts go to a scratch root unless the
# caller (run_all.sh's tests job) set one
if not os.environ.get("BENCHSELECT_EXPERIMENT_ROOT"):
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-facloc-battery-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

import eval_benchpress as ebp
import eval_entropy_vs_mi as emi
import eval_greedy_all as ega
import eval_logit as el
from baseline_select import facility_location_select, greedy_trace_select
from cv_split import cv_splits

SEED = 42


# ── helpers ──────────────────────────────────────────────────────

def _corr(N=9, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, 3))
    S = W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))
    d = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def _toy_dataset(d, M=60, N=6, seed=0):
    """Sparse toy data; benchmark 5 has three models only (ineligible on
    the splits that hold one of them out)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(M, 2))
    X = 1 / (1 + np.exp(-(z @ rng.normal(size=(2, N))
                          + 0.3 * rng.normal(size=(M, N)))))
    O = (rng.random((M, N)) < 0.85).astype(np.float64)
    O[:, 0] = 1
    O[3:, 5] = 0
    np.savez(d / "toy.matrix.npz", B=X * O, O=O,
             model_names=np.array([f"m{i}" for i in range(M)]),
             benchmark_names=np.array([f"t{j}" for j in range(N)]))
    (d / "toy.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [None, None]}}))


def _same_result(a, b):
    """Two selector results agree on picks, residual variances and ties."""
    assert list(a["selected"]) == list(b["selected"])
    assert np.array_equal(np.asarray(a["residual_variance"]),
                          np.asarray(b["residual_variance"]), equal_nan=True)
    assert list(a["tie_breaks"]) == list(b["tie_breaks"])
    assert list(a["index_ties"]) == list(b["index_ties"])


def _rows_equal(a, b):
    """Paired rows equal field by field (NaN == NaN)."""
    return a.keys() == b.keys() and all(
        a[f] == b[f] or (isinstance(a[f], float) and isinstance(b[f], float)
                         and math.isnan(a[f]) and math.isnan(b[f]))
        for f in a)


def _equal(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return (x.shape == y.shape and x.dtype == y.dtype
            and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))


@pytest.fixture
def toy(tmp_path, monkeypatch):
    _toy_dataset(tmp_path)
    for mod in (emi, ega, ebp, el):
        monkeypatch.setattr(mod, "DATA", tmp_path)
    return tmp_path


# ── the facloc / trace selections of every script ────────────────

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_facloc_and_trace_are_the_baseline_select_functions(seed):
    N, k = 9, 6
    R = _corr(N, seed)
    names = np.array([f"b{j}" for j in range(N)])
    eligible = np.ones(N, dtype=bool)
    eligible[[1, 4]] = False
    split = next(iter(cv_splits(50, 10, 1, 0.1, SEED)))
    ref = {"facloc": facility_location_select(R, k, benchmark_names=names,
                                              eligible=eligible),
           "trace": greedy_trace_select(R, k, benchmark_names=names,
                                        eligible=eligible)}
    assert not set(ref["facloc"]["selected"]) & {1, 4}
    for m in ("facloc", "trace"):
        got = {
            "eval_entropy_vs_mi": emi.run_selection(m, R, k, names, split,
                                                    eligible),
            "eval_greedy_all": ega.run_selectors(R, k, names, eligible,
                                                 split)[m],
            "eval_benchpress": ebp.run_selectors(R, k, names, eligible,
                                                 split, False)[m],
            "eval_logit": el.run_selectors(R, k, names, eligible, split,
                                           [m])[m],
        }
        for script, res in got.items():
            _same_result(res, ref[m])


def test_method_lists():
    assert emi.METHODS == ["entropy", "mi", "random", "trace", "facloc"]
    assert ega.METHODS == ["entropy", "random", "trace", "facloc"]
    assert ega.METHOD_PREFIX["facloc"] == "facloc_"
    assert ebp.CV_METHODS == {"entropy": "", "random": "rand_",
                              "trace": "trace_", "facloc": "facloc_"}
    assert ebp.EMI_METHODS == emi.METHODS
    assert el.CV_METHODS == ebp.CV_METHODS and el.EMI_METHODS == emi.METHODS
    # the original pairs come first
    assert ega.PAIRS[0] == ("entropy", "random")
    assert {("facloc", "random"), ("entropy", "facloc"),
            ("trace", "facloc")} <= set(ega.PAIRS)
    assert ("mi", "facloc") in ebp.ADDED_EMI_PAIRS


# ── appending trace and facloc changes no original output ────────

def test_entropy_vs_mi_unchanged(toy, monkeypatch):
    new, meta_new, _ = emi.run_dataset("toy", 1, 10, 15)
    monkeypatch.setattr(emi, "METHODS", ["entropy", "mi", "random"])
    old, meta_old, _ = emi.run_dataset("toy", 1, 10, 15)
    for m in old:
        for key, arr in old[m].items():
            assert _equal(arr, new[m][key]), (m, key)
    for key in meta_old:
        assert _equal(meta_old[key], meta_new[key]), key
    assert np.isfinite(new["facloc"]["r2"][:, 0]).all()


def test_greedy_cv_unchanged(toy, monkeypatch):
    new, logs_new, _ = ega.run_dataset("toy", 1, [0.1, 0.9], 15)
    rows_new = ega.paired_rows("toy", new)
    full = ega.run_selectors
    monkeypatch.setattr(ega, "METHODS", ["entropy", "random"])
    monkeypatch.setattr(ega, "PAIRS", ega.PAIRS[:1])
    monkeypatch.setattr(ega, "run_selectors", lambda *a: {
        m: r for m, r in full(*a).items() if m in ("entropy", "random")})
    old, logs_old, _ = ega.run_dataset("toy", 1, [0.1, 0.9], 15)
    rows_old = ega.paired_rows("toy", old)
    for p in old:
        assert set(old[p]) < set(new[p])
        for key, arr in old[p].items():
            assert _equal(arr, new[p][key]), (p, key)
        assert {k[len("facloc_"):] for k in new[p] if k.startswith(
            "facloc_")} == {"entropy", "resid_var", "r2", "rmse", "sel",
                            "index_ties", "tie_breaks"}
    assert logs_new[:len(logs_old)] == logs_old
    assert {e["method"] for e in logs_new[len(logs_old):]} == {"trace",
                                                                "facloc"}
    assert len(rows_new) > len(rows_old)
    assert all(_rows_equal(a, b) for a, b in zip(rows_old, rows_new))


def test_benchpress_results_unchanged(toy, monkeypatch):
    new, _ = ebp.run_dataset("toy", 5, 10, 1)
    rows_new = ebp.paired_rows("toy", new, 5)
    monkeypatch.setattr(ebp, "CV_METHODS",
                        {"entropy": "", "random": "rand_"})
    monkeypatch.setattr(ebp, "ADDED_CV_PAIRS", [])
    monkeypatch.setattr(ebp, "ADDED_EMI_PAIRS", [])
    old, _ = ebp.run_dataset("toy", 5, 10, 1)
    rows_old = ebp.paired_rows("toy", old, 5)
    for p in old:
        for key, arr in old[p]["meta"].items():
            assert _equal(arr, new[p]["meta"][key]), (p, key)
        for m, ev in old[p]["evals"].items():
            for key, arr in ev.items():
                assert _equal(arr, new[p]["evals"][m][key]), (p, m, key)
        assert {"trace", "facloc"} < set(new[p]["evals"])
    assert len(rows_new) > len(rows_old)
    assert all(_rows_equal(a, b) for a, b in zip(rows_old, rows_new))
    assert all({"trace", "facloc"} & {r["method_a"], r["method_b"]}
               for r in rows_new[len(rows_old):])


def test_logit_unchanged(toy, monkeypatch):
    cv_new, _ = el.run_greedy_cv("toy", 1, {})
    emi_new, meta_new, _ = el.run_entropy_vs_mi("toy", 1, {})
    monkeypatch.setattr(el, "CV_METHODS",
                        {"entropy": "", "random": "rand_"})
    monkeypatch.setattr(el, "EMI_METHODS", ["entropy", "mi", "random"])
    cv_old, _ = el.run_greedy_cv("toy", 1, {})
    emi_old, meta_old, _ = el.run_entropy_vs_mi("toy", 1, {})
    for p in cv_old:
        assert set(cv_old[p]) < set(cv_new[p])
        for key, arr in cv_old[p].items():
            assert _equal(arr, cv_new[p][key]), (p, key)
    for m in emi_old:
        for key, arr in emi_old[m].items():
            assert _equal(arr, emi_new[m][key]), (m, key)
    for key in meta_old:
        assert _equal(meta_old[key], meta_new[key]), key
    assert "facloc_r2" in cv_new[0.1] and "facloc" in emi_new
