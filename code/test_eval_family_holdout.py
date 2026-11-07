"""Tests for eval_family_holdout.py's selection methods (synthetic data,
< 10 s).

Covers facility location (facloc, the paper's headline method since
2026-10-19) as a method of the model-family holdout: run_selection's
facloc is baseline_select.facility_location_select (clipped similarity)
on the same Sigma and eligible mask, and on a small synthetic dataset
(run_dataset with the data path, the family labels and the score bounds
patched) every family split's facloc picks equal facility_location_select
on that split's training estimate and eligible set, the all-model
reference and the stability arrays exist for facloc, facloc's rows repeat
in every repetition (deterministic), and adding facloc leaves every array
of the other methods bit-identical (a run with the pre-facloc METHODS).
The end-to-end run of the script on a sparse toy dataset is in
test_battery_generic.py.
"""

import numpy as np
import pytest

import eval_family_holdout as efh
from baseline_select import facility_location_select
from cv_split import eligible_mask
from eval_entropy_vs_mi import estimate_corr

# METHODS before facloc was added (2026-10-19)
PRE_FACLOC_METHODS = ["entropy", "mi", "random", "hub", "pca", "trace"]


def _factor_cov(N=12, n_factors=3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, n_factors))
    return W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))


def _toy(M=48, N=9, seed=1):
    """(B, O, benchmark_names, model_names, families): factor scores in
    [0, 1], about 15% of the cells missing, one column observed for two
    models only (ineligible everywhere), four keyword families of 12."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((M, 2)) @ rng.standard_normal((2, N))
    B = 1.0 / (1.0 + np.exp(-(Z + 0.3 * rng.standard_normal((M, N)))))
    O = (rng.random((M, N)) > 0.15).astype(float)
    O[:, 0] = 1.0
    O[:, N - 1] = 0.0
    O[[3, 30], N - 1] = 1.0
    B = B * O
    fams = np.array(["llama", "qwen", "mistral", "gemma"])[np.arange(M) % 4]
    names = np.array([f"{f}-{i}" for i, f in enumerate(fams)])
    bench = np.array([f"b{j}" for j in range(N)])
    return B, O, bench, names, fams


@pytest.fixture
def toy_data(tmp_path, monkeypatch):
    B, O, bench, names, fams = _toy()
    path = tmp_path / "toy.matrix.npz"
    np.savez(path, B=B, O=O, benchmark_names=bench, model_names=names)
    monkeypatch.setattr(efh, "matrix_path", lambda ds: path)
    monkeypatch.setattr(efh, "model_family_labels",
                        lambda ds, model_names: fams.copy())
    monkeypatch.setattr(efh, "dataset_bounds", lambda *a, **kw: None)
    return B, O, bench, names, fams


def test_facloc_is_facility_location_select():
    """run_selection("facloc") = facility_location_select (clipped) on the
    same Sigma and eligible mask, picks, ties and gains alike."""
    N = 12
    Sigma = _factor_cov(N, seed=3)
    names = np.array([f"b{j}" for j in range(N)])
    el = np.ones(N, dtype=bool)
    el[[1, 6]] = False
    for eligible in (None, el):
        got = efh.run_selection("facloc", Sigma, 8, names, eligible)
        ref = facility_location_select(Sigma, 8, benchmark_names=names,
                                       eligible=eligible,
                                       similarity="clipped")
        assert got["selected"] == ref["selected"]
        assert got["tie_breaks"] == ref["tie_breaks"]
        assert got["index_ties"] == ref["index_ties"]
        np.testing.assert_array_equal(got["gains"], ref["gains"])
        assert got["similarity"] == "clipped"
    assert not set(efh.run_selection("facloc", Sigma, 10, names,
                                     el)["selected"]) & {1, 6}
    # a first-class method: logged, plotted, labelled
    assert "facloc" in efh.METHODS and "facloc" in efh.PLOT_METHODS
    assert "facloc" not in efh.STOCHASTIC
    for table in (efh.METHOD_LABEL, efh.METHOD_COLOR, efh.METHOD_STYLE):
        assert "facloc" in table
    # appended after the pre-facloc methods (their log order unchanged)
    assert efh.METHODS[:len(PRE_FACLOC_METHODS)] == PRE_FACLOC_METHODS


def test_run_dataset_facloc_rows(toy_data):
    B, O, bench, names, fams = toy_data
    res = efh.run_dataset("toy", 6, n_repeats=2, min_family=5)
    sp = res["splits"]
    F = len(res["families"])
    assert res["families"] == sorted(set(fams))
    for i, fam in enumerate(res["families"]):
        train = np.flatnonzero(fams != fam)
        R, _, _ = estimate_corr(B[train], O[train])
        el = eligible_mask(O[train])
        assert sp["eligible"][i].tolist() == el.tolist()
        ref = facility_location_select(R, 6, eligible=el)["selected"]
        assert sp["facloc_selected"][i].tolist() == efh.padded(ref, 6)
        # deterministic: repetition 1 repeats repetition 0's rows
        np.testing.assert_array_equal(sp["facloc_selected"][F + i],
                                      sp["facloc_selected"][i])
    np.testing.assert_array_equal(sp["facloc_r2_splits"][F:],
                                  sp["facloc_r2_splits"][:F])
    np.testing.assert_array_equal(res["metrics"]["facloc"]["r2"],
                                  sp["facloc_r2_splits"][:F])
    assert np.isfinite(res["metrics"]["facloc"]["r2"][:, :3]).all()
    # all-model reference and stability
    R_full, _, _ = estimate_corr(B, O)
    ref = facility_location_select(R_full, 6,
                                   eligible=eligible_mask(O))["selected"]
    assert sp["full_facloc_selected"].tolist() == efh.padded(ref, 6)
    assert set(res["stability"]["facloc"]) == {5}
    assert np.isfinite(res["stability"]["facloc"][5]).all()
    assert "full_facloc_index_ties" in sp and "facloc_tie_breaks" in sp


def test_facloc_changes_no_other_method(toy_data, monkeypatch):
    """Every array of a run with the pre-facloc METHODS equals the same
    array of the run with facloc (random's draws included)."""
    with_fl = efh.run_dataset("toy", 6, n_repeats=2, min_family=5)
    monkeypatch.setattr(efh, "METHODS", list(PRE_FACLOC_METHODS))
    without = efh.run_dataset("toy", 6, n_repeats=2, min_family=5)
    assert set(without["splits"]) < set(with_fl["splits"])
    assert {k for k in set(with_fl["splits"]) - set(without["splits"])
            if "facloc" not in k} == set()
    for key, arr in without["splits"].items():
        np.testing.assert_array_equal(arr, with_fl["splits"][key],
                                      err_msg=key)
    for m in PRE_FACLOC_METHODS:
        for key in ("r2", "rmse"):
            np.testing.assert_array_equal(without["metrics"][m][key],
                                          with_fl["metrics"][m][key])
    for m, by_k in without["stability"].items():
        for k, vals in by_k.items():
            np.testing.assert_array_equal(vals, with_fl["stability"][m][k])
