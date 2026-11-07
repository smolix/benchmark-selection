"""Tests of eval_tabimpute's scoring and context (CPU only, no torch, no
download; the TabImpute model is replaced by stub imputers; < 10 s).

The scoring of impute_tabimpute is aligned with
eval_entropy_vs_mi.impute_and_evaluate (protocol_changes.md H8): same
cells (validation models without an observed selected benchmark are not
scored), same SS_tot floor and NaN rule, so a stub that returns the
Gaussian conditional mean reproduces impute_and_evaluate's R² exactly.
The context subsample is split-seeded (common random numbers, tag
CONTEXT_SEED_OFFSET = 20, registered in cv_split.SEED_TAGS), and the
prediction clip is counted.  No-leakage checks: the imputer sees training
rows as context and only the observed selected cells of validation rows.
Facility location (2026-10-19, in the default method set): its selections
are baseline_select.facility_location_select's, and --merge-from
assembles the logs of a method set from runs of other sets exactly as one
run of the set writes them (the way to add facloc to the logs of the
default and extra runs from before facloc, whose extra set hub, pca,
trace is unchanged, or to convert the logs of the interim sets of
2026-10-19).
"""

import json
import pathlib
import re

import numpy as np
import pytest

import cv_split
import eval_entropy_vs_mi as emi
import eval_tabimpute as E
from baseline_select import facility_location_select
from cv_split import (ENTROPY_RF_SEED_OFFSET, RANDOM_SEED_OFFSET, cv_splits,
                      eligible_mask, split_seed)

CODE = pathlib.Path(__file__).resolve().parent


# ── stubs and synthetic data ─────────────────────────────────────

class GaussianStub:
    """Fills each row's missing entries with the Gaussian conditional mean
    given its observed entries (impute_and_evaluate's predictor: z-scores
    clipped to ±10, ridge 0.01), rows without observations with the mean.
    Records every batch it is given."""

    def __init__(self, R, cm, cs):
        self.R, self.cm, self.cs = R, cm, cs
        self.batches = []

    def impute(self, X):
        self.batches.append(X.copy())
        out = X.copy()
        for i in range(X.shape[0]):
            o = np.flatnonzero(~np.isnan(X[i]))
            m = np.flatnonzero(np.isnan(X[i]))
            if m.size == 0:
                continue
            if o.size == 0:
                out[i, m] = self.cm[m]
                continue
            z = np.clip((X[i, o] - self.cm[o]) / self.cs[o], -10, 10)
            W = np.linalg.solve(self.R[np.ix_(o, o)] + 0.01 * np.eye(o.size),
                                self.R[np.ix_(m, o)].T).T
            out[i, m] = self.cm[m] + self.cs[m] * (W @ z)
        return out


class ConstStub:
    """Fills every missing entry with fill[j] (a (N,) array)."""

    def __init__(self, fill):
        self.fill = np.asarray(fill, dtype=float)
        self.calls = 0

    def impute(self, X):
        self.calls += 1
        return np.where(np.isnan(X), self.fill[None, :], X)


class NoCallStub:
    def impute(self, X):
        raise AssertionError("TabImpute must not be called")


def _corr(N, seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N, 3))
    S = W @ W.T + np.diag(rng.uniform(0.3, 1.0, N))
    d = 1 / np.sqrt(np.diag(S))
    return S * np.outer(d, d)


def _split_data(M_tr=30, M_te=9, N=8, seed=0, frac=0.6):
    rng = np.random.default_rng(seed)
    B_tr = rng.uniform(0.1, 0.9, (M_tr, N))
    B_te = rng.uniform(0.1, 0.9, (M_te, N))
    O_tr = (rng.random((M_tr, N)) < 0.8).astype(float)
    O_te = (rng.random((M_te, N)) < frac).astype(float)
    cm = rng.normal(size=N) * 0.05 + 0.5
    cs = rng.uniform(0.1, 0.3, N)
    return B_tr * O_tr, O_tr, B_te * O_te, O_te, cm, cs


def _ref_cells(O, sel, N):
    A = set(sel)
    n = 0
    for i in range(O.shape[0]):
        oa = [j for j in sel if O[i, j] == 1]
        ei = [j for j in range(N) if j not in A and O[i, j] == 1]
        if oa and ei:
            n += len(ei)
    return n


# ── scoring: the cells and rules of impute_and_evaluate ──────────

def test_scored_cells_are_impute_and_evaluates():
    rng = np.random.default_rng(1)
    N = 8
    for trial in range(200):
        O = (rng.random((int(rng.integers(1, 7)), N))
             < rng.choice([0.3, 0.6, 1.0])).astype(float)
        sel = [int(j) for j in rng.permutation(N)[:int(rng.integers(1, N))]]
        mask = E.scored_cells(O, sel, N)
        assert int(mask.sum()) == _ref_cells(O, sel, N)
        assert not mask[:, sel].any()
        assert np.all(O[mask] == 1)


def test_gaussian_stub_reproduces_impute_and_evaluate():
    """With the Gaussian conditional mean as 'TabImpute', the TabImpute
    pipeline's R² / RMSE / cell count equal impute_and_evaluate's on sparse
    data (models without an observed selected benchmark included)."""
    N = 8
    R = _corr(N, 2)
    n_skipped = 0
    for seed in range(25):
        B_tr, O_tr, B_te, O_te, cm, cs = _split_data(seed=seed, frac=0.5)
        rng = np.random.default_rng(seed)
        for _ in range(4):
            sel = [int(j) for j in rng.permutation(N)[:int(rng.integers(1, 5))]]
            n_skipped += int(np.sum(~(O_te[:, sel] == 1).any(axis=1)))
            stub = GaussianStub(R, cm, cs)
            got = E.impute_tabimpute(stub, B_tr, O_tr, B_te, O_te, sel, cm,
                                     cs, N)
            ref = emi.impute_and_evaluate(B_te, O_te, sel, R, cm, cs, N)
            assert got["n_eval"] == _ref_cells(O_te, sel, N)
            if np.isnan(ref["r2"]):
                assert np.isnan(got["r2"]) and got["n_eval"] == 0
                assert not stub.batches          # no TabImpute call
                continue
            assert got["r2"] == pytest.approx(ref["r2"], rel=1e-10,
                                              abs=1e-12)
            assert got["rmse"] == pytest.approx(ref["rmse"], rel=1e-10)
            assert got["n_clipped"] == 0
    assert n_skipped > 20                        # the skip rule is exercised


def test_models_without_selected_benchmark_are_not_scored():
    """Validation model 0 observes no selected benchmark: its (wild)
    prediction does not enter R² (it did before 2026-10-18), but its row
    is still part of TabImpute's input."""
    N = 6
    B_tr, O_tr, B_te, O_te, cm, cs = _split_data(N=N, seed=3, frac=1.0)
    O_te[0, :2] = 0
    sel = [0, 1]
    B_te = B_te * O_te
    fill = cm.copy()
    good = E.impute_tabimpute(ConstStub(fill), B_tr, O_tr, B_te, O_te, sel,
                              cm, cs, N)

    class WildRow0(ConstStub):
        def impute(self, X):
            out = super().impute(X)
            out[-B_te.shape[0], 2:] = 1e3        # validation row 0
            return out

    stub = WildRow0(fill)
    wild = E.impute_tabimpute(stub, B_tr, O_tr, B_te, O_te, sel, cm, cs, N)
    assert wild["r2"] == good["r2"] and wild["n_eval"] == good["n_eval"]
    assert good["n_eval"] == (B_te.shape[0] - 1) * (N - 2)
    assert wild["n_clipped"] == 0 and stub.calls == 1


def test_no_cell_gives_nan_without_a_call():
    N = 5
    B_tr, O_tr, B_te, O_te, cm, cs = _split_data(N=N, seed=4, frac=1.0)
    O_te[:, 0] = 0                               # nobody observes the pick
    got = E.impute_tabimpute(NoCallStub(), B_tr, O_tr, B_te * O_te, O_te,
                             [0], cm, cs, N)
    assert np.isnan(got["r2"]) and np.isnan(got["rmse"])
    assert got["n_eval"] == 0 and got["n_clipped"] == 0


def test_ss_tot_floor_not_nan():
    """Truths at the training mean (SS_tot = 0): R² = 1 − SS_res / 1e-12,
    as impute_and_evaluate (NaN before 2026-10-18)."""
    N = 4
    B_tr, O_tr, _, _, cm, cs = _split_data(N=N, seed=5)
    O_te = np.ones((3, N))
    B_te = np.tile(cm, (3, 1))
    R = _corr(N, 5)
    exact = E.impute_tabimpute(ConstStub(cm), B_tr, O_tr, B_te, O_te, [0],
                               cm, cs, N)
    ref = emi.impute_and_evaluate(B_te, O_te, [0], R, cm, cs, N)
    assert exact["r2"] == ref["r2"] == 1.0
    off = E.impute_tabimpute(ConstStub(cm + 1e-3 * cs), B_tr, O_tr, B_te,
                             O_te, [0], cm, cs, N)
    assert np.isfinite(off["r2"]) and off["r2"] < -1e5
    assert off["r2"] == pytest.approx(
        1.0 - off["rmse"] ** 2 * off["n_eval"] / E.SS_TOT_FLOOR, rel=1e-6)


def test_prediction_clip_is_counted():
    """TabImpute's predictions are clipped to the valid score range
    (rule D2), the +-10 SD clip of before 2026-10-18 is gone: a prediction
    25 SDs out but inside the range is kept, one beyond the range is
    clipped to its edge (in standardized units)."""
    N = 6
    B_tr, O_tr, B_te, O_te, cm, cs = _split_data(N=N, seed=6, frac=1.0)
    fill = cm.copy()
    fill[3] = cm[3] + 25 * cs[3]                 # z = +25, inside the range
    fill[4] = cm[4] - 11 * cs[4]                 # z = -11, below lo
    fill[5] = cm[5] + 9 * cs[5]                  # z = 9, above hi
    lo = np.full(N, -np.inf)
    hi = np.full(N, np.inf)
    lo[4] = cm[4] - 3 * cs[4]
    hi[5] = cm[5] + 2 * cs[5]
    sel = [0, 1]
    got = E.impute_tabimpute(ConstStub(fill), B_tr, O_tr, B_te, O_te, sel,
                             cm, cs, N, bounds=(lo, hi))
    assert got["n_clipped"] == 2 * B_te.shape[0]
    z_true = np.clip((B_te[:, 2:] - cm[2:]) / cs[2:], -10, 10)
    z_pred = np.array([0.0, 25.0, -3.0, 2.0])    # after the clip
    ss_res = np.sum((z_true - z_pred) ** 2)
    assert got["r2"] == pytest.approx(1 - ss_res / np.sum(z_true ** 2),
                                      rel=1e-12)
    # no bounds: nothing is clipped
    raw = E.impute_tabimpute(ConstStub(fill), B_tr, O_tr, B_te, O_te, sel,
                             cm, cs, N)
    assert raw["n_clipped"] == 0
    assert not hasattr(E, "PRED_CLIP")


# ── context: split-seeded common random numbers, no leakage ──────

def test_context_index_common_random_numbers():
    splits = list(cv_splits(60, 5, 2, 0.1, E.SEED))
    s = splits[3]
    assert E.context_index(40, s, max_context=40) is None
    idx = E.context_index(54, s, max_context=20)
    assert idx.shape == (20,) and len(set(idx.tolist())) == 20
    assert idx.min() >= 0 and idx.max() < 54
    seed = split_seed(E.SEED + E.CONTEXT_SEED_OFFSET, s.rep, s.fold,
                      s.holdout_frac)
    assert np.array_equal(idx, np.random.default_rng(seed).choice(
        54, 20, replace=False))
    assert np.array_equal(idx, E.context_index(54, s, max_context=20))
    others = [E.context_index(54, t, max_context=20) for t in splits
              if t.split_id != s.split_id]
    assert all(not np.array_equal(idx, o) for o in others)
    assert not np.array_equal(idx, np.random.default_rng(42).choice(
        54, 20, replace=False))                  # not the old fixed draw
    assert E.context_index(E.MAX_CONTEXT, s) is None
    assert len(E.context_index(E.MAX_CONTEXT + 1, s)) == E.MAX_CONTEXT


def test_context_and_test_rows_no_leakage():
    """The imputer sees the context rows ctx_idx of the training matrix and
    validation rows that reveal only their observed selected cells; a
    context above the limit is refused without ctx_idx."""
    N = 6
    B_tr, O_tr, B_te, O_te, cm, cs = _split_data(M_tr=30, N=N, seed=7)
    R = _corr(N, 7)
    sel = [1, 4]
    ctx = np.array([5, 0, 17, 3, 29, 11])
    stub = GaussianStub(R, cm, cs)
    E.impute_tabimpute(stub, B_tr, O_tr, B_te, O_te, sel, cm, cs, N,
                       ctx_idx=ctx)
    X = stub.batches[0]
    X_ctx = np.where(O_tr == 1, B_tr, np.nan)[ctx]
    assert np.array_equal(np.isnan(X[:len(ctx)]), np.isnan(X_ctx))
    assert np.array_equal(np.nan_to_num(X[:len(ctx)]), np.nan_to_num(X_ctx))
    X_te = np.vstack([b[len(ctx):] for b in stub.batches])
    shown = ~np.isnan(X_te)
    allowed = np.zeros_like(shown)
    allowed[:, sel] = O_te[:, sel] == 1
    assert np.array_equal(shown, allowed)
    assert np.array_equal(X_te[shown], B_te[shown])
    big = np.zeros((E.MAX_CONTEXT + 1, N))
    with pytest.raises(ValueError):
        E.impute_tabimpute(stub, big, np.ones_like(big), B_te, O_te, sel,
                           cm, cs, N)


def test_context_tag_registered_and_unique():
    """Tag 20 is eval_tabimpute's, listed in cv_split.SEED_TAGS; every
    seed-tag offset of code/ (*_SEED_OFFSET = n assignments, SEED + n
    seeds) is listed there, and no other module uses 20."""
    assert E.CONTEXT_SEED_OFFSET == 20
    assert E.CONTEXT_SEED_OFFSET not in (RANDOM_SEED_OFFSET,
                                         ENTROPY_RF_SEED_OFFSET)
    assert "eval_tabimpute" in cv_split.SEED_TAGS[20]
    assert cv_split.SEED_TAGS[7].startswith("RANDOM_SEED_OFFSET")
    assert cv_split.SEED_TAGS[11].startswith("ENTROPY_RF_SEED_OFFSET")
    pats = (re.compile(r"^\s*\w*SEED_OFFSET\s*=\s*(\d+)", re.M),
            re.compile(r"\bSEED\s*\+\s*(\d+)\b"))
    used = {}
    for path in sorted(CODE.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        for pat in pats:
            for m in pat.finditer(text):
                used.setdefault(int(m.group(1)), set()).add(path.name)
    assert {7, 11, 20} <= set(used)
    missing = {t: sorted(f) for t, f in used.items()
               if t not in cv_split.SEED_TAGS}
    assert not missing, f"seed tags not in cv_split.SEED_TAGS: {missing}"
    assert used[20] == {"eval_tabimpute.py"}, used[20]


# ── end to end with a stub imputer ───────────────────────────────

def _write_dataset(tmp_path, name="synth", M=40, N=7, seed=11):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(M, 2))
    B = 0.5 + 0.1 * (z @ rng.normal(size=(2, N))) \
        + 0.03 * rng.normal(size=(M, N))
    O = (rng.random((M, N)) < 0.85).astype(float)
    O[:, 0] = 1
    np.savez(tmp_path / f"{name}.matrix.npz", B=B * O, O=O,
             model_names=np.array([f"m{i}" for i in range(M)]),
             benchmark_names=np.array([f"b{j}" for j in range(N)]))
    (tmp_path / f"{name}.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [None, None]}}))
    return B * O, O


def test_run_dataset_logs_alignment_and_context(tmp_path, monkeypatch):
    B, O = _write_dataset(tmp_path)
    monkeypatch.setattr(E, "DATA", tmp_path)
    monkeypatch.setattr(E, "LOGS", tmp_path)
    # a context limit below n_train: every split is subsampled
    monkeypatch.setattr(E, "MAX_ROWS", 30)
    monkeypatch.setattr(E, "TEST_BATCH", 5)
    monkeypatch.setattr(E, "MAX_CONTEXT", 25)

    stub = ConstStub(np.full(B.shape[1], 0.5))
    methods = ["entropy", "random", "trace"]
    metrics, out, names = E.run_dataset("synth", stub, methods, n_repeats=1,
                                        n_folds=4, k_max=3)
    S = len(out["split_id"])
    assert S == 4
    assert out["context_subsampled"].all()
    assert np.all(out["n_context"] == 25)
    splits = list(cv_splits(B.shape[0], 4, 1, E.HOLDOUT_FRAC, E.SEED))
    for i, s in enumerate(splits):
        O_te = O[s.val_idx]
        for m in methods:
            for k in range(1, 4):
                sel = [int(j) for j in out[f"{m}_selected"][i, :k]]
                assert metrics[m]["n_eval"][i, k - 1] == _ref_cells(
                    O_te, sel, B.shape[1])
    for m in methods:
        assert metrics[m]["n_clipped"].shape == (S, 3)
        assert metrics[m]["n_clipped"].dtype == np.int64
    E.save_results("synth", methods, metrics, out, names, 1, 4, 3,
                   {"device": "cpu"})
    d = np.load(tmp_path / "tabimpute_synth.npz")
    for key in ("entropy_n_clipped", "n_context", "context_subsampled",
                "context_seed_offset", "pred_clip", "ss_tot_floor"):
        assert key in d.files, key
    assert int(d["context_seed_offset"]) == 20
    assert float(d["ss_tot_floor"]) == 1e-12


# ── facility location and combined runs ──────────────────────────

class BatchMeanStub:
    """Fills every missing entry with its column's mean over the batch
    (context and test rows; 0.5 for an empty column): deterministic, and
    it depends on the revealed selected cells."""

    def __init__(self):
        self.calls = 0

    def impute(self, X):
        self.calls += 1
        obs = ~np.isnan(X)
        cnt = obs.sum(axis=0)
        mean = np.where(cnt > 0, np.nansum(X, axis=0) / np.maximum(cnt, 1),
                        0.5)
        return np.where(obs, X, mean[None, :])


def test_default_methods_include_facloc(tmp_path, monkeypatch):
    """facloc joins the default set after the original three (key order
    kept); hub, pca and trace stay the extra job's set (run_tabimpute.sh),
    so every array of the earlier logs is unchanged."""
    monkeypatch.delenv("BENCHSELECT_METHODS", raising=False)
    monkeypatch.delenv("BENCHSELECT_DATASETS", raising=False)
    assert E.DEFAULT_METHODS == ["entropy", "mi", "random", "facloc"]
    assert set(E.DEFAULT_METHODS) <= set(E.ALL_METHODS)
    assert {"hub", "pca", "trace"} <= set(E.ALL_METHODS) - set(
        E.DEFAULT_METHODS)
    assert "BENCHSELECT_METHODS=hub,pca,trace " in (
        CODE / "run_tabimpute.sh").read_text()
    assert all(m in E.METHOD_COLOR and m in E.METHOD_LABEL
               and m in E.METHOD_STYLE for m in E.ALL_METHODS)
    _write_dataset(tmp_path)
    monkeypatch.setattr(E, "DATA", tmp_path)
    _, methods, tag = E.parse_args(["--datasets", "synth"])
    assert methods == E.DEFAULT_METHODS and tag == ""
    _, methods, tag = E.parse_args(["--datasets", "synth", "--methods",
                                    "facloc"])
    assert methods == ["facloc"] and tag == E.EXTRA_TAG


def test_facloc_selection_is_facility_location_select(tmp_path,
                                                       monkeypatch):
    B, O = _write_dataset(tmp_path)
    monkeypatch.setattr(E, "DATA", tmp_path)
    _, out, names = E.run_dataset("synth", BatchMeanStub(),
                                  ["facloc", "entropy"], n_repeats=1,
                                  n_folds=4, k_max=5)
    for i, s in enumerate(cv_splits(B.shape[0], 4, 1, E.HOLDOUT_FRAC,
                                    E.SEED)):
        R, _, _ = E.estimate_corr(B[s.train_idx], O[s.train_idx])
        el = eligible_mask(O[s.train_idx])
        sel = list(facility_location_select(
            R, 5, benchmark_names=names, eligible=el)["selected"])
        assert out["facloc_selected"][i].tolist() == (
            sel + [-1] * (5 - len(sel)))


def _run_main(monkeypatch, root, argv):
    """E.main(argv) with BatchMeanStub as TabImpute, outputs in root."""
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(E, "LOGS", root)
    monkeypatch.setattr(E, "OUT", root)
    prov = {"ckpt_repo": E.CKPT_REPO, "ckpt_file": E.CKPT_FILE,
            "ckpt_revision": E.CKPT_REVISION, "ckpt_sha256": "stub",
            "device": "cpu", "torch_version": "0", "tabimpute_version": "0",
            "torch_deterministic": True}
    monkeypatch.setattr(E, "load_imputer",
                        lambda *a, **kw: (BatchMeanStub(), dict(prov)))
    E.main(["--datasets", "synth", "--folds", "4", "--kmax", "4"] + argv)


def _same_log(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    assert sorted(a.files) == sorted(b.files)
    for key in a.files:
        x, y = a[key], b[key]
        assert x.dtype == y.dtype and x.shape == y.shape, key
        assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), key


def test_merge_from_equals_a_single_run(tmp_path, monkeypatch):
    """The logs of the default (entropy, mi, random) and extra (hub, pca,
    trace) runs from before facloc plus a facloc-only run, combined with
    --merge-from, are the logs and paired CSVs of one run of the current
    default set and one of the extra set; the extra log and every array of
    the old default log are unchanged by facloc.  The logs of a run with
    the interim sets of 2026-10-19 (default entropy, mi, random, trace,
    facloc; extra hub, pca) convert to the same logs."""
    monkeypatch.delenv("BENCHSELECT_METHODS", raising=False)
    monkeypatch.delenv("BENCHSELECT_DATASETS", raising=False)
    data = tmp_path / "data"
    data.mkdir()
    _write_dataset(data)
    monkeypatch.setattr(E, "DATA", data)
    old, new, ref, out, mid, out2 = (
        tmp_path / d for d in ("old", "new", "ref", "out", "mid", "out2"))
    extra = ["--methods", "hub", "pca", "trace"]
    with monkeypatch.context() as mp:
        mp.setattr(E, "DEFAULT_METHODS", ["entropy", "mi", "random"])
        _run_main(mp, old, [])
    _run_main(monkeypatch, old, extra)
    _run_main(monkeypatch, new, ["--methods", "facloc"])
    _run_main(monkeypatch, ref, [])
    _run_main(monkeypatch, ref, extra)
    srcs = [str(old / "tabimpute_{ds}.npz"),
            str(old / "tabimpute_extra_{ds}.npz"),
            str(new / "tabimpute_extra_{ds}.npz")]
    _run_main(monkeypatch, out, ["--merge-from"] + srcs)
    _run_main(monkeypatch, out, extra + ["--merge-from"] + srcs)
    names = ("tabimpute_synth.npz", "tabimpute_extra_synth.npz")
    csvs = ("tabimpute_synth_paired.csv", "tabimpute_extra_synth_paired.csv")
    for name in names:
        _same_log(ref / name, out / name)
    for name in csvs:
        assert (ref / name).read_text() == (out / name).read_text()
    d = np.load(out / "tabimpute_synth.npz")
    assert [str(m) for m in d["methods"]] == E.DEFAULT_METHODS
    # the old default log's arrays are the new one's, bit for bit
    o = np.load(old / "tabimpute_synth.npz")
    for key in o.files:
        if key != "methods":
            assert np.array_equal(o[key], d[key],
                                  equal_nan=o[key].dtype.kind == "f"), key
    # the extra job is the one before facloc: same log, same paired CSV
    _same_log(old / names[1], ref / names[1])
    assert (old / csvs[1]).read_text() == (ref / csvs[1]).read_text()
    # a run of the interim sets converts to the same two logs
    with monkeypatch.context() as mp:
        mp.setattr(E, "DEFAULT_METHODS",
                   ["entropy", "mi", "random", "trace", "facloc"])
        _run_main(mp, mid, [])
    _run_main(monkeypatch, mid, ["--methods", "hub", "pca"])
    _run_main(monkeypatch, out2,
              ["--merge-from", str(mid / "tabimpute_{ds}.npz")])
    _run_main(monkeypatch, out2,
              extra + ["--merge-from", str(mid / "tabimpute_extra_{ds}.npz"),
                       str(mid / "tabimpute_{ds}.npz")])
    for name in names:
        _same_log(ref / name, out2 / name)
    for name in csvs:
        assert (ref / name).read_text() == (out2 / name).read_text()
    # another --kmax than the logs', or a method no log has, is refused
    with pytest.raises(SystemExit):
        E.main(["--datasets", "synth", "--folds", "4", "--kmax", "3",
                "--merge-from"] + srcs)
    with pytest.raises(SystemExit):
        E.main(["--datasets", "synth", "--folds", "4", "--kmax", "4",
                "--merge-from", srcs[0]])
