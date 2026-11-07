"""Tests of the protocol rules D1 (imputation targets) and D2 (valid-range
clip of every imputer's predictions): score_bounds.py, cv_split.target_mask
and their one implementation in eval_entropy_vs_mi.impute_and_evaluate, plus
static checks that every scorer of code/ passes both rules.

Fast (< 10 s), synthetic data except the audit checks, which read the
data/ matrices and data/score_bounds.json when present.
"""

import json
import pathlib
import re

import numpy as np
import pytest

import cv_split
import score_bounds as sb
import eval_entropy_vs_mi as emi
import eval_logit as el
from path_config import data_dir

CODE = pathlib.Path(__file__).resolve().parent
DATA = data_dir()


def _corr(N, seed):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(N, 2))
    S = F @ F.T + np.diag(rng.uniform(0.2, 0.6, N))
    d = 1 / np.sqrt(np.diag(S))
    return S * np.outer(d, d)


def _data(M, N, seed, miss=0.2):
    rng = np.random.default_rng(seed)
    B = rng.uniform(0.2, 0.8, size=(M, N))
    O = (rng.random((M, N)) >= miss).astype(float)
    O[:, 0] = 1.0
    cm = rng.uniform(0.4, 0.6, N)
    cs = rng.uniform(0.05, 0.2, N)
    return B * O, O, cm, cs


# ── score_bounds ─────────────────────────────────────────────────

def test_z_limits_and_clip_z():
    lo, hi = np.array([0.0, -np.inf, 1.0]), np.array([1.0, np.inf, 10.0])
    cm, cs = np.array([0.5, 3.0, 5.0]), np.array([0.1, 2.0, 1.0])
    z_lo, z_hi = sb.z_limits((lo, hi), cm, cs)
    np.testing.assert_allclose(z_lo, [-5.0, -np.inf, -4.0])
    np.testing.assert_allclose(z_hi, [5.0, np.inf, 5.0])
    z = np.array([[6.0, 100.0, -5.0], [1.0, -1e9, 2.0]])
    out, n = sb.clip_z(z, z_lo, z_hi)
    assert n == 2
    np.testing.assert_array_equal(out, [[5.0, 100.0, -4.0],
                                        [1.0, -1e9, 2.0]])
    inside = np.array([[1.0, 2.0, 3.0]])
    same, n0 = sb.clip_z(inside, z_lo, z_hi)
    assert n0 == 0 and np.array_equal(same, inside)
    # no bounds or the test-only override: nothing is clipped
    assert np.all(np.isinf(sb.z_limits(None, cm, cs)[0]))


def test_pred_clip_override(monkeypatch):
    monkeypatch.setattr(sb, "PRED_CLIP", False)
    z_lo, z_hi = sb.z_limits((np.zeros(2), np.ones(2)), np.zeros(2),
                             np.ones(2))
    assert np.all(z_lo == -np.inf) and np.all(z_hi == np.inf)
    x, n = sb.clip_raw(np.array([5.0, -5.0]), 0.0, 1.0)
    assert n == 0 and x.tolist() == [5.0, -5.0]


def test_meta_declaration_and_unknown_dataset(tmp_path):
    B = np.array([[0.2, 3.0], [0.4, 7.0]])
    np.savez(tmp_path / "toy.matrix.npz", B=B, O=np.ones_like(B),
             benchmark_names=np.array(["a", "b"]))
    with pytest.raises(KeyError):
        sb.dataset_bounds("toy", data=tmp_path)
    (tmp_path / "toy.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [0, 1], "columns": {"b": [1, None]}}}))
    bd = sb.dataset_bounds("toy", data=tmp_path)
    assert bd.lo.tolist() == [0.0, 1.0] and bd.hi.tolist() == [1.0, np.inf]
    assert "score_bounds.columns" in bd.source[1]
    # an observed score outside its declared range is an error
    (tmp_path / "toy.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [0, 1]}}))
    with pytest.raises(ValueError):
        sb.dataset_bounds("toy", data=tmp_path)


def test_rules_of_the_data_release():
    """Every data/ matrix has a rule; the documented special scales."""
    if not (DATA / "mmlu.matrix.npz").is_file():
        pytest.skip("no data/")
    for ds in sb.DATASETS:
        if not (DATA / f"{ds}.matrix.npz").is_file():
            continue
        bd = sb.dataset_bounds(ds)                 # checks the scores too
        assert len(bd.lo) == len(bd.hi) == len(bd.source)
        assert np.all(bd.lo < bd.hi)
    names = [p.name[:-len(".matrix.npz")] for p in DATA.glob("*.matrix.npz")]
    assert set(names) <= set(sb.RULES) | set(sb.ALIASES), names
    if (DATA / "merged.matrix.npz").is_file():
        d = np.load(DATA / "merged.matrix.npz", allow_pickle=True)
        bd = sb.dataset_bounds("merged")
        col = {str(n): j for j, n in enumerate(d["benchmark_names"])}
        assert (bd.lo[col["mt_bench/math"]], bd.hi[col["mt_bench/math"]]) \
            == (1.0, 10.0)
        assert bd.hi[col["arena_hard/score"]] == 100.0
        assert bd.lo[col["wildbench/Math"]] == -80.0
    if (DATA / "benchpress.matrix.npz").is_file():
        d = np.load(DATA / "benchpress.matrix.npz", allow_pickle=True)
        bd = sb.dataset_bounds("benchpress")
        col = {str(n): j for j, n in enumerate(d["benchmark_names"])}
        for n in ("Chatbot Arena Elo", "Codeforces Rating", "GDP-Val AA"):
            assert np.isinf(bd.lo[col[n]]) and np.isinf(bd.hi[col[n]])
    if (DATA / "mteb.matrix.npz").is_file():
        d = np.load(DATA / "mteb.matrix.npz", allow_pickle=True)
        bd = sb.dataset_bounds("mteb")
        col = {str(n): j for j, n in enumerate(d["benchmark_names"])}
        assert bd.lo[col["STS12"]] == -1.0 and bd.lo[col["ArguAna"]] == 0.0


def test_audit_file_matches_the_rules():
    path = DATA / sb.AUDIT_FILE
    if not path.is_file():
        pytest.skip(f"no {path}")
    obj = json.loads(path.read_text())
    # the public release ships the audit without benchpress and epoch, whose
    # matrices are fetched or rebuilt locally (DATA_LICENSES.md)
    have = [ds for ds in sb.DATASETS if (DATA / f"{ds}.matrix.npz").is_file()
            and ds in obj["datasets"]]
    assert {ds: obj["datasets"][ds] for ds in have} == sb.audit(have)["datasets"]


# ── cv_split.target_mask ─────────────────────────────────────────

def test_target_mask_threshold():
    O = np.zeros((5, 4))
    O[:1, 1] = 1
    O[:2, 2] = 1
    O[:, 3] = 1
    assert cv_split.target_mask(O).tolist() == [False, False, True, True]
    assert cv_split.MIN_TARGET_OBS == 2
    assert not np.any(cv_split.eligible_mask(O) & ~cv_split.target_mask(O))


# ── impute_and_evaluate: one implementation of D1 and D2 ─────────

def _per_model(B, O, sel, R, cm, cs, N, target, lo, hi):
    """Per-model reference with both rules."""
    Z = np.clip(O * (B - cm) / cs, -10, 10) * O
    z_lo, z_hi = (lo - cm) / cs, (hi - cm) / cs
    res = tot = 0.0
    n = 0
    for i in range(B.shape[0]):
        a = [j for j in sel if O[i, j]]
        e = [j for j in range(N) if j not in sel and O[i, j] and target[j]]
        if not a or not e:
            continue
        W = np.linalg.solve(R[np.ix_(a, a)] + 0.01 * np.eye(len(a)),
                            R[np.ix_(e, a)].T).T
        p = np.clip(W @ Z[i, a], z_lo[e], z_hi[e])
        res += np.sum((Z[i, e] - p) ** 2)
        tot += np.sum(Z[i, e] ** 2)
        n += len(e)
    return (np.nan, 0) if n == 0 else (1 - res / max(tot, 1e-12), n)


@pytest.mark.parametrize("seed", range(6))
def test_targets_and_clip_match_reference(seed):
    N = 7
    R = _corr(N, seed)
    B, O, cm, cs = _data(12, N, seed)
    rng = np.random.default_rng(seed)
    target = rng.random(N) < 0.7
    lo = cm - rng.uniform(0.5, 2, N) * cs
    hi = cm + rng.uniform(0.5, 2, N) * cs
    if seed % 2:
        O[:] = O[0]                    # identical rows: vectorized path
    for sel in ([0], [0, 3], [1, 2, 5]):
        got = emi.impute_and_evaluate(B, O, sel, R, cm, cs, N,
                                      target=target, bounds=(lo, hi))
        ref, n = _per_model(B, O, sel, R, cm, cs, N, target, lo, hi)
        assert got["n_eval"] == n
        if np.isnan(ref):
            assert np.isnan(got["r2"])
        else:
            assert got["r2"] == pytest.approx(ref, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_rules_off_are_bit_identical(seed):
    """All-True targets and infinite (or no) bounds give the scores of the
    function without the rules bit for bit: the regression override."""
    N = 6
    R = _corr(N, seed)
    B, O, cm, cs = _data(15, N, 10 + seed, miss=0.3 if seed % 2 else 0.0)
    for sel in ([0], [0, 2], [1, 3, 4]):
        a = emi.impute_and_evaluate(B, O, sel, R, cm, cs, N)
        b = emi.impute_and_evaluate(B, O, sel, R, cm, cs, N,
                                    target=np.ones(N, bool),
                                    bounds=(np.full(N, -np.inf),
                                            np.full(N, np.inf)))
        for key in ("r2", "rmse"):
            assert (a[key] == b[key]) or (np.isnan(a[key])
                                          and np.isnan(b[key]))
        assert b["n_clipped"] == 0


def test_non_target_cells_never_scored():
    N = 5
    R = _corr(N, 1)
    B, O, cm, cs = _data(10, N, 2, miss=0.0)
    t = np.array([True, True, True, True, False])
    a = emi.impute_and_evaluate(B, O, [0], R, cm, cs, N, target=t)
    B2 = B.copy()
    B2[:, 4] = 1e6                        # a non-target's scores are moot
    b = emi.impute_and_evaluate(B2, O, [0], R, cm, cs, N, target=t)
    assert a["r2"] == b["r2"] and a["n_eval"] == 10 * 3


@pytest.mark.parametrize("loop", [False, True])
def test_logit_predictions_only_clipped_to_valid_range(monkeypatch, loop):
    """D2 is the logit pipeline's only prediction clip: a valid prediction
    45 SDs below a high-mean, low-SD column is scored as it is (no +-10
    clip, as for the Gaussian and TabImpute scorers); the test-only
    override PRED_CLIP = False restores the pre-D2 +-10 clip.  loop: a
    second validation row with another observation row (per-model path)."""
    N = 2
    R = np.array([[1.0, 0.9], [0.9, 1.0]])
    O = np.array([[1.0, 1.0], [1.0, 0.0]] if loop else [[1.0, 1.0]])
    B_logit = np.array([[-10.0, 0.0]] * len(O)) * O
    B_raw = np.array([[0.5, 0.85]] * len(O)) * O
    zeros, ones = np.zeros(N), np.ones(N)
    cm_raw, cs_raw = np.array([0.5, 0.9]), np.array([0.1, 0.02])

    def r2():
        return el.impute_and_evaluate_logit(
            B_logit, O, [0], R, zeros, ones, B_raw, cm_raw, cs_raw, ones,
            N, target=np.ones(N, bool), bounds=(zeros, ones))["r2"]

    assert el.same_observation_rows(O) is not loop
    s = el.sigmoid(0.9 / 1.01 * -10.0)        # inside [0, 1]: no D2 clip
    t, p = (0.85 - 0.9) / 0.02, (s - 0.9) / 0.02
    assert p < -40
    assert r2() == pytest.approx(1 - (t - p) ** 2 / t ** 2, rel=1e-12)
    monkeypatch.setattr(sb, "PRED_CLIP", False)
    assert r2() == pytest.approx(1 - (t + 10) ** 2 / t ** 2, rel=1e-12)


# ── static: every scorer passes both rules ───────────────────────

def _calls(src, name):
    """Source text of every call of `name` (balanced parentheses)."""
    out = []
    for m in re.finditer(rf"(?<![\w.]){name}\(|\.{name}\(", src):
        i, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_impute_and_evaluate_call_passes_both_rules():
    for path in sorted(CODE.glob("*.py")):
        # (the legacy predecessors wrap the shared function under its
        # name: test_one_implementation_of_the_rules)
        if path.name.startswith("test_") or path.name in (
                "eval_greedy_mmlu.py", "eval_greedy_merged.py"):
            continue
        src = path.read_text()
        for name in ("impute_and_evaluate", "suite_r2",
                     "impute_and_evaluate_logit"):
            for call in _calls(src, name):
                assert "target=" in call and "bounds=" in call, (
                    path.name, call)


def test_one_implementation_of_the_rules():
    """MIN_TARGET_OBS and the target rule live in cv_split, the ranges in
    score_bounds: no script defines its own threshold, its own target
    mask or its own range rule, and the legacy predecessors call the
    shared imputation."""
    for path in sorted(CODE.glob("*.py")):
        if path.name.startswith("test_") or path.name in ("cv_split.py",
                                                          "score_bounds.py"):
            continue
        src = path.read_text()
        assert not re.search(r"^MIN_TARGET_OBS\s*=", src, re.M), path.name
        assert not re.search(r"^def (target_mask|benchmark_bounds|"
                             r"dataset_bounds|z_limits|clip_z)\b", src,
                             re.M), path.name
        assert "PRED_CLIP = " not in src, path.name
    for name in ("eval_greedy_mmlu.py", "eval_greedy_merged.py"):
        if not (CODE / name).is_file():  # not shipped in the public release
            continue
        src = (CODE / name).read_text()
        assert "_shared_impute(" in src and "target_mask" in src
