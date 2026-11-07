"""Tests of the pooled imputation R² (synthetic data, < 10 s).

eval_entropy_vs_mi.impute_and_evaluate (also used by eval_greedy_all,
eval_benchpress, eval_baselines, eval_family_holdout and the newer
scripts) and eval_logit's impute_and_evaluate_logit against a per-model
reference loop on random sparse observation patterns, including the
pattern that the old vectorized path mis-scored (protocol_changes.md H1:
every validation model observes model 0's benchmarks and some observe
more); a frozen copy of the old function, which drops those cells, and
its bit-identity with the fixed one wherever it drops none; no copy of
the old condition left in code/; the vectorized path
on identical rows; eval_logit.raw_stats against estimate_corr's
standardization (EM and pairwise branches); the training-mean prediction
of targets without training variance (no_info) in logit space, which
reproduces the raw pipeline's contribution of such cells; and that a
split's training statistics do not depend on its validation rows.
"""

import pathlib
import re

import numpy as np
import pytest

import eval_benchpress
import eval_entropy_vs_mi as emi
import eval_greedy_all
import eval_logit as el
import score_bounds
from cv_split import cv_splits

CODE = pathlib.Path(__file__).resolve().parent


def _old_impute_and_evaluate(B_test, O_test, selected, R, col_mean, col_std,
                             N):
    """The pre-2026-10-18 impute_and_evaluate, frozen verbatim (git HEAD
    eval_entropy_vs_mi; eval_greedy_all and eval_benchpress had identical
    copies): its vectorized path is keyed on validation model 0's
    observation row (protocol_changes.md H1)."""
    A = set(selected)
    M_test = B_test.shape[0]
    B_test_std = O_test * (B_test - col_mean[None, :]) / col_std[None, :]
    B_test_std = np.clip(B_test_std, -10, 10) * O_test
    RIDGE = 0.01
    obs_A = np.array([j for j in selected if O_test[0, j] == 1])
    eval_idx = np.array([j for j in range(N) if j not in A and O_test[0, j] == 1])
    all_same = (len(obs_A) > 0 and len(eval_idx) > 0 and
                np.all(O_test[:, list(selected)] == O_test[0, list(selected)]) and
                np.all(O_test[:, eval_idx] == O_test[0, eval_idx]))
    if all_same:
        R_oo = R[np.ix_(obs_A, obs_A)] + RIDGE * np.eye(len(obs_A))
        R_eo = R[np.ix_(eval_idx, obs_A)]
        W = np.linalg.solve(R_oo, R_eo.T).T
        X_obs = B_test_std[:, obs_A]
        X_true = B_test_std[:, eval_idx]
        X_pred = X_obs @ W.T
        ss_res = np.sum((X_true - X_pred) ** 2)
        ss_tot = np.sum(X_true ** 2)
        n_eval = M_test * len(eval_idx)
    else:
        ss_res = 0.0
        ss_tot = 0.0
        n_eval = 0
        for i in range(M_test):
            obs_A_i = np.array([j for j in selected if O_test[i, j] == 1])
            eval_i = np.array([j for j in range(N) if j not in A and O_test[i, j] == 1])
            if len(obs_A_i) == 0 or len(eval_i) == 0:
                continue
            R_oo = R[np.ix_(obs_A_i, obs_A_i)] + RIDGE * np.eye(len(obs_A_i))
            R_eo = R[np.ix_(eval_i, obs_A_i)]
            W = np.linalg.solve(R_oo, R_eo.T).T
            x_obs = B_test_std[i, obs_A_i]
            x_true = B_test_std[i, eval_i]
            x_pred = W @ x_obs
            ss_res += np.sum((x_true - x_pred) ** 2)
            ss_tot += np.sum(x_true ** 2)
            n_eval += len(eval_i)
    if n_eval == 0:
        return {"r2": np.nan, "rmse": np.nan}
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "rmse": np.sqrt(ss_res / n_eval)}


def _old_path_drops(O_test, selected, N):
    """Cells the old vectorized path leaves unscored (0 if not taken): it
    is taken when the rows agree on `selected` and all observe model 0's
    observed unselected benchmarks, and then skips every other observed
    unselected cell."""
    A = set(selected)
    obs_A = [j for j in selected if O_test[0, j] == 1]
    ev = [j for j in range(N) if j not in A and O_test[0, j] == 1]
    taken = (obs_A and ev
             and np.all(O_test[:, list(selected)] == O_test[0, list(selected)])
             and np.all(O_test[:, ev] == O_test[0, ev]))
    if not taken:
        return 0
    rest = [j for j in range(N) if j not in A and O_test[0, j] == 0]
    return int(O_test[:, rest].sum())


def _reference(B_test, O_test, selected, R, cm, cs, N):
    """Per-model pooled R² / RMSE (impute_and_evaluate's definition)."""
    A = set(selected)
    Z = np.clip(O_test * (B_test - cm) / cs, -10, 10) * O_test
    ss_res = ss_tot = 0.0
    n = 0
    for i in range(B_test.shape[0]):
        oa = [j for j in selected if O_test[i, j] == 1]
        ei = [j for j in range(N) if j not in A and O_test[i, j] == 1]
        if not oa or not ei:
            continue
        W = np.linalg.solve(R[np.ix_(oa, oa)] + 0.01 * np.eye(len(oa)),
                            R[np.ix_(ei, oa)].T).T
        p = W @ Z[i, oa]
        ss_res += np.sum((Z[i, ei] - p) ** 2)
        ss_tot += np.sum(Z[i, ei] ** 2)
        n += len(ei)
    if n == 0:
        return np.nan, np.nan
    return 1 - ss_res / max(ss_tot, 1e-12), np.sqrt(ss_res / n)


def _corr(N, seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N, 3))
    S = W @ W.T + np.diag(rng.uniform(0.3, 1.0, N))
    d = 1 / np.sqrt(np.diag(S))
    return S * np.outer(d, d)


def _data(M, N, seed):
    rng = np.random.default_rng(seed)
    B = rng.uniform(0.1, 0.9, size=(M, N))
    return B, rng.normal(size=N) * 0.1 + 0.5, rng.uniform(0.05, 0.2, N)


def test_greedy_all_uses_the_shared_metric():
    assert eval_greedy_all.impute_and_evaluate is emi.impute_and_evaluate


def test_benchpress_uses_the_shared_metric():
    assert eval_benchpress.impute_and_evaluate is emi.impute_and_evaluate


def test_no_copy_of_the_model0_fast_path_in_code():
    """No module of code/ keeps the old condition (validation rows compared
    with model 0's row on a subset of columns only) or its own copy of
    impute_and_evaluate (eval_greedy_mmlu / eval_greedy_merged are the
    single-dataset predecessors with their own definitions and no fast
    path)."""
    old_cond = re.compile(r"O_test\[:, *(eval_idx|list\(selected\))\] *== "
                          r"*O_test\[0")
    for path in sorted(CODE.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        src = path.read_text(encoding="utf-8")
        assert not old_cond.search(src), path.name
        if path.name not in ("eval_entropy_vs_mi.py", "eval_greedy_mmlu.py",
                             "eval_greedy_merged.py"):
            assert not re.search(r"^def impute_and_evaluate\(", src,
                                 re.M), path.name
    for name in ("eval_greedy_mmlu.py", "eval_greedy_merged.py"):
        if (CODE / name).is_file():  # not shipped in the public release
            assert "all_same" not in (CODE / name).read_text()


def test_logit_same_rows_rule_is_the_shared_one():
    rng = np.random.default_rng(8)
    for _ in range(100):
        O = (rng.random((int(rng.integers(1, 5)), 6)) < 0.7).astype(float)
        if rng.random() < 0.4:
            O[:] = O[0]
        assert el.same_observation_rows(O) == emi.same_observation_rows(O)


def _at_risk_matrix(seed=0, M=12, N=10):
    """Sparse validation matrix on which the old path drops cells: every
    model observes benchmarks 0..6, model 0 none of 7..9, the others some
    of them."""
    rng = np.random.default_rng(seed)
    O = np.ones((M, N))
    O[:, 7:] = (rng.random((M, 3)) < 0.5)
    O[0, 7:] = 0
    O[1, 7] = 1
    B, cm, cs = _data(M, N, seed + 1)
    return B * O, O, cm, cs


def test_old_path_drops_cells_new_matches_per_model_loop():
    """Regression test for H1 on a synthetic sparse matrix: the frozen old
    function scores only model 0's benchmarks (it equals the per-model
    loop on the matrix without the dropped cells); the fixed function
    equals the plain per-model loop on every cell."""
    N = 10
    R = _corr(N, 6)
    B, O, cm, cs = _at_risk_matrix()
    for sel in ([0], [0, 3], [2, 4, 6], [1, 2, 3, 4, 5]):
        n_drop = _old_path_drops(O, sel, N)
        assert n_drop == int(O[:, 7:].sum()) > 0
        old = _old_impute_and_evaluate(B, O, sel, R, cm, cs, N)
        new = emi.impute_and_evaluate(B, O, sel, R, cm, cs, N)
        ref = _reference(B, O, sel, R, cm, cs, N)
        assert new["r2"] == pytest.approx(ref[0], rel=1e-12, abs=1e-14)
        assert new["rmse"] == pytest.approx(ref[1], rel=1e-12)
        assert abs(old["r2"] - ref[0]) > 1e-6
        O_kept = O.copy()
        O_kept[:, 7:] = 0                 # the cells the old path scored
        kept = _reference(B * O_kept, O_kept, sel, R, cm, cs, N)
        assert old["r2"] == pytest.approx(kept[0], rel=1e-12)
        assert old["rmse"] == pytest.approx(kept[1], rel=1e-12)
        assert eval_benchpress.impute_and_evaluate(
            B, O, sel, R, cm, cs, N)["r2"] == new["r2"]


def test_old_and_new_bit_identical_unless_cells_dropped():
    """Wherever the old vectorized path drops no cell, the old and the
    fixed function choose the same code path and agree bit for bit (so
    the logs of datasets without such splits are unchanged); where it
    drops cells, only the fixed function matches the per-model loop."""
    rng = np.random.default_rng(12)
    N = 9
    R = _corr(N, 9)
    n_dropping = n_same = 0
    for trial in range(400):
        M = int(rng.integers(1, 8))
        B, cm, cs = _data(M, N, 100 + trial)
        kind = trial % 3
        if kind == 0:                   # independent random rows
            O = (rng.random((M, N))
                 < rng.choice([0.4, 0.8, 1.0])).astype(float)
        else:                           # every row equal to model 0's
            O = np.tile((rng.random(N) < rng.choice([0.5, 0.8])).astype(
                float), (M, 1))
            lacks = np.flatnonzero(O[0] == 0)
            if kind == 2 and M > 1 and lacks.size:
                # plus a cell that model 0 lacks (the H1 pattern)
                O[rng.integers(1, M), rng.choice(lacks)] = 1
        k = int(rng.integers(1, N))
        sel = [int(j) for j in rng.permutation(N)[:k]]
        B = B * O
        old = _old_impute_and_evaluate(B, O, sel, R, cm, cs, N)
        new = emi.impute_and_evaluate(B, O, sel, R, cm, cs, N)
        if _old_path_drops(O, sel, N):
            n_dropping += 1
            ref = _reference(B, O, sel, R, cm, cs, N)
            assert new["r2"] == pytest.approx(ref[0], rel=1e-10, abs=1e-12)
        else:
            n_same += 1
            for key in ("r2", "rmse"):
                assert (old[key] == new[key]
                        or (np.isnan(old[key]) and np.isnan(new[key])))
    assert n_dropping > 25 and n_same > 200


def test_logit_at_risk_pattern_scores_every_cell():
    """impute_and_evaluate_logit on the H1 pattern: equal to the per-model
    logit reference, which differs from the reference without the cells
    model 0 lacks."""
    N = 10
    R = _corr(N, 3)
    B, O, _, _ = _at_risk_matrix(seed=4)
    B = np.clip(B, 0.05, 0.95) * O
    B_logit, col_max = el.logit_transform(B, O, col_max=np.ones(N))
    rng = np.random.default_rng(5)
    sd = {"O_test": O, "B_test_logit": B_logit, "B_test_raw": B,
          "cm_logit": rng.normal(size=N) * 0.1,
          "cs_logit": rng.uniform(0.5, 1.5, N), "R": R, "col_max": col_max,
          "cm_raw": rng.normal(size=N) * 0.1 + 0.5,
          "cs_raw": rng.uniform(0.05, 0.2, N),
          "no_info": np.zeros(N, dtype=bool)}
    for sel in ([0], [1, 5], [2, 3, 6]):
        got = el.impute_split(sd, sel)["r2"]
        ref = _logit_reference(sd, sel)
        assert got == pytest.approx(ref, rel=1e-12)
        O_kept = O.copy()
        O_kept[:, 7:] = 0
        assert _logit_reference(dict(sd, O_test=O_kept), sel) != \
            pytest.approx(ref, rel=1e-6)


def test_at_risk_pattern_scores_every_cell():
    """All models observe model 0's benchmarks, model 2 also benchmark 5:
    the old vectorized path dropped (2, 5); the fixed one scores it."""
    N = 6
    R = _corr(N, 0)
    B, cm, cs = _data(4, N, 1)
    O = np.ones((4, N))
    O[:, 5] = 0
    O[2, 5] = 1
    sel = [0, 1]
    got = emi.impute_and_evaluate(B * O, O, sel, R, cm, cs, N)
    ref = _reference(B * O, O, sel, R, cm, cs, N)
    assert got["r2"] == pytest.approx(ref[0], rel=1e-12)
    assert got["rmse"] == pytest.approx(ref[1], rel=1e-12)
    # the scores without cell (2, 5) differ: the cell is counted
    O2 = O.copy()
    O2[2, 5] = 0
    assert _reference(B * O2, O2, sel, R, cm, cs, N)[0] != \
        pytest.approx(ref[0], rel=1e-9)
    assert not emi.same_observation_rows(O)


def test_random_sparse_patterns_match_reference():
    rng = np.random.default_rng(3)
    N = 8
    R = _corr(N, 2)
    for trial in range(200):
        M = int(rng.integers(1, 6))
        B, cm, cs = _data(M, N, trial)
        O = (rng.random((M, N)) < rng.choice([0.5, 0.9, 1.0])).astype(float)
        if trial % 3 == 0:              # rows that agree on model 0's set
            O[:] = O[0]
            extra = rng.integers(0, N)
            O[rng.integers(0, M), extra] = 1
        k = int(rng.integers(1, N))
        sel = [int(j) for j in rng.permutation(N)[:k]]
        got = emi.impute_and_evaluate(B * O, O, sel, R, cm, cs, N)
        ref = _reference(B * O, O, sel, R, cm, cs, N)
        if np.isnan(ref[0]):
            assert np.isnan(got["r2"]) and np.isnan(got["rmse"])
        else:
            assert got["r2"] == pytest.approx(ref[0], rel=1e-10, abs=1e-12)
            assert got["rmse"] == pytest.approx(ref[1], rel=1e-10)


def test_identical_rows_take_vectorized_path():
    N = 7
    R = _corr(N, 4)
    B, cm, cs = _data(30, N, 5)
    O = np.ones((30, N))
    O[:, 6] = 0                          # one benchmark nobody observes
    assert emi.same_observation_rows(O)
    got = emi.impute_and_evaluate(B * O, O, [2, 4], R, cm, cs, N)
    ref = _reference(B * O, O, [2, 4], R, cm, cs, N)
    assert got["r2"] == pytest.approx(ref[0], rel=1e-12)


# ── eval_logit ───────────────────────────────────────────────────

def _logit_reference(sd, sel):
    """Per-model logit-space imputation, raw-space pooled R² (with the
    training-mean prediction on no_info benchmarks)."""
    O = sd["O_test"]
    N = O.shape[1]
    A = set(sel)
    Z = np.clip(O * (sd["B_test_logit"] - sd["cm_logit"]) / sd["cs_logit"],
                -10, 10) * O
    R = sd["R"]
    ss_res = ss_tot = 0.0
    n = 0
    for i in range(O.shape[0]):
        oa = [j for j in sel if O[i, j] == 1]
        ei = [j for j in range(N) if j not in A and O[i, j] == 1]
        if not oa or not ei:
            continue
        W = np.linalg.solve(R[np.ix_(oa, oa)] + 0.01 * np.eye(len(oa)),
                            R[np.ix_(ei, oa)].T).T
        f = (W @ Z[i, oa]) * sd["cs_logit"][ei] + sd["cm_logit"][ei]
        s = el.sigmoid(f) * sd["col_max"][ei]
        t = np.clip((sd["B_test_raw"][i, ei] - sd["cm_raw"][ei])
                    / sd["cs_raw"][ei], -10, 10)
        p = (s - sd["cm_raw"][ei]) / sd["cs_raw"][ei]
        if not score_bounds.PRED_CLIP:    # pre-D2 +-10 (test-only)
            p = np.clip(p, -10, 10)
        p[sd["no_info"][ei]] = 0.0
        ss_res += np.sum((t - p) ** 2)
        ss_tot += np.sum(t ** 2)
        n += len(ei)
    return 1 - ss_res / max(ss_tot, 1e-12)


def _sparse_matrix(M, N, seed, frac=0.7):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(M, 2))
    X = 1 / (1 + np.exp(-(z @ rng.normal(size=(2, N))
                          + 0.3 * rng.normal(size=(M, N)))))
    O = (rng.random((M, N)) < frac).astype(float)
    O[:, 0] = 1
    return X * O, O


def test_raw_stats_mirror_estimate_corr():
    # EM branch (sparse; fewer models than benchmarks) with benchmarks of
    # 0 and 1 training observations, and the pairwise branch (complete)
    for frac, M, N, sparse in [(0.6, 40, 8, True), (1.0, 40, 8, True),
                               (1.0, 6, 8, True), (1.0, 40, 8, False)]:
        B, O = _sparse_matrix(M, N, 7, frac)
        if sparse:
            O[:, 3] = 0                   # no training observation
            O[0, 4] = 1
            O[1:, 4] = 0                  # one training observation
        else:
            B[:, 5] = 0.5                 # constant: variance floor
        B = B * O
        _, cm, cs = emi.estimate_corr(B, O)
        cm2, cs2 = el.raw_stats(B, O)
        assert np.array_equal(cm, cm2) and np.array_equal(cs, cs2)


def test_logit_at_risk_and_no_info():
    B, O = _sparse_matrix(50, 7, 11, frac=0.85)
    s = next(cv_splits(50, 5, 1, 0.2, 42))
    O = O.copy()
    O[s.train_idx, 6] = 0                 # benchmark 6: no training data
    O[s.train_idx[0], 5] = 1
    O[s.train_idx[1:], 5] = 0             # benchmark 5: one observation
    B = B * O
    sd = el.prepare_split(B, O, s)
    assert sd["no_info"].tolist() == [False] * 5 + [True, True]
    # rule D1: benchmarks 5 and 6 are no imputation targets
    assert sd["target"].tolist() == [True] * 5 + [False, False]
    assert sd["cs_raw"][6] == pytest.approx(0.01)   # EM floor, mean 0
    sd_all = dict(sd, target=None)       # every column a target (pre-D1)
    O_t = sd["O_test"].copy()
    O_t[:, 5:] = 0
    rng = np.random.default_rng(0)
    for _ in range(30):
        sel = [int(j) for j in rng.permutation(5)[:int(rng.integers(1, 4))]]
        got = el.impute_split(sd_all, sel)["r2"]
        assert got == pytest.approx(_logit_reference(sd_all, sel), rel=1e-10)
        # D1: the cells of 5 and 6 are not scored at all
        got = el.impute_split(sd, sel)["r2"]
        assert got == pytest.approx(
            _logit_reference(dict(sd, O_test=O_t), sel), rel=1e-10)
    # validation models that observe only a no-info target: before D1 R²
    # was exactly 0 (prediction = training mean), as in the raw pipeline;
    # with D1 nothing is scored (NaN)
    Ov = sd["O_test"].copy()
    Ov[:, 1:] = 0
    Ov[:, 6] = 1
    sd0 = dict(sd_all, O_test=Ov, B_test_raw=sd["B_test_raw"] * 0 + 0.7 * Ov,
               B_test_logit=sd["B_test_logit"] * Ov)
    assert el.impute_split(sd0, [0])["r2"] == 0.0
    assert np.isnan(el.impute_split(dict(sd0, target=sd["target"]),
                                    [0])["r2"])
    # identical rows (vectorized path) give the same value as the loop
    Ov[:, 2] = 1
    sd1 = dict(sd0, O_test=Ov)
    assert el.same_observation_rows(Ov)
    assert el.impute_split(sd1, [0])["r2"] == pytest.approx(
        _logit_reference(sd1, [0]), rel=1e-12)


def test_logit_split_statistics_ignore_validation_rows():
    B, O = _sparse_matrix(40, 6, 13)
    s = next(cv_splits(40, 5, 1, 0.2, 42))
    sd = el.prepare_split(B, O, s)
    B2 = B.copy()
    B2[s.val_idx] = np.random.default_rng(1).uniform(size=(len(s.val_idx),
                                                           6)) * O[s.val_idx]
    sd2 = el.prepare_split(B2, O, s)
    for key in ["R", "cm_raw", "cs_raw", "cm_logit", "cs_logit", "col_max",
                "no_info"]:
        assert np.array_equal(sd[key], sd2[key]), key
