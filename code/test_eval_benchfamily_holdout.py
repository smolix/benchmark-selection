"""Tests for eval_benchfamily_holdout.py (benchmark-family holdout).

Synthetic data only (a few seconds; one names-only check reads the real
Merged column names if data/merged.matrix.npz exists): the hard-coded MMLU
supercategory mapping (all 57 subjects, sizes 18 / 13 / 12 / 14), the
family schemes (MMLU categories, MTEB task types from a meta file, column
prefixes, Merged's source families with the cross-source copies of MMLU
and MATH merged, output tags), the family-restricted metric (agreement
with impute_and_evaluate and with an independent per-model reference,
also on the observation pattern that the flawed vectorized fast path of
protocol_changes H1 mis-scores; only family cells scored), the arms
(blind selection independent of everything the family contributes to R,
aware / random candidates outside the family, random = the CV random
baseline's common random numbers, no new seed tag), the default refit
blind arm unchanged by the held-out family's training scores on the EM
path (where the sub block does change), the common support and the pooled
"all families" aggregate (table columns subtract to the paired
differences), no leakage of validation rows into any estimate, refit =
sub on a fully observed matrix, the refit-vs-sub comparison, and an
end-to-end run (shapes, split metadata, NaN rules, determinism, paired
rows, letters-only macros, --from-logs and its option checks); facloc (a
method since 2026-10-19): its arms are facility_location_select on the
family's candidates, and adding it leaves every array, paired row, summary
entry, diagnostic and macro of the other arms unchanged (its arms stay out
of their common support).
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import re
import shutil
import tempfile

# Outputs of the imported eval scripts go to a scratch root unless the
# caller (run_all.sh's tests job) set one
if not os.environ.get("BENCHSELECT_EXPERIMENT_ROOT"):
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-benchfam-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

import eval_benchfamily_holdout as E
import eval_baselines
from cv_split import RANDOM_SEED_OFFSET, cv_splits, eligible_mask
from eval_entropy_vs_mi import estimate_corr, impute_and_evaluate
from greedy_select import random_select, selection_at_k

CODE = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _factor_data(M=200, N=12, n_factors=3, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((M, n_factors))
    W = rng.standard_normal((n_factors, N))
    return 0.5 + 0.1 * (Z @ W + noise * rng.standard_normal((M, N)))


def _corr(X):
    R = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(R, 1.0)
    return R


def _reference_family_r2(B_test, O_test, selected, R, mean, std, fam,
                         ridge=0.01, clip=10.0):
    """Independent per-model pooled R^2 over the observed family cells of
    the models observing >= 1 selected benchmark."""
    ss_res = ss_tot = 0.0
    n = 0
    for i in range(B_test.shape[0]):
        a = [j for j in selected if O_test[i, j]]
        t = [j for j in fam if O_test[i, j]]
        if not a or not t:
            continue
        z = np.clip((B_test[i] - mean) / std, -clip, clip)
        W = np.linalg.solve(R[np.ix_(a, a)] + ridge * np.eye(len(a)),
                            R[np.ix_(a, t)])
        pred = z[a] @ W
        ss_res += float(np.sum((z[t] - pred) ** 2))
        ss_tot += float(np.sum(z[t] ** 2))
        n += len(t)
    if n == 0:
        return np.nan, 0
    return 1.0 - ss_res / max(ss_tot, 1e-12), n


class _Split:
    """Stand-in for cv_split.Split (only .seed is used by the selectors)."""

    def __init__(self, rep=0, fold=0, p=0.1):
        self.rep, self.fold, self.holdout_frac = rep, fold, p

    def seed(self, tag):
        from cv_split import split_seed
        return split_seed(tag, self.rep, self.fold, self.holdout_frac)


def _write_matrix(path, B, O, names):
    np.savez(path, B=B * O, O=O, benchmark_names=np.array(names),
             model_names=np.array([f"m{i:03d}" for i in range(B.shape[0])]))
    # valid score ranges of the synthetic data (score_bounds, rule D2)
    meta = pathlib.Path(str(path).replace(".matrix.npz", ".meta.json"))
    meta.write_text(json.dumps({"score_bounds": {"default": [None, None]}}))


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def test_mmlu_mapping_complete():
    assert len(E.MMLU_SUBCATEGORY) == 57
    subs = [s for lst in E.MMLU_CATEGORIES.values() for s in lst]
    assert len(subs) == len(set(subs))
    assert set(E.MMLU_SUBCATEGORY.values()) == set(subs)
    counts = {}
    for subject in E.MMLU_SUBCATEGORY:
        cat = E.mmlu_category(subject)
        counts[cat] = counts.get(cat, 0) + 1
    assert counts == {"stem": 18, "humanities": 13, "social_sciences": 12,
                      "other": 14}
    assert E.mmlu_category("not_a_subject") is None
    # spot checks against hendrycks/test categories.py
    assert E.mmlu_category("formal_logic") == "humanities"
    assert E.mmlu_category("professional_accounting") == "other"
    assert E.mmlu_category("econometrics") == "social_sciences"
    assert E.mmlu_category("high_school_statistics") == "stem"


def test_mmlu_families_partition_and_strictness():
    names = sorted(E.MMLU_SUBCATEGORY)
    for ds in ("mmlu", "helm_mmlu"):
        scheme, fams = E.benchmark_families(ds, names)
        assert scheme == "mmlu_category"
        assert [f for f, _ in fams] == ["stem", "humanities",
                                        "social_sciences", "other"]
        allidx = np.concatenate([idx for _, idx in fams])
        assert sorted(allidx.tolist()) == list(range(57))
        with pytest.raises(ValueError):
            E.benchmark_families(ds, names + ["gsm8k"])
    _, fams = E.benchmark_families("olv1items", names + ["arc_challenge",
                                                         "gsm8k"])
    assert [f for f, _ in fams][-2:] == ["arc_challenge", "gsm8k"]
    assert [len(idx) for _, idx in fams] == [18, 13, 12, 14, 1, 1]


def test_prefix_families():
    names = ["a/x", "b/y", "a/z", "c", "b/w"]
    scheme, fams = E.benchmark_families("merged", names, scheme="prefix")
    assert scheme == "prefix"
    assert [(f, idx.tolist()) for f, idx in fams] == [
        ("a", [0, 2]), ("b", [1, 4]), ("c", [3])]
    # without cross-source copies the source scheme is the prefix scheme
    scheme, fams2 = E.benchmark_families("merged", names)
    assert scheme == "source"
    assert [(f, i.tolist()) for f, i in fams2] == \
        [(f, i.tolist()) for f, i in fams]
    with pytest.raises(ValueError):
        E.benchmark_families("unknown_dataset", names)


def test_mteb_types_from_meta(tmp_path):
    meta = {"benchmarks": [{"name": "T1", "type": "Retrieval"},
                           {"name": "T2", "type": "STS"},
                           {"name": "T3", "type": "Retrieval"},
                           {"name": "T4", "type": "PairClassification"}]}
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta))
    _, fams = E.benchmark_families("mteb", ["T1", "T2", "T3", "T4"],
                                   mteb_meta=path)
    assert [(f, idx.tolist()) for f, idx in fams] == [
        ("Retrieval", [0, 2]), ("STS", [1]), ("PairClassification", [3])]
    with pytest.raises(ValueError):
        E.benchmark_families("mteb", ["T1", "T9"], mteb_meta=path)


def test_mteb_type_counts_add_up():
    assert sum(E.MTEB_TYPE_COUNTS.values()) == 56


MERGED_LIKE = ["mmlu/abstract_algebra", "mmlu/anatomy", "open_llm/IFEval",
               "open_llm/MATH_Lvl5", "open_llm/MMLU-PRO", "helm/GSM8K",
               "helm/MATH", "helm/MMLU", "bigcode/x"]


def test_merged_source_families_merge_cross_source_copies():
    scheme, fams = E.benchmark_families("merged", MERGED_LIKE)
    assert scheme == "source"
    assert [(f, idx.tolist()) for f, idx in fams] == [
        ("mmlu", [0, 1, 4, 7]), ("open_llm", [2]), ("math", [3, 6]),
        ("helm", [5]), ("bigcode", [8])]
    # every copy sits in the family of its benchmark, nowhere else
    fam_of = {j: f for f, idx in fams for j in idx}
    for name, fam in E.MERGED_SHARED.items():
        assert fam_of[MERGED_LIKE.index(name)] == fam
    scheme, fams = E.benchmark_families("merged", MERGED_LIKE,
                                        scheme="prefix")
    assert scheme == "prefix"
    assert [(f, idx.tolist()) for f, idx in fams] == [
        ("mmlu", [0, 1]), ("open_llm", [2, 3, 4]), ("helm", [5, 6, 7]),
        ("bigcode", [8])]


def test_schemes_and_output_tags():
    assert E.dataset_scheme("merged") == "source"
    assert E.dataset_scheme("merged", "prefix") == "prefix"
    assert E.dataset_scheme("olv2sub", "source") == "prefix"
    assert E.dataset_scheme("olv2sub", "prefix") == "prefix"
    assert E.BLIND_COV[0] == "refit"
    stem = "benchfamily_holdout_"
    assert E.output_stem("merged") == stem + "merged"
    assert E.output_stem("merged", "sub") == stem + "merged_sub"
    assert E.output_stem("merged", "refit", "prefix") == stem + "merged_prefix"
    assert E.output_stem("merged", "sub", "prefix") == \
        stem + "merged_prefix_sub"
    assert E.output_stem("olv2sub", "refit", "prefix") == stem + "olv2sub"
    out = {"scheme": np.array("prefix"), "blind_cov": np.array("sub")}
    assert E.macro_prefix("merged", out) == "BenchFamPrefixSub"
    assert E.macro_prefix("olv2sub", out) == "BenchFamSub"
    out = {"scheme": np.array("source"), "blind_cov": np.array("refit")}
    assert E.macro_prefix("merged", out) == "BenchFam"


def test_real_merged_families_as_documented():
    path = E.DATA / "merged.matrix.npz"
    if not path.is_file():
        pytest.skip("no data/merged.matrix.npz")
    with np.load(path, allow_pickle=False) as d:
        names = [str(n) for n in d["benchmark_names"]]
    for name in E.MERGED_SHARED:
        assert name in names
    _, fams = E.benchmark_families("merged", names)
    assert [(f, len(idx)) for f, idx in fams] == [
        ("mmlu", 59), ("open_llm", 4), ("math", 2), ("helm", 9),
        ("alpaca_eval", 3), ("arena_hard", 1), ("livebench", 3),
        ("wildbench", 11), ("mt_bench", 8), ("bigcode", 14)]


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _metric_setup(seed=1):
    X = _factor_data(M=120, N=10, seed=seed)
    train, test = X[:90], X[90:]
    R = _corr(train)
    mean, std = train.mean(axis=0), train.std(axis=0, ddof=1)
    return R, mean, std, test


def test_family_r2_matches_impute_and_reference_fully_observed():
    R, mean, std, B_test = _metric_setup()
    O_test = np.ones_like(B_test)
    sel, fam = [0, 3], np.array([5, 6, 7])
    r2, n_cells, n_models, ss_res, ss_tot = E.family_r2(
        B_test, O_test, sel, R, mean, std, fam)
    assert r2 == pytest.approx(1.0 - ss_res / ss_tot, rel=1e-14)
    O_mask = np.zeros_like(O_test)
    O_mask[:, sel + fam.tolist()] = 1
    direct = impute_and_evaluate(B_test, O_mask, sel, R, mean, std,
                                 R.shape[0])["r2"]
    ref, n_ref = _reference_family_r2(B_test, O_test, sel, R, mean, std, fam)
    assert r2 == pytest.approx(direct, rel=1e-12, abs=1e-12)
    assert r2 == pytest.approx(ref, rel=1e-10, abs=1e-12)
    assert n_cells == n_ref == B_test.shape[0] * 3
    assert n_models == B_test.shape[0]


def test_family_r2_exact_on_fast_path_pattern():
    """The pattern of protocol_changes H1: every validation model observes
    the selected columns and model 0's family columns, others observe one
    more family column.  family_r2 must score those cells too."""
    R, mean, std, B_test = _metric_setup(seed=2)
    O_test = np.ones_like(B_test)
    sel, fam = [1, 2], np.array([6, 7, 8])
    O_test[0, 8] = 0                      # model 0 misses one family column
    O_test[3, 9] = 0                      # a non-family gap (irrelevant)
    r2, n_cells = E.family_r2(B_test, O_test, sel, R, mean, std, fam)[:2]
    ref, n_ref = _reference_family_r2(B_test, O_test, sel, R, mean, std, fam)
    assert n_cells == n_ref == 3 * B_test.shape[0] - 1
    assert r2 == pytest.approx(ref, rel=1e-10, abs=1e-12)


def test_family_r2_sparse_patterns_and_skips():
    R, mean, std, B_test = _metric_setup(seed=3)
    rng = np.random.default_rng(0)
    O_test = (rng.random(B_test.shape) < 0.6).astype(float)
    sel, fam = [0, 4, 9], np.array([2, 5])
    O_test[0, sel] = 0                    # no predictor: skipped
    O_test[0, fam] = 1
    r2, n_cells, n_models = E.family_r2(B_test, O_test, sel, R, mean, std,
                                        fam)[:3]
    ref, n_ref = _reference_family_r2(B_test, O_test, sel, R, mean, std, fam)
    assert n_cells == n_ref
    assert r2 == pytest.approx(ref, rel=1e-10, abs=1e-12)
    exp_models = sum(1 for i in range(len(B_test))
                     if O_test[i, sel].any() and O_test[i, fam].any())
    assert n_models == exp_models
    # nothing scorable -> NaN
    O_none = O_test.copy()
    O_none[:, fam] = 0
    assert np.isnan(E.family_r2(B_test, O_none, sel, R, mean, std, fam)[0])


def test_family_r2_scores_only_family_cells():
    R, mean, std, B_test = _metric_setup(seed=4)
    O_test = np.ones_like(B_test)
    sel, fam = [0, 1], np.array([4, 5])
    base = E.family_r2(B_test, O_test, sel, R, mean, std, fam)[0]
    other = B_test.copy()
    other[:, [2, 3, 6, 7, 8, 9]] += 5.0   # unselected, outside the family
    assert E.family_r2(other, O_test, sel, R, mean, std, fam)[0] == base
    changed = B_test.copy()
    changed[:, 4] += 0.3
    assert E.family_r2(changed, O_test, sel, R, mean, std, fam)[0] != base
    with pytest.raises(ValueError):
        E.family_r2(B_test, O_test, [0, 4], R, mean, std, fam)


def _two_arm_out(S=3, F=3, K=2, seed=11):
    rng = np.random.default_rng(seed)
    out = {"split_id": np.arange(S), "arms": np.array(["x", "y"]),
           "family_in_mean": np.array([True, True, False]), "k_max": K}
    for arm in ("x", "y"):
        out[f"{arm}_ss_res"] = rng.uniform(0.1, 1.0, (S, F, K))
        out[f"{arm}_ss_tot"] = rng.uniform(1.0, 2.0, (S, F, K))
        out[f"{arm}_r2"] = 1 - out[f"{arm}_ss_res"] / out[f"{arm}_ss_tot"]
    return out


def test_pooled_all_families_common_support():
    """"all" = 1 - sum SS_res / sum SS_tot over the families of the mean
    in the common support: a cell where any arm is NaN is dropped for
    every arm, so all arms pool the same families."""
    out = _two_arm_out()
    out["y_r2"][0, 1, 0] = np.nan          # y lacks family 1 on split 0
    out["x_r2"][2, :, 1] = np.nan          # x has nothing on split 2, k = 2
    sup = E.common_support(out, 1)
    assert sup.tolist() == [[True, False, True], [True] * 3, [True] * 3]
    x1 = E.family_mean(out, "x", 1)
    exp = 1 - out["x_ss_res"][:, :2, 0].sum(1) / out["x_ss_tot"][:, :2, 0].sum(1)
    exp[0] = out["x_r2"][0, 0, 0]          # family 1 dropped on split 0
    np.testing.assert_allclose(x1, exp, rtol=1e-14)
    xa, ya = E.family_mean_pair(out, "x", "y", 1)
    np.testing.assert_allclose(xa, x1, rtol=1e-14)
    assert ya[0] == pytest.approx(out["y_r2"][0, 0, 0], rel=1e-14)
    # k = 2: x has nothing on split 2, so neither arm has "all" there
    assert np.isnan(E.family_mean(out, "x", 2)[2])
    assert np.isnan(E.family_mean(out, "y", 2)[2])
    assert np.isfinite(E.family_mean(out, "y", 2)[:2]).all()
    # an explicit support (here: x's own cells) is honoured
    own = np.isfinite(out["x_r2"][:, :, 0])
    x_own = E.family_mean(out, "x", 1, own)
    assert x_own[0] == pytest.approx(
        1 - out["x_ss_res"][0, :2, 0].sum() / out["x_ss_tot"][0, :2, 0].sum())
    with pytest.raises(ValueError):          # y is NaN inside that support
        E.family_mean(out, "y", 1, own)
    # one family with SS_tot ~ 0 cannot dominate the pooled value
    out["x_ss_tot"][:, 1, 0] = 1e-6
    out["x_r2"][:, 1, 0] = 1 - out["x_ss_res"][:, 1, 0] / 1e-6
    assert np.all(E.family_mean(out, "x", 1) > -2.0)


def test_support_diagnostics():
    out = _two_arm_out()
    out["y_r2"][0, 1, 0] = np.nan
    diag = E.support_diagnostics(out, ks=[1, 2])
    assert diag["1"] == {"cells_any": 6, "cells_common": 5, "splits_all": 3,
                         "nan_cells": {"y": 1}}
    assert diag["2"]["cells_common"] == 6 and diag["2"]["nan_cells"] == {}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _arm_setup():
    rng = np.random.default_rng(5)
    X = _factor_data(M=300, N=14, seed=5)
    fam = np.zeros(14, dtype=bool)
    fam[[3, 4, 10]] = True
    # X2: the family columns replaced (non-family block of R unchanged)
    X2 = X.copy()
    X2[:, fam] = X[:, [0]] + 0.05 * rng.standard_normal((300, 3))
    eligible = np.ones(14, dtype=bool)
    eligible[[7]] = False
    names = np.array([f"b{j}" for j in range(14)])
    return _corr(X), _corr(X2), fam, eligible, names


def test_blind_ignores_family_and_aware_uses_it():
    R, R2, fam, eligible, names = _arm_setup()
    keep = np.flatnonzero(~fam)
    assert np.array_equal(R[np.ix_(keep, keep)], R2[np.ix_(keep, keep)])
    split = _Split(rep=1, fold=2)
    differs = 0
    for arm in E.ARMS:
        r_a = E.select_arm(arm, R, eligible, fam, 8, names, split)
        r_b = E.select_arm(arm, R2, eligible, fam, 8, names, split)
        for k in range(1, 9):
            sa, sb = selection_at_k(r_a, k), selection_at_k(r_b, k)
            assert not (set(sa) & set(np.flatnonzero(fam)))
            assert all(eligible[j] for j in sa)
            if arm.endswith("_blind") or arm == "random":
                assert sa == sb, (arm, k)
            elif sa != sb:
                differs += 1
    assert differs > 0          # the aware objectives see the family


def test_blind_equals_selector_on_submatrix():
    R, _, fam, eligible, names = _arm_setup()
    keep = np.flatnonzero(~fam)
    split = _Split()
    for m in E.METHODS:
        res = E.select_arm(f"{m}_blind", R, eligible, fam, 6, names, split)
        sub = eval_baselines.run_selection(m, R[np.ix_(keep, keep)], 6,
                                           names[keep], split, eligible[keep])
        for k in range(1, 7):
            assert selection_at_k(res, k) == [int(keep[j]) for j in
                                              selection_at_k(sub, k)]


def test_random_is_cv_random_baseline_restricted():
    R, _, fam, eligible, names = _arm_setup()
    split = _Split(rep=2, fold=4)
    res = E.select_arm("random", R, eligible, fam, 10, names, split)
    ref = random_select(R, 10, seed=split.seed(E.SEED + RANDOM_SEED_OFFSET),
                        eligible=eligible & ~fam)
    assert res["selected"] == ref["selected"]
    # = the full-candidate CV random baseline's order without F / ineligible
    full = random_select(R, 14, seed=split.seed(E.SEED + RANDOM_SEED_OFFSET))
    expect = [j for j in full["selected"] if eligible[j] and not fam[j]]
    assert res["selected"] == expect[:10]


def test_entropy_blind_aware_agree_given_first_pick():
    """entropy's objective ignores F: with the same first pick the blind
    and aware selections coincide (they differ only by tie-breaks)."""
    R, _, fam, eligible, names = _arm_setup()
    split = _Split()
    rb = E.select_arm("entropy_blind", R, eligible, fam, 8, names, split)
    ra = E.select_arm("entropy_aware", R, eligible, fam, 8, names, split)
    if rb["selected"][0] == ra["selected"][0]:
        assert rb["selected"] == ra["selected"]
    else:
        from greedy_select import greedy_entropy
        forced = greedy_entropy(R, 8, eligible=eligible & ~fam,
                                first=rb["selected"][0])
        assert forced["selected"] == rb["selected"]


def test_no_new_seed_tags():
    src = (CODE / "eval_benchfamily_holdout.py").read_text()
    assert set(re.findall(r"SEED \+ (\d+)", src)) <= {"7"}
    assert "default_rng" not in src


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def _sparse_dataset(tmp_path, name="merged", M=72, seed=7, full=False,
                    p_obs=0.85):
    """Synthetic 'merged'-style matrix (families aa 4, bb 3, cc 3, dd 1);
    p_obs < 0.9 keeps estimate_corr on its EM path (0.85 converges about
    4x faster than 0.75, which keeps the tests fast)."""
    rng = np.random.default_rng(seed)
    names = [f"{p}/t{j}" for p, n in (("aa", 4), ("bb", 3), ("cc", 3),
                                      ("dd", 1)) for j in range(n)]
    N = len(names)
    B = _factor_data(M=M, N=N, seed=seed)
    if full:
        O = np.ones((M, N))
    else:
        O = (rng.random((M, N)) < p_obs).astype(float)
        O[:2, 4:7] = 0               # sparse "bb" family
        O[np.arange(M), rng.integers(0, N, M)] = 1
    _write_matrix(tmp_path / f"{name}.matrix.npz", B, O, names)
    return B * O, O, names


def _patch_dirs(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    monkeypatch.setattr(E, "DATA", tmp_path)
    monkeypatch.setattr(E, "LOGS", out)
    monkeypatch.setattr(E, "OUT", out)
    return out


def test_end_to_end(tmp_path, monkeypatch):
    B, O, names = _sparse_dataset(tmp_path)
    out_dir = _patch_dirs(monkeypatch, tmp_path)
    argv = ["--datasets", "merged", "--repeats", "2", "--folds", "3",
            "--kmax", "4", "--min-family-models", "1"]
    E.main(argv)
    stem = out_dir / "benchfamily_holdout_merged"
    for suffix in (".npz", ".json", "_paired.csv", ".pdf"):
        assert (pathlib.Path(f"{stem}{suffix}")).is_file(), suffix
    d = dict(np.load(f"{stem}.npz", allow_pickle=False))
    S, F, K = 6, 4, 4
    assert d["mi_blind_r2"].shape == (S, F, K)
    assert list(d["family_names"]) == ["aa", "bb", "cc", "dd"]
    assert d["pca_aware_selected_per_k"].shape == (S, F, K, K)
    for key in ("rep", "fold", "split_id", "n_train", "n_val", "n_eligible"):
        assert d[key].shape == (S,)
    splits = list(cv_splits(B.shape[0], 3, 2, E.HOLDOUT_FRAC, E.SEED))
    assert d["split_id"].tolist() == [s.split_id for s in splits]
    assert d["n_train"].tolist() == [len(s.train_idx) for s in splits]
    for s_row, s in enumerate(splits):
        el = eligible_mask(O[s.train_idx])
        assert np.array_equal(d["eligible"][s_row], el)
        for f in range(F):
            fam = d["family_of"] == f
            assert d["n_candidates"][s_row, f] == int(np.sum(el & ~fam))
            for arm in E.ARMS:
                sel = d[f"{arm}_selected"][s_row, f]
                sel = sel[sel >= 0]
                assert not fam[sel].any() and el[sel].all()
                r2 = d[f"{arm}_r2"][s_row, f]
                n_c = d["n_candidates"][s_row, f]
                assert np.all(np.isnan(r2[n_c:]))
                assert np.all(np.isnan(r2) == (d[f"{arm}_n_cells"][s_row, f]
                                               == 0))
    assert np.isfinite(d["trace_blind_r2"]).mean() > 0.5
    assert str(d["blind_cov"]) == "refit" and str(d["scheme"]) == "source"
    # the figure was written once; the reruns below skip it (time)
    monkeypatch.setattr(E, "plot_dataset", lambda *args, **kw: None)
    doc = json.loads(pathlib.Path(f"{stem}.json").read_text())
    assert doc["macros"] and all(k.isalpha() for k in doc["macros"])
    assert "BenchFamRsqMergedAllTraceBlindKone" in doc["macros"]
    assert "BenchFamPairDiffMergedAaMiBlindVsRandomKthree" in doc["macros"]
    n_rows = sum(1 for _ in open(f"{stem}_paired.csv")) - 1
    n_m = len(E.METHODS)                       # 6 with facloc
    assert n_rows == (F + 1) * K * (3 * n_m + 2 * n_m * (n_m - 1) // 2)
    assert "BenchFamRsqMergedAllFaclocBlindKone" in doc["macros"]
    assert "BenchFamPairDiffMergedAllFaclocBlindVsRandomKthree" \
        in doc["macros"]
    # sub: tagged _sub; the aware and random arms do not depend on it
    E.main(argv + ["--blind-cov", "sub"])
    ds = dict(np.load(f"{stem}_sub.npz", allow_pickle=False))
    assert str(ds["blind_cov"]) == "sub"
    # deterministic (the sub run, which shares everything but the refit
    # with the default; the refit's determinism: the leak test)
    E.main(argv + ["--blind-cov", "sub"])
    d2 = dict(np.load(f"{stem}_sub.npz", allow_pickle=False))
    for arm in E.ARMS:
        assert np.array_equal(ds[f"{arm}_r2"], d2[f"{arm}_r2"],
                              equal_nan=True)
        assert np.array_equal(ds[f"{arm}_selected"], d2[f"{arm}_selected"])
    for arm in E.ARMS:
        if not arm.endswith("_blind"):
            assert np.array_equal(d[f"{arm}_r2"], ds[f"{arm}_r2"],
                                  equal_nan=True), arm
    doc_s = json.loads(pathlib.Path(f"{stem}_sub.json").read_text())
    assert "BenchFamSubRsqMergedAllTraceBlindKone" in doc_s["macros"]
    E.main(argv + ["--compare-blind-cov"])
    cmp_doc = json.loads(pathlib.Path(f"{stem}_blindcov.json").read_text())
    assert all(k.isalpha() for k in cmp_doc["macros"])
    for m in E.METHODS:
        for g in cmp_doc["agreement"][m].values():
            assert 0.0 <= g["set_equal"] <= 1.0
            if m == "pca":
                assert g["seq_equal"] is None
            else:            # equal sequences are equal sets
                assert g["seq_equal"] <= g["set_equal"]
    # --from-logs rewrites the same summaries without recomputing
    csv_before = pathlib.Path(f"{stem}_paired.csv").read_text()
    monkeypatch.setattr(E, "run_dataset", None)
    E.main(argv + ["--from-logs"])
    doc2 = json.loads(pathlib.Path(f"{stem}.json").read_text())
    assert doc2["macros"] == doc["macros"]
    assert doc2["summary"] == doc["summary"]
    assert pathlib.Path(f"{stem}_paired.csv").read_text() == csv_before
    # ... and refuses a log written with other options
    shutil.copy(f"{stem}_sub.npz", f"{stem}.npz")
    with pytest.raises(SystemExit):
        E.main(argv + ["--from-logs"])
    with pytest.raises(SystemExit):
        E.main(argv + ["--from-logs", "--merged-families", "prefix"])


def test_table_columns_subtract_to_paired_differences(tmp_path,
                                                       monkeypatch):
    """Common support: with an arm NaN on some cells, every arm of a table
    cell has the same n and the difference of two table means is the
    paired mean difference (per family and for "all").  The APART_METHODS
    arms (facloc) use the support of the other arms and both facloc arms:
    the same holds among them (facloc aware - blind) and for the pairs of
    the other arms."""
    _sparse_dataset(tmp_path, seed=12)
    _patch_dirs(monkeypatch, tmp_path)
    out = E.run_dataset("merged", 1, 3, 3, "sub", verbose=False)
    out["min_family_models"] = np.int64(1)
    out["family_in_mean"] = np.ones(len(out["family_names"]), dtype=bool)
    for key in ("r2", "ss_res", "ss_tot"):          # random misses cells
        out[f"random_{key}"][0, :, 0] = np.nan
        out[f"random_{key}"][1, 0, :] = np.nan
    tables = E.summary_tables(out)
    paired = E.paired_rows("merged", out)
    n_checked = 0
    for r in paired:
        if r["k"] not in E.KS_REPORT or r["n_splits"] == 0:
            continue
        if E.is_apart(r["method_a"]) != E.is_apart(r["method_b"]):
            continue                    # facloc vs a core arm: own support
        t = tables[r["family"]][str(r["k"])]
        assert t[r["method_a"]]["n"] == t[r["method_b"]]["n"] \
            == r["n_splits"]
        assert t[r["method_a"]]["mean"] - t[r["method_b"]]["mean"] == \
            pytest.approx(r["mean_diff"], abs=1e-12)
        n_checked += 1
    assert n_checked > 0
    # split 0 has no k = 1 cell in the support: "all" is NaN there for
    # every arm, not only for random
    for arm in E.ARMS:
        assert np.isnan(E.family_mean(out, arm, 1)[0])
    assert tables[E.ALL]["1"]["trace_blind"]["n"] == len(out["split_id"]) - 1


def test_blind_cov_comparison_of_identical_runs():
    out = _two_arm_out()
    rng = np.random.default_rng(4)
    S, F, K = 3, 3, 2
    full = {"split_id": np.arange(S), "family_names": np.array(["a", "b",
                                                                "c"]),
            "benchmark_names": np.array(list("pqrstu")),
            "family_of": np.array([0, 0, 1, 1, 2, 2]),
            "family_in_mean": np.ones(F, dtype=bool), "k_max": np.int64(K),
            "arms": np.array(E.ARMS), "scheme": np.array("source"),
            "blind_cov": np.array("refit"),
            "n_train": np.full(S, 27), "n_val": np.full(S, 3),
            "rep": np.zeros(S, dtype=int)}
    for arm in E.ARMS:
        full[f"{arm}_ss_res"] = rng.uniform(0.1, 1.0, (S, F, K))
        full[f"{arm}_ss_tot"] = out["x_ss_tot"]
        full[f"{arm}_r2"] = 1 - full[f"{arm}_ss_res"] / full[f"{arm}_ss_tot"]
        full[f"{arm}_selected"] = rng.integers(0, 6, (S, F, K))
    for mode in E.MODES:
        full[f"pca_{mode}_selected_per_k"] = rng.integers(0, 6, (S, F, K, K))
    sub = dict(full, blind_cov=np.array("sub"))
    res = E.blind_cov_comparison("merged", full, sub)
    for m in E.METHODS:
        for g in res["agreement"][m].values():
            assert g["set_equal"] == 1.0 and g["n"] == S * F
    assert res["significant"] == []
    assert res["macros"]["BenchFamCovMaxAbsDiffMerged"] == 0.0
    assert res["macros"]["BenchFamCovNSigMerged"] == 0
    # facloc's rows are counted apart (one method x K rows of "all")
    assert res["macros"]["BenchFamCovMaxAbsDiffMergedFacloc"] == 0.0
    assert res["macros"]["BenchFamCovNPairsMergedFacloc"] == K
    assert res["macros"]["BenchFamCovNPairsMerged"] == \
        (len(E.METHODS) - 1) * K
    # an aware arm that differs between the runs is an error
    bad = dict(sub, mi_aware_r2=sub["mi_aware_r2"] + 0.1)
    with pytest.raises(ValueError):
        E.blind_cov_comparison("merged", full, bad)
    # a changed blind pick lowers the agreement
    sub2 = dict(sub, trace_blind_selected=full["trace_blind_selected"].copy())
    sub2["trace_blind_selected"][0, 0, 0] = 99
    res2 = E.blind_cov_comparison("merged", full, sub2)
    assert res2["agreement"]["trace"]["1"]["set_equal"] == \
        pytest.approx(1 - 1 / (S * F))


def test_estimates_use_training_rows_only(tmp_path, monkeypatch):
    B, O, names = _sparse_dataset(tmp_path, seed=8)
    _patch_dirs(monkeypatch, tmp_path)
    calls = []

    def spy(B_train, O_train):
        calls.append((B_train.copy(), O_train.copy()))
        return estimate_corr(B_train, O_train)
    monkeypatch.setattr(E, "estimate_corr", spy)
    out = E.run_dataset("merged", n_repeats=1, n_folds=3, k_max=3,
                        blind_cov="refit", verbose=False)
    splits = list(cv_splits(B.shape[0], 3, 1, E.HOLDOUT_FRAC, E.SEED))
    F = len(out["family_names"])
    assert len(calls) == len(splits) * (1 + F)
    for s_i, s in enumerate(splits):
        first = calls[s_i * (1 + F)]
        assert np.array_equal(first[0], B[s.train_idx])
        for f in range(F):
            keep = np.flatnonzero(out["family_of"] != f)
            Bc, Oc = calls[s_i * (1 + F) + 1 + f]
            assert np.array_equal(Bc, B[s.train_idx][:, keep])
            assert np.array_equal(Oc, O[s.train_idx][:, keep])


def test_refit_blind_is_blind_to_family_training_scores(tmp_path,
                                                        monkeypatch):
    """EM path (sparse matrix): perturbing the held-out family's scores
    changes the non-family block of the full estimate (what --blind-cov
    sub selects on) but not the refit, so no default blind pick for that
    family changes."""
    B, O, names = _sparse_dataset(tmp_path, seed=10)
    _patch_dirs(monkeypatch, tmp_path)
    fam_cols = np.array([n.startswith("aa/") for n in names])
    keep = np.flatnonzero(~fam_cols)
    rng = np.random.default_rng(3)
    B2 = B.copy()
    B2[:, fam_cols] += 0.05 * rng.standard_normal((B.shape[0],
                                                   fam_cols.sum()))
    B2 *= O
    s = next(iter(cv_splits(B.shape[0], 3, 1, E.HOLDOUT_FRAC, E.SEED)))
    tr = s.train_idx
    assert O[tr].mean() < 0.9                 # estimate_corr takes the EM
    R1 = estimate_corr(B[tr], O[tr])[0]
    R2 = estimate_corr(B2[tr], O[tr])[0]
    assert not np.allclose(R1[np.ix_(keep, keep)], R2[np.ix_(keep, keep)],
                           rtol=0, atol=1e-8)   # sub sees the family
    assert np.array_equal(estimate_corr(B[tr][:, keep], O[tr][:, keep])[0],
                          estimate_corr(B2[tr][:, keep], O[tr][:, keep])[0])
    a = E.run_dataset("merged", 1, 3, 4, verbose=False)
    assert str(a["blind_cov"]) == "refit"
    _write_matrix(tmp_path / "merged.matrix.npz", B2, O, names)
    b = E.run_dataset("merged", 1, 3, 4, verbose=False)
    f = list(a["family_names"]).index("aa")
    for m in E.METHODS:
        arm = f"{m}_blind"
        assert np.array_equal(a[f"{arm}_selected"][:, f],
                              b[f"{arm}_selected"][:, f]), arm
        if m == "pca":
            assert np.array_equal(a[f"{arm}_selected_per_k"][:, f],
                                  b[f"{arm}_selected_per_k"][:, f])
    # the targets did change, so the scores do
    assert not np.allclose(a["trace_blind_r2"][:, f], b["trace_blind_r2"][:, f],
                           equal_nan=True)


def test_refit_equals_sub_on_fully_observed(tmp_path, monkeypatch):
    _sparse_dataset(tmp_path, name="olv2sub", M=90, seed=9, full=True)
    _patch_dirs(monkeypatch, tmp_path)
    a = E.run_dataset("olv2sub", 1, 3, 4, "sub", verbose=False)
    b = E.run_dataset("olv2sub", 1, 3, 4, "refit", verbose=False)
    for arm in E.ARMS:
        assert np.array_equal(a[f"{arm}_selected"], b[f"{arm}_selected"]), arm
        np.testing.assert_allclose(a[f"{arm}_r2"], b[f"{arm}_r2"],
                                   rtol=1e-12, atol=1e-12)


def test_workers_give_the_serial_arrays(tmp_path, monkeypatch):
    """--workers W: the splits run in W spawned processes (map_splits);
    every array equals the serial run's bit for bit."""
    _sparse_dataset(tmp_path)
    _patch_dirs(monkeypatch, tmp_path)
    kw = dict(n_repeats=1, n_folds=3, k_max=3, min_family_models=1,
              verbose=False)
    a = E.run_dataset("merged", **kw)
    b = E.run_dataset("merged", workers=2, **kw)
    assert sorted(a) == sorted(b)
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert x.shape == y.shape, key
        assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), key


# ---------------------------------------------------------------------------
# facloc (facility location, a method since 2026-10-19)
# ---------------------------------------------------------------------------

PRE_FACLOC_METHODS = ["mi", "entropy", "trace", "pca", "hub"]


def test_facloc_arms_are_facility_location_select():
    """facloc_aware = facility_location_select (clipped) on the full R with
    the eligible non-family candidates; facloc_blind = the same on the
    non-family block with its eligible set, picks mapped back."""
    from baseline_select import facility_location_select
    R, _, fam, eligible, names = _arm_setup()
    keep = np.flatnonzero(~fam)
    split = _Split()
    assert E.METHODS[:5] == PRE_FACLOC_METHODS and E.METHODS[5] == "facloc"
    assert E.APART_METHODS == ["facloc"]
    assert E.is_apart("facloc_blind") and E.is_apart("facloc_aware")
    assert not E.is_apart("random") and not E.is_apart("mi_blind")
    aware = E.select_arm("facloc_aware", R, eligible, fam, 8, names, split)
    ref = facility_location_select(R, 8, eligible=eligible & ~fam,
                                   similarity="clipped")
    assert aware["selected"] == ref["selected"]
    blind = E.select_arm("facloc_blind", R, eligible, fam, 8, names, split)
    ref = facility_location_select(R[np.ix_(keep, keep)], 8,
                                   eligible=eligible[keep])
    assert blind["selected"] == [int(keep[j]) for j in ref["selected"]]


def test_facloc_changes_no_other_value(tmp_path, monkeypatch):
    """Every array, paired row, summary entry, diagnostic and macro of a
    run with the pre-facloc METHODS / ARMS equals the run with facloc:
    the facloc arms stay out of the other arms' common support, and the
    refit - sub counts of the other methods keep their rows."""
    _sparse_dataset(tmp_path, seed=12)
    _patch_dirs(monkeypatch, tmp_path)
    kw = dict(n_repeats=1, n_folds=3, k_max=3, min_family_models=1,
              verbose=False)

    def run():
        out = {cov: E.run_dataset("merged", blind_cov=cov, **kw)
               for cov in ("refit", "sub")}
        o = out["refit"]
        paired = E.paired_rows("merged", o)
        tables = E.summary_tables(o)
        return (out, paired, tables,
                E.macro_dict("merged", o, paired, tables),
                E.support_diagnostics(o),
                E.blind_cov_comparison("merged", out["refit"], out["sub"]))

    new = run()
    monkeypatch.setattr(E, "METHODS", list(PRE_FACLOC_METHODS))
    monkeypatch.setattr(E, "ARMS", [f"{m}_{mode}" for m in PRE_FACLOC_METHODS
                                    for mode in E.MODES] + ["random"])
    old = run()
    for cov in ("refit", "sub"):
        o, n = old[0][cov], new[0][cov]
        assert set(n) - set(o) == {k for k in n if "facloc" in k}
        for key, a in o.items():
            if key in ("arms", "methods"):
                assert [x for x in n[key] if "facloc" not in x] == list(a)
                continue
            assert np.array_equal(a, n[key], equal_nan=a.dtype.kind == "f"), \
                key
    keep = [r for r in new[1]
            if "facloc" not in r["method_a"] + r["method_b"]]
    assert json.dumps(keep, default=str) == json.dumps(old[1], default=str)
    for fam, by_k in old[2].items():
        for k, by_arm in by_k.items():
            for arm, entry in by_arm.items():
                assert new[2][fam][k][arm] == entry, (fam, k, arm)
    assert {k: new[3][k] for k in old[3]} == old[3]
    assert set(new[3]) > set(old[3])
    assert all("Facloc" in k for k in set(new[3]) - set(old[3]))
    for k, d in old[4].items():
        assert {key: new[4][k][key] for key in d
                if key != "nan_cells"} == {key: d[key] for key in d
                                           if key != "nan_cells"}
        assert {a: new[4][k]["nan_cells"][a] for a in d["nan_cells"]} \
            == d["nan_cells"]
        assert new[4][k]["cells_common_facloc"] <= d["cells_common"]
    cmp_old, cmp_new = old[5], new[5]
    assert {k: cmp_new["macros"][k] for k in cmp_old["macros"]} \
        == cmp_old["macros"]
    assert all("Facloc" in k
               for k in set(cmp_new["macros"]) - set(cmp_old["macros"]))
    keep = [r for r in cmp_new["paired"] if "facloc" not in r["method_a"]]
    assert json.dumps(keep, default=str) == \
        json.dumps(cmp_old["paired"], default=str)
