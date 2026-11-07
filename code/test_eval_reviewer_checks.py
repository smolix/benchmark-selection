"""Tests for eval_reviewer_checks.py (synthetic data only, a few seconds).

Covers the camera-ready extension of the contamination check (promise
A12): per-dataset results keys ("contam" for MMLU, "contam_{ds}" for any
other dataset, written next to it without replacing it), the
contamination itself (only the target's observed scores of the chosen
models change; missing cells never reach the estimator), the verified
estimator label (pairwise on a fully observed matrix with M >= N, EM
otherwise), the target rule (top MI pick among benchmarks observed for at
least min_obs models; the skipped higher MI picks are listed), the trace /
hub ranks (consistent with their orders), the rank distribution over
contaminated draws (draw 0 = the fixed subset, draws d >= 1 from the
documented stream; reproducible; nested 10% within 30% subsets;
eligibility unchanged), the uniqueness of the draws' seed tag (25) in the
code tree, the default contamination datasets (mmlu, plus the EM datasets
among --datasets), the CLI, the printed paired lines (tiny nonzero
intervals readable, '*' where the CI excludes 0), and impute_rows (no
fast path: under the H1 observation pattern of
protocol_changes.md every observed unselected cell is scored, and it
agrees with impute_and_evaluate where that has no fast-path issue).
--workers: clip, contam and mono (in process and through the CLI) give
the serial results, files and printed lines with two spawned processes.
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

# Outputs of the imported eval scripts go to a scratch root unless the
# caller (run_all.sh's tests job) set one
if not os.environ.get("BENCHSELECT_EXPERIMENT_ROOT"):
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-reviewer-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

import eval_reviewer_checks as erc
from cv_split import ENTROPY_RF_SEED_OFFSET, RANDOM_SEED_OFFSET, eligible_mask
from eval_entropy_vs_mi import (estimate_corr, estimate_corr_em,
                                estimate_corr_pairwise, impute_and_evaluate)

CODE = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scores(M=60, N=8, seed=0, miss=0.0, rank=2):
    """(B, O) of a factor model with scores in (0, 1); B = 0 where
    missing.  Column 0 is always fully observed."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(M, rank))
    W = rng.normal(size=(rank, N))
    X = np.clip(0.5 + 0.08 * (Z @ W) + 0.03 * rng.normal(size=(M, N)),
                0.01, 0.99)
    O = (rng.uniform(size=(M, N)) >= miss).astype(float)
    O[:, 0] = 1.0
    return X * O, O


def _write(data_dir, name, B, O):
    np.savez(data_dir / f"{name}.matrix.npz", B=B, O=O,
             benchmark_names=np.array([f"b{j}" for j in range(B.shape[1])]),
             model_names=np.array([f"m{i}" for i in range(B.shape[0])]))
    # scores in [0, 1] (the contamination unit needs a bounded scale)
    (data_dir / f"{name}.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [0.0, 1.0]}}))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Scratch data and logs directories for the module under test."""
    data, logs = tmp_path / "data", tmp_path / "logs"
    data.mkdir()
    logs.mkdir()
    monkeypatch.setattr(erc, "DATA", data)
    monkeypatch.setattr(erc, "LOGS", logs)
    return data, logs


# ---------------------------------------------------------------------------
# Results keys
# ---------------------------------------------------------------------------

def test_contam_key():
    assert erc.CONTAM_DATASET == "mmlu"
    assert erc.contam_key("mmlu") == "contam"
    assert erc.contam_key("mteb") == "contam_mteb"
    assert erc.contam_key("merged") == "contam_merged"


def test_key_order_puts_dataset_keys_after_their_check():
    keys = ["zzz", "mono", "contam_mteb", "clip", "contam", "contam_merged",
            "aaa"]
    assert sorted(keys, key=erc._key_rank) == [
        "clip", "contam", "contam_merged", "contam_mteb", "mono", "aaa",
        "zzz"]
    # Files with only the three original keys keep their order
    assert sorted(["mono", "contam", "clip"], key=erc._key_rank) == list(
        erc.CHECK_ORDER)


def test_em_dataset_result_does_not_replace_mmlu(dirs):
    _, logs = dirs
    erc.save_results("contam", {"benchmark": "miscellaneous", "x": 1.5})
    before = json.loads((logs / erc.RESULTS_FILE).read_text())["contam"]
    erc.save_results(erc.contam_key("mteb"), {"benchmark": "b"})
    erc.save_results(erc.contam_key("merged"), {"benchmark": "c"})
    out = json.loads((logs / erc.RESULTS_FILE).read_text())
    assert out["contam"] == before
    assert list(out) == ["contam", "contam_merged", "contam_mteb"]


# ---------------------------------------------------------------------------
# Contamination and estimator
# ---------------------------------------------------------------------------

def test_contaminate_changes_only_observed_target_cells():
    B, O = _scores(M=30, N=5, miss=0.4, seed=1)
    B[3, 2], O[3, 2] = 0.95, 1.0          # one score near the cap
    models = np.array([3, 5, 7, 11, 13, 17])
    j = 2
    Bc, inc, n_capped = erc.contaminate(B, O, j, models, 0.1, 1.0, 1.0)
    changed = np.argwhere(Bc != B)
    rows = models[O[models, j] == 1]
    assert set(map(tuple, changed)) == {(i, j) for i in rows}
    assert np.allclose(inc, np.minimum(B[rows, j] + 0.1, 1.0) - B[rows, j])
    assert n_capped == int((B[rows, j] + 0.1 > 1.0).sum()) >= 1
    assert np.all(Bc[O == 0] == 0)        # missing cells untouched
    assert np.array_equal(Bc[:, :j], B[:, :j])


@pytest.mark.parametrize("miss", [0.0, 0.35])
def test_missing_cells_never_reach_the_estimator(miss):
    """The rebuttal-era code also wrote delta into missing cells of the
    contaminated models; the estimators mask them, so R is bit-identical
    (pairwise on the full matrix, EM on the sparse one)."""
    B, O = _scores(M=50, N=6, miss=miss, seed=2)
    models = np.arange(0, 50, 4)
    j = 3
    Bc, _, _ = erc.contaminate(B, O, j, models, 0.2, 1.0, 1.0)
    Bold = B.copy()
    Bold[models, j] = np.minimum(Bold[models, j] + 0.2, 1.0)
    for a, b in zip(estimate_corr(Bc, O), estimate_corr(Bold, O)):
        assert np.array_equal(a, b)


def test_corr_estimator_labels():
    B, O = _scores(M=60, N=6, seed=3)                # full, M >= N
    R = estimate_corr(B, O)[0]
    est = erc.corr_estimator(B, O, R)
    assert est["name"] == "pairwise" and est["verified"]
    assert est["obs_frac"] == 1.0 and est["n_models"] == 60

    B, O = _scores(M=60, N=6, miss=0.3, seed=3)     # < 90% observed
    R = estimate_corr(B, O)[0]
    assert np.array_equal(R, estimate_corr_em(B, O)[0])
    assert erc.corr_estimator(B, O, R)["name"] == "em"

    B, O = _scores(M=5, N=8, seed=4)                 # M < N, fully observed
    R = estimate_corr(B, O)[0]
    assert erc.corr_estimator(B, O, R)["name"] == "em"

    R_bad = estimate_corr_pairwise(*_scores(M=60, N=6, seed=3))[0] * 0.5
    est = erc.corr_estimator(*_scores(M=60, N=6, seed=3), R_bad)
    assert est["name"] == "unknown" and not est["verified"]


def test_default_target_threshold():
    assert erc.CONTAM_MIN_OBS == 30
    assert erc.CONTAM_FRACS == (0.1, 0.3)
    assert erc.CONTAM_DELTAS == (0.1, 0.2)


# Seed tags reserved without a *_SEED_OFFSET assignment in code/: 20 for
# the TabImpute context (eval_selection_stability's docstring)
RESERVED_SEED_TAGS = (20,)


def test_draw_seed_tag_is_unique():
    """CONTAM_DRAW_SEED_OFFSET (25) differs from the CV streams, the
    reserved tags and every other seed tag in the code tree: each
    *SEED_OFFSET = n assignment and each 'SEED + n' of another module (23,
    the first choice, is eval_selection_stability's SUB_SEED_OFFSET)."""
    tag = erc.CONTAM_DRAW_SEED_OFFSET
    assert tag == 25
    assert tag not in (RANDOM_SEED_OFFSET, ENTROPY_RF_SEED_OFFSET)
    assert tag not in RESERVED_SEED_TAGS
    for path in sorted(CODE.glob("*.py")):
        if path.name.startswith("test_") or path.name == \
                "eval_reviewer_checks.py":
            continue
        text = path.read_text()
        for pattern in (r"^\s*\w*SEED_OFFSET\s*=\s*(\d+)",
                        r"\bSEED\s*\+\s*(\d+)\b"):
            for m in re.finditer(pattern, text, re.M):
                assert int(m.group(1)) != tag, (path.name, m.group(0))
    # documented where the draws are described
    assert f"offset {tag})" in erc.__doc__


def test_default_contam_datasets():
    """MMLU always; the EM datasets mteb, merged where --datasets has
    them (so run_all.sh's --check all writes contam_mteb and
    contam_merged, and a DATASETS=mmlu run adds no EM run)."""
    assert erc.CONTAM_EM_DATASETS == ("mteb", "merged")
    assert erc.default_contam_datasets(erc.DATASETS) == [
        "mmlu", "mteb", "merged"]
    assert erc.default_contam_datasets(["mmlu"]) == ["mmlu"]
    assert erc.default_contam_datasets(["benchpress"]) == ["mmlu"]
    assert erc.default_contam_datasets(["merged", "benchpress"]) == [
        "mmlu", "merged"]
    assert erc.default_contam_datasets(["merged", "mteb", "mmlu"]) == [
        "mmlu", "mteb", "merged"]


# ---------------------------------------------------------------------------
# check_contam end to end (synthetic)
# ---------------------------------------------------------------------------

ORIGINAL_TOP = ["config", "dataset", "benchmark", "benchmark_index",
                "selected_as", "score_kind", "delta_unit", "cap",
                "n_eligible", "mean_abs_corr_before", "mi_top15_before",
                "entropy_top15_before", "entropy_tie_breaks_before",
                "mi_tie_breaks_before", "mi_index_ties_before",
                "mi_step0_before", "configs"]
ORIGINAL_CFG = ["frac", "delta", "n_models", "model_indices", "n_capped",
                "effective_delta_mean", "mean_abs_corr_before",
                "mean_abs_corr_after", "mi_step0_after", "mi_rank_before",
                "mi_rank_after", "entropy_rank_before", "entropy_rank_after",
                "mi_jaccard_top5", "mi_jaccard_top15",
                "entropy_jaccard_top5", "entropy_jaccard_top15",
                "mi_top15_before", "mi_top15_after", "entropy_top15_before",
                "entropy_top15_after", "entropy_tie_breaks_after",
                "mi_tie_breaks_after", "mi_index_ties_after"]


def _check(ds, n_draws=3, min_obs=erc.CONTAM_MIN_OBS):
    return erc._jsonable(erc.check_contam(ds, n_draws, min_obs))


@pytest.mark.parametrize("miss,estimator", [(0.0, "pairwise"),
                                            (0.3, "em")])
def test_check_contam_layout_and_ranks(dirs, miss, estimator):
    data, _ = dirs
    B, O = _scores(M=40, N=6, miss=miss, seed=5)
    _write(data, "syn", B, O)
    res = _check("syn", n_draws=2, min_obs=0)
    # The rebuttal-era keys come first, in their original order
    assert list(res)[:len(ORIGINAL_TOP)] == ORIGINAL_TOP
    assert res["estimator"]["name"] == estimator
    assert res["estimator"]["verified"]
    names = [f"b{j}" for j in range(6)]
    b = res["benchmark"]
    # min_obs = 0: the plain top MI pick
    assert res["selected_as"] == "top MI pick on the uncontaminated matrix"
    assert res["mi_top15_before"][0] == b
    assert res["ranks_before"]["mi"] == 1
    assert res["config"]["methods"] == list(erc.CONTAM_ALL)
    assert len(res["configs"]) == 4
    R0 = estimate_corr(B, O)[0]
    orders0 = erc.contam_orders(R0, eligible_mask(O))
    for m in erc.CONTAM_ALL:
        order = [names[j] for j in orders0[m]["selected"]]
        assert res["ranks_before"][m] == order.index(b) + 1
    for m in erc.CONTAM_BASELINES:
        assert res[m + "_top15_before"] == [
            names[j] for j in orders0[m]["selected"][:erc.K_MAX]]
    sub = {}
    for cfg in res["configs"]:
        assert list(cfg)[:len(ORIGINAL_CFG)] == ORIGINAL_CFG
        for m in erc.CONTAM_ALL:
            assert cfg[m + "_rank_before"] == res["ranks_before"][m]
            after = cfg[m + "_top15_after"]
            r = cfg[m + "_rank_after"]
            assert (r <= erc.K_MAX) == (b in after)
            if r <= erc.K_MAX:
                assert after[r - 1] == b
        idx = cfg["model_indices"]
        assert len(idx) == int(cfg["frac"] * 40) == cfg["n_models"]
        assert cfg["n_models_observed"] == int(O[idx, res[
            "benchmark_index"]].sum())
        sub[cfg["frac"]] = set(idx)
    assert sub[0.1] <= sub[0.3]           # nested subsets
    # min_obs = 0: no MI pick is skipped; the key comes last
    assert res["skipped_mi_picks"] == [] and list(res)[-1] == \
        "skipped_mi_picks"
    # Draws: one summary per (frac, delta); draw 0 is the fixed subset,
    # draw d >= 1 the permutation default_rng((SEED + 25, d))
    dr = res["draws"]
    assert dr["n_draws"] == 2 and len(dr["configs"]) == 4
    tag = erc.CONTAM_DRAW_SEED_OFFSET
    assert f"default_rng((SEED + {tag}, d))" in dr["seeds"]
    perm1 = np.random.default_rng((erc.SEED + tag, 1)).permutation(40)
    for summ in dr["configs"]:
        sub1 = perm1[:int(summ["frac"] * 40)]
        assert summ["n_models_observed"][1] == int(
            O[sub1, res["benchmark_index"]].sum())
    for cfg, summ in zip(res["configs"], dr["configs"]):
        assert (summ["frac"], summ["delta"]) == (cfg["frac"], cfg["delta"])
        assert summ["n_models_observed"][0] == cfg["n_models_observed"]
        assert len(summ["n_models_observed"]) == 2
        for m in erc.CONTAM_ALL:
            ranks = summ[m + "_rank_after"]
            assert len(ranks) == 2 and ranks[0] == cfg[m + "_rank_after"]
            assert summ[m + "_rank_after_min"] == min(ranks)
            assert summ[m + "_rank_after_max"] == max(ranks)
            r0 = summ[m + "_rank_before"]
            assert summ[m + "_promoted_frac"] == pytest.approx(
                np.mean([r < r0 for r in ranks]))
            assert summ[m + "_top5_after_frac"] == pytest.approx(
                np.mean([r <= 5 for r in ranks]))


def test_check_contam_target_rule(dirs):
    """The target is the first benchmark of the MI order observed for at
    least min_obs models; the MI pick of a sparse column is skipped."""
    data, _ = dirs
    B, O = _scores(M=60, N=8, miss=0.0, seed=6)
    _write(data, "full", B, O)
    top = _check("full", n_draws=1, min_obs=0)["benchmark_index"]
    # Make the top MI pick sparse (12 observed models) and re-run
    O2 = O.copy()
    O2[12:, top] = 0.0
    B2 = B * O2
    _write(data, "sparse", B2, O2)
    loose = _check("sparse", n_draws=1, min_obs=0)
    strict = _check("sparse", n_draws=1, min_obs=20)
    n_obs = O2.sum(axis=0)
    mi_order = [int(s[1:]) for s in loose["mi_top15_before"]]
    first_ok = next(j for j in mi_order if n_obs[j] >= 20)
    assert strict["benchmark_index"] == first_ok
    assert strict["target_n_obs"] == int(n_obs[first_ok]) >= 20
    assert strict["target_min_obs"] == 20
    assert strict["top_mi_pick"] == loose["mi_top15_before"][0]
    # Every MI pick ahead of the target is listed, with its rank and count
    assert loose["skipped_mi_picks"] == []
    assert strict["skipped_mi_picks"] == [
        {"benchmark": f"b{j}", "mi_rank": r + 1, "n_obs": int(n_obs[j])}
        for r, j in enumerate(mi_order[:mi_order.index(first_ok)])]
    assert all(s["n_obs"] < 20 for s in strict["skipped_mi_picks"])
    assert len(strict["skipped_mi_picks"]) == strict["ranks_before"]["mi"] - 1
    if first_ok != mi_order[0]:
        assert strict["selected_as"].startswith(
            "top MI pick among the benchmarks observed for at least 20")
        assert strict["ranks_before"]["mi"] > 1
        assert strict["skipped_mi_picks"][0]["benchmark"] == \
            strict["top_mi_pick"]
    # Eligibility is that of all models, untouched by contamination
    assert strict["n_eligible"] == int(eligible_mask(O2).sum())


def test_check_contam_is_reproducible(dirs):
    data, _ = dirs
    B, O = _scores(M=40, N=6, miss=0.3, seed=7)
    _write(data, "syn", B, O)
    a = _check("syn", n_draws=2, min_obs=0)
    b = _check("syn", n_draws=2, min_obs=0)
    a["config"].pop("source_sha256")
    b["config"].pop("source_sha256")
    assert a == b


def _cli_env(tmp_path, data):
    """Environment of a CLI run reading `data`, writing under tmp/out."""
    env = dict(os.environ, BENCHSELECT_DATA_DIR=str(data),
               BENCHSELECT_EXPERIMENT_ROOT=str(tmp_path / "out"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    for k in ("BENCHSELECT_LOGS_DIR", "BENCHSELECT_FIGURES_DIR"):
        env.pop(k, None)
    return env


def test_cli_writes_per_dataset_keys(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    B, O = _scores(M=40, N=6, seed=8)
    _write(data, "syn_a", B, O)
    B, O = _scores(M=40, N=6, miss=0.3, seed=9)
    _write(data, "syn_b", B, O)
    logs = tmp_path / "out" / "logs"
    logs.mkdir(parents=True)
    # An MMLU result already in the file must survive
    (logs / "reviewer_checks_t.json").write_text(
        json.dumps({"contam": {"benchmark": "miscellaneous"}}))
    env = _cli_env(tmp_path, data)
    out = subprocess.run(
        [sys.executable, str(CODE / "eval_reviewer_checks.py"), "--check",
         "contam", "--contam-dataset", "syn_a", "syn_b", "--contam-draws",
         "2", "--contam-min-obs", "3", "--tag", "t"],
        cwd=CODE, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    res = json.loads((logs / "reviewer_checks_t.json").read_text())
    assert list(res) == ["contam", "contam_syn_a", "contam_syn_b"]
    assert res["contam"] == {"benchmark": "miscellaneous"}
    assert res["contam_syn_a"]["estimator"]["name"] == "pairwise"
    assert res["contam_syn_b"]["estimator"]["name"] == "em"
    assert res["contam_syn_b"]["draws"]["n_draws"] == 2
    assert "[contam_syn_b] results written" in out.stdout
    bad = subprocess.run(
        [sys.executable, str(CODE / "eval_reviewer_checks.py"), "--check",
         "contam", "--contam-dataset", "syn_a", "--contam-draws", "0"],
        cwd=CODE, env=env, capture_output=True, text=True, timeout=60)
    assert bad.returncode != 0 and "--contam-draws" in bad.stderr


def test_cli_default_contam_datasets_follow_datasets(tmp_path):
    """Without --contam-dataset the check runs on mmlu, plus merged only
    when --datasets includes it (synthetic matrices under those names)."""
    data = tmp_path / "data"
    data.mkdir()
    for name, seed, miss in (("mmlu", 13, 0.0), ("merged", 14, 0.3)):
        B, O = _scores(M=40, N=6, seed=seed, miss=miss)
        _write(data, name, B, O)
    env = _cli_env(tmp_path, data)
    logs = tmp_path / "out" / "logs"
    for tag, datasets, keys in (("a", ["mmlu"], ["contam"]),
                                ("b", ["merged", "mmlu"],
                                 ["contam", "contam_merged"])):
        out = subprocess.run(
            [sys.executable, str(CODE / "eval_reviewer_checks.py"),
             "--check", "contam", "--datasets", *datasets,
             "--contam-draws", "1", "--contam-min-obs", "3", "--tag", tag],
            cwd=CODE, env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        res = json.loads((logs / f"reviewer_checks_{tag}.json").read_text())
        assert list(res) == keys
        assert res["contam"]["dataset"] == "mmlu"


# ---------------------------------------------------------------------------
# Printed paired statistics
# ---------------------------------------------------------------------------

def _st(mean, lo, hi, n=50, win=0.5):
    return {"n_splits": n, "mean_diff": mean, "ci_low": lo, "ci_high": hi,
            "win_rate": win}


def test_paired_line_shows_tiny_nonzero_intervals():
    """MMLU's mi_clipped − mi (+2e-6, CI above 0) must not print as
    [+0.0000, +0.0000]; a CI excluding 0 is marked '*'."""
    line = erc._paired_line("mi_clipped − mi", _st(2.2e-6, 3.7e-7, 3.9e-6,
                                                   win=0.98))
    assert "+2.2e-06 [+3.7e-07, +3.9e-06] *" in line
    line = erc._paired_line("a − b", _st(0.0008, -0.0010, 0.0026))
    assert "+0.0008 [-0.0010, +0.0026]  " in line and "*" not in line
    line = erc._paired_line("a − b", _st(-0.0067, -0.0079, -0.0055))
    assert "-0.0067 [-0.0079, -0.0055] *" in line
    # a small mean with a wide CI keeps the fixed format
    line = erc._paired_line("a − b", _st(1e-6, -0.002, 0.002))
    assert "+0.0000 [-0.0020, +0.0020]  " in line
    line = erc._paired_line("a − b", _st(0.0, 0.0, 0.0, win=0.0))
    assert "+0.0000 [+0.0000, +0.0000]  " in line and "*" not in line
    line = erc._paired_line("a − b", _st(np.nan, np.nan, np.nan, n=1))
    assert "*" not in line
    assert "no common splits" in erc._paired_line("a − b", _st(0, 0, 0, n=0))


# ---------------------------------------------------------------------------
# impute_rows: no fast path (H1)
# ---------------------------------------------------------------------------

def _pooled_r2_reference(B, O, selected, R, cm, cs):
    """Pooled R² over every validation model with >= 1 observed selected
    benchmark and every observed unselected benchmark of it."""
    Z = np.clip(O * (B - cm) / cs, -10, 10) * O
    ss_res = ss_tot = 0.0
    n = 0
    for i in range(B.shape[0]):
        a = [j for j in selected if O[i, j]]
        e = [j for j in range(B.shape[1]) if j not in selected and O[i, j]]
        if not a or not e:
            continue
        W = np.linalg.solve(R[np.ix_(a, a)] + 0.01 * np.eye(len(a)),
                            R[np.ix_(e, a)].T).T
        ss_res += np.sum((Z[i, e] - W @ Z[i, a]) ** 2)
        ss_tot += np.sum(Z[i, e] ** 2)
        n += len(e)
    return 1.0 - ss_res / max(ss_tot, 1e-12), n


def test_impute_rows_scores_every_cell_under_the_h1_pattern():
    """All validation models observe the selected columns and every column
    model 0 observes; models 1.. also observe columns 4, 5 that model 0
    lacks.  A fast path keyed on model 0 would score only model 0's
    columns; impute_rows scores all of them."""
    B, O = _scores(M=60, N=7, seed=10)
    R, cm, cs = estimate_corr(B[:50], O[:50])
    Bv, Ov = B[50:].copy(), O[50:].copy()
    Ov[0, [4, 5]] = 0.0
    Bv *= Ov
    selected = [0, 1]
    zt, zp, jj = erc.impute_rows(Bv, Ov, selected, R, cm, cs, 7)
    expected_cells = int(sum(Ov[i, 2:].sum() for i in range(len(Ov))))
    assert len(zt) == len(jj) == expected_cells
    assert np.sum(jj == 4) == len(Ov) - 1       # models 1.. score column 4
    r2_ref, n_ref = _pooled_r2_reference(Bv, Ov, selected, R, cm, cs)
    assert n_ref == expected_cells
    assert erc.r2(zt, zp) == pytest.approx(r2_ref, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("miss", [0.0, 0.3])
def test_impute_rows_agrees_with_impute_and_evaluate(miss):
    """Where impute_and_evaluate has no fast-path issue (fully observed
    validation rows, or patterns where model 0 is not a subset of the
    others) both give the same pooled R²."""
    B, O = _scores(M=80, N=8, miss=miss, seed=11)
    R, cm, cs = estimate_corr(B[:60], O[:60])
    Bv, Ov = B[60:], O[60:]
    for sel in ([0], [0, 3], [1, 2, 5]):
        out = erc.impute_rows(Bv, Ov, sel, R, cm, cs, 8)
        ref = impute_and_evaluate(Bv, Ov, sel, R, cm, cs, 8)["r2"]
        r2_ref, _ = _pooled_r2_reference(Bv, Ov, sel, R, cm, cs)
        assert erc.r2(out[0], out[1]) == pytest.approx(r2_ref, abs=1e-12)
        if miss == 0.0:
            assert erc.r2(out[0], out[1]) == pytest.approx(ref, abs=1e-12)


def test_impute_rows_uses_no_validation_data_in_the_model():
    """Changing a validation model's scores changes only its own
    predictions (the conditional-mean weights come from R alone)."""
    B, O = _scores(M=70, N=6, seed=12)
    R, cm, cs = estimate_corr(B[:60], O[:60])
    Bv, Ov = B[60:].copy(), O[60:]
    sel = [0, 2]
    zt, zp, _ = erc.impute_rows(Bv, Ov, sel, R, cm, cs, 6)
    Bv2 = Bv.copy()
    Bv2[3] = np.clip(Bv2[3] + 0.2, 0, 1)
    zt2, zp2, _ = erc.impute_rows(Bv2, Ov, sel, R, cm, cs, 6)
    per_row = 6 - len(sel)
    other = np.ones(len(zp), dtype=bool)
    other[3 * per_row:4 * per_row] = False
    assert np.array_equal(zp[other], zp2[other])
    assert not np.array_equal(zp[~other], zp2[~other])


# ---------------------------------------------------------------------------
# check_clip: rule D2 raw vs clipped, every main method, k = 1..K
# ---------------------------------------------------------------------------

def _drop_facloc(obj):
    """obj without every key that names facloc (and the config's method
    list / source hashes), recursively."""
    if isinstance(obj, dict):
        return {k: _drop_facloc(v) for k, v in obj.items()
                if "facloc" not in k and k not in ("methods",
                                                   "source_sha256")}
    if isinstance(obj, list):
        return [_drop_facloc(v) for v in obj]
    return obj


@pytest.mark.parametrize("miss", [0.0, 0.3])
def test_check_contam_facloc(dirs, monkeypatch, miss):
    """facloc's contamination orders are facility_location_select on the
    full-matrix estimate (all eligible benchmarks, eligible_mask of all
    models), before and after contamination; adding facloc (2026-10-19)
    changes no other key or value of the check."""
    from baseline_select import facility_location_select
    data, _ = dirs
    B, O = _scores(M=40, N=7, miss=miss, seed=11)
    _write(data, "syn", B, O)
    assert erc.CONTAM_BASELINES == ("trace", "hub", "facloc")
    assert erc.CONTAM_ALL == ("mi", "entropy", "trace", "hub", "facloc")
    res = _check("syn", n_draws=2, min_obs=0)
    el = eligible_mask(O)
    R0 = estimate_corr(B, O)[0]
    fl0 = facility_location_select(R0, 7, eligible=el)
    names = [f"b{j}" for j in range(7)]
    assert res["facloc_top15_before"] == [names[j] for j in
                                          fl0["selected"][:erc.K_MAX]]
    assert res["facloc_tie_breaks_before"] == [list(t) for t in
                                               fl0["tie_breaks"]]
    b = res["benchmark_index"]
    assert res["ranks_before"]["facloc"] == fl0["selected"].index(b) + 1
    cfg = res["configs"][0]
    Bc, _, _ = erc.contaminate(B, O, b, cfg["model_indices"], cfg["delta"],
                               res["delta_unit"], res["cap"])
    fl1 = facility_location_select(estimate_corr(Bc, O)[0], 7, eligible=el)
    assert cfg["facloc_top15_after"] == [names[j] for j in
                                         fl1["selected"][:erc.K_MAX]]
    assert cfg["facloc_rank_after"] == fl1["selected"].index(b) + 1
    for summ in res["draws"]["configs"]:
        assert len(summ["facloc_rank_after"]) == 2
    # Without facloc (the check before 2026-10-19): the same result minus
    # the facloc keys
    monkeypatch.setattr(erc, "CONTAM_BASELINES", ("trace", "hub"))
    monkeypatch.setattr(erc, "CONTAM_ALL", ("mi", "entropy", "trace", "hub"))
    res7 = _check("syn", n_draws=2, min_obs=0)
    assert res7["config"]["methods"] == ["mi", "entropy", "trace", "hub"]
    assert _drop_facloc(res) == _drop_facloc(res7)
    assert set(_drop_facloc(res)) == set(_drop_facloc(res7))


def test_check_clip_raw_vs_clipped_and_shared_scorer(dirs):
    """The clip check's clipped R² is impute_and_evaluate's (targets of
    rule D1, bounds of rule D2) at every method and k; clipping never
    lowers R² (the truths lie in the range); the k = K_EVAL entries are
    the npz's column K_EVAL - 1; bounds come from the metadata."""
    from eval_baselines import run_selection
    from greedy_select import selection_at_k
    from score_bounds import dataset_bounds
    data, logs = dirs
    B, O = _scores(M=60, N=8, miss=0.2, seed=3)
    B = np.clip(B, 0.4, 0.6) * O              # tight range: clips bind
    _write(data, "syn", B, O)
    (data / "syn.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [0.4, 0.6]}}))
    res = erc._jsonable(erc.check_clip(["syn"], n_repeats=1, n_folds=5,
                                       csv_prefix=logs / "rc", k_max=6))
    d = res["datasets"]["syn"]
    assert set(d["by_k"]) == set(erc.CLIP_METHODS)
    assert "entropy_rf" not in d and "entropy_rf" not in erc.CLIP_METHODS
    assert d["bounds_per_benchmark"]["lo"] == [0.4] * 8
    arr = np.load(logs / "rc_clip_syn.npz")
    bounds = dataset_bounds("syn", [f"b{j}" for j in range(8)], data)
    n_out = 0
    for m in erc.CLIP_METHODS:
        raw, clp = arr[f"{m}_r2_raw"], arr[f"{m}_r2_clip"]
        assert raw.shape == clp.shape == (5, 6)
        ok = np.isfinite(raw)
        assert np.all(clp[ok] >= raw[ok] - 1e-12)
        n_out += int(arr[f"{m}_n_outside"].sum())
        np.testing.assert_allclose(
            np.array(d[m]["r2_clip_by_split"], dtype=float),
            clp[:, erc.K_EVAL - 1], rtol=0, atol=0)
        assert d["by_k"][m]["k"] == list(range(1, 7))
    assert n_out > 0
    # the shared scorer gives the clipped value
    from cv_split import cv_splits, target_mask
    for i, s in enumerate(cv_splits(60, 5, 1, erc.HOLDOUT_FRAC, erc.SEED)):
        R, cm, cs = estimate_corr(B[s.train_idx], O[s.train_idx])
        el = eligible_mask(O[s.train_idx])
        tg = target_mask(O[s.train_idx])
        for m in ("mi", "trace", "random"):
            res_m = run_selection(m, R, 6, None, s, el)
            for k in (1, 3, 6):
                sel = selection_at_k(res_m, k)
                if sel is None:
                    continue
                ref = impute_and_evaluate(B[s.val_idx], O[s.val_idx], sel,
                                          R, cm, cs, 8, target=tg,
                                          bounds=bounds)["r2"]
                assert arr[f"{m}_r2_clip"][i, k - 1] == pytest.approx(
                    ref, rel=1e-10, abs=1e-12)


def test_impute_rows_skips_non_targets():
    B, O = _scores(M=30, N=6, seed=9)
    R = estimate_corr(B, O)[0]
    cm, cs = np.full(6, 0.5), np.full(6, 0.1)
    target = np.array([True, True, True, False, True, True])
    zt, zp, jj = erc.impute_rows(B, O, [0], R, cm, cs, 6, target)
    assert 3 not in set(jj.tolist()) and 0 not in set(jj.tolist())
    zt2, _, jj2 = erc.impute_rows(B, O, [0], R, cm, cs, 6)
    assert 3 in set(jj2.tolist())


def test_contamination_scale_from_shared_bounds():
    """The contamination unit comes from score_bounds (rule D2's ranges),
    the same as the observed-range rule of before for the three targets
    (unit interval: (1, 1), percent: (100, 100))."""
    assert erc.contamination_scale("unit_interval", 0.0, 1.0) == (1.0, 1.0)
    assert erc.contamination_scale("percent", 0.0, 100.0) == (100.0, 100.0)
    assert erc.contamination_scale("judge_1_10", 1.0, 10.0) == (9.0, 10.0)
    with pytest.raises(ValueError):
        erc.contamination_scale("unbounded", -np.inf, np.inf)


# ---------------------------------------------------------------------------
# --workers: the serial results
# ---------------------------------------------------------------------------

def _two_datasets(data):
    """syn_a (sparse: EM) and syn_b (fully observed: pairwise)."""
    _write(data, "syn_a", *_scores(M=60, N=8, miss=0.25, seed=21))
    _write(data, "syn_b", *_scores(M=50, N=7, seed=22))


def _npz_arrays(path):
    with np.load(path) as d:
        return {key: d[key] for key in d.files}


def _assert_same_arrays(a, b):
    assert list(a) == list(b)
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert x.dtype == y.dtype and x.shape == y.shape, key
        assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), key


def _run_split_checks(logs, workers, mono_alone=False):
    """clip, contam and mono (sharing clip's split estimates), or mono
    alone with an empty cache: (results, output files)."""
    erc._CORR_CACHE.clear()
    out = {}
    if not mono_alone:
        out["clip"] = erc.check_clip(["syn_a", "syn_b"], n_repeats=2,
                                     n_folds=3, csv_prefix=logs / "rc",
                                     k_max=6, workers=workers)
        out["contam"] = erc.check_contam("syn_a", 3, 0, workers=workers)
    out["mono"] = erc.check_mono(["syn_a", "syn_b"], n_repeats=2, n_folds=3,
                                 csv_prefix=logs / "rc", workers=workers)
    files = {p.name: (_npz_arrays(p) if p.suffix == ".npz"
                      else p.read_text())
             for p in sorted(logs.glob("rc_*"))}
    for p in logs.glob("rc_*"):
        p.unlink()
    erc._CORR_CACHE.clear()
    return erc._jsonable(out), files


@pytest.mark.parametrize("mono_alone", [False, True])
def test_workers_give_the_serial_results(dirs, capsys, mono_alone):
    """--workers 2: clip and mono evaluate the splits of both datasets,
    contam its contaminated estimates, in spawned processes; every value
    of the results (keys in the same order), every npz array, every CSV
    and every printed line equals the serial run's.  mono gets the clip
    workers' estimates, or estimates them itself when run alone."""
    data, logs = dirs
    _two_datasets(data)
    res1, files1 = _run_split_checks(logs, 1, mono_alone)
    out1 = capsys.readouterr().out
    res2, files2 = _run_split_checks(logs, 2, mono_alone)
    out2 = capsys.readouterr().out
    assert json.dumps(res1) == json.dumps(res2)
    assert list(files1) == list(files2) and files1
    for name in files1:
        if name.endswith(".npz"):
            _assert_same_arrays(files1[name], files2[name])
        else:
            assert files1[name] == files2[name], name
    assert out1 == out2
    assert ("CHECK 1" in out1) != mono_alone and "CHECK 3" in out1


def test_cli_workers_give_the_serial_outputs(tmp_path):
    """The CLI (--check all) with --workers 2 writes the JSON, paired CSVs
    and npz arrays of --workers 1 and prints the same lines."""
    data = tmp_path / "data"
    data.mkdir()
    _two_datasets(data)
    env = _cli_env(tmp_path, data)
    logs = tmp_path / "out" / "logs"
    runs = {}
    for w in (1, 2):
        out = subprocess.run(
            [sys.executable, str(CODE / "eval_reviewer_checks.py"),
             "--datasets", "syn_a", "syn_b", "--repeats", "1", "--folds",
             "3", "--contam-dataset", "syn_a", "--contam-draws", "2",
             "--contam-min-obs", "3", "--tag", f"w{w}", "--workers", str(w)],
            cwd=CODE, env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        files = {p.name.replace(f"_w{w}", ""): p
                 for p in sorted(logs.glob(f"reviewer_checks_w{w}*"))}
        runs[w] = (out.stdout.replace(f"_w{w}", ""), files)
    (out1, files1), (out2, files2) = runs[1], runs[2]
    assert out1 == out2
    assert list(files1) == list(files2)
    assert {"reviewer_checks.json", "reviewer_checks_clip_syn_a.npz",
            "reviewer_checks_mono_syn_b_paired.csv"} <= set(files1)
    for name, p in files1.items():
        if name.endswith(".npz"):
            _assert_same_arrays(_npz_arrays(p), _npz_arrays(files2[name]))
        else:
            assert p.read_bytes() == files2[name].read_bytes(), name
    bad = subprocess.run(
        [sys.executable, str(CODE / "eval_reviewer_checks.py"), "--check",
         "contam", "--contam-dataset", "syn_a", "--workers", "0"],
        cwd=CODE, env=env, capture_output=True, text=True, timeout=60)
    assert bad.returncode != 0 and "--workers" in bad.stderr
