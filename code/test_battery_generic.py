"""Tests of the standard battery on generic datasets (synthetic, < 60 s).

The battery is eval_greedy_all, eval_entropy_vs_mi, plot_mi_selection_all,
eval_baselines, eval_family_holdout, eval_logit and plot_selection_order.
Covers the helpers added for the data-release datasets: the per-k widths
(eval_logit.dataset_k_max, plot_selection_order.figure_k_max), the logit
floor counts, eval_baselines' redundancy report (redundant_groups,
group_picks, hypergeom_multi by brute force), its scored-cell share
(evaluated_share) and the removal of a stale redundancy summary, and an
end-to-end run of every script on a small sparse dataset with N = 6 < 15
benchmarks: per-k arrays of width N (NaN at k = N), the same splits and
picks across scripts (matched design), {m}_cover equal to
evaluated_share recomputed from the logged picks and the split's D1
targets (cv_split.target_mask), the logit no-info
counts, family holdout never training on the held-out family, no
tracebacks or RuntimeWarnings.  Facility location (2026-10-19):
eval_baselines' facloc / facloc_abs / facloc_sq columns, the same facloc
and trace picks in every log that has them, selection-order figures of
facloc, trace, hub and the similarity variants, and plot_selection_order's
log sources (unchanged for entropy, random and mi).
"""

import itertools
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import eval_baselines as eb
import eval_logit as el
import model_families as mf
from cv_split import cv_splits, eligible_mask, target_mask
from plot_selection_order import K_MAX, figure_k_max

CODE = pathlib.Path(__file__).resolve().parent


# ── Helpers ──────────────────────────────────────────────────────

def test_k_max_helpers():
    assert el.dataset_k_max(10) == 10 and el.dataset_k_max(62) == el.K_MAX
    entries = [{"selected_names": list("abcdefghij")},
               {"selected_names": list("abc")}]
    assert figure_k_max(entries) == 10
    assert figure_k_max([{"selected_names": list("x" * 40)}]) == K_MAX
    assert figure_k_max([]) == K_MAX


def test_floor_counts():
    B = np.array([[0.0, 0.5], [0.0005, 1.0], [1.0, 0.0], [2.0, 0.7]])
    O = np.array([[1, 1], [1, 1], [1, 0], [0, 1]], dtype=float)
    n_obs, n_floor = el.floor_counts(B, O)
    # column 0: max 1.0, 0 and 0.0005 at or below EPS; column 1: none
    assert n_obs.tolist() == [3, 3] and n_floor.tolist() == [2, 0]


def test_redundant_groups_and_picks(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "DATA", tmp_path)
    names = np.array(["ifeval/a", "ifeval/b", "bbh/x", "gpqa/d", "gpqa/m",
                      "solo"])
    meta = {"benchmark_names": ["bbh/x", "ifeval/a", "ifeval/b", "gpqa/d",
                                "gpqa/m", "solo", "absent"],
            "benchmarks": {"item_group": ["bbh_x", "ife", "ife", "gp", "gp",
                                          "solo", "solo"]}}
    (tmp_path / "toy.meta.json").write_text(json.dumps(meta))
    groups = eb.redundant_groups("toy", names)
    assert groups == [("ifeval", [0, 1]), ("gpqa", [3, 4])]
    assert eb.redundant_groups("nometa", names) == []
    S, k_max = 3, 4
    out = {"split_id": np.arange(S), "split_ok": np.array([True, True, False]),
           "pca_selected_per_k": np.full((S, k_max, k_max), -1)}
    for m in eb.METHODS:
        out[f"{m}_selected"] = np.array([[0, 1, 3, 2], [2, 5, 0, -1],
                                         [0, 1, 3, 4]])
    out["pca_selected_per_k"][0] = [[0, -1, -1, -1], [3, 4, -1, -1],
                                    [0, 1, 2, -1], [0, 1, 3, 4]]
    picks = eb.group_picks(out, groups, k_max)
    e = picks["entropy"]
    assert e[0, 0].tolist() == [1, 2, 2, 2] and e[0, 1].tolist() == [0, 0, 1, 1]
    assert e[1, 0].tolist() == [0, 0, 1, -1]        # no size-4 selection
    assert (e[2] == -1).all()                       # failed split
    p = picks["pca"]
    assert p[0, 0].tolist() == [1, 0, 2, 2] and p[0, 1].tolist() == [0, 2, 0, 2]
    assert (p[1] == -1).all()                       # no pca sets stored


def test_hypergeom_multi_bruteforce():
    for n, g, k in [(6, 2, 3), (7, 3, 4), (8, 4, 5), (5, 2, 5)]:
        cnt = sum(1 for c in itertools.combinations(range(n), k)
                  if sum(1 for j in c if j < g) > 1)
        assert math.isclose(eb.hypergeom_multi(n, g, k),
                            cnt / math.comb(n, k), rel_tol=1e-12)
    assert np.isnan(eb.hypergeom_multi(4, 2, 5))


def test_evaluated_share():
    O = np.array([[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 1]], dtype=float)
    # selection {0}: models 0 and 2 scored; their unselected observed cells
    # 2 + 1 of 2 + 3 + 1
    assert eb.evaluated_share(O, [0]) == pytest.approx(3 / 6)
    assert eb.evaluated_share(np.ones((3, 4)), [1, 2]) == 1.0
    assert np.isnan(eb.evaluated_share(np.zeros((2, 3)), [0]))


# ── End to end, N = 6 ────────────────────────────────────────────

def _toy_dataset(d, M=60, N=6, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(M, 2))
    W = rng.normal(size=(2, N))
    X = 1 / (1 + np.exp(-(z @ W + 0.3 * rng.normal(size=(M, N)))))
    O = (rng.random((M, N)) < 0.85).astype(np.float64)
    O[:, 0] = 1
    O[3:, 5] = 0                       # benchmark 5: three models only
    B = X * O
    fams = ["llama", "qwen", "mistral", "gemma", "phi", "falcon"]
    names = np.array([f"org{i % 7}/{fams[i % 6]}-{i}b" for i in range(M)])
    np.savez(d / "toy.matrix.npz", B=B, O=O, model_names=names,
             benchmark_names=np.array([f"t{j}" for j in range(N)]))
    # valid score ranges of the synthetic data (score_bounds, rule D2)
    (d / "toy.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [None, None]}}))
    return B, O, names


def test_small_n_end_to_end(tmp_path):
    B, O, names = _toy_dataset(tmp_path)
    out = tmp_path / "out"
    logs = out / "logs"
    logs.mkdir(parents=True)
    stale = logs / "baselines_toy_redundancy.json"
    stale.write_text("{}")             # from an earlier run: must go
    env = dict(os.environ, BENCHSELECT_DATA_DIR=str(tmp_path),
               BENCHSELECT_EXPERIMENT_ROOT=str(out),
               MPLBACKEND="Agg", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    for key in ("BENCHSELECT_LOGS_DIR", "BENCHSELECT_FIGURES_DIR",
                "BENCHSELECT_LOGIT_LOGS_DIR", "BENCHSELECT_LOGIT_FIGURES_DIR",
                "BENCHSELECT_DATASETS"):
        env.pop(key, None)
    cmds = [
        ["eval_greedy_all.py", "--datasets", "toy", "--repeats", "1",
         "--holdouts", "0.1", "0.9", "--selections-json", "none"],
        ["eval_entropy_vs_mi.py", "--datasets", "toy", "--repeats", "1"],
        ["eval_baselines.py", "--datasets", "toy", "--repeats", "1"],
        ["plot_mi_selection_all.py", "--datasets", "toy", "--repeats", "1",
         "--selections-json", "none"],
        ["eval_family_holdout.py", "--datasets", "toy", "--repeats", "2",
         "--min-family", "8"],
        ["eval_logit.py", "--datasets", "toy", "--repeats", "1",
         "--no-eigenspectrum"],
        ["plot_selection_order.py", "--datasets", "toy", "--repeats", "1",
         "--methods", "entropy", "random", "mi", "facloc", "trace", "hub",
         "facloc_abs", "facloc_sq"],
    ]
    for c in cmds:
        r = subprocess.run([sys.executable, *c], cwd=CODE, env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, (c, r.stdout[-2000:], r.stderr[-2000:])
        assert "Traceback" not in r.stdout + r.stderr, c
        assert "RuntimeWarning" not in r.stderr, (c, r.stderr[-1500:])
    assert not stale.exists()
    N = 6
    g = np.load(logs / "greedy_cv_toy.npz")
    assert g["p10_r2"].shape == (10, N) and int(g["k_max"]) == N
    assert np.isnan(g["p10_r2"][:, N - 1]).all()          # nothing to impute
    assert np.isfinite(g["p10_r2"][:, :N - 2]).all()
    e = np.load(logs / "entropy_vs_mi_toy.npz")
    assert e["mi_r2"].shape == (10, N)
    b = np.load(logs / "baselines_toy.npz")
    assert b["trace_r2"].shape == (10, N) and int(b["k_max"]) == N
    m = np.load(logs / "greedy_mi_cv_toy.npz")
    assert m["p10_sel"].shape == (10, N) and int(m["k_max"]) == N
    # the same splits / picks across scripts (matched design)
    assert (m["p10_sel"] == e["mi_selected"]).all()
    assert (g["p10_sel"] == e["entropy_selected"]).all()
    assert (b["mi_selected"] == e["mi_selected"]).all()
    assert np.array_equal(b["mi_r2"], e["mi_r2"], equal_nan=True)

    # {m}_cover = evaluated_share of the logged picks on the same splits
    splits = list(cv_splits(len(names), 10, 1, 0.1, 42))
    for meth in eb.METHODS:
        cov = b[f"{meth}_cover"]
        assert cov.shape == (10, N)
        for i, s in enumerate(splits):
            for k in range(1, N + 1):
                sel = (b["pca_selected_per_k"][i, k - 1, :k] if meth == "pca"
                       else b[f"{meth}_selected"][i, :k])
                if (sel < 0).any():
                    assert np.isnan(cov[i, k - 1])
                    continue
                share = eb.evaluated_share(O[s.val_idx], sel.tolist(),
                                           target_mask(O[s.train_idx]))
                assert np.array_equal(cov[i, k - 1], share, equal_nan=True)
    assert np.nanmin(b["entropy_cover"]) < 1.0             # sparse data

    # family holdout: the held-out family is never in training
    f = np.load(logs / "family_holdout_toy.npz")
    assert f["entropy_r2"].shape[1] == N
    fams = list(f["families"])
    assert fams and all(fam != "other" for fam in fams)
    lab = np.array(mf.assign_families(names, "toy"))
    assert f["model_family"].tolist() == lab.tolist()
    for i, fam in enumerate(fams):
        assert int(f["n_val"][i]) == int((lab == fam).sum())
        train = lab != fam
        el_mask = eligible_mask(O[train])
        assert f["eligible"][i].tolist() == el_mask.tolist()

    # logit: per-k width N and the no-info cell counts per split
    lg = np.load(out / "logs_logit" / "greedy_cv_toy_logit.npz")
    assert lg["p10_r2"].shape == (10, N) and int(lg["k_max"]) == N
    for p in (0.1, 0.9):
        pre = f"p{int(p * 100)}"
        expect = []
        for s in cv_splits(len(names), 10, 1, p, 42):
            ni = O[s.train_idx].sum(axis=0) < el.NO_INFO_MIN_OBS
            expect.append(int(O[s.val_idx][:, ni].sum()))
        assert lg[f"{pre}_n_no_info_cells"].tolist() == expect
    assert lg["p90_n_no_info_cells"].sum() > 0              # benchmark 5
    le = np.load(out / "logs_logit" / "entropy_vs_mi_toy_logit.npz")
    assert le["n_no_info_cells"].shape == (10,)

    figs = out / "figures"
    for name in ["greedy_cv_toy.pdf", "entropy_vs_mi_toy.pdf",
                 "baselines_toy.pdf", "selection_order_toy.pdf",
                 "selection_order_mi_toy.pdf", "family_holdout_toy.pdf",
                 "selection_order_facloc_toy.pdf",
                 "selection_order_trace_toy.pdf",
                 "selection_order_hub_toy.pdf",
                 "selection_order_facloc_abs_toy.pdf",
                 "selection_order_facloc_sq_toy.pdf"]:
        assert (figs / name).stat().st_size > 0, name

    # facility location and its similarity ablation (2026-10-19): the
    # same picks wherever a script logs them on the matched splits
    for meth in ("facloc", "facloc_abs", "facloc_sq"):
        assert b[f"{meth}_selected"].shape == (10, N)
        assert b[f"{meth}_r2"].shape == (10, N)
    assert list(b["methods"][-3:]) == ["facloc", "facloc_abs", "facloc_sq"]
    for meth, pre in (("facloc", "facloc_"), ("trace", "trace_")):
        if f"{meth}_selected" in e.files:          # eval_entropy_vs_mi.py
            assert (e[f"{meth}_selected"] == b[f"{meth}_selected"]).all()
        if f"p10_{pre}sel" in g.files:              # eval_greedy_all.py
            assert (g[f"p10_{pre}sel"] == b[f"{meth}_selected"]).all()


def test_selection_order_sources(tmp_path, monkeypatch):
    """plot_selection_order's log sources: those of entropy, random and mi
    are unchanged by the facility-location methods; trace and facloc read
    the entropy / random sources (eval_greedy_all.py's trace_ / facloc_
    keys and entries), then baselines_{ds}.npz; hub, facloc_abs and
    facloc_sq only baselines_{ds}.npz.  A split log without the method's
    entries falls through to the next source."""
    import plot_selection_order as pso
    monkeypatch.setattr(pso, "LOGS", tmp_path)
    names = {m: [p.name for p, _ in pso.selection_sources("toy", m)]
             for m in pso.METHODS}
    assert names["entropy"] == [
        "greedy_cv_splits_toy.json", "greedy_cv_toy.npz", "toy_results.npz",
        "entropy_vs_mi_toy.npz", "toy_results.npz",
        "greedy_cv_selections.json"]
    assert names["random"] == names["entropy"][:-1]
    assert names["mi"] == [
        "greedy_mi_cv_splits_toy.json", "greedy_mi_cv_toy.npz",
        "entropy_vs_mi_toy.npz", "toy_results.npz",
        "greedy_mi_cv_selections_all.json"]
    for m in ("facloc", "trace"):
        assert names[m] == names["random"] + ["baselines_toy.npz"]
    for m in ("hub", "facloc_abs", "facloc_sq"):
        assert names[m] == ["baselines_toy.npz"]
    # a split log of entropy only: facloc comes from baselines_toy.npz
    (tmp_path / "greedy_cv_splits_toy.json").write_text(json.dumps([
        {"dataset": "toy", "holdout_frac": 0.1, "rep": 0, "fold": 0,
         "split_id": 0, "method": "entropy", "selected_idx": [1, 0],
         "selected_names": ["t1", "t0"], "index_ties": []}]))
    S = 2
    np.savez(tmp_path / "baselines_toy.npz",
             benchmark_names=np.array(["t0", "t1", "t2"]),
             facloc_selected=np.array([[2, 0, -1], [0, 2, 1]]),
             facloc_index_ties=np.zeros((S, 3), dtype=np.int64),
             split_rep=np.zeros(S, dtype=np.int64),
             split_fold=np.arange(S), split_id=np.arange(S),
             n_eligible=np.array([2, 3]), holdout_frac=0.1)
    entries, path, n_dropped = pso.load_selections("toy", "facloc",
                                                   repeats=1)
    assert path.name == "baselines_toy.npz" and n_dropped == 0
    assert [e["selected_names"] for e in entries] == [["t2", "t0"],
                                                      ["t0", "t2", "t1"]]
    entries, path, _ = pso.load_selections("toy", "entropy", repeats=1)
    assert path.name == "greedy_cv_splits_toy.json"
