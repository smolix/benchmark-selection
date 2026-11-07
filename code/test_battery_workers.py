"""--workers of the standard battery's split loops (synthetic, < 90 s).

eval_entropy_vs_mi, eval_greedy_all, plot_mi_selection_all, eval_baselines
and eval_benchpress evaluate their matched splits in W spawned processes
with --workers W (eval_entropy_vs_mi.map_splits).  Checked here on two
small datasets, a sparse one with a [0, 1] score range (rule D2) and an
item group (eval_baselines' redundancy report), and a fully observed one
whose one infinite score makes the correlation estimate fail on every
split that trains on that model (the failed-split rows):

  * per script, run_dataset with workers=2 returns the serial run's
    results bit for bit (every key, value, shape and dtype; the selection
    logs and paired rows too) and prints the serial run's stdout apart
    from the timings;
  * map_splits yields in split order, and its serial path evaluates a
    split only when its result is taken (so a holdout's progress lines
    follow its header);
  * the command lines with --workers 2 write the files of --workers 1:
    every npz key, JSON value and CSV row, and the same stdout apart from
    the timings and the output root.
"""

import atexit
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

# Outputs of the imported eval scripts (and of their worker processes,
# which inherit the environment) go to a scratch root unless the caller
# (run_all.sh's tests job) set one
if not os.environ.get("BENCHSELECT_EXPERIMENT_ROOT"):
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-battery-workers-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

import eval_baselines as eb
import eval_benchpress as ebp
import eval_entropy_vs_mi as emi
import eval_greedy_all as ega
import plot_mi_selection_all as pmi
from cv_split import cv_splits
from score_bounds import dataset_bounds

CODE = pathlib.Path(__file__).resolve().parent
DATASETS = ["toy", "bad"]
# bad's infinite score: NaN column statistics before the estimate fails
pytestmark = pytest.mark.filterwarnings(
    "ignore:invalid value encountered:RuntimeWarning")
# The timings of the progress and completion lines
TIMING = re.compile(r"\(\d+\.\d+s\)|(completed in |Total time: )\d+s")


# ── helpers ──────────────────────────────────────────────────────

def _write_datasets(d, M=60, N=6):
    """toy: sparse scores in (0, 1), valid range [0, 1]; benchmark 5 has
    three models only (ineligible where one of them is held out); t1 and
    t2 share an item group.  bad: fully observed, the score of model 7 on
    t2 infinite, so the estimate fails (LinAlgError) on every split that
    trains on model 7 and succeeds on the others."""
    rng = np.random.default_rng(0)
    names = [f"t{j}" for j in range(N)]
    models = np.array([f"m{i}" for i in range(M)])
    for ds, full in (("toy", False), ("bad", True)):
        z = rng.normal(size=(M, 2))
        X = 1 / (1 + np.exp(-(z @ rng.normal(size=(2, N))
                              + 0.3 * rng.normal(size=(M, N)))))
        if full:
            O = np.ones((M, N))
            X[7, 2] = np.inf
            meta = {"score_bounds": {"default": [None, None]}}
        else:
            O = (rng.random((M, N)) < 0.85).astype(np.float64)
            O[:, 0] = 1
            O[3:, 5] = 0
            meta = {"score_bounds": {"default": [0.0, 1.0]},
                    "benchmark_names": names,
                    "benchmarks": {"item_group": ["a", "g", "g", "b", "c",
                                                  "d"]}}
        np.savez(d / f"{ds}.matrix.npz", B=np.where(O > 0, X, 0.0), O=O,
                 model_names=models, benchmark_names=np.array(names))
        (d / f"{ds}.meta.json").write_text(json.dumps(meta))


def _same(a, b, where="."):
    """a and b equal bit for bit: same types, dict keys in the same order,
    arrays of the same shape and dtype (NaN == NaN), floats equal or both
    NaN."""
    assert type(a) is type(b), (where, type(a), type(b))
    if isinstance(a, dict):
        assert list(a) == list(b), (where, list(a), list(b))
        for k in a:
            _same(a[k], b[k], f"{where}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (where, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape and a.dtype == b.dtype, (where, a.shape,
                                                           b.shape)
        assert np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), where
    elif isinstance(a, (float, np.floating)):
        assert a == b or (math.isnan(a) and math.isnan(b)), (where, a, b)
    else:
        assert a == b, (where, a, b)


def _run(capsys, fn, *args, **kw):
    """fn's result and its stdout with the timings masked."""
    res = fn(*args, **kw)
    return res, TIMING.sub("T", capsys.readouterr().out)


@pytest.fixture
def data(tmp_path, monkeypatch):
    _write_datasets(tmp_path)
    for mod in (emi, ega, pmi, eb, ebp):
        monkeypatch.setattr(mod, "DATA", tmp_path)
    return tmp_path


# ── map_splits ───────────────────────────────────────────────────

def _emi_ctx(d, ds="toy"):
    z = np.load(d / f"{ds}.matrix.npz")
    B, O, names = z["B"], z["O"], z["benchmark_names"]
    return dict(B=B, O=O, N=B.shape[1], k_max=B.shape[1], names=names,
                bounds=dataset_bounds(ds, names, d),
                methods=list(emi.METHODS))


def test_map_splits_order_and_serial_laziness(data, capsys):
    ctx = _emi_ctx(data)
    splits = list(cv_splits(60, 5, 2, 0.1, 42))
    gen = emi.map_splits(emi.emi_split, ctx, splits, 1)
    assert capsys.readouterr().out == ""        # nothing evaluated yet
    next(gen)
    out = capsys.readouterr().out
    assert out.startswith("  Rep 0 fold 0:") and out.count("\n") == 1
    gen.close()
    # closed early, the parallel path prints what it yielded and shuts down
    gen = emi.map_splits(emi.emi_split, ctx, splits, 2)
    next(gen)
    gen.close()
    out = capsys.readouterr().out
    assert out.startswith("  Rep 0 fold 0:") and out.count("\n") == 1
    serial, out1 = _run(capsys, list,
                        emi.map_splits(emi.emi_split, ctx, splits, 1))
    par, out3 = _run(capsys, list,
                     emi.map_splits(emi.emi_split, ctx, splits, 3))
    assert [dict(p["meta"])["split_id"] for p in par] == \
        [s.split_id for s in splits]
    _same(serial, par)
    assert out1 == out3 and out1.count("\n") == len(splits)


# ── run_dataset, script by script ────────────────────────────────

@pytest.mark.parametrize("ds", DATASETS)
def test_entropy_vs_mi_workers(data, capsys, ds):
    res1, out1 = _run(capsys, emi.run_dataset, ds, 2, 5, 15)
    res2, out2 = _run(capsys, emi.run_dataset, ds, 2, 5, 15, workers=2)
    _same(res1, res2)
    assert out1 == out2
    failed = res1[1]["estimation_failed"]
    assert failed.any() == (ds == "bad") and not failed.all()


@pytest.mark.parametrize("ds", DATASETS)
def test_greedy_cv_workers(data, capsys, ds):
    res1, out1 = _run(capsys, ega.run_dataset, ds, 2, [0.1, 0.9], 15)
    res2, out2 = _run(capsys, ega.run_dataset, ds, 2, [0.1, 0.9], 15,
                      workers=2)
    _same(res1, res2)
    _same(ega.paired_rows(ds, res1[0]), ega.paired_rows(ds, res2[0]))
    assert out1 == out2
    assert res1[0][0.1]["failed"].any() == (ds == "bad")


@pytest.mark.parametrize("ds", DATASETS)
def test_mi_selection_workers(data, capsys, ds):
    res1, out1 = _run(capsys, pmi.run_dataset, ds, 2, (0.1, 0.9))
    res2, out2 = _run(capsys, pmi.run_dataset, ds, 2, (0.1, 0.9),
                      workers=2)
    _same(res1, res2)
    assert out1 == out2
    assert res1[0][0.1]["failed"].any() == (ds == "bad")


@pytest.mark.parametrize("ds", DATASETS)
def test_baselines_workers(data, capsys, ds):
    res1, out1 = _run(capsys, eb.run_dataset, ds, 15, 5, 2)
    res2, out2 = _run(capsys, eb.run_dataset, ds, 15, 5, 2, workers=2)
    _same(res1, res2)
    metrics, out, _, k_max, _ = res1
    _same(eb.paired_rows(ds, metrics, out, k_max),
          eb.paired_rows(ds, res2[0], res2[1], k_max))
    assert out1 == out2
    assert (~out["split_ok"]).any() == (ds == "bad") and out["split_ok"].any()


@pytest.mark.parametrize("ds", DATASETS)
def test_benchpress_workers(data, capsys, ds):
    res1, out1 = _run(capsys, ebp.run_dataset, ds, 6, 5, 2)
    res2, out2 = _run(capsys, ebp.run_dataset, ds, 6, 5, 2, workers=2)
    _same(res1, res2)
    _same(ebp.paired_rows(ds, res1[0], 6), ebp.paired_rows(ds, res2[0], 6))
    assert out1 == out2
    assert res1[0][0.1]["meta"]["failed"].any() == (ds == "bad")


# ── the command lines ────────────────────────────────────────────

CLI = [
    ["eval_entropy_vs_mi.py", "--datasets", "toy", "--repeats", "1"],
    ["eval_greedy_all.py", "--datasets", "toy", "--repeats", "1",
     "--holdouts", "0.1", "0.9", "--selections-json", "none"],
    ["plot_mi_selection_all.py", "--datasets", "toy", "--repeats", "1",
     "--selections-json", "none"],
    ["eval_baselines.py", "--datasets", "toy", "--repeats", "1"],
    ["eval_benchpress.py", "--datasets", "toy", "--repeats", "1"],
]


def _cli_run(data, root, workers):
    env = dict(os.environ, BENCHSELECT_DATA_DIR=str(data),
               BENCHSELECT_EXPERIMENT_ROOT=str(root), MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    for key in ("BENCHSELECT_LOGS_DIR", "BENCHSELECT_FIGURES_DIR",
                "BENCHSELECT_LOGIT_LOGS_DIR", "BENCHSELECT_LOGIT_FIGURES_DIR",
                "BENCHSELECT_DATASETS"):
        env.pop(key, None)
    outs = []
    for c in CLI:
        r = subprocess.run([sys.executable, *c, "--workers", str(workers)],
                           cwd=CODE, env=env, capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, (c, r.stdout[-2000:], r.stderr[-2000:])
        assert "Traceback" not in r.stdout + r.stderr, c
        text = r.stdout.replace(str(root.resolve()), "ROOT")
        outs.append(TIMING.sub("T", text.replace(str(root), "ROOT")))
    return outs


def test_command_lines_workers(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_datasets(data)
    roots = {w: tmp_path / f"w{w}" for w in (1, 2)}
    outs = {w: _cli_run(data, root, w) for w, root in roots.items()}
    for c, o1, o2 in zip(CLI, outs[1], outs[2]):
        assert o1 == o2, c[0]
    files = {w: sorted(p.relative_to(root).as_posix()
                       for p in root.rglob("*") if p.is_file())
             for w, root in roots.items()}
    assert files[1] == files[2]
    logs = [f for f in files[1] if f.startswith("logs/")]
    assert {"logs/entropy_vs_mi_toy.npz", "logs/greedy_cv_toy.npz",
            "logs/greedy_cv_splits_toy.json", "logs/greedy_mi_cv_toy.npz",
            "logs/greedy_mi_cv_splits_toy.json", "logs/baselines_toy.npz",
            "logs/baselines_toy_redundancy.json", "logs/toy_results.npz",
            "logs/toy_results_paired.csv"} <= set(logs)
    for f in logs:
        a, b = roots[1] / f, roots[2] / f
        if f.endswith(".npz"):
            za, zb = np.load(a), np.load(b)
            assert za.files == zb.files, f
            for key in za.files:
                _same(za[key], zb[key], f"{f}:{key}")
        elif f.endswith(".json"):
            _same(json.loads(a.read_text()), json.loads(b.read_text()), f)
        else:
            assert a.read_text().splitlines() == b.read_text().splitlines(), f
