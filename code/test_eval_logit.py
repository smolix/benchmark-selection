"""Tests of eval_logit.py's split-level worker processes (--workers).

Synthetic data only (< 60 s): on a sparse dataset (EM estimates; a
benchmark observed by three models, outside the D1 targets of most
splits) and a fully observed one (pairwise estimates), --workers 2 writes
the npz files of --workers 1 (every key, bit for bit, in the same order)
and the same stdout apart from the per-split seconds; the in-process
entry points agree as well when they share no cache (run_greedy_cv
without the entropy-vs-MI part, run_entropy_vs_mi computing its own
splits), and the workers refuse a protocol override of the parent.
"""

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
    _TMP_ROOT = tempfile.mkdtemp(prefix="benchselect-test-logit-")
    atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = _TMP_ROOT

import numpy as np
import pytest

import eval_logit as el

CODE = pathlib.Path(__file__).resolve().parent
LOGS = ("greedy_cv_{}_logit.npz", "entropy_vs_mi_{}_logit.npz")


def _write(d, name, M=50, N=7, seed=0, obs=0.8):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(M, 2))
    W = rng.normal(size=(2, N))
    X = 1 / (1 + np.exp(-(z @ W + 0.3 * rng.normal(size=(M, N)))))
    O = (rng.random((M, N)) < obs).astype(np.float64)
    O[:, 0] = 1
    if obs < 1:
        O[3:, N - 1] = 0               # benchmark N-1: three models only
    np.savez(d / f"{name}.matrix.npz", B=X * O, O=O,
             model_names=np.array([f"m{i}" for i in range(M)]),
             benchmark_names=np.array([f"t{j}" for j in range(N)]))
    (d / f"{name}.meta.json").write_text(json.dumps(
        {"score_bounds": {"default": [None, None]}}))


def _without_seconds(text, root):
    """stdout without its wall-clock fields and with the output root
    replaced (the figure paths)."""
    text = text.replace(str(root), "ROOT")
    return re.sub(r"\(\d+\.\d+s\)|completed in \d+s", "", text)


def _run(tmp_path, workers):
    root = tmp_path / f"root_w{workers}"
    env = dict(os.environ, BENCHSELECT_DATA_DIR=str(tmp_path / "data"),
               BENCHSELECT_EXPERIMENT_ROOT=str(root), MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    for key in ("BENCHSELECT_LOGS_DIR", "BENCHSELECT_FIGURES_DIR",
                "BENCHSELECT_LOGIT_LOGS_DIR", "BENCHSELECT_LOGIT_FIGURES_DIR",
                "BENCHSELECT_DATASETS"):
        env.pop(key, None)
    r = subprocess.run(
        [sys.executable, "eval_logit.py", "--datasets", "toy", "dense",
         "--repeats", "2", "--no-eigenspectrum", "--workers", str(workers)],
        cwd=CODE, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert "Traceback" not in r.stdout + r.stderr
    return root, _without_seconds(r.stdout, root)


def _assert_same(a, b):
    """Two dicts of arrays (npz contents or results): same keys in the
    same order, every array bit for bit."""
    assert list(a) == list(b)
    for key in a:
        u, v = np.asarray(a[key]), np.asarray(b[key])
        assert u.dtype == v.dtype and u.shape == v.shape, key
        assert np.array_equal(u, v, equal_nan=u.dtype.kind == "f"), key


def test_workers_match_the_serial_run(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write(data, "toy")
    _write(data, "dense", N=6, seed=1, obs=1.0)
    root1, out1 = _run(tmp_path, 1)
    root2, out2 = _run(tmp_path, 2)
    for ds in ("toy", "dense"):
        for name in LOGS:
            path = f"logs_logit/{name.format(ds)}"
            with np.load(root1 / path) as x, np.load(root2 / path) as y:
                _assert_same({k: x[k] for k in x.files},
                             {k: y[k] for k in y.files})
    assert out1.count("Rep 1 Fold 9:") == 2 * (len(el.HOLDOUT_FRACS) + 1)
    assert out2 == out1


def test_workers_in_process_without_a_shared_cache(tmp_path, monkeypatch):
    """run_greedy_cv without a cache (no entropy-vs-MI part in its worker
    processes) and run_entropy_vs_mi on its own (its splits in its own
    worker processes) equal their serial runs."""
    _write(tmp_path, "toy", seed=2)
    monkeypatch.setattr(el, "DATA", tmp_path)
    cv1, _ = el.run_greedy_cv("toy", 1)
    cv2, _ = el.run_greedy_cv("toy", 1, workers=2)
    assert list(cv1) == list(cv2) == el.HOLDOUT_FRACS
    for p in el.HOLDOUT_FRACS:
        _assert_same(cv1[p], cv2[p])
    emi1 = el.run_entropy_vs_mi("toy", 1)
    emi2 = el.run_entropy_vs_mi("toy", 1, {}, workers=2)
    for m in el.EMI_METHODS:
        _assert_same(emi1[0][m], emi2[0][m])
    _assert_same(emi1[1], emi2[1])


def test_workers_refuse_a_protocol_override(tmp_path, monkeypatch):
    """A test-only protocol override is module state of this process, which
    a spawned worker does not see: the workers refuse to start instead of
    computing other rows than the serial run."""
    import score_bounds
    _write(tmp_path, "toy", seed=3)
    monkeypatch.setattr(el, "DATA", tmp_path)
    monkeypatch.setattr(score_bounds, "PRED_CLIP", False)
    with pytest.raises(RuntimeError):
        el.run_greedy_cv("toy", 1, workers=2)
