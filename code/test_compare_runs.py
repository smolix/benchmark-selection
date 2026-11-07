"""Tests for compare_runs.py --arrays on logs that gained a method.

When facility location joined the scripts (2026-10-19) some arrays grew
along a method, pair or objective axis (eval_estimator_ablation.py's
methods axis with facloc appended after random, eval_prospective.py's
boot_pairs, eval_token_costs.py's objectives; the synthetic logs below
insert facloc before random, the harder case).  compare_runs pairs such an
array with the old one by the names of a label array of both logs
(status extended when the old entries are unchanged, differs when not,
shape_mismatch when no label array names the axis) and --fail-on-diff
passes on extensions.  Synthetic logs in a scratch root; well under a
second.
"""

from __future__ import annotations

import numpy as np
import pytest

import compare_runs

S = 6                                     # split rows


def _logs():
    """(old, new) npz dicts: new adds facloc (before random) on the method
    axis, a boot pair and an objective; everything else is unchanged."""
    rng = np.random.default_rng(0)
    old_m = ["mi", "entropy", "trace", "random"]
    new_m = ["mi", "entropy", "trace", "facloc", "random"]
    r2 = rng.normal(size=(S, 4, 3))
    r2_new = np.empty((S, 5, 3))
    r2_new[:, [0, 1, 2, 4]] = r2
    r2_new[:, 3] = rng.normal(size=(S, 3))
    # label lengths: methods 4 -> 5, boot_pairs 3 -> 4, objectives 2 -> 3
    pairs = np.array([["mi", "random"], ["mi", "trace"],
                      ["trace", "random"]])
    pairs_new = np.array([["mi", "random"], ["mi", "trace"],
                          ["trace", "random"], ["facloc", "random"]])
    boot = rng.normal(size=(3, 7))
    objs, objs_new = np.array(["mi", "trace"]), np.array(["mi", "trace",
                                                           "facloc"])
    mism = rng.random((4, 2)) < 0.5
    common = {"split_id": np.arange(S), "rep": np.zeros(S, dtype=int),
              "x_r2": rng.normal(size=(S, 5)),
              "unnamed": rng.normal(size=(S, 7))}
    # a method list that grows to exactly the number of split rows (as
    # eval_cost_budget.py's 8 -> 10 methods in a 10-split run)
    cost_m = ["knap_mi", "knap_trace", "cheap_first", "random"]
    old = dict(common, methods=np.array(old_m), r2=r2, boot_pairs=pairs,
               boot_diff=boot, objectives=objs, knap_direct_mismatch=mism,
               cost_methods=np.array(cost_m))
    new = dict(common, methods=np.array(new_m), r2=r2_new,
               cost_methods=np.array(cost_m + ["knap_facloc",
                                               "card_facloc"]),
               boot_pairs=pairs_new,
               boot_diff=np.vstack([boot, rng.normal(size=(1, 7))]),
               objectives=objs_new,
               knap_direct_mismatch=np.hstack([mism,
                                               np.zeros((4, 1), bool)]),
               unnamed=np.hstack([common["unnamed"], np.ones((S, 1))]),
               facloc_r2=rng.normal(size=(S, 5)))
    return old, new


def _roots(tmp_path, old, new):
    for tag, d in (("old", old), ("new", new)):
        (tmp_path / tag / "logs").mkdir(parents=True)
        np.savez(tmp_path / tag / "logs" / "est.npz", **d)
    return tmp_path / "old", tmp_path / "new"


def _status(rows):
    return {r["key"]: r for r in rows}


class TestNamedAxes:
    def test_extended(self, tmp_path):
        o, n = _roots(tmp_path, *_logs())
        rows, old_only, new_only = compare_runs.diff_array_runs(o, n)
        st = _status(rows)
        with np.load(n / "logs" / "est.npz") as d:
            assert d["cost_methods"].size == S    # new length = split rows
        for key in ("methods", "r2", "boot_pairs", "boot_diff", "objectives",
                    "knap_direct_mismatch", "cost_methods"):
            assert st[key]["status"] == "extended", (key, st[key])
            assert "facloc" in st[key]["detail"]
        assert "named by methods" in st["r2"]["detail"]
        assert st["r2"]["split_axis"] == 0 and st["r2"]["n_rows"] == S
        assert "named by boot_pairs" in st["boot_diff"]["detail"]
        assert "new adds facloc/random" in st["boot_diff"]["detail"]
        assert "named by objectives" in st["knap_direct_mismatch"]["detail"]
        # a label array is named by itself first
        assert "named by methods" in st["methods"]["detail"]
        assert st["x_r2"]["status"] == "equal"
        assert st["facloc_r2"]["status"] == "only_new"
        # no label array of lengths 7 / 8: not compared by name
        assert st["unnamed"]["status"] == "shape_mismatch"
        assert old_only == [] and new_only == []

    def test_changed_old_entry_differs(self, tmp_path):
        old, new = _logs()
        new["r2"] = new["r2"].copy()
        new["r2"][2, 4, 1] += 1e-6                # random's row of split 2
        o, n = _roots(tmp_path, old, new)
        st = _status(compare_runs.diff_array_runs(o, n)[0])
        assert st["r2"]["status"] == "differs"
        assert st["r2"]["n_rows_diff"] == 1
        assert st["r2"]["first_split_id"] == 2
        assert st["r2"]["max_abs_delta"] == pytest.approx(1e-6, rel=1e-6)

    def test_reordered_labels_are_not_an_extension(self, tmp_path):
        old, new = _logs()
        new["methods"] = np.array(["entropy", "mi", "trace", "facloc",
                                   "random"])
        o, n = _roots(tmp_path, old, new)
        st = _status(compare_runs.diff_array_runs(o, n)[0])
        assert st["methods"]["status"] == "shape_mismatch"
        assert st["r2"]["status"] == "shape_mismatch"

    def test_fail_on_diff_passes_extensions(self, tmp_path, monkeypatch):
        old, new = _logs()
        new["unnamed"] = old["unnamed"]
        o, n = _roots(tmp_path, old, new)
        monkeypatch.chdir(tmp_path)
        compare_runs.main([str(o), str(n), "--arrays", "--fail-on-diff"])
        new["x_r2"] = new["x_r2"] + 1.0
        o2, n2 = _roots(tmp_path / "b", old, new)
        with pytest.raises(SystemExit) as exc:
            compare_runs.main([str(o2), str(n2), "--arrays",
                               "--fail-on-diff"])
        assert exc.value.code == 1
