"""Tests for make_numbers.py's facility-location wiring (2026-10-19).

Facility location (facloc) is the paper's headline method: every script
that compares selection methods logs it next to the others.  Checks, on
synthetic logs in a scratch root, that

  * the per-split selection log of eval_greedy_all.py gives Sel macros for
    trace and facloc (SEL_SPLIT_LOGS) at the selection-order figures'
    holdout only (SEL_HOLDOUTS), and appending their entries leaves every
    entropy Sel macro unchanged;
  * eval_baselines.py's similarity ablation gets paired macros
    (PairDiffBase{Ds}FaclocVsFaclocAbsK{k}, ...: paired_stats.MACRO_PAIRS)
    equal to the paired differences, facloc's pairs read X - facloc and
    facloc - random, and adding methods to a log only adds macros (every
    macro of the smaller log keeps its value, kind and source).

A second or two.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import make_numbers
import paired_stats

NAMES = [f"b{j}" for j in range(8)]


def _entries(method, n_folds=4, holdouts=(0.1, 0.5), seed=0):
    """Per-split selection entries in eval_greedy_all.py's layout."""
    rng = np.random.default_rng(seed)
    out = []
    for p in holdouts:
        for fold in range(n_folds):
            order = [NAMES[i] for i in rng.permutation(len(NAMES))[:5]]
            out.append({"dataset": "toy", "holdout_frac": p, "fold": fold,
                        "rep": 0, "split_id": fold, "n_eligible": 8,
                        "method": method, "selected_idx": [],
                        "selected_names": order, "tie_breaks": [],
                        "index_ties": [[1, 2]] if fold == 0 else []})
    return out


def _extract(root):
    return make_numbers.extract(root, data_dir=root / "data", eigen="skip",
                                datasets=["toy"]).records


def _write_splits(root, entries):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "greedy_cv_splits_toy.json").write_text(json.dumps(entries))


class TestSelectionMacros:
    def test_trace_and_facloc_sel_macros(self, tmp_path):
        old = _entries("entropy") + _entries("random", seed=1)
        new = old + _entries("trace", seed=2) + _entries("facloc", seed=3)
        _write_splits(tmp_path / "old", old)
        _write_splits(tmp_path / "new", new)
        a, b = _extract(tmp_path / "old"), _extract(tmp_path / "new")
        assert not any(k.startswith(("SelToyFacloc", "SelToyTrace"))
                       for k in a)
        ent = [k for k in a if k.startswith("SelToyEntropy")]
        assert ent and all(k in b and b[k]["value"] == a[k]["value"]
                           and b[k]["kind"] == a[k]["kind"] for k in ent)
        assert set(a) <= set(b)
        fac = [e for e in new if e["method"] == "facloc"]
        assert b["SelToyFaclocPtenFoldZero"]["value"] == \
            ", ".join(fac[0]["selected_names"])
        assert b["SelToyFaclocPtenNSplits"]["value"] == 4
        assert b["SelToyFaclocPtenIndexTies"]["value"] == 1
        assert b["SelToyTracePtenNSplits"]["value"] == 4
        # trace and facloc: the figures' holdout only (SEL_HOLDOUTS);
        # entropy: every holdout, as before
        assert not any(k.startswith(("SelToyFaclocPfifty",
                                     "SelToyTracePfifty")) for k in b)
        assert b["SelToyEntropyPfiftyNSplits"]["value"] == 4
        # random entries give no Sel macros (as before)
        assert not any(k.startswith("SelToyRandom") for k in b)


def _baselines(methods, S=6, k_max=15, seed=0):
    """A baselines_{ds}.npz dict: one (S, k_max) R² array per method
    (each method's values depend on its name only) and the split rows."""
    out = {"methods": np.array(methods),
           "split_rep": np.zeros(S, dtype=int),
           "split_fold": np.arange(S), "split_id": np.arange(S),
           "n_train": np.full(S, 90), "n_val": np.full(S, 10),
           "n_eligible": np.full(S, 20), "benchmark_names": np.array(NAMES)}
    for m in methods:
        rng = np.random.default_rng(seed + sum(map(ord, m)))
        out[f"{m}_r2"] = rng.uniform(0.2, 0.9, size=(S, k_max))
        out[f"{m}_index_ties"] = np.zeros((S, k_max), dtype=int)
    return out


class TestPairedMacros:
    CORE = ["entropy", "mi", "random", "hub", "pca", "trace", "facloc"]

    def _root(self, base, methods):
        logs = base / "logs"
        logs.mkdir(parents=True)
        np.savez(logs / "baselines_toy.npz", **_baselines(methods))
        return base

    def test_ablation_pairs_and_additions_only(self, tmp_path):
        old = self._root(tmp_path / "old", self.CORE)
        new = self._root(tmp_path / "new",
                         self.CORE + ["facloc_abs", "facloc_sq"])
        a, b = _extract(old), _extract(new)
        # adding methods only adds macros
        assert set(a) <= set(b)
        for k, r in a.items():
            assert (b[k]["value"], b[k]["kind"], b[k]["source_key"]) == \
                (r["value"], r["kind"], r["source_key"]), k
        d = _baselines(self.CORE + ["facloc_abs", "facloc_sq"])
        for var, V in (("facloc_abs", "FaclocAbs"), ("facloc_sq", "FaclocSq")):
            for k in paired_stats.K_MACRO:
                name = (f"PairDiffBaseToyFaclocVs{V}"
                        f"K{make_numbers.int_words(k)}")
                diff = d["facloc_r2"][:, k - 1] - d[f"{var}_r2"][:, k - 1]
                assert b[name]["value"] == pytest.approx(diff.mean(),
                                                         abs=1e-15)
                assert name not in a
            assert f"RsqBaseToy{V}Kfive" in b
        # orientation: X - facloc, facloc - random
        for name in ("PairDiffBaseToyMiVsFaclocKfive",
                     "PairDiffBaseToyTraceVsFaclocKfive",
                     "PairDiffBaseToyEntropyVsFaclocKfive",
                     "PairDiffBaseToyFaclocVsRandomKfive"):
            assert name in a and name in b
        assert not any("FaclocVsMi" in k or "FaclocVsTrace" in k for k in b)

    def test_paired_csv_rows_gain_pairs_only(self, tmp_path):
        """numbers/paired_{ds}.csv: the rows of the smaller log are rows of
        the larger one, the new rows all involve the new methods."""
        old = self._root(tmp_path / "old", self.CORE[:-1])
        new = self._root(tmp_path / "new", self.CORE)
        rows = {}
        for tag, root in (("old", old), ("new", new)):
            tables = paired_stats.load_tables(root, ["toy"],
                                              data_dir=root / "data")
            out = paired_stats.write_dataset_csvs(tables, root / "numbers")
            rows[tag] = (out["toy"][0].read_text().splitlines())
        assert rows["old"][0] == rows["new"][0]
        assert set(rows["old"][1:]) <= set(rows["new"][1:])
        extra = set(rows["new"][1:]) - set(rows["old"][1:])
        assert extra and all(",facloc," in r for r in extra)
