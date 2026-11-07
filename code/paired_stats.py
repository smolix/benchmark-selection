r"""Paired comparisons of selection methods over the matched CV splits.

Every evaluation script runs all of its methods on the same matched
splits (cv_split.cv_splits: n_repeats x n_folds train/validation splits
per holdout fraction, one eligibility mask and one set of common random
numbers per split) and logs one row per split together with the split's
metadata (rep, fold, split_id, n_train, n_val, n_eligible).  This module
pairs two methods a and b split by split and summarizes the per-split
differences d_s = R²_a,s - R²_b,s at each k:

  n_splits          splits where both values are finite (NaN rows: failed
                    covariance estimate, or fewer than k eligible
                    benchmarks); n_reps the repetitions among them
  mean_a, mean_b    means of a and b over those splits
  mean_diff         mean of d
  sd_diff           sample standard deviation of d (ddof = 1)
  test_train_ratio  mean(n_val) / mean(n_train) over those splits
  se_nb             Nadeau-Bengio corrected standard error
                    sqrt(sd_diff² · (1/n + n_val/n_train)), which accounts
                    for the overlap of the training sets of different
                    splits (Nadeau & Bengio 2003; Bouckaert & Frank 2004
                    for repeated CV)
  ci_low, ci_high   95% t-interval mean_diff ± t_{0.975, n-1} · se_nb
  t_nb, p_nb        corrected resampled t statistic mean_diff / se_nb and
                    its two-sided p-value (n - 1 degrees of freedom)
  win_rate, tie_rate, loss_rate   fractions of splits with d > 0, = 0, < 0
  n_wins, n_ties, n_losses        the corresponding counts
se_nb, the interval and the test are NaN for fewer than 2 splits (and t_nb,
p_nb also when every difference is 0).  This is the only implementation:
the eval scripts' own *_paired.csv files (eval_greedy_all, eval_baselines,
eval_benchpress, eval_cost_budget, eval_tabimpute, eval_reviewer_checks)
call paired_stats() and orient pairs with orient() / orient_pair(), and
their columns are their context columns (dataset, holdout_frac, k or
budget, method_a, method_b) followed by STAT_FIELDS, so a pair has the
same sign, values and column names (ci_low, ci_high, ...) everywhere.

Sources (ROOT = --root; {ds} any dataset name; tag = the macro tag of
make_numbers.py; P = one table per holdout fraction):

  logs/entropy_vs_mi_{ds}.npz     tag ''      {m}_r2; rep, fold, split_id
  logs/{ds}_results.npz           tag ''      emi_{m}_r2 (eval_benchpress.py)
                                  tag '', P   cv_p{pct}_{pre}r2
  logs/baselines_{ds}.npz         tag 'Base'  {m}_r2; split_rep, split_fold,
                                              split_id
  logs/greedy_cv_{ds}.npz         tag '', P   p{pct}_{pre}r2
  logs/tabimpute_{ds}.npz +       tag 'Tab'   {m}_r2 of both files, paired
  logs/tabimpute_extra_{ds}.npz               on split_id (one table)
  logs_logit/entropy_vs_mi_{ds}_logit.npz     tag 'Logit'
  logs_logit/greedy_cv_{ds}_logit.npz         tag 'Logit', P

and, written to the CSVs only (no make_numbers.py \Pair macros: these
scripts write their own paired macros into their JSON logs):

  logs/estimator_ablation_{ds}.npz  tag 'Est{S}'  one table per estimator
                                    setting S (setting_macro): its
                                    {setting}__{m}_r2 arrays
  logs/prospective_{tag}.npz        tag 'Ctrl{V}' one table per variant V
                                    (variant_macros) of the random-split
                                    control rows ({m}_r2, rows filtered on
                                    `variant`); dataset = the setting's
                                    data (helm -> helm_lite, epoch_bymodel
                                    -> epoch, olv2sub_math -> olv2sub),
                                    holdout_frac = mean n_val / (n_train +
                                    n_val) of the variant's rows
  logs/imputers_{ds}.npz            tag 'Imp{I}'  one table per imputer I
                                    (Gauss, Soft, Hard): {m}_{imputer}_r2,
                                    whose columns are the k of its `ks`
                                    array (1, 2, 3, 5, 7, 10, 15)

Logs whose second axis is a budget, not k (cost_budget, cost_fixed,
token_costs, token_fixed, item_budget, item_fixed), and the per-family /
per-k-subset layouts (benchfamily_holdout, selection_stability,
analyze_objectives) are not read: their scripts write every paired row to
their own *_paired.csv with the budget / family / comparison columns.

Method key prefixes of the per-holdout layouts come from the file's
methods / method_prefixes arrays (cv_methods / cv_method_prefixes for
{ds}_results.npz), else '' = entropy, 'rand_' = random (the entropy_rf
prefixes 'rf_' / 'entropy_rf_' are gone with entropy_rf, protocol rule
D3).  Logs written before the matched design carry
no split metadata: their rows are taken as repetition 0, fold = split_id
= row, and n_train / n_val are reconstructed from the number of models in
{ds}.matrix.npz (--data-dir) exactly as cv_split.cv_train_val sizes them,
so their pairs still get a corrected standard error.  Tables pair methods
of one file (or of the two TabImpute files); different scripts are not
mixed.

Pairs are oriented by METHOD_ORDER (mi before entropy before trace, pca,
hub, random), so every pair of KEY_PAIRS reads as given (mi - random, mi
- entropy, mi - trace, entropy - random, trace - random; also the
facility-location pairs mi / entropy / trace - facloc, facloc - random).
Method names are ranked by their base (method_rank): a knap_ / card_ prefix
(eval_cost_budget.py) and a _clipped suffix (eval_reviewer_checks.py) are
ignored, other methods (facloc, cheap_first, random_greedy, ...) rank
after hub and before random, and equal ranks keep the given order (knap_mi -
card_mi, mi_clipped - mi).  Facility location (facloc) is the paper's
headline method since 2026-10-19, but stays unranked: its pairs logged
before (eval_baselines.py, eval_reviewer_checks.py) keep their sign and
every script's later facloc rows only add pairs, so read "trace - facloc
= -0.01" as facloc ahead by 0.01.  Ranking it before hub would flip the
hub - facloc and pca - facloc rows of those logs (ranking it first, every
facloc row).

Outputs (CLI): OUT_DIR/paired_{ds}.csv per dataset, one row per (source
file, holdout, k, method pair), columns CSV_FIELDS.  make_numbers.py turns
the MACRO_PAIRS at K_MACRO into \Pair... macros (see its docstring).

Usage:
    python paired_stats.py --root ROOT [--out-dir ROOT/numbers]
                           [--datasets NAME ...] [--reps 0 1 ...]
                           [--pairs all|key] [--data-dir DIR]

--reps restricts every table to the given CV repetitions (e.g. --reps 0
for the rows that reproduce the single-repetition runs).
"""

import argparse
import csv
import math
import os
import pathlib
import re
import sys
from typing import NamedTuple

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir as default_data_dir
from cv_split import cv_splits

SEED = 42                    # base seed of every evaluation script
LEVEL = 0.95
DEFAULT_HOLDOUT = 0.1        # holdout of the single-holdout layouts
K_MACRO = [1, 3, 5, 10, 15]  # k of the make_numbers.py paired macros

# Orientation of every pair: a before b in this order (method_rank: other
# methods between hub and random, ties in the given order).
METHOD_ORDER = ["mi", "entropy", "trace", "pca", "hub", "random"]
METHOD_PREFIXES = ("knap_", "card_")     # eval_cost_budget.py variants
METHOD_SUFFIXES = ("_clipped",)          # eval_reviewer_checks.py clip
KEY_PAIRS = [("mi", "random"), ("mi", "entropy"), ("mi", "trace"),
             ("entropy", "random"), ("trace", "random")]
# Pairs that make_numbers.py writes as macros: the key pairs and facility
# location (facloc, not in METHOD_ORDER, so it ranks between hub and
# random), then facloc against the similarity variants of
# eval_baselines.py (facloc_abs, facloc_sq: equal rank, given order); the
# entropy_rf pairs were dropped with entropy_rf (protocol rule D3).
MACRO_PAIRS = (KEY_PAIRS
               + [("mi", "facloc"), ("entropy", "facloc"),
                  ("trace", "facloc"), ("facloc", "random"),
                  ("facloc", "facloc_abs"), ("facloc", "facloc_sq")])

STAT_FIELDS = ["n_splits", "n_reps", "mean_a", "mean_b", "mean_diff",
               "sd_diff", "test_train_ratio", "se_nb", "ci_low", "ci_high",
               "t_nb", "p_nb", "win_rate", "tie_rate", "loss_rate",
               "n_wins", "n_ties", "n_losses"]
CSV_FIELDS = (["dataset", "source", "tag", "holdout_frac", "k", "method_a",
               "method_b", "reps", "n_rows"] + STAT_FIELDS)
INT_FIELDS = {"n_splits", "n_reps", "n_wins", "n_ties", "n_losses", "k",
              "n_rows"}

# Default key prefixes of the per-holdout (greedy CV) layouts
DEFAULT_CV_PREFIXES = {"": "entropy", "rand_": "random"}


# ── Statistics ───────────────────────────────────────────────────

def paired_stats(a, b, n_train, n_val, rep=None, level=LEVEL):
    """Paired comparison of per-split values a - b over matched splits
    (module docstring); n_train, n_val (and rep, for n_reps) are per-split
    arrays aligned with a and b.  Returns a dict over STAT_FIELDS."""
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    out = dict.fromkeys(STAT_FIELDS, math.nan)
    out.update(n_splits=n, n_reps=0, n_wins=0, n_ties=0, n_losses=0)
    if n == 0:
        return out
    if rep is not None:
        out["n_reps"] = int(np.unique(np.asarray(rep)[ok]).size)
    diff = a[ok] - b[ok]
    ratio = float(np.mean(np.asarray(n_val, dtype=float)[ok])
                  / np.mean(np.asarray(n_train, dtype=float)[ok]))
    mean = float(np.mean(diff))
    n_wins, n_ties = int(np.sum(diff > 0)), int(np.sum(diff == 0))
    out.update(mean_a=float(np.mean(a[ok])), mean_b=float(np.mean(b[ok])),
               mean_diff=mean, test_train_ratio=ratio,
               n_wins=n_wins, n_ties=n_ties, n_losses=n - n_wins - n_ties,
               win_rate=n_wins / n, tie_rate=n_ties / n,
               loss_rate=(n - n_wins - n_ties) / n)
    if n > 1:
        sd = float(np.std(diff, ddof=1))
        se = float(np.sqrt(sd ** 2 * (1.0 / n + ratio)))
        half = float(stats.t.ppf(0.5 + level / 2, n - 1)) * se
        out.update(sd_diff=sd, se_nb=se, ci_low=mean - half,
                   ci_high=mean + half)
        if se > 0:
            t = mean / se
            out.update(t_nb=t, p_nb=float(2 * stats.t.sf(abs(t), n - 1)))
    return out


# ── Split tables ─────────────────────────────────────────────────

class SplitTable(NamedTuple):
    """Per-split values of the methods of one source at one holdout.

    meta maps rep, fold, split_id, n_train, n_val, n_eligible to (S,)
    arrays (n_eligible NaN if not logged); r2 / ties map a method to its
    (S, k) R² / index-ties array (ties None if not logged); keys maps a
    method to (file, key) of its R² array.  legacy: the file has no split
    metadata (rows = rep-0 folds).  n_benchmarks: N of the dataset (None
    if unknown).
    """
    dataset: str
    source: str              # file name(s), '+'-joined for merged tables
    path: pathlib.Path       # first file
    tag: str
    holdout_frac: float
    per_holdout: bool        # macros carry a P{pct} tail
    meta: dict
    r2: dict
    ties: dict
    keys: dict
    legacy: bool
    n_benchmarks: object
    failed: object           # (S,) bool or None
    ks: object = None        # k of each R² column (None: column j = k j+1)
    macros: bool = True      # make_numbers.py \Pair / \Split macros

    @property
    def n_rows(self):
        return len(self.meta["split_id"])

    @property
    def methods(self):
        return list(self.r2)

    def k_values(self):
        """The k of the R² columns (the logged range for every method)."""
        n_col = min((v.shape[1] for v in self.r2.values()), default=0)
        if self.ks is None:
            return list(range(1, n_col + 1))
        return [int(k) for k in self.ks][:n_col]

    def column(self, k):
        """Column index of k, or None if k is not logged."""
        ks = self.k_values()
        return ks.index(int(k)) if int(k) in ks else None

    def restrict(self, reps):
        """The rows of the given repetitions (None: all rows)."""
        if reps is None:
            return self
        keep = np.isin(self.meta["rep"], np.asarray(list(reps), dtype=int))
        return self._replace(
            meta={k: v[keep] for k, v in self.meta.items()},
            r2={m: v[keep] for m, v in self.r2.items()},
            ties={m: (None if v is None else v[keep])
                  for m, v in self.ties.items()},
            failed=None if self.failed is None else self.failed[keep])


def _get(d, pre, *names):
    for name in names:
        if pre + name in d:
            return np.asarray(d[pre + name])
    return None


def _scalar(d, *names, default=None):
    for name in names:
        if name in d:
            try:
                return np.asarray(d[name]).item()
            except (ValueError, TypeError):
                pass
    return default


def legacy_sizes(M, n_folds, holdout_frac, seed=SEED):
    """(n_train, n_val) per fold of the single-repetition CV of M models:
    the rep-0 splits of cv_split.cv_splits, i.e. the folds of logs written
    before the matched design."""
    sizes = [(len(s.train_idx), len(s.val_idx))
             for s in cv_splits(M, n_folds, 1, holdout_frac, seed)]
    n_train, n_val = np.array(sizes, dtype=float).T
    return n_train, n_val


def split_meta(d, pre, n_rows, holdout_frac, n_models=None):
    """(meta, legacy) of the n_rows split rows under key prefix `pre`.
    Recognizes rep / split_rep, fold / split_fold, split_id, n_train,
    n_val / n_test and n_eligible; rows of a log without them are rep-0
    folds, with n_train / n_val from legacy_sizes(n_models) (NaN if the
    number of models is unknown)."""
    rep = _get(d, pre, "rep", "split_rep")
    fold = _get(d, pre, "fold", "split_fold")
    sid = _get(d, pre, "split_id")
    n_train = _get(d, pre, "n_train")
    n_val = _get(d, pre, "n_val", "n_test")
    n_el = _get(d, pre, "n_eligible")
    legacy = rep is None and sid is None
    rows = np.arange(n_rows)
    if legacy:
        rep, fold, sid = np.zeros(n_rows, dtype=int), rows, rows
    else:
        if sid is None:
            n_folds = int(np.max(fold)) + 1 if fold is not None else n_rows
            sid = np.asarray(rep) * n_folds + (rows if fold is None else fold)
        n_folds = _scalar(d, pre + "n_folds", "n_folds")
        if rep is None:
            rep = (np.asarray(sid) // int(n_folds) if n_folds
                   else np.zeros(n_rows, dtype=int))
        if fold is None:
            fold = (np.asarray(sid) % int(n_folds) if n_folds else rows)
    if n_train is None or n_val is None:
        if n_models and legacy:
            n_train, n_val = legacy_sizes(n_models, n_rows, holdout_frac)
        else:
            n_train = np.full(n_rows, np.nan)
            n_val = np.full(n_rows, np.nan)
    meta = {"rep": np.asarray(rep, dtype=int),
            "fold": np.asarray(fold, dtype=int),
            "split_id": np.asarray(sid, dtype=int),
            "n_train": np.asarray(n_train, dtype=float),
            "n_val": np.asarray(n_val, dtype=float),
            "n_eligible": (np.full(n_rows, np.nan) if n_el is None
                           else np.asarray(n_el, dtype=float))}
    for key, v in meta.items():
        if v.shape != (n_rows,):
            raise ValueError(f"{pre}{key} has shape {v.shape}, expected "
                             f"({n_rows},)")
    return meta, legacy


def _failed(d, pre):
    v = _get(d, pre, "estimation_failed", "failed")
    if v is not None:
        return np.asarray(v, dtype=bool)
    ok = _get(d, pre, "split_ok")
    return None if ok is None else ~np.asarray(ok, dtype=bool)


def _n_benchmarks(d):
    if "benchmark_names" in d:
        return int(np.asarray(d["benchmark_names"]).size)
    return None


def _str_list(arr):
    return [str(x) for x in np.asarray(arr).ravel()]


def _suffix_methods(d, pre):
    """Methods with a '{pre}{method}_r2' array: the file's {pre}methods
    order where given, else key order."""
    found = [key[len(pre):-len("_r2")] for key in d
             if key.startswith(pre) and key.endswith("_r2")
             and len(key) > len(pre) + len("_r2")]
    declared = _get(d, pre, "methods")
    if declared is not None:
        order = _str_list(declared)
        found = ([m for m in order if m in found]
                 + [m for m in found if m not in order])
    return found


def suffix_table(d, path, ds, tag, pre="", holdout_frac=None, n_models=None):
    """Table of a '{pre}{method}_r2' layout (entropy_vs_mi, baselines,
    tabimpute, emi_), or None if it has no R² arrays."""
    methods = [m for m in _suffix_methods(d, pre)
               if np.asarray(d[f"{pre}{m}_r2"]).ndim == 2]
    if not methods:
        return None
    if holdout_frac is None:
        holdout_frac = _scalar(d, pre + "holdout_frac", "holdout_frac",
                               default=DEFAULT_HOLDOUT)
    holdout_frac = float(holdout_frac)
    n_rows = np.asarray(d[f"{pre}{methods[0]}_r2"]).shape[0]
    meta, legacy = split_meta(d, pre, n_rows, holdout_frac, n_models)
    r2, ties, keys = {}, {}, {}
    for m in methods:
        arr = np.asarray(d[f"{pre}{m}_r2"], dtype=float)
        if arr.shape[0] != n_rows:
            raise ValueError(f"{pre}{m}_r2 has {arr.shape[0]} rows, "
                             f"expected {n_rows}")
        r2[m] = arr
        t = _get(d, pre, f"{m}_index_ties")
        ties[m] = (np.asarray(t) if t is not None and np.asarray(t).ndim == 2
                   and np.asarray(t).shape[0] == n_rows else None)
        keys[m] = (path.name, f"{pre}{m}_r2")
    return SplitTable(ds, path.name, path, tag, holdout_frac, False, meta,
                      r2, ties, keys, legacy, _n_benchmarks(d),
                      _failed(d, pre))


def _cv_prefixes(d, cv):
    """{key prefix: method} of a per-holdout layout."""
    methods = _get(d, cv, "methods")
    prefixes = _get(d, cv, "method_prefixes")
    if methods is not None and prefixes is not None \
            and np.asarray(methods).size == np.asarray(prefixes).size:
        return dict(zip(_str_list(prefixes), _str_list(methods)))
    return dict(DEFAULT_CV_PREFIXES)


_PCT = re.compile(r"^p(\d+)_")


def cv_tables(d, path, ds, tag, cv="", n_models=None):
    """One table per holdout of a per-holdout layout ('{cv}p{pct}_{pre}r2':
    greedy_cv, {ds}_results.npz cv_, logit greedy_cv)."""
    prefixes = _cv_prefixes(d, cv)
    pcts = []
    for key in d:
        if not key.startswith(cv):
            continue
        m = _PCT.match(key[len(cv):])
        if m and int(m.group(1)) not in pcts:
            pcts.append(int(m.group(1)))
    tables = []
    for pct in pcts:
        hp = f"{cv}p{pct}_"
        found = {}
        for key in d:
            if not (key.startswith(hp) and key.endswith("r2")):
                continue
            pre = key[len(hp):-len("r2")]
            if pre in prefixes and np.asarray(d[key]).ndim == 2:
                found[prefixes[pre]] = (pre, key)
        if not found:
            continue
        # method order: the file's (prefix map) order
        methods = [m for m in dict.fromkeys(prefixes.values()) if m in found]
        n_rows = np.asarray(d[found[methods[0]][1]]).shape[0]
        meta, legacy = split_meta(d, hp, n_rows, pct / 100.0, n_models)
        r2, ties, keys = {}, {}, {}
        for m in methods:
            pre, key = found[m]
            arr = np.asarray(d[key], dtype=float)
            if arr.shape[0] != n_rows:
                raise ValueError(f"{key} has {arr.shape[0]} rows, expected "
                                 f"{n_rows}")
            r2[m] = arr
            t = _get(d, hp, f"{pre}index_ties")
            ties[m] = (np.asarray(t) if t is not None
                       and np.asarray(t).ndim == 2
                       and np.asarray(t).shape[0] == n_rows else None)
            keys[m] = (path.name, key)
        tables.append(SplitTable(ds, path.name, path, tag, pct / 100.0, True,
                                 meta, r2, ties, keys, legacy,
                                 _n_benchmarks(d), _failed(d, hp)))
    return tables


def merge_tables(first, second):
    """One table with the methods of both (first wins on a method both
    have), paired on split_id: the rows common to both, in first's
    order."""
    sid_a, sid_b = first.meta["split_id"], second.meta["split_id"]
    if np.array_equal(sid_a, sid_b):
        ia = ib = np.arange(len(sid_a))
    else:
        common = [s for s in sid_a if s in set(sid_b.tolist())]
        pos_a = {s: i for i, s in enumerate(sid_a.tolist())}
        pos_b = {s: i for i, s in enumerate(sid_b.tolist())}
        ia = np.array([pos_a[s] for s in common], dtype=int)
        ib = np.array([pos_b[s] for s in common], dtype=int)
    r2 = {m: v[ia] for m, v in first.r2.items()}
    ties = {m: (None if v is None else v[ia]) for m, v in first.ties.items()}
    keys = dict(first.keys)
    for m, v in second.r2.items():
        if m not in r2:
            r2[m] = v[ib]
            ties[m] = None if second.ties[m] is None else second.ties[m][ib]
            keys[m] = second.keys[m]
    return first._replace(
        source=f"{first.source}+{second.source}",
        meta={k: v[ia] for k, v in first.meta.items()},
        r2=r2, ties=ties, keys=keys,
        legacy=first.legacy and second.legacy,
        failed=None if first.failed is None else first.failed[ia])


# ── CSV-only tables of the reviewer-request experiments ──────────

# Data set of each eval_prospective.py setting tag (the CSV's dataset)
PROSPECTIVE_DATA = {"mteb2026": "mteb2026", "helm": "helm_lite",
                    "epoch": "epoch", "epoch_bymodel": "epoch",
                    "olv2sub": "olv2sub", "olv2sub_math": "olv2sub"}
IMPUTER_TAGS = {"gaussian": "Gauss", "softimpute": "Soft",
                "hardimpute": "Hard"}
_META_KEYS = ("rep", "fold", "split_id", "n_train", "n_val", "n_eligible",
              "estimation_failed")


def _declared(d, key):
    return _str_list(d[key]) if key in d else []


def estimator_tables(d, path, ds, n_models=None):
    """One CSV-only table per estimator setting of an
    eval_estimator_ablation.py log: the {setting}__{m}_r2 arrays, with the
    split metadata every setting shares (n_eligible: the setting's own,
    n_eligible_setting)."""
    settings = _declared(d, "settings")
    labels = _declared(d, "setting_macro")
    if len(labels) != len(settings):
        labels = settings
    declared = _declared(d, "methods")
    holdout = float(_scalar(d, "holdout_frac", default=DEFAULT_HOLDOUT))
    tables = []
    for i, setting in enumerate(settings):
        pre = f"{setting}__"
        methods = [m for m in (declared or _suffix_methods(d, pre))
                   if f"{pre}{m}_r2" in d
                   and np.asarray(d[f"{pre}{m}_r2"]).ndim == 2]
        if not methods:
            continue
        n_rows = np.asarray(d[f"{pre}{methods[0]}_r2"]).shape[0]
        meta, legacy = split_meta(d, "", n_rows, holdout, n_models)
        el = d.get("n_eligible_setting")
        if el is not None and np.asarray(el).shape == (n_rows, len(settings)):
            meta["n_eligible"] = np.asarray(el, dtype=float)[:, i]
        r2 = {}
        for m in methods:
            arr = np.asarray(d[f"{pre}{m}_r2"], dtype=float)
            if arr.shape[0] != n_rows:
                raise ValueError(f"{pre}{m}_r2 has {arr.shape[0]} rows, "
                                 f"expected {n_rows}")
            r2[m] = arr
        failed = d.get("est_failed")
        failed = (np.asarray(failed, dtype=bool)[:, i]
                  if failed is not None
                  and np.asarray(failed).shape == (n_rows, len(settings))
                  else _failed(d, ""))
        tables.append(SplitTable(
            ds, path.name, path, f"Est{labels[i]}", holdout, False, meta, r2,
            {m: None for m in methods},
            {m: (path.name, f"{pre}{m}_r2") for m in methods}, legacy,
            _n_benchmarks(d), failed, None, False))
    return tables


def prospective_tables(d, path, tag):
    """One CSV-only table per variant of an eval_prospective.py log: the
    random-split control rows ({m}_r2, rows of that `variant`), which pair
    on split_id exactly as CV splits do."""
    if "variant" not in d:
        return []
    variant = np.asarray(d["variant"]).astype(int)
    n_all = variant.size
    keys = _declared(d, "variant_keys")
    labels = _declared(d, "variant_macros")
    methods = [m for m in _declared(d, "methods") if f"{m}_r2" in d
               and np.asarray(d[f"{m}_r2"]).ndim == 2
               and np.asarray(d[f"{m}_r2"]).shape[0] == n_all]
    if not methods:
        return []
    tables = []
    for v in sorted(set(variant.tolist())):
        rows = np.flatnonzero(variant == v)
        sub = {k: np.asarray(d[k])[rows] for k in _META_KEYS
               if k in d and np.asarray(d[k]).shape == (n_all,)}
        meta, legacy = split_meta(sub, "", rows.size, DEFAULT_HOLDOUT)
        total = meta["n_train"] + meta["n_val"]
        holdout = (round(float(np.mean(meta["n_val"] / total)), 4)
                   if rows.size and np.all(total > 0) else math.nan)
        label = labels[v] if v < len(labels) else (keys[v] if v < len(keys)
                                                   else str(v))
        failed = (np.asarray(sub["estimation_failed"], dtype=bool)
                  if "estimation_failed" in sub else None)
        tables.append(SplitTable(
            PROSPECTIVE_DATA.get(tag, tag), path.name, path, f"Ctrl{label}",
            holdout, False, meta,
            {m: np.asarray(d[f"{m}_r2"], dtype=float)[rows] for m in methods},
            {m: None for m in methods},
            {m: (path.name, f"{m}_r2") for m in methods}, legacy,
            _n_benchmarks(d), failed, None, False))
    return tables


def imputer_tables(d, path, ds, n_models=None):
    """One CSV-only table per imputer of an eval_imputers.py log: the
    {m}_{imputer}_r2 arrays, whose columns are the k of its `ks` array."""
    ks = ([int(k) for k in np.asarray(d["ks"]).ravel()] if "ks" in d
          else None)
    holdout = float(_scalar(d, "holdout_frac", default=DEFAULT_HOLDOUT))
    tables = []
    for imp in _declared(d, "imputers"):
        methods = [m for m in _declared(d, "methods")
                   if f"{m}_{imp}_r2" in d
                   and np.asarray(d[f"{m}_{imp}_r2"]).ndim == 2]
        if not methods:
            continue
        n_rows = np.asarray(d[f"{methods[0]}_{imp}_r2"]).shape[0]
        meta, legacy = split_meta(d, "", n_rows, holdout, n_models)
        r2 = {}
        for m in methods:
            arr = np.asarray(d[f"{m}_{imp}_r2"], dtype=float)
            if arr.shape[0] != n_rows or (ks and arr.shape[1] != len(ks)):
                raise ValueError(f"{m}_{imp}_r2 has shape {arr.shape}, "
                                 f"expected ({n_rows}, {len(ks or [])})")
            r2[m] = arr
        tables.append(SplitTable(
            ds, path.name, path,
            f"Imp{IMPUTER_TAGS.get(imp, imp.capitalize())}", holdout, False,
            meta, r2, {m: None for m in methods},
            {m: (path.name, f"{m}_{imp}_r2") for m in methods}, legacy,
            _n_benchmarks(d), _failed(d, ""), ks, False))
    return tables


# ── Loading ──────────────────────────────────────────────────────

def _glob(directory, pattern):
    """Sorted matches, ignoring dotfiles (e.g. macOS '._' resource forks)."""
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        return []
    return [p for p in sorted(directory.glob(pattern))
            if not p.name.startswith(".")]


def _load_npz(path, warn):
    try:
        with np.load(path, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:
        warn(f"cannot read {path}: {e}; skipped")
        return None


class _Models:
    """Number of models M of each dataset (rows of {ds}.matrix.npz)."""

    def __init__(self, data_root):
        self.data_root = data_root
        self.cache = {}

    def __call__(self, ds):
        if ds not in self.cache:
            M = None
            path = (pathlib.Path(self.data_root) / f"{ds}.matrix.npz"
                    if self.data_root else None)
            if path is not None and path.is_file():
                try:
                    with np.load(path, allow_pickle=True) as d:
                        M = int(d["O"].shape[0])
                except Exception:
                    M = None
            self.cache[ds] = M
        return self.cache[ds]


def source_files(root):
    """[(kind, path, ds)] of every source under root, in extraction order
    (kind: emi, results, baselines, greedy_cv, tabimpute, logit_emi,
    logit_cv, then the CSV-only estimator, prospective (ds = the setting
    tag) and imputers)."""
    root = pathlib.Path(root).expanduser().resolve()
    logs, logit = root / "logs", root / "logs_logit"

    def stem(p, prefix, suffix=".npz"):
        return p.name[len(prefix):-len(suffix)]

    out = []
    for p in _glob(logs, "entropy_vs_mi_*.npz"):
        if not p.stem.endswith("_logit"):
            out.append(("emi", p, stem(p, "entropy_vs_mi_")))
    for p in _glob(logs, "*_results.npz"):
        out.append(("results", p, stem(p, "", "_results.npz")))
    for p in _glob(logs, "baselines_*.npz"):
        out.append(("baselines", p, stem(p, "baselines_")))
    for p in _glob(logs, "greedy_cv_*.npz"):
        if not p.stem.endswith("_logit"):
            out.append(("greedy_cv", p, stem(p, "greedy_cv_")))
    for p in _glob(logs, "tabimpute_*.npz"):
        if not p.name.startswith("tabimpute_extra_"):
            out.append(("tabimpute", p, stem(p, "tabimpute_")))
    for p in _glob(logs, "tabimpute_extra_*.npz"):
        out.append(("tabimpute", p, stem(p, "tabimpute_extra_")))
    for p in _glob(logit, "entropy_vs_mi_*_logit.npz"):
        out.append(("logit_emi", p, stem(p, "entropy_vs_mi_", "_logit.npz")))
    for p in _glob(logit, "greedy_cv_*_logit.npz"):
        out.append(("logit_cv", p, stem(p, "greedy_cv_", "_logit.npz")))
    for p in _glob(logs, "estimator_ablation_*.npz"):
        out.append(("estimator", p, stem(p, "estimator_ablation_")))
    for p in _glob(logs, "prospective_*.npz"):
        out.append(("prospective", p, stem(p, "prospective_")))
    for p in _glob(logs, "imputers_*.npz"):
        out.append(("imputers", p, stem(p, "imputers_")))
    return out


def source_dataset(kind, ds):
    """Data set of a source_files entry (a prospective tag maps to the
    data of its setting)."""
    return PROSPECTIVE_DATA.get(ds, ds) if kind == "prospective" else ds


def tables_of(kind, d, path, ds, n_models=None):
    """The split tables of one loaded file of the given kind."""
    if kind == "emi":
        t = suffix_table(d, path, ds, "", n_models=n_models)
        return [t] if t else []
    if kind == "results":
        out = []
        if any(k.startswith("emi_") for k in d):
            t = suffix_table(d, path, ds, "", "emi_",
                             _scalar(d, "emi_holdout_frac",
                                     default=DEFAULT_HOLDOUT), n_models)
            if t:
                out.append(t)
        return out + cv_tables(d, path, ds, "", "cv_", n_models)
    if kind == "baselines":
        t = suffix_table(d, path, ds, "Base", n_models=n_models)
        return [t] if t else []
    if kind == "greedy_cv":
        return cv_tables(d, path, ds, "", "", n_models)
    if kind == "tabimpute":
        t = suffix_table(d, path, ds, "Tab", n_models=n_models)
        return [t] if t else []
    if kind == "logit_emi":
        t = suffix_table(d, path, ds, "Logit", n_models=n_models)
        return [t] if t else []
    if kind == "logit_cv":
        return cv_tables(d, path, ds, "Logit", "", n_models)
    if kind == "estimator":
        return estimator_tables(d, path, ds, n_models)
    if kind == "prospective":
        return prospective_tables(d, path, ds)
    if kind == "imputers":
        return imputer_tables(d, path, ds, n_models)
    raise ValueError(kind)


def load_tables(root, datasets=None, reps=None, data_dir=None, warn=None):
    """Every split table under root (source_files order; the two TabImpute
    files of a dataset merged into one), restricted to `reps` (None: all
    rows) and to `datasets` (None: all).  Unreadable or malformed files are
    reported through warn(msg) and skipped."""
    if warn is None:
        def warn(msg):
            print(f"WARNING: {msg}", file=sys.stderr, flush=True)
    models = _Models(default_data_dir() if data_dir is None else data_dir)
    tables, tab_index = [], {}
    for kind, path, ds in source_files(root):
        if datasets is not None and source_dataset(kind, ds) not in datasets:
            continue
        d = _load_npz(path, warn)
        if d is None:
            continue
        try:
            found = tables_of(kind, d, path, ds,
                              None if kind == "prospective" else models(ds))
        except Exception as e:
            warn(f"{path}: paired-table extraction failed "
                 f"({type(e).__name__}: {e}); file skipped")
            continue
        for t in found:
            if t.legacy and not np.isfinite(t.meta["n_train"]).all():
                warn(f"{path.name}: no split sizes and no {ds}.matrix.npz; "
                     f"corrected standard errors are NaN")
            if kind == "tabimpute":
                key = (ds, t.holdout_frac)
                if key in tab_index:
                    i = tab_index[key]
                    tables[i] = merge_tables(tables[i], t)
                    continue
                tab_index[key] = len(tables)
            tables.append(t)
    return [t.restrict(reps) for t in tables]


# ── Pairs and rows ───────────────────────────────────────────────

def method_rank(method):
    """Position of a method in the pair orientation: METHOD_ORDER's index of
    its base name (METHOD_PREFIXES / METHOD_SUFFIXES stripped); a base not
    in METHOD_ORDER ranks between hub and random."""
    base = str(method)
    for pre in METHOD_PREFIXES:
        if base.startswith(pre):
            base = base[len(pre):]
            break
    for suf in METHOD_SUFFIXES:
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    if base in METHOD_ORDER:
        return float(METHOD_ORDER.index(base))
    return METHOD_ORDER.index("random") - 0.5


def orient(methods):
    """All unordered pairs of `methods`, each as (a, b) with a before b by
    method_rank (equal ranks in the given order)."""
    methods = list(methods)
    order = sorted(range(len(methods)),
                   key=lambda i: (method_rank(methods[i]), i))
    ordered = [methods[i] for i in order]
    return [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:]]


def orient_pair(a, b):
    """(a, b) in the orientation of orient(): swapped if b ranks before a
    (equal ranks keep the given order)."""
    return (b, a) if method_rank(b) < method_rank(a) else (a, b)


def compare(table, a, b, k):
    """paired_stats of methods a and b of `table` at k (1-based; the
    table's ks labels if it has them); NaN statistics if k is not in the
    logged k range."""
    ra, rb = table.r2[a], table.r2[b]
    if table.ks is None:
        col = k - 1 if 1 <= k <= min(ra.shape[1], rb.shape[1]) else None
    else:
        col = table.column(k)
    if col is None:
        nan = np.full(table.n_rows, np.nan)
        return paired_stats(nan, nan, table.meta["n_train"],
                            table.meta["n_val"], table.meta["rep"])
    return paired_stats(ra[:, col], rb[:, col], table.meta["n_train"],
                        table.meta["n_val"], table.meta["rep"])


def reps_text(table):
    return " ".join(str(r) for r in np.unique(table.meta["rep"]))


def table_rows(table, pairs="all", ks=None):
    """CSV rows (dicts over CSV_FIELDS) of a table: every oriented pair
    ('all') or the KEY_PAIRS / MACRO_PAIRS present ('key' / 'macro'), at
    every logged k (or ks)."""
    if pairs == "all":
        todo = orient(table.methods)
    else:
        wanted = KEY_PAIRS if pairs == "key" else MACRO_PAIRS
        todo = [(a, b) for a, b in wanted if a in table.r2 and b in table.r2]
    if ks is None and table.ks is not None:
        ks = table.k_values()
    elif ks is None:
        k_max = max((v.shape[1] for v in table.r2.values()), default=0)
        ks = range(1, k_max + 1)
    rows = []
    for a, b in todo:
        for k in ks:
            rows.append({"dataset": table.dataset, "source": table.source,
                         "tag": table.tag,
                         "holdout_frac": table.holdout_frac, "k": k,
                         "method_a": a, "method_b": b,
                         "reps": reps_text(table), "n_rows": table.n_rows,
                         **compare(table, a, b, k)})
    return rows


def _cell(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return v


def write_csv(rows, path, fields=None):
    """Rows over `fields` (default CSV_FIELDS; floats at full precision,
    NaN as 'nan', missing fields NaN), written atomically (temporary file
    in the same directory, then os.replace)."""
    fields = CSV_FIELDS if fields is None else list(fields)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({key: _cell(row.get(key, math.nan))
                            for key in fields})
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_csv(path):
    """Rows of a CSV written by write_csv, with numeric fields parsed."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in CSV_FIELDS:
                if key in INT_FIELDS:
                    row[key] = int(row[key])
                elif key in STAT_FIELDS or key == "holdout_frac":
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def write_dataset_csvs(tables, out_dir, pairs="all"):
    """OUT_DIR/paired_{ds}.csv per dataset; returns {ds: (path, n_rows)}."""
    by_ds = {}
    for t in tables:
        by_ds.setdefault(t.dataset, []).extend(table_rows(t, pairs))
    written = {}
    for ds, rows in by_ds.items():
        path = pathlib.Path(out_dir) / f"paired_{ds}.csv"
        write_csv(rows, path)
        written[ds] = (path, len(rows))
    return written


def print_summary(tables, ks=(1, 5, 10), pairs="macro"):
    """The MACRO_PAIRS (facility location's included; pairs="key": the
    KEY_PAIRS only) at a few k, per table, to stdout."""
    for t in tables:
        rows = table_rows(t, pairs, list(ks))
        if not rows:
            continue
        hp = f", holdout {t.holdout_frac:g}" if t.per_holdout else ""
        hp += f", {t.tag}" if t.tag else ""
        print(f"\n{t.dataset} [{t.source}{hp}] {t.n_rows} splits "
              f"(reps {reps_text(t)})"
              + (" legacy: rows = rep-0 folds" if t.legacy else ""))
        for r in rows:
            if r["n_splits"] == 0:
                print(f"  k={r['k']:2d} {r['method_a']:>10} − "
                      f"{r['method_b']:<10} no common splits")
                continue
            print(f"  k={r['k']:2d} {r['method_a']:>10} − "
                  f"{r['method_b']:<10} Δ={r['mean_diff']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
                  f"win {r['win_rate']:.2f} (n={r['n_splits']})")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="experiment root with logs/ and logs_logit/")
    parser.add_argument("--out-dir", help="directory for paired_{ds}.csv "
                        "(default: ROOT/numbers)")
    parser.add_argument("--datasets", nargs="+",
                        help="dataset names (default: every dataset found)")
    parser.add_argument("--reps", nargs="+", type=int,
                        help="CV repetitions to use (default: all rows)")
    parser.add_argument("--pairs", default="all", choices=["all", "key"],
                        help="every method pair (default) or KEY_PAIRS only")
    parser.add_argument("--data-dir", help="directory with {ds}.matrix.npz "
                        "(split sizes of pre-matched-design logs; default: "
                        "path_config.data_dir())")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the key-pair summary")
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root).expanduser().resolve()
    data_root = (pathlib.Path(args.data_dir).expanduser().resolve()
                 if args.data_dir else None)
    tables = load_tables(root, args.datasets, args.reps, data_root)
    if not tables:
        print(f"paired_stats.py: error: no per-split R² logs under {root}",
              file=sys.stderr)
        sys.exit(2)
    out_dir = (pathlib.Path(args.out_dir).expanduser() if args.out_dir
               else root / "numbers")
    if not args.quiet:
        print_summary(tables)
    written = write_dataset_csvs(tables, out_dir, args.pairs)
    print()
    for ds, (path, n) in written.items():
        print(f"wrote {path} ({n} rows)")


if __name__ == "__main__":
    main()
