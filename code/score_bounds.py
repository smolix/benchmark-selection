"""Valid score range of every benchmark (protocol rule D2, 2026-10-18).

The predictions of every imputer (the Gaussian conditional mean of
eval_entropy_vs_mi.impute_and_evaluate and every script that scores its
own predictions, TabImpute, the fitted completion imputers, the
prospective, cost, token and item scripts) are clipped to the valid score
range of their benchmark before scoring.  The range comes from the
benchmark's scale -- from data/<ds>.meta.json where the metadata names the
metric, otherwise from the documented per-dataset rule below -- and never
from held-out (or any) observed scores; unbounded scales (Elo / arena
ratings) are not clipped.  The clip is applied in train-standardized units,
which is the raw-unit clip mapped through the split's training mean and SD
(z_limits: z_lo = (lo - mean) / sd, z_hi = (hi - mean) / sd; clip_z), so a
prediction inside the range keeps its value bit for bit.  The true
standardized scores keep their +-10 clip (Z_CLIP of the scorers).  The
only unclipped numbers in the outputs are those of eval_reviewer_checks'
clip check (raw vs clipped, the appendix).

Rules (RULES; dataset -> column -> (lo, hi), the source of each bound in
the audit, data/score_bounds.json):
  mmlu          [0, 1]     every column: accuracy on an MMLU subject (Open
                           LLM Leaderboard v1 harness acc)
  mteb, mteb2026  by the task's MTEB main_score (benchmarks[].main_score
                           of data/mteb2026.meta.json; data/mteb.matrix.npz
                           has the same 56 columns): cosine_spearman (STS,
                           Summarization; a Spearman correlation) [-1, 1],
                           accuracy, v_measure, max_ap, map_at_1000,
                           max_over_subqueries_map_at_1000, ndcg_at_10
                           [0, 1]
  merged, merged_v1  by the source prefix of the column: mmlu/ (subject
                           accuracy), open_llm/ (lm-eval-harness metrics in
                           [0, 1], data/README.md), helm/ (HELM Lite
                           leaderboard values: EM, F1, BLEU-4, mean win
                           rate, all fractions), livebench/ (category
                           scores stored as fractions) [0, 1];
                           alpaca_eval/ (win rates in %), arena_hard/
                           (win rate against the baseline in %), bigcode/
                           (BigCodeBench pass@1 in %) [0, 100]; mt_bench/
                           (GPT-4 judge score on the 1-10 scale) [1, 10];
                           wildbench/ (WB-Score = 20 (mean judge score -
                           5) for judge scores on the 1-10 scale, WildBench
                           v2 all_stat_wildbench WB_score.original_task=*)
                           [-80, 100]
  benchpress    by benchmarks[].metric of data/benchpress.json: a metric
                           in % ("% correct", "pass@1 %", "% resolved",
                           "% of 42 points", ...) [0, 100]; "index score"
                           (Artificial Analysis Intelligence Index) and
                           "overall score" (LiveBench) are averages of
                           percentages [0, 100]; "Elo rating" (Chatbot
                           Arena Elo, Codeforces Rating) and "score"
                           (GDP-Val AA, an Elo-style rating around 1,600)
                           unbounded
  olv2sub       [0, 1]     raw lm-eval-harness metrics (meta
                           score_definition.scale)
  olv1items     [0, 1]     task score = mean of item scores in [0, 1]
                           (binary items; TruthfulQA mc2 continuous in
                           [0, 1]; meta description)
  epoch         [0, 1]     normalized performance max(0, (s - baseline) /
                           (ceiling - baseline)) (meta value_definition)
  helm_lite, helm_mmlu, helm_classic  [0, 1]  HELM main metrics
                           (benchmarks[].metric_name: exact_match,
                           quasi_exact_match, f1_score, bleu_4, rouge_2,
                           NDCG@10, RR@10, math_equiv_chain_of_thought, ...;
                           all fractions, none lower_is_better)
A dataset may instead declare its ranges in data/<ds>.meta.json,
"score_bounds": {"default": [lo, hi], "columns": {name: [lo, hi]}} (null
= unbounded on that side; meta_rule): the declaration takes precedence
over RULES.  No meta file in data/ has one (the tests' synthetic
datasets do).  A dataset with neither raises KeyError, so the clip is
never skipped silently.  Every rule is checked against the data when it
is applied: an observed score outside its range raises (load-time sanity
check of the rule, not a source of the bound).

Usage:
    python score_bounds.py [--datasets ...] [--write data/score_bounds.json]
                           [--check data/score_bounds.json]

--write writes the audit (every column of every dataset: bound, source of
the bound, observed range and count); --check compares a written audit
with the rules and exits 1 on a difference.  PRED_CLIP = False (test-only
override) disables the clip everywhere: z_limits returns -inf / +inf.
"""

import argparse
import json
import math
import pathlib
import sys
from typing import NamedTuple, Optional, Sequence

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir

# Protocol rule D2: clip every imputer's predictions to the valid range.
# Test-only override: False disables the clip (z_limits returns +-inf).
PRED_CLIP = True

INF = math.inf
UNIT = (0.0, 1.0)
PERCENT = (0.0, 100.0)
CORRELATION = (-1.0, 1.0)
MT_BENCH = (1.0, 10.0)
WB_SCORE = (-80.0, 100.0)
UNBOUNDED = (-INF, INF)
# Names of the ranges (eval_reviewer_checks' clip check groups by them;
# unit_interval and percent are the names of its rebuttal-era kinds)
BOUND_KIND = {UNIT: "unit_interval", PERCENT: "percent",
              CORRELATION: "correlation", MT_BENCH: "judge_1_10",
              WB_SCORE: "wb_score", UNBOUNDED: "unbounded"}
# Tolerance of the load-time check that observed scores lie in their range
CHECK_TOL = 1e-9

DATASETS = ["mmlu", "mteb", "merged", "merged_v1", "benchpress", "olv2sub",
            "olv1items", "epoch", "helm_lite", "helm_mmlu", "helm_classic",
            "mteb2026"]
AUDIT_FILE = "score_bounds.json"

MTEB_MAIN_SCORE = {
    "cosine_spearman": (CORRELATION, "Spearman correlation"),
    "accuracy": (UNIT, "accuracy"),
    "v_measure": (UNIT, "V-measure"),
    "max_ap": (UNIT, "average precision"),
    "map_at_1000": (UNIT, "MAP@1000"),
    "max_over_subqueries_map_at_1000": (UNIT, "MAP@1000"),
    "ndcg_at_10": (UNIT, "nDCG@10"),
}
MERGED_PREFIX = {
    "mmlu": (UNIT, "MMLU subject accuracy"),
    "open_llm": (UNIT, "lm-eval-harness metric in [0, 1] (data/README.md)"),
    "helm": (UNIT, "HELM Lite leaderboard value (EM / F1 / BLEU-4 / mean "
                   "win rate, fractions)"),
    "livebench": (UNIT, "LiveBench category score stored as a fraction"),
    "alpaca_eval": (PERCENT, "AlpacaEval 2 win rate in %"),
    "arena_hard": (PERCENT, "Arena-Hard-Auto win rate against the baseline "
                            "in %"),
    "bigcode": (PERCENT, "BigCodeBench pass@1 in %"),
    "mt_bench": (MT_BENCH, "MT-Bench GPT-4 judge score on the 1-10 scale"),
    "wildbench": (WB_SCORE, "WildBench WB-Score = 20 (mean judge score - 5), "
                            "judge scores on the 1-10 scale"),
}
BENCHPRESS_METRIC = {
    "index score": (PERCENT, "Artificial Analysis Intelligence Index, an "
                             "average of percentages"),
    "overall score": (PERCENT, "LiveBench overall score, an average of "
                               "percentages"),
    "Elo rating": (UNBOUNDED, "Elo rating (unbounded)"),
    "score": (UNBOUNDED, "Elo-style rating (unbounded)"),
}
HELM_METRICS = {"exact_match", "quasi_exact_match", "f1_score", "bleu_4",
                "rouge_2", "NDCG@10", "RR@10", "final_number_exact_match",
                "math_equiv_chain_of_thought"}


class Bounds(NamedTuple):
    """Valid raw-score range per column: lo, hi (N,) float (+-inf where
    unbounded) and source (N,) str, where each bound comes from."""
    lo: np.ndarray
    hi: np.ndarray
    source: list


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _const(names, rng, source):
    return [(rng, source)] * len(names)


def _rule_mteb(names, data):
    meta = _read_json(data / "mteb2026.meta.json")
    by_name = {b["name"]: b["main_score"] for b in meta["benchmarks"]}
    out = []
    for n in names:
        if n not in by_name:
            raise KeyError(f"MTEB task {n!r} not in mteb2026.meta.json")
        ms = by_name[n]
        if ms not in MTEB_MAIN_SCORE:
            raise KeyError(f"no range for MTEB main_score {ms!r} ({n})")
        rng, what = MTEB_MAIN_SCORE[ms]
        out.append((rng, f"meta mteb2026.meta.json benchmarks[].main_score "
                         f"= {ms} ({what})"))
    return out


def _rule_merged(names, data):
    out = []
    for n in names:
        prefix = n.split("/", 1)[0]
        if "/" not in n or prefix not in MERGED_PREFIX:
            raise KeyError(f"merged column {n!r}: no rule for its source")
        rng, what = MERGED_PREFIX[prefix]
        out.append((rng, f"rule merged source {prefix}/: {what}"))
    return out


def _rule_benchpress(names, data):
    meta = _read_json(data / "benchpress.json")
    by_name = {b["name"]: b["metric"] for b in meta["benchmarks"]}
    out = []
    for n in names:
        if n not in by_name:
            raise KeyError(f"BenchPress benchmark {n!r} not in "
                           f"benchpress.json")
        metric = by_name[n]
        if metric in BENCHPRESS_METRIC:
            rng, what = BENCHPRESS_METRIC[metric]
        elif "%" in metric:
            rng, what = PERCENT, "percentage"
        else:
            raise KeyError(f"no range for BenchPress metric {metric!r} ({n})")
        out.append((rng, f"meta benchpress.json benchmarks[].metric = "
                         f"{metric!r} ({what})"))
    return out


def _rule_helm(ds):
    def rule(names, data):
        meta = _read_json(data / f"{ds}.meta.json")
        bench = meta["benchmarks"]
        out = []
        for n in names:
            b = bench[n]
            metric = b["metric_name"]
            if metric not in HELM_METRICS or b.get("lower_is_better"):
                raise KeyError(f"{ds} {n!r}: no range for metric {metric!r}")
            out.append((UNIT, f"meta {ds}.meta.json benchmarks.{n}"
                              f".metric_name = {metric} (fraction)"))
        return out
    return rule


def _rule_unit(ds, what):
    return lambda names, data: _const(names, UNIT, f"rule {ds}: {what}")


RULES = {
    "mmlu": _rule_unit("mmlu", "MMLU subject accuracy (Open LLM Leaderboard "
                               "v1 harness acc)"),
    "mteb": _rule_mteb,
    "mteb2026": _rule_mteb,
    "merged": _rule_merged,
    "merged_v1": _rule_merged,
    "benchpress": _rule_benchpress,
    "olv2sub": _rule_unit("olv2sub", "raw lm-eval-harness metric in [0, 1] "
                                     "(meta score_definition.scale)"),
    "olv1items": _rule_unit("olv1items", "task score = mean item score, "
                                         "items in [0, 1] (meta)"),
    "epoch": _rule_unit("epoch", "normalized performance in [0, 1] (meta "
                                 "value_definition)"),
    "helm_lite": _rule_helm("helm_lite"),
    "helm_mmlu": _rule_helm("helm_mmlu"),
    "helm_classic": _rule_helm("helm_classic"),
}
# Names under which scripts address a dataset (eval_prospective's settings)
ALIASES = {"helm": "helm_lite", "olv2sub_math": "olv2sub",
           "epoch_bymodel": "epoch"}


def _load_matrix(ds, data):
    d = np.load(data / f"{ds}.matrix.npz", allow_pickle=True)
    return (d["B"].astype(np.float64), d["O"] != 0,
            [str(b) for b in d["benchmark_names"]])


def _pair(v, what):
    """(lo, hi) of a JSON [lo, hi] pair, null = unbounded on that side."""
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"{what}: expected [lo, hi], got {v!r}")
    return (-INF if v[0] is None else float(v[0]),
            INF if v[1] is None else float(v[1]))


def meta_rule(ds, data):
    """The rule of a dataset whose data/<ds>.meta.json declares its valid
    ranges, "score_bounds": {"default": [lo, hi], "columns": {name: [lo,
    hi]}} (null = unbounded on that side; a column without an entry takes
    the default; a missing default is an error for such a column); None
    without such a declaration.  A declaration takes precedence over
    RULES (none of the meta files in data/ has one: their datasets use
    RULES; synthetic datasets in the tests declare theirs)."""
    path = data / f"{ds}.meta.json"
    if not path.is_file():
        return None
    decl = _read_json(path).get("score_bounds")
    if decl is None:
        return None
    default = decl.get("default")
    cols = decl.get("columns", {})

    def rule(names, data):
        out = []
        for n in names:
            if n in cols:
                out.append((_pair(cols[n], f"{path} {n}"),
                            f"meta {ds}.meta.json score_bounds.columns"))
            elif default is not None:
                out.append((_pair(default, f"{path} default"),
                            f"meta {ds}.meta.json score_bounds.default"))
            else:
                raise KeyError(f"{path}: no score_bounds for {n!r}")
        return out
    return rule


def dataset_bounds(ds: str, benchmark_names: Optional[Sequence] = None,
                   data: Optional[pathlib.Path] = None,
                   check: bool = True) -> Bounds:
    """Bounds of the columns `benchmark_names` (default: every column of
    data/<ds>.matrix.npz, in order) of dataset `ds` (or its alias): the
    score_bounds declaration of data/<ds>.meta.json if it has one
    (meta_rule), else RULES.  check: every observed score of
    data/<ds>.matrix.npz must lie in its column's range (ValueError
    otherwise).  KeyError for a dataset or column without a rule: the clip
    is never skipped silently."""
    data = data_dir() if data is None else pathlib.Path(data)
    ds = ALIASES.get(ds, ds)
    rule = meta_rule(ds, data)
    if rule is None and ds not in RULES:
        raise KeyError(f"no score-range rule for dataset {ds!r}; add one to "
                       f"score_bounds.RULES (or a score_bounds declaration "
                       f"to data/{ds}.meta.json)")
    if benchmark_names is None or check:
        B, O, names = _load_matrix(ds, data)
    if benchmark_names is None:
        benchmark_names = names
    benchmark_names = [str(b) for b in benchmark_names]
    rules = (rule or RULES[ds])(benchmark_names, data)
    lo = np.array([r[0][0] for r in rules], dtype=np.float64)
    hi = np.array([r[0][1] for r in rules], dtype=np.float64)
    bounds = Bounds(lo, hi, [r[1] for r in rules])
    if check:
        col = {n: j for j, n in enumerate(names)}
        for c, n in enumerate(benchmark_names):
            if n not in col:
                continue
            x = B[O[:, col[n]], col[n]]
            if x.size and (x.min() < lo[c] - CHECK_TOL
                           or x.max() > hi[c] + CHECK_TOL):
                raise ValueError(f"{ds} {n!r}: observed scores "
                                 f"[{x.min():g}, {x.max():g}] outside the "
                                 f"valid range [{lo[c]:g}, {hi[c]:g}]")
    return bounds


def bound_kind(lo, hi):
    """Name of the range [lo, hi] (BOUND_KIND; "other" if unnamed)."""
    return BOUND_KIND.get((float(lo), float(hi)), "other")


def unbounded(N: int) -> Bounds:
    """Bounds that clip nothing (synthetic data in tests)."""
    return Bounds(np.full(N, -INF), np.full(N, INF), ["unbounded"] * N)


def z_limits(bounds, col_mean, col_std):
    """(z_lo, z_hi): the valid range in the train-standardized units of a
    split, (lo - mean) / sd and (hi - mean) / sd (+-inf where unbounded).
    bounds: a Bounds or (lo, hi), or None (no clip).  -inf / +inf
    everywhere when PRED_CLIP is off (test-only override)."""
    col_mean = np.asarray(col_mean, dtype=np.float64)
    N = col_mean.shape[0]
    if bounds is None or not PRED_CLIP:
        return np.full(N, -INF), np.full(N, INF)
    lo, hi = (np.asarray(b, dtype=np.float64) for b in tuple(bounds)[:2])
    col_std = np.asarray(col_std, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (lo - col_mean) / col_std, (hi - col_mean) / col_std


def clip_z(z, z_lo, z_hi):
    """(z clipped to [z_lo, z_hi], number of entries the clip changed);
    z_lo / z_hi broadcast against z (columns last).  Entries inside keep
    their value bit for bit."""
    z = np.asarray(z, dtype=np.float64)
    n = int(np.count_nonzero((z < z_lo) | (z > z_hi)))
    if n == 0:
        return z, 0
    return np.clip(z, z_lo, z_hi), n


def clip_raw(x, lo, hi):
    """(x clipped to [lo, hi] in raw units, number changed); unchanged
    when PRED_CLIP is off."""
    x = np.asarray(x, dtype=np.float64)
    if not PRED_CLIP:
        return x, 0
    n = int(np.count_nonzero((x < lo) | (x > hi)))
    if n == 0:
        return x, 0
    return np.clip(x, lo, hi), n


def _num(x):
    return None if not math.isfinite(x) else float(x)


def audit(datasets=DATASETS, data=None):
    """{"config", "datasets": {ds: {"columns": [...], "counts"}}} for the
    audit file: per column the bound, its source and the observed range
    and count (JSON: null for an infinite bound)."""
    data = data_dir() if data is None else pathlib.Path(data)
    out = {"description": "valid score range of every benchmark column "
                          "(protocol rule D2): the bound and its source "
                          "(score_bounds.RULES); observed_* are a sanity "
                          "check, never a source.  null = unbounded.",
           "generated_by": "code/score_bounds.py --write",
           "datasets": {}}
    for ds in datasets:
        B, O, names = _load_matrix(ds, data)
        bd = dataset_bounds(ds, names, data, check=True)
        cols, counts = [], {}
        for j, n in enumerate(names):
            x = B[O[:, j], j]
            key = f"[{_num(bd.lo[j])}, {_num(bd.hi[j])}]"
            counts[key] = counts.get(key, 0) + 1
            cols.append({"name": n, "lo": _num(bd.lo[j]),
                         "hi": _num(bd.hi[j]), "source": bd.source[j],
                         "n_obs": int(x.size),
                         "observed_min": float(x.min()) if x.size else None,
                         "observed_max": float(x.max()) if x.size else None})
        out["datasets"][ds] = {"n_columns": len(names),
                               "counts_by_bound": counts, "columns": cols}
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Valid score ranges (protocol rule D2): audit table.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--write", metavar="PATH",
                        help="write the audit JSON (e.g. data/"
                             f"{AUDIT_FILE})")
    parser.add_argument("--check", metavar="PATH",
                        help="compare an audit JSON with the rules; exit 1 "
                             "on a difference")
    args = parser.parse_args()
    obj = audit(args.datasets)
    for ds, entry in obj["datasets"].items():
        print(f"{ds}: {entry['n_columns']} columns, "
              + ", ".join(f"{k} x{v}" for k, v in
                          entry["counts_by_bound"].items()))
    if args.write:
        with open(args.write, "w") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")
        print(f"audit written to {args.write}")
    if args.check:
        old = _read_json(args.check)
        if old.get("datasets") != obj["datasets"]:
            bad = [ds for ds in obj["datasets"]
                   if old.get("datasets", {}).get(ds) != obj["datasets"][ds]]
            print(f"audit {args.check} differs from the rules: {bad}")
            sys.exit(1)
        print(f"audit {args.check} matches the rules")


if __name__ == "__main__":
    main()
