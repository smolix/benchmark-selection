r"""Derived keys of the camera-ready (blueprint section 6.1, source 3).

Usage (on a server; never on the laptop):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ~/venvs/benchselect/bin/python derive_numbers.py \
        --root ~/benchselect-runs/canonical-20261020/out \
        --out numbers_derived.csv

Reads only ROOT/numbers/paired_<ds>.csv, ROOT/numbers/numbers.csv (matrix
sizes), ROOT/logs/{baselines,analyze_objectives}_<ds>.npz and, for the
cover counts of the MMLU example, the run's data/mmlu.matrix.npz and its
estimator (TREE = --tree, default ROOT/../tree: TREE/data, TREE/code), and
writes one
CSV with the columns of numbers.csv plus `formula`:
key,value,source_file,source_key,index,aggregation,kind,scope,formula.
paper/cr/tools/paper_numbers.sh reads it as its third source.

The list is closed (blueprint 6.1.3):

  ScoreFaclocVs{X}{SigBetter,NotSig,SigWorse}  int, X in Trace Mi Entropy Hub
      Pca Random.  Cells (ds, k): the 11 independent matrices (merged_v1
      excluded), k in {3, 5, 10}, k <= N - 3.  Row: the paired_<ds>.csv row
      of baselines_<ds>.npz, tag Base, holdout 0.1, {a, b} = {facloc, x};
      oriented to FL - X ((lo, hi) if a = facloc, else (-hi, -lo));
      SigBetter if lo > 0, SigWorse if hi < 0, else NotSig; NaN rows skipped.
  ScoreNCells, ScoreNMatrices  int (cells and matrices counted).
  ScoreFaclocVsTraceMaxDeficit  float, max of EV - FL over the SigWorse
      cells of X = Trace.
  TieFaclocSplits{Ds}K{k}  int, Ds in Mmlu Mteb Merged Benchpress, k in 5 10
      15: splits with facloc_tie_breaks[:, k-1] > 0 (baselines_<ds>.npz;
      tie_breaks[:, k-1] counts the picks among the first k made among
      near-tied facility-location gains, greedy_select.tie_break_count).
  TieFaclocMean{Ds}Kfifteen  float, mean over splits of
      facloc_tie_breaks[:, 14] (tie-broken picks among the first 15).
  FaclocCover{Ds}K{k}  float, k in 5 10: mean over splits of
      facloc_facloc_tr[:, k-1] / N (analyze_objectives_<ds>.npz: F_FL of
      facility location's train-selected set under the training R; N = all
      benchmarks of the matrix).

Added for the revision of 2026-10-20 (reviews of the draft):

  ScoreMiVsEntropy{SigBetter,NotSig,SigWorse}  int, the 32 scoreboard cells,
      oriented to MI - entropy.
  ScoreAllKNCells, ScoreAllKFaclocVsTrace{SigBetter,NotSig,SigWorse},
  ScoreAllKFaclocVsTraceMaxDeficit, ScoreAllKFaclocVsTraceSigWorseNMatrices
      the FL - EV comparison over every k = 2..min(15, N - 3) of the 11
      matrices (the scoreboard rule without the k grid; k = 1 is left out as
      in the scoreboard, because first picks often coincide).
  SelMmlu{Facloc,Trace,Mi}KfiveSameSplits  int, splits (of 50) whose first
      five picks form the modal 5-set (baselines_mmlu.npz {m}_selected).
  SelMmlu{Facloc,Mi}KfivePick{One..Five}  text, the picks of the modal
      5-set in the selection order of its first split, underscores as
      spaces.
  SelMmlu{Trace,Mi}KfiveSharedWithFacloc  int, size of the intersection
      of the modal 5-sets of m and of facility location.
  CoverMmluFaclocKfivePick{One..Five}  int, on split 0 (training R of
      eval_entropy_vs_mi.estimate_corr): number of unselected subjects
      whose largest clipped correlation with the five picks is with that
      pick (ties: earlier pick).  CoverMmluFaclocKfiveSplitsSame: splits
      whose counts equal split 0's.
  ObjModalFirstMerged{Mi,Trace}{RestRsq,SingleRsq}  float, the point of
      figure 4b for the method's most frequent first pick: nanmean over
      splits of single_rest_r2_tr / single_r2 (analyze_objectives_merged.npz).
  FigMechMergedNPlotted, FigMechMergedNOmitted  int, benchmarks of
      figure 4b with finite / non-finite coordinates.

EXPECTED holds the values the blueprint recorded on 2026-10-20; a value that
is not reproduced is reported on stderr (exit status 3) and never adjusted.
"""

import argparse
import csv
import math
import pathlib
import sys

import numpy as np

SCORE_DATASETS = ["mmlu", "mteb", "merged", "benchpress", "olv2sub",
                  "olv1items", "epoch", "helm_lite", "helm_mmlu",
                  "helm_classic", "mteb2026"]
SCORE_KS = [3, 5, 10]
SCORE_METHODS = [("trace", "Trace"), ("mi", "Mi"), ("entropy", "Entropy"),
                 ("hub", "Hub"), ("pca", "Pca"), ("random", "Random")]
MAIN = [("mmlu", "Mmlu"), ("mteb", "Mteb"), ("merged", "Merged"),
        ("benchpress", "Benchpress")]
KWORD = {3: "three", 5: "five", 10: "ten", 15: "fifteen"}
DS_CAMEL = {"mmlu": "Mmlu", "mteb": "Mteb", "merged": "Merged",
            "benchpress": "Benchpress", "olv2sub": "OlvTwoSub",
            "olv1items": "OlvOneItems", "epoch": "Epoch",
            "helm_lite": "HelmLite", "helm_mmlu": "HelmMmlu",
            "helm_classic": "HelmClassic",
            "mteb2026": "MtebTwothousandtwentysix"}

# blueprint 6.1.3, recorded 2026-10-20 (orientation values; tolerance below)
EXPECTED = {
    "ScoreFaclocVsTraceSigBetter": 0, "ScoreFaclocVsTraceNotSig": 26,
    "ScoreFaclocVsTraceSigWorse": 6,
    "ScoreFaclocVsMiSigBetter": 15, "ScoreFaclocVsMiNotSig": 17,
    "ScoreFaclocVsMiSigWorse": 0,
    "ScoreFaclocVsEntropySigBetter": 15, "ScoreFaclocVsEntropyNotSig": 17,
    "ScoreFaclocVsEntropySigWorse": 0,
    "ScoreFaclocVsHubSigBetter": 22, "ScoreFaclocVsHubNotSig": 10,
    "ScoreFaclocVsHubSigWorse": 0,
    "ScoreFaclocVsPcaSigBetter": 17, "ScoreFaclocVsPcaNotSig": 15,
    "ScoreFaclocVsPcaSigWorse": 0,
    "ScoreFaclocVsRandomSigBetter": 16, "ScoreFaclocVsRandomNotSig": 16,
    "ScoreFaclocVsRandomSigWorse": 0,
    "ScoreNCells": 32, "ScoreNMatrices": 11,
    "ScoreFaclocVsTraceMaxDeficit": 0.00450955,
    "TieFaclocSplitsMergedKten": 24, "TieFaclocSplitsMergedKfifteen": 44,
    "TieFaclocSplitsBenchpressKten": 25,
    "TieFaclocSplitsMmluKfive": 0, "TieFaclocSplitsMmluKten": 0,
    "TieFaclocSplitsMmluKfifteen": 0,
    # The blueprint recorded 8.3 / 4.8 / 5.2, a different quantity (the sum
    # of the cumulative tie count over k); the defined mean is about one
    # (checked 2026-10-20, see camera_ready/workflow_reports/paper_draft.md).
    "TieFaclocMeanMtebKfifteen": 1.02, "TieFaclocMeanMergedKfifteen": 1.26,
    "TieFaclocMeanBenchpressKfifteen": 0.98,
    "FaclocCoverMmluKten": 0.969, "FaclocCoverMtebKten": 0.938,
    "FaclocCoverMergedKten": 0.824, "FaclocCoverBenchpressKten": 0.813,
    # revision of 2026-10-20 (values checked by hand on 4090-1)
    "ScoreMiVsEntropySigBetter": 3, "ScoreMiVsEntropyNotSig": 22,
    "ScoreMiVsEntropySigWorse": 7,
    "ScoreAllKFaclocVsTraceSigBetter": 0, "ScoreAllKFaclocVsTraceSigWorse": 22,
    "ScoreAllKNCells": 144, "ScoreAllKFaclocVsTraceSigWorseNMatrices": 3,
    "ScoreAllKFaclocVsTraceMaxDeficit": 0.0062051,
    "SelMmluFaclocKfiveSameSplits": 50, "CoverMmluFaclocKfiveSplitsSame": 50,
    "FigMechMergedNPlotted": 106, "FigMechMergedNOmitted": 8,
}
# absolute tolerance of the check: the recorded values are rounded
EXPECTED_TOL = {"ScoreFaclocVsTraceMaxDeficit": 5e-9,
                "TieFaclocMeanMtebKfifteen": 0.05,
                "TieFaclocMeanMergedKfifteen": 0.05,
                "TieFaclocMeanBenchpressKfifteen": 0.05}
DEFAULT_TOL = 5e-4

FIELDS = ["key", "value", "source_file", "source_key", "index",
          "aggregation", "kind", "scope", "formula"]


def read_sizes(numbers_csv):
    """{ds: N} from the Data{Ds}N keys of numbers.csv."""
    want = {f"Data{c}N": ds for ds, c in DS_CAMEL.items()}
    sizes = {}
    with open(numbers_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["key"] in want:
                sizes[want[row["key"]]] = int(row["value"])
    missing = sorted(set(DS_CAMEL) - set(sizes))
    if missing:
        sys.exit(f"derive_numbers: no Data*N key for {missing}")
    return sizes


def scoreboard(numbers_dir, sizes, rows):
    counts = {x: {"SigBetter": 0, "NotSig": 0, "SigWorse": 0}
              for x, _ in SCORE_METHODS}
    deficit = []
    cells = set()
    for ds in SCORE_DATASETS:
        path = numbers_dir / f"paired_{ds}.csv"
        with open(path, newline="") as fh:
            table = [r for r in csv.DictReader(fh)
                     if r["source"] == f"baselines_{ds}.npz"
                     and r["tag"] == "Base"
                     and abs(float(r["holdout_frac"]) - 0.1) < 1e-12]
        for k in SCORE_KS:
            if k > sizes[ds] - 3:
                continue
            for x, _ in SCORE_METHODS:
                hit = [r for r in table if int(r["k"]) == k and
                       {r["method_a"], r["method_b"]} == {"facloc", x}]
                if len(hit) != 1:
                    sys.exit(f"derive_numbers: {len(hit)} rows for "
                             f"{ds} k={k} facloc/{x}")
                r = hit[0]
                lo, hi, diff = (float(r["ci_low"]), float(r["ci_high"]),
                                float(r["mean_diff"]))
                if r["method_a"] != "facloc":
                    lo, hi, diff = -hi, -lo, -diff
                if not (math.isfinite(lo) and math.isfinite(hi)):
                    continue
                cells.add((ds, k))
                cls = ("SigBetter" if lo > 0 else
                       "SigWorse" if hi < 0 else "NotSig")
                counts[x][cls] += 1
                if x == "trace" and cls == "SigWorse":
                    deficit.append(-diff)          # EV - FL
    src = "numbers/paired_*.csv"
    agg = (f"count over {len(cells)} cells ({len(SCORE_DATASETS)} matrices "
           f"x k in {{3, 5, 10}}, k <= N - 3)")
    for x, X in SCORE_METHODS:
        for cls, n in counts[x].items():
            rows.append(dict(
                key=f"ScoreFaclocVs{X}{cls}", value=n,
                source_file=src,
                source_key=f"ci_low, ci_high of facloc vs {x}",
                index="source baselines_<ds>.npz, tag Base, holdout 0.1",
                aggregation=agg, kind="int", scope="derived",
                formula=("orient to FL - X; SigBetter: lo > 0, SigWorse: "
                         "hi < 0, NotSig otherwise; NaN rows skipped")))
    rows.append(dict(key="ScoreNCells", value=len(cells), source_file=src,
                     source_key="", index="", aggregation=agg, kind="int",
                     scope="derived",
                     formula="number of (matrix, k) cells counted"))
    rows.append(dict(key="ScoreNMatrices",
                     value=len({ds for ds, _ in cells}), source_file=src,
                     source_key="", index="", aggregation=agg, kind="int",
                     scope="derived", formula="number of matrices counted"))
    rows.append(dict(key="ScoreFaclocVsTraceMaxDeficit",
                     value=repr(max(deficit)) if deficit else "nan",
                     source_file=src, source_key="mean_diff of trace vs facloc",
                     index="", aggregation=f"max over {len(deficit)} cells "
                     "where facloc is significantly below trace",
                     kind="float", scope="derived",
                     formula="max of (EV - FL) over the SigWorse cells"))


def ties_and_cover(logs, rows):
    for ds, D in MAIN:
        with np.load(logs / f"baselines_{ds}.npz") as d:
            tb = np.asarray(d["facloc_tie_breaks"])
        if np.any(np.diff(tb, axis=1) < 0):
            sys.exit(f"derive_numbers: facloc_tie_breaks of {ds} is not "
                     "cumulative")
        for k in (5, 10, 15):
            rows.append(dict(
                key=f"TieFaclocSplits{D}K{KWORD[k]}",
                value=int((tb[:, k - 1] > 0).sum()),
                source_file=f"logs/baselines_{ds}.npz",
                source_key="facloc_tie_breaks", index=f"[:, {k - 1}]",
                aggregation=f"splits (of {tb.shape[0]}) with > 0",
                kind="int", scope="derived",
                formula=("number of splits whose first k facility-location "
                         "picks include a pick among near-tied gains")))
        rows.append(dict(
            key=f"TieFaclocMean{D}Kfifteen", value=repr(float(tb[:, 14].mean())),
            source_file=f"logs/baselines_{ds}.npz",
            source_key="facloc_tie_breaks", index="[:, 14]",
            aggregation=f"mean over {tb.shape[0]} splits", kind="float",
            scope="derived",
            formula=("mean number of tie-broken picks among the first 15 "
                     "facility-location picks")))
        with np.load(logs / f"analyze_objectives_{ds}.npz") as a:
            f = np.asarray(a["facloc_facloc_tr"], dtype=float)
            n_bench = int(np.asarray(a["eligible"]).shape[1])
        for k in (5, 10):
            col = f[:, k - 1] / n_bench
            ok = np.isfinite(col)
            rows.append(dict(
                key=f"FaclocCover{D}K{KWORD[k]}",
                value=repr(float(col[ok].mean())),
                source_file=f"logs/analyze_objectives_{ds}.npz",
                source_key="facloc_facloc_tr", index=f"[:, {k - 1}]",
                aggregation=(f"nanmean over {int(ok.sum())}/{len(col)} splits "
                             f"of F_FL / N, N = {n_bench}"),
                kind="float", scope="derived",
                formula=("F_FL(A_k)/N: mean best clipped correlation of a "
                         "benchmark with facility location's k-set")))


def _pair_rows(numbers_dir, ds):
    path = numbers_dir / f"paired_{ds}.csv"
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if r["source"] == f"baselines_{ds}.npz"
                and r["tag"] == "Base"
                and abs(float(r["holdout_frac"]) - 0.1) < 1e-12]


def _oriented(table, ds, k, a, b):
    """(lo, hi, diff) of a - b at k, or None if the row has NaN bounds."""
    hit = [r for r in table if int(r["k"]) == k and
           {r["method_a"], r["method_b"]} == {a, b}]
    if len(hit) != 1:
        sys.exit(f"derive_numbers: {len(hit)} rows for {ds} k={k} {a}/{b}")
    r = hit[0]
    lo, hi, diff = (float(r["ci_low"]), float(r["ci_high"]),
                    float(r["mean_diff"]))
    if r["method_a"] != a:
        lo, hi, diff = -hi, -lo, -diff
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo, hi, diff


def scoreboard_extra(numbers_dir, sizes, rows):
    src = "numbers/paired_*.csv"
    # MI - entropy over the 32 scoreboard cells
    counts = {"SigBetter": 0, "NotSig": 0, "SigWorse": 0}
    n = 0
    detail = []
    for ds in SCORE_DATASETS:
        table = _pair_rows(numbers_dir, ds)
        for k in SCORE_KS:
            if k > sizes[ds] - 3:
                continue
            o = _oriented(table, ds, k, "mi", "entropy")
            if o is None:
                continue
            n += 1
            cls = "SigBetter" if o[0] > 0 else "SigWorse" if o[1] < 0 else "NotSig"
            counts[cls] += 1
            if cls != "NotSig":
                detail.append(f"{ds} k={k} {cls} {o[2]:+.4f}")
    print("MI - entropy significant cells: " + "; ".join(detail))
    agg = f"count over {n} cells (11 matrices x k in {{3, 5, 10}}, k <= N - 3)"
    for cls, v in counts.items():
        rows.append(dict(key=f"ScoreMiVsEntropy{cls}", value=v,
                         source_file=src,
                         source_key="ci_low, ci_high of mi vs entropy",
                         index="source baselines_<ds>.npz, tag Base, holdout 0.1",
                         aggregation=agg, kind="int", scope="derived",
                         formula=("orient to MI - entropy; SigBetter: lo > 0, "
                                  "SigWorse: hi < 0, NotSig otherwise")))
    # FL - EV over every k = 2..min(15, N - 3)
    counts = {"SigBetter": 0, "NotSig": 0, "SigWorse": 0}
    deficit, mats, detail = [], set(), []
    n = 0
    for ds in SCORE_DATASETS:
        table = _pair_rows(numbers_dir, ds)
        for k in range(2, min(15, sizes[ds] - 3) + 1):
            o = _oriented(table, ds, k, "facloc", "trace")
            if o is None:
                continue
            n += 1
            cls = "SigBetter" if o[0] > 0 else "SigWorse" if o[1] < 0 else "NotSig"
            counts[cls] += 1
            if cls == "SigWorse":
                deficit.append(-o[2])
                mats.add(ds)
                detail.append(f"{ds} k={k} {-o[2]:.4f}")
    print("FL < EV (all k) cells: " + "; ".join(detail))
    agg = f"count over {n} cells (11 matrices x k = 2..min(15, N - 3))"
    for cls, v in counts.items():
        rows.append(dict(key=f"ScoreAllKFaclocVsTrace{cls}", value=v,
                         source_file=src,
                         source_key="ci_low, ci_high of facloc vs trace",
                         index="source baselines_<ds>.npz, tag Base, holdout 0.1",
                         aggregation=agg, kind="int", scope="derived",
                         formula=("orient to FL - EV; SigBetter: lo > 0, "
                                  "SigWorse: hi < 0, NotSig otherwise")))
    rows.append(dict(key="ScoreAllKNCells", value=n, source_file=src,
                     source_key="", index="", aggregation=agg, kind="int",
                     scope="derived", formula="number of (matrix, k) cells"))
    rows.append(dict(key="ScoreAllKFaclocVsTraceMaxDeficit",
                     value=repr(max(deficit)) if deficit else "nan",
                     source_file=src, source_key="mean_diff of facloc vs trace",
                     index="", aggregation=f"max over {len(deficit)} cells",
                     kind="float", scope="derived",
                     formula="max of (EV - FL) over the SigWorse cells, all k"))
    rows.append(dict(key="ScoreAllKFaclocVsTraceSigWorseNMatrices",
                     value=len(mats), source_file=src, source_key="",
                     index="", aggregation=agg, kind="int", scope="derived",
                     formula="matrices with at least one SigWorse cell: "
                             + ", ".join(sorted(mats))))


def _human(name):
    return name.replace("_", " ")


def mmlu_example(logs, data_dir, code_dir, rows):
    from collections import Counter
    with np.load(logs / "baselines_mmlu.npz", allow_pickle=True) as d:
        names = [str(x) for x in d["benchmark_names"]]
        sel = {m: np.asarray(d[f"{m}_selected"])
               for m in ("facloc", "trace", "mi")}
    modal = {}
    for m, s in sel.items():
        sets = Counter(frozenset(r[:5].tolist()) for r in s)
        top, cnt = sets.most_common(1)[0]
        first = next(i for i, r in enumerate(s)
                     if frozenset(r[:5].tolist()) == top)
        modal[m] = (top, [int(j) for j in s[first][:5]])
        M = {"facloc": "Facloc", "trace": "Trace", "mi": "Mi"}[m]
        rows.append(dict(key=f"SelMmlu{M}KfiveSameSplits", value=int(cnt),
                         source_file="logs/baselines_mmlu.npz",
                         source_key=f"{m}_selected", index="[:, :5]",
                         aggregation=f"splits (of {len(s)})", kind="int",
                         scope="derived",
                         formula="splits whose first five picks form the "
                                 "modal 5-set"))
    words = ["One", "Two", "Three", "Four", "Five"]
    for m, M in (("facloc", "Facloc"), ("mi", "Mi")):
        for w, j in zip(words, modal[m][1]):
            rows.append(dict(key=f"SelMmlu{M}KfivePick{w}",
                             value=_human(names[j]),
                             source_file="logs/baselines_mmlu.npz",
                             source_key=f"{m}_selected",
                             index="modal 5-set, order of its first split",
                             aggregation="", kind="text", scope="derived",
                             formula="benchmark name, underscores as spaces"))
    for m, M in (("trace", "Trace"), ("mi", "Mi")):
        rows.append(dict(key=f"SelMmlu{M}KfiveSharedWithFacloc",
                         value=len(modal[m][0] & modal["facloc"][0]),
                         source_file="logs/baselines_mmlu.npz",
                         source_key=f"{m}_selected, facloc_selected",
                         index="[:, :5]", aggregation="modal 5-sets",
                         kind="int", scope="derived",
                         formula="size of the intersection of the modal "
                                 "5-sets"))
    # covers of facility location's picks
    sys.path.insert(0, str(code_dir))
    from cv_split import cv_splits
    from eval_entropy_vs_mi import estimate_corr
    d = np.load(data_dir / "mmlu.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    if [str(x) for x in d["benchmark_names"]] != names:
        sys.exit("derive_numbers: mmlu benchmark names differ from the log")
    Mm = B.shape[0]
    counts_all = []
    for i, s in enumerate(cv_splits(Mm, 10, 5, 0.1, 42)):
        A = [int(j) for j in sel["facloc"][i][:5]]
        R, _, _ = estimate_corr(B[s.train_idx], O[s.train_idx])
        rp = np.maximum(R, 0.0)
        rest = [j for j in range(len(names)) if j not in A]
        best = np.argmax(rp[np.ix_(A, rest)], axis=0)
        counts_all.append(tuple(int((best == t).sum()) for t in range(5)))
    c0 = counts_all[0]
    same = sum(1 for c in counts_all if c == c0)
    print(f"MMLU FL covers split 0: {c0}; identical on {same}/50 splits; "
          f"distinct: {sorted(set(counts_all))}")
    if [int(j) for j in sel["facloc"][0][:5]] != modal["facloc"][1]:
        sys.exit("derive_numbers: split 0 is not the modal FL order")
    for w, c in zip(words, c0):
        rows.append(dict(key=f"CoverMmluFaclocKfivePick{w}", value=c,
                         source_file="logs/baselines_mmlu.npz; data/mmlu.matrix.npz",
                         source_key="facloc_selected; estimate_corr(train of split 0)",
                         index="split 0", aggregation="count of unselected subjects",
                         kind="int", scope="derived",
                         formula="unselected subjects whose largest clipped "
                                 "correlation with the five picks is with "
                                 "this pick (ties: earlier pick)"))
    rows.append(dict(key="CoverMmluFaclocKfiveSplitsSame", value=same,
                     source_file="logs/baselines_mmlu.npz; data/mmlu.matrix.npz",
                     source_key="as CoverMmluFaclocKfivePick*", index="",
                     aggregation="splits (of 50)", kind="int", scope="derived",
                     formula="splits whose cover counts equal split 0's"))


def fig4_points(logs, rows):
    from collections import Counter
    with np.load(logs / "analyze_objectives_merged.npz") as a:
        sr = np.asarray(a["single_rest_r2_tr"], float)
        s1 = np.asarray(a["single_r2"], float)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x = np.nanmean(sr, axis=0)
            y = np.nanmean(s1, axis=0)
    ok = np.isfinite(x) & np.isfinite(y)
    for key, v, f in (("FigMechMergedNPlotted", int(ok.sum()),
                       "benchmarks with finite x and y in figure 4b"),
                      ("FigMechMergedNOmitted", int((~ok).sum()),
                       "benchmarks without finite x or y in figure 4b")):
        rows.append(dict(key=key, value=v,
                         source_file="logs/analyze_objectives_merged.npz",
                         source_key="single_rest_r2_tr, single_r2", index="",
                         aggregation="nanmean over splits", kind="int",
                         scope="derived", formula=f))
    with np.load(logs / "baselines_merged.npz", allow_pickle=True) as d:
        for m, M in (("mi", "Mi"), ("trace", "Trace")):
            first = np.asarray(d[f"{m}_selected"])[:, 0]
            first = first[first >= 0]
            c = Counter(first.tolist())
            best = max(c.values())
            j = min(i for i, v in c.items() if v == best)
            for part, val in (("RestRsq", x[j]), ("SingleRsq", y[j])):
                rows.append(dict(
                    key=f"ObjModalFirstMerged{M}{part}", value=repr(float(val)),
                    source_file="logs/analyze_objectives_merged.npz",
                    source_key=("single_rest_r2_tr" if part == "RestRsq"
                                else "single_r2"),
                    index=f"[:, {j}] (most frequent first pick of {m})",
                    aggregation="nanmean over splits", kind="float",
                    scope="derived",
                    formula="coordinate of the method's marker in figure 4b"))


def check(rows):
    got = {r["key"]: float(r["value"]) for r in rows if r["kind"] != "text"}
    bad = []
    for key, want in EXPECTED.items():
        tol = EXPECTED_TOL.get(key, DEFAULT_TOL)
        if key not in got:
            bad.append(f"{key}: missing (expected {want})")
        elif not abs(got[key] - want) <= tol:
            bad.append(f"{key}: got {got[key]:.6g}, expected {want}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, type=pathlib.Path,
                    help="canonical run output (with numbers/ and logs/)")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--tree", type=pathlib.Path, default=None,
                    help="code/data tree of the run (default ROOT/../tree)")
    args = ap.parse_args(argv)
    numbers = args.root / "numbers"
    tree = args.tree or (args.root.parent / "tree")
    rows = []
    sizes = read_sizes(numbers / "numbers.csv")
    scoreboard(numbers, sizes, rows)
    ties_and_cover(args.root / "logs", rows)
    scoreboard_extra(numbers, sizes, rows)
    mmlu_example(args.root / "logs", tree / "data", tree / "code", rows)
    fig4_points(args.root / "logs", rows)
    keys = [r["key"] for r in rows]
    if len(keys) != len(set(keys)):
        sys.exit("derive_numbers: duplicate keys")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"derive_numbers: wrote {len(rows)} keys to {args.out}")
    bad = check(rows)
    for msg in bad:
        print(f"EXPECTED VALUE NOT REPRODUCED: {msg}", file=sys.stderr)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
