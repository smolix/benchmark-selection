"""Compare submodular selection against simple baseline heuristics.

Runs nine selection methods under the matched (paired) repeated-CV design
of cv_split.cv_splits (10% holdout; --holdout for others):

  entropy     — Algorithm 1 (greedy entropy / pivoted Cholesky)
  mi          — Algorithm 2 (greedy mutual information)
  random      — uniform random selection
  hub         — top-k by mean |correlation| (Reviewer QZ19's heuristic)
  pca         — PCA-based subset selection (pivoted QR on top-k loadings)
  trace       — forward selection minimizing total residual variance
  facloc      — greedy facility location on the correlation matrix clipped
                at 0 (Yao et al. 2026, arXiv 2607.09739;
                baseline_select.facility_location_select), the paper's
                headline method since 2026-10-19.  Deterministic, no random
                numbers: appending it shifts no other stream.
  facloc_abs, facloc_sq
              — the similarity ablation of facloc (FACLOC_VARIANTS):
                facility location on |r| (anti-correlated proxies credited)
                and on r^2 (each target credited with the variance its best
                single proxy explains; at k = 1 the trace score), on the
                same splits, so the paper can say why the clipped
                correlation is used.  Deterministic and appended after
                facloc: no other method's rows change.
(entropy_rf, entropy with a uniformly random first pivot, was dropped on
2026-10-18, protocol rule D3; no other stream changed.)

Covariance estimation and imputation are imported from
eval_entropy_vs_mi so every method sees the same splits, the same
estimated correlation matrix, and the same evaluation.

Matched design.  The splits are cv_splits(M, folds, repeats, 0.1, SEED):
--repeats repetitions of --folds-fold CV (default 5 x 10 = 50 splits).
Repetition 0 is the original single run (cv_folds(M, 10, 42) and the
seed-42 training subsample), so its rows reproduce earlier logs and
--repeats 1 is the original protocol.  In every split all methods share

  * the train / validation indices,
  * one eligibility mask, eligible_mask(O_train): only benchmarks observed
    for at least MIN_TRAIN_OBS training models can be selected.
    Ineligible benchmarks stay in the correlation matrix (MI complement,
    trace objective, hub scores, PCA loadings) and remain imputation
    targets.  When fewer than k benchmarks are eligible, R², RMSE and
    residual variance are NaN at that k;
  * one target mask, target_mask(O_train) (protocol rule D1): the cells
    of benchmarks observed by fewer than MIN_TARGET_OBS = 2 training
    models are never scored, and every prediction is clipped to its
    benchmark's valid range (rule D2, score_bounds.py) before scoring;
  * common random numbers: the random baseline's permutation is seeded
    with Split.seed(SEED + RANDOM_SEED_OFFSET) (rep 0: (49, fold, 1000),
    as before).

Near-ties of every selector's score are broken by explained variance,
then by the lowest index (greedy_select / baseline_select); the picks the
index decided are counted in {m}_index_ties.

Sparse data.  impute_and_evaluate scores a validation model only if it
observes at least one selected benchmark, so on sparse matrices (Merged,
Epoch AI) a method's R² at small k covers fewer cells when it picks
rarely run benchmarks: methods are then pooled over different cells
(Epoch at k = 1, 50 splits: 64% of the validation cells for entropy, 11%
for mi).
The share of validation cells scored (evaluated_share) is logged per
split and k ({m}_cover) and printed when it is below 1 somewhere; filter
or weight paired comparisons with it where it differs between methods.

Outputs (LOGS = path_config.logs_dir(), OUT = path_config.figures_dir()):

  LOGS/baselines_{ds}.npz
    benchmark_names
    {m}_r2, {m}_rmse, {m}_resid_var   (n_splits, k_max) float, row i is
                                      split i (split order of cv_splits)
    {m}_cover                         (n_splits, k_max) float, share of
                                      the validation cells that {m}_r2
                                      scores (evaluated_share; 1 on fully
                                      observed data; NaN where the split
                                      has no size-k selection)
    methods                           (n_methods,) method names
    split_rep, split_fold, split_id   (n_splits,) int; pair methods (and
                                      scripts) on split_id = rep·folds + fold
    n_eligible, n_train, n_val        (n_splits,) int
    split_ok                          (n_splits,) bool, estimation succeeded
                                      (rows of failed splits are NaN)
    eligible                          (n_splits, N) bool selectable mask
    n_targets                         (n_splits,) int, imputation targets
                                      (target_mask, rule D1)
    {m}_n_clipped                     (n_splits, k_max) int, predictions
                                      the D2 clip changed
    {m}_selected                      (n_splits, k_max) int, selection order,
                                      -1 padded (pca: its largest set)
    pca_selected_per_k                (n_splits, k_max, k_max) int, row kk-1
                                      = pca's set of size kk, -1 padded
    {m}_index_ties                    (n_splits, k_max) int, picks of the
                                      size-k selection decided by the lowest
                                      index (greedy_select.index_tie_count)
    {m}_tie_breaks                    (n_splits, k_max) int, picks of the
                                      size-k selection made among near-tied
                                      scores (greedy_select.tie_break_count;
                                      the layout of {m}_index_ties, as in
                                      every log)
    pca_n_eigvecs, pca_eig_skipped    (n_splits, k_max) int, eigenvector
                                      directions spanning PCA's size-k
                                      subspace (k unless a repeated
                                      eigenvalue's eigenspace was completed)
                                      and directions without loading on the
                                      eligible benchmarks skipped above them
                                      (baseline_select._pca_basis; 0 if
                                      fewer than k eligible)
    holdout_frac, n_folds, n_repeats, seed, min_train_obs,
    min_target_obs, pred_clip, k_max  scalars
    Datasets with columns redundant by construction (redundant_groups:
    the item_group of data/{ds}.meta.json shared by two or more columns,
    so far olv2sub's four ifeval/* metrics of one run and its three
    nested gpqa/* subsets) also store
    redundant_groups                  (G,) group labels (shared name prefix)
    redundant_members                 (G, max size) int, -1 padded
    {m}_group_picks                   (n_splits, G, k_max) int, members of
                                      group g in the size-k selection (pca:
                                      its size-k set), -1 where the split
                                      has no size-k selection
  LOGS/baselines_{ds}_redundancy.json  (those datasets only) per group,
    method and k in REDUNDANCY_K: fraction of splits whose size-k selection
    holds more than one member of the group and mean members picked, the
    hypergeometric P(> 1) of a uniformly random size-k subset of the
    eligible benchmarks, config, and a flat "macros" dict
    (RedundBase{Ds}{Group}{M}K{k}, RedundPicksBase..., RedundHyperBase
    {Ds}{Group}K{k}; make_numbers.camel naming).  Deleted at the start
    of every run of the dataset, so a failed run leaves no stale file.
    Not yet declared as an output of run_all.sh's baselines job nor read
    by make_numbers.py (their owners' changes)
  LOGS/baselines_{ds}_paired.csv
    paired R² differences method_a − method_b for every method pair and k
    over the matched splits (paired_stats.paired_stats: mean, Nadeau–Bengio
    corrected standard error, 95% t-interval, win rate; pairs oriented by
    paired_stats.orient, e.g. mi − entropy; columns dataset, holdout_frac,
    k, method_a, method_b, then paired_stats.STAT_FIELDS), the same rows
    paired_stats.py writes for this log
  OUT/baselines_{ds}.pdf

Datasets are addressed by name: --datasets NAME reads
data/NAME.matrix.npz (B with missing = 0, mask O, benchmark_names).

Usage:
    python eval_baselines.py [--datasets mmlu mteb merged benchpress ...]
                             [--folds 10] [--repeats 5] [--kmax 15]
                             [--holdout 0.1]
                             [--workers 1]

--holdout P (default 0.1) runs the matched splits of holdout P (e.g. 0.9
for the protocol-effect tables); its outputs keep the file names above,
so give such a run its own BENCHSELECT_EXPERIMENT_ROOT.

--workers W evaluates the splits in W spawned single-threaded processes
(eval_entropy_vs_mi.map_splits): every split is a pure function of the
data and the split (common random numbers from Split.seed), so the logs do
not depend on W, and the per-split progress lines are printed in split
order (stdout differs from the serial run's in the timings only).  The
default W = 1 runs the serial loop, as before.
"""

import argparse
import json
import math
import pathlib
import sys
import time
import warnings
import numpy as np
from contextlib import closing

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import (greedy_entropy, greedy_mi, index_tie_count,
                           random_select, selection_at_k, tie_break_count)
from baseline_select import (facility_location_select, greedy_trace_select,
                             hub_select, pca_cssp_select)
from cv_split import (MIN_TRAIN_OBS, N_REPEATS, RANDOM_SEED_OFFSET,
                      cv_splits, eligible_mask, target_mask)
from eval_entropy_vs_mi import (estimate_corr, impute_and_evaluate,
                                map_splits, protocol_config, say_now)
from score_bounds import dataset_bounds
from paired_stats import STAT_FIELDS, orient, paired_stats, write_csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42

DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
METHODS = ["entropy", "mi", "random", "hub", "pca", "trace", "facloc",
           "facloc_abs", "facloc_sq"]
# facility_location_select's similarity of each facility-location method
# (baseline_select.FACLOC_SIMILARITIES): facloc and its ablation variants
FACLOC_VARIANTS = {"facloc": "clipped", "facloc_abs": "abs",
                   "facloc_sq": "sq"}
METRICS = ["r2", "rmse", "resid_var"]

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "mteb": "MTEB",
    "merged": "Merged",
    "benchpress": "BenchPress",
    # data release (data/build_*.py)
    "olv2sub": "OLL v2 sub-tasks",
    "olv1items": "OLL v1 tasks",
    "epoch": "Epoch AI",
    "helm_lite": "HELM Lite",
    "helm_mmlu": "HELM MMLU",
    "helm_classic": "HELM Classic",
    "mteb2026": "MTEB 2026",
}

METHOD_LABEL = {
    "entropy": "Entropy (Alg. 1)",
    "mi": "Mutual Info (Alg. 2)",
    "random": "Random",
    "hub": "Hub correlation",
    "pca": "PCA subset (CSSP)",
    "trace": "Greedy trace min.",
    "facloc": "Facility location",
    "facloc_abs": "Facility location, |r|",
    "facloc_sq": "Facility location, r²",
}

METHOD_COLOR = {
    "entropy": "#4477AA",
    "mi": "#CC3311",
    "random": "#888888",
    "hub": "#228833",
    "pca": "#AA3377",
    "trace": "#EE7733",
    "facloc": "#009988",
    "facloc_abs": "#33BBEE",
    "facloc_sq": "#44AA99",
}

METHOD_STYLE = {
    "entropy": "o-",
    "mi": "o-",
    "random": "x--",
    "hub": "s-",
    "pca": "d-",
    "trace": "^-",
    "facloc": "v-",
    "facloc_abs": "v--",
    "facloc_sq": "v:",
}

PAIRED_FIELDS = (["dataset", "holdout_frac", "k", "method_a", "method_b"]
                 + STAT_FIELDS)

# k of the redundancy report (redundant_groups)
REDUNDANCY_K = [5, 10]


def display_name(ds_name):
    """Display name of a dataset (the raw name if none is registered)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def run_selection(method, R, k_max, benchmark_names, split, eligible):
    """Run `method` on one split: shared eligibility mask and the split's
    common random numbers (the random baseline's permutation)."""
    if method == "entropy":
        return greedy_entropy(R, k_max, benchmark_names=benchmark_names,
                              eligible=eligible)
    if method == "mi":
        return greedy_mi(R, k_max, benchmark_names=benchmark_names,
                         eligible=eligible)
    if method == "random":
        return random_select(R, k_max,
                             seed=split.seed(SEED + RANDOM_SEED_OFFSET),
                             benchmark_names=benchmark_names,
                             eligible=eligible)
    if method == "hub":
        return hub_select(R, k_max, benchmark_names=benchmark_names,
                          eligible=eligible)
    if method == "pca":
        return pca_cssp_select(R, k_max, benchmark_names=benchmark_names,
                               eligible=eligible)
    if method == "trace":
        return greedy_trace_select(R, k_max, benchmark_names=benchmark_names,
                                   eligible=eligible)
    if method in FACLOC_VARIANTS:
        return facility_location_select(R, k_max,
                                        benchmark_names=benchmark_names,
                                        eligible=eligible,
                                        similarity=FACLOC_VARIANTS[method])
    raise ValueError(method)


def _padded(indices, length):
    """Integer array of `indices`, padded with -1 to `length`."""
    out = np.full(length, -1, dtype=np.int64)
    out[:len(indices)] = indices
    return out


def evaluated_share(O_test, selected, target=None):
    """Share of the validation cells that impute_and_evaluate scores for
    the selection `selected`: the observed unselected target cells of the
    models that observe at least one selected benchmark, over the observed
    unselected target cells of all validation models (1 on fully observed
    data).  target: the split's target_mask (D1; None: every column).
    On sparse data (Epoch AI: 18% observed) a selection of rarely run
    benchmarks is scored on few models, so R² of different methods covers
    different cells; print_summary reports this share."""
    O_test = np.asarray(O_test) > 0
    cols = (np.ones(O_test.shape[1], dtype=bool) if target is None
            else np.array(target, dtype=bool))
    cols[list(selected)] = False
    cells = O_test[:, cols].sum(axis=1)
    scored = O_test[:, list(selected)].any(axis=1)
    total = cells.sum()
    return float(cells[scored].sum() / total) if total else np.nan


def redundancy_json_path(ds_name):
    """LOGS/baselines_{ds}_redundancy.json (redundant_groups datasets)."""
    return LOGS / f"baselines_{ds_name}_redundancy.json"


def _nan_mean_std(a, axis=0):
    """np.nanmean / np.nanstd (ddof=0) without all-NaN-slice warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)


def baselines_split(ctx, s, say=say_now):
    """One matched split of run_dataset (its loop body): the split's
    eligible mask, n_eligible and n_targets, ok (estimation succeeded)
    and, if ok, rows[m] = method m's (k_max,) rows of every per-split
    array of run_dataset (METRICS, cover, n_clipped, selected,
    index_ties, tie_breaks; pca also selected_per_k, n_eigvecs,
    eig_skipped), filled exactly as run_dataset fills row i; progress via
    say (eval_entropy_vs_mi.map_splits)."""
    t0 = time.time()
    B, O, N, k_max = ctx["B"], ctx["O"], ctx["N"], ctx["k_max"]
    B_train, O_train = B[s.train_idx], O[s.train_idx]
    B_test, O_test = B[s.val_idx], O[s.val_idx]
    el = eligible_mask(O_train)
    target = target_mask(O_train)        # D1, shared by every method
    n_el = int(el.sum())
    part = {"eligible": el, "n_eligible": n_el,
            "n_targets": int(target.sum()), "ok": False}

    say(f"  Split {s.split_id} (rep {s.rep}, fold {s.fold}): "
        f"{len(s.train_idx)} train / {len(s.val_idx)} test, "
        f"{n_el}/{N} eligible", end="")

    try:
        R, col_mean, col_std = estimate_corr(B_train, O_train)
    except Exception as e:
        say(f" — estimation failed: {e}")
        return part
    part["ok"] = True

    rows = {}
    for method in ctx["methods"]:
        result = run_selection(method, R, k_max, ctx["names"], s, el)
        row = {key: np.full(k_max, np.nan) for key in METRICS}
        row["cover"] = np.full(k_max, np.nan)
        row["n_clipped"] = np.zeros(k_max, dtype=np.int64)
        row["selected"] = _padded(result["selected"], k_max)
        row["index_ties"] = np.zeros(k_max, dtype=np.int64)
        row["tie_breaks"] = np.zeros(k_max, dtype=np.int64)
        if method == "pca":
            row["selected_per_k"] = np.full((k_max, k_max), -1,
                                            dtype=np.int64)
            for kk, sel_kk in enumerate(result["selected_per_k"]):
                row["selected_per_k"][kk] = _padded(sel_kk, k_max)
            row["n_eigvecs"] = np.zeros(k_max, dtype=np.int64)
            row["eig_skipped"] = np.zeros(k_max, dtype=np.int64)
            n_pca = len(result["n_eigvecs"])
            row["n_eigvecs"][:n_pca] = result["n_eigvecs"]
            row["eig_skipped"][:n_pca] = result["eig_skipped"]
        for k in range(1, k_max + 1):
            row["index_ties"][k - 1] = index_tie_count(result, k)
            row["tie_breaks"][k - 1] = tie_break_count(result, k)
            row["resid_var"][k - 1] = result["residual_variance"][k - 1]
            selected = selection_at_k(result, k)
            if selected is None:      # fewer than k eligible: NaN
                continue
            imp = impute_and_evaluate(B_test, O_test, selected, R,
                                      col_mean, col_std, N,
                                      target=target, bounds=ctx["bounds"])
            row["r2"][k - 1] = imp["r2"]
            row["rmse"][k - 1] = imp["rmse"]
            row["n_clipped"][k - 1] = imp["n_clipped"]
            row["cover"][k - 1] = evaluated_share(O_test, selected, target)
        rows[method] = row
    part["rows"] = rows

    elapsed = time.time() - t0
    at5 = min(5, k_max) - 1
    say("  " + "  ".join(
        f"{m}:{rows[m]['r2'][at5]:.3f}" for m in ctx["methods"])
        + f"  ({elapsed:.1f}s)")
    return part


def run_dataset(ds_name, k_max, n_folds, n_repeats, workers=1):
    """Every method on the matched splits of one dataset: (metrics, out,
    benchmark_names, k_max, cover), the per-split arrays (rows in
    cv_splits order) of the module docstring.  workers > 1 evaluates the
    splits in that many processes (eval_entropy_vs_mi.map_splits; the
    arrays and progress lines are those of the serial run)."""
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)     # D2
    obs_frac = O.sum() / (M * N)
    print(f"{M} models × {N} tasks, {obs_frac*100:.1f}% observed", flush=True)

    p = HOLDOUT_FRAC
    k_max = min(k_max, N)
    splits = list(cv_splits(M, n_folds, n_repeats, p, SEED))
    S = len(splits)
    proto = protocol_config()
    print(f"{n_repeats} × {n_folds}-fold CV = {S} matched splits, holdout "
          f"{p:g}, selectable: >= {MIN_TRAIN_OBS} training observations, "
          f"targets: >= {proto['min_target_obs']}, prediction clip: "
          f"{proto['pred_clip']}", flush=True)

    metrics = {m: {key: np.full((S, k_max), np.nan) for key in METRICS}
               for m in METHODS}
    # Share of the validation cells R² covers (evaluated_share), per split
    cover = {m: np.full((S, k_max), np.nan) for m in METHODS}
    # Predictions changed by the D2 clip, per split and k
    clipped = {m: np.zeros((S, k_max), dtype=np.int64) for m in METHODS}
    out = {
        "split_rep": np.array([s.rep for s in splits], dtype=np.int64),
        "split_fold": np.array([s.fold for s in splits], dtype=np.int64),
        "split_id": np.array([s.split_id for s in splits], dtype=np.int64),
        "n_eligible": np.zeros(S, dtype=np.int64),
        "n_train": np.array([len(s.train_idx) for s in splits],
                            dtype=np.int64),
        "n_val": np.array([len(s.val_idx) for s in splits], dtype=np.int64),
        "split_ok": np.zeros(S, dtype=bool),
        "eligible": np.zeros((S, N), dtype=bool),
        "n_targets": np.zeros(S, dtype=np.int64),
        "pca_selected_per_k": np.full((S, k_max, k_max), -1, dtype=np.int64),
        "pca_n_eigvecs": np.zeros((S, k_max), dtype=np.int64),
        "pca_eig_skipped": np.zeros((S, k_max), dtype=np.int64),
    }
    for m in METHODS:
        out[f"{m}_selected"] = np.full((S, k_max), -1, dtype=np.int64)
        out[f"{m}_index_ties"] = np.zeros((S, k_max), dtype=np.int64)
        out[f"{m}_tie_breaks"] = np.zeros((S, k_max), dtype=np.int64)

    ctx = dict(B=B, O=O, N=N, k_max=k_max, names=benchmark_names,
               bounds=bounds, methods=list(METHODS))
    with closing(map_splits(baselines_split, ctx, splits, workers)) as parts:
        for i, part in enumerate(parts):
            out["eligible"][i] = part["eligible"]
            out["n_eligible"][i] = part["n_eligible"]
            out["n_targets"][i] = part["n_targets"]
            if not part["ok"]:
                continue
            out["split_ok"][i] = True
            for method in METHODS:
                row = part["rows"][method]
                out[f"{method}_selected"][i] = row["selected"]
                if method == "pca":
                    out["pca_selected_per_k"][i] = row["selected_per_k"]
                    out["pca_n_eigvecs"][i] = row["n_eigvecs"]
                    out["pca_eig_skipped"][i] = row["eig_skipped"]
                out[f"{method}_index_ties"][i] = row["index_ties"]
                out[f"{method}_tie_breaks"][i] = row["tie_breaks"]
                for key in METRICS:
                    metrics[method][key][i] = row[key]
                clipped[method][i] = row["n_clipped"]
                cover[method][i] = row["cover"]

    for m in METHODS:
        out[f"{m}_n_clipped"] = clipped[m]
    return metrics, out, benchmark_names, k_max, cover


# ── Paired statistics over matched splits ────────────────────────

def paired_rows(ds_name, metrics, out, k_max):
    """paired_stats.paired_stats of R² for every method pair (oriented by
    paired_stats.orient, e.g. mi − entropy) and every k, as CSV rows
    (PAIRED_FIELDS)."""
    rows = []
    for k in range(1, k_max + 1):
        for ma, mb in orient(METHODS):
            st = paired_stats(metrics[ma]["r2"][:, k - 1],
                              metrics[mb]["r2"][:, k - 1],
                              out["n_train"], out["n_val"], out["split_rep"])
            rows.append({"dataset": ds_name, "holdout_frac": HOLDOUT_FRAC,
                         "k": k, "method_a": ma, "method_b": mb, **st})
    return rows


def write_paired_csv(path, rows):
    write_csv(rows, path, PAIRED_FIELDS)
    print(f"Paired statistics saved to {path}", flush=True)


# ── Columns redundant by construction ────────────────────────────

def redundant_groups(ds_name, benchmark_names):
    """[(label, member column indices)] of the columns of ds_name sharing
    an item_group in data/{ds_name}.meta.json ("benchmarks": {"item_group":
    [...]}, aligned with its "benchmark_names"): several metrics of one
    evaluation run or nested question subsets, redundant by construction
    (olv2sub: the four ifeval/* metrics, the three gpqa/* subsets).  The
    label is the members' common name prefix before "/" (else the
    item_group).  [] for datasets without such metadata."""
    path = DATA / f"{ds_name}.meta.json"
    if not path.is_file():
        return []
    with open(path) as f:
        meta = json.load(f)
    bench = meta.get("benchmarks")
    item_group = bench.get("item_group") if isinstance(bench, dict) else None
    meta_names = meta.get("benchmark_names")
    if (not isinstance(item_group, list) or not isinstance(meta_names, list)
            or len(item_group) != len(meta_names)):
        return []
    col = {str(n): j for j, n in enumerate(benchmark_names)}
    members = {}
    for name, grp in zip(meta_names, item_group):
        if grp is not None and str(name) in col:
            members.setdefault(str(grp), []).append(col[str(name)])
    groups = []
    for grp, idx in members.items():
        if len(idx) < 2:
            continue
        names = [str(benchmark_names[j]) for j in idx]
        prefixes = {n.split("/")[0] for n in names}
        label = (prefixes.pop() if len(prefixes) == 1
                 and all("/" in n for n in names) else grp)
        groups.append((label, sorted(idx)))
    return groups


def group_picks(out, groups, k_max):
    """{m: (n_splits, G, k_max) int}: members of group g in method m's
    size-k selection (pca: its size-k set selected_per_k), -1 where the
    split has no size-k selection (failed estimate, fewer than k
    eligible)."""
    S = len(out["split_id"])
    picks = {}
    for m in METHODS:
        arr = np.full((S, len(groups), k_max), -1, dtype=np.int64)
        for k in range(1, k_max + 1):
            sel = (out["pca_selected_per_k"][:, k - 1, :k] if m == "pca"
                   else out[f"{m}_selected"][:, :k])
            has = np.all(sel >= 0, axis=1) & out["split_ok"]
            for g, (_, idx) in enumerate(groups):
                arr[:, g, k - 1] = np.where(
                    has, np.isin(sel, idx).sum(axis=1), -1)
        picks[m] = arr
    return picks


def hypergeom_multi(n, g, k):
    """P(more than one of g marked items among k drawn uniformly without
    replacement from n): the redundancy rate of a random size-k subset."""
    if k > n:
        return np.nan
    total = math.comb(n, k)
    p0 = math.comb(n - g, k) / total
    p1 = g * math.comb(n - g, k - 1) / total
    return 1.0 - p0 - p1


def redundancy_summary(ds_name, picks, groups, out, k_max):
    """Rows (group, k, method, n_splits, n_multi, frac_multi, mean_picks,
    hypergeom_multi) for k in REDUNDANCY_K and the flat macros dict
    {RedundBase{Ds}{Group}{M}K{k}: frac_multi, RedundPicksBase...:
    mean_picks, RedundHyperBase{Ds}{Group}K{k}: hypergeom_multi}."""
    from make_numbers import camel, int_words
    D = camel(ds_name)
    el = out["eligible"]
    n_el = el.sum(axis=1)
    rows, macros = [], {}
    for g, (label, idx) in enumerate(groups):
        G = camel(label)
        g_el = el[:, idx].sum(axis=1)
        for k in REDUNDANCY_K:
            if k > k_max:
                continue
            ok = out["split_ok"] & (n_el >= k)
            hyper = (float(np.mean([hypergeom_multi(int(n), int(gg), k)
                                    for n, gg in zip(n_el[ok], g_el[ok])]))
                     if ok.any() else np.nan)
            macros[f"RedundHyperBase{D}{G}K{int_words(k)}"] = hyper
            for m in METHODS:
                c = picks[m][:, g, k - 1]
                v = c[c >= 0]
                frac = float(np.mean(v > 1)) if v.size else np.nan
                mean = float(np.mean(v)) if v.size else np.nan
                rows.append({"group": label, "k": k, "method": m,
                             "n_splits": int(v.size),
                             "n_multi": int(np.sum(v > 1)),
                             "frac_multi": frac, "mean_picks": mean,
                             "hypergeom_multi": hyper})
                macros[f"RedundBase{D}{G}{camel(m)}K{int_words(k)}"] = frac
                macros[f"RedundPicksBase{D}{G}{camel(m)}K{int_words(k)}"] = \
                    mean
    return rows, macros


def print_redundancy(ds_name, rows, groups, benchmark_names):
    print(f"\n== {display_name(ds_name)}: selections with more than one "
          f"column of a redundant group (share of splits; mean members "
          f"picked) ==")
    for label, idx in groups:
        print(f"  {label}: " + ", ".join(str(benchmark_names[j])
                                         for j in idx))
    print(f"  {'group':<10}{'k':<4}" + "".join(f"{m:<14}" for m in METHODS)
          + "random subset")
    for label, _ in groups:
        for k in REDUNDANCY_K:
            sel = [r for r in rows if r["group"] == label and r["k"] == k]
            if not sel:
                continue
            by_m = {r["method"]: r for r in sel}
            print(f"  {label:<10}{k:<4}" + "".join(
                f"{by_m[m]['frac_multi']:.2f} ({by_m[m]['mean_picks']:.2f})"
                .ljust(14) for m in METHODS)
                + f"{sel[0]['hypergeom_multi']:.2f}")


def _json_safe(obj):
    """NaN -> None (strict JSON), recursively."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_redundancy_json(path, ds_name, rows, macros, groups,
                          benchmark_names, out, n_folds, n_repeats, k_max):
    obj = {
        "dataset": ds_name,
        "script": "eval_baselines.py",
        "description": "share of matched splits whose size-k selection "
                       "holds more than one column of a group redundant by "
                       "construction (data/{ds}.meta.json item_group); "
                       "hypergeom_multi: the same probability for a "
                       "uniformly random size-k subset of the eligible "
                       "benchmarks, averaged over splits",
        "config": {"holdout_frac": HOLDOUT_FRAC, "n_folds": n_folds,
                   "n_repeats": n_repeats, "seed": SEED,
                   **protocol_config(), "k_max": k_max,
                   "ks": [k for k in REDUNDANCY_K if k <= k_max],
                   "methods": METHODS,
                   "n_splits": int(len(out["split_id"]))},
        "groups": {label: [str(benchmark_names[j]) for j in idx]
                   for label, idx in groups},
        "summary": rows,
        "macros": macros,
    }
    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=1)
    print(f"Redundancy summary saved to {path}", flush=True)


# ── Reporting ────────────────────────────────────────────────────

def print_summary(ds_name, metrics, out, k_max, paired, cover=None):
    ks_report = [k for k in [1, 3, 5, 10, 15] if k <= k_max]
    S = len(out["split_id"])
    print(f"\n== {display_name(ds_name)}: "
          f"imputation R² (mean ± std over {S} splits) ==")
    header = f"{'method':<26}" + "".join(f"k={k:<12}" for k in ks_report)
    print(header)
    for m in METHODS:
        r2 = metrics[m]["r2"]
        valid = ~np.all(np.isnan(r2), axis=1)
        if not valid.any():
            print(f"{METHOD_LABEL[m]:<26}" + "n/a")
            continue
        cells = []
        for k in ks_report:
            mu, sd = _nan_mean_std(r2[valid, k - 1])
            cells.append(f"{mu:.3f}±{sd:.3f}")
        print(f"{METHOD_LABEL[m]:<26}" + "".join(f"{c:<14}" for c in cells))
    if cover is not None and any(np.nanmin(c, initial=1.0) < 1.0
                                 for c in cover.values()):
        print("share of validation cells scored (models observing >= 1 "
              "selected benchmark; mean over splits):")
        for m in METHODS:
            means = [_nan_mean_std(cover[m][:, k - 1])[0] for k in ks_report]
            print(f"{METHOD_LABEL[m]:<26}"
                  + "".join(f"{v:<14.2f}" for v in means))

    n_el = out["n_eligible"]
    print(f"eligible benchmarks per split: min {n_el.min()}, "
          f"median {int(np.median(n_el))}, max {n_el.max()} of "
          f"{out['eligible'].shape[1]}"
          + (f"; {int((n_el < k_max).sum())} splits with fewer than "
             f"k_max = {k_max} (NaN beyond)" if (n_el < k_max).any() else ""))
    print(f"index ties (picks decided by column order) at k = {k_max}, "
          "splits affected / total picks: " + ", ".join(
              f"{m} {int(np.count_nonzero(out[f'{m}_index_ties'][:, -1]))}/"
              f"{int(out[f'{m}_index_ties'][:, -1].sum())}" for m in METHODS))
    ks_all = np.arange(1, k_max + 1)
    n_skip = int(np.count_nonzero(out["pca_eig_skipped"].any(axis=1)))
    n_ext = int(np.count_nonzero(
        (out["pca_n_eigvecs"] > ks_all).any(axis=1)))
    print(f"PCA basis: {n_skip} splits with eigenvector directions skipped "
          f"for lack of eligible loading (max {out['pca_eig_skipped'].max()}"
          f"), {n_ext} with a repeated eigenvalue's eigenspace completed")

    ks_pair = [k for k in [5, 10] if k <= k_max] or [k_max]
    print(f"\n== {display_name(ds_name)}: paired R² differences a − b "
          f"(Nadeau–Bengio 95% CI, win rate) ==")
    for row in paired:
        if (row["k"] not in ks_pair or row["n_splits"] == 0
                or (row["method_a"] not in ("entropy", "mi")
                    and "facloc" not in (row["method_a"],
                                         row["method_b"]))):
            continue
        print(f"  k={row['k']:<3}{row['method_a']:>8} − {row['method_b']:<11}"
              f"{row['mean_diff']:+.3f}  [{row['ci_low']:+.3f}, "
              f"{row['ci_high']:+.3f}]  win {row['win_rate']:.2f}  "
              f"(n={row['n_splits']})")


def plot_comparison(ds_name, metrics, k_max):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ks = np.arange(1, k_max + 1)

    for method in METHODS:
        c = METHOD_COLOR[method]
        lab = METHOD_LABEL[method]
        ls = METHOD_STYLE[method]
        main = method in ("entropy", "mi", "facloc")
        lw = 1.5 if main else 1.0
        ms = 4 if main else 3

        r2 = metrics[method]["r2"]
        valid = ~np.all(np.isnan(r2), axis=1)
        if not valid.any():
            continue
        mean_r2, std_r2 = _nan_mean_std(r2[valid], axis=0)
        mean_rv, _ = _nan_mean_std(metrics[method]["resid_var"][valid],
                                   axis=0)

        axes[0].plot(ks, mean_r2, ls, color=c, label=lab,
                     markersize=ms, linewidth=lw)
        axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                             color=c, alpha=0.10)
        axes[1].semilogy(ks, mean_rv, ls, color=c, label=lab,
                         markersize=ms, linewidth=lw)

    axes[0].set_ylabel("Imputation $R^2$")
    axes[1].set_ylabel("Residual variance fraction")
    for ax in axes:
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xlim(0.5, k_max + 0.5)
        ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= k_max])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = OUT / f"baselines_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {out_path}", flush=True)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names, read from "
                             "data/<name>.matrix.npz (default: %(default)s)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="CV repetitions; splits = repeats × folds, "
                             "rep 0 = the original seed-42 folds "
                             "(default: %(default)s)")
    parser.add_argument("--kmax", type=int, default=K_MAX)
    parser.add_argument("--holdout", type=float, default=HOLDOUT_FRAC,
                        help="holdout fraction of the matched splits "
                             "(default %(default)s; outputs keep their "
                             "names: use a separate experiment root)")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes over the splits (results do not "
                             "depend on it; default %(default)s)")
    args = parser.parse_args()
    if args.folds < 2 or args.repeats < 1 or args.kmax < 1:
        parser.error("need --folds >= 2, --repeats >= 1 and --kmax >= 1")
    if not 0.0 < args.holdout < 1.0:
        parser.error("--holdout must lie in (0, 1)")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    HOLDOUT_FRAC = args.holdout
    missing = [ds for ds in args.datasets
               if not (DATA / f"{ds}.matrix.npz").is_file()]
    if missing:
        avail = sorted(p.name[:-len(".matrix.npz")]
                       for p in DATA.glob("*.matrix.npz"))
        parser.error(f"no {DATA}/<name>.matrix.npz for {missing}; "
                     f"available: {' '.join(avail)}")

    total_t0 = time.time()
    for ds_name in args.datasets:
        ds_t0 = time.time()
        # A redundancy summary of an earlier run must not outlive this one
        # (run_all.sh does not list it among the job's outputs)
        redundancy_json_path(ds_name).unlink(missing_ok=True)
        metrics, out, bench_names, k_max, cover = run_dataset(
            ds_name, args.kmax, args.folds, args.repeats, args.workers)
        paired = paired_rows(ds_name, metrics, out, k_max)
        print_summary(ds_name, metrics, out, k_max, paired, cover)
        plot_comparison(ds_name, metrics, k_max)

        save_dict = {"benchmark_names": bench_names}
        for m in METHODS:
            for key, arr in metrics[m].items():
                save_dict[f"{m}_{key}"] = arr
        for m in METHODS:
            save_dict[f"{m}_cover"] = cover[m]
        save_dict["methods"] = np.array(METHODS)
        save_dict.update(out)
        save_dict.update(holdout_frac=HOLDOUT_FRAC, n_folds=args.folds,
                         n_repeats=args.repeats, seed=SEED,
                         **protocol_config(), k_max=k_max)
        groups = redundant_groups(ds_name, bench_names)
        if groups:
            picks = group_picks(out, groups, k_max)
            red_rows, red_macros = redundancy_summary(ds_name, picks, groups,
                                                      out, k_max)
            print_redundancy(ds_name, red_rows, groups, bench_names)
            width = max(len(idx) for _, idx in groups)
            save_dict["redundant_groups"] = np.array([g for g, _ in groups])
            save_dict["redundant_members"] = np.array(
                [list(idx) + [-1] * (width - len(idx)) for _, idx in groups],
                dtype=np.int64)
            for m in METHODS:
                save_dict[f"{m}_group_picks"] = picks[m]
            write_redundancy_json(
                redundancy_json_path(ds_name), ds_name,
                red_rows, red_macros, groups, bench_names, out, args.folds,
                args.repeats, k_max)
        np.savez(LOGS / f"baselines_{ds_name}.npz", **save_dict)
        write_paired_csv(LOGS / f"baselines_{ds_name}_paired.csv", paired)
        print(f"{ds_name.upper()} completed in {time.time() - ds_t0:.0f}s",
              flush=True)

    print(f"\nAll datasets completed in {time.time() - total_t0:.0f}s",
          flush=True)
