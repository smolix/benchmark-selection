"""Model-family holdout: OOD generalization of benchmark selection.

Instead of random CV folds, hold out an entire model family (llama,
qwen, claude, ... — see model_families.py), estimate the covariance
from the remaining models, select benchmarks, and impute the held-out
family's scores.  This tests whether selections remain informative for
model families never seen during covariance estimation.

Also reports selection stability: Jaccard overlap between the selection
computed on ALL models and the selection computed with each family
removed (Reviewer QSWq's "adding/removing model families" question).

Matched design (the family analogue of cv_split.cv_splits): the splits
are the held-out families, fold f = families[f] (every family with at
least --min-family models, "other" excepted).  Every method sees the
same train/test models, the same eligible set and the same common random
numbers.  Eligible benchmarks are those observed for at least
MIN_TRAIN_OBS training models (cv_split.eligible_mask, one mask per
split); the others stay in the covariance, in the MI / trace objectives,
in the hub scores and among the imputation targets, but cannot be
selected.  Imputation targets are the benchmarks observed for at least
MIN_TARGET_OBS training models (cv_split.target_mask, protocol rule D1;
the held-out family's cells of the others are never scored), and every
prediction is clipped to its benchmark's valid score range
(score_bounds.py, rule D2).  A family split has no random subsample, so
the --repeats repetitions only redraw the random baseline's permutation
(tag SEED + RANDOM_SEED_OFFSET), seeded by family_seed: (tag, f) in rep 0
— the seed of the single-repetition runs — and (tag, r, f) in rep r > 0.
The deterministic methods (entropy, mi, hub, pca, trace, facloc) are run
once per family; their split rows repeat in every repetition.

Methods: entropy (Alg. 1), mi (Alg. 2), random, hub, pca (CSSP), trace and
facloc, greedy facility location on the clipped correlation matrix
(baseline_select.facility_location_select), the paper's headline method
since 2026-10-19: deterministic, it draws no random numbers and comes
last, so adding it changed no other method's rows (entropy_rf was dropped
on 2026-10-18, protocol rule D3).  Near-ties of every selector are broken
by explained variance, then by the lowest index (greedy_select.TIE_RTOL).

Usage:
    python eval_family_holdout.py [--datasets mmlu mteb merged benchpress]
                                  [--kmax 15] [--repeats 5]
                                  [--min-family N] [--plot-entropy-rf]

--datasets takes any name with a {name}.matrix.npz in the data directory.
Families come from model_families.assign_families, whose scheme depends
on the name (model_families.family_scheme): the HF org prefix for MTEB
and the MTEB-derived mteb* (mteb2026), the developer of the base model
for epoch and helm_* (the organization of data/<name>.meta.json, Epoch's
"organization" and HELM's "creator_organization", with one label per
developer; fine-tunes such as Vicuna, Alpaca or Stable Beluga go with
their base model's developer, and models without an organization with
their keyword family's developer: model_families.meta_org_assignments),
base-architecture keywords for every other name (mmlu, merged,
benchpress, olv1items, olv2sub).  The families of the paper's datasets
are unchanged.  The minimum family size is
MIN_FAMILY for the paper's datasets and max(MIN_FAMILY_FLOOR,
ceil(M / MIN_FAMILY_DIV)) for others (olv2sub: 30, the other new
datasets: 6), unless --min-family is given.

Output logs/family_holdout_{ds}.npz.  Family rows (F = held-out families),
the layout of the single-repetition logs and of the paper's macros:
    families (F,), benchmark_names (N,)
    {m}_r2, {m}_rmse (F, k)        R² / RMSE on the held-out family; for
                                   random the mean over the
                                   repetitions (with --repeats 1 the rep-0
                                   draw).  Pair methods on these rows (one
                                   per family; the split rows with rep == 0
                                   give each family's n_train / n_val).
    stability_{m}_k{k} (F,)        Jaccard vs the all-model selection
                                   (deterministic m; NaN where estimation
                                   failed or fewer than k were eligible)
Split rows (S = repeats·F, rep-major; row = split_id = rep·F + fold), as in
the CV logs:
    rep, fold, split_id, n_train, n_val, n_eligible, estimation_failed (S,)
    eligible (S, N)                the selectable set of the split
    {m}_r2_splits, {m}_rmse_splits (S, k)
    {m}_selected (S, k)            picks in order, -1 beyond the last (PCA:
                                   its largest selection; pca_selected_per_k
                                   (S, k, k) holds every size, row kk-1 the
                                   size-kk set)
    {m}_index_ties (S, k)          picks of the size-k selection decided by
                                   the lowest index (index_tie_count)
    {m}_tie_breaks (S, k)          picks of the size-k selection made among
                                   near-tied scores
All-model reference: full_eligible (N,), full_{m}_selected (k,),
full_{m}_index_ties (k,), full_pca_selected_per_k (k, k).
model_family (M,): the family label of every model of the matrix (row
order of model_names; "other" is never held out).
Scalars: seed, n_repeats, min_train_obs, min_target_obs, pred_clip,
min_family, k_max.
"""

import argparse
import math
import pathlib
import sys
import time
import warnings
from collections import Counter
from typing import NamedTuple

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import (greedy_entropy, greedy_mi,
                           index_tie_count, random_select, selection_at_k,
                           tie_break_count)
from baseline_select import (facility_location_select, greedy_trace_select,
                             hub_select, pca_cssp_select)
from cv_split import (MIN_TRAIN_OBS, N_REPEATS, RANDOM_SEED_OFFSET,
                      eligible_mask, target_mask)
import model_families
from model_families import assign_families
from eval_entropy_vs_mi import (estimate_corr, impute_and_evaluate,
                                protocol_config)
from score_bounds import dataset_bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
SEED = 42
STABILITY_K = [5, 10, 15]

DATASETS = ["mmlu", "mteb", "merged", "benchpress"]

# Log order: the six methods of the single-repetition logs, then facloc
# (facility location, the paper's headline method since 2026-10-19)
METHODS = ["entropy", "mi", "random", "hub", "pca", "trace", "facloc"]
PLOT_METHODS = ["entropy", "mi", "random", "hub", "pca", "trace", "facloc"]
# Methods drawing on the split's common random numbers; the others are
# deterministic given the training models (the same in every repetition).
STOCHASTIC = ("random",)

# Minimum family size to be held out
MIN_FAMILY = {"mmlu": 40, "mteb": 7, "merged": 7, "benchpress": 6}
# Other datasets: at least MIN_FAMILY_FLOOR models and 1/MIN_FAMILY_DIV
# of all models (bounds the number of held-out families for large M).
MIN_FAMILY_FLOOR = 6
MIN_FAMILY_DIV = 150

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
}

METHOD_COLOR = {
    "entropy": "#4477AA",
    "mi": "#CC3311",
    "random": "#888888",
    "hub": "#228833",
    "pca": "#AA3377",
    "trace": "#EE7733",
    "facloc": "#009988",
}

METHOD_STYLE = {
    "entropy": "o-",
    "mi": "o-",
    "random": "x--",
    "hub": "s-",
    "pca": "d-",
    "trace": "^-",
    "facloc": "v-",
}
# Methods drawn with the thicker line and larger markers
MAIN_METHODS = ("entropy", "mi", "facloc")


def display_name(ds_name):
    """Display name of a dataset (the raw name if it has none)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def matrix_path(ds_name):
    return DATA / f"{ds_name}.matrix.npz"


def family_scheme(ds_name):
    """Family strategy of ds_name (model_families.family_scheme): "org"
    (the HF org prefix: mteb and the MTEB-derived mteb*), "meta_org"
    (the base model's developer, from data/<name>.meta.json: epoch,
    helm_*) or "keyword" (every other name)."""
    return model_families.family_scheme(ds_name)


def model_family_labels(ds_name, model_names):
    """(M,) family label of every model (model_families.assign_families
    with the data directory's metadata)."""
    return np.array(assign_families(model_names, ds_name, data_dir=DATA))


def min_family_size(ds_name, M, override=None):
    """Smallest family that is held out: override, else MIN_FAMILY, else
    max(MIN_FAMILY_FLOOR, ceil(M / MIN_FAMILY_DIV))."""
    if override is not None:
        return int(override)
    if ds_name in MIN_FAMILY:
        return MIN_FAMILY[ds_name]
    return max(MIN_FAMILY_FLOOR, math.ceil(M / MIN_FAMILY_DIV))


# ── Matched family splits ────────────────────────────────────────

def family_seed(tag, rep, fold):
    """Common-random-number seed of family split `fold` in repetition
    `rep` for the stream `tag`: (tag, fold) for rep 0 (the seed of the
    single-repetition runs), (tag, rep, fold) for rep > 0 — split_seed of
    cv_split.py without the holdout-fraction tag."""
    if rep == 0:
        return (int(tag), int(fold))
    return (int(tag), int(rep), int(fold))


class FamilySplit(NamedTuple):
    """One held-out-family split; fields as cv_split.Split, with the
    held-out family in place of the holdout fraction.  split_id = rep · F
    + fold, F = number of held-out families."""
    rep: int
    fold: int
    split_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    family: str

    def seed(self, tag):
        """Seed of this split for stream `tag` (family_seed); tag = SEED +
        RANDOM_SEED_OFFSET for the random baseline."""
        return family_seed(tag, self.rep, self.fold)


def family_splits(fams, holdout_fams, n_repeats=N_REPEATS):
    """Iterate over the n_repeats × F family splits, rep-major, families
    in the order of holdout_fams: fold f holds out every model with
    fams == holdout_fams[f] and trains on all others ("other" included).
    The models are the same in every repetition; only the seeds differ."""
    F = len(holdout_fams)
    for rep in range(n_repeats):
        for fold, fam in enumerate(holdout_fams):
            test_mask = fams == fam
            yield FamilySplit(rep, fold, rep * F + fold,
                              np.where(~test_mask)[0],
                              np.where(test_mask)[0], str(fam))


# ── Selection and evaluation ─────────────────────────────────────

def run_selection(method, R, k_max, benchmark_names, eligible=None,
                  split=None):
    """Selection of `method` restricted to `eligible`; the stochastic
    methods draw from the split's common random numbers (split.seed)."""
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
    if method == "facloc":
        return facility_location_select(R, k_max,
                                        benchmark_names=benchmark_names,
                                        eligible=eligible)
    raise ValueError(method)


def padded(sel, k_max):
    sel = list(sel)[:k_max]
    return sel + [-1] * (k_max - len(sel))


def evaluate_selection(result, k_max, B_test, O_test, R, col_mean, col_std,
                       target=None, bounds=None):
    """Per-k imputation R² / RMSE on the test models (NaN where fewer than
    k benchmarks were eligible; impute_and_evaluate with the split's D1
    targets and the dataset's D2 bounds) and the tie counts of each
    size-k set."""
    N = R.shape[0]
    out = {key: [] for key in ("r2", "rmse", "index_ties", "tie_breaks")}
    for k in range(1, k_max + 1):
        selected = selection_at_k(result, k)
        if selected is None:
            out["r2"].append(np.nan)
            out["rmse"].append(np.nan)
        else:
            imp = impute_and_evaluate(B_test, O_test, selected, R,
                                      col_mean, col_std, N, target=target,
                                      bounds=bounds)
            out["r2"].append(imp["r2"])
            out["rmse"].append(imp["rmse"])
        out["index_ties"].append(index_tie_count(result, k))
        out["tie_breaks"].append(tie_break_count(result, k))
    return out


def jaccard(a, b):
    if a is None or b is None:
        return np.nan
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def rep_mean(arr, n_repeats):
    """Mean over repetitions of split rows (n_repeats·F, ...) -> (F, ...),
    ignoring NaN (NaN where every repetition is NaN); exactly the rep-0
    rows for n_repeats = 1."""
    a = arr.reshape((n_repeats, arr.shape[0] // n_repeats) + arr.shape[1:])
    finite = ~np.isnan(a)
    total = np.where(finite, a, 0.0).sum(axis=0)
    count = finite.sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def run_dataset(ds_name, k_max, n_repeats=N_REPEATS, min_family=None):
    """All methods on every family split of one dataset.

    Returns a dict: metrics[m]["r2" | "rmse"] (F, k) family rows,
    stability[m][k] (F,), families, benchmark_names, k_max, and splits
    (the split-row and reference arrays of the log, see module doc).
    """
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()} — family holdout")
    print(f"{'#'*70}")

    d = np.load(matrix_path(ds_name), allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    model_names = d["model_names"]
    M, N = B.shape
    k_max = min(k_max, N)
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)     # D2

    fams = model_family_labels(ds_name, model_names)
    counts = Counter(fams)
    min_fam = min_family_size(ds_name, M, min_family)
    holdout_fams = sorted(
        f for f, c in counts.items()
        if f != "other" and c >= min_fam)
    F = len(holdout_fams)
    S = n_repeats * F
    print(f"{M} models × {N} tasks; {family_scheme(ds_name)} families, "
          f"{counts.get('other', 0)} models 'other' (always trained on)",
          flush=True)
    print(f"holding out families: "
          + ", ".join(f"{f}({counts[f]})" for f in holdout_fams), flush=True)
    print(f"Families of ≥ {min_fam} models; selectable: benchmarks with ≥ "
          f"{MIN_TRAIN_OBS} training models; {n_repeats} repetition(s) of "
          f"the random streams", flush=True)

    # Reference selections on ALL models (for stability analysis), with
    # the eligible set of all models
    R_full, _, _ = estimate_corr(B, O)
    full_eligible = eligible_mask(O)
    full_sel = {}
    for method in METHODS:
        if method in STOCHASTIC:
            continue
        full_sel[method] = run_selection(method, R_full, k_max,
                                         benchmark_names, full_eligible)

    # Split rows (row = split_id) and family rows of stability
    rows = {key: np.zeros(S, dtype=np.int64)
            for key in ("rep", "fold", "split_id", "n_train", "n_val",
                        "n_eligible")}
    rows["estimation_failed"] = np.zeros(S, dtype=bool)
    rows["eligible"] = np.zeros((S, N), dtype=bool)
    for m in METHODS:
        rows[f"{m}_r2_splits"] = np.full((S, k_max), np.nan)
        rows[f"{m}_rmse_splits"] = np.full((S, k_max), np.nan)
        rows[f"{m}_selected"] = np.full((S, k_max), -1, dtype=np.int64)
        rows[f"{m}_index_ties"] = np.zeros((S, k_max), dtype=np.int64)
        rows[f"{m}_tie_breaks"] = np.zeros((S, k_max), dtype=np.int64)
    rows["pca_selected_per_k"] = np.full((S, k_max, k_max), -1,
                                         dtype=np.int64)
    stability = {m: {k: np.full(F, np.nan) for k in STABILITY_K
                     if k <= k_max}
                 for m in METHODS if m not in STOCHASTIC}
    fits = {}                       # fold -> (R, col_mean, col_std) | None
    at5 = min(5, k_max) - 1

    for s in family_splits(fams, holdout_fams, n_repeats):
        t0 = time.time()
        i = s.split_id
        # One eligibility mask per split, shared by every method
        eligible = eligible_mask(O[s.train_idx])
        target = target_mask(O[s.train_idx])     # D1, shared by all
        n_eligible = int(eligible.sum())
        for key, val in [("rep", s.rep), ("fold", s.fold),
                         ("split_id", s.split_id),
                         ("n_train", len(s.train_idx)),
                         ("n_val", len(s.val_idx)),
                         ("n_eligible", n_eligible), ("eligible", eligible)]:
            rows[key][i] = val

        if s.rep == 0:
            print(f"  Holdout {s.family}: {len(s.train_idx)} train / "
                  f"{len(s.val_idx)} test, {n_eligible}/{N} eligible",
                  end="", flush=True)
            try:
                fits[s.fold] = estimate_corr(B[s.train_idx], O[s.train_idx])
            except Exception as e:
                print(f" — estimation failed: {e}", flush=True)
                fits[s.fold] = None
            methods = METHODS
        else:
            print(f"  Holdout {s.family} rep {s.rep}:", end="", flush=True)
            methods = STOCHASTIC
        if fits[s.fold] is None:
            rows["estimation_failed"][i] = True
            if s.rep > 0:
                print(" estimation failed in rep 0, skipped", flush=True)
            continue

        R, col_mean, col_std = fits[s.fold]
        B_test, O_test = B[s.val_idx], O[s.val_idx]
        for method in methods:
            result = run_selection(method, R, k_max, benchmark_names,
                                   eligible, s)
            out = evaluate_selection(result, k_max, B_test, O_test, R,
                                     col_mean, col_std, target, bounds)
            rows[f"{method}_r2_splits"][i] = out["r2"]
            rows[f"{method}_rmse_splits"][i] = out["rmse"]
            rows[f"{method}_index_ties"][i] = out["index_ties"]
            rows[f"{method}_tie_breaks"][i] = out["tie_breaks"]
            rows[f"{method}_selected"][i] = padded(result["selected"], k_max)
            if method == "pca":
                for kk, sel in enumerate(result["selected_per_k"]):
                    rows["pca_selected_per_k"][i, kk] = padded(sel, k_max)
            for k in stability.get(method, {}):
                stability[method][k][s.fold] = jaccard(
                    selection_at_k(result, k),
                    selection_at_k(full_sel[method], k))

        elapsed = time.time() - t0
        print("  " + "  ".join(
            f"{m}:{rows[f'{m}_r2_splits'][i][at5]:.3f}" for m in methods)
            + f"  ({elapsed:.1f}s)", flush=True)

    # Deterministic methods: their rep-0 rows hold in every repetition
    for m in METHODS:
        if m in STOCHASTIC:
            continue
        keys = [f"{m}_r2_splits", f"{m}_rmse_splits", f"{m}_selected",
                f"{m}_index_ties", f"{m}_tie_breaks"]
        keys += ["pca_selected_per_k"] if m == "pca" else []
        for key in keys:
            for rep in range(1, n_repeats):
                rows[key][rep * F:(rep + 1) * F] = rows[key][:F]

    # Family rows: the stochastic methods averaged over the repetitions
    metrics = {}
    for m in METHODS:
        metrics[m] = {}
        for key in ("r2", "rmse"):
            arr = rows[f"{m}_{key}_splits"]
            metrics[m][key] = (rep_mean(arr, n_repeats) if m in STOCHASTIC
                               else arr[:F].copy())

    rows["full_eligible"] = full_eligible
    for m, res in full_sel.items():
        rows[f"full_{m}_selected"] = np.array(padded(res["selected"], k_max),
                                              dtype=np.int64)
        rows[f"full_{m}_index_ties"] = np.array(
            [index_tie_count(res, k) for k in range(1, k_max + 1)],
            dtype=np.int64)
    rows["full_pca_selected_per_k"] = np.array(
        [padded(sel, k_max) for sel in full_sel["pca"]["selected_per_k"]]
        + [[-1] * k_max] * (k_max - len(full_sel["pca"]["selected_per_k"])),
        dtype=np.int64).reshape(k_max, k_max)
    rows.update(seed=np.int64(SEED), n_repeats=np.int64(n_repeats),
                **{key: (np.bool_(val) if isinstance(val, bool) else
                         np.int64(val))
                   for key, val in protocol_config().items()},
                min_family=np.int64(min_fam), k_max=np.int64(k_max))
    rows["model_family"] = np.array([str(f) for f in fams])

    return {"metrics": metrics, "stability": stability,
            "families": list(holdout_fams),
            "benchmark_names": benchmark_names, "k_max": k_max,
            "splits": rows}


def print_summary(ds_name, metrics, stability, family_list, k_max,
                  n_repeats=1):
    ks_report = [k for k in [1, 3, 5, 10, 15] if k <= k_max]
    print(f"\n== {display_name(ds_name)}: family-holdout "
          f"imputation R² (mean ± std over {len(family_list)} families) ==")
    if n_repeats > 1:
        print("   (" + ", ".join(STOCHASTIC) + ": per family, mean over "
              f"{n_repeats} draws)")
    header = f"{'method':<22}" + "".join(f"k={k:<11}" for k in ks_report)
    print(header)
    with warnings.catch_warnings():
        # All-NaN columns (fewer than k eligible benchmarks) print as nan
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for m in METHODS:
            r2 = metrics[m]["r2"]
            valid = ~np.all(np.isnan(r2), axis=1)
            if not valid.any():
                continue
            cells = []
            for k in ks_report:
                mu = np.nanmean(r2[valid, k - 1])
                sd = np.nanstd(r2[valid, k - 1])
                cells.append(f"{mu:.3f}±{sd:.3f}")
            print(f"{METHOD_LABEL[m]:<22}"
                  + "".join(f"{c:<13}" for c in cells))

    print(f"\n   Per-family R² at k=5:")
    at5 = min(5, k_max) - 1
    print(f"   {'family':<16}" + "".join(f"{m:<12}" for m in METHODS))
    for i, fam in enumerate(family_list):
        cells = []
        for m in METHODS:
            v = metrics[m]["r2"][i][at5] if i < len(metrics[m]["r2"]) else np.nan
            cells.append(f"{v:.3f}" if np.isfinite(v) else "nan")
        print(f"   {fam:<16}" + "".join(f"{c:<12}" for c in cells))

    print(f"\n   Selection stability (Jaccard vs full-data selection):")
    for m in stability:
        cells = []
        for k in sorted(stability[m]):
            vals = stability[m][k]
            if np.any(np.isfinite(vals)):
                cells.append(f"k={k}: {np.nanmean(vals):.2f}")
        print(f"   {METHOD_LABEL[m]:<22}" + "  ".join(cells))


def print_matched_summary(res):
    """Eligible sets and near-tie counts of the matched splits."""
    sp = res["splits"]
    fams, k_max = res["families"], res["k_max"]
    N = len(res["benchmark_names"])
    F = len(fams)
    n_el = sp["n_eligible"][:F]
    restricted = [f"{f} {n}" for f, n in zip(fams, n_el) if n < N]
    print(f"\n   Eligible benchmarks (≥ {MIN_TRAIN_OBS} training models): "
          f"all models {int(sp['full_eligible'].sum())}/{N}; families "
          + (", ".join(restricted) + f" of {N}, others all"
             if restricted else f"all {N}"))
    S = len(sp["split_id"])
    for m in METHODS:
        ties = sp[f"{m}_index_ties"][:, -1]
        tb = sp[f"{m}_tie_breaks"][:, -1]
        print(f"   {m:<11} k ≤ {k_max}: index ties {int(ties.sum())} picks "
              f"in {int(np.count_nonzero(ties))}/{S} splits, near-tie "
              f"breaks {int(tb.sum())} in {int(np.count_nonzero(tb))}/{S}; "
              f"all-model reference: "
              + (f"{int(sp[f'full_{m}_index_ties'][-1])} index ties"
                 if f"full_{m}_index_ties" in sp else "n/a"))


def plot_comparison(ds_name, metrics, k_max, methods=PLOT_METHODS):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ks = np.arange(1, k_max + 1)

    with warnings.catch_warnings():
        # All-NaN columns (fewer than k eligible benchmarks) plot as gaps
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for method in methods:
            c = METHOD_COLOR[method]
            lw = 1.5 if method in MAIN_METHODS else 1.0
            ms = 4 if method in MAIN_METHODS else 3
            r2 = metrics[method]["r2"]
            valid = ~np.all(np.isnan(r2), axis=1)
            if not valid.any():
                continue
            mean_r2 = np.nanmean(r2[valid], axis=0)
            std_r2 = np.nanstd(r2[valid], axis=0)
            ax.plot(ks, mean_r2, METHOD_STYLE[method], color=c,
                    label=METHOD_LABEL[method], markersize=ms, linewidth=lw)
            ax.fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                            color=c, alpha=0.10)

    ax.set_xlabel("Selected benchmarks $k$")
    ax.set_ylabel("Imputation $R^2$ (held-out family)")
    ax.set_xlim(0.5, k_max + 0.5)
    ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= k_max])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    out_path = OUT / f"family_holdout_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {out_path}", flush=True)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model-family holdout of every selection method on the "
                    "matched family splits.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="dataset names, each read from "
                             "{name}.matrix.npz in the data directory "
                             f"(default: {' '.join(DATASETS)})")
    parser.add_argument("--kmax", type=int, default=K_MAX)
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="repetitions of the random baseline's stream "
                             "(default %(default)s; 1 = rep 0 only, the "
                             "single-repetition seeds)")
    parser.add_argument("--min-family", type=int, default=None,
                        help="smallest family held out (default: "
                             "MIN_FAMILY for the paper's datasets, else "
                             f"max({MIN_FAMILY_FLOOR}, "
                             f"ceil(M / {MIN_FAMILY_DIV})))")
    parser.add_argument("--plot-entropy-rf", action="store_true",
                        help="ignored: entropy_rf was dropped (protocol "
                             "rule D3); kept so old command lines run")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.kmax < 1:
        parser.error("--kmax must be at least 1")
    if args.min_family is not None and args.min_family < 1:
        parser.error("--min-family must be at least 1")
    missing = [ds for ds in args.datasets if not matrix_path(ds).is_file()]
    if missing:
        parser.error("no data matrix for " + ", ".join(
            f"{ds} ({matrix_path(ds)})" for ds in missing))
    if args.plot_entropy_rf:
        print("note: --plot-entropy-rf is ignored (entropy_rf was dropped, "
              "protocol rule D3)", flush=True)
    plot_methods = PLOT_METHODS

    total_t0 = time.time()
    for ds_name in args.datasets:
        ds_t0 = time.time()
        res = run_dataset(ds_name, args.kmax, args.repeats, args.min_family)
        metrics, stability = res["metrics"], res["stability"]
        family_list, k_max = res["families"], res["k_max"]
        print_summary(ds_name, metrics, stability, family_list, k_max,
                      args.repeats)
        print_matched_summary(res)
        plot_comparison(ds_name, metrics, k_max, plot_methods)

        save_dict = {"benchmark_names": res["benchmark_names"],
                     "families": np.array(family_list)}
        for m in METHODS:
            for key, arr in metrics[m].items():
                save_dict[f"{m}_{key}"] = arr
        for m in stability:
            for k, vals in stability[m].items():
                save_dict[f"stability_{m}_k{k}"] = np.array(vals)
        save_dict.update(res["splits"])
        np.savez(LOGS / f"family_holdout_{ds_name}.npz", **save_dict)
        print(f"{ds_name.upper()} completed in {time.time() - ds_t0:.0f}s",
              flush=True)

    print(f"\nAll datasets completed in {time.time() - total_t0:.0f}s",
          flush=True)
