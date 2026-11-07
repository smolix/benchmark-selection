"""Evaluate benchmark selection with TabImpute V2 as the imputation backend.

Same matched-split protocol as eval_entropy_vs_mi.py and eval_baselines.py
(cv_split.cv_splits, 10-fold CV, 10% holdout, the same correlation
estimate and selectors), but the unselected benchmarks of each validation
model are imputed by TabImpute V2 instead of the Gaussian conditional mean.
R² is computed in standardized space (training z-scores), as there.

Matched design.  The splits are cv_splits(M, folds, repeats, 0.1, SEED).
GPU time is scarce, so --repeats defaults to TABIMPUTE_REPEATS = 1:
repetition 0 only, i.e. the original seed-42 folds and training
subsamples, which are exactly rep 0 (split_id 0..9) of the CPU runs
(eval_entropy_vs_mi.py, eval_baselines.py, ...); --repeats 5 (N_REPEATS)
is the full 5 x 10 design.  In every split all methods share

  * the train / validation indices and the estimated correlation matrix,
  * one eligibility mask, eligible_mask(O_train): only benchmarks observed
    for at least MIN_TRAIN_OBS training models can be selected.
    Ineligible benchmarks stay in the correlation matrix (MI complement,
    trace objective, hub scores, PCA loadings), in TabImpute's context and
    among the imputation targets.  Where fewer than k benchmarks are
    eligible, R², RMSE are NaN at that k (and TabImpute is not called);
  * common random numbers: the random baseline's permutation is seeded
    with Split.seed(SEED + RANDOM_SEED_OFFSET) (rep 0: (49, fold, 1000), as
    before).  (entropy_rf was dropped on 2026-10-18, protocol rule D3.)

Near-ties of every selector's score are broken by explained variance, then
by the lowest index (greedy_select / baseline_select); {m}_index_ties
counts the picks the index decided.  Within a split, TabImpute's input
depends only on the selected set, so a set that two methods (or two k)
share is imputed once and reused (identical R², a paired tie).

Scoring (aligned with eval_entropy_vs_mi.impute_and_evaluate on
2026-10-18, protocol_changes.md H8, so that TabImpute and the Gaussian
conditional mean are scored on the same cells and paired comparisons
between the two pipelines are like for like):
  * cells: every observed unselected target benchmark of every validation
    model that observes at least one selected benchmark (scored_cells);
    targets are the benchmarks observed for at least MIN_TARGET_OBS = 2
    training models (cv_split.target_mask, protocol rule D1).  A
    non-target column stays in TabImpute's context, its cells are not
    scored.  A model
    without an observed selected benchmark is still part of TabImpute's
    input (an all-missing row, as before), but it is not scored; until
    2026-10-18 it was scored with TabImpute's context-only prediction.
  * R² = 1 − SS_res / max(SS_tot, SS_TOT_FLOOR), NaN only if no cell is
    scored (until 2026-10-18: NaN also if SS_tot < 1e-12).  SS_tot is
    about the training mean in training-standardized units, truths
    clipped to ±Z_CLIP, as there.
  * TabImpute's predictions are clipped to the benchmark's valid score
    range (protocol rule D2, score_bounds.py: the same rule as for the
    Gaussian predictions of impute_and_evaluate), in train-standardized
    units; {m}_n_clipped counts the scored predictions the clip changed,
    per split and k.  Until 2026-10-18 the standardized predictions were
    clipped to +-10 instead (and the Gaussian ones not at all).
  * context: TabImpute sees all training rows if they fit next to a test
    batch (MAX_ROWS − TEST_BATCH = 3,800 rows), else 3,800 training rows
    drawn without replacement by default_rng(Split.seed(SEED +
    CONTEXT_SEED_OFFSET)) (context_index): one draw per split, shared by
    every method and k (common random numbers; the tag is registered in
    cv_split.SEED_TAGS).  Until 2026-10-18 every split drew with the fixed
    default_rng(42).  Only MMLU (n_train ≈ 4,906) exceeds the limit among
    the paper datasets (OLL v2 sub-tasks, n_train ≈ 4,044, would as well).

Methods: entropy, mi, random, facloc (the default set: the original three,
then since 2026-10-19 facloc, greedy facility location on the correlation
matrix clipped at 0, baseline_select.facility_location_select, the
paper's headline method) and hub, pca, trace (baseline_select.py; trace
is greedy trace minimization), which run_tabimpute.sh runs as its extra
job (BENCHSELECT_METHODS=hub,pca,trace); entropy_rf was dropped (protocol
rule D3) and is rejected.  BENCHSELECT_METHODS (or --methods) evaluates
another set; any such override saves under the "extra_" tag, so the
default tabimpute_{ds} files are kept (make_numbers and paired_stats read
the methods of both files as one table, paired on split_id: facloc's
pairs with hub, pca and trace come from there).  The two jobs keep the
method sets they had before facloc, facloc joining the default one, so
every array of their earlier logs is unchanged.  No method draws from a
shared generator (random has its own Split.seed stream, trace and facloc
draw none) and TabImpute's input depends only on the split and the
selected set, so every per-method array is the same whichever other
methods share the run.

Combining runs (--merge-from).  A log of some methods can therefore be
assembled from earlier runs' logs instead of being recomputed:
--merge-from PATH ... (no TabImpute, no GPU) takes each method of the
run's method set (--methods / the default set, as for a run) from the
first PATH whose log has it ('{ds}' in a PATH is the dataset), checks
that all PATHs agree on the splits, the per-split context and
eligibility, the scalars and the checkpoint / torch / tabimpute versions
(the device may differ: the merged log records the devices joined by
'+'), and writes the npz, paired CSV and figure exactly as one run of
those methods would.  E.g. a root OLD with the logs of the default
(entropy, mi, random) and extra (hub, pca, trace) runs from before facloc
needs only facloc on the GPU:

    BENCHSELECT_EXPERIMENT_ROOT=NEW python eval_tabimpute.py \
        --datasets mmlu --methods facloc   # NEW/logs/tabimpute_extra_mmlu.npz
    BENCHSELECT_EXPERIMENT_ROOT=OUT python eval_tabimpute.py --datasets mmlu \
        --merge-from OLD/logs/tabimpute_{ds}.npz \
                     NEW/logs/tabimpute_extra_{ds}.npz      # default set
    BENCHSELECT_EXPERIMENT_ROOT=OUT python eval_tabimpute.py --datasets mmlu \
        --methods hub pca trace \
        --merge-from OLD/logs/tabimpute_extra_{ds}.npz      # extra_ set

(use the --repeats / --folds / --kmax of those runs; OUT then holds the
logs a run of the current default and extra sets would write; the extra_
log is OLD's).  Logs of a run with the interim sets of 2026-10-19
(default entropy, mi, random, trace, facloc; extra hub, pca) convert the
same way: --merge-from RUN/logs/tabimpute_{ds}.npz for the default set,
--methods hub pca trace --merge-from RUN/logs/tabimpute_extra_{ds}.npz
RUN/logs/tabimpute_{ds}.npz for the extra one.  bf16 results can differ
across GPU models, CUDA and torch versions: combine runs of one
environment only (the check covers the versions, not the GPU model).

TabImpute weights and determinism.  tabimputev2.ckpt is downloaded from
the Hugging Face Hub repo CKPT_REPO at the pinned commit CKPT_REVISION
(hf_hub_download(revision=...), cached; --revision, or --checkpoint for a
local file), and its sha256 must equal CKPT_SHA256 (override:
BENCHSELECT_TABIMPUTE_CKPT_SHA256 or --ckpt-sha256; empty = record only).
The file is the same LFS object (sha256 b48b5bfd...) as at 96d7a0f1, the
commit that last changed it and the cached revision of the earlier,
unpinned runs.  Inference draws no random numbers (eval mode, dropout 0,
bar-distribution median); torch deterministic algorithms are requested
(warn_only: an op without a deterministic CUDA kernel warns instead of
failing), cuDNN deterministic with benchmarking off, and
CUBLAS_WORKSPACE_CONFIG=:4096:8 unless set.  bf16 results can still differ
across GPU models, CUDA and torch versions.

Usage:
    python eval_tabimpute.py [--datasets mmlu mteb merged benchpress ...]
                             [--methods entropy mi random facloc]
                             [--repeats 1] [--folds 10] [--kmax 15]
                             [--device cuda:0] [--revision SHA]
                             [--checkpoint PATH] [--ckpt-sha256 HEX]
                             [--merge-from LOG ...]

--datasets accepts any name with a {name}.matrix.npz in the data directory
(path_config.data_dir()).  Environment (the defaults of the flags, as set by
run_tabimpute.sh): BENCHSELECT_DATASETS and BENCHSELECT_METHODS (comma
separated), BENCHSELECT_TABIMPUTE_DEVICE (default: cuda if available, else
cpu), BENCHSELECT_TABIMPUTE_CKPT_SHA256.

Outputs, per dataset ({tag} = "" for the default methods, else "extra_"):
  logs/tabimpute_{tag}{ds}.npz
    {m}_r2, {m}_rmse        (n_splits, kmax) imputation R² / RMSE on the
                            validation models given the size-k selection
    {m}_n_eval              (n_splits, kmax) int, entries scored
    {m}_n_clipped           (n_splits, kmax) int, scored predictions
                            the valid-range clip (rule D2) changed
    n_context               (n_splits,) int, TabImpute context rows
    context_subsampled      (n_splits,) bool, context drawn by
                            context_index (n_train above the limit)
    {m}_selected            (n_splits, kmax) int, picks in order, -1 beyond
                            the last (pca: its largest set)
    {m}_index_ties          (n_splits, kmax) int, picks of the size-k
                            selection decided by the lowest index
                            (greedy_select.index_tie_count)
    {m}_tie_breaks          (n_splits, kmax) int, picks of the size-k
                            selection made among near-tied scores
    pca_selected_per_k      (n_splits, kmax, kmax) int, row kk-1 = pca's
                            set of size kk, -1 padded (if pca ran)
    methods                 (n_methods,) method names
    rep, fold, split_id     (n_splits,) int; pair methods (and scripts) on
                            split_id = rep * folds + fold
    n_train, n_val, n_eligible   (n_splits,) int
    eligible                (n_splits, N) bool, the selectable set
    estimation_failed       (n_splits,) bool (rows all NaN)
    holdout_frac, n_folds, n_repeats, seed, min_train_obs, min_target_obs,
    k_max, context_seed_offset, max_context, pred_clip (bool: rule D2
    on), z_clip, ss_tot_floor
                            scalars
    ckpt_repo, ckpt_file, ckpt_revision, ckpt_sha256, device,
    torch_version, tabimpute_version, torch_deterministic   provenance
                            (device of a --merge-from log: its sources'
                            devices joined by '+')
    benchmark_names         (N,)
  logs/tabimpute_{tag}{ds}_paired.csv
    paired R² differences method_a − method_b for every method pair and k
    over the matched splits (paired_stats.paired_stats: mean, Nadeau–Bengio
    corrected standard error, 95% t-interval, win rate; pairs oriented by
    paired_stats.orient, e.g. mi − entropy; columns dataset, holdout_frac,
    k, method_a, method_b, then paired_stats.STAT_FIELDS)
  figures/tabimpute_r2_{tag}{ds}.pdf   R² mean ± std over splits
"""

import argparse
import hashlib
import os
import pathlib
import sys
import time
import warnings
import numpy as np

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
from em_cov import em_cov
import cv_split
import score_bounds
from cv_split import (MIN_TRAIN_OBS, RANDOM_SEED_OFFSET, cv_splits,
                      eligible_mask, target_mask)
from score_bounds import clip_z, dataset_bounds, z_limits
from paired_stats import STAT_FIELDS, orient, paired_stats, write_csv

# torch and tabimpute are imported by load_imputer(), after the
# determinism settings, so the selection code imports without them.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42
# CV repetitions by default: rep 0 only (GPU time); N_REPEATS for the full
# matched design.
TABIMPUTE_REPEATS = 1
MAX_ROWS = 4000    # max total rows per TabImpute call (GPU memory limit)
TEST_BATCH = 200   # test rows per batch; context rows = MAX_ROWS - TEST_BATCH
MAX_CONTEXT = MAX_ROWS - TEST_BATCH
# Common random numbers of the context subsample (context_index):
# Split.seed(SEED + CONTEXT_SEED_OFFSET).  Tag 20 was reserved for it
# (cv_split.SEED_TAGS); distinct from RANDOM_SEED_OFFSET = 7,
# ENTROPY_RF_SEED_OFFSET = 11 and every other tag in code/.
CONTEXT_SEED_OFFSET = 20
# Scoring constants of eval_entropy_vs_mi.impute_and_evaluate (module
# docstring, "Scoring"); the predictions are clipped to the valid score
# range (score_bounds, rule D2), which replaced the +-10 prediction clip
Z_CLIP = 10.0
SS_TOT_FLOOR = 1e-12

# Pinned TabImpute V2 weights: the repo's commit at the time of pinning
# (https://huggingface.co/api/models/Tabimpute/TabImpute, 2026-10-17) and
# the checkpoint's sha256 (its LFS oid, unchanged since 96d7a0f1).
CKPT_REPO = "Tabimpute/TabImpute"
CKPT_FILE = "tabimputev2.ckpt"
CKPT_REVISION = "727574989ddebe71860141c9a8b9f6e7958e3cce"
CKPT_SHA256 = "b48b5bfdaf496bb949b762377518cd4ddeea7629e44f02aebb9a18ca9e711bce"

DEFAULT_DATASETS = ["mmlu", "mteb", "merged", "benchpress"]

# Selection methods: the default set (the original three, whose key order
# is kept, then facility location, 2026-10-19) or any of ALL_METHODS (the
# extra job of run_tabimpute.sh: hub, pca, trace).  An override (via
# BENCHSELECT_METHODS or --methods) saves under the "extra_" tag so the
# default tabimpute_{ds}.npz files are preserved.
DEFAULT_METHODS = ["entropy", "mi", "random", "facloc"]
ALL_METHODS = ["entropy", "mi", "random", "hub", "pca", "trace", "facloc"]
EXTRA_TAG = "extra_"

# Log keys (save_results) that --merge-from takes per method, checks for
# equality across the merged logs, or reads as provenance (the device may
# differ between them)
METRIC_KEYS = ("r2", "rmse", "n_eval", "n_clipped")
METHOD_OUT_KEYS = ("selected", "index_ties", "tie_breaks")
SPLIT_KEYS = ("rep", "fold", "split_id", "n_train", "n_val", "n_eligible",
              "eligible", "estimation_failed", "n_context",
              "context_subsampled")
SCALAR_KEYS = ("holdout_frac", "n_folds", "n_repeats", "seed",
               "min_train_obs", "min_target_obs", "k_max",
               "context_seed_offset", "max_context", "pred_clip", "z_clip",
               "ss_tot_floor")
PROVENANCE_KEYS = ("ckpt_repo", "ckpt_file", "ckpt_revision", "ckpt_sha256",
                   "device", "torch_version", "tabimpute_version",
                   "torch_deterministic")

PAIRED_FIELDS = (["dataset", "holdout_frac", "k", "method_a", "method_b"]
                 + STAT_FIELDS)

DISPLAY_NAME = {
    "mmlu": "MMLU", "mteb": "MTEB",
    "merged": "Merged", "benchpress": "BenchPress",
}

METHOD_COLOR = {"entropy": "#4477AA", "mi": "#CC3311", "random": "#888888",
                "hub": "#228833", "pca": "#AA3377", "trace": "#EE7733",
                "facloc": "#009988"}
METHOD_LABEL = {"entropy": "Entropy (Alg. 1)", "mi": "Mutual Info (Alg. 2)",
                "random": "Random", "hub": "Hub correlation",
                "pca": "PCA subset (CSSP)", "trace": "Greedy trace min.",
                "facloc": "Facility location"}
METHOD_STYLE = {"entropy": "o-", "mi": "o-", "random": "x--",
                "hub": "s-", "pca": "d-", "trace": "^-", "facloc": "v-"}


def display_name(ds_name):
    """Display name of a dataset (the raw name if it has none)."""
    return DISPLAY_NAME.get(ds_name, ds_name)


def matrix_path(ds_name):
    return DATA / f"{ds_name}.matrix.npz"


def run_selection(method, R, k_max, benchmark_names, split, eligible):
    """Run `method` on split `split` with the split's eligible mask and
    common random numbers (identical for every script on that split)."""
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


def _padded(indices, length):
    """Integer array of `indices`, padded with -1 to `length`."""
    out = np.full(length, -1, dtype=np.int64)
    out[:len(indices)] = indices
    return out


def _nan_mean_std(a, axis=0):
    """np.nanmean / np.nanstd (ddof=0) without all-NaN-slice warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)

# ── Covariance estimation (same as other scripts) ─────────────────

def estimate_corr_pairwise(B_train, O_train):
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]
    gram = B_std.T @ B_std
    pair_counts = O_train.T @ O_train
    denom = np.maximum(pair_counts - 1.0, 1.0)
    R = gram / denom
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = (eigvecs * eigvals[None, :]) @ eigvecs.T
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(R), 1e-12))
    R = R * np.outer(d_inv, d_inv)
    return R, col_mean, col_std


def estimate_corr_em(B_train, O_train):
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std_floor = 0.01 * (np.abs(col_mean) + 1.0)
    col_std = np.sqrt(np.maximum(col_var, col_std_floor ** 2))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]
    regularized = M_tr < N_tr or obs_frac < 0.5
    use_shrink = "auto" if M_tr < N_tr else 0.0
    em_eps = 1e-3 if regularized else 1e-6
    em_iter = 1000 if regularized else 500
    result = em_cov(B_std, O_train, max_iter=em_iter, tol=5e-4,
                    shrinkage=use_shrink, eps_psd=em_eps, verbose=False)
    Sigma = result["Sigma"]
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    return R, col_mean, col_std


def estimate_corr(B_train, O_train):
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)
    if obs_frac < 0.9 or M_tr < N_tr:
        return estimate_corr_em(B_train, O_train)
    else:
        return estimate_corr_pairwise(B_train, O_train)


# ── TabImpute model (pinned weights, deterministic settings) ─────

def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_checkpoint(revision=CKPT_REVISION, checkpoint=None,
                       expected_sha256=CKPT_SHA256):
    """(path, sha256) of the TabImpute V2 checkpoint.

    `checkpoint` (a local file) if given, else CKPT_FILE of CKPT_REPO at
    commit `revision` via hf_hub_download (from the local cache when that
    snapshot is there).  Exits unless the file's sha256 equals
    expected_sha256 (no check if it is empty).
    """
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILE,
                               revision=revision)
    else:
        path = checkpoint
    digest = _sha256_file(path)
    if expected_sha256 and digest != expected_sha256.lower():
        raise SystemExit(f"eval_tabimpute.py: {path}: sha256 {digest} is "
                         f"not the pinned {expected_sha256.lower()}")
    return str(path), digest


def set_deterministic(torch):
    """Deterministic torch settings for inference: deterministic
    algorithms (warn_only, so an op without a deterministic CUDA kernel
    warns instead of failing), cuDNN deterministic without autotuning, and
    a fixed global seed (inference draws no random numbers)."""
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_imputer(device=None, revision=CKPT_REVISION, checkpoint=None,
                 expected_sha256=CKPT_SHA256):
    """TabImputeV2 with the pinned checkpoint, and its provenance dict
    (the ckpt_* / device / version entries saved with the results)."""
    # Read by cuBLAS when its first handle is created (deterministic GEMM
    # workspace); run_tabimpute.sh sets the same default.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from importlib import metadata
    from tabimpute.tabimpute_v2 import TabImputeV2

    set_deterministic(torch)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path, digest = resolve_checkpoint(revision, checkpoint, expected_sha256)
    imputer = TabImputeV2(device=device, checkpoint_path=path)
    try:
        tabimpute_version = metadata.version("tabimpute")
    except metadata.PackageNotFoundError:
        tabimpute_version = ""
    info = {
        "ckpt_repo": CKPT_REPO,
        "ckpt_file": CKPT_FILE,
        "ckpt_revision": revision if checkpoint is None else "",
        "ckpt_sha256": digest,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "tabimpute_version": tabimpute_version,
        "torch_deterministic": bool(
            torch.are_deterministic_algorithms_enabled()),
    }
    print(f"Checkpoint {path}\n  {CKPT_REPO}/{CKPT_FILE} revision "
          f"{info['ckpt_revision'] or '(local file)'}, sha256 {digest}")
    print(f"torch {info['torch_version']}, tabimpute {tabimpute_version}, "
          f"deterministic algorithms {info['torch_deterministic']}, "
          f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}")
    return imputer, info


# ── TabImpute-based imputation ────────────────────────────────────

def context_index(M_train, split, max_context=None):
    """Training rows of TabImpute's context on split `split`: None (all
    M_train rows) if they fit, else max_context (default MAX_CONTEXT) row
    indices drawn without replacement by default_rng(split.seed(SEED +
    CONTEXT_SEED_OFFSET)).  One draw per split, shared by every method and
    k (common random numbers); it depends on the split only, not on the
    selection."""
    if max_context is None:
        max_context = MAX_CONTEXT
    if M_train <= max_context:
        return None
    rng = np.random.default_rng(split.seed(SEED + CONTEXT_SEED_OFFSET))
    return rng.choice(M_train, max_context, replace=False)


def scored_cells(O_test, selected, N, target=None):
    """(M_test, N) bool, the cells impute_and_evaluate scores for the
    selection `selected`: the observed unselected target benchmarks of
    every validation model that observes at least one selected benchmark
    (and at least one unselected target).  target: the split's D1 mask
    (cv_split.target_mask; None: every column)."""
    obs = np.asarray(O_test) == 1
    in_sel = np.zeros(N, dtype=bool)
    in_sel[np.asarray(list(selected), dtype=np.int64)] = True
    targets = obs & ~in_sel[None, :]
    if target is not None:
        targets &= np.asarray(target, dtype=bool)[None, :]
    has_sel = obs[:, in_sel].any(axis=1)
    return targets & has_sel[:, None]


def impute_tabimpute(imputer, B_train, O_train, B_test, O_test,
                     selected, col_mean, col_std, N, ctx_idx=None,
                     target=None, bounds=None):
    """Impute test models' unselected benchmarks using TabImpute V2.

    Constructs a matrix with training rows (context) and test rows
    (only selected benchmarks observed), runs TabImpute and evaluates the
    pooled R² in standardized (z-scored) space on the cells and with the
    rules of eval_entropy_vs_mi.impute_and_evaluate (module docstring,
    "Scoring"): cells scored_cells(O_test, selected, N, target) (target:
    the split's D1 mask), truths clipped to ±Z_CLIP, R² = 1 − SS_res /
    max(SS_tot, SS_TOT_FLOOR), NaN (and no TabImpute call) if no cell is
    scored.  Predictions are clipped to the valid score range `bounds`
    (score_bounds.dataset_bounds, rule D2; None: no clip) in standardized
    units; n_clipped counts the scored ones the clip changed.

    The context is every training row, or the rows ctx_idx
    (context_index(M_train, split)) if they do not fit next to a test
    batch; then ctx_idx is required.  Test rows are batched so that each
    batch sees the whole context.  Returns {"r2", "rmse", "n_eval",
    "n_clipped"}.
    """
    M_train = B_train.shape[0]
    M_test = B_test.shape[0]
    scored = scored_cells(O_test, selected, N, target)
    n_eval = int(scored.sum())
    z_lo, z_hi = z_limits(bounds, col_mean, col_std)
    if n_eval == 0:
        return {"r2": np.nan, "rmse": np.nan, "n_eval": 0, "n_clipped": 0}

    # Context rows: all training data with observed entries
    X_ctx = np.where(O_train == 1, B_train, np.nan)
    if ctx_idx is not None:
        X_ctx = X_ctx[np.asarray(ctx_idx, dtype=np.int64)]
    elif M_train > MAX_CONTEXT:
        raise ValueError(f"{M_train} training rows exceed the context limit "
                         f"{MAX_CONTEXT}: pass ctx_idx = context_index(...)")
    n_ctx = X_ctx.shape[0]
    if n_ctx > MAX_CONTEXT:
        raise ValueError(f"context of {n_ctx} rows exceeds {MAX_CONTEXT}")

    # Test rows: only reveal selected benchmarks
    X_test = np.full((M_test, N), np.nan)
    for j in selected:
        for i in range(M_test):
            if O_test[i, j] == 1:
                X_test[i, j] = B_test[i, j]

    # Process test rows in batches, each seeing all context rows
    max_test_per_batch = max(1, min(TEST_BATCH, MAX_ROWS - n_ctx))
    X_test_imputed = np.empty((M_test, N), dtype=np.float64)
    for start in range(0, M_test, max_test_per_batch):
        end = min(start + max_test_per_batch, M_test)
        X_batch = np.vstack([X_ctx, X_test[start:end]])
        X_batch_imp = imputer.impute(X_batch)
        X_test_imputed[start:end] = X_batch_imp[n_ctx:]

    # Evaluate in standardized space (as impute_and_evaluate): z-scores
    # in training units, pooled over the scored cells of every model
    ss_res = 0.0
    ss_tot = 0.0
    n_clipped = 0
    for i in range(M_test):
        eval_idx = np.flatnonzero(scored[i])
        if len(eval_idx) == 0:
            continue

        true_std = (B_test[i, eval_idx] - col_mean[eval_idx]) / col_std[eval_idx]
        pred_std = (X_test_imputed[i, eval_idx] - col_mean[eval_idx]) / col_std[eval_idx]
        true_std = np.clip(true_std, -Z_CLIP, Z_CLIP)
        pred_std, n_c = clip_z(pred_std, z_lo[eval_idx], z_hi[eval_idx])
        n_clipped += n_c

        ss_res += np.sum((true_std - pred_std) ** 2)
        ss_tot += np.sum(true_std ** 2)  # mean is 0 in standardized space

    r2 = 1.0 - ss_res / max(ss_tot, SS_TOT_FLOOR)
    rmse = np.sqrt(ss_res / n_eval)
    return {"r2": r2, "rmse": rmse, "n_eval": n_eval,
            "n_clipped": n_clipped}


# ── Run one dataset ───────────────────────────────────────────────

def run_dataset(ds_name, imputer, methods=DEFAULT_METHODS,
                n_repeats=TABIMPUTE_REPEATS, n_folds=N_FOLDS, k_max=K_MAX):
    """Every method on every matched split of one dataset.

    Returns (metrics, out, benchmark_names): metrics[method][key] is an
    (n_splits, k_max) array (key in r2, rmse, n_eval, n_clipped); out
    holds the per-split selections, tie counts, context sizes and split
    metadata (the npz keys other than {m}_r2 / _rmse / _n_eval /
    _n_clipped), rows in cv_splits order.
    """
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")

    d = np.load(matrix_path(ds_name), allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    bounds = dataset_bounds(ds_name, benchmark_names, DATA)     # D2
    obs_frac = O.sum() / (M * N)
    print(f"{M} models × {N} tasks, {obs_frac*100:.1f}% observed", flush=True)

    p = HOLDOUT_FRAC
    splits = list(cv_splits(M, n_folds, n_repeats, p, SEED))
    S = len(splits)
    print(f"{n_repeats} × {n_folds}-fold CV = {S} matched splits, "
          f"{p:.0%} holdout, selectable: benchmarks with ≥ {MIN_TRAIN_OBS} "
          f"training models; methods: {', '.join(methods)}", flush=True)

    metrics = {m: {"r2": np.full((S, k_max), np.nan),
                   "rmse": np.full((S, k_max), np.nan),
                   "n_eval": np.zeros((S, k_max), dtype=np.int64),
                   "n_clipped": np.zeros((S, k_max), dtype=np.int64)}
               for m in methods}
    out = {
        "rep": np.array([s.rep for s in splits], dtype=np.int64),
        "fold": np.array([s.fold for s in splits], dtype=np.int64),
        "split_id": np.array([s.split_id for s in splits], dtype=np.int64),
        "n_train": np.array([len(s.train_idx) for s in splits],
                            dtype=np.int64),
        "n_val": np.array([len(s.val_idx) for s in splits], dtype=np.int64),
        "n_eligible": np.zeros(S, dtype=np.int64),
        "eligible": np.zeros((S, N), dtype=bool),
        "estimation_failed": np.zeros(S, dtype=bool),
        "n_context": np.zeros(S, dtype=np.int64),
        "context_subsampled": np.zeros(S, dtype=bool),
    }
    for m in methods:
        out[f"{m}_selected"] = np.full((S, k_max), -1, dtype=np.int64)
        out[f"{m}_index_ties"] = np.zeros((S, k_max), dtype=np.int64)
        out[f"{m}_tie_breaks"] = np.zeros((S, k_max), dtype=np.int64)
    if "pca" in methods:
        out["pca_selected_per_k"] = np.full((S, k_max, k_max), -1,
                                            dtype=np.int64)

    at5 = min(5, k_max) - 1
    n_calls = n_reused = 0
    for i, s in enumerate(splits):
        t0 = time.time()
        B_train, O_train = B[s.train_idx], O[s.train_idx]
        B_test, O_test = B[s.val_idx], O[s.val_idx]
        M_train = len(s.train_idx)
        M_test = len(s.val_idx)

        # One eligibility mask per split, shared by every method
        el = eligible_mask(O_train)
        target = target_mask(O_train)        # D1, shared by every method
        n_el = int(el.sum())
        out["eligible"][i] = el
        out["n_eligible"][i] = n_el

        print(f"  Rep {s.rep} fold {s.fold}: {M_train} train / {M_test} test, "
              f"{n_el}/{N} eligible", end="", flush=True)

        # Estimate correlation (still needed for selection)
        try:
            R, col_mean, col_std = estimate_corr(B_train, O_train)
        except Exception as e:
            print(f" — estimation failed: {e}", flush=True)
            out["estimation_failed"][i] = True
            continue

        # The context: one draw per split for every method and k (CRN)
        ctx_idx = context_index(M_train, s)
        out["n_context"][i] = M_train if ctx_idx is None else len(ctx_idx)
        out["context_subsampled"][i] = ctx_idx is not None

        # TabImpute's input depends only on the selected set
        imputed = {}
        for method in methods:
            result = run_selection(method, R, k_max, benchmark_names, s, el)
            out[f"{method}_selected"][i] = _padded(result["selected"][:k_max],
                                                   k_max)
            if method == "pca":
                for kk, sel_kk in enumerate(result["selected_per_k"]):
                    out["pca_selected_per_k"][i, kk] = _padded(sel_kk, k_max)
            met = metrics[method]
            for k in range(1, k_max + 1):
                out[f"{method}_index_ties"][i, k - 1] = index_tie_count(
                    result, k)
                out[f"{method}_tie_breaks"][i, k - 1] = tie_break_count(
                    result, k)
                selected = selection_at_k(result, k)
                if selected is None:      # fewer than k eligible: NaN
                    continue
                key = frozenset(selected)
                if key in imputed:
                    n_reused += 1
                else:
                    imputed[key] = impute_tabimpute(
                        imputer, B_train, O_train, B_test, O_test, selected,
                        col_mean, col_std, N, ctx_idx=ctx_idx,
                        target=target, bounds=bounds)
                    n_calls += 1
                imp = imputed[key]
                met["r2"][i, k - 1] = imp["r2"]
                met["rmse"][i, k - 1] = imp["rmse"]
                met["n_eval"][i, k - 1] = imp["n_eval"]
                met["n_clipped"][i, k - 1] = imp["n_clipped"]

        elapsed = time.time() - t0
        print("  " + "  ".join(
            f"{m} R²@{at5 + 1}={metrics[m]['r2'][i, at5]:.3f}"
            for m in methods) + f"  ({elapsed:.1f}s)", flush=True)

    print(f"TabImpute: {n_calls} imputations, {n_reused} reused "
          "(same selected set within a split)", flush=True)
    return metrics, out, benchmark_names


# ── Paired statistics over matched splits ────────────────────────

def paired_rows(ds_name, methods, metrics, out, k_max):
    """paired_stats.paired_stats of R² for every method pair (oriented by
    paired_stats.orient, e.g. mi − entropy) and every k, as CSV rows
    (PAIRED_FIELDS)."""
    rows = []
    for k in range(1, k_max + 1):
        for ma, mb in orient(methods):
            st = paired_stats(metrics[ma]["r2"][:, k - 1],
                              metrics[mb]["r2"][:, k - 1],
                              out["n_train"], out["n_val"], out["rep"])
            rows.append({"dataset": ds_name, "holdout_frac": HOLDOUT_FRAC,
                         "k": k, "method_a": ma, "method_b": mb, **st})
    return rows


def write_paired_csv(path, rows):
    write_csv(rows, path, PAIRED_FIELDS)
    print(f"Paired statistics saved to {path}", flush=True)


# ── Reporting ─────────────────────────────────────────────────────

def print_summary(ds_name, methods, metrics, out, k_max, paired):
    S = len(out["split_id"])
    n_el = out["n_eligible"]
    N = out["eligible"].shape[1]
    print(f"\n{display_name(ds_name)}: eligible benchmarks per split "
          f"{n_el.min()}–{n_el.max()} of {N} "
          f"({int(np.sum(n_el < N))}/{S} splits restricted"
          + (f", {int(np.sum(n_el < k_max))} with fewer than k_max = "
             f"{k_max}: NaN beyond" if (n_el < k_max).any() else "") + ")")
    for m in methods:
        ties = out[f"{m}_index_ties"][:, -1]
        print(f"index ties ({m}, k ≤ {k_max}): {int(ties.sum())} picks "
              f"in {int(np.count_nonzero(ties))}/{S} splits")
    n_sub = int(out["context_subsampled"].sum())
    print(f"context: {out['n_context'].min()}–{out['n_context'].max()} "
          f"training rows ({n_sub}/{S} splits subsampled, Split.seed(SEED + "
          f"{CONTEXT_SEED_OFFSET}))")
    for m in methods:
        clipped = metrics[m]["n_clipped"]
        n_ev = int(metrics[m]["n_eval"].sum())
        print(f"valid-range prediction clip ({m}): {int(clipped.sum())} "
              f"of {n_ev} scored predictions, in "
              f"{int(np.count_nonzero(clipped.sum(axis=1)))}/{S} splits")
    ks_pair = [k for k in [5, 10] if k <= k_max] or [k_max]
    print(f"Paired R² differences a − b over {S} splits "
          "(Nadeau–Bengio 95% CI, win rate):")
    for row in paired:
        if (row["k"] not in ks_pair or row["n_splits"] < 2
                or (row["method_a"] not in ("entropy", "mi")
                    and "facloc" not in (row["method_a"],
                                         row["method_b"]))):
            continue
        print(f"  k={row['k']:<3}{row['method_a']:>8} − {row['method_b']:<11}"
              f"{row['mean_diff']:+.3f}  [{row['ci_low']:+.3f}, "
              f"{row['ci_high']:+.3f}]  win {row['win_rate']:.2f}  "
              f"(n={row['n_splits']})")


# ── Plotting ──────────────────────────────────────────────────────

def plot_comparison(ds_name, metrics, methods, k_max=K_MAX, tag=""):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ks = np.arange(1, k_max + 1)

    for method in methods:
        c = METHOD_COLOR[method]
        ls = METHOD_STYLE[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        valid = ~np.all(np.isnan(metrics[method]["r2"]), axis=1)
        if not valid.any():
            continue

        # All-NaN columns (fewer than k eligible benchmarks) plot as gaps
        mean_r2, std_r2 = _nan_mean_std(metrics[method]["r2"][valid], axis=0)

        ax.plot(ks, mean_r2, ls, color=c, label=METHOD_LABEL[method],
                markersize=ms, linewidth=lw)
        ax.fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                        color=c, alpha=0.15)

    ax.set_ylabel("Imputation $R^2$ (TabImpute V2)")
    ax.set_xlabel("Selected benchmarks $k$")
    ax.set_xlim(0.5, k_max + 0.5)
    ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= k_max])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{display_name(ds_name)}")

    plt.tight_layout()
    out_path = OUT / f"tabimpute_r2_{tag}{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}", flush=True)
    plt.close()


def save_results(ds_name, methods, metrics, out, bench_names, n_repeats,
                 n_folds, k_max, provenance, tag=""):
    save_dict = {"benchmark_names": bench_names}
    for method in methods:
        for key, arr in metrics[method].items():
            save_dict[f"{method}_{key}"] = arr
    save_dict["methods"] = np.array(methods)
    save_dict.update(out)
    save_dict.update(holdout_frac=np.float64(HOLDOUT_FRAC),
                     n_folds=np.int64(n_folds), n_repeats=np.int64(n_repeats),
                     seed=np.int64(SEED),
                     min_train_obs=np.int64(MIN_TRAIN_OBS),
                     min_target_obs=np.int64(cv_split.MIN_TARGET_OBS),
                     k_max=np.int64(k_max),
                     context_seed_offset=np.int64(CONTEXT_SEED_OFFSET),
                     max_context=np.int64(MAX_CONTEXT),
                     pred_clip=np.bool_(score_bounds.PRED_CLIP),
                     z_clip=np.float64(Z_CLIP),
                     ss_tot_floor=np.float64(SS_TOT_FLOOR))
    save_dict.update({key: np.array(val) for key, val in provenance.items()})
    path = LOGS / f"tabimpute_{tag}{ds_name}.npz"
    np.savez(path, **save_dict)
    print(f"Results saved to {path}", flush=True)


# ── Combining the logs of earlier runs ────────────────────────────

def merge_logs(ds_name, methods, paths, n_repeats, n_folds, k_max):
    """(metrics, out, benchmark_names, provenance) of `methods` on dataset
    ds_name, assembled from the logs `paths` of earlier runs of this script
    (npz files; '{ds}' in a path is replaced by ds_name) instead of
    running TabImpute (module docstring, "Combining runs"): each method's
    arrays come from the first log that has the method.  Every log must
    agree with the first on benchmark_names, SPLIT_KEYS, SCALAR_KEYS and
    PROVENANCE_KEYS other than the device (recorded as the devices of the
    used logs joined by '+'), and the scalars must be this script's and
    the run's (n_repeats, n_folds, k_max), so the result is what one run of
    `methods` writes.  ValueError otherwise, or if no log has a method."""
    logs = []
    for p in paths:
        path = pathlib.Path(str(p).replace("{ds}", ds_name))
        if not path.is_file():
            raise ValueError(f"no log {path}")
        with np.load(path, allow_pickle=True) as d:
            logs.append((path, {key: d[key] for key in d.files}))
    ref_path, ref = logs[0]
    same = (("benchmark_names",) + SPLIT_KEYS + SCALAR_KEYS
            + tuple(k for k in PROVENANCE_KEYS if k != "device"))
    for path, log in logs:
        for key in same + ("methods", "device"):
            if key not in log:
                raise ValueError(f"{path} has no {key!r} (not a log of "
                                 f"this script?)")
        for key in same:
            a, b = np.asarray(ref[key]), np.asarray(log[key])
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError(f"{path} and {ref_path} disagree on "
                                 f"{key!r}")
    current = {"holdout_frac": HOLDOUT_FRAC, "seed": SEED,
               "min_train_obs": MIN_TRAIN_OBS,
               "min_target_obs": cv_split.MIN_TARGET_OBS,
               "context_seed_offset": CONTEXT_SEED_OFFSET,
               "max_context": MAX_CONTEXT,
               "pred_clip": bool(score_bounds.PRED_CLIP), "z_clip": Z_CLIP,
               "ss_tot_floor": SS_TOT_FLOOR, "n_repeats": n_repeats,
               "n_folds": n_folds, "k_max": k_max}
    bad = [key for key, val in current.items() if ref[key].item() != val]
    if bad:
        raise ValueError(f"{ref_path} was written with other settings "
                         f"than this run's ({', '.join(bad)})")

    metrics = {}
    out = {key: ref[key] for key in SPLIT_KEYS}
    devices = []
    for m in methods:
        src = next((log for _, log in logs
                    if m in [str(x) for x in np.ravel(log["methods"])]),
                   None)
        if src is None:
            raise ValueError(f"no log has method {m!r} "
                             f"({', '.join(str(p) for p, _ in logs)})")
        metrics[m] = {key: src[f"{m}_{key}"] for key in METRIC_KEYS}
        for key in METHOD_OUT_KEYS:
            out[f"{m}_{key}"] = src[f"{m}_{key}"]
        if m == "pca":
            pca_sets = src["pca_selected_per_k"]
        if str(src["device"]) not in devices:
            devices.append(str(src["device"]))
    if "pca" in methods:
        out["pca_selected_per_k"] = pca_sets
    provenance = {key: ref[key].item() for key in PROVENANCE_KEYS}
    provenance["device"] = "+".join(devices)
    return metrics, out, ref["benchmark_names"], provenance


# ── Main ──────────────────────────────────────────────────────────

def _env_list(name):
    """Comma-separated list from the environment ([] if unset or empty)."""
    return [v for v in os.environ.get(name, "").split(",") if v]


def parse_args(argv=None):
    """Arguments, the method list and the output tag ("" or "extra_")."""
    env_datasets = _env_list("BENCHSELECT_DATASETS")
    env_methods = _env_list("BENCHSELECT_METHODS")
    parser = argparse.ArgumentParser(
        description="Benchmark selection with TabImpute V2 imputation on "
                    "the matched repeated-CV splits.")
    parser.add_argument("--datasets", nargs="+",
                        default=env_datasets or DEFAULT_DATASETS,
                        help="dataset names, each read from "
                             "{name}.matrix.npz in the data directory "
                             "(default: $BENCHSELECT_DATASETS, else "
                             f"{' '.join(DEFAULT_DATASETS)})")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"selection methods out of {' '.join(ALL_METHODS)} "
                             "(default: $BENCHSELECT_METHODS, else "
                             f"{' '.join(DEFAULT_METHODS)}); any override "
                             f"saves under the '{EXTRA_TAG}' tag")
    parser.add_argument("--repeats", type=int, default=TABIMPUTE_REPEATS,
                        help="CV repetitions (default %(default)s = "
                             "repetition 0 only, the original folds; 5 = "
                             "the full matched design)")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--kmax", type=int, default=K_MAX)
    parser.add_argument("--device",
                        default=os.environ.get("BENCHSELECT_TABIMPUTE_DEVICE"),
                        help="torch device (default: "
                             "$BENCHSELECT_TABIMPUTE_DEVICE, else cuda if "
                             "available, else cpu)")
    parser.add_argument("--revision", default=CKPT_REVISION,
                        help=f"commit of {CKPT_REPO} to load {CKPT_FILE} "
                             "from (default: the pinned %(default)s)")
    parser.add_argument("--checkpoint", default=None,
                        help="local checkpoint file instead of the Hub "
                             "download (its sha256 is checked as well)")
    parser.add_argument("--ckpt-sha256",
                        default=os.environ.get(
                            "BENCHSELECT_TABIMPUTE_CKPT_SHA256", CKPT_SHA256),
                        help="required sha256 of the checkpoint (default: "
                             "$BENCHSELECT_TABIMPUTE_CKPT_SHA256, else the "
                             "pinned one); '' records it without a check")
    parser.add_argument("--merge-from", nargs="+", default=None,
                        metavar="LOG",
                        help="run no TabImpute: assemble the logs of the "
                             "method set from these logs of earlier runs "
                             "('{ds}' = the dataset; module docstring, "
                             "'Combining runs')")
    parser.add_argument("--plot-entropy-rf", action="store_true",
                        help="ignored: entropy_rf was dropped (protocol "
                             "rule D3); kept so old command lines run")
    args = parser.parse_args(argv)

    if args.methods is not None:
        methods = [m for tok in args.methods for m in tok.split(",") if m]
    else:
        methods = env_methods or list(DEFAULT_METHODS)
    tag = EXTRA_TAG if (args.methods is not None or env_methods) else ""
    if "entropy_rf" in methods:
        parser.error("entropy_rf was dropped (protocol rule D3, "
                     "2026-10-18); choose from " + " ".join(ALL_METHODS))
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown or not methods:
        parser.error(f"unknown or no methods {unknown} (choose from "
                     f"{' '.join(ALL_METHODS)})")
    if len(set(methods)) != len(methods):
        parser.error(f"duplicate methods in {methods}")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.kmax < 1:
        parser.error("--kmax must be at least 1")
    missing = [ds for ds in args.datasets if not matrix_path(ds).is_file()]
    if missing:
        avail = sorted(p.name[:-len(".matrix.npz")]
                       for p in DATA.glob("*.matrix.npz"))
        parser.error("no data matrix for " + ", ".join(
            f"{ds} ({matrix_path(ds)})" for ds in missing)
            + f"; available: {' '.join(avail)}")
    return args, methods, tag


def main(argv=None):
    args, methods, tag = parse_args(argv)
    plot_methods = methods
    total_t0 = time.time()

    if args.merge_from:
        print(f"Combining the logs {' '.join(args.merge_from)} "
              f"(no TabImpute)")
    else:
        print("Loading TabImputeV2...")
        imputer, provenance = load_imputer(args.device, args.revision,
                                           args.checkpoint, args.ckpt_sha256)
        print(f"Model loaded on {provenance['device']}.\n")

    all_metrics = {}
    for ds_name in args.datasets:
        ds_t0 = time.time()
        if args.merge_from:
            try:
                metrics, out, bench_names, provenance = merge_logs(
                    ds_name, methods, args.merge_from, args.repeats,
                    args.folds, args.kmax)
            except ValueError as e:
                raise SystemExit(f"eval_tabimpute.py --merge-from "
                                 f"({ds_name}): {e}")
        else:
            metrics, out, bench_names = run_dataset(
                ds_name, imputer, methods, args.repeats, args.folds,
                args.kmax)
        all_metrics[ds_name] = metrics
        paired = paired_rows(ds_name, methods, metrics, out, args.kmax)
        print_summary(ds_name, methods, metrics, out, args.kmax, paired)
        plot_comparison(ds_name, metrics, plot_methods, args.kmax, tag)
        save_results(ds_name, methods, metrics, out, bench_names,
                     args.repeats, args.folds, args.kmax, provenance, tag)
        write_paired_csv(LOGS / f"tabimpute_{tag}{ds_name}_paired.csv",
                         paired)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{ds_name.upper()} completed in {ds_elapsed:.0f}s", flush=True)

    # Print summary table
    ks_report = [k for k in (5, 10, 15) if k <= args.kmax]
    print(f"\n{'='*70}")
    print("SUMMARY: Mean R² at k = " + ", ".join(map(str, ks_report))
          + " (TabImpute V2 imputation)")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Method':<10} "
          + " ".join(f"{f'R²@{k}':>8}" for k in ks_report))
    print("-" * (24 + 9 * len(ks_report)))
    for ds_name in args.datasets:
        for method in methods:
            r2 = all_metrics[ds_name][method]["r2"]
            valid = ~np.all(np.isnan(r2), axis=1)
            if valid.any():
                mr2, _ = _nan_mean_std(r2[valid], axis=0)
                print(f"{display_name(ds_name):<12} {method:<10} "
                      + " ".join(f"{mr2[k - 1]:>8.4f}" for k in ks_report))
        print("-" * (24 + 9 * len(ks_report)))

    total_elapsed = time.time() - total_t0
    print(f"\nAll datasets completed in {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
