#!/usr/bin/env python3
"""Build the Open LLM Leaderboard v2 SUB-TASK score matrix (``olv2sub``).

Rows are models, columns are the 42 leaf scores that the leaderboard v2
aggregates are computed from:

    IFEval    4 metrics  prompt/inst level x strict/loose accuracy   (n=541 prompts)
    BBH      24 tasks    acc_norm                                     (3-shot)
    MATH-hard 7 tasks    exact_match  (Minerva MATH, level-5 subset)  (4-shot)
    GPQA      3 tasks    acc_norm     (diamond / extended / main)     (0-shot)
    MuSR      3 tasks    acc_norm                                     (0-shot)
    MMLU-Pro  1 task     acc                                          (5-shot)

All values are RAW metric values in [0, 1] as written by lm-evaluation-harness
(not the leaderboard's baseline-normalised 0-100 scores).  The legacy
``leaderboard_arc_challenge`` task (present in ~6% of files) is dropped.

Sources (both pinned by commit sha, anonymous access):
    open-llm-leaderboard/results   per-model ``<org>/<model>/results_<ts>.json``
                                   (10,509 files; frozen 2025-03-15)
    open-llm-leaderboard/contents  ``data/train-00000-of-00001.parquet``: one row
                                   per leaderboard entry (model x precision) with
                                   metadata and the six raw aggregates.

Expected layout of ``--raw-dir`` (see data/raw_manifests/olv2sub.json):
    results_git/                       ``git clone --depth 1`` of the results repo
    contents/train-00000-of-00001.parquet
    contents/README.md
    api_results.json, api_contents.json   HF API responses used to pin revisions
    fetch_info.json                    optional download provenance written by the
                                       person fetching (fetched_at, method); only
                                       copied into the meta/manifest, never needed
                                       to compute the matrix

Fetching the pinned raw inputs (anonymous, no token; ~1.3 GB; 5 HTTP requests
plus one git clone -- stay well below HF's anonymous rate limit):

    RAW=~/benchselect-data/raw/olv2sub; mkdir -p $RAW/contents && cd $RAW
    HF=https://huggingface.co
    curl -fsS -o api_results.json  $HF/api/datasets/open-llm-leaderboard/results
    curl -fsS -o api_contents.json $HF/api/datasets/open-llm-leaderboard/contents
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-tags \\
        $HF/datasets/open-llm-leaderboard/results results_git
    C=$HF/datasets/open-llm-leaderboard/contents/resolve/9c09a7cae43334062a82cb164f2ef255013dafa2
    curl -fsSL -o contents/train-00000-of-00001.parquet $C/data/train-00000-of-00001.parquet
    curl -fsSL -o contents/README.md $C/README.md

The contents file is fetched by commit, so it cannot drift.  The results clone
gets the current HEAD; the builder refuses any HEAD other than ``RESULTS_REV``.
If the HF repo has moved on, check out the pinned commit instead, e.g.
``git -C results_git fetch --depth 1 origin <RESULTS_REV> && git -C results_git
checkout --detach <RESULTS_REV>`` (needs server support for fetching by sha),
or a full-history ``git clone`` followed by ``git checkout <RESULTS_REV>``, or
``huggingface_hub.snapshot_download(..., repo_type="dataset",
revision=RESULTS_REV, max_workers=1)`` (one GET per file, ~10.5k requests --
throttle it).  Independently of git, the builder verifies the checked-out tree
file by file: the sha256 of the sorted ``sha256sum`` listing of every non-.git
file must equal ``RESULTS_TREE_SHA256`` (10,510 files, per-file hashes in
raw_manifests/olv2sub.json); on a mismatch it names the differing paths if an
earlier manifest is present in ``--manifest-dir``.  The contents parquet is
checked against ``CONTENTS_SHA256``.

Row selection
-------------
Many models have several result files (original run plus bulk re-uploads on
2024-10-24 and 2025-02-13 that re-score MATH-hard).  For every non-flagged
``contents`` row we compute, for each result file with the same ``model_name``,
the six aggregates exactly the way the leaderboard defines its "Raw" columns:

    IFEval Raw  = (prompt_level_strict_acc + inst_level_strict_acc) / 2
    BBH Raw     = unweighted mean of the 24 BBH sub-task acc_norm
    MATH Raw    = sample-weighted mean of the 7 MATH-hard exact_match
    GPQA Raw    = sample-weighted mean of the 3 GPQA acc_norm
    MuSR Raw    = unweighted mean of the 3 MuSR acc_norm
    MMLU-PRO Raw= acc

(these definitions reproduce the contents table to <= 2.3e-16; 4,485 of the
4,493 selected models match exactly, the other 8 are leaderboard rows that mix
values from two files or a MATH value found in no file; see meta counts), and
pick the file with the smallest L_inf distance to the contents row.  Ties
(|d1 - d2| <= 1e-12) go to the latest file timestamp, then the path; for 166
models the tied files differ in some leaf values (MATH-hard sub-tasks re-scored
with identical totals, IFEval loose metrics), flagged per row by
``models.match_tied_leaf_differs`` / ``match_tied_max_abs_leaf_diff``.  Files run
with a sample ``limit`` and files missing any of the six aggregates are not
eligible.  The 79 models listed under two precisions (bfloat16 and float16)
keep the row with the later Submission Date, then bfloat16, then eval_name.
The one ``Flagged`` model is dropped, as are result-only models that never
appear in ``contents``.

Order
-----
Rows: model ``fullname`` (HF repo id) sorted by Python ``sorted`` (Unicode
code point order).  Columns: groups in leaderboard order (IFEval, BBH, MATH,
GPQA, MuSR, MMLU-Pro); within a group, sub-task names sorted alphabetically,
IFEval metrics in the fixed order listed in ``IFEVAL_METRICS``.

Outputs (in ``--output-dir``):
    olv2sub.matrix.npz / olv2sub.matrix.csv  (``build_matrices.save_matrix``)
    olv2sub.meta.json                        model / benchmark metadata, rules,
                                             provenance, consistency checks
    raw_manifests/olv2sub.json               every raw input with size + sha256

Caveats: MATH-hard scorer versions are mixed exactly as on the leaderboard
(task version 3.0 for the 2025-02-13 bulk re-scoring, 2.0 / 1.0 otherwise);
``models.math_hard_task_version`` in the meta file records it per row.  The
three GPQA columns share questions (diamond 198 within main 448 within extended
546, Rein et al. 2023), so they are correlated by construction;
``benchmarks.item_group`` marks columns drawn from one question pool (GPQA, and
the four IFEval metrics, which are also one run: ``benchmarks.cost_group``).

Only json / pandas / numpy are used to parse the (untrusted) downloads.
Run with ``python -I`` from a directory that holds no downloaded files, e.g.

    cd /tmp && python -I .../data/build_olv2sub.py \\
        --raw-dir ~/benchselect-data/raw/olv2sub --output-dir .../data \\
        --reference-csv .../data/open_llm.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ``python -I`` does not put the script directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_matrices import save_matrix  # noqa: E402

NAME = "olv2sub"

RESULTS_REPO = "open-llm-leaderboard/results"
RESULTS_REV = "aa81ecc38fdc5708254b833923368970efdf5ef5"
RESULTS_LAST_MODIFIED = "2025-03-15T05:57:14Z"
# sha256 of the `LC_ALL=C sort`ed `sha256sum` listing of every non-.git file of
# the checkout at RESULTS_REV (10,509 results JSONs + .gitattributes)
RESULTS_TREE_SHA256 = "db2d994cafad7c9719ed4fa5034489b47c2d41164da7f79635f7ffeb34481db7"
RESULTS_TREE_N_FILES = 10510
CONTENTS_REPO = "open-llm-leaderboard/contents"
CONTENTS_REV = "9c09a7cae43334062a82cb164f2ef255013dafa2"
CONTENTS_LAST_MODIFIED = "2025-03-20T12:17:27Z"
CONTENTS_FILE = "data/train-00000-of-00001.parquet"
CONTENTS_SHA256 = "f13a7061e63133bc8745d89f8c7a56148584940ede22d3b27e21cad1e078dca5"
HF = "https://huggingface.co/datasets"
HF_TERMS = "https://huggingface.co/terms-of-service"

# raw inputs other than the results clone: local path -> (url, revision).  The
# method / fetched_at come from the optional fetch_info.json.
RAW_FILES = {
    "contents/train-00000-of-00001.parquet":
        (f"{HF}/{CONTENTS_REPO}/resolve/{CONTENTS_REV}/{CONTENTS_FILE}", CONTENTS_REV),
    "contents/README.md": (f"{HF}/{CONTENTS_REPO}/resolve/{CONTENTS_REV}/README.md",
                           CONTENTS_REV),
    "api_results.json": (f"https://huggingface.co/api/datasets/{RESULTS_REPO}", RESULTS_REV),
    "api_contents.json": (f"https://huggingface.co/api/datasets/{CONTENTS_REPO}",
                          CONTENTS_REV),
}

# GPQA subsets are nested (Rein et al. 2023, arXiv:2311.12022):
# diamond (198) within main (448) within extended (546).
GPQA_NESTING = "gpqa/diamond within gpqa/main within gpqa/extended"

IFEVAL_TASK = "leaderboard_ifeval"
IFEVAL_METRICS = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]
BBH_TASKS = sorted([
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation", "navigate",
    "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies",
])
MATH_TASKS = sorted([
    "algebra_hard", "counting_and_prob_hard", "geometry_hard",
    "intermediate_algebra_hard", "num_theory_hard", "prealgebra_hard",
    "precalculus_hard",
])
GPQA_TASKS = ["diamond", "extended", "main"]
MUSR_TASKS = ["murder_mysteries", "object_placements", "team_allocation"]

# (group label, harness group key, metric, [(column name, harness task key)])
GROUPS = [
    ("IFEval", IFEVAL_TASK, None,
     [(f"ifeval/{m}", IFEVAL_TASK) for m in IFEVAL_METRICS]),
    ("BBH", "leaderboard_bbh", "acc_norm",
     [(f"bbh/{t}", f"leaderboard_bbh_{t}") for t in BBH_TASKS]),
    ("MATH_Lvl5", "leaderboard_math_hard", "exact_match",
     [(f"math_hard/{t[:-5]}", f"leaderboard_math_{t}") for t in MATH_TASKS]),
    ("GPQA", "leaderboard_gpqa", "acc_norm",
     [(f"gpqa/{t}", f"leaderboard_gpqa_{t}") for t in GPQA_TASKS]),
    ("MUSR", "leaderboard_musr", "acc_norm",
     [(f"musr/{t}", f"leaderboard_musr_{t}") for t in MUSR_TASKS]),
    ("MMLU-PRO", "leaderboard_mmlu_pro", "acc",
     [("mmlu_pro", "leaderboard_mmlu_pro")]),
]

COLUMNS = []  # (column, group label, harness task, metric)
for _label, _gkey, _metric, _cols in GROUPS:
    for _i, (_col, _task) in enumerate(_cols):
        COLUMNS.append((_col, _label, _task,
                        IFEVAL_METRICS[_i] if _label == "IFEval" else _metric))
COLUMN_NAMES = [c[0] for c in COLUMNS]
AGG_NAMES = [g[0] for g in GROUPS]
CONTENTS_RAW = {
    "IFEval": "IFEval Raw", "BBH": "BBH Raw", "MATH_Lvl5": "MATH Lvl 5 Raw",
    "GPQA": "GPQA Raw", "MUSR": "MUSR Raw", "MMLU-PRO": "MMLU-PRO Raw",
}
TIE_TOL = 1e-12
EXACT_TOL = 1e-9

CONTENTS_META = {  # meta key -> contents column
    "precision": "Precision",
    "upload_to_hub_date": "Upload To Hub Date",
    "submission_date": "Submission Date",
    "params_b": "#Params (B)",
    "type": "Type",
    "architecture": "Architecture",
    "base_model": "Base Model",
    "hub_license": "Hub License",
    "weight_type": "Weight type",
    "model_sha": "Model sha",
    "available_on_hub": "Available on the hub",
    "moe": "MoE",
    "merged": "Merged",
    "official_providers": "Official Providers",
    "chat_template": "Chat Template",
    "generation": "Generation",
    "hub_hearts": "Hub ❤️",
    "co2_cost_kg": "CO₂ cost (kg)",
    "leaderboard_average_normalized": "Average ⬆️",
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(v):
    """Finite float or None (rejects bools, strings such as 'N/A', NaN)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def clean(v):
    """JSON-safe scalar from a pandas cell."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        return v if v != "" else None
    return v


def _is_dict_list(v) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)


def to_json(v, ind: int = 0) -> str:
    """Deterministic, strict JSON: dicts indented one space per level, lists of
    scalars on one line, lists of dicts one compact object per line."""
    pad = " " * ind
    kw = dict(ensure_ascii=False, allow_nan=False, separators=(", ", ": "))
    if isinstance(v, dict) and v:
        items = [f"{pad} {json.dumps(k, **kw)}: {to_json(x, ind + 1)}" for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if _is_dict_list(v):
        items = [pad + " " + (to_json(x, ind + 1) if any(_is_dict_list(y) for y in x.values())
                              else json.dumps(x, **kw)) for x in v]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(v, **kw)


def git_head(repo: Path) -> str:
    """Resolve HEAD of a local clone by reading .git files (no subprocess)."""
    git = repo / ".git"
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    p = git / ref
    if p.exists():
        return p.read_text().strip()
    for line in (git / "packed-refs").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    raise RuntimeError(f"cannot resolve {ref}")


def list_result_files(root: Path) -> list[str]:
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d != ".git")
        for f in fn:
            if f.startswith("results_") and f.endswith(".json"):
                out.append(os.path.relpath(os.path.join(dp, f), root))
    return sorted(out)


def tree_listing(root: Path) -> tuple[list[dict], str]:
    """(path, size, sha256) of every non-.git file under ``root`` in byte order
    (== ``LC_ALL=C sort``), and the sha256 of the equivalent listing
    ``find . -type f ! -path './.git/*' | cut -c3- | LC_ALL=C sort | xargs sha256sum``."""
    files = []
    for p in root.rglob("*"):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".git/") or rel == ".git" or not p.is_file():
            continue
        files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256_file(p)})
    files.sort(key=lambda f: f["path"].encode())
    digest = hashlib.sha256("".join(
        f"{f['sha256']}  {f['path']}\n" for f in files).encode()).hexdigest()
    return files, digest


def check_tree(files: list[dict], digest: str, manifest_path: Path) -> None:
    """Refuse a results checkout whose files differ from the pinned tree."""
    if digest == RESULTS_TREE_SHA256 and len(files) == RESULTS_TREE_N_FILES:
        return
    msg = [f"results tree digest {digest} ({len(files)} files) != pinned "
           f"{RESULTS_TREE_SHA256} ({RESULTS_TREE_N_FILES} files)"]
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        pinned = {f["path"]: f["sha256"] for f in old["inputs"][0]["files"]}
        have = {f["path"]: f["sha256"] for f in files}
        bad = sorted(p for p in set(pinned) | set(have) if pinned.get(p) != have.get(p))
        msg.append(f"{len(bad)} paths differ from {manifest_path}: {bad[:20]}")
    raise SystemExit("\n".join(msg))


def declared_license(api: dict, readme: str) -> dict:
    """License declared on the HF Hub, from the pinned API response / card."""
    tags = [t for t in (api.get("tags") or []) if t.startswith("license:")]
    card = api.get("cardData") or {}
    front = readme.split("---", 2)[1] if readme.startswith("---") else ""
    front_keys = [ln.split(":", 1)[0] for ln in front.splitlines()
                  if ln and not ln[0].isspace() and not ln.startswith("-") and ":" in ln]
    return {"license_tags": tags, "card_license": card.get("license"),
            "readme_front_matter_keys": front_keys}


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------

# Mirror of data/download_open_llm.py (BENCHMARK_MAP + _extract_scores), which
# produced data/open_llm.csv.  Copied rather than imported because that module
# loads data/.env and needs huggingface_hub/tqdm at import time.
OLD_BENCHMARK_MAP = {
    "IFEval": ("leaderboard_ifeval", "prompt_level_strict_acc,none"),
    "BBH": ("leaderboard_bbh", "acc_norm,none"),
    "MATH_Lvl5": ("leaderboard_math_hard", "exact_match,none"),
    "GPQA": ("leaderboard_gpqa", "acc_norm,none"),
    "MUSR": ("leaderboard_musr", "acc_norm,none"),
    "MMLU-PRO": ("leaderboard_mmlu_pro", "acc,none"),
}


def old_extract_scores(data: dict) -> dict | None:
    results = data.get("results") or {}
    if not results:
        return None
    scores = {}
    for bench_name, (group_key, metric_key) in OLD_BENCHMARK_MAP.items():
        val = None
        if group_key in results:
            val = results[group_key].get(metric_key)
        if val is None:  # substring fallback: may hit a sub-task (e.g. bbh_<first>)
            for k, v in results.items():
                if group_key in k and isinstance(v, dict):
                    val = v.get(metric_key)
                    if val is not None:
                        break
        scores[bench_name] = val
    if all(v is None for v in scores.values()):
        return None
    return scores


def direct_group_scores(data: dict) -> dict:
    """What ``old_extract_scores`` reads WITHOUT its substring fallback: the
    file's own results[<group>][<metric>] entries (None where blank)."""
    results = data.get("results") or {}
    out = {}
    for bench_name, (group_key, metric_key) in OLD_BENCHMARK_MAP.items():
        e = results.get(group_key)
        out[bench_name] = as_float(e.get(metric_key)) if isinstance(e, dict) else None
    return out


def parse_result_file(root: Path, rel: str) -> dict:
    path = root / rel
    data = json.loads(path.read_bytes())
    res = data.get("results") or {}
    nsamp = data.get("n-samples") or {}
    cfg = data.get("config") or {}
    groups = data.get("groups") or {}

    leaf = {}
    n_eff = {}
    limited = cfg.get("limit") is not None
    for col, _label, task, metric in COLUMNS:
        entry = res.get(task)
        leaf[col] = as_float(entry.get(f"{metric},none")) if isinstance(entry, dict) else None
        ns = nsamp.get(task)
        if isinstance(ns, dict):
            n_eff[task] = ns.get("effective")
            if ns.get("effective") != ns.get("original"):
                limited = True

    # harness group entries (sample-weighted means), where present.  NB: in the
    # MATH re-scoring re-uploads only results[...] was updated; the copy under
    # "groups" (and the combined results["leaderboard"]) keeps the old value.
    group_vals, groups_copy = {}, {}
    for label, gkey, metric, _cols in GROUPS:
        if metric is None or label == "MMLU-PRO":
            continue
        g = res.get(gkey)
        group_vals[label] = as_float(g.get(f"{metric},none")) if isinstance(g, dict) else None
        g = groups.get(gkey)
        groups_copy[label] = as_float(g.get(f"{metric},none")) if isinstance(g, dict) else None

    ts = rel.rsplit("/", 1)[-1][len("results_"):-len(".json")]
    t_eval = data.get("total_evaluation_time_seconds")
    try:
        t_eval = float(t_eval)
        t_eval = t_eval if math.isfinite(t_eval) else None
    except (TypeError, ValueError):
        t_eval = None

    return {
        "rel": rel,
        "model_name": data.get("model_name") or os.path.dirname(rel),
        "ts": ts,
        "limited": limited,
        "leaf": leaf,
        "n_eff": n_eff,
        "group_vals": group_vals,
        "groups_copy": groups_copy,
        "old_scores": old_extract_scores(data),
        "direct_scores": direct_group_scores(data),
        "total_evaluation_time_seconds": t_eval,
        "eval_unix_time": as_float(data.get("date")),
        "model_dtype": cfg.get("model_dtype"),
        "model_revision": cfg.get("model_revision"),
        "model_num_parameters": cfg.get("model_num_parameters"),
        "n_shot": {k: v for k, v in (data.get("n-shot") or {}).items()},
        "versions": {k: v for k, v in (data.get("versions") or {}).items()},
        "output_type": {k: (v or {}).get("output_type")
                        for k, v in (data.get("configs") or {}).items()
                        if isinstance(v, dict)},
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def leaderboard_aggregates(rec: dict) -> dict:
    """Six aggregates as defined by the contents table's 'Raw' columns."""
    leaf, n = rec["leaf"], rec["n_eff"]
    out = {}
    for label, _gkey, _metric, cols in GROUPS:
        vals = [leaf[c] for c, _ in cols]
        if any(v is None for v in vals):
            out[label] = None
            continue
        if label == "IFEval":
            out[label] = (leaf["ifeval/prompt_level_strict_acc"]
                          + leaf["ifeval/inst_level_strict_acc"]) / 2
        elif label in ("BBH", "MUSR"):
            out[label] = float(np.mean(vals))
        elif label in ("MATH_Lvl5", "GPQA"):
            w = [n.get(t) for _, t in cols]
            if any(not isinstance(x, int) for x in w):
                out[label] = None
                continue
            out[label] = float(np.dot(vals, w) / np.sum(w))
        else:
            out[label] = vals[0]
    return out


def harness_aggregates(row: pd.Series, n_eff: dict) -> dict:
    """The six aggregates as lm-eval-harness group entries report them
    (sample-weighted means; IFEval = prompt_level_strict_acc), i.e. the
    definition used by data/open_llm.csv."""
    out = {}
    for label, _gkey, _metric, cols in GROUPS:
        if label == "IFEval":
            out[label] = row["ifeval/prompt_level_strict_acc"]
            continue
        vals = np.array([row[c] for c, _ in cols], float)
        w = np.array([n_eff[t] for _, t in cols], float)
        out[label] = float(np.dot(vals, w) / w.sum())
    return out


# ----------------------------------------------------------------------------
# main build
# ----------------------------------------------------------------------------

def build(raw_dir: Path, output_dir: Path, manifest_dir: Path,
          reference_csv: Path | None) -> None:
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_root = raw_dir / "results_git"
    contents_path = raw_dir / "contents" / "train-00000-of-00001.parquet"

    head = git_head(results_root)
    if head != RESULTS_REV:
        raise SystemExit(f"results clone is at {head}, expected {RESULTS_REV}")
    tree_files, tree_digest = tree_listing(results_root)
    check_tree(tree_files, tree_digest, manifest_dir / f"{NAME}.json")
    csha = sha256_file(contents_path)
    if csha != CONTENTS_SHA256:
        raise SystemExit(f"contents parquet sha256 {csha} != pinned {CONTENTS_SHA256}")
    print(f"results @ {head}; tree of {len(tree_files)} files matches pinned digest; "
          "contents sha256 ok")

    fetch_path = raw_dir / "fetch_info.json"
    fetch = json.loads(fetch_path.read_text(encoding="utf-8")) if fetch_path.exists() else {}
    if not fetch:
        print("  note: no fetch_info.json; fetched_at / method recorded as null")
    for rel, (url, rev) in [("results_git", (f"{HF}/{RESULTS_REPO}", RESULTS_REV)),
                            *RAW_FILES.items()]:
        f = fetch.setdefault(rel, {})
        for k, v in (("url", url), ("revision", rev)):
            if f.get(k, v) != v:
                raise SystemExit(f"fetch_info.json[{rel!r}][{k!r}] = {f[k]!r}, expected {v!r}")
        f.setdefault("method", None)
        f.setdefault("fetched_at", None)

    # ---- contents table --------------------------------------------------
    contents = pd.read_parquet(contents_path)
    contents = contents.sort_values("eval_name", kind="mergesort").reset_index(drop=True)
    stats = {"contents_rows": int(len(contents)),
             "contents_unique_models": int(contents["fullname"].nunique())}
    flagged = sorted(contents.loc[contents["Flagged"].astype(bool), "fullname"].unique())
    contents_ok = contents[~contents["fullname"].isin(flagged)]
    stats["flagged_models_dropped"] = flagged

    # ---- result files ----------------------------------------------------
    rels = list_result_files(results_root)
    print(f"parsing {len(rels)} result files …")
    recs = [parse_result_file(results_root, r) for r in rels]
    for r in recs:
        r["agg"] = leaderboard_aggregates(r)
        r["complete"] = all(r["agg"][a] is not None for a in AGG_NAMES)
    by_model: dict[str, list[dict]] = {}
    for r in recs:
        by_model.setdefault(r["model_name"], []).append(r)
    stats["result_files"] = len(recs)
    stats["result_models"] = len(by_model)
    stats["result_files_with_limit"] = sorted(r["rel"] for r in recs if r["limited"])
    stats["result_files_incomplete"] = sum(not r["complete"] for r in recs)
    stats["result_files_dir_ne_model_name"] = sorted(
        r["rel"] for r in recs if os.path.dirname(r["rel"]) != r["model_name"])
    only_results = sorted(set(by_model) - set(contents["fullname"]))
    stats["result_models_not_in_contents_dropped"] = only_results

    # ---- match each contents row to its best file --------------------------
    matches = []
    no_candidate = []
    for _, crow in contents_ok.iterrows():
        target = np.array([crow[CONTENTS_RAW[a]] for a in AGG_NAMES], float)
        cands = [r for r in by_model.get(crow["fullname"], [])
                 if r["complete"] and not r["limited"]]
        if not cands:
            no_candidate.append(crow["eval_name"])
            continue
        scored = []
        for r in cands:
            v = np.array([r["agg"][a] for a in AGG_NAMES], float)
            scored.append((float(np.max(np.abs(v - target))), r))
        best_d = min(d for d, _ in scored)
        tied = [r for d, r in scored if d - best_d <= TIE_TOL]
        tied.sort(key=lambda r: (r["ts"], r["rel"]))
        chosen = tied[-1]
        tie_diff = {c: max(abs(t["leaf"][c] - chosen["leaf"][c]) for t in tied)
                    for c in COLUMN_NAMES}
        tie_diff = {c: d for c, d in tie_diff.items() if d != 0.0}
        matches.append({"crow": crow, "rec": chosen, "linf": best_d,
                        "n_tied": len(tied), "tied_leaf_differs": bool(tie_diff),
                        "tie_diff": tie_diff, "n_candidates": len(cands)})
    stats["contents_rows_without_eligible_file"] = sorted(no_candidate)

    # ---- one row per model -------------------------------------------------
    per_model: dict[str, list[dict]] = {}
    for m in matches:
        per_model.setdefault(m["crow"]["fullname"], []).append(m)

    def prec_key(m):
        c = m["crow"]
        sub = c["Submission Date"] or ""
        return (sub, 1 if c["Precision"] == "bfloat16" else 0, c["eval_name"])

    chosen = {}
    alt_rows = {}
    for name, ms in per_model.items():
        ms = sorted(ms, key=prec_key)
        chosen[name] = ms[-1]
        alt_rows[name] = [x["crow"]["eval_name"] for x in ms[:-1]]
    stats["models_with_two_precisions"] = sum(1 for v in alt_rows.values() if v)

    model_names = sorted(chosen)
    sel = [chosen[m] for m in model_names]

    # ---- matrix ------------------------------------------------------------
    df = pd.DataFrame(
        [[np.nan if m["rec"]["leaf"][c] is None else m["rec"]["leaf"][c]
          for c in COLUMN_NAMES] for m in sel],
        index=pd.Index(model_names, name="model"), columns=COLUMN_NAMES, dtype=np.float64)
    save_matrix(NAME, df, output_dir)

    # ---- per-benchmark metadata -------------------------------------------
    task_n = {}
    for _c, _l, task, _m in COLUMNS:
        vals = sorted({m["rec"]["n_eff"].get(task) for m in sel}, key=str)
        task_n[task] = vals
    nonuniform = {t: v for t, v in task_n.items() if len(v) != 1}
    if nonuniform:
        raise SystemExit(f"non-uniform n-samples among selected files: {nonuniform}")

    def uniform(field, task):
        vals = {m["rec"][field].get(task) for m in sel}
        if field == "versions":  # 2 and 2.0 both occur
            vals = {float(v) for v in vals}
        vals = sorted(vals, key=json.dumps)
        return vals[0] if len(vals) == 1 else vals

    def math_version(rec):
        vs = {float(rec["versions"][t]) for _c, label, t, _m in COLUMNS
              if label == "MATH_Lvl5" and t in rec["versions"]}
        return vs.pop() if len(vs) == 1 else None

    bench_meta = {
        "group": [], "harness_task": [], "metric": [], "n_samples": [],
        "n_shot": [], "output_type": [], "task_version": [], "cost_group": [],
        "item_group": [],
    }
    for col, label, task, metric in COLUMNS:
        bench_meta["group"].append(label)
        bench_meta["harness_task"].append(task)
        bench_meta["metric"].append(f"{metric},none")
        bench_meta["n_samples"].append(task_n[task][0])
        bench_meta["n_shot"].append(uniform("n_shot", task))
        bench_meta["output_type"].append(uniform("output_type", task))
        bench_meta["task_version"].append(uniform("versions", task))
        bench_meta["cost_group"].append(task)
        bench_meta["item_group"].append("leaderboard_gpqa" if label == "GPQA" else task)
    # one run per cost_group; one question pool per item_group (nested GPQA
    # subsets: the pool is the largest subset)
    runs = dict(zip(bench_meta["cost_group"], bench_meta["n_samples"]))
    pools = {}
    for g, n in zip(bench_meta["item_group"], bench_meta["n_samples"]):
        pools[g] = max(pools.get(g, 0), n)
    gpqa_n = {c: n for c, n in zip(COLUMN_NAMES, bench_meta["n_samples"])
              if c.startswith("gpqa/")}
    if not gpqa_n["gpqa/diamond"] < gpqa_n["gpqa/main"] < gpqa_n["gpqa/extended"]:
        raise SystemExit(f"unexpected GPQA subset sizes {gpqa_n}")

    # ---- per-model metadata (columnar, aligned with model_names) -----------
    mm = {k: [] for k in CONTENTS_META}
    extra = ["result_file",
             "eval_unix_time", "total_evaluation_time_seconds", "model_dtype",
             "model_num_parameters", "math_hard_task_version", "n_result_files",
             "n_eligible_result_files", "match_linf", "match_n_tied",
             "match_tied_leaf_differs", "match_tied_max_abs_leaf_diff",
             "other_precision_eval_names"]
    for k in extra:
        mm[k] = []
    contents_raw = {a: [] for a in AGG_NAMES}
    for name, m in zip(model_names, sel):
        c, r = m["crow"], m["rec"]
        for k, col in CONTENTS_META.items():
            v = clean(c[col])
            if k == "params_b" and v is not None and v < 0:  # contents uses -1 = unknown
                v = None
            mm[k].append(v)
        mm["result_file"].append(r["rel"])
        mm["eval_unix_time"].append(r["eval_unix_time"])
        mm["total_evaluation_time_seconds"].append(r["total_evaluation_time_seconds"])
        mm["model_dtype"].append(r["model_dtype"])
        mm["model_num_parameters"].append(clean(r["model_num_parameters"]))
        mm["math_hard_task_version"].append(math_version(r))
        mm["n_result_files"].append(len(by_model[name]))
        mm["n_eligible_result_files"].append(m["n_candidates"])
        mm["match_linf"].append(m["linf"])
        mm["match_n_tied"].append(m["n_tied"])
        mm["match_tied_leaf_differs"].append(m["tied_leaf_differs"])
        mm["match_tied_max_abs_leaf_diff"].append(
            max(m["tie_diff"].values()) if m["tie_diff"] else 0.0)
        mm["other_precision_eval_names"].append(alt_rows[name])
        for a in AGG_NAMES:
            contents_raw[a].append(clean(c[CONTENTS_RAW[a]]))

    linf = np.array([m["linf"] for m in sel])
    stats.update({
        "models_in_matrix": len(model_names),
        "match_linf_exact_le_1e-15": int((linf <= 1e-15).sum()),
        "match_linf_le_1e-9": int((linf <= EXACT_TOL).sum()),
        "match_linf_gt_1e-9": int((linf > EXACT_TOL).sum()),
        "match_linf_max": float(linf.max()),
        "match_models_gt_1e-9": [n for n, m in zip(model_names, sel) if m["linf"] > EXACT_TOL],
        "models_with_multiple_files": int(sum(len(by_model[n]) > 1 for n in model_names)),
        "selected_file_is_latest_eligible": int(sum(
            m["rec"]["ts"] == max(r["ts"] for r in by_model[n]
                                  if r["complete"] and not r["limited"])
            for n, m in zip(model_names, sel))),
        "selected_timestamp_counts": dict(sorted(
            pd.Series([m["rec"]["ts"] for m in sel]).map(
                lambda t: t if t in ("2024-10-24T00-00-00.000000",
                                     "2025-02-13T18-27-04.338360") else "original_run"
            ).value_counts().items())),
        "ties_with_differing_leaf_values": int(sum(m["tied_leaf_differs"] for m in sel)),
        "ties_differing_cells_by_group": dict(sorted(pd.Series(
            [c.split("/")[0] for m in sel for c in m["tie_diff"]],
            dtype=object).value_counts().items())),
        "ties_max_abs_leaf_diff_median_over_models": float(np.median(
            [max(m["tie_diff"].values()) for m in sel if m["tie_diff"]])),
        "ties_max_abs_leaf_diff_max": float(max(
            [max(m["tie_diff"].values()) for m in sel if m["tie_diff"]])),
        "eval_name_eq_fullname_slash_to_underscore_plus_precision": int(sum(
            m["crow"]["eval_name"] == m["crow"]["fullname"].replace("/", "_") + "_"
            + m["crow"]["Precision"] for m in sel)),
        "json_model_revision_eq_contents_model_sha": int(sum(
            m["rec"]["model_revision"] == m["crow"]["Model sha"] for m in sel)),
        "dtype_matches_precision": int(sum(
            (m["rec"]["model_dtype"] or "").replace("torch.", "") == m["crow"]["Precision"]
            for m in sel)),
        "math_hard_task_version_counts": {str(k): v for k, v in sorted(
            pd.Series([math_version(m["rec"]) for m in sel]).value_counts().items())},
    })
    rescoring_ts = "2025-02-13T18-27-04.338360"
    stats["math_hard_v2_original_runs_after_2025-02-13_rescoring"] = int(sum(
        math_version(m["rec"]) == 2.0 and m["rec"]["ts"] > rescoring_ts for m in sel))
    # size of the MATH re-scorings: same model, same run, different scorer version
    shift = {}
    for n in model_names:
        by_v = {}
        for r in by_model[n]:
            if r["complete"] and not r["limited"]:
                by_v.setdefault(math_version(r), r["agg"]["MATH_Lvl5"])
        for a, b in [(1.0, 2.0), (2.0, 3.0), (1.0, 3.0)]:
            if a in by_v and b in by_v:
                shift.setdefault(f"v{a:g}->v{b:g}", []).append(by_v[b] - by_v[a])
    stats["math_hard_aggregate_shift_between_scorer_versions"] = {
        k: {"n_models": len(v), "mean": float(np.mean(v)),
            "median_abs": float(np.median(np.abs(v))),
            "p90_abs": float(np.quantile(np.abs(v), 0.9)),
            "max_abs": float(np.max(np.abs(v)))}
        for k, v in sorted(shift.items())}

    # ---- consistency checks -----------------------------------------------
    n_eff = {task: task_n[task][0] for _c, _l, task, _m in COLUMNS}
    harness = pd.DataFrame([harness_aggregates(df.loc[n], n_eff) for n in model_names],
                           index=model_names)
    # (a) our recomputation vs the selected file's own results[<group>] entries
    grp_cmp = {}
    for a in ["BBH", "MATH_Lvl5", "GPQA", "MUSR"]:
        g = np.array([np.nan if m["rec"]["group_vals"].get(a) is None
                      else m["rec"]["group_vals"][a] for m in sel])
        ok = ~np.isnan(g)
        d = np.abs(harness[a].values[ok] - g[ok])
        gc = np.array([np.nan if m["rec"]["groups_copy"].get(a) is None
                       else m["rec"]["groups_copy"][a] for m in sel])
        stale = (~np.isnan(gc)) & ok & (np.abs(gc - g) > EXACT_TOL)
        grp_cmp[a] = {"n_with_group_entry": int(ok.sum()),
                      "n_blank_group_entry": int((~ok).sum()),
                      "n_le_1e-9": int((d <= EXACT_TOL).sum()),
                      "max_abs_diff": float(d.max()) if d.size else None,
                      "n_stale_copy_under_groups_key": int(stale.sum())}
    consistency = {"recomputed_vs_selected_file_group_entries": grp_cmp}

    # (b) vs data/open_llm.csv (6 harness aggregates written by
    #     download_open_llm.py: keyed by model_name, last file in repo listing
    #     order kept, substring fallback when a group entry is blank)
    def file_harness(r):
        nn = r["n_eff"]
        out_ = {}
        for label, _g, _m, cols in GROUPS:
            if label == "IFEval":
                v = r["leaf"]["ifeval/prompt_level_strict_acc"]
                out_[label] = np.nan if v is None else v
                continue
            v = [r["leaf"][c] for c, _ in cols]
            w = [nn.get(t) for _, t in cols]
            if any(x is None for x in v) or any(not isinstance(x, int) for x in w):
                out_[label] = np.nan
                continue
            out_[label] = float(np.dot(v, w) / np.sum(w))
        return out_

    def same6(x, y):
        return all((np.isnan(x[a]) and np.isnan(y[a])) or
                   (not np.isnan(x[a]) and not np.isnan(y[a]) and
                    abs(x[a] - y[a]) <= EXACT_TOL) for a in AGG_NAMES)

    def as_nan(s):
        return {a: (np.nan if s is None or s.get(a) is None else float(s[a]))
                for a in AGG_NAMES}

    if reference_csv is not None:
        ref = pd.read_csv(reference_csv, index_col=0, float_precision="round_trip")
        # emulate the old script on the same raw files (listing order = path order)
        old_last = {}
        for r in recs:
            if r["old_scores"] is not None:
                old_last[r["model_name"]] = r
        emu = {n: as_nan(r["old_scores"]) for n, r in old_last.items()}
        emu_cells_equal = sum(
            1 for n in ref.index if n in emu
            for a in AGG_NAMES
            if (np.isnan(ref.loc[n, a]) and np.isnan(emu[n][a])) or
               (not np.isnan(ref.loc[n, a]) and ref.loc[n, a] == emu[n][a]))
        out = {"reference": "data/" + str(reference_csv.name),
               "reference_models": int(len(ref)),
               "reference_emulation": {
                   "models_emulated": len(emu),
                   "same_model_set": set(emu) == set(ref.index),
                   "cells_bitwise_equal": emu_cells_equal,
                   "cells_total": int(ref.size)},
               "common_models": None, "per_aggregate": {}}
        common = [n for n in model_names if n in ref.index]
        out["common_models"] = len(common)
        out["reference_models_not_in_olv2sub"] = sorted(set(ref.index) - set(model_names))
        refc = ref.loc[common]
        hc = harness.loc[common]
        any_bad = np.zeros(len(common), bool)
        for a in AGG_NAMES:
            d = np.abs(hc[a].values - refc[a].values)
            both = ~np.isnan(d)
            bad = both & (d > EXACT_TOL)
            any_bad |= bad | (np.isnan(refc[a].values) != np.isnan(hc[a].values))
            out["per_aggregate"][a] = {
                "n_compared": int(both.sum()),
                "n_reference_missing": int(np.isnan(refc[a].values).sum()),
                "n_agree_le_1e-9": int((both & (d <= EXACT_TOL)).sum()),
                "n_disagree": int(bad.sum()),
                "max_abs_diff": float(np.nanmax(d)) if both.any() else None,
                "median_abs_diff_of_disagreeing": float(np.median(d[bad])) if bad.any() else None,
            }
        def fallback_used(r):
            """Aggregates where the file's group entry is blank and the old
            script's substring fallback substituted a sub-task value."""
            return [a for a in AGG_NAMES
                    if r["direct_scores"][a] is None and r["old_scores"].get(a) is not None]

        def classify(n):
            r_old = old_last[n]
            if fallback_used(r_old):
                return "old_value_from_subtask_fallback"
            if not same6(as_nan(r_old["old_scores"]), as_nan(r_old["direct_scores"])):
                return "other"  # cannot happen if the emulation is faithful
            if not r_old["complete"]:
                return "old_file_partial_reupload"
            if r_old["rel"] != chosen[n]["rec"]["rel"]:
                return "different_complete_file_selected"
            if not same6(as_nan(r_old["direct_scores"]), file_harness(r_old)):
                return "source_group_value_ne_own_subtasks"
            return "other"

        cause = {k: [] for k in ["old_value_from_subtask_fallback",
                                 "old_file_partial_reupload",
                                 "different_complete_file_selected",
                                 "source_group_value_ne_own_subtasks", "other"]}
        for i, n in enumerate(common):
            if any_bad[i]:
                cause[classify(n)].append(n)
        out["models_all_six_agree"] = int((~any_bad).sum())
        out["models_with_any_disagreement"] = int(any_bad.sum())
        out["disagreement_causes_counts"] = {k: len(v) for k, v in cause.items()}
        out["disagreement_causes"] = cause
        fb = {n: fallback_used(r) for n, r in sorted(old_last.items())}
        out["reference_rows_with_subtask_fallback_values"] = {
            "n_rows": sum(bool(v) for v in fb.values()),
            "n_cells": sum(len(v) for v in fb.values()),
            "aggregates_affected": dict(sorted(pd.Series(
                [a for v in fb.values() for a in v], dtype=object).value_counts().items())),
            "rows": [n for n, v in fb.items() if v]}
        out["reference_rows_from_partial_reupload"] = sorted(
            n for n, r in old_last.items() if not r["complete"])
        out["note"] = (
            "Cause of each disagreement between data/open_llm.csv (written by "
            "download_open_llm.py, emulated above bit for bit) and olv2sub's harness "
            "aggregates, assigned in this order. old_value_from_subtask_fallback: the "
            "file kept by the old script has blank BBH/GPQA/MUSR group entries and its "
            "substring fallback stored the FIRST sub-task's value instead (wrong values "
            "in open_llm.csv). old_file_partial_reupload: the old script kept a partial "
            "BBH+MATH re-upload as the last file; the values it read are that file's "
            "group entries (MATH equals the leaderboard's), the other aggregates are NaN "
            "(incomplete rows, not misreads). different_complete_file_selected: the old "
            "script kept another complete file (last in path order), read correctly; "
            "olv2sub keeps the file matching the leaderboard contents table. "
            "source_group_value_ne_own_subtasks: same file, read correctly, but the "
            "file's own group value differs from the sample-weighted mean of its "
            "sub-tasks (open_llm.csv holds the group value, which is also what the "
            "leaderboard shows).")
        consistency["vs_open_llm_csv"] = out

    # ---- licence as declared on the Hub at the pinned revisions -------------
    lic_results = declared_license(
        json.loads((raw_dir / "api_results.json").read_text(encoding="utf-8")), "")
    lic_contents = declared_license(
        json.loads((raw_dir / "api_contents.json").read_text(encoding="utf-8")),
        (raw_dir / "contents" / "README.md").read_text(encoding="utf-8"))
    readme_listed = any(s.get("rfilename") == "README.md" for s in json.loads(
        (raw_dir / "api_results.json").read_text(encoding="utf-8")).get("siblings", []))

    def lic_text(lic, rev, card):
        declared = lic["license_tags"] or lic["card_license"]
        if declared:
            return f"declared on the HF Hub at {rev[:7]}: {declared}"
        return (f"none declared on the HF Hub at {rev[:7]} ({card}; API tags carry no "
                f"license:* tag). Access is governed by the Hugging Face Terms of "
                f"Service ({HF_TERMS}). Per-model licences of the evaluated models "
                f"(model weights, not these scores) are in models.hub_license")

    license_info = {
        "results": lic_text(lic_results, RESULTS_REV,
                            "no dataset card: the repo has no README.md"
                            if not readme_listed else "dataset card without a license key"),
        "contents": lic_text(lic_contents, CONTENTS_REV,
                             "dataset card front matter has keys "
                             + ", ".join(lic_contents["readme_front_matter_keys"])
                             + " and no license key"),
        "benchmark_items": ("no benchmark questions or model outputs are redistributed, "
                            "only per-task scores; the underlying benchmarks (IFEval, BBH, "
                            "MATH, GPQA, MuSR, MMLU-Pro) keep their own licences"),
    }

    # ---- meta.json -----------------------------------------------------------
    meta = {
        "name": NAME,
        "description": ("Open LLM Leaderboard v2 at sub-task level: raw lm-eval-harness "
                        "scores for the 42 leaf metrics behind the six v2 aggregates, one "
                        "row per leaderboard model."),
        "shape": [len(model_names), len(COLUMN_NAMES)],
        "row_order": "model fullname (HF repo id), Python sorted() / Unicode code-point order",
        "column_order": ("groups in leaderboard order IFEval, BBH, MATH_Lvl5, GPQA, MUSR, "
                         "MMLU-PRO; sub-tasks alphabetical within group; IFEval metrics in "
                         "the order " + ", ".join(IFEVAL_METRICS)),
        "score_definition": {
            "scale": "raw metric in [0,1] from lm-eval-harness results JSON (NOT the "
                     "leaderboard's baseline-normalised 0-100 score)",
            "ifeval/*": "leaderboard_ifeval <metric>,none; prompt-level metrics average "
                        "over 541 prompts, inst-level over the instructions they contain",
            "bbh/*": "leaderboard_bbh_<task> acc_norm,none (3-shot, multiple choice)",
            "math_hard/*": "leaderboard_math_<task>_hard exact_match,none (4-shot, "
                           "generative); value from the selected file, i.e. after the "
                           "leaderboard's MATH re-scoring when the selected file is a "
                           "2024-10-24 / 2025-02-13 re-upload. Scorer versions are MIXED "
                           "exactly as on the leaderboard: v3.0 = 2025-02-13 bulk re-scoring "
                           "(consistent with HF's Feb-2025 Math-Verify fix), v2.0 = 2024-10-24 "
                           "fix or runs after 2025-02-13 that were not re-scored, v1.0 = "
                           "original 2024 scorer; see models.math_hard_task_version, "
                           "counts.math_hard_task_version_counts and "
                           "counts.math_hard_aggregate_shift_between_scorer_versions. "
                           "Temporal splits at >= 2025-02-13 put v2.0 models on the test side",
            "gpqa/*": "leaderboard_gpqa_<task> acc_norm,none (0-shot)",
            "musr/*": "leaderboard_musr_<task> acc_norm,none (0-shot)",
            "mmlu_pro": "leaderboard_mmlu_pro acc,none (5-shot)",
            "harness_group_aggregate": "sample-weighted mean of sub-tasks "
                                       "(lm-eval weight_by_size); IFEval group value = "
                                       "prompt_level_strict_acc (as in data/open_llm.csv)",
            "leaderboard_raw_aggregate": {
                "IFEval": "(prompt_level_strict_acc + inst_level_strict_acc)/2",
                "BBH": "unweighted mean of 24 sub-tasks",
                "MATH_Lvl5": "sample-weighted mean of 7 sub-tasks",
                "GPQA": "sample-weighted mean of 3 sub-tasks",
                "MUSR": "unweighted mean of 3 sub-tasks",
                "MMLU-PRO": "acc",
            },
            "missing": "NaN in CSV, B=0 and O=0 in npz",
            "models.params_b": "contents '#Params (B)'; the sentinel -1 is mapped to null "
                               "(models.model_num_parameters from the result JSON is exact)",
        },
        "selection_rules": {
            "row_per_contents_entry": ("best result file = min L_inf between the six "
                                       "leaderboard-raw aggregates recomputed from the "
                                       "file's sub-tasks and the contents row; ties "
                                       f"(<= {TIE_TOL}) -> latest file timestamp, then path"),
            "tie_sensitivity": ("where several eligible files tie, their leaf values can "
                                "still differ (MATH-hard sub-tasks re-scored with identical "
                                "totals, IFEval loose metrics); those rows depend on the "
                                "tie-break, not only on the contents table: "
                                "models.match_tied_leaf_differs / "
                                "match_tied_max_abs_leaf_diff, counts.ties_* "
                                "(ties_differing_cells_by_group counts (model, column) "
                                "pairs where any tied file differs from the kept one)"),
            "eligible_files": "same model_name; all six aggregates computable; no sample limit",
            "row_per_model": ("models listed under two precisions keep the contents row "
                              "with the later Submission Date, then bfloat16, then "
                              "eval_name; the other row's eval_name is recorded"),
            "dropped": "contents rows with Flagged=True; result-only models absent from contents",
        },
        "cost": {
            "per_benchmark": "benchmarks.n_samples = effective n-samples of the harness task "
                             "(items evaluated per model). The four ifeval/* columns share one "
                             "run (cost_group) of 541 prompts. Generative tasks (IFEval, MATH) "
                             "cost more per item than log-likelihood multiple-choice tasks.",
            "shared_items": (f"benchmarks.item_group: columns with the same value draw on one "
                             f"question pool. The three gpqa/* columns are separate harness "
                             f"runs (separate cost_group, so their costs add as run on the "
                             f"leaderboard) over NESTED subsets of the same questions: "
                             f"{GPQA_NESTING} (n = {gpqa_n['gpqa/diamond']}, "
                             f"{gpqa_n['gpqa/main']}, {gpqa_n['gpqa/extended']}; Rein et al. "
                             f"2023, arXiv:2311.12022), so they are correlated by construction "
                             f"and a model scored on gpqa/extended has already been asked "
                             f"every diamond/main question (in a separate run, so per-item "
                             f"results need not coincide). The four ifeval/* columns "
                             f"are four metrics of one run over the same 541 prompts."),
            "item_evaluations_per_model": int(sum(runs.values())),
            "distinct_items_per_model": int(sum(pools.values())),
            "totals_note": ("item_evaluations_per_model = sum of n_samples over distinct "
                            "cost_group (IFEval counted once, GPQA 198+448+546); "
                            "distinct_items_per_model = sum over item_group of the largest "
                            "subset (IFEval 541 once, GPQA 546)"),
            "per_model": "models.total_evaluation_time_seconds = wall-clock time of the "
                         "selected file's whole leaderboard run (all tasks); bulk "
                         "re-uploads carry the original run's time",
        },
        "counts": stats,
        "benchmark_names": COLUMN_NAMES,
        "benchmarks": bench_meta,
        "model_names": model_names,
        "models": mm,
        "contents_raw_aggregates": contents_raw,
        "consistency_checks": consistency,
        "license": license_info,
        "sources": {
            "results": {"repo": RESULTS_REPO, "url": f"{HF}/{RESULTS_REPO}",
                        "revision": RESULTS_REV,
                        "last_modified": RESULTS_LAST_MODIFIED,
                        "fetched_at": fetch["results_git"]["fetched_at"],
                        "method": fetch["results_git"]["method"],
                        "n_files": len(recs),
                        "tree_sha256_of_sha256sum_listing": RESULTS_TREE_SHA256,
                        "license": license_info["results"],
                        "declared_license_evidence": lic_results},
            "contents": {"repo": CONTENTS_REPO,
                         "url": f"{HF}/{CONTENTS_REPO}/resolve/{CONTENTS_REV}/{CONTENTS_FILE}",
                         "revision": CONTENTS_REV,
                         "last_modified": CONTENTS_LAST_MODIFIED,
                         "sha256": CONTENTS_SHA256,
                         "fetched_at": fetch["contents/train-00000-of-00001.parquet"]["fetched_at"],
                         "license": license_info["contents"],
                         "declared_license_evidence": lic_contents},
            "terms_of_service": HF_TERMS,
            "manifest": f"raw_manifests/{NAME}.json",
            "builder": "data/build_olv2sub.py",
        },
    }
    meta_path = output_dir / f"{NAME}.meta.json"
    meta_path.write_text(to_json(meta) + "\n", encoding="utf-8")
    print(f"  wrote {meta_path}")

    # ---- raw manifest ----------------------------------------------------------
    # tree_listing order is plain byte order == `LC_ALL=C sort`, so the digest equals
    # `find . -type f ! -path './.git/*' | cut -c3- | LC_ALL=C sort | xargs sha256sum | sha256sum`
    files = tree_files
    inputs = [{
        "local_path": "results_git/",
        "url": f"{HF}/{RESULTS_REPO}",
        "file_url_template": f"{HF}/{RESULTS_REPO}/resolve/{RESULTS_REV}/{{path}}",
        "revision": RESULTS_REV,
        "method": fetch["results_git"]["method"],
        "fetched_at": fetch["results_git"]["fetched_at"],
        "license": license_info["results"],
        "n_files": len(files),
        "size": sum(f["size"] for f in files),
        "sha256_of_sha256sum_listing": tree_digest,
        "files": files,
    }]
    for rel, (url, rev) in RAW_FILES.items():
        p = raw_dir / rel
        inputs.append({"local_path": rel, "url": url, "revision": rev,
                       "method": fetch[rel]["method"],
                       "fetched_at": fetch[rel]["fetched_at"],
                       "size": p.stat().st_size, "sha256": sha256_file(p)})
    if fetch_path.exists():
        inputs.append({"local_path": "fetch_info.json", "url": None, "revision": None,
                       "method": "written by the person fetching (not downloaded): "
                                 "provenance only, read for fetched_at / method",
                       "fetched_at": None, "size": fetch_path.stat().st_size,
                       "sha256": sha256_file(fetch_path)})
    manifest = {"name": NAME,
                "raw_root": "the builder's --raw-dir; layout and fetch recipe in the "
                            "docstring of data/build_olv2sub.py",
                "license": license_info,
                "inputs": inputs}
    mpath = manifest_dir / f"{NAME}.json"
    mpath.write_text(to_json(manifest) + "\n", encoding="utf-8")
    print(f"  wrote {mpath}")

    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, list)}, indent=1))
    print(json.dumps(consistency, indent=1)[:4000])


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="Directory with results_git/, contents/, api_*.json "
                         "(and optionally fetch_info.json)")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                    help="Where to write olv2sub.matrix.{npz,csv} and olv2sub.meta.json")
    ap.add_argument("--manifest-dir", type=Path, default=None,
                    help="Where to write olv2sub.json (default: <output-dir>/raw_manifests)")
    ap.add_argument("--reference-csv", type=Path, default=None,
                    help="data/open_llm.csv for the 6-aggregate consistency check")
    a = ap.parse_args()
    if a.manifest_dir is None:
        a.manifest_dir = a.output_dir / "raw_manifests"
    return a


if __name__ == "__main__":
    args = parse_args()
    build(args.raw_dir, args.output_dir, args.manifest_dir, args.reference_csv)
