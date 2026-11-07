#!/usr/bin/env python3
"""Build HELM score matrices with per-(model, scenario) evaluation-cost metadata.

One builder for three HELM leaderboards, selected with ``--suite``:

    lite     -> helm_lite     HELM Lite v1.13.0, all models x the 10 core
                              scenarios (same main metric and column names as
                              data/download_helm.py / data/helm.csv, which hold
                              the Lite v1.0.0 snapshot)
    mmlu     -> helm_mmlu     HELM MMLU v1.13.0, all models x the 57 subjects
                              (exact match; columns named like data/mmlu.matrix.npz)
    classic  -> helm_classic  HELM Classic v0.4.0, all models x the 16 core
                              scenarios (one main metric per scenario)

Produces in --output-dir (default: this directory):
    <name>.matrix.csv / <name>.matrix.npz   via build_matrices.save_matrix
                                            (B, O, model_names, benchmark_names)
    <name>.meta.json                        models, benchmarks, costs, provenance,
                                            consistency checks
    raw_manifests/<name>.json               every raw input with URL, pinned
                                            revision, size, sha256, fetched_at

Sources (all public, no credentials)
------------------------------------
* HELM release JSON from the public GCS bucket ``crfm-helm-public``:
  ``<project>/benchmark_output/releases/<release>/{summary,schema,groups,
  groups_metadata,run_specs,runs_to_run_suites,costs,runs}.json`` and
  ``.../groups/*.json``.  Objects are stored gzip-encoded; we keep the stored
  bytes (``*.json.gz``), check them against the GCS md5, and pin each one by its
  GCS object generation (the generation is part of the recorded URL).
  ``summary.json`` of every release of the project gives the release dates of
  the evaluation suites named in ``runs_to_run_suites.json``.
* Model release dates: ``src/helm/config/model_metadata.yaml`` of
  github.com/stanford-crfm/helm at commit HELM_COMMIT (raw.githubusercontent,
  anonymous).  Models missing there (old Classic ids such as together/bloom)
  are matched by display name, else the release's schema.json date is used;
  ``release_date_source`` records which.  A few upstream dates contradict the
  date encoded in the model id (e.g. mistral-large-2402 dated 2023-02-26); they
  are corrected via RELEASE_DATE_OVERRIDES (``release_date_upstream`` keeps the
  original) and every id-encoded date is checked (meta consistency
  ``release_date_vs_model_id``).

Scores
------
lite / mmlu: B[m, b] is the value HELM shows in the release's leaderboard
table (group ``core_scenarios`` for lite, ``mmlu_subjects`` for mmlu), table
"Accuracy": the scenario's main metric (schema run_group environment
``main_name`` on ``main_split``), macro-averaged by HELM over the scenario's
sub-runs (e.g. 5 LegalBench subsets, 5 WMT language pairs, 7 MATH subjects).
Every cell is re-derived from runs.json (consistency ``runs_json_cross_check``).

classic: B[m, b] is the main metric of the cell's few-shot leaderboard runs in
runs.json (the ``data_augmentation=canonical`` runs, without HELM's ablation
runs ``groups=ablation_*``, contrast sets, zero-shot ``max_train_instances=0``
variants and the duplicate ``device=cpu/cuda`` summarization runs), merged as
HELM merges stats (sum / count over the trial entries, i.e. a mean over the
sub-runs -- 5 MMLU subjects, 9 CivilComments demographics, 11 RAFT subsets --
and the in-context-example trials; sub_split entries such as CivilComments'
toxic / non-toxic are merged the same way).  This equals HELM's leaderboard
value except where HELM's table also pools a zero-shot ablation run into the
cell (TruthfulQA for 11 models, CNN/DailyMail and XSUM for 4); those cells are
listed in meta ``leaderboard_differences`` with HELM's value, and
``cell_costs.leaderboard_value`` holds it per cell.  Where the two agree
(|diff| <= SCORE_TOL) B keeps HELM's displayed value bit for bit.
Aggregate columns ("Mean win rate", MMLU "All Subjects") are not benchmarks;
they are kept per model in the meta file.

Costs
-----
Per (model, scenario), one evaluation pass (one in-context-example trial).
lite / mmlu: from the scenario's own group file, summing over its leaf tables
(one per sub-run; the "only_contrast:True" robustness tables are excluded).
classic: from runs.json, summing over the same runs as the score and over their
sub_splits (Classic's leaf tables pool the zero-shot runs above and average
CivilComments' sub_splits without instance weights, so they are only a
cross-check here).
    instances               sum of "# eval" / num_instances (evaluation instances)
    prompt_tokens_mean      instance-weighted mean of "# prompt tokens"
    completion_tokens_mean  instance-weighted mean of "# output tokens"
    cost_tokens             sum of instances x (prompt tokens + output tokens)
    inference_time_s_mean   instance-weighted mean of the efficiency metric
                            ("Observed inference time (s)" for lite/mmlu,
                            "Denoised inference time (s)" for classic)
    inference_time_s_total  sum of instances x time
    num_train_trials        "# trials" / num_train_trials (classic: 3 or 1)
    num_requests            HELM's own request count, summed over the cell's
                            runs (runs.json stat ``num_requests``: all splits,
                            trials and perturbations of those runs)
Per benchmark (aligned with benchmark_names): the mean over models with an
observed cell of each quantity; ``cost_tokens_mean`` is the cost vector for the
budget experiments.  Token counts use each model's own tokenizer.  The leaf
sums are checked against the "General information" table of the leaderboard
and against runs.json (see meta "consistency").

Order and determinism
---------------------
Rows are the HELM display names (unique per release) sorted by Python ``str``
order (Unicode code points); columns are the benchmark names sorted the same
way.  The build reads only the raw files (verified against the manifest's
sha256), so re-running it on the same raw inputs gives identical B/O/name
arrays, an identical CSV and an identical meta JSON.

Usage (run with ``python -I`` from a directory without downloaded files):
    python -I build_helm.py --suite lite --fetch --raw-dir ~/benchselect-data/raw/helm_lite
    python -I build_helm.py --suite lite --raw-dir RAW --output-dir OUT \\
        [--snapshot-csv data/helm.csv]
    # re-fetch exactly the pinned objects of an existing manifest:
    python -I build_helm.py --suite lite --fetch --pin-manifest raw_manifests/helm_lite.json \\
        --raw-dir RAW

Requirements: numpy, pandas, requests, pyyaml (requests only for --fetch).
"""
from __future__ import annotations

import argparse
import base64
import collections
import datetime as _dt
import gzip
import hashlib
import importlib.util
import json
import math
import re
import time
import urllib.parse
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load_save_matrix():
    """Load save_matrix from build_matrices.py next to this file by explicit path, without
    putting this directory on sys.path (so ``python -I`` isolation holds for all imports)."""
    spec = importlib.util.spec_from_file_location("build_matrices", HERE / "build_matrices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.save_matrix


save_matrix = _load_save_matrix()

GCS_BUCKET = "crfm-helm-public"
GCS_LIST = f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o"
GCS_MEDIA = f"https://storage.googleapis.com/download/storage/v1/b/{GCS_BUCKET}/o"
HELM_REPO = "stanford-crfm/helm"
HELM_COMMIT = "63754d05db6f874e41a395880fb573890a13e791"  # main HEAD on 2026-10-17 (commit of 2026-06-05)
MODEL_METADATA = "src/helm/config/model_metadata.yaml"

SUITES = {
    "lite": {
        "name": "helm_lite", "project": "lite", "release": "v1.13.0",
        "core_group": "core_scenarios", "snapshot_release": "v1.0.0",
        "title": "HELM Lite v1.13.0 (core scenarios)",
        "caveats": [
            "'Mean win rate' (column 11 of data/helm.csv) is not a benchmark and depends on the "
            "model set, so it is not a column here; it is kept as models[*].mean_win_rate.",
            "Release dates of Mistral Large/Small 2402, Mistral Large 2407 and GPT-4o (2024-05-13) "
            "are corrected from upstream typos (2023-02-26, 2023-07-24, 2024-04-09) to the date "
            "in the model id (RELEASE_DATE_OVERRIDES; models[*].release_date_upstream).",
            "HELM renamed 6 of the 30 snapshot models (e.g. 'Anthropic Claude 2.0' -> 'Claude 2.0', "
            "'Cohere Command' -> 'Command'); models[*].snapshot_display_name gives the name used "
            "in data/helm.csv and data/canonical_mapping.json.",
            "One missing cell: Phi-3 (7B) has no GSM8K run.",
            "Instances differ by model for LegalBench (1562 instead of 2047 for Mistral Large/Small "
            "2402) and WMT 2014 (Jamba Instruct lacks fr-en: 2341 instead of 2844).",
            "Token counts use each model's own tokenizer; the per-benchmark cost vector averages "
            "over models.",
        ],
    },
    "mmlu": {
        "name": "helm_mmlu", "project": "mmlu", "release": "v1.13.0",
        "core_group": "mmlu_subjects", "aggregate_groups": ["mmlu"],
        "strip_prefix": "mmlu_",
        "title": "HELM MMLU v1.13.0 (57 subjects, exact match)",
        "caveats": [
            "HELM's own 5-shot multiple_choice_joint MMLU on the full test split (14,042 "
            "questions), not the Open LLM Leaderboard harness behind data/mmlu.matrix.npz; "
            "subject names match that matrix.",
            "The 'MMLU All Subjects' aggregate is kept as models[*].mmlu_score, not as a column.",
            "Release dates of Mistral Large/Small 2402, Mistral Large 2407, GPT-4o (2024-05-13) "
            "and gpt-3.5-turbo-0125 are corrected from upstream typos (2023-02-26, 2023-07-24, "
            "2024-04-09, 2023-06-13) to the date in the model id (RELEASE_DATE_OVERRIDES; "
            "models[*].release_date_upstream).",
            "Token counts use each model's own tokenizer; the per-benchmark cost vector averages "
            "over models.",
        ],
    },
    "classic": {
        "name": "helm_classic", "project": "classic", "release": "v0.4.0",
        "core_group": "core_scenarios",
        "title": "HELM Classic v0.4.0 (core scenarios)",
        "min_models": 50, "min_benchmarks": 15,
        "scores_from_runs": True, "costs_from_runs": True,
        "caveats": [
            "Scores and costs come from the few-shot leaderboard runs in runs.json. In 19 cells "
            "HELM's leaderboard table also pools a zero-shot ablation run (max_train_instances=0) "
            "into the cell: TruthfulQA for Anthropic-LM v4-s3, BLOOM, GLM, GPT-J, GPT-NeoX, "
            "OPT 175B, OPT 66B, T0pp, T5, UL2, YaLM, and CNN/DailyMail + XSUM for curie, "
            "davinci, text-curie-001, text-davinci-002. There B is the few-shot value and differs "
            "from the HELM website by up to 0.0135; HELM's value is kept in "
            "leaderboard_differences and cell_costs.leaderboard_value.",
            "Scores are means over 3 in-context-example trials for most cells (1 trial for "
            "HellaSwag, OpenbookQA and 25 of the 30 models added in 2023); "
            "cell_costs.num_train_trials. Costs are for one trial.",
            "Fill is 81.7%: HellaSwag, OpenbookQA (32 models), MS MARCO (30/31) and the two "
            "summarization scenarios (42) were not run for every model.",
            "Anthropic-LM v4-s3 (52B) has complete 3-trial MS MARCO (regular) and (TREC) runs in "
            "runs.json (RR@10 0.382, NDCG@10 0.642), but HELM's leaderboard shows no value for "
            "them; they are left missing as published and listed in "
            "consistency.runs_json_values_missing_from_leaderboard.",
            "HELM marks 58 cells as contaminated (54 weak, 4 strong, e.g. T0pp); scores are kept "
            "and flagged in cell_costs.contamination_level.",
            "CivilComments: HELM's tables average the toxic/non-toxic sub-splits without weights "
            "('# eval' is half the instance count); here instances, token means and times are "
            "instance-weighted over the sub-splits from runs.json (cell_costs.instances_table "
            "keeps the table's '# eval').",
            "HellaSwag (multiple_choice_separate_original), OpenbookQA (_calibrated) and MS MARCO "
            "(ranking_binary) send several requests per instance (4, 8, 30.5 and 224.6); "
            "cost_tokens counts one request per instance as specified, "
            "benchmark_cost.cost_tokens_requests_mean multiplies by requests_per_instance.",
            "OpenbookQA's reported '# prompt tokens' (~5) appears to cover only the calibration "
            "requests, so its token cost is underestimated even after the request correction.",
            "num_requests (runs.json) is reported only for the 37 original 2022 models; the 30 "
            "models added in 2023 (LLaMA, Llama 2, Falcon, Jurassic-2, gpt-3.5-turbo, ...) lack "
            "the stat. It counts all splits, trials and perturbed copies of a run.",
            "Ten old model ids (together/bloom, together/opt-175b, ...) are missing from "
            "model_metadata.yaml and were matched by display name (release_date_source).",
        ],
    },
}
SNAPSHOT_SKIP = {"runs.json"}  # the snapshot release is only used for score comparison
EXCLUDED_LEAF = "only_contrast:True"
CONTAMINATION_MARKS = "\u26a0\u2620"  # per-scenario tables suffix model names with these
GI = {"# eval": "eval", "# prompt tokens": "prompt", "# output tokens": "output",
      "# train": "train", "truncated": "truncated", "# trials": "trials"}
# runs.json stat names of the same quantities (schema.json metrics with those display names)
RUN_STATS = {"eval": "num_instances", "prompt": "num_prompt_tokens", "output": "num_output_tokens",
             "train": "num_train_instances", "truncated": "prompt_truncated",
             "trials": "num_train_trials"}
SCORE_TOL = 1e-9    # runs.json vs leaderboard value; real disagreements are >= 3e-4
COST_FIELDS = ["instances", "prompt_tokens_mean", "completion_tokens_mean", "cost_tokens",
               "inference_time_s_mean", "inference_time_s_total", "num_train_trials"]
LICENSE = ("HELM code is Apache-2.0 (github.com/stanford-crfm/helm); the benchmark_output "
           "JSON states no separate data license")
COMMON_CAVEATS = [
    "Data license unresolved: HELM's code is Apache-2.0, but the benchmark_output JSON states "
    "no terms for redistributing the score tables; check before the data release.",
    "Raw inputs are pinned by GCS object generation, but the bucket keeps only the current "
    "generation (HELM rewrote all release files in 2026). After HELM's next rewrite "
    "'--fetch --pin-manifest' will fail; the sha256 values (source.raw_inputs, "
    "raw_manifests/<name>.json) then only verify archived copies, so the raw directory "
    "(~92 MB for the three HELM matrices) should be archived with the data release.",
]
# Upstream release dates that contradict the date encoded in the model id: model id ->
# (upstream date in model_metadata.yaml@HELM_COMMIT and the release schema.json, corrected
# date, reason).  Applied only while the upstream value is unchanged.
RELEASE_DATE_OVERRIDES = {
    "mistralai/mistral-large-2402": (
        "2023-02-26", "2024-02-26", "version 2402 = 2024-02; Mistral Large released 2024-02-26"),
    "mistralai/mistral-small-2402": (
        "2023-02-26", "2024-02-26", "version 2402 = 2024-02; released with Mistral Large 2024-02-26"),
    "mistralai/mistral-large-2407": (
        "2023-07-24", "2024-07-24", "version 2407 = 2024-07; Mistral Large 2 released 2024-07-24"),
    "openai/gpt-4o-2024-05-13": (
        "2024-04-09", "2024-05-13", "snapshot date 2024-05-13 in the id; upstream repeats the "
        "gpt-4-turbo-2024-04-09 date"),
    "openai/gpt-3.5-turbo-0125": (
        "2023-06-13", "2024-01-25", "snapshot 0125 = 2024-01-25; upstream repeats the "
        "gpt-3.5-turbo-0613 date"),
}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_path(root, rel_path):
    """root / rel_path for a relative POSIX path without '..', '.', empty or absolute parts."""
    p = PurePosixPath(rel_path)
    if (not rel_path or "\\" in rel_path or "\0" in rel_path or p.is_absolute()
            or any(part in ("", ".", "..") for part in rel_path.split("/"))):
        raise ValueError(f"unsafe raw path {rel_path!r}")
    out = root.joinpath(*p.parts)
    if root.resolve() not in out.resolve().parents:
        raise ValueError(f"raw path {rel_path!r} escapes {root}")
    return out


def _gcs_list(prefix):
    import requests
    items, prefixes, token = [], [], None
    while True:
        params = {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = requests.get(GCS_LIST, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        items += d.get("items", [])
        prefixes += d.get("prefixes", [])
        token = d.get("nextPageToken")
        time.sleep(0.2)
        if not token:
            return items, prefixes


def _get_stored(url):
    """GET the stored (possibly gzip-encoded) bytes, without transcoding."""
    import requests
    with requests.get(url, headers={"Accept-Encoding": "gzip"}, stream=True, timeout=300) as r:
        r.raise_for_status()
        return r.raw.read(decode_content=False)


def _fetch_object(obj, raw_dir, rel_path):
    url = (f"{GCS_MEDIA}/{urllib.parse.quote(obj['name'], safe='')}"
           f"?alt=media&generation={obj['generation']}")
    data = _get_stored(url)
    if base64.b64encode(hashlib.md5(data).digest()).decode() != obj["md5Hash"]:
        raise RuntimeError(f"md5 mismatch for {obj['name']}")
    enc = obj.get("contentEncoding")
    decoded = gzip.decompress(data) if enc == "gzip" else data
    if enc == "gzip":
        rel_path += ".gz"
    out = _safe_path(raw_dir, rel_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"  {rel_path}: {len(data):,} B")
    time.sleep(0.1)
    return {
        "path": rel_path, "url": url,
        "revision": f"gcs-generation:{obj['generation']}",
        "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "content_encoding": enc, "decoded_size": len(decoded),
        "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
        "gcs_md5": obj["md5Hash"], "gcs_crc32c": obj["crc32c"],
        "gcs_updated": obj["updated"], "fetched_at": _now(),
    }


def fetch(suite, raw_dir, pin_manifest=None):
    """Download every raw input of ``suite`` into raw_dir and write _manifest.json."""
    import requests
    cfg = SUITES[suite]
    raw_dir = raw_dir.expanduser()
    raw_dir.mkdir(parents=True, exist_ok=True)
    if pin_manifest is not None:  # re-fetch the pinned objects and verify sha256
        man = json.loads(Path(pin_manifest).read_text())
        inputs = []
        for e in man["inputs"]:
            out = _safe_path(raw_dir, e["path"])
            if not (e["url"].startswith(GCS_MEDIA + "/")
                    or e["url"].startswith(f"https://raw.githubusercontent.com/{HELM_REPO}/")):
                raise ValueError(f"unexpected pinned URL {e['url']!r}")
            data = _get_stored(e["url"]) if e["url"].startswith(GCS_MEDIA) else \
                requests.get(e["url"], timeout=60).content
            if hashlib.sha256(data).hexdigest() != e["sha256"]:
                raise RuntimeError(f"sha256 mismatch for pinned {e['path']}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            # this copy's own download time; the pinned manifest's time is kept alongside
            inputs.append({**e, "fetched_at": _now(),
                           "first_fetched_at": e.get("first_fetched_at", e["fetched_at"])})
            print(f"  {e['path']}: ok")
            time.sleep(0.1)
        man = {**man, "inputs": inputs, "local_root": str(raw_dir).replace(str(Path.home()), "~", 1)}
        (raw_dir / "_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        return
    rel_root = f"{cfg['project']}/benchmark_output/releases/"
    entries = []
    _, release_prefixes = _gcs_list(rel_root)
    releases = sorted(p[len(rel_root):].strip("/") for p in release_prefixes)
    full = [cfg["release"]] + ([cfg["snapshot_release"]] if "snapshot_release" in cfg else [])
    for rel in full:
        base = rel_root + rel + "/"
        root_items, _ = _gcs_list(base)
        group_items, _ = _gcs_list(base + "groups/")
        skip = SNAPSHOT_SKIP if rel != cfg["release"] else set()
        objs = [o for o in root_items if o["name"].rsplit("/", 1)[-1] not in skip] + group_items
        for obj in sorted(objs, key=lambda o: o["name"]):
            entries.append(_fetch_object(obj, raw_dir, obj["name"][len(rel_root):]))
    for rel in releases:  # release dates of all evaluation suites
        if rel in full:
            continue
        items, _ = _gcs_list(rel_root + rel + "/summary.json")
        for obj in items:
            if obj["name"] == rel_root + rel + "/summary.json":
                entries.append(_fetch_object(obj, raw_dir, obj["name"][len(rel_root):]))
    url = f"https://raw.githubusercontent.com/{HELM_REPO}/{HELM_COMMIT}/{MODEL_METADATA}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out = raw_dir / "github" / "model_metadata.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    entries.append({"path": "github/model_metadata.yaml", "url": url,
                    "revision": f"git:{HELM_COMMIT}", "size": len(r.content),
                    "sha256": hashlib.sha256(r.content).hexdigest(), "fetched_at": _now()})
    man = {
        "name": cfg["name"],
        "description": f"{cfg['title']}: HELM release JSON (gs://{GCS_BUCKET}/{rel_root}"
                       f"{cfg['release']}/) and model_metadata.yaml at {HELM_REPO}@{HELM_COMMIT}",
        "license": LICENSE,
        "local_root": str(raw_dir).replace(str(Path.home()), "~", 1),
        "inputs": sorted(entries, key=lambda e: e["path"]),
    }
    (raw_dir / "_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Raw access
# ---------------------------------------------------------------------------

class Raw:
    """Read-only access to manifest-verified raw files (JSON/YAML parsed as data)."""

    def __init__(self, raw_dir):
        self.dir = raw_dir.expanduser()
        self.manifest = json.loads((self.dir / "_manifest.json").read_text())
        self.files = {e["path"]: e for e in self.manifest["inputs"]}
        for path, e in self.files.items():
            if hashlib.sha256(_safe_path(self.dir, path).read_bytes()).hexdigest() != e["sha256"]:
                raise RuntimeError(f"raw file {path} does not match its manifest sha256")

    def has(self, path):
        return path in self.files

    def json(self, path):
        if path + ".gz" in self.files:
            path += ".gz"
        if path not in self.files:
            raise KeyError(f"{path} is not in the raw manifest")
        data = _safe_path(self.dir, path).read_bytes()
        if path.endswith(".gz"):
            data = gzip.decompress(data)
        return json.loads(data)

    def yaml(self, path):
        import yaml
        if path not in self.files:
            raise KeyError(f"{path} is not in the raw manifest")
        return yaml.safe_load(_safe_path(self.dir, path).read_text())


def _num(cell):
    v = cell.get("value") if isinstance(cell, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def _count(cell):
    m = re.search(r"\((\d+)\)\s*$", cell.get("description") or "")
    return int(m.group(1)) if m else None


def _table(tables, name):
    hits = [t for t in tables if t.get("name") == name]
    if len(hits) != 1:
        raise KeyError(f"table {name!r}: {len(hits)} matches")
    return hits[0]


def _rows(table):
    out = {}
    for row in table["rows"]:
        name = row[0]["value"].rstrip(CONTAMINATION_MARKS)
        if name in out:
            raise ValueError(f"duplicate model row {name!r} in table {table.get('name')}")
        out[name] = row
    return out


def _version_key(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def _iso(d):
    if d is None:
        return None
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _resolve_columns(cfg, schema, acc):
    """Map accuracy-table columns to HELM run groups below the leaderboard group."""
    groups = {g["name"]: g for g in schema["run_groups"]}
    below, todo = set(), [cfg["core_group"]]
    while todo:
        for s in groups[todo.pop()].get("subgroups", []):
            if s not in below:
                below.add(s)
                todo.append(s)
    cols = []
    for j, h in enumerate(acc["header"]):
        md = h.get("metadata") or {}
        if j == 0 or "run_group" not in md:
            continue
        hits = [g for g in below if groups[g].get("environment", {}).get("main_name")
                and md["run_group"] in (groups[g].get("display_name"),
                                        groups[g].get("short_display_name"))]
        if len(hits) != 1:
            raise ValueError(f"column {h['value']!r}: run groups {hits}")
        g = groups[hits[0]]
        cols.append({"col": j, "header": h["value"], "group": g["name"],
                     "display_name": md["run_group"], "metric": md["metric"],
                     "main_name": g["environment"]["main_name"],
                     "main_split": g["environment"]["main_split"],
                     "lower_is_better": bool(h.get("lower_is_better", False)),
                     "description": g.get("description")})
    return cols


def _leaf_stats(raw, rel, group, metric, time_metric):
    """Per-model leaf-table values for one scenario: {model: [leaf dicts]}."""
    tables = raw.json(f"{rel}/groups/{group}.json")
    leaves = tables if len(tables) == 1 else [
        t for t in tables if t.get("name") != group and EXCLUDED_LEAF not in t.get("name", "")]
    per_model = collections.defaultdict(list)
    for t in sorted(leaves, key=lambda t: t["name"]):
        hdr = [h["value"] for h in t["header"]]
        idx = {k: hdr.index(k) for k in list(GI) + [metric, time_metric] if k in hdr}
        for model, row in _rows(t).items():
            if "# eval" not in idx or _num(row[idx["# eval"]]) is None:
                continue
            d = {"leaf": t["name"]}
            for k, short in GI.items():
                if k in idx:
                    d[short] = _num(row[idx[k]])
                    d["n_" + short] = _count(row[idx[k]])
            if metric in idx:
                d["score"], d["n_score"] = _num(row[idx[metric]]), _count(row[idx[metric]])
            if time_metric in idx:
                d["time"] = _num(row[idx[time_metric]])
            per_model[model].append(d)
    return [t["name"] for t in sorted(leaves, key=lambda t: t["name"])], per_model


def _cell_cost(leaves):
    ev = sum(l["eval"] for l in leaves)
    if not ev:
        return None
    out = {"instances": ev, "n_leaf_runs": len(leaves)}
    if all(l.get("prompt") is not None and l.get("output") is not None for l in leaves):
        out["prompt_tokens_mean"] = sum(l["eval"] * l["prompt"] for l in leaves) / ev
        out["completion_tokens_mean"] = sum(l["eval"] * l["output"] for l in leaves) / ev
        out["cost_tokens"] = sum(l["eval"] * (l["prompt"] + l["output"]) for l in leaves)
    if all(l.get("time") is not None for l in leaves):
        out["inference_time_s_total"] = sum(l["eval"] * l["time"] for l in leaves)
        out["inference_time_s_mean"] = out["inference_time_s_total"] / ev
    if all(l.get("train") is not None for l in leaves):
        out["num_train_mean"] = sum(l["eval"] * l["train"] for l in leaves) / ev
    if all(l.get("truncated") is not None for l in leaves):
        out["truncated_frac"] = sum(l["eval"] * l["truncated"] for l in leaves) / ev
    if all(l.get("trials") is not None for l in leaves):
        out["num_train_trials"] = sum(l["eval"] * l["trials"] for l in leaves) / ev
    return out


def _run_stats(runs):
    """Per run from runs.json: {"num_requests": total requests (if reported),
    ("num_instances", split): unperturbed instances per trial,
    "instance_variants": instances per trial summed over splits and perturbed copies}."""
    out = {}
    for r in runs:
        agg, sub, d = {}, collections.defaultdict(float), {}
        for s in r["stats"]:
            n = s["name"]
            p = n.get("perturbation")
            if n["name"] == "num_requests" and p is None:
                d["num_requests"] = s["sum"]
            if n["name"] != "num_instances" or (p is not None and p.get("computed_on") != "perturbed"):
                continue
            key = (n.get("split"), p["name"] if p else None)
            if n.get("sub_split"):
                sub[key] += s["mean"]
            else:
                agg[key] = s["mean"]
        for k, v in sub.items():  # some runs report instances only per sub_split
            agg.setdefault(k, v)
        for (split, pert), v in agg.items():
            if pert is None:
                d[("num_instances", split)] = v
        d["instance_variants"] = sum(agg.values())
        out[r["run_spec"]["name"]] = d
    return out


def _stat_index(run):
    """Unperturbed stats of one runs.json run, keyed by (stat name, split, sub_split)."""
    idx = {}
    for s in run["stats"]:
        n = s["name"]
        if n.get("perturbation") is not None:
            continue
        key = (n["name"], n.get("split"), n.get("sub_split"))
        if key in idx:
            raise ValueError(f"duplicate stat {key} in run {run['run_spec']['name']}")
        idx[key] = s
    return idx


def _runs_cell(idxs, metric, split, time_stat):
    """Score and one-pass cost of a cell from the runs.json stats of its runs (``idxs``).

    Score as HELM merges stats: sum / count of the metric's trial entries over the runs; a
    run without an aggregate entry contributes its sub_split entries (CivilComments).
    Costs: per run and sub_split (or the aggregate), num_instances x per-instance means,
    summed, i.e. instance-weighted over sub_splits (HELM's tables weight them equally)."""
    s_sum = s_cnt = 0.0
    parts = []
    for st in idxs:
        agg = st.get((metric, split, None))
        ents = [agg] if agg is not None else [
            st[k] for k in sorted(k for k in st if k[0] == metric and k[1] == split and k[2])]
        if not ents:
            return None
        s_sum += sum(e["sum"] for e in ents)
        s_cnt += sum(e["count"] for e in ents)
        subs = [None] if (RUN_STATS["eval"], split, None) in st else sorted(
            k[2] for k in st if k[0] == RUN_STATS["eval"] and k[1] == split and k[2])
        if not subs:
            return None
        for sub in subs:
            part = {}
            for short, stat in [*RUN_STATS.items(), ("time", time_stat)]:
                e = st.get((stat, split, sub))
                part[short] = float(e["mean"]) if e is not None else None
            if part["eval"] is None:
                return None
            parts.append(part)
    cc = _cell_cost(parts)
    if cc is None or not s_cnt:
        return None
    cc["n_leaf_runs"] = len(idxs)
    cc["score"] = s_sum / s_cnt
    cc["n_score_entries"] = s_cnt
    cc["n_sub_splits"] = len(parts)
    return cc


def _id_date(mid):
    """Date encoded in a HELM model id: ("ymd", date) for YYYYMMDD / YYYY-MM-DD / YYMMDD
    suffixes, ("ym", first of month) for Mistral YYMM versions, ("md", (month, day)) for
    OpenAI MMDD snapshots (year not encoded); None if the id encodes no date."""
    org, _, nm = mid.partition("/")
    pats = [("ymd", r"(20\d\d)-?([01]\d)-?([0-3]\d)(?!\d)", lambda m: m.groups()),
            ("ymd", r"-(\d\d)([01]\d)([0-3]\d)$", lambda m: ("20" + m.group(1),) + m.groups()[1:])]
    if org == "mistralai":
        pats.append(("ym", r"-(\d\d)([01]\d)$", lambda m: ("20" + m.group(1), m.group(2), "01")))
    for kind, pat, parts in pats:
        m = re.search(pat, nm)
        if m:
            try:
                return kind, _dt.date(*(int(x) for x in parts(m)))
            except ValueError:
                continue
    if org == "openai":
        m = re.search(r"-([01]\d)([0-3]\d)(?:-preview)?$", nm)
        if m and 1 <= int(m.group(1)) <= 12 and 1 <= int(m.group(2)) <= 31:
            return "md", (int(m.group(1)), int(m.group(2)))
    return None


def _date_conflict(kind, idd, date):
    """True if release ``date`` (ISO string) contradicts the id-encoded date: earlier than it by
    more than a week, more than 60 days after it (month end for "ym"), or another MM-DD."""
    d = _dt.date.fromisoformat(str(date)[:10])
    if kind == "md":
        return (d.month, d.day) != idd
    end = idd if kind == "ymd" else (idd.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)
    return d < idd - _dt.timedelta(days=7) or d > end + _dt.timedelta(days=60)


def _model_ids_from_cells(acc_rows, specs):
    ids, runs = {}, {}
    for model, row in acc_rows.items():
        names = sorted({r for c in row[1:] if isinstance(c, dict) for r in c.get("run_spec_names", [])})
        mids = sorted({specs[r]["adapter_spec"]["model"] for r in names})
        if len(mids) != 1:
            raise ValueError(f"{model!r}: model ids {mids}")
        ids[model] = mids[0]
        runs[model] = names
    return ids, runs


def build(suite, raw_dir, output_dir, snapshot_csv=None, mmlu_counts=None):
    cfg = SUITES[suite]
    name, rel = cfg["name"], cfg["release"]
    raw = Raw(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building {name} from {raw.dir}")

    schema = raw.json(f"{rel}/schema.json")
    summary = raw.json(f"{rel}/summary.json")
    specs = {s["name"]: s for s in raw.json(f"{rel}/run_specs.json")}
    run2suite = raw.json(f"{rel}/runs_to_run_suites.json")
    core = raw.json(f"{rel}/groups/{cfg['core_group']}.json")
    acc, gi = _table(core, "accuracy"), _table(core, "general_information")
    eff = _table(core, "efficiency")
    acc_rows, gi_rows = _rows(acc), _rows(gi)
    gi_hdr = {h["value"]: j for j, h in enumerate(gi["header"])}
    eff_metric = sorted({(h.get("metadata") or {}).get("metric") for h in eff["header"][1:]}
                        - {None})
    if len(eff_metric) != 1:
        raise ValueError(f"efficiency metrics {eff_metric}")
    time_metric = eff_metric[0]

    # suite (evaluation batch) dates from every release's summary.json
    suite_dates = {}
    for path in sorted(raw.files):
        m = re.match(r"(v[\d.]+)/summary\.json(\.gz)?$", path)
        if m:
            suite_dates[m.group(1)] = raw.json(f"{m.group(1)}/summary.json").get("date")

    # ---- columns ---------------------------------------------------------
    cols = _resolve_columns(cfg, schema, acc)
    aggregates = [c for c in cols if c["group"] in cfg.get("aggregate_groups", [])]
    cols = [c for c in cols if c["group"] not in cfg.get("aggregate_groups", [])]
    for c in cols:
        c["name"] = (c["group"][len(cfg["strip_prefix"]):] if "strip_prefix" in cfg
                     else c["display_name"])
    if len({c["name"] for c in cols}) != len(cols):
        raise ValueError("duplicate benchmark names")
    cols.sort(key=lambda c: c["name"])
    bench = [c["name"] for c in cols]
    wr_col = next((j for j, h in enumerate(acc["header"]) if h["value"] == "Mean win rate"), None)

    # ---- models ----------------------------------------------------------
    schema_models = {m["name"]: m for m in schema["models"]}
    if any("run_spec_names" in c for row in acc_rows.values() for c in row[1:] if isinstance(c, dict)):
        model_id, model_runs = _model_ids_from_cells(acc_rows, specs)
    else:  # Classic tables carry no run names: map display name -> schema model name
        by_disp = collections.defaultdict(list)
        for m in schema["models"]:
            by_disp[m["display_name"]].append(m["name"])
        model_id = {}
        for model in acc_rows:
            if len(by_disp[model]) != 1:
                raise ValueError(f"{model!r}: schema models {by_disp[model]}")
            model_id[model] = by_disp[model][0]
        model_runs = None
    models = sorted(acc_rows)

    # runs of each (model, scenario) cell
    specs_by_model = collections.defaultdict(list)
    for rn in sorted(specs):
        specs_by_model[specs[rn]["adapter_spec"]["model"]].append(rn)

    def table_runs(model, c):
        """Classic: every non-ablation, non-contrast run of the model in the scenario, i.e.
        the runs HELM's leaderboard table pools (incl. zero-shot and device duplicates)."""
        return sorted(rn for rn in specs_by_model[model_id[model]]
                      if c["group"] in specs[rn]["groups"] and "groups=" not in rn
                      and "only_contrast=True" not in rn)

    def cell_runs(model, c):
        if model_runs is not None:
            cell = acc_rows[model][c["col"]]
            return sorted(set(cell.get("run_spec_names", [])))
        # Classic: the few-shot leaderboard runs of a scenario, without HELM's ablation runs
        # ("groups=ablation_*"), contrast sets and variants (e.g. max_train_instances=0);
        # summarization runs exist twice (device=cpu/cuda, same requests), keep one.
        cands = table_runs(model, c)
        if any("data_augmentation=canonical" in rn for rn in cands):
            cands = [rn for rn in cands if "data_augmentation=canonical" in rn]
        out = {}
        for rn in cands:
            out.setdefault(re.sub(r",?device=[^,]+", "", rn), rn)
        if not out:
            return []
        nargs = {k: k.count(",") for k in out}
        return sorted(rn for k, rn in out.items() if nargs[k] == min(nargs.values()))
    all_runs = collections.defaultdict(set)

    # ---- scores and costs ------------------------------------------------
    M, N = len(models), len(bench)
    S = np.full((M, N), np.nan)
    fields = ["instances", "instances_table", "prompt_tokens_mean", "completion_tokens_mean", "cost_tokens",
              "inference_time_s_mean", "inference_time_s_total", "num_train_mean",
              "truncated_frac", "num_train_trials", "num_requests", "requests_per_instance",
              "n_leaf_runs"]
    cost = {f: [[None] * N for _ in range(M)] for f in fields}
    eval_suite = [[None] * N for _ in range(M)]
    adapt = [collections.Counter() for _ in range(N)]
    leaf_names = {}
    chk = collections.Counter()
    flags = [[None] * N for _ in range(M)]
    lb_value = [[None] * N for _ in range(M)]
    rescaled = collections.Counter()
    worst = {"eval_rel": 0.0, "score_abs": 0.0, "runs_instances_rel": 0.0}
    runs_raw = raw.json(f"{rel}/runs.json") if raw.has(f"{rel}/runs.json.gz") else []
    runstats = _run_stats(runs_raw)
    run_idx = {r["run_spec"]["name"]: _stat_index(r) for r in runs_raw}
    del runs_raw
    time_stat = [m["name"] for m in schema.get("metrics", [])
                 if time_metric in (m.get("display_name"), m.get("short_display_name"))]
    if len(time_stat) != 1:
        raise ValueError(f"efficiency metric {time_metric!r}: stats {time_stat}")
    time_stat = time_stat[0]
    scores_from_runs, costs_from_runs = cfg.get("scores_from_runs"), cfg.get("costs_from_runs")
    lb_diffs, suppressed = [], []
    xc = {"n_cells_compared": 0, "n_cells_without_runs": 0,
          "max_abs_score_diff_agreeing": 0.0, "n_score_diff_gt_tol": 0,
          "cost_fields": {f: {"n_compared": 0, "n_differ_gt_1e-9": 0, "max_rel_diff": 0.0,
                              "differ_by_benchmark": collections.Counter()}
                          for f in COST_FIELDS}}
    unexplained_cost_diffs = []

    def rel_diff(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-12)

    for j, c in enumerate(cols):
        leaf_names[c["name"]], leaves = _leaf_stats(raw, rel, c["group"], c["metric"], time_metric)
        gcol = gi_hdr.get(f"{c['header'].rsplit(' - ', 1)[0]} - # eval")
        for i, model in enumerate(models):
            s = _num(acc_rows[model][c["col"]])
            runs = cell_runs(model, c)
            rc = None
            if run_idx and runs and all(rn in run_idx for rn in runs):
                rc = _runs_cell([run_idx[rn] for rn in runs], c["main_name"], c["main_split"],
                                time_stat)
            if s is None:
                if rc is not None:  # runs exist, but HELM's leaderboard shows no value
                    suppressed.append({
                        "model": model, "benchmark": c["name"], "runs_json_value": rc["score"],
                        "n_score_entries": rc["n_score_entries"], "runs": runs,
                        "leaderboard_cell": {k: acc_rows[model][c["col"]].get(k)
                                             for k in ("description",)}})
                continue
            S[i, j] = s
            flags[i][j] = acc_rows[model][c["col"]].get("contamination_level")
            # ---- runs.json score vs the leaderboard value
            if rc is None:
                xc["n_cells_without_runs"] += 1
            else:
                xc["n_cells_compared"] += 1
                d = abs(rc["score"] - s)
                if d <= SCORE_TOL:
                    xc["max_abs_score_diff_agreeing"] = max(xc["max_abs_score_diff_agreeing"], d)
                else:
                    xc["n_score_diff_gt_tol"] += 1
                    pooled = table_runs(model, c) if model_runs is None else runs
                    pc = (_runs_cell([run_idx[rn] for rn in pooled], c["main_name"],
                                     c["main_split"], time_stat)
                          if all(rn in run_idx for rn in pooled) else None)
                    lb_diffs.append({
                        "model": model, "benchmark": c["name"],
                        "B": rc["score"] if scores_from_runs else s,
                        "few_shot_runs_value": rc["score"], "leaderboard_value": s,
                        "diff_few_shot_minus_leaderboard": rc["score"] - s,
                        "leaderboard_description": acc_rows[model][c["col"]].get("description"),
                        "runs_used": runs,
                        "runs_pooled_by_leaderboard_only": sorted(set(pooled) - set(runs)),
                        "pooled_runs_value": pc["score"] if pc else None,
                        "pooled_runs_reproduce_leaderboard":
                            bool(pc is not None and abs(pc["score"] - s) <= SCORE_TOL),
                    })
                    if scores_from_runs:
                        S[i, j] = rc["score"]
                        lb_value[i][j] = s
            lc = _cell_cost(leaves.get(model, []))
            cc = rc if costs_from_runs and rc is not None else lc
            if cc is None:
                chk["cells_without_costs"] += 1
                continue
            if costs_from_runs and rc is None:
                chk["cells_costs_from_leaf_tables_fallback"] += 1
            for f, v in cc.items():
                if f in cost:
                    cost[f][i][j] = v
            # ---- leaf-table costs vs runs.json costs
            if rc is not None and lc is not None:
                differs = []
                for f in COST_FIELDS:
                    if rc.get(f) is None or lc.get(f) is None:
                        continue
                    e = xc["cost_fields"][f]
                    e["n_compared"] += 1
                    r_ = rel_diff(rc[f], lc[f])
                    e["max_rel_diff"] = max(e["max_rel_diff"], r_)
                    if r_ > 1e-9:
                        e["n_differ_gt_1e-9"] += 1
                        e["differ_by_benchmark"][c["name"]] += 1
                        differs.append(f)
                blended = bool(lb_diffs) and lb_diffs[-1]["model"] == model \
                    and lb_diffs[-1]["benchmark"] == c["name"]
                if differs and not blended and rc["n_sub_splits"] == rc["n_leaf_runs"]:
                    # neither a pooled zero-shot run nor sub_split weighting explains it
                    unexplained_cost_diffs.append([model, c["name"], differs])
            if lc is not None:
                # consistency with the leaderboard's General information / Accuracy tables
                lv = leaves[model]
                if len(lv) == 1:  # contaminated cells carry no "(n)" description
                    for k in ("n_eval", "n_score"):
                        lv[0][k] = lv[0].get(k) or 1
                if gcol is not None and _num(gi_rows[model][gcol]) is not None \
                        and _count(gi_rows[model][gcol]) and all(l.get("n_eval") for l in lv):
                    # HELM's cell value is the mean over its (n) stat entries
                    core_mean = _num(gi_rows[model][gcol])
                    leaf_mean = (sum(l["eval"] * l["n_eval"] for l in lv)
                                 / sum(l["n_eval"] for l in lv))
                    rel_err = abs(core_mean - leaf_mean) / max(core_mean, 1.0)
                    worst["eval_rel"] = max(worst["eval_rel"], rel_err)
                    chk["eval_checked"] += 1
                    chk["eval_mismatch"] += rel_err > 1e-6
                if all(l.get("score") is not None and l.get("n_score") for l in lv):
                    # the leaf tables reproduce HELM's displayed value (not necessarily B)
                    rep = sum(l["score"] * l["n_score"] for l in lv) / sum(l["n_score"] for l in lv)
                    worst["score_abs"] = max(worst["score_abs"], abs(rep - s))
                    chk["score_checked"] += 1
                    chk["score_mismatch"] += abs(rep - s) > 1e-9
            all_runs[model].update(runs)
            for rn in runs:
                adapt[j][specs[rn]["adapter_spec"]["method"]] += 1
            sts = sorted({run2suite[rn] for rn in runs if rn in run2suite}, key=_version_key)
            eval_suite[i][j] = sts[-1] if sts else None
            if runstats and runs and all(rn in runstats for rn in runs):
                if all("num_requests" in runstats[rn] for rn in runs):
                    req = sum(runstats[rn]["num_requests"] for rn in runs)
                    var = sum(runstats[rn]["instance_variants"]
                              * specs[rn]["adapter_spec"].get("num_train_trials", 1) for rn in runs)
                    cost["num_requests"][i][j] = req
                    cost["requests_per_instance"][i][j] = req / var if var else None
                inst = sum(runstats[rn].get(("num_instances", c["main_split"]), 0.0) for rn in runs)
                if lc is None:
                    continue
                rel_err = abs(inst - lc["instances"]) / max(lc["instances"], 1.0)
                worst["runs_instances_rel"] = max(worst["runs_instances_rel"], rel_err)
                chk["runs_checked"] += 1
                if rel_err > 1e-6 and inst > 0:
                    # HELM's "# eval" is a mean over sub_split stat entries (CivilComments:
                    # toxic / non-toxic), so the table undercounts.  Costs from runs.json
                    # already use the run totals; leaf-table costs are rescaled.
                    cost["instances_table"][i][j] = lc["instances"]
                    if cc is lc:
                        ratio = inst / lc["instances"]
                        cost["instances"][i][j] = inst
                        for f in ("cost_tokens", "inference_time_s_total"):
                            if cost[f][i][j] is not None:
                                cost[f][i][j] *= ratio
                    rescaled[c["name"]] += 1
                chk["runs_instances_mismatch"] += rel_err > 1e-6
            elif runstats:
                chk["runs_missing"] += 1

    for f in list(cost):  # e.g. newer runs.json carry no num_requests, Lite has no "# trials"
        if all(v is None for row in cost[f] for v in row):
            del cost[f]
    keep = ~np.isnan(S).all(axis=1)
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} models without any score")
    df = pd.DataFrame(S, index=pd.Index(models, name="model"), columns=bench)
    df = df.loc[keep]
    kept = [i for i in range(M) if keep[i]]
    models = [models[i] for i in kept]
    for f in cost:
        cost[f] = [cost[f][i] for i in kept]
    eval_suite = [eval_suite[i] for i in kept]
    flags = [flags[i] for i in kept]
    lb_value = [lb_value[i] for i in kept]

    # ---- per-benchmark cost vectors -------------------------------------
    def colmean(f):
        out, cnt = [], []
        for j in range(N):
            v = [cost[f][i][j] for i in range(len(models)) if cost[f][i][j] is not None]
            out.append(sum(v) / len(v) if v else None)
            cnt.append(len(v))
        return out, cnt
    cost_def = ("sum over the cell's few-shot runs (runs.json) and their sub_splits of "
                "num_instances x (num_prompt_tokens + num_output_tokens)" if costs_from_runs
                else "sum over sub-runs (leaf tables) of '# eval' x ('# prompt tokens' + "
                     "'# output tokens')")
    bcost = {"definition": ("per benchmark, mean over models with an observed cell of the "
                            f"per-(model, benchmark) value; cost_tokens = {cost_def}, one pass "
                            "(one in-context-example trial), model-specific tokenizers"),
             "benchmark_names": bench}
    for f in ["cost_tokens", "instances", "prompt_tokens_mean", "completion_tokens_mean",
              "inference_time_s_total", "num_requests"]:
        if f not in cost:
            continue
        key = f + ("_mean" if not f.endswith("_mean") else "")
        bcost[key], n_models = colmean(f)
        bcost["n_models" if f == "cost_tokens" else f"n_models_{f}"] = n_models
    # requests per evaluated instance: a property of the adapter (1 for generation and
    # multiple_choice_joint; #choices for multiple_choice_separate_original, 2x #choices for
    # _calibrated; candidate passages per query for ranking_binary)
    rpi, rpi_spread = [], []
    for j in range(N):
        meths = set(adapt[j])
        v = sorted(cost["requests_per_instance"][i][j] for i in range(len(models))
                   if "requests_per_instance" in cost and cost["requests_per_instance"][i][j])
        if v:
            rpi.append(float(np.median(v)))
            rpi_spread.append(v[-1] - v[0])
        elif meths and meths <= {"generation", "multiple_choice_joint"}:
            rpi.append(1.0)
            rpi_spread.append(0.0)
        else:
            rpi.append(None)
            rpi_spread.append(None)
    bcost["requests_per_instance"] = rpi
    bcost["requests_per_instance_range"] = rpi_spread
    bcost["requests_per_instance_note"] = (
        "median over models of num_requests / (instances x trials summed over all splits and "
        "perturbed copies), from runs.json; 1 where every run uses generation or "
        "multiple_choice_joint and runs.json reports no num_requests")
    bcost["cost_tokens_requests_mean"] = [
        None if bcost["cost_tokens_mean"][j] is None or rpi[j] is None
        else bcost["cost_tokens_mean"][j] * rpi[j] for j in range(N)]

    # ---- model metadata --------------------------------------------------
    ymeta = raw.yaml("github/model_metadata.yaml")["models"]
    y_by_name, y_by_disp = {}, collections.defaultdict(list)
    for m in ymeta:
        y_by_name.setdefault(m["name"], m)
        y_by_disp[m.get("display_name")].append(m)
    mmeta = {}
    for i, model in enumerate(models):
        mid = model_id[model]
        sm = schema_models.get(mid, {})
        y, src = y_by_name.get(mid), f"model_metadata.yaml@{HELM_COMMIT[:12]}"
        if y is None and len(y_by_disp.get(model, [])) == 1:
            y = y_by_disp[model][0]
            src += f" (matched by display_name to {y['name']})"
        date = _iso(y.get("release_date")) if y else None
        if date is None:
            date, src = sm.get("release_date"), f"{rel} schema.json"
        upstream_date = date
        ov = RELEASE_DATE_OVERRIDES.get(mid)
        if ov is not None and str(upstream_date) == ov[0]:
            date, src = ov[1], f"override of {src}: {ov[2]} (build_helm.RELEASE_DATE_OVERRIDES)"
        sts = sorted({s for s in eval_suite[i] if s}, key=_version_key)
        deployments = sorted({specs[r]["adapter_spec"].get("model_deployment")
                              for r in all_runs[model]} - {None})
        mm = {
            "helm_model_id": mid,
            "model_deployments": deployments,
            "creator_organization": (y or {}).get("creator_organization_name",
                                                  sm.get("creator_organization")),
            "access": (y or {}).get("access", sm.get("access")),
            "num_parameters": (y or {}).get("num_parameters", sm.get("num_parameters")),
            "release_date": date,
            "release_date_source": src,
            "release_date_release_schema": sm.get("release_date"),
            "first_eval_suite": sts[0] if sts else None,
            "first_eval_suite_date": suite_dates.get(sts[0]) if sts else None,
            "last_eval_suite": sts[-1] if sts else None,
            "n_observed": int(df.loc[model].notna().sum()),
        }
        if date != upstream_date:
            mm["release_date_upstream"] = upstream_date
        if wr_col is not None:
            mm["mean_win_rate"] = _num(acc_rows[model][wr_col])
        for a in aggregates:
            mm[a["group"] + "_score"] = _num(acc_rows[model][a["col"]])
        mmeta[model] = mm

    # ---- benchmark metadata ----------------------------------------------
    metric_desc = {m["name"]: m for m in schema.get("metrics", [])}
    bmeta = {}
    for j, c in enumerate(cols):
        md = metric_desc.get(c["main_name"], {})
        bmeta[c["name"]] = {
            "helm_group": c["group"], "display_name": c["display_name"],
            "leaderboard_column": c["header"], "metric": c["metric"],
            "metric_name": c["main_name"], "metric_description": md.get("description"),
            "main_split": c["main_split"], "lower_is_better": c["lower_is_better"],
            "adaptation_methods": dict(sorted(adapt[j].items())),
            "leaf_tables": leaf_names[c["name"]],
            "group_description": c["description"],
            "n_models": int(df[c["name"]].notna().sum()),
        }

    # ---- consistency -----------------------------------------------------
    consistency = {"leaf_vs_leaderboard": dict(sorted(chk.items())),
                   "max_rel_err_eval_sum": worst["eval_rel"],
                   "max_abs_err_score_reproduction": worst["score_abs"],
                   "max_rel_err_runs_json_instances": worst["runs_instances_rel"],
                   "scores_source": ("runs.json few-shot leaderboard runs (HELM's displayed value "
                                     "where equal within SCORE_TOL)" if scores_from_runs
                                     else "leaderboard Accuracy table"),
                   "costs_source": "runs.json" if costs_from_runs else "leaf tables",
                   "leaf_eval_differs_from_runs_json_instances": dict(sorted(rescaled.items()))}
    xc["cost_fields"] = {f: {**v, "differ_by_benchmark": dict(sorted(v["differ_by_benchmark"].items()))}
                         for f, v in xc["cost_fields"].items()}
    xc.update({
        "score_tol": SCORE_TOL,
        "n_leaderboard_differences": len(lb_diffs),
        "min_abs_score_diff_differing": (min(abs(d["diff_few_shot_minus_leaderboard"])
                                             for d in lb_diffs) if lb_diffs else None),
        "max_abs_score_diff_differing": (max(abs(d["diff_few_shot_minus_leaderboard"])
                                             for d in lb_diffs) if lb_diffs else None),
        "all_differences_reproduced_by_pooling_leaderboard_runs":
            all(d["pooled_runs_reproduce_leaderboard"] for d in lb_diffs),
        "unexplained_cost_differences": unexplained_cost_diffs,
        "note": ("runs.json value of each observed cell (the cell's runs; see score_definition) "
                 "vs the leaderboard value, and runs.json costs vs leaf-table costs; cost "
                 "differences are expected only in leaderboard_differences cells and in cells "
                 "with sub_splits (CivilComments)"),
    })
    consistency["runs_json_cross_check"] = xc
    consistency["runs_json_values_missing_from_leaderboard"] = suppressed
    if lb_diffs and not scores_from_runs:
        print(f"  WARNING: {len(lb_diffs)} cells where runs.json disagrees with the leaderboard")
    if unexplained_cost_diffs:
        print(f"  WARNING: unexplained cost differences {unexplained_cost_diffs[:5]}")
    if suite == "mmlu" and mmlu_counts is not None and Path(mmlu_counts).exists():
        ref = json.loads(Path(mmlu_counts).read_text())
        inst = dict(zip(bench, bcost["instances_mean"]))
        consistency["instances_vs_mmlu_test_counts"] = {
            "path": Path(mmlu_counts).name,
            "n_subjects_equal": sum(1 for b in bench if ref.get(b) == inst[b]),
            "n_subjects": len(bench),
            "differences": {b: [inst[b], ref.get(b)] for b in bench if ref.get(b) != inst[b]},
        }
    date_disagree = sorted(m for m, v in mmeta.items() if v["release_date_release_schema"]
                           and v.get("release_date_upstream", v["release_date"])
                           != v["release_date_release_schema"])
    consistency["release_date_yaml_vs_schema_disagreements"] = date_disagree
    n_id, conf_up, conf_now, overridden = 0, [], [], []
    for m, v in mmeta.items():
        idd = _id_date(v["helm_model_id"])
        if idd is None or v["release_date"] is None:
            continue
        n_id += 1
        kind, d = idd
        d_str = d.isoformat() if kind != "md" else f"--{d[0]:02d}-{d[1]:02d}"
        up = v.get("release_date_upstream", v["release_date"])
        if _date_conflict(kind, d, up):
            conf_up.append([m, v["helm_model_id"], str(up), d_str])
        if _date_conflict(kind, d, v["release_date"]):
            conf_now.append([m, v["helm_model_id"], str(v["release_date"]), d_str])
        if "release_date_upstream" in v:
            overridden.append([m, v["helm_model_id"], str(up), str(v["release_date"])])
    ids_here = {v["helm_model_id"]: v for v in mmeta.values()}
    consistency["release_date_vs_model_id"] = {
        "rule": ("a release date conflicts with the date in the model id if it is more than 7 "
                 "days earlier or more than 60 days later (YYMM ids: than the month), or, for "
                 "OpenAI MMDD ids, has another month-day"),
        "n_models_with_date_in_id": n_id,
        "conflicts_upstream": sorted(conf_up),
        "overridden": sorted(overridden),
        "conflicts_after_overrides": sorted(conf_now),
        "overrides_not_applied": sorted(
            mid for mid in RELEASE_DATE_OVERRIDES
            if mid in ids_here and "release_date_upstream" not in ids_here[mid]),
    }
    if conf_now:
        print(f"  WARNING: release dates contradict the model id: {conf_now}")
    if "snapshot_release" in cfg:
        consistency["snapshot"] = _snapshot_check(cfg, raw, df, model_id, model_runs, run2suite,
                                                  cols, mmeta, snapshot_csv)
    qualification = None
    if "min_models" in cfg:
        tok = [v for v in bcost["prompt_tokens_mean"] if v is not None]
        per_inst = [bcost["cost_tokens_mean"][j] / bcost["instances_mean"][j]
                    for j in range(N) if bcost["cost_tokens_mean"][j] and bcost["instances_mean"][j]]
        qualification = {
            "n_models": len(models), "n_benchmarks": N,
            "single_main_metric_per_scenario": True,
            "tokens_per_instance_min": min(per_inst), "tokens_per_instance_max": max(per_inst),
            "tokens_per_instance_ratio": max(per_inst) / min(per_inst),
            "passes": bool(len(models) >= cfg["min_models"] and N >= cfg["min_benchmarks"]
                           and tok and max(per_inst) / min(per_inst) >= 10),
        }
        if not qualification["passes"]:
            print(f"  {name} does not qualify: {qualification}; nothing written")
            return None

    # ---- write -----------------------------------------------------------
    save_matrix(name, df, output_dir)
    man = raw.manifest
    meta = {
        "name": name,
        "title": cfg["title"],
        "source": {
            "project": cfg["project"], "release": rel, "release_date": summary.get("date"),
            "release_suites": summary.get("suites"),
            "url": f"https://storage.googleapis.com/{GCS_BUCKET}/{cfg['project']}/"
                   f"benchmark_output/releases/{rel}/",
            "leaderboard_group": cfg["core_group"],
            "leaderboard_url": f"https://crfm.stanford.edu/helm/{cfg['project']}/{rel}/",
            "model_metadata": {"repo": f"https://github.com/{HELM_REPO}", "commit": HELM_COMMIT,
                               "path": MODEL_METADATA},
            "raw_manifest": f"raw_manifests/{name}.json",
            "raw_inputs": [{k: e.get(k) for k in ("path", "url", "revision", "size", "sha256",
                                                  "decoded_sha256", "fetched_at")}
                           for e in sorted(man["inputs"], key=lambda e: e["path"])],
            "fetched_at": min(e["fetched_at"] for e in man["inputs"]),
            "license": LICENSE,
            "builder": "data/build_helm.py",
        },
        "order": ("rows: HELM display names sorted by Python str order (Unicode code points); "
                  "columns: benchmark names sorted the same way"),
        "score_definition": (
            "the scenario's main metric on its main split over the cell's few-shot leaderboard "
            "runs in runs.json (data_augmentation=canonical; no ablation, contrast, zero-shot "
            "max_train_instances=0 or duplicate device runs), merged as HELM merges stats "
            "(sum / count over trial entries: mean over sub-runs and the in-context-example "
            "trials). Equal to the release leaderboard 'Accuracy' table value (kept bit for "
            "bit) except in leaderboard_differences, where HELM's table also pools a zero-shot "
            "run" if scores_from_runs else
            "value of the release leaderboard 'Accuracy' table: the scenario's main metric on "
            "its main split, macro-averaged by HELM over the scenario's sub-runs (re-derived "
            "from runs.json in consistency.runs_json_cross_check)"),
        "leaderboard_differences": lb_diffs,
        "efficiency_metric": time_metric,
        "suite_dates": {k: suite_dates[k] for k in sorted(suite_dates, key=_version_key)},
        "model_names": models,
        "benchmark_names": bench,
        "models": mmeta,
        "benchmarks": bmeta,
        "benchmark_cost": bcost,
        "cell_costs": {"layout": "lists [model][benchmark] aligned with model_names x "
                                 "benchmark_names; null where unobserved (leaderboard_value: "
                                 "HELM's displayed score, only where it differs from B)",
                       **cost, "eval_suite": eval_suite,
                       **({"contamination_level": flags}
                          if any(v for row in flags for v in row) else {}),
                       **({"leaderboard_value": lb_value}
                          if any(v is not None for row in lb_value for v in row) else {})},
        "consistency": consistency,
    }
    if qualification:
        meta["qualification"] = qualification
    meta["shape"] = [len(models), N]
    meta["fill_fraction"] = float(df.notna().values.mean())
    meta["caveats"] = cfg.get("caveats", []) + COMMON_CAVEATS
    (output_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    (output_dir / "raw_manifests").mkdir(exist_ok=True)
    (output_dir / "raw_manifests" / f"{name}.json").write_text(
        json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"  wrote {name}.meta.json and raw_manifests/{name}.json")
    print("  consistency:", json.dumps(consistency, sort_keys=True)[:2000])
    return meta


def _snapshot_check(cfg, raw, df, model_id, model_runs, run2suite, cols, mmeta, snapshot_csv):
    """Compare the release with the snapshot release (Lite v1.0.0 = data/helm.csv)."""
    srel = cfg["snapshot_release"]
    sspecs = {s["name"]: s for s in raw.json(f"{srel}/run_specs.json")}
    score = _table(raw.json(f"{srel}/groups/{cfg['core_group']}.json"), "accuracy")
    srows = _rows(score)
    sid, sruns = _model_ids_from_cells(srows, sspecs)
    scols = {(h.get("metadata") or {}).get("run_group"): j for j, h in enumerate(score["header"])}
    wr = next((j for j, h in enumerate(score["header"]) if h["value"] == "Mean win rate"), None)
    id2new = {model_id[m]: m for m in df.index}
    id2old = {mid: old for old, mid in sid.items()}
    for m in df.index:
        mmeta[m]["in_snapshot"] = model_id[m] in id2old
        if mmeta[m]["in_snapshot"]:
            mmeta[m]["snapshot_display_name"] = id2old[model_id[m]]
    diffs, changed, n_cmp, renamed = [], [], 0, []
    same_runs, rerun = 0, []
    for old, mid in sorted(sid.items()):
        new = id2new.get(mid)
        if new is None:
            continue
        if new != old:
            renamed.append([old, new])
        if set(sruns[old]) <= set(model_runs[new]):
            same_runs += 1
        suites = sorted({run2suite.get(r) for r in model_runs[new]} - {None}, key=_version_key)
        if suites != [srel]:
            rerun.append([new, suites])
        for c in cols:
            j = scols.get(c["display_name"])
            a, b = (_num(srows[old][j]) if j is not None else None), df.loc[new, c["name"]]
            if a is None or np.isnan(b):
                continue
            n_cmp += 1
            diffs.append(abs(a - b))
            if abs(a - b) > 1e-9:
                changed.append([new, c["name"], a, float(b)])
    out = {
        "snapshot_release": srel,
        "n_snapshot_models": len(sid),
        "n_snapshot_models_in_release": sum(1 for m in sid.values() if m in id2new),
        "snapshot_models_missing_from_release": sorted(o for o, m in sid.items() if m not in id2new),
        "display_name_changes": renamed,
        "n_cells_compared": n_cmp,
        "n_cells_changed_gt_1e-9": len(changed),
        "max_abs_score_diff": max(diffs) if diffs else None,
        "changed_cells": changed,
        "n_models_with_snapshot_runs_retained": same_runs,
        "models_with_runs_from_later_suites": rerun,
        "conclusion": None,
    }
    out["conclusion"] = (
        "identical: every snapshot model keeps its v1.0.0 runs and scores" if not changed and not rerun
        else "HELM changed or re-ran some snapshot cells; see changed_cells / "
             "models_with_runs_from_later_suites")
    if snapshot_csv is not None and Path(snapshot_csv).exists():
        snap = pd.read_csv(snapshot_csv, index_col=0)
        dd, unmatched = [], []
        for old in snap.index:
            if old not in srows:
                unmatched.append(old)
                continue
            for col in snap.columns:
                j = wr if col == "Mean win rate" else scols.get(col)
                a = _num(srows[old][j]) if j is not None else None
                if a is not None and pd.notna(snap.loc[old, col]):
                    dd.append(abs(a - float(snap.loc[old, col])))
        out["snapshot_csv"] = {
            "path": Path(snapshot_csv).name, "shape": list(snap.shape),
            "rows_not_in_snapshot_release": unmatched, "n_cells_compared": len(dd),
            "max_abs_diff_vs_snapshot_release": max(dd) if dd else None,
            "sha256": hashlib.sha256(Path(snapshot_csv).read_bytes()).hexdigest(),
        }
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--raw-dir", type=Path, help="raw input directory "
                    "(default: ~/benchselect-data/raw/<name>)")
    ap.add_argument("--output-dir", type=Path, default=HERE)
    ap.add_argument("--fetch", action="store_true", help="download raw inputs, then exit")
    ap.add_argument("--pin-manifest", type=Path,
                    help="with --fetch: re-download exactly the objects of this manifest")
    ap.add_argument("--snapshot-csv", type=Path, default=HERE / "helm.csv",
                    help="lite only: Lite v1.0.0 snapshot to cross-check (default: helm.csv)")
    ap.add_argument("--mmlu-counts", type=Path, default=HERE / "mmlu_test_counts.json",
                    help="mmlu only: per-subject test counts to cross-check instances")
    a = ap.parse_args()
    if a.raw_dir is None:
        a.raw_dir = Path.home() / "benchselect-data" / "raw" / SUITES[a.suite]["name"]
    return a


if __name__ == "__main__":
    args = parse_args()
    if args.fetch:
        fetch(args.suite, args.raw_dir, args.pin_manifest)
    else:
        build(args.suite, args.raw_dir, args.output_dir.resolve(), args.snapshot_csv,
              args.mmlu_counts)
