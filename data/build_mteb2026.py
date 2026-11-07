#!/usr/bin/env python3
"""Build the MTEB prospective matrix ``mteb2026`` (models x 56 MTEB(eng, classic) tasks).

``data/mteb.matrix.npz`` (263 x 56) is our March-2026 snapshot, produced by
``download_mteb.py`` (2026-03-13T23:29Z) + ``build_matrices.py``.  This script
rebuilds the same 56 columns from a later, pinned state of the MTEB results
and adds every model that has >= 2 of the 56 tasks, so that models released
after the snapshot can serve as a prospective (temporal) test set.

Score definition (identical to download_mteb.py, which called
``mteb.load_results(tasks=MTEB(eng, classic).tasks)`` and ``TaskResult.get_score()``):

* cell(model, task) = mean of ``main_score`` over the (split, hf_subset)
  entries restricted to the benchmark task's ``eval_splits`` x ``hf_subsets``
  (``validate_and_filter_scores``).  For all 56 tasks this is a single entry:
  split ``test`` (``dev`` for MSMARCO), subset ``default`` (``en`` for
  AmazonCounterfactual/AmazonReviews/MTOP*/Massive*/STS22, ``en-en`` for STS17).
  A missing split/subset makes the cell missing.  Scores are on the native 0-1
  scale (no rescaling).
* Several result revisions of one model are merged per cell.  The snapshot
  used pandas ``pivot_table(aggfunc="first")`` in filesystem order; here the
  order is deterministic (revisions with model_meta.json first, then more
  complete revisions, then revision string).  On the pinned inputs no cell has
  conflicting values across revisions (reported in meta).

Inputs (``--raw-dir``, written by steps 1 and 2 below into an empty directory,
see ``raw_manifests/mteb2026.json``):

* ``hf_mteb_results/data/train-0000{0..3}-of-00004.parquet``: HF dataset
  ``mteb/results`` at a pinned commit (long format, one row per model,
  revision, task, split, subset).  This is the score source.  ``README.md``
  (dataset card, declares the license) at the same commit.
* ``results_git/``: blobless git clone of github.com/embeddings-benchmark/results
  with the 56 task JSONs, model_meta.json blobs and the top-level LICENSE
  fetched for the snapshot commit and the pinned HEAD (sparse checkout).  It is
  used through git plumbing only (ls-tree, cat-file, log; lazy fetching
  disabled) for model_meta.json release dates, first-seen dates, duplicate-row
  detection (identical result blobs), the license text, and an exact
  JSON-level reproduction of the snapshot.
* ``mteb_pkg/``: JSON dumps of the MTEB(eng, classic) task metadata, the
  ModelMeta registry and the package / task-dataset licenses from the
  pip-installed ``mteb`` package (dumped offline).

No credentials are used: git runs with GIT_CONFIG_GLOBAL=/dev/null,
GIT_CONFIG_NOSYSTEM=1 and ``-c credential.helper=`` (no gh/credential helper),
Hugging Face downloads pass ``token=False``, and the HF API is queried with a
requests session that ignores ~/.netrc.

Row order: model names sorted by Unicode code point (same convention as
mteb.matrix.npz).  Column order: exactly the 56 columns of mteb.matrix.npz.

Prospective test set: use ``meta['prospective_test_set']`` (status
``added_after_snapshot`` and not a duplicate of a snapshot row).  Two such
models (jina-embeddings-v5-omni-nano/-small) re-use the result files of
snapshot rows (identical git blobs) and are flagged by
``duplicate_of_snapshot_model``.  ``release_date`` is unreliable as the sole
filter (it is sometimes later than the first appearance of the results);
``release_date_after_first_seen`` flags those rows.

Reproduce (from a directory that contains no downloaded files; nothing downloaded is executed):
    # 1. dump task + ModelMeta metadata from the installed mteb package (offline)
    HF_HUB_OFFLINE=1 ~/venvs/mteb/bin/python -I build_mteb2026.py --dump-mteb-pkg --raw-dir RAW
    # 2. fetch pinned HF parquet + blobless clone / sparse checkout of the results repo
    #    (RAW must contain only mteb_pkg/ from step 1; ~360 MB, ~15 HF requests + 1 git clone)
    ~/venvs/benchselect-data/bin/python -I build_mteb2026.py --fetch --raw-dir RAW
    # 3. build (offline; git is used read-only with lazy fetching disabled)
    ~/venvs/benchselect-data/bin/python -I build_mteb2026.py --raw-dir RAW \
        --snapshot-dir data/ --output-dir data/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_matrices import save_matrix  # noqa: E402

NAME = "mteb2026"
HF_REPO = "mteb/results"
HF_REVISION = "9d3f92b432d3e1d9c35b375650747f0afb4d35c5"
GITHUB_URL = "https://github.com/embeddings-benchmark/results"
GIT_HEAD = "039756031720de6d3f2e084e562f2b52233eeeda"
# last commit on main before download_mteb.py wrote data/mteb.npz (2026-03-13T23:29:05Z)
GIT_SNAPSHOT = "37dfa3b5a3adb370d181ad168470cd86d35a4ead"
SNAPSHOT_BUILT_AT = "2026-03-13T23:29:05Z"
RELEASE_CUTOFF = "2026-03-14"
PARQUETS = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]

# keys of legacy (pre mteb 1.11) result files that are not hf_subsets
_LEGACY_NONSUBSET = {"v_measures", "cos_sim", "euclidean", "manhattan", "dot", "max"}
_LEGACY_RENAMES = [("cosine", "cos_sim"), ("manhattan", "manhattan"), ("euclidean", "euclidean"),
                   ("dot", "dot"), ("max", "max"), ("similarity", "similarity")]


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def _clean(x):
    """Make a value JSON-serialisable and deterministic."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if math.isnan(x) else x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _git_env():
    """Environment for every git call: no global/system config (so no gh credential helper or
    other user settings), no terminal prompts."""
    return dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                GIT_ASKPASS="/bin/false", SSH_ASKPASS="/bin/false")


def spdx_by_keyword(text):
    """Coarse license identification of a LICENSE text (the full text is recorded as well)."""
    if not text:
        return None
    t = " ".join(text.split())
    rules = [("CC0-1.0", ["CC0 1.0 Universal"]), ("CC-BY-4.0", ["Attribution 4.0 International"]),
             ("Apache-2.0", ["Apache License", "Version 2.0"]),
             ("MIT", ["Permission is hereby granted, free of charge"]),
             ("BSD-3-Clause", ["Redistribution and use in source and binary forms", "Neither the name"])]
    for spdx, keys in rules:
        if all(k in t for k in keys):
            return spdx
    return "unrecognized (see text)"


def hf_card_license(readme_text):
    """``license:`` field of the YAML front matter of a HF dataset card (regex, no YAML parser)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", readme_text or "", re.S)
    if not m:
        return None
    lm = re.search(r"^license:\s*(\S.*?)\s*$", m.group(1), re.M)
    return lm.group(1).strip("'\"") if lm else None


class Git:
    """Read-only access to the blobless clone. Lazy fetching is disabled, so a
    missing object raises instead of touching the network."""

    def __init__(self, repo):
        self.repo = str(repo)
        self.base = ["git", "-C", self.repo, "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=",
                     "-c", "protocol.allow=never", "-c", "remote.origin.promisor=false"]
        self.env = dict(_git_env(), GIT_NO_LAZY_FETCH="1")

    def run(self, *args):
        return subprocess.run(self.base + list(args), check=True, capture_output=True,
                              text=True, env=self.env).stdout

    def commit_info(self, rev):
        h, date, title = self.run("log", "-1", "--format=%H%x09%cI%x09%s", rev).strip().split("\t", 2)
        return {"sha": h, "committer_date": date, "title": title}

    def tree(self, commit):
        """(model_dir, rev_dir, file, oid) for every results/<dir>/<rev>/<file>."""
        rows = []
        for line in self.run("ls-tree", "-r", "--full-tree", commit, "--", "results/").splitlines():
            meta, path = line.split("\t", 1)
            _, typ, oid = meta.split()
            parts = path.split("/")
            if typ == "blob" and len(parts) == 4:
                rows.append((parts[1], parts[2], parts[3], oid))
        return pd.DataFrame(rows, columns=["model_dir", "rev_dir", "file", "oid"])

    def path_oid(self, commit, path):
        """git oid of one blob at ``commit`` (None if the path does not exist)."""
        out = self.run("ls-tree", "--full-tree", commit, "--", path).strip()
        return out.split("\t", 1)[0].split()[2] if out else None

    def blobs(self, oids):
        """Return {oid: bytes} using one ``cat-file --batch`` process."""
        oids = sorted(set(oids))
        if not oids:
            return {}
        p = subprocess.run(self.base + ["cat-file", "--batch"], input="\n".join(oids).encode() + b"\n",
                           check=True, capture_output=True, env=self.env)
        out, buf, i = {}, p.stdout, 0
        for oid in oids:
            j = buf.index(b"\n", i)
            header = buf[i:j].decode().split()
            if len(header) < 3 or header[1] != "blob":
                raise RuntimeError(f"missing blob {oid}: {header}")
            size = int(header[2])
            out[oid] = buf[j + 1:j + 1 + size]
            i = j + 1 + size + 1
        return out

    def first_parent_order(self, commit):
        """{sha: position} along the first-parent history (0 = root)."""
        shas = self.run("rev-list", "--first-parent", "--reverse", commit).split()
        return {s: i for i, s in enumerate(shas)}

    def first_added(self, commit):
        """{(model_dir, rev_dir, file): (position, sha, committer_date)} for the first-parent
        commit that first added each path (no rename detection)."""
        order = self.first_parent_order(commit)
        first = {}
        cur = None
        for line in self.run("log", "--first-parent", "--reverse", "--no-renames", "--diff-filter=A",
                             "--name-only", "--format=@@%H%x09%cI", commit, "--", "results/").splitlines():
            if line.startswith("@@"):
                sha, date = line[2:].split("\t")
                cur = (order[sha], sha, date)
            elif line.strip():
                parts = line.split("/")
                if len(parts) == 4:
                    first.setdefault((parts[1], parts[2], parts[3]), cur)
        return first, order

    def path_log(self, since, until, path):
        out = self.run("log", "--first-parent", "--format=%H%x09%cI%x09%s", f"{since}..{until}", "--", path)
        return [dict(zip(["sha", "committer_date", "title"], l.split("\t", 2))) for l in out.splitlines() if l]


# ----------------------------------------------------------------------------
# score definition (mirror of mteb TaskResult.validate_and_filter_scores + get_score)
# ----------------------------------------------------------------------------
def mteb_cell_score(split_entries, tm):
    """split_entries: {split: [ {hf_subset, languages, main_score}, ...]}.
    Returns float (possibly NaN) or None (main_score missing -> no value)."""
    splits, hs, agg = tm["eval_splits"], set(tm["hf_subsets"]), tm["is_aggregate"]
    s2l = tm["hf_subsets_to_langscripts"]
    new = {}
    for split, entries in split_entries.items():
        if split not in splits:
            continue
        if agg:
            kept = [e for e in entries if e["hf_subset"] == "default"]
            seen = {"default"} | {e["hf_subset"] for e in kept}
        else:
            kept = [e for e in entries if e["hf_subset"] in hs]
            seen = {e["hf_subset"] for e in kept}
        if seen != hs and not (agg and "default" in seen):
            kept = kept + [{"hf_subset": m, "main_score": float("nan"), "languages": s2l.get(m, [])}
                           for m in sorted(hs - seen)]
        new[split] = kept
    for split in splits:
        if split not in new:
            new[split] = [{"hf_subset": m, "main_score": float("nan"), "languages": s2l.get(m, [])}
                          for m in sorted(hs)]
    values = [e["main_score"] for split in new for e in new[split] if len(e["languages"]) > 0]
    if any(v is None or isinstance(v, (str, bool, dict, list)) for v in values):
        return None
    return float(np.mean(values)) if values else float("nan")


def _version_key(v):
    if v is None:
        return None
    v = str(v).split("-", 1)[0] if re.match(r"^\d", str(v)) else str(v)
    m = re.match(r"^(\d+(?:\.\d+)*)(.*)$", v)
    if not m:
        return None
    nums = tuple(int(x) for x in m.group(1).split("."))
    nums = (nums + (0, 0, 0))[:3] if len(nums) < 3 else nums
    pre = 0 if re.match(r"^\.?(dev|a|b|rc)", m.group(2)) else 1
    return nums + (pre,)


def parse_result_json(data, tasks_lc, renamed, task_meta):
    """Mirror of mteb TaskResult.from_disk (incl. pre-1.11 conversion), restricted to
    the main score.  Returns (task_name, {split: [entries]}) or None if not one of our tasks."""
    vk = _version_key(data.get("mteb_version"))
    pre111 = (vk is not None and vk < (1, 11, 0, 1)) or ("mteb_version" not in data)
    sc = data.get("scores")
    if isinstance(sc, dict) and "task_name" in data and all(isinstance(v, list) for v in sc.values()):
        task = data["task_name"]
        if task not in task_meta:
            return None
        return task, {split: [{"hf_subset": e.get("hf_subset"), "languages": e.get("languages") or [],
                               "main_score": e.get("main_score")} for e in lst] for split, lst in sc.items()}
    if not pre111:
        raise ValueError("unparseable post-1.11 result file")
    scores = {k: v for k, v in data.items()
              if k not in ("dataset_revision", "mteb_dataset_name", "mteb_version") and isinstance(v, dict)}
    task = data["mteb_dataset_name"]
    task = renamed.get(task, task)
    task = tasks_lc.get(task.lower())
    if task is None:
        return None
    tm = task_meta[task]
    for split, ss in scores.items():
        if isinstance(ss, dict):
            ss.pop("evaluation_time", None)
    contains_hf_subset = any(isinstance(v, dict) for ss in scores.values() if isinstance(ss, dict)
                             for k, v in ss.items() if k not in _LEGACY_NONSUBSET)
    if not contains_hf_subset:
        scores = {split: {"default": dict(ss)} for split, ss in scores.items()}
    s2l = tm["hf_subsets_to_langscripts"]
    all_langs = []
    for langs in s2l.values():
        all_langs.extend(l for l in langs if l not in all_langs)
    out = {}
    for split, ss in scores.items():
        out[split] = []
        for subset, hss in ss.items():
            hss = dict(hss)
            for name, prev in _LEGACY_RENAMES:
                p = hss.pop(prev, None)
                if isinstance(p, dict):
                    for k, v in p.items():
                        hss[f"{name}_{k}"] = v
            main = hss.get("main_score", hss.get(tm["main_score"]))
            out[split].append({"hf_subset": subset, "languages": s2l.get(subset, all_langs), "main_score": main})
    return task, out


def json_matrix_at(git, tree, task_meta, renamed, require_model_meta, known_stems):
    """Reproduce ``mteb.load_results`` + download_mteb.py for one commit from the JSON files.

    mteb reads every ``*.json`` in a revision dir and filters on the task name *inside* the
    file.  We read every file whose name is one of the 56 tasks, plus every file whose name is
    not a task registered in the mteb package (e.g. ``TRECCOVID copy.json``); files named after
    another registered task are assumed to contain that task and are skipped.
    Returns long DataFrame (model_name, model_revision, model_dir, rev_dir, file, task, score, oid,
    mteb_version) and the {(dir, rev): (name, revision)} map from model_meta.json."""
    tasks_lc = {t.lower(): t for t in task_meta}
    meta_rows = tree[tree.file == "model_meta.json"]
    metas = git.blobs(meta_rows.oid)
    names = {}
    for r in meta_rows.itertuples():
        j = json.loads(metas[r.oid])
        names[(r.model_dir, r.rev_dir)] = (j["name"].replace("__", "/"), str(j["revision"]))
    files = tree[(tree.file != "model_meta.json") & tree.file.str.endswith(".json")]
    stems = files.file.str[:-5].str.lower()
    files = files[stems.isin(tasks_lc.keys()) | ~stems.isin(known_stems)]
    blobs = git.blobs(files.oid)
    recs = []
    for r in files.itertuples():
        key = (r.model_dir, r.rev_dir)
        if key in names:
            mname, mrev = names[key]
        elif require_model_meta:
            continue
        else:
            mname, mrev = r.model_dir.replace("__", "/"), r.rev_dir
        data = json.loads(blobs[r.oid])
        parsed = parse_result_json(data, tasks_lc, renamed, task_meta)
        if parsed is None:
            continue
        task, entries = parsed
        s = mteb_cell_score(entries, task_meta[task])
        if s is None:
            continue
        recs.append((mname, mrev, r.model_dir, r.rev_dir, r.file, task, s, r.oid, str(data.get("mteb_version"))))
    return pd.DataFrame(recs, columns=["model_name", "model_revision", "model_dir", "rev_dir", "file", "task",
                                       "score", "oid", "mteb_version"]), names


# ----------------------------------------------------------------------------
# step 1: dump metadata from the installed mteb package (run with the mteb venv)
# ----------------------------------------------------------------------------
def dump_mteb_pkg(raw_dir):
    import importlib.metadata as im
    import logging
    import warnings

    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    import mteb
    from mteb.get_tasks import _TASKS_REGISTRY
    from mteb.results import task_result as _tr

    out_dir = Path(raw_dir) / "mteb_pkg"
    out_dir.mkdir(parents=True, exist_ok=True)
    b = mteb.get_benchmark("MTEB(eng, classic)")
    tasks = []
    for t in b.tasks:
        md = t.metadata
        tasks.append({
            "name": md.name, "type": md.type, "main_score": md.main_score,
            "eval_splits": list(t.eval_splits), "hf_subsets": list(t.hf_subsets),
            "is_aggregate": bool(getattr(t, "is_aggregate", False)),
            "hf_subsets_to_langscripts": {k: list(v) for k, v in md.hf_subsets_to_langscripts.items()},
            "dataset_path": md.dataset.get("path"), "dataset_revision": md.dataset.get("revision"),
        })
    tasks.sort(key=lambda x: x["name"])
    json.dump({
        "mteb_version": mteb.__version__,
        "benchmark": b.name,
        "renamed_tasks": dict(getattr(_tr, "renamed_tasks", {})),
        "outdated_tasks": sorted(getattr(_tr, "outdated_tasks", {}).keys()),
        "all_registered_task_names": sorted(_TASKS_REGISTRY.keys()),
        "tasks": tasks,
    }, open(out_dir / "mteb_eng_classic_tasks.json", "w"), indent=1, sort_keys=True)
    keep = ["name", "revision", "release_date", "n_parameters", "embed_dim", "max_tokens", "open_weights",
            "modalities", "model_type", "reference", "superseded_by", "adapted_from", "languages", "license",
            "is_cross_encoder", "use_instructions", "framework", "experiment_kwargs"]
    rows = []
    for m in mteb.get_model_metas():
        d = m.model_dump(mode="json", exclude={"loader", "loader_kwargs"}, warnings=False)
        r = {k: d.get(k) for k in keep if k in d}
        if r.get("release_date") is not None:
            r["release_date"] = str(r["release_date"])
        rows.append(r)
    rows.sort(key=lambda r: (r["name"], str(r.get("revision"))))
    json.dump({"mteb_version": mteb.__version__, "dist_name": im.distribution("mteb").metadata["Name"],
               "n_model_metas": len(rows), "model_metas": rows},
              open(out_dir / "mteb_model_metas.json", "w"), indent=1, sort_keys=True)
    # licenses: the mteb package itself (metadata + bundled license files) and each task's dataset
    dist = im.distribution("mteb")
    lic_files = []
    for f in sorted(dist.files or [], key=str):
        if re.search(r"(^|/)(LICEN[CS]E|COPYING)[^/]*$", str(f), re.I) and ".dist-info/" in str(f):
            txt = f.read_text()
            lic_files.append({"path": str(f), "sha256": hashlib.sha256(txt.encode()).hexdigest(), "text": txt})
    md_ = dist.metadata
    json.dump({
        "mteb_version": mteb.__version__, "dist_name": md_["Name"],
        "license": md_.get("License"), "license_expression": md_.get("License-Expression"),
        "license_classifiers": sorted(c for c in (md_.get_all("Classifier") or []) if c.startswith("License")),
        "license_files": lic_files,
        "task_dataset_licenses": {t.metadata.name: t.metadata.license for t in sorted(b.tasks, key=lambda t: t.metadata.name)},
    }, open(out_dir / "mteb_package_license.json", "w"), indent=1, sort_keys=True)
    print(f"mteb {mteb.__version__}: {len(tasks)} tasks, {len(rows)} model metas, "
          f"{len(lic_files)} license files -> {out_dir}")


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gitignore_escape(path):
    return "/" + re.sub(r"([\\*?\[\]!#])", r"\\\1", path)


# ----------------------------------------------------------------------------
# step 2: fetch raw inputs (network)
# ----------------------------------------------------------------------------
def fetch(raw_dir):
    import shutil
    import time

    import requests
    from huggingface_hub import hf_hub_download

    raw_dir = Path(raw_dir)
    tdump = json.load(open(raw_dir / "mteb_pkg" / "mteb_eng_classic_tasks.json"))  # from --dump-mteb-pkg
    repo = raw_dir / "results_git"
    for p in (repo, raw_dir / "hf_mteb_results", raw_dir / "hf_commits.json"):
        if p.exists():
            raise SystemExit(f"{p} exists: --fetch must run on a raw dir that holds only mteb_pkg/")
    # HF parquet at the pinned revision (5 one-shot file downloads, anonymous)
    (raw_dir / "hf_fetched_at.txt").write_text(_utcnow() + "\n")
    for p in PARQUETS + ["README.md"]:
        hf_hub_download(HF_REPO, p, repo_type="dataset", revision=HF_REVISION,
                        local_dir=raw_dir / "hf_mteb_results", token=False)
    shutil.rmtree(raw_dir / "hf_mteb_results" / ".cache", ignore_errors=True)
    sess = requests.Session()
    sess.trust_env = False  # no ~/.netrc credentials, no env-provided auth
    commits = []
    for page in range(20):
        r = sess.get(f"https://huggingface.co/api/datasets/{HF_REPO}/commits/main",
                     params={"p": page, "limit": 50}, timeout=60)
        r.raise_for_status()
        commits += r.json()
        time.sleep(1)
        if len(r.json()) < 50:
            break
    json.dump(commits, open(raw_dir / "hf_commits.json", "w"), indent=1)
    (raw_dir / "hf_commits_fetched_at.txt").write_text(_utcnow() + "\n")
    # results repo: blobless clone (commits + trees only), then sparse checkouts fetch the blobs we read
    env = _git_env()
    g = ["git", "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=", "-c", "advice.detachedHead=false"]
    (raw_dir / "github_clone_started_at.txt").write_text(_utcnow() + "\n")
    subprocess.run(g + ["clone", "--filter=blob:none", "--no-checkout", "--single-branch", "--branch", "main",
                        GITHUB_URL + ".git", str(repo)], check=True, env=env)
    (raw_dir / "github_clone_finished_at.txt").write_text(_utcnow() + "\n")
    g += ["-C", str(repo)]
    known = {x.lower() for x in tdump["all_registered_task_names"]} | \
        {x.lower() for x in tdump["renamed_tasks"]} | {x.lower() for x in tdump["outdated_tasks"]}
    pats = ["/LICENSE", "/results/*/*/model_meta.json"] + \
        [f"/results/*/*/{t['name']}.json" for t in tdump["tasks"]]
    extra = set()
    for commit in (GIT_SNAPSHOT, GIT_HEAD):
        out = subprocess.run(g + ["ls-tree", "-r", "-z", "--name-only", commit, "--", "results/"],
                             check=True, capture_output=True, text=True, env=env).stdout
        for path in out.split("\0"):
            parts = path.split("/")
            if len(parts) == 4 and parts[3].endswith(".json") and parts[3] != "model_meta.json" \
                    and parts[3][:-5].lower() not in known:
                extra.add(path)
    patterns = "\n".join(pats + sorted(_gitignore_escape(p) for p in extra)) + "\n"
    subprocess.run(g + ["sparse-checkout", "init", "--no-cone"], check=True, env=env)
    subprocess.run(g + ["sparse-checkout", "set", "--no-cone", "--stdin"], input=patterns, text=True,
                   check=True, env=env)
    for commit in (GIT_SNAPSHOT, GIT_HEAD):
        subprocess.run(g + ["checkout", "--quiet", commit], check=True, env=env)
    print(f"Fetched raw inputs into {raw_dir} ({len(pats)} + {len(extra)} sparse patterns)")


# ----------------------------------------------------------------------------
# step 3: main build
# ----------------------------------------------------------------------------
def build(raw_dir, snapshot_dir, output_dir, min_tasks=2):
    raw_dir, snapshot_dir, output_dir = Path(raw_dir), Path(snapshot_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_manifests").mkdir(exist_ok=True)

    # --- snapshot -------------------------------------------------------------
    snap = np.load(snapshot_dir / "mteb.matrix.npz")
    snap_raw = np.load(snapshot_dir / "mteb.npz")
    columns = [str(x) for x in snap["benchmark_names"]]
    assert columns == [str(x) for x in snap_raw["benchmark_names"]]
    snap_models = [str(x) for x in snap["model_names"]]
    snap_raw_models = [str(x) for x in snap_raw["model_names"]]
    S = pd.DataFrame(np.where(snap["O"] > 0, snap["B"], np.nan), index=snap_models, columns=columns)
    S_raw = pd.DataFrame(np.where(snap_raw["O"] > 0, snap_raw["B"], np.nan), index=snap_raw_models, columns=columns)
    print(f"Snapshot: {len(snap_models)} models (matrix), {len(snap_raw_models)} (raw), {len(columns)} tasks")

    # --- mteb package dumps ----------------------------------------------------
    tdump = json.load(open(raw_dir / "mteb_pkg" / "mteb_eng_classic_tasks.json"))
    task_meta = {t["name"]: t for t in tdump["tasks"]}
    assert set(task_meta) == set(columns), "task set differs from snapshot columns"
    renamed = tdump.get("renamed_tasks", {})
    known_stems = {x.lower() for x in tdump["all_registered_task_names"]} | \
        {x.lower() for x in renamed} | {x.lower() for x in tdump.get("outdated_tasks", [])}
    mdump = json.load(open(raw_dir / "mteb_pkg" / "mteb_model_metas.json"))
    registry = {m["name"]: m for m in mdump["model_metas"]}

    # --- parquet (score source) ------------------------------------------------
    pq = [raw_dir / "hf_mteb_results" / p for p in PARQUETS]
    cols = ["model_name", "model_revision", "task_name", "split", "language", "subset", "score",
            "is_public", "trained_on"]
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in pq], ignore_index=True)
    n_rows_total = len(d)
    d = d[d.task_name.isin(columns)].reset_index(drop=True)
    print(f"Parquet: {n_rows_total} rows, {len(d)} rows for the 56 tasks, {d.model_name.nunique()} models")

    # per (model, revision, task) score, exactly as mteb_cell_score
    entries = {}
    trained = {}
    for r in d.itertuples(index=False):
        key = (r.model_name, r.model_revision, r.task_name)
        langs = list(r.language) if r.language is not None else []
        sc = None if r.score is None or (isinstance(r.score, float) and math.isnan(r.score)) else float(r.score)
        entries.setdefault(key, {}).setdefault(r.split, []).append(
            {"hf_subset": r.subset, "languages": langs, "main_score": sc})
        if r.trained_on:
            trained[(r.model_name, r.task_name)] = True
    recs = []
    for (m, rev, t), e in entries.items():
        s = mteb_cell_score(e, task_meta[t])
        if s is not None and not math.isnan(s):
            recs.append((m, rev, t, s))
    L = pd.DataFrame(recs, columns=["model_name", "model_revision", "task", "score"])
    L = L.sort_values(["model_name", "model_revision", "task"]).reset_index(drop=True)

    # --- git: trees, model_meta, first-seen --------------------------------------
    git = Git(raw_dir / "results_git")
    head_info, snap_info = git.commit_info(GIT_HEAD), git.commit_info(GIT_SNAPSHOT)
    assert head_info["sha"] == GIT_HEAD and snap_info["sha"] == GIT_SNAPSHOT
    tree_head, tree_snap = git.tree(GIT_HEAD), git.tree(GIT_SNAPSHOT)
    first, fp_order = git.first_added(GIT_HEAD)
    snap_pos = fp_order[GIT_SNAPSHOT]

    meta_head = tree_head[tree_head.file == "model_meta.json"]
    mm_blobs = git.blobs(meta_head.oid)
    dirrev_meta = {}
    for r in meta_head.itertuples():
        dirrev_meta[(r.model_dir, r.rev_dir)] = json.loads(mm_blobs[r.oid])

    def dirrev_name(model_dir, rev_dir, metas):
        j = metas.get((model_dir, rev_dir))
        if j is not None:
            return j["name"].replace("__", "/"), str(j["revision"]), True
        return model_dir.replace("__", "/"), rev_dir, False

    # map (model_name, model_revision) as used in the parquet -> (dir, rev) at HEAD
    head_dirrevs = tree_head[["model_dir", "rev_dir"]].drop_duplicates()
    name_rev_to_dirrev, name_to_dirrevs = {}, {}
    for r in head_dirrevs.itertuples():
        nm, rv, has = dirrev_name(r.model_dir, r.rev_dir, dirrev_meta)
        name_rev_to_dirrev.setdefault((nm, rv), []).append((r.model_dir, r.rev_dir, has))
        name_to_dirrevs.setdefault(nm, []).append((r.model_dir, r.rev_dir, has))
    first_by_dir, first56_by_dir = {}, {}
    task_file_set = {f"{t}.json" for t in columns}
    for (md, rd, f), rec in first.items():
        if md not in first_by_dir or rec[0] < first_by_dir[md][0]:
            first_by_dir[md] = rec
        if f in task_file_set and (md not in first56_by_dir or rec[0] < first56_by_dir[md][0]):
            first56_by_dir[md] = rec

    rev_has_meta = {}
    for (nm, rv), lst in name_rev_to_dirrev.items():
        rev_has_meta[(nm, rv)] = any(h for _, _, h in lst)
    unmapped = sorted({(m, r) for m, r in zip(L.model_name, L.model_revision)} - set(name_rev_to_dirrev))
    print(f"Parquet (model, revision) pairs not found in GitHub HEAD tree: {len(unmapped)}")

    # --- merge revisions per cell (deterministic) ------------------------------------
    L["has_meta"] = [rev_has_meta.get((m, r), False) for m, r in zip(L.model_name, L.model_revision)]
    ncomp = L.groupby(["model_name", "model_revision"]).task.nunique().rename("n_rev_tasks")
    L = L.join(ncomp, on=["model_name", "model_revision"])
    L = L.sort_values(["model_name", "task", "has_meta", "n_rev_tasks", "model_revision"],
                      ascending=[True, True, False, False, True], kind="mergesort")
    conflicts = L.groupby(["model_name", "task"]).score.nunique()
    n_conflicts = int((conflicts > 1).sum())
    conflict_list = [list(k) for k in conflicts[conflicts > 1].index]
    chosen = L.drop_duplicates(["model_name", "task"], keep="first")
    W = chosen.pivot(index="model_name", columns="task", values="score")
    W = W.reindex(columns=columns)
    n_obs = W.notna().sum(axis=1)
    W = W[n_obs >= min_tasks]
    W = W.loc[sorted(W.index)]
    W.index.name = "model"
    print(f"Models with >= {min_tasks} of the 56 tasks: {len(W)} "
          f"(dropped {(n_obs < min_tasks).sum()} with fewer; cells with conflicting revisions: {n_conflicts})")

    # --- consistency check A: pinned parquet vs snapshot cells -----------------------
    def compare(ref, new):
        common_m = [m for m in ref.index if m in new.index]
        a = ref.loc[common_m]
        b = new.reindex(index=common_m, columns=ref.columns)
        both = a.notna() & b.notna()
        diff = (a - b).abs()[both]
        n = int(both.values.sum())
        exact = int(((a == b) & both).values.sum())
        ulp2 = 2 * np.spacing(np.maximum(a.abs(), b.abs()))
        within = int((((a - b).abs() <= ulp2) & both).values.sum())
        return {
            "within_2ulp": within,
            "ref_models": int(len(ref)), "models_in_both": len(common_m),
            "ref_cells": int(ref.notna().values.sum()),
            "ref_cells_of_common_models": int(a.notna().values.sum()),
            "cells_in_both": n, "exact_match": exact,
            "exact_match_fraction": (exact / n) if n else None,
            "max_abs_diff": float(np.nanmax(diff.values)) if n else None,
            "ref_cells_missing_in_new": int((a.notna() & b.isna()).values.sum()),
            "new_cells_missing_in_ref": int((a.isna() & b.notna()).values.sum()),
        }, a, b

    # full upstream matrix without the min_tasks filter, for comparisons
    Wall = chosen.pivot(index="model_name", columns="task", values="score").reindex(columns=columns)
    chkA, a, b = compare(S_raw, Wall)
    mism = []
    for m in a.index:
        for t in columns:
            x, y = a.at[m, t], b.at[m, t]
            if pd.notna(x) and pd.notna(y) and x != y:
                mism.append({"model": m, "task": t, "snapshot": float(x), "pinned": float(y),
                             "abs_diff": float(abs(x - y)), "row_in_snapshot_matrix": m in set(snap_models)})
    chkA["absent_upstream_models"] = sorted(set(S_raw.index) - set(Wall.index))
    print(f"[A] parquet@{HF_REVISION[:8]} vs snapshot: {chkA['exact_match']}/{chkA['cells_in_both']} exact, "
          f"max|d|={chkA['max_abs_diff']:.4g}, {len(chkA['absent_upstream_models'])} snapshot models absent")

    # --- consistency check B: JSON at the snapshot commit reproduces the snapshot exactly ---
    Jsnap, snap_names = json_matrix_at(git, tree_snap, task_meta, renamed, True, known_stems)
    Js_nonan = Jsnap[~Jsnap.score.isna()].sort_values(["model_name", "task", "model_revision", "file"],
                                                       kind="mergesort")
    cand = Js_nonan.groupby(["model_name", "task"]).score.agg(lambda s: sorted(set(s)))
    ref_cells = S_raw.stack()
    chkB = {"ref_models": int(len(S_raw)), "ref_cells": int(len(ref_cells)),
            "reproduced_models": int(cand.index.get_level_values(0).nunique()),
            "reproduced_cells": int(len(cand))}
    chkB["model_set_equal"] = set(cand.index.get_level_values(0)) == set(S_raw.index)
    chkB["cell_set_equal"] = set(cand.index) == set(ref_cells.index)
    obs_cnt = cand.groupby(level=0).size()
    chkB["matrix_rows_after_le1_drop_equal"] = set(obs_cnt.index[obs_cnt >= 2]) == set(S.index)
    exact, diffs = 0, []
    for idx, val in ref_cells.items():
        if idx in cand.index:
            c = cand[idx]
            exact += int(val in c)
            diffs.append(min(abs(val - x) for x in c))
    chkB["exact_match"] = exact
    chkB["exact_match_fraction"] = exact / len(ref_cells)
    chkB["max_abs_diff"] = float(max(diffs)) if diffs else None
    amb = cand[cand.map(len) > 1]
    chkB["cells_with_multiple_candidate_values"] = []
    for (m, t), vals in amb.items():
        rows = Js_nonan[(Js_nonan.model_name == m) & (Js_nonan.task == t)]
        chkB["cells_with_multiple_candidate_values"].append({
            "model": m, "task": t, "snapshot": float(S_raw.at[m, t]) if m in S_raw.index else None,
            "candidates": [{"path": f"results/{r.model_dir}/{r.rev_dir}/{r.file}", "value": float(r.score)}
                           for r in rows.itertuples()],
            "note": "mteb.load_results returns one TaskResult per JSON file; pivot_table(aggfunc='first') "
                    "kept whichever came first in filesystem (APFS) order"})
    # the snapshot matrix file (mteb.matrix.npz) went through a CSV round trip in build_matrices.py
    # (pandas read_csv default float parser), so compare it with a 2-ulp tolerance
    ex_m, tol_m, n_m = 0, 0, 0
    for idx, val in S.stack().items():
        if idx in cand.index:
            n_m += 1
            ex_m += int(val in cand[idx])
            tol_m += int(any(abs(val - x) <= 2 * np.spacing(max(abs(val), abs(x))) for x in cand[idx]))
    chkB["vs_snapshot_matrix_npz"] = {"cells": int(S.notna().values.sum()), "cells_reproduced": n_m,
                                      "exact_match": ex_m, "within_2ulp": tol_m}
    chkB["exact_match_single_candidate_cells"] = int(sum(
        1 for idx, val in ref_cells.items() if idx in cand.index and len(cand[idx]) == 1 and val == cand[idx][0]))
    chkB["single_candidate_cells"] = int(sum(1 for idx in ref_cells.index if idx in cand.index
                                             and len(cand[idx]) == 1))
    print(f"[B] JSON@{GIT_SNAPSHOT[:8]} vs snapshot: {chkB['exact_match']}/{chkB['ref_cells']} cells equal a "
          f"candidate value, model set equal={chkB['model_set_equal']}, cell set equal={chkB['cell_set_equal']}, "
          f"{len(amb)} cells with duplicate result files")

    # explain check-A mismatches with git history of the JSON files that produced the snapshot value
    jsel = Js_nonan.set_index(["model_name", "task"])
    head_oid = {(r.model_dir, r.rev_dir, r.file): r.oid for r in tree_head.itertuples()}
    for mm in mism:
        try:
            rows = jsel.loc[[(mm["model"], mm["task"])]]
        except KeyError:
            continue
        ev = []
        for r in rows.itertuples():
            path = f"results/{r.model_dir}/{r.rev_dir}/{r.file}"
            ev.append({"path": path, "snapshot_oid": r.oid, "head_oid": head_oid.get((r.model_dir, r.rev_dir, r.file)),
                       "snapshot_json_value": float(r.score),
                       "commits_since_snapshot": git.path_log(GIT_SNAPSHOT, GIT_HEAD, path)})
        mm["evidence"] = ev
        mm["explained_by_upstream_change"] = any(e["snapshot_json_value"] == mm["snapshot"] and
                                                 e["snapshot_oid"] != e["head_oid"] for e in ev)
    chkA["mismatches"] = mism
    chkA["mismatches_explained_by_upstream_file_change"] = int(sum(m.get("explained_by_upstream_change", False)
                                                                   for m in mism))
    chkA["mismatches_in_snapshot_matrix_rows"] = int(sum(m["row_in_snapshot_matrix"] for m in mism))
    chkA["mismatches_note"] = ("data/mteb.npz (raw snapshot, 268 models) vs the pinned parquet under the same model "
                               "name; mismatches in rows that build_matrices.py dropped (<= 1 task) do not affect "
                               "data/mteb.matrix.npz (see check D for the matrix rows)")

    # --- snapshot models that are absent under their old name: renamed / moved? -----------------
    snap_name_to_dirrevs = {}
    for (md, rd), (nm, rv) in snap_names.items():
        snap_name_to_dirrevs.setdefault(nm, set()).add((md, rd))
    task_files = {f"{t}.json" for t in columns}
    head_by_fileoid = {}
    for r in tree_head.itertuples():
        if r.file in task_files:
            head_by_fileoid.setdefault((r.file, r.oid), set()).add((r.model_dir, r.rev_dir))
    renames, absent = {}, []
    for m in chkA["absent_upstream_models"]:
        rows = Js_nonan[Js_nonan.model_name == m]
        votes = {}
        for r in rows.itertuples():
            for md, rd in head_by_fileoid.get((r.file, r.oid), ()):
                nm = dirrev_name(md, rd, dirrev_meta)[0]
                votes[nm] = votes.get(nm, 0) + 1
        method, new = None, None
        if votes:
            new = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            method = "identical_result_blobs"
        else:  # value-based: identical values on >= 3 common cells and >= 90% of common cells
            best = None
            for cand_m in Wall.index:
                if cand_m in S_raw.index:
                    continue
                a_, b_ = S_raw.loc[m], Wall.loc[cand_m]
                both = a_.notna() & b_.notna()
                n_b = int(both.sum())
                n_eq = int((a_[both] == b_[both]).sum())
                if n_b >= 3 and n_eq >= 0.9 * n_b and (best is None or n_eq > best[1]):
                    best = (cand_m, n_eq, n_b)
            if best:
                new, method = best[0], f"identical_values_{best[1]}_of_{best[2]}_common_cells"
        logs = []
        for md in sorted({md for md, _ in snap_name_to_dirrevs.get(m, set())}):
            logs += git.path_log(GIT_SNAPSHOT, GIT_HEAD, f"results/{md}")
        detail = {"model": m, "snapshot_dirs": sorted(f"{a}/{b}" for a, b in snap_name_to_dirrevs.get(m, set())),
                  "commits_since_snapshot": logs[:5], "current_name": new, "match_method": method,
                  "matching_blobs": votes.get(new) if votes else None, "n_snapshot_cells": int(len(rows))}
        if new is not None and new in Wall.index:
            a_, b_ = S_raw.loc[m], Wall.loc[new]
            both = a_.notna() & b_.notna()
            detail["common_cells"] = int(both.sum())
            detail["equal_cells"] = int((a_[both] == b_[both]).sum())
            renames[m] = new
        absent.append(detail)
    chkA["absent_upstream_detail"] = absent
    rename_inv = {v: k for k, v in renames.items()}
    assert len(rename_inv) == len(renames), "two snapshot models map to one current name"
    print(f"Renamed snapshot models: {len(renames)} of {len(chkA['absent_upstream_models'])} absent")

    # --- consistency check C: JSON at HEAD reproduces the parquet ----------------------
    Jhead, _ = json_matrix_at(git, tree_head, task_meta, renamed, False, known_stems)
    jh = Jhead[~Jhead.score.isna()].groupby(["model_name", "model_revision", "task"]).score.agg(
        lambda s: sorted(set(s)))
    lp = L.set_index(["model_name", "model_revision", "task"]).score
    common = lp.index.intersection(jh.index)
    chkC = {"parquet_cells": int(len(lp)), "json_head_cells": int(len(jh)), "cells_in_both": int(len(common)),
            "exact_match": int(sum(lp[i] in jh[i] for i in common)),
            "max_abs_diff": float(max(min(abs(lp[i] - x) for x in jh[i]) for i in common)) if len(common) else None,
            "json_cells_with_duplicate_files": int((jh.map(len) > 1).sum()),
            "parquet_only": int(len(lp.index.difference(jh.index))),
            "json_only": int(len(jh.index.difference(lp.index)))}
    chkC["json_only_examples"] = [list(x) for x in list(jh.index.difference(lp.index))[:10]]
    chkC["parquet_only_examples"] = [list(x) for x in list(lp.index.difference(jh.index))[:10]]
    print(f"[C] JSON@{GIT_HEAD[:8]} vs parquet: {chkC['exact_match']}/{chkC['cells_in_both']} exact, "
          f"parquet-only={chkC['parquet_only']}, json-only={chkC['json_only']}")

    # --- duplicate rows: non-snapshot rows whose results are copies of snapshot-matrix rows ----------
    # Evidence 1: identical git blob oids (byte-identical result files) for task files of this model
    # and task files of a snapshot-matrix row, at the snapshot commit or at HEAD.
    # Evidence 2: identical values on the common observed cells of W.
    snap_set_m = set(snap_models)
    blob_owner = {"snapshot_commit": {}, "head": {}}
    for r in tree_snap[tree_snap.file.isin(task_files)].itertuples():
        nm = snap_names.get((r.model_dir, r.rev_dir), (r.model_dir.replace("__", "/"), r.rev_dir))[0]
        if nm in snap_set_m:
            blob_owner["snapshot_commit"].setdefault((r.file, r.oid), set()).add(nm)
    for r in tree_head[tree_head.file.isin(task_files)].itertuples():
        nm = dirrev_name(r.model_dir, r.rev_dir, dirrev_meta)[0]
        nm = rename_inv.get(nm, nm)
        if nm in snap_set_m:
            blob_owner["head"].setdefault((r.file, r.oid), set()).add(nm)
    head_task_files = {}
    for r in tree_head[tree_head.file.isin(task_files)].itertuples():
        head_task_files.setdefault((r.model_dir, r.rev_dir), set()).add((r.file, r.oid))
    Wv, Wo = W.values, W.notna().values
    w_row = {m: i for i, m in enumerate(W.index)}
    snap_rows_in_W = {rename_inv.get(m, m): w_row[m] for m in W.index if rename_inv.get(m, m) in snap_set_m}

    def duplicate_info(m):
        mine = set()
        for md, rd, _ in name_to_dirrevs.get(m, []):
            mine |= head_task_files.get((md, rd), set())
        n_files = len({f for f, _ in mine})
        votes = {}
        for where, owner in blob_owner.items():
            per = {}
            for f, oid in mine:
                for s in owner.get((f, oid), ()):
                    per.setdefault(s, set()).add(f)
            for s, fs in per.items():
                votes.setdefault(s, {"snapshot_commit": 0, "head": 0})[where] = len(fs)
        i = w_row[m]
        vals = {}
        for s, j in snap_rows_in_W.items():
            both = Wo[i] & Wo[j]
            nb = int(both.sum())
            ne = int((Wv[i][both] == Wv[j][both]).sum())
            if ne:
                vals[s] = (ne, nb)
        n_tasks = int(Wo[i].sum())
        best, best_key = None, None
        for s in sorted(set(votes) | set(vals)):
            nblob = max(votes.get(s, {}).values(), default=0)
            ne, nb = vals.get(s, (0, 0))
            key = (nblob, ne)
            if best_key is None or key > best_key:
                best, best_key = s, key
        if best is None:
            return None, None
        nblob = max(votes.get(best, {}).values(), default=0)
        ne, nb = vals.get(best, (0, 0))
        ev = {"snapshot_row": best,
              "identical_result_blobs_vs_snapshot_commit": votes.get(best, {}).get("snapshot_commit", 0),
              "identical_result_blobs_vs_head": votes.get(best, {}).get("head", 0),
              "task_result_files_of_this_model": n_files,
              "equal_values": ne, "common_cells": nb, "n_tasks": n_tasks}
        is_dup = (nblob >= 3 and nblob >= 0.5 * n_files) or (nb >= 3 and ne >= 0.5 * nb)
        return (best if is_dup else None), ev

    # --- consistency check E: near-identical row pairs anywhere in the matrix -------------------
    eq = (Wv[:, None, :] == Wv[None, :, :]) & Wo[:, None, :] & Wo[None, :, :]
    co = Wo[:, None, :] & Wo[None, :, :]
    n_eq, n_co = eq.sum(-1), co.sum(-1)
    pairs = []
    for i, j in zip(*np.nonzero(np.triu((n_co >= 3) & (n_eq >= 0.5 * n_co), 1))):
        a_, b_ = W.index[i], W.index[j]
        pairs.append({"a": a_, "b": b_, "equal_cells": int(n_eq[i, j]), "common_cells": int(n_co[i, j]),
                      "a_in_snapshot": rename_inv.get(a_, a_) in snap_set_m,
                      "b_in_snapshot": rename_inv.get(b_, b_) in snap_set_m})
    chkE = {"rule": "pairs of rows with >= 3 common observed cells and exactly equal values on >= 50% of them",
            "n_pairs": len(pairs), "pairs": pairs,
            "note": "pairs with one row outside the snapshot are flagged per model (duplicate_of_snapshot_model); "
                    "pairs of two snapshot rows already exist in data/mteb.matrix.npz and are reported only"}
    del eq, co
    print(f"[E] near-identical row pairs: {len(pairs)}: " + ", ".join(f"{p['a']} ~ {p['b']}" for p in pairs))

    # --- per-model metadata ------------------------------------------------------------
    snap_set, snap_raw_set = set(snap_models), set(snap_raw_models)
    # models present at the snapshot commit (any naming / with or without model_meta.json)
    Jsnap_any, _ = json_matrix_at(git, tree_snap, task_meta, renamed, False, known_stems)
    snap_any_cnt = Jsnap_any[~Jsnap_any.score.isna()].groupby("model_name").task.nunique()
    snap_meta_cnt = Js_nonan.groupby("model_name").task.nunique()
    mtebver = Jhead.groupby("model_name").mteb_version.agg(lambda s: sorted(set(s)))

    def to_utc(s):
        return _dt.datetime.fromisoformat(s).astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    models_meta = []
    for m in W.index:
        cm = chosen[chosen.model_name == m]
        rev_tasks = {}
        for r in cm.itertuples():
            rev_tasks.setdefault(r.model_revision, []).append(columns.index(r.task))
        former = rename_inv.get(m)  # snapshot name of a renamed / moved snapshot model
        dirrevs = sorted(name_to_dirrevs.get(m, []))
        # model directories that hold (or held at the snapshot commit) results under this name
        hist_dirs = {md for md, _, _ in dirrevs} | {md for md, _ in snap_name_to_dirrevs.get(former or m, set())}
        fs = [first_by_dir[md] for md in hist_dirs if md in first_by_dir]
        fs56 = [first56_by_dir[md] for md in hist_dirs if md in first56_by_dir]
        first_seen = min(fs) if fs else None  # (position, sha, date): earliest along first-parent history
        first_seen56 = min(fs56) if fs56 else None
        reg = registry.get(m) or (registry.get(former) if former else None)
        rm_dates = sorted({str(dirrev_meta[(md, rd)].get("release_date")) for md, rd, has in dirrevs
                           if has and dirrev_meta[(md, rd)].get("release_date")})
        rd_pkg = reg.get("release_date") if reg else None
        rd_res = rm_dates[0] if rm_dates else None
        rd_first = to_utc(first_seen[2])[:10] if first_seen else None
        if rd_pkg:
            rd, src = rd_pkg, "mteb_package_ModelMeta"
        elif rd_res:
            rd, src = rd_res, "results_repo_model_meta_json"
        elif rd_first:
            rd, src = rd_first, "results_repo_first_seen"
        else:
            rd, src = None, None
        snap_name = m if m in snap_raw_set else former
        if m in snap_set:
            status = "in_snapshot"
        elif former in snap_set:
            status = "in_snapshot_renamed"
        elif snap_name in snap_raw_set:
            status = "snapshot_raw_dropped_le1_task"
        elif int(snap_any_cnt.get(m, 0)) == 0:
            status = "added_after_snapshot"
        elif int(snap_meta_cnt.get(m, 0)) == 0:
            status = "pre_snapshot_excluded_no_model_meta_json"
        else:
            status = "pre_snapshot_other"
        in_snap = snap_name in snap_set
        dup, dup_ev = (None, None) if in_snap else duplicate_info(m)
        fs_utc = to_utc(first_seen[2]) if first_seen else None
        models_meta.append({
            "name": m,
            "in_snapshot": in_snap,
            "in_snapshot_raw": snap_name in snap_raw_set,
            "snapshot_name": snap_name,
            "status": status,
            "duplicate_of_snapshot_model": dup,
            "duplicate_evidence": dup_ev if dup else None,
            "recommended_prospective": status == "added_after_snapshot" and dup is None,
            "n_tasks": int(W.loc[m].notna().sum()),
            "release_date": rd,
            "release_date_source": src,
            "release_date_mteb_package": rd_pkg,
            "release_date_results_model_meta": rd_res,
            "release_date_after_first_seen": (rd[:10] > fs_utc[:10]) if (rd and fs_utc) else None,
            "first_seen_results_repo": fs_utc,
            "first_seen_results_repo_commit": first_seen[1] if first_seen else None,
            "first_seen_56task_results": to_utc(first_seen56[2]) if first_seen56 else None,
            "first_seen_after_snapshot_commit": (first_seen[0] > snap_pos) if first_seen else None,
            "first_seen_56task_after_snapshot_commit": (first_seen56[0] > snap_pos) if first_seen56 else None,
            "snapshot_commit_n_tasks_any": int(snap_any_cnt.get(snap_name or m, 0)),
            "snapshot_commit_n_tasks_with_model_meta": int(snap_meta_cnt.get(snap_name or m, 0)),
            "has_model_meta_json": any(h for _, _, h in dirrevs),
            "in_mteb_package_registry": reg is not None,
            "mteb_package_revision": reg.get("revision") if reg else None,
            "n_parameters": reg.get("n_parameters") if reg else None,
            "open_weights": reg.get("open_weights") if reg else None,
            "result_revisions": {k: sorted(v) for k, v in sorted(rev_tasks.items())},
            "result_dirs": sorted({f"{md}/{rd}" for md, rd, _ in dirrevs}),
            "result_mteb_versions": mtebver.get(m, []),
            "trained_on_tasks": [i for i, t in enumerate(columns) if trained.get((m, t))],
        })

    # --- save matrix -------------------------------------------------------------------------
    save_matrix(NAME, W, output_dir)

    is_new = np.array([not x["in_snapshot"] for x in models_meta])
    is_added = np.array([x["status"] == "added_after_snapshot" for x in models_meta])
    late = np.array([bool(x["release_date"]) and x["release_date"] > RELEASE_CUTOFF for x in models_meta])
    late_mm = np.array([bool(x["release_date_mteb_package"] or x["release_date_results_model_meta"]) and
                        (x["release_date_mteb_package"] or x["release_date_results_model_meta"]) > RELEASE_CUTOFF
                        for x in models_meta])
    O = W.notna().values

    def fill(mask):
        return float(O[mask].mean()) if mask.any() else None

    # snapshot rows: changes between snapshot matrix and the pinned rebuild
    W_snapnames = W.rename(index={k: rename_inv[k] for k in W.index if k in rename_inv})
    chkD, a3, b3 = compare(S, W_snapnames)
    chkD["note"] = "renamed models compared under their snapshot name"
    chkD["renamed"] = {k: v for k, v in sorted(renames.items())}
    # mteb.matrix.npz went through a CSV round trip (pandas default float parser): errors up to ~1e-16
    # absolute on these 0-1 scores, i.e. sometimes several ulp for small values; 1e-12 separates
    # those from real upstream changes
    both3 = a3.notna() & b3.notna()
    absd3 = (a3 - b3).abs()
    changed3 = both3 & (absd3 > 1e-12)
    chkD["changed_threshold_abs"] = 1e-12
    chkD["changed_cells"] = [
        {"model": m, "task": t, "snapshot_matrix": float(a3.at[m, t]), "mteb2026": float(b3.at[m, t]),
         "abs_diff": float(absd3.at[m, t])}
        for m in a3.index for t in columns if changed3.at[m, t]]
    chkD["n_changed_cells"] = len(chkD["changed_cells"])
    chkD["n_models_with_changed_cells"] = len({c["model"] for c in chkD["changed_cells"]})
    chkD["max_abs_diff_other_common_cells"] = float(np.nanmax(absd3[both3 & ~changed3].values))
    chkD["snapshot_cells_missing_in_mteb2026"] = [
        {"model": m, "task": t} for m in a3.index for t in columns if pd.notna(a3.at[m, t]) and pd.isna(b3.at[m, t])]
    status_counts = pd.Series([x["status"] for x in models_meta]).value_counts().sort_index().to_dict()
    counts = {
        "models": int(len(W)), "tasks": len(columns),
        "fill_fraction": float(O.mean()),
        "snapshot_models_matrix": len(snap_models),
        "snapshot_models_present": int((~is_new).sum()),
        "snapshot_models_present_renamed": int(sum(x["status"] == "in_snapshot_renamed" for x in models_meta)),
        "snapshot_models_missing": sorted(set(snap_models) - {x["snapshot_name"] for x in models_meta
                                                              if x["in_snapshot"]}),
        "snapshot_rows_fill_fraction": fill(~is_new),
        "new_models": int(is_new.sum()),
        "new_models_fill_fraction": fill(is_new),
        "new_models_status": {k: int(v) for k, v in status_counts.items()
                              if k not in ("in_snapshot", "in_snapshot_renamed")},
        "added_after_snapshot": int(is_added.sum()),
        "added_after_snapshot_fill_fraction": fill(is_added),
        f"new_models_release_date_after_{RELEASE_CUTOFF}": int((is_new & late).sum()),
        f"added_after_snapshot_and_release_date_after_{RELEASE_CUTOFF}": int((is_added & late).sum()),
        f"new_models_release_date_after_{RELEASE_CUTOFF}_fill_fraction": fill(is_new & late),
        f"new_models_modelmeta_release_date_after_{RELEASE_CUTOFF}": int((is_new & late_mm).sum()),
        "new_models_release_date_source": pd.Series([x["release_date_source"] for x in models_meta])[is_new]
        .fillna("none").value_counts().sort_index().to_dict(),
        "new_models_with_release_date": int(sum(1 for x, n in zip(models_meta, is_new) if n and x["release_date"])),
        f"added_after_snapshot_release_date_on_or_before_{RELEASE_CUTOFF}": int((is_added & ~late).sum()),
        "duplicates_of_snapshot_rows": [x["name"] for x in models_meta if x["duplicate_of_snapshot_model"]],
        "release_date_after_first_seen": [x["name"] for x in models_meta if x["release_date_after_first_seen"]],
    }
    is_rec = np.array([x["recommended_prospective"] for x in models_meta])
    counts["recommended_prospective"] = int(is_rec.sum())
    counts["recommended_prospective_fill_fraction"] = fill(is_rec)
    counts[f"recommended_prospective_and_release_date_after_{RELEASE_CUTOFF}"] = int((is_rec & late).sum())
    counts[f"recommended_prospective_and_release_date_after_{RELEASE_CUTOFF}_fill_fraction"] = fill(is_rec & late)
    print(json.dumps(counts, indent=1))
    prospective = {
        "rule": "status == 'added_after_snapshot' and duplicate_of_snapshot_model is null "
                "(per-model field recommended_prospective)",
        "why": "status is defined from the git history of the results repo (no 56-task result at the snapshot "
               f"commit {GIT_SNAPSHOT[:8]}), which is reliable; release_date is author-supplied and sometimes later "
               "than the first appearance of the results (release_date_after_first_seen) or much earlier than the "
               "snapshot. Duplicates re-use the result files of snapshot rows and would leak training rows.",
        "n_models": counts["recommended_prospective"],
        "fill_fraction": counts["recommended_prospective_fill_fraction"],
        "models": [x["name"] for x in models_meta if x["recommended_prospective"]],
        "excluded_duplicates": {x["name"]: x["duplicate_of_snapshot_model"] for x in models_meta
                                if x["duplicate_of_snapshot_model"]},
        "optional_release_date_filter": {
            "rule": f"additionally release_date > {RELEASE_CUTOFF}",
            "n_models": counts[f"recommended_prospective_and_release_date_after_{RELEASE_CUTOFF}"],
            "fill_fraction": counts[f"recommended_prospective_and_release_date_after_{RELEASE_CUTOFF}_fill_fraction"],
            "models": [x["name"] for x, l in zip(models_meta, late) if x["recommended_prospective"] and l]},
    }

    # --- licenses ---------------------------------------------------------------------------
    readme_path = raw_dir / "hf_mteb_results" / "README.md"
    hf_lic = hf_card_license(readme_path.read_text())
    lic_oid = git.path_oid(GIT_HEAD, "LICENSE")
    lic_oid_snap = git.path_oid(GIT_SNAPSHOT, "LICENSE")
    lic_bytes = git.blobs([lic_oid])[lic_oid] if lic_oid else None
    lic_text = lic_bytes.decode("utf-8") if lic_bytes is not None else None
    lic_added = [dict(zip(["sha", "committer_date", "title"], l.split("\t", 2))) for l in git.run(
        "log", "--diff-filter=A", "--format=%H%x09%cI%x09%s", GIT_HEAD, "--", "LICENSE").splitlines() if l]
    pkg_lic = json.load(open(raw_dir / "mteb_pkg" / "mteb_package_license.json"))
    licenses = {
        "scores_hf_mteb_results": {
            "source": f"huggingface.co/datasets/{HF_REPO}@{HF_REVISION[:8]}", "license": hf_lic,
            "evidence": "license field of the dataset card YAML front matter (hf_mteb_results/README.md, sha256 "
                        f"{sha256_file(readme_path)})"},
        "results_repo_github": {
            "source": f"{GITHUB_URL}@{GIT_HEAD[:8]}", "path": "LICENSE", "git_oid": lic_oid,
            "git_oid_at_snapshot_commit": lic_oid_snap,
            "license_file_added_in": lic_added,
            "note": None if lic_oid_snap else "the repo had no LICENSE file at the snapshot commit; this LICENSE "
                                              "(added later, see license_file_added_in) is the one at the pinned "
                                              "HEAD, the state this matrix is built from",
            "sha256": hashlib.sha256(lic_bytes).hexdigest() if lic_bytes is not None else None,
            "size": len(lic_bytes) if lic_bytes is not None else None,
            "license_detected": spdx_by_keyword(lic_text), "text": lic_text},
        "mteb_package": {
            "source": f"pypi mteb=={pkg_lic['mteb_version']} (task metadata and ModelMeta dumps)",
            "license": pkg_lic.get("license_expression") or pkg_lic.get("license"),
            "license_classifiers": pkg_lic.get("license_classifiers"),
            "license_files": [{"path": f["path"], "sha256": f["sha256"],
                               "license_detected": spdx_by_keyword(f["text"])} for f in pkg_lic["license_files"]]},
        "task_datasets": "the matrix contains only evaluation scores, no task data; each task's dataset license "
                         "as declared in mteb TaskMetadata is given per benchmark (benchmarks[].dataset_license)",
        "model_weights": "not recorded here (the scores are not model weights); see mteb ModelMeta.license",
    }
    print("Licenses: HF card", hf_lic, "| GitHub LICENSE", licenses["results_repo_github"]["license_detected"],
          lic_oid, "| mteb package", licenses["mteb_package"]["license"])

    # --- raw manifest -------------------------------------------------------------------------
    hf_fetched = read_text(raw_dir / "hf_fetched_at.txt")
    inputs = []
    for p in PARQUETS + ["README.md"]:
        fp = raw_dir / "hf_mteb_results" / p
        inputs.append({"name": f"hf:{HF_REPO}/{p}", "local_path": f"hf_mteb_results/{p}",
                       "url": f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{p}",
                       "revision": HF_REVISION, "size": fp.stat().st_size, "sha256": sha256_file(fp),
                       "fetched_at": hf_fetched, "license": hf_lic})
    fp = raw_dir / "hf_commits.json"
    n_hf_commits = len(json.load(open(fp)))
    inputs.append({"name": f"hf:{HF_REPO} commit list (API)", "local_path": "hf_commits.json",
                   "url": f"https://huggingface.co/api/datasets/{HF_REPO}/commits/main"
                          f"?p=0..{n_hf_commits // 50}&limit=50",
                   "revision": HF_REVISION, "size": fp.stat().st_size, "sha256": sha256_file(fp),
                   "fetched_at": read_text(raw_dir / "hf_commits_fetched_at.txt"), "n_commits": n_hf_commits,
                   "license": hf_lic,
                   "note": "HF dataset history starts 2026-06-02; used only to document that it cannot "
                           "reproduce the March-2026 snapshot. The list grows with every new upstream commit, "
                           "so its sha256 identifies this fetch only."})
    packs = []
    for pp in sorted(glob.glob(str(raw_dir / "results_git" / ".git" / "objects" / "pack" / "*.pack"))):
        packs.append({"file": os.path.basename(pp), "size": os.path.getsize(pp), "sha256": sha256_file(pp)})
    used = pd.concat([tree_head.assign(commit=GIT_HEAD), tree_snap.assign(commit=GIT_SNAPSHOT)])
    ustem = used.file.str[:-5].str.lower()
    used = used[(used.file == "model_meta.json") | (used.file.str.endswith(".json") &
                (ustem.isin({c.lower() for c in columns}) | ~ustem.isin(known_stems)))]
    listing = "\n".join(sorted(f"{c} results/{a}/{b}/{f} {o}" for c, a, b, f, o in
                               zip(used.commit, used.model_dir, used.rev_dir, used.file, used.oid)))
    inputs.append({"name": "github:embeddings-benchmark/results (blobless clone + sparse checkout of "
                           "model_meta.json, the 56 task JSONs and result files not named after a registered "
                           "mteb task, at both commits)",
                   "local_path": "results_git", "url": GITHUB_URL + ".git",
                   "revision": GIT_HEAD, "snapshot_commit": GIT_SNAPSHOT,
                   "head_commit_info": head_info, "snapshot_commit_info": snap_info,
                   "fetched_at": read_text(raw_dir / "github_clone_started_at.txt"),
                   "size": int(sum(p["size"] for p in packs)), "pack_files": packs,
                   "blobs_used": int(len(used)),
                   "sha256": hashlib.sha256(listing.encode()).hexdigest(),
                   "sha256_note": "sha256 of the sorted '<commit> <path> <git blob oid>' listing of all "
                                  "blobs read at both commits (git oids are content hashes); together with the "
                                  "two commit shas this is the reproducible identity of the clone content",
                   "pack_files_note": "pack file names, count, sizes and sha256 depend on server-side pack "
                                      "generation and on the number of fetches (one per sparse checkout); they "
                                      "identify this local copy only and differ on every re-fetch",
                   "license": licenses["results_repo_github"]["license_detected"]})
    inputs.append({"name": "github:embeddings-benchmark/results LICENSE", "local_path": "results_git (git blob)",
                   "url": f"{GITHUB_URL}/blob/{GIT_HEAD}/LICENSE", "revision": GIT_HEAD,
                   "git_oid": lic_oid, "size": licenses["results_repo_github"]["size"],
                   "sha256": licenses["results_repo_github"]["sha256"],
                   "fetched_at": read_text(raw_dir / "github_clone_started_at.txt"),
                   "license": licenses["results_repo_github"]["license_detected"]})
    for fn, note in [("mteb_eng_classic_tasks.json", "MTEB(eng, classic) task metadata"),
                     ("mteb_model_metas.json", "ModelMeta registry (name, revision, release_date, ...)"),
                     ("mteb_package_license.json", "mteb package license metadata/files and task dataset licenses")]:
        fp = raw_dir / "mteb_pkg" / fn
        inputs.append({"name": f"mteb=={tdump['mteb_version']} package dump: {fn}", "local_path": f"mteb_pkg/{fn}",
                       "url": f"https://pypi.org/project/mteb/{tdump['mteb_version']}/",
                       "revision": tdump["mteb_version"], "size": fp.stat().st_size, "sha256": sha256_file(fp),
                       "fetched_at": _dt.datetime.fromtimestamp(fp.stat().st_mtime, _dt.timezone.utc)
                       .strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "license": licenses["mteb_package"]["license"],
                       "note": note + "; generated offline (HF_HUB_OFFLINE=1) from the installed package, "
                                      "not downloaded"})
    manifest = {"name": NAME,
                "fetched_by": "build_mteb2026.py --dump-mteb-pkg (mteb venv, offline) then --fetch into an empty "
                              "raw dir; anonymous (no HF token, git credential helpers disabled)",
                "licenses": {"hf:" + HF_REPO: hf_lic,
                             "github:embeddings-benchmark/results": licenses["results_repo_github"]["license_detected"],
                             f"pypi:mteb=={tdump['mteb_version']}": licenses["mteb_package"]["license"]},
                "inputs": inputs}
    with open(output_dir / "raw_manifests" / f"{NAME}.json", "w") as f:
        json.dump(_clean(manifest), f, indent=1, sort_keys=False)
        f.write("\n")

    # --- meta ----------------------------------------------------------------------------------
    task_sizes = {}
    ts_path = snapshot_dir / "mteb_task_sizes.json"
    if ts_path.exists():
        task_sizes = json.load(open(ts_path))
    benchmarks = []
    for t in columns:
        tm = task_meta[t]
        benchmarks.append({"name": t, "type": tm["type"], "main_score": tm["main_score"],
                           "eval_splits": tm["eval_splits"], "hf_subsets": tm["hf_subsets"],
                           "is_aggregate": tm["is_aggregate"], "dataset_path": tm["dataset_path"],
                           "dataset_revision": tm["dataset_revision"],
                           "dataset_license": pkg_lic["task_dataset_licenses"].get(t),
                           "n_items": task_sizes.get(t),
                           "n_models_observed": int(W[t].notna().sum())})
    meta = {
        "name": NAME,
        "description": "MTEB(eng, classic) 56-task matrix rebuilt from a pinned post-snapshot state of the MTEB "
                       "results, with every model that has >= 2 of the 56 tasks. Rows flagged in_snapshot "
                       "are the models of data/mteb.matrix.npz; the others are candidates for a prospective "
                       "(temporal) test set.",
        "shape": list(W.shape),
        "row_order": "model names sorted by Unicode code point (Python sorted), as in mteb.matrix.npz",
        "column_order": "identical to data/mteb.matrix.npz benchmark_names",
        "score_definition": {
            "value": "mteb TaskResult.get_score() after validate_and_filter_scores(task) with the "
                     "MTEB(eng, classic) task objects: mean of main_score over eval_splits x hf_subsets "
                     "(one entry per task here); missing split/subset -> missing cell; native 0-1 scale.",
            "revision_merge": "per cell, first available value over result revisions ordered by "
                              "(has model_meta.json desc, #complete tasks desc, revision asc); "
                              "snapshot used pivot_table(aggfunc='first') in filesystem order",
            "cells_with_conflicting_revision_values": n_conflicts,
            "conflicting_cells": conflict_list,
            "model_inclusion": f">= {min_tasks} observed of the 56 tasks; models without model_meta.json in "
                               "the results repo are included (snapshot required it) and flagged via "
                               "has_model_meta_json",
            "same_as": "data/download_mteb.py (mteb.load_results + TaskResult.get_score) + "
                       "data/build_matrices.py (drop models with <= 1 task)",
        },
        "sources": {
            "scores": {"hf_dataset": HF_REPO, "revision": HF_REVISION,
                       "files": [{"path": i["local_path"], "sha256": i["sha256"], "size": i["size"]}
                                 for i in inputs[:4]],
                       "fetched_at": hf_fetched},
            "results_repo": {"url": GITHUB_URL, "head_commit": head_info, "snapshot_commit": snap_info,
                             "snapshot_built_at": SNAPSHOT_BUILT_AT,
                             "fetched_at": read_text(raw_dir / "github_clone_started_at.txt")},
            "mteb_package": {"version": tdump["mteb_version"], "benchmark": tdump["benchmark"],
                             "n_model_metas": mdump["n_model_metas"]},
            "raw_manifest": f"raw_manifests/{NAME}.json",
            "raw_inputs_sha256": {i["local_path"]: i["sha256"] for i in inputs},
        },
        "release_date_rule": "release_date = mteb package ModelMeta.release_date if the model name is in the "
                             f"mteb=={tdump['mteb_version']} registry; else release_date from the results repo "
                             "model_meta.json (a serialized ModelMeta) at the pinned HEAD; else the UTC date of "
                             "the first results-repo commit adding any file of the model (left-censored at the "
                             "repo's first commit, 2024-07); else null.",
        "status_values": {
            "in_snapshot": "row of data/mteb.matrix.npz",
            "in_snapshot_renamed": "row of data/mteb.matrix.npz whose results were later moved to a new model "
                                   "name (snapshot_name gives the old name); in_snapshot is True",
            "snapshot_raw_dropped_le1_task": "in data/mteb.npz but dropped by build_matrices.py (<= 1 task then)",
            "added_after_snapshot": f"no 56-task result in the results repo at the snapshot commit {GIT_SNAPSHOT[:8]}",
            "pre_snapshot_excluded_no_model_meta_json": "had 56-task results at the snapshot commit only in "
                                                        "revision dirs without model_meta.json "
                                                        "(mteb.load_results(require_model_meta=True) skipped them)",
            "pre_snapshot_other": "had 56-task results at the snapshot commit under this name but was not in "
                                  "the snapshot for another reason",
        },
        "duplicate_rule": "duplicate_of_snapshot_model = a row of data/mteb.matrix.npz (snapshot name) such that "
                          ">= 3 and >= 50% of this model's task result files are byte-identical (same git blob "
                          "oid) to that row's files at the snapshot commit or at HEAD, or such that the two rows "
                          "have equal values on >= 50% of >= 3 common cells; evaluated for rows not in_snapshot",
        "prospective_test_set": prospective,
        "licenses": licenses,
        "counts": counts,
        "consistency_checks": {
            "A_pinned_parquet_vs_snapshot_raw": chkA,
            "B_json_at_snapshot_commit_vs_snapshot_raw": chkB,
            "C_json_at_head_vs_pinned_parquet": chkC,
            "D_snapshot_matrix_rows_vs_mteb2026": chkD,
            "E_near_identical_row_pairs": chkE,
        },
        "caveats": [
            "Use prospective_test_set (status added_after_snapshot, duplicates removed) for temporal evaluation; "
            f"in_snapshot=False also contains {int((is_new & ~is_added).sum())} old models (no model_meta.json / "
            "dropped with <= 1 task).",
            "Duplicates of snapshot rows (excluded from prospective_test_set): " + ("; ".join(
                f"{k} re-uses the result files of {v}" for k, v in prospective["excluded_duplicates"].items())
                or "none") + ".",
            "release_date is taken as published and is not a reliable temporal filter on its own: "
            f"{len(counts['release_date_after_first_seen'])} rows have release_date later than the first "
            "appearance of their results in the results repo (release_date_after_first_seen), and "
            f"{counts[f'added_after_snapshot_release_date_on_or_before_{RELEASE_CUTOFF}']} added_after_snapshot "
            f"models have release_date <= {RELEASE_CUTOFF}.",
            f"{len(renames)} snapshot models now appear under new names (status in_snapshot_renamed): "
            "join with data/mteb.matrix.npz via snapshot_name.",
            "Snapshot rows carry the current upstream values: see consistency_checks.D for the "
            f"{chkD['n_changed_cells']} changed cells ({chkD['n_models_with_changed_cells']} models), "
            f"{len(chkD['snapshot_cells_missing_in_mteb2026'])} lost and {chkD['new_cells_missing_in_ref']} added "
            f"cells; all other common cells agree to {chkD['max_abs_diff_other_common_cells']:.2g} "
            "(data/mteb.matrix.npz went through a CSV round trip).",
        ],
        "benchmarks": benchmarks,
        "models": models_meta,
    }
    with open(output_dir / f"{NAME}.meta.json", "w") as f:
        json.dump(_clean(meta), f, indent=1)
        f.write("\n")
    print(f"Wrote {output_dir / (NAME + '.matrix.npz')}, .csv, .meta.json and raw_manifests/{NAME}.json")
    return W, meta


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("~/benchselect-data/raw/mteb2026").expanduser(),
                   help="Directory with hf_mteb_results/, results_git/, mteb_pkg/ and fetch timestamps.")
    p.add_argument("--snapshot-dir", type=Path, default=here,
                   help="Directory with mteb.matrix.npz, mteb.npz and mteb_task_sizes.json (default: data/).")
    p.add_argument("--output-dir", type=Path, default=here, help="Directory for the generated files.")
    p.add_argument("--min-tasks", type=int, default=2, help="Minimum number of observed tasks per model.")
    p.add_argument("--dump-mteb-pkg", action="store_true",
                   help="Step 1: write RAW/mteb_pkg/*.json from the installed mteb package and exit.")
    p.add_argument("--fetch", action="store_true", help="Step 2: download the pinned raw inputs into RAW and exit.")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.dump_mteb_pkg:
        dump_mteb_pkg(a.raw_dir)
    elif a.fetch:
        fetch(a.raw_dir)
    else:
        build(a.raw_dir, a.snapshot_dir, a.output_dir, a.min_tasks)
