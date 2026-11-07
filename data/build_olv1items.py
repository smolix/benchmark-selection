#!/usr/bin/env python3
"""Build the Open LLM Leaderboard v1 item-level dataset (olv1items).

Produces (in --output-dir):
    olv1items.matrix.csv   395 models x 62 tasks, NaN for missing (none here)
    olv1items.matrix.npz   B, O, model_names, benchmark_names (save_matrix format)
    olv1items.items.npz    per-item scores (item x model), bit-packed, so that
                           any subset of items can be scored
    olv1items.meta.json    score definitions, per-task item counts / few-shot /
                           request counts, per-model provenance (matched results
                           file, eval timestamp, model sha, submission date),
                           overlap with mmlu.matrix.npz, consistency checks
    raw_manifests/olv1items.json
                           every raw input: URL, revision, size, sha256,
                           git blob sha1, fetched_at

Task score = mean item score over all items of the task.  57 MMLU subjects
(5-shot, acc), ARC-Challenge (25-shot, acc_norm), HellaSwag (10-shot,
acc_norm), Winogrande (5-shot, acc) and GSM8K (5-shot, exact-match acc) have
0/1 items.  TruthfulQA (0-shot) items are NOT binary: each item carries its
mc2 score (normalised probability mass on the true answers), so the task score
is the leaderboard's mc2.  All six definitions are verified against the
official v1 aggregate results (see meta.json -> consistency).

Order (deterministic, independent of dict order in the source):
    models      sorted by HF id ('<org>/<model>') with Python str ordering
                (code-point order, case-sensitive)
    benchmarks  the 57 MMLU subjects in alphabetical order (identical to
                data/mmlu.matrix.npz), then arc_challenge, hellaswag,
                truthfulqa_mc2, winogrande, gsm8k (leaderboard v1 order)
    items       benchmark order, then source row order within each task

Raw inputs (in --raw-dir; ``--fetch`` downloads whatever is missing):
    tinyBenchmarks/lb.pickle   https://github.com/felipemaiapolo/tinyBenchmarks
                               tutorials/data/lb.pickle @ TB_COMMIT (MIT)
    tinyBenchmarks/LICENSE, README.md (same commit)
    olv1_results/              shallow git fetch of exactly RESULTS_COMMIT of HF
                               dataset open-llm-leaderboard-old/results
    olv1_requests/             shallow git fetch of exactly REQUESTS_COMMIT of HF
                               dataset open-llm-leaderboard-old/requests
    olv1_details/<repo>/       shallow git fetch of the pinned commit of the HF
                               details dataset of each model in DETAILS_REPOS
                               (models whose original run is missing from
                               olv1_results; LFS files stay pointers)
    olv1_details_api/<repo>.json
                               HF API card of a fixed sample of details repos
                               (LICENSE_SAMPLE), for the license record only
    *fetched_at.txt            UTC download timestamps written by --fetch
    metabench/                 optional, inspected only and not read by the build
                               (recorded in the manifest; hash-checked if present)

Every git input is fetched by commit id (``git fetch --depth 1 <url> <sha>``),
not by the remote HEAD, so a re-fetch keeps working if the remote moves.
Paths of results files in meta.json are relative to --raw-dir.

Safety: lb.pickle is untrusted.  Before unpickling, every opcode is scanned
statically (pickletools.genops, no code executed) and every GLOBAL /
STACK_GLOBAL must resolve to an allowlisted numpy/builtin reconstructor;
forbidden opcodes (PERSID, EXT*, INST, OBJ) and unresolvable STACK_GLOBALs
abort.  Loading then uses a restricted Unpickler whose find_class returns
objects from the same explicit allowlist, narrowed further to the globals the
scan found (no imports by name).  The allowlist is TASK_ALLOWLIST (the list
given for this build) plus ALLOWLIST_EXTENSIONS (numpy's protocol-5 array
rebuild function, wrapped so that object dtypes are refused); both are
recorded in meta.json -> pickle_safety.  The leaderboard JSONs are parsed with
json only.  Run with ``python -I`` from a directory without downloaded files.

Usage:
    python -I data/build_olv1items.py --raw-dir ~/benchselect-data/raw/olv1items \
        --output-dir data [--fetch]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import io
import json
import pathlib
import pickle
import pickletools
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import zipfile

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I does not add the script dir
from build_matrices import save_matrix  # noqa: E402

NAME = "olv1items"

TB_REPO = "felipemaiapolo/tinyBenchmarks"
TB_COMMIT = "e9a8b1031b0340571beb6c9ca3a27891be09a8fd"   # main HEAD, fetched 2026-10-17
TB_LB_LAST_COMMIT = "ce9f7024248685ff1a5b507b955051e9a0e6e074"  # last change to lb.pickle
TB_FILES = {  # local name -> repo path
    "lb.pickle": "tutorials/data/lb.pickle",
    "LICENSE": "LICENSE",
    "README.md": "README.md",
}
LB_SHA256 = "34f44d6a819512ef74d00a95288d252fa679288a10ca167cd97fdbc3aae66437"
LB_GIT_BLOB = "c6f010d63c08779624ad6b95b518e7d58f369283"   # tree entry at TB_COMMIT

RESULTS_REPO = "open-llm-leaderboard-old/results"
RESULTS_COMMIT = "23474373f8874f9057d23b97e5a41e911d2721c5"
REQUESTS_REPO = "open-llm-leaderboard-old/requests"
REQUESTS_COMMIT = "56486a8023577caeb30ba13ff89d0d35af13de89"

DETAILS_PREFIX = "open-llm-leaderboard/details_"   # model names in lb.pickle
DETAILS_ORG = "open-llm-leaderboard-old"           # current home of the details repos
# Models whose original evaluation run (the one tinyBenchmarks used for all
# tasks except winogrande/gsm8k) has no results file in olv1_results: take the
# aggregate results_*.json of that run from the model's details repo.  Other
# results_*.json files in these repos must equal the olv1_results file.
DETAILS_REPOS = {  # details repo -> (pinned commit, [results files used])
    "details_CHIH-HUNG__llama-2-13b-huangyt_FINETUNE2_3w": (
        "63aa8508ce031392d2407c8eab3e3b965c834bb1",
        ["results_2023-09-01T02:00:37.235761.json"]),
    "details_CHIH-HUNG__llama-2-13b-huangyt_Fintune_1_17w-q_k_v_o_proj": (
        "bc8b0f7ac0d8051843f2f8e53b236b9298619f41",
        ["results_2023-09-04T02:19:44.261303.json"]),
}
# License check of the details repos (upstream of lb.pickle): the two repos
# above (git, pinned) plus the HF API card of every 50th model in matrix order.
LICENSE_SAMPLE = [
    "details_922-CA__monika-ddlc-7b-v1",
    "details_HenryJJ__dolphin-2.6-mistral-7b-dpo-orca-v3",
    "details_PulsarAI__EnsembleV5-Nova-13B",
    "details_Undi95__Emerald-13B",
    "details_bofenghuang__vigostral-7b-chat",
    "details_fireballoon__baichuan-vicuna-chinese-7b",
    "details_lilloukas__Platypus-30B",
    "details_rishiraj__CatPPT-base",
]

# metabench (Kipnis et al.): inspected while building, NOT used by the outputs.
METABENCH = {
    "name": "metabench - Paper Data (Zenodo record 12819251)",
    "record": "https://zenodo.org/records/12819251",
    "doi": "10.5281/zenodo.12819251", "license": "cc-by-4.0",
    "license_declared_in": "Zenodo record metadata.license.id",
    "fetched_at": "2026-10-18T00:08:43Z",
    "files": [
        {"path": "metabench/data.tgz",
         "url": "https://zenodo.org/api/records/12819251/files/data.tgz/content",
         "size": 620415543, "md5": "9f2d5d6bbf6cf730494e0c29507850c7",
         "sha256": "24de1a3f387ee3787c163981e8ea6bb441f625e4f173158cc6a4316d06f8283e"},
        {"path": "metabench/record.json", "url": "https://zenodo.org/api/records/12819251",
         "size": 3987,
         "sha256": "a86ff4525473a061cd1e66ac4478098708c1c47c2fb4c20109712f9f06e2c029"},
    ],
    "use": ("inspected only (same 62 tasks and item order as lb.pickle, item vectors equal "
            "for 219 of 221 overlapping models, TruthfulQA stored as binary mc1); not read "
            "by this builder and not needed to rebuild the outputs"),
}

# The 57 MMLU subjects (mirrors download_mmlu.MMLU_SUBJECTS; not imported
# because that module loads data/.env at import time).
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

MMLU_DEF = ("4-way multiple choice, loglikelihood of the answer letter; item = "
            "1 if the gold choice has the highest loglikelihood (acc)")
# name, lb.pickle key, results-JSON key, metric, n_shot, binary, definition
TASKS = [(s, f"harness_hendrycksTest_{s}_5", f"harness|hendrycksTest-{s}|5",
          "acc", 5, True, MMLU_DEF) for s in MMLU_SUBJECTS] + [
    ("arc_challenge", "harness_arc_challenge_25", "harness|arc:challenge|25",
     "acc_norm", 25, True,
     "multiple choice (3-5 options), item = 1 if the gold choice has the highest "
     "length-normalised loglikelihood (acc_norm)"),
    ("hellaswag", "harness_hellaswag_10", "harness|hellaswag|10",
     "acc_norm", 10, True,
     "4-way sentence completion, item = 1 if the gold ending has the highest "
     "length-normalised loglikelihood (acc_norm)"),
    ("truthfulqa_mc2", "harness_truthfulqa_mc_0", "harness|truthfulqa:mc|0",
     "mc2", 0, False,
     "NOT binary: item value = mc2 = sum of normalised probabilities of the true "
     "reference answers among all (true+false) answers, in [0,1]; 0-shot "
     "(the harness task prepends its fixed QA instruction prompt)"),
    ("winogrande", "harness_winogrande_5", "harness|winogrande|5",
     "acc", 5, True,
     "2-way pronoun resolution by partial loglikelihood, item = 1 if correct (acc)"),
    ("gsm8k", "harness_gsm8k_5", "harness|gsm8k|5",
     "acc", 5, True,
     "generative math word problems, item = 1 if the extracted final answer "
     "matches exactly (acc)"),
]
BENCH_NAMES = [t[0] for t in TASKS]
MATCH_TOL = 1e-9

# ---------------------------------------------------------------- pickle safety
def _frombuffer_guarded(buf, dtype, shape, order, axis_order=None):
    """numpy's protocol-5 array rebuild (frombuffer + reshape), refusing object
    dtypes (np.frombuffer refuses them as well; checked here explicitly)."""
    dt = np.dtype(dtype)
    if dt.hasobject:
        raise pickle.UnpicklingError(f"object dtype {dt!r} in _frombuffer")
    return np._core.numeric._frombuffer(buf, dt, shape, order, axis_order)


# The allowlist given for this build (numpy array reconstruction + plain builtins).
TASK_ALLOWLIST = {
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("builtins", "dict"): dict, ("builtins", "list"): list,
    ("builtins", "set"): set, ("builtins", "tuple"): tuple,
    ("builtins", "str"): str, ("builtins", "int"): int,
    ("builtins", "float"): float,
    ("collections", "OrderedDict"): collections.OrderedDict,
}
# Additions beyond TASK_ALLOWLIST, each with its justification (-> meta.json).
_FROMBUFFER_WHY = ("numpy's rebuild function for arrays pickled with protocol 5 "
                   "(ndarray.__reduce_ex__(5)): frombuffer(buf, dtype).reshape(shape); "
                   "it only views a byte buffer as a non-object array.  Mapped to "
                   "_frombuffer_guarded, which also refuses object dtypes.")
ALLOWLIST_EXTENSIONS = {
    ("numpy.core.numeric", "_frombuffer"): (_frombuffer_guarded, _FROMBUFFER_WHY),
    ("numpy._core.numeric", "_frombuffer"): (_frombuffer_guarded, _FROMBUFFER_WHY),
}
SAFE_GLOBALS = {**TASK_ALLOWLIST, **{k: v[0] for k, v in ALLOWLIST_EXTENSIONS.items()}}
_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
               "SHORT_BINSTRING", "BINSTRING", "STRING"}
_FORBIDDEN_OPS = {"PERSID", "BINPERSID", "EXT1", "EXT2", "EXT4", "INST", "OBJ"}


def scan_pickle(path):
    """Static opcode scan.  Returns (globals Counter, opcode Counter, problems)."""
    found = collections.Counter()
    opcounts = collections.Counter()
    problems = []
    memo = {}
    unk = object()
    tail = []  # last pushed stack values: str or unk
    with open(path, "rb") as f:
        for op, arg, pos in pickletools.genops(f):
            n = op.name
            opcounts[n] += 1
            if n in _FORBIDDEN_OPS:
                problems.append(f"{n}@{pos}")
            if n in _STRING_OPS:
                tail.append(arg)
            elif n in ("GET", "BINGET", "LONG_BINGET"):
                tail.append(memo.get(arg, unk))
            elif n == "MEMOIZE":
                memo[len(memo)] = tail[-1] if tail else unk
            elif n in ("PUT", "BINPUT", "LONG_BINPUT"):
                memo[arg] = tail[-1] if tail else unk
            elif n in ("FRAME", "PROTO"):
                pass
            elif n == "GLOBAL":
                found[tuple(arg.split(" ", 1))] += 1
                tail.append(unk)
            elif n == "STACK_GLOBAL":
                if len(tail) >= 2 and isinstance(tail[-1], str) and isinstance(tail[-2], str):
                    found[(tail[-2], tail[-1])] += 1
                else:
                    problems.append(f"UNRESOLVED_STACK_GLOBAL@{pos}")
                tail.append(unk)
            else:
                tail.append(unk)
            del tail[:-8]
    for g in found:
        if g not in SAFE_GLOBALS:
            problems.append(f"GLOBAL_NOT_ALLOWED:{g[0]}.{g[1]}")
    return found, opcounts, problems


class _RestrictedUnpickler(pickle.Unpickler):
    """find_class returns only objects of SAFE_GLOBALS, and only for the
    (module, name) pairs in ``allowed`` (the globals the static scan found)."""

    def __init__(self, f, allowed):
        super().__init__(f)
        self._allowed = frozenset(allowed) & frozenset(SAFE_GLOBALS)

    def find_class(self, module, name):
        if (module, name) not in self._allowed:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return SAFE_GLOBALS[(module, name)]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent ids are not allowed")


def load_lb_pickle(path):
    found, opcounts, problems = scan_pickle(path)
    if problems:
        raise RuntimeError(f"unsafe pickle {path}: {problems[:10]}")
    with open(path, "rb") as f:
        obj = _RestrictedUnpickler(f, allowed=found).load()
    if not (isinstance(obj, dict) and set(obj) == {"data", "models"}):
        raise ValueError("unexpected lb.pickle top-level structure")
    models = obj["models"]
    if not (isinstance(models, list) and all(isinstance(m, str) for m in models)):
        raise ValueError("models must be a list of str")
    data = {}
    for key, v in obj["data"].items():
        if not (isinstance(v, dict) and set(v) == {"correctness"}):
            raise ValueError(f"unexpected entry for {key}")
        c = v["correctness"]
        if not (type(c) is np.ndarray and c.dtype == np.float64 and c.ndim == 2
                and c.shape[1] == len(models)):
            raise ValueError(f"bad correctness array for {key}")
        data[key] = c
    scan_info = {
        "method": ("static scan with pickletools.genops (no code executed); then "
                   "pickle.Unpickler with find_class restricted to the globals found "
                   "by the scan that are also in the allowlist"),
        "globals_found": [f"{m}.{n}" for (m, n) in sorted(found)],
        "global_counts": {f"{m}.{n}": c for (m, n), c in sorted(found.items())},
        "opcode_counts": dict(sorted(opcounts.items())),
        "forbidden_opcodes_checked": sorted(_FORBIDDEN_OPS),
        "forbidden_opcodes_found": [],
        "unresolved_stack_globals": 0,
        "task_allowlist": sorted(f"{m}.{n}" for (m, n) in TASK_ALLOWLIST),
        "allowlist_extensions": {f"{m}.{n}": why for (m, n), (_, why)
                                 in sorted(ALLOWLIST_EXTENSIONS.items())},
        "globals_found_outside_task_allowlist": [f"{m}.{n}" for (m, n) in sorted(found)
                                                 if (m, n) not in TASK_ALLOWLIST],
        "allowlist": sorted(f"{m}.{n}" for (m, n) in SAFE_GLOBALS),
        "unpickler_allowed": [f"{m}.{n}" for (m, n) in sorted(found) if (m, n) in SAFE_GLOBALS],
        "post_load_validation": ("top level {'data','models'}; models list[str]; every task "
                                 "{'correctness': 2-D float64 ndarray, n_models columns}"),
    }
    return models, data, scan_info


# ---------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    data = pathlib.Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git_head(repo_dir):
    git = pathlib.Path(repo_dir) / ".git"
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    if (git / ref).exists():
        return (git / ref).read_text().strip()
    for line in (git / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise RuntimeError(f"cannot resolve HEAD of {repo_dir}")


def git_tree_blobs(repo_dir):
    """{path: blob sha1} of HEAD (via the system git binary; no hooks run)."""
    out = subprocess.run(["git", "-C", str(repo_dir), "ls-tree", "-r", "-z", "HEAD"],
                         check=True, capture_output=True).stdout
    blobs = {}
    for rec in out.split(b"\0"):
        if rec:
            meta, path = rec.split(b"\t", 1)
            blobs[path.decode()] = meta.split()[2].decode()
    return blobs


def read_fetched_at(path):
    p = pathlib.Path(path)
    return p.read_text().strip() if p.exists() else None


def card_license(readme_path):
    """The top-level ``license`` of a HF dataset card's YAML front matter
    (README.md), parsed with re only; None if the card declares none."""
    text = pathlib.Path(readme_path).read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    lines = text[3:end if end >= 0 else len(text)].splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"license:\s*(.*)$", line)  # column 0 = top-level key
        if m:
            v = m.group(1).strip().strip("'\"")
            if v:
                return v
            vals = []
            for nxt in lines[i + 1:]:
                mm = re.match(r"\s*-\s*(.+)$", nxt)
                if not mm:
                    break
                vals.append(mm.group(1).strip().strip("'\""))
            return ",".join(vals) or None
    return None


def savez_deterministic(path, arrays):
    """np.savez_compressed equivalent with fixed zip timestamps (byte-stable)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, arr in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())


def load_item_scores(path):
    """Decode olv1items.items.npz -> (scores[n_items, n_models] float64, npz).

    Binary items are unpacked from ``correct_bits`` (np.packbits along the
    model axis, bitorder='big'); non-binary items (TruthfulQA mc2) are taken
    from ``float_scores`` at rows ``float_item_index``.  Items of task k are
    rows task_offsets[k]:task_offsets[k+1]; the score of model m on any item
    subset S is scores[S, m].mean().
    """
    z = np.load(path, allow_pickle=False)
    n_models = int(z["n_models"])
    s = np.unpackbits(z["correct_bits"], axis=1, count=n_models,
                      bitorder="big").astype(np.float64)
    s[z["float_item_index"]] = z["float_scores"]
    return s, z


_TS_RE = re.compile(r"results_(\d{4}-\d\d-\d\d)T(\d\d)[-: ](\d\d)[-: ](\d\d)(\.\d+)?\.json$")


def results_timestamp(fname):
    m = _TS_RE.search(fname)
    if not m:
        return None
    return f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}{m.group(5) or ''}"


def details_to_hf(name):
    if not name.startswith(DETAILS_PREFIX):
        raise ValueError(f"unexpected model name {name!r}")
    rest = name[len(DETAILS_PREFIX):]
    if rest.count("__") != 1:
        raise ValueError(f"ambiguous org/model split for {name!r}")
    return rest.replace("__", "/", 1)


# ---------------------------------------------------------------- fetch
def _stamp():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


GIT_FETCH_DESC = ("git init + git fetch --depth 1 <url> <pinned commit> + checkout --detach "
                  "(GIT_LFS_SKIP_SMUDGE=1, no credential helper, no global git config)")


def git_fetch_pinned(url, commit, dest):
    """Shallow-fetch exactly ``commit`` (not the remote HEAD) of ``url`` into
    ``dest`` and check it out detached.  Returns True if it downloaded.  No
    credentials: credential helpers are disabled and HOME is not passed, so no
    global git config (or LFS filter) applies; LFS files stay pointers."""
    dest = pathlib.Path(dest)
    if dest.exists():
        if git_head(dest) != commit:
            raise RuntimeError(f"{dest} is at {git_head(dest)}, not the pinned {commit}; "
                               "move it away and re-fetch")
        return False
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    env = {"GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin"}
    git = ["git", "-c", "credential.helper=", "-c", "core.hooksPath=/dev/null"]
    subprocess.run(git + ["init", "--quiet", str(tmp)], check=True, env=env)
    subprocess.run(git + ["-C", str(tmp), "fetch", "--quiet", "--depth", "1", url, commit],
                   check=True, env=env)
    subprocess.run(git + ["-C", str(tmp), "-c", "advice.detachedHead=false", "checkout",
                          "--quiet", "--detach", "FETCH_HEAD"], check=True, env=env)
    if git_head(tmp) != commit:
        raise RuntimeError(f"{url}: fetched {git_head(tmp)}, expected {commit}")
    tmp.rename(dest)
    return True


def fetch(raw_dir):
    import requests  # only needed for --fetch

    sess = requests.Session()
    sess.trust_env = False  # no ~/.netrc credentials, no proxies from the environment
    raw_dir.mkdir(parents=True, exist_ok=True)
    tb = raw_dir / "tinyBenchmarks"
    tb.mkdir(exist_ok=True)
    if not ((tb / "lb.pickle").exists() and sha256_file(tb / "lb.pickle") == LB_SHA256
            and all((tb / local).exists() for local in TB_FILES)):
        for local, rpath in TB_FILES.items():
            url = f"https://raw.githubusercontent.com/{TB_REPO}/{TB_COMMIT}/{rpath}"
            r = sess.get(url, timeout=300)
            r.raise_for_status()
            (tb / local).write_bytes(r.content)
        (tb / "fetched_at.txt").write_text(_stamp() + "\n")
    hf = "https://huggingface.co/datasets"
    for url, commit, dest in (
            [(f"{hf}/{RESULTS_REPO}", RESULTS_COMMIT, raw_dir / "olv1_results"),
             (f"{hf}/{REQUESTS_REPO}", REQUESTS_COMMIT, raw_dir / "olv1_requests")]
            + [(f"{hf}/{DETAILS_ORG}/{repo}", commit, raw_dir / "olv1_details" / repo)
               for repo, (commit, _) in sorted(DETAILS_REPOS.items())]):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if git_fetch_pinned(url, commit, dest):
            dest.with_name(dest.name + ".fetched_at.txt").write_text(_stamp() + "\n")
    api = raw_dir / "olv1_details_api"
    if not all((api / f"{repo}.json").exists() for repo in LICENSE_SAMPLE):
        api.mkdir(exist_ok=True)
        for repo in LICENSE_SAMPLE:  # 8 API requests, 1 per second
            r = sess.get(f"https://huggingface.co/api/datasets/{DETAILS_ORG}/{repo}", timeout=60)
            r.raise_for_status()
            (api / f"{repo}.json").write_bytes(r.content)
            time.sleep(1.0)
        (api / "fetched_at.txt").write_text(_stamp() + "\n")


# ---------------------------------------------------------------- build
def load_results(raw_dir, hf_ids, details_repo_of):
    """{hf_id: [(path relative to raw_dir, timestamp, json dict)]} sorted by
    (timestamp, path): all results files of the model in olv1_results plus the
    DETAILS_REPOS files of its details repo (olv1_details/<repo>/...)."""
    res_root = raw_dir / "olv1_results"
    out = {}
    bad = []
    for h in hf_ids:
        d = res_root / h
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        det = details_repo_of[h]
        if det in DETAILS_REPOS:
            files += [raw_dir / "olv1_details" / det / f for f in DETAILS_REPOS[det][1]]
        rows = []
        for f in files:
            rel = f.relative_to(raw_dir).as_posix()
            try:
                js = json.loads(f.read_text())
            except (ValueError, UnicodeDecodeError) as e:
                bad.append((rel, str(e)[:80]))
                continue
            rows.append((rel, results_timestamp(f.name), js))
        rows.sort(key=lambda r: (r[1] or "", r[0]))
        out[h] = rows
    return out, bad


def check_details_repos(raw_dir, details_repo_of):
    """Verify the pinned details repos: HEAD, used files in the tree, and every
    other top-level results_*.json byte-identical (git blob) to a file of the
    same model in olv1_results.  Returns per-repo records for the manifest/meta."""
    res_tree = git_tree_blobs(raw_dir / "olv1_results")
    hf_of = {v: k for k, v in details_repo_of.items()}
    recs = []
    for repo, (commit, used) in sorted(DETAILS_REPOS.items()):
        root = raw_dir / "olv1_details" / repo
        if git_head(root) != commit:
            raise RuntimeError(f"{root}: HEAD is not pinned commit {commit}")
        tree = git_tree_blobs(root)
        h = hf_of[repo]
        mine = {b for p, b in res_tree.items() if p.startswith(h + "/")}
        others = sorted(p for p in tree if re.fullmatch(r"results_[^/]*\.json", p)
                        and p not in used)
        for p in others:
            if tree[p] not in mine:
                raise RuntimeError(f"{repo}/{p} is not identical to a file in olv1_results/{h}")
        for p in used + ["README.md"]:
            if tree.get(p) != git_blob_sha1(root / p):
                raise RuntimeError(f"{repo}/{p} differs from the pinned tree")
        recs.append({"repo": repo, "model": h, "commit": commit, "root": root, "tree": tree,
                     "used": used, "other_results_files_equal_olv1_results": others,
                     "card_license": card_license(root / "README.md")})
    return recs


def load_requests(req_root, hf_ids):
    """{hf_id: [(relpath, json)]}, matched on the 'model' field; a model without
    an exact match falls back to a case-insensitive match (flagged)."""
    want = set(hf_ids)
    lower = {h.lower(): h for h in hf_ids}
    exact = collections.defaultdict(list)
    ci = collections.defaultdict(list)
    for f in sorted(req_root.rglob("*.json")):
        if ".git" in f.parts:
            continue
        try:
            js = json.loads(f.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if not (isinstance(js, dict) and isinstance(js.get("model"), str)):
            continue
        rel = f.relative_to(req_root).as_posix()
        if js["model"] in want:
            exact[js["model"]].append((rel, js))
        elif js["model"].lower() in lower:
            ci[lower[js["model"].lower()]].append((rel, js))
    out, how = {}, {}
    for h in hf_ids:
        if exact.get(h):
            out[h], how[h] = exact[h], "exact"
        elif ci.get(h):
            out[h], how[h] = ci[h], "case_insensitive"
    return out, how


def norm_model_type(x):
    s = " ".join(x) if isinstance(x, list) else str(x or "")
    s = s.lower()
    for key, val in [("pretrained", "pretrained"), ("instruction", "instruction-tuned"),
                     ("rl-tuned", "RL-tuned"), ("fine-tuned", "fine-tuned"),
                     ("finetuned", "fine-tuned")]:
        if key in s:
            return val
    return None


def dump_json(obj, line_keys=()):
    """json.dumps(indent=1), except that the top-level lists named in
    line_keys are written one compact element per line."""
    o = dict(obj)
    lists = {k: o.pop(k) for k in line_keys}
    for k in line_keys:
        o[k] = f"@@{k}@@"
    s = json.dumps(o, indent=1)
    for k, lst in lists.items():
        body = "[\n" + ",\n".join("  " + json.dumps(x, separators=(",", ":"))
                                  for x in lst) + "\n ]"
        s = s.replace(json.dumps(f"@@{k}@@"), body)
    return s + "\n"


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build(raw_dir, output_dir, mmlu_matrix):
    raw_dir = pathlib.Path(raw_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_manifests").mkdir(exist_ok=True)
    tb = raw_dir / "tinyBenchmarks"
    res_root = raw_dir / "olv1_results"
    req_root = raw_dir / "olv1_requests"

    # --- verify pinned raw inputs
    lb_path = tb / "lb.pickle"
    lb_sha = sha256_file(lb_path)
    if lb_sha != LB_SHA256 or git_blob_sha1(lb_path) != LB_GIT_BLOB:
        raise RuntimeError(f"lb.pickle hash mismatch: {lb_sha}")
    for root, commit in [(res_root, RESULTS_COMMIT), (req_root, REQUESTS_COMMIT)]:
        if git_head(root) != commit:
            raise RuntimeError(f"{root}: HEAD is not pinned commit {commit}")

    # --- item data
    print(f"Scanning and loading {lb_path} ...")
    src_models, data, scan_info = load_lb_pickle(lb_path)
    print(f"  globals found: {scan_info['globals_found']}")
    if set(data) != {t[1] for t in TASKS}:
        raise ValueError(f"task set mismatch: {sorted(set(data) ^ {t[1] for t in TASKS})}")
    hf_src = [details_to_hf(m) for m in src_models]
    if len(set(hf_src)) != len(hf_src):
        raise ValueError("duplicate model ids")
    order = sorted(range(len(hf_src)), key=lambda j: hf_src[j])
    models = [hf_src[j] for j in order]
    M = len(models)
    details_repo_of = {hf_src[j]: src_models[j][len("open-llm-leaderboard/"):]
                       for j in range(len(hf_src))}
    if not set(DETAILS_REPOS) | set(LICENSE_SAMPLE) <= set(details_repo_of.values()):
        raise ValueError("DETAILS_REPOS / LICENSE_SAMPLE name repos of models not in lb.pickle")
    details_recs = check_details_repos(raw_dir, details_repo_of)

    # --- license evidence (dataset cards at the pinned commits, API sample)
    tb_license = (tb / "LICENSE").read_text(encoding="utf-8")
    if not tb_license.startswith("MIT License"):
        raise RuntimeError("tinyBenchmarks LICENSE is not the MIT license")
    api_dir = raw_dir / "olv1_details_api"
    api_fetched = read_fetched_at(api_dir / "fetched_at.txt")
    api_sample = []
    for repo in LICENSE_SAMPLE:
        p = api_dir / f"{repo}.json"
        js = json.loads(p.read_text())
        if js.get("id") != f"{DETAILS_ORG}/{repo}":
            raise ValueError(f"{p}: unexpected id {js.get('id')!r}")
        card = js.get("cardData") if isinstance(js.get("cardData"), dict) else {}
        api_sample.append({
            "repo": f"{DETAILS_ORG}/{repo}", "sha": js.get("sha"),
            "lastModified": js.get("lastModified"),
            "card_license": card.get("license"),
            "license_tags": sorted(t for t in js.get("tags") or []
                                   if isinstance(t, str) and t.startswith("license:")),
            "file": f"olv1_details_api/{repo}.json"})
    det_licenses = ([{"repo": f"{DETAILS_ORG}/{r['repo']}", "commit": r["commit"],
                      "card_license": r["card_license"],
                      "evidence": f"olv1_details/{r['repo']}/README.md front matter"}
                     for r in details_recs]
                    + [{"repo": a["repo"], "commit": a["sha"],
                        "card_license": a["card_license"] or (",".join(a["license_tags"]) or None),
                        "evidence": f"{a['file']} (HF API cardData.license and license: tags)"}
                       for a in api_sample])
    det_declared = sorted({d["card_license"] for d in det_licenses if d["card_license"]})
    results_license = card_license(res_root / "README.md")
    requests_license = card_license(req_root / "README.md")

    # --- metabench: not used; hash-checked if still present
    for f in METABENCH["files"]:
        p = raw_dir / f["path"]
        if p.exists() and (p.stat().st_size != f["size"] or sha256_file(p) != f["sha256"]):
            raise RuntimeError(f"{p} differs from the recorded metabench file")

    blocks, B_cols, item_ids, item_task, item_row = [], [], [], [], []
    float_rows, float_vals = [], []
    offsets = [0]
    for k, (bname, key, _, _, _, binary, _) in enumerate(TASKS):
        C = np.ascontiguousarray(data[key][:, order])  # items x models
        n = C.shape[0]
        is01 = bool(np.isin(C, (0.0, 1.0)).all())
        if is01 != binary:
            raise ValueError(f"{bname}: expected binary={binary}, got {is01}")
        if binary:
            B_cols.append(np.count_nonzero(C, axis=0) / n)
            blocks.append(np.packbits(C.astype(bool), axis=1, bitorder="big"))
        else:
            if not (np.isfinite(C).all() and C.min() >= 0.0 and C.max() <= 1.0 + 1e-12):
                raise ValueError(f"{bname}: non-binary values outside [0,1]")
            B_cols.append(C.mean(axis=0))
            blocks.append(np.zeros((n, (M + 7) // 8), dtype=np.uint8))
            float_rows.append(np.arange(offsets[-1], offsets[-1] + n, dtype=np.int64))
            float_vals.append(C)
        item_ids += [f"{bname}/{i:05d}" for i in range(n)]
        item_task.append(np.full(n, k, dtype=np.int16))
        item_row.append(np.arange(n, dtype=np.int32))
        offsets.append(offsets[-1] + n)
    B = np.column_stack(B_cols)
    df = pd.DataFrame(B, index=pd.Index(models, name="model"), columns=BENCH_NAMES)

    items = {
        "model_names": np.array(models),
        "task_names": np.array(BENCH_NAMES),
        "task_n_items": np.diff(np.array(offsets, dtype=np.int64)),
        "task_offsets": np.array(offsets, dtype=np.int64),
        "task_binary": np.array([t[5] for t in TASKS]),
        "item_ids": np.array(item_ids),
        "item_task": np.concatenate(item_task),
        "item_source_row": np.concatenate(item_row),
        "correct_bits": np.concatenate(blocks, axis=0),
        "float_item_index": np.concatenate(float_rows),
        "float_scores": np.concatenate(float_vals, axis=0),
        "n_models": np.array(M, dtype=np.int64),
    }
    n_items = offsets[-1]
    print(f"  {M} models x {len(TASKS)} tasks, {n_items} items "
          f"({n_items - len(items['float_item_index'])} binary)")

    # --- results repo: which official run do the item data come from?
    print("Matching against open-llm-leaderboard-old/results (+ pinned details repos) ...")
    results, bad_results = load_results(raw_dir, models, details_repo_of)
    req, req_how = load_requests(req_root, models)
    model_meta, task_check = [], {t[0]: {"n_models_with_metric": 0, "n_exact": 0,
                                         "max_abs_diff_closest": 0.0} for t in TASKS}
    summary_tasks = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for i, h in enumerate(models):
        rows = results[h]
        matched = collections.defaultdict(list)  # file idx -> tasks
        unmatched = []
        for k, (bname, _, rkey, metric, *_rest) in enumerate(TASKS):
            diffs = []
            for fi, (_, _, js) in enumerate(rows):
                v = _num(((js.get("results") or {}).get(rkey) or {}).get(metric))
                if v is not None:
                    diffs.append((abs(v - B[i, k]), fi))
            if not diffs:
                unmatched.append(bname)
                continue
            tc = task_check[bname]
            tc["n_models_with_metric"] += 1
            best = min(d for d, _ in diffs)
            tc["max_abs_diff_closest"] = max(tc["max_abs_diff_closest"], best)
            hits = [fi for d, fi in diffs if d <= MATCH_TOL]
            if hits:
                tc["n_exact"] += 1
                for fi in hits:
                    matched[fi].append(bname)
            else:
                unmatched.append(bname)
        entry = {"name": h, "details_repo": src_models[order[i]],
                 "source_index": order[i], "n_results_files": len(rows)}
        if matched:
            pf = min(matched, key=lambda fi: (-len(matched[fi]), rows[fi][1] or "", rows[fi][0]))
            rel, ts, js = rows[pf]
            cg = js.get("config_general") or {}
            entry.update({
                "primary_results_file": rel,
                "primary_results_source": ("details_repo" if rel.startswith("olv1_details/")
                                           else "results_repo"),
                "eval_timestamp": ts,
                "model_sha": cg.get("model_sha"),
                "model_dtype": cg.get("model_dtype"),
                "model_size": cg.get("model_size"),
                "lighteval_sha": cg.get("lighteval_sha"),
                "total_evaluation_time_s": _num(cg.get("total_evaluation_time_secondes")),
            })
            primary_tasks = set(matched[pf])
            other = {}
            for k, t in enumerate(TASKS):
                if t[0] in primary_tasks or t[0] in unmatched:
                    continue
                fis = [fi for fi in matched if t[0] in matched[fi]]
                fi = min(fis, key=lambda x: (rows[x][1] or "", rows[x][0]))
                other[t[0]] = {"file": rows[fi][0], "eval_timestamp": rows[fi][1],
                               "model_sha": (rows[fi][2].get("config_general") or {})
                               .get("model_sha")}
            entry["tasks_from_other_results_file"] = other
            st = js.get("summary_tasks") or {}
            for t in TASKS:
                s = st.get(t[2])
                if t[0] in primary_tasks and isinstance(s, dict):
                    if "padded" in s and "non_padded" in s:
                        summary_tasks[t[0]]["n_requests"][s["padded"] + s["non_padded"]] += 1
                    if "truncated" in s and "non_truncated" in s:
                        summary_tasks[t[0]]["n_docs"][s["truncated"] + s["non_truncated"]] += 1
                    if "effective_few_shots" in s:
                        summary_tasks[t[0]]["effective_few_shots"][s["effective_few_shots"]] += 1
        else:
            entry.update({"primary_results_file": None, "eval_timestamp": None})
        entry["tasks_without_matching_results"] = unmatched

        # requests repo: submission metadata, same precision as the matched run
        rq = req.get(h, [])
        dtype = (entry.get("model_dtype") or "").replace("torch.", "")
        same = [r for r in rq if r[1].get("precision") == dtype] or rq
        fin = [r for r in same if r[1].get("status") == "FINISHED"] or same
        fin.sort(key=lambda r: (str(r[1].get("submitted_time") or "9999"), r[0]))
        if fin:
            rj = fin[0][1]
            entry["request"] = {"file": f"olv1_requests/{fin[0][0]}", "match": req_how[h], **{
                k: rj.get(k) for k in ("submitted_time", "precision", "weight_type",
                                       "status", "params", "model_type", "license",
                                       "architectures", "base_model", "revision")}}
            entry["request"]["model_type_norm"] = norm_model_type(rj.get("model_type"))
            sub = str(rj.get("submitted_time") or "").replace("Z", "")
            entry["request"]["submitted_after_eval"] = bool(
                sub and entry.get("eval_timestamp") and sub > entry["eval_timestamp"])
        else:
            entry["request"] = None
        entry["n_request_files"] = len(rq)
        entry["eval_date"] = (entry.get("eval_timestamp") or "")[:10] or None
        model_meta.append(entry)

    # --- overlap with data/mmlu.matrix.npz + MMLU consistency
    print(f"Comparing MMLU subjects with {mmlu_matrix} ...")
    mm = np.load(mmlu_matrix, allow_pickle=False)
    mm_names = [str(x) for x in mm["model_names"]]
    mm_bench = [str(x) for x in mm["benchmark_names"]]
    if mm_bench != MMLU_SUBJECTS:
        raise ValueError("mmlu.matrix benchmark order differs from MMLU_SUBJECTS")
    mm_idx = {n: r for r, n in enumerate(mm_names)}
    mm_lower = collections.defaultdict(list)
    for n in mm_names:
        mm_lower[n.lower()].append(n)
    diffs, per_model = [], []
    for i, h in enumerate(models):
        e = model_meta[i]
        if h in mm_idx:
            e["in_mmlu_matrix"], e["mmlu_matrix_name"], e["mmlu_matrix_match"] = True, h, "exact"
            r = mm_idx[h]
            if not (mm["O"][r] == 1).all():
                raise ValueError(f"missing MMLU cells for {h}")
            d = np.abs(mm["B"][r] - B[i, :57])
            diffs.append(d)
            per_model.append((float(d.max()), h))
        elif len(mm_lower.get(h.lower(), [])) == 1:
            e["in_mmlu_matrix"], e["mmlu_matrix_name"], e["mmlu_matrix_match"] = \
                False, mm_lower[h.lower()][0], "case_insensitive"
        else:
            e["in_mmlu_matrix"], e["mmlu_matrix_name"], e["mmlu_matrix_match"] = False, None, None
    D = np.concatenate(diffs) if diffs else np.zeros(0)
    outliers = []
    for mx, h in sorted(per_model, reverse=True):
        if mx <= 1e-6:
            break
        # which results file does mmlu.matrix agree with?
        r = mm_idx[h]
        agree = []
        for rel, ts, js in results[h]:
            res = js.get("results") or {}
            vals = [_num((res.get(f"harness|hendrycksTest-{s}|5") or {}).get("acc_norm"))
                    or _num((res.get(f"harness|hendrycksTest-{s}|5") or {}).get("acc"))
                    for s in MMLU_SUBJECTS]
            if all(v is not None for v in vals) and \
                    np.abs(np.array(vals) - mm["B"][r]).max() <= 1e-9:
                agree.append(rel)
        outliers.append({"model": h, "max_abs_diff": mx,
                         "mean_abs_diff": float(np.abs(mm["B"][r] - B[models.index(h), :57]).mean()),
                         "olv1items_results_file": model_meta[models.index(h)]["primary_results_file"],
                         "mmlu_matrix_agrees_with": agree})
    mmlu_check = {
        "n_models_exact_name_match": len(per_model),
        "n_models_case_insensitive_only": sum(e["mmlu_matrix_match"] == "case_insensitive"
                                              for e in model_meta),
        "n_cells_compared": int(D.size),
        "max_abs_diff": float(D.max()) if D.size else None,
        "mean_abs_diff": float(D.mean()) if D.size else None,
        "median_abs_diff": float(np.median(D)) if D.size else None,
        "n_models_max_abs_diff_le_1e-6": sum(mx <= 1e-6 for mx, _ in per_model),
        "mean_abs_diff_excluding_outlier_models": float(np.concatenate(
            [d for d, (mx, _) in zip(diffs, per_model) if mx <= 1e-6]).mean()) if diffs else None,
        "outlier_models": outliers,
        "n_mmlu_entries_acc_ne_acc_norm": None,  # filled below
        "note": ("data/mmlu.matrix.npz (download_mmlu.py) reads acc_norm, falling back to "
                 "acc, from the last results file of each model; olv1items reproduces acc "
                 "of the run whose details tinyBenchmarks used.  Outlier models differ "
                 "because mmlu.matrix took a later re-evaluation (mmlu_matrix_agrees_with)."),
    }
    n_ne = n_tot = 0
    for h in models:
        for _, _, js in results[h]:
            res = js.get("results") or {}
            for s in MMLU_SUBJECTS:
                e = res.get(f"harness|hendrycksTest-{s}|5") or {}
                a, an = _num(e.get("acc")), _num(e.get("acc_norm"))
                if a is not None and an is not None:
                    n_tot += 1
                    n_ne += abs(a - an) > MATCH_TOL
    mmlu_check["n_mmlu_entries_acc_ne_acc_norm"] = f"{n_ne} of {n_tot}"

    # --- duplicates
    S_full = np.unpackbits(items["correct_bits"], axis=1, count=M,
                           bitorder="big").astype(np.float64)
    S_full[items["float_item_index"]] = items["float_scores"]
    cols = collections.defaultdict(list)
    for i in range(M):
        cols[hashlib.sha256(S_full[:, i].tobytes()).hexdigest()].append(models[i])
    dup_groups = sorted(sorted(v) for v in cols.values() if len(v) > 1)
    # self-check: decoding reproduces the matrix exactly (binary) / to 1e-12 (mc2)
    off = items["task_offsets"]
    rec = np.column_stack([S_full[off[k]:off[k + 1]].mean(axis=0) for k in range(len(TASKS))])
    rec_err = float(np.abs(rec - B).max())
    if rec_err > 1e-12:
        raise RuntimeError(f"items.npz does not reproduce the matrix: {rec_err}")

    # --- per-task metadata
    benchmarks = []
    for k, (bname, key, rkey, metric, nshot, binary, definition) in enumerate(TASKS):
        st = summary_tasks[bname]

        def summ(c):
            if not c:
                return None
            v, n = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            return {"mode": v, "n_models_at_mode": n, "n_models": sum(c.values()),
                    "min": min(c), "max": max(c)}
        benchmarks.append({
            "name": bname, "group": "mmlu" if k < 57 else bname,
            "lb_pickle_key": key, "results_key": rkey, "metric": metric,
            "n_shot": nshot, "n_items": int(off[k + 1] - off[k]),
            "binary_items": binary, "item_range": [int(off[k]), int(off[k + 1])],
            "score_definition": definition,
            "n_requests": summ(st["n_requests"]),
            "n_docs": summ(st["n_docs"]),
            "effective_few_shots": summ(st["effective_few_shots"]),
            "results_check": task_check[bname],
        })

    # --- write matrix + items
    print("Writing outputs ...")
    save_matrix(NAME, df, output_dir)
    savez_deterministic(output_dir / f"{NAME}.items.npz", items)

    # --- manifest
    tb_fetched = read_fetched_at(tb / "fetched_at.txt")
    manifest_inputs = []
    for local, rpath in TB_FILES.items():
        p = tb / local
        manifest_inputs.append({
            "path": f"tinyBenchmarks/{local}",
            "url": f"https://raw.githubusercontent.com/{TB_REPO}/{TB_COMMIT}/{rpath}",
            "repo": f"https://github.com/{TB_REPO}", "revision": TB_COMMIT,
            "size": p.stat().st_size, "sha256": sha256_file(p),
            "git_blob_sha1": git_blob_sha1(p), "fetched_at": tb_fetched})
    git_inputs = []
    used_results = {rel for h in models for rel, _, _ in results[h]}
    git_specs = [
        (RESULTS_REPO, RESULTS_COMMIT, res_root, None,
         sorted(r[len("olv1_results/"):] for r in used_results
                if r.startswith("olv1_results/")) + ["README.md"]),
        (REQUESTS_REPO, REQUESTS_COMMIT, req_root, None,
         sorted({rel for h in models for rel, _ in req.get(h, [])}) + ["README.md"])]
    for d in details_recs:
        git_specs.append((f"{DETAILS_ORG}/{d['repo']}", d["commit"], d["root"], d["tree"],
                          sorted(d["used"]) + ["README.md"]))
    for repo, commit, root, tree, used in git_specs:
        rel_root = root.relative_to(raw_dir).as_posix()
        fetched = read_fetched_at(root.parent / f"{root.name}.fetched_at.txt")
        tree = tree or git_tree_blobs(root)
        files = []
        for rel in used:
            p = root / rel
            blob = git_blob_sha1(p)
            if tree.get(rel) != blob:
                raise RuntimeError(f"{rel_root}/{rel} differs from pinned tree")
            files.append({
                "path": f"{rel_root}/{rel}",
                "url": f"https://huggingface.co/datasets/{repo}/resolve/{commit}/"
                       + urllib.parse.quote(rel),
                "revision": commit, "size": p.stat().st_size, "sha256": sha256_file(p),
                "git_blob_sha1": blob, "fetched_at": fetched})
        git_inputs.append({
            "path": rel_root, "repo": f"https://huggingface.co/datasets/{repo}",
            "revision": commit, "fetch": GIT_FETCH_DESC,
            "fetched_at": fetched, "n_files_in_repo": len(tree),
            "n_files_used": len(files), "files_verified_against_tree": True})
        manifest_inputs.extend(files)
    for a in api_sample:
        p = raw_dir / a["file"]
        manifest_inputs.append({
            "path": a["file"], "url": f"https://huggingface.co/api/datasets/{a['repo']}",
            "revision": a["sha"], "size": p.stat().st_size, "sha256": sha256_file(p),
            "fetched_at": api_fetched,
            "note": "HF API dataset info (revision = its 'sha'); used for the license record only"})
    manifest = {"name": NAME, "builder": "data/build_olv1items.py",
                "repositories": git_inputs, "inputs": manifest_inputs,
                "inspected_not_used": [{
                    **{k: METABENCH[k] for k in ("name", "record", "doi", "license", "use")},
                    "files": [{**f, "fetched_at": METABENCH["fetched_at"]}
                              for f in METABENCH["files"]],
                    "note": ("kept in --raw-dir/metabench/ on the build server; hash-checked by "
                             "the builder if present, never extracted or executed (the archive "
                             "holds CSVs and R data files; only the CSVs were streamed)")}]}

    # --- meta
    n_obs = int(np.isfinite(B).sum())
    meta = {
        "name": NAME,
        "description": ("Open LLM Leaderboard v1 item-level correctness from tinyBenchmarks "
                        "(Maia Polo et al., 2024) for 395 models on 62 tasks (57 MMLU subjects, "
                        "ARC-Challenge, HellaSwag, TruthfulQA, Winogrande, GSM8K); task score "
                        "= mean item score."),
        "builder": "data/build_olv1items.py",
        "license": {
            "per_source": [
                {"source": f"tinyBenchmarks ({TB_REPO}, lb.pickle)", "license": "MIT",
                 "declared_in": f"LICENSE at {TB_COMMIT}",
                 "used_for": "item-level scores (items.npz) and the matrix",
                 "notice_required": ("MIT: the copyright and permission notice below must be "
                                     "included in all copies or substantial portions"),
                 "notice": tb_license},
                {"source": f"HF dataset {RESULTS_REPO}", "license": results_license,
                 "declared_in": (f"dataset card front matter (README.md) at {RESULTS_COMMIT}"
                                 + ("" if results_license else ": no license key")),
                 "used_for": "consistency checks and run metadata in meta.json only"},
                {"source": f"HF dataset {REQUESTS_REPO}", "license": requests_license,
                 "declared_in": (f"dataset card front matter (README.md) at {REQUESTS_COMMIT}"
                                 + ("" if requests_license else ": no license key")),
                 "used_for": "submission metadata in meta.json only"},
                {"source": (f"HF datasets {DETAILS_ORG}/details_* (formerly "
                            "open-llm-leaderboard/details_*): the per-model evaluation outputs "
                            "from which tinyBenchmarks built lb.pickle"),
                 "license": ",".join(det_declared) or None,
                 "declared_in": (f"dataset cards of {len(det_licenses)} of the {M} details repos "
                                 "(the 2 fetched at pinned commits + a fixed sample of every "
                                 "50th model via the HF API): "
                                 + ("licenses declared: " + ", ".join(det_declared)
                                    if det_declared else "none declares a license")),
                 "checked": det_licenses,
                 "used_for": ("upstream of lb.pickle; results_*.json of 2 repos used for "
                              "consistency checks and run metadata")},
                {"source": f"metabench, {METABENCH['record']} (inspected only, not used)",
                 "license": METABENCH["license"],
                 "declared_in": METABENCH["license_declared_in"], "used_for": "nothing"},
            ],
            "note": ("Item scores are model outputs on public benchmarks, produced by the Open "
                     "LLM Leaderboard (Hugging Face) and redistributed by tinyBenchmarks under "
                     "MIT; benchmark questions are not redistributed here (their own licenses "
                     "apply to them).  The leaderboard results and details datasets declare no "
                     "license on their cards; the requests dataset is apache-2.0.  Keep the "
                     "MIT notice above with any copy of the item data."),
        },
        "matrix": {"file": f"{NAME}.matrix.npz", "csv": f"{NAME}.matrix.csv",
                   "shape": list(B.shape), "n_observed": n_obs,
                   "fill_fraction": n_obs / B.size,
                   "model_order": "sorted by HF id with Python str ordering (code points)",
                   "benchmark_order": "57 MMLU subjects alphabetical (= mmlu.matrix.npz), then "
                                      "arc_challenge, hellaswag, truthfulqa_mc2, winogrande, gsm8k",
                   "score": "binary tasks: count_correct / n_items (exact); truthfulqa_mc2: "
                            "float64 mean of per-item mc2"},
        "items": {
            "file": f"{NAME}.items.npz", "n_items": int(n_items),
            "n_binary_items": int(n_items - len(items["float_item_index"])),
            "n_float_items": int(len(items["float_item_index"])),
            "layout": {
                "model_names": "(395,) same order as the matrix",
                "task_names": "(62,) same order as matrix benchmark_names",
                "task_offsets": "(63,) items of task k are rows task_offsets[k]:task_offsets[k+1]",
                "task_n_items": "(62,)", "task_binary": "(62,) bool",
                "item_ids": "(n_items,) '<task>/<row:05d>', row = item row in lb.pickle",
                "item_task": "(n_items,) int16 task index", "item_source_row": "(n_items,) int32",
                "correct_bits": "(n_items, 50) uint8 = np.packbits(correct, axis=1, "
                                "bitorder='big'); zero rows for non-binary items",
                "float_item_index": "(817,) rows of non-binary items (truthfulqa_mc2)",
                "float_scores": "(817, 395) float64 per-item mc2",
                "n_models": "scalar 395"},
            "decode": ("S = np.unpackbits(z['correct_bits'], axis=1, count=int(z['n_models']), "
                       "bitorder='big').astype(float); S[z['float_item_index']] = "
                       "z['float_scores']  # (n_items, n_models); or build_olv1items."
                       "load_item_scores(path)"),
            "item_identity": ("The source has no question ids or text: an item is identified by "
                              "its row in lb.pickle (the example order of the leaderboard details "
                              "files used by tinyBenchmarks)."),
            "reproduces_matrix_max_abs_err": rec_err,
        },
        "benchmarks": benchmarks,
        "models": model_meta,
        "duplicate_item_vectors": dup_groups,
        "sources": [
            {"name": "tinyBenchmarks lb.pickle", "repo": f"https://github.com/{TB_REPO}",
             "path": TB_FILES["lb.pickle"], "revision": TB_COMMIT,
             "last_commit_touching_file": TB_LB_LAST_COMMIT, "license": "MIT",
             "sha256": LB_SHA256, "git_blob_sha1": LB_GIT_BLOB, "fetched_at": tb_fetched,
             "citation": "Maia Polo et al., tinyBenchmarks: evaluating LLMs with fewer "
                         "examples, ICML 2024 (arXiv:2402.14992)"},
            {"name": "Open LLM Leaderboard v1 results (aggregates, run metadata)",
             "repo": f"https://huggingface.co/datasets/{RESULTS_REPO}",
             "revision": RESULTS_COMMIT, "fetched_at": git_inputs[0]["fetched_at"],
             "license": results_license,
             "use": "consistency check, eval timestamps, model sha/dtype, request counts"},
            {"name": "Open LLM Leaderboard v1 requests (submission metadata)",
             "repo": f"https://huggingface.co/datasets/{REQUESTS_REPO}",
             "revision": REQUESTS_COMMIT, "fetched_at": git_inputs[1]["fetched_at"],
             "license": requests_license,
             "use": "submitted_time, params, model_type, license"},
        ] + [
            {"name": f"Open LLM Leaderboard v1 details repo of {d['model']}",
             "repo": f"https://huggingface.co/datasets/{DETAILS_ORG}/{d['repo']}",
             "revision": d["commit"], "fetched_at": gi["fetched_at"],
             "license": d["card_license"], "files_used": d["used"],
             "other_results_files_identical_to_olv1_results":
                 d["other_results_files_equal_olv1_results"],
             "use": ("aggregate results of the original run, which is missing from "
                     f"{RESULTS_REPO}: consistency check of 60 tasks and eval_timestamp")}
            for d, gi in zip(details_recs, git_inputs[2:])
        ] + [
            {"name": "HF API dataset cards of sampled details repos",
             "repos": [a["repo"] for a in api_sample], "fetched_at": api_fetched,
             "use": "license record only (license.per_source)"},
            {"name": METABENCH["name"], "repo": METABENCH["record"], "doi": METABENCH["doi"],
             "license": METABENCH["license"], "fetched_at": METABENCH["fetched_at"],
             "files": METABENCH["files"], "use": METABENCH["use"]},
        ],
        "pickle_safety": scan_info,
        "consistency": {
            "mmlu_matrix": mmlu_check,
            "official_v1_results": {
                "tolerance": MATCH_TOL,
                "note": ("per task: models whose task score equals the official aggregate "
                         "(metric column) of at least one of their results files"),
                "per_group": {},
                "unparseable_results_files": bad_results,
            },
        },
        "caveats": [],
    }
    grp = collections.defaultdict(lambda: {"n_cells": 0, "n_with_metric": 0, "n_exact": 0,
                                           "max_abs_diff_closest": 0.0})
    for b in benchmarks:
        g = grp[b["group"]]
        tc = b["results_check"]
        g["metric"] = b["metric"]
        g["n_cells"] += M
        g["n_with_metric"] += tc["n_models_with_metric"]
        g["n_exact"] += tc["n_exact"]
        g["max_abs_diff_closest"] = max(g["max_abs_diff_closest"], tc["max_abs_diff_closest"])
    meta["consistency"]["official_v1_results"]["per_group"] = dict(grp)
    no_res = [e["name"] for e in model_meta if e["tasks_without_matching_results"]]
    det_models = [e["name"] for e in model_meta if e.get("primary_results_source") == "details_repo"]
    n_rerun = sum(bool(e.get("tasks_from_other_results_file")) for e in model_meta)
    n_rerun_sha = sum(any(o.get("model_sha") != e.get("model_sha")
                          for o in (e.get("tasks_from_other_results_file") or {}).values())
                      for e in model_meta)
    n_cells_checked = sum(g["n_exact"] for g in grp.values())
    meta["consistency"]["official_v1_results"]["n_cells_exact"] = f"{n_cells_checked} of {B.size}"
    meta["consistency"]["official_v1_results"]["details_repo_models"] = det_models
    meta["caveats"] = [
        "TruthfulQA items are continuous mc2 values, not 0/1; item-level selection that "
        "assumes Bernoulli items must treat truthfulqa_mc2 separately.",
        f"Total items = {n_items} (MMLU 14,042 + ARC 1,172 + HellaSwag 10,042 + TruthfulQA 817 + "
        f"Winogrande 1,267 + GSM8K 1,319); {M} x {len(TASKS)} = {B.size} is the number of "
        "model-task cells, not items.",
        "Items carry no question ids/text in the source; ids are lb.pickle row indices.",
        "The 395 models are the tinyBenchmarks selection of leaderboard v1 models evaluated "
        "2023-07..2024-01, not a random sample of the leaderboard.",
        f"For {len(det_models)} model(s) ({', '.join(det_models)}) {RESULTS_REPO} holds only "
        "the late-2023 winogrande/gsm8k re-run; the aggregate results file of their original "
        "run was taken from their details repo at a pinned commit (primary_results_source = "
        "'details_repo'), so their other 60 tasks are checked against official values and "
        "eval_timestamp/eval_date date the original run.",
        f"For {n_rerun} models winogrande and gsm8k come from the leaderboard's late-2023 re-run "
        "(tasks_from_other_results_file); eval_timestamp/eval_date refer to the run that "
        "produced the remaining tasks (primary_results_file).  For "
        f"{n_rerun_sha} of them the re-run's model_sha differs from the primary run's.",
        f"request.submitted_time is later than eval_timestamp for "
        f"{sum(bool((e['request'] or {}).get('submitted_after_eval')) for e in model_meta)} "
        "models: request files are keyed by model+precision and re-submissions overwrite "
        "them, so use eval_date as the model date for temporal splits.",
        "request.model_type is free text from the leaderboard; model_type_norm maps it to "
        "pretrained / fine-tuned / instruction-tuned / RL-tuned (None if unknown).",
    ]
    if no_res:
        meta["caveats"].append(f"{len(no_res)} model(s) have tasks without a matching official "
                               "results file: " + ", ".join(no_res))
    if dup_groups:
        meta["caveats"].append(f"{len(dup_groups)} group(s) of models have identical item "
                               "vectors (see duplicate_item_vectors).")

    (output_dir / f"{NAME}.meta.json").write_text(dump_json(meta, ("models",)))
    (output_dir / "raw_manifests" / f"{NAME}.json").write_text(
        dump_json(manifest, ("inputs",)))

    print(f"  MMLU vs mmlu.matrix: {mmlu_check['n_models_exact_name_match']} exact matches, "
          f"max |diff| {mmlu_check['max_abs_diff']:.3g}, mean {mmlu_check['mean_abs_diff']:.3g}, "
          f"{mmlu_check['n_models_max_abs_diff_le_1e-6']} models agree to 1e-6")
    for g, v in grp.items():
        print(f"  official results {g:15s} {v['metric']:9s} exact {v['n_exact']}/{v['n_cells']} "
              f"(with metric {v['n_with_metric']}), max closest diff {v['max_abs_diff_closest']:.2e}")
    print(f"  cells equal to an official aggregate: {n_cells_checked}/{B.size}; "
          f"primary run from details repo: {det_models}")
    print(f"  licenses: results={results_license} requests={requests_license} "
          f"details={det_declared or None} ({len(det_licenses)} checked)")
    print(f"  duplicate item-vector groups: {len(dup_groups)}")
    print(f"Done. Files saved to {output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=pathlib.Path,
                   default=pathlib.Path.home() / "benchselect-data" / "raw" / NAME,
                   help="Directory with the raw inputs (see docstring).")
    p.add_argument("--output-dir", type=pathlib.Path, default=HERE,
                   help="Directory for the generated files (default: data/).")
    p.add_argument("--mmlu-matrix", type=pathlib.Path, default=HERE / "mmlu.matrix.npz",
                   help="mmlu.matrix.npz used for the overlap / consistency check.")
    p.add_argument("--fetch", action="store_true",
                   help="Download the pinned raw inputs into --raw-dir first.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fetch:
        fetch(args.raw_dir.expanduser())
    build(args.raw_dir, args.output_dir, args.mmlu_matrix)
