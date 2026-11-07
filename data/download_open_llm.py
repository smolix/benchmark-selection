#!/usr/bin/env python3
"""Download and assemble the Open LLM Leaderboard v2 score matrix.

Produces (in --output-dir, default: this directory):
    open_llm.csv   — tidy CSV (rows = models, columns = benchmarks)
    open_llm.npz   — B (score matrix), O (observation mask), model_names,
                      benchmark_names

Requirements:
    pip install pandas numpy pyarrow          (offline, --raw-dir)
    pip install huggingface_hub python-dotenv (online download)

The leaderboard v2 benchmarks and the value stored for each (lm-evaluation-
harness metrics in [0, 1], not the leaderboard's normalised 0-100 scores):

    IFEval      leaderboard_ifeval     prompt_level_strict_acc
    BBH         leaderboard_bbh        acc_norm     group entry
    MATH_Lvl5   leaderboard_math_hard  exact_match  group entry
    GPQA        leaderboard_gpqa       acc_norm     group entry
    MUSR        leaderboard_musr       acc_norm     group entry
    MMLU-PRO    leaderboard_mmlu_pro   acc

A "group entry" is the file's own results[<group>][<metric>] value (the
harness's sample-weighted mean over the group's sub-tasks).  Where a file
leaves it blank, the sample-weighted mean of the group's sub-tasks in that
file is stored instead; a sub-task value is never stored as a group score.

Sources (pinned by commit sha, anonymous access; the pins, the result-file
parser and the aggregate definitions are imported from build_olv2sub.py):
    open-llm-leaderboard/results   ``<org>/<model>/results_<ts>.json``
    open-llm-leaderboard/contents  one row per leaderboard entry with the six
                                   "Raw" aggregates the leaderboard shows

Result-file selection, per results ``model_name``:
    1. The model has a non-flagged contents row and at least one complete
       (all six aggregates present), unlimited (no sample ``limit``) file:
       keep the file whose recomputed leaderboard aggregates are closest to
       the contents row in L_inf; ties (<= 1e-12) go to the latest timestamp,
       then the path; a model listed under two precisions keeps the row with
       the later Submission Date, then bfloat16, then eval_name.  This is the
       rule of build_olv2sub.py, so open_llm and olv2sub use the same file per
       model (checked against olv2sub.meta.json when it is present).
    1b. A leaderboard row that this file does not reproduce (L_inf > 1e-9;
       8 rows at the pinned revisions) can combine two files, e.g. BBH and
       MATH from the 2025-02-13 partial re-upload and the other four from
       the 2024-10-24 complete file.  Each aggregate that differs from the
       contents row by more than 1e-9 is taken from the latest other
       unlimited file of the model, complete or partial, whose recomputed
       aggregate equals the contents value (within 1e-9); MATH-hard
       sub-tasks without n-samples in that file use the sample counts of the
       rule-1 file (same 1,324 questions).  If no file matches, the rule-1
       value stays.  At the pinned revisions this takes BBH (5 rows) and
       MATH (3 rows) from the partial re-uploads and IFEval + MATH from a
       second complete file (HuggingFaceTB/SmolLM2-135M).
    2. Otherwise (model not in contents, flagged, or only sample-limited
       runs): the latest complete file, unlimited runs first.
       --leaderboard-only drops these rows.
    Files missing any of the six aggregates (e.g. the partial BBH + MATH
    re-uploads) are never a row's rule-1 / rule-2 file; they only supply the
    aggregates of rule 1b.

Rows are in the order of each model's last result file in path order (the
row order of the original download); columns in the order listed above.

History: the original version of this script (snapshot of 2026-03-13, frozen
as open_llm_v1.{csv,npz}) kept the LAST result file in path order and, where a
group entry was blank, fell back to the first sub-task whose key contains the
group key (e.g. leaderboard_bbh_boolean_expressions for BBH).  At the pinned
revisions that gave 47 wrong rows / 200 cells; see README.md in this directory.

Usage:
    python download_open_llm.py             # online; skips if outputs exist
    cd /tmp && python -I .../data/download_open_llm.py \\
        --raw-dir ~/benchselect-data/raw/olv2sub --output-dir DIR --force

``--raw-dir`` has the layout described in build_olv2sub.py (``results_git/``,
a clone of the results repo at RESULTS_REV, and
``contents/train-00000-of-00001.parquet``).  The results files are checked
against RESULT_FILES_SHA256, the sha256 of their sorted ``sha256sum`` listing
pinned below, and the contents parquet against CONTENTS_SHA256, so offline and
online runs read the same bytes; there is no way to skip the check.  The
per-file listing in raw_manifests/olv2sub.json (--manifest-dir) only serves to
name the differing files.  Only json / pandas / numpy parse the (untrusted)
downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent

# ``python -I`` does not put the script directory on sys.path.
sys.path.insert(0, str(HERE))
import build_olv2sub as olv2  # noqa: E402  (pins, parser, aggregate definitions)

REPO_ID = olv2.RESULTS_REPO
RESULTS_REV = olv2.RESULTS_REV
CONTENTS_REPO = olv2.CONTENTS_REPO
CONTENTS_REV = olv2.CONTENTS_REV
CONTENTS_FILE = olv2.CONTENTS_FILE
# sha256 of the `LC_ALL=C sort`ed `sha256sum` listing of the results_*.json files
# at RESULTS_REV, i.e. in the clone: `find . -name 'results_*.json' ! -path
# './.git/*' | cut -c3- | LC_ALL=C sort | xargs -d '\n' sha256sum | sha256sum`
# (per-file hashes in raw_manifests/olv2sub.json)
RESULT_FILES_SHA256 = "2784cd40ec7911c423bfe213e30e44b922c7d1cbfba0fb7b1530c097ffeb7503"
RESULT_FILES_N = 10509

# Mapping: benchmark display name → (group key in "results" dict, metric key)
BENCHMARK_MAP = {
    "IFEval": ("leaderboard_ifeval", "prompt_level_strict_acc,none"),
    "BBH": ("leaderboard_bbh", "acc_norm,none"),
    "MATH_Lvl5": ("leaderboard_math_hard", "exact_match,none"),
    "GPQA": ("leaderboard_gpqa", "acc_norm,none"),
    "MUSR": ("leaderboard_musr", "acc_norm,none"),
    "MMLU-PRO": ("leaderboard_mmlu_pro", "acc,none"),
}
assert list(BENCHMARK_MAP) == olv2.AGG_NAMES

RULE_LEADERBOARD = "leaderboard_match"
RULE_LATEST = "latest_complete"
RULE_LATEST_LIMITED = "latest_complete_limited_run"


def _with_counts(rec: dict, n_eff: dict) -> dict:
    """``rec`` with the sample counts of tasks it has no n-samples for taken
    from ``n_eff`` (another file of the same model; rule 1b)."""
    merged = dict(n_eff)
    merged.update({t: n for t, n in rec["n_eff"].items() if n is not None})
    return {**rec, "n_eff": merged}


def _extract_scores(rec: dict) -> dict:
    """The six values stored in open_llm.csv for one parsed result file
    (``build_olv2sub.parse_result_file`` record); None where unavailable."""
    leaf, n_eff = rec["leaf"], rec["n_eff"]
    scores = {}
    for label, _gkey, _metric, cols in olv2.GROUPS:
        if label == "IFEval":
            val = leaf["ifeval/prompt_level_strict_acc"]
        elif label == "MMLU-PRO":
            val = leaf["mmlu_pro"]
        else:
            val = rec["group_vals"].get(label)
            if val is None:  # blank group entry: sample-weighted sub-task mean
                v = [leaf[c] for c, _ in cols]
                w = [n_eff.get(t) for _, t in cols]
                if all(x is not None for x in v) and all(isinstance(x, int) for x in w):
                    val = float(np.dot(v, w) / np.sum(w))
        scores[label] = val
    return scores


def parse_results(results_root: pathlib.Path) -> list[dict]:
    """Parse every results_*.json under ``results_root`` (sorted path order)."""
    rels = olv2.list_result_files(results_root)
    print(f"  Parsing {len(rels)} result files …")
    recs = [olv2.parse_result_file(results_root, r) for r in rels]
    for r in recs:
        r["agg"] = olv2.leaderboard_aggregates(r)
        r["complete"] = all(r["agg"][a] is not None for a in olv2.AGG_NAMES)
    return recs


def result_files_digest(recs: list[dict]) -> str:
    """sha256 of the ``sha256sum`` listing of the parsed files in byte order."""
    listing = "".join(f"{r['sha256']}  {r['rel']}\n"
                      for r in sorted(recs, key=lambda r: r["rel"].encode()))
    return hashlib.sha256(listing.encode()).hexdigest()


def verify_inputs(recs: list[dict], contents_path: pathlib.Path,
                  manifest_path: pathlib.Path) -> None:
    """Refuse result files or a contents table other than the pinned ones.

    The check uses the digests pinned in this module and does not depend on
    ``manifest_path``; the manifest, when present, names the differing files.
    """
    csha = olv2.sha256_file(contents_path)
    if csha != olv2.CONTENTS_SHA256:
        raise SystemExit(f"contents parquet sha256 {csha} != pinned {olv2.CONTENTS_SHA256}")
    digest = result_files_digest(recs)
    if digest == RESULT_FILES_SHA256 and len(recs) == RESULT_FILES_N:
        print(f"  {len(recs)} result files and the contents table match the pinned "
              f"revisions ({RESULTS_REV[:7]}, {CONTENTS_REV[:7]})")
        return
    msg = [f"result files digest {digest} ({len(recs)} files) != pinned "
           f"{RESULT_FILES_SHA256} ({RESULT_FILES_N} files, revision {RESULTS_REV})"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pinned = {f["path"]: f["sha256"] for f in manifest["inputs"][0]["files"]
                  if os.path.basename(f["path"]).startswith("results_")
                  and f["path"].endswith(".json")}
        have = {r["rel"]: r["sha256"] for r in recs}
        bad = sorted(p for p in set(pinned) | set(have) if pinned.get(p) != have.get(p))
        msg.append(f"{len(bad)} result files differ from {manifest_path}: {bad[:20]}")
    else:
        msg.append(f"({manifest_path} not found: cannot name the differing files)")
    raise SystemExit("\n".join(msg))


def _group_sources(base: dict, files: list[dict], target: np.ndarray) -> dict:
    """Rule 1b: {aggregate: record} for the aggregates of a leaderboard row that
    ``base`` (the rule-1 file) does not reproduce within EXACT_TOL, each from
    the latest other unlimited file whose recomputed aggregate does."""
    out = {}
    for a, t in zip(olv2.AGG_NAMES, target):
        if abs(base["agg"][a] - t) <= olv2.EXACT_TOL:
            continue
        hits = []
        for r in files:
            if r is base or r["limited"]:
                continue
            v = olv2.leaderboard_aggregates(_with_counts(r, base["n_eff"]))[a]
            if v is not None and abs(v - t) <= olv2.EXACT_TOL:
                hits.append(r)
        if hits:
            out[a] = max(hits, key=lambda r: (r["ts"], r["rel"]))
    return out


def select_files(recs: list[dict], contents: pd.DataFrame,
                 leaderboard_only: bool = False) -> tuple[dict, dict]:
    """{model_name: (record, rule, {aggregate: record of rule 1b})} and
    selection counts (see module doc)."""
    by_model: dict[str, list[dict]] = {}
    for r in recs:
        by_model.setdefault(r["model_name"], []).append(r)

    contents = contents.sort_values("eval_name", kind="mergesort").reset_index(drop=True)
    flagged = set(contents.loc[contents["Flagged"].astype(bool), "fullname"])
    contents_ok = contents[~contents["fullname"].isin(flagged)]

    # rule 1: the file matching the leaderboard row (build_olv2sub.py)
    matched: dict[str, list] = {}
    for _, crow in contents_ok.iterrows():
        cands = [r for r in by_model.get(crow["fullname"], [])
                 if r["complete"] and not r["limited"]]
        if not cands:
            continue
        target = np.array([crow[olv2.CONTENTS_RAW[a]] for a in olv2.AGG_NAMES], float)
        scored = [(float(np.max(np.abs(
            np.array([r["agg"][a] for a in olv2.AGG_NAMES], float) - target))), r)
            for r in cands]
        best_d = min(d for d, _ in scored)
        tied = [r for d, r in scored if d - best_d <= olv2.TIE_TOL]
        tied.sort(key=lambda r: (r["ts"], r["rel"]))
        matched.setdefault(crow["fullname"], []).append((crow, tied[-1], target, best_d))

    def prec_key(m):
        c = m[0]
        return (c["Submission Date"] or "", 1 if c["Precision"] == "bfloat16" else 0,
                c["eval_name"])

    chosen = {}
    mixed = {}
    for name, ms in matched.items():
        _crow, rec, target, best_d = sorted(ms, key=prec_key)[-1]
        extra = {}
        if best_d > olv2.EXACT_TOL:  # rule 1b
            extra = _group_sources(rec, by_model[name], target)
            mixed[name] = {"linf": best_d, "file": rec["rel"],
                           "aggregates_from_other_files":
                               {a: r["rel"] for a, r in extra.items()}}
        chosen[name] = (rec, RULE_LEADERBOARD, extra)

    # rule 2: models the leaderboard table cannot place
    unplaced = {"not_in_contents": [], "flagged": [], "only_limited_runs": [],
                "no_complete_file": []}
    in_contents = set(contents["fullname"])
    for name, rs in by_model.items():
        if name in chosen:
            continue
        reason = ("flagged" if name in flagged else
                  "not_in_contents" if name not in in_contents else
                  "only_limited_runs")
        complete = [r for r in rs if r["complete"]]
        if not complete:
            unplaced["no_complete_file"].append(name)
            continue
        unplaced[reason].append(name)
        if leaderboard_only:
            continue
        rec = max(complete, key=lambda r: (not r["limited"], r["ts"], r["rel"]))
        chosen[name] = (rec, RULE_LATEST_LIMITED if rec["limited"] else RULE_LATEST, {})

    counts = {
        "result_files": len(recs),
        "result_files_incomplete": sum(not r["complete"] for r in recs),
        "result_files_with_limit": sum(r["limited"] for r in recs),
        "models_in_results": len(by_model),
        "rows_by_rule": {k: sum(1 for _, rule, _ in chosen.values() if rule == k)
                         for k in (RULE_LEADERBOARD, RULE_LATEST, RULE_LATEST_LIMITED)},
        "leaderboard_rows_not_reproduced_by_one_file": dict(sorted(mixed.items())),
        "models_not_placed_by_leaderboard": {k: sorted(v) for k, v in unplaced.items()},
    }
    return chosen, counts


def row_scores(rec: dict, extra: dict) -> dict:
    """Stored values of one row: the rule-1/2 file, with the rule-1b aggregates
    taken from their files (missing n-samples from ``rec``)."""
    scores = _extract_scores(rec)
    for a, r in extra.items():
        scores[a] = _extract_scores(_with_counts(r, rec["n_eff"]))[a]
    return scores


def build_frame(recs: list[dict], chosen: dict) -> pd.DataFrame:
    """Score DataFrame, rows ordered by each model's last file in path order."""
    last = {}
    for i, r in enumerate(recs):
        last[r["model_name"]] = i
    names = sorted(chosen, key=last.get)
    rows = [row_scores(chosen[n][0], chosen[n][2]) for n in names]
    df = pd.DataFrame(rows, index=pd.Index(names, name="model"),
                      columns=list(BENCHMARK_MAP), dtype=np.float64)
    return df.dropna(how="all")


def check_against_olv2sub(chosen: dict, meta_path: pathlib.Path) -> None:
    """Rule-1 rows must use the file that olv2sub.meta.json records."""
    if not meta_path.exists():
        print(f"  note: {meta_path.name} not found; olv2sub cross-check skipped")
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ref = dict(zip(meta["model_names"], meta["models"]["result_file"]))
    mine = {n: rec["rel"] for n, (rec, rule, _) in chosen.items() if rule == RULE_LEADERBOARD}
    bad = sorted(n for n in set(ref) | set(mine) if ref.get(n) != mine.get(n))
    if bad:
        raise SystemExit(f"{len(bad)} models select a different file than "
                         f"{meta_path.name}: {bad[:10]}")
    print(f"  same result file as {meta_path.name} for all {len(mine)} leaderboard rows")


def compare_reference(df: pd.DataFrame, ref_csv: pathlib.Path) -> None:
    """Print how ``df`` differs from a reference open_llm.csv (cells, tol 1e-9)."""
    ref = pd.read_csv(ref_csv, index_col=0, float_precision="round_trip")
    common = df.index.intersection(ref.index)
    a, b = df.loc[common, ref.columns].values, ref.loc[common].values
    filled = np.isnan(b) & ~np.isnan(a)
    emptied = ~np.isnan(b) & np.isnan(a)
    changed = ~np.isnan(a) & ~np.isnan(b) & (np.abs(a - b) > 1e-9)
    bad = filled | emptied | changed
    print(f"  vs {ref_csv.name}: {len(df)} vs {len(ref)} rows "
          f"({len(df.index.difference(ref.index))} new, "
          f"{len(ref.index.difference(df.index))} missing); "
          f"{int(bad.any(axis=1).sum())} common rows / {int(bad.sum())} cells differ "
          f"({int(changed.sum())} changed, {int(filled.sum())} filled, "
          f"{int(emptied.sum())} emptied)")


def fetch_online(cache_dir: pathlib.Path | None) -> tuple[pathlib.Path, pathlib.Path]:
    """Download the pinned results JSONs and contents parquet (HF Hub cache)."""
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download, snapshot_download

    # Load HF_TOKEN (and any other vars) from .env next to this script
    load_dotenv(HERE / ".env")
    print(f"Downloading {REPO_ID} @ {RESULTS_REV[:7]} (*.json) …")
    results_root = pathlib.Path(snapshot_download(
        REPO_ID, repo_type="dataset", revision=RESULTS_REV,
        allow_patterns="*.json", cache_dir=cache_dir))
    print(f"Downloading {CONTENTS_REPO} @ {CONTENTS_REV[:7]} ({CONTENTS_FILE}) …")
    contents_path = pathlib.Path(hf_hub_download(
        CONTENTS_REPO, CONTENTS_FILE, repo_type="dataset", revision=CONTENTS_REV,
        cache_dir=cache_dir))
    return results_root, contents_path


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()

    # ── Skip if data already exists ───────────────────────────────
    csv_path = out_dir / "open_llm.csv"
    npz_path = out_dir / "open_llm.npz"
    if not args.force and csv_path.exists() and npz_path.exists():
        df_check = pd.read_csv(csv_path, index_col=0)
        if df_check.shape[0] > 0 and df_check.shape[1] > 0:
            print(f"Open LLM data already exists ({df_check.shape[0]} models × "
                  f"{df_check.shape[1]} benchmarks).  Skipping download "
                  f"(--force rebuilds).")
            return

    if args.raw_dir is not None:
        raw = args.raw_dir.resolve()
        results_root = raw / "results_git"
        contents_path = raw / "contents" / "train-00000-of-00001.parquet"
        if (results_root / ".git").exists():
            head = olv2.git_head(results_root)
            if head != RESULTS_REV:
                raise SystemExit(f"results clone is at {head}, expected {RESULTS_REV}")
        print(f"Reading pinned raw data from {raw}")
    else:
        results_root, contents_path = fetch_online(args.cache_dir)

    recs = parse_results(results_root)
    verify_inputs(recs, contents_path, args.manifest_dir / "olv2sub.json")
    contents = pd.read_parquet(contents_path)
    chosen, counts = select_files(recs, contents, args.leaderboard_only)
    print(f"  {counts['models_in_results']} models in {counts['result_files']} files "
          f"({counts['result_files_incomplete']} incomplete and "
          f"{counts['result_files_with_limit']} sample-limited files are not "
          f"eligible for rule 1)")
    print(f"  Rows by selection rule: {counts['rows_by_rule']}")
    mixed = counts["leaderboard_rows_not_reproduced_by_one_file"]
    print(f"  {len(mixed)} leaderboard rows match no single file (L_inf > "
          f"{olv2.EXACT_TOL:g}); rule 1b takes aggregates from a second file in "
          f"{sum(bool(m['aggregates_from_other_files']) for m in mixed.values())}:")
    for name, m in mixed.items():
        other = {a: rel.rsplit("/", 1)[-1] for a, rel in m["aggregates_from_other_files"].items()}
        print(f"    {name}: L_inf {m['linf']:.2g}, {m['file'].rsplit('/', 1)[-1]}"
              + (f" + {other}" if other else ""))
    for reason, names in counts["models_not_placed_by_leaderboard"].items():
        if names:
            print(f"    {reason}: {len(names)} {names}")
    check_against_olv2sub(chosen, args.olv2sub_meta)

    df = build_frame(recs, chosen)
    if args.reference_csv is not None:
        compare_reference(df, args.reference_csv)
    _save(df, out_dir)


def _save(df: pd.DataFrame, out_dir: pathlib.Path = HERE):
    benchmarks = list(BENCHMARK_MAP.keys())
    df = df[benchmarks]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Score matrix: {df.shape[0]} models × {df.shape[1]} benchmarks")
    n_missing = df.isna().sum().sum()
    n_total = df.shape[0] * df.shape[1]
    print(f"  Missing entries: {n_missing}/{n_total} ({n_missing/n_total:.1%})")
    print(f"  Benchmarks: {list(df.columns)}")

    csv_path = out_dir / "open_llm.csv"
    df.to_csv(csv_path)
    print(f"  Saved {csv_path}")

    B = df.values.astype(np.float64)
    O = (~np.isnan(B)).astype(np.float64)
    B = np.nan_to_num(B, nan=0.0)
    npz_path = out_dir / "open_llm.npz"
    np.savez(
        npz_path,
        B=B,
        O=O,
        model_names=np.array(df.index.tolist()),
        benchmark_names=np.array(benchmarks),
    )
    print(f"  Saved {npz_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=pathlib.Path,
                        help="offline: pinned raw data (results_git/ + contents/), "
                             "e.g. ~/benchselect-data/raw/olv2sub; default downloads "
                             "the pinned revisions from the HF Hub")
    parser.add_argument("--output-dir", type=pathlib.Path, default=HERE,
                        help="where open_llm.{csv,npz} are written (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the outputs already exist")
    parser.add_argument("--leaderboard-only", action="store_true",
                        help="keep only rows matched to the leaderboard contents table "
                             "(rule 1); drops the rule-2 rows")
    parser.add_argument("--cache-dir", type=pathlib.Path,
                        help="online: HF Hub cache directory (default: HF_HOME)")
    parser.add_argument("--manifest-dir", type=pathlib.Path, default=HERE / "raw_manifests",
                        help="directory with olv2sub.json, the per-file sha256 listing "
                             "used to name the files that fail the pinned-digest "
                             "check; the check itself does not need it "
                             "(default: %(default)s)")
    parser.add_argument("--olv2sub-meta", type=pathlib.Path, default=HERE / "olv2sub.meta.json",
                        help="cross-check the selected files against this olv2sub meta "
                             "file if it exists (default: %(default)s)")
    parser.add_argument("--reference-csv", type=pathlib.Path,
                        help="print the differences to this open_llm.csv, "
                             "e.g. open_llm_v1.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
