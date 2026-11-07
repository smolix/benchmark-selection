"""Build the Epoch AI Benchmarking Hub score matrix (models x benchmarks).

Source
------
Epoch AI, "Capabilities & benchmarking" (Benchmarking Hub), CC BY 4.0,
https://epoch.ai/data/benchmark_data.zip.  This is a living dataset without
versioned releases, so a snapshot is pinned by the sha256 of the zip (plus
the HTTP ETag and the generation timestamp of the zip members), all recorded
in raw_manifests/epoch.json and epoch.meta.json.  Fetch it once with

    curl -sS -L -D headers.txt -o benchmark_data.zip \\
        https://epoch.ai/data/benchmark_data.zip
    date -u +%Y-%m-%dT%H:%M:%SZ > fetched_at.txt

into --raw-dir.  The builder refuses any zip whose sha256 differs from the
pinned snapshot (PINNED_SHA256, fetched 2026-10-17T23:56:28Z); a re-fetch of
the URL will usually differ, so keep the pinned zip with the data release.
Pass --expected-sha256 any to build from another snapshot on purpose.  The
zip is read in place (zipfile + pandas): nothing is extracted and no
downloaded code is executed.  All CSVs are parsed with
float_precision='round_trip' (and string-typed numeric columns with Python
float()), so every value is the correctly rounded double of Epoch's decimal
string.

Matrix
------
Values come from the curated, name-normalized table
``epoch_capabilities_index/processed_data_for_eci.csv`` that Epoch uses to fit
the Epoch Capabilities Index (ECI).  It has one row per (model, benchmark):

    performance = max(0, (max_r s_r - random_baseline)
                         / (score_ceiling - random_baseline))

where s_r = row[score_column] * scale for a row r of the benchmark's raw CSV,
and the max runs over every row whose ``Model version`` maps to the same
``model_group`` in model_metadata.csv.  That covers dated snapshots,
reasoning-effort / thinking-budget variants (``_high``, ``_xhigh``,
``_16K``, ...), provider-hosted copies, agents/scaffolds, and in some cases
the base and instruct checkpoints of one release.  score_column, scale,
random_baseline and score_ceiling come from benchmark_metadata.csv.  The
builder re-derives every cell from the raw CSVs and records the cells it
cannot reproduce (``cells_not_reproducible_from_raw`` in the meta file).

    row (model)      Epoch ``model_group`` (= ``model`` column of the table)
    column           ``benchmark`` column of the table (ECI benchmarks)
    value            ``performance`` in [0, 1]: chance-corrected,
                     ceiling-normalized best score over the merged variants
    missing          no row in the processed table.  Mostly not evaluated /
                     not reported, but some missing cells do have raw rows:
                     Epoch drops a superseded benchmark (benchmark_metadata
                     ``superseded_by``, the old FrontierMath versions) for every
                     model scored on its successor, so newer models are absent
                     from the old columns by design, and a few raw rows are
                     newer than the processed table.  These cells are listed
                     in ``cells_missing_but_in_raw`` (not filled in).

Order: rows sorted by (name.casefold(), name), columns sorted the same way.
Models with fewer than --min-obs observed benchmarks are dropped (default 2,
as for the other matrices; ECI itself keeps only models with >= 4 benchmarks,
so the default drops nothing at the pinned snapshot).

Metadata (epoch.meta.json)
--------------------------
models      release_date (ECI ``date`` = latest model_metadata date among the
            versions behind the row), release_date_first (earliest
            model_metadata date in the group), release quarter, organization,
            country, accessibility, open_weights, training compute, ECI score,
            and the merged model versions.
benchmarks  release date, score definition (file, column, scale, baseline,
            ceiling), EDI difficulty, item count where one is available, and
            the per-row cost/token columns of the raw CSV.
cells       for each cell the raw-CSV row achieving the max (its model
            version, raw score and cost/token values), Epoch's own
            model_version label and attribution ``source`` of the processed
            row, all cells that do not reproduce from the raw CSVs, and all
            missing cells that have raw rows (cells_missing_but_in_raw).
overlap     Epoch models that match a canonical model of
            canonical_mapping.json / merged.matrix.npz by model-version string.

Outputs (in --output-dir)
-------------------------
    epoch.matrix.csv, epoch.matrix.npz   via build_matrices.save_matrix
    epoch.meta.json                      metadata described above
    raw_manifests/epoch.json             raw-input manifest (zip, headers.txt,
                                         fetched_at.txt, and all zip members)

Running the builder twice on the same raw inputs gives byte-identical CSV,
.npz (np.savez writes fixed zip-member timestamps), meta and manifest.
"""

import argparse
import hashlib
import io
import json
import platform
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from build_matrices import save_matrix  # noqa: E402

NAME = "epoch"
SOURCE_URL = "https://epoch.ai/data/benchmark_data.zip"
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
CITATION = ("Epoch AI, 'Capabilities & benchmarking'. Published online at epoch.ai. "
            "Retrieved from 'https://epoch.ai/benchmarks' [online resource].")
ZIP_NAME = "benchmark_data.zip"
FETCH_SIDE_FILES = {
    "headers.txt": "HTTP response headers of the fetch (curl -D); source of etag/http_date",
    "fetched_at.txt": "UTC fetch timestamp (date -u); source of fetched_at",
}
# Pinned snapshot (fetched 2026-10-17T23:56:28Z, ETag 42074755daa96fd78abe382ba976dbe6).
PINNED_SHA256 = "e1ce578cc525c6e565ae31bd65618027f8bbf68309a8e6dd168ee75c61100818"
PINNED_SIZE = 2401423
PROCESSED = "epoch_capabilities_index/processed_data_for_eci.csv"
ECI_SCORES = "epoch_capabilities_index/eci_scores.csv"
EDI_SCORES = "epoch_capabilities_index/edi_scores.csv"
MODEL_META = "model_metadata.csv"
BENCH_META = "benchmark_metadata.csv"

DEFAULT_RAW = Path("~/benchselect-data/raw/epoch").expanduser()
TOL = 1e-6

# Raw-CSV columns reported as per-run cost / token / effort figures.
COST_COL_RE = re.compile(r"cost|spend|tokens|latency|duration|time per case|steps|step budget", re.I)
# Item-count inference from stderr: accepted only if the runs agree this well.
N_ITEMS_MAX_SPREAD = 0.05
N_ITEMS_MIN_RUNS = 5
# Release-date cutoffs reported for prospective (temporal) splits.
SPLIT_CUTOFFS = ["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
                 "2026-01-01", "2026-04-01", "2026-07-01"]
# Model-version suffixes stripped before matching canonical model names.
EFFORT_SUFFIX_RE = re.compile(
    r"[_](minimal|low|medium|high|xhigh|max|promax|unknown|none|thinking|\d+k)$", re.I)


def sort_key(s):
    return (s.casefold(), s)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(z, member):
    # round_trip: correctly rounded doubles (pandas' default C parser is off by
    # a few ulp for strings with >= 17 significant digits).
    return pd.read_csv(io.BytesIO(z.read(member)), float_precision="round_trip")


NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _parse_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = re.sub(r"[$,\s]", "", str(x))
    return float(s) if NUM_RE.fullmatch(s) else np.nan


def to_num(series):
    """Numeric parse of a raw column, tolerating '$' and thousands separators.

    String cells are converted with Python float() (correctly rounded) after a
    strict decimal-number regex; anything else becomes NaN.
    """
    if series.dtype.kind in "fiu":
        return series.astype(float)
    return series.map(_parse_num).astype(float)


def file_entry(path):
    """size + sha256 of a local side file, or None if absent."""
    if not path.exists():
        return None
    data = path.read_bytes()
    return {"size": len(data), "sha256": sha256(data)}


def jv(x):
    """JSON-safe scalar: numpy -> python, NaN/NaT/None -> None."""
    if x is None:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, str):
        return x
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x


def quarter(date):
    if not isinstance(date, str) or len(date) < 7:
        return None
    y, m = int(date[:4]), int(date[5:7])
    return f"{y}Q{(m - 1) // 3 + 1}"


def norm_model_id(s):
    """Normalize a model identifier for exact-match lookup against canonical names."""
    s = str(s).strip().lower()
    s = s.split("/")[-1]
    s = re.sub(r"\s*\(.*?\)\s*", "", s)
    s = re.sub(r":\d+$", "", s)
    prev = None
    while prev != s:
        prev = s
        s = EFFORT_SUFFIX_RE.sub("", s)
    return s.replace("_", "-").replace(" ", "-")


def read_fetch_info(raw_dir):
    info = {"fetched_at": None, "etag": None, "http_date": None, "content_length": None}
    p = raw_dir / "fetched_at.txt"
    if p.exists():
        info["fetched_at"] = p.read_text().strip() or None
    p = raw_dir / "headers.txt"
    if p.exists():
        # Last response block (after redirects), plain "key: value" lines.
        block = re.split(r"\r?\n\r?\n", p.read_text(errors="replace").strip())[-1]
        for line in block.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip().lower(), v.strip()
            if k == "etag":
                info["etag"] = v.strip('"')
            elif k == "date":
                info["http_date"] = v
            elif k == "content-length":
                info["content_length"] = int(v) if v.isdigit() else None
    return info


def infer_n_items(raw, score_col, scale):
    """Item count per benchmark: explicit column, else stderr inversion if runs agree."""
    if "Total tasks" in raw.columns:
        vals = sorted(set(to_num(raw["Total tasks"]).dropna().astype(int)))
        if len(vals) == 1:
            return vals[0], "column 'Total tasks'"
    if "stderr" in raw.columns:
        col = "mean_score" if "mean_score" in raw.columns else score_col
        p = to_num(raw[col]) * (scale if col == score_col else 1.0)
        se = to_num(raw["stderr"])
        ok = p.between(0.05, 0.95) & (se > 0)
        n = (p[ok] * (1 - p[ok]) / se[ok] ** 2 + 1)
        if len(n) >= N_ITEMS_MIN_RUNS:
            med, lo, hi = n.median(), n.quantile(0.1), n.quantile(0.9)
            if (hi - lo) / med <= N_ITEMS_MAX_SPREAD:
                return int(round(med)), (
                    f"inferred: median over {len(n)} runs of p(1-p)/stderr^2 + 1 "
                    f"(runs with 0.05<p<0.95; p10={lo:.1f}, p90={hi:.1f})")
    return None, None


def canonical_overlap(models, group_versions, canonical_path, merged_path):
    if canonical_path is None or not Path(canonical_path).exists():
        return {"available": False}
    with open(canonical_path) as f:
        canon = json.load(f)["canonical_models"]
    merged_names = None
    if merged_path is not None and Path(merged_path).exists():
        with np.load(merged_path) as m:
            merged_names = set(m["model_names"].tolist())
    ckeys = {}
    for g in canon:
        keys = {norm_model_id(g["canonical_name"])}
        keys |= {norm_model_id(v) for v in g["datasets"].values()}
        ckeys[g["canonical_name"]] = keys
    pairs = []
    for m in models:
        ekeys = {norm_model_id(v) for v in group_versions.get(m, [])}
        for cname in sorted(ckeys, key=sort_key):
            hit = sorted(ekeys & ckeys[cname])
            if hit:
                pairs.append({"epoch_model": m, "canonical_name": cname,
                              "in_merged_matrix": (None if merged_names is None
                                                   else cname in merged_names),
                              "matched_keys": hit})
    em = sorted({p["epoch_model"] for p in pairs}, key=sort_key)
    cm = sorted({p["canonical_name"] for p in pairs}, key=sort_key)
    return {
        "available": True,
        "method": ("exact match after normalization (lowercase, drop provider prefix "
                   "'.../', parenthetical, ':N' suffix and effort/budget suffixes such as "
                   "_high/_xhigh/_16K, '_'->'-') between every model_metadata "
                   "model_version of an Epoch model_group and every canonical_name / "
                   "per-collection alias in canonical_mapping.json"),
        "n_epoch_models": len(models),
        "n_canonical_models": len(canon),
        "n_epoch_models_matched": len(em),
        "n_canonical_models_matched": len(cm),
        "n_canonical_matched_in_merged_matrix": (None if merged_names is None else
                                                 sum(c in merged_names for c in cm)),
        "epoch_models_matching_several_canonical": sorted(
            [k for k, v in Counter(p["epoch_model"] for p in pairs).items() if v > 1],
            key=sort_key),
        "pairs": pairs,
    }


def build_epoch(raw_dir, output_dir, manifest_dir=None, canonical_path=None,
                merged_path=None, min_obs=2, expected_sha256=PINNED_SHA256):
    raw_dir = Path(raw_dir).expanduser().resolve()
    output_dir = Path(output_dir).resolve()
    manifest_dir = Path(manifest_dir).resolve() if manifest_dir else output_dir / "raw_manifests"

    zip_path = raw_dir / ZIP_NAME
    blob = zip_path.read_bytes()
    zip_sha = sha256(blob)
    check_pin = expected_sha256 is not None and expected_sha256.lower() != "any"
    if check_pin and zip_sha != expected_sha256.lower():
        raise ValueError(
            f"{zip_path}: sha256 {zip_sha} (size {len(blob)}) != expected {expected_sha256}. "
            "Epoch's URL is a living file; use the pinned snapshot, or pass "
            "--expected-sha256 any to build from a different one on purpose.")
    side_files = {n: file_entry(raw_dir / n) for n in FETCH_SIDE_FILES}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(blob))
    members = []
    for zi in sorted(z.infolist(), key=lambda i: i.filename):
        name = zi.filename
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"unsafe member name in zip: {name!r}")
        if zi.is_dir():
            continue
        members.append({"name": name, "size": zi.file_size, "sha256": sha256(z.read(name)),
                        "date_time": "%04d-%02d-%02dT%02d:%02d:%02d" % zi.date_time})
    generated_at = max(m["date_time"] for m in members)
    fetch = read_fetch_info(raw_dir)
    print(f"Reading {zip_path} ({len(blob)} bytes, sha256 {zip_sha})")

    mm = read_csv(z, MODEL_META)
    bm = read_csv(z, BENCH_META)
    proc = read_csv(z, PROCESSED)
    eci = read_csv(z, ECI_SCORES)
    edi = read_csv(z, EDI_SCORES)

    checks = {}
    # --- consistency checks on the processed table ---
    checks["processed_rows"] = int(len(proc))
    checks["processed_unique_model_benchmark"] = bool(
        not proc.duplicated(["model", "benchmark"]).any())
    checks["processed_model_eq_Model"] = bool((proc["model"] == proc["Model"]).all())
    checks["processed_models_all_in_model_group"] = bool(
        proc["model"].isin(set(mm["model_group"].dropna())).all())
    checks["processed_versions_all_in_model_metadata"] = bool(
        proc["model_version"].isin(set(mm["model_version"].dropna())).all())
    checks["processed_benchmarks_all_in_benchmark_metadata"] = bool(
        proc["benchmark"].isin(set(bm["benchmark"])).all())
    checks["processed_performance_range"] = [float(proc["performance"].min()),
                                             float(proc["performance"].max())]
    bdate = bm.set_index("benchmark")["release_date"]
    checks["benchmark_release_date_matches_metadata"] = bool(
        (proc["benchmark_release_date"] == proc["benchmark"].map(bdate)).all())
    for k, v in checks.items():
        if v is False:
            raise ValueError(f"consistency check failed: {k}")

    # --- matrix ---
    df = proc.pivot(index="model", columns="benchmark", values="performance")
    df = df.reindex(index=sorted(df.index, key=sort_key),
                    columns=sorted(df.columns, key=sort_key))
    df.index.name = None
    df.columns.name = None
    obs = df.notna().sum(axis=1)
    dropped = sorted(obs[obs < min_obs].index.tolist(), key=sort_key)
    if dropped:
        print(f"  Dropping {len(dropped)} models with < {min_obs} observations: {dropped}")
        df = df.drop(index=dropped)
    df = df.dropna(axis=1, how="all")
    print("Building Epoch matrix...")
    save_matrix(NAME, df, output_dir)
    models, benches = df.index.tolist(), df.columns.tolist()

    # --- per-model metadata ---
    eci_i = eci.set_index("Model")
    checks["eci_scores_models_eq_matrix_models"] = bool(set(eci_i.index) == set(models) | set(dropped))
    pver = proc.groupby("model")["date"].max()
    checks["eci_date_eq_latest_processed_version_date"] = int(
        (eci_i["date"].reindex(models) == pver.reindex(models)).sum())
    mm_g = mm[mm["model_group"].isin(models)]
    group_versions = {g: sorted(set(d["model_version"].dropna()), key=sort_key)
                      for g, d in mm_g.groupby("model_group")}
    model_meta = []
    for m in models:
        g = mm_g[mm_g["model_group"] == m]
        e = eci_i.loc[m] if m in eci_i.index else None
        dates = sorted(g["date"].dropna().unique())
        comp = sorted(set(g["training_compute_flop"].dropna().astype(float)))
        acc_group = jv(e["Accessibility group"]) if e is not None else None
        model_meta.append({
            "name": m,
            "release_date": jv(e["date"]) if e is not None else (dates[-1] if dates else None),
            "release_date_first": dates[0] if dates else None,
            "release_quarter": quarter(jv(e["date"]) if e is not None else None),
            "organization": jv(e["Organization"]) if e is not None else None,
            "country": jv(e["Country (of organization)"]) if e is not None else None,
            "accessibility": jv(e["Model accessibility"]) if e is not None else None,
            "accessibility_group": acc_group,
            "open_weights": {"Open weights": True, "Closed weights": False}.get(acc_group),
            "training_compute_flop": comp[-1] if comp else None,
            "training_compute_flop_ambiguous": len(comp) > 1,
            "eci": jv(e["eci"]) if e is not None else None,
            "eci_ci": [jv(e["eci_ci_low"]), jv(e["eci_ci_high"])] if e is not None else None,
            "n_observed": int(df.loc[m].notna().sum()),
            "model_versions": group_versions.get(m, []),
        })

    # --- per-benchmark metadata and per-cell provenance from raw CSVs ---
    v2g = (mm.dropna(subset=["model_version"]).drop_duplicates("model_version")
           .set_index("model_version")["model_group"])
    bm_i = bm.set_index("benchmark")
    edi_i = edi.set_index("benchmark_name")
    bid = proc.drop_duplicates("benchmark").set_index("benchmark")["benchmark_id"]
    proc_models = {b: set(d["model"]) for b, d in proc.groupby("benchmark")}
    proc_i = proc.set_index(["benchmark", "model"]).sort_index()
    model_set = set(models)
    bench_meta, not_repro, missing_in_raw = [], [], []
    cell_version, cell_raw, cell_costs = {}, {}, {}
    n_repro = 0
    raw_groups_not_in_matrix = Counter()
    raw_rows_unmapped_version = {}
    for b in benches:
        meta = bm_i.loc[b]
        rb, ceil, scale = float(meta["random_baseline"]), float(meta["score_ceiling"]), float(meta["scale"])
        entry = {
            "name": b,
            "epoch_benchmark_id": jv(bid.get(b)),
            "release_date": jv(meta["release_date"]),
            "release_quarter": quarter(jv(meta["release_date"])),
            "source_file": jv(meta["source_file"]),
            "score_column": jv(meta["score_column"]),
            "scale": scale,
            "random_baseline": rb,
            "score_ceiling": ceil,
            "superseded_by": jv(meta["superseded_by"]),
            "edi": jv(edi_i["edi"].get(b)) if b in edi_i.index else None,
            "edi_is_anchor": jv(edi_i["is_anchor"].get(b)) if b in edi_i.index else None,
            "edi_slope_scaled": jv(edi_i["estimated_slope_scaled"].get(b)) if b in edi_i.index else None,
            "n_models": int(df[b].notna().sum()),
            "n_items": None, "n_items_source": None,
            "cost_columns": [],
            "n_cells_reproduced_from_raw": 0, "n_cells_not_reproduced": 0,
            "n_missing_cells_with_raw_rows": 0,
        }
        observed = df[b].dropna()
        src = entry["source_file"]
        if src is None or src not in z.namelist():
            for m, v in observed.items():
                not_repro.append({"model": m, "benchmark": b, "processed": float(v),
                                  "raw_derived": None, "reason": "no raw source file",
                                  "processed_model_version": proc_i.loc[(b, m), "model_version"],
                                  "source": jv(proc_i.loc[(b, m), "source"])})
            entry["n_cells_not_reproduced"] = int(len(observed))
            bench_meta.append(entry)
            continue
        raw = read_csv(z, src)
        entry["n_items"], entry["n_items_source"] = infer_n_items(raw, entry["score_column"], scale)
        raw = raw.assign(_s=to_num(raw[entry["score_column"]]) * scale,
                         _g=raw["Model version"].map(v2g),
                         _v=raw["Model version"].astype(str),
                         _id=(raw["id"].astype(str) if "id" in raw.columns
                              else pd.Series(range(len(raw)), index=raw.index).astype(str)))
        n_unmapped = int((raw["_s"].notna() & raw["_g"].isna()).sum())
        if n_unmapped:
            raw_rows_unmapped_version[b] = n_unmapped
        raw = raw.dropna(subset=["_s", "_g"])
        raw = raw.sort_values(["_g", "_s", "_v", "_id"], ascending=[True, False, True, True],
                              kind="mergesort")
        best = raw.drop_duplicates("_g", keep="first").set_index("_g")
        for g in best.index:
            if g not in model_set:
                raw_groups_not_in_matrix[g] += 1
        # Missing cells (no processed row) for which the raw CSV has a scored row.
        sup = entry["superseded_by"]
        nm = 0
        for m in models:
            if m in observed.index or m not in best.index:
                continue
            row = best.loc[m]
            has_successor = bool(sup) and m in proc_models.get(sup, set())
            missing_in_raw.append({
                "model": m, "benchmark": b,
                "raw_derived": max(0.0, (float(row["_s"]) - rb) / (ceil - rb)),
                "raw_score": float(row["_s"]),
                "raw_model_version": row["_v"],
                "reason": "superseded" if has_successor else "not_in_processed_table",
                "superseded_by": sup if has_successor else None,
            })
            nm += 1
        entry["n_missing_cells_with_raw_rows"] = nm
        ccols = [c for c in raw.columns
                 if not c.startswith("_") and COST_COL_RE.search(c) and "score" not in c.lower()]
        entry["cost_columns"] = ccols
        cell_version[b], cell_raw[b] = {}, {}
        costs = {c: {} for c in ccols}
        nr = 0
        for m, v in observed.items():
            pver = proc_i.loc[(b, m), "model_version"]
            psrc = jv(proc_i.loc[(b, m), "source"])
            if m not in best.index:
                not_repro.append({"model": m, "benchmark": b, "processed": float(v),
                                  "raw_derived": None, "reason": "no raw row for model_group",
                                  "processed_model_version": pver, "source": psrc})
                continue
            row = best.loc[m]
            derived = max(0.0, (float(row["_s"]) - rb) / (ceil - rb))
            if abs(derived - float(v)) <= TOL:
                nr += 1
                cell_version[b][m] = row["_v"]
                cell_raw[b][m] = float(row["_s"])
                for c in ccols:
                    cv = to_num(pd.Series([row[c]])).iloc[0]
                    if pd.notna(cv):
                        costs[c][m] = float(cv)
            else:
                not_repro.append({"model": m, "benchmark": b, "processed": float(v),
                                  "raw_derived": derived, "reason": "value differs",
                                  "raw_derived_model_version": row["_v"],
                                  "processed_model_version": pver, "source": psrc})
        cell_costs[b] = {c: d for c, d in costs.items() if d}
        if not cell_costs[b]:
            del cell_costs[b]
        entry["n_cells_reproduced_from_raw"] = nr
        entry["n_cells_not_reproduced"] = int(len(observed) - nr)
        n_repro += nr
        bench_meta.append(entry)

    n_obs = int(df.notna().sum().sum())
    checks["cells_reproduced_from_raw"] = n_repro
    checks["cells_not_reproduced_from_raw"] = len(not_repro)
    checks["cells_not_reproduced_by_benchmark"] = dict(sorted(
        Counter(r["benchmark"] for r in not_repro).items()))
    print(f"  Re-derived {n_repro}/{n_obs} cells from raw CSVs; "
          f"{len(not_repro)} not reproducible: {checks['cells_not_reproduced_by_benchmark']}")
    miss_reason = Counter(r["reason"] for r in missing_in_raw)
    checks["missing_cells_with_raw_rows"] = len(missing_in_raw)
    checks["missing_cells_with_raw_rows_by_reason"] = dict(sorted(miss_reason.items()))
    checks["missing_cells_with_raw_rows_by_benchmark"] = dict(sorted(
        Counter(r["benchmark"] for r in missing_in_raw).items()))
    # Epoch keeps at most one of (superseded benchmark, its successor) per model.
    checks["superseded_pairs_models_with_both"] = {
        f"{b['name']} -> {b['superseded_by']}": len(
            proc_models.get(b["name"], set()) & proc_models.get(b["superseded_by"], set()))
        for b in bench_meta if b["superseded_by"]}
    print(f"  {len(missing_in_raw)} missing cells have raw rows: {dict(sorted(miss_reason.items()))}")

    # --- per-cell processed-table label and attribution ---
    cell_pver, cell_source = {}, {}
    for b in benches:
        cell_pver[b], cell_source[b] = {}, {}
        for m in df[b].dropna().index:
            r = proc_i.loc[(b, m)]
            cell_pver[b][m] = r["model_version"]
            s = jv(r["source"])
            if s is not None:
                cell_source[b][m] = s
        if not cell_source[b]:
            del cell_source[b]
    n_same = sum(cell_pver[b][m] == v for b, d in cell_version.items() for m, v in d.items())
    checks["processed_model_version_eq_best_raw_row_version"] = {
        "equal": n_same, "differ": n_repro - n_same}
    source_counts = Counter(s for d in cell_source.values() for s in d.values())
    n_with_source = sum(source_counts.values())

    # --- temporal summaries ---
    rq = Counter(m["release_quarter"] for m in model_meta)
    bq = Counter(b["release_quarter"] for b in bench_meta)
    rel = pd.Series({m["name"]: m["release_date"] for m in model_meta})
    split = {}
    for c in SPLIT_CUTOFFS:
        late = rel[rel >= c].index
        split[c] = {"n_models_on_or_after": int(len(late)),
                    "n_cells_on_or_after": int(df.loc[late].notna().sum().sum()),
                    "n_benchmarks_released_on_or_after": int(sum(
                        (b["release_date"] or "") >= c for b in bench_meta))}

    overlap = canonical_overlap(models, group_versions, canonical_path, merged_path)

    # --- excluded inputs ---
    used_files = {b["source_file"] for b in bench_meta if b["source_file"]}
    excluded_bench = [{"benchmark": r["benchmark"], "in_eci": bool(r["in_eci"]),
                       "source_file": jv(r["source_file"])}
                      for _, r in bm.iterrows() if r["benchmark"] not in set(benches)]
    excluded_bench.sort(key=lambda d: sort_key(d["benchmark"]))
    unused_csv = sorted(m["name"] for m in members
                        if m["name"].endswith(".csv") and "/" not in m["name"]
                        and m["name"] not in used_files | {MODEL_META, BENCH_META})

    side_meta = {n: (None if e is None else {**e, "role": FETCH_SIDE_FILES[n]})
                 for n, e in side_files.items()}
    source = {
        "url": SOURCE_URL,
        "revision": ("unversioned living dataset; snapshot pinned by sha256 "
                     "(plus HTTP ETag and zip-member generation time)"),
        "sha256": zip_sha,
        "pinned_sha256": PINNED_SHA256,
        "sha256_matches_pinned": zip_sha == PINNED_SHA256,
        "size_bytes": len(blob),
        "etag": fetch["etag"],
        "http_date": fetch["http_date"],
        "fetched_at": fetch["fetched_at"],
        "fetch_side_files": side_meta,
        "zip_generated_at": generated_at,
        "license": LICENSE,
        "citation": CITATION,
        "files_used": {n: next(m["sha256"] for m in members if m["name"] == n)
                       for n in sorted({PROCESSED, ECI_SCORES, EDI_SCORES, MODEL_META,
                                        BENCH_META} | used_files)},
    }
    meta = {
        "name": NAME,
        "title": "Epoch AI Benchmarking Hub (ECI benchmark table)",
        "source": source,
        "builder": "data/build_epoch.py",
        "builder_args": {"min_obs": min_obs,
                         "expected_sha256": expected_sha256 if check_pin else "any"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__},
        "shape": [len(models), len(benches)],
        "n_observed": n_obs,
        "fill_fraction": n_obs / df.size,
        "row_order": "models sorted by (name.casefold(), name)",
        "column_order": "benchmarks sorted by (name.casefold(), name)",
        "model_identity": ("Epoch model_group (model_metadata.csv), identical to the 'model' "
                           "column of processed_data_for_eci.csv.  A group merges all "
                           "model_versions of one release: dated snapshots, reasoning-effort / "
                           "thinking-budget variants, provider-hosted copies and, for some "
                           "open models, base and instruct checkpoints (see models[].model_versions)."),
        "value_definition": ("processed_data_for_eci.csv 'performance' = max(0, (max_r s_r - "
                             "random_baseline) / (score_ceiling - random_baseline)), with s_r = "
                             "row[score_column] * scale and the max over all raw rows of the "
                             "benchmark whose model_version belongs to the group (i.e. best "
                             "variant / agent / run).  In [0, 1]; higher is better.  Raw score "
                             "of the best row: cells.raw_score (units of score_column*scale)."),
        "missing_definition": (
            "no row in Epoch's processed ECI table (NaN in CSV, O=0, B=0).  Usually the "
            "model was not evaluated / not reported on the benchmark, but "
            f"{len(missing_in_raw)} missing cells do have scored rows in the raw CSVs: "
            f"{miss_reason.get('superseded', 0)} are 'superseded' (Epoch drops a benchmark "
            "that benchmark_metadata marks superseded_by another, here the old "
            "FrontierMath versions, for every model that has a score on the successor; "
            f"models with both: {sum(checks['superseded_pairs_models_with_both'].values())}, "
            "see consistency_checks.superseded_pairs_models_with_both; "
            "so models evaluated after the successor appeared are absent from the old "
            "column by design and the missingness is time-dependent), and "
            f"{miss_reason.get('not_in_processed_table', 0)} are "
            "'not_in_processed_table' (raw rows the processed table does not include, "
            "apparently newer than it).  All are listed in cells_missing_but_in_raw with "
            "the raw-derived value; they are NOT filled into the matrix."),
        "release_date_rule": ("release_date = eci_scores.csv 'date' = latest model_metadata "
                              "'date' among the versions behind the processed rows; "
                              "release_date_first = earliest model_metadata 'date' of any "
                              "version in the group.  release_quarter uses release_date."),
        "dropped_models": dropped,
        "consistency_checks": checks,
        "release_quarter_counts": dict(sorted(rq.items())),
        "benchmark_release_quarter_counts": dict(sorted(bq.items())),
        "prospective_split_cutoffs": split,
        "canonical_overlap": overlap,
        "excluded": {
            "benchmarks_not_in_matrix": excluded_bench,
            "raw_csv_files_not_used": unused_csv,
            "raw_model_groups_not_in_matrix": {
                "note": ("model_groups with a scored raw row on at least one matrix benchmark "
                         "but no row in the processed table (ECI keeps only models with "
                         ">= 4 benchmarks, among other filters); n_benchmarks counts matrix "
                         "benchmarks with a raw row"),
                "n": len(raw_groups_not_in_matrix),
                "groups": [{"model_group": g, "n_benchmarks": n} for g, n in
                           sorted(raw_groups_not_in_matrix.items(), key=lambda t: sort_key(t[0]))],
            },
            "raw_rows_with_model_version_not_in_model_metadata": dict(sorted(
                raw_rows_unmapped_version.items(), key=lambda t: sort_key(t[0]))),
        },
        "models": model_meta,
        "benchmarks": bench_meta,
        "cells": {
            "source_model_version": cell_version,
            "source_model_version_note": ("model_version of the raw-CSV row attaining the max "
                                          "score (ties: model_version, then row id); only "
                                          "cells reproduced from raw are filled"),
            "processed_model_version": cell_pver,
            "processed_model_version_note": (
                "'model_version' column of processed_data_for_eci.csv, as given by Epoch, for "
                "every observed cell.  It is a label of the group, not necessarily the version "
                "attaining the max: it differs from source_model_version in "
                f"{n_repro - n_same} of {n_repro} reproduced cells"),
            "source": cell_source,
            "source_note": (
                "'source' column of processed_data_for_eci.csv: the report / leaderboard "
                f"behind the value.  Given for {n_with_source} of {n_obs} cells "
                f"({len(source_counts)} distinct); cells without a source are omitted"),
            "source_counts": dict(sorted(source_counts.items(), key=lambda t: sort_key(t[0]))),
            "raw_score": cell_raw,
            "cost_token_columns": cell_costs,
            "cost_token_note": ("values copied from the raw row that attains the cell's max "
                                "score (ties: model_version, then row id); units and scope "
                                "(per task / per run / total) as named by the source column; "
                                "only cells reproduced from raw are filled"),
        },
        "cells_not_reproducible_from_raw": not_repro,
        "cells_missing_but_in_raw": missing_in_raw,
    }
    meta_path = output_dir / f"{NAME}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False, allow_nan=False) + "\n")

    manifest = {
        "name": NAME,
        "description": "Epoch AI Benchmarking Hub bulk download (benchmark_data.zip)",
        "license": LICENSE,
        "inputs": [{
            "url": SOURCE_URL,
            "local_path": f"~/benchselect-data/raw/{NAME}/{ZIP_NAME}",
            "revision": source["revision"],
            "etag": fetch["etag"],
            "http_date": fetch["http_date"],
            "zip_generated_at": generated_at,
            "size": len(blob),
            "sha256": zip_sha,
            "sha256_matches_pinned": zip_sha == PINNED_SHA256,
            "fetched_at": fetch["fetched_at"],
        }] + [{
            "url": None,
            "derived_from": SOURCE_URL,
            "local_path": f"~/benchselect-data/raw/{NAME}/{n}",
            "role": FETCH_SIDE_FILES[n],
            "revision": "written by the fetch of the zip above",
            "size": e["size"],
            "sha256": e["sha256"],
            "fetched_at": fetch["fetched_at"],
        } for n, e in side_files.items() if e is not None],
        "zip_members": members,
    }
    man_path = manifest_dir / f"{NAME}.json"
    man_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"  Wrote {meta_path} and {man_path}")
    print(f"  Release quarters: {dict(sorted(rq.items()))}")
    if overlap.get("available"):
        print(f"  Canonical overlap: {overlap['n_epoch_models_matched']} Epoch models <-> "
              f"{overlap['n_canonical_models_matched']} canonical models")
    return df, meta


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW,
                        help="Directory with benchmark_data.zip (+ fetched_at.txt, headers.txt).")
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="Directory for epoch.matrix.{csv,npz} and epoch.meta.json.")
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        help="Directory for the raw manifest (default: <output-dir>/raw_manifests).")
    parser.add_argument("--canonical-mapping", type=Path, default=HERE / "canonical_mapping.json",
                        help="canonical_mapping.json for the overlap report.")
    parser.add_argument("--merged-matrix", type=Path, default=HERE / "merged.matrix.npz",
                        help="merged.matrix.npz for the overlap report (optional).")
    parser.add_argument("--min-obs", type=int, default=2,
                        help="Drop models with fewer observed benchmarks.")
    parser.add_argument("--expected-sha256", default=PINNED_SHA256,
                        help="Required sha256 of benchmark_data.zip (default: the pinned "
                             "snapshot); 'any' disables the check.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_epoch(args.raw_dir, args.output_dir, args.manifest_dir, args.canonical_mapping,
                args.merged_matrix, args.min_obs, args.expected_sha256)
