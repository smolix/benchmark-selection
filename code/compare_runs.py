"""Diff the paper-facing numbers of two experiment runs.

Runs make_numbers.py's extraction on OLD and NEW (each an experiment root
as accepted by make_numbers.py --root, or a CSV previously written by
make_numbers.py --out-csv) and reports, keyed by macro name,

  changed           numeric quantities with |new - old| > tol, sorted by
                    |delta| (NaN on one side only counts as an infinite
                    change)
  coverage_changed  |delta| <= tol, but the mean/std now covers a different
                    number of finite folds or families (e.g. a fold failed:
                    'nanmean over 10/10 folds' -> '9/10 folds')
  display_changed   |delta| <= tol, but the rounded macro text differs
                    (e.g. 0.3547 -> 0.3551 turns \\Rsq...{0.35} into {0.36})
  text_changed      text quantities (family lists, benchmark names,
                    selection orders, index lists) that differ
  only_old          quantities present in OLD only
  only_new          quantities present in NEW only

The report is printed and written to --out-csv with columns
  key,status,old_value,new_value,delta,abs_delta,old_source,new_source,
  source_key,index,aggregation,detail
(detail notes coverage and macro-text changes on any numeric row).

The tolerance is absolute and applies to every numeric quantity (R²,
counts, budgets alike); --keys restricts the comparison to matching names.
Every macro group of make_numbers.py is compared (the paper's numbers.tex
and the numbers_{group}.tex files alike: the macros dicts of the
experiment JSON logs, the data-release datasets), so restrict a diff to
one experiment by its prefix, e.g. --keys '^(Stab|Est)'.

Matched splits.  --old-reps / --new-reps (or --reps for both) restrict
each root's per-split quantities to the given CV repetitions
(make_numbers.py --reps): a root run with the 5 x 10 matched design
compares with a root that predates it (10 folds = repetition 0) through
--new-reps 0.  Coverage counts then agree ('10/10 folds' vs '10/10
splits') and only genuine changes remain.

Per-split arrays (--arrays).  Instead of the macros, compare the raw
arrays of every logs/*.npz and logs_logit/*.npz file both roots have, row
by row on split_id: each file's split-row groups are the key prefixes
with a {prefix}split_id array (p10_, cv_p10_, emi_, ''); a group without
one (a log that predates the matched design) has split_id = row (rep-0
folds).  The split axis of an array is the first axis whose length is
the group's number of rows (axis 1 of cost_budget's (budgets, splits)
arrays).  A split_id that repeats within a log (eval_prospective.py's
control rows, variant-major) pairs by occurrence: the n-th row with an id
in OLD with the n-th in NEW.  Only split_ids present in both roots are
compared, so the
rep-0 rows of a 5-repetition run compare with a 1-repetition or a
historical run.  Statuses: equal (|delta| <= tol + rtol·|old| everywhere,
NaN in the same places, ints and strings identical), differs,
shape_mismatch, only_old, only_new (e.g. keys added by the matched
design), extended (NEW has more entries along one named axis and equals
OLD on OLD's entries, see below), not_compared (wall-clock timings such
as time_s and split_seconds, TIMING_KEYS, unless --keys is given); per
array the CSV has the rows compared, the rows that differ, max |delta|
and the first differing split_id.

Named axes.  When a method joins a script (facility location, 2026-10-19)
its arrays grow along a method, pair or objective axis: methods 4 -> 5
(eval_estimator_ablation.py, facloc appended after random), boot_pairs
5 -> 9 (eval_prospective.py), objectives 3 -> 4 (eval_token_costs.py's
knap_direct_mismatch).  An array whose shape differs from OLD's in
exactly one axis besides the split axis is compared by name along it:
a label array of both logs (strings, 1-D or one label row per entry,
e.g. methods, boot_pairs, objectives) whose OLD entries are an ordered
subsequence of NEW's and whose lengths are the two axis lengths picks
NEW's entries of OLD's names; the label arrays themselves compare the
same way (OLD's names in order inside NEW's), also when NEW's length
happens to equal the number of split rows (eval_cost_budget.py's methods,
8 -> 10 in a 10-split run).  extended if those entries
are equal, differs if not (detail: the label array and NEW's added
entries); shape_mismatch if no label array names the axis.  --tol defaults to 1e-12
and --rtol to 1e-10 here.  The relative term absorbs rounding between
machines (CPU kernels, BLAS builds): it is up to ~2e-12 relative on
random-baseline log-determinants (values ~ -27, |delta| up to 4e-11
between the repository logs and a 4090 rerun, 0 between two runs on one
host), so an rtol of 1e-12 would flag arrays the code does not change;
genuine changes of selections or R² are many orders of magnitude larger.
The macro comparison's --tol is absolute: below ~1e-10 it flags log-det
macros (LogDet...) for the same cross-machine rounding.
Quantities that both runs read from the same file outside their roots
(scope 'shared': Data*, and EigK* when recomputed from --data-dir) cannot
differ and are listed in a note; give each root its own cov files
(ROOT/cov, or --old-cov-dir / --new-cov-dir) to compare the eigenspectrum.
A root that yields nothing from its own logs/, logs_logit/ or stdout/ is
an error (exit status 2), so a mistyped path never compares clean.

Usage:
    python compare_runs.py OLD_ROOT NEW_ROOT [--tol 0.005]
                           [--out-csv numbers_diff.csv] [--keys REGEX]
                           [--eigen auto|cov|recompute|skip] [--data-dir DIR]
                           [--old-cov-dir DIR] [--new-cov-dir DIR]
                           [--digits 2] [--std-digits 3]
                           [--reps R ...] [--old-reps R ...] [--new-reps R ...]
                           [--max-print N] [--fail-on-diff]
    python compare_runs.py OLD_ROOT NEW_ROOT --arrays [--tol 1e-12]
                           [--rtol 1e-10] [--out-csv arrays_diff.csv]
                           [--keys REGEX]
                           [--max-print N] [--fail-on-diff]
"""

import argparse
import csv
import math
import pathlib
import re
import sys

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import make_numbers

DIFF_FIELDS = ["key", "status", "old_value", "new_value", "delta",
               "abs_delta", "old_source", "new_source", "source_key",
               "index", "aggregation", "detail"]
STATUS_ORDER = {"changed": 0, "coverage_changed": 1, "display_changed": 2,
                "text_changed": 3, "only_old": 4, "only_new": 5}
ARRAY_FIELDS = ["file", "key", "status", "group", "split_axis", "n_rows",
                "n_rows_diff", "max_abs_delta", "first_split_id", "old_shape",
                "new_shape", "detail"]
ARRAY_STATUS_ORDER = {"differs": 0, "shape_mismatch": 1, "only_old": 2,
                      "only_new": 3, "extended": 4, "equal": 5,
                      "not_compared": 6}
ARRAY_TOL = 1e-12
ARRAY_RTOL = 1e-10      # cross-machine rounding (module docstring)
ARRAY_DIRS = ("logs", "logs_logit")
# Keys of the npz logs that record wall-clock time, never reproducible
# (eval_selection_stability time_s, eval_imputers split_seconds,
# eval_token_costs runtime_s): status not_compared unless --keys is given
TIMING_KEYS = re.compile(r"(^|_)(time_s|split_seconds|runtime_s|runtime_sec)$")


def _die(msg):
    print(f"compare_runs.py: error: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def load_run(spec, eigen="auto", data_dir=None, cov_dir=None, reps=None):
    """Records of one run: extract from a root (per-split quantities
    restricted to the CV repetitions `reps`, None: all), or read a numbers
    CSV.  Exits with status 2 if the run has no results of its own."""
    path = pathlib.Path(spec).expanduser()
    if path.is_file() and path.suffix == ".csv":
        if reps is not None:
            _die(f"--reps cannot be applied to the numbers CSV {path}; "
                 f"give the experiment root")
        print(f"reading {path}", flush=True)
        records = make_numbers.read_csv(path)
        legacy = any(r["legacy"] for r in records.values())
    elif path.is_dir():
        note = "" if reps is None else f" (reps {' '.join(map(str, reps))})"
        print(f"extracting {path}{note}", flush=True)
        records = make_numbers.extract(path, cov_dir=cov_dir,
                                       data_dir=data_dir, eigen=eigen,
                                       reps=reps).records
        legacy = False
    else:
        _die(f"{path} is neither an experiment root nor a numbers CSV")
    if not records:
        _die(f"no quantities in {path}")
    # CSVs without the kind/scope columns cannot tell results from inputs
    if not legacy and not any(make_numbers.from_results(r)
                              for r in records.values()):
        _die(f"nothing from {path}'s own "
             f"{', '.join(d + '/' for d in make_numbers.RESULT_DIRS)}; "
             f"wrong root?")
    return records


def _source(r):
    return f"{r['source_file']}:{r['source_key']}"


def _raw(r):
    """The value as its CSV cell (exact for text, round-trip for floats)."""
    return r["raw"] if "raw" in r else make_numbers.csv_value(r["value"],
                                                               r["kind"])


def _cov_text(cov):
    return f"{cov[0]}/{cov[1]}"


def _row(key, status, a, b, delta=math.nan, abs_delta=math.nan, detail=""):
    ref = b if b is not None else a
    return {"key": key, "status": status,
            "old_value": "" if a is None else a["value"],
            "new_value": "" if b is None else b["value"],
            "delta": delta, "abs_delta": abs_delta,
            "old_source": "" if a is None else _source(a),
            "new_source": "" if b is None else _source(b),
            "source_key": ref["source_key"], "index": ref["index"],
            "aggregation": ref["aggregation"], "detail": detail}


def diff_runs(old, new, tol=0.005, digits=2, std_digits=None):
    """Rows for every quantity that differs (by more than tol, in finite
    fold coverage, or in its macro text at digits/std_digits) or exists in
    only one run, ordered by status, then |delta| descending."""
    rows = []
    keys = list(old) + [k for k in new if k not in old]
    for key in keys:
        a, b = old.get(key), new.get(key)
        if b is None:
            rows.append(_row(key, "only_old", a, None))
            continue
        if a is None:
            rows.append(_row(key, "only_new", None, b))
            continue
        if a["kind"] == "text" or b["kind"] == "text":
            if _raw(a) != _raw(b):
                rows.append(_row(key, "text_changed", a, b))
            continue
        x, y = float(a["value"]), float(b["value"])
        if x == y or (math.isnan(x) and math.isnan(y)):
            delta, moved = 0.0, False
        elif math.isnan(x) or math.isnan(y):
            delta, moved = math.nan, True
        else:
            delta = y - x
            moved = not abs(delta) <= tol
        abs_delta = math.inf if math.isnan(delta) else abs(delta)
        notes = []
        cov_a = make_numbers.coverage(a["aggregation"])
        cov_b = make_numbers.coverage(b["aggregation"])
        # None on one side only: a CSV from an older make_numbers.py
        coverage_moved = None not in (cov_a, cov_b) and cov_a != cov_b
        if coverage_moved:
            notes.append(f"coverage {_cov_text(cov_a)} -> {_cov_text(cov_b)}")
        shown = None
        if not (a.get("legacy") or b.get("legacy")):    # format unknown
            shown = [make_numbers.format_value(r["value"], r["kind"], digits,
                                               std_digits) for r in (a, b)]
            if shown[0] != shown[1]:
                notes.append(f"macro {shown[0]} -> {shown[1]}")
        if moved:
            status = "changed"
        elif coverage_moved:
            status = "coverage_changed"
        elif shown and shown[0] != shown[1]:
            status = "display_changed"
        else:
            continue
        rows.append(_row(key, status, a, b, delta, abs_delta,
                         "; ".join(notes)))
    # Stable sort: status group, then largest |delta| first
    rows.sort(key=lambda r: (STATUS_ORDER[r["status"]],
                             0.0 if math.isnan(r["abs_delta"])
                             else -r["abs_delta"]))
    return rows


def shared_keys(old, new):
    """Keys both runs read from the same file outside their roots."""
    return [k for k, a in old.items()
            if k in new and a.get("scope") == "shared"
            and new[k].get("scope") == "shared"
            and a["source_file"] == new[k]["source_file"]]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if math.isfinite(v) else str(v)
    return str(v)


def print_report(rows, n_old, n_new, tol, max_print=0, shared=()):
    counts = {s: sum(r["status"] == s for r in rows) for s in STATUS_ORDER}
    print(f"\nold: {n_old} quantities, new: {n_new} quantities, tol {tol:g}")
    print("  " + ", ".join(f"{s}: {n}" for s, n in counts.items()))
    if shared:
        print(f"note: {len(shared)} quantities come from the same files "
              f"outside both roots and cannot differ "
              f"(e.g. {', '.join(shared[:4])})")
    if not rows:
        print("no differences")
        return
    shown = rows if max_print <= 0 else rows[:max_print]
    width = max(len(r["key"]) for r in shown)
    print(f"\n{'key':<{width}}  {'status':<16} {'old':>12} {'new':>12} "
          f"{'delta':>10}  detail")
    for r in shown:
        old, new = _fmt(r["old_value"]), _fmt(r["new_value"])
        if r["status"] == "text_changed":
            print(f"{r['key']:<{width}}  {r['status']:<16} {old!r} -> {new!r}")
            continue
        numeric = r["status"] in ("changed", "coverage_changed",
                                  "display_changed")
        delta = _fmt(r["delta"]) if numeric else ""
        print(f"{r['key']:<{width}}  {r['status']:<16} {old:>12} {new:>12} "
              f"{delta:>10}  {r['detail']}".rstrip())
    if len(shown) < len(rows):
        print(f"... {len(rows) - len(shown)} more rows in the CSV")


def write_diff_csv(rows, path, fields=DIFF_FIELDS):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v)
                        for k, v in r.items()})


# ── Per-split arrays (--arrays) ──────────────────────────────────

def array_files(root):
    """{'logs/NAME.npz': path} of the npz logs under root."""
    root = pathlib.Path(root).expanduser().resolve()
    out = {}
    for sub in ARRAY_DIRS:
        d = root / sub
        if d.is_dir():
            for p in sorted(d.glob("*.npz")):
                if not p.name.startswith("."):
                    out[f"{sub}/{p.name}"] = p
    return out


def split_groups(d):
    """{key prefix: split_id array} of a log's split-row groups."""
    return {k[:-len("split_id")]: np.asarray(d[k]).astype(int)
            for k in d if k.endswith("split_id")
            and np.asarray(d[k]).ndim == 1}


def _group_of(key, groups):
    """Longest prefix in groups that key starts with ('' if none)."""
    best = ""
    for g in groups:
        if key.startswith(g) and len(g) > len(best):
            best = g
    return best


def _occurrences(sid):
    """[(split_id, n)] of a split_id array: n counts the earlier rows with
    the same id, so repeated ids still give unique keys."""
    seen, out = {}, []
    for s in np.asarray(sid).tolist():
        out.append((s, seen.get(s, 0)))
        seen[s] = seen.get(s, 0) + 1
    return out


def _split_axis(shape, n_rows):
    """First axis of length n_rows (None if there is none)."""
    for ax, n in enumerate(shape):
        if n == n_rows:
            return ax
    return None


def _row_diffs(a, b, axis, tol, rtol=0.0):
    """(n_rows_diff, max_abs_delta, differing-row mask) of two equally
    shaped arrays along `axis` (NaN == NaN; NaN on one side differs; floats
    differ where |a - b| > tol + rtol·|a|)."""
    if a.dtype.kind in "OUSb" or b.dtype.kind in "OUSb":
        neq = (a.astype(str) != b.astype(str)) if a.dtype.kind in "OUS" \
            or b.dtype.kind in "OUS" else (a != b)
        mx = math.nan
    elif a.dtype.kind in "iu" and b.dtype.kind in "iu":
        neq = a != b
        mx = float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))) \
            if a.size else 0.0
    else:
        x, y = a.astype(float), b.astype(float)
        nan_x, nan_y = np.isnan(x), np.isnan(y)
        with np.errstate(invalid="ignore"):
            delta = np.abs(x - y)
        delta = np.where(nan_x & nan_y, 0.0, delta)
        with np.errstate(invalid="ignore"):
            bound = tol + rtol * np.abs(np.where(nan_x, 0.0, x))
        neq = (nan_x != nan_y) | (delta > bound)
        finite = delta[~(nan_x | nan_y)]
        mx = float(finite.max()) if finite.size else 0.0
        if (nan_x != nan_y).any():
            mx = math.inf
    if axis is None:
        rows = np.array([bool(np.any(neq))])
    else:
        rows = np.moveaxis(neq, axis, 0).reshape(neq.shape[axis], -1) \
            .any(axis=1) if neq.size else np.zeros(neq.shape[axis], bool)
    return int(rows.sum()), mx, rows


def _labels(arr):
    """Entries of a label array as hashable values (1-D: the strings; 2-D:
    one tuple per row, e.g. eval_prospective.py's (a, b) boot_pairs), or
    None if it holds anything but strings."""
    arr = np.asarray(arr)
    if arr.ndim not in (1, 2) or arr.dtype.kind not in "USO":
        return None
    items = arr.tolist()
    flat = items if arr.ndim == 1 else [x for row in items for x in row]
    if not all(isinstance(x, (str, bytes)) for x in flat):
        return None
    if arr.ndim == 1:
        return [str(x) for x in items]
    return [tuple(str(x) for x in row) for row in items]


def _positions(old, new):
    """Indices of OLD's entries in NEW, in order, if OLD (without repeats)
    is an ordered subsequence of NEW; else None."""
    if len(set(old)) != len(old):
        return None
    pos, j = [], 0
    for x in old:
        while j < len(new) and new[j] != x:
            j += 1
        if j == len(new):
            return None
        pos.append(j)
        j += 1
    return pos


def _named_axes(a, b, old, new, key=None):
    """[(axis, label key, positions, added labels)] for every label array
    of both logs that names the one axis along which b extends a (module
    docstring, "Named axes"), the array `key` itself first; [] if the
    shapes differ otherwise or no label array fits."""
    if a.ndim != b.ndim:
        return []
    axes = [ax for ax in range(a.ndim) if a.shape[ax] != b.shape[ax]]
    if len(axes) != 1 or b.shape[axes[0]] < a.shape[axes[0]]:
        return []
    ax = axes[0]
    out = []
    for lk in sorted(old, key=lambda k: k != key):       # stable
        if lk not in new:
            continue
        lo, ln = _labels(old[lk]), _labels(new[lk])
        if lo is None or ln is None or len(lo) != a.shape[ax] \
                or len(ln) != b.shape[ax]:
            continue
        pos = _positions(lo, ln)
        if pos is not None:
            kept = set(pos)
            out.append((ax, lk, pos, [x for j, x in enumerate(ln)
                                      if j not in kept]))
    return out


def _compare_named(a, b, old, new, tol, rtol, split_axis=None, key=None):
    """(status, n_rows_diff, max_abs_delta, differing-row mask, detail) of
    b against a (the arrays `key` of old and new) by name along the axis b
    extends (status extended or differs; the mask along split_axis, one
    entry without one), or None if no label array names it.  The first
    label array under which the entries agree decides; else the first one
    found."""
    first = None
    for ax, lk, pos, added in _named_axes(a, b, old, new, key):
        n_diff, mx, bad = _row_diffs(a, np.take(b, pos, axis=ax), split_axis,
                                     tol, rtol)
        names = ", ".join("/".join(x) if isinstance(x, tuple) else x
                          for x in added)
        res = ("differs" if n_diff else "extended", n_diff, mx, bad,
               f"axis {ax} named by {lk}; new adds {names}")
        if not n_diff:
            return res
        first = first or res
    return first


def diff_arrays(old_path, new_path, rel, tol=ARRAY_TOL, keys=None,
                rtol=ARRAY_RTOL):
    """Rows (ARRAY_FIELDS) comparing every array of two npz logs, the
    split rows paired on split_id (module docstring)."""
    with np.load(old_path, allow_pickle=True) as f:
        old = {k: f[k] for k in f.files}
    with np.load(new_path, allow_pickle=True) as f:
        new = {k: f[k] for k in f.files}
    g_old, g_new = split_groups(old), split_groups(new)
    rows = []

    def row(key, status, **kw):
        r = dict.fromkeys(ARRAY_FIELDS, "")
        r.update(file=rel, key=key, status=status, **kw)
        rows.append(r)

    names = list(old) + [k for k in new if k not in old]
    for key in names:
        if keys is not None and not keys.search(key):
            continue
        if key not in new:
            row(key, "only_old", old_shape=str(old[key].shape))
            continue
        if key not in old:
            row(key, "only_new", new_shape=str(new[key].shape))
            continue
        if keys is None and TIMING_KEYS.search(key):
            row(key, "not_compared", old_shape=str(old[key].shape),
                new_shape=str(new[key].shape), detail="wall-clock time")
            continue
        a, b = np.asarray(old[key]), np.asarray(new[key])
        group = _group_of(key, g_new) if g_new else ""
        sid_new = g_new.get(group)
        if g_old:
            sid_old = g_old.get(_group_of(key, g_old))
        else:
            sid_old = None
        info = dict(group=group, old_shape=str(a.shape),
                    new_shape=str(b.shape))
        ax_new = None if sid_new is None else _split_axis(b.shape,
                                                          len(sid_new))
        if ax_new is not None and a.ndim == b.ndim and sid_old is not None \
                and a.shape[ax_new] != len(sid_old) and a.shape != b.shape:
            # NEW's axis only happens to have the split count (a methods
            # list grown from 8 to the 10 splits of a run): by name
            named = _compare_named(a, b, old, new, tol, rtol, key=key)
            if named is not None:
                status, n_diff, mx, _, how = named
                row(key, status, n_rows=1, n_rows_diff=n_diff,
                    max_abs_delta=mx, detail=how, **info)
                continue
        if ax_new is not None and a.ndim == b.ndim:
            if sid_old is None:          # log before the matched design
                sid_old = np.arange(a.shape[ax_new])
            if a.shape[ax_new] != len(sid_old):
                row(key, "shape_mismatch", split_axis=ax_new,
                    detail="split axis length differs from the old split ids",
                    **info)
                continue
            # A split_id may repeat (eval_prospective.py's control rows
            # repeat 0..C-1 per variant): pair the n-th occurrence of an
            # id in OLD with its n-th occurrence in NEW
            pos_new = {key: i for i, key in enumerate(_occurrences(sid_new))}
            common = [(i, pos_new[key])
                      for i, key in enumerate(_occurrences(sid_old))
                      if key in pos_new]
            if not common:
                row(key, "shape_mismatch", split_axis=ax_new,
                    detail="no common split_id", **info)
                continue
            io = np.array([i for i, _ in common])
            jn = np.array([j for _, j in common])
            a_c = np.take(a, io, axis=ax_new)
            b_c = np.take(b, jn, axis=ax_new)
            missing = len(sid_old) - len(common)
            notes = [f"{missing} old split rows not in new"] if missing \
                else []
            if a_c.shape != b_c.shape:
                named = _compare_named(a_c, b_c, old, new, tol, rtol, ax_new,
                                       key)
                if named is None:
                    row(key, "shape_mismatch", split_axis=ax_new, **info)
                    continue
                status, n_diff, mx, bad, how = named
                notes.insert(0, how)
            else:
                n_diff, mx, bad = _row_diffs(a_c, b_c, ax_new, tol, rtol)
                status = "differs" if n_diff else "equal"
            first = (int(sid_old[io[np.flatnonzero(bad)[0]]]) if n_diff
                     else "")
            row(key, status, split_axis=ax_new,
                n_rows=len(common), n_rows_diff=n_diff, max_abs_delta=mx,
                first_split_id=first, detail="; ".join(notes), **info)
            continue
        if a.shape != b.shape:
            named = _compare_named(a, b, old, new, tol, rtol, key=key)
            if named is None:
                row(key, "shape_mismatch", **info)
                continue
            status, n_diff, mx, _, how = named
            row(key, status, n_rows=1, n_rows_diff=n_diff, max_abs_delta=mx,
                detail=how, **info)
            continue
        n_diff, mx, _ = _row_diffs(a, b, None, tol, rtol)
        row(key, "differs" if n_diff else "equal", n_rows=1,
            n_rows_diff=n_diff, max_abs_delta=mx, **info)
    return rows


def diff_array_runs(old_root, new_root, tol=ARRAY_TOL, keys=None,
                    rtol=ARRAY_RTOL):
    """(rows, files_old_only, files_new_only) over the npz logs of both
    roots."""
    fo, fn = array_files(old_root), array_files(new_root)
    if not fo:
        _die(f"no npz logs under {old_root}/{{{','.join(ARRAY_DIRS)}}}")
    if not fn:
        _die(f"no npz logs under {new_root}/{{{','.join(ARRAY_DIRS)}}}")
    rows = []
    for rel in fo:
        if rel in fn:
            try:
                rows.extend(diff_arrays(fo[rel], fn[rel], rel, tol, keys,
                                        rtol))
            except Exception as e:
                r = dict.fromkeys(ARRAY_FIELDS, "")
                r.update(file=rel, key="", status="shape_mismatch",
                         detail=f"cannot compare ({type(e).__name__}: {e})")
                rows.append(r)
    rows.sort(key=lambda r: (ARRAY_STATUS_ORDER[r["status"]], r["file"]))
    return (rows, [f for f in fo if f not in fn], [f for f in fn
                                                   if f not in fo])


def print_array_report(rows, old_only, new_only, tol, max_print=0,
                       rtol=ARRAY_RTOL):
    files = sorted({r["file"] for r in rows})
    print(f"\n{len(files)} common npz logs, tol {tol:g}, rtol {rtol:g}")
    if old_only:
        print(f"  only in OLD: {', '.join(old_only)}")
    if new_only:
        print(f"  only in NEW: {', '.join(new_only)}")
    for rel in files:
        mine = [r for r in rows if r["file"] == rel]
        counts = {s: sum(r["status"] == s for r in mine)
                  for s in ARRAY_STATUS_ORDER}
        n_rows = max([r["n_rows"] for r in mine
                      if isinstance(r["n_rows"], int)] or [0])
        print(f"  {rel}: " + ", ".join(f"{s} {n}" for s, n in counts.items()
                                       if n)
              + f"  (up to {n_rows} common split rows)")
    grown = [r for r in rows if r["status"] == "extended"]
    if grown:
        print(f"\n{len(grown)} arrays extended along a named axis (OLD's "
              f"entries equal in NEW):")
        for r in grown if max_print <= 0 else grown[:max_print]:
            print(f"  {r['file']}:{r['key']}  {r['old_shape']} -> "
                  f"{r['new_shape']}  {r['detail']}")
    shown = [r for r in rows if r["status"] in ("differs", "shape_mismatch")]
    if not shown:
        print("no differences in the common arrays")
        return
    if max_print > 0:
        shown = shown[:max_print]
    width = max(len(f"{r['file']}:{r['key']}") for r in shown)
    print(f"\n{'file:key':<{width}}  {'status':<14} {'rows':>9} "
          f"{'max|delta|':>11} first_split  detail")
    for r in shown:
        mx = r["max_abs_delta"]
        mx = f"{mx:.3g}" if isinstance(mx, float) else str(mx)
        nr = (f"{r['n_rows_diff']}/{r['n_rows']}" if r["n_rows"] != ""
              else "")
        print(f"{r['file'] + ':' + r['key']:<{width}}  {r['status']:<14} "
              f"{nr:>9} {mx:>11} {str(r['first_split_id']):>11}  "
              f"{r['detail'] or ''} {r['old_shape']} -> {r['new_shape']}"
              .rstrip())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", help="OLD experiment root or numbers CSV")
    parser.add_argument("new", help="NEW experiment root or numbers CSV")
    parser.add_argument("--tol", type=float, default=None,
                        help="absolute tolerance (default 0.005; with "
                        f"--arrays {ARRAY_TOL:g})")
    parser.add_argument("--out-csv", default=None,
                        help="diff CSV to write (default: ./numbers_diff.csv;"
                        " with --arrays ./arrays_diff.csv)")
    parser.add_argument("--keys", help="compare only macro names (with "
                        "--arrays: npz keys) matching this regular "
                        "expression (re.search)")
    parser.add_argument("--arrays", action="store_true",
                        help="compare the per-split arrays of the npz logs, "
                        "paired on split_id, instead of the macros")
    parser.add_argument("--rtol", type=float, default=ARRAY_RTOL,
                        help="with --arrays: relative tolerance (default "
                        f"{ARRAY_RTOL:g})")
    parser.add_argument("--reps", nargs="+", type=int,
                        help="CV repetitions of both runs (make_numbers.py "
                        "--reps)")
    parser.add_argument("--old-reps", nargs="+", type=int,
                        help="CV repetitions of OLD (overrides --reps)")
    parser.add_argument("--new-reps", nargs="+", type=int,
                        help="CV repetitions of NEW (overrides --reps; e.g. "
                        "0 against a pre-matched-design root)")
    parser.add_argument("--eigen", default="auto",
                        choices=["auto", "cov", "recompute", "skip"],
                        help="eigenspectrum source, as in make_numbers.py")
    parser.add_argument("--data-dir", help="directory with {ds}.matrix.npz")
    parser.add_argument("--old-cov-dir", help="OLD run's {ds}.cov.npz "
                        "directory (default: OLD/cov, OLD/data, OLD)")
    parser.add_argument("--new-cov-dir", help="NEW run's {ds}.cov.npz "
                        "directory (default: NEW/cov, NEW/data, NEW)")
    parser.add_argument("--digits", type=int, default=2,
                        help="macro decimals for display_changed, as in "
                        "make_numbers.py")
    parser.add_argument("--std-digits", type=int, default=None,
                        help="macro decimals of stds (default: digits + 1)")
    parser.add_argument("--max-print", type=int, default=0,
                        help="print at most N rows (0 = all)")
    parser.add_argument("--fail-on-diff", action="store_true",
                        help="exit with status 1 if anything differs")
    args = parser.parse_args(argv)

    if args.arrays:
        if any(r is not None for r in (args.reps, args.old_reps,
                                       args.new_reps)):
            parser.error("--arrays pairs rows on split_id; --reps does not "
                         "apply")
        for spec in (args.old, args.new):
            if not pathlib.Path(spec).expanduser().is_dir():
                parser.error(f"--arrays needs experiment roots, not {spec}")
        tol = ARRAY_TOL if args.tol is None else args.tol
        out_csv = args.out_csv or "arrays_diff.csv"
        pattern = re.compile(args.keys) if args.keys else None
        rows, old_only, new_only = diff_array_runs(args.old, args.new, tol,
                                                   pattern, args.rtol)
        print_array_report(rows, old_only, new_only, tol, args.max_print,
                           args.rtol)
        write_diff_csv(rows, out_csv, ARRAY_FIELDS)
        print(f"\nwrote {out_csv} ({len(rows)} rows)")
        if args.fail_on_diff and any(r["status"] in ("differs",
                                                     "shape_mismatch")
                                     for r in rows):
            sys.exit(1)
        return

    tol = 0.005 if args.tol is None else args.tol
    out_csv = args.out_csv or "numbers_diff.csv"
    old_reps = args.old_reps if args.old_reps is not None else args.reps
    new_reps = args.new_reps if args.new_reps is not None else args.reps
    old = load_run(args.old, args.eigen, args.data_dir, args.old_cov_dir,
                   old_reps)
    new = load_run(args.new, args.eigen, args.data_dir, args.new_cov_dir,
                   new_reps)
    if args.keys:
        pattern = re.compile(args.keys)
        old = {k: r for k, r in old.items() if pattern.search(k)}
        new = {k: r for k, r in new.items() if pattern.search(k)}
    rows = diff_runs(old, new, tol, args.digits, args.std_digits)
    print_report(rows, len(old), len(new), tol, args.max_print,
                 shared_keys(old, new))
    write_diff_csv(rows, out_csv)
    print(f"\nwrote {out_csv} ({len(rows)} rows)")
    if args.fail_on_diff and rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
