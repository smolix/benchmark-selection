"""Plot the benchmark overlap network of the canonical model mapping.

Regenerates figures/benchmark_overlap.pdf (paper Appendix E, Figure
fig:overlap), which no script in the repository generated before.

Usage:
    python plot_benchmark_overlap.py [--mapping PATH] [--collections-dir DIR]
        [--label-min 20] [--out PATH] [--summary PATH]

Input: data/canonical_mapping.json, {"canonical_models": [{"canonical_name":
..., "datasets": {collection: raw model name}}]}, the mapping from which
build_matrices.py builds the Merged matrix.  The figure has one node per
collection with at least one canonical model, showing how many canonical
models appear in it, and one edge per pair of collections that share
canonical models, max(n / 8, 0.5) pt wide for n shared models and labeled
with n when n >= --label-min.  Collections without a canonical model (MTEB)
are omitted.  Layout, colors and sizes are those of the submitted figure
(node positions read off paper/figures/benchmark_overlap.pdf of 2026-03-16);
a collection outside that layout is placed on a ring around it.  With
matplotlib 3.10.8, which made the submitted PDF, the output is identical to
it except for /CreationDate; with the pinned 3.11.1 text positions differ by
less than 1 pt.

Counts are of canonical models, as the caption says.  A raw leaderboard row
mapped to two canonical models (a base and a chat model sharing one entry)
counts for both, so a node can exceed the number of matched rows in Table
tab:unmatched; such rows are printed as "shared raw rows".

Optional cross-check against the raw collection tables <collection>.csv in
--collections-dir (default data/; download_*.py outputs, first column = model
name): every mapped raw name must be a row of its table, and per collection
the rows, the rows matched by the mapping and the unmatched rows are printed
(Table tab:unmatched).  The figure needs only the mapping; collections
without a table are skipped.

--summary PATH also writes all counts as JSON.
"""

import argparse
import csv
import itertools
import json
import pathlib
import sys
from collections import Counter, OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir

LABEL_MIN = 20            # label edges shared by at least this many models
EDGE_PT_PER_MODEL = 1 / 8  # edge width in pt per shared model ...
EDGE_MIN_WIDTH = 0.5      # ... but at least this
EDGE_COLOR = "#888888"
EDGE_ALPHA = 0.6

# Column order of Table tab:canon (= build_matrices.py's collection order).
COLLECTIONS = ["mmlu", "open_llm", "helm", "alpaca_eval", "arena_hard",
               "livebench", "wildbench", "mt_bench", "bigcode", "mteb"]

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "open_llm": "Open LLM v2",
    "helm": "HELM Lite",
    "alpaca_eval": "AlpacaEval 2",
    "arena_hard": "Arena-Hard",
    "livebench": "LiveBench",
    "wildbench": "WildBench",
    "mt_bench": "MT-Bench",
    "bigcode": "BigCodeBench",
    "mteb": "MTEB",
}

CATEGORY = {
    "mmlu": "knowledge",
    "open_llm": "knowledge",
    "helm": "knowledge",
    "alpaca_eval": "chat",
    "arena_hard": "chat",
    "livebench": "chat",
    "wildbench": "chat",
    "mt_bench": "chat",
    "bigcode": "code",
    "mteb": "embedding",
}

# category -> (legend label, node color); legend in this order.
CATEGORIES = OrderedDict([
    ("knowledge", ("Knowledge & reasoning", "#4477AA")),
    ("chat", ("Instruction & chat", "#EE7733")),
    ("code", ("Code generation", "#228833")),
    ("embedding", ("Embedding", "#AA3377")),
    ("other", ("Other", "#BBBBBB")),
])

# Node centers in data units (equal aspect) and axes limits of the submitted
# figure: in its PDF, 1 unit = 208.48 pt (9.7 in axes width / 3.35 units).
POS = {
    "mmlu": (-0.3, 1.1),
    "open_llm": (0.8, 0.8),
    "helm": (-1.1, -0.5),
    "alpaca_eval": (-0.6, 0.0),
    "arena_hard": (0.0, 0.5),
    "livebench": (0.5, 0.0),
    "wildbench": (0.0, -0.5),
    "mt_bench": (-1.3, 0.5),
    "bigcode": (1.2, -0.4),
}
XLIM = (-1.65, 1.7)
YLIM = (-0.9, 1.5)
NODE_RADIUS = 0.0875
COUNT_OFFSET = 0.005      # count label (va="center") above the node center
NAME_OFFSET = 0.16        # top of the name label below the node center
FIGSIZE = (10, 7.5)       # tight_layout: axes 9.7 in wide, as submitted
# Collections without a POS entry: evenly on this ring (center, radius).
RING = ((0.0, 0.3), 1.9)
RING_MARGIN = 0.4


def display_name(coll):
    """Display name of a collection (the raw name if none is registered)."""
    return DISPLAY_NAME.get(coll, coll)


def category(coll):
    return CATEGORY.get(coll, "other")


def load_mapping(path):
    """Canonical model groups of the mapping file (canonical names unique)."""
    with open(path, encoding="utf-8") as f:
        groups = json.load(f)["canonical_models"]
    dup = [n for n, c in Counter(g["canonical_name"] for g in groups).items()
           if c > 1]
    if dup:
        raise ValueError(f"{path}: duplicate canonical names {dup}")
    return groups


def collection_order(groups):
    """Collections with at least one canonical model: COLLECTIONS order,
    then any others sorted by name."""
    present = set()
    for g in groups:
        present.update(g["datasets"])
    return ([c for c in COLLECTIONS if c in present]
            + sorted(present - set(COLLECTIONS)))


def overlap_counts(groups, order):
    """(nodes, edges): canonical models per collection, and per collection
    pair (a, b) the canonical models in both.  Edges are listed as a graph
    lists its adjacency (networkx Graph.edges()): a in `order`, then a's
    neighbours b not listed before, in order of their first co-occurrence
    with a in the mapping.  This is the drawing order of the submitted
    figure; it only affects the PDF's byte order, not its appearance."""
    nodes = Counter()
    adj = OrderedDict((c, Counter()) for c in order)
    for g in groups:
        colls = list(g["datasets"])
        nodes.update(colls)
        for a, b in itertools.combinations(colls, 2):
            adj[a][b] += 1
            adj[b][a] += 1
    edges, listed = OrderedDict(), set()
    for a in order:
        for b, n in adj[a].items():
            if b not in listed:
                edges[(a, b)] = n
        listed.add(a)
    return nodes, edges


def shared_raw_rows(groups):
    """{(collection, raw name): canonical names} of the raw rows that the
    mapping assigns to more than one canonical model."""
    owners = OrderedDict()
    for g in groups:
        for coll, raw in g["datasets"].items():
            owners.setdefault((coll, raw), []).append(g["canonical_name"])
    return OrderedDict((key, names) for key, names in owners.items()
                       if len(names) > 1)


def read_table_models(coll, root):
    """Model names (first column) of root/<coll>.csv, or None if absent."""
    path = root / f"{coll}.csv"
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)                      # header
        return [row[0] for row in reader if row]


def check_collections(groups, order, root):
    """{collection: counts} of the raw tables in `root` (tab:unmatched)."""
    out = OrderedDict()
    for coll in order + [c for c in COLLECTIONS if c not in order]:
        models = read_table_models(coll, root)
        if models is None:
            continue
        rows = set(models)
        mapped = {g["datasets"][coll] for g in groups if coll in g["datasets"]}
        matched = len(mapped & rows)
        out[coll] = {
            "rows": len(models),
            "duplicate_rows": len(models) - len(rows),
            "matched_rows": matched,
            "unmatched_rows": len(rows) - matched,
            "canonical_models": sum(coll in g["datasets"] for g in groups),
            "missing_raw_names": sorted(mapped - rows),
        }
    return out


def layout(order):
    """(positions, xlim, ylim): POS, plus a ring for other collections."""
    pos = {c: POS[c] for c in order if c in POS}
    extra = [c for c in order if c not in POS]
    xlim, ylim = XLIM, YLIM
    if extra:
        (cx, cy), r = RING
        for i, c in enumerate(extra):
            a = -np.pi / 2 + 2 * np.pi * (i + 0.5) / len(extra)
            pos[c] = (cx + r * np.cos(a), cy + r * np.sin(a))
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        xlim = (min(XLIM[0], min(xs) - RING_MARGIN),
                max(XLIM[1], max(xs) + RING_MARGIN))
        ylim = (min(YLIM[0], min(ys) - RING_MARGIN),
                max(YLIM[1], max(ys) + RING_MARGIN))
    return pos, xlim, ylim


def edge_width(n):
    return max(n * EDGE_PT_PER_MODEL, EDGE_MIN_WIDTH)


def plot(nodes, edges, order, label_min, out_path):
    pos, xlim, ylim = layout(order)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    for (a, b), n in edges.items():
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ax.plot([xa, xb], [ya, yb], color=EDGE_COLOR, alpha=EDGE_ALPHA,
                linewidth=edge_width(n), zorder=1)
    for (a, b), n in edges.items():
        if n >= label_min:
            (xa, ya), (xb, yb) = pos[a], pos[b]
            ax.text((xa + xb) / 2, (ya + yb) / 2, str(n), fontsize=7,
                    ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.8))

    for c in order:
        ax.add_patch(Circle(pos[c], NODE_RADIUS,
                            facecolor=CATEGORIES[category(c)][1],
                            edgecolor="white", linewidth=1.5, zorder=6))
    for c in order:
        x, y = pos[c]
        ax.text(x, y + COUNT_OFFSET, str(nodes[c]), color="white", fontsize=9,
                fontweight="bold", ha="center", va="center", zorder=7)
        ax.text(x, y - NAME_OFFSET, display_name(c), fontsize=9,
                ha="center", va="top", zorder=7)

    used = [k for k in CATEGORIES if any(category(c) == k for c in order)]
    ax.legend(handles=[Patch(color=CATEGORIES[k][1], label=CATEGORIES[k][0])
                       for k in used],
              loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mapping", type=pathlib.Path, default=None,
                        help="canonical mapping JSON "
                             "(default: data/canonical_mapping.json)")
    parser.add_argument("--collections-dir", type=pathlib.Path, default=None,
                        help="directory of the raw <collection>.csv tables "
                             "for the cross-check (default: data/)")
    parser.add_argument("--label-min", type=int, default=LABEL_MIN,
                        help=f"label edges with >= this many shared models "
                             f"(default {LABEL_MIN})")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="output PDF "
                             "(default: figures/benchmark_overlap.pdf)")
    parser.add_argument("--summary", type=pathlib.Path, default=None,
                        help="also write the counts to this JSON file")
    args = parser.parse_args()

    mapping = args.mapping or data_dir() / "canonical_mapping.json"
    tables = args.collections_dir or data_dir()
    out_path = args.out or figures_dir() / "benchmark_overlap.pdf"

    groups = load_mapping(mapping)
    order = collection_order(groups)
    nodes, edges = overlap_counts(groups, order)
    omitted = [c for c in COLLECTIONS if c not in order]
    multiplicity = Counter(len(g["datasets"]) for g in groups)
    shared = shared_raw_rows(groups)
    tables_check = check_collections(groups, order, tables)
    edge_list = sorted(edges.items(),
                       key=lambda e: (-e[1], order.index(e[0][0]),
                                      order.index(e[0][1])))
    n_labeled = sum(n >= args.label_min for n in edges.values())

    print(f"{mapping}: {len(groups)} canonical models; by number of "
          f"collections: " + ", ".join(f"{m}: {multiplicity[m]}"
                                       for m in sorted(multiplicity)))
    print("\nNodes (canonical models per collection):")
    for c in order:
        print(f"  {display_name(c):14s} {nodes[c]:4d}   [{category(c)}]")
    if omitted:
        print("Omitted (no canonical model): "
              + ", ".join(display_name(c) for c in omitted))
    print(f"\nEdges: {len(edges)} of {len(order) * (len(order) - 1) // 2} "
          f"collection pairs share models; {n_labeled} labeled "
          f"(>= {args.label_min})")
    for (a, b), n in edge_list:
        print(f"  {display_name(a):>14s} - {display_name(b):14s} {n:4d}  "
              f"width {edge_width(n):.3f} pt"
              + ("  labeled" if n >= args.label_min else ""))
    if shared:
        print(f"\nShared raw rows (one leaderboard row, several canonical "
              f"models; counted for each): {len(shared)}")
        for (coll, raw), names in shared.items():
            print(f"  {coll}/{raw} <- {', '.join(names)}")
    if tables_check:
        print(f"\nCollection tables in {tables} (Table tab:unmatched):")
        print(f"  {'collection':14s} {'rows':>6s} {'matched':>8s} "
              f"{'unmatched':>9s} {'canonical':>9s}")
        for coll, t in tables_check.items():
            print(f"  {display_name(coll):14s} {t['rows']:6d} "
                  f"{t['matched_rows']:8d} {t['unmatched_rows']:9d} "
                  f"{t['canonical_models']:9d}")
            if t["duplicate_rows"]:
                print(f"  WARNING {coll}: {t['duplicate_rows']} duplicate "
                      f"model names in {coll}.csv")
            if t["missing_raw_names"]:
                print(f"  WARNING {coll}: {len(t['missing_raw_names'])} "
                      f"mapped names not in {coll}.csv: "
                      f"{t['missing_raw_names']}")
        missing = [c for c in order if c not in tables_check]
        if missing:
            print(f"  (no table for: {', '.join(missing)})")
    else:
        print(f"\nNo collection tables in {tables}; cross-check skipped")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(nodes, edges, order, args.label_min, out_path)
    print(f"\nSaved to {out_path}")

    if args.summary is not None:
        summary = OrderedDict([
            ("mapping", str(mapping)),
            ("n_canonical", len(groups)),
            ("by_n_collections", {str(m): multiplicity[m]
                                  for m in sorted(multiplicity)}),
            ("nodes", OrderedDict((c, nodes[c]) for c in order)),
            ("omitted", omitted),
            ("label_min", args.label_min),
            ("n_edges", len(edges)),
            ("n_labeled", n_labeled),
            ("edges", [[a, b, n] for (a, b), n in edge_list]),
            ("shared_raw_rows", [[coll, raw, names]
                                 for (coll, raw), names in shared.items()]),
            ("collection_tables", tables_check),
        ])
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")
        print(f"Summary written to {args.summary}")


if __name__ == "__main__":
    main()
