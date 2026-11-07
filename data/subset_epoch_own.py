"""Extract the Epoch-own columns of the Epoch matrix for redistribution.

Usage (after build_epoch.py has written epoch.matrix.npz and epoch.meta.json):

    python data/subset_epoch_own.py [--data-dir DIR] [--out-dir DIR]

Epoch AI's data are CC BY 4.0, but 46 of the 61 columns of epoch.matrix.npz
come from Epoch's ``*_external.csv`` files, i.e. from external projects that
keep their own licenses.  The public release therefore ships only the 15
columns whose ``benchmarks[*].source_file`` (epoch.meta.json) is one of
Epoch's own benchmark files, and the full matrix is rebuilt locally with
data/build_epoch.py from Epoch's download.

Writes (in --out-dir, default --data-dir):
  epoch_own.npz        B (M x 15, 0 where missing), O (mask), model_names,
                       benchmark_names; rows: the models of epoch.matrix.npz
                       with at least one observed cell in these columns,
                       in the order of epoch.matrix.npz
  epoch_own.csv        the same matrix, empty where missing
  epoch_own.meta.json  the subset of epoch.meta.json for these rows and
                       columns (source, value definition, models, benchmarks,
                       per-cell model version / source / raw score)

These files are not an input of any experiment (the experiments read
epoch.matrix.npz); they are named so that no ``*.matrix.npz`` glob sees them.
"""

import argparse
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def own_columns(meta):
    return [b["name"] for b in meta["benchmarks"]
            if b.get("source_file") and "external" not in b["source_file"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or args.data_dir
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data_dir / "epoch.matrix.npz", allow_pickle=False)
    meta = json.loads((args.data_dir / "epoch.meta.json").read_text())
    names = [str(x) for x in d["benchmark_names"]]
    models = [str(x) for x in d["model_names"]]
    keep_cols = own_columns(meta)
    cidx = [names.index(c) for c in keep_cols]
    B = d["B"][:, cidx]
    O = d["O"][:, cidx]
    ridx = np.flatnonzero(O.sum(axis=1) > 0)
    B, O = B[ridx], O[ridx]
    keep_models = [models[i] for i in ridx]

    np.savez(out / "epoch_own.npz", B=B, O=O,
             model_names=np.array(keep_models),
             benchmark_names=np.array(keep_cols))
    with open(out / "epoch_own.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + keep_cols)
        for i, m in enumerate(keep_models):
            w.writerow([m] + [repr(float(B[i, j])) if O[i, j] else ""
                              for j in range(len(keep_cols))])

    km, kc = set(keep_models), set(keep_cols)
    cells = {}
    for key, val in meta["cells"].items():
        if isinstance(val, dict) and key != "source_counts":
            def rows(x):
                return {m: v for m, v in x.items() if m in km}
            cells[key] = {
                c: ({sub_k: rows(sub_v) for sub_k, sub_v in col.items()}
                    if key == "cost_token_columns" else rows(col))
                for c, col in val.items() if c in kc}
        elif key != "source_counts":
            cells[key] = val
    sub = {
        "name": "epoch_own",
        "title": meta["title"] + ": the 15 columns from Epoch's own files",
        "license": "CC BY 4.0 (Epoch AI); see DATA_LICENSES.md",
        "derived_from": "epoch.matrix.npz / epoch.meta.json (data/build_epoch.py)",
        "builder": "data/subset_epoch_own.py",
        "selection": "columns whose benchmarks[*].source_file is not an "
                     "*_external.csv file; rows with >= 1 observed cell in them",
        "source": meta["source"],
        "value_definition": meta["value_definition"],
        "missing_definition": "no row in Epoch's processed ECI table (O = 0, B = 0, "
                              "empty in the CSV)",
        "shape": [len(keep_models), len(keep_cols)],
        "n_observed": int(O.sum()),
        "models": [m for m in meta["models"] if m["name"] in km],
        "benchmarks": [b for b in meta["benchmarks"] if b["name"] in kc],
        "cells": cells,
    }
    (out / "epoch_own.meta.json").write_text(json.dumps(sub, indent=1) + "\n")
    print(f"epoch_own: {len(keep_models)} models x {len(keep_cols)} benchmarks, "
          f"{int(O.sum())} observed cells -> {out}")


if __name__ == "__main__":
    main()
