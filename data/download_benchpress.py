"""Download the BenchPress score table at its pinned revision and build the matrix.

BenchPress (Papailiopoulos et al., 2026) collects scores of frontier models
from papers, model cards and leaderboards, each with a source URL:
https://github.com/anadim/llm-benchmark-matrix.  The repository's README
states "License: MIT" but it has no license file and names no copyright
holder, so this release does not redistribute the JSON (nor the matrix built
from it).  This script fetches it from GitHub at the commit used in the
paper, verifies its SHA-256 and builds data/benchpress.matrix.npz:

    python data/download_benchpress.py            # fetch, verify, build
    python data/download_benchpress.py --no-build # fetch and verify only

Pinned input (retrieved 2026-04-10):
    commit  e93b33f11f4a818d0e18f6914e256bfe320be54b
    file    data/llm_benchmark_data.json  -> data/benchpress.json
    sha256  255ea00914119403032f90f9568e9b6236483ff8b6858f18c22b62fd5bebe449
A file whose checksum differs is refused (and not written).  The built
benchpress.matrix.npz is listed in data/SHA256SUMS
(f72d3a44b912e0314884189c596231480ecaf86c1d09d7e3a4001335edaa2f42).
Standard library only for the download; numpy for the build.
"""

import argparse
import hashlib
import pathlib
import sys
import urllib.request

COMMIT = "e93b33f11f4a818d0e18f6914e256bfe320be54b"
URL = ("https://raw.githubusercontent.com/anadim/llm-benchmark-matrix/"
       f"{COMMIT}/data/llm_benchmark_data.json")
SHA256 = "255ea00914119403032f90f9568e9b6236483ff8b6858f18c22b62fd5bebe449"
HERE = pathlib.Path(__file__).resolve().parent


def fetch(dest, url=URL, expected=SHA256):
    dest = pathlib.Path(dest)
    if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
        print(f"{dest}: present, sha256 ok")
        return dest
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read()
    got = hashlib.sha256(body).hexdigest()
    if got != expected:
        sys.exit(f"download_benchpress: sha256 mismatch for {url}: "
                 f"got {got}, expected {expected}; nothing written")
    dest.write_bytes(body)
    print(f"{dest}: {len(body)} bytes from {url}, sha256 ok")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    fetch(args.data_dir / "benchpress.json")
    if not args.no_build:
        sys.path.insert(0, str(HERE))
        from build_benchpress import build_benchpress
        build_benchpress(args.data_dir, args.data_dir)


if __name__ == "__main__":
    main()
