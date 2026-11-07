"""Check the toy suite of Section 3 with the paper's own selectors.

Usage (from the repo root; needs numpy and scipy):
    python paper_tools/check_toy_suite.py

The suite: three near-duplicate chemistry subjects C1-C3 (correlation rho),
two benchmarks x1, x2 with correlation c to each chemistry subject (through
a general factor on which chemistry loads 0.5, so corr(x1, x2) = 4 c^2), a
noisy average Y of x1 and x2 (noise variance s2 relative to the average),
and an outlier O that correlates with no other benchmark.  The selectors
are those of code/greedy_select.py and code/baseline_select.py, tie rule
included.  The paper states the picks for rho = 0.9, c = 0.2, s2 = 0.1:
    FL, EV: C1, Y, O    entropy: C1, O, ...    MI: Y, C1, C2
and the grid below checks that they hold for c in [0.15, 0.3] and
s2 in [0.01, 0.1] at rho = 0.9.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from greedy_select import greedy_entropy, greedy_mi  # noqa: E402
from baseline_select import facility_location_select, greedy_trace_select  # noqa: E402


def suite(rho=0.9, c=0.2, s2=0.1):
    names = ["C1", "C2", "C3", "x1", "x2", "Y", "O"]
    L = 2 + 3 + 2 + 2          # general, chemistry, 3 unique, 2 unique, noise, outlier
    gc = 0.5
    rows = []
    for i in range(3):
        v = np.zeros(L); v[0] = gc; v[1] = np.sqrt(rho - gc**2); v[2 + i] = np.sqrt(1 - rho)
        rows.append(v)
    gx = c / gc
    xs = []
    for i in range(2):
        v = np.zeros(L); v[0] = gx; v[5 + i] = np.sqrt(1 - gx**2)
        rows.append(v); xs.append(v)
    a = (xs[0] + xs[1]) / np.linalg.norm(xs[0] + xs[1])
    a[7] = np.sqrt(s2)
    rows.append(a)
    v = np.zeros(L); v[8] = 1.0; rows.append(v)
    V = np.array(rows)
    R = V @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d), names


def picks(R, names, k=4):
    out = {}
    for nm, f in [("FL", facility_location_select), ("EV", greedy_trace_select),
                  ("entropy", greedy_entropy), ("MI", greedy_mi)]:
        out[nm] = [names[j] for j in f(R, k)["selected"]]
    return out


def as_stated(p):
    return (p["FL"][:3] == ["C1", "Y", "O"] and p["EV"][:3] == ["C1", "Y", "O"]
            and p["entropy"][:2] == ["C1", "O"] and p["MI"][:3] == ["Y", "C1", "C2"])


if __name__ == "__main__":
    R, names = suite()
    np.set_printoptions(precision=3, suppress=True)
    print(names)
    print(R)
    p = picks(R, names)
    print(p, "as stated:", as_stated(p))
    bad = [(c, s2) for c in (0.15, 0.2, 0.25, 0.3) for s2 in (0.01, 0.05, 0.1)
           if not as_stated(picks(*suite(0.9, c, s2)))]
    print("grid at rho = 0.9:", "all as stated" if not bad else f"fails at {bad}")
    sys.exit(1 if bad or not as_stated(p) else 0)
