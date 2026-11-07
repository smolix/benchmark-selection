r"""Main-text figures of the camera-ready paper (paper/figures_cr/).

Usage:
    python plot_paper.py [--figures 0 1 2 3 4] [--logs LOGS] [--out-dir OUT]
                         [--numbers NUMBERS_CSV] [--png]

Reads only the result logs of a run (LOGS = path_config.logs_dir(): the
experiment root's logs/ when BENCHSELECT_EXPERIMENT_ROOT is set; figure 0
reads the covariance files of LOGS/../cov) and, for the panel titles and
the cross-check, the run's numbers/numbers.csv (make_numbers.py; default
LOGS/../numbers/numbers.csv, skipped if absent).  Writes
OUT/fig{0,1,2,3,4}_*.pdf (OUT = path_config.figures_dir(); --png adds
300-dpi PNGs for proofreading).  Figures 1-4 are 5.5 in wide (the NeurIPS
text width), so \includegraphics[width=\textwidth] prints them at 1:1 with
7.5-9 pt text; figure 0 is 0.42 x 5.5 in wide, for a wrapfigure of
0.42\textwidth, with 7-8 pt text; fonts are embedded as TrueType
(pdf.fonttype 42, no Type 3).
No figure draws error bands: the paired intervals are in the paper's
tables and text.  Every plotted value that numbers.csv also carries is
checked against it (|difference| <= 1e-12); a mismatch is an error.

Methods, colours and markers (identical in every figure; facility location
is drawn thickest and its line on top; markers sit above all lines, so a
method that ties with facility location stays visible):
    facility location  (facloc)   #00897B solid, filled down-triangle
    explained variance (trace)    #F28E2B solid, filled up-triangle
    mutual information (mi)       #D62728 solid, filled circle
    entropy                       #4E79A7 solid, hollow circle
    hub                           #8C564B (brown) thin, square, muted
    PCA (pca)                     #B07AA1 thin, diamond, muted
    random                        #999999 dashed, x  (lighter than the
                                  #7F7F7F of the canonical figures, so that
                                  it stays apart from facility location's
                                  teal under protanopia: OKLab dE 8.3)
    random-split control of facility location: teal dotted, no marker
    cost-blind facility location:  teal dashed, hollow down-triangle
    cheapest first:                #CC79A7 dash-dot, plus
    random items:                  black dash-dot, hollow diamond

Figure 0 (fig0_spectrum.pdf): benchmark scores are nearly low-rank.  For
    MMLU, MTEB and Merged (LOGS/../cov/{ds}.cov.npz, Sigma: the full-data
    EM estimate), the fraction of variance not explained by the top k
    principal components of the correlation matrix, 1 - rho(k), against
    k = 1..25 on a log axis, with a light reference line at 10% (90%
    explained) and a dot at the first k with rho(k) >= 0.9.  These k and
    those for 95% are checked against EigK{Ds}Var{ninety,ninetyfive}
    (99%: EigK{Ds}Varninetynine) and N against Data{Ds}N.  The lines are
    data sets, not methods: greys and a dash pattern, no method colour.
    Beyond k = 28 the Merged curve flattens at the eigenvalue floor of its
    regularized EM estimate, hence the axis ends at 25.
Figure 1 (fig1_main_curves.pdf): R^2 against the number of selected
    benchmarks k = 1..15 on the four main matrices (MMLU, MTEB, Merged,
    BenchPress), one panel each with its own y-range.  Each curve is the
    mean over the 50 matched splits of logs/baselines_{ds}.npz {m}_r2
    (nanmean over the splits where R^2 is finite: BenchPress random at
    k = 1 has 41/50, MI 48/50), so the values at k = 1, 5, 10 are the
    cells of the main results table (RsqBase{Ds}{M}K{k}).  Panel titles:
    matrix, models x benchmarks and % of observed cells (Data{Ds}M, N,
    ObsPct of numbers.csv).
Figure 2 (fig2_prospective.pdf): selection on old models, imputation of
    new ones.  (a) HELM Lite v1.0 -> later releases, k = 1..7 (N - 3);
    (b) MTEB snapshot -> models added in 2026, k = 1..15; (c) Open LLM
    Leaderboard v2, submissions before -> after 2025-01-01, k = 1..15.
    y: R^2 on the new models (logs/prospective_{helm,mteb2026,olv2sub}.npz
    prosp_{m}_r2 of the primary variant; random: the mean over its 50
    draws, prosp_random_r2).  Dotted teal: facility location on 50 random
    splits of the same sizes (ctrl_facloc_mean), the iid reference.
    Titles: old -> new models and benchmarks (prosp_n_train, prosp_n_val,
    variant_n_bench; checked against NTrainProsp/NTestProsp/NBenchProsp).
Figure 3 (fig3_budgets.pdf): budgets.  (a) MMLU and (b) MTEB: R^2 (mean
    over 50 splits, logs/cost_budget_{ds}.npz {m}_r2) against the cost
    budget in % of the whole suite's cost (log axis; MMLU cost = test
    questions, MTEB cost = evaluation samples) for knapsack selection
    (cost-effective greedy plus best single benchmark) with facility
    location and explained variance, cost-blind facility location (its
    greedy order filled to the budget), cheapest first and random
    (knapsack MI, discussed nowhere in the text, was dropped on 2026-10-20).
    (c) Open LLM Leaderboard v1 items: R^2 over all 62 benchmarks (run
    benchmarks count exactly) against the number of questions per model Q
    (log axis, Q = 100..12,800; at Q = 50 no benchmark fits) for whole
    benchmarks chosen by knapsack facility location / explained variance,
    random whole benchmarks, and uniformly random items scored with a
    Gaussian posterior (items_center); logs/item_budget_olv1items.npz.
Figure 4 (fig4_mechanism.pdf): why coverage beats information.
    (a) MMLU and (b) Merged: one point per benchmark j (eligible in at
    least one split): x = R^2_{j|rest}, how well all other benchmarks
    predict j under the training correlation matrix; y = realized R^2 when
    j is the only selected benchmark (both means over the 50 splits,
    logs/analyze_objectives_{ds}.npz single_rest_r2_tr and single_r2).
    Coloured outline markers: the most frequent first pick
    (logs/baselines_{ds}.npz {m}_selected[:, 0]) of facility location,
    explained variance and MI (entropy's first pick is explained
    variance's by the tie rule; on MMLU facility location and explained
    variance pick the same benchmark).
    (c) Horizontal bars: within-split Spearman correlation between an
    objective's value and the realized R^2 over 30 uniformly random 5-sets
    per split, averaged over the 50 splits (analyze_objectives rand_* at
    k = 5: facloc_tr, pop_r2_tr for explained variance, logdet_tr, mi_tr;
    numbers.csv ObjSpearman{Ds}{M}Kfive), for the four main matrices.

HEIGHT holds the figure heights; every legend takes one row (LEGEND_H); the
random-split control of figure 2 is explained in the caption, and figure 2b
starts at YFLOOR with MI's off-scale k = 1 value marked by an arrow.
Panel (c) of figure 4 has its own legend for the bars; the bottom legend
explains the scatter markers.
"""

import argparse
import csv
import pathlib
import sys
import warnings
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import figures_dir, logs_dir

WIDTH = 5.5                      # NeurIPS text width (in)
# figure heights (in): panels, two-line panel titles, x label, legend
HEIGHT = {1: 2.0, 2: 1.85, 3: 1.85, 4: 1.8}
LEGEND_H = 0.2                  # one-row legends (revision of 2026-10-20)
MAIN_DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
DS_NAME = {"mmlu": "MMLU", "mteb": "MTEB", "merged": "MERGED",
           "benchpress": "BenchPress"}
DS_CAMEL = {"mmlu": "Mmlu", "mteb": "Mteb", "merged": "Merged",
            "benchpress": "Benchpress"}
METHODS = ["facloc", "trace", "mi", "entropy", "hub", "pca", "random"]
K_MAX = 15
TOL = 1e-12

LABEL = {"facloc": "facility location", "trace": "explained variance",
         "mi": "MI", "entropy": "entropy", "hub": "hub",
         "pca": "PCA", "random": "random",
         "control": "random-split control",
         "knap_facloc": "knapsack FL", "knap_trace": "knapsack EV",
         "knap_mi": "knapsack MI", "card_facloc": "cost-blind FL",
         "cheap_first": "cheapest first", "items_center": "random items"}
TEAL, ORANGE, RED, BLUE = "#00897B", "#F28E2B", "#D62728", "#4E79A7"
HUB_BROWN, PURPLE, GRAY, PINK = "#8C564B", "#B07AA1", "#999999", "#CC79A7"
STYLE = {
    "facloc": dict(color=TEAL, ls="-", marker="v", lw=2.1, ms=4.4,
                   mfc=TEAL, zorder=10),
    "trace": dict(color=ORANGE, ls="-", marker="^", lw=1.4, ms=3.8,
                  mfc=ORANGE, zorder=8),
    "mi": dict(color=RED, ls="-", marker="o", lw=1.2, ms=3.2, mfc=RED,
               zorder=7),
    "entropy": dict(color=BLUE, ls="-", marker="o", lw=1.2, ms=3.6,
                    mfc="white", zorder=6),
    "hub": dict(color=HUB_BROWN, ls="-", marker="s", lw=0.8, ms=2.8,
                mfc=HUB_BROWN, alpha=0.75, zorder=4),
    "pca": dict(color=PURPLE, ls="-", marker="D", lw=0.8, ms=2.6,
                mfc=PURPLE, alpha=0.75, zorder=4),
    "random": dict(color=GRAY, ls="--", marker="x", lw=1.2, ms=3.6,
                   zorder=5),
    "control": dict(color=TEAL, ls=":", marker=None, lw=1.1, zorder=9),
    "card_facloc": dict(color=TEAL, ls="--", marker="v", lw=1.2, ms=4.0,
                        mfc="white", zorder=6),
    "cheap_first": dict(color=PINK, ls="-.", marker="P", lw=1.2, ms=3.8,
                        mfc=PINK, zorder=5),
    "items_center": dict(color="#000000", ls="-.", marker="D", lw=1.4,
                         ms=3.6, mfc="white", zorder=9),
}
STYLE["knap_facloc"] = STYLE["facloc"]
STYLE["knap_trace"] = STYLE["trace"]
STYLE["knap_mi"] = STYLE["mi"]
# markers are drawn above all lines; facility location's (the largest)
# lowest, so that markers of methods it ties with stay visible on top of it
MARKER_Z = {"facloc": 20, "knap_facloc": 20, "trace": 21, "knap_trace": 21,
            "mi": 22, "knap_mi": 22, "entropy": 23, "hub": 24, "pca": 25,
            "random": 26, "card_facloc": 27, "cheap_first": 28,
            "items_center": 29}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def int_words(n):
    """95 -> 'ninetyfive' (make_numbers.int_words)."""
    n = int(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (_ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + "hundred" + (int_words(rest) if rest else "")
    rest = n % 1000
    return (int_words(n // 1000) + "thousand"
            + (int_words(rest) if rest else ""))


def num_words(x):
    """0.2 -> 'zeropointtwo', 5 -> 'five' (make_numbers.num_words)."""
    s = np.format_float_positional(round(float(x), 6), trim="-")
    whole, _, frac = s.partition(".")
    out = int_words(int(whole or 0))
    if frac:
        out += "point" + "".join(_ONES[int(c)] for c in frac)
    return out


def camel(s):
    return "".join(t[:1].upper() + t[1:] for t in s.split("_"))


# ── Style and helpers ────────────────────────────────────────────

def setup_style():
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E3E3E3",
        "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.titlepad": 3, "axes.labelpad": 2,
        "figure.dpi": 100, "savefig.dpi": 300,
    })


def nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def plot_line(ax, x, y, method, markevery=None, clip=False):
    """One curve: the line at the method's z-order (facility location on
    top of all lines), its markers above every line (MARKER_Z, facility
    location lowest but largest), so a method hidden under the thick
    facility-location line still shows its markers.  clip: clip line and markers
    to the axes (a panel with an off-scale value, figure 2b)."""
    st = STYLE[method]
    color = st["color"]
    alpha = st.get("alpha", 1.0)
    ax.plot(x, y, color=color, linestyle=st["ls"], linewidth=st["lw"],
            zorder=st["zorder"], alpha=alpha, clip_on=clip)
    if st["marker"]:
        x, y = np.asarray(x), np.asarray(y)
        idx = (np.arange(len(x)) if markevery is None
               else np.asarray([i for i in markevery if i < len(x)]))
        ax.plot(x[idx], y[idx], linestyle="none", marker=st["marker"],
                markersize=st["ms"], markerfacecolor=st.get("mfc", color),
                markeredgecolor=color, markeredgewidth=0.8, alpha=alpha,
                zorder=MARKER_Z.get(method, 30), clip_on=clip)


def legend_handle(method):
    st = STYLE[method]
    return Line2D([], [], color=st["color"], linestyle=st["ls"],
                  linewidth=min(st["lw"], 1.8), marker=st["marker"],
                  markersize=(st.get("ms", 4) + 0.6) if st["marker"] else 0,
                  markerfacecolor=st.get("mfc", st["color"]),
                  markeredgecolor=st["color"], markeredgewidth=0.8,
                  alpha=st.get("alpha", 1.0))


def make_figure(n_panels, height, legend_height, width_ratios=None):
    """(fig, axes, top, bottom): a WIDTH x height figure whose top subfigure
    holds the panels (and the shared x label) and whose bottom subfigure
    of legend_height inches holds the shared legend."""
    fig = plt.figure(figsize=(WIDTH, height), layout="constrained")
    top, bottom = fig.subfigures(
        2, 1, height_ratios=[height - legend_height, legend_height],
        hspace=0.0)
    axes = top.subplots(1, n_panels, gridspec_kw=(
        {"width_ratios": width_ratios} if width_ratios else None))
    return fig, list(np.atleast_1d(axes)), top, bottom


def bottom_legend(sub, handles, labels, ncol):
    sub.legend(handles, labels, loc="center", ncol=ncol, fontsize=7.5,
               handlelength=1.6, handletextpad=0.35, columnspacing=0.8,
               borderaxespad=0.0)


def method_legend(sub, methods, labels=None, ncol=4):
    bottom_legend(sub, [legend_handle(m) for m in methods],
                  labels or [LABEL[m] for m in methods], ncol)


def panel_title(ax, letter, name, detail):
    ax.set_title(f"({letter}) {name}", loc="left", fontsize=9, pad=10)
    ax.text(0.0, 1.015, detail, transform=ax.transAxes, fontsize=7.5,
            color="#333333", ha="left", va="bottom")


def savefig(fig, out_dir, stem, png):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pdf"
    fig.savefig(path, metadata={"CreationDate": None})
    if png:
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def fmt_int(n):
    return f"{int(n):,}"


# ── numbers.csv (titles and cross-check) ─────────────────────────

class Numbers:
    """numbers.csv lookup; check(key, value) records plotted values that
    the CSV also carries and fails on a mismatch."""

    def __init__(self, path):
        self.values = {}
        self.n_checked = 0
        self.bad = []
        self.path = path
        if path is not None and path.is_file():
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    self.values[row["key"]] = (row["value"], row["kind"])
        elif path is not None:
            print(f"warning: {path} not found; titles from the logs only, "
                  "no cross-check", file=sys.stderr)

    def get(self, key, default=None):
        if key not in self.values:
            return default
        raw, kind = self.values[key]
        if kind == "text":
            return raw
        return float(raw)

    def check(self, key, value):
        if key not in self.values:
            return False
        ref = self.get(key)
        if isinstance(ref, str):
            ok = ref == str(value)
        else:
            v = float(value)
            ok = (np.isnan(ref) and np.isnan(v)) or abs(ref - v) <= TOL
        self.n_checked += 1
        if not ok:
            self.bad.append(f"{key}: plotted {value!r}, numbers.csv {ref!r}")
        return True

    def require(self, key, value):
        if not self.check(key, value) and self.values:
            self.bad.append(f"{key}: not in {self.path}")


# ── Figure 0: spectrum of the correlation matrices ───────────────

FIG0_SIZE = (0.42 * WIDTH, 1.75)    # wrapfigure of 0.42 \textwidth, 1:1
# k = 1..25: beyond k = 28 the Merged curve flattens at the eigenvalue
# floor of its regularized EM estimate (eps_psd = 1e-3), a property of the
# estimator, not of the scores; every 90% and 95% crossing is far inside
SPEC_KMAX = 25
SPEC_LEVELS = [0.90, 0.95, 0.99]    # EigK{Ds}Var{90,95,99} of numbers.csv
SPEC_REF = 0.10                     # reference line: 90% explained
SPEC_YLIM = (3e-3, 1.0)
# (cov stem, label stem, style); legend order = curve order at k = 1 (top
# first).  The lines are data sets, not methods: neutral greys and dash
# patterns, none of them a method's colour or the random baseline's dash.
SPECTRUM = [
    ("merged", "merged leaderboards",
     dict(color="#8A8A8A", ls="-", lw=1.5)),
    ("mteb", "MTEB tasks",
     dict(color="#4A4A4A", ls=(0, (3.6, 1.4)), lw=1.25)),
    ("mmlu", "MMLU subjects",
     dict(color="#000000", ls="-", lw=1.25)),
]


def corr_spectrum(Sigma):
    """Cumulative explained-variance fraction rho(k), k = 1..N, of the
    correlation matrix of Sigma (descending eigvalsh), exactly as
    make_numbers._corr_eig_k and plot_eigenspectrum.correlation_spectrum."""
    diag_std = np.maximum(np.sqrt(np.diag(Sigma)), 1e-12)
    D_inv = 1.0 / diag_std
    R = Sigma * np.outer(D_inv, D_inv)
    eigvals = np.linalg.eigvalsh(R)[::-1]
    return np.cumsum(eigvals) / eigvals.sum()


def fig0_spectrum(logs, out_dir, nums, png):
    cov_dir = logs.parent / "cov"
    fig, ax = plt.subplots(figsize=FIG0_SIZE, layout="constrained")
    handles, labels = [], []
    for ds, name, st in SPECTRUM:
        path = cov_dir / f"{ds}.cov.npz"
        with np.load(path, allow_pickle=True) as d:
            Sigma = np.asarray(d["Sigma"], float)
            n_names = len(d["benchmark_names"])
        N = Sigma.shape[0]
        if Sigma.shape != (N, N) or n_names != N:
            nums.bad.append(f"{path}: Sigma {Sigma.shape}, "
                            f"{n_names} benchmark_names")
        D = DS_CAMEL[ds]
        nums.require(f"Data{D}N", N)
        rho = corr_spectrum(Sigma)
        resid = 1.0 - rho
        crossing = {}
        for t in SPEC_LEVELS:
            k = int(np.searchsorted(rho, t)) + 1      # smallest k, rho >= t
            crossing[t] = k
            key = f"EigK{D}Var{int_words(round(100 * t))}"
            if t < 0.99:
                nums.require(key, k)
            else:
                nums.check(key, k)
        k90 = crossing[0.90]
        # the dot is the first plotted point on or below the 10% line
        if not (resid[k90 - 1] <= SPEC_REF + 1e-12
                and (k90 == 1 or resid[k90 - 2] > SPEC_REF)):
            nums.bad.append(f"fig0 {ds}: 90% crossing k = {k90} is not the "
                            "first point below the reference line")
        print(f"  fig0 {ds}: N = {N}, k(90%) = {k90}, k(95%) = "
              f"{crossing[0.95]}, k(99%) = {crossing[0.99]}, unexplained "
              f"at k = 1: {100 * resid[0]:.1f}%, at k = {SPEC_KMAX}: "
              f"{100 * resid[SPEC_KMAX - 1]:.2f}%")
        ks = np.arange(1, min(N, SPEC_KMAX) + 1)
        ax.plot(ks, resid[:len(ks)], color=st["color"], linestyle=st["ls"],
                linewidth=st["lw"], solid_capstyle="round",
                dash_capstyle="butt", zorder=5)
        ax.plot([k90], [resid[k90 - 1]], ls="none", marker="o", ms=4.2,
                mfc=st["color"], mec="white", mew=0.6, zorder=8)
        handles.append(Line2D([], [], color=st["color"], linestyle=st["ls"],
                              linewidth=st["lw"]))
        labels.append(f"{name} ({N})")
    ax.axhline(SPEC_REF, color="#B5B5B5", lw=0.8, zorder=2)
    ax.text(SPEC_KMAX, SPEC_REF / 1.12, "90% explained", fontsize=7,
            color="#666666", ha="right", va="top")
    ax.grid(False)
    ax.set_yscale("log")
    ax.set_ylim(*SPEC_YLIM)
    ax.yaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1.0]))
    ax.set_yticklabels(["1%", "10%", "100%"])
    ax.set_xlim(0.3, SPEC_KMAX + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("number of principal components", fontsize=8)
    ax.set_ylabel("unexplained variance", fontsize=8)
    ax.legend(handles, labels, loc="upper right", fontsize=7,
              handlelength=2.2, handletextpad=0.5, labelspacing=0.25,
              borderaxespad=0.1)
    return savefig(fig, out_dir, "fig0_spectrum", png)


# ── Figure 1: R^2 vs k on the four main matrices ─────────────────

def fig1_main_curves(logs, out_dir, nums, png):
    fig, axes, top, bottom = make_figure(4, HEIGHT[1], LEGEND_H)
    ks = np.arange(1, K_MAX + 1)
    for i, (ax, ds) in enumerate(zip(axes, MAIN_DATASETS)):
        with np.load(logs / f"baselines_{ds}.npz") as d:
            curves = {m: nanmean(d[f"{m}_r2"][:, :K_MAX], axis=0)
                      for m in METHODS}
            n_bench = d["eligible"].shape[1]
        D = DS_CAMEL[ds]
        for m in METHODS[::-1]:
            plot_line(ax, ks, curves[m], m, markevery=[0, 2, 4, 6, 9, 14])
            for k in ks:
                nums.check(f"RsqBase{D}{camel(m)}K{int_words(k)}",
                           curves[m][k - 1])
            for k in (1, 5, 10):
                nums.require(f"RsqBase{D}{camel(m)}K{int_words(k)}",
                             curves[m][k - 1])
        M = nums.get(f"Data{D}M")
        N = nums.get(f"Data{D}N", n_bench)
        obs = nums.get(f"Data{D}ObsPct")
        detail = (f"{fmt_int(M)} × {int(N)}, {obs:.0f}% obs."
                  if M is not None and obs is not None
                  else f"{n_bench} benchmarks")
        panel_title(ax, "abcd"[i], DS_NAME[ds], detail)
        ax.set_xlim(0.5, K_MAX + 0.5)
        ax.set_xticks([1, 5, 10, 15])
        lo = min(np.nanmin(c) for c in curves.values())
        hi = max(np.nanmax(c) for c in curves.values())
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        if lo < 0 < hi:
            ax.axhline(0, color="#777777", lw=0.6, zorder=1)
    axes[0].set_ylabel("$R^2$ (validation models)")
    top.supxlabel("number of selected benchmarks $k$", fontsize=9)
    method_legend(bottom, METHODS, ncol=len(METHODS))
    return savefig(fig, out_dir, "fig1_main_curves", png)


# ── Figure 2: prospective ────────────────────────────────────────

PROSPECTIVE = [
    # (log stem, variant key, macro stem, title, k_max)
    ("prospective_helm", "main", "Helm", "HELM Lite: v1.0 → later", 7),
    ("prospective_mteb2026", "main", "Mteb", "MTEB: snapshot → 2026", 15),
    ("prospective_olv2sub", "cut_2025-01-01", "OlvTwoSubCutJanTwentyfive",
     "OLL v2: cutoff 2025-01-01", 15),
]


# lower y-limit of a prospective panel whose curves of interest would be
# squeezed by one off-scale value (MTEB-2026: MI at k = 1 is about 0); the
# off-scale points are marked by an arrow and their value (mark_off_scale)
YFLOOR = {"Mteb": 0.3}


def mark_off_scale(ax, ks, curves, ctrl, floor):
    """Set the y-range [floor, top] and mark every point below the floor by
    a downward arrow at the bottom edge with the method and its value."""
    finite = [v for c in list(curves.values()) + [ctrl] for v in c
              if np.isfinite(v)]
    top = max(finite)
    ax.set_ylim(floor, top + 0.04)
    for m, y in curves.items():
        for k, v in zip(ks, y):
            if np.isfinite(v) and v < floor:
                color = STYLE[m]["color"]
                label = f"{LABEL[m]}: {v:.2f}".replace("-", "−")
                ax.annotate("", xy=(k, floor), xytext=(k, floor + 0.09),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=0.9, mutation_scale=7),
                            annotation_clip=False, zorder=40)
                ax.text(k + 0.9, floor + 0.045, label, fontsize=7,
                        color=color, ha="left", va="center", zorder=40,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white",
                                  ec="none", alpha=0.85))


def fig2_prospective(logs, out_dir, nums, png):
    fig, axes, top, bottom = make_figure(3, HEIGHT[2], LEGEND_H)
    for i, (ax, (stem, variant, S, title, kmax)) in enumerate(
            zip(axes, PROSPECTIVE)):
        with np.load(logs / f"{stem}.npz") as d:
            keys = [str(v) for v in d["variant_keys"]]
            v = keys.index(variant)
            curves = {m: np.asarray(d[f"prosp_{m}_r2"][v], float)
                      for m in METHODS}
            draws = nanmean(d["prosp_random_r2_draws"][v], axis=0)
            ctrl = np.asarray(d["ctrl_facloc_mean"][v], float)
            n_old, n_new = int(d["prosp_n_train"][v]), int(d["prosp_n_val"][v])
            n_bench = int(d["variant_n_bench"][v])
        if not np.allclose(draws, curves["random"], equal_nan=True,
                           atol=TOL, rtol=0):
            nums.bad.append(f"{stem}: prosp_random_r2 is not the mean of "
                            "its draws")
        for key, val in ((f"NTrainProsp{S}", n_old), (f"NTestProsp{S}", n_new),
                         (f"NBenchProsp{S}", n_bench)):
            nums.require(key, val)
        ks = np.arange(1, kmax + 1)
        floor = YFLOOR.get(S)
        plot_line(ax, ks, ctrl[:kmax], "control", clip=floor is not None)
        for m in METHODS[::-1]:
            y = curves[m][:kmax]
            plot_line(ax, ks, y, m,
                      markevery=None if kmax <= 7 else [0, 2, 4, 6, 9, 14],
                      clip=floor is not None)
            for k in ks:
                nums.check(f"RsqProsp{S}{camel(m)}K{int_words(k)}", y[k - 1])
            nums.require(f"RsqProsp{S}{camel(m)}Kfive", y[4])
        for k in ks:
            nums.check(f"RsqCtrl{S}FaclocK{int_words(k)}", ctrl[k - 1])
        nums.require(f"RsqCtrl{S}FaclocKfive", ctrl[4])
        panel_title(ax, "abc"[i], title,
                    f"{fmt_int(n_old)} → {fmt_int(n_new)} models, "
                    f"{n_bench} benchmarks")
        ax.set_xlim(0.6, kmax + 0.4)
        ax.set_xticks([1, 3, 5, 7] if kmax <= 7 else [1, 5, 10, 15])
        if floor is not None:
            mark_off_scale(ax, ks, {m: curves[m][:kmax] for m in METHODS},
                           ctrl[:kmax], floor)
    axes[0].set_ylabel("$R^2$ (new models)")
    top.supxlabel("number of selected benchmarks $k$", fontsize=9)
    # the dotted random-split control is explained in the caption
    method_legend(bottom, METHODS, ncol=len(METHODS))
    return savefig(fig, out_dir, "fig2_prospective", png)


# ── Figure 3: budgets ────────────────────────────────────────────

COST_METHODS = ["knap_facloc", "knap_trace", "card_facloc",
                "cheap_first", "random"]
ITEM_METHODS = [("bench_facloc", "knap_facloc"), ("bench_trace", "knap_trace"),
                ("bench_random", "random"), ("items_center", "items_center")]


def fig3_budgets(logs, out_dir, nums, png):
    fig, axes, top, bottom = make_figure(3, HEIGHT[3], LEGEND_H)
    unit = {"mmlu": "questions", "mteb": "evaluation samples"}
    for i, ds in enumerate(["mmlu", "mteb"]):
        ax = axes[i]
        with np.load(logs / f"cost_budget_{ds}.npz") as d:
            pct = 100 * np.asarray(d["budgets_frac"], float)
            curves = {m: nanmean(d[f"{m}_r2"], axis=1) for m in COST_METHODS}
        D = DS_CAMEL[ds]
        for m in COST_METHODS[::-1]:
            plot_line(ax, pct, curves[m], m)
            for j, p in enumerate(pct):
                nums.require(f"CostRsq{D}{camel(m)}B{num_words(p)}",
                             curves[m][j])
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(pct))
        ax.xaxis.set_minor_locator(NullLocator())
        shown = {1, 2, 5, 20} if ds == "mmlu" else {0.2, 1, 5, 20}
        ax.set_xticklabels([f"{p:g}%" if round(p, 6) in shown else ""
                            for p in pct])
        ax.set_xlim(pct[0] / 1.25, pct[-1] * 1.25)
        panel_title(ax, "ab"[i], DS_NAME[ds], f"cost = {unit[ds]}")
        ax.set_xlabel("budget (% of the suite's cost)")
    ax = axes[2]
    with np.load(logs / "item_budget_olv1items.npz") as d:
        q = np.asarray(d["budgets"], float)
        curves = {m: nanmean(d[f"{m}_r2"], axis=0) for m, _ in ITEM_METHODS}
        n_bench = int(np.asarray(d["costs"]).shape[0])
    keep = q >= 100                     # at Q = 50 no benchmark fits
    for m, style in ITEM_METHODS[::-1]:
        plot_line(ax, q[keep], curves[m][keep], style)
        for j in np.flatnonzero(keep):
            nums.require(f"ItemBudgetRsqOlvOneItems{camel(m)}"
                         f"Q{int_words(q[j])}", curves[m][j])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(q[keep]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([fmt_int(x) if x in (200, 800, 3200, 12800) else ""
                        for x in q[keep]])
    ax.set_xlim(q[keep][0] / 1.3, q[keep][-1] * 1.3)
    panel_title(ax, "c", "OLL v1 items",
                f"scored on all {n_bench} benchmarks")
    ax.set_xlabel("questions per model $Q$")
    axes[0].set_ylabel("$R^2$ (validation models)")
    method_legend(bottom, COST_METHODS + ["items_center"],
                  ncol=len(COST_METHODS) + 1)
    return savefig(fig, out_dir, "fig3_budgets", png)


# ── Figure 4: mechanism ──────────────────────────────────────────

SPEARMAN_Q = [("facloc", "facloc_tr"), ("trace", "pop_r2_tr"),
              ("entropy", "logdet_tr"), ("mi", "mi_tr")]
PICK_STYLE = {   # first-pick markers of the scatter panels (outlines, so that
    # coinciding picks stay readable: on MMLU the down- and up-triangle of
    # facility location and explained variance mark the same benchmark)
    "trace": dict(marker="^", s=46, facecolor="none", edgecolor=ORANGE,
                  linewidths=1.3, zorder=11),
    "facloc": dict(marker="v", s=46, facecolor="none", edgecolor=TEAL,
                   linewidths=1.3, zorder=12),
    "mi": dict(marker="o", s=46, facecolor="none", edgecolor=RED,
               linewidths=1.3, zorder=13),
}


def _spearman(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok])[0])


def _first_pick(logs, ds, method):
    with np.load(logs / f"baselines_{ds}.npz") as d:
        first = np.asarray(d[f"{method}_selected"])[:, 0]
        names = [str(n) for n in d["benchmark_names"]]
    first = first[first >= 0]
    counts = Counter(first.tolist())
    best = max(counts.values())
    j = min(i for i, c in counts.items() if c == best)   # ties: lowest index
    return j, names[j], best, len(first)


def fig4_mechanism(logs, out_dir, nums, png):
    fig, axes, top, bottom = make_figure(3, HEIGHT[4], LEGEND_H,
                                         width_ratios=[1, 1, 1.5])
    for i, ds in enumerate(["mmlu", "merged"]):
        ax = axes[i]
        with np.load(logs / f"analyze_objectives_{ds}.npz") as a:
            x = nanmean(a["single_rest_r2_tr"], axis=0)
            y = nanmean(a["single_r2"], axis=0)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=7, color="#A0A0A0", edgecolor="none",
                   zorder=3, clip_on=False)
        D = DS_CAMEL[ds]
        for m in ("trace", "facloc", "mi"):
            j, name, cnt, n = _first_pick(logs, ds, m)
            nums.check(f"Sel{D}{camel(m)}PtenFirstName", name)
            nums.check(f"Sel{D}{camel(m)}PtenFirstFolds", cnt)
            print(f"  fig4 {ds}: {m} first pick {name} ({cnt}/{n} splits), "
                  f"x = {x[j]:.3f}, y = {y[j]:.3f}")
            ax.scatter([x[j]], [y[j]], clip_on=False, **PICK_STYLE[m])
        span_x = np.ptp(x[ok])
        span_y = np.ptp(y[ok])
        ax.set_xlim(x[ok].min() - 0.05 * span_x, x[ok].max() + 0.06 * span_x)
        ax.set_ylim(y[ok].min() - 0.05 * span_y, y[ok].max() + 0.08 * span_y)
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel(r"$R^2_{j\,|\,\mathrm{rest}}$ (training)")
        ax.grid(True, axis="both")
        n_all = int(nums.get(f"Data{D}N", len(ok)))
        count = (f"{int(ok.sum())}" if int(ok.sum()) == n_all
                 else f"{int(ok.sum())} of {n_all}")
        panel_title(ax, "ab"[i], DS_NAME[ds],
                    f"one point per benchmark $j$ ({count})")
    axes[0].set_ylabel(r"$R^2$ with only $j$ selected")
    # (c) Spearman bars
    ax = axes[2]
    height = 0.2
    ys = np.arange(len(MAIN_DATASETS))
    lo = 0.0
    for qi, (m, q) in enumerate(SPEARMAN_Q):
        vals = []
        for ds in MAIN_DATASETS:
            with np.load(logs / f"analyze_objectives_{ds}.npz") as a:
                ks = [int(k) for k in a["rand_ks"]]
                ik = ks.index(5)
                rho = [_spearman(a[f"rand_{q}"][s, :, ik],
                                 a["rand_r2"][s, :, ik])
                       for s in range(a["rand_r2"].shape[0])]
            val = nanmean(np.array(rho))
            nums.require(f"ObjSpearman{DS_CAMEL[ds]}{camel(m)}Kfive", val)
            vals.append(val)
        lo = min(lo, min(vals))
        yq = ys + (qi - 1.5) * height
        ax.barh(yq, vals, height * 0.9,
                color=STYLE[m]["color"], linewidth=0, zorder=3)
        # a dot at each bar end keeps near-zero bars visible
        ax.plot(vals, yq, ls="none", marker="o", ms=2.2,
                color=STYLE[m]["color"], zorder=5, clip_on=False)
    ax.axvline(0, color="#444444", lw=0.6, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([DS_NAME[d] for d in MAIN_DATASETS])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(ys) - 0.45, -0.55)          # MMLU on top
    ax.set_xlim(min(-0.1, lo - 0.05), 1.0)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel(r"Spearman $\rho$ with $R^2$")
    panel_title(ax, "c", "Objective vs. $R^2$", "30 random sets of 5 per split")
    # (c) has its own legend for the bars (top to bottom: FL, EV, entropy, MI)
    bar_handles = [Patch(facecolor=STYLE[m]["color"], edgecolor="none")
                   for m, _ in SPEARMAN_Q]
    ax.legend(bar_handles, ["FL", "EV", "entropy", "MI"], loc="lower right",
              ncol=1, fontsize=7, handlelength=1.0, handletextpad=0.3,
              borderaxespad=0.1, labelspacing=0.25, frameon=False)
    # the bottom legend explains the scatter panels (a, b) only
    handles = [Line2D([], [], ls="none", marker="v", ms=6.5, color=TEAL,
                      mfc="none", mew=1.3),
               Line2D([], [], ls="none", marker="^", ms=6.5, color=ORANGE,
                      mfc="none", mew=1.3),
               Line2D([], [], ls="none", marker="o", ms=6, color=RED,
                      mfc="none", mew=1.3),
               Line2D([], [], ls="none", marker="o", ms=3.2, color="#A0A0A0",
                      mfc="#A0A0A0")]
    labels = ["first pick of facility location", "of explained variance",
              "of MI", "other benchmarks"]
    bottom.legend(handles, labels, loc="center", ncol=4, fontsize=7.5,
                  handletextpad=0.2, columnspacing=1.2, borderaxespad=0.0,
                  handlelength=1.2)
    return savefig(fig, out_dir, "fig4_mechanism", png)


FIGURES = {0: fig0_spectrum, 1: fig1_main_curves, 2: fig2_prospective,
           3: fig3_budgets, 4: fig4_mechanism}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Main-text figures of the camera-ready paper.")
    parser.add_argument("--figures", nargs="+", type=int,
                        default=sorted(FIGURES), choices=sorted(FIGURES))
    parser.add_argument("--logs", type=pathlib.Path, default=None,
                        help="result logs (default: path_config.logs_dir())")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None,
                        help="output directory (default: "
                             "path_config.figures_dir())")
    parser.add_argument("--numbers", type=pathlib.Path, default=None,
                        help="numbers.csv for titles and the cross-check "
                             "(default: LOGS/../numbers/numbers.csv)")
    parser.add_argument("--png", action="store_true",
                        help="also write 300-dpi PNGs")
    args = parser.parse_args(argv)
    logs = args.logs or logs_dir()
    out_dir = args.out_dir or figures_dir()
    numbers = args.numbers or (logs.parent / "numbers" / "numbers.csv")
    setup_style()
    nums = Numbers(numbers)
    for f in args.figures:
        FIGURES[f](logs, out_dir, nums, args.png)
    print(f"cross-check: {nums.n_checked} plotted values compared with "
          f"{numbers}")
    if nums.bad:
        for msg in nums.bad:
            print(f"MISMATCH {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
