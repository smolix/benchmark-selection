r"""Table bodies of the camera-ready (blueprint sections 4.3 and 5.1).

Usage (on a server; never on the laptop):
    ~/venvs/benchselect/bin/python make_tables.py \
        --numbers ~/benchselect-runs/canonical-20261020/out/numbers \
        --derived numbers_derived.csv --out tables/

Every number in a body is a number command of cr/numbers_cmds.tex
(\nv{Key}, \nvd, \nvneg, \nvci, \nvcineg, \nvdiff), so the bodies go through
paper_numbers.sh like the text (audit trail: numbers_used.csv, file =
tables/<name>.tex) and print exactly the strings of the text.  This script
only decides the layout from the full-precision values: which cell is bold
(the row maximum at 3 decimals, and every cell equal to it at 3 decimals),
which cell carries a significance mark, which cell is missing.  It reads
numbers.csv, the paired_<ds>.csv rows (imported under the names that
paper_numbers.sh gives them) and numbers_derived.csv, and fails on any key
it cannot find unless the key is in ALLOWED_MISSING (printed as ---).

Bodies written to --out:
  tab_main_body.tex        rows only (Table 2, tab:main): the 12 rows of the
                           four main matrices x k in {1, 5, 10}, then
                           \midrule and the scoreboard (3 count rows); the
                           section file holds tabular, header and rules.
  tab_app_results_body.tex complete tabulars, panels (a) k = 5 on all
                           matrices and (b) p = 0.9
  tab_app_prosp_body.tex   complete tabular (prospective settings, k = 5)
  tab_app_family_body.tex  complete tabulars, panels (a) and (b)
  tab_app_benchfam_body.tex, tab_app_estimator_body.tex,
  tab_app_imputers_body.tex, tab_app_mech_body.tex   complete tabulars
  tab_app_costs_body.tex, tab_app_tokens_body.tex    significance marks
                           over budgets (Appendix F), complete tabulars
"""

import argparse
import csv
import math
import pathlib
import re
import sys

import numpy as np

# ── names (make_numbers.camel / int_words, as paper_numbers.awk) ─────────

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_TOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+(?:\.\d+)?")


def int_words(n):
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
    s = np.format_float_positional(round(float(x), 6), trim="-")
    whole, _, frac = s.partition(".")
    out = int_words(int(whole or 0))
    if frac:
        out += "point" + "".join(_ONES[int(c)] for c in frac)
    return out


def camel(s):
    out = []
    for tok in _TOKEN.findall(str(s).replace("²", "2")):
        if tok[0].isdigit():
            tok = num_words(tok)
        out.append(tok[0].upper() + tok[1:].lower())
    return "".join(out)


# ── key store ────────────────────────────────────────────────────

STATS = [("Diff", "mean_diff"), ("Lo", "ci_low"), ("Hi", "ci_high"),
         ("Se", "se_nb"), ("Win", "win_rate"), ("Wins", "n_wins"),
         ("Ties", "n_ties"), ("Losses", "n_losses"), ("N", "n_splits")]

# keys that may be missing; their cells print as ---
ALLOWED_MISSING = {
    "RsqProspEpochCutJanTwentysixPcaKfive",
}


class Keys:
    def __init__(self, numbers_dir, derived):
        self.v = {}
        with open(numbers_dir / "numbers.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                self.v[row["key"]] = (row["value"], row["kind"])
        groups = {}
        paired = sorted(numbers_dir.glob("paired_*.csv"))
        rows = []
        for path in paired:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    g = (row["dataset"], row["source"], row["tag"])
                    groups.setdefault(g, set()).add(row["holdout_frac"])
                    rows.append(row)
        for row in rows:
            g = (row["dataset"], row["source"], row["tag"])
            name = (row["tag"] + camel(row["dataset"])
                    + camel(row["method_a"]) + "Vs" + camel(row["method_b"])
                    + "K" + int_words(row["k"]))
            if len(groups[g]) > 1 or float(row["holdout_frac"]) != 0.1:
                name += "P" + int_words(int(100 * float(row["holdout_frac"])
                                            + 0.5))
            for s, col in STATS:
                key = "Pair" + s + name
                if key not in self.v:
                    self.v[key] = (row[col], "float")
        if derived is not None and derived.is_file():
            with open(derived, newline="") as fh:
                for row in csv.DictReader(fh):
                    self.v[row["key"]] = (row["value"], row["kind"])
        self.used = []

    def has(self, key):
        return key in self.v

    def num(self, key):
        self.used.append(key)
        if key not in self.v:
            if key in ALLOWED_MISSING:
                return math.nan
            raise KeyError(f"unknown key {key}")
        raw, kind = self.v[key]
        if kind == "text":
            raise KeyError(f"text key {key} used as a number")
        return float(raw)

    def text(self, key):
        self.used.append(key)
        raw, kind = self.v[key]
        if kind != "text":
            raise KeyError(f"{key} is not text")
        return raw


def r3(v):
    """value as printed with 3 decimals (as the numbers tool rounds)."""
    s = format(v, ".3f")
    return 0.0 if float(s) == 0 else float(s)


def digits_for(values, base=3, most=5):
    """Smallest number of decimals >= base at which no nonzero value of
    `values` prints as zero (the numbers tool prints such a value as an
    unsigned 0.000; a signed column needs its sign)."""
    vals = [float(v) for v in values if np.isfinite(v) and v != 0]
    d = base
    while d < most and any(float(format(v, f".{d}f")) == 0 for v in vals):
        d += 1
    return d


def dopt(d, base=3):
    """the optional-argument string of a number command: '' at the
    default, '[d]' otherwise."""
    return "" if d == base else f"[{d}]"


def bold_set(values):
    """indices of the row maximum at 3 decimals (NaN ignored)."""
    vals = [r3(v) if np.isfinite(v) else None for v in values]
    finite = [v for v in vals if v is not None]
    if not finite:
        return set()
    best = max(finite)
    return {i for i, v in enumerate(vals) if v is not None and v == best}


def cell(K, key, bold=False, mark="", digits=None, missing="---"):
    v = K.num(key)
    if not np.isfinite(v):
        if key in ALLOWED_MISSING:
            return missing
        raise ValueError(f"non-finite value for {key}")
    if digits is None and K.v[key][1] in ("float", "std"):
        d = digits_for([v])                  # a tiny nonzero value keeps a digit
        digits = None if d == 3 else d
    s = rf"\nv{{{key}}}" if digits is None else rf"\nv[{digits}]{{{key}}}"
    if bold:
        s = rf"\textbf{{{s}}}"
    return s + mark


def diff_ci(K, key, negate=False, digits=None, small=False):
    """One cell '+0.012 [lo, hi]' (interval in \\scriptsize; small: the
    difference too) for a Diff key (negate: an X - FL key printed as
    FL - X).  Without `digits`, the cell gets a fourth decimal if the
    difference or a bound would otherwise print as an unsigned 0.000."""
    v = K.num(key)
    if digits is None:
        lo = K.num(key.replace("Diff", "Lo", 1))
        hi = K.num(key.replace("Diff", "Hi", 1))
        digits = digits_for([v, lo, hi])
    d = dopt(digits)
    pre = r"\scriptsize" if small else ""
    if negate:
        return (rf"{{{pre}\nvneg{d}{{{key}}}\,{{\scriptsize"
                rf"\nvcineg{d}{{{key}}}}}}}")
    return rf"{{{pre}\nvd{d}{{{key}}}\,{{\scriptsize\nvci{d}{{{key}}}}}}}"


def header_comment(name, numbers_dir):
    return (f"% {name} -- GENERATED by paper/cr/tools/make_tables.py from\n"
            f"% {numbers_dir}; do not edit.  Numbers are number commands "
            "(cr/numbers_cmds.tex).\n")


MAIN = [("mmlu", "Mmlu", r"\MMLU"), ("mteb", "Mteb", r"\MTEB"),
        ("merged", "Merged", r"\Merged"),
        ("benchpress", "Benchpress", r"\BenchPress")]
EXTENDED = [("olv2sub", "OlvTwoSub", r"\OLLtwo"),
            ("olv1items", "OlvOneItems", r"\OLLone"),
            ("epoch", "Epoch", r"\Epoch"),
            ("helm_lite", "HelmLite", r"\HELMLite"),
            ("helm_mmlu", "HelmMmlu", r"\HELMMMLU"),
            ("helm_classic", "HelmClassic", r"\HELMClassic"),
            ("mteb2026", "MtebTwothousandtwentysix", r"\MTEBnew")]
METHODS = ["Facloc", "Trace", "Mi", "Entropy", "Hub", "Pca", "Random"]
KW = {1: "one", 3: "three", 5: "five", 10: "ten"}
TRI_DOWN = r"$^{\triangledown}$"
TRI_UP = r"$^{\blacktriangle}$"
DAGGER = r"$^{\dagger}$"


def fl_minus_x_interval(K, D, X, k):
    """(lo, hi) of FL - X on the matched splits (paired keys, tag Base)."""
    if X == "Random":
        lo = K.num(f"PairLoBase{D}FaclocVsRandomK{KW[k]}")
        hi = K.num(f"PairHiBase{D}FaclocVsRandomK{KW[k]}")
        return lo, hi
    lo = K.num(f"PairLoBase{D}{X}VsFaclocK{KW[k]}")
    hi = K.num(f"PairHiBase{D}{X}VsFaclocK{KW[k]}")
    return -hi, -lo


# ── Table 2 ──────────────────────────────────────────────────────

def tab_main(K, numbers_dir, aggregation):
    lines = [header_comment("tab_main_body.tex", numbers_dir),
             "% Rows only, 9 columns: Matrix & $k$ & FL & EV & MI & Entropy "
             "& Hub & PCA & Random.\n"
             "% Bold: row maximum at 3 decimals.  "
             f"{TRI_DOWN}: significantly below FL (FL - X interval > 0); "
             f"{TRI_UP}: significantly above FL;\n"
             f"% {DAGGER}: mean over the splits with finite R^2 only "
             "(fewer than 50).\n"]
    daggers = []
    for b, (ds, D, name) in enumerate(MAIN):
        if b:
            lines.append(r"\addlinespace" + "\n")
        for r, k in enumerate((1, 5, 10)):
            keys = [f"RsqBase{D}{M}K{KW[k]}" for M in METHODS]
            vals = [K.num(key) for key in keys]
            bold = bold_set(vals)
            cells = []
            for i, (M, key) in enumerate(zip(METHODS, keys)):
                mark = ""
                if M != "Facloc":
                    lo, hi = fl_minus_x_interval(K, D, M, k)
                    if lo > 0:
                        mark = TRI_DOWN
                    elif hi < 0:
                        mark = TRI_UP
                agg = aggregation.get(key, "")
                m = re.search(r"over (\d+)/(\d+) ", agg)
                if m and m.group(1) != m.group(2):
                    mark += DAGGER
                    daggers.append(f"{key} ({m.group(1)}/{m.group(2)})")
                cells.append(cell(K, key, bold=i in bold, mark=mark))
            label = name if r == 0 else ""
            lines.append(f"{label} & {k} & " + " & ".join(cells) + r" \\"
                         + "\n")
    lines.append(r"\midrule" + "\n")
    lines.append(r"\multicolumn{9}{@{}l}{\emph{Facility location against "
                 r"each column on \nv{ScoreNMatrices} matrices, "
                 r"$k \in \{3, 5, 10\}$ (\nv{ScoreNCells} settings):}} \\"
                 + "\n")
    K.num("ScoreNMatrices")
    K.num("ScoreNCells")
    for cls, label in (("SigBetter", "FL significantly better"),
                       ("NotSig", "no significant difference"),
                       ("SigWorse", "FL significantly worse")):
        cells = []
        for X in METHODS[1:]:
            key = f"ScoreFaclocVs{X}{cls}"
            K.num(key)
            cells.append(rf"\nv{{{key}}}")
        lines.append(rf"\multicolumn{{2}}{{@{{}}l}}{{{label}}} & --- & "
                     + " & ".join(cells) + r" \\" + "\n")
    return "".join(lines), daggers


# ── Appendix D ───────────────────────────────────────────────────

def tab_app_results(K, numbers_dir):
    out = [header_comment("tab_app_results_body.tex", numbers_dir)]
    out.append(r"{\footnotesize\setlength{\tabcolsep}{2.2pt}" "\n"
               r"\textit{(a) All matrices, $k = 5$, holdout $p = 0.1$}"
               "\\\\[2pt]\n"
               r"\begin{tabular}{@{}lrrrrrrrll@{}}" "\n" r"\toprule" "\n"
               r"Matrix & FL & EV & MI & Entropy & Hub & PCA & Random & "
               r"FL$\,-\,$EV \scriptsize [95\% CI] & "
               r"FL$\,-\,$Random \scriptsize [95\% CI] \\" "\n"
               r"\midrule" "\n")
    rows = MAIN + EXTENDED + [("merged_v1", "MergedVOne", r"\MergedVone")]
    for ds, D, name in rows:
        if ds == "merged_v1":
            out.append(r"\midrule" "\n")
        keys = [f"RsqBase{D}{M}Kfive" for M in METHODS]
        vals = [K.num(k) for k in keys]
        bold = bold_set(vals)
        cells = [cell(K, k, bold=i in bold) for i, k in enumerate(keys)]
        label = name                      # Merged-v1 upright (small caps)
        out.append(f"{label} & " + " & ".join(cells) + " & "
                   + diff_ci(K, f"PairDiffBase{D}TraceVsFaclocKfive",
                             negate=True, small=True) + " & "
                   + diff_ci(K, f"PairDiffBase{D}FaclocVsRandomKfive",
                             small=True)
                   + r" \\" "\n")
    out.append(r"\bottomrule" "\n" r"\end{tabular}" "\n\n" r"\medskip" "\n")
    out.append(r"\textit{(b) Fewer training models: $k = 5$, holdout "
               r"$p = 0.9$}\\[2pt]" "\n"
               r"\begin{tabular}{@{}lrrrrl@{}}" "\n" r"\toprule" "\n"
               r"Matrix & FL & EV & Entropy & Random & "
               r"FL$\,-\,$Random \scriptsize [95\% CI] \\" "\n"
               r"\midrule" "\n")
    for ds, D, name in [MAIN[0], MAIN[1], MAIN[2], MAIN[3],
                        ("epoch", "Epoch", r"\Epoch"),
                        ("olv2sub", "OlvTwoSub", r"\OLLtwo"),
                        ("mteb2026", "MtebTwothousandtwentysix",
                         r"\MTEBnew")]:
        keys = [f"Rsq{D}{M}KfivePninety"
                for M in ("Facloc", "Trace", "Entropy", "Random")]
        vals = [K.num(k) for k in keys]
        bold = bold_set(vals)
        cells = [cell(K, k, bold=i in bold) for i, k in enumerate(keys)]
        out.append(f"{name} & " + " & ".join(cells) + " & "
                   + diff_ci(K, f"PairDiff{D}FaclocVsRandomKfivePninety")
                   + r" \\" "\n")
    out.append(r"\bottomrule" "\n" r"\end{tabular}}" "\n")
    return "".join(out)


# ── Appendix E ───────────────────────────────────────────────────

PROSP_ROWS = [
    # (group header or None, row label, macro stem); labels from the
    # variant labels and cutoffs of logs/prospective_*.npz
    (r"\HELMLite", r"v1.0\,$\to$\,later", "Helm"),
    (r"\MTEBnew", "all new", "Mteb"),
    (None, "late releases", "MtebLate"),
    (r"\OLLtwo, cutoff", "2024-10", "OlvTwoSubCutOctTwentyfour"),
    (None, "2025-01", "OlvTwoSubCutJanTwentyfive"),
    (None, "2025-03", "OlvTwoSubCutMarTwentyfive"),
    (None, "2025-01, MATH", "OlvTwoSubMathCutJanTwentyfive"),
    (r"\Epoch, cutoff", "2024-07", "EpochCutJulTwentyfour"),
    (None, "2025-01", "EpochCutJanTwentyfive"),
    (None, "2025-07", "EpochCutJulTwentyfive"),
    (None, "2026-01", "EpochCutJanTwentysix"),
    (None, "2025-01, model", "EpochByModelCutJanTwentyfive"),
]


def tab_app_prosp(K, numbers_dir):
    out = [header_comment("tab_app_prosp_body.tex", numbers_dir)]
    out.append(r"{\footnotesize\setlength{\tabcolsep}{2.2pt}" "\n"
               r"\begin{tabular}{@{}lrrrrrrrrrl@{}}" "\n" r"\toprule" "\n"
               r"Setting & Old\,$\to$\,new & FL & EV & MI & Entropy & Hub & "
               r"PCA & Random & Control & "
               r"FL$\,-\,$EV \scriptsize [95\% CI] \\" "\n" r"\midrule" "\n")
    first = True
    for group, label, S in PROSP_ROWS:
        if group is not None:
            if not first:
                out.append(r"\addlinespace" "\n")
            out.append(rf"\multicolumn{{11}}{{@{{}}l}}{{{group}}} \\" "\n")
            first = False
        keys = [f"RsqProsp{S}{M}Kfive" for M in METHODS]
        vals = [K.num(k) for k in keys]
        bold = bold_set(vals)
        cells = [cell(K, k, bold=i in bold) for i, k in enumerate(keys)]
        K.num(f"NTrainProsp{S}")
        K.num(f"NTestProsp{S}")
        sizes = rf"\nv{{NTrainProsp{S}}}\,$\to$\,\nv{{NTestProsp{S}}}"
        out.append(rf"\quad {label} & {sizes} & " + " & ".join(cells)
                   + " & " + cell(K, f"RsqCtrl{S}FaclocKfive") + " & "
                   + diff_ci(K, f"BootDiffProsp{S}TraceVsFaclocKfive",
                             negate=True, small=True) + r" \\" "\n")
    out.append(r"\bottomrule" "\n" r"\end{tabular}}" "\n")
    return "".join(out)


def tab_app_family(K, numbers_dir):
    out = [header_comment("tab_app_family_body.tex", numbers_dir)]
    out.append(r"{\footnotesize\setlength{\tabcolsep}{3.2pt}" "\n"
               r"\textit{(a) All matrices, $k = 5$}\\[2pt]" "\n"
               r"\begin{tabular}{@{}lrrrrrrrrrr@{}}" "\n" r"\toprule" "\n"
               r"Matrix & Families & FL & EV & MI & Entropy & Hub & PCA & "
               r"Random & Gap & Jaccard \\" "\n" r"\midrule" "\n")
    for ds, D, name in MAIN + EXTENDED:
        keys = [f"RsqFam{D}{M}Kfive" for M in METHODS]
        vals = [K.num(k) for k in keys]
        bold = bold_set(vals)
        cells = [cell(K, k, bold=i in bold) for i, k in enumerate(keys)]
        K.num(f"NFam{D}")
        K.num(f"RsqBase{D}FaclocKfive")
        K.num(f"JaccFam{D}FaclocKfive")
        out.append(f"{name} & " + rf"\nv{{NFam{D}}} & " + " & ".join(cells)
                   + rf" & \nvdiff{dopt(digits_for([K.num(f'RsqFam{D}FaclocKfive') - K.num(f'RsqBase{D}FaclocKfive')]))}{{RsqFam{D}FaclocKfive}}"
                     rf"{{RsqBase{D}FaclocKfive}}"
                   + rf" & \nv[2]{{JaccFam{D}FaclocKfive}} \\" "\n")
    out.append(r"\bottomrule" "\n" r"\end{tabular}" "\n\n" r"\medskip" "\n")
    out.append(r"\textit{(b) Facility location's $R^2$ on each held-out "
               r"family, $k = 5$}\\[2pt]" "\n"
               "% family names: FamList{Ds} of numbers.csv "
               "(logs/family_holdout_<ds>.npz families)\n"
               r"\begin{tabularx}{\linewidth}"
               r"{@{}l>{\raggedright\arraybackslash}X@{}}"
               "\n" r"\toprule" "\n")
    for i, (ds, D, name) in enumerate(MAIN):
        fams = [f.strip() for f in K.text(f"FamList{D}").split(",")]
        items = []
        for fam in fams:
            key = f"RsqFam{D}FaclocKfive{camel(fam)}"
            K.num(key)
            items.append(rf"{fam}~\nv{{{key}}}")
        if i:
            out.append(r"\addlinespace" "\n")
        out.append(f"{name} & " + "; ".join(items) + r" \\" "\n")
    out.append(r"\bottomrule" "\n" r"\end{tabularx}}" "\n")
    return "".join(out)


# ── Appendix F ───────────────────────────────────────────────────

MARK_UP = r"$\blacktriangle$"       # facility location significantly ahead
MARK_DOWN = r"$\triangledown$"      # facility location significantly behind
MARK_NS = r"$\cdot$"                # no significant difference

# budget grids (design: eval_cost_budget.py --fixed-cost, eval_token_costs.py)
COST_BUDGETS = {"Mmlu": [1, 2, 3, 5, 8, 12, 20],
                "Mteb": [0.2, 0.5, 1, 2, 5, 10, 20]}
TOKEN_BUDGETS = [1, 2, 5, 10, 20, 30, 50]


def _mark_row(K, keys_neg):
    """Marks and the largest |FL - X| cell of one comparison over budgets.
    keys_neg: [(Diff key, negate)], negate = the key is X - FL."""
    marks, best = [], None
    for key, neg in keys_neg:
        if not K.has(key):
            marks.append("--")
            continue
        v = K.num(key)
        lo = K.num(key.replace("Diff", "Lo", 1))
        hi = K.num(key.replace("Diff", "Hi", 1))
        if not (np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi)):
            marks.append("--")
            continue
        if neg:
            v, lo, hi = -v, -hi, -lo
        marks.append(MARK_UP if lo > 0 else MARK_DOWN if hi < 0 else MARK_NS)
        if best is None or abs(v) > abs(best[0]):
            best = (v, key, neg)
    if best is None:
        return marks, "--"
    _, key, neg = best
    return marks, (rf"\nvneg{{{key}}}" if neg else rf"\nvd{{{key}}}")


def tab_app_costs(K, numbers_dir):
    """Appendix F.1: knapsack facility location against five alternatives
    at every budget, without and with a fixed overhead per benchmark."""
    out = [header_comment("tab_app_costs_body.tex", numbers_dir)]
    comps = [("cheapest-first", "CheapFirstVsKnapFacloc", True),
             ("random", "KnapFaclocVsRandom", False),
             ("cost-blind FL", "KnapFaclocVsCardFacloc", False),
             ("knapsack MI", "KnapMiVsKnapFacloc", True),
             ("knapsack EV", "KnapTraceVsKnapFacloc", True)]
    out.append(r"{\footnotesize\setlength{\tabcolsep}{3.4pt}" "\n")
    for b, (D, name) in enumerate((("Mmlu", r"\MMLU"), ("Mteb", r"\MTEB"))):
        budgets = COST_BUDGETS[D]
        if b:
            out.append(r"\medskip" "\n\n")
        out.append(r"\begin{tabular}{@{}ll" + "c" * len(budgets) + "r@{}}\n"
                   r"\toprule" "\n"
                   rf" & knapsack FL against & "
                   + " & ".join(f"{p:g}\\%" for p in budgets)
                   + r" & largest gap \\" "\n" r"\midrule" "\n")
        for f, (F, flabel) in enumerate((("Fzero", "no overhead"),
                                         ("Fone", "overhead"))):
            if f:
                out.append(r"\addlinespace" "\n")
            for c, (clabel, stem, neg) in enumerate(comps):
                keys = [(f"CostFixedPairDiff{D}{stem}{F}B{num_words(p)}", neg)
                        for p in budgets]
                marks, gap = _mark_row(K, keys)
                lab = (rf"{name}, {flabel}" if c == 0 else "")
                out.append(f"{lab} & {clabel} & " + " & ".join(marks)
                           + f" & {gap}" + r" \\" "\n")
        out.append(r"\bottomrule" "\n" r"\end{tabular}" "\n")
    out.append("}\n")
    return "".join(out)


def tab_app_tokens(K, numbers_dir):
    """Appendix F.2: facility location priced in tokens against three
    alternatives at every token budget on the three HELM suites."""
    out = [header_comment("tab_app_tokens_body.tex", numbers_dir)]
    comps = [("FL priced by instances", "FaclocTokensVsFaclocInstances", False),
             ("random", "FaclocTokensVsRandom", False),
             ("EV priced by tokens", "TraceTokensVsFaclocTokens", True)]
    out.append(r"{\footnotesize\setlength{\tabcolsep}{3.4pt}" "\n"
               r"\begin{tabular}{@{}ll" + "c" * len(TOKEN_BUDGETS) + "r@{}}\n"
               r"\toprule" "\n"
               r" & FL priced by tokens against & "
               + " & ".join(f"{p:g}\\%" for p in TOKEN_BUDGETS)
               + r" & largest gap \\" "\n" r"\midrule" "\n")
    for s, (S, name) in enumerate((("HelmMmlu", r"\HELMMMLU"),
                                   ("HelmLite", r"\HELMLite"),
                                   ("HelmClassic", r"\HELMClassic"))):
        if s:
            out.append(r"\addlinespace" "\n")
        for c, (clabel, stem, neg) in enumerate(comps):
            keys = [(f"TokSuitePairDiff{S}{stem}B{num_words(p)}", neg)
                    for p in TOKEN_BUDGETS]
            marks, gap = _mark_row(K, keys)
            out.append(f"{name if c == 0 else ''} & {clabel} & "
                       + " & ".join(marks) + f" & {gap}" + r" \\" "\n")
    out.append(r"\bottomrule" "\n" r"\end{tabular}}" "\n")
    return "".join(out)


# ── Appendix G ───────────────────────────────────────────────────

BENCHFAM = [("Mmlu", r"\MMLU"), ("Mteb", r"\MTEB"), ("Merged", r"\Merged"),
            ("OlvTwoSub", r"\OLLtwo"), ("OlvOneItems", r"\OLLone"),
            ("HelmMmlu", r"\HELMMMLU")]


def tab_app_benchfam(K, numbers_dir):
    out = [header_comment("tab_app_benchfam_body.tex", numbers_dir)]
    out.append(r"{\footnotesize\setlength{\tabcolsep}{3.2pt}" "\n"
               r"\begin{tabular}{@{}lrrrrrll@{}}" "\n" r"\toprule" "\n"
               r" & \multicolumn{2}{c}{FL} & EV & MI & & "
               r"\multicolumn{2}{c}{paired differences, FL "
               r"\scriptsize [95\% CI]} \\" "\n"
               r"\cmidrule(lr){2-3}\cmidrule(l){7-8}" "\n"
               r"Matrix & blind & aware & blind & blind & Random & "
               r"blind$\,-\,$Random & aware$\,-\,$blind \\" "\n"
               r"\midrule" "\n")
    for D, name in BENCHFAM:
        keys = [f"BenchFamRsq{D}AllFaclocBlindKfive",
                f"BenchFamRsq{D}AllFaclocAwareKfive",
                f"BenchFamRsq{D}AllTraceBlindKfive",
                f"BenchFamRsq{D}AllMiBlindKfive",
                f"BenchFamRsq{D}AllRandomKfive"]
        vals = [K.num(k) for k in keys]
        bold = bold_set(vals)
        cells = [cell(K, k, bold=i in bold) for i, k in enumerate(keys)]
        blind = f"BenchFamPairDiff{D}AllFaclocBlindVsRandomKfive"
        aware = f"BenchFamPairDiff{D}AllFaclocAwareVsBlindKfive"
        out.append(f"{name} & " + " & ".join(cells) + " & "
                   + diff_ci(K, blind) + " & " + diff_ci(K, aware)
                   + r" \\" "\n")
    out.append(r"\bottomrule" "\n" r"\end{tabular}}" "\n")
    return "".join(out)


EST_ROWS = {
    # source of the labels: logs/estimator_ablation_<ds>.json config.settings
    "Benchpress": [("Default", "default"), ("Pairwise", "pairwise (forced)"),
                   ("AlphaPointOne", r"$\alpha = 0.1$ fixed"),
                   ("AlphaPointThree", r"$\alpha = 0.3$ fixed"),
                   ("TolLoose", r"EM tol.\ $5\cdot 10^{-3}$"),
                   ("TolTight", r"EM tol.\ $5\cdot 10^{-5}$, $4\times$ iter."),
                   ("EligTen", r"eligibility $m = 10$")],
    "Merged": [("Default", "default"), ("Pairwise", "pairwise (forced)"),
               ("AlphaPointThree", r"$\alpha = 0.3$ fixed"),
               ("ShrinkFour", r"$\alpha = 4\alpha_0$"),
               ("TolLoose", r"EM tol.\ $5\cdot 10^{-3}$"),
               ("PsdMinusSix", r"PSD floor $10^{-6}$"),
               ("EligOne", r"eligibility $m = 1$"),
               ("EligTwo", r"eligibility $m = 2$")],
}


def tab_app_estimator(K, numbers_dir):
    out = [header_comment("tab_app_estimator_body.tex", numbers_dir)]
    out.append(r"{\footnotesize\setlength{\tabcolsep}{4pt}" "\n"
               r"\begin{tabular}{@{}lrrrr@{}}" "\n" r"\toprule" "\n"
               r"Setting & FL & EV & MI & Random \\" "\n" r"\midrule" "\n")
    for b, (D, name) in enumerate((("Benchpress", r"\BenchPress"),
                                   ("Merged", r"\Merged"))):
        if b:
            out.append(r"\addlinespace" "\n")
        out.append(rf"\multicolumn{{5}}{{@{{}}l}}{{{name}}} \\" "\n")
        for tag, label in EST_ROWS[D]:
            keys = [f"EstRsq{D}{tag}{M}Kfive"
                    for M in ("Facloc", "Trace", "Mi", "Random")]
            vals = [K.num(k) for k in keys]
            bold = bold_set(vals)
            cells = [cell(K, k, bold=i in bold) for i, k in enumerate(keys)]
            out.append(rf"\quad {label} & " + " & ".join(cells) + r" \\"
                       "\n")
    out.append(r"\bottomrule" "\n" r"\end{tabular}}" "\n")
    return "".join(out)


def tab_app_imputers(K, numbers_dir):
    out = [header_comment("tab_app_imputers_body.tex", numbers_dir)]
    out.append(r"{\footnotesize\setlength{\tabcolsep}{3.5pt}" "\n"
               r"\begin{tabular}{@{}llrrrrrl@{}}" "\n" r"\toprule" "\n"
               r"Matrix & Imputer & FL & EV & MI & Hub & Random & "
               r"FL: imputer$\,-\,$Gaussian \scriptsize [95\% CI] \\" "\n"
               r"\midrule" "\n")
    for b, (ds, D, name) in enumerate(MAIN):
        if b:
            out.append(r"\addlinespace" "\n")
        for r, (imp, label) in enumerate((("Gauss", "Gaussian"),
                                          ("Soft", "SoftImpute"),
                                          ("Hard", "HardImpute"))):
            keys = [f"RsqImp{imp}{D}{M}Kfive"
                    for M in ("Facloc", "Trace", "Mi", "Hub", "Random")]
            vals = [K.num(k) for k in keys]
            bold = bold_set(vals)
            cells = [cell(K, k, bold=i in bold) for i, k in enumerate(keys)]
            last = (r"\multicolumn{1}{c}{---}" if imp == "Gauss" else
                    diff_ci(K, f"PairDiffImp{D}Facloc{imp}VsGaussKfive"))
            out.append(f"{name if r == 0 else ''} & {label} & "
                       + " & ".join(cells) + f" & {last}" + r" \\" "\n")
    out.append(r"\bottomrule" "\n" r"\end{tabular}}" "\n")
    return "".join(out)


# ── Appendix H ───────────────────────────────────────────────────

def tab_app_mech(K, numbers_dir):
    Ds = [D for _, D, _ in MAIN]
    out = [header_comment("tab_app_mech_body.tex", numbers_dir)]
    out.append(r"{\footnotesize\setlength{\tabcolsep}{4pt}" "\n"
               r"\begin{tabular}{@{}rllrrrr@{}}" "\n" r"\toprule" "\n"
               r" & Quantity & & \MMLU & \MTEB & \Merged & \BenchPress \\" "\n"
               r"\midrule" "\n")

    def row(num, quantity, sub, keys, fmt="v2"):
        cells = []
        for key in keys:
            v = K.num(key)
            if not np.isfinite(v):
                cells.append("---")
            elif fmt == "v2":
                cells.append(rf"\nv[{digits_for([v], 2)}]{{{key}}}")
            elif fmt == "v3":
                cells.append(rf"\nv{{{key}}}")
            elif fmt == "d3":
                cells.append(rf"\nvd{dopt(digits_for([v]))}{{{key}}}")
            elif fmt == "pct":
                cells.append(rf"\nvpct{{{key}}}")
            elif fmt == "int":
                cells.append(rf"\nv{{{key}}}")
        out.append(f"{num} & {quantity} & {sub} & " + " & ".join(cells)
                   + r" \\" "\n")

    names = {"Facloc": "FL", "Trace": "EV", "Mi": "MI", "Entropy": "entropy",
             "Hub": "hub", "Pca": "PCA", "Random": "random"}
    for i, M in enumerate(("Facloc", "Trace", "Mi")):
        row("1" if i == 0 else "", r"Spearman $\rho$, single benchmarks"
            if i == 0 else "", names[M],
            [f"ObjSpearman{D}{M}Kone" for D in Ds])
    for i, M in enumerate(("Facloc", "Trace", "Entropy", "Mi", "Hub")):
        row("2" if i == 0 else "", r"Spearman $\rho$, random sets of five"
            if i == 0 else "", names[M],
            [f"ObjSpearman{D}{M}Kfive" for D in Ds])
    out.append(r"\addlinespace" "\n")
    row("3", r"MI's first pick: $R^2_{j\mid\mathrm{rest}}$", "",
        [f"ObjFirst{D}MiRestRsq" for D in Ds], fmt="v3")
    for i, M in enumerate(("Mi", "Trace")):
        row("4" if i == 0 else "", r"share of $P_{jj}$ from the smallest "
            r"10\% eigenvalues" if i == 0 else "", f"{names[M]} first pick",
            [f"ObjFirst{D}{M}PLowShare" for D in Ds], fmt="pct")
    out.append(r"\addlinespace" "\n")
    for i, M in enumerate(METHODS):
        row("5" if i == 0 else "", r"mean $|\hat R_{ij}|$ within the five picks"
            if i == 0 else "", names[M],
            [f"ObjRedund{D}{M}Kfive" for D in Ds])
    out.append(r"\addlinespace" "\n")
    row("6", r"FL $\cap$ EV at $k = 5$", "shared picks",
        [f"ObjFlVsTraceShared{D}Kfive" for D in Ds], fmt="pct")
    row("", "", "Jaccard", [f"ObjFlVsTraceJacc{D}Kfive" for D in Ds])
    row("7", r"greedy / optimal $F_{\mathrm{FL}}$, $k = 3$", "mean",
        [f"ObjBrute{D}FaclocKthreeFRatio" for D in Ds], fmt="v3")
    row("", "", "worst split",
        [f"ObjBrute{D}FaclocKthreeFRatioMin" for D in Ds], fmt="v3")
    row("8", "EV: candidate gains that grow after a pick", "",
        [f"ObjTheory{D}TraceIncreaseFrac" for D in Ds], fmt="pct")
    row("", r"EV: $\gamma$ at $k = 2$", "",
        [f"ObjTheory{D}TraceGamma" for D in Ds])
    row("", r"EV: bound $1 - e^{-\gamma}$", "",
        [f"ObjTheory{D}TraceGammaBound" for D in Ds])
    out.append(r"\addlinespace" "\n")
    for i, (part, M) in enumerate((("Total", "Facloc"), ("Total", "Trace"),
                                   ("Mask", "Facloc"), ("Mask", "Trace"))):
        row("9" if i == 0 else "", r"realized $-$ plug-in $R^2$, $k = 5$"
            if i == 0 else "",
            f"{'total' if part == 'Total' else 'mask part'}, {names[M]}",
            [f"ObjGap{part}Train{D}{M}Kfive" for D in Ds], fmt="d3")
    row("10", r"FL cover $F_{\mathrm{FL}}(\mathcal{A}_{10})/N$", "",
        [f"FaclocCover{D}Kten" for D in Ds], fmt="v3")
    row("11", "splits with a tie-broken FL pick (first 10)", "",
        [f"TieFaclocSplits{D}Kten" for D in Ds], fmt="int")
    out.append(r"\bottomrule" "\n" r"\end{tabular}}" "\n")
    return "".join(out)


# ── main ─────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--numbers", required=True, type=pathlib.Path)
    ap.add_argument("--derived", type=pathlib.Path, default=None)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args(argv)
    K = Keys(args.numbers, args.derived)
    aggregation = {}
    with open(args.numbers / "numbers.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["key"].startswith("RsqBase"):
                aggregation[row["key"]] = row["aggregation"]
    args.out.mkdir(parents=True, exist_ok=True)
    main_body, daggers = tab_main(K, args.numbers, aggregation)
    bodies = {
        "tab_main_body.tex": main_body,
        "tab_app_results_body.tex": tab_app_results(K, args.numbers),
        "tab_app_prosp_body.tex": tab_app_prosp(K, args.numbers),
        "tab_app_family_body.tex": tab_app_family(K, args.numbers),
        "tab_app_benchfam_body.tex": tab_app_benchfam(K, args.numbers),
        "tab_app_estimator_body.tex": tab_app_estimator(K, args.numbers),
        "tab_app_imputers_body.tex": tab_app_imputers(K, args.numbers),
        "tab_app_mech_body.tex": tab_app_mech(K, args.numbers),
        "tab_app_costs_body.tex": tab_app_costs(K, args.numbers),
        "tab_app_tokens_body.tex": tab_app_tokens(K, args.numbers),
    }
    for name, text in bodies.items():
        (args.out / name).write_text(text)
        print(f"wrote {args.out / name}")
    print("Table 2 daggers (partial split counts): " + "; ".join(daggers))
    missing = sorted({k for k in K.used if not K.has(k)})
    if missing:
        print("allowed missing keys (printed as ---): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
