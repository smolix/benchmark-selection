"""Greedy benchmark selection via entropy and mutual information.

Implements:
  - Algorithm 1 (greedy entropy): select the benchmark with largest
    conditional variance at each step (= pivoted Cholesky).
  - Algorithm 2 (greedy MI): select the benchmark maximizing
    log σ²_{v|S} + log P_{vv}, where P_{vv} is the complement precision
    diagonal.

Numerical near-ties (TIE_RTOL) in the greedy scores of both algorithms
are broken deterministically by explained variance, then by the lowest
column index (see _tie_break, _entropy_pivot).

All selectors take an optional `eligible` (N,) bool mask: only eligible
benchmarks can be picked (cv_split.eligible_mask: observed in at least
MIN_TRAIN_OBS training models), while ineligible ones stay in Σ, in the
MI complement and in the residual-variance trajectories.  When fewer than
k benchmarks are eligible, all eligible ones are selected and the per-step
arrays are NaN beyond.  eligible=None (or all True) is the unrestricted
selection, bit-identical to the selection without the argument.
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

# Tolerance below which two pivot scores count as tied.  EM / BLAS rounding
# perturbs a unit diagonal at the ~1e-16 level (EM correlation entries at
# up to ~1e-12 across thread counts); genuine differences between
# conditional variances are many orders of magnitude larger.  It is applied
# relative to max(|best score|, noise floor of the score), see _near_max.
TIE_RTOL = 1e-9


def _near_max(
    values: NDArray, scale: float = 1.0, rtol: float = TIE_RTOL
) -> NDArray:
    """Boolean mask of entries within rtol·max(|max|, scale) of the maximum.

    `scale` is the magnitude below which rounding in `values` is absolute
    rather than relative: the variance scale max_i Σ_ii for conditional and
    explained variances, 1 for dimensionless log-values, and 0 for a purely
    relative test (e.g. gain/cost ratios, whose size depends on the cost
    units).
    """
    vmax = np.max(values)
    return values >= vmax - rtol * max(abs(vmax), scale)


def _explained_variance(
    Sigma: NDArray,
    L: NDArray,
    d: NDArray,
    t: int,
    cand: NDArray,
    unsel: NDArray,
) -> NDArray:
    """Residual variance of the unselected set explained by each candidate.

    For candidate pivot j after t pivots, returns
        sum_{i ∈ unsel} r_ij² / d_j,   r_ij = Σ_ij − L[i, :t] · L[j, :t],
    i.e. the drop in the pivoted-Cholesky residual trace if j were pivoted
    next (the i = j term contributes d_j itself).  Only rows of unselected
    benchmarks are used: a selected benchmark's L row stops at its own
    pivot column, so its residual covariance is zero in exact arithmetic
    and would only add rounding residue.  `cand` must be a subset of
    `unsel`.  At t = 0 on a correlation matrix this is the hub score
    sum_i R_ij².
    """
    r = Sigma[np.ix_(unsel, cand)] - L[unsel, :t] @ L[cand, :t].T
    return np.sum(r ** 2, axis=0) / np.maximum(d[cand], 1e-300)


def _tie_break(
    Sigma: NDArray,
    L: NDArray,
    d: NDArray,
    t: int,
    cand: NDArray,
    unsel: NDArray,
) -> tuple:
    """Pick one of the near-tied candidates `cand` (ascending indices).

    Shared second and third criteria of every selector: the candidate
    explaining the most residual variance of the unselected set `unsel`
    given the t pivots in (L, d) (_explained_variance, tied within
    TIE_RTOL · max(ev_max, s), s = max_i Σ_ii), then the lowest index.

    Returns (j_star, n_candidates, n_index_tied); n_index_tied > 1 is the
    number of candidates still tied on explained variance when the lowest
    index decided, else 1.
    """
    if len(cand) == 1:
        return int(cand[0]), 1, 1
    scale = float(np.max(np.diag(Sigma)))
    ev = _explained_variance(Sigma, L, d, t, cand, unsel)
    ev_tied = _near_max(ev, scale)
    return int(cand[np.argmax(ev_tied)]), len(cand), int(ev_tied.sum())


def _eligible(eligible: Optional[NDArray], N: int) -> Optional[NDArray]:
    """Validated (N,) bool copy of an eligibility mask (None stays None)."""
    if eligible is None:
        return None
    el = np.asarray(eligible, dtype=bool)
    if el.shape != (N,):
        raise ValueError(f"eligible has shape {el.shape}, expected ({N},)")
    return el


def _pad_nan(values: list, k: int) -> NDArray:
    """np.array(values), padded with NaN to length k (fewer eligible
    benchmarks than k)."""
    a = np.array(values, dtype=np.float64)
    if len(a) < k:
        a = np.concatenate([a, np.full(k - len(a), np.nan)])
    return a


def _cholesky_state(Sigma: NDArray, selected: list) -> tuple:
    """Forward Cholesky state (L, d) of Σ after pivoting `selected` in
    order, with greedy_entropy's update: L is (N, t), d the conditional
    variances given the selection (meaningful for unselected rows)."""
    N = Sigma.shape[0]
    t = len(selected)
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, t), dtype=np.float64)
    active = np.ones(N, dtype=bool)
    for s, j_star in enumerate(selected):
        active[j_star] = False
        rows = np.flatnonzero(active)
        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        L[j_star, s] = (Sigma[j_star, j_star]
                        - L[j_star, :s] @ L[j_star, :s]) / sqrt_d
        L[rows, s] = (Sigma[rows, j_star] - L[rows, :s] @ L[j_star, :s]) / sqrt_d
        d[rows] -= L[rows, s] ** 2
    return L, d


def selection_at_k(result: dict, k: int) -> Optional[list]:
    """Selected set of size k from any selector's result (PCA/CSSP's
    non-nested "selected_per_k" included), or None when fewer than k
    benchmarks were eligible (report NaN for that k)."""
    per_k = result.get("selected_per_k")
    if per_k is not None:
        return list(per_k[k - 1]) if k <= len(per_k) else None
    sel = result["selected"]
    return list(sel[:k]) if k <= len(sel) else None


def _tie_count(result: dict, key: str, k: Optional[int]) -> int:
    ties = result.get(key, [])
    if k is None:
        return len(ties)
    if "selected_per_k" in result:
        return sum(1 for kk, _, _ in ties if kk == k)
    return sum(1 for step, _ in ties if step < k)


def index_tie_count(result: dict, k: Optional[int] = None) -> int:
    """Number of picks decided by the lowest column index (index_ties).

    With k, only those affecting selection_at_k(result, k): steps < k for
    nested selectors, entries (kk, step, n) with kk == k for PCA/CSSP.
    """
    return _tie_count(result, "index_ties", k)


def tie_break_count(result: dict, k: Optional[int] = None) -> int:
    """Number of picks made among near-tied scores (tie_breaks), counted
    like index_tie_count: with k, only the picks of selection_at_k(result,
    k).  Every log stores {m}_tie_breaks with the layout of its
    {m}_index_ties (e.g. (n_splits, k_max): picks of the size-k
    selection)."""
    return _tie_count(result, "tie_breaks", k)


def _entropy_pivot(
    Sigma: NDArray,
    L: NDArray,
    d: NDArray,
    t: int,
    selected: list,
    eligible: Optional[NDArray] = None,
) -> tuple:
    """Next greedy-entropy pivot with a deterministic tie-break.

    The entropy gain of adding j is ½ log(2πe d_j), so the greedy rule is
    argmax_j d_j over unselected j.  On standardized data (a correlation
    matrix) every conditional variance starts at 1: the first gain is
    identical for every benchmark, and np.argmax resolves the tie by
    rounding noise from EM / BLAS in the estimated diagonal (1 ± ~1e-16).
    That noise differs across runs, thread counts and machines, and when
    it vanishes argmax falls back to the lowest index (alphabetical bias).

    Candidates are therefore all unselected j with
        d_j ≥ d_max − TIE_RTOL · s,   s = max_i Σ_ii ≥ d_max.
    The tolerance is relative to the variance scale s of Σ, not to d_max,
    because rounding in the Cholesky downdate d_j = Σ_jj − Σ_t L_jt² is
    absolute at that scale; conditional variances below TIE_RTOL · s (an
    essentially singular residual) are treated as indistinguishable.
    Among several candidates we pick the one explaining the most variance
    of the remaining benchmarks (_explained_variance, tied within
    TIE_RTOL · max(ev_max, s)): among pivots adding equal entropy, prefer
    the one that most reduces the pivoted-Cholesky residual trace.  At
    step 1 this selects the hub argmax_j sum_i R_ij².

    The rule is deterministic and invariant to a global rescaling of Σ.
    It is permutation-equivariant except where candidates also tie on
    explained variance, i.e. exchangeable columns such as benchmarks
    observed in too few training models for EM to tell their correlations
    apart.  The lowest column index then decides, so the pick depends on
    column order rather than on the data.

    With an eligibility mask only eligible unselected j are candidates;
    explained variance is still measured on all unselected benchmarks.

    Returns (j_star, n_candidates, n_index_tied); n_index_tied > 1 is the
    number of candidates still tied on explained variance when the lowest
    index decided, else 1.
    """
    N = len(d)
    sel_set = set(selected)
    unsel = np.array([j for j in range(N) if j not in sel_set])
    pool = unsel if eligible is None else unsel[eligible[unsel]]
    scale = float(np.max(np.diag(Sigma)))
    cand = pool[_near_max(d[pool], scale)]
    return _tie_break(Sigma, L, d, t, cand, unsel)


def random_select(
    Sigma: NDArray[np.floating],
    k: int,
    seed: Union[int, tuple] = 0,
    benchmark_names: Optional[NDArray] = None,
    eligible: Optional[NDArray] = None,
) -> dict:
    """Select k benchmarks in uniformly random order.

    Computes the same metrics as greedy_entropy (residual variance,
    log-determinants) but picks benchmarks uniformly at random without
    replacement, serving as a baseline for comparison.

    Parameters
    ----------
    Sigma : (N, N) covariance (or correlation) matrix, must be PSD.
    k : number of benchmarks to select.
    seed : random seed (int or tuple for deterministic sub-seeding).
    benchmark_names : optional (N,) array of names for reporting.
    eligible : optional (N,) bool mask of selectable benchmarks.  The
        random order is the permutation of all N indices drawn from `seed`
        restricted to the eligible ones, so splits with different masks
        still share their random numbers, and an all-True mask (or None)
        gives the unrestricted selection.

    Returns
    -------
    dict with keys:
        selected : list of min(k, n_eligible) selected indices (random order).
        names    : list of selected names (if benchmark_names provided).
        log_dets : (k,) cumulative log-det values after each selection.
        residual_variance : (k,) residual variance fraction after each step.
        tie_breaks, index_ties : always [] (no scores to tie).
        (Per-step arrays are NaN beyond the last selection.)
    """
    N = Sigma.shape[0]
    assert k >= 1
    el = _eligible(eligible, N)

    rng = np.random.default_rng(seed)
    total_var = np.trace(Sigma)

    perm = rng.permutation(N)
    if el is not None:
        perm = perm[el[perm]]
    selected = perm[:k].tolist()

    log_dets = []
    residual_variance = []
    RIDGE = 1e-8

    sel_set = set()
    for t in range(1, len(selected) + 1):
        sel_set.add(selected[t - 1])
        sel = selected[:t]
        R_AA = Sigma[np.ix_(sel, sel)]

        sign, logdet = np.linalg.slogdet(R_AA)
        log_dets.append(logdet if sign > 0 else -np.inf)

        unsel = np.array([j for j in range(N) if j not in sel_set])
        if len(unsel) > 0:
            R_UA = Sigma[np.ix_(unsel, sel)]
            R_AA_reg = R_AA + RIDGE * np.eye(t)
            W = np.linalg.solve(R_AA_reg, R_UA.T).T
            cond_var = np.diag(Sigma)[unsel] - np.sum(W * R_UA, axis=1)
            resid = np.sum(np.maximum(cond_var, 0.0))
        else:
            resid = 0.0
        residual_variance.append(resid / total_var)

    result = {
        "selected": selected,
        "log_dets": _pad_nan(log_dets, k),
        "residual_variance": _pad_nan(residual_variance, k),
        "tie_breaks": [],
        "index_ties": [],
    }
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in selected]

    return result


def greedy_entropy(
    Sigma: NDArray[np.floating],
    k: int,
    benchmark_names: Optional[NDArray] = None,
    eligible: Optional[NDArray] = None,
) -> dict:
    """Select k benchmarks by greedy entropy maximization.

    Pivots on the largest conditional variance (pivoted Cholesky).  Pivots
    whose conditional variances agree to within TIE_RTOL · max_i Σ_ii are
    tied (always the case at step 1 on a correlation matrix); such ties
    are broken by explained variance, see _entropy_pivot.  At steps
    without such a tie the pick is plain argmax pivoting, with identical
    arithmetic.

    Parameters
    ----------
    Sigma : (N, N) covariance (or correlation) matrix, must be PSD.
    k : number of benchmarks to select.
    benchmark_names : optional (N,) array of names for reporting.
    eligible : optional (N,) bool mask; only eligible benchmarks can be
        pivots (ineligible ones still count in residual_variance).

    Returns
    -------
    dict with keys:
        selected : list of min(k, n_eligible) selected indices (in
                   selection order).
        names    : list of selected names (if benchmark_names provided).
        d        : (N,) final residual diagonal (conditional variances).
        log_dets : (k,) cumulative log-det values after each selection.
        residual_variance : (k,) residual variance fraction after each step.
                   (Per-step arrays are NaN beyond the last selection.)
        tie_breaks : list of (step, n_candidates) for every step (0-based
                     position in `selected`) at which n_candidates > 1
                     pivots were tied and the tie-break decided.
        index_ties : list of (step, n_tied) for the subset of those steps
                     at which n_tied > 1 candidates also tied on explained
                     variance, so the lowest column index decided (the
                     pick there depends on column order).
    """
    N = Sigma.shape[0]
    assert Sigma.shape == (N, N)
    assert k >= 1
    el = _eligible(eligible, N)
    n_steps = min(k, N if el is None else int(el.sum()))

    total_var = np.trace(Sigma)

    # Residual diagonal = conditional variances
    d = np.diag(Sigma).copy().astype(np.float64)
    # Cholesky rows: L[j] stores the partial row for benchmark j
    L = np.zeros((N, k), dtype=np.float64)

    selected = []
    log_dets = []
    residual_variance = []
    tie_breaks = []
    index_ties = []
    cum_logdet = 0.0

    for t in range(n_steps):
        # Select pivot with largest conditional variance among unselected
        # (eligible), breaking numerical ties by explained variance
        j_star, n_cand, n_index = _entropy_pivot(Sigma, L, d, t, selected,
                                                 el)
        if n_cand > 1:
            tie_breaks.append((t, n_cand))
        if n_index > 1:
            index_ties.append((t, n_index))
        selected.append(j_star)

        cum_logdet += np.log(max(d[j_star], 1e-300))
        log_dets.append(cum_logdet)

        # Cholesky update for all remaining benchmarks
        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2

        # Residual variance = sum of conditional variances of unselected
        resid = sum(d[j] for j in range(N) if j not in selected)
        residual_variance.append(resid / total_var)

    result = {
        "selected": selected,
        "d": d,
        "log_dets": _pad_nan(log_dets, k),
        "residual_variance": _pad_nan(residual_variance, k),
        "tie_breaks": tie_breaks,
        "index_ties": index_ties,
    }
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in selected]

    return result


# entropy_rf (greedy_entropy_rf: greedy entropy from a uniformly random
# first pivot, with its helper random_first_pivot) was removed on
# 2026-10-18 (protocol rule D3); nothing else used it.


# ── Complement precision diagonal ────────────────────────────────

def _complement_precision_diag(Sigma: NDArray, comp: NDArray) -> NDArray:
    """Compute diagonal of (Sigma[comp, comp])^{-1}.

    Uses Cholesky with eigendecomposition fallback for near-singular blocks.
    Returns array of length len(comp).
    """
    from scipy.linalg import solve_triangular

    m = len(comp)
    Sigma_sub = Sigma[np.ix_(comp, comp)]

    try:
        L = np.linalg.cholesky(Sigma_sub)
        # diag(Σ^{-1}) = column norms² of L^{-1}
        L_inv = solve_triangular(L, np.eye(m), lower=True)
        return np.sum(L_inv ** 2, axis=0)
    except np.linalg.LinAlgError:
        # Eigendecomposition fallback: Σ^{-1} = V Λ^{-1} V^T
        eigvals, eigvecs = np.linalg.eigh(Sigma_sub)
        eigvals = np.maximum(eigvals, 1e-10)
        return np.sum(eigvecs ** 2 / eigvals[None, :], axis=1)


# ── Greedy MI maximization ───────────────────────────────────────

def greedy_mi(
    Sigma: NDArray[np.floating],
    k: int,
    benchmark_names: Optional[NDArray] = None,
    eligible: Optional[NDArray] = None,
) -> dict:
    """Select k benchmarks by greedy mutual information maximization.

    At each step, selects the benchmark v maximizing the MI gain
        Δ(v | S) = ½[log σ²_{v|S} + log P_{vv}]
    where σ²_{v|S} is the forward conditional variance (Cholesky) and
    P_{vv} = [Σ_{V\\S, V\\S}^{-1}]_{vv} is the complement precision diagonal.

    The complement precision is recomputed from scratch at each step via
    Cholesky factorization, avoiding the numerically unstable rank-1
    downdate of the explicit inverse.

    Gains within TIE_RTOL of the best (absolute below 1, relative above:
    _near_max with scale 1, as for any dimensionless log-value) are tied;
    such ties occur between exchangeable columns (e.g. merged's
    mt_bench/*, whose gains agree to ~1e-14 and whose argmax otherwise
    depends on BLAS rounding).  They are broken by explained variance of
    the unselected benchmarks, then by the lowest index (_tie_break).
    Without near-ties the pick is the plain argmax, with identical
    arithmetic.

    Parameters
    ----------
    Sigma : (N, N) covariance (or correlation) matrix, must be PSD.
    k : number of benchmarks to select.
    benchmark_names : optional (N,) array of names for reporting.
    eligible : optional (N,) bool mask; only eligible benchmarks can be
        picked.  The complement V\\S of the MI objective (and P_vv) still
        contains every unselected benchmark, eligible or not.

    Returns
    -------
    dict with keys:
        selected : list of min(k, n_eligible) selected indices (in
                   selection order).
        names    : list of selected names (if benchmark_names provided).
        d        : (N,) final residual diagonal (conditional variances).
        log_dets : (k,) cumulative log-det of selected block.
        residual_variance : (k,) residual variance fraction after each step.
        mi_values : (k,) mutual information I(X_A; X_{V\\A}) after each step.
        gains    : (k,) MI gain of each pick.
                   (Per-step arrays are NaN beyond the last selection.)
        tie_breaks : list of (step, n_candidates) for steps at which
                     n_candidates > 1 gains were near-tied.
        index_ties : list of (step, n_tied) for those steps at which n_tied
                     > 1 candidates also tied on explained variance, so the
                     lowest column index decided.
    """
    N = Sigma.shape[0]
    assert Sigma.shape == (N, N)
    assert k >= 1
    el = _eligible(eligible, N)
    n_steps = min(k, N if el is None else int(el.sum()))

    total_var = np.trace(Sigma)

    # log det(Σ_V) — constant term in MI formula
    sign_full, logdet_full = np.linalg.slogdet(Sigma)
    if sign_full <= 0:
        eigvals_full = np.linalg.eigvalsh(Sigma)
        logdet_full = np.sum(np.log(np.maximum(eigvals_full, 1e-300)))

    # Forward Cholesky state (identical to greedy_entropy)
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, k), dtype=np.float64)

    selected = []
    selected_set = set()
    log_dets = []
    residual_variance = []
    gains = []
    tie_breaks = []
    index_ties = []
    cum_logdet = 0.0

    for t in range(n_steps):
        # Complement indices (sorted for reproducibility)
        comp = np.array(sorted(set(range(N)) - selected_set))

        # Complement precision diagonal via fresh factorization
        P_diag_comp = _complement_precision_diag(Sigma, comp)

        # Map back to full index space
        P_full = np.zeros(N)
        for i, idx in enumerate(comp):
            P_full[idx] = P_diag_comp[i]

        # Scores log d_v + log P_vv of the (eligible) complement
        pool = comp if el is None else comp[el[comp]]
        scores = np.full(len(pool), -np.inf)
        for i, j in enumerate(pool):
            dj = d[j]
            pj = P_full[j]
            if dj > 1e-300 and pj > 1e-300:
                scores[i] = np.log(dj) + np.log(pj)

        # Selection: argmax score; near-ties (on the gain ½·score) broken
        # by explained variance, then lowest index
        if np.max(scores) == -np.inf:
            j_star, n_cand, n_index = int(pool[0]), 1, 1
            best_score = -np.inf
        else:
            cand = pool[_near_max(0.5 * scores, 1.0)]
            j_star, n_cand, n_index = _tie_break(Sigma, L, d, t, cand, comp)
            best_score = scores[np.searchsorted(pool, j_star)]
        if n_cand > 1:
            tie_breaks.append((t, n_cand))
        if n_index > 1:
            index_ties.append((t, n_index))
        selected.append(j_star)
        selected_set.add(j_star)
        gains.append(0.5 * best_score)

        # Forward Cholesky update
        cum_logdet += np.log(max(d[j_star], 1e-300))
        log_dets.append(cum_logdet)

        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected_set and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2

        # Residual variance
        resid = sum(d[j] for j in range(N) if j not in selected_set)
        residual_variance.append(resid / total_var)

    # Compute MI = ½[log det(Σ_A) + log det(Σ_{V\A}) - log det(Σ_V)]
    mi_values = []
    for kk in range(1, n_steps + 1):
        logdet_A = log_dets[kk - 1]
        comp_kk = np.array(sorted(set(range(N)) - set(selected[:kk])))
        if len(comp_kk) == 0:
            mi_values.append(0.0)
            continue
        Sigma_comp = Sigma[np.ix_(comp_kk, comp_kk)]
        sign_c, logdet_c = np.linalg.slogdet(Sigma_comp)
        if sign_c <= 0:
            eig_c = np.linalg.eigvalsh(Sigma_comp)
            logdet_c = np.sum(np.log(np.maximum(eig_c, 1e-300)))
        mi_values.append(0.5 * (logdet_A + logdet_c - logdet_full))

    result = {
        "selected": selected,
        "d": d,
        "log_dets": _pad_nan(log_dets, k),
        "residual_variance": _pad_nan(residual_variance, k),
        "mi_values": _pad_nan(mi_values, k),
        "gains": _pad_nan(gains, k),
        "tie_breaks": tie_breaks,
        "index_ties": index_ties,
    }
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in selected]

    return result
