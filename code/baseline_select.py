"""Baseline benchmark selection heuristics for reviewer comparisons.

Implements:
  - hub_select: rank benchmarks once by mean |correlation| with all
    others and take the top k (Reviewer QZ19's suggested heuristic).
  - pca_cssp_select: PCA-based column subset selection — pivoted QR on
    the top-k eigenvector loadings of the correlation matrix
    (deterministic CSSP).  Selections are recomputed per k and need not
    be nested.
  - greedy_trace_select: forward selection minimizing the total
    conditional variance of the unselected benchmarks, i.e. the "total
    residual variance" objective of Section 3.  In standardized space
    this is equivalent to maximizing the model-implied average R² of
    predicting the complement, so it is the strongest supervised
    forward-selection baseline under the Gaussian model.
  - facility_location_select: greedy facility location on the clipped
    correlation matrix, F(A) = sum_j max_{i ∈ A} max(R_ij, 0) (Yao et al.
    2026, "Coresets Before Score Sets", arXiv 2607.09739, Sec. 4.4: their
    benchmark-level comparison with greedy entropy / MI on MMLU and MTEB);
    the paper's headline method since 2026-10-19.  similarity="abs" (|R_ij|)
    and "sq" (R_ij²) are the ablation variants of the similarity
    (FACLOC_SIMILARITIES, facloc_similarity).

All functions return the same dict schema as greedy_select.greedy_entropy
(selected, log_dets, residual_variance, tie_breaks, index_ties, optional
names) so they plug into the existing CV harness.  pca_cssp_select
additionally returns "selected_per_k" since its selections are not nested.

Like the greedy selectors, every function takes an optional `eligible`
(N,) bool mask: only eligible benchmarks can be picked, while all of them
stay in Σ (hub correlations, PCA loadings, the trace objective, the
facility-location targets).  With fewer eligible benchmarks than k, all
eligible ones are selected and the per-step arrays are NaN beyond.

Near-ties (TIE_RTOL, greedy_select._near_max) of each method's score are
broken by explained variance of the unselected benchmarks given the
current selection (greedy_select._tie_break), then by the lowest index;
index_ties records the picks the index decided.  Without near-ties the
selections are those of the plain argmax / argsort / pivoted-QR rules.

PCA/CSSP pivots on a basis that is unique even where eigh's is not
(_pca_basis): a cut through a repeated eigenvalue takes its whole
eigenspace, and eigenvector directions without loading on the eligible
benchmarks are skipped; pca_cssp_select reports both per k ("n_eigvecs",
"eig_skipped").  Without repeated eigenvalues and without such directions
(none is skipped without an eligibility mask) its basis is eigh's top-kk
eigenvectors, column for column, and the selections are unchanged.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from greedy_select import (TIE_RTOL, _cholesky_state, _eligible, _near_max,
                           _pad_nan, _tie_break)

RIDGE = 1e-8


def _selection_metrics(Sigma: NDArray, selections: list,
                       k: Optional[int] = None) -> dict:
    """Compute log-det and residual-variance trajectories for a list of
    selections (selections[t] is the full selected set at step t+1),
    NaN-padded to length k."""
    N = Sigma.shape[0]
    total_var = np.trace(Sigma)
    log_dets = []
    residual_variance = []

    for sel in selections:
        sel = list(sel)
        R_AA = Sigma[np.ix_(sel, sel)]
        sign, logdet = np.linalg.slogdet(R_AA)
        log_dets.append(logdet if sign > 0 else -np.inf)

        unsel = np.array([j for j in range(N) if j not in set(sel)])
        if len(unsel) > 0:
            R_UA = Sigma[np.ix_(unsel, sel)]
            R_AA_reg = R_AA + RIDGE * np.eye(len(sel))
            W = np.linalg.solve(R_AA_reg, R_UA.T).T
            cond_var = np.diag(Sigma)[unsel] - np.sum(W * R_UA, axis=1)
            resid = np.sum(np.maximum(cond_var, 0.0))
        else:
            resid = 0.0
        residual_variance.append(resid / total_var)

    k = len(selections) if k is None else k
    return {
        "log_dets": _pad_nan(log_dets, k),
        "residual_variance": _pad_nan(residual_variance, k),
    }


def hub_select(
    Sigma: NDArray[np.floating],
    k: int,
    benchmark_names: Optional[NDArray] = None,
    eligible: Optional[NDArray] = None,
) -> dict:
    """Top-k benchmarks by mean absolute correlation with all others.

    One-shot ranking (not adaptive): score_j = mean_{i != j} |R_ij|,
    computed on the correlation matrix implied by Sigma (over all N
    benchmarks, eligible or not).  The ranking is taken over the eligible
    benchmarks.  Scores within TIE_RTOL (absolute below 1) of the best
    remaining one are tied and ordered by unconditional explained variance
    sum_i Σ_ij² / Σ_jj (keeping the ranking one-shot), then by index;
    without such ties this is np.argsort(-scores)[:k].
    """
    N = Sigma.shape[0]
    assert k >= 1
    el = _eligible(eligible, N)

    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    abs_R = np.abs(R)
    np.fill_diagonal(abs_R, 0.0)
    scores = abs_R.sum(axis=1) / (N - 1)

    remaining = np.arange(N) if el is None else np.flatnonzero(el)
    n_steps = min(k, len(remaining))
    L0 = np.zeros((N, 0))
    d0 = np.diag(Sigma).copy().astype(np.float64)
    all_idx = np.arange(N)

    selected, tie_breaks, index_ties = [], [], []
    for t in range(n_steps):
        cand = remaining[_near_max(scores[remaining], 1.0)]
        j_star, n_cand, n_index = _tie_break(Sigma, L0, d0, 0, cand, all_idx)
        if n_cand > 1:
            tie_breaks.append((t, n_cand))
        if n_index > 1:
            index_ties.append((t, n_index))
        selected.append(j_star)
        remaining = remaining[remaining != j_star]

    result = {"selected": selected, "hub_scores": scores,
              "tie_breaks": tie_breaks, "index_ties": index_ties}
    result.update(_selection_metrics(
        Sigma, [selected[:t] for t in range(1, n_steps + 1)], k))
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in selected]
    return result


def _cssp_pivots(
    Sigma: NDArray,
    Vk: NDArray,
    n_piv: int,
    pool: NDArray,
) -> tuple:
    """Businger–Golub pivoted QR of Vk^T (kk × N), pivots restricted to
    `pool` (ascending), with near-tie handling.

    Each step pivots the column v_j = Vk[j] with the largest squared norm
    of its residual after projecting out the previous pivots, i.e. pivoted
    Cholesky of the leverage Gram matrix Vk Vk^T.  Residual rows are kept
    explicitly (modified Gram–Schmidt), so the norms are recomputed, not
    downdated.  Norms within TIE_RTOL · max_j |v_j|² of the best are tied
    and broken by explained variance of the unselected benchmarks in Σ
    given the pivots so far, then by the lowest index.

    Returns (pivots, tie_breaks, index_ties) with (step, n) entries.
    """
    res = np.array(Vk, dtype=np.float64)
    scale = float(np.max(np.sum(res ** 2, axis=1)))
    N = Vk.shape[0]
    avail = np.asarray(pool)
    unsel = np.ones(N, dtype=bool)
    piv, tie_breaks, index_ties = [], [], []
    for t in range(n_piv):
        norm2 = np.sum(res[avail] ** 2, axis=1)
        cand = avail[_near_max(norm2, scale)]
        if len(cand) > 1:
            L, d = _cholesky_state(Sigma, piv)
            j_star, n_cand, n_index = _tie_break(
                Sigma, L, d, t, cand, np.flatnonzero(unsel))
            tie_breaks.append((t, n_cand))
            if n_index > 1:
                index_ties.append((t, n_index))
        else:
            j_star = int(cand[0])
        piv.append(j_star)
        unsel[j_star] = False
        avail = avail[avail != j_star]
        q = res[j_star] / max(np.linalg.norm(res[j_star]), 1e-300)
        res -= np.outer(res @ q, q)
    return piv, tie_breaks, index_ties


def _pca_basis(Sigma: NDArray, pool: NDArray) -> tuple:
    """Eigenvector basis of Σ for PCA/CSSP pivoting restricted to `pool`.

    Columns are the eigenvectors of Σ in descending eigenvalue order
    (np.linalg.eigh, then argsort(-eigvals)), except where eigh's basis is
    not determined by Σ or cannot be represented by an eligible pivot:

      * Repeated eigenvalues.  Consecutive eigenvalues whose gap is at
        most TIE_RTOL · max|λ| form one cluster.  eigh returns an
        arbitrary (BLAS- and thread-dependent) orthonormal basis of a
        repeated eigenvalue's eigenspace, so a top-kk subspace that cuts
        through a cluster is not a function of Σ.  cut[i] is the end
        (exclusive) of the cluster of column i: the subspace of size kk is
        V[:, :cut[kk − 1]], the whole cluster, on which the pivoted-QR
        residual norms do not depend on the basis.
      * Directions without eligible loading.  An eigenvector (within a
        cluster: a right singular direction of the cluster's eligible rows)
        whose squared norm on the rows `pool` is at most TIE_RTOL cannot be
        represented by any eligible pivot.  EM gives benchmarks seen by
        only one or two training models (ineligible) an uncorrelated block,
        an eigenvalue of exactly 1 supported on their rows; kept among the
        top kk, such a vector makes the eligible rows of V rank-deficient,
        and once the pivots have used up the other directions every
        remaining candidate has residual ~1e-25: all tie, and the
        explained-variance tie-break (the trace rule) picks instead of PCA.
        These directions go behind all others; they are used only if fewer
        than kk directions are visible, which cannot happen for
        kk <= len(pool): the pool rows of the full orthonormal basis are
        orthonormal, so the squared pool norms of all N directions add up
        to len(pool), each at most 1 and each skipped one at most
        TIE_RTOL.

    Without skipped directions (always without an eligibility mask) V is
    eigh's sorted basis, the same array; without repeated eigenvalues
    cut[i] = i + 1, so V[:, :cut[kk − 1]] is the plain top-kk basis.

    Returns (V, cut, n_skipped): V (N, N); cut (N,) int; n_skipped (N,)
    int, the number of directions without eligible loading whose
    eigenvalue ranks above column i's.
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    N = len(eigvals)
    pool = np.asarray(pool, dtype=int)
    tol = TIE_RTOL * float(np.max(np.abs(eigvals)))
    bounds = ([0] + [i + 1 for i in range(N - 1)
                     if eigvals[i] - eigvals[i + 1] > tol] + [N])

    visible, hidden = [], []          # column blocks
    cut, n_skipped = [], []           # per visible column
    n_vis = n_hid = 0
    for start, end in zip(bounds[:-1], bounds[1:]):
        Vc = eigvecs[:, start:end]
        m = end - start
        if m == 1:
            n_ok = int(np.sum(Vc[pool, 0] ** 2) > TIE_RTOL)
        else:
            _, s_vals, Wt = np.linalg.svd(Vc[pool], full_matrices=True)
            n_ok = int(np.sum(s_vals ** 2 > TIE_RTOL))
            if 0 < n_ok < m:
                # Rotate the eigenspace so that its first n_ok directions
                # carry all of its eligible loading
                Vc = Vc @ Wt.T
        if n_ok:
            visible.append(Vc[:, :n_ok])
            n_vis += n_ok
            cut += [n_vis] * n_ok
            n_skipped += [n_hid] * n_ok
        if n_ok < m:
            hidden.append(Vc[:, n_ok:])
            n_hid += m - n_ok
    if not hidden:
        V = eigvecs
    else:
        V = np.concatenate(visible + hidden, axis=1)
        cut += list(range(n_vis + 1, N + 1))
        n_skipped += list(range(n_hid))
    return V, np.asarray(cut, dtype=int), np.asarray(n_skipped, dtype=int)


def pca_cssp_select(
    Sigma: NDArray[np.floating],
    k: int,
    benchmark_names: Optional[NDArray] = None,
    eligible: Optional[NDArray] = None,
) -> dict:
    """PCA-based subset selection via pivoted QR on top-k loadings.

    For each kk = 1..k: take the top-kk eigenvectors V of Sigma and run
    QR with column pivoting on V^T (kk x N); the first kk pivots are the
    selected benchmarks.  This is the standard deterministic CSSP
    heuristic (Golub & Van Loan §5.4).  Selections are not nested across
    kk, so "selected_per_k" holds the per-size selections and "selected"
    holds the largest one.

    The eigenvectors come from the full Sigma; pivots are restricted to
    eligible benchmarks (kk ≤ n_eligible).  The basis is _pca_basis's:
    eigh's eigenvectors, except that a kk cutting through a repeated
    eigenvalue takes its whole eigenspace (V then has more than kk
    columns; still kk pivots) and that directions without loading on the
    eligible benchmarks are skipped (the next eigenvectors take their
    place).  Pivoting is _cssp_pivots (explicit residual norms, near-ties
    broken by explained variance, then index), which picks the same
    columns as LAPACK's geqp3 wherever the residual norms are not
    near-tied.  tie_breaks / index_ties hold (kk, step, n) entries;
    "n_eigvecs"[kk − 1] is the number of eigenvector directions spanning
    the size-kk subspace (kk unless a cluster was completed) and
    "eig_skipped"[kk − 1] the number of directions without eligible
    loading ranked above them (0 without an eligibility mask).
    """
    N = Sigma.shape[0]
    assert k >= 1
    el = _eligible(eligible, N)
    pool = np.arange(N) if el is None else np.flatnonzero(el)
    n_steps = min(k, len(pool))

    selected_per_k, tie_breaks, index_ties = [], [], []
    n_eigvecs, eig_skipped = [], []
    if n_steps:
        V, cut, n_skipped = _pca_basis(Sigma, pool)
    for kk in range(1, n_steps + 1):
        n_vec = int(cut[kk - 1])
        Vk = V[:, :n_vec]
        piv, tb, it = _cssp_pivots(Sigma, Vk, kk, pool)
        selected_per_k.append(piv)
        tie_breaks.extend((kk, t, n) for t, n in tb)
        index_ties.extend((kk, t, n) for t, n in it)
        n_eigvecs.append(n_vec)
        eig_skipped.append(int(n_skipped[n_vec - 1]))

    result = {
        "selected": selected_per_k[-1] if selected_per_k else [],
        "selected_per_k": selected_per_k,
        "tie_breaks": tie_breaks,
        "index_ties": index_ties,
        "n_eigvecs": n_eigvecs,
        "eig_skipped": eig_skipped,
    }
    result.update(_selection_metrics(Sigma, selected_per_k, k))
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in result["selected"]]
    return result


def greedy_trace_select(
    Sigma: NDArray[np.floating],
    k: int,
    benchmark_names: Optional[NDArray] = None,
    eligible: Optional[NDArray] = None,
) -> dict:
    """Forward selection minimizing total residual variance.

    At each step, pick the benchmark j whose selection minimizes
    tr(Sigma_{comp|A ∪ {j}}), i.e. maximizes the variance explained in
    the remaining benchmarks.  Maintains the conditional covariance S of
    all benchmarks given the current selection via Schur-complement
    updates; O(k N²) total.

    Only eligible j are candidates; the trace runs over all unselected
    benchmarks.  The score is itself the explained variance, so
    reductions within TIE_RTOL · max(best, max_i Σ_ii) of the best go
    straight to the lowest index (tie_breaks == index_ties).
    """
    N = Sigma.shape[0]
    assert k >= 1
    el = _eligible(eligible, N)
    n_steps = min(k, N if el is None else int(el.sum()))

    total_var = np.trace(Sigma)
    scale = float(np.max(np.diag(Sigma)))
    S = Sigma.astype(np.float64).copy()
    unselected = np.ones(N, dtype=bool)

    selected = []
    log_dets = []
    residual_variance = []
    tie_breaks = []
    index_ties = []
    cum_logdet = 0.0

    for t in range(n_steps):
        cand = np.where(unselected)[0]
        pool = cand if el is None else cand[el[cand]]
        d = np.diag(S)[pool]

        # Trace reduction from selecting j: sum_i S_ij² / S_jj over
        # unselected i (including i = j, which removes S_jj itself).
        S_sub = S[np.ix_(cand, pool)]
        col_sq = np.sum(S_sub ** 2, axis=0)
        safe_d = np.maximum(d, 1e-12)
        reduction = col_sq / safe_d
        reduction[d <= 1e-12] = -np.inf

        if np.max(reduction) == -np.inf:
            j_star = int(pool[0])
        else:
            tied = pool[_near_max(reduction, scale)]
            j_star = int(tied[0])
            if len(tied) > 1:
                tie_breaks.append((t, len(tied)))
                index_ties.append((t, len(tied)))
        selected.append(j_star)
        unselected[j_star] = False

        d_star = max(S[j_star, j_star], 1e-300)
        cum_logdet += np.log(d_star)
        log_dets.append(cum_logdet)

        # Schur complement update on the remaining block
        s_col = S[:, j_star].copy()
        S -= np.outer(s_col, s_col) / d_star
        S[j_star, :] = 0.0
        S[:, j_star] = 0.0

        resid = np.sum(np.maximum(np.diag(S)[unselected], 0.0))
        residual_variance.append(resid / total_var)

    result = {
        "selected": selected,
        "log_dets": _pad_nan(log_dets, k),
        "residual_variance": _pad_nan(residual_variance, k),
        "tie_breaks": tie_breaks,
        "index_ties": index_ties,
    }
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in selected]
    return result


# Similarities of facility_location_select: "clipped" max(R_ij, 0) is Yao et
# al.'s (arXiv 2607.09739, Sec. 4.4); "abs" |R_ij| is a sensitivity variant
# that also credits anti-correlated proxies (the imputation does); "sq"
# R_ij² (added 2026-10-19 for the similarity ablation, eval_baselines'
# facloc_sq) credits a target with the variance its best single proxy
# explains (on a correlation matrix its k = 1 gain 1 + Σ_{i≠j} R_ij² is
# greedy trace's first-step score; after that it keeps the best single
# proxy where trace regresses on all of A).  "clipped" and "abs" are
# computed exactly as before "sq" existed.
FACLOC_SIMILARITIES = ("clipped", "abs", "sq")


def facloc_similarity(Sigma: NDArray, similarity: str = "clipped") -> NDArray:
    """Similarity matrix of facility_location_select: R = Sigma rescaled to
    a unit diagonal, then max(R, 0) ("clipped"), |R| ("abs") or R² ("sq"),
    with the diagonal set to exactly 1."""
    if similarity not in FACLOC_SIMILARITIES:
        raise ValueError(f"similarity {similarity!r} not in "
                         f"{FACLOC_SIMILARITIES}")
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    if similarity == "abs":
        S = np.abs(R)
    elif similarity == "sq":
        S = R ** 2
    else:
        S = np.maximum(R, 0.0)
    np.fill_diagonal(S, 1.0)
    return S


def facility_location_select(
    Sigma: NDArray[np.floating],
    k: int,
    benchmark_names: Optional[NDArray] = None,
    eligible: Optional[NDArray] = None,
    similarity: str = "clipped",
) -> dict:
    """Greedy facility location on the clipped correlation matrix.

    Benchmark-level facility location of Yao et al. (2026), "Coresets
    Before Score Sets", arXiv 2607.09739: Sec. 4.4 runs FL "on the same
    Σ̂" as greedy entropy / MI (the correlation matrix estimated on the
    training split), "applying the truncation Σ̂⁺_ij = max(Σ̂_ij, 0) since
    FL requires non-negative pairwise similarities", with the objective of
    their Eq. (1), f(A) = Σ_{i ∈ V} max_{j ∈ A} w_ij over the whole ground
    set V, maximized by the plain greedy algorithm.  Here

        F(A) = Σ_{j=1..N} max_{i ∈ A} s_ij,   s_ij = max(R_ij, 0),

    with R the correlation matrix implied by Sigma (Sigma rescaled to a
    unit diagonal, as in hub_select; the EM / pairwise estimates of
    eval_entropy_vs_mi.estimate_corr already are correlations) and
    F(∅) = 0.  Details the paper leaves open, fixed here:

      * The diagonal is set to s_jj = 1 exactly (a correlation's
        diagonal; their cosine similarity gives w_ii = 1 too): a selected
        benchmark is its own facility, contributing 1 to F.
      * The targets j run over all N benchmarks, selected ones included
        (Eq. (1)'s sum over V) and ineligible ones included: like Σ in
        the other selectors, eligibility restricts only the candidates.
      * No sparsification or threshold beyond the clipping at 0.
      * similarity="abs" uses s_ij = |R_ij| instead (sensitivity variant,
        not theirs); similarity="sq" uses s_ij = R_ij² (the variance of j
        explained by i alone; ablation variant, not theirs).  On a
        correlation matrix the "sq" step-1 gain 1 + Σ_{i≠j} R_ij² is
        greedy trace's first-step score, so the two differ in crediting
        each target with its best single proxy instead of the joint
        regression on A (and in their near-tie rules).
        facloc_similarity computes S.

    F is monotone submodular, so the greedy gains are non-increasing and
    the greedy selection is nested across k (selected[:k] is the size-k
    selection).  At step 1 the gain of j is 1 + Σ_{i≠j} max(R_ij, 0): with
    no negative correlations this is hub_select's score (N − 1)·mean|R_ij|
    plus 1, the same ranking.  Gains within TIE_RTOL · max(|best|, 1) of the
    best (_near_max with scale 1: sums of correlations, dimensionless) are
    tied and broken by explained variance of the unselected benchmarks in
    Sigma given the current selection, then by the lowest index
    (greedy_select._tie_break); without near-ties the pick is the plain
    argmax.  Unlike the variance-based scores, FL gains also tie exactly
    for generic Σ: two candidates i, j that add value only through their
    own terms and through covering each other both gain
    (1 − c_i) + (s_ij − c_j), c the current cover max_{a ∈ A} s_·a; the
    explained-variance rule decides such pairs (eval_baselines on MTEB,
    Merged and BenchPress: about one of the 15 picks per split, at most
    three; none on MMLU).  Once F saturates (every candidate's gain
    within tolerance of 0, e.g. duplicated columns already covered) every
    candidate ties and the explained-variance rule decides.
    Deterministic, O(k N²).

    Returns the dict of the other selectors (selected, log_dets,
    residual_variance, tie_breaks, index_ties, optional names) plus
    "facloc_values" (k,) F(A_t) after each step, "gains" (k,) the gain of
    each pick (NaN-padded beyond the last selection) and "similarity".
    """
    if similarity not in FACLOC_SIMILARITIES:
        raise ValueError(f"similarity {similarity!r} not in "
                         f"{FACLOC_SIMILARITIES}")
    N = Sigma.shape[0]
    assert Sigma.shape == (N, N)
    assert k >= 1
    el = _eligible(eligible, N)
    n_steps = min(k, N if el is None else int(el.sum()))

    S = facloc_similarity(Sigma, similarity)

    cover = np.zeros(N, dtype=np.float64)   # max_{i ∈ A} s_ij per target j
    unselected = np.ones(N, dtype=bool)
    selected, values, gains, tie_breaks, index_ties = [], [], [], [], []

    for t in range(n_steps):
        pool = np.flatnonzero(unselected if el is None
                              else unselected & el)
        # Gain of candidate i: sum_j max(s_ij − cover_j, 0) (S symmetric)
        gain = np.sum(np.maximum(S[:, pool] - cover[:, None], 0.0), axis=0)
        tied = _near_max(gain, 1.0)
        cand = pool[tied]
        if len(cand) > 1:
            L, d = _cholesky_state(Sigma, selected)
            j_star, n_cand, n_index = _tie_break(
                Sigma, L, d, t, cand, np.flatnonzero(unselected))
            tie_breaks.append((t, n_cand))
            if n_index > 1:
                index_ties.append((t, n_index))
        else:
            j_star = int(cand[0])
        gains.append(float(gain[np.searchsorted(pool, j_star)]))
        selected.append(j_star)
        unselected[j_star] = False
        cover = np.maximum(cover, S[:, j_star])
        values.append(float(np.sum(cover)))

    result = {
        "selected": selected,
        "facloc_values": _pad_nan(values, k),
        "gains": _pad_nan(gains, k),
        "similarity": similarity,
        "tie_breaks": tie_breaks,
        "index_ties": index_ties,
    }
    result.update(_selection_metrics(
        Sigma, [selected[:t] for t in range(1, n_steps + 1)], k))
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in selected]
    return result
