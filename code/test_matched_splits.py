"""Tests for the matched-split design and the shared selector API.

Covers cv_split (repeated CV, common-random-number seeds, eligibility,
imputation targets), the `eligible` argument of greedy_select and
baseline_select, the removal of entropy_rf (protocol rule D3), and the
near-tie handling of every selector (explained variance, then lowest
index).

The real-data check (TestMergedThreads) recomputes the rep-0 Merged
correlation estimates (EM, ~10 s per fold and thread setting on a 4090
host) under 1 and 16 BLAS threads in subprocesses; BENCHSELECT_SKIP_SLOW=1
skips it.  To compare hosts, run on each and diff the digests:

    python test_matched_splits.py --merged-digest [--threads 1]
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from cv_split import (
    ENTROPY_RF_SEED_OFFSET,
    MIN_TRAIN_OBS,
    N_REPEATS,
    RANDOM_SEED_OFFSET,
    Split,
    cv_folds,
    cv_splits,
    cv_train_val,
    eligible_mask,
    fold_seed,
    repeat_folds,
    split_seed,
)
from greedy_select import (
    TIE_RTOL,
    greedy_entropy,
    greedy_mi,
    index_tie_count,
    random_select,
    selection_at_k,
)
from baseline_select import greedy_trace_select, hub_select, pca_cssp_select

SEED = 42
CODE = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _factor_corr(N=30, n_factors=3, seed=0):
    """Random factor-model correlation matrix with an exactly unit diagonal."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, n_factors))
    S = W @ W.T + np.diag(rng.uniform(0.2, 1.0, N))
    s = 1.0 / np.sqrt(np.diag(S))
    R = S * np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return R


def _factor_cov(N=30, seed=0):
    """Factor-model covariance with distinct variances (no entropy ties)."""
    R = _factor_corr(N, seed=seed)
    sd = np.random.default_rng(1000 + seed).uniform(0.5, 3.0, N)
    return R * np.outer(sd, sd)


BLOCK = np.array([3, 7, 9, 12])   # exchangeable columns of _exch_corr


def _exch_corr(seed=0, N=14):
    """Correlation matrix with an exactly exchangeable block BLOCK.

    The block rows share one loading vector of squared norm 0.9 on the
    dominant factor (so every method picks among them early, and their
    within-block contrast eigenvalue 0.1 is the smallest); all other rows
    are generic.  Swapping two block indices leaves R bit-identical.
    """
    rng = np.random.default_rng(seed)
    W = rng.uniform(-0.25, 0.25, (N, 3))
    W[:, 0] = rng.uniform(0.3, 0.7, N)
    W *= np.sqrt(rng.uniform(0.3, 0.75, N) / np.sum(W ** 2, axis=1))[:, None]
    w = np.array([0.93, 0.15, -0.1])
    W[BLOCK] = w * np.sqrt(0.9 / np.sum(w ** 2))
    R = W @ W.T
    np.fill_diagonal(R, 1.0)
    return R


def _perturb_sym(R, rng, scale=1e-15):
    """R plus symmetric U(-scale, scale) noise on every entry."""
    E = rng.uniform(-scale, scale, R.shape)
    return R + (E + E.T) / 2


def _canon(sel, block=BLOCK):
    """Replace the i-th block member appearing in `sel` by block[i]
    (exchangeable columns are interchangeable)."""
    it = iter(sorted(int(b) for b in block))
    bset = set(int(b) for b in block)
    return [next(it) if int(j) in bset else int(j) for j in sel]


METHODS_DET = {
    "entropy": greedy_entropy,
    "mi": greedy_mi,
    "hub": hub_select,
    "pca": pca_cssp_select,
    "trace": greedy_trace_select,
}


def _all_selections(Sigma, k, eligible=None):
    """{method: selection} for the deterministic selectors (PCA: per k)."""
    out = {}
    for name, fn in METHODS_DET.items():
        res = fn(Sigma, k, eligible=eligible)
        out[name] = (res["selected_per_k"] if name == "pca"
                     else res["selected"])
    return out


# ---------------------------------------------------------------------------
# Frozen copies of the selectors before eligibility / decision-3 ties
# (greedy_entropy already had its explained-variance tie-break)
# ---------------------------------------------------------------------------

def _frozen_near_max(values, scale=1.0, rtol=TIE_RTOL):
    vmax = np.max(values)
    return values >= vmax - rtol * max(abs(vmax), scale)


def _frozen_explained_variance(Sigma, L, d, t, cand, unsel):
    r = Sigma[np.ix_(unsel, cand)] - L[unsel, :t] @ L[cand, :t].T
    return np.sum(r ** 2, axis=0) / np.maximum(d[cand], 1e-300)


def _frozen_entropy_pivot(Sigma, L, d, t, selected):
    N = len(d)
    sel_set = set(selected)
    unsel = np.array([j for j in range(N) if j not in sel_set])
    scale = float(np.max(np.diag(Sigma)))
    cand = unsel[_frozen_near_max(d[unsel], scale)]
    if len(cand) == 1:
        return int(cand[0]), 1, 1
    ev = _frozen_explained_variance(Sigma, L, d, t, cand, unsel)
    ev_tied = _frozen_near_max(ev, scale)
    return int(cand[np.argmax(ev_tied)]), len(cand), int(ev_tied.sum())


def _frozen_greedy_entropy(Sigma, k):
    N = Sigma.shape[0]
    total_var = np.trace(Sigma)
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, k), dtype=np.float64)
    selected, log_dets, residual_variance = [], [], []
    tie_breaks, index_ties = [], []
    cum_logdet = 0.0
    for t in range(k):
        j_star, n_cand, n_index = _frozen_entropy_pivot(Sigma, L, d, t,
                                                        selected)
        if n_cand > 1:
            tie_breaks.append((t, n_cand))
        if n_index > 1:
            index_ties.append((t, n_index))
        selected.append(j_star)
        cum_logdet += np.log(max(d[j_star], 1e-300))
        log_dets.append(cum_logdet)
        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2
        resid = sum(d[j] for j in range(N) if j not in selected)
        residual_variance.append(resid / total_var)
    return {"selected": selected, "d": d, "log_dets": np.array(log_dets),
            "residual_variance": np.array(residual_variance),
            "tie_breaks": tie_breaks, "index_ties": index_ties}


def _frozen_complement_precision_diag(Sigma, comp):
    from scipy.linalg import solve_triangular
    m = len(comp)
    Sigma_sub = Sigma[np.ix_(comp, comp)]
    try:
        L = np.linalg.cholesky(Sigma_sub)
        L_inv = solve_triangular(L, np.eye(m), lower=True)
        return np.sum(L_inv ** 2, axis=0)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma_sub)
        eigvals = np.maximum(eigvals, 1e-10)
        return np.sum(eigvecs ** 2 / eigvals[None, :], axis=1)


def _frozen_greedy_mi(Sigma, k):
    N = Sigma.shape[0]
    total_var = np.trace(Sigma)
    sign_full, logdet_full = np.linalg.slogdet(Sigma)
    if sign_full <= 0:
        eigvals_full = np.linalg.eigvalsh(Sigma)
        logdet_full = np.sum(np.log(np.maximum(eigvals_full, 1e-300)))
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, k), dtype=np.float64)
    selected, selected_set = [], set()
    log_dets, residual_variance, gains = [], [], []
    cum_logdet = 0.0
    for t in range(k):
        comp = np.array(sorted(set(range(N)) - selected_set))
        P_diag_comp = _frozen_complement_precision_diag(Sigma, comp)
        P_full = np.zeros(N)
        for i, idx in enumerate(comp):
            P_full[idx] = P_diag_comp[i]
        best_j, best_score = int(comp[0]), -np.inf
        for j in comp:
            dj, pj = d[j], P_full[j]
            if dj > 1e-300 and pj > 1e-300:
                score = np.log(dj) + np.log(pj)
            else:
                score = -np.inf
            if score > best_score:
                best_score, best_j = score, int(j)
        j_star = best_j
        selected.append(j_star)
        selected_set.add(j_star)
        gains.append(0.5 * best_score)
        cum_logdet += np.log(max(d[j_star], 1e-300))
        log_dets.append(cum_logdet)
        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected_set and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2
        resid = sum(d[j] for j in range(N) if j not in selected_set)
        residual_variance.append(resid / total_var)
    mi_values = []
    for kk in range(1, k + 1):
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
    return {"selected": selected, "d": d, "log_dets": np.array(log_dets),
            "residual_variance": np.array(residual_variance),
            "mi_values": np.array(mi_values), "gains": np.array(gains)}


def _frozen_random_select(Sigma, k, seed=0):
    N = Sigma.shape[0]
    rng = np.random.default_rng(seed)
    total_var = np.trace(Sigma)
    perm = rng.permutation(N)
    selected = perm[:k].tolist()
    log_dets, residual_variance = [], []
    RIDGE = 1e-8
    sel_set = set()
    for t in range(1, k + 1):
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
    return {"selected": selected, "log_dets": np.array(log_dets),
            "residual_variance": np.array(residual_variance)}


def _frozen_selection_metrics(Sigma, selections):
    N = Sigma.shape[0]
    total_var = np.trace(Sigma)
    log_dets, residual_variance = [], []
    RIDGE = 1e-8
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
    return {"log_dets": np.array(log_dets),
            "residual_variance": np.array(residual_variance)}


def _frozen_hub_select(Sigma, k):
    N = Sigma.shape[0]
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    abs_R = np.abs(R)
    np.fill_diagonal(abs_R, 0.0)
    scores = abs_R.sum(axis=1) / (N - 1)
    selected = np.argsort(-scores)[:k].tolist()
    result = {"selected": selected, "hub_scores": scores}
    result.update(_frozen_selection_metrics(
        Sigma, [selected[:t] for t in range(1, k + 1)]))
    return result


def _frozen_pca_cssp_select(Sigma, k):
    from scipy.linalg import qr
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    selected_per_k = []
    for kk in range(1, k + 1):
        _, _, piv = qr(eigvecs[:, :kk].T, pivoting=True, mode="economic")
        selected_per_k.append([int(j) for j in piv[:kk]])
    result = {"selected": selected_per_k[-1], "selected_per_k": selected_per_k}
    result.update(_frozen_selection_metrics(Sigma, selected_per_k))
    return result


def _frozen_greedy_trace_select(Sigma, k):
    N = Sigma.shape[0]
    total_var = np.trace(Sigma)
    S = Sigma.astype(np.float64).copy()
    unselected = np.ones(N, dtype=bool)
    selected, log_dets, residual_variance = [], [], []
    cum_logdet = 0.0
    for _ in range(k):
        cand = np.where(unselected)[0]
        d = np.diag(S)[cand]
        S_sub = S[np.ix_(cand, cand)]
        col_sq = np.sum(S_sub ** 2, axis=0)
        safe_d = np.maximum(d, 1e-12)
        reduction = col_sq / safe_d
        reduction[d <= 1e-12] = -np.inf
        j_star = int(cand[int(np.argmax(reduction))])
        selected.append(j_star)
        unselected[j_star] = False
        d_star = max(S[j_star, j_star], 1e-300)
        cum_logdet += np.log(d_star)
        log_dets.append(cum_logdet)
        s_col = S[:, j_star].copy()
        S -= np.outer(s_col, s_col) / d_star
        S[j_star, :] = 0.0
        S[:, j_star] = 0.0
        resid = np.sum(np.maximum(np.diag(S)[unselected], 0.0))
        residual_variance.append(resid / total_var)
    return {"selected": selected, "log_dets": np.array(log_dets),
            "residual_variance": np.array(residual_variance)}


FROZEN = {
    "entropy": _frozen_greedy_entropy,
    "mi": _frozen_greedy_mi,
    "hub": _frozen_hub_select,
    "pca": _frozen_pca_cssp_select,
    "trace": _frozen_greedy_trace_select,
}


def _assert_same_result(new, old, name):
    for key, val in old.items():
        if isinstance(val, np.ndarray):
            assert np.array_equal(new[key], val), f"{name}: {key}"
        else:
            assert new[key] == val, f"{name}: {key}"


# ===================================================================
# cv_split: matched repeated-CV splits
# ===================================================================

MS = [7, 23, 50, 83, 118, 263]
PS = [0.1, 0.15, 0.2, 0.29, 0.3, 0.5, 0.7, 0.9]


class TestSplits:
    @pytest.mark.parametrize("M", MS)
    @pytest.mark.parametrize("p", PS)
    def test_rep0_equals_single_repetition(self, M, p):
        """Rep 0 reproduces cv_folds(M, 10, 42) + cv_train_val exactly."""
        folds = cv_folds(M, 10, SEED)
        splits = [s for s in cv_splits(M, 10, N_REPEATS, p, SEED) if s.rep == 0]
        assert len(splits) == 10
        for s in splits:
            tr, va = cv_train_val(folds, s.fold, p, M, SEED)
            assert np.array_equal(s.train_idx, tr)
            assert np.array_equal(s.val_idx, va)
            assert s.split_id == s.fold

    @pytest.mark.parametrize("M", [23, 118])
    @pytest.mark.parametrize("p", [0.1, 0.3, 0.9])
    def test_rep0_matches_literal_old_code(self, M, p):
        """Rep 0 against the original seeds written out literally."""
        perm = np.random.default_rng(42).permutation(M)
        folds = np.array_split(perm, 10)
        for s in cv_splits(M, 10, 1, p, SEED):
            pool = np.concatenate([folds[j] for j in range(10) if j != s.fold])
            n_train = min(round((1 - p) * M), len(pool))
            if n_train >= len(pool):
                tr = pool
            else:
                rng = np.random.default_rng((42, s.fold, int(p * 10000)))
                tr = rng.permutation(pool)[:n_train]
            assert np.array_equal(s.train_idx, tr)
            assert np.array_equal(s.val_idx, folds[s.fold])

    @pytest.mark.parametrize("M", [23, 118])
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_rep_r_seeds(self, M, p):
        """Rep r > 0: folds from (42, r), subsample from (42, r, fold, p)."""
        for s in cv_splits(M, 10, N_REPEATS, p, SEED):
            if s.rep == 0:
                continue
            perm = np.random.default_rng((42, s.rep)).permutation(M)
            folds = np.array_split(perm, 10)
            assert np.array_equal(s.val_idx, folds[s.fold])
            pool = np.concatenate([folds[j] for j in range(10) if j != s.fold])
            n_train = min(round((1 - p) * M), len(pool))
            if n_train < len(pool):
                rng = np.random.default_rng((42, s.rep, s.fold,
                                             int(p * 10000)))
                assert np.array_equal(s.train_idx,
                                      rng.permutation(pool)[:n_train])
            else:
                assert np.array_equal(s.train_idx, pool)

    @pytest.mark.parametrize("M", MS)
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_balanced_and_disjoint(self, M, p):
        splits = list(cv_splits(M, 10, N_REPEATS, p, SEED))
        assert len(splits) == 10 * N_REPEATS
        assert [s.split_id for s in splits] == list(range(10 * N_REPEATS))
        val_count = np.zeros(M, dtype=int)
        for rep in range(N_REPEATS):
            rs = [s for s in splits if s.rep == rep]
            assert [s.fold for s in rs] == list(range(10))
            vals = np.concatenate([s.val_idx for s in rs])
            assert np.array_equal(np.sort(vals), np.arange(M))
            sizes = [len(s.val_idx) for s in rs]
            assert max(sizes) - min(sizes) <= 1
            for s in rs:
                assert not set(s.train_idx) & set(s.val_idx)
                assert len(set(s.train_idx)) == len(s.train_idx)
                assert len(s.train_idx) == min(round((1 - p) * M),
                                               M - len(s.val_idx))
                val_count[s.val_idx] += 1
        assert np.all(val_count == N_REPEATS)

    @pytest.mark.parametrize("M", [50, 118, 263])
    def test_repeats_differ(self, M):
        """Different repetitions partition and subsample differently."""
        parts = []
        for rep in range(N_REPEATS):
            folds = repeat_folds(M, 10, rep, SEED)
            parts.append(frozenset(frozenset(f.tolist()) for f in folds))
        assert len(set(parts)) == N_REPEATS
        trains = {}
        for s in cv_splits(M, 10, N_REPEATS, 0.5, SEED):
            trains.setdefault(s.fold, set()).add(
                tuple(np.sort(s.train_idx).tolist()))
        assert all(len(v) == N_REPEATS for v in trains.values())

    def test_reps_subset_and_split_ids(self):
        full = list(cv_splits(83, 10, N_REPEATS, 0.3, SEED))
        sub = list(cv_splits(83, 10, N_REPEATS, 0.3, SEED, reps=[0, 3]))
        assert [s.split_id for s in sub] == list(range(10)) + list(range(30, 40))
        for s in sub:
            f = full[s.split_id]
            assert (f.rep, f.fold) == (s.rep, s.fold)
            assert np.array_equal(f.train_idx, s.train_idx)
            assert np.array_equal(f.val_idx, s.val_idx)
        with pytest.raises(ValueError):
            list(cv_splits(83, 10, N_REPEATS, 0.3, SEED, reps=[N_REPEATS]))

    def test_iteration_order_independent(self):
        """Splits do not depend on how many others were drawn before."""
        a = list(cv_splits(118, 10, N_REPEATS, 0.7, SEED))[37]
        b = list(cv_splits(118, 10, N_REPEATS, 0.7, SEED, reps=[3]))[7]
        assert (a.rep, a.fold, a.split_id) == (b.rep, b.fold, b.split_id)
        assert np.array_equal(a.train_idx, b.train_idx)

    @pytest.mark.parametrize("p", PS)
    def test_split_seed(self, p):
        tag = SEED + RANDOM_SEED_OFFSET
        assert tag == 49 and SEED + ENTROPY_RF_SEED_OFFSET == 53
        # rep 0: the random baseline's existing seed (SEED + 7, fold, p)
        assert split_seed(tag, 0, 4, p) == (SEED + 7, 4, int(p * 10000))
        assert split_seed(tag, 2, 4, p) == (SEED + 7, 2, 4, int(p * 10000))
        assert split_seed(SEED, 0, 4, p) == (42, 4, int(p * 10000))
        s = Split(3, 4, 34, np.arange(3), np.arange(3, 5), p)
        assert s.seed(tag) == (49, 3, 4, int(p * 10000))
        assert all(isinstance(x, int) for x in s.seed(tag))

    def test_split_seed_truncates_like_old_code(self):
        # 0.57 * 10000 = 5699.999999999999: int() truncates to 5699
        assert split_seed(49, 0, 1, 0.57)[2] == int(0.57 * 10000) == 5699
        assert split_seed(49, 2, 1, 0.69)[3] == 6899

    def test_fold_seed(self):
        assert fold_seed(0, 42) == 42
        assert fold_seed(3, 42) == (42, 3)
        assert all(np.array_equal(a, b) for a, b in
                   zip(repeat_folds(77, 10, 0, 42), cv_folds(77, 10, 42)))

    def test_cv_train_val_rep_default(self):
        folds = cv_folds(83, 10, SEED)
        a = cv_train_val(folds, 2, 0.6, 83, SEED)
        b = cv_train_val(folds, 2, 0.6, 83, SEED, rep=0)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        c = cv_train_val(folds, 2, 0.6, 83, SEED, rep=1)
        assert not np.array_equal(a[0], c[0])


class TestEligibleMask:
    def test_threshold(self):
        O = np.zeros((6, 5))
        for j, n in enumerate([0, 2, 3, 4, 6]):
            O[:n, j] = 1.0
        assert MIN_TRAIN_OBS == 3
        assert eligible_mask(O).tolist() == [False, False, True, True, True]
        assert eligible_mask(O, min_obs=1).tolist() == [False, True, True,
                                                         True, True]
        assert eligible_mask(O.astype(bool)).dtype == bool
        assert eligible_mask(O.astype(np.int8)).tolist() == eligible_mask(O).tolist()

    def test_uses_training_rows_only(self):
        rng = np.random.default_rng(0)
        O = (rng.uniform(size=(40, 12)) < 0.1).astype(float)
        for s in cv_splits(40, 10, 2, 0.5, SEED):
            el = eligible_mask(O[s.train_idx])
            assert np.array_equal(el, O[s.train_idx].sum(axis=0) >= 3)


# ===================================================================
# Bit-identity with the previous selectors on inputs without near-ties
# ===================================================================

def _generic_inputs():
    out = []
    for seed in range(6):
        out.append(("corr", seed, _factor_corr(N=24 + 3 * seed, seed=seed)))
        out.append(("cov", seed, _factor_cov(N=20 + 2 * seed, seed=seed)))
    return out


GENERIC = _generic_inputs()


class TestBitIdentity:
    @pytest.mark.parametrize("kind,seed,Sigma", GENERIC,
                             ids=[f"{k}{s}" for k, s, _ in GENERIC])
    @pytest.mark.parametrize("name", list(FROZEN))
    def test_matches_frozen(self, name, kind, seed, Sigma):
        N = Sigma.shape[0]
        k = min(15, N) if name == "pca" else N
        old = FROZEN[name](Sigma, k)
        for el in [None, np.ones(N, dtype=bool)]:
            new = METHODS_DET[name](Sigma, k, eligible=el)
            _assert_same_result(new, old, name)
            assert new["index_ties"] == []
            if name not in ("entropy",):
                assert new["tie_breaks"] == []

    @pytest.mark.parametrize("seed", range(4))
    def test_random_matches_frozen(self, seed):
        Sigma = _factor_corr(N=30, seed=seed)
        for rep in range(3):
            rs = split_seed(SEED + 7, rep, seed, 0.1)
            old = _frozen_random_select(Sigma, 15, seed=rs)
            for el in [None, np.ones(30, dtype=bool)]:
                new = random_select(Sigma, 15, seed=rs, eligible=el)
                _assert_same_result(new, old, "random")

    def test_greedy_mi_k_full(self):
        R = _factor_corr(N=25, seed=11)
        _assert_same_result(greedy_mi(R, 25), _frozen_greedy_mi(R, 25), "mi")


# ===================================================================
# Eligibility
# ===================================================================

def _cond_cov(Sigma, sel):
    if not sel:
        return Sigma.copy()
    S = np.array(sel)
    return Sigma - Sigma[:, S] @ np.linalg.solve(Sigma[np.ix_(S, S)],
                                                 Sigma[S, :])


def _mask(N, seed, frac=0.6):
    rng = np.random.default_rng(seed)
    el = rng.uniform(size=N) < frac
    el[rng.integers(N)] = True
    return el


ELIG_CASES = [(kind, seed) for kind in ["corr", "cov"] for seed in range(4)]


class TestEligibility:
    def _sigma(self, kind, seed):
        return (_factor_corr(N=26, seed=seed) if kind == "corr"
                else _factor_cov(N=26, seed=seed))

    @pytest.mark.parametrize("kind,seed", ELIG_CASES)
    def test_selected_subset_of_eligible(self, kind, seed):
        Sigma = self._sigma(kind, seed)
        N = Sigma.shape[0]
        el = _mask(N, 50 + seed)
        k = 10
        n_el = int(el.sum())
        runs = {name: fn(Sigma, k, eligible=el)
                for name, fn in METHODS_DET.items()}
        runs["random"] = random_select(Sigma, k, seed=(49, seed), eligible=el)
        for name, res in runs.items():
            sels = res.get("selected_per_k", [res["selected"]])
            for sel in sels:
                assert all(el[j] for j in sel), name
                assert len(set(sel)) == len(sel), name
            assert len(res["selected"]) == min(k, n_el), name

    @pytest.mark.parametrize("kind,seed", ELIG_CASES)
    def test_entropy_optimal_among_eligible(self, kind, seed):
        Sigma = self._sigma(kind, seed)
        el = _mask(Sigma.shape[0], 60 + seed)
        sel = greedy_entropy(Sigma, 8, eligible=el)["selected"]
        scale = np.max(np.diag(Sigma))
        for t, j in enumerate(sel):
            dv = np.diag(_cond_cov(Sigma, sel[:t]))
            pool = [i for i in np.flatnonzero(el) if i not in sel[:t]]
            assert dv[j] >= max(dv[pool]) - 1e-9 * scale

    @pytest.mark.parametrize("kind,seed", ELIG_CASES)
    def test_mi_optimal_among_eligible(self, kind, seed):
        """Each pick maximizes ½[log σ²_v|S + log P_vv] over eligible v,
        with P from the FULL complement (ineligible benchmarks included)."""
        Sigma = self._sigma(kind, seed)
        N = Sigma.shape[0]
        el = _mask(N, 70 + seed)
        sel = greedy_mi(Sigma, 8, eligible=el)["selected"]
        for t, j in enumerate(sel):
            dv = np.diag(_cond_cov(Sigma, sel[:t]))
            comp = [i for i in range(N) if i not in sel[:t]]
            P = np.diag(np.linalg.inv(Sigma[np.ix_(comp, comp)]))
            gain = {v: 0.5 * (np.log(dv[v]) + np.log(P[i]))
                    for i, v in enumerate(comp)}
            best = max(gain[v] for v in comp if el[v])
            assert gain[j] >= best - 1e-9

    @pytest.mark.parametrize("kind,seed", ELIG_CASES)
    def test_trace_optimal_among_eligible(self, kind, seed):
        Sigma = self._sigma(kind, seed)
        N = Sigma.shape[0]
        el = _mask(N, 80 + seed)
        res = greedy_trace_select(Sigma, 8, eligible=el)
        sel = res["selected"]
        for t, j in enumerate(sel):
            base = np.trace(_cond_cov(Sigma, sel[:t]))
            red = {v: base - np.trace(_cond_cov(Sigma, sel[:t] + [v]))
                   for v in np.flatnonzero(el) if v not in sel[:t]}
            assert red[j] >= max(red.values()) - 1e-9 * np.max(np.diag(Sigma))
        # residual variance counts the ineligible benchmarks too
        expected = []
        for t in range(1, len(sel) + 1):
            C = _cond_cov(Sigma, sel[:t])
            unsel = [i for i in range(N) if i not in sel[:t]]
            expected.append(np.trace(C[np.ix_(unsel, unsel)]) / np.trace(Sigma))
        np.testing.assert_allclose(res["residual_variance"], expected,
                                   rtol=1e-8, atol=1e-12)

    @pytest.mark.parametrize("kind,seed", ELIG_CASES)
    def test_hub_is_top_eligible_scores(self, kind, seed):
        Sigma = self._sigma(kind, seed)
        el = _mask(Sigma.shape[0], 90 + seed)
        res = hub_select(Sigma, 8, eligible=el)
        pool = np.flatnonzero(el)
        expected = pool[np.argsort(-res["hub_scores"][pool])][:8].tolist()
        assert res["selected"] == expected
        # scores still use all benchmarks
        assert np.array_equal(res["hub_scores"],
                              hub_select(Sigma, 8)["hub_scores"])

    @pytest.mark.parametrize("kind,seed", ELIG_CASES)
    def test_pca_is_qr_on_eligible_columns(self, kind, seed):
        """CSSP over eligible columns == pivoted QR of the eligible
        columns of the full-Σ loadings."""
        from scipy.linalg import qr
        Sigma = self._sigma(kind, seed)
        el = _mask(Sigma.shape[0], 100 + seed)
        pool = np.flatnonzero(el)
        res = pca_cssp_select(Sigma, 8, eligible=el)
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]
        for kk in range(1, min(8, len(pool)) + 1):
            _, _, piv = qr(eigvecs[:, :kk][pool].T, pivoting=True,
                           mode="economic")
            assert res["selected_per_k"][kk - 1] == pool[piv[:kk]].tolist()

    @pytest.mark.parametrize("kind,seed", ELIG_CASES)
    def test_random_is_restricted_permutation(self, kind, seed):
        """Random order = the seed's full permutation restricted to the
        eligible set (common random numbers across masks)."""
        Sigma = self._sigma(kind, seed)
        N = Sigma.shape[0]
        el = _mask(N, 110 + seed)
        rs = split_seed(49, 1, seed, 0.3)
        perm = np.random.default_rng(rs).permutation(N)
        res = random_select(Sigma, 6, seed=rs, eligible=el)
        assert res["selected"] == [int(j) for j in perm if el[j]][:6]

    def test_fewer_eligible_than_k(self):
        Sigma = _factor_cov(N=20, seed=3)
        el = np.zeros(20, dtype=bool)
        el[[2, 5, 11, 17]] = True
        k = 7
        runs = {name: fn(Sigma, k, eligible=el)
                for name, fn in METHODS_DET.items()}
        runs["random"] = random_select(Sigma, k, seed=1, eligible=el)
        for name, res in runs.items():
            assert sorted(res["selected"]) == [2, 5, 11, 17], name
            for key in ["log_dets", "residual_variance"]:
                arr = res[key]
                assert arr.shape == (k,), (name, key)
                assert np.all(np.isfinite(arr[:4])), (name, key)
                assert np.all(np.isnan(arr[4:])), (name, key)
            assert selection_at_k(res, 4) is not None, name
            assert selection_at_k(res, 5) is None, name
        assert len(runs["pca"]["selected_per_k"]) == 4
        for key in ["mi_values", "gains"]:
            assert np.all(np.isnan(runs["mi"][key][4:]))

    def test_no_eligible(self):
        Sigma = _factor_corr(N=10, seed=1)
        el = np.zeros(10, dtype=bool)
        for name, fn in METHODS_DET.items():
            res = fn(Sigma, 3, eligible=el)
            assert res["selected"] == [], name
            assert np.all(np.isnan(res["residual_variance"])), name
        assert random_select(Sigma, 3, eligible=el)["selected"] == []

    def test_bad_mask(self):
        Sigma = _factor_corr(N=10, seed=1)
        with pytest.raises(ValueError):
            greedy_mi(Sigma, 3, eligible=np.ones(9, dtype=bool))

    def test_ineligible_changes_objective_not_candidates(self):
        """Removing an ineligible benchmark from Σ changes MI picks: it
        stays in the complement, it is not just dropped."""
        changed = 0
        for seed in range(10):
            Sigma = _factor_corr(N=16, seed=seed)
            el = np.ones(16, dtype=bool)
            el[:4] = False
            sel = greedy_mi(Sigma, 6, eligible=el)["selected"]
            keep = np.arange(4, 16)
            sub = greedy_mi(Sigma[np.ix_(keep, keep)], 6)["selected"]
            changed += sel != [int(keep[j]) for j in sub]
        assert changed > 0


# ===================================================================
# Protocol rules D1 (targets) and D3 (entropy_rf retired)
# ===================================================================

class TestTargetsAndRetiredEntropyRF:
    def test_target_mask(self):
        import cv_split
        O = np.zeros((6, 5))
        O[:1, 1] = 1                      # 1 training observation
        O[:2, 2] = 1                      # 2
        O[:3, 3] = 1                      # 3
        O[:, 4] = 2.0                     # any nonzero entry counts
        assert cv_split.MIN_TARGET_OBS == 2 <= MIN_TRAIN_OBS
        assert cv_split.target_mask(O).tolist() == [False, False, True, True,
                                                    True]
        assert cv_split.target_mask(O, 0).all()
        # non-targets are never eligible
        assert not np.any(eligible_mask(O) & ~cv_split.target_mask(O))

    def test_target_mask_reads_constant_at_call_time(self, monkeypatch):
        """The test-only override MIN_TARGET_OBS = 0 makes every column a
        target (the cells before rule D1)."""
        import cv_split
        O = np.zeros((4, 3))
        O[:1, 0] = 1
        monkeypatch.setattr(cv_split, "MIN_TARGET_OBS", 0)
        assert cv_split.target_mask(O).all()

    def test_entropy_rf_removed(self):
        import cv_split
        import greedy_select
        assert not hasattr(greedy_select, "greedy_entropy_rf")
        assert not hasattr(greedy_select, "random_first_pivot")
        # the seed tag stays reserved, documented as retired
        assert SEED + ENTROPY_RF_SEED_OFFSET == 53
        assert "retired" in cv_split.SEED_TAGS[11]
        assert "retired" in cv_split.SEED_TAGS[14]
        with pytest.raises(TypeError):
            greedy_entropy(_factor_corr(N=8, seed=0), 3, first=2)

    def test_no_entropy_rf_method_anywhere(self):
        """No script runs, logs, plots or names entropy_rf any more: no
        method list carries it and no selector call remains (rule D3)."""
        import re
        for path in sorted(CODE.glob("*.py")) + sorted(CODE.glob("*.sh")):
            if path.name.startswith("test_"):
                continue
            src = path.read_text()
            assert not re.search(r"greedy_entropy_rf\(|random_first_pivot\(",
                                 src), path.name
            assert not re.search(r"[\"']entropy_rf[\"']\s*[:,\]]", src), \
                path.name
            assert "rf_first" not in src, path.name


# ===================================================================
# Decision-3 tie handling: exchangeable block
# ===================================================================

K_EXCH = 6   # ≤ N − |BLOCK|: PCA never splits the block's eigenspace


class TestExchangeableBlock:
    @pytest.mark.parametrize("seed", range(3))
    def test_block_is_exchangeable(self, seed):
        R = _exch_corr(seed)
        for a in BLOCK:
            for b in BLOCK:
                p = np.arange(R.shape[0])
                p[[a, b]] = p[[b, a]]
                assert np.array_equal(R[np.ix_(p, p)], R)
        ev = np.sort(np.linalg.eigvalsh(R))
        assert np.allclose(ev[:3], 0.1) and ev[3] > 0.1 + 1e-3

    @pytest.mark.parametrize("seed", range(3))
    def test_ties_hit_block_and_lowest_index(self, seed):
        R = _exch_corr(seed)
        b0 = int(BLOCK[0])
        sels = _all_selections(R, K_EXCH)
        for name in ["mi", "hub", "trace", "entropy"]:
            res = METHODS_DET[name](R, K_EXCH)
            assert res["index_ties"], name
            block_picks = [j for j in sels[name] if j in set(BLOCK)]
            assert block_picks == sorted(block_picks), name
            assert block_picks and block_picks[0] == b0, name
        pca = pca_cssp_select(R, K_EXCH)
        assert pca["index_ties"]
        assert all(kk >= 1 and n >= 2 for kk, _, n in pca["index_ties"])
        assert index_tie_count(pca) == len(pca["index_ties"])

    @pytest.mark.parametrize("seed", range(3))
    def test_invariant_to_1e15_perturbations(self, seed):
        R = _exch_corr(seed)
        ref = _all_selections(R, K_EXCH)
        rng = np.random.default_rng(700 + seed)
        frozen_seen = {name: set() for name in ["mi", "hub", "trace", "pca"]}
        for _ in range(30):
            Rp = _perturb_sym(R, rng)
            assert _all_selections(Rp, K_EXCH) == ref
            for name in frozen_seen:
                res = FROZEN[name](Rp, K_EXCH)
                sel = res.get("selected_per_k", res["selected"])
                frozen_seen[name].add(json.dumps(sel))
        # Control: rounding-level noise moved the previous selectors
        moved = [n for n, s in frozen_seen.items() if len(s) > 1]
        assert "mi" in moved and "hub" in moved and "trace" in moved, moved

    @pytest.mark.parametrize("seed", range(3))
    def test_permutation_equivariant(self, seed):
        """Permuted inputs give the permuted selection, up to relabeling
        within the exchangeable block; exactly when the permutation keeps
        the block's index order."""
        R = _exch_corr(seed)
        N = R.shape[0]
        ref = _all_selections(R, K_EXCH)
        rng = np.random.default_rng(800 + seed)
        for trial in range(8):
            p = rng.permutation(N)
            if trial % 2 == 0:   # keep block members in index order
                pos = np.sort(np.flatnonzero(np.isin(p, BLOCK)))
                p[pos] = BLOCK
            sels = _all_selections(R[np.ix_(p, p)], K_EXCH)
            for name, sel in sels.items():
                seqs = sel if name == "pca" else [sel]
                refs = ref[name] if name == "pca" else [ref[name]]
                for s_new, s_ref in zip(seqs, refs):
                    mapped = [int(p[j]) for j in s_new]
                    if trial % 2 == 0:
                        assert mapped == s_ref, name
                    else:
                        assert _canon(mapped) == _canon(s_ref), name

    @pytest.mark.parametrize("seed", range(3))
    def test_permutation_equivariant_generic(self, seed):
        """No exchangeable columns: exact permutation equivariance."""
        R = _factor_corr(N=20, seed=seed)
        ref = _all_selections(R, 8)
        rng = np.random.default_rng(900 + seed)
        for _ in range(5):
            p = rng.permutation(20)
            sels = _all_selections(R[np.ix_(p, p)], 8)
            for name, sel in sels.items():
                seqs = sel if name == "pca" else [sel]
                refs = ref[name] if name == "pca" else [ref[name]]
                for s_new, s_ref in zip(seqs, refs):
                    assert [int(p[j]) for j in s_new] == s_ref, name

    @pytest.mark.parametrize("seed", range(2))
    def test_eligibility_with_block(self, seed):
        """Masking the lowest block member moves the pick to the next."""
        R = _exch_corr(seed)
        el = np.ones(R.shape[0], dtype=bool)
        el[BLOCK[0]] = False
        for name, fn in METHODS_DET.items():
            res = fn(R, K_EXCH, eligible=el)
            sel = res["selected"]
            assert BLOCK[0] not in sel, name
            if name != "pca":
                bp = [j for j in sel if j in set(BLOCK)]
                assert bp == sorted(bp), name

    def test_mi_tie_counts(self):
        R = _exch_corr(0)
        res = greedy_mi(R, K_EXCH)
        n_cand = dict(res["tie_breaks"])
        for step, n in res["index_ties"]:
            assert 2 <= n <= n_cand[step]
        assert index_tie_count(res) == len(res["index_ties"])
        assert index_tie_count(res, 1) == sum(s < 1 for s, _ in res["index_ties"])


# ===================================================================
# PCA/CSSP basis: repeated eigenvalues, directions without eligible loading
# ===================================================================

def _with_null_block(A, m):
    """blockdiag(A, I_m) and the mask selecting A's rows: I_m is EM's
    estimate for benchmarks seen by only one or two training models
    (ineligible, uncorrelated with everything, eigenvalue exactly 1)."""
    n_a = A.shape[0]
    S = np.eye(n_a + m)
    S[:n_a, :n_a] = A
    el = np.zeros(n_a + m, dtype=bool)
    el[:n_a] = True
    return S, el


def _spectrum_cov(eigvals, seed):
    """Q diag(eigvals) Q^T for a random orthogonal Q (exact repeats in
    eigvals give repeated eigenvalues up to rounding)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((len(eigvals), len(eigvals))))
    S = (Q * np.asarray(eigvals, dtype=float)) @ Q.T
    return (S + S.T) / 2


def _plain_pca_picks(Sigma, kk):
    """geqp3 pivots on eigh's top-kk eigenvectors (the selection before
    _pca_basis), all benchmarks eligible."""
    from scipy.linalg import qr
    w, V = np.linalg.eigh(Sigma)
    V = V[:, np.argsort(-w)]
    _, _, piv = qr(V[:, :kk].T, pivoting=True, mode="economic")
    return [int(j) for j in piv[:kk]]


class TestPCABasis:
    @pytest.mark.parametrize("seed", range(4))
    def test_generic_basis_is_eighs(self, seed):
        R = _factor_corr(N=26, seed=seed)
        for el in [None, np.ones(26, dtype=bool), _mask(26, seed)]:
            res = pca_cssp_select(R, 12, eligible=el)
            n = len(res["selected_per_k"])
            assert res["n_eigvecs"] == list(range(1, n + 1))
            assert res["eig_skipped"] == [0] * n

    @pytest.mark.parametrize("seed", range(4))
    def test_null_block_is_skipped(self, seed):
        """Eigenvalue-1 directions of an ineligible uncorrelated block rank
        inside the top k; PCA on the full Σ must equal PCA on the eligible
        block alone, with no tie-decided picks."""
        A = _factor_corr(N=22, seed=seed)
        m, k = 6, 15
        S, el = _with_null_block(A, m)
        w = np.sort(np.linalg.eigvalsh(A))[::-1]
        assert np.sum(w > 1.0) < k        # the block ranks inside the top k
        res = pca_cssp_select(S, k, eligible=el)
        ref = pca_cssp_select(A, k)
        assert res["selected_per_k"] == ref["selected_per_k"]
        assert res["tie_breaks"] == [] and res["index_ties"] == []
        assert res["n_eigvecs"] == list(range(1, k + 1))
        n_above = int(np.sum(w > 1.0))
        assert res["eig_skipped"] == [0] * n_above + [m] * (k - n_above)

    @pytest.mark.parametrize("seed", range(3))
    def test_visible_direction_inside_null_cluster(self, seed):
        """The eligible block has an eigenvalue of exactly 1 too: the
        eigenvalue-1 eigenspace mixes it with the null block (eigh's
        basis is arbitrary there); the SVD rotation recovers the one
        direction with eligible loading."""
        lam = [6.0, 4.0, 2.5, 1.7, 1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12,
               0.08]
        A = _spectrum_cov(lam, 40 + seed)
        S, el = _with_null_block(A, 5)
        res = pca_cssp_select(S, 10, eligible=el)
        ref = pca_cssp_select(A, 10)
        assert res["selected_per_k"] == ref["selected_per_k"]
        assert res["tie_breaks"] == [] and res["index_ties"] == []
        assert res["eig_skipped"] == [0] * 5 + [5] * 5

    @pytest.mark.parametrize("seed", range(3))
    def test_repeated_eigenvalue_takes_whole_eigenspace(self, seed):
        """kk = 3 cuts through a repeated eigenvalue (positions 3-4): the
        size-3 subspace is the whole eigenspace, so the picks no longer
        depend on eigh's (rounding-dependent) basis of it."""
        lam = [5.0, 3.0, 2.0, 2.0, 1.5, 1.1, 0.9, 0.7, 0.5, 0.35, 0.25,
               0.15, 0.1, 0.05]
        S = _spectrum_cov(lam, 60 + seed)
        res = pca_cssp_select(S, 6)
        assert res["n_eigvecs"] == [1, 2, 4, 4, 5, 6]
        assert res["eig_skipped"] == [0] * 6
        rng = np.random.default_rng(70 + seed)
        plain = set()
        for _ in range(20):
            Sp = _perturb_sym(S, rng, 1e-13)
            assert pca_cssp_select(Sp, 6)["selected_per_k"] == \
                res["selected_per_k"]
            plain.add(tuple(_plain_pca_picks(Sp, 3)))
        # Control: eigh's top-3 basis rotates inside the eigenspace
        assert len(plain) > 1

    def test_no_eligible(self):
        R = _factor_corr(N=10, seed=1)
        res = pca_cssp_select(R, 5, eligible=np.zeros(10, dtype=bool))
        assert res["selected_per_k"] == [] and res["n_eigvecs"] == []


# ===================================================================
# Paired statistics (paired_stats.py, the one shared implementation)
# ===================================================================

class TestPairedStats:
    def test_nadeau_bengio_by_hand(self):
        from scipy import stats
        from paired_stats import paired_stats
        a = np.array([0.50, 0.42, 0.61, 0.55, np.nan, 0.47, 0.52])
        b = np.array([0.45, 0.44, 0.58, 0.55, 0.40, 0.41, np.inf])
        n_train = np.array([90, 90, 91, 90, 90, 91, 90], dtype=float)
        n_val = np.array([10, 10, 9, 10, 10, 9, 10], dtype=float)
        rep = np.array([0, 0, 0, 1, 1, 2, 2])
        st = paired_stats(a, b, n_train, n_val, rep)
        keep = [0, 1, 2, 3, 5]
        d = a[keep] - b[keep]
        n = len(d)
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / (n - 1)
        ratio = (sum(n_val[keep]) / n) / (sum(n_train[keep]) / n)
        se = np.sqrt(var * (1.0 / n + ratio))
        half = stats.t.ppf(0.975, n - 1) * se
        t = mean / se
        assert st["n_splits"] == 5 and st["n_reps"] == 3
        assert st["mean_diff"] == pytest.approx(mean, abs=1e-15)
        assert st["mean_a"] == pytest.approx(np.mean(a[keep]), abs=1e-15)
        assert st["mean_b"] == pytest.approx(np.mean(b[keep]), abs=1e-15)
        assert st["sd_diff"] == pytest.approx(np.sqrt(var), rel=1e-12)
        assert st["test_train_ratio"] == pytest.approx(ratio, rel=1e-15)
        assert st["se_nb"] == pytest.approx(se, rel=1e-12)
        assert st["ci_low"] == pytest.approx(mean - half, rel=1e-12)
        assert st["ci_high"] == pytest.approx(mean + half, rel=1e-12)
        assert st["t_nb"] == pytest.approx(t, rel=1e-12)
        assert st["p_nb"] == pytest.approx(
            2 * stats.t.sf(abs(t), n - 1), rel=1e-10)
        # d = 0.05, -0.02, 0.03, 0.00, 0.06
        assert (st["n_wins"], st["n_ties"], st["n_losses"]) == (3, 1, 1)
        assert st["win_rate"] == 0.6 and st["tie_rate"] == 0.2
        assert st["loss_rate"] == 0.2
        # NB correction: wider than the naive standard error
        assert st["se_nb"] > np.sqrt(var / n)

    def test_antisymmetric(self):
        from paired_stats import paired_stats
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=12), rng.normal(size=12)
        nt, nv = np.full(12, 80.0), np.full(12, 9.0)
        ab, ba = paired_stats(a, b, nt, nv), paired_stats(b, a, nt, nv)
        assert ab["mean_diff"] == -ba["mean_diff"]
        assert ab["se_nb"] == ba["se_nb"]
        assert ab["ci_low"] == pytest.approx(-ba["ci_high"], abs=1e-15)
        assert (ab["n_wins"], ab["n_losses"]) == (ba["n_losses"],
                                                  ba["n_wins"])

    def test_degenerate(self):
        from paired_stats import STAT_FIELDS, paired_stats
        nt, nv = np.full(3, 50.0), np.full(3, 5.0)
        st = paired_stats([np.nan] * 3, [1, 2, 3], nt, nv)
        assert st["n_splits"] == 0 and np.isnan(st["mean_diff"])
        assert set(st) == set(STAT_FIELDS)
        st = paired_stats([1.0, np.nan, 2], [0.5, 1, np.nan], nt, nv)
        assert st["n_splits"] == 1 and st["mean_diff"] == 0.5
        assert np.isnan(st["se_nb"]) and np.isnan(st["ci_low"])
        st = paired_stats([1.0, 2, 3], [1.0, 2, 3], nt, nv)
        assert st["se_nb"] == 0 and st["ci_low"] == st["ci_high"] == 0
        assert np.isnan(st["t_nb"]) and st["tie_rate"] == 1.0

    def test_orientation(self):
        from paired_stats import orient, orient_pair
        pairs = orient(["entropy", "mi", "random", "hub", "pca", "trace",
                        "facloc"])
        assert pairs[:3] == [("mi", "entropy"), ("mi", "trace"),
                             ("mi", "pca")]
        assert all(b != "mi" for _, b in pairs)
        assert all(a != "random" for a, _ in pairs)
        assert ("hub", "facloc") in pairs and ("facloc", "random") in pairs
        assert orient_pair("entropy", "mi") == ("mi", "entropy")
        assert orient_pair("knap_entropy", "knap_mi") == ("knap_mi",
                                                          "knap_entropy")
        assert orient_pair("knap_mi", "card_mi") == ("knap_mi", "card_mi")
        assert orient_pair("random", "cheap_first") == ("cheap_first",
                                                        "random")
        assert orient_pair("entropy_clipped", "mi_clipped") == (
            "mi_clipped", "entropy_clipped")
        assert orient_pair("mi_clipped", "mi") == ("mi_clipped", "mi")
        assert orient_pair("mi", "random_greedy") == ("mi", "random_greedy")

    def test_macro_pairs_facloc(self, capsys):
        """Facility location, the headline method since 2026-10-19, stays
        unranked (its pairs logged before keep their sign): MACRO_PAIRS =
        the key pairs, then facloc against mi / entropy / trace / random
        and against eval_baselines' similarity variants, each read as
        given; the CLI summary prints them."""
        import paired_stats as ps
        assert "facloc" not in ps.METHOD_ORDER
        n = len(ps.KEY_PAIRS)
        assert ps.MACRO_PAIRS[:n] == ps.KEY_PAIRS
        assert ps.MACRO_PAIRS[n:] == [
            ("mi", "facloc"), ("entropy", "facloc"), ("trace", "facloc"),
            ("facloc", "random"), ("facloc", "facloc_abs"),
            ("facloc", "facloc_sq")]
        for a, b in ps.MACRO_PAIRS:
            assert ps.orient_pair(a, b) == (a, b), (a, b)
        rng = np.random.default_rng(1)
        S = 5
        d = {f"{m}_r2": rng.normal(size=(S, 6))
             for m in ("mi", "trace", "facloc", "random")}
        d.update(split_rep=np.zeros(S, int), split_fold=np.arange(S),
                 split_id=np.arange(S), n_train=np.full(S, 90.0),
                 n_val=np.full(S, 10.0))
        t = ps.suffix_table(d, pathlib.Path("baselines_toy.npz"), "toy",
                            "Base")
        ps.print_summary([t])
        out = capsys.readouterr().out
        for a, b in [("trace", "facloc"), ("facloc", "random"),
                     ("mi", "trace")]:
            assert f"{a:>10} − {b:<10}" in out, (a, b)
        ps.print_summary([t], pairs="key")
        assert "facloc" not in capsys.readouterr().out

    def test_single_implementation(self):
        """Every eval script's *_paired.csv comes from paired_stats.py:
        no local copy of the estimator, the shared column names."""
        import re
        writers = []
        for path in sorted(CODE.glob("eval_*.py")):
            src = path.read_text()
            assert not re.search(r"^def paired_stats\b", src, re.M), path.name
            assert "ci95_" not in src, path.name
            if re.search(r"_paired[_.{].*csv", src):
                writers.append(path.name)
                assert re.search(r"^from paired_stats import .*paired_stats",
                                 src, re.M), path.name
                assert "STAT_FIELDS" in src, path.name
        assert {"eval_baselines.py", "eval_greedy_all.py",
                "eval_benchpress.py", "eval_cost_budget.py",
                "eval_tabimpute.py", "eval_reviewer_checks.py"} <= \
            set(writers)

    def test_no_eligibility_override(self):
        """One eligibility threshold for every script (cv_split's).  The
        one exception is eval_estimator_ablation.py, whose elig_{1,2,5,10}
        arm ablates the threshold on purpose: there exactly the calls
        below may pass one."""
        import re
        ablation = {"eval_estimator_ablation.py": {
            "eligible_mask(O_train, s.min_obs)",     # the ablation arm
            "eligible_mask(O_train, m)",             # its docstring
            "eligible_mask(O, m)"}}                  # its docstring
        for path in sorted(CODE.glob("*.py")):
            if path.name.startswith("test_") or path.name == "cv_split.py":
                continue
            src = path.read_text()
            assert "--min-train-obs" not in src, path.name
            # eligible_mask(O_train) only, never with a threshold argument
            calls = re.findall(r"eligible_mask\([^)\n]*,[^)\n]*\)?", src)
            if path.name in ablation:
                assert calls, path.name             # the arm is still there
                assert set(calls) <= ablation[path.name], (path.name, calls)
                continue
            assert not re.search(r"eligible_mask\([^)\n]*,", src), path.name


# ===================================================================
# Real data: Merged, rep-0 folds, greedy_mi under 1 vs 16 BLAS threads
# ===================================================================

def _merged_fold_selections(fold: int, k: int = 15) -> dict:
    """Selections of every method on rep-0 fold `fold` of Merged (10%
    holdout), plus the frozen pre-change greedy_mi as a control."""
    from path_config import data_dir
    from eval_entropy_vs_mi import estimate_corr

    d = np.load(data_dir() / "merged.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    M, N = B.shape
    s = [s for s in cv_splits(M, 10, 1, 0.1, SEED) if s.fold == fold][0]
    R, _, _ = estimate_corr(B[s.train_idx], O[s.train_idx])
    el = eligible_mask(O[s.train_idx])
    k = min(k, N)
    out = {"fold": fold, "n_ineligible": int((~el).sum())}
    for tag, mask in [("", None), ("_elig", el)]:
        mi = greedy_mi(R, k, eligible=mask)
        out["mi" + tag] = mi["selected"]
        out["mi" + tag + "_index_ties"] = [list(x) for x in mi["index_ties"]]
        out["entropy" + tag] = greedy_entropy(R, k, eligible=mask)["selected"]
        out["hub" + tag] = hub_select(R, k, eligible=mask)["selected"]
        out["trace" + tag] = greedy_trace_select(R, k, eligible=mask)["selected"]
        out["pca" + tag] = pca_cssp_select(R, k, eligible=mask)["selected_per_k"]
    out["mi_frozen"] = _frozen_greedy_mi(R, k)["selected"]
    return out


def _run_merged_folds(threads: int, folds=range(10)) -> dict:
    """Run _merged_fold_selections in one subprocess per fold, all with
    `threads` BLAS threads, at most cpu_count // threads at a time
    (oversubscribed OpenBLAS threads spin and slow EM down badly);
    returns {fold: result}."""
    env = dict(os.environ)
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[var] = str(threads)
    env.setdefault("MPLBACKEND", "Agg")
    folds = list(folds)
    width = max(1, min(len(folds), (os.cpu_count() or 1) // threads))
    out = {}
    for start in range(0, len(folds), width):
        procs = {f: subprocess.Popen(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--merged-fold", str(f)],
            cwd=str(CODE), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
            for f in folds[start:start + width]}
        for f, proc in procs.items():
            stdout, stderr = proc.communicate(timeout=1800)
            assert proc.returncode == 0, stderr[-2000:]
            out[f] = json.loads(stdout.strip().splitlines()[-1])
    return out


def _digest(results: dict, keys=None) -> str:
    payload = {f: {k: v for k, v in r.items() if keys is None or k in keys}
               for f, r in sorted(results.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


NEW_KEYS = [m + t for m in ["mi", "entropy", "hub", "trace", "pca"]
            for t in ["", "_elig"]]


def _merged_available():
    try:
        from path_config import data_dir
        return (data_dir() / "merged.matrix.npz").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _merged_available(), reason="data/merged.matrix.npz missing")
@pytest.mark.skipif(os.environ.get("BENCHSELECT_SKIP_SLOW") == "1",
                    reason="BENCHSELECT_SKIP_SLOW=1")
class TestMergedThreads:
    def test_selections_identical_1_vs_16_threads(self):
        r1 = _run_merged_folds(1)
        r16 = _run_merged_folds(16)
        for f in range(10):
            for key in NEW_KEYS:
                assert r1[f][key] == r16[f][key], (f, key)
        # rep-0 folds with eligible mt_bench/* columns see MI ties
        assert any(r1[f]["mi_index_ties"] for f in range(10))
        print("\nmerged digest (1 thread): ", _digest(r1, NEW_KEYS))
        print("frozen greedy_mi differs 1 vs 16 threads in folds",
              [f for f in range(10) if r1[f]["mi_frozen"] != r16[f]["mi_frozen"]])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-fold", type=int)
    ap.add_argument("--merged-digest", action="store_true",
                    help="print the digest of all rep-0 Merged selections")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.merged_fold is not None:
        print(json.dumps(_merged_fold_selections(args.merged_fold)))
    elif args.merged_digest:
        res = _run_merged_folds(args.threads)
        print(json.dumps({"threads": args.threads,
                          "digest_new": _digest(res, NEW_KEYS),
                          "digest_frozen_mi": _digest(res, ["mi_frozen"]),
                          "mi_index_ties": {f: r["mi_index_ties"]
                                            for f, r in res.items()},
                          "mi_elig_index_ties": {f: r["mi_elig_index_ties"]
                                                 for f, r in res.items()}}))
