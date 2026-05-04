"""Run normality diagnostics on all three datasets.

For each dataset:
  1. Standardize observed scores to zero mean, unit variance per benchmark.
  2. Run per-column Shapiro-Wilk tests (on observed entries only).
  3. Run Mardia's multivariate test on the fully observed sub-block
     (rows with no missing entries) when feasible.
  4. Generate a combined figure: histogram of W statistics + Q-Q plot
     of Mahalanobis distances.

Saves figures to figures/ and a summary JSON to logs/.
"""

import pathlib
import sys
import json
import numpy as np
from scipy import stats

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from normality import mardia_test, shapiro_wilk_columns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["mmlu", "mteb", "merged"]
DISPLAY = {"mmlu": "MMLU", "mteb": "MTEB", "merged": "Merged"}


def run_dataset(ds_name):
    """Run normality diagnostics for one dataset."""
    print(f"\n{'#'*60}")
    print(f"# {ds_name.upper()}")
    print(f"{'#'*60}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    names = list(d["benchmark_names"])
    M, N = B.shape
    obs_frac = O.sum() / (M * N)
    print(f"{M} models × {N} tasks, {obs_frac:.1%} observed", flush=True)

    # Standardize per column (observed entries only)
    col_count = np.maximum(O.sum(axis=0), 1.0)
    col_mean = (B * O).sum(axis=0) / col_count
    B_c = O * (B - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))
    B_std = O * (B - col_mean[None, :]) / col_std[None, :]

    # ── Per-column Shapiro-Wilk on observed entries ──────────────
    W_stats = np.empty(N)
    pvals_raw = np.empty(N)
    n_obs_col = np.empty(N, dtype=int)

    for j in range(N):
        obs_mask = O[:, j] == 1
        n_obs = int(obs_mask.sum())
        n_obs_col[j] = n_obs
        if n_obs < 3:
            W_stats[j] = np.nan
            pvals_raw[j] = np.nan
            continue
        col = B_std[obs_mask, j]
        # Shapiro-Wilk limited to 5000 samples; subsample if needed
        if n_obs > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_obs, 5000, replace=False)
            col = col[idx]
        w, p = stats.shapiro(col)
        W_stats[j] = w
        pvals_raw[j] = p

    # Multiple testing correction (BH)
    valid = ~np.isnan(pvals_raw)
    n_valid = int(valid.sum())
    pvals_bonf = np.full(N, np.nan)
    pvals_bh = np.full(N, np.nan)
    if n_valid > 0:
        pv = pvals_raw[valid]
        pvals_bonf[valid] = np.minimum(pv * n_valid, 1.0)
        # BH
        order = np.argsort(pv)
        ranked = np.empty(n_valid)
        ranked[order] = np.arange(1, n_valid + 1)
        adj = pv * n_valid / ranked
        adj = np.minimum(adj, 1.0)
        rev_order = np.argsort(-ranked)
        cummin = np.minimum.accumulate(adj[rev_order])
        adj[rev_order] = cummin
        pvals_bh[valid] = adj

    reject_bh = pvals_bh < 0.05
    reject_bonf = pvals_bonf < 0.05
    n_reject_bh = int(np.nansum(reject_bh))
    n_reject_bonf = int(np.nansum(reject_bonf))

    print(f"\nShapiro-Wilk (α=0.05):")
    print(f"  W range: [{np.nanmin(W_stats):.4f}, {np.nanmax(W_stats):.4f}]")
    print(f"  Median W: {np.nanmedian(W_stats):.4f}")
    print(f"  Rejections: {n_reject_bonf}/{n_valid} (Bonferroni), "
          f"{n_reject_bh}/{n_valid} (BH)")

    # Print worst benchmarks
    if n_valid > 0:
        worst_idx = np.argsort(W_stats[valid])[:5]
        valid_indices = np.where(valid)[0]
        print(f"  Worst 5 benchmarks:")
        for idx in worst_idx:
            j = valid_indices[idx]
            rej = "FAIL" if reject_bh[j] else "ok"
            print(f"    {names[j]:<35s} W={W_stats[j]:.4f}  "
                  f"p_BH={pvals_bh[j]:.4e}  {rej}")

    # ── Mardia's test on complete rows ───────────────────────────
    # Find rows with all entries observed
    complete_rows = np.all(O == 1, axis=1)
    n_complete = int(complete_rows.sum())
    mardia_result = None
    maha_diag = None

    if n_complete > N + 5:
        print(f"\nMardia test ({n_complete} complete rows):", flush=True)
        B_complete = B_std[complete_rows]
        try:
            mardia_result = mardia_test(B_complete)
            print(f"  Skewness β̂₁ = {mardia_result.beta_1:.2f}, "
                  f"p = {mardia_result.skew_pvalue:.4e} "
                  f"{'REJECT' if mardia_result.skew_pvalue < 0.05 else 'PASS'}")
            print(f"  Kurtosis β̂₂ = {mardia_result.beta_2:.2f} "
                  f"(expected {N*(N+2):.0f}), "
                  f"z = {mardia_result.kurt_stat:.2f}, "
                  f"p = {mardia_result.kurt_pvalue:.4e} "
                  f"{'REJECT' if mardia_result.kurt_pvalue < 0.05 else 'PASS'}")

            # Mahalanobis distances for Q-Q plot
            X_c = B_complete - B_complete.mean(axis=0)
            Sigma_hat = (X_c.T @ X_c) / n_complete
            # Regularize for stability
            eigvals, eigvecs = np.linalg.eigh(Sigma_hat)
            eigvals = np.maximum(eigvals, 1e-8)
            Sigma_inv = (eigvecs / eigvals) @ eigvecs.T
            maha_diag = np.sum((X_c @ Sigma_inv) * X_c, axis=1)
            print(f"  Mahalanobis d² range: [{maha_diag.min():.1f}, "
                  f"{maha_diag.max():.1f}]")
        except Exception as e:
            print(f"  Mardia test failed: {e}", flush=True)
    else:
        print(f"\nMardia test: skipped ({n_complete} complete rows, "
              f"need > {N+5})", flush=True)

    return {
        "ds_name": ds_name,
        "M": M, "N": N, "obs_frac": obs_frac,
        "n_complete": n_complete,
        "W_stats": W_stats, "pvals_bh": pvals_bh,
        "n_reject_bh": n_reject_bh, "n_reject_bonf": n_reject_bonf,
        "n_valid": n_valid,
        "names": names,
        "mardia": mardia_result,
        "maha_diag": maha_diag,
    }


def plot_diagnostics(results_list):
    """Create combined normality diagnostics figure (Shapiro-Wilk histograms)."""
    n_ds = len(results_list)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 3.5))
    if n_ds == 1:
        axes = [axes]

    for col, res in enumerate(results_list):
        ds_name = res["ds_name"]
        W = res["W_stats"]
        reject = res["pvals_bh"] < 0.05

        ax = axes[col]
        valid_W = W[~np.isnan(W)]
        pass_W = W[(~np.isnan(W)) & (~reject)]
        fail_W = W[(~np.isnan(W)) & reject]

        w_min = np.nanmin(W)
        bins = np.linspace(max(0.0, w_min - 0.02), 1.001, 30)
        if len(pass_W) > 0:
            ax.hist(pass_W, bins=bins, color="#4477AA", alpha=0.7,
                    label=f"Pass ({len(pass_W)})", edgecolor="white")
        if len(fail_W) > 0:
            ax.hist(fail_W, bins=bins, color="#CC3311", alpha=0.7,
                    label=f"Reject ({len(fail_W)})", edgecolor="white")
        ax.axvline(x=0.95, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Shapiro-Wilk $W$")
        ax.set_ylabel("Number of benchmarks")
        ax.set_title(f"{DISPLAY[ds_name]}  ({res['M']}×{res['N']}, "
                     f"{res['obs_frac']:.0%} obs.)", fontweight="bold")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT / "normality_diagnostics.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}", flush=True)
    plt.close()


if __name__ == "__main__":
    results = []
    summary = {}

    for ds_name in DATASETS:
        res = run_dataset(ds_name)
        results.append(res)

        # Serializable summary
        summary[ds_name] = {
            "M": res["M"], "N": res["N"],
            "obs_frac": float(res["obs_frac"]),
            "n_complete": res["n_complete"],
            "n_reject_bh": res["n_reject_bh"],
            "n_reject_bonf": res["n_reject_bonf"],
            "n_valid": res["n_valid"],
            "W_median": float(np.nanmedian(res["W_stats"])),
            "W_min": float(np.nanmin(res["W_stats"])),
            "W_max": float(np.nanmax(res["W_stats"])),
        }
        if res["mardia"] is not None:
            mr = res["mardia"]
            summary[ds_name]["mardia_skew_p"] = float(mr.skew_pvalue)
            summary[ds_name]["mardia_kurt_p"] = float(mr.kurt_pvalue)
            summary[ds_name]["mardia_beta1"] = float(mr.beta_1)
            summary[ds_name]["mardia_beta2"] = float(mr.beta_2)

        # Per-benchmark details
        worst = []
        order = np.argsort(res["W_stats"])
        for j in order[:10]:
            if not np.isnan(res["W_stats"][j]):
                worst.append({
                    "name": res["names"][j],
                    "W": float(res["W_stats"][j]),
                    "p_bh": float(res["pvals_bh"][j]) if not np.isnan(res["pvals_bh"][j]) else None,
                    "reject": bool(res["pvals_bh"][j] < 0.05) if not np.isnan(res["pvals_bh"][j]) else False,
                })
        summary[ds_name]["worst_benchmarks"] = worst

    plot_diagnostics(results)

    with open(LOGS / "normality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {LOGS / 'normality_summary.json'}")
    print("\nDone.")
