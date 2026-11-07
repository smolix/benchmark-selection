# code/

Everything here is driven by two entry points; run them from any directory (paths resolve relative to the
scripts, `path_config.py`):

- `run_all.sh`: all CPU experiments (138 jobs), the figures and the paper numbers. Its header documents every
  option, the job graph, the outputs and the manifest. `--list` shows the jobs, `--dry-run` the commands.
- `run_tabimpute.sh`: the GPU experiment (TabImpute, appendix note only); sources `run_all.sh` for its helpers.

Outputs go to `$BENCHSELECT_EXPERIMENT_ROOT/{logs,logs_logit,figures,figures_logit,cov,numbers,stdout}` and
`manifest.json`. Inputs are read from `../data` (override with `BENCHSELECT_DATA_DIR`).

## Modules

| File | Role |
|---|---|
| `greedy_select.py` | entropy and MI greedy selection, random selection, near-tie rule (`TIE_RTOL`) |
| `baseline_select.py` | facility location (`facility_location_select`), explained variance (`greedy_trace_select`), hub, PCA |
| `cv_split.py` | the matched 5 x 10 CV splits, eligibility (>= 3 training observations) and target rule (>= 2) |
| `em_cov.py`, `estimate_covariance.py` | EM correlation estimate for incomplete matrices; the cov files of Figure 1 |
| `score_bounds.py` | valid score range per benchmark; predictions are clipped to it (rule D2) |
| `paired_stats.py` | paired differences with Nadeau-Bengio-corrected intervals, win counts |
| `model_families.py` | model families for the family holdout |
| `make_numbers.py` | every paper number from the logs: `numbers.csv`, `numbers*.tex`, `paired_<ds>.csv` |
| `compare_runs.py` | diff the numbers of two experiment roots |
| `plot_paper.py` | the five main-text figures (`fig0`-`fig4`, paper Figures 1-5) from a root's logs |
| `path_config.py` | data/log/figure directories from environment variables |

## Experiments (job names of `run_all.sh --list`)

| Script | Jobs | Paper |
|---|---|---|
| `eval_baselines.py` | `baselines_<ds>`, `baselines_merged_v1` | Figure 2, Tables 2 and 4 |
| `eval_greedy_all.py`, `eval_entropy_vs_mi.py`, `plot_mi_selection_all.py`, `plot_selection_order.py` | `greedy_cv_*`, `entropy_vs_mi_*`, `mi_selection_*`, `selection_order*` | standard battery, selection orders |
| `eval_benchpress.py` | `benchpress` | BenchPress battery |
| `analyze_objectives.py` | `analyze_objectives[_fulloracle]` | Figure 3, Table 12 |
| `eval_prospective.py` | `prospective_*` | Figure 4, Table 5 |
| `eval_family_holdout.py` | `family_<ds>` | Table 6 |
| `eval_cost_budget.py` | `cost_<ds>`, `cost_fixed_<ds>` | Figure 5, Table 7 |
| `eval_item_budget.py` | `item_budget`, `item_fixed` | Figure 5(c) |
| `eval_token_costs.py` | `token_costs*`, `token_fixed` | Table 8 |
| `eval_estimator_ablation.py` | `estimator_<ds>` | Table 9 |
| `eval_benchfamily_holdout.py` | `benchfam_*` | Table 10 |
| `eval_imputers.py` | `imputers_<ds>_rep<r>`, `imputers_<ds>`, `imputers_all` | Table 11 |
| `eval_tabimpute.py` | `run_tabimpute.sh` | App. G (TabImpute note) |
| `eval_reviewer_checks.py`, `eval_selection_stability.py`, `eval_logit.py`, `run_normality.py` (`normality.py`) | `reviewer_checks`, `stability_*`, `logit*`, `normality` | App. G checks |
| `plot_eigenspectrum.py`, `plot_benchmark_overlap.py` | `eigenspectrum*`, `benchmark_overlap` | Figure 1 (run figure), App. C |

## Tests

`test_*.py` are pytest modules (`test_tabimpute_sanity.py` is a GPU script, not a test module):

```
cd code && OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -n auto -q --ignore=test_tabimpute_sanity.py
```

(`pytest-xdist` from `../requirements-test.txt`; `RUN_ONLY=tests bash run_all.sh` runs them serially.)
Tests that need a matrix that is not shipped (BenchPress, full Epoch) skip.

## Not in the release

The single-dataset predecessors `eval_greedy_mmlu.py`, `eval_greedy_merged.py`, `plot_mi_selection_mmlu.py` and
`run_greedy_mmlu.py` of the submission are not shipped; `run_all.sh` mentions them only in a comment and never runs
them. The two tests that inspect them skip when they are absent.
