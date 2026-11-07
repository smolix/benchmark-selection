# Submodular Benchmark Selection: code and data

Code, data and results of the NeurIPS 2026 paper *Submodular Benchmark Selection* (Alex Smola).

The paper asks which few benchmarks of a suite to run so that the scores on the rest can be predicted, using
a Gaussian model of the benchmarks' correlations. It compares seven selectors on eleven public score matrices
in a paired design, and finds that the coverage objectives (facility location, explained variance) predict
best, while entropy and mutual information pick poor proxies.

## Layout

```
README.md               this file
LICENSE                 Apache-2.0 (all code)
LICENSE-DATA            CC BY 4.0 (our own data additions and results)
DATA_LICENSES.md        every data source: upstream license, what we ship, attribution
requirements.txt        pinned CPU environment (all experiments except TabImpute)
requirements-tabimpute.txt  pinned GPU environment (run_tabimpute.sh)
requirements-data.txt   pinned environment of the data download/build scripts
requirements-test.txt   requirements.txt + pytest-xdist, for running the tests in parallel
code/                   experiments, selectors, number and figure generation, unit tests (code/README.md)
data/                   score matrices, cost tables, download and build scripts (data/README.md)
paper_tools/            table bodies and derived counts of the paper, from numbers.csv
results/canonical-20261020/   outputs of the canonical camera-ready run (logs, numbers, figures, manifests)
```

## Install

Python 3.12 (validated with 3.12.3 on Linux x86_64 and 3.12.14 on macOS arm64).

```
python3.12 -m venv ~/venvs/benchselect
~/venvs/benchselect/bin/pip install -r requirements.txt
```

`requirements.txt` is the full lock (`uv pip freeze`) of the venv that ran the canonical experiments;
`code/run_all.sh` refuses an interpreter whose packages differ from it (exit 3) unless
`BENCHSELECT_ALLOW_UNPINNED=1`. To run the unit tests in parallel install `requirements-test.txt` instead
(it adds only pytest-xdist and execnet). The GPU experiment has its own lock, `requirements-tabimpute.txt`
(torch 2.13.0, tabimpute 0.0.4, CUDA 13 wheels).

## Data you have to fetch first

Two inputs of the paper are not redistributed here (see `DATA_LICENSES.md`):

```
python data/download_benchpress.py   # BenchPress at commit e93b33f1, SHA-256 checked, builds benchpress.matrix.npz
python data/build_epoch.py --raw-dir DIR   # full Epoch matrix from Epoch's benchmark_data.zip in DIR (needs pandas)
```

`download_benchpress.py` reproduces the paper's `benchpress.matrix.npz` bit for bit. `build_epoch.py` checks the zip
against the pinned SHA-256 (`e1ce578c...`, fetched 2026-10-17); Epoch's download is a living dataset that is
overwritten in place, so a fresh download will usually differ and the builder then refuses it unless you pass
`--expected-sha256 any` (which builds a different snapshot). The shipped `data/epoch_own.*` holds the 15 columns of
the pinned snapshot that come from Epoch's own files. Without the two matrices, drop them from the run:
`DATASETS="mmlu mteb merged" DATASETS_EXTRA="olv2sub olv1items helm_lite helm_mmlu helm_classic mteb2026"`
(and the prospective Epoch jobs, see `bash code/run_all.sh --list`).

## Reproduce everything

All CPU experiments (138 jobs: every log, every CPU figure and the paper-number tables):

```
BENCHSELECT_EXPERIMENT_ROOT=/scratch/benchselect-run \
PYTHON=~/venvs/benchselect/bin/python \
    bash code/run_all.sh --repeats 5
```

`--dry-run` prints every command, `--list` lists the jobs with their dependencies, outputs and cores,
`RUN_ONLY="baselines numbers"` runs a subset, `RUN_ONLY=tests` runs the unit tests. The driver needs bash >= 4.4,
writes only under `BENCHSELECT_EXPERIMENT_ROOT`, runs every job single-threaded (`MAX_JOBS` jobs in parallel,
`WORKERS` processes for the split-parallel jobs) and records a manifest of git state, host, package versions and the
SHA-256 of every input, code file and output. The full header of `code/run_all.sh` documents every option.

Runtime of the canonical run (AMD EPYC 7502, 32 cores / 64 threads, ~250 GB RAM, `MAX_JOBS=60 WORKERS=16`):
5,798 s wall (1 h 37 min) on 4090-1, 2026-10-20, run id `run_all-20261020T014458Z-17996`, with another user's jobs
on the host for the first hour. The shipped code (the same tree plus split-parallel workers and a backfilling
scheduler) ran the same 138 jobs in 3,938 s on an identical host and reproduced every log, figure and number bit for
bit (except runtimes, worker counts, code hashes and the manifest).

The GPU experiment (TabImpute, App. G of the paper; one GPU with 24 GB):

```
BENCHSELECT_EXPERIMENT_ROOT=/scratch/benchselect-run \
PYTHON=~/venvs/tabimpute/bin/python \
    bash code/run_tabimpute.sh
RUN_ONLY=numbers PYTHON=~/venvs/benchselect/bin/python BENCHSELECT_EXPERIMENT_ROOT=/scratch/benchselect-run \
    bash code/run_all.sh --repeats 5     # adds the TabImpute numbers
```

It took 23,497 s for all 50 splits and 7 methods on one RTX 4090. The checkpoint is pinned by revision and SHA-256
in `code/run_tabimpute.sh`.

The paper's five main figures come from the logs of a run:

```
BENCHSELECT_EXPERIMENT_ROOT=/scratch/benchselect-run python code/plot_paper.py --figures 0 1 2 3 4
```

## Results of the canonical run

`results/canonical-20261020/` holds the outputs of the canonical run (CC BY 4.0):

- `numbers/numbers.csv`: every number the paper uses, at full precision, with its source file and key
  (from `code/make_numbers.py`); `numbers/paired_<ds>.csv`: all paired method differences (mean,
  Nadeau-Bengio-corrected SE, 95% interval, wins) over the 50 matched splits.
- `logs-part1.tar.xz`, `logs-part2.tar.xz`: the result logs (`logs/`, 279 files, 221 MB unpacked), split in two
  archives to stay under GitHub's file-size limit. Unpack with
  `cd results/canonical-20261020 && for f in logs-part*.tar.xz; do tar -xJf $f; done` and check them with
  `shasum -a 256 -c logs.SHA256SUMS`.
- `logs_logit/`, `stdout/` (stdout of every job), `cov/` (EM covariance estimates), `figures/`, `figures_logit/`,
  `figures_paper/` (the paper's Figures 1-5), `tabimpute_provenance/` (logs of the GPU run and its conversion).
- `manifest.json` (latest invocation, the `RUN_ONLY=numbers` pass after TabImpute) and `manifests/` (the full run and
  that pass), `run_summary.txt` (the run's checks). Absolute server paths in these files have been shortened to `~`.

To rebuild the numbers from the shipped logs (about a minute):

```
cd results/canonical-20261020 && for f in logs-part*.tar.xz; do tar -xJf $f; done && cd ../..
python code/make_numbers.py --root results/canonical-20261020 --cov-dir results/canonical-20261020/cov \
    --data-dir data --datasets mmlu mteb merged benchpress olv2sub olv1items epoch helm_lite helm_mmlu \
    helm_classic mteb2026 merged_v1 --out-csv /tmp/numbers.csv --paired-dir /tmp/paired
python code/compare_runs.py --help        # diff two roots or two numbers.csv files
```

The matrix sizes (`Data*` keys) are read from `data/`, so the BenchPress and Epoch sizes appear only after the two
matrices have been fetched or built (above).

## Paper figures and tables

| Paper | Content | Script (log) |
|---|---|---|
| Figure 1 | eigenspectrum of the correlation matrices | `estimate_covariance.py` (cov/), `plot_paper.py --figures 0`; also `plot_eigenspectrum.py` |
| Figure 2 | R² against k on the four main matrices | `eval_baselines.py` (`baselines_<ds>.npz`), `plot_paper.py --figures 1` |
| Figure 3 | mechanism: why MI and entropy pick poor proxies | `analyze_objectives.py` (`analyze_objectives_<ds>.npz`), `plot_paper.py --figures 4` |
| Figure 4 | new models (prospective splits) | `eval_prospective.py` (`prospective_<setting>.npz`), `plot_paper.py --figures 2` |
| Figure 5 (App. F) | cost and item budgets | `eval_cost_budget.py` (`cost_budget_<ds>.npz`), `eval_item_budget.py` (`item_budget_olv1items.npz`), `plot_paper.py --figures 3` |
| Table 2 | scoreboard: FL against every selector | `eval_baselines.py`, `paired_stats.py`, `paper_tools/derive_numbers.py`, `paper_tools/make_tables.py` |
| Table 3 (App. C) | score matrices and sources | `data/` (see `DATA_LICENSES.md`) |
| Table 4 (App. D) | R² at k = 5 on all eleven matrices | `eval_baselines.py` |
| Table 5 (App. E) | prospective evaluation | `eval_prospective.py` |
| Table 6 (App. E) | model-family holdout | `eval_family_holdout.py` (`model_families.py`) |
| Table 7 (App. F) | knapsack facility location by cost | `eval_cost_budget.py` |
| Table 8 (App. F) | token-priced selection on HELM | `eval_token_costs.py` |
| Table 9 (App. G) | sensitivity to the correlation estimate | `eval_estimator_ablation.py` |
| Table 10 (App. G) | held-out benchmark groups | `eval_benchfamily_holdout.py` |
| Table 11 (App. G) | fitted imputers at fixed selections | `eval_imputers.py`; TabImpute: `eval_tabimpute.py` |
| Table 12 (App. H) | objective diagnostics | `analyze_objectives.py` |

Further checks of App. G (clipping, monotonicity, contamination, normality, logit transform, selection stability)
come from `eval_reviewer_checks.py`, `run_normality.py`, `eval_logit.py` and `eval_selection_stability.py`.
`paper_tools/numbers_used.csv` lists every number printed in the paper with the key of `numbers.csv` (or of
`paper_tools/numbers_derived.csv`) and the log it comes from; `paper_tools/make_tables.py` builds the table bodies
from `numbers/` and `numbers_derived.csv`, and `paper_tools/check_toy_suite.py` checks the toy example of Section 3.

## Tests

```
pip install -r requirements-test.txt
cd code && OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -n auto -q --ignore=test_tabimpute_sanity.py
```

(`test_tabimpute_sanity.py` is a GPU script, not a pytest module.) Tests that need a matrix that is not shipped
(BenchPress, full Epoch) skip.

## Data licenses (summary)

The code is Apache-2.0. Each data file keeps the terms of its upstream source; we do not relicense upstream data.
Our own additions (the canonical model-name mapping, metadata, manifests and all results) are CC BY 4.0
(`LICENSE-DATA`). The AlpacaEval scores, including the three AlpacaEval columns of Merged, are CC BY-NC 4.0
(non-commercial). Several sources declare no license; we ship their derived numeric scores with attribution and
remove files on request (alex@smola.org). BenchPress is not redistributed (pinned download script instead), and of
Epoch AI only the 15 columns from Epoch's own files are shipped. `DATA_LICENSES.md` has the full table, notices and
attributions. The files contain no benchmark questions, answers, prompts or model outputs.

The GPU experiment downloads the TabImpute V2 checkpoint (`tabimputev2.ckpt`) from the Hugging Face repository
`Tabimpute/TabImpute` at revision `727574989ddebe71860141c9a8b9f6e7958e3cce`, which is licensed Apache-2.0. The
checkpoint is not redistributed here.

`data/open_llm_v1.*` and `data/merged_v1.*` are the submission-era (pre-fix) versions of the Open LLM v2 and Merged
tables. They reproduce the submission's numbers and serve as a sensitivity check; all camera-ready results use
`open_llm.*` and `merged.*` (App. C.3 of the paper and `data/README.md`).

## Citation

```bibtex
@inproceedings{smola2026submodular,
  title     = {Submodular Benchmark Selection},
  author    = {Smola, Alex},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026}
}
```

Contact: Alex Smola, alex@smola.org.
