# data/

Score matrices, cost tables and the scripts that download and build them. Licenses and attributions of every
file: `../DATA_LICENSES.md` (summary at the end of this file).

## Files

| Files | What | Written by |
|---|---|---|
| `<source>.csv`, `<source>.npz` | per-leaderboard snapshot of 2026-03-13 (`mmlu`, `open_llm`, `helm`, `alpaca_eval`, `arena_hard`, `livebench`, `wildbench`, `mt_bench`, `bigcode`, `mteb`, `bigbench`) | `download_<source>.py` |
| `{mmlu,mteb,merged}.matrix.{csv,npz}` | the paper's main matrices MMLU, MTEB, Merged (`B` with missing = 0, mask `O`, `model_names`, `benchmark_names`) | `build_matrices.py` (with `load_data.py`, `canonical_mapping.json`) |
| `benchpress.matrix.npz` (**not shipped**) | BenchPress, the fourth main matrix | `download_benchpress.py` (fetch + SHA-256 check) → `build_benchpress.py` |
| `{mmlu,mteb,merged}.cov.npz` | EM covariance estimates (Figure 1) | `code/estimate_covariance.py` |
| `<name>.matrix.{csv,npz}`, `<name>.meta.json`, `raw_manifests/<name>.json` | the extended matrices built from pinned inputs: `olv2sub` (OLL v2), `olv1items` (+ `olv1items.items.npz`, OLL v1 items), `helm_lite`, `helm_mmlu`, `helm_classic`, `mteb2026` | `build_<name>.py` |
| `epoch.matrix.{csv,npz}`, `epoch.meta.json` (**not shipped**) | Epoch, 277 x 61 | `build_epoch.py` from Epoch's `benchmark_data.zip` (SHA-256 pinned in `raw_manifests/epoch.json`) |
| `epoch_own.{npz,csv}`, `epoch_own.meta.json` | the 15 Epoch columns from Epoch's own files (203 models, 1,063 cells); not an input of any experiment | `subset_epoch_own.py` (from the full Epoch matrix) |
| `open_llm_v1.{csv,npz}`, `merged_v1.matrix.{csv,npz}`, `merged_v1.meta.json` | **submission-era (pre-fix) versions** of Open LLM v2 and Merged; they reproduce the submission's numbers and serve as the sensitivity check `merged_v1` | frozen copies (2026-10-18) |
| `canonical_mapping.json` | model-name mapping across collections (original name of every model in every collection) | ours (LLM-assisted matching, see the paper, App. C.4) |
| `mmlu_test_counts.json`, `mteb_task_sizes.json` | cost tables: test questions per MMLU subject, `num_samples` per MTEB task | `build_cost_tables.py` (`requirements-data.txt`) |
| `score_bounds.json` | valid score range of every benchmark column (protocol rule D2): bound, its source, observed range and count. In this release without the `benchpress` and `epoch` entries; `python code/score_bounds.py --write data/score_bounds.json` regenerates the full file once both matrices exist. The bounds used by the code come from `score_bounds.RULES`, not from this file | `code/score_bounds.py --write` |
| `SHA256SUMS` | SHA-256 of the snapshot, matrix, covariance and metadata files of the canonical run, including the two that are fetched or rebuilt (BenchPress, Epoch), so `shasum -a 256 -c --ignore-missing SHA256SUMS` checks both what is shipped and what you rebuild. The `score_bounds.json` line is that of this release's copy (the canonical run's full file: `f9a8a16a...`) | |

`bigbench.*` (BIG-bench Lite) is shipped but used in no experiment: its models appear in no other collection.

SHA-256 of the `epoch_own` files (not in `SHA256SUMS`, which lists the inputs of the canonical run):

```
3ad73b3866ff9cb8953b50ba42b9601e354d7410ea02e5a317c1a93e00ce488a  epoch_own.csv
5d179dc4216f2c2a4ce1a50a0966eda8fd3b4525a08219038de5373e05226530  epoch_own.meta.json
37a7b8836a6a0a5e25f158325b5cf0db1f34e51b13b69eaa86dbda8bb814229a  epoch_own.npz
```

## Fetching what is not shipped

```bash
python data/download_benchpress.py      # anadim/llm-benchmark-matrix @ e93b33f1, sha256 255ea009..., builds benchpress.matrix.npz
pip install -r requirements-data.txt
python data/build_epoch.py --raw-dir DIR --output-dir data   # DIR holds benchmark_data.zip (+ headers.txt, fetched_at.txt)
python data/subset_epoch_own.py         # optional: re-derive epoch_own.* from it
```

`build_epoch.py --help` shows how to fetch the zip. Epoch overwrites it in place, so a fresh download will usually
not match the pinned SHA-256 `e1ce578c...` and the builder refuses it; `--expected-sha256 any` builds the current
snapshot instead, which differs from the paper's.

## Open LLM Leaderboard v2: `open_llm.csv` fixed on 2026-10-18

**Defect.** The original `download_open_llm.py` kept the *last* result file per model in path order.
Where that file had a blank BBH/GPQA/MuSR group entry, a substring fallback stored the first
sub-task's value as the group score (e.g. `leaderboard_bbh_boolean_expressions` for BBH).
At the pinned revisions (`open-llm-leaderboard/results@aa81ecc`, `contents@9c09a7c`) this gave
47 wrong rows / 200 cells (180 values changed, 20 missing cells filled; |change| median 0.0098,
max 0.451). Each row has one cause:

| cause | rows | cells |
|---|---:|---:|
| sub-task value stored as BBH/GPQA/MuSR score | 22 | 66 |
| another complete file kept (last in path order, not the one the leaderboard shows) | 20 | 114 |
| partial re-upload (BBH + MATH only) kept, so IFEval/GPQA/MuSR/MMLU-PRO were missing | 5 | 20 |

For the 5 partial re-upload rows, v1's BBH and MATH_Lvl5 are the leaderboard's values and stay as
they are; only the four missing aggregates are filled. `HuggingFaceTB/SmolLM2-135M` is counted
under the sub-task fallback (BBH, GPQA, MuSR). Its leaderboard GPQA also comes from a different
file than the one v1 kept.

**Fix.** Rule 1: per model, keep the result file whose recomputed aggregates match the
leaderboard `contents` row: minimum L_inf; ties go to the latest timestamp, then the path. This
is the rule of `build_olv2sub.py`, and all 4,493 matched models use the same file as `olv2sub`.
A blank group entry becomes the sample-weighted mean of the file's own sub-tasks.

Rule 1b: 8 leaderboard rows match no single file (L_inf > 1e-9). Some of them combine two files.
Each aggregate that the rule-1 file does not reproduce is taken from the latest other unlimited
file of the model whose recomputed aggregate equals the leaderboard value (within 1e-9). The
2025-02-13 partial re-uploads have no n-samples for the MATH-hard tasks, so for them the
rule-1 file's sample counts (the same 1,324 questions) are used. Partial files are never a row's
main file.

| model | L_inf | rule-1 file | taken from another file |
|---|---:|---|---|
| `google/gemma-7b-it` | 0.0113 | 2024-10-24 | BBH, MATH from the 2025-02-13 partial re-upload |
| `HuggingFaceH4/zephyr-7b-alpha` | 0.0023 | 2024-10-24 | BBH, MATH from the 2025-02-13 partial re-upload |
| `mistralai/Mistral-7B-v0.1` | 0.0023 | 2024-10-24 | BBH, MATH from the 2025-02-13 partial re-upload |
| `google/gemma-2b` | 0.0015 | 2024-10-24 | BBH from the 2025-02-13 partial re-upload |
| `mistralai/Mistral-7B-v0.3` | 0.0007 | 2024-10-24 | BBH from the 2025-02-13 partial re-upload |
| `HuggingFaceTB/SmolLM2-135M` | 0.0098 | 2024-11-06 | IFEval, MATH from the 2025-02-13 complete file |
| `microsoft/Phi-3-small-{128k,8k}-instruct` | 0.021–0.022 | 2025-02-13 | none: the file's MATH group entry equals the leaderboard, its sub-tasks do not |

In the three rows that take MATH from the partial re-upload, the leaderboard's MATH is the
2025-02-13 re-scoring (task version 3.0). The 2024-10-24 file holds version 2.0, which is lower
by up to 0.0113. In `gemma-2b` and `Mistral-7B-v0.3` both versions give the same MATH. Result: every
stored value comes from a file that reproduces the leaderboard's aggregate within 1e-9. The two
exceptions are the Phi-3 MATH cells, whose stored group entry equals the leaderboard value
exactly. MATH_Lvl5, GPQA and MMLU-PRO, which are stored in the leaderboard's own definition,
equal the `contents` values within 3.5e-18 in all 4,493 rows.

Caveat for `olv2sub`: it takes all 42 sub-task columns from the rule-1 file. In the 6 rule-1b
rows, its BBH or MATH sub-tasks (IFEval and MATH for SmolLM2-135M) therefore come from a
different file than the leaderboard's aggregate.

The stored metrics and the file format are unchanged (harness metrics in [0, 1]; IFEval =
`prompt_level_strict_acc`). `--raw-dir` rebuilds offline from the pinned clone. The 10,509 results
files are checked against `RESULT_FILES_SHA256`, the sha256 of their sorted `sha256sum` listing
pinned in the script, and the contents parquet against `CONTENTS_SHA256`. The check has no
opt-out and does not depend on `raw_manifests/olv2sub.json`, which only names the differing files.
The online path downloads the same pinned revisions and applies the same code; `.env` is read only
there. The rebuild is byte-deterministic: two offline runs and the network-stubbed online path give
identical bytes.

**Rows the leaderboard table cannot place (14, unchanged).** These rows keep the latest
complete file, preferring unlimited runs. This reproduces their v1 values exactly:
10 models not in `contents`, 1 flagged model (`newsbang/Homer-v0.5-Qwen2.5-7B`), and
3 models whose only file is a sample-`limit`ed run, i.e. not a full evaluation
(`databricks/dbrx-base`, `internlm/internlm2-7b`, `mistral-community/Mixtral-8x22B-v0.1`).
None is in Merged; `--leaderboard-only` drops them.

**Check.** An independent re-derivation from the raw files equals the rebuilt file bitwise in every
cell. Against v1, 201 cells differ bitwise, all in the 47 rows: the 200 above, plus
`mistralai/Mistral-7B-v0.3` MATH_Lvl5, which moves by 1 ulp. That cell now holds the rule-1 file's
0.03021148036253777 instead of the partial re-upload's 0.030211480362537766; both equal the
leaderboard within 3.5e-18. Row set (4,507) and row order are unchanged.

The investigation's `open_llm.corrected.csv` and an earlier, uncommitted version of this fix took
all six aggregates from the rule-1 file. That moved 10 cells away from the leaderboard: BBH in the
5 partial re-upload rows, MATH in 3 of them (version 2.0 instead of 3.0), and IFEval and MATH of
SmolLM2-135M. Those 10 cells now hold their v1 values again.

`olv2sub.meta.json`'s `consistency_checks.vs_open_llm_csv` describes the pre-fix file,
`open_llm_v1.csv`. `build_olv2sub.py --reference-csv .../data/open_llm_v1.csv` reproduces the meta
file byte for byte. Its cause labels refer to the old rule, so do not pass the fixed file there.

## Merged: rebuilt with the fixed `open_llm.csv`

Only 2 Merged models use an affected row (`gemma-7b-it`, `zephyr-7b-alpha`). Both are partial
re-upload rows. Exactly 8 cells change, all `open_llm/*`: IFEval, GPQA, MuSR and MMLU-PRO go from
missing to observed. BBH and MATH_Lvl5 of both models are unchanged (the leaderboard's values).
Observed cells go from 4,185 to 4,193; models and columns are unchanged (`merged_v1.meta.json`
lists the cells). Of the 60 Merged models with an `open_llm` row, 58 now carry MATH from the
version 3.0 re-scoring. The other 2 (`dbrx-instruct`, `Mixtral-8x22B-Instruct-v0.1`) carry
version 1.0, exactly as the leaderboard shows them.

The effect was checked on the matched splits (10 folds, holdout 0.1, seed 42; mi, entropy, trace,
random; EM correlation, pooled R²), with mean R² at k = 1 / 5 / 10:

| method | rep 0, v1 | rep 0, fixed | 5 reps (50 splits), v1 | 5 reps, fixed |
|---|---|---|---|---|
| MI | .206 / .515 / .560 | .209 / .469 / .530 | .134 / .485 / .560 | .084 / .464 / .527 |
| entropy | .558 / .560 / .539 | .563 / .523 / .529 | .502 / .514 / .546 | .490 / .502 / .548 |
| trace | .558 / .564 / .589 | .563 / .582 / .621 | .502 / .595 / .627 | .490 / .591 / .624 |
| random | .337 / .545 / .553 | .337 / .541 / .555 | .259 / .562 / .590 | .260 / .561 / .591 |

No paired difference fixed − v1 is significant (Nadeau–Bengio p ≥ 0.36 over 50 splits, ≥ 0.21 on
rep 0). The k = 5 sets change in 40 (MI), 41 (entropy) and 40 (trace) of 50 splits, and in none
for random. Selections on Merged are therefore sensitive to a handful of cells. On the fixed
matrix, trace − MI is +0.127 [+0.022, +0.232] (p = 0.019) at k = 5 and +0.097 [−0.001, +0.195]
(p = 0.052) at k = 10 over 50 splits; on v1 it was +0.110 (p = 0.030) and +0.068 (p = 0.088).
R² at k = 1 is undefined in 5 of 50 random splits and 1 MI split; those splits are excluded from
the means and the pairs.

**Platform note.** `build_matrices.py` reproduces `merged_v1` value for value on the Mac where it
was built. On x86-64 Linux, 55 WildBench cells come out 1 or 2 ulps different (43 by 1 ulp, 12 by
2 ulps; max 1.42e-14, all at values ≥ 3.3). The cause is pandas' default CSV float parser, which
is not correctly rounded. It accumulates digits as `number * 10 + d`; clang on arm64 contracts this
into a fused multiply-add, the x86-64 wheels do not. The two results differ only for
17-significant-digit values ≥ 1. An exact emulation of the parser reproduces `merged_v1` (with FMA)
and the Linux rebuild (without FMA) in all 4,185 observed cells. `merged.matrix.npz` is therefore
`merged_v1` with the 8 cells filled. It equals the emulated arm64 build of the fixed sources in
every cell, and the CSV differs from `merged_v1.matrix.csv` only in the two rows. The 8 values
are `build_matrices.py`'s own parse of `open_llm.csv`, as in v1, and differ from the correctly
rounded values by up to 8.3e-17. `mmlu.matrix.*` and `mteb.matrix.*` rebuild byte-identically on both platforms.

`merged.cov.npz` was regenerated with `code/estimate_covariance.py` (venv `benchselect`, numpy 2.4.6,
single-threaded, x86-64). Regenerating the v1 file there reproduces the stored one to 1e-12 in Sigma
(same 274 EM iterations). The fixed matrix converges in 225 iterations; k90/95/99 stay at
6/10/20. The largest entry changes are among the 8 MT-Bench columns and `helm/GSM8K`, which are
co-observed for only 1–3 models.

## MMLU: `mmlu.csv` frozen; opt-in recovery of earlier files

**Defect.** `download_mmlu.py` reads only the last results file per model directory of
`open-llm-leaderboard-old/results`. Every value of the 5,452 rows in `mmlu.matrix.npz` is the
value in that last file, but 1,640 models whose MMLU results sit in an earlier file are lost:
1,639 because their last file has no MMLU, and 1 because its last file has too few subjects for
`build_matrices.py`'s 50 % rule. The paper's matrix keeps the default rule.
`--recover-earlier-files` takes each subject from the latest file that contains it.

**Check** (clone at `23474373`, `--results-dir`). With the flag, 1,640 matrix rows are added and
every row of the default matrix (after the 50 % rule) is bitwise unchanged. In the per-source
`mmlu.csv`, one default row changes: `google/recurrentgemma-9b` goes from 1 subject to 57 (its one
value unchanged). It is the model recovered because its last file has too few subjects. The added
rows equal the investigation's `mmlu.corrected.matrix.npz` (frozen 5,452 + 1,640 = 7,092 rows)
bitwise in the script's npz, and within 1.1e-16 after `build_matrices.py` parses the CSV.

**Case-insensitive download.** On the case-sensitive clone, the default rule gives
5,470 / 5,468 rows instead of 5,454 / 5,452. The frozen files were downloaded to a case-insensitive
macOS filesystem, where 18 model directories that differ only by case were merged. That merge
lost 16 models, filled `aihub-app/zyte-1.1b` from the merged pair's last file (max |diff| 0.119)
and spelled 22 names with the other casing. Emulating the merge on the clone reproduces the
frozen `mmlu.csv` bit for bit. The frozen files are kept as they are.

## Rebuilding the other matrices

Every downloader and builder documents its own options (`--help`); the extended builders verify the SHA-256 of
every raw input listed in `raw_manifests/<name>.json`. For example:

```bash
cd /tmp   # a directory without downloaded files
python -I .../data/download_open_llm.py --raw-dir RAW/olv2sub \
    --output-dir OUT --force --reference-csv .../data/open_llm_v1.csv
python -I .../data/build_matrices.py --input-dir SRC --output-dir OUT   # SRC: data/ sources
python -I .../data/download_mmlu.py --results-dir RAW/olv1full/results \
    --output-dir OUT [--recover-earlier-files]
```

`download_all.sh` runs the ten downloaders of the 2026-03-13 snapshot in a conda environment. Some Hugging Face
downloads need a token in the environment (`HF_TOKEN`, optionally read from a `.env` file next to the scripts,
which is never part of the release). The raw inputs themselves are not redistributed.

## sha256 of the files changed or added on 2026-10-18

`SHA256SUMS` lists all of them:

```
612a77b6cbc26f8cc6806272e385acc4322d15a587c55ead93d36de9fed0b4e3  open_llm.csv
948c68146e6c034b0d71c79a71c8382489fbec2dd6807de1cad7b315c1188c12  open_llm.npz
43216e82174b090252b43b18a433957d4a80b82a2e761854a60686bc27bfbbdd  merged.matrix.npz
801d77830de73dfe9d35e86fc54a6dca2a0b04ad7cafb7807496c87b27421479  merged.matrix.csv
391b5ba1be303598a30d1aefd1d58e0b471ff321d417fea41fd63b541e7bcc85  merged.cov.npz
375ee4d9a56d3a72ccf51b27972f60a2d758694734f1598e61bdf6356f5bc09e  open_llm_v1.csv
8d0a4d949e927d3dc853e4f527752236c44bbeabade356b4a4e779fbab15304c  open_llm_v1.npz
68ced5b62799c5cd5cac9058f93614ec77ca9f7399b4d401a8b3a0d1e27deb91  merged_v1.matrix.npz
fcb1a53d1efcb7a8d9cc26c9ee123e4c28b6ab844a380b0c5b866c81a8de5c22  merged_v1.matrix.csv
a6c30429de4f8059b25a73941b73993d97a0db8ede17efb3482f0cc71312eaf9  merged_v1.meta.json
```

The pre-fix `open_llm.*` and `merged.matrix.*` hashes are those of the `*_v1` copies; the pre-fix
`merged.cov.npz` was `abbb88c8d47b4a3779f3af8c824fdbf7141546ef52d870ca13bb1bea693d2a2a`.

## Licenses (summary)

Each file keeps the terms of its upstream source (see `../DATA_LICENSES.md`). Our own additions (mapping, metadata,
manifests, cost tables, derived arrangements) are CC BY 4.0 (`../LICENSE-DATA`); the scripts are Apache-2.0.
The AlpacaEval scores, i.e. `alpaca_eval.*` and the three `alpaca_eval/*` columns of `merged.matrix.*` and
`merged_v1.matrix.*`, are **CC BY-NC 4.0: non-commercial use only**, with attribution, changes indicated (we dropped
the non-numeric `mode` column and, in Merged, keep only three win-rate columns under canonical model names).
Sources without a declared license (Open LLM Leaderboard v1/v2, LiveBench, WildBench, BigCodeBench, MT-Bench) are
shipped as derived numeric scores with attribution; files are removed on request (alex@smola.org).
