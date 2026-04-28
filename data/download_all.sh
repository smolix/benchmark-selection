#!/usr/bin/env bash
# Download all benchmark datasets.
#
# Creates a conda environment "benchselect" with the required packages,
# then runs each download script inside it.
#
# Run from the data/ directory:
#   cd data && bash download_all.sh

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

ENV_NAME="benchselect"

# ── Create / update conda environment ────────────────────────────
if conda info --envs 2>/dev/null | grep -qw "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists – reusing it."
else
    echo "Creating conda environment '$ENV_NAME' …"
    conda create -y -n "$ENV_NAME" python=3.11
fi

echo "Installing / updating dependencies …"
conda run --no-capture-output -n "$ENV_NAME" pip install \
    datasets huggingface_hub pandas numpy requests mteb python-dotenv tqdm

# ── Download datasets ────────────────────────────────────────────
echo
echo "============================================================"
echo "1/10  Open LLM Leaderboard v2"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_open_llm.py
echo

echo "============================================================"
echo "2/10  HELM Lite"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_helm.py
echo

echo "============================================================"
echo "3/10  MTEB"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_mteb.py
echo

echo "============================================================"
echo "4/10  MMLU (per-subject)"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_mmlu.py
echo

echo "============================================================"
echo "5/10  AlpacaEval 2"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_alpaca_eval.py
echo

echo "============================================================"
echo "6/10  MT-Bench"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_mt_bench.py
echo

echo "============================================================"
echo "7/10  Arena-Hard-Auto"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_arena_hard.py
echo

echo "============================================================"
echo "8/10  LiveBench"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_livebench.py
echo

echo "============================================================"
echo "9/10  WildBench"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_wildbench.py
echo

echo "============================================================"
echo "10/10  BigCode / EvalPlus"
echo "============================================================"
conda run --no-capture-output -n "$ENV_NAME" python3 download_bigcode.py
echo

echo "============================================================"
echo "Done.  Summary of downloaded files:"
echo "============================================================"
ls -lh *.csv *.npz 2>/dev/null || echo "(no files found)"
