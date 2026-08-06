#!/usr/bin/env bash
set -euo pipefail

readonly DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
readonly REPRFORGE_ROOT="${DATA_ROOT}/reprforge/repo"
readonly INPUT_ROOT="${DATA_ROOT}/datasets/vidore-v3-computer-science"
readonly OUTPUT_ROOT="${AUTODL_OUTPUT_ROOT:-${DATA_ROOT}/outputs/omni-cs-p1-full-hpool-agc-cascade-50a6693-rtx5090}"

export REPRFORGE_ROOT
export OMNI_SOURCE="${DATA_ROOT}/reprforge/omni-col-press"
export OMNI_CORPUS_JSONL="${INPUT_ROOT}/corpus.jsonl"
export OMNI_QUERY_JSONL="${INPUT_ROOT}/queries.jsonl"
export OMNI_QRELS="${INPUT_ROOT}/qrels.jsonl"
export OMNI_ASSETS="${INPUT_ROOT}/assets"
export OMNI_MODEL_CACHE="${DATA_ROOT}/models"
export OMNI_PYTHON_BIN="${DATA_ROOT}/venvs/omni-cu128/bin/python"
export OMNI_OUTPUT_ROOT="${OUTPUT_ROOT}"
export OMNI_MAX_CORPUS_ROWS=1360
export OMNI_MAX_QUERY_ROWS=215
export OMNI_EXPECTED_GPU_NAME_SUBSTRING="RTX 5090"
export OMNI_WRAPPER_REVISION="50a66938265a551844b3af1e3393ca32b01bc831"
export OMNI_PROFILE_BATCHES=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export TMPDIR="${DATA_ROOT}/cache/tmp"
export PYTHONUNBUFFERED=1

exec bash "${REPRFORGE_ROOT}/experiments/omni-col-press-tier-baseline/run_a100_domain_p1.sh"
