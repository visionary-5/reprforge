#!/usr/bin/env bash
set -euo pipefail

readonly DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
readonly DOMAIN="${AUTODL_DOMAIN:?set AUTODL_DOMAIN to the dataset suffix}"
readonly MAX_CORPUS_ROWS="${AUTODL_MAX_CORPUS_ROWS:?set AUTODL_MAX_CORPUS_ROWS}"
readonly MAX_QUERY_ROWS="${AUTODL_MAX_QUERY_ROWS:?set AUTODL_MAX_QUERY_ROWS}"
readonly REPRFORGE_ROOT="${DATA_ROOT}/reprforge/repo"
readonly INPUT_ROOT="${AUTODL_INPUT_ROOT:-${DATA_ROOT}/datasets/vidore-v3-${DOMAIN}}"
readonly OUTPUT_ROOT="${AUTODL_OUTPUT_ROOT:-${DATA_ROOT}/outputs/omni-${DOMAIN}-p1-full-hpool-agc-cascade-12bfcb2-rtx5090}"

if [[ ! "${DOMAIN}" =~ ^[a-z0-9-]+$ ]]; then
  echo "AUTODL_DOMAIN must be a lowercase dataset suffix: ${DOMAIN}" >&2
  exit 2
fi
if [[ ! "${MAX_CORPUS_ROWS}" =~ ^[1-9][0-9]*$ || ! "${MAX_QUERY_ROWS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AutoDL row limits must be positive integers" >&2
  exit 2
fi

export REPRFORGE_ROOT
export OMNI_SOURCE="${DATA_ROOT}/reprforge/omni-col-press"
export OMNI_CORPUS_JSONL="${INPUT_ROOT}/corpus.jsonl"
export OMNI_QUERY_JSONL="${INPUT_ROOT}/queries.jsonl"
export OMNI_QRELS="${INPUT_ROOT}/qrels.jsonl"
export OMNI_ASSETS="${INPUT_ROOT}/assets"
export OMNI_MODEL_CACHE="${DATA_ROOT}/models"
export OMNI_PYTHON_BIN="${DATA_ROOT}/venvs/omni-cu128/bin/python"
export OMNI_OUTPUT_ROOT="${OUTPUT_ROOT}"
export OMNI_MAX_CORPUS_ROWS="${MAX_CORPUS_ROWS}"
export OMNI_MAX_QUERY_ROWS="${MAX_QUERY_ROWS}"
export OMNI_EXPECTED_GPU_NAME_SUBSTRING="RTX 5090"
export OMNI_WRAPPER_REVISION="12bfcb260b15b95f136db59074bd396a379139af"
export OMNI_PROFILE_BATCHES="${OMNI_PROFILE_BATCHES:-0}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export TMPDIR="${DATA_ROOT}/cache/tmp"
export PYTHONUNBUFFERED=1

exec bash "${REPRFORGE_ROOT}/experiments/omni-col-press-tier-baseline/run_a100_domain_p1.sh"
