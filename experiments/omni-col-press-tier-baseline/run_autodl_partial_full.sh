#!/usr/bin/env bash
set -euo pipefail

readonly DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
readonly DOMAIN="${AUTODL_DOMAIN:?set AUTODL_DOMAIN}"
readonly SUBSET_CORPUS="${AUTODL_SUBSET_CORPUS:?set AUTODL_SUBSET_CORPUS}"
readonly OUTPUT_ROOT="${AUTODL_OUTPUT_ROOT:?set AUTODL_OUTPUT_ROOT}"
readonly REPRFORGE_ROOT="${AUTODL_REPRFORGE_ROOT:-${DATA_ROOT}/reprforge/partial-vlm-audit}"
readonly INPUT_ROOT="${DATA_ROOT}/datasets/vidore-v3-${DOMAIN}"
readonly CORPUS_ROWS="$(wc -l < "${SUBSET_CORPUS}")"
readonly QUERY_ROWS="$(wc -l < "${INPUT_ROOT}/queries.jsonl")"

if [[ ! "${DOMAIN}" =~ ^[a-z0-9-]+$ ]]; then
  echo "AUTODL_DOMAIN must be a lowercase dataset suffix" >&2
  exit 2
fi
for path in "${SUBSET_CORPUS}" "${INPUT_ROOT}/queries.jsonl" "${INPUT_ROOT}/qrels.jsonl"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing input file: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite existing output root: ${OUTPUT_ROOT}" >&2
  exit 2
fi

export OMNI_SOURCE="${DATA_ROOT}/reprforge/omni-col-press"
export OMNI_CORPUS_JSONL="${SUBSET_CORPUS}"
export OMNI_QUERY_JSONL="${INPUT_ROOT}/queries.jsonl"
export OMNI_QRELS="${INPUT_ROOT}/qrels.jsonl"
export OMNI_ASSETS="${INPUT_ROOT}/assets"
export OMNI_MODEL_CACHE="${DATA_ROOT}/models"
export OMNI_OUTPUT_ROOT="${OUTPUT_ROOT}"
export OMNI_MAX_CORPUS_ROWS="${CORPUS_ROWS}"
export OMNI_MAX_QUERY_ROWS="${QUERY_ROWS}"
export OMNI_EXPECTED_GPU_NAME_SUBSTRING="RTX 5090"
export OMNI_WRAPPER_REVISION="physical-partial-visual-index-v0-2026-08-07"
export OMNI_PROFILE_BATCHES="${OMNI_PROFILE_BATCHES:-1}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export TMPDIR="${DATA_ROOT}/cache/tmp"
export PYTHONUNBUFFERED=1
export PATH="${DATA_ROOT}/venvs/omni-cu128/bin:${PATH}"

OMNI_CASES=full bash \
  "${REPRFORGE_ROOT}/experiments/omni-col-press-tier-baseline/run_a100_smoke.sh"
