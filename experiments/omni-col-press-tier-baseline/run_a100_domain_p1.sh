#!/usr/bin/env bash
set -euo pipefail

required_vars=(
  REPRFORGE_ROOT
  OMNI_SOURCE
  OMNI_CORPUS_JSONL
  OMNI_QUERY_JSONL
  OMNI_QRELS
  OMNI_ASSETS
  OMNI_MODEL_CACHE
  OMNI_PYTHON_BIN
  OMNI_OUTPUT_ROOT
  OMNI_MAX_CORPUS_ROWS
  OMNI_MAX_QUERY_ROWS
  CUDA_VISIBLE_DEVICES
)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    exit 2
  fi
done
for path_name in REPRFORGE_ROOT OMNI_SOURCE OMNI_CORPUS_JSONL OMNI_QUERY_JSONL OMNI_QRELS OMNI_ASSETS OMNI_MODEL_CACHE OMNI_PYTHON_BIN OMNI_OUTPUT_ROOT; do
  path_value="${!path_name}"
  if [[ "${path_value}" != /* ]]; then
    echo "${path_name} must be absolute: ${path_value}" >&2
    exit 2
  fi
done
if [[ ! -x "${OMNI_PYTHON_BIN}" ]]; then
  echo "OMNI_PYTHON_BIN must be an executable file: ${OMNI_PYTHON_BIN}" >&2
  exit 2
fi
export PATH="$(dirname "${OMNI_PYTHON_BIN}"):${PATH}"
if [[ "$(command -v python)" != "${OMNI_PYTHON_BIN}" ]]; then
  echo "failed to pin python to OMNI_PYTHON_BIN: $(command -v python)" >&2
  exit 2
fi

readonly DOMAIN_RUNNER="${REPRFORGE_ROOT}/experiments/omni-col-press-tier-baseline/run_a100_smoke.sh"
if [[ ! -f "${DOMAIN_RUNNER}" ]]; then
  echo "missing frozen Omni runner: ${DOMAIN_RUNNER}" >&2
  exit 2
fi
if [[ -e "${OMNI_OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite existing output root: ${OMNI_OUTPUT_ROOT}" >&2
  exit 2
fi

OMNI_CASES=full,hpool,agc \
OMNI_MAX_CORPUS_ROWS="${OMNI_MAX_CORPUS_ROWS:?}" \
OMNI_MAX_QUERY_ROWS="${OMNI_MAX_QUERY_ROWS:?}" \
OMNI_WRAPPER_REVISION="${OMNI_WRAPPER_REVISION:-unknown}" \
bash "${DOMAIN_RUNNER}"

(
  cd "${REPRFORGE_ROOT}"
  python -m tools.rerank_omni_candidates \
    --index "${OMNI_OUTPUT_ROOT}/full/index" \
    --query-embeddings "${OMNI_OUTPUT_ROOT}/full/result/query_embeddings.pkl" \
    --query-masks "${OMNI_OUTPUT_ROOT}/full/result/query_masks.pkl" \
    --candidate-ranking "${OMNI_OUTPUT_ROOT}/hpool/result/ranking.txt" \
    --output-root "${OMNI_OUTPUT_ROOT}/cascade" \
    --candidate-depth 100 \
    --rerank-depths 20 50 100 \
    --score-chunk-size "${OMNI_CASCADE_CHUNK_SIZE:-25}" \
    --device cuda
)

echo "P1 domain run completed: ${OMNI_OUTPUT_ROOT}"
