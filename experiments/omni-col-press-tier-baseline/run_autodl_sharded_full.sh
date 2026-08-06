#!/usr/bin/env bash
set -euo pipefail

readonly DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
readonly DOMAIN="${AUTODL_DOMAIN:?set AUTODL_DOMAIN to the dataset suffix}"
readonly MAX_CORPUS_ROWS="${AUTODL_MAX_CORPUS_ROWS:?set AUTODL_MAX_CORPUS_ROWS}"
readonly MAX_QUERY_ROWS="${AUTODL_MAX_QUERY_ROWS:?set AUTODL_MAX_QUERY_ROWS}"
readonly SHARD_COUNT="${AUTODL_FULL_SHARDS:?set AUTODL_FULL_SHARDS}"
readonly REPRFORGE_ROOT="${DATA_ROOT}/reprforge/repo"
readonly INPUT_ROOT="${AUTODL_INPUT_ROOT:-${DATA_ROOT}/datasets/vidore-v3-${DOMAIN}}"
readonly OUTPUT_ROOT="${AUTODL_OUTPUT_ROOT:-${DATA_ROOT}/outputs/omni-${DOMAIN}-full-sharded}"
readonly DOMAIN_RUNNER="${REPRFORGE_ROOT}/experiments/omni-col-press-tier-baseline/run_a100_smoke.sh"

if [[ ! "${DOMAIN}" =~ ^[a-z0-9-]+$ ]]; then
  echo "AUTODL_DOMAIN must be a lowercase dataset suffix: ${DOMAIN}" >&2
  exit 2
fi
for value in "${MAX_CORPUS_ROWS}" "${MAX_QUERY_ROWS}" "${SHARD_COUNT}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "AutoDL row limits and shard count must be positive integers" >&2
    exit 2
  fi
done
if (( SHARD_COUNT < 2 )); then
  echo "AUTODL_FULL_SHARDS must be at least 2" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite existing output root: ${OUTPUT_ROOT}" >&2
  exit 2
fi

export OMNI_SOURCE="${DATA_ROOT}/reprforge/omni-col-press"
export OMNI_CORPUS_JSONL="${INPUT_ROOT}/corpus.jsonl"
export OMNI_QUERY_JSONL="${INPUT_ROOT}/queries.jsonl"
export OMNI_QRELS="${INPUT_ROOT}/qrels.jsonl"
export OMNI_ASSETS="${INPUT_ROOT}/assets"
export OMNI_MODEL_CACHE="${DATA_ROOT}/models"
export OMNI_MAX_CORPUS_ROWS="${MAX_CORPUS_ROWS}"
export OMNI_MAX_QUERY_ROWS="${MAX_QUERY_ROWS}"
export OMNI_EXPECTED_GPU_NAME_SUBSTRING="RTX 5090"
export OMNI_WRAPPER_REVISION="${OMNI_WRAPPER_REVISION:-unknown}"
export OMNI_PROFILE_BATCHES="${OMNI_PROFILE_BATCHES:-1}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export TMPDIR="${DATA_ROOT}/cache/tmp"
export PYTHONUNBUFFERED=1
export PATH="${DATA_ROOT}/venvs/omni-cu128/bin:${PATH}"

mkdir -p "${OUTPUT_ROOT}"
for (( shard_index=0; shard_index<SHARD_COUNT; shard_index++ )); do
  shard_name="$(printf 'shard-%03d-of-%03d' "${shard_index}" "${SHARD_COUNT}")"
  OMNI_CASES=full \
  OMNI_DATASET_NUMBER_OF_SHARDS="${SHARD_COUNT}" \
  OMNI_DATASET_SHARD_INDEX="${shard_index}" \
  OMNI_OUTPUT_ROOT="${OUTPUT_ROOT}/${shard_name}" \
  bash "${DOMAIN_RUNNER}"
done

merge_args=()
for (( shard_index=0; shard_index<SHARD_COUNT; shard_index++ )); do
  shard_name="$(printf 'shard-%03d-of-%03d' "${shard_index}" "${SHARD_COUNT}")"
  merge_args+=(
    --shard-ranking "${OUTPUT_ROOT}/${shard_name}/full/result/ranking.txt"
  )
done

cd "${REPRFORGE_ROOT}"
python -m tools.merge_omni_shard_rankings \
  "${merge_args[@]}" \
  --qrels "${OMNI_QRELS}" \
  --top-k 100 \
  --output "${OUTPUT_ROOT}/merged/ranking.txt" \
  --report "${OUTPUT_ROOT}/merged/results.json"

echo "sharded Full run completed: ${OUTPUT_ROOT}"
