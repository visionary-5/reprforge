#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_SOURCE_COMMIT="4a559677bbc8a3ea0c10322a721b52bb70d382ec"
readonly FULL_MODEL_ID="hltcoe/ColBERT_qwen2.5-vl_colpali"
readonly FULL_MODEL_REVISION="14a7bb3328187705ff153e3511a47f9abb144054"
readonly AGC_MODEL_ID="hltcoe/AGC_qwen2.5-vl_colpali"
readonly AGC_MODEL_REVISION="14ba8fb11de7d15d5a87c7fa17e893bffcdd9020"
readonly ATTN_IMPLEMENTATION="${OMNI_ATTN_IMPLEMENTATION:-sdpa}"

IFS=',' read -r -a requested_cases <<< "${OMNI_CASES:-full,hpool,agc}"
for case_name in "${requested_cases[@]}"; do
  case "${case_name}" in
    full|hpool|agc) ;;
    *)
      echo "unsupported OMNI_CASES entry: ${case_name}" >&2
      exit 2
      ;;
  esac
done

required_vars=(
  OMNI_SOURCE
  OMNI_CORPUS_JSONL
  OMNI_QUERY_JSONL
  OMNI_QRELS
  OMNI_ASSETS
  OMNI_MODEL_CACHE
  OMNI_OUTPUT_ROOT
)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    exit 2
  fi
done

for path_name in OMNI_SOURCE OMNI_CORPUS_JSONL OMNI_QUERY_JSONL OMNI_QRELS OMNI_ASSETS OMNI_MODEL_CACHE OMNI_OUTPUT_ROOT; do
  path_value="${!path_name}"
  if [[ "${path_value}" != /* ]]; then
    echo "${path_name} must be an absolute path: ${path_value}" >&2
    exit 2
  fi
done

for input_file in "${OMNI_CORPUS_JSONL}" "${OMNI_QUERY_JSONL}" "${OMNI_QRELS}"; do
  if [[ ! -f "${input_file}" ]]; then
    echo "required input file does not exist: ${input_file}" >&2
    exit 2
  fi
done
if [[ ! -d "${OMNI_SOURCE}" || ! -d "${OMNI_ASSETS}" ]]; then
  echo "OMNI_SOURCE and OMNI_ASSETS must be existing directories" >&2
  exit 2
fi
if [[ -e "${OMNI_OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite existing output path: ${OMNI_OUTPUT_ROOT}" >&2
  exit 2
fi

actual_commit="$(git -C "${OMNI_SOURCE}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "official source commit mismatch: ${actual_commit}" >&2
  exit 2
fi

for executable in python torchrun nvidia-smi /usr/bin/time; do
  if ! command -v "${executable}" >/dev/null 2>&1; then
    echo "missing executable: ${executable}" >&2
    exit 2
  fi
done

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
gpu_count="$(printf '%s\n' "${gpu_names}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
if [[ "${gpu_count}" != "1" || "${gpu_names}" != *A100* ]]; then
  echo "this preregistration requires exactly one visible A100; visible GPUs: ${gpu_names}" >&2
  exit 2
fi

python - "${OMNI_CORPUS_JSONL}" "${OMNI_QUERY_JSONL}" "${OMNI_QRELS}" <<'PY'
import importlib
import json
import sys
from pathlib import Path

for module in (
    "accelerate", "datasets", "faiss", "jsonlines", "numpy", "peft",
    "qwen_vl_utils", "safetensors", "scipy", "torch", "transformers",
):
    importlib.import_module(module)

corpus_path, query_path, qrels_path = map(Path, sys.argv[1:])

def load_jsonl(path):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}")
    return rows

corpus = load_jsonl(corpus_path)
queries = load_jsonl(query_path)
if not 1 <= len(corpus) <= 32:
    raise SystemExit(f"smoke corpus must contain 1..32 rows, got {len(corpus)}")
if not 1 <= len(queries) <= 16:
    raise SystemExit(f"smoke query file must contain 1..16 rows, got {len(queries)}")
for i, row in enumerate(corpus, 1):
    if not (row.get("docid") or row.get("id")) or not row.get("image"):
        raise SystemExit(f"corpus row {i} requires docid/id and image")
for i, row in enumerate(queries, 1):
    if not (row.get("query_id") or row.get("id") or row.get("query-id")):
        raise SystemExit(f"query row {i} requires query_id/id/query-id")
    if not (row.get("query") or row.get("query_text")):
        raise SystemExit(f"query row {i} requires query/query_text")
if qrels_path.stat().st_size == 0:
    raise SystemExit("qrels file is empty")
PY

mkdir -p "${OMNI_MODEL_CACHE}" "${OMNI_OUTPUT_ROOT}/timing" "${OMNI_OUTPUT_ROOT}/profiles"
readonly FULL_MODEL_PATH="${OMNI_MODEL_CACHE}/colbert-qwen2.5-vl-colpali-${FULL_MODEL_REVISION}"
readonly AGC_MODEL_PATH="${OMNI_MODEL_CACHE}/agc-qwen2.5-vl-colpali-${AGC_MODEL_REVISION}"

download_model() {
  local model_id="$1"
  local revision="$2"
  local model_path="$3"
  if [[ -f "${model_path}/config.json" ]]; then
    return
  fi
  python - "${model_id}" "${revision}" "${model_path}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_dir=sys.argv[3],
)
PY
}

needs_full=false
needs_agc=false
for case_name in "${requested_cases[@]}"; do
  if [[ "${case_name}" == "full" || "${case_name}" == "hpool" ]]; then
    needs_full=true
  elif [[ "${case_name}" == "agc" ]]; then
    needs_agc=true
  fi
done
if [[ "${needs_full}" == true ]]; then
  download_model "${FULL_MODEL_ID}" "${FULL_MODEL_REVISION}" "${FULL_MODEL_PATH}"
fi
if [[ "${needs_agc}" == true ]]; then
  download_model "${AGC_MODEL_ID}" "${AGC_MODEL_REVISION}" "${AGC_MODEL_PATH}"
fi

run_case() {
  local case_name="$1"
  local model_path="$2"
  local pooling="$3"
  shift 3
  local extra_args=("$@")
  local index_path="${OMNI_OUTPUT_ROOT}/${case_name}/index"
  local result_path="${OMNI_OUTPUT_ROOT}/${case_name}/result"

  mkdir -p "${index_path}" "${result_path}" "${OMNI_OUTPUT_ROOT}/profiles/${case_name}"
  (
    cd "${OMNI_SOURCE}"
    ENABLE_PROFILING=1 \
    PROFILE_BATCHES="${OMNI_PROFILE_BATCHES:-2}" \
    PROFILE_OUTPUT_DIR="${OMNI_OUTPUT_ROOT}/profiles/${case_name}" \
    /usr/bin/time -p -o "${OMNI_OUTPUT_ROOT}/timing/${case_name}-build.time" \
      torchrun --nproc_per_node=1 -m src.build_index \
        --model_name_or_path "${model_path}" \
        --processor_name_or_path "${model_path}" \
        --dtype bfloat16 \
        --attn_implementation "${ATTN_IMPLEMENTATION}" \
        --pooling "${pooling}" \
        --dataset_name json \
        --corpus_name json \
        --corpus_path "${OMNI_CORPUS_JSONL}" \
        --assets_path "${OMNI_ASSETS}" \
        --passage_prefix "Passage: " \
        --encode_modalities '{"default":{"text":true,"image":true},"query":{"image":false}}' \
        --index_output_path "${index_path}" \
        --index_type multivec \
        --batch_size "${OMNI_BATCH_SIZE:-1}" \
        "${extra_args[@]}"

    /usr/bin/time -p -o "${OMNI_OUTPUT_ROOT}/timing/${case_name}-eval.time" \
      torchrun --nproc_per_node=1 -m src.evaluate \
        --model_name_or_path "${model_path}" \
        --processor_name_or_path "${model_path}" \
        --dtype bfloat16 \
        --attn_implementation "${ATTN_IMPLEMENTATION}" \
        --pooling "${pooling}" \
        --dataset_name json \
        --query_path "${OMNI_QUERY_JSONL}" \
        --qrels_path "${OMNI_QRELS}" \
        --assets_path "${OMNI_ASSETS}" \
        --query_prefix "Query: " \
        --encode_is_query \
        --encode_modalities '{"default":{"text":true},"query":{"image":false}}' \
        --index_path "${index_path}" \
        --index_type multivec \
        --output_path "${result_path}" \
        --batch_size "${OMNI_QUERY_BATCH_SIZE:-4}" \
        --top_k 1 5 10 \
        "${extra_args[@]}"
  )
}

for case_name in "${requested_cases[@]}"; do
  case "${case_name}" in
    full)
      run_case full "${FULL_MODEL_PATH}" colbert
      ;;
    hpool)
      run_case hpool "${FULL_MODEL_PATH}" hierarchical_clustering --num_repr_vectors 64
      ;;
    agc)
      run_case agc "${AGC_MODEL_PATH}" select \
        --num_repr_vectors 64 \
        --num_appending_token 64 \
        --use_parametric_appending_tokens \
        --use_cluster_pooling \
        --use_attn_weight_cluster_pooling
      ;;
  esac
done

echo "smoke completed; raw official outputs are under ${OMNI_OUTPUT_ROOT}"
echo "do not report paper numbers or get_stats() byte estimates as local measurements"
