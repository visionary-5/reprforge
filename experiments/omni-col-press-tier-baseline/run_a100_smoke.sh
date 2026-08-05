#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_SOURCE_COMMIT="4a559677bbc8a3ea0c10322a721b52bb70d382ec"
readonly EXPECTED_COMPAT_PATCH_SHA256="5448707eaf3b49357784177fc46aea9be6a209d85a86c2cb0d76351099d73721"
readonly EXPECTED_COMPONENTS_FACTORY_SHA256="9f635a72d7ecf29e36c43a0f115beb099317787613d6bd79fd5b1c2553b659a9"
readonly EXPECTED_LOADERS_PATCH_SHA256="cc6624be900caf2a4a45918443f9b0293e410c0386b270eaaaf650d4c7fa1210"
readonly EXPECTED_LOADERS_SHA256="7de8abf3e19d318d942fc1c40fc71b6dcce9d9852d3b4397c997e99c8f53c41d"
readonly EXPECTED_MASK_PATCH_SHA256="c2ce55b8e89a7155fa4cf01b6e9a62348bb401c037d1282aeb6b6cd71d8ed72d"
readonly EXPECTED_MASKING_UTILS_SHA256="bb5aea0104c75e5fce99ddb6377d93905ee69ce4345cd6195f5e8a6c8ecdcf0f"
readonly EXPECTED_ATTN_MASK_PATCH_SHA256="9b684103f82c318da6821b80639106f81482a5f42cefeeda7bb9ac56e15453d7"
readonly EXPECTED_QWEN25VL_MODELING_SHA256="dbd2ece6823661da99a9337019888af86d313d7e3f301ffb71c65a04eefe32d3"
readonly FULL_MODEL_ID="hltcoe/ColBERT_qwen2.5-vl_colpali"
readonly FULL_MODEL_REVISION="14a7bb3328187705ff153e3511a47f9abb144054"
readonly FULL_CONFIG_SHA256="b4b7d8e29a63cbd33d30b761a4fa09f1bdb3126e07ace2af1280bc7ad2c69f7d"
readonly FULL_INDEX_SHA256="709aaeedb9fce8dca08d95042870a8ff133fc43562e9b979d5d04527f8392d4c"
readonly FULL_TOKENIZER_SHA256="9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"
readonly FULL_SHARD_1_SHA256="b53e17cd524ec03250abd7feaa7a2c41b4173c38367ed7953a9627752a71a9dc"
readonly FULL_SHARD_2_SHA256="23a2d511b1c6d2f9eac158cb6c944e90abd3ed5aacf5282c97ac2dca69d518ad"
readonly AGC_MODEL_ID="hltcoe/AGC_qwen2.5-vl_colpali"
readonly AGC_MODEL_REVISION="14ba8fb11de7d15d5a87c7fa17e893bffcdd9020"
readonly ATTN_IMPLEMENTATION="${OMNI_ATTN_IMPLEMENTATION:-sdpa}"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly COMPAT_PATCH="${SCRIPT_DIR}/image_only_optional_fast_plaid.patch"
readonly LOADERS_PATCH="${SCRIPT_DIR}/image_only_optional_fast_plaid_loaders.patch"
readonly MASK_PATCH="${SCRIPT_DIR}/torch25_mask_device.patch"
readonly ATTN_MASK_PATCH="${SCRIPT_DIR}/qwen25vl_attention_mask_device.patch"

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

if [[ ! -f "${COMPAT_PATCH}" ]]; then
  echo "missing image-only optional Fast-Plaid compatibility patch: ${COMPAT_PATCH}" >&2
  exit 2
fi
actual_patch_sha256="$(sha256sum "${COMPAT_PATCH}" | awk '{print $1}')"
if [[ "${actual_patch_sha256}" != "${EXPECTED_COMPAT_PATCH_SHA256}" ]]; then
  echo "compatibility patch SHA-256 mismatch: ${actual_patch_sha256}" >&2
  exit 2
fi
components_factory="${OMNI_SOURCE}/src/factory/components_factory.py"
actual_factory_sha256="$(sha256sum "${components_factory}" | awk '{print $1}')"
if [[ "${actual_factory_sha256}" != "${EXPECTED_COMPONENTS_FACTORY_SHA256}" ]]; then
  if ! git -C "${OMNI_SOURCE}" diff --quiet -- src/factory/components_factory.py; then
    echo "refusing to patch an unexpectedly modified components_factory.py" >&2
    exit 2
  fi
  git -C "${OMNI_SOURCE}" apply "${COMPAT_PATCH}"
  actual_factory_sha256="$(sha256sum "${components_factory}" | awk '{print $1}')"
fi

if [[ ! -f "${LOADERS_PATCH}" ]]; then
  echo "missing image-only optional Fast-Plaid loaders patch: ${LOADERS_PATCH}" >&2
  exit 2
fi
actual_loaders_patch_sha256="$(sha256sum "${LOADERS_PATCH}" | awk '{print $1}')"
if [[ "${actual_loaders_patch_sha256}" != "${EXPECTED_LOADERS_PATCH_SHA256}" ]]; then
  echo "loaders patch SHA-256 mismatch: ${actual_loaders_patch_sha256}" >&2
  exit 2
fi
loaders_file="${OMNI_SOURCE}/src/utils/loaders.py"
actual_loaders_sha256="$(sha256sum "${loaders_file}" | awk '{print $1}')"
if [[ "${actual_loaders_sha256}" != "${EXPECTED_LOADERS_SHA256}" ]]; then
  if ! git -C "${OMNI_SOURCE}" diff --quiet -- src/utils/loaders.py; then
    echo "refusing to patch an unexpectedly modified loaders.py" >&2
    exit 2
  fi
  git -C "${OMNI_SOURCE}" apply "${LOADERS_PATCH}"
  actual_loaders_sha256="$(sha256sum "${loaders_file}" | awk '{print $1}')"
fi
if [[ "${actual_loaders_sha256}" != "${EXPECTED_LOADERS_SHA256}" ]]; then
  echo "patched loaders.py SHA-256 mismatch: ${actual_loaders_sha256}" >&2
  exit 2
fi
if [[ ! -f "${MASK_PATCH}" ]]; then
  echo "missing Torch 2.5 mask-device compatibility patch: ${MASK_PATCH}" >&2
  exit 2
fi
actual_mask_patch_sha256="$(sha256sum "${MASK_PATCH}" | awk '{print $1}')"
if [[ "${actual_mask_patch_sha256}" != "${EXPECTED_MASK_PATCH_SHA256}" ]]; then
  echo "mask-device patch SHA-256 mismatch: ${actual_mask_patch_sha256}" >&2
  exit 2
fi
masking_utils_file="${OMNI_SOURCE}/src/models/masking_utils.py"
actual_masking_utils_sha256="$(sha256sum "${masking_utils_file}" | awk '{print $1}')"
if [[ "${actual_masking_utils_sha256}" != "${EXPECTED_MASKING_UTILS_SHA256}" ]]; then
  if ! git -C "${OMNI_SOURCE}" diff --quiet -- src/models/masking_utils.py; then
    echo "refusing to patch an unexpectedly modified masking_utils.py" >&2
    exit 2
  fi
  git -C "${OMNI_SOURCE}" apply "${MASK_PATCH}"
  actual_masking_utils_sha256="$(sha256sum "${masking_utils_file}" | awk '{print $1}')"
fi
if [[ "${actual_masking_utils_sha256}" != "${EXPECTED_MASKING_UTILS_SHA256}" ]]; then
  echo "patched masking_utils.py SHA-256 mismatch: ${actual_masking_utils_sha256}" >&2
  exit 2
fi
if [[ ! -f "${ATTN_MASK_PATCH}" ]]; then
  echo "missing Qwen2.5-VL attention-mask device compatibility patch: ${ATTN_MASK_PATCH}" >&2
  exit 2
fi
actual_attn_mask_patch_sha256="$(sha256sum "${ATTN_MASK_PATCH}" | awk '{print $1}')"
if [[ "${actual_attn_mask_patch_sha256}" != "${EXPECTED_ATTN_MASK_PATCH_SHA256}" ]]; then
  echo "attention-mask device patch SHA-256 mismatch: ${actual_attn_mask_patch_sha256}" >&2
  exit 2
fi
qwen25vl_modeling_file="${OMNI_SOURCE}/src/models/qwen2_5_vl_embed/modeling_qwen2_5_vl.py"
actual_qwen25vl_modeling_sha256="$(sha256sum "${qwen25vl_modeling_file}" | awk '{print $1}')"
if [[ "${actual_qwen25vl_modeling_sha256}" != "${EXPECTED_QWEN25VL_MODELING_SHA256}" ]]; then
  if ! git -C "${OMNI_SOURCE}" diff --quiet -- src/models/qwen2_5_vl_embed/modeling_qwen2_5_vl.py; then
    echo "refusing to patch an unexpectedly modified modeling_qwen2_5_vl.py" >&2
    exit 2
  fi
  git -C "${OMNI_SOURCE}" apply "${ATTN_MASK_PATCH}"
  actual_qwen25vl_modeling_sha256="$(sha256sum "${qwen25vl_modeling_file}" | awk '{print $1}')"
fi
if [[ "${actual_qwen25vl_modeling_sha256}" != "${EXPECTED_QWEN25VL_MODELING_SHA256}" ]]; then
  echo "patched modeling_qwen2_5_vl.py SHA-256 mismatch: ${actual_qwen25vl_modeling_sha256}" >&2
  exit 2
fi
if [[ "${actual_factory_sha256}" != "${EXPECTED_COMPONENTS_FACTORY_SHA256}" ]]; then
  echo "patched components_factory.py SHA-256 mismatch: ${actual_factory_sha256}" >&2
  exit 2
fi

for executable in git python sha256sum nvidia-smi /usr/bin/time; do
  if ! command -v "${executable}" >/dev/null 2>&1; then
    echo "missing executable: ${executable}" >&2
    exit 2
  fi
done

if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi
readonly PHYSICAL_GPU_ID="${CUDA_VISIBLE_DEVICES}"
gpu_names="$(nvidia-smi --id="${PHYSICAL_GPU_ID}" --query-gpu=name --format=csv,noheader)"
if [[ "${gpu_names}" != *A100* ]]; then
  echo "this preregistration requires one A100; GPU ${PHYSICAL_GPU_ID}: ${gpu_names}" >&2
  exit 2
fi
gpu_memory_used="$(nvidia-smi --id="${PHYSICAL_GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
gpu_compute_pids="$(nvidia-smi --id="${PHYSICAL_GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "${gpu_compute_pids}" || "${gpu_memory_used}" -ge 100 ]]; then
  echo "refusing to share occupied GPU ${PHYSICAL_GPU_ID}: memory=${gpu_memory_used} MiB pids=${gpu_compute_pids:-none}" >&2
  exit 2
fi

python - "${OMNI_CORPUS_JSONL}" "${OMNI_QUERY_JSONL}" "${OMNI_QRELS}" <<'PY'
import importlib
import json
import sys
from pathlib import Path
from packaging.version import Version

for module in (
    "accelerate", "datasets", "faiss", "jsonlines", "numpy", "peft",
    "qwen_vl_utils", "safetensors", "scipy", "torch", "transformers",
):
    importlib.import_module(module)

transformers = importlib.import_module("transformers")
if Version(transformers.__version__) < Version("4.57.1"):
    raise SystemExit(
        f"OmniColPress source requires transformers>=4.57.1; got {transformers.__version__}"
    )
importlib.import_module("transformers.models.qwen3_vl")

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

mkdir -p "${OMNI_MODEL_CACHE}"
readonly FULL_MODEL_PATH="${OMNI_MODEL_CACHE}/colbert-qwen2.5-vl-colpali-${FULL_MODEL_REVISION}"
readonly AGC_MODEL_PATH="${OMNI_MODEL_CACHE}/agc-qwen2.5-vl-colpali-${AGC_MODEL_REVISION}"

download_model() {
  local model_id="$1"
  local revision="$2"
  local model_path="$3"
  if [[ -f "${model_path}/config.json" && -f "${model_path}/model-00001-of-00002.safetensors" && -f "${model_path}/model-00002-of-00002.safetensors" ]]; then
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

verify_sha256() {
  local path="$1"
  local expected="$2"
  if [[ ! -f "${path}" ]]; then
    echo "missing fixed model artifact: ${path}" >&2
    exit 2
  fi
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "model artifact SHA-256 mismatch for ${path}: ${actual}" >&2
    exit 2
  fi
}

verify_full_model() {
  verify_sha256 "${FULL_MODEL_PATH}/config.json" "${FULL_CONFIG_SHA256}"
  verify_sha256 "${FULL_MODEL_PATH}/model.safetensors.index.json" "${FULL_INDEX_SHA256}"
  verify_sha256 "${FULL_MODEL_PATH}/tokenizer.json" "${FULL_TOKENIZER_SHA256}"
  verify_sha256 "${FULL_MODEL_PATH}/model-00001-of-00002.safetensors" "${FULL_SHARD_1_SHA256}"
  verify_sha256 "${FULL_MODEL_PATH}/model-00002-of-00002.safetensors" "${FULL_SHARD_2_SHA256}"
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
  verify_full_model
fi
if [[ "${needs_agc}" == true ]]; then
  download_model "${AGC_MODEL_ID}" "${AGC_MODEL_REVISION}" "${AGC_MODEL_PATH}"
fi

mkdir -p "${OMNI_OUTPUT_ROOT}/timing" "${OMNI_OUTPUT_ROOT}/profiles"
python - \
  "${OMNI_OUTPUT_ROOT}/run-manifest.json" \
  "${actual_commit}" \
  "${actual_patch_sha256}" \
  "${actual_factory_sha256}" \
  "${actual_loaders_patch_sha256}" \
  "${actual_loaders_sha256}" \
  "${actual_mask_patch_sha256}" \
  "${actual_masking_utils_sha256}" \
  "${actual_attn_mask_patch_sha256}" \
  "${actual_qwen25vl_modeling_sha256}" \
  "${FULL_MODEL_ID}" \
  "${FULL_MODEL_REVISION}" \
  "${needs_full}" \
  "${AGC_MODEL_ID}" \
  "${AGC_MODEL_REVISION}" \
  "${needs_agc}" \
  "${OMNI_CORPUS_JSONL}" \
  "${OMNI_QUERY_JSONL}" \
  "${OMNI_QRELS}" \
  "${gpu_names}" \
  "${ATTN_IMPLEMENTATION}" \
  "${requested_cases[*]}" \
  "${OMNI_WRAPPER_REVISION:-unknown}" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

(
    output_path,
    upstream_commit,
    factory_patch_sha256,
    factory_file_sha256,
    loaders_patch_sha256,
    loaders_file_sha256,
    mask_patch_sha256,
    masking_utils_sha256,
    attention_mask_patch_sha256,
    qwen25vl_modeling_sha256,
    model_id,
    model_revision,
    needs_full,
    agc_model_id,
    agc_model_revision,
    needs_agc,
    corpus_path,
    query_path,
    qrels_path,
    gpu_names,
    attention_implementation,
    cases,
    wrapper_revision,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

packages = {}
for package in (
    "accelerate", "datasets", "faiss-cpu", "huggingface-hub", "peft",
    "qwen-vl-utils", "torch", "torchvision", "transformers",
):
    packages[package] = importlib.metadata.version(package)

manifest = {
    "attention_implementation": attention_implementation,
    "cases": cases.split(),
    "compatibility_patches": {
        "components_factory_patch_sha256": factory_patch_sha256,
        "components_factory_sha256": factory_file_sha256,
        "loaders_patch_sha256": loaders_patch_sha256,
        "loaders_sha256": loaders_file_sha256,
        "mask_device_patch_sha256": mask_patch_sha256,
        "masking_utils_sha256": masking_utils_sha256,
        "attention_mask_device_patch_sha256": attention_mask_patch_sha256,
        "qwen25vl_modeling_sha256": qwen25vl_modeling_sha256,
    },
    "gpu_names": gpu_names.splitlines(),
    "physical_gpu_id": int(__import__("os").environ["CUDA_VISIBLE_DEVICES"]),
    "inputs": {
        "corpus": {"path": corpus_path, "sha256": sha256(corpus_path)},
        "qrels": {"path": qrels_path, "sha256": sha256(qrels_path)},
        "queries": {"path": query_path, "sha256": sha256(query_path)},
    },
    "models": [
        model
        for model, enabled in (
            ({"id": model_id, "revision": model_revision}, needs_full == "true"),
            ({"id": agc_model_id, "revision": agc_model_revision}, needs_agc == "true"),
        )
        if enabled
    ],
    "packages": packages,
    "upstream_commit": upstream_commit,
    "wrapper_revision": wrapper_revision,
}
Path(output_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

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
      python -m torch.distributed.run --nproc_per_node=1 -m src.build_index \
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
        "${extra_args[@]}" \
        > "${OMNI_OUTPUT_ROOT}/${case_name}/build.log" 2>&1

    /usr/bin/time -p -o "${OMNI_OUTPUT_ROOT}/timing/${case_name}-eval.time" \
      python -m torch.distributed.run --nproc_per_node=1 -m src.evaluate \
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
        "${extra_args[@]}" \
        > "${OMNI_OUTPUT_ROOT}/${case_name}/eval.log" 2>&1
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
