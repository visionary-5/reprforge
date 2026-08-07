#!/usr/bin/env bash
set -euo pipefail

readonly DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
readonly DOMAIN="${AUTODL_DOMAIN:?set AUTODL_DOMAIN}"
readonly PREP_ROOT="${AUTODL_PREP_ROOT:?set AUTODL_PREP_ROOT}"
readonly OUTPUT_ROOT="${AUTODL_OUTPUT_ROOT:?set AUTODL_OUTPUT_ROOT}"
readonly REPRFORGE_ROOT="${AUTODL_REPRFORGE_ROOT:-${DATA_ROOT}/reprforge/partial-vlm-audit}"
readonly DATASET_ROOT="${AUTODL_DATASET_ROOT:-${DATA_ROOT}/datasets/vidore-v3-${DOMAIN}}"
readonly STRATEGIES="${AUTODL_STRATEGIES:-sha256_random,text_scarcity,visual_complexity,cheap_locator_disagreement,history_candidate_frequency,risk_cover_plus_history_benefit}"
readonly BUDGETS="${AUTODL_BUDGETS:-005,020,040}"
readonly INCLUDE_FULL="${AUTODL_INCLUDE_FULL:-1}"
readonly ANALYZE_MATRIX="${AUTODL_ANALYZE_MATRIX:-1}"
readonly RELEASE_INDEX="${AUTODL_RELEASE_INDEX_AFTER_RECEIPT:-0}"
readonly FULL_CASE_ROOT="${AUTODL_FULL_CASE_ROOT:-${OUTPUT_ROOT}/full-100}"
readonly QUERY_SPLITS="${AUTODL_QUERY_SPLITS:-${PREP_ROOT}/query-splits.json}"

if [[ ! "${DOMAIN}" =~ ^[a-z0-9-]+$ ]]; then
  echo "invalid domain: ${DOMAIN}" >&2
  exit 2
fi
for path in "${PREP_ROOT}/manifest.json" "${QUERY_SPLITS}" "${DATASET_ROOT}/corpus.jsonl" \
  "${DATASET_ROOT}/queries.jsonl" "${DATASET_ROOT}/qrels.jsonl"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required file: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${DATASET_ROOT}/assets" || ! -d "${REPRFORGE_ROOT}" ]]; then
  echo "dataset assets or ReprForge root is missing" >&2
  exit 2
fi
IFS=',' read -r -a strategy_values <<< "${STRATEGIES}"
IFS=',' read -r -a budget_values <<< "${BUDGETS}"
for value in "${strategy_values[@]}"; do
  if [[ ! "${value}" =~ ^[a-z0-9_]+$ ]]; then
    echo "invalid strategy: ${value}" >&2
    exit 2
  fi
done
for value in "${budget_values[@]}"; do
  if [[ ! "${value}" =~ ^[0-9]{3}$ || "${value}" == "000" || "${value}" == "100" ]]; then
    echo "invalid partial budget: ${value}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"
if [[ -f "${OUTPUT_ROOT}/preparation-manifest.json" ]]; then
  if ! cmp -s "${PREP_ROOT}/manifest.json" "${OUTPUT_ROOT}/preparation-manifest.json"; then
    echo "refusing resume with a different preparation manifest" >&2
    exit 2
  fi
else
  cp "${PREP_ROOT}/manifest.json" "${OUTPUT_ROOT}/preparation-manifest.json"
fi

run_direct_case() {
  local case_name="$1"
  local corpus="$2"
  local case_root="${OUTPUT_ROOT}/${case_name}"
  if [[ -e "${case_root}" ]]; then
    if [[ -f "${case_root}/run-manifest.json" && \
          -f "${case_root}/timing/full-build.time" && \
          ( -f "${case_root}/full/result/ranking.txt" || \
            -f "${case_root}/full/result-ranking-top100/ranking.txt" ) && \
          -f "${case_root}/case-receipt.json" ]]; then
      echo "resume: completed case ${case_name}"
      return
    fi
    echo "incomplete case exists; preserving it and stopping: ${case_root}" >&2
    exit 2
  fi
  AUTODL_DOMAIN="${DOMAIN}" \
  AUTODL_SUBSET_CORPUS="${corpus}" \
  AUTODL_OUTPUT_ROOT="${case_root}" \
  AUTODL_REPRFORGE_ROOT="${REPRFORGE_ROOT}" \
  AUTODL_DATA_ROOT="${DATA_ROOT}" \
  bash "${REPRFORGE_ROOT}/experiments/omni-col-press-tier-baseline/run_autodl_partial_full.sh"

  local receipt_args=(
    --case-root "${case_root}"
    --output "${case_root}/case-receipt.json"
  )
  local selection_manifest="${OUTPUT_ROOT}/${case_name}-input-manifest.json"
  if [[ -f "${selection_manifest}" ]]; then
    receipt_args+=(--selection-manifest "${selection_manifest}")
  fi
  PYTHONPATH="${REPRFORGE_ROOT}" "${DATA_ROOT}/venvs/omni-cu128/bin/python" \
    "${REPRFORGE_ROOT}/tools/record_physical_case_receipt.py" "${receipt_args[@]}"

  if [[ "${RELEASE_INDEX}" == "1" ]]; then
    local index_path="${case_root}/full/index"
    local resolved_case_root resolved_index expected_index symlinks
    resolved_case_root="$(realpath -e "${case_root}")"
    resolved_index="$(realpath -e "${index_path}")"
    expected_index="${resolved_case_root}/full/index"
    if [[ "${resolved_index}" != "${expected_index}" || ! -d "${resolved_index}" || -L "${index_path}" ]]; then
      echo "refusing unsafe index release: ${index_path}" >&2
      exit 2
    fi
    symlinks="$(find "${resolved_index}" -type l -print -quit)"
    if [[ -n "${symlinks}" ]]; then
      echo "refusing index release because it contains a symlink: ${symlinks}" >&2
      exit 2
    fi
    if [[ ! -s "${case_root}/case-receipt.json" ]]; then
      echo "refusing index release without a non-empty receipt" >&2
      exit 2
    fi
    echo "releasing reproducible index after receipt: ${resolved_index}"
    rm -rf -- "${resolved_index}"
  fi
}

# Full is built once. Each partial case is a direct build from its selected
# corpus; no complete embedding bank is sliced to fabricate ingestion savings.
if [[ "${INCLUDE_FULL}" == "1" ]]; then
  run_direct_case "full-100" "${DATASET_ROOT}/corpus.jsonl"
fi

for strategy in "${strategy_values[@]}"; do
  for budget in "${budget_values[@]}"; do
    subset="${PREP_ROOT}/subsets/${strategy}/budget-${budget}/corpus.jsonl"
    manifest="${PREP_ROOT}/subsets/${strategy}/budget-${budget}/manifest.json"
    if [[ ! -f "${subset}" || ! -f "${manifest}" ]]; then
      echo "missing prepared subset or manifest: ${strategy}/${budget}" >&2
      exit 2
    fi
    copied_manifest="${OUTPUT_ROOT}/${strategy}-${budget}-input-manifest.json"
    if [[ -f "${copied_manifest}" ]]; then
      if ! cmp -s "${manifest}" "${copied_manifest}"; then
        echo "refusing resume with changed case manifest: ${strategy}/${budget}" >&2
        exit 2
      fi
    else
      cp "${manifest}" "${copied_manifest}"
    fi
    run_direct_case "${strategy}-${budget}" "${subset}"
  done
done

if [[ "${ANALYZE_MATRIX}" != "1" ]]; then
  echo "physical analysis deferred by AUTODL_ANALYZE_MATRIX=${ANALYZE_MATRIX}"
elif [[ -f "${OUTPUT_ROOT}/physical-static-summary.json" ]]; then
  echo "resume: physical static summary already exists"
else
  PYTHONPATH="${REPRFORGE_ROOT}" "${DATA_ROOT}/venvs/omni-cu128/bin/python" \
    "${REPRFORGE_ROOT}/tools/analyze_physical_static_materialization_v0.py" \
    --config "${REPRFORGE_ROOT}/configs/progressive-visual-materialization-v0.json" \
    --dataset-root "${DATASET_ROOT}" \
    --matrix-root "${OUTPUT_ROOT}" \
    --full-case-root "${FULL_CASE_ROOT}" \
    --query-splits "${QUERY_SPLITS}" \
    --output "${OUTPUT_ROOT}/physical-static-summary.json"
fi

echo "static physical matrix complete: ${OUTPUT_ROOT}"
