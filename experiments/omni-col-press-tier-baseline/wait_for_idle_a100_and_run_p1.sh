#!/usr/bin/env bash
set -euo pipefail

readonly GPU_CANDIDATES="${OMNI_GPU_CANDIDATES:-0,1,2,3}"
readonly POLL_SECONDS="${OMNI_GPU_POLL_SECONDS:-30}"
readonly REQUIRED_IDLE_POLLS="${OMNI_GPU_REQUIRED_IDLE_POLLS:-2}"
readonly MAX_POLLS="${OMNI_GPU_MAX_POLLS:-960}"
readonly MAX_IDLE_MEMORY_MIB="${OMNI_GPU_MAX_IDLE_MEMORY_MIB:-100}"

for value_name in POLL_SECONDS REQUIRED_IDLE_POLLS MAX_POLLS MAX_IDLE_MEMORY_MIB; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer: ${value}" >&2
    exit 2
  fi
done

if [[ -z "${REPRFORGE_ROOT:-}" || "${REPRFORGE_ROOT}" != /* ]]; then
  echo "REPRFORGE_ROOT must be an absolute path" >&2
  exit 2
fi
readonly DOMAIN_RUNNER="${REPRFORGE_ROOT}/experiments/omni-col-press-tier-baseline/run_a100_domain_p1.sh"
if [[ ! -f "${DOMAIN_RUNNER}" ]]; then
  echo "missing domain runner: ${DOMAIN_RUNNER}" >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<< "${GPU_CANDIDATES}"
if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
  echo "OMNI_GPU_CANDIDATES must contain at least one physical GPU index" >&2
  exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "invalid physical GPU index: ${gpu_id}" >&2
    exit 2
  fi
done

selected_gpu=""
consecutive_idle_polls=0
for ((poll_number = 1; poll_number <= MAX_POLLS; poll_number++)); do
  idle_gpu=""
  for gpu_id in "${gpu_ids[@]}"; do
    gpu_name="$(nvidia-smi --id="${gpu_id}" --query-gpu=name --format=csv,noheader)"
    if [[ "${gpu_name}" != *A100* ]]; then
      continue
    fi
    memory_used="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    compute_pids="$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
    if [[ -z "${compute_pids}" && "${memory_used}" -lt "${MAX_IDLE_MEMORY_MIB}" ]]; then
      idle_gpu="${gpu_id}"
      break
    fi
  done

  if [[ -n "${idle_gpu}" && "${idle_gpu}" == "${selected_gpu}" ]]; then
    consecutive_idle_polls=$((consecutive_idle_polls + 1))
  elif [[ -n "${idle_gpu}" ]]; then
    selected_gpu="${idle_gpu}"
    consecutive_idle_polls=1
  else
    selected_gpu=""
    consecutive_idle_polls=0
  fi

  printf '%s poll=%d idle_gpu=%s consecutive=%d/%d\n' \
    "$(date --iso-8601=seconds)" \
    "${poll_number}" \
    "${idle_gpu:-none}" \
    "${consecutive_idle_polls}" \
    "${REQUIRED_IDLE_POLLS}"

  if [[ "${consecutive_idle_polls}" -ge "${REQUIRED_IDLE_POLLS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${selected_gpu}"
    echo "starting frozen P1 domain run on physical GPU ${selected_gpu}"
    exec bash "${DOMAIN_RUNNER}"
  fi
  sleep "${POLL_SECONDS}"
done

echo "no A100 became safely idle within ${MAX_POLLS} polls" >&2
exit 3
