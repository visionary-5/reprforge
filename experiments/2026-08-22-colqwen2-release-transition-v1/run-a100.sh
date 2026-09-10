#!/usr/bin/env bash

set -u
set -o pipefail

root=/data/ldf/reprforge/scratch/colqwen2-release-transition-v1
python=/data/ldf/reprforge/vidore-official-a70f23a/env/bin/python
model_root=/data/ldf/reprforge/experiments/conditioned-transport-a100-20260815
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${used}" -gt 100 ]; then
  printf 'GPU0 is no longer idle: %s MiB\n' "${used}" >&2
  exit 96
fi

cd "${root}/code" || exit 97
CUDA_VISIBLE_DEVICES=0 "${python}" run_gpu.py \
  --protocol gpu-protocol.json \
  --cpu-result "${root}/cpu-result.json" \
  --slice "${model_root}/data/energy-qrel-closed-128/slice.parquet" \
  --base-model "${model_root}/models/Qwen2-VL-2B-Instruct" \
  --base-projection "${model_root}/models/colqwen2-base/custom_text_proj.safetensors" \
  --source-adapter "${root}/models/source" \
  --target-adapter "${root}/models/target" \
  --ir-root "${root}/source-ir" \
  --output "${root}/gpu-result.json" \
  --device cuda:0
status=$?
printf '%s\n' "${status}" > "${root}/gpu-status"
exit "${status}"
