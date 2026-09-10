#!/usr/bin/env bash
# usage: run-a100.sh <gpu> <smoke:0|N>
set -euo pipefail
gpu=${1:?gpu}; smoke=${2:-0}
root=/data/ldf/reprforge/scratch/colqwen2-pooled-upgrade-v1
pooled=/data/ldf/reprforge/scratch/pooled-gallery-upgrade-matrix-v1/code
python=/data/ldf/reprforge/envs/reprforge-py311/bin/python
matrix=/data/ldf/reprforge/scratch/sigir-version-evolution-matrix-v1/code
support=/data/ldf/reprforge/scratch/mmdocir-versioned-ir-transfer-v1/code
M=/data/ldf/reprforge/experiments/conditioned-transport-a100-20260815/models
A=/data/ldf/reprforge/scratch/colqwen2-release-transition-v1/models
distractor=/home/ldf/codex-workspaces/reprforge-mmdocir-native-stage1-20260815/data/mmdocir-page-stage1.parquet
if [ "${smoke}" != "0" ]; then output=${root}/smoke-output; else output=${root}/output; fi
test ! -e "${output}"
CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
"${python}" "${pooled}/pooled_upgrade_matrix.py" \
  --protocol "${root}/code/protocol.json" --matrix-code-root "${matrix}" --support-code-root "${support}" \
  --collections "${pooled}/collections.json" --distractor-parquet "${distractor}" \
  --family colqwen2 --base-model "$M/Qwen2-VL-2B-Instruct" --processor-model "$M/Qwen2-VL-2B-Instruct" --max-pixels 602112 \
  --deployed-adapter "$A/source" --deployed-projection "$M/colqwen2-base/custom_text_proj.safetensors" --targets "${root}/code/targets.json" \
  --output-root "${output}" --device cuda:0 --smoke "${smoke}"
