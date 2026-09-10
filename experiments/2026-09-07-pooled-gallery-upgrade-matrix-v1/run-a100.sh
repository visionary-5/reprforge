#!/usr/bin/env bash
# usage: run-a100.sh <gpu> <max_pixels> <smoke:0|N>
set -euo pipefail
gpu=${1:?gpu index}; pixels=${2:?max pixels}; smoke=${3:-0}
root=/data/ldf/reprforge/scratch/pooled-gallery-upgrade-matrix-v1
python=/data/ldf/reprforge/envs/reprforge-py311/bin/python
matrix=/data/ldf/reprforge/scratch/sigir-version-evolution-matrix-v1/code
support=/data/ldf/reprforge/scratch/mmdocir-versioned-ir-transfer-v1/code
base=/home/ldf/codex-workspaces/rag-partial-enrichment/models/qwen2.5-vl-3b
deployed=/data/ldf/reprforge/external/colqwen2.5-v0.1-d8bacd9
deployed_proj=/home/ldf/codex-workspaces/rag-latent-bridge/models/colqwen2.5-v0.2/base-shard2-header.bin
distractor=/home/ldf/codex-workspaces/reprforge-mmdocir-native-stage1-20260815/data/mmdocir-page-stage1.parquet
if [ "${smoke}" != "0" ]; then output=${root}/smoke-${pixels}; else output=${root}/output-${pixels}; fi
test ! -e "${output}"
CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
"${python}" "${root}/code/pooled_upgrade_matrix.py" \
  --protocol "${root}/code/protocol.json" --matrix-code-root "${matrix}" --support-code-root "${support}" \
  --collections "${root}/code/collections.json" --distractor-parquet "${distractor}" \
  --base-model "${base}" --processor-model "${base}" --max-pixels "${pixels}" \
  --deployed-adapter "${deployed}" --deployed-projection "${deployed_proj}" --targets "${root}/code/targets.json" \
  --output-root "${output}" --device cuda:0 --smoke "${smoke}"
