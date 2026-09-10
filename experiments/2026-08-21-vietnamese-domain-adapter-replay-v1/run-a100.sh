#!/usr/bin/env bash

set -u
set -o pipefail

root=/data/ldf/reprforge/scratch/vietnamese-domain-adapter-replay-v1
python=/data/ldf/reprforge/vidore-official-a70f23a/env/bin/python

cd "${root}/code" || exit 97
CUDA_VISIBLE_DEVICES=0 "${python}" run_colqwen_recompile.py \
  --slice /data/ldf/reprforge/experiments/conditioned-transport-a100-20260815/data/energy-complete/slice.parquet \
  --manifest /data/ldf/reprforge/scratch/incremental-index-recompile-v1/energy-portrait-128-manifest-v1.json \
  --protocol protocol.json \
  --base-model /home/ldf/codex-workspaces/rag-partial-enrichment/models/qwen2.5-vl-3b \
  --adapter /data/ldf/reprforge/external/colqwen2.5-vietnamese-861105f \
  --base-shard-prefix /home/ldf/codex-workspaces/rag-latent-bridge/models/colqwen2.5-v0.2/base-shard2-header.bin \
  --cache-dir /data/ldf/reprforge/scratch/incremental-index-recompile-v1/vision-ir-v1 \
  --reuse-cache \
  --output "${root}/result.json" \
  --device cuda:0
status=$?
printf '%s\n' "${status}" > "${root}/status"
exit "${status}"
