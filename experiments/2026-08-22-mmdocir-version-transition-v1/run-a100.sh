#!/usr/bin/env bash

set -u
set -o pipefail

root=/data/ldf/reprforge/scratch/mmdocir-version-transition-v1
legacy=/data/ldf/reprforge/scratch/mmdocir-versioned-ir-transfer-v1
generation=${root}/generation-colqwen25-v02
model_python=/data/ldf/reprforge/vidore-official-a70f23a/env/bin/python
serving_python=/data/ldf/reprforge/experiments/strong-baselines-20260805/omni-env/bin/python

cd "${root}/code" || exit 97
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${legacy}/code" "${model_python}" replay_terminal.py \
  --protocol protocol.json \
  --annotations /data/ldf/reprforge/external/MMDocIR/dataset/MMDocIR_annotations.jsonl \
  --base-model /home/ldf/codex-workspaces/rag-partial-enrichment/models/qwen2.5-vl-3b \
  --adapter /home/ldf/codex-workspaces/rag-latent-bridge/models/colqwen2.5-v0.2 \
  --base-shard-prefix /home/ldf/codex-workspaces/rag-latent-bridge/models/colqwen2.5-v0.2/base-shard2-header.bin \
  --pca256 /data/ldf/reprforge/scratch/versioned-ir-codec-v1/pca256-compact-measure.pt \
  --ir-root "${legacy}/full-ir" \
  --reference-result "${legacy}/full-result.json" \
  --reference-documents "${legacy}/full-documents" \
  --terminal-root "${generation}/terminal" \
  --document-output "${root}/quality-documents" \
  --output "${root}/replay-result.json" \
  --device cuda:0
replay_status=$?
if [ "${replay_status}" -ne 0 ]; then
  printf '%s\n' "${replay_status}" > "${root}/status"
  exit "${replay_status}"
fi

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ionice -c 3 nice -n 10 "${serving_python}" build_publish.py \
  --protocol protocol.json \
  --replay-result "${root}/replay-result.json" \
  --terminal-root "${generation}/terminal" \
  --reference-root /data/ldf/reprforge/scratch/mmdocir-faiss-search-fidelity-v1/reference \
  --generation-root "${generation}" \
  --deployment-root "${root}/deployment" \
  --output "${root}/result.json"
status=$?
printf '%s\n' "${status}" > "${root}/status"
exit "${status}"
