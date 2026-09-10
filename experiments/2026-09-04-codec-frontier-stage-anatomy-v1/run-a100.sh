#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?gpu index}
smoke=${2:-0}
root=/data/ldf/reprforge/scratch/codec-frontier-stage-anatomy-v1
python=/data/ldf/reprforge/envs/reprforge-py311/bin/python
matrix=/data/ldf/reprforge/scratch/sigir-version-evolution-matrix-v1/code
support=/data/ldf/reprforge/scratch/mmdocir-versioned-ir-transfer-v1/code
arxiv=/home/ldf/.cache/huggingface/hub/datasets--vidore--arxivqa_test_subsampled/snapshots/b8a106812c8682bab08935cf5d1b4566c82562de/data/test-00000-of-00001.parquet
docvqa=/home/ldf/.cache/huggingface/hub/datasets--vidore--docvqa_test_subsampled/snapshots/49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6/data/test-00000-of-00001.parquet
flickr=/home/ldf/.cache/huggingface/hub/datasets--nlphuji--flickr_1k_test_image_text_retrieval/snapshots/c4a4cdc67b77fce148b45484a067957bf75ec4c3
base=/home/ldf/codex-workspaces/rag-partial-enrichment/models/qwen2.5-vl-3b
old=/data/ldf/reprforge/external/colqwen2.5-v0.1-d8bacd9
new=/home/ldf/codex-workspaces/rag-latent-bridge/models/colqwen2.5-v0.2
projection=/home/ldf/codex-workspaces/rag-latent-bridge/models/colqwen2.5-v0.2/base-shard2-header.bin
if [ "${smoke}" != "0" ]; then output=${root}/smoke-output; else output=${root}/output; fi
test ! -e "${output}"
CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
"${python}" "${root}/code/run_codec_frontier.py" \
  --protocol "${root}/code/protocol.json" \
  --matrix-code-root "${matrix}" --support-code-root "${support}" \
  --arxivqa-parquet "${arxiv}" --docvqa-parquet "${docvqa}" --flickr-root "${flickr}" \
  --base-model "${base}" --processor-model "${base}" \
  --old-adapter "${old}" --new-adapter "${new}" --projection "${projection}" \
  --output-root "${output}" --device cuda:0 --image-batch-size 1 --query-batch-size 32 \
  --smoke "${smoke}" --corpora arxivqa docvqa flickr
