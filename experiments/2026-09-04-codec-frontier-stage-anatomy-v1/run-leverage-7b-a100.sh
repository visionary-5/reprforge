#!/usr/bin/env bash
# Timing-only leverage on Qwen2.5-VL-7B with the ColNomic-7B adapter (architecture identical; not a legality claim).
set -uo pipefail
gpu=${1:?gpu}
root=/data/ldf/reprforge/scratch/codec-frontier-stage-anatomy-v1
python=/data/ldf/reprforge/envs/reprforge-py311/bin/python
matrix=/data/ldf/reprforge/scratch/sigir-version-evolution-matrix-v1/code
hf=/home/ldf/.cache/huggingface/hub
arxiv=$hf/datasets--vidore--arxivqa_test_subsampled/snapshots/b8a106812c8682bab08935cf5d1b4566c82562de/data/test-00000-of-00001.parquet
docvqa=$hf/datasets--vidore--docvqa_test_subsampled/snapshots/49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6/data/test-00000-of-00001.parquet
flickr=$hf/datasets--nlphuji--flickr_1k_test_image_text_retrieval/snapshots/c4a4cdc67b77fce148b45484a067957bf75ec4c3
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
$python $root/code/stage_leverage.py --family colqwen2.5 \
  --base-model /home/ldf/codex-workspaces/rag-partial-enrichment/models/qwen2.5-vl-7b \
  --adapter /data/ldf/reprforge/external/third-party-adapters/colnomic-7b \
  --processor-model /home/ldf/codex-workspaces/rag-partial-enrichment/models/qwen2.5-vl-7b \
  --matrix-code-root $matrix --arxivqa-parquet $arxiv --docvqa-parquet $docvqa --flickr-root $flickr --pages 40 --device cuda:0 \
  --output $root/leverage-output/colqwen2.5-7b.json
echo "LEVERAGE7B-DONE $?"
