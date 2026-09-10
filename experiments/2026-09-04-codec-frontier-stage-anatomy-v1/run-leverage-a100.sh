#!/usr/bin/env bash
set -uo pipefail
gpu=${1:?gpu}
root=/data/ldf/reprforge/scratch/codec-frontier-stage-anatomy-v1
python=/data/ldf/reprforge/envs/reprforge-py311/bin/python
matrix=/data/ldf/reprforge/scratch/sigir-version-evolution-matrix-v1/code
arxiv=/home/ldf/.cache/huggingface/hub/datasets--vidore--arxivqa_test_subsampled/snapshots/b8a106812c8682bab08935cf5d1b4566c82562de/data/test-00000-of-00001.parquet
docvqa=/home/ldf/.cache/huggingface/hub/datasets--vidore--docvqa_test_subsampled/snapshots/49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6/data/test-00000-of-00001.parquet
flickr=/home/ldf/.cache/huggingface/hub/datasets--nlphuji--flickr_1k_test_image_text_retrieval/snapshots/c4a4cdc67b77fce148b45484a067957bf75ec4c3
common="--matrix-code-root $matrix --arxivqa-parquet $arxiv --docvqa-parquet $docvqa --flickr-root $flickr --pages 40 --device cuda:0"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
mkdir -p $root/leverage-output
$python $root/code/stage_leverage.py --family colpali \
  --base-model /data/ldf/reprforge/models/colpaligemma-3b-mix-448-base-verified-b4d673f \
  --adapter /data/ldf/reprforge/models/colpali-v1.1 $common --output $root/leverage-output/colpali.json
$python $root/code/stage_leverage.py --family colsmol \
  --base-model /data/ldf/reprforge/models/colsmol-500m-base-650243e9 \
  --adapter /data/ldf/reprforge/models/colsmol-500m-adapter-1aa9325c $common --output $root/leverage-output/colsmol.json
$python $root/code/stage_leverage.py --family colqwen2.5 \
  --base-model /home/ldf/codex-workspaces/rag-partial-enrichment/models/qwen2.5-vl-3b \
  --adapter /home/ldf/codex-workspaces/rag-latent-bridge/models/colqwen2.5-v0.2 \
  --processor-model /home/ldf/codex-workspaces/rag-partial-enrichment/models/qwen2.5-vl-3b $common --output $root/leverage-output/colqwen2.5.json
echo LEVERAGE-DONE
