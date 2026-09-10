#!/usr/bin/env bash
# usage: run-extra-a100.sh <gpu> batch4|mlp
set -uo pipefail
gpu=${1:?gpu}; job=${2:?job}
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
common="--matrix-code-root $matrix --support-code-root $support --arxivqa-parquet $arxiv --docvqa-parquet $docvqa --flickr-root $flickr --base-model $base --processor-model $base --old-adapter $old --new-adapter $new --projection $projection --device cuda:0"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $job in
  batch4) $python $root/code/run_codec_frontier.py --protocol $root/code/protocol-batch4.json $common --output-root $root/output-batch4 --image-batch-size 4 --query-batch-size 32 --corpora arxivqa docvqa flickr ;;
  mlp)    $python $root/code/indomain_mlp_bridge.py $common --output-root $root/output-mlp --corpora arxivqa docvqa flickr ;;
esac
echo "EXTRA-DONE $job $?"
