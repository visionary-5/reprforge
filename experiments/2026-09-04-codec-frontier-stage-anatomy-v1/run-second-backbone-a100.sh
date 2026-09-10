#!/usr/bin/env bash
# usage: run-second-backbone-a100.sh <gpu> colqwen2|infovqa
set -uo pipefail
gpu=${1:?gpu}; job=${2:?job}
root=/data/ldf/reprforge/scratch/codec-frontier-stage-anatomy-v1
python=/data/ldf/reprforge/envs/reprforge-py311/bin/python
matrix=/data/ldf/reprforge/scratch/sigir-version-evolution-matrix-v1/code
support=/data/ldf/reprforge/scratch/mmdocir-versioned-ir-transfer-v1/code
hf=/home/ldf/.cache/huggingface/hub
arxiv=$hf/datasets--vidore--arxivqa_test_subsampled/snapshots/b8a106812c8682bab08935cf5d1b4566c82562de/data/test-00000-of-00001.parquet
docvqa=$hf/datasets--vidore--docvqa_test_subsampled/snapshots/49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6/data/test-00000-of-00001.parquet
flickr=$hf/datasets--nlphuji--flickr_1k_test_image_text_retrieval/snapshots/c4a4cdc67b77fce148b45484a067957bf75ec4c3
infovqa=$(ls $hf/datasets--vidore--infovqa_test_subsampled/snapshots/*/data/test-00000-of-00001.parquet | head -1)
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $job in
  colqwen2)
    M=/data/ldf/reprforge/experiments/conditioned-transport-a100-20260815/models
    A=/data/ldf/reprforge/scratch/colqwen2-release-transition-v1/models
    $python $root/code/run_codec_frontier.py --protocol $root/code/protocol-colqwen2.json \
      --matrix-code-root $matrix --support-code-root $support \
      --arxivqa-parquet $arxiv --docvqa-parquet $docvqa --flickr-root $flickr \
      --family colqwen2 --projection-format safetensors \
      --base-model $M/Qwen2-VL-2B-Instruct --processor-model $M/Qwen2-VL-2B-Instruct \
      --old-adapter $A/source --new-adapter $A/target --projection $M/colqwen2-base/custom_text_proj.safetensors \
      --output-root $root/output-colqwen2 --device cuda:0 --image-batch-size 1 --query-batch-size 32 \
      --corpora arxivqa docvqa flickr ;;
  infovqa)
    base=/home/ldf/codex-workspaces/rag-partial-enrichment/models/qwen2.5-vl-3b
    $python $root/code/run_codec_frontier.py --protocol $root/code/protocol-infovqa.json \
      --matrix-code-root $matrix --support-code-root $support \
      --arxivqa-parquet $arxiv --docvqa-parquet $docvqa --flickr-root $flickr --infovqa-parquet $infovqa \
      --base-model $base --processor-model $base \
      --old-adapter /data/ldf/reprforge/external/colqwen2.5-v0.1-d8bacd9 \
      --new-adapter /home/ldf/codex-workspaces/rag-latent-bridge/models/colqwen2.5-v0.2 \
      --projection /home/ldf/codex-workspaces/rag-latent-bridge/models/colqwen2.5-v0.2/base-shard2-header.bin \
      --output-root $root/output-infovqa --device cuda:0 --image-batch-size 1 --query-batch-size 32 \
      --corpora infovqa ;;
esac
echo "SECOND-DONE $job $?"
