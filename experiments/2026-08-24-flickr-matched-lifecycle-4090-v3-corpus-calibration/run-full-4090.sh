#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/reprforge
code=${root}/code/flickr-matched-lifecycle-4090-v3-corpus-calibration
python=${root}/env/bin/python
output=${root}/artifacts/flickr-matched-lifecycle-4090-v3-corpus-calibration
upstream=${root}/code/MarginMerge

test "$(sha256sum "${code}/protocol.json" | cut -d' ' -f1)" = 55f05fc6314c26203c5ded09cb5e9bb9bf40b3f3f14633f2c0adfd9be5f01458
test "$(sha256sum "${code}/run_matched_lifecycle.py" | cut -d' ' -f1)" = fdb0a73e6fecd7be595453e7f06dfa8f2be1993090895f7e887945a6038934c5
test "$(sha256sum "${code}/analyze.py" | cut -d' ' -f1)" = b7d0e45432704a6f0d167d79cd7459ad98d95080f2801dbe11de26300cef7f51
test "$(git -C "${upstream}" rev-parse HEAD)" = 94cdafbc5e4c17d17aaeb0c5cdf3f0238462b42a
test ! -e "${output}"

used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${used}" -gt 100 ]; then
  echo "GPU0 is not free: ${used} MiB" >&2
  exit 73
fi
available_kb=$(df --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
if [ "${available_kb}" -lt 52428800 ]; then
  echo "less than 50 GiB free on /root/autodl-tmp" >&2
  exit 74
fi

HF_HOME=${root}/hf-cache \
HF_DATASETS_CACHE=${root}/hf-cache/datasets \
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
"${python}" "${code}/run_matched_lifecycle.py" \
  --protocol "${code}/protocol.json" \
  --output-root "${output}" \
  --dataset-root "${root}/data/flickr" \
  --base-model "${root}/models/colqwen2.5-base" \
  --processor-model "${root}/models/colqwen2.5-v0.2" \
  --v01-adapter "${root}/models/colqwen2.5-v0.1" \
  --v02-adapter "${root}/models/colqwen2.5-v0.2" \
  --pca256 "${root}/artifacts/corpus-calibrated-pca4090-v1/flickr/basis.pt" \
  --support-code-root "${root}/code" \
  --image-batch-size 8

CUDA_VISIBLE_DEVICES=0 \
CATTS_CACHE="${output}/raw-cache" \
CATTS_RES="${output}/raw-baselines" \
PYTHONPATH="${upstream}/src" \
"${python}" "${upstream}/src/baselines_memory.py" flickr

CUDA_VISIBLE_DEVICES=0 \
CATTS_CACHE="${output}/replay-cache" \
CATTS_RES="${output}/replay-baselines" \
PYTHONPATH="${upstream}/src" \
"${python}" "${upstream}/src/baselines_memory.py" flickr

"${python}" "${code}/analyze.py" \
  --protocol "${code}/protocol.json" \
  --build-result "${output}/result.json" \
  --raw-baseline "${output}/raw-baselines/ckpt/flickr.json" \
  --replay-baseline "${output}/replay-baselines/ckpt/flickr.json" \
  --output "${output}/analysis-output"
