#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/reprforge
code=${root}/code/arxivqa-matched-lifecycle-4090-v3-corpus-calibration
python=${root}/env/bin/python
output=${root}/artifacts/arxivqa-matched-lifecycle-4090-v3-corpus-calibration
upstream=${root}/code/MarginMerge

test "$(sha256sum "${code}/protocol.json" | cut -d' ' -f1)" = 993180d908c3ddc10acaa96901ff2fb31217ad56fb664fd56e971f76ff8428ab
test "$(sha256sum "${code}/run_matched_lifecycle.py" | cut -d' ' -f1)" = d0211f6d63eb1f8d62fbda22a9ae22b81f7e16de43be6406b8c222198b98ace9
test "$(sha256sum "${code}/analyze.py" | cut -d' ' -f1)" = 4790306aa8581cbde2d74a91ae900f3261cb5328b798a60b5cee6913f10f2d6d
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
  --dataset-parquet "${root}/data/arxivqa/data/test-00000-of-00001.parquet" \
  --base-model "${root}/models/colqwen2.5-base" \
  --processor-model "${root}/models/colqwen2.5-v0.2" \
  --v01-adapter "${root}/models/colqwen2.5-v0.1" \
  --v02-adapter "${root}/models/colqwen2.5-v0.2" \
  --pca256 "${root}/artifacts/corpus-calibrated-pca4090-v1/arxivqa/basis.pt" \
  --support-code-root "${root}/code" \
  --image-batch-size 4

CUDA_VISIBLE_DEVICES=0 \
CATTS_CACHE="${output}/raw-cache" \
CATTS_RES="${output}/raw-baselines" \
PYTHONPATH="${upstream}/src" \
"${python}" "${upstream}/src/baselines_memory.py" arxivqa

CUDA_VISIBLE_DEVICES=0 \
CATTS_CACHE="${output}/replay-cache" \
CATTS_RES="${output}/replay-baselines" \
PYTHONPATH="${upstream}/src" \
"${python}" "${upstream}/src/baselines_memory.py" arxivqa

"${python}" "${code}/analyze.py" \
  --protocol "${code}/protocol.json" \
  --build-result "${output}/result.json" \
  --raw-baseline "${output}/raw-baselines/ckpt/arxivqa.json" \
  --replay-baseline "${output}/replay-baselines/ckpt/arxivqa.json" \
  --output "${output}/analysis-output"
