#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/reprforge
code=${root}/code/docvqa-matched-lifecycle-4090-v3-corpus-calibration
python=${root}/env/bin/python
output=${root}/artifacts/docvqa-matched-lifecycle-4090-v3-corpus-calibration-smoke
upstream=${root}/code/MarginMerge

test "$(sha256sum "${code}/protocol.json" | cut -d' ' -f1)" = fd2b3431170bc88749074523eb3e66c2337e23f78766cdc84737705a780d601b
test "$(sha256sum "${code}/run_matched_lifecycle.py" | cut -d' ' -f1)" = 554c605c50e214d6cf88b6c641325119a7fc8cc55bb3ba876da557182179350a
test "$(sha256sum "${code}/analyze.py" | cut -d' ' -f1)" = c58d0aee5ca0d182b10c09d5a7fe8f38d5c1c897a1adecf5456e810ebc4749cc
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
  --dataset-parquet "${root}/data/docvqa/data/test-00000-of-00001.parquet" \
  --base-model "${root}/models/colqwen2.5-base" \
  --processor-model "${root}/models/colqwen2.5-v0.2" \
  --v01-adapter "${root}/models/colqwen2.5-v0.1" \
  --v02-adapter "${root}/models/colqwen2.5-v0.2" \
  --pca256 "${root}/artifacts/corpus-calibrated-pca4090-v1/docvqa/basis.pt" \
  --support-code-root "${root}/code" \
  --limit 4 \
  --image-batch-size 1
