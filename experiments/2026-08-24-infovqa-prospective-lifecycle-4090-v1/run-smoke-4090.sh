#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/reprforge
code=${root}/code/infovqa-prospective-lifecycle-4090-v1
python=${root}/env/bin/python
output=${root}/artifacts/infovqa-prospective-lifecycle-4090-v1-smoke
upstream=${root}/code/MarginMerge

test "$(sha256sum "${code}/protocol.json" | cut -d' ' -f1)" = 15e90952943756cdcc3a235e6b38fc2793d356797165ee48405d03a21e258f11
test "$(sha256sum "${code}/run_matched_lifecycle.py" | cut -d' ' -f1)" = f92af35c707427d4834febd61d45492a1db819b0c321ed53e8af64ebf9486c03
test "$(sha256sum "${code}/analyze.py" | cut -d' ' -f1)" = 0330853d3f30a8bd3d0a013e50762de9cd359f4dacf864c095d02daa9693b2cc
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
  --dataset-parquet "${root}/data/infovqa/data/test-00000-of-00001.parquet" \
  --base-model "${root}/models/colqwen2.5-base" \
  --processor-model "${root}/models/colqwen2.5-v0.2" \
  --v01-adapter "${root}/models/colqwen2.5-v0.1" \
  --v02-adapter "${root}/models/colqwen2.5-v0.2" \
  --pca256 "${root}/artifacts/infovqa-corpus-pca256-calibration-4090-v1/basis.pt" \
  --support-code-root "${root}/code" \
  --limit 4 \
  --image-batch-size 1
