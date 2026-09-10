#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/reprforge
code=${root}/code/arxivqa-matched-lifecycle-4090-v3-corpus-calibration
python=${root}/env/bin/python
output=${root}/artifacts/arxivqa-matched-lifecycle-4090-v3-corpus-calibration-smoke

test "$(sha256sum "${code}/protocol.json" | cut -d' ' -f1)" = 993180d908c3ddc10acaa96901ff2fb31217ad56fb664fd56e971f76ff8428ab
test "$(sha256sum "${code}/run_matched_lifecycle.py" | cut -d' ' -f1)" = d0211f6d63eb1f8d62fbda22a9ae22b81f7e16de43be6406b8c222198b98ace9
test "$(sha256sum "${code}/run_smoke.py" | cut -d' ' -f1)" = 1ef43e52b0db8c26d966f136b29d6cfc2cdf6f33286f486cd2fefacd88363a60
test ! -e "${output}"

used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${used}" -gt 100 ]; then
  echo "GPU0 is not free: ${used} MiB" >&2
  exit 73
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
  --image-batch-size 4 \
  --limit 16
