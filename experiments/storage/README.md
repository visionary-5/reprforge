# Retained-state storage

Compare BF16, INT8 and PCA using matched pages, calibration splits and processors.

```bash
python experiments/storage/run_codec_frontier.py --help
```

Pass the selected `protocol*.json`, local model/data paths, a new external
`--output-root`, and `experiments/support` as `--support-code-root`. The CLI lists
required arguments. Protocols specify page counts, calibration and A100 settings.
Outputs include bytes per page, target agreement, retrieval quality and stage
times. Stage timing excludes persistent state reads and writes.

For InfoVQA alone, use the inputs prepared by the reconstruction guide:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/storage/run_codec_frontier.py \
  --protocol experiments/storage/protocol-infovqa.json \
  --matrix-code-root experiments/support --support-code-root experiments/support \
  --corpora infovqa --infovqa-parquet /data/reprforge-inputs/datasets/infovqa/test.parquet \
  --base-model /data/reprforge-inputs/qwen2.5-vl-3b \
  --processor-model /data/reprforge-inputs/qwen2.5-vl-3b \
  --old-adapter /data/reprforge-inputs/colqwen2.5-v0.1 \
  --new-adapter /data/reprforge-inputs/vidore-v0.2 \
  --projection /data/reprforge-inputs/vidore-v0.2/base-shard2-header.bin \
  --output-root /data/reprforge-storage
```
