# Retrieval quality

Compare the target reference, stale source index, representation bridges,
partial refresh and retained-state replay on the same gallery. Prepare inputs
with the [reconstruction guide](../../docs/reproduction.md). Copy the `collections`
and `targets` objects from the generated endpoint config into separate JSON files,
so their paths point to your downloaded inputs.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/quality/pooled_upgrade_matrix.py \
  --protocol experiments/quality/protocol.json \
  --matrix-code-root experiments/support --support-code-root experiments/support \
  --collections /data/collections.json --targets /data/targets.json \
  --base-model /data/reprforge-inputs/qwen2.5-vl-3b \
  --processor-model /data/reprforge-inputs/qwen2.5-vl-3b \
  --deployed-adapter /data/reprforge-inputs/colqwen2.5-v0.1 \
  --deployed-projection /data/reprforge-inputs/vidore-v0.2/base-shard2-header.bin \
  --max-pixels 12845056 --output-root /data/quality-run
python experiments/quality/analyze.py /data/quality-run --output /data/quality-metrics
```

For the additional MMDocIR distractor gallery, prepare the pinned public pages
with `experiments/prepare_mmdocir_distractors.py` and pass `--distractor-parquet`.
Record the resulting gallery size; omitting those pages changes the benchmark.

The analyzer uses stable target rankings and excludes Shift queries while
retaining its pages. The reference shares a visual prefix; native raw correctness
is evaluated in `reconstruction/`. Quality labels use a single-page relevance
proxy, not official graded benchmark qrels. `official_upgrade/` supplies the
ColQwen2 release-pair configuration for `--family colqwen2`.

Bridge fitting in this runner skips empty query text. Historical paper bridge
baselines were fitted before that filter; their exact values may differ. This
change does not alter the replay definition.
