# Use ReprForge

The public command-line integration currently supports the ColQwen2.5
retrievers evaluated in the paper. ReprForge's state and validation interfaces
are model-independent, but each model family needs an explicit capture,
continuation and dependency contract.

Install the GPU environment and prepare the verified ColQwen2.5 configuration
as described in [reproduction](../docs/reproduction.md). Create a `pages.json`
list of image paths, relative to that JSON file. Use an available GPU and new
output directories:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_source_index.py \
  --config /data/reprforge-inputs/endpoint-config.json \
  --pages /data/pages.json --output /data/source-index
CUDA_VISIBLE_DEVICES=0 python scripts/replay_target.py \
  --config /data/reprforge-inputs/endpoint-config.json \
  --source-index /data/source-index --target vidore-v0.2 \
  --output /data/target-index
python scripts/evaluate.py \
  --reference /data/raw-target/representations.pt \
  --candidate /data/target-index/representations.pt \
  --output /data/comparison.json
```

The index is a logical ordered list of page token matrices; these commands do
not construct a physical ANN index. Keep the raw page files for fallback. The
source command captures the visual prefix once and runs the source continuation
before persisting states and final representations. These commands are usage
entry points, not timing benchmarks.

Target reuse requires matching loaded visual weights, processor/tokenizer,
model configuration, package versions, continuation code and page content.
Changed or unavailable state files fall back to raw target encoding. The
conservative contract may reject harmless changes; it does not certify arbitrary
model programs. Full config changes also reject replay.

`evaluate.py` checks representations against a separately generated raw target
bank in identical page order. For an automatically constructed independent raw
reference, ordered top-k checks and measured timings, run
[`experiments/reconstruction`](../experiments/reconstruction/). For relevance
metrics and query evaluation, use [`experiments/quality`](../experiments/quality/).
