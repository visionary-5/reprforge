# Complete-collection reconstruction

Extend the independent endpoint to every unique image in the six prepared
ViDoRe evaluation collection files. Use a completed endpoint run as input:

```bash
python experiments/reconstruction/full_collection/prepare.py \
  --prior /data/reprforge-endpoint --output /data/reprforge-full-inputs
CUDA_VISIBLE_DEVICES=0 python experiments/reconstruction/full_collection/run.py \
  --config /data/reprforge-full-inputs/config.json \
  --protocol /data/reprforge-full-inputs/protocol.json \
  --output /data/reprforge-full
```

Preparation freezes page identities and the gallery size before encoding.
The setup is A100, BF16, batch 1, a common processor and the v0.2 target.
These are the selected evaluation splits, not the full original source datasets.

`verify.py` checks output completeness and recorded equality. `repeat_timing.py`
accepts a completed verification report and runs a separate repeated-timing
protocol. Use each script's `--help` for its required arguments. The default reconstruction protocol selects 200 pages; this runner uses all
unique pages from the supplied evaluation collection files.
