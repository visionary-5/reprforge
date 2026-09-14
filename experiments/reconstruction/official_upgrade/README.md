# Official ColQwen2 upgrade

Evaluate ColQwen2 v0.1 to v1.0 with its shipped processor contract.
Prepare the Energy slice using [data preparation](../data_preparation/), then
inspect checkpoint dependencies before GPU execution:

```bash
python experiments/reconstruction/official_upgrade/inspect_cpu.py --help
python experiments/reconstruction/official_upgrade/run_gpu.py --help
```

Pass the freshly generated CPU inspection as `--cpu-result`. The GPU runner
independently checks model, adapter, processor, projection and dataset hashes
against `gpu-protocol.json`. Use the certified successor protocol when preparing
the Energy Parquet container from public data. Write all outputs to new external
paths. The protocol specifies warmup, repeats, batch size and timing scope.
