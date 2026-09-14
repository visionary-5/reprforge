# Streaming reconstruction and index construction

Reconstruct MMDocIR representations from retained PCA-256 states, then build a
serving index. Inputs are generated states and a trained PCA basis from a matching
encoding run; these are not included in Git.

```bash
python experiments/reconstruction/large_scale/replay_terminal.py --help
python experiments/reconstruction/large_scale/build_publish.py --help
```

Use `protocol.json` for the model and collection settings. Supply your own input
paths and new external output roots through the listed arguments. Measure raw
encoding separately under the same environment when comparing total costs.
Report replay, index construction and publication times separately.
