# State design and validation

Compare visual-state replay, a source-fused-input baseline and incomplete
position reconstruction against independent target raw encoding.

First run the [reconstruction experiment](../reconstruction/). Then:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ablation/run.py \
  --prior /data/reprforge-reconstruction --output /data/reprforge-ablation --pages 200
```

The page count cannot exceed the prior gallery. Outputs include tensor errors,
validator decisions, parameter hashes and independent raw fallback checks.
Text, vision and merger interventions are controlled counterfactuals, not public
model releases. The fused baseline reconstructs source embeddings in the target
process; it does not test an independently persisted source-fused artifact.
