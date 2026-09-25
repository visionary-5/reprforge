# Reproduce the experiments

Install the [GPU environment](../docs/reproduction.md), prepare the public inputs,
and run the relevant evaluation below. New runs produce their own outputs. Selected historical summaries are supplied
in the accompanying supplementary archive, outside Git; see the [paper-to-code map](../docs/paper-map.md) for coverage.

| Directory | Evaluation |
|---|---|
| [adapter_census](adapter_census/) | Changed components in public retriever adapters |
| [quality](quality/) | Retrieval quality and target agreement after upgrades |
| [reconstruction](reconstruction/) | Independent exactness checks and rebuild timing |
| [storage](storage/) | BF16, INT8 and PCA storage–fidelity trade-offs |
| [ablation](ablation/) | State design, target reconstruction and invalidation |
| [paper](paper/) | Released-pair audit, architecture discovery and lifecycle |
| [final_pass](final_pass/) | Complete manual interfaces, HR quality and validation timing |

`support/` contains shared model and scoring utilities. Keep numerical settings,
page selection, calibration splits, warmup and timing scope from the selected
protocol when comparing runs. Model weights, datasets and generated results
should be written outside the repository.
