# Experiments

Start with the [reproduction guide](../docs/reproduction.md) for environments and
public inputs. Each workflow writes new outputs outside the checkout. The
[paper map](../docs/paper-map.md) identifies the scope of each experiment.

## Main paper workflows

| Directory | Evaluation |
|---|---|
| [paper](paper/) | Released-pair validity, interface discovery and lifecycle timing |
| [reconstruction](reconstruction/) | Independent exactness checks and rebuild timing |
| [final_pass](final_pass/) | Manual/automatic interfaces, native/common HR quality and validation cost |
| [ablation](ablation/) | State design, target reconstruction and invalidation |
| [adapter_census](adapter_census/) | Changed components in public retriever adapters |

## Auxiliary workflows

| Directory | Evaluation |
|---|---|
| [quality](quality/) | Earlier quality and target-agreement protocols |
| [storage](storage/) | Collection-specific BF16, INT8 and PCA storage–fidelity experiments |
| [reconstruction/official_upgrade](reconstruction/official_upgrade/) | Earlier ColQwen2 release-pair reconstruction |
| [reconstruction/large_scale](reconstruction/large_scale/) | Streaming reconstruction and serving-index utilities |

These auxiliary workflows have their own protocols; they do not replace the
final HR quality experiment or reproduce every appendix figure. `support/`
contains shared input, scoring and codec utilities. Preserve page selection,
processor budgets, calibration splits, warmup and timing scope when comparing
runs. Historical measurements are distributed with the paper supplement.
