# Audit, tracing and lifecycle experiments

Use the [reproduction guide](../../docs/reproduction.md) for environments and
input preparation, and the [paper map](../../docs/paper-map.md) for coverage.
The directory names below retain the original experiment identifiers.

| Directory | Entry | Purpose |
|---|---|---|
| `item1/` | `sweep_pairs.py` | Released source–target pairs, loaded dependencies, page-input checks and independent full encoding |
| `item3/` | `trace_colqwen25.py`, `trace_colpali.py`, `trace_qwen3_2b.py`, `trace_qwen3_4b.py` | Executed-path visual interface and dependency discovery |
| `e2e/` | `e2e_rebuild.py`, `analyze_e2e.py` | Fixed-contract lifecycle measurements and aggregate analysis |

Configuration templates describe the required local model and corpus layout.
`item1/sweep_config.json` and `e2e/config.template.json` can be materialized
with `scripts/materialize_config.py`; this resolves paths but does not download
or validate their inputs. Qwen3 probes require their documented model-specific
integrations; the 2B producer/probe modules are not bundled.

The audit's historical `native` result field uses the release processor when it
loads, or the family-base processor on a recorded loading failure. The paper
calls successful pairs on this route **Direct**. The common-configuration route
is evaluated separately. See [protocol notes](../../docs/protocol-notes.md).

The lifecycle runner uses the fixed ColQwen2.5 integration. Generic validation
and manual/automatic interface comparisons are in
[`final_pass/`](../final_pass/); their timings belong to separate experiments.
