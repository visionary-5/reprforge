# Frozen result summaries

This directory contains compact, reviewable summaries used by the current
documentation. Raw datasets, score banks, embeddings, and physical indexes are
not committed.

## Quality

- `quality/mmdocir-protocol.json`: frozen 30-document role assignment and
  dataset contract.
- `quality/typed-capacity-v1.json`: sealed Typed-Capacity V1 evaluation,
  including per-document and aggregate metrics.

## Systems

- `systems/candidate-scaling-summary.json`: 1×/4×/16× physical candidate
  scaling summary.
- `systems/reprforge-token-budget.json`: compressed-index token-work latency.
- `systems/full-token-budget.json`: full-index token-work latency.
- `systems/*-correctness.json`: fixed-batch versus token-work score and Top-k
  equivalence.

Machine-specific source paths have been removed from these summaries.
