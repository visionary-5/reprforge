# Frozen result summaries

This directory contains compact, reviewable summaries used by the current
documentation. Raw datasets, score banks, embeddings, and physical indexes are
not committed.

## Quality

- `quality/mmdocir-protocol.json`: frozen 30-document role assignment and
  dataset contract.
- `quality/typed-capacity-v1.json`: sealed Typed-Capacity V1 evaluation,
  including per-document and aggregate metrics.
- `quality/pool25-visual-transfer.json`: role-separated transfer result for a
  pool-25 base with full visual capacity on visual layouts.

## Systems

- `systems/candidate-scaling-summary.json`: 1×/4×/16× physical candidate
  scaling summary.
- `systems/reprforge-token-budget.json`: compressed-index token-work latency.
- `systems/full-token-budget.json`: full-index token-work latency.
- `systems/*-correctness.json`: fixed-batch versus token-work score and Top-k
  equivalence.
- `systems/versioned-visual-smoke.json`: real embedding-bank correctness smoke
  for the immutable text base, visual delta, cache-hit, and rollback contract.
- `systems/public-mmdocir-a100-final/versioned-suite.json`: final public
  MMDocIR A100 comparison of text, full visual, pool-25, fixed hybrid,
  equivalent compiled plans, and versioned base-plus-delta execution.

Machine-specific source paths have been removed from these summaries.
