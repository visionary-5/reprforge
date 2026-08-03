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
- `quality/baseline-transfer.json`: three-role quality comparison of pool-25,
  exact two-stage reranking, static visual upgrades, and query-scoped
  activation.

## Systems

- `candidate-fusion/`: two-dataset BM25/visual candidate-relative fusion
  quality, representation work, and frozen trace digests.
- `systems/cohort-compiler-hr.json`: official HR full-visual, synchronous
  no-reuse, batch-1 resident, and batch-8 resident A100 comparison.
- `systems/cohort-compiler-finance.json`: held-out Finance-EN batch-8 resident
  transfer against the official full-visual A100 baseline.
- `systems/cohort-trace-parity.json`: Top-100 online/offline rank parity on HR
  and Finance-EN, with raw replay digests.
- `systems/irpapers-transfer.json`: independent 3,230-page IRPAPERS BM25,
  full-visual, static-fusion, resident K=10/K=20 and oracle-headroom result.
- `systems/representation-view-control-plane.json`: deterministic IRPAPERS- and
  ViDoRe-v3-scale metadata lifecycle measurements; no quality claim.
- `systems/complementary-view-v3.json`: source-paper-disjoint V3 objective,
  exact-oracle, solver and predictive-validity audit on IRPAPERS.

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
- `systems/public-baseline-comparison/versioned-suite.json`: adds pooled
  candidate generation plus exact full-vector reranking at K=10/20/50.
- `systems/lifecycle-a100/`: no-cache pageable/pinned host transfer,
  globally-active cache, query-scoped activation, and three pinned-host
  stability repetitions.

Machine-specific source paths have been removed from these summaries.
