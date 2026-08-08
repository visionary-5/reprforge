# Frozen result summaries

Start with [`CURRENT.md`](CURRENT.md). It identifies the one compact summary
that supports the active materialization direction; the catalog below is an
audit record and includes many historical or negative experiments.

This directory contains compact, reviewable summaries used by the current
documentation. Raw datasets, score banks, embeddings, and physical indexes are
not committed.

## Compression-risk protocol

- `compression-risk/hr-pool9-v1.json`: configuration-level pool-9 risk,
  ranking fidelity, tail harm, safety gate, and physical cost on ViDoRe v3 HR.
- `compression-risk/finance-en-pool9-v1.json`: the aligned FinanceEN result.
- `compression-risk/computer-science-pool9-v1.json`: the aligned Computer
  Science result.
- `compression-risk/{hr,finance-en,computer-science}-pool{4,25}-v1.json`: the
  denser 24.95% and aggressive 3.98% semantic hierarchical-pooling capacity
  points. Pool-4 passes the frozen safety gate only on Computer Science;
  pool-25 passes none.
- `compression-risk/pharmaceuticals-pool{4,9,25}-v1.json`: the first held-out
  validation curve. The frozen qrel-free gate accepts only pool-4 and the
  later relevance safety gate makes the same three decisions.
- `compression-risk/industrial-pool{4,9,25}-v1.json`: the second held-out
  validation curve. The frozen gate abstains on all three rates and all three
  later fail the relevance safety gate.
- `compression-risk/certificates/pharmaceuticals-pool{4,9,25}-v1.json`: the
  pre-qrel ranking decisions; `pharmaceuticals-label-materialization-v1.json`
  binds their hashes to the later oracle-label artifact.
- `compression-risk/certificates/industrial-pool{4,9,25}-v1.json`: the aligned
  Industrial pre-qrel decisions; `industrial-label-materialization-v1.json`
  binds their hashes before evaluation.
- `compression-risk/validation-gate-transfer-v1.json`: cross-collection
  summary over six held-out configurations: 6/6 decision matches, zero
  false-safe decisions, and 62.48% selected macro resident fraction versus
  100% for the best fixed safe state.
- `systems/pharmaceuticals-{full-surface,pool4-bank,pool9-bank,pool25-bank}-v1.json`:
  full-bank construction and semantic-pooling physical artifacts.
- `systems/industrial-{full-surface,pool4-bank,pool9-bank,pool25-bank}-v1.json`:
  the aligned Industrial physical artifacts.

The frozen definitions and interpretation are in
`docs/compression-risk-contract.md` and
`docs/compression-risk-baseline-result.md`. Qrels are evaluation-only; positive
regret means the compressed candidate is worse than full.

## Phenomenon diagnostics

- `diagnostics/heterogeneity-atlas-v0.json`: global, static-document,
  query-route, and interaction-level representation headroom on three frozen
  score surfaces.
- `diagnostics/query-route-probe-v2-grouped.json`: query-only,
  cheap-retrieval, candidate-identity, and source-document connectivity
  probes. Label-using utility targets are cross-fitted and diagnostic only.
- `diagnostics/landmark-probe-v0.json`: rejected query-local
  landmark-completion budget curve.
- `diagnostics/cohort-certificates-v4.json`: exact set/order certificate
  headroom on HR, Finance, IRPAPERS and MMDocIR, including comparable Top-5
  results and measured/attributed physical costs.
- `diagnostics/cohort-selectors-v1-active.json`: query-holdout ridge and learned
  active-acquisition probes; scalar cheap-surface selection is rejected as the
  main method.
- `diagnostics/landmark-probe-v1-top5.json`: four-dataset Top-5 coverage and
  boundary-acquisition curves.
- `diagnostics/workload-compiler-v2-selected.json`: qrel-free fit-workload
  physical-plan selection and held-out quality/cost results.
- `diagnostics/workload-compiler-v3-stability.json`: leave-one-state-out
  abstention control; it reduces low-budget harm but is too conservative to be
  the final safety mechanism.
- `diagnostics/cohort-selectors-v3-nonlinear.json`: nonlinear semantic
  candidate selector on the original query holdout; its apparent two-domain
  gain motivates, but does not survive, stricter source-document analysis.
- `diagnostics/cohort-selectors-v4-group-crossfit.json`: source-document-
  disjoint selector audit; semantic selectors do not consistently beat fixed
  Top-B and are rejected as the main contribution.
- `diagnostics/workload-compiler-v4-greedy.json`: direct listwise teacher-
  fidelity greedy search; near-perfect fit fidelity but weak held-out quality
  demonstrates document-identity overfitting.
- `diagnostics/ladder-error-bound-expansion30-v1.json`: five-state MMDocIR
  ladder and rejected full-teacher error-bounded compiler.
- `diagnostics/type-policy-compiler-expansion30-v2.json`: exact 625-policy,
  query- and source-document-cross-fitted type-to-state compiler with paired
  bootstrap comparisons to uniform full and pool-9.
- `diagnostics/group-policy-compiler-expansion30-v1.json`: rejected entropy/
  edge feature-group extension; it is dominated by the simpler type policy.
- `diagnostics/hr-pool9-transfer-analysis.json` and
  `diagnostics/finance-pool9-transfer-analysis.json`: full-corpus evidence that
  pool-9 is an 11.07%-byte high-recall visual substrate, plus its post-hoc
  pooling construction-time penalty.
- `diagnostics/hr-token-witness-crossfit-v1.json`: qrel-free empirical MaxSim
  witness pruning and matched-random control.
- `diagnostics/hr-competitive-token-witness-v1.json`: rejected Top-K-only
  witness compiler; removing a corpus-wide coarse cover harms unseen queries.
- `diagnostics/hr-residual-token-witness-v1.json` and
  `diagnostics/hr-residual-token-witness-high-epsilon-v1.json`: HR pool-9 cover
  plus sparse full-token residual witness Pareto curves.
- `diagnostics/hr-residual-token-witness-significance-v1.json`: paired-query
  bootstrap comparisons to uniform full and pool-9.
- `diagnostics/finance-residual-token-witness-transfer-v1.json` and
  `diagnostics/finance-residual-token-witness-significance-v1.json`: frozen
  Finance transfer that rejects the raw residual method's general gate.
- `diagnostics/hr-boundary-residual-token-witness-exploratory-v1.json` and
  `diagnostics/finance-boundary-residual-token-witness-exploratory-v1.json`:
  hard Top-K boundary residuals; strong HR behavior does not transfer.
- `diagnostics/hr-probe-residual-exploratory-v1.json` and
  `diagnostics/finance-probe-residual-exploratory-v1.json`: development-domain
  spherical workload-probe residual curves and matched-random controls.
- `diagnostics/cs-probe-residual-frozen-v1.json` and
  `diagnostics/cs-probe-residual-significance-frozen-v1.json`: sealed
  Computer Science transfer that rejects the fixed 32-probe method.
- `diagnostics/cs-raw-residual-diagnostic-v1.json`,
  `diagnostics/cs-adaptive-probe-count-diagnostic-v1.json`, and
  `diagnostics/cs-probe-residual-score-sketch*-v1.json`: post-failure
  diagnostics; none rescues the third-domain result.
- `diagnostics/pool-score-error-v1.json`: three-domain full-minus-pool score
  decomposition, rejecting pooled-score overshoot as the failure cause.
- `diagnostics/physical-compression-{hr,finance,cs}-v5-pareto.json`: the
  development query-holdout curve for the parameter-free boundary coreset and
  residual-affine dual-view scorer. The 65%-anchor point passes all three
  domains at about 89.9% of full persistent document-vector bytes.
- `diagnostics/physical-compression-cs-v6-materialized.json`: post-
  materialization Computer Science parity, quality, tail-risk, certificate,
  and byte result for the real pool4-cover plus full-anchor banks.
- `compression-risk/physical-plan-{hr,finance,cs}-v1.json`: strict three-fold
  qrel-free cross-certificates and final refit plans for the development
  physical compiler.
- `compression-risk/physical-plan-industrial-v2.json` and
  `physical-plan-pharmaceuticals-v1.json`: frozen validation plans written
  without qrels. Industrial v1 is intentionally not committed because its
  fp32-runtime/fp16-bank cost normalization was invalid; v2 preserves the
  identical anchor selection with the corrected physical byte denominator.
- `compression-risk/certificates/physical-{industrial,pharmaceuticals}-v1.json`:
  reserved-query runtime certificates written before physical-method label
  evaluation. Both pass the qrel-free gate.
- `diagnostics/physical-compression-{industrial,pharmaceuticals}-v1-validation.json`:
  frozen validation evaluations. Both narrowly fail relevance safety despite
  passing the qrel-free certificate, establishing a 2/2 false-safe result.

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
- `systems/hr-pool9-full-result.json` and
  `systems/finance-pool9-full-result.json`: measured A100 full-corpus pool-9
  construction, storage, query and trace manifests.
- `systems/cs-full-witness-result.json`: third-domain Computer Science full
  bank and trace manifest; raw embeddings remain on the A100 data volume.
- `systems/cs-dual-view-anchor-bank-v1.json`: measured 884-anchor Computer
  Science bank; combined with the pool-4 cover it occupies 322,570,240 bytes,
  or 89.951% of full.
- `systems/{industrial,pharmaceuticals}-dual-view-anchor-bank-v1.json`: actual
  validation anchor banks and measured combined byte fractions (89.940% and
  89.932% of full).

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
