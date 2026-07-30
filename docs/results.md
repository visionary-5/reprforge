# Current Results

All values below are frozen A100 or MMDocIR results. They are evidence for the
current prototype, not a final paper claim.

## Sealed representation allocation

The sealed split has 9 documents and 38 evaluable queries.

| Plan | Vector bytes | R@1 | R@5 | R@10 | nDCG@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform pool-25 | 41.61 MB | 0.277 | 0.573 | 0.715 | 0.480 | 0.531 |
| Typed-Capacity V1 | 59.46 MB | **0.386** | 0.585 | 0.693 | **0.539** | **0.578** |
| uniform pool-9 | 115.69 MB | 0.349 | **0.595** | **0.703** | 0.527 | 0.572 |
| uniform pool-4 | 260.80 MB | 0.375 | 0.583 | 0.676 | 0.527 | 0.564 |

Typed-Capacity V1 uses 51.4% of uniform pool-9 bytes. Relative to pool-9:

- Recall@1: +0.0370;
- Recall@5: -0.0099;
- Recall@10: -0.0097;
- nDCG@5: +0.0123;
- nDCG@10: +0.0061.

The preregistered storage gate passed and the Recall@5 regression gate passed
narrowly. The required `+0.01` nDCG@10 gain did not pass. Document-macro
nDCG@10 is -0.0031, so this is a positive query-weighted trade-off rather than
stable cross-document dominance.

## Mechanism evidence

Exact one-layout interventions on the mechanism-design split distinguish:

- evidence recovery;
- evidence loss;
- distractor inflation;
- distractor suppression.

Full visual capacity helps some tables and evidence-bearing figures but can
hurt ordinary text regions by increasing distractors. Aggressive pooling can
suppress distractors but loses evidence in other layouts. Nearest-token
embedding cover loss has weak rank correlation with intervention utility
(maximum observed Spearman correlation about 0.10), so it was rejected as a
planner feature.

This motivates an evidence–risk model tied to workload utility rather than a
generic compression-fidelity model.

## Candidate scaling

The systems-only scaling probe physically materializes 1×, 4×, and 16×
candidate sets. At 16× there are 12,496 candidates.

With fixed 64-document batching:

| Index | Vector storage | Batches | P50 |
| --- | ---: | ---: | ---: |
| compressed V0 | 695.07 MB | 196 | 35.10 ms |
| full visual | 6.59 GB | 196 | 35.62 ms |

Storage compression alone does not create latency improvement because both
indexes launch the same number of padded batches.

## Token-work scheduling

The frozen token budget is 65,920 padded vectors, equal to 64 full ColPali
layouts of 1,030 vectors each.

| Index | Batches | P50 | P95 | QPS |
| --- | ---: | ---: | ---: | ---: |
| compressed V0, fixed-64 | 196 | 35.10 ms | — | 28.37 |
| compressed V0, token budget | **23** | **9.03 ms** | **9.67 ms** | **109.5** |
| full visual, token budget | 196 | 36.28 ms | 38.62 ms | 27.3 |

Token-work scheduling makes the compressed index 3.89× faster than its own
fixed-batch execution and 4.02× faster than full visual execution. Across 46
queries, fixed and token-work scheduling have identical Top-10 results; the
maximum absolute score difference is \(3.81\times10^{-6}\).

Machine-readable summaries are indexed in `results/README.md`.
