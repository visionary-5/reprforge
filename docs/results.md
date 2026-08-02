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

## Versioned visual delta smoke

A real 781-item embedding bank was compiled into a 20.57 MB immutable text
base. Materializing three visual items created a 1.58 MB version-1 delta.
Requesting two already cached items created no new version.

For one real query embedding:

- all three active delta scores exactly matched direct image MaxSim;
- all 778 untouched item scores exactly matched the text base;
- switching the active pointer from version 1 to 0 and back to 1 preserved the
  expected cache state.

This is a physical-state and score-correctness milestone. It is not yet a GPU
latency result: the smoke used the NumPy reference runtime and copied existing
route embeddings rather than running the visual model.

## Online candidate-relative cohort compiler

The fixed online policy is BM25 Top-20 followed by candidate-local normalized
BM25/ColPali fusion.  A resident compiler constructs each touched visual page
once and activates it only for queries whose BM25 cohort contains that page.

| Dataset | Full visual nDCG@10 | ReprForge nDCG@10 | Full visual end-to-end | ReprForge end-to-end | Visual pages built |
|---|---:|---:|---:|---:|---:|
| ViDoRe v3 HR | 0.5178 | **0.5373** | 108.74 s | **98.03 s** | 895 / 1,110 |
| ViDoRe v3 Finance-EN | 0.4732 | **0.5628** | 320.83 s | **190.50 s** | 1,855 / 2,942 |

On HR, synchronous no-reuse takes 590.86 seconds and batch-1 resident takes
104.78 seconds.  Thus reuse explains most of the final 6.04x search speedup;
batch-8 adds only 1.07x over batch-1 resident and misses its 1.10x standalone
gate.  Batch-8 P95 completion remains 9.04 seconds, so the result is a
cold-stream time-to-quality improvement, not a low-latency serving claim.

Frozen-score replay produces zero Top-100 mismatches on all 318 HR and 309
Finance queries.  Real bf16 execution has a 0.00148 nDCG@10 span across HR
batching policies, so request batch size is part of the execution contract.

Machine-readable summaries are indexed in `results/README.md`.
