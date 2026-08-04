# Workload-Quantized Residual Index Contract

## Method hypothesis

A fixed pooled cover is a robust but lossy document representation. Raw
workload witnesses repair that loss on HR but overfit Finance, while restricting
witnesses to fit-workload Top-K pages amplifies collection-specific competition
structure. The next compiler therefore quantizes the **query-token
distribution**, rather than memorizing query--page pairs.

For fit-workload query-token directions `Q`, deterministic spherical k-means
produces `P` unit probes `C = {c_1, ..., c_P}` without qrels. For document full
tokens `D`, pool-9 cover `B(D)`, and probe `c`, define

```text
r(c,D) = max_{d in D} c dot d - max_{b in B(D)} c dot b.
```

When `r(c,D) > epsilon`, the compiler appends one full token attaining the
first maximum. Each physical page is its pool-9 cover plus the union of its
distinct residual winners. Its residual cardinality is page-specific and
bounded above by `P`.

For a compiled probe, the same one-sided guarantee as the raw residual index
holds:

```text
M_hybrid(c,D) >= M_full(c,D) - epsilon.
```

For a unit query token `q` and nearest unit probe `c`, normalized document
vectors give a Lipschitz extension:

```text
M_hybrid(q,D) >= M_full(q,D) - epsilon - 2 ||q-c||_2.
```

This is a correctness statement, not yet a useful worst-case certificate: the
observed 32-probe quantization radii are loose. The paper must report their
distribution and rely on held-out listwise retrieval for the quality claim.

## Development evidence

HR and Finance were inspected while developing the method. The shared
`P=32, epsilon=0.05` point was selected because it has nearly identical
behavior on both collections:

| Collection | tokens/full | pool-9 nDCG@10 | probe residual | full | matched random |
|---|---:|---:|---:|---:|---:|
| HR | 11.94% | 0.4931 | 0.5065 | 0.5178 | 0.4934 |
| Finance-EN | 11.98% | 0.4503 | 0.4622 | 0.4732 | 0.4473 |

On both collections, paired-query bootstrap says the probe index is
significantly above pool-9 and statistically indistinguishable from full for
nDCG@10. Recall@100 is statistically indistinguishable from full. These are
development results, not a sealed transfer.

## Frozen Computer Science transfer

The following protocol is frozen before deriving or evaluating the
Computer Science pool-9 bank:

- dataset: `vidore/vidore_v3_computer_science`, all 1,360 public pages and
  215 public queries;
- exact ColPali v1.1 full/query embeddings and hierarchical pool factor 9;
- five deterministic SHA-256 query-ID folds;
- `P=32` deterministic spherical probes fitted independently within each
  outer fold, 20 Lloyd iterations;
- residual threshold `epsilon=0.05`, one full winner per eligible probe/page,
  no mandatory residual tokens;
- qrels excluded from compilation and used only by final metrics;
- matched-random full tokens at exactly the same per-page cardinality;
- nDCG@5, nDCG@10, Recall@100, token fraction and paired-query bootstrap;
- no Computer Science-specific threshold, probe count, reranking depth,
  query type or relevance feature may be introduced after observing results.

The transfer supports continued paper development only if all are true:

1. persistent token fraction is at most 13% of full;
2. nDCG@10 improves pool-9 by at least 0.01 or has a paired 95% interval above
   zero;
3. nDCG@10 is within 0.015 absolute of full or statistically
   indistinguishable from full;
4. the probe index beats its matched-random residual control in nDCG@10;
5. Recall@100 loses no more than 0.01 absolute from the better of full/pool-9.

Failure rejects the current fixed-probe method as a three-domain claim. No
post-hoc Computer Science sweep may be used to rescue it.

## Sealed outcome

The transfer failed. At 11.97% of full tokens, the frozen probe residual index
obtained 0.7058 nDCG@10 versus 0.7027 pool-9, 0.7170 full and 0.7034 matched
random. The pool-9 delta was +0.0031 with paired 95% CI
[-0.0069, 0.0132]; the full delta was -0.0112 with CI
[-0.0223, -0.0001]. Recall@100 was 0.9627 versus 0.9700 pool-9. Conditions 2
and 4 were not supported, so the fixed-probe method is rejected as a
three-domain paper claim.

Post-failure diagnostics are not transfer results. Raw residual witnesses,
64/128 probes and a scalar residual score sketch all failed to close the gap.

## Novelty boundary and required baselines

The contribution cannot be generic token pruning, token merging, spherical
clustering, or decoupling. Closest baselines include:

- Light-ColPali/Light-ColQwen2 query-agnostic token merging;
- the ACL 2026 Prune-then-Merge adaptive pruning and hierarchical merging
  pipeline;
- OmniColPress attention-guided clustering with learned universal query
  tokens and a fixed output budget;
- SIGIR 2026 uniform-sphere Voronoi influence pruning;
- raw empirical residual witnesses, hard Top-K boundary witnesses and
  matched random residuals;
- full ColPali, pool-9 and text-only late interaction.

The prospective novelty is a qrel-free workload-distribution quantizer used
to compile a pooled cover plus page-specific exact residual overlay, with
cross-fit and physical-index evaluation. It must beat modern merging methods
or expose a distinct workload-adaptation/heterogeneous-capacity tradeoff to be
paper-worthy.
