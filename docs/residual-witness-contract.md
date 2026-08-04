# Residual Witness Index Contract

## Hypothesis

A pooled visual cover and a sparse set of workload-conditioned full-token
residual witnesses can preserve late-interaction retrieval quality at a small
fraction of the full persistent index. The mechanism is qrel-free and differs
from replacing the full representation with either pruning or pooling alone.

For document-token set `D`, pooled cover `P(D)`, and query token `q`, define

```text
M_full(q,D) = max_{d in D} q dot d
M_pool(q,D) = max_{p in P(D)} q dot p
residual(q,D) = M_full(q,D) - M_pool(q,D).
```

For every fit-workload `(q,D)` event with `residual(q,D) > epsilon`, the
compiler retains one full token attaining `M_full`. The physical page index is
the union of `P(D)` and the distinct retained full tokens.

This gives a deterministic fit-workload one-sided guarantee. For every query
token,

```text
M_hybrid(q,D) >= M_full(q,D) - epsilon,
```

and therefore for a query containing `L` valid tokens,

```text
S_hybrid(query,D) >= S_full(query,D) - L * epsilon.
```

The guarantee is deliberately one-sided. Pooled centroids can score above an
individual full token and may suppress or introduce different distractors;
final listwise quality remains an empirical held-out-workload question.

## Frozen transfer protocol

HR was the development collection. The following choices are frozen before
opening the Finance residual-witness results:

- full visual ColPali v1.1 document and query token embeddings;
- published hierarchical token pooling with factor 9 as the cover;
- five deterministic SHA-256 query-ID folds;
- qrels excluded from compilation and accepted only by final metrics;
- one retained full winner for every eligible query-token/document event;
- minimum winner frequency 1 and no mandatory full residual tokens;
- residual thresholds `{0.12, 0.15, 0.20, 0.30}` transferred unchanged;
- matched-random full tokens at the exact per-document residual cardinality;
- nDCG@5, nDCG@10, Recall@100, persistent token fraction, and scoring time;
- paired-query bootstrap with 4,000 resamples against full and pool-9.

No Finance-specific threshold, candidate depth, page type, relevance feature,
or calibration may be introduced after observing the transfer.

## Transfer gate

The method remains a paper candidate only if Finance has at least one point at
or below 20% of full tokens that satisfies all of the following:

1. nDCG@10 is statistically above pool-9 or improves it by at least 0.01;
2. nDCG@10 is within 0.01 absolute of full, with a paired interval reported;
3. the witness index beats its matched-random residual control;
4. Recall@100 loses no more than 0.01 absolute from the better of full/pool-9.

Failure does not invalidate the HR phenomenon, but it rejects a general
two-dataset method claim. Passing authorizes the next systems stage: compile a
contiguous base-plus-residual index and measure actual bytes, build/compile
time, MaxSim latency, update cost, and version rollback.

## Required prior-art baselines

- pool-9 and the complete Light-ColPali/Light-ColQwen2 capacity curve;
- uniform random tokens at matched total and per-page capacities;
- workload-wide empirical winners without a pooled cover;
- uniform-sphere Voronoi pruning from SIGIR 2026;
- full ColPali and text-only late interaction;
- query-time Col-Bandit as an orthogonal scoring optimization when code is
  available.

The paper claim must remain about workload-conditioned pooled residuals and
heterogeneous physical cardinality, not token pruning in general.
