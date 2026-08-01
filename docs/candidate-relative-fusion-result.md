# Candidate-Relative Fusion Result

## Decision

ReprForge now has a concrete positive algorithm, but not yet a complete paper
system.  The algorithm is a fixed, query-driven two-stage path:

1. BM25 over public Markdown retrieves 20 candidates;
2. only those candidates receive a full visual ColPali representation;
3. BM25 and visual scores are standardized **within the same 20-page
   cohort**;
4. the two standardized scores are added and the cohort is reranked.

This is called **candidate-relative normalized fusion**.  It uses no learned
router, qrels, future queries, or per-page content classifier.  `K=20` is held
fixed across both datasets.

The important insight is that representation allocation and score semantics
cannot be separated.  A visual upgrade changes the competitors in a query's
ranked set.  Its value is therefore a property of a candidate cohort, not an
independent property of a page.

## Public A100 result

The model is the frozen ColPali v1.1 adapter over the public
ColPaliGemma-3B base.  Visual traces were measured on one A100-SXM4-80GB.
The BM25 locator was built on the same server CPU at low priority.  HR uses
318 English queries and 1,110 pages; Finance-EN uses 309 base English queries
and 2,942 pages.

| Dataset | BM25 | Full visual | Visual-only rerank K=20 | RRF K=20 | Normalized fusion K=20 |
|---|---:|---:|---:|---:|---:|
| ViDoRe v3 HR | 0.4860 | 0.5178 | 0.5175 | 0.5294 | **0.5373** |
| ViDoRe v3 Finance-EN | 0.5285 | 0.4732 | 0.5045 | 0.5486 | **0.5628** |

The reported quality metric is official nDCG@10.  Relative to the best
single representation, normalized fusion improves HR by 0.0195 absolute
(3.77% relative) and Finance-EN by 0.0343 absolute (6.49% relative).  It also
beats rank-only RRF on both datasets, so the gain is not explained only by
candidate union.

## Cost and the remaining bottleneck

| Dataset | BM25 build | Full visual build | Unique K=20 visual pages | Visual work | BM25 + visual build-equivalent | Ratio to full visual |
|---|---:|---:|---:|---:|---:|---:|
| HR | 0.19 s | 100.73 s | 895 / 1,110 (80.6%) | 80.5% | 80.33 s | 0.797 |
| Finance-EN | 0.56 s | 308.39 s | 1,855 / 2,942 (63.1%) | 62.3% | 190.15 s | 0.617 |

The build-equivalent column is not a measured online wall clock. It adds
measured server-side BM25 construction to the measured
per-page visual encode time for every page first touched in the complete
query stream.  It excludes query scoring and assumes each visual page is
retained without eviction.

This establishes a quality--work improvement, but it also exposes the next
system challenge.  If the current query waits for every missing visual page,
the estimated first-touch visual work has P95 1.18 s on HR and 1.70 s on
Finance-EN.  Unconditional persistence eventually covers 81% and 63% of the
corpora.  Thus the current implementation is neither a low-latency transient
cascade nor a sufficiently selective persistent index.

## Why the earlier utility model was rejected

The first attempted controller assigned an exact delta-nDCG label to every
individual page intervention.  It failed for a structural reason:

- at K=20, the actual cohort delta is +0.00347 per query, while the sum of
  individual deltas is -0.03114;
- mean absolute interaction is 0.1584;
- 76.7% of queries have absolute interaction above 0.01;
- cohort and additive signs disagree on 20.1% of queries;
- the held-out linear intervention model has only 0.0145 Pearson correlation
  with exact page utility.

A query-text k-nearest-neighbour controller was also rejected.  Across five
held-out folds it reaches 0.5091 nDCG@10, below the best fixed cohort at
0.5133 and full visual at 0.5178.  A more complex selector did not solve the
incorrect independent-page abstraction.

## What is and is not a contribution yet

The positive result is real and reproducible: a cheap text locator plus
cohort-calibrated heterogeneous evidence beats either representation alone
on two public datasets while touching less than the full visual corpus.
However, BM25 candidate generation, z-score fusion, RRF, and visual reranking
are individually known techniques.  These two datasets alone do not make the
combination a paper contribution.

The prospective system contribution is narrower:

> compile visual representations in query-generated cohorts, calibrate them
> in the cohort where they will compete, and schedule their construction so
> the quality gain does not appear as multi-second cold-query latency.

The next design must compare synchronous transient refinement, full visual
prebuild, unconditional caching, simple admission, and an asynchronous
cohort compiler under the same quality and GPU-time contract.  It becomes a
paper direction only if the compiler improves time-to-quality or end-to-end
cost without hiding the first-touch work.  ViDoRe has no natural query
timestamps, so temporal or workload-drift claims still require an additional
public trace.

## Reproducibility

The implementation is in `reprforge/bm25_locator.py` and
`reprforge/candidate_fusion.py`.  Compact result summaries are under
`results/candidate-fusion/`.  Images, models, Parquet data, and full score
matrices remain outside Git; committed hashes bind the summaries to the
frozen traces.
