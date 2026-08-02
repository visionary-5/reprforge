# Online Cohort Compiler Result

## Decision

ReprForge now executes candidate-relative fusion as a real online system.  It
is no longer only an offline score replay.  The minimal system builds a BM25
locator, turns a request batch into query-specific visual cohorts, constructs
only missing ColPali representations, scores a query--candidate-union matrix,
and publishes reusable visual state after the complete request batch succeeds.

The first complete A100 result is positive but narrower than the original
"asynchronous compiler" hypothesis:

- candidate-relative fusion and query-driven persistence improve the measured
  quality--time point over full visual prebuild;
- physical reuse, not microbatch deduplication, explains most of the speedup;
- batching improves aggregate throughput modestly while substantially
  increasing synchronous batch-completion latency;
- no background worker, admission algorithm, or temporal-workload claim has
  been established.

This is a working system milestone, not yet a complete paper contribution.

## What the system does

```text
Markdown corpus
      |
      v
 deterministic BM25 locator (built once)
      |
query batch -- Top-20 candidate cohorts
      |             |
      |             +-- resident visual hits
      |             +-- visual misses
      |                         |
      v                         v
deduplicated cohort union --> batched ColPali construction
      |                         |
      +----------- staged generation
                         |
query embeddings --> query x union MaxSim
                         |
candidate-local z(BM25) + z(visual)
                         |
fused candidates + untouched BM25 tail --> Top-100
                         |
publish staged visual state only after the batch succeeds
```

The resident store is host-resident float32 ColPali state in the current
backend, not persistent GPU HBM.  Only pages in the current query's Top-20 are
logically activated, even if other pages remain physically cached.  A failed
request batch does not expose a partial generation.

Two execution modes make the mechanism measurable:

- `bm25-fusion-sync`: one query at a time, no reuse;
- `bm25-fusion-batched`: configurable request batching and either no cache or
  an unbounded resident cache.

The compiler intentionally remains synchronous at the request-batch boundary.
Calling it asynchronous would hide the cold work that the current system still
makes queries wait for.

## Correctness

CPU contract tests cover synchronous/batched rank equivalence, candidate-union
deduplication, cross-call reuse, request-batch atomicity, deterministic BM25
tail order, and failure rollback.  The repository currently passes 71 tests,
with seven environment-dependent tests skipped locally.

A separate replay tool rebuilt BM25 from the public Markdown and replayed the
frozen ColPali score surface through the online compiler.  It compared every
Top-100 identifier, not only aggregate quality:

| Dataset | Queries | Corpus pages | Request batch | Top-100 mismatches |
|---|---:|---:|---:|---:|
| ViDoRe v3 HR | 318 | 1,110 | 8 | **0** |
| ViDoRe v3 Finance-EN | 309 | 2,942 | 8 | **0** |

Real bf16 re-encoding is not bitwise invariant to request batching.  On HR,
the observed nDCG@10 span across sync, batch-1 resident, and batch-8 resident
is 0.00148.  Frozen-score rank parity and real-model numerical reproducibility
are therefore reported separately.

## A100 result

The hardware is one NVIDIA A100-SXM4-80GB.  The model is the frozen ColPali
v1.1 adapter over ColPaliGemma-3B.  All modes use image encode batch 4 and
MaxSim scoring batch 16.  Model loading and dataset I/O are outside the
official ViDoRe indexing/search timers for every mode.

### ViDoRe v3 HR

| Execution | nDCG@10 | Recall@100 | Visual pages encoded | End-to-end | Mean search/query |
|---|---:|---:|---:|---:|---:|
| Full visual prebuild | 0.51778 | 0.87459 | 1,110 | 108.74 s | 25.19 ms |
| Sync, no reuse | 0.53666 | 0.85213 | 6,360 page-events | 590.86 s | 1,857.35 ms |
| Batch 1, resident | 0.53581 | 0.85213 | 895 unique pages | 104.78 s | 328.81 ms |
| Batch 8, resident | **0.53729** | 0.85213 | 895 unique pages | **98.03 s** | **307.56 ms** |

Batch-8 resident is 1.11x faster end-to-end than full visual and improves
nDCG@10 by 0.01951 absolute (3.77% relative).  It does not dominate every
quality metric: Recall@100 is 0.02245 lower than full visual.  Relative to the
synchronous no-reuse path it is 6.04x faster in search, but batch-1 resident is
already 5.65x faster.  The incremental batch-8 speedup over batch-1 resident
is only 1.07x, below the registered 1.10x mechanism gate.

The execution traces explain why:

- persistence removes 5,465 repeated visual page constructions;
- batch-8 cohort union removes only 11.1% of candidate events within batches;
- both resident variants end with 895 pages and 471,987,200 bytes, 80.6% of
  the corpus and 80.6% of the full visual vector bytes;
- batch-8 reduces visual encoder calls from 199 to 40 and batches query
  encoding, but it does not reduce the final resident set.

The latency trade-off is not hidden.  Batch-1 resident has P95 completion
1.39 s and returns the first query after 2.75 s.  Batch-8 resident has P95
batch completion 9.04 s and returns its first eight queries after 15.54 s.
The reported 307.56 ms/query is throughput, not independent request latency.

There is nevertheless a useful cold-start result.  Full visual becomes ready
after 100.73 s of indexing.  By that same deadline, batch-8 resident has
already returned all 318 queries and materialized 895 pages; batch-1 resident
has returned 263 queries.  This is the concrete benefit of progressive
construction, distinct from a steady-state latency claim.

### ViDoRe v3 Finance-EN

The transfer uses the same K=20, fusion equation, request batch 8, and
resident policy as HR; no Finance labels were used to alter the mechanism.

| Execution | nDCG@10 | Recall@100 | Visual pages encoded | End-to-end | Mean search/query |
|---|---:|---:|---:|---:|---:|
| Full visual prebuild | 0.47316 | 0.82554 | 2,942 | 320.83 s | 40.24 ms |
| Batch 8, resident | **0.56280** | **0.88045** | 1,855 | **190.50 s** | 614.54 ms |

The online system is 1.68x faster end-to-end and improves nDCG@10 by 0.08964
absolute (18.95% relative).  Unlike HR, Recall@100 also improves by 0.05491.
It stores 978,252,800 bytes of visual state, 63.1% of the 1,551,493,120-byte
full visual index.  Within-batch deduplication is only 6.5%, while cross-batch
resident hits account for 69.0% of candidate events; this independently
supports the HR conclusion that persistence, not cohort overlap inside one
batch, is the main resource lever.

The latency limitation transfers too: first-batch completion is 15.98 s and
P95 batch completion is 9.13 s.  However, all 309 queries finish by 190.50 s,
well before the full visual index becomes ready at 308.39 s.

## Failed canonical-packing hypothesis

An observed 0.00148 nDCG span motivated one bounded diagnostic: pad every
visual construction call to a multiple of four, so a short final image
microbatch could not change the bf16 kernel shape.  On HR this encoded 69
additional padding pages (+7.7%), increased end-to-end time from 98.03 s to
102.96 s, and left batch-8 nDCG@10 exactly unchanged at 0.537292.  The
hypothesis was rejected and the padding mechanism was removed from the system.

The remaining small quality variation tracks query batching.  Request batch
size is therefore part of the reproducibility contract rather than an
invisible performance knob.

## What is established

The current evidence supports this bounded claim:

> Query-generated candidate cohorts can be compiled into reusable visual
> state while preserving candidate-relative fusion semantics.  On a cold
> public query stream, the resulting progressive index can reach a better
> Top-10 quality point before a full visual index finishes building.

It does not establish that BM25, normalized fusion, caching, or batching is a
new algorithm.  It does not establish an asynchronous serving policy, an
optimal cache, a production P99 benefit, or a temporal workload advantage.

## Design review

The implementation changes the next research question.  More batching is not
the priority: it failed the standalone 1.10x mechanism gate and worsens
completion latency.  The material bottleneck is that admit-on-first-touch
still grows to 63--81% of the corpus and makes cold requests wait for ColPali.

The next system mechanism should target one of two measured gaps:

1. **bounded admission/persistence:** retain only cohorts whose expected reuse
   amortizes construction and storage, evaluated on a defensible public query
   episode rather than arbitrary ViDoRe order;
2. **serve-then-refine:** immediately return the BM25 result, compile visual
   evidence in the background, and measure time-to-quality plus revision
   semantics instead of pretending cold visual work disappeared.

Both require a new execution contract and strong transient/full-prebuild
baselines.  Neither should be implemented until the current public traces can
express query arrival or repeated workload episodes.

## Artifacts

- `reprforge/cohort_compiler.py`: online compiler and batch trace;
- `reprforge/vidore_pipeline.py`: official pipeline modes;
- `reprforge/cohort_trace_parity.py`: frozen online/offline rank verifier;
- `reprforge/cohort_benchmark_summary.py`: compact official-result analyzer;
- `results/systems/cohort-compiler-hr.json`: frozen HR summary and raw-result
  digests.
- `results/systems/cohort-compiler-finance.json`: frozen Finance transfer
  summary and raw-result digests.
- `results/systems/cohort-trace-parity.json`: two-dataset Top-100 parity and
  raw replay digests.

Large official JSON files, images, models, and score matrices remain outside
Git.
