# Baseline Comparison and First Design Revision

## Outcome

The comparison changed the system design.

The initial versioned runtime made every physically cached full-visual
representation override the compressed base for every query. That coupling
is unnecessary and sometimes harmful: a full representation useful to one
query can become a distractor for another query in the same document.

The revised runtime separates:

```text
physical cache membership
    which full vectors are resident and reusable

logical query activation
    which resident full vectors override pool-25 for this query
```

This query-scoped activation preserves the encoding-reuse benefit of a
versioned cache while matching the ranking semantics of a query-specific
no-cache execution.

## Compared systems

1. Uniform pool-25: one compressed visual representation for every layout.
2. Uniform full visual: full ColPali-style vectors for every layout.
3. Pool-25 + exact rerank: the
   [Visual RAG Toolkit](https://arxiv.org/abs/2602.12510)-style baseline;
   pool-25 generates Top-K candidates and full MaxSim reranks them.
4. Static pool-25 + visual-full: pool-25 for nonvisual layouts and full
   capacity for every visual layout.
5. Preencoded no-cache: pool-25 is GPU resident; query-selected full vectors
   are gathered from pageable or pinned host memory, copied to GPU, scored,
   and discarded. This excludes image encoding and is therefore a lower
   bound, not a faithful end-to-end DVI measurement.
6. Versioned cache, globally active: cached full vectors always override the
   pool-25 base.
7. Versioned cache, query scoped: cached full vectors override the base only
   when the current structural locator activates them.

The no-cache framing is motivated by
[Deferred Visual Ingestion](https://arxiv.org/abs/2602.14162), but DVI sends
original images to a VLM at query time. ReprForge reports preencoded-vector
latency and replays visual encoding cost separately.

## Public quality transfer

Quality uses MMDocIR's official within-document candidate pools. The same
direction is observed on the prior, mechanism, and sealed final roles.

| Role | Pool-25 | Pool→full rerank K=10 | Static visual-full | Query-scoped K=10 |
|---|---:|---:|---:|---:|
| Prior development | 0.6303 | 0.6345 | 0.6839 | **0.6990** |
| Mechanism design | 0.5323 | 0.5336 | **0.6122** | 0.5793 |
| Sealed final | 0.5311 | 0.5782 | 0.5865 | **0.5992** |

The query-scoped mechanism beats the two-stage K=10 baseline on all three
roles. It does not beat static visual-full on mechanism design, so there is no
claim of universal quality dominance. Hotset replay also contains episodes
where activating full vectors hurts relative to staying pool-25. Admission
still needs a utility gate.

### Hotset counterexample

The failure is systematic enough to shape the next design. With K=10,
query-scoped activation changes nDCG@10 relative to pool-25 by:

| Role | Hottest 25% repeated | Hottest 10% repeated |
|---|---:|---:|
| Prior development | +0.032 | -0.058 |
| Mechanism design | +0.051 | -0.058 |
| Sealed final | +0.006 | -0.039 |

Therefore neither visual content type nor a high structural-locator score is
sufficient evidence that full activation helps. The workload can repeatedly
target a region where the compressed representation is already better.
Caching solves acquisition reuse; it does not solve representation utility.

## A100 retrieval result

Hardware and measurement contract:

- one NVIDIA A100-SXM4-80GB;
- PyTorch 2.5.1, CUDA 12.4, float32;
- 781 public layouts and 46 queries;
- 5 warmup passes and 20 measured repetitions;
- physical search over all persisted bank items;
- PDF decoding and model encoding excluded.

| System | P50 | P95 | Stored/resident representation |
|---|---:|---:|---:|
| Uniform pool-25 | 0.497 ms | 0.510 ms | 16.4 MB GPU |
| Uniform full visual | 2.411 ms | 2.438 ms | 411.9 MB GPU |
| Pool→full rerank K=10 | 0.616 ms | 0.633 ms | 428.3 MB GPU |
| Static pool-25 + visual-full | 0.984 ms | 1.003 ms | 99.9 MB GPU |
| Versioned, all cached active | 1.092 ms | 1.113 ms | 86.5 MB GPU for the K=10 cache union |
| Versioned, query-scoped active | **0.659 ms** | **0.680 ms** | same 86.5 MB GPU |

The query-scoped and no-cache executions have identical result digests. The
new mechanism is 1.66x faster than globally activating the same cache because
it avoids scoring unrelated cached full vectors.

The pinned-host no-cache baseline is CPU/NUMA sensitive on the shared server.
Across all observed runs its P50 ranged from 0.987 to 2.741 ms; three isolated
repetitions produced 2.322, 2.325, and 2.741 ms. Query-scoped resident
activation was stable at 0.644--0.660 ms in those repetitions. The paper
should report the distribution and host topology, not select the slowest
pinned run.

## Lifecycle result

For the prior-role cyclic-4 stream:

- 184 query steps;
- structural BM25 Top-10 visual activation;
- 133 unique visual layouts selected from 165 visual layouts.

| Policy | Full-vector encode calls | Replayed full encode time | Persistent GPU representation |
|---|---:|---:|---:|
| Deferred, no cache | 1,644 | 91.1 s | 16.4 MB + up to 5.3 MB transient |
| Versioned cache | 133 | 7.52 s | 86.5 MB |

Caching reduces repeated full-vector encoding calls by 12.36x and replayed
encoding time by 12.11x. However, a prebuilt pinned host index avoids those
encoding calls too, at the cost of building and retaining the complete
411.9 MB full-vector bank. The actual design space is therefore:

```text
full offline host index
    low GPU residency, high upfront build/storage

deferred re-encoding
    low upfront build, repeated query-time model cost

versioned selective cache
    incremental build, bounded resident delta, reuse across queries
```

## What this establishes

The result supports a concrete system mechanism rather than a generic
"dynamic representation" claim:

> Physical full-vector residency and logical representation activation should
> be separate decisions in a maintained multimodal index.

The mechanism has two measured effects:

1. physical residency removes repeated encoding and host-transfer work;
2. query-scoped activation avoids cache-induced ranking pollution and
   unnecessary full-vector MaxSim.

## Remaining contribution gap

The current structural locator decides *which visual items are plausible* but
not whether switching from pool-25 to full visual has positive utility. The
next algorithmic component must predict the sign and magnitude of that
intervention using deployable features, then jointly decide:

- activate an already resident full representation;
- admit a missing full representation into the versioned cache;
- leave the item on pool-25;
- evict a low-reuse representation under a memory budget.

The next public transfer should use the official
[ViDoRe v3 pipeline framework](https://github.com/illuin-tech/vidore-benchmark),
which supports multi-stage/hybrid pipelines and records indexing and search
compute time. MMDocRAG remains the later end-to-end answer-use benchmark.
