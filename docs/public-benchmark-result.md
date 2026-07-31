# Public Benchmark Result: Versioned Visual Delta

## Result

The first physical A100 benchmark supports the *mechanism*, but not yet the
complete adaptive-policy claim.

The strongest current construction is:

```text
pool-25 representation for every MMDocIR layout
                    +
full visual representation for visual layouts in an immutable delta
```

On the public MMDocIR pilot bank (781 layouts, 46 queries), this construction
improves nDCG@10 from `0.6303` for uniform pool-25 and `0.6236` for uniform
full visual to `0.6839`. Its single-compiled form uses 99.9 MB and has 0.987 ms
P50 latency; uniform full visual uses 411.9 MB and has 2.434 ms P50 latency.
Thus it is 75.7% smaller, 2.46x faster, and 0.0603 absolute nDCG@10 better than
uniform full visual on this split.

The versioned base-plus-delta implementation reproduces all 46 full rankings
of the equivalent single-compiled plan exactly. It uses 103.4 MB and has
1.085 ms P50 / 1.106 ms P95 latency. The cost of independent base and delta
execution is therefore 9.8% P50 relative to the ideal compiled form, while it
remains 2.24x faster than uniform full visual.

## Benchmark contract

- Public workload: [MMDocIR](https://aclanthology.org/2025.emnlp-main.1576/),
  layout-level retrieval.
- Hardware: one NVIDIA A100-SXM4-80GB.
- Model artifacts: MMDocIR's published ColPali-compatible embeddings.
- Physical search: all 781 persisted layouts for every query.
- Quality: MMDocIR's official within-document candidate pools and overlap
  relevance.
- Runtime: PyTorch 2.5.1, CUDA 12.4, float32, token batch budget 65,536.
- Measurement: 5 full warmup passes followed by 20 repetitions, or 920 timed
  queries per system.
- Excluded: image decoding and model encoding. Reported create/publish times
  are embedding-copy and index-serialization costs, not end-to-end ingestion.

## A100 comparison

| System | nDCG@10 | R@5 | Compact bytes | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| Uniform text | 0.0797 | 0.0870 | 20.6 MB | 0.670 | 0.719 |
| Uniform full visual | 0.6236 | 0.6899 | 411.9 MB | 2.434 | 2.486 |
| Uniform pool-25 | 0.6303 | 0.6796 | 16.4 MB | 0.496 | 0.509 |
| MMDocIR fixed hybrid | 0.6362 | 0.7231 | 106.3 MB | 1.158 | 1.185 |
| Pool-25 + visual-full, compiled | **0.6839** | **0.7362** | 99.9 MB | 0.987 | 1.002 |
| Pool-25 base + visual delta | **0.6839** | **0.7362** | 103.4 MB | 1.085 | 1.106 |

All physical-score comparisons use the same query and item embeddings. The
versioned pool-25 construction has zero maximum absolute error and identical
full rankings relative to the equivalent compiled plan.

## Transfer over the frozen 30-document corpus

The representation effect has the same direction on all three roles frozen
before final evaluation:

| Role | Documents | Queries | Pool-25 nDCG@10 | Pool-25 + visual-full | Gain |
|---|---:|---:|---:|---:|---:|
| Prior development | 10 | 46 | 0.6303 | 0.6839 | +0.0536 |
| Mechanism design | 11 | 57 | 0.5323 | 0.6122 | +0.0799 |
| Sealed final evaluation | 9 | 38 | 0.5311 | 0.5865 | +0.0554 |

This is evidence that the quality effect is not confined to the A100 pilot
split. It does **not** show that upgrading every visual layout is the optimal
workload-aware policy.

## Strong baselines and collision boundary

- MMDocIR supplies the full-visual, text, and content-type hybrid baselines.
  Its paper reports that visual retrieval generally outperforms text retrieval.
- [Visual RAG Toolkit](https://arxiv.org/abs/2602.12510) is the closest static
  efficiency baseline: training-free spatial pooling for candidate generation
  followed by exact full-vector reranking. Its two-stage search must be added
  before making a state-of-the-art throughput claim.
- [Deferred Visual Ingestion](https://arxiv.org/abs/2602.14162) is the closest
  index-light baseline: structural/BM25 location followed by query-time visual
  reasoning. It is the correct no-cache/on-demand comparison, although its
  target is end-to-end document QA rather than late-interaction index
  maintenance.
- The official [ViDoRe pipeline framework](https://github.com/illuin-tech/vidore-benchmark)
  supports multi-stage and hybrid systems and records indexing/search compute
  time. ViDoRe v3 is the next independent public benchmark; MMDocRAG is the
  later end-to-end answer-use benchmark.

## What is established and what remains open

Established:

1. A compressed visual base is much stronger than a text base for this
   workload.
2. Selective full-capacity visual representations add consistent quality on
   three frozen MMDocIR roles.
3. An immutable base plus versioned visual delta can realize the exact target
   ranking with a measured 9.8% composition overhead.
4. Full visual ingestion is not a necessary default on this public workload.

Not established:

1. The current content-type rule knows visual labels in advance and upgrades
   all 165 visual layouts. It is not yet a workload-aware allocation policy.
2. The benchmark does not include live ColPali encoding, OCR, PDF parsing, or
   answer generation.
3. It does not yet compare against Visual RAG Toolkit's exact two-stage
   implementation or DVI's no-cache execution.
4. Only one persisted public embedding bank has been physically measured on
   A100; the 30-document quality transfer uses frozen score replay.

The next system experiment should therefore hold the pool-25 base fixed and
compare, under identical dynamic query traces: no upgrades, no-cache
full-visual reranking, static all-visual upgrades, versioned cached upgrades,
and pooled candidate generation plus exact reranking. The main question is no
longer whether compression works; it is whether a versioned workload-aware
system can decide which full representations are worth keeping and recover
the measured quality gain at lower cumulative ingestion and maintenance cost.
