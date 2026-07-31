# Official ViDoRe v3 Integration

## Upstream contract

The integration is pinned against
[`illuin-tech/vidore-benchmark`](https://github.com/illuin-tech/vidore-benchmark)
commit `a70f23af8bb3b33efe8a4a6c6c15a6e2d978035e` (audited
2026-07-31).  ReprForge implements the official complex-pipeline contract:

- `index(corpus_ids, corpus_images, corpus_texts, dataset_name)` receives each
  page as both a PIL image and Markdown;
- `search(query_ids, queries)` returns pytrec-eval score dictionaries plus an
  optional cost dictionary;
- the unmodified evaluator reports indexing time, search time, nDCG, recall,
  precision, MAP, and reciprocal rank.

The upstream repository is not vendored.  Its commit is recorded here so an
evaluation result identifies the exact external evaluator.

When a compute node cannot reach Hugging Face, `reprforge.vidore_local_eval`
loads the three official Parquet files locally and passes their unmodified
query/corpus/qrel semantics to the same upstream `evaluate_retrieval` and
`aggregate_results` functions.  Every result records the three source
SHA-256 digests.  This transport fallback is not a replacement evaluator.

For oracle/replay work, add `--score-trace-dir /path/to/trace`. This is
supported only for `text` and `visual` modes. It writes `runtime.npz` with the
complete query--corpus score surface and per-page costs, plus a separate
`oracle-labels.npz` containing qrels. The separation is deliberate: qrels are
not runtime-visible policy features. See
[`progressive-visual-index-contract.md`](progressive-visual-index-contract.md)
for the frozen oracle and online replay semantics.

## What is being tested

ViDoRe is the first external full-corpus test of ReprForge's representation
lifecycle.  It is not a replay of the MMDocIR within-document candidate pool.
The official corpus forces the system to locate pages globally.

`reprforge.vidore_pipeline.ReprForgeViDoRePipeline` exposes five policies:

| Mode | Build-time representation | Query-time action | What it isolates |
|---|---|---|---|
| `text` | Markdown for every page | score text | cheap-build quality floor |
| `visual` | full visual for every page | score visual | quality/build-cost reference |
| `visual-pool` | full visual, then token pool | score pooled visual | storage and MaxSim savings only |
| `two-stage` | Markdown for every page | text Top-K, encode/rerank images | no-cache deferred ingestion |
| `tiered-selective` | Markdown for every page | text Top-K, cache visual, query-scoped score replacement | ReprForge lifecycle design |

The distinction between `visual-pool` and selective modes is essential:
pooling does **not** avoid the original visual encoder call.  Only `text`,
`two-stage`, and `tiered-selective` avoid full-corpus visual materialization
during indexing.

## Cost fields

The pipeline returns the official ranking plus:

- model load time (outside the official index timer);
- representation construction time measured inside `index`;
- compact vector count and bytes;
- visual materializations during index and search;
- visual encoder call count;
- candidate and full-visual score-pair count;
- cache hits, misses, current/peak items, and current/peak vector bytes.

Official evaluator timing and ReprForge's decomposed counters must both be
retained.  Neither is a substitute for the other.

## First experiment contract

Start with `vidore/vidore_v3_hr`, English queries, one idle A100, and the
existing local MMDocIR ColPali checkpoint.  Run a bounded correctness slice
before a complete dataset.  The first complete comparison is:

1. `text`;
2. `visual`;
3. `visual-pool` with factor 25;
4. `two-stage` with candidate K in `{10, 20, 50}`;
5. `tiered-selective` with the same K and unbounded cache.

Report official nDCG@10 and Recall@10/100 together with build time, search
time, index bytes, visual materializations, and cache hit rate.

This is a transfer test, not a preregistered positive claim.  The mechanism is
supported only if at least one selective setting:

- avoids at least 80% of index-time visual materializations;
- retains at least 95% of the full-visual nDCG@10;
- reduces end-to-end time for a stated query horizon; and
- beats the no-cache two-stage baseline on repeated-query or update replay
  without increasing logical score activation.

If Markdown cannot locate visually relevant pages, the result identifies a
locator bottleneck rather than validating the lifecycle policy.
