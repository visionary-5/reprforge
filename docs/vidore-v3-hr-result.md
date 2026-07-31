# ViDoRe v3 HR Transfer Result

## Outcome

The official ViDoRe v3 HR consumer confirms the problem but rejects the
current K=20 policy.

On 318 English queries over 1,110 pages, full visual indexing improves
nDCG@10 by 0.023 absolute over the Markdown representation, but costs 3.06x
as much corrected build time and 1.86x as many compact vector bytes.  This is a real
quality--construction-cost gap on a public full-corpus benchmark.

The current `tiered-selective` design does not close that gap.  It uses
Markdown Top-20 candidates to trigger visual encoding, caches the result, and
replaces Markdown scores only for the current query.  Across the complete
query stream it materializes 932 of 1,110 pages, consumes more combined
representation bytes than the full-visual index, and remains below
full-visual quality.

## Measured result

One NVIDIA A100-SXM4-80GB, PyTorch 2.5.1 CUDA 12.4, Transformers 4.53.2,
ColPali Engine 0.3.12, and the public MMDocIR ColPali v1.1 checkpoint were
used.  Results use the unmodified official evaluator pinned at
[`a70f23a`](https://github.com/illuin-tech/vidore-benchmark/commit/a70f23af8bb3b33efe8a4a6c6c15a6e2d978035e).

| Policy | nDCG@10 | Recall@10 | Recall@100 | Build | Search | Representation bytes |
|---|---:|---:|---:|---:|---:|---:|
| Markdown | 0.4947 | 0.5389 | 0.8444 | 32.91 s | 6.72 s | 315 MB |
| Full visual | 0.5178 | 0.5584 | 0.8746 | 100.73 s | 8.01 s | 585 MB |
| Tiered K=20 | 0.5009 | 0.5230 | 0.8239 | 32.71 s | 100.51 s | 315 MB base + 491 MB peak cache |

Tiered K=20 observes 6,360 candidate events.  It has 5,428 cache hits and 932
misses, an 85.3% event hit rate, but the misses cover 84.0% of the entire
corpus.  Its final build-plus-search time is 133.23 seconds, 22.5% above full
visual's 108.74 seconds, while its nDCG@10 is 0.0169 lower. It also invokes
the visual encoder in 189 small query-sized batches instead of one efficient
corpus batch.

The earlier 276.7-second visual result included a redundant PIL--PNG--PIL
round trip in the local adapter. Passing decoded images directly reduces the
visual build to 100.7 seconds; the score matrix is bitwise identical. The
old run is retained as provenance, but it is superseded for cost claims.

Corrected compact measurements and raw-result digests are stored in
`results/vidore-progressive-oracle/system-summary.json`. The original run is
retained in `results/vidore-v3-hr/summary.json`. Models, images, embeddings,
and machine-specific logs are intentionally excluded from Git.

## What this proves

1. Full visual representation has measurable retrieval value on an official
   public full-corpus workload.
2. Paying for it uniformly is expensive: the observed build cost differs by
   more than the search cost does.
3. Cross-query reuse is real: 85% of Top-20 candidate events hit the cache.
4. Reuse alone is insufficient.  A broad text Top-K eventually touches most
   pages, and unconditional visual score replacement can reduce recall.

The result does **not** show that deferred visual ingestion is ineffective.
It shows that “materialize every text Top-K miss” is not a sufficient utility
policy.

## Position against current official baselines

This run uses an older checkpoint to isolate representation lifecycle, not to
claim a leaderboard result.  The pinned official repository contains stronger
models and pipelines on the same HR split:

- the 32M textual
  [mxbai edge ColBERT result](https://github.com/illuin-tech/vidore-benchmark/blob/a70f23af8bb3b33efe8a4a6c6c15a6e2d978035e/results/metrics/mxbai-edge-colbert-v0-32m/vidore_v3_hr.json)
  reports 0.5261 nDCG@10;
- the 1B
  [Nemotron image+text result](https://github.com/illuin-tech/vidore-benchmark/blob/a70f23af8bb3b33efe8a4a6c6c15a6e2d978035e/results/metrics/nvidia_llama-nemotron-embed-vl-1b-v2_mod_image_text/vidore_v3_hr.json)
  reports 0.6080;
- the 3B
  [Nemotron visual late-interaction result](https://github.com/illuin-tech/vidore-benchmark/blob/a70f23af8bb3b33efe8a4a6c6c15a6e2d978035e/results/metrics/nvidia_llama-nemotron-colembed-vl-3b-v2/vidore_v3_hr.json)
  reports 0.6548.

Any paper-level evaluation must therefore transfer the final policy to at
least one current official model.  Improving ColPali v1.1 itself is not the
claim.

## Design review

The next system mechanism should be a progressive visual-index controller,
not a larger cache. The rejected K=20 run is already a query-driven
materialization baseline: its failure shows that "first candidate touch"
cannot double as a persistent-admission decision.

```text
Markdown retrieval
       |
       v
candidate evidence + verified score intervention + lifecycle state
       |
       +---- stay Markdown
       |
       +---- visualize for this query only
       |
       +---- visualize and admit to cache
       |
       +---- evict when reuse value decays
```

The controller must predict an action, not merely a content label. It should
first estimate which candidates justify transient refinement, then use the
observed visual score intervention plus reuse state to decide whether to admit
the result. Candidate rank, Markdown margin, prior access count, recency, and
remaining cache budget are runtime-visible. Full-visual scores and qrels may
be used only by the offline oracle or a disjoint training split.

The score trace is now persisted and the offline oracle passes the registered
headroom gate. A 25-page (2.25%) witness already exceeds the 95%-gain target;
the best tested point below 30% residency uses 150 pages (13.5%) and reaches
0.5436 nDCG@10. Simple qrel frequency and full-stream text Top-20 frequency do
not pass. This establishes that the representation pair contains selective
value, but only the successful selector uses qrels and complete future rank
outcomes.

The next experiment is therefore an online estimate--verify--commit replay.
It must ask whether a runtime-visible policy can:

- materialize at most 30% of pages over the full HR query stream;
- retain at least 95% of the full-visual nDCG@10 gain over Markdown;
- keep combined base-plus-cache bytes below the full-visual index; and
- beat both fixed Markdown and full visual at a stated query horizon.

The frozen experiment and baseline contract is specified in
[`progressive-visual-index-contract.md`](progressive-visual-index-contract.md),
and the measured oracle result is reported in
[`progressive-visual-oracle-result.md`](progressive-visual-oracle-result.md).
