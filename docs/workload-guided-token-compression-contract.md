# Workload-Guided Homogeneous Token Compression

Date: 2026-08-04. Status: bounded opened-domain development probe. Computer
Science was opened by the earlier residual-witness transfer and is not a
sealed collection.

## Hypothesis

The failed residual index combines hierarchical centroids with selected full
tokens. Those two token populations do not share exact MaxSim semantics:
centroid pooling can overshoot a full-token maximum, while adding exact tokens
only repairs underestimation. Document-level full/pool mixing exposed the same
score-comparability problem.

This probe instead emits one homogeneous representation. Unlabeled fit-query
tokens are quantized into 32 spherical workload probes. For every document,
each probe ranks the original document tokens by cosine similarity. A
weighted-fair scheduler allocates a fixed token budget across probes in
proportion to their fit-workload frequency and retains the highest-ranked
distinct original tokens. There is no pooled cover and no full-token overlay.

Because the compressed representation is a subset of original document
tokens, for every possible query token `q` and document `D`,

```text
max(workload_subset(D) dot q) <= max(full(D) dot q).
```

This one-sided score property removes centroid overshoot; it does not guarantee
ranking or relevance quality. Qrels are absent from probe fitting, weighting,
token selection, and scoring policy.

## Frozen first probe

- dataset: ViDoRe v3 Computer Science, 1,360 pages and 215 queries;
- full ColPali-v1.1 bank and exact query embeddings already materialized;
- five deterministic query-ID folds;
- 32 spherical query-token probes, 20 Lloyd iterations;
- fixed per-document budgets 64 and 128 original tokens;
- weighted-fair selection with no temperature, epsilon, or content heuristic;
- matched per-document random original-token selection;
- pool-9 and full visual baselines;
- nDCG@5, nDCG@10, Recall@100, persistent token fraction, compile time, and
  score time;
- 4,000 paired-query bootstrap samples.

The selection branch continues only if at least one point at no more than 15%
of full tokens:

1. improves pool-9 nDCG@10 by at least .01 or has a paired interval above
   zero;
2. is within .015 nDCG@10 of full or statistically indistinguishable;
3. beats matched random token selection in nDCG@10;
4. loses no more than .01 Recall@100 from the better of full and pool-9.

If workload selection beats random but misses pool-9 because discarded token
content is not summarized, a single preregistered extension is allowed:
assign all original tokens to the selected seeds and emit normalized cluster
means at the same budgets. If it does not beat random, the branch stops.

## Prior-art boundary

This is not a claim for token pruning, clustering, query prototypes, or
workload conditioning in isolation. OmniColPress AGC uses learned universal
query tokens and attention to select centroids under a constant budget;
Prune-then-Merge uses global-token attention followed by hierarchical merging;
uniform-sphere Voronoi pruning estimates query-agnostic token influence. The
prospective distinction is empirical workload-query distribution as the
centroid/coverage measure, a homogeneous fixed-budget physical index, and
held-out workload compilation without relevance labels. AGC,
Prune-then-Merge, pool-9, matched random, and full remain required baselines
for a paper claim.

## Opened-domain outcome

The selection branch fails its minimum mechanism gate and is stopped. At 64
tokens per page (6.21% of full tokens), workload-guided selection obtains
0.5439 nDCG@10 versus 0.6285 for matched random, 0.7027 for pool-9, and 0.7170
for full. At 128 tokens (12.43%), it obtains 0.5994 versus 0.6688 matched
random. Its 128-token loss to pool-9 is -.1034 nDCG@10 with paired 95% CI
[-.1328, -.0749]. Recall@100 is also lower than both full and pool-9.

This is not a capacity-only failure: the exact matched-random control is much
stronger at both budgets. Empirical query probes concentrate the budget on
frequent workload directions and remove document-side diversity needed to
separate candidate pages. The prerequisite for the preregistered merge
extension---beating matched random---is false, so seed merging is not run.
The compact result is
`results/diagnostics/workload-guided-token-cs-v0.json`.
