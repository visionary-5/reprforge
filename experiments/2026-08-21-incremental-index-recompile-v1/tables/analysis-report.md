# Incremental Index Recompilation: Analysis Report

## Analysis question

For an adapter/projection update that leaves ColQwen2.5's image processor,
input embedding table, and vision tower unchanged, does replaying a materialized
post-vision representation reduce index rebuild time without changing the
resulting index?

The repeated-measure unit is one complete 128-page build. There are three
order-alternated timing repeats per method on one A100-SXM4-80GB. These repeats
measure systems variability; they are not model seeds, datasets, or independent
machines.

## Key findings

1. **The reusable boundary is computationally valid in this configuration.**
   The loaded adapter changes 504 decoder LoRA tensors and two terminal
   projection tensors, and changes zero vision tensors. Raw-image and reusable-IR
   execution produced bitwise-identical Full indexes and bitwise-identical L12
   indexes across all 128 pages (maximum absolute error 0.0).

2. **Recompilation payoff is large on this bounded workload.** Full compilation
   fell from 39.566 ± 0.621 s to 6.236 ± 0.057 s (mean ± run SD), a mean paired
   saving of 33.330 s and 84.24%. L12 compilation fell from 37.931 ± 0.601 s to
   4.378 ± 0.003 s, a mean paired saving of 33.553 s and 88.46%. Both comparisons
   won in all three paired repeats.

3. **Version reuse and in-flight lowering are complementary, not competing.**
   From raw images, L12 only reduces total time by 4.13% relative to Full because
   the unchanged visual front end dominates. Once that front end is reused, L12
   reduces the remaining recompile time by 29.79% relative to IR-Full. This is a
   more coherent compiler story: dependency reuse removes invalidated work at
   stage granularity; physical lowering reduces the work and storage that remain.

4. **The first representation is too large to be the final system design.** The
   BF16 IR occupies 397.64 MB for 128 pages, or 3.107 MB/page. That is 9.25× the
   compressed source images, 8.12× the Full float32 terminal index, and 16.01×
   the L12 terminal index. The wall-time overhead of writing it was only 0.453 s
   and is recovered in less than one recompile, but persistent storage—not
   serialization time—is now the bottleneck.

## What changed in the research understanding

The earlier one-shot operator addressed only the language suffix and therefore
had an observed end-to-end ceiling near 7%. This experiment shows that the
larger cost is not inherently unavoidable: for a common update class, the vision
front end is outside the invalidation set and can be reused exactly. The stronger
research object is therefore a version-aware multimodal index compiler with two
different operator classes:

- exact dependency-aware stage reuse for model/index versions; and
- optional lossy physical lowering for the invalidated suffix and durable index.

The experiment does **not** yet prove that long-lived IR caching is economically
attractive at corpus scale. It establishes the mechanism and exposes the next
optimization target: reduce or selectively materialize reusable state while
retaining most of the 84–88% recompile saving.

## Claim candidates

### Claim 1 — exact stage reuse

- Claim: Adapter/projection updates can reuse ColQwen2.5 post-vision states exactly.
- Source evidence: `result-v1.json`; adapter-scope diagnostics and 128-page
  bitwise equivalence checks.
- Allowed wording: "For the evaluated ColQwen2.5 adapter update, post-vision IR
  reuse produced an identical index."
- Forbidden stronger wording: "Adapters never invalidate visual states" or
  "the method is exact for all multi-vector VLMs."
- Uncertainty: one adapter architecture and one processor/model version.
- Next check: repeat with a second adapter/backbone whose dependency manifest is
  independently derived.
- Decision: keep.

### Claim 2 — recompilation speedup

- Claim: Reusing the valid visual IR materially accelerates this rebuild class.
- Source evidence: three order-alternated 128-page repeats; 84.24% Full and
  88.46% L12 mean savings; Figure 1.
- Allowed wording: "On one A100 and this homogeneous Energy subset, warm-storage
  recompilation was 6.34× faster for Full and 8.66× faster for L12."
- Forbidden stronger wording: "ReprForge accelerates multimodal indexing by 8×"
  without the update class, warm-storage condition, hardware, and workload.
- Uncertainty: only three systems repeats, one host, one page shape, warm OS cache.
- Next check: cold/warm NVMe separation and a heterogeneous complete corpus.
- Decision: keep, bounded.

### Claim 3 — compiler composition

- Claim: Dependency reuse makes the existing L12 suffix operator materially more
  useful during rebuilds.
- Source evidence: raw L12 saves 4.13% versus raw Full, while IR L12 saves 29.79%
  versus IR Full under the same repeated protocol.
- Allowed wording: "After exact visual-stage reuse, suffix lowering reduced the
  remaining execution time by about 30% in this workload."
- Forbidden stronger wording: "The combined method is Pareto-optimal" until
  retrieval quality, alternative cache representations, and baselines are joined.
- Uncertainty: the known L12 quality loss remains; this run measures build output
  equivalence between source paths, not L12-vs-Full retrieval parity.
- Next check: join complete-benchmark quality with versioned total-cost curves.
- Decision: revise into the central systems-mechanism hypothesis.

### Claim 4 — storage feasibility

- Claim: The naïve materialized BF16 IR is already an acceptable permanent cache.
- Source evidence: contradicted by 3.107 MB/page and 8.12× Full-index storage.
- Allowed wording: "The naïve IR identifies storage amplification as the next
  bottleneck."
- Forbidden stronger wording: any corpus-scale storage-efficiency claim.
- Uncertainty: no quantized, tiered, sparse, or temporary-cache policy evaluated.
- Next check: frozen int8/FP8 or topology-reduced IR replay with retrieval and
  recompilation measurements.
- Decision: discard the permanent-BF16-cache design; retain the reusable-boundary
  mechanism.

## Main caveats

- Timing is warm-storage: the operating-system page cache was not flushed.
- Source images were already present as in-memory compressed Parquet cells, while
  IR shards were loaded from files. This is conservative for IR I/O but is not a
  production storage-stack comparison.
- Three repeats are adequate for a mechanism screen but too few for a strong
  population-level significance claim.
- The corpus subset has a single processor shape and one benchmark domain.
- A policy-only index change can legally reuse final terminal embeddings and is
  a stronger baseline than suffix replay. The reported advantage applies to
  decoder-adapter/projection changes that invalidate terminal embeddings.

## Decision

Promote version-aware dependency reuse to the active research hypothesis. Do
not promote the current BF16 IR as the final method. The next decision-changing
experiment should test a storage/recompute frontier for reusable IR (BF16,
quantized, and selectively materialized) and must compare against raw rebuild
and terminal-cache baselines under their legal invalidation scopes.
