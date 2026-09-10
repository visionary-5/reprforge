# Strict analysis: official ColQwen2 v0.1 → v1.0 transition

## Analysis question

For the official named ColQwen2 successor transition, can the v0.1 post-vision state be reused to materialize the v1.0 multi-vector index exactly, while avoiding enough repeated vision work to matter physically?

The frozen comparison is `raw target build` versus `source post-vision IR → target suffix`. The measured unit is one complete 128-page Energy slice build. The three pairs are alternating warm technical repetitions on one A100, not three datasets, seeds, or hardware replicas.

## Key findings

| Evidence | Raw target | Exact replay | Interpretation |
|---|---:|---:|---|
| Median wall time | 34.692 s | 3.687 s | 9.409× speedup; 89.372% median saving |
| Mean ± SD | 34.725 ± 0.219 s | 3.691 ± 0.011 s | Replay wins all 3 matched pairs |
| Upstream/source stage | 30.914–31.349 s | 0.076–0.099 s | Saved time comes from processor/vision reuse |
| Target suffix + projection + D2H | 3.603–3.611 s | 3.603–3.606 s | Target-version work is retained, not skipped |
| Target output comparison | — | 12,058,624 / 12,058,624 elements equal | Bitwise equality; maximum absolute error 0 |
| Old versus target terminal | — | 99.778% elements changed; cosine 0.69139 | Old terminal index is genuinely stale |

The paired absolute saving is 31.035 ± 0.229 s. Even the weakest observed pair saves 89.274%. The effect is therefore physically large and stable within this fixed warm-worker scope.

The mechanism test is stronger than a timing-only cache result. Both paths execute the complete v1.0 decoder adapter, retrieval projection, and terminal copy. Their target tensors are bitwise identical, while the v0.1 terminal tensors differ substantially. The legal reusable object is therefore the state before the changed decoder/projection cut—not the stale terminal index.

## Main caveat

The exact BF16 post-vision IR is 297.15 MB for 128 pages: 4.360× the compressed input images and 6.161× the float32 target terminal tensors. This run establishes the exact dependency and compute upper bound; it does not make the uncompressed IR a storage-admissible deployment design. The already validated PCA route addresses that separate storage question, but this release experiment did not retest PCA or full retrieval quality.

## What changed in the research understanding

The official-release evidence is no longer confined to ColQwen2.5 v0.1→v0.2 or a third-party domain adapter. A second named ColQwen release generation, v0.1→v1.0, independently exhibits the same backend-only invalidation structure and admits exact post-vision replay. This strengthens the formulation that release identity is coarser than the actual invalidation boundary.

It does not establish how frequently such updates occur in production, generalize the 89.37% number beyond this workload/hardware, or show that every adapter update is replayable. Vision/base changes remain mandatory raw fallbacks.

## Claim candidates

- Claim: An official ColQwen2 v0.1 post-vision state can exactly materialize the v1.0 index for the frozen 128-page slice.
  - Source evidence: `gpu-result.json`, SHA-256 `ad497dd90f5fea5229beeeedd5b2ecc0688af0b4c72f1a94c48a6705f5bb45c6`; 12,058,624 elements bitwise equal.
  - Allowed wording: “For one official ColQwen2 v0.1→v1.0 transition and frozen 128-page Energy slice, post-vision replay reproduced the v1.0 terminal tensors bitwise exactly.”
  - Forbidden stronger wording: “All ColQwen releases are compatible” or “retrieval quality was benchmarked across Energy.”
  - Uncertainty: One slice, one A100, one processor contract.
  - Next check: Cold/object-storage route or a full release-transition trace using the compact IR.
  - Decision: keep

- Claim: Exact replay materially reduces target rebuild time for this transition.
  - Source evidence: Three alternating warm pairs; median 34.692→3.687 s, 89.372% saving, minimum pair saving 89.274%.
  - Allowed wording: “Replay was 9.41× faster in the frozen warm A100 canary and won all three matched repetitions.”
  - Forbidden stronger wording: “ReprForge is 9.41× faster in production” or a cross-hardware speedup claim.
  - Uncertainty: Technical repetitions are not independent external replications; exact sign-test p=0.25 at n=3.
  - Next check: Cold/object-store profile with validation, load, and transfer included.
  - Decision: keep

- Claim: The old terminal index cannot substitute for target materialization.
  - Source evidence: v0.1 versus v1.0 terminal changed in 99.778% of elements; mean token cosine 0.69139.
  - Allowed wording: “The successor invalidated the terminal representation while preserving an earlier exact dependency cut.”
  - Forbidden stronger wording: “Every changed element causes a retrieval ranking change.”
  - Uncertainty: Ranking impact was not separately measured in this 128-page canary.
  - Next check: None for dependency legality; full retrieval belongs to benchmark validation.
  - Decision: keep

- Claim: The exact IR is a deployable storage format.
  - Source evidence: IR is 6.161× the float32 terminal tensor.
  - Allowed wording: “Exact BF16 IR exposes the compute upper bound and a storage bottleneck.”
  - Forbidden stronger wording: “Exact replay reduces both build time and storage.”
  - Uncertainty: None; this storage claim is directly rejected by the measured bytes.
  - Next check: Use the already frozen compact-IR design for deployment-facing traces.
  - Decision: discard
