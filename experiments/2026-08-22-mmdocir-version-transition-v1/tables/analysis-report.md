# MMDocIR complete version-transition analysis

## Decision

**PROMOTE** the dependency-aware version transition under the frozen contract. The experiment completed the physical path from a post-vision IR to a quality-admitted, SQ8-indexed, atomically published target generation.

## What was tested

The target change updates the ColQwen2.5 retrieval decoder adapter and projection while preserving the processor, base embedding, and vision dependencies compiled into the post-vision IR. All 313 MMDocIR documents, 20,395 pages, and 12,441,160 terminal vectors were regenerated. Official quality uses 1,658 source-document queries; serving fidelity uses 20 domain-stratified queries over all pages.

This is an execution of one method, not a sum of unrelated best numbers: replay writes a new terminal generation; that exact generation is scanned to train and populate SQ8; its files are hashed into one manifest; publication occurs only after the quality, candidate, scope, and timing gates pass.

## Decisive results

- Official Recall@5 is 0.856011 versus Full 0.852885; paired compact-minus-Full 95% query-bootstrap interval is [-0.003559, +0.009892].
- Official nDCG@10 is 0.790830 versus Full 0.789233 (delta +0.001597). These are quality-preservation results, not improvements.
- The complete warm transition takes 1350.86 s versus 5077.86 s for the raw-page route with identical downstream validation and serving work: 73.40% saving.
- SQ8 returns 2197.2 candidate pages on average (10.77% of the collection) and covers 98.50% of Full exact global Top-10 pages. Median candidate search is 5.07 ms.
- The new generation was published through a single manifest-addressed `ACTIVE` pointer. This run tests first publication into a new deployment root; the separate five-trial cutover experiment is the evidence for concurrent old/new switching.

## Gate audit

- `replay_admitted`: PASS
- `complete_scope`: PASS
- `candidate_fidelity`: PASS
- `end_to_end_transition`: PASS
- `artifact_hashes`: PASS
- `atomic_publication`: PASS

## Mechanism interpretation

The payoff comes from dependency scope, not generic feature compression. The adapter/projection update invalidates terminal vectors but does not invalidate the frozen visual prefix. Replaying the deepest valid artifact avoids the vision path while still materializing the true target-version endpoint. A query bridge is cheaper but serves stale document semantics and was already significantly worse on the same complete benchmark. A terminal rewrite is faster but is legal only when embedding semantics do not change.

The lossy in-flight merge is not required for this result. Its one-shot 7% build saving and the failed ViDoSeek query-free risk certificate make it an optional workload-admitted operator, not the default ReprForge method.

## What this can support

1. A problem contribution: multimodal multi-vector indexes require dependency-aware version maintenance, not unconditional re-embedding.
2. A method contribution: fingerprinted dependency cuts plus measured materialization planning choose the deepest quality-valid rebuild source.
3. A systems contribution: the selected route produces and admits one immutable serving generation, including terminal replay, compact candidate construction, validation, manifest hashing, and publication.
4. A mechanism result: embedding-space alignment is not a MaxSim compatibility certificate, and query-free merge geometry is not a ranking-safety certificate.

PCA, SQ8, atomic rename, and DAG caching are components and baselines; none is claimed as standalone novelty.

## Limits

- One complete adapter/projection transition, one target backbone, one local A100/NVMe environment.
- MMDocIR official evaluation is source-document page ranking; the global candidate probe is internal serving fidelity.
- The reported transition peak is a lower bound because it excludes filesystem and process overhead and any separately retained v0.1 ANN.
- Distributed/object-store publication, garbage collection, cold-cache performance, and production update-frequency traces remain outside this experiment.
