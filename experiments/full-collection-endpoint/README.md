# Full-collection independent endpoint

Question: does persisted source-state replay match independently executed native
raw targets over every unique page in the six already used public collections,
rather than a 200-page subsample? These are the frozen ViDoRe evaluation
splits (including published subsampled datasets), not the full original DocVQA
or ArxivQA benchmarks. Does this hold for ordered retrieval on the
original 1,033 query holdouts?

Use ColQwen2.5 v0.1 -> v0.2, common pinned processor at 12,845,056 pixels,
A100, BF16, page batch 1. Source capture and target execution use separate
processes. Target raw/replay are paired with alternating execution order.
No new benchmark, model family, training or tuned intervention is introduced.
Exclude the previous MMDocIR distractors from this complete six-collection run;
report the actual gallery size and do not call it the old 3,420-page protocol.

Expand the existing independent endpoint protocol to every unique image-byte
SHA-256 in those collections, preserving its deterministic round-robin order.
The exact page count, data hashes and protocol must be frozen before GPU output.

Expected scope: a few thousand pages; approximately 2–4 A100 hours including
source capture, target encoding, replay and scoring; no claim of measured time
until completion. Raw/reference and replay time exclude initial state capture,
index construction and validation; warm state reads are included.

Failure interpretation: any admitted nonidentical page limits representation
exactness; ranking differences must be reported separately. No failed pages
may be dropped. Complete collection coverage does not establish arbitrary-model
correctness. Synthetic mechanism interventions remain a separate experiment.

## Versioned entry point

`run.py` is the expanded successor of the frozen independent endpoint script.
The original `experiments/independent-endpoint/run.py` is retained byte-for-byte
so historical evidence verification remains valid. Run this successor with the
`config.json` and `protocol.json` emitted by `prepare.py`.

The already launched 2026-09-11 run used this same code under the old deployment
path; its manifest and archived executed source must retain that original path.
Renaming the publication entry point does not change the historical manifest.

Analysis uses the provided page associated with each frozen query row as its
single gold page. nDCG@5 and Recall@10 are descriptive under this annotation
protocol; they are not substituted for target agreement. Multiple queries may
share a gold page, so queries are not independent sampling units for uncertainty.
The companion `verify.py --tensors` checks actual saved tensor bytes and accepts
negative outcomes while rejecting missing, duplicated or inconsistent evidence.

## Matched repeated timing (prepared, not yet executed)

`repeat-protocol.json` freezes three fresh target processes over all prior pages.
`repeat_timing.py` requires a completed `verify.py --kind full --tensors` report
bound to the same input and model manifest before launching. It records every
page, including mismatches, and samples GPU process activity outside timed regions.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/full-collection-endpoint/repeat_timing.py \
  --prior /completed/full-collection --verification /completed/full-verification.json \
  --output /new/three-round-output
```

This successor includes image-byte validation on raw and state-hash validation
on replay. Its numbers must not be mixed with the earlier validation-excluded
replay clocks. Report all three rounds, not the best one. Timing excludes source
capture, static model validation/loading, parquet loading, queries, output
serialization and physical index construction. No global cache flush is used.
