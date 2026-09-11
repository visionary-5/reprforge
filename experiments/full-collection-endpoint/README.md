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
