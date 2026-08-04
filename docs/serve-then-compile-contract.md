# Serve-Then-Compile Multimodal RAG Index

Date: 2026-08-04. Status: initial metric gate passed; cohort-frontier system
validation in progress.

## Research hypothesis

Post-hoc token compression begins after the expensive visual encoder has
already processed every page, so it cannot address the motivating delay:
time-to-first-usable multimodal RAG. ReprForge should instead publish a cheap
locator immediately, serve provisional evidence, compile expensive visual
candidate cohorts in the background, and atomically revise evidence when a
generation finishes.

The physical state is persistent and reusable across queries. A batch failure
publishes no partial generation; readers may pin an old generation. This is
different from hiding synchronous build latency and from query-time reranking
over an already complete visual bank.

## Closest-work boundary

Deferred Visual Ingestion (DVI, arXiv:2602.14162) already argues for zero-VLM
pre-ingestion, BM25/metadata localization, and query-time VLM inspection of
original pages. Deferred ingestion alone is therefore not novel. ReprForge's
prospective contribution must be the lifecycle of reusable multi-vector
retrieval state, atomic evidence revision, and the measured quality trajectory
while the physical index is being constructed.

The first validation reuses the frozen HR and Finance-EN A100 batch-8 resident
traces. No model run is needed. For every query, BM25 nDCG@10 is its provisional
quality. At the completion of the atomic batch containing that query, its
quality becomes the frozen candidate-relative BM25+ColPali fusion quality.

## New primary metrics

1. **Time to evidence-quality target:** first wall-clock checkpoint whose
   population mean nDCG@10 reaches the better of BM25 and full visual.
   Because this target can be trivial when BM25 is already the better route,
   also report time to 50% and 90% of the final fusion gain above that route.
2. **Quality-time AUC:** time integral of current population nDCG@10 divided
   by the comparison horizon.
3. **Gain-over-base AUC:** the same integral after subtracting the immediately
   available BM25 quality.
4. **Evidence stabilization latency:** P50/P95 time when each query's refined
   evidence is atomically published.
5. **Revision safety:** fractions of queries improved, unchanged, and harmed,
   plus the fifth percentile of per-query nDCG change.
6. Existing build work, resident bytes, encoder calls, and final retrieval
   metrics remain mandatory; AUC does not replace them.

The fair full-prebuild baseline may also serve BM25 while its visual index is
building. Its refined full-visual results become available only when the
complete frozen indexing-and-retrieval run finishes. The comparison horizon
is that full-visual completion time.

## Bounded validation gate

The serve-then-compile direction proceeds to answer-level RAG only if both HR
and Finance-EN satisfy all of the following without changing K=20, fusion, or
batch size:

- time to the better-single-route quality target is earlier than the full
  visual end-to-end completion;
- mean quality over the full-prebuild horizon exceeds the fair full-prebuild
  baseline by at least .005 nDCG@10;
- final evidence quality exceeds the better single route;
- revision harm and its tail are reported, not hidden by the population mean.

Passing is a metric/mechanism result, not yet a paper claim. The next stage
must add answer correctness, answer stabilization time, contradiction/revision
rate, and at least one defensible workload ordering. Official ViDoRe row order
is explicitly not a natural temporal trace.

## Initial outcome

The fixed FIFO compiler passes the bounded metric gate on both frozen domains:
quality--time gain over a fair full-prebuild baseline is +0.01340 nDCG@10 on
HR and +0.02051 on Finance-EN.  The qrel-free cohort-frontier scheduler then
reduces exact mean completion work by 19.3% and 28.7%, respectively.  A real
A100 HR comparison beats both FIFO and static full-stream popularity in total
time and frozen-score quality--time AUC.  See
[`cohort-frontier-scheduler-result.md`](cohort-frontier-scheduler-result.md).
