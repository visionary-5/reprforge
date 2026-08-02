# Benchmark Transfer Contract

This document turns the benchmark landscape into a bounded evaluation plan.
It does not authorize downloading data or running GPUs; each phase is started
only after its artifact and resource preflight is recorded.

## Research claim under test

The current constructive claim is:

> A cheap corpus-wide locator can define query-relative visual cohorts, and a
> progressive compiler can build and reuse expensive representations for
> those cohorts quickly enough to reach useful retrieval quality before a
> uniform full-visual index is ready.

The transfer tests must determine whether this is a property of two ViDoRe
splits and BM25 wording, or a reusable index design.

## Frozen metrics

Retrieval quality:

- nDCG@10 when graded or multi-page relevance is available;
- Recall@1/5/20 for single-gold-page datasets;
- Recall@100 for first-stage candidate coverage;
- per-query reciprocal-rank delta and failure category.

Construction and storage:

- source parsing/OCR seconds, visual encoding seconds, and physical index
  build seconds reported separately;
- encoded pages and page-events;
- base, resident, transient-peak, and total index bytes;
- time until 50%, 90%, 95%, and 100% of final quality is reached.

Serving:

- cold and warm P50/P95 query completion;
- throughput only when requests are actually concurrent;
- locator, construction, score/fusion, and reader times;
- cumulative GPU-seconds and host-to-device bytes.

End-to-end:

- answer correctness or the benchmark's official answer metric;
- evidence/citation recall;
- answer quality at equal build budget and equal wall-clock deadline.

No combined “end-to-end” number may omit a stage for one system and charge it
to another.

## Baselines frozen before transfer

Every transfer must implement or import the strongest available member of
these roles:

1. BM25 over supplied text;
2. modern dense or late-interaction text retrieval;
3. static full visual late interaction;
4. static fixed compact visual representation;
5. static text+visual fusion with both complete indexes;
6. transient candidate refinement with no cross-query reuse;
7. admit-on-first-touch resident cache;
8. bounded resident cache using LRU, LFU and two-hit admission;
9. ReprForge's current cohort compiler;
10. an offline oracle, reported only as a diagnostic upper bound.

Current runnable substitutions must be labeled.  In particular, ColPali v1.1
can preserve mechanism comparability but cannot stand in for current visual
SOTA.  LightSTAR and MURE remain paper-contract baselines until their official
code and weights are released.

## Phase A: IRPAPERS mechanism transfer

Why first:

- 3,230 pages fit a bounded one-GPU experiment;
- image and text forms are both public;
- open BM25, dense text, image late-interaction, MUVERA and hybrid reference
  results exist;
- questions exhibit measured text-only and image-only successes.

Required runs:

- text-only BM25 and dense+BM25 hybrid;
- full visual;
- static full text+visual fusion;
- transient Top-K visual refinement for K in {10, 20, 50};
- resident and no-cache ReprForge modes under the same K;
- an equal-byte compact visual baseline if MURE is still unavailable.

Pass condition:

- at equal or better Recall@5, ReprForge must reduce visual construction by at
  least 25% and improve cold-stream completion by at least 20% over uniform
  full visual; and
- it must retain at least 95% of the static full-hybrid Recall@1 gain over the
  best text-only baseline; and
- the result must not depend on one fixed query ordering across three frozen
  permutations.

Failing this closes the claim that the existing BM25-cohort mechanism
transfers beyond ViDoRe.  It does not close heterogeneous representation as a
research direction.

## Phase B: hard-negative mechanism tests

### Invoice Haystack

Use corpus sizes 500, 1,000, and 1,500.  Compare BM25, full visual,
full text+visual fusion, and progressive fusion.  The key curve is not only
Recall@1; it is visual work versus Recall@1 as homogeneity and corpus size
increase.

The useful result may be *less* visual allocation: if full visual collapses
while text remains discriminative, the compiler should avoid admitting
visual state that cannot change the ranking.

### MultiDocR

Run only after an official artifact is available.  Break results down by
original versus paraphrased query, question type, and five-level relevance.
The principal gate is that ReprForge retains its improvement on low-overlap
paraphrases.  Failure means the current locator/fusion mechanism is tied to
lexical overlap.

## Phase C: scale transfer

MIRACL-VISION proceeds in two steps:

1. English plus one high-resource and one low-resource non-English split;
2. all 18 languages only after correctness and storage preflight passes.

The scale experiment must use an ANN or sparse first stage.  Exhaustively
scoring 338K pages per query is not an acceptable ReprForge baseline.  Report
the effect of the benchmark's easy-negative removal and keep language macro
averages separate from corpus-weighted averages.

At least one current visual model with an official MIRACL-VISION result must
be included; Argus and Nemotron ColEmbed V2 are current candidates, subject to
4xA100 memory preflight.

## Phase D: downstream answer-use transfer

Choose one of:

- M3DocVQA/M3DocRAG for open-domain corpus retrieval followed by VLM answer;
- MMDocRAG for fine-grained multimodal quote selection and answer assembly.

M3DocVQA is the cleaner full-index transfer.  MMDocRAG is the richer evidence
use test but begins from 15/20 candidate quotes and therefore cannot replace a
first-stage retrieval experiment.

The reader and prompt are frozen across index policies.  A system result must
show either equal answer quality at lower build/latency cost or better answer
quality at the same cost.  Retrieval improvement without answer/citation
preservation is insufficient.

## Deferred temporal claim

None of the selected visual benchmarks contains natural query and update
timestamps.  The first four phases therefore evaluate cold construction,
static quality, and repeated-query reuse only.

A later temporal phase needs a separately reviewed trace with:

- explicit document arrival/update/delete times;
- the document version visible to each query;
- repeated topics or users;
- time-scoped relevance labels.

Until then, LRU/LFU results are cache sensitivity studies, not evidence for an
adaptive production lifecycle.

## Immediate stopping point

The next implementation task is limited to an IRPAPERS adapter and a dry-run
manifest.  Before GPU execution it must prove:

- all 3,230 page IDs map consistently across images, text, and qrels;
- the official 180-query Recall evaluator is reproduced on a supplied run;
- every baseline receives the same corpus and query set;
- preprocessing supplied by the dataset is separated from work performed by
  the evaluated system;
- projected new storage is below 8 GiB and the first smoke uses one A100 for
  no more than 30 minutes.

No new admission algorithm is justified before this transfer exposes a
measured miss in the current compiler.

