# Research Roadmap

## Paper direction

The working direction is:

> **Incremental workload-aware compilation of heterogeneous multimodal
> retrieval indexes.**

The paper should not be framed as “compress ColPali vectors.” Compression is
one action inside a larger resource-allocation system. The research question
is whether an evolving query workload changes which layouts deserve text,
compressed visual, or full visual capacity—and whether the useful changes can
be deployed cheaply enough to improve end-to-end service.

## Current proof-of-mechanism

The complete Markdown/full-visual score and cost trace is now frozen for the
official ViDoRe HR run. The static headroom gate passes: an oracle-only set of
25 pages retains at least 95% of the full-visual nDCG gain, and the best tested
set below 30% residency uses 150 pages (13.5%) and reaches 0.5436 nDCG@10.
Uniform full visual reaches 0.5178. This proves that selective representation
has headroom, not that the useful pages can be predicted online.

The proposed page-level estimator has now been tested and rejected. Exact
single-page utilities are strongly non-additive, and neither a linear
intervention model nor a semantic nearest-neighbour cohort controller beats
robust fixed baselines. It must not remain the default next step.

The new proof-of-mechanism is candidate-relative normalized fusion. With a
fixed BM25 Top-20 cohort it reaches 0.5373 nDCG@10 on HR and 0.5628 on
Finance-EN, above both single-representation baselines. The complete streams
use 79.7% and 61.7% of full-visual build-equivalent work, respectively.

## Implemented mechanism

The minimal synchronous cohort compiler now:

1. BM25 immediately produces a Top-20 cohort;
2. resident visual candidates are fused with candidate-relative calibration;
3. missing candidates from a request batch are deduplicated and encoded in one
   compiler call;
4. compiled visual state becomes visible only after the request batch
   succeeds;
5. every experiment reports time-to-quality, cold-query P95, total GPU work,
   and final resident coverage.

Synchronous no-reuse, batch-1 resident, and full visual prebuild have been
measured on HR.  Resident reuse accounts for most of the 6.04x improvement
over synchronous no-reuse; batch-8 adds only 1.07x over batch-1 resident and
increases P95 completion from 1.39 to 9.04 seconds.  On Finance-EN the final
system is 1.68x faster end-to-end than full visual and improves both nDCG@10
and Recall@100.

The next mechanism is not a larger batch queue. LFU/two-hit or another bounded
admission rule is added only when a defensible repeated or temporal workload
is available. Otherwise the constructive path is serve-then-refine with
explicit result revision and time-to-quality metrics.

The required first-stage baselines now include a LightSTAR-style transient
selection/refinement cascade and a fixed compact visual representation such as
MURE when reproducible artifacts are available. These are stronger than the
current ColPali-only comparison and determine whether persistence has any
incremental value.

## Evaluation expansion

The public benchmark audit is recorded in `benchmark-landscape.md`, and the
bounded execution gates are frozen in `benchmark-transfer-contract.md`.  The
priority order is now:

1. IRPAPERS for a text/image complementary-failure transfer;
2. Invoice Haystack for visually homogeneous hard negatives;
3. MIRACL-VISION for multilingual 338K-page scale;
4. M3DocVQA or MMDocRAG for downstream answer use.

MultiDocR is a high-value paraphrase and graded-relevance test, but remains
blocked on a public artifact.  MMLongBench is a reader stress test rather than
an index benchmark and cannot substitute for these transfers.

The next complete evaluation needs:

- at least two public multimodal document workloads or independently defined
  MMDocIR workload families;
- static uniform, static typed, full reoptimization, and migration-aware
  baselines;
- retrieval and downstream Agent answer/citation metrics;
- storage, build, migration, P50/P95, throughput, and GPU-hours;
- ablations for evidence, risk, migration cost, hysteresis, and token-work
  scheduling.

## Decision boundary

The current compiler provides a cold-stream Pareto improvement over full
prebuild on two datasets, but it does not provide low request latency. A
dynamic lifecycle or asynchronous serving system is motivated only if
workload episodes produce a meaningful plan boundary:

- a static deployable plan has material regret;
- the regret is not removed by a single robust fixed plan;
- useful adaptation changes a small enough part of the index to amortize;
- improvements persist on documents not used to design the estimator.

If plan movement is negligible, ReprForge remains a static heterogeneous index
and the research should focus on a stronger static evidence–risk allocator
rather than adding lifecycle machinery.
