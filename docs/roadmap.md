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

## Candidate mechanism

The next minimal system is an asynchronous cohort compiler:

1. BM25 immediately produces a Top-20 cohort;
2. resident visual candidates are fused with candidate-relative calibration;
3. missing candidates enter a GPU batch queue rather than twenty small
   synchronous calls;
4. compiled visual state becomes visible through the existing versioned delta
   index;
5. every experiment reports time-to-quality, cold-query P95, total GPU work,
   and final resident coverage.

Synchronous refine-and-wait, full visual prebuild, and unconditional
admit-on-first-touch are required baselines. LFU/two-hit admission is added
only when a defensible repeated or temporal workload is available.

The required first-stage baselines now include a LightSTAR-style transient
selection/refinement cascade and a fixed compact visual representation such as
MURE when reproducible artifacts are available. These are stronger than the
current ColPali-only comparison and determine whether persistence has any
incremental value.

## Evaluation expansion

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

The asynchronous compiler is motivated only if it produces a stable Pareto
improvement over full prebuild and synchronous refinement. A dynamic
lifecycle system is motivated only if workload episodes produce a meaningful
plan boundary:

- a static deployable plan has material regret;
- the regret is not removed by a single robust fixed plan;
- useful adaptation changes a small enough part of the index to amortize;
- improvements persist on documents not used to design the estimator.

If plan movement is negligible, ReprForge remains a static heterogeneous index
and the research should focus on a stronger static evidence–risk allocator
rather than adding lifecycle machinery.
