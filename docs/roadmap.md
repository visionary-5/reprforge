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

The next proof is whether observed interventions plus past reuse can predict
admissions without future queries or test qrels. It must also quantify:

1. how non-additive page upgrade utilities are under ranking competition;
2. whether transient refinement dominates persistence when all costs are
   charged; and
3. whether the oracle headroom survives chronological or frozen shuffled
   train/validation/test query prefixes.

The runtime-visible sketch may contain query-cluster frequencies, retrieval
boundary statistics, and score changes observed after a paid transient
refinement. It must not contain relevance labels or future queries. The
offline oracle can use labels only to measure headroom.

## Candidate mechanism

For each page and representation action, estimate:

\[
U_t(i,r) = \operatorname{Evidence}_t(i,r)
- \lambda \operatorname{DistractorRisk}_t(i,r).
\]

Promising pages are first refined transiently. Their measured intervention
and estimated future reuse then drive admission under a storage/serving
budget. Charge:

\[
\operatorname{MigrationCost}(r_{\text{old}}, r_{\text{new}}).
\]

Use hysteresis so small estimated utility changes do not churn the index.
Compile changed layouts into a delta generation, validate it, and atomically
switch the active manifest. The existing token-work scheduler executes mixed
generations without changing retrieval semantics.

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

The dynamic system is motivated only if workload episodes produce a
meaningful plan boundary:

- a static deployable plan has material regret;
- the regret is not removed by a single robust fixed plan;
- useful adaptation changes a small enough part of the index to amortize;
- improvements persist on documents not used to design the estimator.

If plan movement is negligible, ReprForge remains a static heterogeneous index
and the research should focus on a stronger static evidence–risk allocator
rather than adding lifecycle machinery.
