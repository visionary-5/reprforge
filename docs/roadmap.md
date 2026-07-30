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

## Next proof-of-mechanism

Before expanding the runtime, construct public, document-disjoint workload
episodes from query clusters or domains and answer:

1. Does the per-episode optimal representation plan move?
2. How much quality or resource regret does one static plan incur?
3. How many layouts and bytes must move to recover that regret?
4. Can a cheap workload sketch predict the useful migrations?

The runtime-visible sketch may contain query-cluster frequencies and retrieval
boundary statistics. It must not contain relevance labels or future queries.
The offline oracle can use labels only to measure headroom.

## Candidate mechanism

For each layout and route, estimate:

\[
U_t(i,r) = \operatorname{Evidence}_t(i,r)
- \lambda \operatorname{DistractorRisk}_t(i,r).
\]

Allocate routes under a storage/serving budget and charge:

\[
\operatorname{MigrationCost}(r_{\text{old}}, r_{\text{new}}).
\]

Use hysteresis so small estimated utility changes do not churn the index.
Compile changed layouts into a delta generation, validate it, and atomically
switch the active manifest. The existing token-work scheduler executes mixed
generations without changing retrieval semantics.

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
