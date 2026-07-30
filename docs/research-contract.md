# Research Contract

## Problem

A multimodal document layout can be represented as native text, a full visual
late-interaction embedding, or one of several compressed visual embeddings.
These choices change:

- whether query-relevant evidence is recoverable;
- how many distractors compete in MaxSim;
- index storage and GPU-resident bytes;
- offline encoding and migration work;
- online padded token work and execution batches.

ReprForge treats representation assignment as a constrained compilation
problem. For layouts \(i\), routes \(r\), and budget \(B\), the static form is:

\[
\max_{\{r_i\}} \sum_i U(i,r_i)
\quad \text{subject to} \quad
\sum_i C(i,r_i) \le B.
\]

`U` cannot be defined only as compression fidelity. A useful model must
separate evidence recovery from distractor risk under the expected query
workload. `C` is reported as explicit dimensions—bytes, build time, and GPU
work—not as an arbitrary weighted “total cost.”

## Current system claim

The present artifact demonstrates that:

1. heterogeneous representation plans can be compiled into a compact,
   executable late-interaction index;
2. representation capacity changes retrieval quality in ways that raw visual
   complexity and embedding-cover loss do not adequately predict;
3. storage compression becomes query speed only when the runtime schedules
   padded token work rather than a fixed number of documents.

The current Typed-Capacity V1 planner is a transparent heuristic and does not
establish a general allocation algorithm.

## Intended contribution

The target system is a workload-aware representation compiler with three
connected mechanisms:

1. **Evidence–risk estimation:** estimate which route preserves useful
   evidence without inflating distractors for the current workload.
2. **Migration-aware allocation:** change only layouts whose expected utility
   gain justifies rebuild and deployment cost.
3. **Work-normalized execution:** schedule the compiled heterogeneous index by
   padded vector work so resource reduction yields GPU latency reduction.

For workload episode \(t\), a candidate dynamic objective is:

\[
\max_{\{r_i^t\}}
\sum_i U_t(i,r_i^t)
- \mu \sum_i M(r_i^{t-1}, r_i^t)
\]

subject to storage, build, and serving budgets. Here `M` is measured
migration cost. Versioning is an implementation mechanism for atomic
deployment and rollback; it is not a novelty claim by itself.

## Required evidence

A paper-level claim requires all of the following:

- public, document-disjoint workloads and frozen model/data revisions;
- strong fixed and uniform representation baselines;
- an offline per-episode oracle used only as a diagnostic upper bound;
- retrieval quality, Agent answer quality, storage, build cost, latency, and
  migration cost reported separately;
- stable gains across documents or workload episodes, not only a
  query-weighted aggregate;
- an ablation linking each proposed mechanism to an observable failure of a
  simpler policy;
- reproducible physical indexes and GPU measurements.

## Non-claims

ReprForge does not currently claim:

- that content-type routing, token pooling, or MaxSim is new;
- universal dominance over full visual representations;
- that the sealed V1 quality result passes its preregistered quality gate;
- global-corpus retrieval quality from an unjudged candidate pool;
- that physically replicated candidates are a valid quality benchmark;
- a dynamic-workload benefit before workload episodes are evaluated.
