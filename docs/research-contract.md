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

## Updated mechanism after intervention testing

Independent page utility is not a valid first-order model for the current
retrieval path. On the frozen HR trace, replacing the scores of a Top-20
cohort has an average delta nDCG@10 of +0.00347, while the sum of the same 20
single-page interventions is -0.03114. Ranking competition creates a mean
absolute interaction of 0.1584. A held-out intervention learner consequently
has only 0.0145 Pearson correlation with exact page utility.

The current positive mechanism is instead set-level: use a cheap Markdown
BM25 locator, form a fixed candidate cohort, normalize BM25 and visual scores
inside that cohort, and fuse the two evidence sources. With K=20 fixed across
datasets it beats both single representations on HR and Finance-EN. This
changes the target system from a per-page utility allocator to a
query-generated cohort compiler.

## Intended contribution

The target system is a workload-aware progressive representation index with
three connected mechanisms:

1. **Candidate-relative evidence composition:** compare heterogeneous scores
   only in the cohort where they compete, rather than replacing globally
   incomparable raw scores.
2. **Asynchronous cohort compilation:** build query-requested visual state in
   batches without turning representation savings into cold-query stalls.
3. **Work-normalized execution:** schedule the compiled heterogeneous index by
   padded vector work so resource reduction yields GPU latency reduction.

Admission and eviction become a later mechanism only if a public temporal
workload demonstrates that persistence beats transient refinement and simple
caches. ViDoRe query order is not a natural production trace.

For workload episode \(t\), a candidate dynamic objective is:

\[
\max_{\{r_i^t\}}
\sum_i U_t(i,r_i^t)
- \mu \sum_i M(r_i^{t-1}, r_i^t)
\]

subject to storage, build, and serving budgets. Here `M` is measured
migration cost. Versioning is an implementation mechanism for atomic
deployment and rollback; it is not a novelty claim by itself.

The rejected `tiered-selective K=20` policy is the unconditional
admit-on-first-touch baseline. It already provides the query-driven
materialization pattern, so database cracking alone is not a contribution.
The open challenge is no longer a more elaborate per-page what-if estimator.
It is scheduling cohort construction under a latency/resource budget while
preserving candidate-relative fusion quality. See
[`progressive-visual-index-contract.md`](progressive-visual-index-contract.md).

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
