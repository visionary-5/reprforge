# Cohort-Frontier Scheduling Contract

Date: 2026-08-04. Status: bounded frozen-trace mechanism probe.

## Hypothesis and prior-art boundary

Deferred ingestion, caching, and query grouping are established ideas.  The
bounded hypothesis here is narrower: while a costly multi-vector index is
being physically constructed, schedule atomic query cohorts by their shared
missing representations so useful evidence becomes available with less
construction work.  The final candidate-relative ranking and final resident
page set are invariant; only the publication trajectory changes.

CaGR-RAG groups queries to improve disk-vector locality.  This probe is not a
first-query-grouping claim.  A surviving ReprForge mechanism must optimize and
measure evidence-quality progress under charged visual representation
construction, then demonstrate that the benefit remains under bounded arrival
windows and real A100 timing.

## Frozen algorithm

Given the BM25 Top-20 query--page graph and request batch size eight:

1. choose a pending query with the fewest pages absent from resident state;
2. break cold-start ties by future page reuse in the pending window;
3. pack the remainder of the atomic batch by the fewest pages added to the
   staged union, then by resident/staged overlap;
4. publish only after the complete batch has been encoded and scored.

The scheduler cannot inspect visual scores, qrels, or downstream answers.
Compare FIFO, ten deterministic random orders, static full-stream page
popularity, and the dynamic cohort frontier.  Static popularity is explicitly
future-aware and therefore an optimistic batch-workload baseline.

## Metrics and stop gate

First replay exact unique visual pages rather than fitting a noisy latency
model.  Report mean/P50/P95 encoded pages at query completion, completed-query
AUC over full-corpus construction work, and post-hoc nDCG@10--page-work AUC.

Do not run new GPU executions unless the frontier scheduler, on both HR and
Finance-EN:

- reduces mean completion page-work by at least 10% versus FIFO;
- is no worse than static popularity on mean completion work;
- improves nDCG@10--work AUC over FIFO without using qrels; and
- preserves the exact final cohort union and final evidence quality.

Passing this gate authorizes real A100 replays for FIFO, frontier, and the
strongest simple schedule.  It does not authorize an online latency claim:
unrestricted reordering assumes a queued batch workload.  The next gate must
bound scheduling windows or replay a public arrival trace.
