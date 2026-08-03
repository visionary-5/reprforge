# Reusable Pair-Probe Contract

## Mechanism

ReprForge V2 removes the separate training-probe bill. A what-if observation
is now the same physical action as index construction:

```text
unresolved Top-5 boundary edge
        -> materialize missing endpoint pages
        -> score the incumbent/challenger pair
        -> retain both page views in the resident index
        -> update the risk of unresolved edges
```

The prior probability for an unresolved edge is
`P(N(0, 2) > locator_margin)`, derived from the difference of two normalized
visual scores. After each materialization round, observed raw ColPali pair
deltas estimate their own scale using `Var(X-Y)=2 Var(X)`. A smoothed empirical
survival probability then reprioritizes later edges. Pages shared across edges
or queries are encoded once.

The physical budget is an operator input, not a learned candidate-depth
constant. The selector maximizes newly observable boundary risk per marginal
predicted A100 millisecond. The physical development point uses two-page
warm-up rounds until eight pair deltas are observed, then admits at most eight
pages per round. Every round is published atomically and all probe artifacts
remain resident. Missing visual views receive a zero-mean prior.

The eight-observation transition was chosen after an IRPAPERS sensitivity
sweep (`4, 8, 16, 32, 64`). It is frozen before physical execution but is not
independent IRPAPERS quality evidence. Any cross-dataset claim must transfer
this value unchanged or tune it on a disjoint development benchmark.

## Leakage boundary

The planner may inspect BM25 candidates, ranks, margins, already materialized
visual scores and physical cost counters. It may not inspect:

- any visual score whose page representation is not resident;
- the complete Top-10 or corpus visual score surface;
- qrels or answers.

The frozen score-surface provider must reject unmaterialized reads. In the GPU
executor, every score used for adaptation must be produced from a page encoded
in the same charged run.

## Offline public gate

Use the complete IRPAPERS benchmark with five source-paper-disjoint folds,
BM25 Top-10 and a Top-5 output. Compare at 10%, 15%, 20% and 25% episode-page
budgets:

1. frequency admission;
2. independent incident-risk admission;
3. static pair admission using only the normal prior;
4. reusable active pair admission with empirical updates.

Proceed to A100 only if one operating point satisfies:

- pair-only visual delta has at least 0.80 AUC for the full-cohort boundary
  event;
- active admission beats the same-budget static pair policy by at least one
  Recall@5 query without lower aggregate exact-teacher agreement, or matches
  its quality using at least 10% less predicted physical work;
- it matches or exceeds the previous train-only pair controller's 82.22%
  Recall@5 while materializing at least 15% fewer than its 125 pages;
- it is no worse than the strongest same-budget cheap baseline in at least
  four of five folds, allowing at most one-query loss in any other fold;
- predicted visual work improves by at least 1.15x.

## Physical gate

The A100 probe compares:

- reusable active pair at the frozen 15% budget and eight-page rounds;
- static pair at the same budget and round schedule;
- independent incident risk at 20%, atomically prebuilt in one round.

Use one idle A100, the existing read-only ColPali environment, two
fold-interleaved repetitions and alternating policy order. Charge query
encoding, every representation build, every materialization call and every
visual score. Stop after 30 minutes or two GPU-hours, whichever arrives first.

A physical systems claim requires active pair to preserve the offline adaptive
quality gain over static pair and beat independent-20 by at least 1.15x in
both repetitions. If the
multi-round executor erases the page reduction, retain the offline mechanism
but reject the performance claim; do not replace the eight-page round with an
unmeasured atomic replay.
