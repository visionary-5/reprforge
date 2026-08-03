# Risk-Limited Acquisition on ViDoRe v3 Finance-EN

## Decision

The frozen Finance-EN transfer is **NO-GO for the current paper claim**.  It
does contain a real variable-depth signal: the conformal candidate set uses
10.13% less measured visual construction work than the cheapest
quality-matched fixed depth.  That improvement is below the preregistered 20%
offline gate, does not transfer to HR, and therefore does not justify a live
progressive executor or a context-adaptive second stage.

## Experiment

The A100 run regenerated token-aware text and full-visual traces for 309
queries and 2,942 pages from the fixed official Finance-EN Parquet revision.
The corpus-wide text late-interaction index is the cheap locator and supplies
the Top-100 pool.  The full ColPali-v1.1 trace is used only as a score teacher
and as the source of measured per-page visual encoding costs.  Five
deterministic query-disjoint folds use the algorithm, features and
`alpha=0.05` frozen before the HR transfer.  Qrels are excluded from every
policy feature and used only for final retrieval metrics.

| Policy | Mean visual pages/query | Unique visual pages | Summed visual build ms | Exact teacher Top-10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Fixed K=10 | 10.00 | 1,136 | 106,750.6 | 4.53% | 0.4387 |
| Fixed K=20 | 20.00 | 1,619 | 152,660.0 | 74.43% | 0.4552 |
| **Conformal candidate set** | **34.50** | **2,030** | **191,278.9** | **95.15%** | **0.4627** |
| Fixed K=50 | 50.00 | 2,256 | 212,843.5 | 99.35% | 0.4646 |
| Score-envelope acquisition | 67.22 | 2,470 | 233,344.0 | 100% | 0.4654 |
| Full Top-100 teacher | 100.00 | 2,612 | 246,597.2 | 100% | 0.4654 |

The candidate set passes its 93% Top-k coverage gate and remains within 0.005
nDCG@10 of the full teacher.  Fixed K=20 is too inaccurate to be considered
quality matched under the frozen rule, while fixed K=50 is the cheapest
eligible fixed baseline.  Relative to K=50, the candidate set saves 226
unique visual pages and 21,564.6 ms of summed measured page-encoding work, or
10.13%.

The corpus-wide text representation takes 85.0 seconds to build; the complete
visual representation takes 281.6 seconds.  Those full-index timings establish
the physical cost asymmetry, but they are not added to the table's incremental
page sums.  No live progressive latency is claimed because the offline 20%
work-reduction gate failed.

Recall@100 is 0.7222 for every policy because every row shares the same text
Top-100 candidate pool.  The experiment evaluates visual acquisition inside
that pool; it cannot repair locator misses.

## What the transfer establishes

Finance-EN rejects the strongest version of the earlier negative conclusion.
A single fixed depth is not always sufficient: per-query acquisition reduces
visual work by about 10% at matched quality.  However, the effect is neither
large enough nor consistent enough to support the proposed system.  HR favors
fixed K=20 by 1.7% construction work, whereas Finance favors the variable set
over fixed K=50 by 10.1%.

The score-envelope path is again too conservative.  Simultaneous score
coverage requires 67.2 pages/query, while direct decision coverage needs 34.5.
The remaining limitation is not conformal calibration alone.  The protected
event is still exact teacher Top-k membership rather than retrieval evidence.
On Finance, 10 of the candidate set's 15 teacher-set disagreements leave
per-query nDCG unchanged.  On HR the mismatch is stronger: 19 of fixed K=20's
28 disagreements and 11 of the candidate set's 12 disagreements are
nDCG-neutral.

The next mechanism must therefore estimate **evidence risk per unit of visual
construction**, not reproduce every score transition.  A valid policy may use
train-split relevance or downstream evidence labels to learn which uncertain
rank changes matter, but runtime decisions must remain qrel-free.  Its first
gate is to beat the quality-matched fixed depth by at least 20% construction
work on both HR and Finance without dataset-specific threshold repair.  Until
that happens, lifecycle context, caching and asynchronous execution remain
out of scope.

The compact artifact is
`results/systems/risk-limited-vidore-finance.json`.  Full score matrices,
models and public dataset files remain in the independent server workspace.
Two independent analysis invocations are byte-identical; the compact result's
SHA-256 is
`be52f8489aa2f51ca5eccdf87605f5a49f842cccb15876d0e5ea73f95ba7f494`.
