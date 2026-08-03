# Risk-Limited Acquisition on IRPAPERS

## Decision

The first risk-limited mechanism is implemented and the frozen IRPAPERS gate
is **NO-GO for a performance claim**.  The result is nevertheless decisive:
score-level uncertainty is much too conservative, while a decision-level
conformal candidate set nearly collapses to the already-strong fixed K=10.

This closes two tempting but insufficient algorithms without rejecting the
multimodal representation-acquisition problem.  The next valid transfer is a
workload where one robust fixed K does not already remove the action value.

## Controlled result

The replay uses the complete 180 x 3,230 A100 ColPali-v1.1 score surface,
BM25 Top-100 candidates, five source-paper-disjoint fit/calibration/test
folds, cutoff 5 and `alpha=0.05`.  Policy fitting and calibration do not use
qrels.

| Policy | Mean visual pages/query | Page-events | Unique pages | Exact Top-5 teacher agreement | Recall@5 |
|---|---:|---:|---:|---:|---:|
| Fixed K=5 | 5.00 | 900 | 275 | 55.56% | 78.33% |
| **Fixed K=10** | **10.00** | **1,800** | **511** | **96.11%** | **83.33%** |
| Fixed K=20 | 20.00 | 3,600 | 987 | 100% | 83.33% |
| Score-envelope acquisition | 48.73 | 8,772 | 1,908 | 99.44% | 83.33% |
| Conformal Top-k candidate set | 10.27 | 1,849 | 544 | 96.67% | 83.33% |
| Full Top-100 teacher | 100.00 | 18,000 | 2,318 | 100% | 83.33% |

The one-sided score envelope attains 96.11% simultaneous upper coverage and
misses only one teacher Top-5 set, but it builds 4.87x as many page-events as
fixed K=10.  Directly calibrating a candidate set is far tighter: it covers
the full-score Top-5 on 96.67% of held-out queries and preserves Recall@5.
However, it still uses 2.72% more events and 6.46% more unique pages than
fixed K=10.

## Mechanism interpretation

The score-envelope failure is structural.  Certifying 100 noisy visual
scores is harder than protecting five rank members, and the largest residual
in each calibration query forces broad intervals.  One-sided bounds improve
the mean only from 52.97 to 48.73 pages.

The candidate-set result demonstrates that decision-level uncertainty is the
right abstraction, but also that IRPAPERS provides no useful adaptive policy
gap under this score contract.  BM25 Top-10 already achieves the full Top-100
teacher's Recall@5 and almost the same set agreement at lower work.  Adding a
learned adaptive set cannot create value when a fixed prefix is already this
robust.

No A100 executor timing is run for this policy because the preregistered event
gate fails.  Batching or caching cannot reverse a policy that requests more
unique representations than the fixed baseline.

## What remains open

The algorithm transfers without threshold repair to ViDoRe HR and Finance,
where fixed K=20 accumulates 80.6% and 63.1% corpus coverage respectively.
Those workloads can still contain a query-dependent acquisition gap that the
single-gold, BM25-friendly IRPAPERS workload masks.  Both score-envelope and
candidate-set methods remain frozen baselines during transfer; hand-tuned
rank or content rules are forbidden.

The compact result is
`results/systems/risk-limited-irpapers.json` with SHA-256
`fbc703737b2c28496ddb993e2407877d8c4acfa494e59b7236cc38a1c0be9fed`.
