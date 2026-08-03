# Risk-Limited Acquisition on ViDoRe v3 HR

## Decision

Frozen transfer to the official ViDoRe v3 HR English workload is **NO-GO for
a performance claim**.  The decision-level conformal set preserves the
full-candidate quality within the frozen tolerance, but it does not beat the
quality-matched fixed K=20 baseline in physical construction work.

## Experiment

The A100 run regenerated complete token-aware text and full-visual score
traces for 318 queries and 1,110 pages.  The locator is the corpus-wide cheap
text late-interaction representation; the candidate pool is its Top-100.
ColPali-v1.1 visual MaxSim is normalized with the true query vector count.
Five deterministic query-disjoint folds use the same fit/calibration/test
algorithm and `alpha=0.05` frozen on IRPAPERS.  Qrels are used only for the
final nDCG and Recall evaluation.

| Policy | Mean visual pages/query | Unique visual pages | Visual build ms | Exact teacher Top-10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Fixed K=10 | 10.00 | 796 | 83,816.9 | 11.01% | 0.5143 |
| **Fixed K=20** | **20.00** | **932** | **98,011.3** | **91.19%** | **0.5262** |
| Conformal candidate set | 23.36 | 948 | 99,700.1 | 96.23% | 0.5266 |
| Score-envelope acquisition | 50.52 | 1,001 | 105,241.8 | 99.69% | 0.5269 |
| Fixed K=50 | 50.00 | 1,004 | 105,528.0 | 99.69% | 0.5269 |
| Full Top-100 teacher | 100.00 | 1,044 | 109,753.5 | 100% | 0.5269 |

Candidate-set acquisition satisfies the 93% coverage and 0.005 nDCG quality
gates.  It fails the resource gate: the quality-matched fixed K=20 builds
1.7% less visual time.  The score-envelope method again collapses to an
approximately K=50 policy.

Recall@100 is 0.8444 for every row because all policies share the same
Top-100 locator pool and only change its Top-10 visual ordering.  Candidate
recall is therefore reported as a separate locator ceiling, not an
acquisition gain.

## Interpretation

The IRPAPERS result was not a single-dataset accident.  Direct conformal
Top-k coverage is far tighter than simultaneous score coverage, but the set
size concentrates around a robust fixed depth.  The current cheap features
identify a safe cohort, not a cohort whose variable size provides material
systems value.

More importantly, exact teacher fidelity is misaligned with retrieval value.
Fixed K=20 differs from the full teacher on 28 queries, but 19 differences
leave per-query nDCG unchanged and 4 actually improve it; only 5 are harmful.
The conformal candidate set removes 16 of those set differences by building
more pages, yet 11 of its remaining 12 differences are also nDCG-neutral.
Protecting every teacher Top-10 transition therefore overprices changes that
do not affect relevant evidence.

No live progressive executor is timed because the offline unique-build gate
fails.  A scheduler cannot create a speedup when the proposed policy requests
more unique visual pages than fixed K=20.

The compact result is
`results/systems/risk-limited-vidore-hr.json`.  Its runtime traces remain in
the independent server workspace and record source, model and score hashes.
