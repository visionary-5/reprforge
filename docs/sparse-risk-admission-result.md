# Sparse Cost--Risk Admission Result

## Verdict

The first joint value--cost--risk controller exposes useful online headroom,
but fails the cold-start systems gate:

> **algorithmic-headroom-positive; probe-debt-unresolved**

The new planner is more selective than the previous exact-teacher controller
and does not use held-out visual scores or qrels. Its sparse evidence is still
far more expensive than the work saved by one IRPAPERS workload episode.

## Algorithm implemented

The planner first compiles probed queries into an interpretable boundary-risk
table over challenger rank and locator margin. It uses the posterior mean plus
one uncertainty standard deviation to weight the existing complementary-view
graph. A measured physical model then evaluates an atomic plan as:

```text
setup + page_ms * materialized_pages
      + batch_ms * encoder_batches
      + score_ms * visual_candidate_events
```

The graph selector greedily buys the complete incumbent--challenger comparison
with the largest newly covered risk per marginal predicted millisecond.
Source-paper leave-one-group-out predictions calibrate a permitted five-point
teacher-agreement loss. A failed or under-supported calibration returns the
independent baseline itself rather than silently executing an unverified plan.

This is a real algorithmic change from the previous system: it jointly chooses
what to build and how much physical work to spend, while exposing its evidence,
uncertainty, cost and fallback decision.

## IRPAPERS outer-fold result

| Historical probe fraction | Selected-page reduction | Predicted online speedup | Exact-teacher delta | Recall@5 baseline -> planner | Fold Recall regressions |
|---:|---:|---:|---:|---:|---:|
| 10% | 0.0% | 1.00x | 0.00 pp | 79.44% -> 79.44% | 0/5 |
| 20% | 24.44% | 1.32x | -1.67 pp | 79.44% -> 80.56% | 0/5 |
| 40% | 28.15% | 1.39x | -2.78 pp | 79.44% -> 81.67% | 0/5 |

The 10% controller has fewer than 20 calibration queries and correctly falls
back in every fold. At 20%, only two folds pass the empirical risk gate; at
40%, four pass. Recall@5 never falls in a held-out fold, but exact-teacher
agreement can move unevenly (the largest fold loss is 17.14 points). This
prevents a strong risk-generalization claim even though aggregate agreement
stays inside the configured five-point allowance.

The physical model is fit from 20 prior eager A100 observations. Non-negative
least squares assigns 268.80 ms per visual page and zero measurable marginal
weight to encoder-batch and MaxSim-event terms on this small corpus. Its
relative RMSE is 9.54%. This independently confirms that construction, not
candidate scoring, is the lever, but the model is not accurate enough to
replace a physical run.

## Probe accounting

| Probe fraction | Outer-fold probe pages | Estimated probe work | Estimated online work saved | Break-even episodes |
|---:|---:|---:|---:|---:|
| 20% | 978 | 262.89 s | 8.87 s/episode | 29.64 |
| 40% | 1,410 | 379.01 s | 10.21 s/episode | 37.11 |

These sums are cross-validation accounting: each outer deployment is charged
for its own historical probes. They demonstrate why “use sparse probes” is not
yet a contribution. Complete Top-10 query probes are sparse relative to a full
3,230-page visual index, but expensive relative to the 12--33 page plans being
optimized.

## Design review

The result rejects two shortcuts:

1. A full historical visual score surface cannot be hidden behind the word
   “training.” Its physical acquisition cost changes the system verdict.
2. A lower page budget chosen from a small aggregate calibration set is unsafe.
   Source-group cross-fitting and exact baseline fallback materially change the
   reported result.

The next mechanism should reduce probe debt rather than tune the graph
selector. The most direct path is **pair-level estimate--then--verify with
reusable probe artifacts**: score only an incumbent and selected challengers,
retain those physical views if admitted, and buy another probe only when its
expected boundary information exceeds its incremental build cost. That would
turn probing and materialization from two separate bills into one progressive
physical-design action.

No new A100 run is justified by this result alone. First, the pair-level probe
must reduce the estimated break-even below the intended public workload
horizon while preserving the outer-fold quality gate. Only then should the
predicted 1.32--1.39x online speedup be physically tested.

