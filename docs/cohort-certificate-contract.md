# Cohort certificate oracle contract

## Purpose

The oracle asks whether a complete expensive Top-20 candidate cohort contains
a substantially smaller set of exact representation states that is sufficient
to reproduce the complete teacher's Top-10 result. We report two objectives:
exact set preservation and exact ordered-list preservation.

It is a headroom test for selective construction, not a deployable selector.

## Partial-fusion semantics

1. The cheap route freezes the Top-20 candidate set.
2. Candidate cheap scores are standardized within that set.
3. Three deterministic rank-coverage anchors are always observed.
4. A query-local quadratic model completes unobserved expensive scores.
5. Materialized candidates replace their prediction with the exact expensive
   score.
6. Completed expensive scores are standardized and added to cheap scores.
7. The candidate set is reranked above the untouched cheap tail.

The complete teacher observes all 20 expensive scores under the same fusion
equation. This contract therefore isolates representation selection from
candidate generation.

## Oracle

Backward elimination begins with all 20 candidates observed. At each step it
removes the candidate that best preserves the teacher order among legal
removals. A legal removal retains either the exact teacher Top-10 set or the
exact teacher Top-10 ordered list, depending on the objective. It stops when
no further single removal is legal. Qrels are never consulted.

The result is inclusion-minimal, not necessarily globally minimum. A
deterministic subset of queries is exhaustively enumerated below the greedy
upper bound to measure the optimization gap.

## Gate

Selective construction remains viable only if both HR and Finance satisfy:

- median certificate size no more than 8 of 20;
- the order-preserving certificate retains 100% of full-fusion retrieval gain;
- exact audit does not expose a large greedy optimization gap;
- unique physical states and measured build time materially decrease.

If the oracle fails, no learned selector can repair the missing headroom under
this partial-fusion contract.

## Outcome on frozen surfaces

The gate passes for exact ordered Top-5 headroom on HR and Finance (median
7/20 and 8/20) but fails for exact ordered Top-10 (12/20 and 13/20). IRPAPERS
also passes at Top-5 with median 7/20. MMDocIR pool-25 to full-image refinement
needs median 4.5/10, although its full-refinement quality gain is only 0.0037
nDCG@5 and therefore does not justify refinement by itself.

This is an oracle statement only. Query-holdout scalar selectors fail to
recover the small certificates consistently, so certificate existence must not
be reported as deployable low-budget selection.
