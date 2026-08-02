# Sparse Cost--Risk Admission Contract

## Scope

This stage upgrades pairwise representation admission from a full-history
teacher analysis into a deployable planning contract.  It asks whether a
small, explicitly charged sample of visual what-if observations can support a
cost-aware plan on an unseen source-paper workload.

The mechanism has four frozen parts:

1. **Sparse probes.** Queries are selected from locator-visible boundary
   margins, balanced across source papers. A probe opens its complete Top-10
   visual cohort and charges every unique representation it touches.
2. **Interpretable value model.** A Beta posterior indexed by challenger rank
   and BM25 boundary-margin bin estimates the probability that a candidate
   crosses Top-5. Its upper confidence estimate weights the incumbent--
   challenger graph.
3. **Physical cost model.** Historical A100 runs fit a non-negative model over
   atomic page construction, encoder batches and MaxSim events. The selector
   purchases completed boundary comparisons per marginal predicted
   millisecond, not per page.
4. **Risk controller.** Every probed source paper is predicted by a model fit
   without that paper. A group-bootstrap upper bound chooses the smallest cost
   fraction whose extra teacher disagreement is at most five percentage
   points at 90% empirical confidence. Insufficient support or a failed grid
   falls back exactly to the independent 20% baseline.

No qrel is accepted by the planner. The held-out fold's visual scores are
opened only after its page set and cost fraction are frozen. Qrels are used
only to report Recall@5.

## Public protocol

- complete public IRPAPERS score surface: 3,230 pages and 180 questions;
- official ColPali-v1.1 visual adapter scores;
- BM25 Top-10 candidate cohort and Top-5 result boundary;
- five deterministic source-paper-disjoint outer folds;
- sparse probe fractions of 10%, 20% and 40% of historical queries;
- 20% independent incident-risk admission as the deployment baseline;
- previously measured eager atomic A100 runs as cost-model observations.

The full score surface is an offline evaluator. An experiment is invalid if
the planner indexes unprobed training visual values, held-out visual values or
qrels.

## Decision gates

The mechanism may proceed to a new A100 execution only if the offline result
satisfies all of the following:

- at least 15% fewer selected physical pages than the independent baseline;
- at least 1.15x predicted visual construction-plus-scoring speedup;
- aggregate exact-teacher agreement loses no more than five percentage
  points;
- no held-out fold loses Recall@5;
- source-paper cross-fitting, minimum-support fallback and probe charges are
  enabled.

A complete cold-start systems claim additionally requires probe cost to be no
larger than the work it saves in the first evaluation horizon. If probes need
amortization, ReprForge must report the break-even episode count and supply a
public recurring-query or update trace that actually reaches it. A predicted
online speedup alone is not an end-to-end result.

The next physical gate, if authorized, requires two fold-interleaved A100
repetitions and a measured speedup above 1.15x. The cost model itself cannot
establish that claim.

