# Candidate Representation View Contract

## System question

ReprForge treats an expensive multimodal representation as a database
physical-design object, rather than an embedding that must exist before query
execution begins.

The bounded question for this slice is:

> Can the system propose, probe, verify and publish representation views under
> explicit build and storage budgets, without making unverified state visible
> to retrieval?

This slice validates the control plane. It does not claim that the current
utility estimate predicts retrieval quality.

## View semantics

A candidate representation view is identified by `(item_id, route)`. For
example, `(page-17, full-visual)` is a derived physical view of page 17. The
catalog stores metadata before the expensive vector exists:

- parent representation;
- expected utility and uncertainty;
- expected cross-query reuse;
- probe, full-build, storage and maintenance cost estimates;
- lifecycle state and revision.

The legal lifecycle is:

```text
hypothetical -> probing -> verified -> materializing -> materialized
                         \-> rejected
```

An interrupted probe returns to `hypothetical`; an interrupted materialization
returns to `verified`. A failed or incomplete executor result cannot publish a
view.

## Planning contract

The first probe planner uses optimistic utility density:

```text
(expected utility + beta * uncertainty) * expected reuse / probe cost
```

This is an intentionally transparent scheduling baseline, not the proposed
paper algorithm. The materialization planner ranks verified views by observed
workload utility over remaining lifecycle cost. It enforces:

- a hard remaining-build budget;
- a hard storage budget;
- at most one representation in the same `(item, slot)` for this prototype;
- deterministic ordering and tie breaking.

The database precedent is candidate generation followed by hypothetical and
actual verification in
[Automatic Indexing in Oracle](https://www.vldb.org/pvldb/vol18/p4924-chakkappen.pdf).
The controllable per-query refinement budget follows the motivation of
[Progressive Indexes](https://www.vldb.org/pvldb/vol12/p2366-holanda.pdf).
ReprForge differs in the unresolved estimator: an index cost model predicts
latency, while a missing visual representation can change a non-additive
ranking and downstream answer.

## Query execution contract

The existing BM25 cohort compiler now accepts a published page set and a
historical visual prior by locator rank. It physically encodes only admitted
pages, while keeping the complete BM25 cohort as the logical candidate set.
Unbuilt candidates use the train-only rank prior.

At least two observed pages are required for within-query visual z-score
normalization. If only one admitted page appears in a query, its representation
can remain resident for future reuse, but the current executor cannot safely
derive a candidate-relative visual score from that single value. This is a
measured design constraint, not silently patched with a global score scale.

## Scale gate

The CPU control plane must complete both profiles without illegal states or a
snapshot mismatch:

1. IRPAPERS-like: 3,230 corpus pages, 180 queries, candidate depth 10;
2. ViDoRe-v3-like: 26,000 corpus pages, 3,100 queries, candidate depth 20.

The second profile should remain below 2 seconds of candidate generation,
0.25 seconds of materialization planning and 128 MiB peak Python allocation on
the current local development machine. These thresholds test engineering
feasibility only.

## Next empirical gate

The next quality experiment must use a public score surface or real encoder,
source-disjoint calibration and a cold physical execution. It must compare:

- full visual prebuild;
- fixed candidate depth;
- frequency-only admission;
- boundary-risk admission;
- probe--verify--materialize;
- an oracle that may inspect the complete visual surface.

The initial pass criterion remains at least 20% fewer physical encodes, no
more than one Recall@5 query lost, at least 1.15x lower construction plus
retrieval time, and no regression against frequency admission. A control-plane
scale pass does not satisfy this gate.
