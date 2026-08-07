# Defer--Materialize Phase Diagram v0

## Decision

**NO-GO for the current method; GO for the underlying research problem.**

The measured results support a non-empty *oracle* region in which a small
persistent visual tier amortizes sooner than Full ingestion and sooner than
repeated DVI-like page inspection. They do not show that the current realizable
`full-corpus ColSmol locator + partial Omni tier` occupies that region.

Two independent failures block the current method:

1. its full-corpus ColSmol construction time already exceeds Full Omni
   construction time before any high-fidelity page is added;
2. the history-selected residual pages reduce, rather than preserve, future
   mean nDCG in both evaluated domains.

The paper may use this phase diagram to define the required operating regime,
but it may not claim that the present selector/locator solves it.

## Frozen accounting

The comparison cancels final answer-generation cost. A persistent retrieval
embedding does not replace query-conditioned reasoning over the final evidence
page. It can only repair candidate escape, reduce the number of raw pages sent
to a query-time VLM, or avoid rebuilding retrieval state.

For diagnostic purposes, let `delta_k` be the number of query-conditioned raw
page inspections avoided per query at the same retrieval-quality target:

```text
DVI incremental cost       = Q * delta_k * verifier_page_seconds
oracle partial build cost  = selected_fraction * Full build seconds
current ReprForge build    = full ColSmol build + oracle partial build
Full build cost            = Full build seconds
```

The measured page-verifier value is GPU forward time, not complete I/O,
preprocessing, H2D, answer-generation, or end-to-end latency. The analysis is a
headroom boundary, not a production latency claim.

## Quality-qualified oracle plans

The frozen quality gate requires at least 90% recovery of Full's nDCG gain and
at most one percentage point loss in query-hit@20. The smallest future-aware
plans satisfying both are:

| Domain | Oracle pages | Corpus fraction | Projected Omni build | Full build |
|---|---:|---:|---:|---:|
| Pharmaceuticals | 116 | 5.02% | 30.40 s | 606.15 s |
| Industrial | 53 | 1.01% | 13.54 s | 1,339.67 s |

This establishes substantial static headroom. It does not make the page set
predictable before future queries are observed.

## Crossover boundaries

The table reports the number of queries after which an endpoint amortizes
relative to repeated DVI-like verification, for several values of `delta_k`.

### Pharmaceuticals

| Avoided pages/query | Oracle partial vs DVI | Full vs DVI | Current stack vs DVI |
|---:|---:|---:|---:|
| 1 | 193.2 | 3,852.7 | 5,617.0 |
| 5 | 38.6 | 770.5 | 1,123.4 |
| 20 | 9.7 | 192.6 | 280.8 |
| 80 | 2.4 | 48.2 | 70.2 |

### Industrial

| Avoided pages/query | Oracle partial vs DVI | Full vs DVI | Current stack vs DVI |
|---:|---:|---:|---:|
| 1 | 88.0 | 8,705.9 | 16,488.5 |
| 5 | 17.6 | 1,741.2 | 3,297.7 |
| 20 | 4.4 | 435.3 | 824.4 |
| 80 | 1.1 | 108.8 | 206.1 |

For example, if persistent retrieval avoids five raw-page VLM checks per query,
the ideal partial plan creates a wide middle interval: about 39--771 queries in
Pharmaceuticals and 18--1,741 in Industrial. This is the strongest evidence
that the research problem itself is real.

## Why the current stack does not occupy the middle

| Domain | ColSmol + qualified partial build | Ratio to Full build | Ratio to Full bytes |
|---|---:|---:|---:|
| Pharmaceuticals | 883.72 s | 1.46x | 6.94% |
| Industrial | 2,537.28 s | 1.89x | 3.54% |

The current stack is strongly storage-efficient but is dominated by Full on
ingestion GPU time. On the frozen grid, comparing only deployable current
endpoints gives `DVI -> Full`; current ColSmol+partial is never the cheapest.

Transfer also fails. Across the history/future residual selector curves, the
best nonzero budget changes future mean nDCG by:

- Pharmaceuticals: `-0.001559` versus the unmaterialized base;
- Industrial: `-0.000765` versus the unmaterialized base.

Thus future-aware page sparsity is not yet a realizable compiler.

## Correction to the three-region story

Query volume alone does not imply that Full becomes optimal. If a fixed 5% page
set preserves quality and has the same per-query execution cost as Full, it
remains cheaper than Full at every horizon. Full is the high-load endpoint only
when at least one of the following occurs:

1. useful visual demand broadens toward the complete corpus;
2. partial coverage no longer meets the quality target;
3. online promotion creates unacceptable p95/p99 latency spikes;
4. workload drift makes the active visual working set unpredictable;
5. partial-stack fixed overhead exceeds the cost of simply building Full.

The meaningful axes are therefore not just low, medium, and high query count.
They are **query horizon, candidate-inspection savings, and visual working-set
breadth/locality**.

## Method consequence

Do not continue optimizing the current ColSmol+Omni stack as the paper method.
The next solution must make the oracle region realizable by one of these routes:

1. a genuinely cheap, efficiently batched coverage representation whose
   measured corpus build time is below Full, not merely smaller in parameter or
   byte count;
2. a structure-aware ingestion compiler that identifies visual-risk objects
   without a full-corpus VLM retrieval pass;
3. a transient estimate--verify--admit mechanism, with low-reuse visual
   long-tail coverage separated from high-reuse workload benefit;
4. a real or defensible temporal workload where reuse and working-set drift are
   measured rather than assumed.

The next GPU gate should not reopen until a CPU/structure policy beats static
type, frequency, and history-residual selection on a held-out workload. If that
gate passes, physical partial-index and end-to-end latency measurements become
worth running.

## Artifacts

- `configs/defer-materialize-phase-diagram-v0.json`
- `results/compiler-feasibility/defer-materialize-phase-diagram-v0-2026-08-07.json`
- `tools/analyze_defer_materialize_phase_diagram.py`
