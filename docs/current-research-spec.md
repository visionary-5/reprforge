# ReprForge current research specification

Status: active specification, 2026-08-07  
Scope: problem and experimental contract; method remains a hypothesis until
the physical and transfer gates pass.

## 1. Research problem

Multimodal RAG ingestion is not a single embedding call. A document is parsed
into pages, text blocks, tables, figures, captions, and relations; each unit
can be represented by native/OCR text, structural metadata, a compact visual
embedding, a high-fidelity visual late-interaction embedding, or no persistent
visual state at all. These choices alter both retrieval capability and the
cost paid before the first query.

Let `D` be the document corpus, `U(D)` a hierarchy of searchable knowledge
units, `R` the available representation operators, and `W` a query workload.
For every unit `u`, the compiler chooses a representation state over time:

```text
cheap searchable state
    -> transient query-conditioned visual inspection
    -> persistent visual retrieval representation
    -> optional higher-fidelity persistent representation
```

The research problem is to choose unit boundaries, representation states, and
promotion times under measured build, storage, and query budgets while
preserving retrieval and downstream answer quality.

This is an index physical-design problem. Query-side candidate depth and cache
state are observations available to the planner, not the primary contribution.

## 2. Why the problem is not already answered

The closest endpoints make different assumptions:

- **Full visual ingestion** materializes visual retrieval state for every
  page. It maximizes immediate visual discoverability but pays before knowing
  which pages or representations the workload will use.
- **Deferred Visual Ingestion (DVI)** builds a structural/text locator and
  sends located raw pages to a VLM at query time. It assumes cheap indexes can
  find the evidence and that repeated visual work need not become a persistent
  retrieval asset.
- **AgenticOCR** moves OCR/region parsing to query time. It optimizes which
  regions are read for the current answer, not which units acquire reusable
  visual retrieval state.
- **EdgeRAG** removes embeddings within an IVF organization and regenerates
  the same representation on demand, then caches it. Its central constraint is
  memory for an already defined representation, not capability differences
  between text discovery and visual discovery.
- **LightSTAR-style cascades** use a lightweight selector and expensive
  refinement only for current candidates. Persistence and future reuse are
  not automatically beneficial.

ReprForge therefore asks when a system should continue to defer visual work
and when it should pay once to create a persistent visual retrieval asset.
The expected contribution is a quality- and cost-aware physical design across
these endpoints, not the generic observation that indexing can be delayed.

## 3. Representation and cost model

The first complete protocol uses pages as the physical materialization unit
because all current public retrieval labels and physical score banks are page
aligned. Page is not assumed to be the final semantic unit. Layout/region
granularity remains a controlled extension and must beat its additional
vector, build, and fusion cost.

For each page `u`, the catalog contains:

| State | Persistent | Corpus-wide discovery | Query-conditioned | Main cost |
|---|---|---|---|---|
| text/structure `T` | yes | yes | no | parsing/OCR, sparse or dense index |
| compact visual `V_c` | optional | yes among represented pages | no | compact visual build/storage |
| full visual `V_f` | optional | yes among represented pages | no | high-fidelity build/storage/search |
| transient inspection `X` | no | no; depends on another locator | yes | raw-page I/O and VLM time per use |

Costs are reported as a vector, not hidden inside one arbitrary weighted sum:

- ingestion wall time and GPU-seconds;
- serialized and resident bytes;
- query-time raw pages processed and VLM GPU-seconds;
- retrieval and reranking latency, including p50/p95/p99;
- promotion stalls, evictions, and rebuilt bytes;
- final retrieval and answer quality.

Pareto frontiers and quality-constrained cost minimization are primary. A
single scalar objective may be used only for an ablation with disclosed
weights.

## 4. Method hypothesis: progressive visual materialization

The candidate compiler has three logically separate decisions.

### 4.1 Cheap coverage layer

Build a corpus-wide text/structure locator for every unit. A compact visual
locator is included only if its *measured* build cost and unique discovery gain
justify it. The current full-corpus ColSmol implementation is one candidate,
not a required component of the method.

### 4.2 Signed materialization value

A persistent visual state is valuable only when it supplies one or both of:

1. **discovery coverage**: it prevents evidence that cheap locators cannot
   place in the query candidate set;
2. **workload benefit**: it improves future ranking or avoids repeated
   query-time visual work enough to amortize construction and storage.

Visual state can also be harmful by introducing high-scoring distractors.
Selection therefore estimates signed listwise benefit, not a binary page type
or embedding reconstruction loss:

```text
value(u) = expected future evidence repair
         - expected ranking interference
         + avoided repeated transient work
         - build, storage, and maintenance cost
```

The deployable estimator may use only information available before the future
test workload: document structure, OCR quality, cheap-locator scores and
disagreement, historical query/page interactions, previous transient outcomes,
and measured operator cost. Test qrels and complete visual score surfaces are
forbidden. Oracle selectors may use them only to measure headroom.

### 4.3 Defer, inspect, or promote

For a query, the runtime can accept the cheap result, inspect located raw pages
transiently, or use already materialized visual retrieval state. After an
inspection, an admission rule may persist a representation only when estimated
future reuse and discovery value exceed its one-time and ongoing costs.

The initial method baselines are deliberately simple: never promote, promote
on first touch, promote on second touch, frequency/LFU admission, and a
cost-blind residency policy. A learned or optimization-based policy is added
only after these baselines expose a stable gap.

## 5. Falsifiable hypotheses and present evidence

| ID | Hypothesis | Present status | What would settle it |
|---|---|---|---|
| H1 | Cheap and visual locators have material, complementary discovery failures. | Partially supported on CS, Pharma, and Industrial by the DVI-like verifier pilot. | Strong text/structure/dense baselines and sealed-domain transfer. |
| H2 | Visual representation value is sparse enough that partial physical materialization has headroom. | Supported only by future-aware oracle curves on Pharma and Industrial. | Physical 1/2/5/10/20/40% curves with gain recovery and candidate escape. |
| H3 | Useful units are predictable without future qrels or Full visual scores. | Not established; current history-residual and simple visual-risk selectors are weak. | Document/query-disjoint policy transfer against random, coverage, type, and frequency. |
| H4 | Persistence beats repeated transient analysis for realistic repeated workloads. | Not established; public benchmark order is not a temporal trace. | Measured locality/drift traces, multiple random orders, and cumulative cost crossover. |
| H5 | Logical savings translate to physical ingestion, I/O, and tail-latency gains. | Not established. Existing verifier timing excludes full I/O/preprocessing. | Same-hardware physical execution with decomposed and end-to-end timing. |
| H6 | Sub-page organization improves value density enough to justify its cost. | Current mechanical fixed/XY-cut regions fail this gate. | A structure-aware region operator with downstream quality and cost benefit. |

No single hypothesis determines the entire direction. A failed locator, unit
granularity, or admission estimator is a bottleneck to diagnose, not evidence
that multimodal index physical design is meaningless.

## 6. Paper contribution boundary

A paper-level result requires all three contributions below.

1. **Problem and benchmark contract:** a defer--materialize physical-design
   formulation with explicit capability, cumulative cost, locality, drift, and
   quality measurements.
2. **Compiler method:** a realizable policy that jointly accounts for discovery
   risk, signed ranking value, reuse, and physical cost. Decoupling, caching,
   routing, or page typing alone is not sufficient.
3. **Physical evidence:** stable quality-constrained savings over DVI-like
   defer, Full materialization, transient cascades, static partial selection,
   and simple cache/admission policies on multiple public domains and at least
   one independently constructed benchmark family.

ReprForge need not dominate every endpoint. A valid result may identify a
measured operating region: defer wins under low reuse and reliable cheap
localization; Full wins when visual demand is broad and sustained; progressive
materialization wins when visual discovery matters but the useful working set
is concentrated or evolves gradually.

## 7. Decision discipline

- Development domains may shape the estimator; sealed transfer domains may
  not.
- A benchmark serialization order is never described as a natural workload.
- “Full” is a strong fixed representation, not a relevance oracle.
- “99% of Full quality” is always accompanied by absolute quality and visual
  gain recovery relative to text-only.
- Projected page-fraction cost is never substituted for measured physical
  build time in a final systems claim.
- A negative gate closes a concrete method/configuration, not the broad problem
  unless the strongest oracle also shows no useful headroom.

The executable protocol and experiment order are defined in
[`progressive-materialization-experiment-matrix.md`](progressive-materialization-experiment-matrix.md).
