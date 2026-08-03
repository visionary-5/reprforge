# Elastic Representation V4: Result

## Result

The bounded two-state mechanism passes its feasibility gate, but leaves a
large algorithmic gap.

At the cost crossover, `verified_ski_ttl` beats the better of recompute-every-
time and retain-forever without changing candidate sets, scores, or retrieval
quality:

| Dataset | Memory-price point | No cache | Resident | Verified elastic | Gain over best fixed | Offline oracle | Elastic over oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViDoRe HR | 4 | 571,398 | 532,159 | **497,107** | **6.59%** | 333,987 | 1.49x |
| ViDoRe Finance | 2 | 635,942 | 557,162 | **504,287** | **9.49%** | 354,716 | 1.42x |

Costs are encode-millisecond equivalents plus the explicit integrated-memory
term from the contract. They are not wall-clock query latency.

The quality contract is invariant across retention policies:

- HR candidate-relative fusion: **0.53729 nDCG@10**;
- Finance candidate-relative fusion: **0.56280 nDCG@10**.

This invariance matters: the saving is caused only by changing the lifetime of
already-required visual state, not by removing visual work and accepting a
hidden quality loss.

## Query-order sensitivity

ViDoRe does not provide a natural production arrival order. We therefore
replayed ten deterministic random permutations as a sensitivity test, not as
a substitute for a real trace.

At the same cost points, `verified_ski_ttl` improves over the better fixed
extreme by:

- HR: mean **7.43%**, range **6.43%--8.89%**;
- Finance: mean **11.46%**, range **10.49%--12.96%**.

The crossover is therefore not an accident of the published benchmark order.
It still does not establish behavior under temporal locality, drift, bursts,
or document updates.

## What the experiment teaches us

The useful insight is not “use BM25 Top-20” and not “use a TTL.” It is a
three-part decomposition:

1. **representation activation creates a demand stream**: candidate selection
   determines which expensive states are needed now;
2. **representation retention is physical design**: measured rebuild cost,
   state size, and reuse distance determine whether that state should survive;
3. **the correct action changes by operating regime**: retain-forever wins
   when memory is cheap, verified elastic retention wins near the crossover,
   and transient recomputation wins when memory is expensive.

Thus no single “always cache” or “never cache” implementation is sufficient.
A deployable envelope can select the appropriate regime from measured physical
costs; inside the elastic regime, per-page lifecycle decisions add another
6.6%--9.5% on the original traces.

## What failed or remains weak

The simple online algorithm remains **42%--49% above the offline lower bound**
at its best crossover points. The gap is the next real algorithmic problem.
Break-even TTL knows item size and recomputation cost but does not estimate the
next reuse distance or distinguish stable popularity, bursts, and drift. The
two-hit admission rule filters one-off pages but deliberately misses the first
reuse. Adding more thresholds would be curve fitting, not a contribution.

The experiment also does not establish end-to-end speedup. The current trace
has per-page encode cost and bytes, but not a concurrent arrival process,
batched GPU queueing, overlap, or update maintenance. Those components must be
measured before describing the cost reduction as latency or throughput.

## Decision and next design

**Conditional GO** for one integrated multi-state prototype; no paper claim
yet.

The next algorithm should be a representation physical-design controller,
not a larger page classifier. For each page and representation state it should
maintain a measured tuple:

`(quality intervention, build cost, query-time serve cost, bytes, reuse hazard)`.

It should then:

1. activate a representation only when the current query needs its evidence;
2. use observed score/rank intervention to verify that the representation was
   useful;
3. estimate a time-decayed reuse hazard from the request stream;
4. retain, downgrade, or evict the state when expected avoided rebuild work no
   longer pays for memory and maintenance;
5. operate over at least three states: text/coarse, compact visual, and full
   visual, using reproducible representation artifacts rather than simulated
   cost ratios.

This design borrows the online cost discipline of elastic database caching
and the representation ladder established by MetaEmbed/Reminisce. A future
contribution would have to lie in their joint realization for multimodal RAG:
quality-aware state transitions, update-safe physical maintenance, and an
end-to-end implementation that improves the quality--build--memory--latency
frontier against full visual indexing, transient refinement, fixed compact
representations, and standard cache policies.

The frozen result is
`results/elastic-representation-v4/vidore.json` with SHA-256
`404005265f3615074480e940e6cfb2b2dd18b3f46b447884280b4b8132af0418`.
