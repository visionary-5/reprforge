# Elastic Representation: Strong-Baseline Result

## Decision

The binary retention policy is **not a ReprForge contribution**. Standard
capacity-aware database caching closes its apparent advantage.

This result supersedes the initial V4 comparison against only `no_cache` and
unbounded `resident`. It does not invalidate the implementation or cost model;
it changes what they may be used to claim.

## Why this baseline is required

[Linear Elastic Caching via Ski Rental (CIDR 2025)](https://www.vldb.org/cidrdb/papers/2025/p22-kumar.pdf)
combines a per-page TTL with a conventional capacity eviction policy. Its
evaluation uses GDSF as the main policy, verifies qualitatively similar results
with S3FIFO, and compares deterministic Breakeven, randomized ski rental, and
learned variants. The paper reports deployment in Google Spanner and about 5%
total-cost reduction. ReprForge's original `ski_ttl` is the published
Breakeven idea applied to visual representations; it is a baseline, not a new
algorithm.

The new replay implements a common contract for:

- transient recomputation (`no_cache`);
- fixed-capacity LRU;
- fixed-capacity Greedy-Dual-Size-Frequency (GDSF);
- GDSF plus deterministic Breakeven TTL;
- GDSF plus the classical randomized ski-rental TTL, averaged over five seeds;
- GDSF plus ReprForge's reuse-verified Breakeven admission.

All methods receive the same BM25 Top-20 request stream, per-page measured
ColPali encode time, visual bytes, and memory price. Capacities sweep 5%, 10%,
20%, 40%, 60%, 80%, and 100% of the full visual index. Every family is allowed
to report its post-hoc best capacity. This is an optimistic lower envelope,
not a deployable tuning protocol, but it gives the strongest possible test of
whether the mechanism itself has headroom.

## Results at the previously claimed points

| Dataset | Price | Previous weak comparison | Strongest capacity baseline | ReprForge verified+GDSF | ReprForge gain |
|---|---:|---:|---:|---:|---:|
| ViDoRe HR | 4 | +6.59% | GDSF 491,053 | 494,928 | **-0.79%** |
| ViDoRe Finance | 2 | +9.49% | GDSF 498,817 | 500,604 | **-0.36%** |

Across the complete nine-point memory-price sweep:

- HR wins at **0/9** points;
- Finance wins at **1/9** points, by only **1.38%** at price 4;
- at high prices, `no_cache` is stronger than retaining pages until a TTL;
- at low and medium prices, capacity-aware GDSF removes weak residents without
  paying the second-miss penalty of reuse-verified admission.

Five deterministic query-order permutations at the two original claim points
do not rescue the mechanism:

- HR loses all five, by 1.26%--2.79% (mean 1.76%);
- Finance wins four of five, but only by 0.54% on average and at most 1.16%;
- GDSF is the strongest baseline in all ten reordered runs.

The preregistered 5% cross-dataset gate therefore fails decisively. Retrieval
quality remains 0.53729 nDCG@10 on HR and 0.56280 on Finance for all retention
policies because activation and fusion are unchanged.

## Mechanism explanation

The original improvement did not reveal a missing RAG-specific context.
`resident` admitted every first-touch page forever, including one-off and weak
reuse pages. `verified_ski_ttl` corrected that obvious pollution. GDSF already
solves the more general problem: it combines access frequency, refault cost,
item size, and capacity pressure when deciding what to evict. Once GDSF is
present, the two-hit rule adds no stable information advantage.

This is precisely the kind of baseline correction the repository should keep.
It prevents a valid implementation improvement from being mistaken for a
research contribution.

## What remains outside existing binary caching

The broader ReprForge system is not reduced to binary caching. Standard cache
policies assume that a page is either present or absent and that every hit
returns the same object. A multimodal index can instead hold several physical
representations of the same content:

```text
text/OCR  ->  compact visual  ->  full late-interaction visual
 cheap           medium                    expensive
 coarse          partial                    strongest
```

A compact-state hit may be sufficient for one query but not another. Moving
to a richer state changes ranking quality, GPU scoring work, bytes, and future
upgrade cost. An update may invalidate only some states. These semantics are
not expressed by LRU, GDSF, or binary elastic caching.

The next research question is therefore:

> Can a quality-aware physical-design controller choose and maintain one of
> several representation states per content unit, using observed ranking
> intervention and reuse, and improve the end-to-end quality--build--memory--
> latency frontier over both fixed representations and strong binary caches?

Before implementing that controller, the next benchmark must add at least one
real compact visual state to the same HR and Finance traces. A multi-state
offline oracle must exceed the best fixed representation plus GDSF envelope by
at least 10% cost at matched quality on both datasets. Otherwise the added
state machine has no measured reason to exist.

The compact frozen artifact is
`results/elastic-representation-v4/vidore-strong-baselines.json` with SHA-256
`b1a7d814dc64e9cb721e8109f5d399801f2f174bbfb24d097cd29b7538022032`.
The bounded order-sensitivity artifacts are `hr-p4-strong-order.json`
(`461cbc19484112ba8809d0a5cdd361bc5f3b00ffe17d903c668b9be0ae0389cd`)
and `finance-p2-strong-order.json`
(`14e90d2cef870e8bc206c2f7e5636def7e7dfa7224d001495e043b9a6d4de29d`).
