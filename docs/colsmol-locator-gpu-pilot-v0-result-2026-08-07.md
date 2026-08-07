# ColSmol locator GPU pilot v0 result (2026-08-07)

## Decision

The frozen continuation gate is **NO-GO**. A full-corpus 256M visual locator is a
useful candidate-escape repair layer, but it is not a drop-in low-cost replacement
for either BM25/DVI-style location or the existing high-fidelity visual index.

The positive result is narrower and more useful: a small learned visual index
exposes a real intermediate operating point between text-only deferral and a full
Omni/ColPali index. Its physical index is roughly two orders of magnitude smaller,
and its deeper candidate lists recover most Full-visual repairs. Its current GPU
build path, however, is slower per page than the high-fidelity reference, and a
fixed Top-20 is not reliable on Industrial.

The protocol was frozen in
`configs/colsmol-locator-gpu-pilot-v0.json` before either test-domain run. No
quality threshold, candidate depth, batch size, or route was changed after seeing
the results.

## Candidate coverage

| Domain | Main BM25@20 hit | Main ColSmol@20 hit | Main BM10 + ColSmol10 hit | Full-visual repair queries | ColSmol@10 repair | @20 | @50 | @100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pharma | 91% | 92% | 94% | 25 | 48% | 80% | 96% | 96% |
| Industrial | 78% | 78% | 83% | 33 | 45.5% | 48.5% | 63.6% | 90.9% |

The stress cohort contains queries missed by BM25 Top-20 but repaired by the
existing Full visual Top-20. ColSmol Top-20 repairs 20/25 Pharma escapes but only
16/33 Industrial escapes. The Industrial gate required at least 50% and misses by
one query. The Top-20 to Top-100 gap is large in both domains, so locator access
depth is itself query- and domain-dependent.

The main-set 10+10 route improves candidate hit over BM25 Top-20 in both domains,
so the visual locator is complementary to text. On the escape-only cohort, however,
ColSmol Top-10 repairs less than half of the queries. A fixed shallow fusion hides
the very pages for which visual retrieval is needed most.

## Measured system cost on RTX 5090

| Domain | Pages | Build GPU seconds | Seconds/page | Physical index | Bytes/page | Peak allocated CUDA memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pharma | 2,313 | 853.3 | 0.369 | 518 MB | 224 KB | 1.95 GiB |
| Industrial | 5,244 | 2,523.7 | 0.481 | 1.51 GB | 288 KB | 2.41 GiB |

The previous Omni full builds took about 0.262 and 0.256 seconds/page on Pharma
and Industrial. ColSmol is therefore about 1.41x and 1.88x slower in this frozen
implementation, rather than below the preregistered 0.5x ceiling. This rejects the
shortcut “smaller model implies cheaper ingestion.” Image tokenization and
multi-vector encoding must be measured directly.

By contrast, the physical ColSmol representation is about 0.22--0.29 MB/page,
versus roughly 11 MB/page for the existing Omni representation. It easily passes
the 20% storage gate and is around 2--3% of the high-fidelity index per page.

## Gate accounting

| Frozen condition | Pharma | Industrial | Overall |
| --- | --- | --- | --- |
| ColSmol@20 repairs at least 50% of Full-visual escapes | pass (80%) | fail (48.5%) | fail |
| BM10 + ColSmol10 does not reduce main BM25@20 hit | pass | pass | pass |
| Index is at most 20% of Omni full | pass | pass | pass |
| Build seconds/page are at most 50% of Omni full | fail | fail | fail |

Because the confirmatory continuation gate failed, the frozen protocol does not
authorize presenting a raw-page verifier follow-up as a successful continuation.
Such a run can still be performed later as an explicitly labelled diagnostic, but
it must not erase this NO-GO result.

## What this says about DVI, EdgeRAG, AgenticOCR, and LightSTAR

This experiment does not claim an end-to-end win over those systems. It isolates
the page-location boundary that a DVI-style system depends on:

- Text Top-20 misses 25 Pharma and 33 Industrial queries that Full visual Top-20
  can repair. Query-time visual understanding cannot recover evidence that the
  cheap locator never sends to it.
- A small full-corpus visual locator repairs many, but not all, such misses. It
  also pays a real full-corpus ingestion cost, so “always build the small visual
  index” is not yet a dominant answer.
- EdgeRAG's same-representation regeneration/cache mechanism and AgenticOCR's
  query-time regional parsing address different stages. They remain relevant
  baselines for storage/recomputation and deferred parsing, respectively.
- LightSTAR already occupies the lightweight visual selection plus semantic
  refinement design space. ReprForge cannot claim that two-stage structure as
  novelty.

The remaining research space is heterogeneous persistent index state: start from
text and/or a compact visual locator, identify the residual pages for which deeper
or high-fidelity visual retrieval changes candidate coverage, and materialize that
capability only when its future quality and reuse benefit amortizes build and
storage cost.

## Next falsifiable experiment

Before another expensive verifier run, use the now-complete BM25, ColSmol, and
Full-visual score surfaces for a residual-materialization headroom audit:

1. Treat BM25 + ColSmol as the deployed cheap locator stack.
2. Measure the remaining query/page escapes at depths 20, 50, and 100.
3. Under 1%, 2%, 5%, 10%, and 20% high-fidelity page budgets, compare random,
   static visual risk, historical workload, risk-plus-workload greedy, and oracle
   page materialization.
4. Report gain recovery over the cheap stack, not only quality retained from Full.
5. Continue only if a realizable selector closes a material part of the oracle gap
   in both domains and promoted pages receive enough future reuse to amortize their
   build cost.

This audit directly tests whether lifecycle management has algorithmic headroom
above a strong small learned locator. If it fails, the honest conclusion is to use
DVI/LightSTAR-style deferral or a full visual index according to workload scale,
not to add a compiler between them.

## Artifacts

- `results/compiler-feasibility/colsmol-locator-gpu-pilot-v0/pharmaceuticals/result.json`
- `results/compiler-feasibility/colsmol-locator-gpu-pilot-v0/pharmaceuticals/ranking.txt`
- `results/compiler-feasibility/colsmol-locator-gpu-pilot-v0/industrial/result.json`
- `results/compiler-feasibility/colsmol-locator-gpu-pilot-v0/industrial/ranking.txt`

The large resumable embedding shards remain on the rental server and are not
required to reproduce the reported metrics from the saved rankings.
