# Representation value sparsity atlas v0 result

Date: 2026-08-07  
Frozen protocol commit: `abff430`  
Decision: **page-value sparsity GO; deployable selector not yet established**

## Question

Above an always-present BM25 plus ColSmol index, this audit adds exactly one page's
high-fidelity Omni representation and measures its signed mean nDCG@10 change over
the complete workload. A page can help some queries and harm others; its label is
therefore its **net workload value**, not whether it is visually complex, relevant
once, or able to repair one candidate escape.

The protocol and thresholds were frozen before either domain result. Future qrels
and complete Omni ranks are used only to construct oracle labels.

## Value is sparse and signed

| Domain | Pages | Positive | Negative | Neutral | Top 2% positive-mass recovery | Top 5% recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pharma | 2,313 | 158 (6.83%) | 1,537 (66.45%) | 618 (26.72%) | 63.66% | 95.18% |
| Industrial | 5,244 | 185 (3.53%) | 1,673 (31.90%) | 3,386 (64.57%) | 88.56% | 100.00% |

All preregistered checks pass in both domains:

- positive pages occupy at most 20% of the corpus;
- the best 5% pages contain at least 70% of positive singleton value;
- at least 1% of pages have negative net value.

This is stronger than the earlier candidate-escape result. Visual value is not
only concentrated; it has a large negative side. Full materialization exposes many
representations that degrade the shared ranking surface.

## Relevant or visually useful once does not imply worth materializing

The signed decomposition exposes why simple selectors failed:

| Diagnostic | Pharma | Industrial |
| --- | ---: | ---: |
| Positive pages that also harm at least one query | 83 / 158 (52.5%) | 98 / 185 (53.0%) |
| Negative pages that are relevant to at least one query | 660 / 1,537 (42.9%) | 340 / 1,673 (20.3%) |
| Escape-repair events attributed to positive pages | 8 | 19 |
| Escape-repair events attributed to negative pages | 30 | 24 |

A page can repair a difficult query yet remain negative overall because its visual
rank contribution disturbs many other queries. Conversely, the highest-net-value
pages are not necessarily the residual pages. This separates two objectives that
must both appear in a method:

1. coverage of rare visual evidence and candidate escape;
2. control of cross-query ranking interference.

Optimizing only either term is insufficient.

## What can be predicted cheaply

Univariate AUC for identifying positive/negative singleton value:

| Signal | Pharma positive | Pharma negative | Industrial positive | Industrial negative |
| --- | ---: | ---: | ---: | ---: |
| BM25 Top-100 workload frequency | 0.506 | 0.781 | 0.791 | 0.816 |
| ColSmol Top-100 workload frequency | 0.487 | 0.812 | 0.783 | 0.819 |
| OCR text scarcity | 0.537 | 0.409 | 0.418 | 0.342 |
| Edge energy | 0.559 | 0.607 | 0.501 | 0.587 |
| Image entropy | 0.507 | 0.562 | 0.515 | 0.600 |
| Qrel frequency, oracle-only | 0.827 | 0.585 | 0.940 | 0.578 |

Static visual complexity is close to random. Base-index workload visibility is a
surprisingly stable signal for **negative** value in both domains, while its
positive-value signal transfers poorly from Industrial to Pharma. Qrel frequency
is strong but unavailable at ingestion time unless converted into historical click
or success feedback.

The result rules out a page-type heuristic as the core method. The realizable
selector needs workload features and must explicitly predict harm, not only likely
benefit.

## Relation to the residual greedy oracle

The interference-aware future-information greedy oracle selected at most 5% pages:

| Domain | Base nDCG@10 | Full stack | Greedy at 1% | Greedy at 2% | Greedy at 5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pharma | 0.57472 | 0.61630 | 0.60969 | 0.62744 | 0.66097 |
| Industrial | 0.46717 | 0.51298 | 0.56275 | 0.60162 | 0.62367 |

Selective materialization exceeds Full because it avoids many negative pages. The
singleton atlas explains this result, while the greedy run confirms that positive
first-order values remain useful under interactions. Singleton mass is not assumed
additive and is not itself a deployable policy.

## Updated research problem

The evidence supports a build-time physical-design problem rather than a query
Top-k routing problem:

> Given a document collection, a calibration workload, representation operators,
> and an ingestion/storage budget, decide which knowledge units should be promoted
> to which persistent multimodal representation state so that positive evidence is
> exposed while cross-query interference is controlled.

For a unit `u`, representation `r`, and workload `W`, the relevant target is signed:

```text
Value(u, r | W)
  = retrieval benefit on helped queries
  - ranking interference on harmed queries
  - build cost
  - storage cost.
```

This differs from Full VLM indexing, always-lazy DVI processing, and same-
representation cache/regeneration. The contribution is not the observation that
some pages are visual; it is workload-aware compilation of heterogeneous,
multi-granularity persistent representation state.

## Granularity boundary

The current ViDoRe artifacts contain page images and page OCR strings, but no
original PDFs, document boundaries, layout boxes, OCR confidence, or region labels.
Page is therefore the only faithful unit in v0. It must not silently become the
final method granularity.

The next granularity audit should compare, on positive, negative, and random control
pages:

1. complete page representation;
2. fixed overlapping tiles as a model-free control;
3. OCR/layout-derived regions;
4. logical text chunks where visual localization is unnecessary.

It should measure retained positive value, removed negative interference, build
seconds, representation bytes, and number of regions. A region method is useful
only if it preserves page-level benefit at lower cost or removes page-level noise.

## Next method gate

Before designing a complex compiler, train a small cross-domain value ranker using
only build-time and historical-base signals:

- OCR/text statistics and visual-layout features;
- BM25 and ColSmol workload visibility;
- text/compact-visual rank disagreement;
- historical success or correction feedback when available;
- predicted negative exposure as a separate target.

Freeze on one domain and transfer to the other. Compare random, physical risk,
frequency-only, benefit-only, harm-filter-only, and signed-value ranking at fixed
1%, 2%, and 5% budgets. Report quality, oracle gain recovery, escape repair, false-
positive harm rate, build seconds, and bytes. If a simple signed-value ranker cannot
beat frequency and risk baselines across domains, collect a benchmark with real
PDF structure and workload traces before adding more system components.

## Artifacts

- `results/compiler-feasibility/representation-value-sparsity-atlas-v0/pharmaceuticals.json`
- `results/compiler-feasibility/representation-value-sparsity-atlas-v0/industrial.json`
- `results/compiler-feasibility/residual-materialization-oracles/`

