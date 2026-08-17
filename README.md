# ReprForge

**Build smaller multimodal late-interaction indexes without treating every
visual token as permanent.**

[![Tests](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml/badge.svg)](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Multimodal retrievers such as ColPali build a page index from hundreds or
thousands of visual vectors. Those vectors are expensive twice: they all pass
through the document encoder, then remain in the index for every future query.

ReprForge treats index construction as a representation-lifecycle problem:

```text
page ── Full prefix ── coalesce ── compact suffix ── persistent compact index
                         │
small Full canary ───────┴── query-free trajectory alignment

query ── compact search ── Top-K pages ── on-demand Full materialization ── rerank
```

The compact index is the persistent locator. Full page representations are
temporary and are built only for a small candidate set when the workload needs
Full-level quality.

## Install

```bash
git clone https://github.com/visionary-5/reprforge.git
cd reprforge
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Quick start

ReprForge is model-agnostic at the endpoint boundary. A model integration
provides compact suffix endpoints and a small set of spatially matched Full
endpoints; the public API fits the query-free correction, builds the compact
index, and performs candidate refinement.

```python
from reprforge import BackboneProfile, CompilerConfig, ReprForgeCompiler

compiler = ReprForgeCompiler(
    CompilerConfig(
        profile=BackboneProfile(
            name="my-multi-vector-vlm",
            total_layers=18,
            split_after_layer=12,
            full_visual_tokens=1024,
            compact_visual_tokens=512,
        ),
        rank=8,
        fit_steps=25,
    )
)

# Lists of [slots, dimension] arrays. No queries, qrels, or answers are used.
compiler.fit(compact_canary, full_canary_in_compact_slots)

index = compiler.build(zip(page_ids, compact_page_endpoints))
candidates = index.search(query_vectors, top_k=20)
ranking = index.refine(query_vectors, candidates, materialize_full_page, top_k=5)
```

An executable synthetic example is available at
[`examples/quickstart.py`](examples/quickstart.py).

## Method

ReprForge separates three decisions that are usually entangled:

1. **Where evidence becomes stable.** A model-specific boundary exposes a full
   visual state before the retrieval geometry has finished evolving.
2. **What remains active.** Topology-preserving coalescing creates fewer worker
   slots for the remaining encoder layers. A small query-free, rank-8 residual
   aligns their final directions using 3.125% Full canary pages.
3. **What remains persistent.** The compact slots form the long-lived index.
   A measured workload policy chooses compact-only, compact-plus-refinement, or
   eager Full construction.

The query encoder and MaxSim scoring rule are unchanged.

## Current evidence

All comparisons use the same retriever checkpoint and candidate collection
within each benchmark. Large embeddings, indexes, models, and raw datasets are
not stored in this repository.

| Evaluation | Scale | Persistent vectors | Quality | First-build saving |
|---|---:|---:|---:|---:|
| ViDoRe-v3, four complete tasks | 4 domains | 50.29% of Full | 98.42%–102.88% of Full nDCG@10 | 17.04%–19.87% |
| MMLongBench-Doc page retrieval | 5,777 pages / 810 held-out queries | 50.29% | 0.5325 vs Full 0.5259 nDCG@10 | 17.65% |
| ViDoSeek global retrieval | 5,385 pages / 1,142 queries | 50.32% serialized bytes | 0.9221 vs Full 0.9212 Recall@5 after Top-20 refinement | 17.87% |

On ViDoSeek, compact-only retrieval has 23 Recall@5 regressions relative to
Full. Top-20 refinement recovers all 23, introduces no regression, and improves
one query. The real online path reads only the compact index and raw pages; its
current median latency is 1.97 seconds per query, so refinement is useful for
cold or storage-constrained indexes rather than every workload.

A 500M-parameter ColSmol control is quality-competitive on the same task, but
its measured index is 1.70× larger and its build is 2.45× slower. Model size and
multi-vector index lifecycle cost are different axes.

## Scope

ReprForge is an index-construction method, not a general Transformer token
compression library. The current positive build-time result is strongest for
ColPali-style encoders. ColModernVBERT currently supports a storage-only plan,
and ColQwen2 is an observed abstention case at the useful physical boundary.

The repository is a compact research preview. Model hooks, concurrent
candidate materialization, and additional complete benchmark integrations are
still being stabilized.

## Repository

```text
src/reprforge/   compiler, alignment, index, and lifecycle policy
examples/        small executable examples
tests/           public API and invariant tests
```

The design builds on late-interaction retrieval and visual document retrieval,
especially [ColBERT](https://github.com/stanford-futuredata/ColBERT) and
[ColPali](https://github.com/illuin-tech/colpali).

## License

Apache-2.0. See [LICENSE](LICENSE).
