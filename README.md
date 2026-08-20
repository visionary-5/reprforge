# ReprForge

**A physical-plan compiler for multimodal RAG indexes.**

[![Tests](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml/badge.svg)](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

## Why an index compiler?

A conventional database index has explicit source fields, update semantics,
and an exact operator contract. A multimodal RAG index is different: it is a
lossy, model-compiled view of unstructured evidence. The stored vectors mean
something only together with the document encoder, query encoder, adapter,
visual resolution, retrieval granularity, and score function. Change that
contract and the collection may need to be compiled again.

ReprForge treats that repeated indexing work as a compilation problem:

```text
logical evidence       physical plan        model lowering       artifact
page / layout / chart → boundary + workers → prefix / coalesce / suffix → index
                              │                                        │
                              └──── workload and quality contract ─────┘
```

This learned semantic view is precisely what RAG contributes beyond ordinary
lookup: a query can retrieve relevant text, layout, tables, and figures without
an exact schema or key. It also creates the systems problem. The goal is not to
make one Transformer forward pass look cheaper, but to reduce the total cost
of constructing, storing, refreshing, and using that model-dependent view.

## Current physical operator

For a ColPali-style visual late-interaction encoder, the current plan:

1. lets the Full visual state evolve to an evidence-maturation boundary;
2. reserves two stable suffix positions from every visual 2×2 cell;
3. assigns every visual hidden state to its most similar reserved anchor;
4. averages the raw states owned by each variable-size cluster;
5. continues the original frozen suffix with half as many visual workers;
6. stores only the compact retrieval endpoints and their plan manifest.

The fixed anchors preserve positional identity for the suffix. Global semantic
assignment decides which evidence each worker owns. Query encoding and MaxSim
remain unchanged. No query, qrel, answer, or task-specific training is used by
the operator. The boundary and persistent capacity are separate physical-plan
choices: a vector can remain transiently active long enough to mature without
being written to the long-lived index.

## Architecture

The repository root is the project boundary; the inner `reprforge/` directory
is the installable Python namespace. Keeping that namespace is conventional
Python packaging. The meaningful architecture is inside it:

```text
reprforge/
├── planning/    backbone admission and serializable physical CompilePlan
├── execution/   evidence assignment, hidden-state coalescing, build compiler
├── adapters/    contract for lowering a plan into a real model prefix/suffix
├── indexing/    MaxSim index plus checksummed, plan-aware persistent artifacts
└── runtime/     optional Full refinement and workload lifecycle decisions

examples/        executable reference pipeline
tests/           subsystem and end-to-end contract tests
```

These boundaries follow the index lifecycle rather than arbitrary file size:

- `planning` decides what representation should exist;
- `execution` changes which states remain active during index construction;
- `adapters` isolate model-specific attention, position, and layer APIs;
- `indexing` makes the compiled representation durable and reproducible;
- `runtime` decides when the compact view is sufficient or should defer to Full.

This follows the same separation of indexing, search, infrastructure, and model
integration used by mature late-interaction projects such as
[ColBERT](https://github.com/stanford-futuredata/ColBERT),
[ColPali](https://github.com/nomic-ai/colpali), and
[RAGatouille](https://github.com/AnswerDotAI/RAGatouille), while keeping raw
experiments and paper drafts outside the public package.

## Install

```bash
git clone https://github.com/visionary-5/reprforge.git
cd reprforge
python -m pip install -e .
```

Development checks:

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m pytest -q
python examples/quickstart.py
```

## API walkthrough

First freeze a physical plan for one backbone and collection:

```python
from reprforge import BackboneProfile, CompilerConfig, ReprForgeCompiler

profile = BackboneProfile(
    name="colpali-v1.1",
    total_layers=18,
    split_after_layer=9,
    full_visual_tokens=1024,
    compact_visual_tokens=512,
)
compiler = ReprForgeCompiler(
    CompilerConfig(profile=profile, grid_shape=(32, 32))
)

print(compiler.plan.fingerprint)
```

A model adapter implements `run_prefix` and `run_suffix`. The compiler then
owns the collection build rather than accepting unexplained endpoint arrays:

```python
index = compiler.build_documents(
    adapter,
    [(page_id, image) for page_id, image in pages],
)
candidates = index.search(query_vectors, top_k=20)
```

The compiled artifact records both vectors and the exact physical plan:

```python
from reprforge import load_index, save_index

manifest = save_index("indexes/my-collection", index, compiler.plan)
reloaded, observed_manifest = load_index("indexes/my-collection")
assert observed_manifest.plan.fingerprint == manifest.plan.fingerprint
```

For a versioned collection, a reusable boundary is worthwhile only when it
survives the expected update, meets the quality and storage contract, and
amortizes its own materialization cost. ReprForge plans that decision from
measured costs rather than assuming every intermediate state should be kept:

```python
from reprforge import (
    MaterializationOption,
    UpdateScenario,
    choose_materializations,
)

post_vision = MaterializationOption(
    name="post_vision_ir",
    depends_on=frozenset({"processor", "vision", "base_embedding"}),
    storage_bytes=6_266_593_554,
    replay_seconds=1_243.8,
    materialization_seconds=29.8,
    quality_fraction=0.999,
)
adapter_update = UpdateScenario(
    "adapter_v2",
    frozenset({"adapter", "projection"}),
    expected_count=2,
)
decision = choose_materializations(
    (post_vision,),
    (adapter_update,),
    raw_rebuild_seconds=5_000.2,
    storage_budget_bytes=6_369_873_920,
)
```

The planner can also return an empty portfolio: when the valid prefix is cheap,
the artifact is too large, or updates are too rare, rebuilding from the source
is the correct physical plan.

Optional query-time recovery is a separate runtime decision:

```python
from reprforge import refine_candidates

ranking = refine_candidates(
    index,
    query_vectors,
    candidates,
    materialize_full_page,
    top_k=5,
)
```

[`examples/quickstart.py`](examples/quickstart.py) runs the complete contract
with a synthetic adapter. A production adapter must update the model's attention
mask, position state, and suffix inputs using the returned compact positions.

## Evidence and limits

The frozen ColPali v1.1 operating point compiles after layer 9 of 18 and retains
50.29% of Full index vectors. It was selected by a preregistered split-depth
frontier: layers 3/6 were too early, layer 8 missed the cross-task quality gate,
and layers 10/11/12 did not dominate layer 9 on quality and construction cost.

Across six complete ViDoRe-v3 domains—15,194 pages and 10,782 queries—moving the
unchanged topology-global compiler from layer 6 to layer 9 improves every task.
Macro nDCG@10 rises from **0.42503 to 0.43414**: **+0.00911**, exact task
bootstrap 95% interval **[+0.00620, +0.01172]**. The equal-capacity post-hoc and
Full references are 0.43824 and 0.45253. Layer 9 therefore recovers about 69% of
the former layer-6-to-post-hoc gap while keeping the same half-size index.

The unchanged point also transfers to complete MP-DocVQA (741 pages, 591
queries): nDCG@10 rises from **0.85134 to 0.86148**, paired-query bootstrap 95%
interval for the gain **[+0.00178, +0.01872]**. Post-hoc and Full score 0.86722
and 0.87001.

Three uncontended, order-alternated MP-DocVQA measurements give **6.82% mean
document-build saving** versus Full (sample SD 0.54 percentage points; all three
runs positive), **49.71% tensor-storage saving**, and a 3.2% lower peak allocated
GPU-memory point. The build result did not pass the experiment's stricter 8%
promotion gate, so it is reported as a modest but repeatable payoff rather than
a large systems speedup.

The boundary remains explicit. Evidence covers two benchmark families but only
one backbone family. A storage-matched compact-native model dominates this
backbone on ViDoSeek and MP-DocVQA, while the compiled large model remains much
stronger across most ViDoRe domains. ReprForge is useful when the chosen large
backbone has a real capability premium; it is not a reason to use a large model
when a smaller one is already better.

## Research status

The repository contains the validated reference operator and its physical-plan
contracts. A paper-level artifact still requires:

- at least one maintained real-model adapter;
- reproducible complete benchmark entry points;
- a second backbone family and adapter;
- repeated measurements beyond one A100 host, plus query-runtime accounting;
- comparison with post-hoc pruning, local in-flight pooling, Full, and a
  compact-native smaller model.

Large datasets, checkpoints, raw result bundles, research logs, and paper
drafts are deliberately excluded from this repository.

## License

Apache-2.0. See [LICENSE](LICENSE).
