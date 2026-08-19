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

1. runs the Full document prefix to a frozen hidden-state boundary;
2. reserves two stable suffix positions from every visual 2×2 cell;
3. assigns every visual hidden state to its most similar reserved anchor;
4. averages the raw states owned by each variable-size cluster;
5. continues the original frozen suffix with half as many visual workers;
6. stores only the compact retrieval endpoints and their plan manifest.

The fixed anchors preserve positional identity for the suffix. Global semantic
assignment decides which evidence each worker owns. Query encoding and MaxSim
remain unchanged. No query, qrel, answer, or task-specific training is used by
the operator.

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
    split_after_layer=6,
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

The frozen ColPali v1.1 operating point uses layer 6 and retains 50.29% of Full
index vectors. Across six complete ViDoRe-v3 domains—15,194 pages and 10,782
queries—topology-anchored global assignment improves capacity-matched local
pooling on all six tasks: **+0.0049 macro nDCG@10**, task-bootstrap 95% CI
**[+0.0031, +0.0066]**. Measured raw-image document-build savings are
**7.48%–10.95%**.

The remaining gap is also explicit: −0.0132 macro nDCG@10 versus equal-capacity
post-hoc pooling and −0.0275 versus Full. Current evidence supports task/domain
transfer within one benchmark suite, not cross-backbone or cross-benchmark
generality.

Controls have ruled out several tempting patches at this operating point:
diverse anchors, exact cluster balance, endpoint-only low-rank correction,
tail-conditioned correction, and simple spatial assignment penalties. The next
scientific step is a model-aware lowering that preserves the physical-plan
abstraction—not another endpoint mapping or pooling sweep.

## Research status

The repository now contains a coherent project-level system skeleton and the
validated reference operator. A paper-level artifact still requires:

- at least one maintained real-model adapter;
- reproducible complete benchmark entry points;
- cross-benchmark validation of the unchanged physical plan;
- repeated build, memory, serialized-size, and query-runtime measurements;
- comparison with post-hoc pruning, local in-flight pooling, Full, and a
  compact-native smaller model.

Large datasets, checkpoints, raw result bundles, research logs, and paper
drafts are deliberately excluded from this repository.

## License

Apache-2.0. See [LICENSE](LICENSE).
