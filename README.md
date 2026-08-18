# ReprForge

**Compile multimodal document encoders into smaller long-lived indexes.**

[![Tests](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml/badge.svg)](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Late-interaction retrievers build hundreds or thousands of vectors for every
page. Those vectors are costly during document encoding and remain costly for
the lifetime of the index. ReprForge asks a systems question: **must every
intermediate visual representation remain active through the entire index
construction pipeline?**

ReprForge shortens that lifecycle inside the document encoder:

```text
page → full prefix → topology-anchored coalescing → compact suffix → index
                         1024 visual states              512 states

query → unchanged query encoder → unchanged MaxSim search
```

This is index construction, not generic Transformer acceleration. The query
path and retrieval score do not change.

## Method

At a model boundary where visual tokens still form a grid, ReprForge:

1. reserves two fixed suffix positions from every spatial 2×2 cell;
2. assigns every visual hidden state to its most similar reserved anchor;
3. averages raw hidden states within each cluster;
4. continues all remaining frozen encoder layers at the compact positions;
5. persists only the resulting compact retrieval endpoints.

The fixed anchors preserve the execution layout expected by the suffix. Global
assignment lets each slot gather semantically similar evidence beyond its local
2×2 cell. The implementation is query-free and training-free.

`TopologyAnchoredPlan` exposes the assignments, anchors, cluster sizes, and
compact positions needed by a PyTorch/JAX model hook. Empty clusters fall back
to their fixed anchor state, so output capacity and suffix positions never
change.

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

## Quick start

```python
import numpy as np

from reprforge import BackboneProfile, CompilerConfig, ReprForgeCompiler

compiler = ReprForgeCompiler(
    CompilerConfig(
        profile=BackboneProfile(
            name="colpali-style-encoder",
            total_layers=18,
            split_after_layer=6,
            full_visual_tokens=1024,
            compact_visual_tokens=512,
        )
    )
)

# Hidden states at the selected boundary: visual states first, auxiliaries last.
hidden = np.random.default_rng(0).normal(size=(1030, 2048))
state = compiler.compile_hidden_state(
    hidden,
    grid_shape=(32, 32),
    auxiliary_tokens=6,
)

# A model adapter continues the original suffix using both values below.
compact_hidden = state.hidden_states
compact_positions = state.plan.compact_positions(auxiliary_tokens=6)

# Persist normalized retrieval endpoints produced by that compact suffix.
index = compiler.build(compact_page_endpoints)
candidates = index.search(query_vectors, top_k=20)
ranking = index.refine(
    query_vectors,
    candidates,
    materialize_full_page,
    top_k=5,
)
```

[`examples/quickstart.py`](examples/quickstart.py) is executable and uses small
synthetic arrays. Framework-specific hooks remain model-owned because attention
masks, rotary positions, and layer APIs differ across backbone families.

## Evidence

The current frozen operating point uses a ColPali v1.1 document encoder,
coalesces after layer 6, and retains 50.29% of persistent tokens.

Across six complete ViDoRe-v3 domains (15,194 pages and 10,782 queries), the
topology-anchored global operator improved over capacity-matched local pooling
on all six tasks: **+0.0049 macro nDCG@10**, task-bootstrap 95% CI
**[+0.0031, +0.0066]**. Measured document-build savings were **7.48%–10.95%**.

The remaining quality gap is explicit: global compilation is −0.0132 macro
nDCG@10 versus same-capacity post-hoc pooling and −0.0275 versus Full. On an
opened 9-document MMDocIR mechanism set, it reached 0.6076 nDCG@10 versus
0.5787 for local in-flight pooling and 0.6084 for post-hoc pooling, with 22.1%
measured build saving. MMDocIR is algorithm-development evidence, not a held-out
generalization claim.

Recent controls narrowed the method rather than adding unsupported machinery:

- diverse anchors and exact balance both reduced retrieval quality;
- assignment-aligned rank-8 endpoint correction recovered only +0.0014 macro
  nDCG@10 across two domains;
- geometry-only tail prediction reached 0.647 held-out-document AUROC, below
  its frozen 0.70 gate;
- tail-conditioned correction reduced held-out p95 endpoint error by only
  2.32%, and spatial assignment penalties did not improve aggregate nDCG.

The public default is therefore the simplest operator supported by the current
evidence: fixed topology anchors, global semantic assignment, no learned
correction.

## Scope

ReprForge currently demonstrates task/domain generalization within one
benchmark family. It does **not** yet establish cross-backbone or
cross-benchmark generalization. ColPali-style encoders satisfy the execution
contract; other multi-vector VLMs require a verified hidden boundary, visual
topology, and compact suffix path.

The repository contains the reusable algorithm, reference index, lifecycle
policy, tests, and one example. Datasets, checkpoints, raw result bundles,
paper drafts, and research logs stay outside the public tree.

```text
reprforge/   coalescing algorithm, compiler, index, and lifecycle policy
examples/    executable synthetic integration
tests/       API and invariant tests
```

ReprForge builds on late-interaction retrieval and visual document retrieval,
especially [ColBERT](https://github.com/stanford-futuredata/ColBERT) and
[ColPali](https://github.com/illuin-tech/colpali).

## License

Apache-2.0. See [LICENSE](LICENSE).
