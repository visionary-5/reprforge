# ReprForge

**Semantic recompilation cuts for evolving visual late-interaction indexes.**

[![Tests](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml/badge.svg)](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

A visual document retriever (ColPali, ColQwen2/2.5, ColSmol, ...) is upgraded far
more often than the collection it indexes. Every upgrade forces a choice between
serving a stale index and re-encoding every page through a multi-billion
parameter vision-language model. ReprForge is the planner and contract layer
for a third option: keep the deepest intermediate state that the new version has
not changed, replay only the target suffix from it, admit the result by measured
retrieval quality, and publish it as an immutable generation.

The library is small and CPU-only. Model integrations live outside it and plug in
through one protocol.

## Why this works

Public ColPali-family releases change the decoder LoRA and the retrieval head;
none of the 19 adapter checkpoints we parsed touches the vision tower or the
vision-language merger, and three vendors ship a byte-identical visual prefix.
An upgrade therefore lives in the suffix. Replaying the target suffix from an
exact post-vision cut reproduces the target index bit for bit (Target Agreement
1.000 on three benchmarks); compressed cuts trade retained bytes for ranking
fidelity along a measured frontier. The share of raw encode time spent before
the cut (0.85–0.88 on document pages with ColQwen2.5) equals the saving, and
predicts which backbones benefit: ColPali's small vision tower is 23% of its
forward pass, so its cut never pays.

Query-side compatibility bridges (Procrustes, affine, Drift-Adapter MLP) keep
the *old* ranking searchable; fitted cross-domain or in-domain they reach
0.60–0.83 Target Agreement. Constructing the target index is a different
contract, and this library is about that contract.

## Install

```bash
pip install -e '.[dev]'
python -m pytest -q
python examples/quickstart.py
```

Only `numpy` is required at runtime.

## What the library does

```text
version tuple ──diff──▶ changed components ──▶ legal cuts ──▶ planner ──▶ route
 (h_C,h_P,h_E,h_V,       (tensor census,          (depends_on ∩      (storage,      raw | cut
  h_A,h_R,h_I)            output certificates)     changed = ∅)       quality,
                                                                     leverage,
                                                                     break-even)
                                      route ──▶ resume target suffix ──▶ validate ──▶ seal ──▶ publish
```

| Module | Concern |
|---|---|
| `reprforge.versions` | The version tuple (collection, processor, vision, base embedding, adapter, projection, index policy) and its exact dependency delta. |
| `reprforge.dependencies` | Classify an adapter checkpoint's tensor names by encoder stage. Unknown names fail closed. |
| `reprforge.equivalence` | Collection-scoped output certificates that discharge processor-file changes which do not change outputs. |
| `reprforge.planning` | Select the cut portfolio that minimises expected upgrade time under storage and quality contracts; `cut_leverage` and `break_even_upgrades` as pre-codec predictors. |
| `reprforge.adapter` | `DocumentEncoderAdapter`: `encode` (raw route), `emit_cut`, `resume`. `CutState` carries the stored tensor, its compiled dependencies and the resume contract. |
| `reprforge.index` | Reference MaxSim index, `target_agreement`, checksummed storage that records the rebuild source. |
| `reprforge.generation` | Immutable, hashed generations and atomic active-pointer publication. |

## Minimal walkthrough

```python
from reprforge import (VersionManifest, inspect_adapter_tensor_keys,
                       MaterializationOption, choose_materializations)

# 1. What changed? Read the target checkpoint's tensor names (safetensors header).
scope = inspect_adapter_tensor_keys(tensor_names)        # decoder + head only
assert scope.post_vision_cut_legal

# 2. Which source? Costs come from a small calibration run on this collection.
decision = choose_materializations(
    (MaterializationOption("post_vision",
                           depends_on=frozenset({"processor", "vision", "base_embedding"}),
                           storage_bytes=6_267_000_000, replay_seconds=1_244,
                           quality_fraction=0.997),),
    (scope.to_update_scenario("colqwen2.5-v0.2"),),
    raw_rebuild_seconds=5_000, storage_budget_bytes=7_000_000_000,
)
decision.routes[0].source    # "post_vision"  -> replay; "raw" -> rebuild from pages

# 3. Execute through your model integration, then seal and publish.
vectors = adapter.resume(stored_cut)                     # target suffix from the cut
```

`examples/quickstart.py` runs this end to end with a synthetic encoder and
checks the exact cut against the raw target index with Target Agreement;
`examples/versioned_update.py` shows the planning and publication path alone.

## Reproducing the paper

The GPU experiments (public-release census, codec frontier, build-time anatomy,
cross-backbone leverage, MMDocIR generation transition) are model-specific and
live in the research repository released with the paper, each as a frozen
protocol, runner, raw output and analysis report. This package contains the
model-agnostic decision and publication logic those experiments exercise, plus
the numbers they measured as test fixtures (`tests/test_planning.py`).

Headline measurements (one A100 unless stated):

| Claim | Evidence |
|---|---|
| 19 / 19 public adapters change only decoder + head | safetensors header census, 2026-09-04 |
| Exact post-vision cut reproduces target index | TA@10 = 1.000 on ArxivQA, DocVQA, Flickr |
| Compressed cuts: fidelity vs bytes | INT8 .96–.98 at 4× terminal; PCA-256/INT8 .85–.90 at 0.5× |
| Cut leverage equals replay saving | .845 / .876 leverage → 84.6% / 87.5% saving (ArxivQA / DocVQA) |
| Small images depend on suffix batching | Flickr 54% at batch 1, 77% at batch 4 (A100); 78% on RTX 4090 |
| Complete 20,395-page transition | 1,351 s from cut vs 5,078 s raw incl. validation, sealing, publication (73%) |

## Scope and limits

- Alpha research package, not a serving system. The reference index is in-memory;
  production ANN engines keep the manifest and publication contract.
- Positive lifecycle evidence covers ColQwen2.5 under a pinned processor
  contract; the shipped v0.2 release also lowers `max_pixels`, which under the
  version tuple is a processor change requiring a raw rebuild.
- Timings are single-host (A100, RTX 4090) with warm local storage.

## Citation

See `CITATION.cff`.

## License

Apache-2.0.
