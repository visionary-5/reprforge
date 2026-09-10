# ReprForge

**Reuse valid intermediate states when rebuilding visual document representations after a retriever upgrade.**

[![Tests](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml/badge.svg)](https://github.com/visionary-5/reprforge/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Visual document retrievers encode pages offline and store their representations
in an index. An upgrade may change only part of that computation. ReprForge
checks whether an existing intermediate state remains valid for the target
version. If its upstream dependencies are unchanged or certified equivalent
for the relevant pages, the target suffix resumes
from that state; otherwise the route falls back to raw-page encoding.

The core package provides dependency and version contracts, replay integration
interfaces, and a small reference index. Optional GPU experiment runners expose
the actual ColQwen cut and its target suffix. The cut stores post-merger visual
token representations at the language-model entrance, together with resume
metadata. It is not just the vision tower's output.

## Minimal CPU run

From a checkout of this repository:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/quickstart.py
```

Only NumPy is required by the library. The quickstart uses a synthetic encoder;
it demonstrates contracts and execution, not evidence about a real checkpoint.

## Find the implementation

[Replay correctness conditions](docs/replay-contract.md) explain which values
must be retained, how their dependencies differ, and what the library actually
checks. [Controlled mechanism evidence](experiments/core-mechanism/README.md)
tests those conditions through component changes and reconstruction ablations.

| Location | Purpose |
|---|---|
| [`reprforge/versions.py`](reprforge/versions.py), [`dependencies.py`](reprforge/dependencies.py) | Version differences and the dependencies of reusable states. Adapter tensor names alone do not certify the entire upstream contract. |
| [`reprforge/equivalence.py`](reprforge/equivalence.py) | Collection-scoped certificates for processor changes with equivalent observed outputs. |
| [`reprforge/page_reuse.py`](reprforge/page_reuse.py) | Refine processor invalidation per page using complete output evidence; other upstream blockers remain. |
| [`reprforge/adapter.py`](reprforge/adapter.py) | `encode`, `emit_cut`, `resume`, and the state contract. |
| [`experiments/independent-endpoint/run.py`](experiments/independent-endpoint/run.py) | Real ColQwen2.5 integration: `capture`, `replay`, and independent native raw encoding in separate source/target processes. |
| [`reprforge/index.py`](reprforge/index.py) | Reference MaxSim and TA@k (set overlap, not ordered equality). |
| [`reprforge/planning.py`](reprforge/planning.py), [`generation.py`](reprforge/generation.py) | Optional cost selection and reference publication machinery; these are not the paper's locality evidence. |

## Reproduce and inspect evidence

Start with [the reproduction guide](experiments/README.md),
[experiment-to-claim index](docs/evidence.md), and
[claim ledger](docs/claim-ledger.md). Frozen historical runners and protocols
are under `experiments/<date>-<experiment>/`; shared dependencies are under
`experiments/support/`. [Provenance](docs/provenance.json) maps published code and
external output files to their SHA-256 digests. Model weights, datasets, retained
states and embedding banks are external artifacts and are not committed.

The experiments address three questions:

1. **Upgrade locality:** the public-release census inspects changed stages and
   base/processor contracts. Absence of vision LoRA does not itself prove reuse.
2. **Target fidelity:** independent raw/replay checks compare document tensors;
   pooled and codec experiments report ranking agreement with explicitly stated
   reference paths. Representation equality, ordered ranking equality and
   serialized index-byte equality are different claims.
3. **Recomputation savings:** stage measurements isolate preprocessing, prefix
   and suffix; complete-build measurements have separately documented boundaries.
   Compressed routes are approximate and must retain their quality conditions.

The [new independent endpoint check](experiments/independent-endpoint/summary.json)
passed for vidore v0.2, Metric-AI 3B, T-Systems 3B and ColNomic 3B: each has
200/200 equal document representations (95,869,056 elements, zero maximum error)
and 1,033/1,033 equal ordered top-10 lists on the 200-page gallery, under a
pinned common base and processor. This is a bounded correctness result.

The [page-validity extension](experiments/page-validity/README.md) examines an
isolated processor-budget change: 15/200 pages retain identical processed
inputs. A stratified 16-page independent endpoint verifies selective replay
and fallback. This is a method/boundary result, not an additional speed claim.

## Evidence boundaries

- The 19 inspected retrieval adapter releases contained no vision-tower or
  merger LoRA tensors. This is a scoped census, not a statement about all
  adapters. A Turkish ColPali adapter in the dependency audit changes vision.
- The clean official ColQwen2 v0.1 → v1.0 transition retains the recorded base
  and shipped processor contract. ColQwen2.5 v0.2 changes the pixel budget;
  pooled comparisons pin a common processor and do not test vendor defaults.
- Historical mixed-pool `target_full` rows run the target suffix on a shared
  prefix. Their BF16 rows are consistency checks, not independent raw-target
  endpoint tests. Historical Energy checks independently reload the target.
- A100 batch-1 suffix costs are about 9.3% of raw page encoding at 12,845,056
  max pixels and 20% at 602,112. These stage ratios exclude state IO, index
  construction, validation and serialization; they are not full rebuild times.
- MMDocIR's 73.4% complete-transition saving uses approximate PCA-256 replay and
  a frozen raw baseline from a prior run. RTX 4090 build savings also use
  approximate routes. Neither is an exact-replay headline.
- No experiment establishes physical serialized-index byte equality.

The library is a research reference, with an in-memory index and single-host
experiments. See [environment and timing conditions](docs/environment.md).

## Citation and license

[CITATION.cff](CITATION.cff) cites the software. Paper title and author metadata
will be added when finalized. Code is licensed under [Apache-2.0](LICENSE);
external models and datasets retain their own licenses.
