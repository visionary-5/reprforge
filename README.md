<h1 align="center">ReprForge</h1>
<p align="center"><b>Efficient Index Rebuilding for Visual Document Retriever Upgrades</b></p>
<p align="center">
  <a href="https://github.com/visionary-5/reprforge/actions/workflows/tests.yml"><img src="https://github.com/visionary-5/reprforge/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="Apache 2.0"></a>
</p>
<p align="center">
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/reproduction.md">Reproduction</a> ·
  <a href="docs/paper-map.md">Experiments</a>
</p>

ReprForge rebuilds visual document indexes by reusing intermediate computation
that remains valid across retriever upgrades. It identifies the visual state
needed by later computation, validates its dependencies and page inputs, and
lets the target retriever complete the remaining computation with its own
parameters.

```text
Initial build    page → visual computation → retained state → source representation
Upgrade          validate retained state → target continuation → target representation
```

The implementation includes PyTorch execution tracing, state capture and replay,
a ColQwen2.5 integration, and experiment runners. See the
[architecture](docs/architecture.md) and [implementation notes](docs/implementation.md)
for the replay contract and supported execution paths.

## Paper results

On **20,946 pages**, four released retriever upgrades achieve **4.35–4.48× rebuild
speedups** and reduce total initial-build-and-upgrade time by **61%**, including
initial state retention. These measurements use a common processing
configuration and a fixed input contract; all rebuilt representations are
bitwise identical to full target encoding. Timing covers representation
generation and publication, excluding physical ANN index construction.

A separate 200-page experiment measures generic input validation, which reruns
the target processor: **1.878× speedup**, or **1.818×** including the one-time
dependency check. Protocols and implementation paths are documented in the
[reproduction guide](docs/reproduction.md) and [paper map](docs/paper-map.md).

## Quick start

From a checkout, install the library and run the CPU example:

```bash
git clone https://github.com/visionary-5/reprforge.git
cd reprforge
python -m pip install -e '.[dev]'
python examples/basic_replay.py
python -m pytest -q
```

The core library requires Python 3.10+ and NumPy. The example uses synthetic
tensors to demonstrate capture, validation, replay and publication. Install
PyTorch to include the tracing and replay tests:

```bash
python -m pip install torch==2.5.1
python -m pytest tests/test_traced_replay.py -q
```

For real models, follow the [GPU reproduction guide](docs/reproduction.md).
The [command-line guide](scripts/README.md) covers source indexing, target
replay and comparison with independently encoded target representations.

## Experiments

| Evaluation | Entry |
|---|---|
| Released-pair validity and interface discovery | [Audit and tracing](experiments/paper/) |
| Exact target reconstruction | [Reconstruction](experiments/reconstruction/) |
| Full lifecycle rebuild cost | [Lifecycle](experiments/paper/e2e/) |
| Manual interfaces, retrieval quality and validation cost | [Interface and validation](experiments/final_pass/) |
| Controlled state and dependency changes | [Ablation](experiments/ablation/) |

The [experiment index](experiments/README.md) also lists auxiliary workflows.
The [paper map](docs/paper-map.md) identifies coverage, required inputs and
historical workflows that are not fully distributed. Each experiment compares
routes under its own input contract and timing scope.

## Repository

- [`reprforge/`](reprforge/): tracing, replay, model integration and index utilities.
- [`scripts/`](scripts/), [`examples/`](examples/): command-line tools and CPU examples.
- [`experiments/`](experiments/), [`configs/`](configs/): evaluation runners and input specifications.
- [`tests/`](tests/): implementation checks.
- [`docs/`](docs/): architecture, protocols and reproduction instructions.

Keep datasets, full model checkpoints, retained states and generated results
outside the checkout. Four small projection inputs needed for input preparation
are bundled with their [source and license information](THIRD_PARTY_NOTICES.md).

## Citation and license

Software citation metadata is available in [CITATION.cff](CITATION.cff).
Paper citation details will be added when the manuscript is publicly available.
Code is licensed under [Apache-2.0](LICENSE); third-party models and datasets
retain their own terms.
