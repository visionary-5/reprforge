<h1 align="center">ReprForge</h1>
<p align="center"><b>Efficient Index Rebuilding for Visual Document Retriever Upgrades</b></p>
<p align="center">
  <a href="https://github.com/visionary-5/reprforge/actions/workflows/tests.yml"><img src="https://github.com/visionary-5/reprforge/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="Apache 2.0"></a>
</p>
<p align="center">
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/reproduction.md">Reproduction</a>
</p>

ReprForge is an index-rebuilding system for visual document retriever upgrades.
It retains the output of expensive visual computation and runs the target
retriever's remaining computation to produce its document representations.
When an upstream dependency changes, reconstruction falls back to the raw page.

```text
Source indexing:  page → processor → vision + merger → retained state
                                                           ↓
Target upgrade:   dependency validation → target continuation → target index
                          ↓ invalid
                     raw target encoding ────────────────────────┘
```

The design has three parts: **replayable visual states**, **state-specific
dependency validation**, and **target-side reconstruction**. Exact replay aims
to reproduce target document representations. Optional INT8 and PCA compression
trade storage for reconstruction fidelity.

## Results at a glance

On a 200-page gallery with 1,033 queries, one retained ColQwen2.5 v0.1 visual
state supported four independently loaded target retrievers. Every replayed
page representation was elementwise equal to native target encoding, and every
ordered top-10 result matched the target reference.

| Target retriever | Exact pages | Ordered top-10 | Replay / full encoding time |
|---|---:|---:|---:|
| ViDoRe v0.2 | 200 / 200 | 1,033 / 1,033 | 12.9% |
| Metric-AI 3B | 200 / 200 | 1,033 / 1,033 | 13.3% |
| T-Systems 3B | 200 / 200 | 1,033 / 1,033 | 13.2% |
| ColNomic 3B | 200 / 200 | 1,033 / 1,033 | 13.0% |

These A100 measurements use BF16, batch size 1 and one fixed processor
contract. Time includes warm retained-state reads and excludes initial state
retention, retrieval scoring and physical index construction. See the
[reproduction protocol](experiments/reconstruction/protocol.json) before
comparing new runs.

## Installation

The core library requires Python 3.10+ and NumPy. From this checkout:

```bash
python -m pip install -e '.[dev]'
python examples/basic_replay.py
python -m pytest -q
```

The quickstart runs a synthetic encoder on CPU to demonstrate the API.
For real ColQwen models, use the [A100 reproduction guide](docs/reproduction.md).

## Reproduce the paper

The [reproduction guide](docs/reproduction.md) provides the A100 environment,
public input preparation, and commands for independent target reconstruction.
Experiments generate their own results; model weights, datasets and outputs stay
outside Git.

| Evaluation | Entry |
|---|---|
| Adapter component inspection | [Adapter census](experiments/adapter_census/) |
| Retrieval quality and target agreement | [Quality](experiments/quality/) |
| Exact reconstruction and rebuild time | [Reconstruction](experiments/reconstruction/) |
| State compression | [Storage](experiments/storage/) |
| State design and validation | [Ablation](experiments/ablation/) |

## Code

- [`reprforge/`](reprforge/): state contracts, dependency validation and index utilities.
- [`examples/`](examples/): small CPU API examples.
- [`scripts/`](scripts/): source indexing, target replay and evaluation commands.
- [`configs/`](configs/): model configuration templates.
- [`experiments/`](experiments/): paper evaluation code and protocols.
- [`tests/`](tests/): implementation correctness tests.

## Citation and license

Software citation metadata is in [CITATION.cff](CITATION.cff).
Paper bibliographic information will be added with the manuscript release.
Code is licensed under [Apache-2.0](LICENSE). Models and datasets retain their
original licenses.
