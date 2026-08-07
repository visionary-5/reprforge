# ReprForge

ReprForge studies the physical design of multimodal RAG indexes. Its current
question is:

> Given parsed documents, several representation operators, a build/storage
> budget, and a query workload, which knowledge units should remain cheap and
> searchable, which should be inspected visually only for the current query,
> and which should be persistently materialized as reusable visual retrieval
> state?

The project does **not** currently claim a finished paper method. Existing GPU
results establish two useful facts: text and visual locators have complementary
candidate failures, and a future-aware oracle can concentrate much of the
visual retrieval gain on a small page set. They do not yet establish that an
online, realizable selector can predict that set or that logical savings become
end-to-end latency savings.

## Current research state

The authoritative documents are deliberately small:

- [Current research specification](docs/current-research-spec.md): problem,
  method hypothesis, contribution boundary, and decision gates;
- [Experiment matrix](docs/progressive-materialization-experiment-matrix.md):
  benchmarks, baselines, metrics, workload traces, and GPU stages;
- [Evidence registry](docs/evidence-registry.md): which results are active
  evidence, negative evidence, infrastructure, or historical exploration;
- [Public benchmark landscape](docs/benchmark-landscape.md): adjacent papers
  and public artifact audit.

Older documents and results remain in Git for auditability. Unless an older
file is explicitly promoted in the evidence registry, it is not a current
claim or instruction.

## Repository map

```text
reprforge/    tested analysis and runtime modules
tools/        reproducible data preparation and experiment entry points
experiments/  hardware-specific wrappers and pinned compatibility patches
configs/      frozen or explicitly draft experiment contracts
tests/        CPU correctness and protocol tests
results/      compact summaries; no raw datasets, models, or large indexes
docs/         current specification plus historical research record
```

Large models, public datasets, full embedding banks, compiled indexes, and
machine-specific environments are intentionally stored outside Git. Every
paper-facing result must retain a compact manifest with input hashes, model
revision, command/config identity, hardware, timing scope, and output hashes.

## Quick check

Python 3.11 or newer is required.

```bash
python -m pip install -e '.[dev]'
pytest -q
bash scripts/run_replay_smoke.sh
```

GPU experiments additionally require the `gpu` extra and explicit local paths
for datasets, models, and outputs:

```bash
python -m pip install -e '.[gpu,dev]'
python -m reprforge.run_end_to_end --help
```

The repository never assumes a shared-server path and its experiment wrappers
refuse to share an occupied GPU or overwrite an existing output directory.

ReprForge is currently private. A release license and public artifact package
will be selected only after the paper claims and reproducibility boundary are
settled.
