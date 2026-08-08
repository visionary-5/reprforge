# ReprForge

ReprForge studies the physical design of multimodal RAG indexes. Its current
question is:

> Given parsed documents, several representation operators, a build/storage
> budget, and a query workload, which knowledge units should remain cheap and
> searchable, which should be inspected visually only for the current query,
> and which should be persistently materialized as reusable visual retrieval
> state?

The project does **not** currently claim a finished paper method. Current GPU
and NVMe results establish that retrieval and query-conditioned visual states
have different quality roles, and that reusable query-independent visual
features can preserve verifier scores while removing about three quarters of
the repeated raw-page execution cost. They do not yet establish a realizable
policy that jointly predicts discovery benefit, reuse, and ranking harm on an
unseen domain.

## Current research state

Start with [the active handoff](docs/HANDOFF.md). The authoritative documents
are deliberately small:

- [Current research specification](docs/current-research-spec.md): problem,
  method hypothesis, contribution boundary, and decision gates;
- [Experiment matrix](docs/progressive-materialization-experiment-matrix.md):
  benchmarks, baselines, metrics, workload traces, and GPU stages;
- [Evidence registry](docs/evidence-registry.md): which results are active
  evidence, negative evidence, infrastructure, or historical exploration;
- [Public benchmark landscape](docs/benchmark-landscape.md): adjacent papers
  and public artifact audit.
- [Documentation map](docs/README.md): what is current and what is historical.

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
