# ReprForge

ReprForge is a research prototype for **budgeted multimodal document
representation**. It compiles each document layout into one of several
retrieval representations—native text, a full visual late-interaction
representation, or a semantically pooled visual representation—and executes
the resulting heterogeneous index on GPU.

The central question is not whether document embeddings can be compressed.
It is:

> Which representation capacity should each content unit receive under
> storage, build, and serving constraints, without losing the evidence needed
> by downstream retrieval and agents?

## System

```text
MMDocIR layouts + route embeddings
            |
            v
  representation planner
  text / pool-25 / pool-9 / pool-4 / full
            |
            v
 heterogeneous index compiler
 compact route-local shards + offsets
            |
            v
 token-work GPU scheduler
 MaxSim + Top-k + quality/latency report
```

The repository currently provides:

- a reproducible MMDocIR adapter and frozen within-document evaluation;
- score replay for studying representation plans without re-encoding;
- explainable route-intervention analysis;
- a budgeted representation allocator;
- a compact physical heterogeneous index;
- a versioned compressed-base + full-visual-delta index;
- NumPy and PyTorch MaxSim runtimes;
- a token-work scheduler that turns fewer vectors into fewer GPU batches.

## Current evidence

The current results are promising, but not yet a finished paper claim.

- On a sealed 9-document, 38-query MMDocIR split, Typed-Capacity V1 uses
  **51.4%** of the vectors of uniform pool-9. It improves query-weighted
  nDCG@10 from **0.572 to 0.578**, while Recall@5 changes from **0.595 to
  0.585**. The preregistered `+0.01` nDCG@10 gate was not met, and
  document-macro results are not uniformly positive.
- At 12,496 physically materialized candidates, token-work batching reduces
  the compressed index from **196 to 23 GPU batches** and P50 latency from
  **35.10 ms to 9.03 ms**. Under the same scheduler it is **4.02× faster**
  than the full-visual index, with identical Top-10 results for all 46
  evaluated queries.
- On the public 781-layout MMDocIR bank, a pool-25 base plus full visual
  representations for visual layouts reaches **0.684 nDCG@10**, versus
  **0.624** for uniform full visual. Its compiled form is **75.7% smaller**
  and **2.46× faster**. The versioned physical index reproduces all 46 full
  rankings exactly and remains **2.24× faster** than uniform full visual.
- Query-scoped delta activation improves pilot nDCG@10 from **0.635** for
  pooled Top-10 + exact-full reranking to **0.699**, while using 86.5 MB
  rather than 428.3 MB of resident representations. It separates physical
  cache residency from per-query score replacement, avoiding cache-induced
  ranking pollution.

These findings establish a working end-to-end system and a real
quality–resource trade-off. The planner is still a heuristic; the next
research step is a workload-aware, migration-aware allocation mechanism.

## Repository

```text
reprforge/   system, planners, analysis tools, and command-line modules
tests/       CPU unit and integration tests
examples/    tiny replay fixture
results/     compact, frozen result summaries only
docs/        research contract, system design, evaluation, and roadmap
```

Large models, datasets, embedding banks, compiled indexes, machine-specific
environments, and exploratory logs are intentionally excluded.

## Quick check

Python 3.11 or newer is required.

```bash
python -m pip install -e '.[dev]'
pytest -q
bash scripts/run_replay_smoke.sh
```

GPU experiments additionally require the packages in the `gpu` extra and
locally available MMDocIR/ColPali artifacts:

```bash
python -m pip install -e '.[gpu,dev]'
python -m reprforge.run_end_to_end --help
```

The exact benchmark inputs are passed explicitly to the command-line tools;
the repository does not assume a server path or modify an existing
environment.

## Documentation

- [Research contract](docs/research-contract.md)
- [System design](docs/system.md)
- [Versioned visual delta index](docs/versioned-visual-index.md)
- [Public A100 benchmark](docs/public-benchmark-result.md)
- [Strong baseline comparison](docs/baseline-comparison-result.md)
- [Evaluation protocol](docs/evaluation.md)
- [Current results](docs/results.md)
- [Research roadmap](docs/roadmap.md)

ReprForge is currently a private research project. A public license and
release package will be selected when the artifact and paper claims are
ready.
