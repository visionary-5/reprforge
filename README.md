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
- an official ViDoRe v3 complex-pipeline adapter for full-corpus transfer;
- an online BM25-to-visual cohort compiler with atomic resident generations;
- NumPy and PyTorch MaxSim runtimes;
- a token-work scheduler that turns fewer vectors into fewer GPU batches.
- a database-style candidate representation catalog with budgeted
  probe--verify--materialize transitions and admission-aware execution.

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
- On the official ViDoRe v3 HR English split, full visual improves nDCG@10
  from **0.495** to **0.518** over Markdown at **3.06x** corrected build time.
  Removing a redundant PIL--PNG--PIL adapter round-trip reduced visual build
  from 276.7 to 100.7 seconds without changing any score. Tiered K=20 still
  visualizes 84% of pages, is 22.5% slower end-to-end than full visual, and
  reaches only **0.501**. An oracle-only 13.5% visual witness reaches
  **0.544**, establishing allocation headroom but not a deployable policy.
- Candidate-relative fusion now runs through a real online compiler on two
  official ViDoRe v3 datasets. BM25 Top-20 plus cohort-normalized visual
  evidence reaches **0.537** nDCG@10 on HR and **0.563** on Finance-EN. With
  resident visual state, complete cold streams finish in **98.03 s** and
  **190.50 s**, versus **108.74 s** and **320.83 s** for full visual prebuild.
  The system constructs 80.6% and 63.1% of pages. HR ablation shows that
  persistence supplies most of the gain: batch-8 adds only 1.07x over batch-1
  resident and misses its 1.10x mechanism gate. Synchronous batch P95 remains
  9.0--9.1 seconds, so low-latency asynchronous construction is not claimed.
- On the independent 3,230-page IRPAPERS benchmark, resident K=10 reaches
  **48.9/82.2/92.8 Recall@1/5/20** while visually constructing 511 pages.
  It completes in **118.1 s**, versus 695.9 s for the controlled full-visual
  ColPali-v1.1 path, and uses 6.16x fewer combined index bytes. A qrel-only
  minimum-action oracle reaches 52.8% Recall@1 with only 110 visual page-events,
  identifying per-query action estimation as the next algorithmic gap.
- A first paper-disjoint progressive-acquisition probe replaces fixed K with
  observed ranking intervention. It reduces visual candidate events from
  **1,800 to 992** and unique candidate pages from **511 to 309**, while
  matching K=10's **48.9% Recall@1**. It does not preserve Recall@5 (78.3%
  versus 82.2%), so it is a mechanism signal rather than the final algorithm.
- Boundary-weighted admission then combines Top-5 transition risk with page
  reuse. Across five source-paper-disjoint workload folds it retains **57.2%**
  of eligible visual pages and reaches **83.3% Recall@5**, versus 69.7% and
  81.7% for frequency-only admission under the same train-only risk target.
  This passes its offline mechanism gate.
- Pairwise admission exposes a complementary-view effect at the Top-5
  boundary. At the preregistered 20% page budget it improves exact teacher
  agreement from **50.6% to 56.1%** over independent boundary weighting with
  identical Recall@5. Train-only calibration builds **125 rather than 135
  pages**, reaches **82.2% rather than 81.7% Recall@5**, and removes 21.2% of
  score pairs. Two eager, interleaved A100 repetitions reach only **1.121x**
  and **1.032x**, missing the frozen 1.15x speed gate. The mechanism is kept;
  the current performance claim is rejected.
- A sparse cost--risk controller now replaces full-history exact matching with
  source-paper-cross-fitted what-if evidence, an interpretable rank/margin risk
  table and a measured A100 page-cost model. At 20% historical query probes it
  selects 24.4% fewer pages than sparse independent admission, predicts 1.32x
  less online visual work and raises held-out Recall@5 from 79.44% to 80.56%.
  The probes themselves require an estimated 29.6 workload episodes to
  amortize, so the cold-start system gate fails and no new speed claim is made.
- Reusable pair probes remove that separate cold-start bill: every probed page
  is retained as a final index view. On 40 interleaved A100 runs, active pair
  15% does not reproduce its offline quality gain and misses the speed gate in
  one repetition. The online update is rejected. Atomic static
  complementary-pair admission is the constructive result: it builds **25.9%
  fewer pages**, improves Recall@5 from **80.56% to 81.67%**, and is
  **1.228x/1.240x faster** than independent-risk 20% in the two repetitions.
- V3 formalizes complementary-view admission as weighted densest
  `k`-subgraph, adds an exact oracle and a sparse published Frank--Wolfe
  baseline, and exposes an objective mismatch: additive edge coverage has
  only **0.213** Spearman correlation with held-out Recall@5. A sparse
  query-saturated greedy planner reaches **82.78%** Recall@5 at the 20% budget
  versus **81.67%** for conditional admission, with no extra pages and no loss
  in any fold. An expensive local-search diagnostic reaches 83.33%. The
  deployable gain is only two of 180 queries, so graded ViDoRe transfer is
  required before it becomes an algorithm claim.
- The candidate-view control plane scales to 30,594 hypothetical views at a
  ViDoRe-v3-like workload size in 1.17 seconds of candidate generation and
  0.10 seconds of materialization planning, using 80 MB peak Python memory.
  This validates systems feasibility only; its utility values are synthetic.

These findings establish a working end-to-end system and a real
quality–resource trade-off. The independent per-page utility abstraction has
been rejected: rank interactions make representation value cohort-dependent.
The current mechanism compiles incumbent--challenger comparisons into a
weighted boundary graph and admits complete, reusable comparisons under a
physical-build budget. V3 shows that stronger optimization of raw edge mass
does not reliably improve retrieval; query-level saturation is the frozen
transfer candidate. The next research step is graded HR and Finance-EN
transfer without threshold repair. Online pair-delta adaptation is no longer
on the main path. Persistence remains a separate action, and lifecycle
adaptation remains conditional on a real temporal workload.

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
- [Official ViDoRe v3 integration](docs/vidore-integration.md)
- [ViDoRe v3 HR transfer result](docs/vidore-v3-hr-result.md)
- [Progressive visual-index contract](docs/progressive-visual-index-contract.md)
- [Progressive visual oracle result](docs/progressive-visual-oracle-result.md)
- [Candidate-relative fusion result](docs/candidate-relative-fusion-result.md)
- [Online cohort compiler result](docs/cohort-compiler-result.md)
- [Public benchmark and baseline landscape](docs/benchmark-landscape.md)
- [Benchmark transfer contract](docs/benchmark-transfer-contract.md)
- [IRPAPERS transfer result](docs/irpapers-transfer-result.md)
- [Progressive evidence mechanism probe](docs/progressive-evidence-probe.md)
- [Boundary-weighted admission gate](docs/boundary-admission-gate.md)
- [Candidate representation view contract](docs/representation-view-contract.md)
- [Candidate view scale result](docs/representation-view-scale-result.md)
- [Pairwise what-if contract](docs/pairwise-what-if-contract.md)
- [Pairwise representation admission result](docs/pairwise-view-admission-result.md)
- [Sparse cost--risk admission contract](docs/sparse-risk-admission-contract.md)
- [Sparse cost--risk admission result](docs/sparse-risk-admission-result.md)
- [Reusable pair-probe contract](docs/reusable-pair-probe-contract.md)
- [Reusable pair-probe A100 result](docs/reusable-pair-probe-result.md)
- [Complementary-view V3 contract](docs/complementary-view-v3-contract.md)
- [Complementary-view V3 result](docs/complementary-view-v3-result.md)
- [Evaluation protocol](docs/evaluation.md)
- [Current results](docs/results.md)
- [Research roadmap](docs/roadmap.md)

ReprForge is currently a private research project. A public license and
release package will be selected when the artifact and paper claims are
ready.
