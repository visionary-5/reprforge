# Evaluation Protocol

## Public workload

The main benchmark is MMDocIR layout-level retrieval. Dataset and model
revisions are frozen in `results/quality/mmdocir-protocol.json`.

The current corpus contains:

- 30 public documents;
- 4,643 layouts;
- 141 evaluable queries;
- 9 domains.

Documents were assigned roles before final scores were opened:

- 10 prior-development documents;
- 11 mechanism-design documents;
- 9 final-evaluation documents.

The final split is sealed. It must not be used to tune the next planner.

## Candidate semantics

Quality uses MMDocIR's official within-document candidate pool and overlap
relevance. This avoids treating unjudged layouts from other documents as
negatives.

Candidate amplification creates distinct physical vector replicas only to
measure systems scaling. It is not used for a retrieval-quality claim.

## Baselines

The frozen route baselines are:

- uniform full visual;
- uniform pool-4;
- uniform pool-9;
- uniform pool-25;
- uniform text;
- fixed content-type routing where applicable;
- the frozen Typed-Capacity V1 rule.

Lifecycle evaluation additionally compares:

- a pool-25 base with a versioned full-visual delta;
- the identical representation plan compiled into one static index;
- a text base with a full-visual delta;
- static full visual, static pool-25, and MMDocIR's fixed hybrid.

The implemented external-mechanism baselines are:

- Visual RAG Toolkit-style pooled candidate generation followed by exact
  full-vector reranking at K=10/20/50;
- structural location followed by preencoded full-vector host transfer with
  both pageable and pinned memory.

The latter is only a systems lower bound for Deferred Visual Ingestion.
Faithful DVI comparison still requires query-time image/VLM execution.

An offline oracle may inspect all route outcomes to decompose headroom. It has
information unavailable to a deployable runtime and is never reported as a
deployable baseline.

## Metrics

Quality:

- Recall@1, @5, and @10;
- nDCG@5 and @10;
- query-weighted aggregate and document-macro deltas.

Resources:

- serialized vector bytes;
- GPU-resident bytes and padding;
- encoding/build seconds;
- number of physical candidates and vectors.

Serving:

- execution batch count;
- P50 and P95 latency;
- queries per second;
- score tolerance and Top-k equivalence.

Future workload adaptation additionally requires:

- static-plan regret per workload episode;
- fraction of layouts migrated;
- bytes and GPU-seconds rebuilt;
- adaptation delay and time to recover quality.

## Correctness

The NumPy implementation is the reference for physical-index scoring. PyTorch
results must satisfy explicit absolute and relative tolerances. Scheduler
comparisons additionally require identical Top-k rankings.

The compact artifacts retained in `results/` are summaries and correctness
reports. Raw images, models, embeddings, compiled indexes, and exploratory
logs are intentionally excluded.
