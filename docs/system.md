# System Design

## Representation routes

Each MMDocIR layout can use one of five routes:

- `text`: native layout text encoded into a late-interaction representation;
- `image-pool-25`: aggressively pooled visual vectors;
- `image-pool-9`: medium-capacity pooled visual vectors;
- `image-pool-4`: high-capacity pooled visual vectors;
- `image`: the full visual representation.

Pooling uses ColPali semantic hierarchical token pooling. The routes share a
backbone so the experiment isolates representation choice rather than a
collection of unrelated models.

## Pipeline

```text
layout metadata + rendered crops + queries
                    |
                    v
          mmdocir_route_runner
        encode each route once
                    |
                    v
             replay score bank
      planner / intervention analysis
                    |
                    v
        heterogeneous index compiler
    route-local contiguous vector shards
                    |
                    v
      NumPy reference or PyTorch runtime
       MaxSim -> Top-k -> evaluation
```

The score bank decouples expensive encoding from planner iteration. It stores
scores and compact route costs, not model weights or the generated physical
index.

## Physical index

`reprforge.heterogeneous_index` compiles a plan into:

- a manifest describing the model, dtype, dimensions, and route shards;
- compact contiguous vectors for each selected route;
- document offsets and stable external IDs;
- query vectors in a separate bank.

Storage accounting uses serialized vector bytes. GPU-resident accounting also
reports padding and temporary memory. Route boundaries are an index-layout
choice, not a semantic change to MaxSim.

## Runtime

The reference runtime scores one query against every candidate:

\[
\operatorname{MaxSim}(q,d)
= \sum_{q_j \in q}\max_{d_k \in d} q_j^\top d_k.
\]

Two schedulers are available:

- fixed-document batching groups a fixed number of candidates;
- token-work batching grows a batch while
  `batch_documents × max_document_tokens` remains below a work budget.

The second policy matters because padded matrix work is set by the longest
document in a batch. It sorts candidates by vector length and groups similar
lengths, without changing scores or rankings.

## Main modules

| Module | Responsibility |
| --- | --- |
| `mmdocir_data` | Normalize public MMDocIR layouts and queries |
| `mmdocir_route_runner` | Encode representation routes and route diagnostics |
| `policy_replay` | Evaluate plans over a frozen score bank |
| `route_mechanism_analysis` | Exact one-layout evidence/risk interventions |
| `representation_allocator` | Budgeted route allocation |
| `heterogeneous_index` | Compile, load, amplify, and execute physical indexes |
| `run_end_to_end` | Compile, verify, evaluate, and benchmark in one run |

All command-line tools expose `--help` through `python -m`, for example:

```bash
python -m reprforge.policy_replay --help
python -m reprforge.heterogeneous_index --help
python -m reprforge.run_end_to_end --help
```

Server paths, GPU IDs, model caches, and Conda environments are deliberately
outside the repository contract.
