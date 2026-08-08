# ReprForge handoff

Status: active, 2026-08-08  
Branch: `exp/value-aware-materialization-compiler`

This is the first file a new agent should read. It separates the active paper
question from the large historical research record.

## Current paper question

ReprForge studies multimodal RAG index construction between document arrival
and query execution:

> Under build, storage, and query-time budgets, which knowledge units should
> remain as raw pages, which should acquire reusable query-independent visual
> features, and which need a corpus-searchable visual retrieval
> representation?

The primary contribution is not Top-k routing, caching by itself, or a new
compression operator. The intended method is a representation-state compiler
that jointly considers:

1. **discovery value**: whether visual retrieval prevents candidate escape;
2. **reuse value**: whether persistent visual features amortize repeated raw
   page processing;
3. **signed quality value**: whether a visual representation repairs ranking
   or introduces distractors;
4. **physical cost**: build time, serialized bytes, reads, H2D, GPU execution,
   invalidation, and query latency.

## Current representation states

```text
raw page / deferred inspection
        |
        | repeated query-conditioned use
        v
query-independent visual feature
        |
        | corpus-wide discovery value
        v
visual retrieval representation
```

Feature and retrieval materialization are separate actions. A cached visual
feature preserves the query-conditioned verifier but does not make an escaped
page discoverable. A retrieval embedding can improve discovery but may change
ranking quality. Future algorithms must not collapse these two effects.

## Established evidence

The compact authoritative result is
`results/compiler-feasibility/closure-materialization-v0-2026-08-08.json`.
Raw artifacts are under the workspace-level
`reprforge-experiment-artifacts/value-aware-2026-08-08/` directory.

- On identical H-Pool Top-20 candidates, a Qwen2.5-VL raw-page verifier beats
  persistent Full Omni reranking on Pharma and Industrial. Persistent and
  transient visual representations are therefore not quality-equivalent.
- Reusing query-independent Qwen2.5-VL visual features preserves the exact
  YES-minus-NO score over 1024 query-page pairs in Pharma, Industrial, and the
  Computer Science calibration domain.
- The host-feature path costs about 52 ms per pair versus 229--235 ms from raw
  pages on the two held-out physical domains, a reduction of about 77%.
- A serialized NVMe path with best-effort page-cache eviction costs about
  56.8--56.9 ms per pair over 256-pair pilots in all three domains. The raw
  path costs 226--237 ms and the score error remains zero.
- Batch size eight does not materially improve the optimized raw-page verifier
  over batch size four, so the crossover is not explained by a weak small-batch
  baseline.
- At a 5% persistent-page budget, history-frequency replay produces a stable
  defer--partial--full cumulative-cost phase under the real benchmark order and
  five random orders in Pharma and Industrial.

## What is not established

- No deployable materialization policy currently closes the future-frequency
  oracle gap.
- Computer Science is a verifier calibration domain, not an unseen transfer
  domain for the final method.
- Benchmark permutations are controlled traces, not natural production
  workloads.
- `POSIX_FADV_DONTNEED` is a best-effort cold-read hint, not strict direct I/O
  or a concurrent serving measurement.
- The current work does not yet jointly solve discovery materialization and
  exact feature materialization.
- Visual-feature reuse itself is not the novelty; the compiler decision and
  its quality/cost transfer evidence must carry the contribution.

## Next implementation milestone

Freeze temporal and domain splits, then implement the simplest realizable
two-action page-value policy. For page `d`, estimate separately:

```text
reuse_value(d) = predicted_future_uses(d) * (raw_cost - feature_cost)
                  - feature_build_cost - feature_storage_cost

discovery_value(d) = predicted_escape_repair(d)
                      - predicted_ranking_interference(d)
                      - retrieval_build_and_storage_cost
```

Select feature and retrieval states under independent build/storage budgets.
The first comparison must include never materialize (DVI-like), materialize
all, first-touch, second-touch, history frequency, visual-risk-only, their
simple union, the realizable joint policy, and a future-aware oracle.

## Reading order

1. `docs/HANDOFF.md` -- current state and next milestone;
2. `docs/current-research-spec.md` -- problem and claim boundary;
3. `docs/progressive-materialization-experiment-matrix.md` -- evaluation plan;
4. `docs/evidence-registry.md` -- active, negative, and historical evidence;
5. `docs/benchmark-landscape.md` -- benchmark and prior-work map.

All other documents are historical evidence unless the registry explicitly
promotes them. Do not continue an older method merely because it has more code
or more result files.

## Verification and Git state

```bash
pytest -q
python -m json.tool \
  results/compiler-feasibility/closure-materialization-v0-2026-08-08.json \
  >/dev/null
```

Latest local commits at the time of this handoff:

- `548b4db` -- persistent visual-feature NVMe measurements;
- `e2496fc` -- 1024-pair equivalence and batch-eight baseline;
- `9669e17` -- exact visual-feature materialization prototype.

Large artifacts are intentionally outside Git. Remote push requires explicit
confirmation that the configured private repository is owned by the user.
