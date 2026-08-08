# Active code architecture

Status: implementation map, 2026-08-08.

The repository contains many correct modules from rejected research branches.
File count is not a signal of current importance. This document identifies the
small execution slice that supports the active materialization question.

## Current experimental flow

```text
document pages
    -> complete cheap locator (BM25 and/or compact visual locator)
    -> query candidate cohort
    -> raw-page Qwen verifier OR reusable visual feature
    -> query-conditioned relevance score
    -> cumulative defer / partial / full materialization accounting
```

Discovery materialization is evaluated separately through complete-corpus
visual ranking surfaces. Feature materialization accelerates the verifier but
does not add pages to the candidate cohort.

## Active library modules

| File | Responsibility |
|---|---|
| `reprforge/partial_vlm_materialization.py` | Immutable score-surface loading and shared retrieval metrics. |
| `reprforge/closure_materialization.py` | Query-scope candidate closure and persistent/transient page planning. |
| `reprforge/defer_materialize_phase.py` | Pure cumulative-cost and crossover calculations. |
| `reprforge/dvi_page_verifier.py` | Deterministic query sampling, ranking metrics, reranking, and verifier helpers. |
| `reprforge/visual_page_features.py` | Cheap page-image features used only by realizable selectors or diagnostics. |
| `reprforge/materialization/states.py` | Orthogonal feature and retrieval state, rather than one overloaded "visual" bit. |
| `reprforge/materialization/costs.py` | Measured build, storage, raw-query, and cached-query cost catalog. |
| `reprforge/materialization/split.py` | Frozen fit/calibration/test split and deterministic query-order permutations. |
| `reprforge/materialization/policy.py` | Leakage-safe v0 feature-reuse and retrieval-value plan compiler. |
| `reprforge/materialization/replay.py` | Exact-quality feature-state replay for static and touch-based baselines. |

These modules are measurement substrate, not the final paper algorithm.

## Active experiment entry points

| File | Responsibility |
|---|---|
| `tools/run_dvi_page_verifier_pilot.py` | DVI-like raw-page quality and timing baseline. |
| `tools/benchmark_qwen_visual_feature_cache.py` | Exact host/NVMe visual-feature materialization benchmark. |
| `tools/evaluate_closure_materialization.py` | Frozen candidate closure and workload replay generation. |
| `tools/analyze_visual_feature_cache_phase.py` | Exact-quality defer--partial--full phase analysis. |
| `tools/benchmark_omni_page_construction.py` | Real Full Omni build cost. |
| `tools/benchmark_omni_closure_runtime.py` | Physical candidate-cohort execution. |
| `tools/evaluate_materialization_compiler_v0.py` | Two-action v0, core baselines, query-order repeats, and sparse-support fusion audit. |

Each GPU tool writes a new output and refuses to overwrite an existing result.
Paper-facing summaries bind raw outputs by SHA-256.

## Historical code boundary

Modules for RBRC, frontier scheduling, residency-aware Top-k routing,
compression certificates, token witnesses, and earlier value-aware selectors
remain tested historical evidence. In particular, do not extend
`reprforge/rbrc_v0.py` or the existing 900-line
`reprforge/value_aware_materialization.py` as the new method. Their assumptions
were shaped by earlier query-side or single-representation questions.

When the two-action materialization policy is implemented, create a small new
package with explicit boundaries rather than adding another monolithic module:

```text
reprforge/materialization/
    states.py       # independent feature/retrieval state and transitions
    costs.py        # measured per-operator costs
    split.py        # frozen workload splits and query orders
    policy.py       # realizable feature/retrieval admission
    replay.py       # exact-quality feature cost replay
```

The package now exists and v0 has passed implementation tests, but not the
method gate: its static retrieval selector leaves a large oracle gap. Only
after a replacement passes frozen cross-domain evaluation should obsolete
modules and tests be physically moved into a release archive.
