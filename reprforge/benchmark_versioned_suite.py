#!/usr/bin/env python3
"""Benchmark the versioned visual index against public MMDocIR baselines.

The suite deliberately separates:

* logical retrieval quality, measured with MMDocIR's public candidate pools;
* physical A100 query cost, measured over every item in the persisted bank;
* lifecycle cost, measured while creating the base and publishing the visual
  generation; and
* composition overhead, measured against an equivalent single compiled index.

Large compiled indexes live in a temporary directory and are removed after
the compact JSON result has been written.
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from reprforge.heterogeneous_index import (
    TorchMaxSimRuntime,
    benchmark_runtime,
    compile_heterogeneous_index,
    evaluate_runtime,
    load_query_bank,
)
from reprforge.policy_replay import (
    fixed_hybrid_plan,
    load_replay_data,
    uniform_plan,
)
from reprforge.retrieval_baselines import (
    PooledExactRerankRuntime,
    evaluate_two_stage_runtime,
)
from reprforge.versioned_visual_index import (
    VersionedVisualIndex,
    create_versioned_visual_index,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _ranking(scores: np.ndarray, item_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        item_ids[index]
        for index in sorted(
            range(len(item_ids)),
            key=lambda index: (-float(scores[index]), item_ids[index]),
        )
    )


def _compare_runtimes(
    observed_runtime: Any,
    expected_runtime: Any,
    *,
    query_embeddings: list[np.ndarray],
    top_k: int,
    reference: str,
) -> dict[str, Any]:
    max_absolute_error = 0.0
    max_relative_error = 0.0
    equal_full_rankings = 0
    equal_top_k = 0
    for embedding in query_embeddings:
        observed = observed_runtime.score(embedding)
        expected = expected_runtime.score(embedding)
        difference = np.abs(observed - expected)
        max_absolute_error = max(
            max_absolute_error, float(difference.max(initial=0.0))
        )
        relative = difference / np.maximum(np.abs(expected), 1e-12)
        max_relative_error = max(
            max_relative_error, float(relative.max(initial=0.0))
        )
        observed_ranking = _ranking(observed, observed_runtime.item_ids)
        expected_ranking = _ranking(expected, expected_runtime.item_ids)
        equal_full_rankings += int(observed_ranking == expected_ranking)
        equal_top_k += int(
            observed_ranking[:top_k] == expected_ranking[:top_k]
        )
    return {
        "reference": reference,
        "queries": len(query_embeddings),
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "equal_full_rankings": equal_full_rankings,
        "equal_top_k_rankings": equal_top_k,
        "all_full_rankings_equal": (
            equal_full_rankings == len(query_embeddings)
        ),
        "all_top_k_rankings_equal": equal_top_k == len(query_embeddings),
    }


def run_suite(
    *,
    route_bank_directory: Path,
    output: Path,
    device: str,
    storage_dtype: str,
    document_batch_size: int,
    token_batch_budget: int | None,
    warmup: int,
    repetitions: int,
    top_k: int,
) -> dict[str, Any]:
    bank = route_bank_directory / "embedding-bank"
    replay = load_replay_data(
        route_bank_directory / "items.jsonl",
        route_bank_directory / "queries.jsonl",
        route_bank_directory / "scores.jsonl",
    )
    query_ids, query_embeddings = load_query_bank(bank)
    fixed_hybrid = fixed_hybrid_plan(replay.items)
    pool25_visual_full = {
        item_id: ("image" if route == "image" else "image-pool-25")
        for item_id, route in fixed_hybrid.items()
    }
    plans = {
        "uniform-text": uniform_plan(replay.items, "text"),
        "uniform-image": uniform_plan(replay.items, "image"),
        "uniform-image-pool-25": uniform_plan(
            replay.items, "image-pool-25"
        ),
        "fixed-hybrid": fixed_hybrid,
        "pool25-visual-full": pool25_visual_full,
    }
    runtime_options = {
        "device": device,
        "document_batch_size": document_batch_size,
        "token_batch_budget": token_batch_budget,
    }
    output.mkdir(parents=True, exist_ok=True)
    baselines: dict[str, Any] = {}
    two_stage_results: dict[str, Any] = {}
    lifecycle_results: dict[str, Any] = {}
    equivalence_results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(
        prefix=".versioned-suite-",
        dir=output,
    ) as temporary:
        work = Path(temporary)
        static_paths: dict[str, Path] = {}
        for name, plan in plans.items():
            index_path = work / name
            began = time.perf_counter()
            manifest = compile_heterogeneous_index(
                bank=bank,
                plan=plan,
                output=index_path,
                storage_dtype=storage_dtype,
            )
            build_ms = (time.perf_counter() - began) * 1000.0
            runtime = TorchMaxSimRuntime(index_path, **runtime_options)
            quality = evaluate_runtime(
                runtime,
                query_ids=query_ids,
                query_embeddings=query_embeddings,
                replay_directory=route_bank_directory,
                source_plan=plan,
            )
            performance = benchmark_runtime(
                runtime,
                query_ids=query_ids,
                query_embeddings=query_embeddings,
                warmup=warmup,
                repetitions=repetitions,
                top_k=top_k,
            )
            baselines[name] = {
                "build_ms_embedding_copy_only": build_ms,
                "index": manifest,
                "quality": quality,
                "performance": performance,
            }
            static_paths[name] = index_path
            del runtime
            _clear_cuda_cache()

        for candidate_k in (10, 20, 50):
            name = f"pool25-exact-rerank-{candidate_k}"
            runtime = PooledExactRerankRuntime(
                static_paths["uniform-image-pool-25"],
                static_paths["uniform-image"],
                device=device,
                candidate_k=candidate_k,
                document_batch_size=document_batch_size,
                token_batch_budget=token_batch_budget,
            )
            quality = evaluate_two_stage_runtime(
                runtime,
                query_ids=query_ids,
                query_embeddings=query_embeddings,
                replay_directory=route_bank_directory,
            )
            performance = benchmark_runtime(
                runtime,
                query_ids=query_ids,
                query_embeddings=query_embeddings,
                warmup=warmup,
                repetitions=repetitions,
                top_k=top_k,
            )
            two_stage_results[name] = {
                "candidate_k": candidate_k,
                "quality": quality,
                "performance": performance,
            }
            del runtime
            _clear_cuda_cache()

        lifecycle_specs = {
            "text-base-visual-delta": {
                "base_route": "text",
                "target_plan": "fixed-hybrid",
            },
            "pool25-base-visual-delta": {
                "base_route": "image-pool-25",
                "target_plan": "pool25-visual-full",
            },
        }
        for name, spec in lifecycle_specs.items():
            tiered_path = work / name
            began = time.perf_counter()
            root_manifest = create_versioned_visual_index(
                bank=bank,
                output=tiered_path,
                base_route=spec["base_route"],
                visual_route="image",
                storage_dtype=storage_dtype,
            )
            create_ms = (time.perf_counter() - began) * 1000.0
            target_plan = plans[spec["target_plan"]]
            visual_item_ids = [
                item_id
                for item_id, route in target_plan.items()
                if route == "image"
            ]
            controller = VersionedVisualIndex(tiered_path)
            began = time.perf_counter()
            materialization = controller.materialize(visual_item_ids)
            materialize_ms = (time.perf_counter() - began) * 1000.0
            runtime = controller.torch_runtime(**runtime_options)
            performance = benchmark_runtime(
                runtime,
                query_ids=query_ids,
                query_embeddings=query_embeddings,
                warmup=warmup,
                repetitions=repetitions,
                top_k=top_k,
            )
            lifecycle_results[name] = {
                "target_plan": spec["target_plan"],
                "create_base_ms_embedding_copy_only": create_ms,
                "publish_delta_ms_embedding_copy_only": materialize_ms,
                "requested_visual_items": len(visual_item_ids),
                "root_manifest": root_manifest,
                "materialization": materialization,
                "status": controller.status(),
                "performance": performance,
            }
            compiled = TorchMaxSimRuntime(
                static_paths[spec["target_plan"]],
                **runtime_options,
            )
            equivalence_results[name] = _compare_runtimes(
                runtime,
                compiled,
                query_embeddings=query_embeddings,
                top_k=top_k,
                reference=f"single-compiled-{spec['target_plan']}",
            )
            del compiled
            del runtime
            _clear_cuda_cache()

    result = {
        "contract": {
            "dataset": "MMDocIR/MMDocIR_Evaluation_Dataset",
            "candidate_semantics": "official-within-document-for-quality",
            "physical_search_scope": "all-persisted-bank-items",
            "device": device,
            "storage_dtype": storage_dtype,
            "document_batch_size": document_batch_size,
            "token_batch_budget": token_batch_budget,
            "warmup": warmup,
            "repetitions": repetitions,
            "top_k": top_k,
            "embedding_time_included": False,
            "temporary_indexes_retained": False,
        },
        "corpus": {
            "items": len(replay.items),
            "queries": len(replay.queries),
            "routes": list(replay.routes),
        },
        "baselines": baselines,
        "two_stage_baselines": two_stage_results,
        "versioned_systems": lifecycle_results,
        "physical_equivalence": equivalence_results,
    }
    _write_json(output / "versioned-suite.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-bank-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float32",
    )
    parser.add_argument("--document-batch-size", type=int, default=64)
    parser.add_argument("--token-batch-budget", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    result = run_suite(
        route_bank_directory=args.route_bank_directory,
        output=args.output,
        device=args.device,
        storage_dtype=args.storage_dtype,
        document_batch_size=args.document_batch_size,
        token_batch_budget=args.token_batch_budget,
        warmup=args.warmup,
        repetitions=args.repetitions,
        top_k=args.top_k,
    )
    summary = {
        "corpus": result["corpus"],
        "baselines": {
            name: {
                "ndcg_at_10": row["quality"]["physical_metrics"][
                    "ndcg_at_10"
                ],
                "recall_at_5": row["quality"]["physical_metrics"][
                    "recall_at_5"
                ],
                "compact_vector_bytes": row["performance"][
                    "compact_vector_bytes"
                ],
                "latency_p50_ms": row["performance"]["latency_ms"]["p50"],
                "latency_p95_ms": row["performance"]["latency_ms"]["p95"],
            }
            for name, row in result["baselines"].items()
        },
        "two_stage_baselines": {
            name: {
                "candidate_k": row["candidate_k"],
                "ndcg_at_10": row["quality"]["ndcg_at_10"],
                "recall_at_5": row["quality"]["recall_at_5"],
                "compact_vector_bytes": row["performance"][
                    "compact_vector_bytes"
                ],
                "latency_p50_ms": row["performance"]["latency_ms"]["p50"],
                "latency_p95_ms": row["performance"]["latency_ms"]["p95"],
            }
            for name, row in result["two_stage_baselines"].items()
        },
        "versioned_systems": {
            name: {
                "target_plan": row["target_plan"],
                "cached_items": row["status"]["cached_items"],
                "compact_vector_bytes": row["performance"][
                    "compact_vector_bytes"
                ],
                "latency_p50_ms": row["performance"]["latency_ms"]["p50"],
                "latency_p95_ms": row["performance"]["latency_ms"]["p95"],
            }
            for name, row in result["versioned_systems"].items()
        },
        "physical_equivalence": result["physical_equivalence"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
