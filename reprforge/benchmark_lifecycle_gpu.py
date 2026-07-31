#!/usr/bin/env python3
"""Physical A100 comparison of no-cache and versioned visual upgrades."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from reprforge.heterogeneous_index import (
    benchmark_runtime,
    compile_heterogeneous_index,
    load_query_bank,
)
from reprforge.lifecycle_replay import VISUAL_TYPES, make_stream
from reprforge.policy_replay import load_replay_data, uniform_plan
from reprforge.retrieval_baselines import (
    PreencodedNoCacheRuntime,
    benchmark_resident_selected_runtime,
    benchmark_selected_runtime,
)
from reprforge.versioned_visual_index import (
    VersionedVisualIndex,
    create_versioned_visual_index,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_gpu_lifecycle(
    *,
    route_bank_directory: Path,
    locator_path: Path,
    output: Path,
    workload: str,
    locator_k: int,
    device: str,
    storage_dtype: str,
    document_batch_size: int,
    token_batch_budget: int | None,
    warmup: int,
    repetitions: int,
    top_k: int,
    no_cache_host_mode: str,
) -> dict[str, Any]:
    if locator_k <= 0:
        raise ValueError("locator_k must be positive")
    bank = route_bank_directory / "embedding-bank"
    data = load_replay_data(
        route_bank_directory / "items.jsonl",
        route_bank_directory / "queries.jsonl",
        route_bank_directory / "scores.jsonl",
    )
    query_rows = _read_jsonl(route_bank_directory / "queries.jsonl")
    query_rows_by_id = {
        str(row["query_id"]): row for row in query_rows
    }
    ordered_query_ids = [
        str(row["query_id"])
        for row in sorted(
            query_rows,
            key=lambda row: (
                int(row.get("document_index", 0)),
                int(row.get("source_query_index", 0)),
                str(row["query_id"]),
            ),
        )
    ]
    stream_ids = make_stream(ordered_query_ids, workload)
    locator = json.loads(locator_path.read_text(encoding="utf-8"))
    content_types = {item.item_id: item.content_type for item in data.items}
    selections_by_query: dict[str, tuple[str, ...]] = {}
    for query_id in ordered_query_ids:
        row = query_rows_by_id[query_id]
        scores = locator["scores"][query_id]
        visual = [
            str(item_id)
            for item_id in row["candidate_item_ids"]
            if content_types[str(item_id)] in VISUAL_TYPES
        ]
        selections_by_query[query_id] = tuple(
            sorted(
                visual,
                key=lambda item_id: (-float(scores[item_id]), item_id),
            )[:locator_k]
        )
    query_ids, query_embeddings = load_query_bank(bank)
    embeddings = dict(zip(query_ids, query_embeddings, strict=True))
    stream_embeddings = [embeddings[query_id] for query_id in stream_ids]
    stream_selections = [
        selections_by_query[query_id] for query_id in stream_ids
    ]
    if any(not selected for selected in stream_selections):
        missing = [
            query_id
            for query_id, selected in zip(
                stream_ids, stream_selections, strict=True
            )
            if not selected
        ]
        raise ValueError(
            f"queries contain no visual locator candidates: {missing[:5]}"
        )
    items = {item.item_id: item for item in data.items}
    base_encode_ms = sum(
        item.route_costs["image-pool-25"].encode_ms for item in data.items
    )
    no_cache_encode_ms = sum(
        items[item_id].route_costs["image"].encode_ms
        for selected in stream_selections
        for item_id in selected
    )
    cached_ids = tuple(
        item.item_id
        for item in data.items
        if item.item_id
        in {
            item_id
            for selected in stream_selections
            for item_id in selected
        }
    )
    cache_encode_ms = sum(
        items[item_id].route_costs["image"].encode_ms
        for item_id in cached_ids
    )
    runtime_options = {
        "device": device,
        "document_batch_size": document_batch_size,
        "token_batch_budget": token_batch_budget,
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".lifecycle-gpu-",
        dir=output,
    ) as temporary:
        work = Path(temporary)
        pool_index = work / "uniform-pool25"
        full_index = work / "uniform-image"
        compile_heterogeneous_index(
            bank=bank,
            plan=uniform_plan(data.items, "image-pool-25"),
            output=pool_index,
            storage_dtype=storage_dtype,
        )
        compile_heterogeneous_index(
            bank=bank,
            plan=uniform_plan(data.items, "image"),
            output=full_index,
            storage_dtype=storage_dtype,
        )
        no_cache_results: dict[str, Any] = {}
        host_modes = {
            "pageable": (False,),
            "pinned": (True,),
            "both": (False, True),
        }
        if no_cache_host_mode not in host_modes:
            raise ValueError(
                f"unsupported no-cache host mode: {no_cache_host_mode}"
            )
        for pinned_host in host_modes[no_cache_host_mode]:
            name = "pinned" if pinned_host else "pageable"
            no_cache = PreencodedNoCacheRuntime(
                pool_index,
                full_index,
                pinned_host=pinned_host,
                **runtime_options,
            )
            no_cache_results[name] = benchmark_selected_runtime(
                no_cache,
                query_ids=stream_ids,
                query_embeddings=stream_embeddings,
                selections=stream_selections,
                warmup=warmup,
                repetitions=repetitions,
                top_k=top_k,
            )
            del no_cache

        tiered_path = work / "versioned-cache"
        began = time.perf_counter()
        create_versioned_visual_index(
            bank=bank,
            output=tiered_path,
            base_route="image-pool-25",
            visual_route="image",
            storage_dtype=storage_dtype,
        )
        create_ms = (time.perf_counter() - began) * 1000.0
        controller = VersionedVisualIndex(tiered_path)
        began = time.perf_counter()
        materialization = controller.materialize(cached_ids)
        materialize_ms = (time.perf_counter() - began) * 1000.0
        cache_runtime = controller.torch_runtime(**runtime_options)
        cache_result = benchmark_runtime(
            cache_runtime,
            query_ids=stream_ids,
            query_embeddings=stream_embeddings,
            warmup=warmup,
            repetitions=repetitions,
            top_k=top_k,
        )
        selective_runtime = controller.selective_torch_runtime(
            **runtime_options
        )
        selective_result = benchmark_resident_selected_runtime(
            selective_runtime,
            query_ids=stream_ids,
            query_embeddings=stream_embeddings,
            selections=stream_selections,
            warmup=warmup,
            repetitions=repetitions,
            top_k=top_k,
        )

    result = {
        "contract": {
            "dataset": "MMDocIR/MMDocIR_Evaluation_Dataset",
            "workload": workload,
            "locator": str(locator["manifest"]["method"]),
            "locator_content_sha256": str(
                locator["manifest"]["content_sha256"]
            ),
            "locator_k": locator_k,
            "device": device,
            "storage_dtype": storage_dtype,
            "warmup": warmup,
            "repetitions": repetitions,
            "top_k": top_k,
            "selection_scope": "official-document-local-candidate-pool",
            "physical_search_scope": "all-bank-items-after-selection",
            "model_encoding_measured": False,
            "model_encoding_cost_replayed": True,
            "no_cache_host_mode": no_cache_host_mode,
        },
        "corpus": {
            "items": len(data.items),
            "queries": len(data.queries),
            "stream_steps": len(stream_ids),
            "unique_cached_items": len(cached_ids),
        },
        "no_cache": {
            "interpretation": "preencoded-vector-lower-bound",
            "base_encode_ms_replayed": base_encode_ms,
            "visual_encode_ms_replayed": no_cache_encode_ms,
            "visual_encode_calls": sum(map(len, stream_selections)),
            "performance": no_cache_results,
        },
        "versioned_cache": {
            "base_encode_ms_replayed": base_encode_ms,
            "visual_encode_ms_replayed": cache_encode_ms,
            "visual_encode_calls": len(cached_ids),
            "create_base_ms_embedding_copy_only": create_ms,
            "publish_delta_ms_embedding_copy_only": materialize_ms,
            "materialization": materialization,
            "performance": cache_result,
            "query_scoped_activation_performance": selective_result,
        },
    }
    _write_json(output / "gpu-lifecycle.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-bank-directory", type=Path, required=True)
    parser.add_argument("--locator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workload",
        choices=("single-pass", "cyclic-4", "hotset-25-4", "hotset-10-4"),
        required=True,
    )
    parser.add_argument("--locator-k", type=int, required=True)
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
    parser.add_argument(
        "--no-cache-host-mode",
        choices=("pageable", "pinned", "both"),
        default="both",
    )
    args = parser.parse_args()
    result = run_gpu_lifecycle(
        route_bank_directory=args.route_bank_directory,
        locator_path=args.locator,
        output=args.output,
        workload=args.workload,
        locator_k=args.locator_k,
        device=args.device,
        storage_dtype=args.storage_dtype,
        document_batch_size=args.document_batch_size,
        token_batch_budget=args.token_batch_budget,
        warmup=args.warmup,
        repetitions=args.repetitions,
        top_k=args.top_k,
        no_cache_host_mode=args.no_cache_host_mode,
    )
    print(
        json.dumps(
            {
                "corpus": result["corpus"],
                "no_cache": result["no_cache"],
                "versioned_cache": result["versioned_cache"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
