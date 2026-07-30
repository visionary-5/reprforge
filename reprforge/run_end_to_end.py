#!/usr/bin/env python3
"""Compile, validate, and benchmark one ReprForge representation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reprforge.heterogeneous_index import (
    NumpyMaxSimRuntime,
    TorchMaxSimRuntime,
    benchmark_runtime,
    compile_heterogeneous_index,
    evaluate_runtime,
    load_plan,
    load_query_bank,
)
from reprforge.policy_replay import load_replay_data, plan_cost


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_end_to_end(
    *,
    route_bank_directory: Path,
    plan_path: Path | None,
    plan_key: str | None,
    uniform_route: str | None,
    output: Path,
    storage_dtype: str,
    engine: str,
    device: str,
    document_batch_size: int,
    warmup: int,
    repetitions: int,
    top_k: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict:
    """Materialize a plan and return one complete quality/cost/runtime record."""

    embedding_bank = route_bank_directory / "embedding-bank"
    replay = load_replay_data(
        route_bank_directory / "items.jsonl",
        route_bank_directory / "queries.jsonl",
        route_bank_directory / "scores.jsonl",
    )
    if (plan_path is None) == (uniform_route is None):
        raise ValueError("provide exactly one plan path or uniform route")
    if uniform_route is not None:
        if uniform_route not in replay.routes:
            raise ValueError(
                f"uniform route {uniform_route!r} is unavailable; "
                f"choose from {list(replay.routes)}"
            )
        plan = {
            item.item_id: uniform_route for item in replay.items
        }
    else:
        assert plan_path is not None
        plan = load_plan(plan_path, dotted_key=plan_key)
    index_directory = output / "index"
    index_manifest = compile_heterogeneous_index(
        bank=embedding_bank,
        plan=plan,
        output=index_directory,
        storage_dtype=storage_dtype,
    )
    query_ids, query_embeddings = load_query_bank(embedding_bank)
    if engine == "numpy":
        runtime = NumpyMaxSimRuntime(index_directory)
    elif engine == "torch":
        runtime = TorchMaxSimRuntime(
            index_directory,
            device=device,
            document_batch_size=document_batch_size,
        )
    else:
        raise ValueError(f"unsupported engine: {engine}")

    correctness = evaluate_runtime(
        runtime,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        replay_directory=route_bank_directory,
        source_plan=plan,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    performance = benchmark_runtime(
        runtime,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        warmup=warmup,
        repetitions=repetitions,
        top_k=top_k,
    )
    logical_cost = plan_cost(replay.items, plan)
    result = {
        "system": "ReprForge",
        "route_bank_directory": str(route_bank_directory),
        "plan_path": str(plan_path) if plan_path is not None else None,
        "plan_key": plan_key,
        "uniform_route": uniform_route,
        "engine": engine,
        "device": device if engine == "torch" else "cpu",
        "storage_dtype": storage_dtype,
        "document_batch_size": (
            document_batch_size if engine == "torch" else None
        ),
        "index": index_manifest,
        "logical_planner_cost": logical_cost,
        "correctness": correctness,
        "performance": performance,
    }
    _write_json(output / "end-to-end.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-bank-directory", type=Path, required=True)
    plan_group = parser.add_mutually_exclusive_group(required=True)
    plan_group.add_argument("--plan", type=Path)
    plan_group.add_argument("--uniform-route")
    parser.add_argument("--plan-key")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float32",
    )
    parser.add_argument("--engine", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--document-batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    result = run_end_to_end(
        route_bank_directory=args.route_bank_directory,
        plan_path=args.plan,
        plan_key=args.plan_key,
        uniform_route=args.uniform_route,
        output=args.output,
        storage_dtype=args.storage_dtype,
        engine=args.engine,
        device=args.device,
        document_batch_size=args.document_batch_size,
        warmup=args.warmup,
        repetitions=args.repetitions,
        top_k=args.top_k,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    summary = {
        "index_bytes": result["index"]["compact_vector_bytes"],
        "latency_p50_ms": result["performance"]["latency_ms"]["p50"],
        "latency_p95_ms": result["performance"]["latency_ms"]["p95"],
        "qps": result["performance"]["qps"],
        "score_all_close": result["correctness"]["score_contract"]["all_close"],
        "ranking_all_equal": result["correctness"]["ranking_contract"]["all_equal"],
        "recall_at_5": result["correctness"]["physical_metrics"]["recall_at_5"],
        "ndcg_at_10": result["correctness"]["physical_metrics"]["ndcg_at_10"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
