#!/usr/bin/env python3
"""Compare no-cache and versioned visual upgrades on public query streams."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from reprforge.policy_replay import (
    ReplayData,
    load_replay_data,
    uniform_plan,
)
from reprforge.route_mechanism_analysis import subset_by_documents


VISUAL_TYPES = frozenset({"chart", "figure", "image", "table"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_stream(query_ids: Sequence[str], workload: str) -> list[str]:
    ordered = list(query_ids)
    if workload == "single-pass":
        return ordered
    if workload == "cyclic-4":
        return ordered * 4
    if workload == "hotset-25-4":
        size = max(1, math.ceil(len(ordered) * 0.25))
        return ordered[:size] * 4
    if workload == "hotset-10-4":
        size = max(1, math.ceil(len(ordered) * 0.10))
        return ordered[:size] * 4
    raise ValueError(f"unsupported workload: {workload}")


def _rank(
    data: ReplayData,
    plan: Mapping[str, str],
    query_id: str,
) -> list[str]:
    query = next(query for query in data.queries if query.query_id == query_id)
    candidates = list(query.candidate_item_ids or ())
    return sorted(
        candidates,
        key=lambda item_id: (
            -data.scores[plan[item_id]][query_id][item_id],
            item_id,
        ),
    )


def _query_metrics(
    data: ReplayData,
    ranked: Sequence[str],
    query_id: str,
) -> tuple[float, float]:
    query = next(query for query in data.queries if query.query_id == query_id)
    dcg = sum(
        query.relevance.get(item_id, 0.0) / math.log2(rank + 1)
        for rank, item_id in enumerate(ranked[:10], start=1)
    )
    ideal = sorted(query.relevance.values(), reverse=True)[:10]
    idcg = sum(
        value / math.log2(rank + 1)
        for rank, value in enumerate(ideal, start=1)
    )
    denominator = (
        query.relevance_denominator
        if query.relevance_denominator is not None
        else sum(query.relevance.values())
    )
    recall = sum(
        query.relevance.get(item_id, 0.0) for item_id in ranked[:5]
    ) / denominator
    return (dcg / idcg if idcg else 0.0), recall


def _visual_selection(
    data: ReplayData,
    locator: Mapping[str, Any],
    query_id: str,
    *,
    locator_k: int,
) -> list[str]:
    query = next(query for query in data.queries if query.query_id == query_id)
    content_types = {item.item_id: item.content_type for item in data.items}
    scores = locator["scores"][query_id]
    visual = [
        item_id
        for item_id in (query.candidate_item_ids or ())
        if content_types[item_id] in VISUAL_TYPES
    ]
    return sorted(
        visual,
        key=lambda item_id: (-float(scores[item_id]), item_id),
    )[:locator_k]


def _base_bytes(data: ReplayData) -> int:
    return sum(
        item.route_costs["image-pool-25"].index_bytes for item in data.items
    )


def _delta_bytes(data: ReplayData, item_ids: Sequence[str] | set[str]) -> int:
    items = {item.item_id: item for item in data.items}
    return sum(
        items[item_id].route_costs["image"].index_bytes
        for item_id in item_ids
    )


def _encode_ms(data: ReplayData, item_ids: Sequence[str] | set[str]) -> float:
    items = {item.item_id: item for item in data.items}
    return sum(
        items[item_id].route_costs["image"].encode_ms
        for item_id in item_ids
    )


def _summarize(
    *,
    name: str,
    steps: Sequence[Mapping[str, Any]],
    base_encode_ms: float,
    upgrade_encode_ms: float,
    encode_calls: int,
    unique_upgrades: int,
    final_active_bytes: int,
    peak_active_bytes: int,
) -> dict[str, Any]:
    return {
        "policy": name,
        "stream_steps": len(steps),
        "stream_ndcg_at_10": sum(
            float(step["ndcg_at_10"]) for step in steps
        )
        / len(steps),
        "stream_recall_at_5": sum(
            float(step["recall_at_5"]) for step in steps
        )
        / len(steps),
        "base_encode_ms": base_encode_ms,
        "upgrade_encode_ms": upgrade_encode_ms,
        "total_encode_ms": base_encode_ms + upgrade_encode_ms,
        "visual_encode_calls": encode_calls,
        "unique_visual_upgrades": unique_upgrades,
        "final_active_index_bytes": final_active_bytes,
        "peak_active_index_bytes": peak_active_bytes,
        "steps": list(steps),
    }


def compare_lifecycle(
    data: ReplayData,
    *,
    stream: Sequence[str],
    locator: Mapping[str, Any],
    locator_k: int,
) -> dict[str, Any]:
    if locator_k <= 0:
        raise ValueError("locator_k must be positive")
    base_plan = uniform_plan(data.items, "image-pool-25")
    base_bytes = _base_bytes(data)
    base_encode_ms = sum(
        item.route_costs["image-pool-25"].encode_ms for item in data.items
    )

    static_visual_ids = {
        item.item_id
        for item in data.items
        if item.content_type in VISUAL_TYPES
    }
    static_plan = dict(base_plan)
    static_plan.update({item_id: "image" for item_id in static_visual_ids})

    def static_steps(plan: Mapping[str, str]) -> list[dict[str, Any]]:
        result = []
        for step, query_id in enumerate(stream):
            ndcg, recall = _query_metrics(
                data, _rank(data, plan, query_id), query_id
            )
            result.append(
                {
                    "step": step,
                    "query_id": query_id,
                    "ndcg_at_10": ndcg,
                    "recall_at_5": recall,
                }
            )
        return result

    base_steps = static_steps(base_plan)
    static_steps_result = static_steps(static_plan)

    no_cache_steps: list[dict[str, Any]] = []
    no_cache_encode_ms = 0.0
    no_cache_calls = 0
    no_cache_unique: set[str] = set()
    no_cache_peak_bytes = base_bytes
    for step, query_id in enumerate(stream):
        selected = _visual_selection(
            data, locator, query_id, locator_k=locator_k
        )
        temporary = dict(base_plan)
        temporary.update({item_id: "image" for item_id in selected})
        ndcg, recall = _query_metrics(
            data, _rank(data, temporary, query_id), query_id
        )
        encode_ms = _encode_ms(data, selected)
        no_cache_encode_ms += encode_ms
        no_cache_calls += len(selected)
        no_cache_unique.update(selected)
        active_bytes = base_bytes + _delta_bytes(data, selected)
        no_cache_peak_bytes = max(no_cache_peak_bytes, active_bytes)
        no_cache_steps.append(
            {
                "step": step,
                "query_id": query_id,
                "selected_items": len(selected),
                "ndcg_at_10": ndcg,
                "recall_at_5": recall,
                "visual_encode_ms": encode_ms,
                "active_index_bytes": active_bytes,
            }
        )

    cached: set[str] = set()
    cache_steps: list[dict[str, Any]] = []
    cache_encode_ms = 0.0
    for step, query_id in enumerate(stream):
        selected = _visual_selection(
            data, locator, query_id, locator_k=locator_k
        )
        new_items = [item_id for item_id in selected if item_id not in cached]
        cached.update(new_items)
        encode_ms = _encode_ms(data, new_items)
        cache_encode_ms += encode_ms
        plan = dict(base_plan)
        plan.update({item_id: "image" for item_id in cached})
        ndcg, recall = _query_metrics(
            data, _rank(data, plan, query_id), query_id
        )
        cache_steps.append(
            {
                "step": step,
                "query_id": query_id,
                "requested_items": len(selected),
                "new_items": len(new_items),
                "cached_items": len(cached),
                "ndcg_at_10": ndcg,
                "recall_at_5": recall,
                "visual_encode_ms": encode_ms,
                "active_index_bytes": base_bytes + _delta_bytes(data, cached),
            }
        )

    return {
        "uniform-pool25": _summarize(
            name="uniform-pool25",
            steps=base_steps,
            base_encode_ms=base_encode_ms,
            upgrade_encode_ms=0.0,
            encode_calls=0,
            unique_upgrades=0,
            final_active_bytes=base_bytes,
            peak_active_bytes=base_bytes,
        ),
        "static-pool25-visual-full": _summarize(
            name="static-pool25-visual-full",
            steps=static_steps_result,
            base_encode_ms=base_encode_ms,
            upgrade_encode_ms=_encode_ms(data, static_visual_ids),
            encode_calls=len(static_visual_ids),
            unique_upgrades=len(static_visual_ids),
            final_active_bytes=(
                base_bytes + _delta_bytes(data, static_visual_ids)
            ),
            peak_active_bytes=(
                base_bytes + _delta_bytes(data, static_visual_ids)
            ),
        ),
        "locator-no-cache-preencoded-lower-bound": _summarize(
            name="locator-no-cache-preencoded-lower-bound",
            steps=no_cache_steps,
            base_encode_ms=base_encode_ms,
            upgrade_encode_ms=no_cache_encode_ms,
            encode_calls=no_cache_calls,
            unique_upgrades=len(no_cache_unique),
            final_active_bytes=base_bytes,
            peak_active_bytes=no_cache_peak_bytes,
        ),
        "locator-versioned-cache": _summarize(
            name="locator-versioned-cache",
            steps=cache_steps,
            base_encode_ms=base_encode_ms,
            upgrade_encode_ms=cache_encode_ms,
            encode_calls=len(cached),
            unique_upgrades=len(cached),
            final_active_bytes=base_bytes + _delta_bytes(data, cached),
            peak_active_bytes=base_bytes + _delta_bytes(data, cached),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--locator", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("prior-development", "mechanism-design", "final-evaluation"),
        required=True,
    )
    parser.add_argument(
        "--workload",
        choices=("single-pass", "cyclic-4", "hotset-25-4", "hotset-10-4"),
        required=True,
    )
    parser.add_argument("--locator-k", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    documents = {
        int(row["document_index"])
        for row in protocol["documents"]
        if str(row["role"]) == args.role
    }
    item_rows = _read_jsonl(args.data / "items.jsonl")
    query_rows = _read_jsonl(args.data / "queries.jsonl")
    full = load_replay_data(
        args.data / "items.jsonl",
        args.data / "queries.jsonl",
        args.data / "scores.jsonl",
    )
    data, _ = subset_by_documents(
        full,
        query_rows=query_rows,
        item_rows=item_rows,
        document_indices=documents,
    )
    selected_query_rows = [
        row for row in query_rows if int(row["document_index"]) in documents
    ]
    query_order = [
        str(row["query_id"])
        for row in sorted(
            selected_query_rows,
            key=lambda row: (
                int(row["document_index"]),
                int(row["source_query_index"]),
            ),
        )
    ]
    locator = json.loads(args.locator.read_text(encoding="utf-8"))
    result = {
        "contract": {
            "dataset": "MMDocIR/MMDocIR_Evaluation_Dataset",
            "role": args.role,
            "workload": args.workload,
            "locator": str(locator["manifest"]["method"]),
            "locator_content_sha256": str(
                locator["manifest"]["content_sha256"]
            ),
            "locator_k": args.locator_k,
            "full_visual_encoding_cost_included": True,
            "gpu_query_latency_included": False,
            "no_cache_interpretation": "preencoded-vector-lower-bound",
        },
        "corpus": {
            "documents": len(documents),
            "items": len(data.items),
            "queries": len(data.queries),
            "stream_steps": len(make_stream(query_order, args.workload)),
        },
        "results": compare_lifecycle(
            data,
            stream=make_stream(query_order, args.workload),
            locator=locator,
            locator_k=args.locator_k,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        name: {
            key: row[key]
            for key in (
                "stream_ndcg_at_10",
                "stream_recall_at_5",
                "total_encode_ms",
                "visual_encode_calls",
                "unique_visual_upgrades",
                "final_active_index_bytes",
                "peak_active_index_bytes",
            )
        }
        for name, row in result["results"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
