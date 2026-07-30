#!/usr/bin/env python3
"""Replay heterogeneous representation plans over frozen retrieval scores.

The first ReprForge system slice intentionally separates expensive model
execution from policy research:

1. Each representation route produces a score for every query/item pair.
2. A plan chooses exactly one route for each indexed item.
3. This module composes the chosen scores, ranks items, and reports retrieval
   quality plus *offline* construction/storage cost.

The replay format is model-independent and makes it possible to compare
MMDocIR's fixed layout-type hybrid with budget-matched alternatives without
re-encoding the corpus for every candidate plan.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TEXT_ROUTE = "text"
IMAGE_ROUTE = "image"
SUPPORTED_ROUTES = (TEXT_ROUTE, IMAGE_ROUTE)
VISUAL_LAYOUT_TYPES = frozenset({"chart", "figure", "image", "table"})


@dataclass(frozen=True)
class RouteCost:
    """Per-item offline cost for one representation route."""

    index_bytes: int
    encode_ms: float

    def __post_init__(self) -> None:
        if self.index_bytes < 0 or self.encode_ms < 0:
            raise ValueError("route costs must be non-negative")


@dataclass(frozen=True)
class Item:
    item_id: str
    content_type: str
    route_costs: Mapping[str, RouteCost]

    def __post_init__(self) -> None:
        if not self.route_costs:
            raise ValueError(f"{self.item_id} has no representation routes")


@dataclass(frozen=True)
class Query:
    query_id: str
    relevance: Mapping[str, float]
    relevance_denominator: float | None = None
    candidate_item_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.relevance:
            raise ValueError(f"{self.query_id} has no relevant items")
        if any(value <= 0 for value in self.relevance.values()):
            raise ValueError(f"{self.query_id} relevance weights must be positive")
        if self.relevance_denominator is not None and self.relevance_denominator <= 0:
            raise ValueError(f"{self.query_id} relevance denominator must be positive")

    @property
    def relevant_item_ids(self) -> frozenset[str]:
        return frozenset(self.relevance)


@dataclass(frozen=True)
class ReplayData:
    items: tuple[Item, ...]
    queries: tuple[Query, ...]
    scores: Mapping[str, Mapping[str, Mapping[str, float]]]

    @property
    def routes(self) -> tuple[str, ...]:
        return tuple(sorted(self.scores))

    def validate(self) -> None:
        if not self.routes:
            raise ValueError("score cube has no representation routes")
        item_ids = {item.item_id for item in self.items}
        if len(item_ids) != len(self.items):
            raise ValueError("item identifiers must be unique")
        query_ids = {query.query_id for query in self.queries}
        if len(query_ids) != len(self.queries):
            raise ValueError("query identifiers must be unique")
        for query in self.queries:
            unknown = query.relevant_item_ids - item_ids
            if unknown:
                raise ValueError(
                    f"{query.query_id} references unknown relevant items: {sorted(unknown)}"
                )
            if query.candidate_item_ids is not None:
                candidate_ids = set(query.candidate_item_ids)
                unknown_candidates = candidate_ids - item_ids
                if unknown_candidates:
                    raise ValueError(
                        f"{query.query_id} references unknown candidates: "
                        f"{sorted(unknown_candidates)[:5]}"
                    )
                if not query.relevant_item_ids <= candidate_ids:
                    raise ValueError(
                        f"{query.query_id} has relevant items outside its candidate pool"
                    )
        expected_routes = set(self.routes)
        for item in self.items:
            item_routes = set(item.route_costs)
            if item_routes != expected_routes:
                raise ValueError(
                    f"{item.item_id} route costs {sorted(item_routes)} do not "
                    f"match score routes {sorted(expected_routes)}"
                )
        for route in self.routes:
            for query in self.queries:
                route_query = self.scores[route].get(query.query_id)
                if route_query is None:
                    raise ValueError(f"missing {route} scores for query {query.query_id}")
                required_items = (
                    set(query.candidate_item_ids)
                    if query.candidate_item_ids is not None
                    else item_ids
                )
                missing_items = required_items - set(route_query)
                if missing_items:
                    raise ValueError(
                        f"missing {route} scores for {query.query_id}: "
                        f"{sorted(missing_items)[:5]}"
                    )


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_replay_data(items_path: Path, queries_path: Path, scores_path: Path) -> ReplayData:
    item_rows = _load_jsonl(items_path)
    items = tuple(
        Item(
            item_id=str(row["item_id"]),
            content_type=str(row["content_type"]).lower(),
            route_costs={
                route: RouteCost(
                    index_bytes=int(row["route_costs"][route]["index_bytes"]),
                    encode_ms=float(row["route_costs"][route]["encode_ms"]),
                )
                for route in sorted(row["route_costs"])
            },
        )
        for row in item_rows
    )
    queries = tuple(
        Query(
            query_id=str(row["query_id"]),
            relevance=(
                {
                    str(item_id): float(value)
                    for item_id, value in row["relevance"].items()
                }
                if "relevance" in row
                else {
                    str(item_id): 1.0 for item_id in row["relevant_item_ids"]
                }
            ),
            relevance_denominator=(
                float(row["relevance_denominator"])
                if row.get("relevance_denominator") is not None
                else None
            ),
            candidate_item_ids=(
                tuple(str(item_id) for item_id in row["candidate_item_ids"])
                if row.get("candidate_item_ids") is not None
                else None
            ),
        )
        for row in _load_jsonl(queries_path)
    )
    score_rows = _load_jsonl(scores_path)
    score_routes = sorted({str(row["route"]) for row in score_rows})
    scores: dict[str, dict[str, dict[str, float]]] = {
        route: {} for route in score_routes
    }
    for row in score_rows:
        route = str(row["route"])
        query_id = str(row["query_id"])
        scores[route].setdefault(query_id, {})[str(row["item_id"])] = float(row["score"])
    data = ReplayData(items=items, queries=queries, scores=scores)
    data.validate()
    return data


def uniform_plan(items: Sequence[Item], route: str) -> dict[str, str]:
    missing = [item.item_id for item in items if route not in item.route_costs]
    if missing:
        raise ValueError(
            f"route {route!r} is unavailable for items: {missing[:5]}"
        )
    return {item.item_id: route for item in items}


def fixed_hybrid_plan(
    items: Sequence[Item],
    *,
    image_route: str = IMAGE_ROUTE,
) -> dict[str, str]:
    """Reproduce MMDocIR's public layout-type rule.

    Tables and images use rendered-image embeddings. Other layouts use native
    text through the same visual retriever's language path. ``image_route``
    permits a budget-matched type rule using a registered compressed visual
    route instead of the full representation.
    """

    missing = [
        item.item_id
        for item in items
        if item.content_type in VISUAL_LAYOUT_TYPES
        and image_route not in item.route_costs
    ]
    if missing:
        raise ValueError(
            f"route {image_route!r} is unavailable for visual items: {missing[:5]}"
        )
    return {
        item.item_id: (
            image_route if item.content_type in VISUAL_LAYOUT_TYPES else TEXT_ROUTE
        )
        for item in items
    }


def typed_capacity_plan_v1(items: Sequence[Item]) -> dict[str, str]:
    """Frozen mechanism-derived ReprForge V1 policy.

    Tables retain full visual capacity because text/strong compression loses
    relevant evidence. Other visual regions keep the strong 9x pooled base.
    Predominantly textual and remaining region types use 25x pooling, which
    reduces distractor strength without crossing into the lossy text substrate.
    """

    routes = {
        item.item_id: (
            IMAGE_ROUTE
            if item.content_type == "table"
            else (
                "image-pool-9"
                if item.content_type in {"chart", "figure", "image"}
                else "image-pool-25"
            )
        )
        for item in items
    }
    missing = [
        item.item_id
        for item in items
        if routes[item.item_id] not in item.route_costs
    ]
    if missing:
        raise ValueError(
            f"typed capacity routes are unavailable for items: {missing[:5]}"
        )
    return routes


def plan_cost(items: Sequence[Item], plan: Mapping[str, str]) -> dict:
    index_bytes = 0
    encode_ms = 0.0
    route_counts = {
        route: 0
        for route in sorted(
            {route for item in items for route in item.route_costs}
        )
    }
    for item in items:
        route = plan.get(item.item_id)
        if route not in item.route_costs:
            raise ValueError(f"plan has no valid route for {item.item_id}")
        cost = item.route_costs[route]
        index_bytes += cost.index_bytes
        encode_ms += cost.encode_ms
        route_counts[route] += 1
    return {
        "offline_index_bytes": index_bytes,
        "offline_encode_ms": encode_ms,
        "route_counts": route_counts,
    }


def random_plan_under_budget(
    items: Sequence[Item],
    *,
    index_budget_bytes: int,
    seed: int,
) -> dict[str, str]:
    """Upgrade a deterministic random subset from text to image under budget."""

    plan = uniform_plan(items, TEXT_ROUTE)
    base_cost = plan_cost(items, plan)["offline_index_bytes"]
    if base_cost > index_budget_bytes:
        raise ValueError("index budget is smaller than the all-text index")
    current = base_cost
    order = list(items)
    random.Random(seed).shuffle(order)
    for item in order:
        delta = (
            item.route_costs[IMAGE_ROUTE].index_bytes
            - item.route_costs[TEXT_ROUTE].index_bytes
        )
        if delta <= 0 or current + delta <= index_budget_bytes:
            plan[item.item_id] = IMAGE_ROUTE
            current += delta
    return plan


def _percentile_scores(values: Mapping[str, float]) -> dict[str, float]:
    """Rank-normalize one route for one query to [0, 1].

    This is available for cross-model experiments. The primary MMDocIR
    text/image comparison should use ``none`` because both routes are emitted
    by the same retriever and are intended to share a score space.
    """

    ranked = sorted(values.items(), key=lambda pair: (pair[1], pair[0]))
    if len(ranked) == 1:
        return {ranked[0][0]: 1.0}
    return {
        item_id: rank / (len(ranked) - 1)
        for rank, (item_id, _) in enumerate(ranked)
    }


def calibrated_scores(
    data: ReplayData,
    calibration: str,
) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    if calibration == "none":
        return data.scores
    if calibration != "percentile":
        raise ValueError(f"unsupported calibration: {calibration}")
    return {
        route: {
            query.query_id: _percentile_scores(data.scores[route][query.query_id])
            for query in data.queries
        }
        for route in data.routes
    }


def _ndcg(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    dcg = sum(
        float(relevance.get(item_id, 0.0)) / math.log2(rank + 1)
        for rank, item_id in enumerate(ranked[:k], start=1)
        if item_id in relevance
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate_plan(
    data: ReplayData,
    plan: Mapping[str, str],
    *,
    ks: Sequence[int] = (1, 5, 10),
    calibration: str = "none",
) -> dict:
    data.validate()
    score_cube = calibrated_scores(data, calibration)
    all_item_ids = [item.item_id for item in data.items]
    recalls = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    per_query: dict[str, dict] = {}
    for query in data.queries:
        item_ids = (
            list(query.candidate_item_ids)
            if query.candidate_item_ids is not None
            else all_item_ids
        )
        ranked = sorted(
            item_ids,
            key=lambda item_id: (
                -score_cube[plan[item_id]][query.query_id][item_id],
                item_id,
            ),
        )
        query_result: dict[str, float] = {}
        for k in ks:
            retrieved_relevance = sum(
                query.relevance.get(item_id, 0.0) for item_id in ranked[:k]
            )
            denominator = (
                query.relevance_denominator
                if query.relevance_denominator is not None
                else sum(query.relevance.values())
            )
            recall = retrieved_relevance / denominator
            ndcg = _ndcg(ranked, query.relevance, k)
            recalls[k] += recall
            ndcgs[k] += ndcg
            query_result[f"recall_at_{k}"] = recall
            query_result[f"ndcg_at_{k}"] = ndcg
        per_query[query.query_id] = query_result
    count = len(data.queries)
    result = {
        **{f"recall_at_{k}": recalls[k] / count for k in ks},
        **{f"ndcg_at_{k}": ndcgs[k] / count for k in ks},
        "queries": count,
        "calibration": calibration,
        "cost": plan_cost(data.items, plan),
        "per_query": per_query,
    }
    return result


def exact_budget_oracle(
    data: ReplayData,
    *,
    index_budget_bytes: int,
    target_metric: str = "recall_at_5",
    calibration: str = "none",
    max_items: int = 20,
) -> tuple[dict[str, str], dict]:
    """Enumerate every binary plan for a small problem and return the exact best.

    This is deliberately bounded. It is suitable for unit tests and small,
    stratified oracle slices; it must not be mislabeled as a scalable policy.
    """

    if len(data.items) > max_items:
        raise ValueError(
            f"exact oracle supports at most {max_items} items, got {len(data.items)}"
        )
    best_plan: dict[str, str] | None = None
    best_result: dict | None = None
    best_value = float("-inf")
    best_cost = math.inf
    for choices in itertools.product(data.routes, repeat=len(data.items)):
        plan = {
            item.item_id: route for item, route in zip(data.items, choices, strict=True)
        }
        cost = plan_cost(data.items, plan)["offline_index_bytes"]
        if cost > index_budget_bytes:
            continue
        result = evaluate_plan(data, plan, calibration=calibration)
        value = float(result[target_metric])
        if value > best_value or (value == best_value and cost < best_cost):
            best_plan = plan
            best_result = result
            best_value = value
            best_cost = cost
    if best_plan is None or best_result is None:
        raise ValueError("no plan fits the index budget")
    best_result["oracle"] = {
        "exact": True,
        "enumerated_items": len(data.items),
        "target_metric": target_metric,
    }
    return best_plan, best_result


def write_result(path: Path, policy: str, plan: Mapping[str, str], metrics: dict) -> None:
    payload = {
        "policy": policy,
        "plan": dict(sorted(plan.items())),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("all-text", "all-image", "fixed-hybrid", "random", "exact-oracle"),
        required=True,
    )
    parser.add_argument("--index-budget-bytes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-metric", default="recall_at_5")
    parser.add_argument("--calibration", choices=("none", "percentile"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load_replay_data(args.items, args.queries, args.scores)
    if args.policy == "all-text":
        plan = uniform_plan(data.items, TEXT_ROUTE)
        metrics = evaluate_plan(data, plan, calibration=args.calibration)
    elif args.policy == "all-image":
        plan = uniform_plan(data.items, IMAGE_ROUTE)
        metrics = evaluate_plan(data, plan, calibration=args.calibration)
    elif args.policy == "fixed-hybrid":
        plan = fixed_hybrid_plan(data.items)
        metrics = evaluate_plan(data, plan, calibration=args.calibration)
    elif args.policy == "random":
        if args.index_budget_bytes is None:
            parser.error("--index-budget-bytes is required for random")
        plan = random_plan_under_budget(
            data.items,
            index_budget_bytes=args.index_budget_bytes,
            seed=args.seed,
        )
        metrics = evaluate_plan(data, plan, calibration=args.calibration)
    else:
        if args.index_budget_bytes is None:
            parser.error("--index-budget-bytes is required for exact-oracle")
        plan, metrics = exact_budget_oracle(
            data,
            index_budget_bytes=args.index_budget_bytes,
            target_metric=args.target_metric,
            calibration=args.calibration,
        )
    write_result(args.output, args.policy, plan, metrics)
    print(
        json.dumps(
            {
                "policy": args.policy,
                args.target_metric: metrics.get(args.target_metric),
                "cost": metrics["cost"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
