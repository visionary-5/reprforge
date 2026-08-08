"""Cumulative feature-materialization replay with exact quality semantics."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .costs import CostCatalog


def replay_feature_policy(
    candidates: np.ndarray,
    query_order: Sequence[int],
    costs: CostCatalog,
    *,
    capacity_pages: int,
    policy: str,
    initial_pages: Sequence[int] = (),
) -> dict[str, object]:
    costs.validate()
    matrix = np.asarray(candidates, dtype=np.int32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("candidates must have shape [queries, depth]")
    order = np.asarray(list(map(int, query_order)), dtype=np.int32)
    if order.size == 0 or order.min() < 0 or order.max() >= matrix.shape[0]:
        raise ValueError("query order is empty or outside candidate matrix")
    if capacity_pages < 0:
        raise ValueError("capacity_pages must be nonnegative")
    supported = {"never", "static", "first_touch", "second_touch"}
    if policy not in supported:
        raise ValueError(f"unsupported feature policy: {policy}")

    resident = set(map(int, initial_pages))
    if len(resident) > capacity_pages:
        raise ValueError("initial feature pages exceed capacity")
    if resident and min(resident) < 0:
        raise ValueError("initial feature pages must be nonnegative")
    touches: dict[int, int] = {}
    cumulative = len(resident) * costs.offline_feature_cost
    initial_seconds = cumulative
    cumulative_rows: list[float] = []
    per_query_seconds: list[float] = []
    hits = 0
    events = 0
    promotions = 0

    threshold = 1 if policy == "first_touch" else 2
    for query in order:
        pages = list(dict.fromkeys(map(int, matrix[int(query)])))
        query_seconds = 0.0
        missing: list[int] = []
        for page in pages:
            events += 1
            if page in resident:
                hits += 1
                query_seconds += costs.feature_query_seconds
            else:
                query_seconds += costs.raw_query_seconds
                missing.append(page)
            touches[page] = touches.get(page, 0) + 1
        if policy in {"first_touch", "second_touch"}:
            for page in missing:
                if len(resident) >= capacity_pages:
                    break
                if touches[page] >= threshold and page not in resident:
                    resident.add(page)
                    promotions += 1
                    query_seconds += costs.feature_write_seconds
        cumulative += query_seconds
        per_query_seconds.append(query_seconds)
        cumulative_rows.append(cumulative)

    values = np.asarray(per_query_seconds, dtype=np.float64)
    return {
        "policy": policy,
        "queries": int(len(order)),
        "candidate_events": int(events),
        "capacity_pages": int(capacity_pages),
        "initial_materialized_pages": int(len(set(map(int, initial_pages)))),
        "final_materialized_pages": int(len(resident)),
        "online_promotions": int(promotions),
        "initial_materialization_seconds": float(initial_seconds),
        "total_seconds": float(cumulative),
        "mean_query_seconds": float(values.mean()),
        "p95_query_seconds": float(np.percentile(values, 95)),
        "p99_query_seconds": float(np.percentile(values, 99)),
        "feature_hit_fraction": float(hits / events) if events else 0.0,
        "cumulative_seconds": cumulative_rows,
        "quality_invariant_to_feature_storage": True,
        "online_promotion_cost_scope": "write/fsync after raw forward exposes reusable visual features",
    }
