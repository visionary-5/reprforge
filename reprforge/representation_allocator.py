#!/usr/bin/env python3
"""Deterministic multiple-choice allocation for heterogeneous representations."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RouteOption:
    item_id: str
    route: str
    cost_bytes: int
    predicted_utility: float

    def __post_init__(self) -> None:
        if not self.item_id or not self.route:
            raise ValueError("item and route identifiers must be non-empty")
        if self.cost_bytes < 0:
            raise ValueError("route cost must be non-negative")
        if not math.isfinite(self.predicted_utility):
            raise ValueError("predicted utility must be finite")


@dataclass(frozen=True)
class Allocation:
    plan: Mapping[str, str]
    total_cost_bytes: int
    predicted_utility: float
    budget_bytes: int
    cost_quantum_bytes: int
    capacity_units: int
    quantized: bool


def _group_options(options: Sequence[RouteOption]) -> list[tuple[str, list[RouteOption]]]:
    grouped: dict[str, dict[str, RouteOption]] = {}
    for option in options:
        routes = grouped.setdefault(option.item_id, {})
        if option.route in routes:
            raise ValueError(
                f"duplicate route {option.route!r} for item {option.item_id!r}"
            )
        routes[option.route] = option
    if not grouped:
        raise ValueError("allocator requires at least one item")
    return [
        (
            item_id,
            sorted(
                routes.values(),
                key=lambda option: (
                    option.cost_bytes,
                    option.route,
                ),
            ),
        )
        for item_id, routes in sorted(grouped.items())
    ]


def allocate_routes(
    options: Sequence[RouteOption],
    *,
    budget_bytes: int,
    cost_quantum_bytes: int = 512,
    max_capacity_units: int = 100_000,
) -> Allocation:
    """Solve a quantized multiple-choice knapsack with a hard byte guarantee.

    The cheapest route for each item forms the mandatory base plan. Alternative
    route costs are rounded *up* to a byte quantum before dynamic programming.
    Consequently every returned plan is within the real byte budget. Quantized
    rounding can leave budget unused, so the result is exact only when the
    effective quantum is one byte.
    """

    if budget_bytes < 0:
        raise ValueError("budget must be non-negative")
    if cost_quantum_bytes <= 0:
        raise ValueError("cost quantum must be positive")
    if max_capacity_units <= 0:
        raise ValueError("max capacity units must be positive")

    grouped = _group_options(options)
    base_options = [routes[0] for _, routes in grouped]
    base_cost = sum(option.cost_bytes for option in base_options)
    if base_cost > budget_bytes:
        raise ValueError(
            f"minimum representation cost {base_cost} exceeds budget {budget_bytes}"
        )
    remaining_bytes = budget_bytes - base_cost
    effective_quantum = max(
        cost_quantum_bytes,
        math.ceil(remaining_bytes / max_capacity_units)
        if remaining_bytes
        else cost_quantum_bytes,
    )
    capacity = remaining_bytes // effective_quantum

    # Utilities are represented relative to the mandatory cheapest route.
    dp = np.full(capacity + 1, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    choice_rows: list[np.ndarray] = []
    delta_units_by_item: list[list[int]] = []
    route_options_by_item: list[list[RouteOption]] = []

    for (_, routes), base in zip(grouped, base_options, strict=True):
        deltas = [
            math.ceil((option.cost_bytes - base.cost_bytes) / effective_quantum)
            for option in routes
        ]
        utility_deltas = [
            option.predicted_utility - base.predicted_utility
            for option in routes
        ]
        next_dp = np.full_like(dp, -np.inf)
        choices = np.zeros(capacity + 1, dtype=np.uint16)
        for route_index, (delta, utility_delta) in enumerate(
            zip(deltas, utility_deltas, strict=True)
        ):
            if delta > capacity:
                continue
            source = dp[: capacity + 1 - delta]
            candidate = source + utility_delta
            target = next_dp[delta:]
            # Strict improvement preserves the deterministic earlier-route tie
            # break established by cost then route name.
            improved = candidate > target
            target[improved] = candidate[improved]
            choices[delta:][improved] = route_index
        dp = next_dp
        choice_rows.append(choices)
        delta_units_by_item.append(deltas)
        route_options_by_item.append(routes)

    best_capacity = int(np.argmax(dp))
    if not np.isfinite(dp[best_capacity]):
        raise RuntimeError("no feasible allocation survived dynamic programming")
    plan: dict[str, str] = {}
    capacity_cursor = best_capacity
    selected: list[RouteOption] = []
    for item_index in range(len(grouped) - 1, -1, -1):
        route_index = int(choice_rows[item_index][capacity_cursor])
        option = route_options_by_item[item_index][route_index]
        plan[option.item_id] = option.route
        selected.append(option)
        capacity_cursor -= delta_units_by_item[item_index][route_index]
    if capacity_cursor != 0:
        raise RuntimeError("allocation reconstruction did not return to base state")

    total_cost = sum(option.cost_bytes for option in selected)
    if total_cost > budget_bytes:
        raise AssertionError("quantized allocator exceeded the real byte budget")
    return Allocation(
        plan=dict(sorted(plan.items())),
        total_cost_bytes=total_cost,
        predicted_utility=sum(option.predicted_utility for option in selected),
        budget_bytes=budget_bytes,
        cost_quantum_bytes=effective_quantum,
        capacity_units=capacity,
        quantized=effective_quantum != 1,
    )


def _load_options(path: Path) -> list[RouteOption]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        RouteOption(
            item_id=str(row["item_id"]),
            route=str(row["route"]),
            cost_bytes=int(row["cost_bytes"]),
            predicted_utility=float(row["predicted_utility"]),
        )
        for row in payload
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--cost-quantum-bytes", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    allocation = allocate_routes(
        _load_options(args.options),
        budget_bytes=args.budget_bytes,
        cost_quantum_bytes=args.cost_quantum_bytes,
    )
    payload = {
        "plan": allocation.plan,
        "total_cost_bytes": allocation.total_cost_bytes,
        "predicted_utility": allocation.predicted_utility,
        "budget_bytes": allocation.budget_bytes,
        "cost_quantum_bytes": allocation.cost_quantum_bytes,
        "capacity_units": allocation.capacity_units,
        "quantized": allocation.quantized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "plan"}))


if __name__ == "__main__":
    main()
