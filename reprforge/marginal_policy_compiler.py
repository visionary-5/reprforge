"""Counterfactual listwise value compiler for heterogeneous index states."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube, paired_bootstrap_ci
from reprforge.type_policy_compiler import _candidate_ndcg


def single_switch_gains(
    cube: ScoreCube,
    *,
    base_route: str,
    target_k: int = 5,
) -> dict[str, np.ndarray]:
    """Return exact per-query nDCG gains for one item changing route.

    Each returned array is query-by-item aligned.  Only one item's score is
    changed at a time, so the exact new top-k can be recovered from the base
    top-(k+1) plus that item.
    """

    cube.validate()
    if base_route not in cube.routes:
        raise ValueError("base route is unavailable")
    query_count = len(cube.query_ids)
    item_count = len(cube.corpus_ids)
    positions = np.arange(item_count)
    base = cube.scores[base_route]
    base_orders = np.stack(
        [
            np.lexsort((positions, -base[query_index]))[: target_k + 1]
            for query_index in range(query_count)
        ]
    )
    discounts = 1.0 / np.log2(np.arange(2, target_k + 2))
    denominators = np.asarray(
        [
            np.dot(
                np.asarray(sorted(values.values(), reverse=True)[:target_k]),
                discounts[: min(target_k, len(values))],
            )
            if values
            else 0.0
            for values in cube.relevance
        ],
        dtype=np.float64,
    )

    def quality(query_index: int, order: np.ndarray) -> float:
        denominator = denominators[query_index]
        if denominator <= 0.0:
            return 0.0
        relevance = cube.relevance[query_index]
        values = np.fromiter(
            (relevance.get(int(item), 0.0) for item in order),
            dtype=np.float64,
            count=len(order),
        )
        return float(np.dot(values, discounts[: len(order)]) / denominator)

    baseline = np.asarray(
        [quality(query_index, base_orders[query_index, :target_k]) for query_index in range(query_count)]
    )
    output = {}
    for route in cube.routes:
        if route == base_route:
            continue
        gains = np.zeros((query_count, item_count), dtype=np.float32)
        alternative = cube.scores[route]
        for query_index in range(query_count):
            shortlist = base_orders[query_index]
            for item in range(item_count):
                pool = shortlist[shortlist != item][:target_k].tolist()
                pool.append(item)
                order = np.asarray(
                    sorted(
                        pool,
                        key=lambda candidate: (
                            -(
                                alternative[query_index, candidate]
                                if candidate == item
                                else base[query_index, candidate]
                            ),
                            candidate,
                        ),
                    )[:target_k],
                    dtype=np.int32,
                )
                gains[query_index, item] = quality(query_index, order) - baseline[
                    query_index
                ]
        output[route] = gains
    return output


def _compile_marginal_plan(
    *,
    base_route: str,
    gains: Mapping[str, np.ndarray],
    fit_mask: np.ndarray,
    route_costs: Mapping[str, np.ndarray],
    budget: float,
    risk_z: float,
) -> tuple[tuple[str, ...], dict]:
    item_count = len(route_costs[base_route])
    base_cost = float(np.sum(route_costs[base_route]))
    if base_cost > budget + 1e-6:
        raise ValueError("base plan exceeds budget")
    candidates = []
    for route, per_query in gains.items():
        deltas = route_costs[route] - route_costs[base_route]
        fit_values = per_query[fit_mask]
        means = fit_values.mean(axis=0)
        standard_errors = fit_values.std(axis=0, ddof=1) / np.sqrt(fit_mask.sum())
        utilities = means - risk_z * standard_errors
        for item, (utility, cost) in enumerate(zip(utilities, deltas, strict=True)):
            # This pass measures the conservative headroom from spending more
            # than the uniform base.  Cost-decreasing substitutions require a
            # multiple-choice solver because they can finance later switches.
            if utility > 0.0 and cost > 0.0:
                candidates.append(
                    (float(utility / cost), float(utility), -float(cost), -item, route)
                )
    candidates.sort(reverse=True)
    plan = [base_route] * item_count
    used = base_cost
    selected_utility = 0.0
    for _, utility, negative_cost, negative_item, route in candidates:
        item = -negative_item
        if plan[item] != base_route:
            continue
        cost = -negative_cost
        if used + cost > budget + 1e-6:
            continue
        plan[item] = route
        used += cost
        selected_utility += utility
    return tuple(plan), {
        "cost": used,
        "fit_single_switch_utility_sum": selected_utility,
        "route_counts": dict(sorted(Counter(plan).items())),
    }


def crossfit_marginal_policy_compiler(
    cube: ScoreCube,
    *,
    base_route: str,
    route_costs: Mapping[str, Sequence[float]],
    fold_ids: Sequence[int],
    budget_fractions: Sequence[float],
    teacher_route: str = "image",
    target_k: int = 5,
    risk_z: float = 0.0,
) -> dict:
    """Compile on past queries and replay the static plan on unseen queries."""

    costs = {
        route: np.asarray(values, dtype=np.float64)
        for route, values in route_costs.items()
    }
    item_count = len(cube.corpus_ids)
    if set(costs) != set(cube.routes) or any(
        values.shape != (item_count,) for values in costs.values()
    ):
        raise ValueError("route costs must be corpus-aligned and cover all routes")
    folds = np.asarray(fold_ids, dtype=np.int16)
    if folds.shape != (len(cube.query_ids),) or len(set(folds.tolist())) < 2:
        raise ValueError("fold IDs must be query-aligned and non-degenerate")
    if risk_z < 0.0:
        raise ValueError("risk_z must be non-negative")
    gains = single_switch_gains(cube, base_route=base_route, target_k=target_k)
    all_candidates = (np.arange(item_count, dtype=np.int32),) * len(cube.query_ids)
    uniform_values = {
        route: _candidate_ndcg(
            cube,
            (route,) * item_count,
            all_candidates,
            target_k=target_k,
        )
        for route in cube.routes
    }
    teacher_cost = float(np.sum(costs[teacher_route]))
    reports = {}
    for fraction in budget_fractions:
        budget = teacher_cost * float(fraction)
        if float(np.sum(costs[base_route])) > budget + 1e-6:
            reports[str(fraction)] = {"feasible": False}
            continue
        predictions = np.zeros(len(cube.query_ids), dtype=np.float64)
        selected = []
        for fold in sorted(set(folds.tolist())):
            fit = folds != fold
            test = folds == fold
            plan, details = _compile_marginal_plan(
                base_route=base_route,
                gains=gains,
                fit_mask=fit,
                route_costs=costs,
                budget=budget,
                risk_z=risk_z,
            )
            values = _candidate_ndcg(
                cube, plan, all_candidates, target_k=target_k
            )
            predictions[test] = values[test]
            selected.append(
                {
                    "fold": int(fold),
                    "test_queries": int(test.sum()),
                    "cost_fraction": details["cost"] / teacher_cost,
                    "fit_ndcg_at_5": float(values[fit].mean()),
                    "test_ndcg_at_5": float(values[test].mean()),
                    "fit_single_switch_utility_sum": details[
                        "fit_single_switch_utility_sum"
                    ],
                    "route_counts": details["route_counts"],
                }
            )
        reports[str(fraction)] = {
            "feasible": True,
            "crossfit_ndcg_at_5": float(predictions.mean()),
            "mean_cost_fraction": float(
                np.mean([value["cost_fraction"] for value in selected])
            ),
            "per_query_ndcg_at_5": predictions.tolist(),
            "vs_uniform_image": paired_bootstrap_ci(
                predictions, uniform_values[teacher_route]
            ),
            "vs_uniform_image_pool_9": paired_bootstrap_ci(
                predictions, uniform_values[base_route]
            ),
            "selected_by_fold": selected,
        }
    return {
        "protocol": "single-switch listwise marginal value; held-out-query crossfit",
        "compiler_uses_fit_qrels": True,
        "base_route": base_route,
        "risk_z": risk_z,
        "routes": cube.routes,
        "folds": len(set(folds.tolist())),
        "uniform_ndcg_at_5": {
            route: float(values.mean()) for route, values in uniform_values.items()
        },
        "budget_curve": reports,
    }
