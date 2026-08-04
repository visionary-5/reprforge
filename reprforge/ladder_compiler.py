"""Qrel-free error-bounded compilation over a representation ladder."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube, query_metrics


def _mixed_surface(
    cube: ScoreCube,
    route_by_item: Sequence[str],
    candidate_indices: Sequence[Sequence[int]],
) -> np.ndarray:
    """Compose one physical route per item while respecting candidate scope."""

    routes = np.asarray(route_by_item, dtype=object)
    if routes.shape != (len(cube.corpus_ids),):
        raise ValueError("route plan must be corpus-aligned")
    unknown = set(routes) - set(cube.routes)
    if unknown:
        raise ValueError(f"route plan contains unknown routes: {sorted(unknown)}")
    surface = np.full(
        (len(cube.query_ids), len(cube.corpus_ids)), -np.inf, dtype=np.float64
    )
    for query_index, candidates_value in enumerate(candidate_indices):
        candidates = np.asarray(candidates_value, dtype=np.int32)
        for route in cube.routes:
            selected = candidates[routes[candidates] == route]
            surface[query_index, selected] = cube.scores[route][query_index, selected]
        if not np.isfinite(surface[query_index, candidates]).all():
            raise ValueError("candidate plan produced non-finite scores")
        floor = float(surface[query_index, candidates].min())
        surface[query_index, ~np.isfinite(surface[query_index])] = floor - max(
            abs(floor), 1.0
        )
    return surface


def _teacher_fidelity(
    plan: np.ndarray,
    teacher: np.ndarray,
    candidate_indices: Sequence[Sequence[int]],
    *,
    target_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(plan.shape[1])
    overlap = np.zeros(plan.shape[0], dtype=np.float64)
    exact_position = np.zeros(plan.shape[0], dtype=np.float64)
    for query_index, candidates_value in enumerate(candidate_indices):
        candidates = np.asarray(candidates_value, dtype=np.int32)
        plan_order = candidates[
            np.lexsort((positions[candidates], -plan[query_index, candidates]))
        ][:target_k]
        teacher_order = candidates[
            np.lexsort((positions[candidates], -teacher[query_index, candidates]))
        ][:target_k]
        overlap[query_index] = len(set(plan_order) & set(teacher_order)) / target_k
        exact_position[query_index] = np.mean(plan_order == teacher_order)
    return overlap, exact_position


def compile_error_bounded_plan(
    cube: ScoreCube,
    *,
    teacher_route: str,
    candidate_indices: Sequence[Sequence[int]],
    route_costs: Mapping[str, Sequence[float]],
    error_budget: float,
    fit_quantile: float = 1.0,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Choose the cheapest item state under a workload score-error bound.

    Error is the absolute route--teacher score residual divided by the teacher
    score standard deviation inside that query's candidate set.  The bound for
    one item/route is a fit-workload quantile over queries containing the item.
    Full teacher scores and qrels are not needed at inference; qrels are never
    used by the compiler.
    """

    cube.validate()
    if teacher_route not in cube.routes:
        raise ValueError("teacher route is unavailable")
    if not 0.0 < fit_quantile <= 1.0 or error_budget < 0.0:
        raise ValueError("invalid fit quantile or error budget")
    item_count = len(cube.corpus_ids)
    if set(route_costs) != set(cube.routes):
        raise ValueError("route costs must cover the score cube routes")
    costs = {
        route: np.asarray(values, dtype=np.float64) for route, values in route_costs.items()
    }
    if any(values.shape != (item_count,) for values in costs.values()):
        raise ValueError("route costs must be corpus-aligned")
    fit = np.flatnonzero(np.asarray(cube.split_roles) == "fit")
    observations: list[list[list[float]]] = [
        [[] for _ in cube.routes] for _ in range(item_count)
    ]
    route_position = {route: index for index, route in enumerate(cube.routes)}
    teacher = cube.scores[teacher_route]
    for query_index in fit:
        candidates = np.asarray(candidate_indices[query_index], dtype=np.int32)
        scale = max(float(teacher[query_index, candidates].std()), 1e-6)
        for route in cube.routes:
            residuals = np.abs(
                cube.scores[route][query_index, candidates]
                - teacher[query_index, candidates]
            ) / scale
            column = route_position[route]
            for item, residual in zip(candidates, residuals, strict=True):
                observations[int(item)][column].append(float(residual))
    bounds = np.full((item_count, len(cube.routes)), np.inf, dtype=np.float64)
    bounds[:, route_position[teacher_route]] = 0.0
    plan = []
    for item in range(item_count):
        for route, column in route_position.items():
            if observations[item][column]:
                bounds[item, column] = float(
                    np.quantile(observations[item][column], fit_quantile)
                )
        feasible = [
            route for route in cube.routes
            if bounds[item, route_position[route]] <= error_budget
        ]
        plan.append(min(feasible, key=lambda route: (costs[route][item], route)))
    return tuple(plan), bounds


def analyze_error_bounded_ladder(
    cube: ScoreCube,
    *,
    teacher_route: str,
    candidate_indices: Sequence[Sequence[int]],
    route_costs: Mapping[str, Sequence[float]],
    error_budgets: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6),
    fit_quantile: float = 1.0,
    target_k: int = 5,
    target_metric: str = "ndcg_at_5",
    reference_plans: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Compile an error-budget curve and evaluate it on held-out queries."""

    eval_indices = np.flatnonzero(np.asarray(cube.split_roles) == "eval")
    teacher_plan = (teacher_route,) * len(cube.corpus_ids)
    teacher_surface = _mixed_surface(cube, teacher_plan, candidate_indices)
    quality = query_metrics(teacher_surface, cube.relevance, ks=(target_k,))[
        target_metric
    ]
    teacher_cost = float(np.sum(route_costs[teacher_route]))

    def evaluate(plan: Sequence[str]) -> dict:
        surface = _mixed_surface(cube, plan, candidate_indices)
        values = query_metrics(surface, cube.relevance, ks=(target_k,))[target_metric]
        overlap, exact_position = _teacher_fidelity(
            surface, teacher_surface, candidate_indices, target_k=target_k
        )
        total_cost = float(
            sum(route_costs[route][item] for item, route in enumerate(plan))
        )
        return {
            "quality": float(values[eval_indices].mean()),
            "teacher_quality": float(quality[eval_indices].mean()),
            "mean_topk_overlap": float(overlap[eval_indices].mean()),
            "mean_exact_position_agreement": float(
                exact_position[eval_indices].mean()
            ),
            "cost": total_cost,
            "teacher_cost": teacher_cost,
            "cost_fraction": total_cost / teacher_cost,
            "route_counts": dict(sorted(Counter(plan).items())),
        }

    baselines = {
        f"uniform-{route}": evaluate((route,) * len(cube.corpus_ids))
        for route in cube.routes
    }
    for name, plan in (reference_plans or {}).items():
        baselines[name] = evaluate(plan)
    reports = {}
    for budget in error_budgets:
        plan, bounds = compile_error_bounded_plan(
            cube,
            teacher_route=teacher_route,
            candidate_indices=candidate_indices,
            route_costs=route_costs,
            error_budget=float(budget),
            fit_quantile=fit_quantile,
        )
        reports[str(budget)] = evaluate(plan)
        reports[str(budget)]["finite_bound_fraction"] = float(
            np.isfinite(bounds).mean()
        )
    return {
        "protocol": "fit-workload residual bound; held-out-query evaluation",
        "compiler_uses_qrels": False,
        "teacher_route": teacher_route,
        "fit_quantile": fit_quantile,
        "fit_queries": int(len(cube.query_ids) - len(eval_indices)),
        "eval_queries": int(len(eval_indices)),
        "baselines": baselines,
        "cost_curve": reports,
    }
