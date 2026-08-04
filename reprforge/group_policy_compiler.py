"""Cost-regularized coordinate search over observable item groups."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube
from reprforge.type_policy_compiler import _candidate_ndcg


def _plan_for_mapping(
    groups: np.ndarray, mapping: Mapping[str, str]
) -> tuple[str, ...]:
    return tuple(mapping[str(group)] for group in groups)


def fit_group_policy(
    cube: ScoreCube,
    *,
    item_groups: Sequence[str],
    candidate_indices: Sequence[Sequence[int]],
    route_costs: Mapping[str, Sequence[float]],
    fit_mask: Sequence[bool],
    cost_penalty: float,
    teacher_route: str = "image",
    initial_mappings: Sequence[Mapping[str, str]] = (),
    target_k: int = 5,
    max_passes: int = 4,
) -> dict:
    """Coordinate-optimize listwise fit quality minus normalized index cost."""

    groups = np.asarray(item_groups, dtype=object)
    unique_groups = tuple(sorted(set(str(value) for value in groups)))
    fit = np.asarray(fit_mask, dtype=bool)
    teacher_cost = float(np.sum(route_costs[teacher_route]))
    costs = {
        route: np.asarray(values, dtype=np.float64) for route, values in route_costs.items()
    }

    def evaluate(mapping: Mapping[str, str]) -> tuple[float, float, np.ndarray, float]:
        plan = _plan_for_mapping(groups, mapping)
        values = _candidate_ndcg(
            cube, plan, candidate_indices, target_k=target_k
        )
        cost = float(sum(costs[route][item] for item, route in enumerate(plan)))
        objective = float(values[fit].mean()) - cost_penalty * cost / teacher_cost
        return objective, cost, values, float(values[fit].mean())

    starts = [
        {group: route for group in unique_groups} for route in cube.routes
    ]
    starts.extend(
        {group: str(mapping[group]) for group in unique_groups}
        for mapping in initial_mappings
    )
    best_result = None
    for start in starts:
        mapping = dict(start)
        objective, cost, values, fit_quality = evaluate(mapping)
        passes = 0
        for passes in range(1, max_passes + 1):
            changed = False
            for group in unique_groups:
                candidates = []
                for route in cube.routes:
                    proposal = dict(mapping)
                    proposal[group] = route
                    result = evaluate(proposal)
                    candidates.append((result[0], -result[1], route, proposal, result))
                _, _, _, proposal, result = max(candidates, key=lambda value: value[:3])
                if proposal[group] != mapping[group]:
                    mapping = proposal
                    objective, cost, values, fit_quality = result
                    changed = True
            if not changed:
                break
        result = {
            "mapping": mapping,
            "objective": objective,
            "cost": cost,
            "cost_fraction": cost / teacher_cost,
            "per_query_ndcg": values,
            "fit_ndcg_at_5": fit_quality,
            "passes": passes,
            "route_counts": dict(
                sorted(Counter(_plan_for_mapping(groups, mapping)).items())
            ),
        }
        if best_result is None or (
            result["objective"], -result["cost"], tuple(sorted(result["mapping"].items()))
        ) > (
            best_result["objective"],
            -best_result["cost"],
            tuple(sorted(best_result["mapping"].items())),
        ):
            best_result = result
    assert best_result is not None
    return best_result


def crossfit_group_policy_compiler(
    cube: ScoreCube,
    *,
    item_groups: Sequence[str],
    candidate_indices: Sequence[Sequence[int]],
    route_costs: Mapping[str, Sequence[float]],
    fold_ids: Sequence[int],
    cost_penalties: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.5),
    initial_mappings: Sequence[Mapping[str, str]] = (),
) -> dict:
    folds = np.asarray(fold_ids, dtype=np.int16)
    reports = {}
    for penalty in cost_penalties:
        predictions = np.zeros(len(cube.query_ids), dtype=np.float64)
        selected = []
        for fold in sorted(set(folds.tolist())):
            fit = folds != fold
            test = folds == fold
            result = fit_group_policy(
                cube,
                item_groups=item_groups,
                candidate_indices=candidate_indices,
                route_costs=route_costs,
                fit_mask=fit,
                cost_penalty=float(penalty),
                initial_mappings=initial_mappings,
            )
            predictions[test] = result["per_query_ndcg"][test]
            selected.append(
                {
                    "fold": int(fold),
                    "test_queries": int(test.sum()),
                    "fit_ndcg_at_5": result["fit_ndcg_at_5"],
                    "test_ndcg_at_5": float(result["per_query_ndcg"][test].mean()),
                    "cost_fraction": result["cost_fraction"],
                    "route_counts": result["route_counts"],
                    "mapping": result["mapping"],
                }
            )
        reports[str(penalty)] = {
            "crossfit_ndcg_at_5": float(predictions.mean()),
            "mean_cost_fraction": float(
                np.mean([record["cost_fraction"] for record in selected])
            ),
            "selected_by_fold": selected,
        }
    return {
        "protocol": "source-document cross-fit; listwise qrel objective minus index cost",
        "compiler_uses_fit_qrels": True,
        "groups": len(set(str(value) for value in item_groups)),
        "folds": len(set(folds.tolist())),
        "penalty_curve": reports,
    }
