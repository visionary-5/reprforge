"""Exact low-cardinality type-policy search for ladder physical design."""

from __future__ import annotations

import itertools
from typing import Mapping, Sequence

import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube, paired_bootstrap_ci


def _candidate_ndcg(
    cube: ScoreCube,
    route_by_item: Sequence[str],
    candidate_indices: Sequence[Sequence[int]],
    *,
    target_k: int,
) -> np.ndarray:
    plan = np.asarray(route_by_item, dtype=object)
    output = np.zeros(len(cube.query_ids), dtype=np.float64)
    for query_index, candidates_value in enumerate(candidate_indices):
        candidates = np.asarray(candidates_value, dtype=np.int32)
        scores = np.asarray(
            [
                cube.scores[str(plan[item])][query_index, item]
                for item in candidates
            ],
            dtype=np.float64,
        )
        order = candidates[np.lexsort((candidates, -scores))][:target_k]
        relevance = cube.relevance[query_index]
        dcg = sum(
            relevance.get(int(item), 0.0) / np.log2(rank + 1)
            for rank, item in enumerate(order, start=1)
        )
        ideal = sorted(relevance.values(), reverse=True)[:target_k]
        idcg = sum(
            value / np.log2(rank + 1)
            for rank, value in enumerate(ideal, start=1)
        )
        output[query_index] = dcg / idcg if idcg else 0.0
    return output


def enumerate_type_policies(
    cube: ScoreCube,
    *,
    item_types: Sequence[str],
    candidate_indices: Sequence[Sequence[int]],
    route_costs: Mapping[str, Sequence[float]],
    target_k: int = 5,
) -> tuple[tuple[str, ...], tuple[dict, ...]]:
    """Enumerate every type-to-route mapping and cache per-query quality."""

    types = tuple(sorted(set(str(value) for value in item_types)))
    if len(types) > 8:
        raise ValueError("exact type-policy enumeration is limited to eight types")
    item_types_array = np.asarray(item_types, dtype=object)
    if item_types_array.shape != (len(cube.corpus_ids),):
        raise ValueError("item types must be corpus-aligned")
    costs = {
        route: np.asarray(values, dtype=np.float64) for route, values in route_costs.items()
    }
    records = []
    for choices in itertools.product(cube.routes, repeat=len(types)):
        mapping = dict(zip(types, choices, strict=True))
        plan = tuple(mapping[str(value)] for value in item_types_array)
        cost = float(sum(costs[route][item] for item, route in enumerate(plan)))
        records.append(
            {
                "mapping": mapping,
                "cost": cost,
                "per_query_ndcg": _candidate_ndcg(
                    cube, plan, candidate_indices, target_k=target_k
                ),
            }
        )
    return types, tuple(records)


def crossfit_type_policy_compiler(
    cube: ScoreCube,
    *,
    item_types: Sequence[str],
    candidate_indices: Sequence[Sequence[int]],
    route_costs: Mapping[str, Sequence[float]],
    fold_ids: Sequence[int],
    budget_fractions: Sequence[float] = (0.04, 0.06, 0.09, 0.11, 0.25, 1.0),
    teacher_route: str = "image",
    target_k: int = 5,
) -> dict:
    """Select exact type mappings on training folds and evaluate held-out folds."""

    types, records = enumerate_type_policies(
        cube,
        item_types=item_types,
        candidate_indices=candidate_indices,
        route_costs=route_costs,
        target_k=target_k,
    )
    folds = np.asarray(fold_ids, dtype=np.int16)
    if folds.shape != (len(cube.query_ids),) or len(set(folds.tolist())) < 2:
        raise ValueError("fold IDs must be query-aligned and non-degenerate")
    teacher_cost = float(np.sum(route_costs[teacher_route]))
    uniform_values = {}
    for route in cube.routes:
        uniform_values[route] = next(
            record["per_query_ndcg"]
            for record in records
            if set(record["mapping"].values()) == {route}
        )
    reports = {}
    for fraction in budget_fractions:
        budget = teacher_cost * fraction
        feasible = [record for record in records if record["cost"] <= budget + 1e-6]
        if not feasible:
            reports[str(fraction)] = {"feasible": False}
            continue
        predictions = np.zeros(len(cube.query_ids), dtype=np.float64)
        selected = []
        fit_means = []
        costs = []
        for fold in sorted(set(folds.tolist())):
            fit = folds != fold
            test = folds == fold
            best = max(
                feasible,
                key=lambda record: (
                    float(record["per_query_ndcg"][fit].mean()),
                    -record["cost"],
                    tuple(record["mapping"][value] for value in types),
                ),
            )
            predictions[test] = best["per_query_ndcg"][test]
            fit_means.append(float(best["per_query_ndcg"][fit].mean()))
            costs.append(best["cost"])
            selected.append(
                {
                    "fold": int(fold),
                    "test_queries": int(test.sum()),
                    "mapping": best["mapping"],
                    "cost_fraction": best["cost"] / teacher_cost,
                    "fit_ndcg_at_5": float(best["per_query_ndcg"][fit].mean()),
                    "test_ndcg_at_5": float(best["per_query_ndcg"][test].mean()),
                }
            )
        report = {
            "feasible": True,
            "crossfit_ndcg_at_5": float(predictions.mean()),
            "mean_selected_fit_ndcg_at_5": float(np.mean(fit_means)),
            "mean_cost_fraction": float(np.mean(costs) / teacher_cost),
            "per_query_ndcg_at_5": predictions.tolist(),
            "vs_uniform_image": paired_bootstrap_ci(
                predictions, uniform_values[teacher_route]
            ),
            "selected_by_fold": selected,
        }
        if "image-pool-9" in uniform_values:
            report["vs_uniform_image_pool_9"] = paired_bootstrap_ci(
                predictions, uniform_values["image-pool-9"]
            )
        reports[str(fraction)] = report
    return {
        "protocol": "exact type-to-route policy search; cross-fitted qrel objective",
        "compiler_uses_fit_qrels": True,
        "types": types,
        "routes": cube.routes,
        "enumerated_policies": len(records),
        "folds": len(set(folds.tolist())),
        "uniform_ndcg_at_5": {
            route: float(values.mean()) for route, values in uniform_values.items()
        },
        "budget_curve": reports,
    }
