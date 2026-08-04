"""Query-local landmark probing for expensive representation completion."""

from __future__ import annotations

import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube, query_metrics


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    scale = max(float(values.std()), 1e-12)
    return (values - mean) / scale


def _fit_completion(x: np.ndarray, y: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fit a query-local quadratic response surface from observed landmarks."""

    degree = min(2, int(observed.sum()) - 1)
    columns = [np.ones(len(x)), x]
    if degree >= 2:
        columns.append(x**2)
    design = np.column_stack(columns)
    fit = design[observed]
    target = y[observed]
    regularizer = np.eye(fit.shape[1]) * 1e-3
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        fit.T @ fit + regularizer, fit.T @ target
    )
    predicted = design @ coefficients
    predicted[observed] = y[observed]
    return predicted


def _initial_order(count: int) -> list[int]:
    """Nested farthest-point coverage over cheap-rank positions."""

    if count < 2:
        return list(range(count))
    selected = [0, count - 1]
    remaining = set(range(1, count - 1))
    while remaining:
        candidate = max(
            remaining,
            key=lambda value: (
                min(abs(value - chosen) for chosen in selected),
                -value,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def _completed_rerank_scores(
    base_scores: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_values: np.ndarray,
) -> np.ndarray:
    """Rerank a frozen candidate set while keeping it above the untouched tail."""

    output = np.asarray(base_scores, dtype=np.float64).copy()
    order = np.lexsort((candidate_indices, -candidate_values))
    width = len(candidate_indices)
    # The candidate band is separated from the base-score tail. Only candidate
    # order changes, matching a conventional retrieve-then-rerank contract.
    output[candidate_indices[order]] = 2.0 + (
        width - np.arange(width, dtype=np.float64)
    ) / width
    outside = np.ones(len(output), dtype=bool)
    outside[candidate_indices] = False
    if outside.any():
        output[outside] = output[outside] - output[outside].max()
    return output


def landmark_completion_surface(
    base_scores: np.ndarray,
    expensive_scores: np.ndarray,
    *,
    candidate_k: int,
    budget: int,
    policy: str = "coverage",
    target_k: int = 10,
) -> np.ndarray:
    """Return a coherent reranked surface using ``budget`` exact landmarks."""

    base = np.asarray(base_scores, dtype=np.float64)
    expensive = np.asarray(expensive_scores, dtype=np.float64)
    if base.shape != expensive.shape or base.ndim != 2:
        raise ValueError("base and expensive score surfaces must align")
    if not 2 <= budget <= candidate_k <= base.shape[1]:
        raise ValueError("require 2 <= budget <= candidate_k <= corpus")
    if not 1 <= target_k <= candidate_k:
        raise ValueError("target_k must lie inside the candidate set")
    if policy not in {"coverage", "boundary"}:
        raise ValueError("unsupported landmark policy")

    result = np.empty_like(base)
    positions = np.arange(base.shape[1])
    coverage_order = _initial_order(candidate_k)
    for query_index in range(base.shape[0]):
        candidates = np.lexsort((positions, -base[query_index]))[:candidate_k]
        x = _zscore(base[query_index, candidates])
        y = expensive[query_index, candidates]
        observed = np.zeros(candidate_k, dtype=bool)
        observed[coverage_order[: min(3, budget)]] = True
        while int(observed.sum()) < budget:
            if policy == "coverage":
                observed[coverage_order[int(observed.sum())]] = True
                continue
            predicted = _fit_completion(x, y, observed)
            fused = x + _zscore(predicted)
            boundary = np.sort(fused)[-target_k]
            observed_positions = np.flatnonzero(observed)
            rank_distance = np.asarray(
                [
                    min(abs(index - chosen) for chosen in observed_positions)
                    for index in range(candidate_k)
                ],
                dtype=np.float64,
            )
            rank_distance /= max(candidate_k - 1, 1)
            priority = rank_distance / (np.abs(fused - boundary) + 0.05)
            priority[observed] = -np.inf
            observed[int(np.argmax(priority))] = True
        completed = _fit_completion(x, y, observed)
        fused = x + _zscore(completed)
        result[query_index] = _completed_rerank_scores(
            base[query_index], candidates, fused
        )
    return result


def full_candidate_fusion_surface(
    base_scores: np.ndarray,
    expensive_scores: np.ndarray,
    *,
    candidate_k: int,
    target_k: int = 10,
) -> np.ndarray:
    """Exact candidate-local normalized fusion with all expensive scores."""

    return landmark_completion_surface(
        base_scores,
        expensive_scores,
        candidate_k=candidate_k,
        budget=candidate_k,
        policy="coverage",
        target_k=target_k,
    )


def analyze_landmark_budgets(
    cube: ScoreCube,
    *,
    base_route: str,
    expensive_route: str,
    candidate_k: int = 20,
    budgets: tuple[int, ...] = (2, 4, 8, 12, 20),
    target_metric: str = "ndcg_at_10",
    target_k: int = 10,
) -> dict:
    cube.validate()
    base = cube.scores[base_route]
    expensive = cube.scores[expensive_route]
    base_metrics = query_metrics(base, cube.relevance, ks=(target_k,))
    full_surface = full_candidate_fusion_surface(
        base, expensive, candidate_k=candidate_k, target_k=target_k
    )
    full_metrics = query_metrics(full_surface, cube.relevance, ks=(target_k,))
    rows = []
    for policy in ("coverage", "boundary"):
        for budget in budgets:
            surface = landmark_completion_surface(
                base,
                expensive,
                candidate_k=candidate_k,
                budget=budget,
                policy=policy,
                target_k=target_k,
            )
            values = query_metrics(surface, cube.relevance, ks=(target_k,))[
                target_metric
            ]
            base_values = base_metrics[target_metric]
            full_values = full_metrics[target_metric]
            gain = float((values - base_values).mean())
            full_gain = float((full_values - base_values).mean())
            rows.append(
                {
                    "policy": policy,
                    "budget": budget,
                    "candidate_k": candidate_k,
                    "probe_fraction": budget / candidate_k,
                    target_metric: float(values.mean()),
                    "gain_over_base": gain,
                    "full_fusion_gain_recovery": gain / full_gain if abs(full_gain) > 1e-12 else 0.0,
                }
            )
    return {
        "schema_version": 1,
        "queries": len(cube.query_ids),
        "corpus": len(cube.corpus_ids),
        "base_route": base_route,
        "expensive_route": expensive_route,
        "candidate_k": candidate_k,
        "target_metric": target_metric,
        "base": float(base_metrics[target_metric].mean()),
        "full_candidate_fusion": float(full_metrics[target_metric].mean()),
        "rows": rows,
    }
