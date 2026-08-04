"""Qrel-free physical allocation for a two-state compression ladder.

The compiler starts from a cheap corpus-wide representation and upgrades
physical documents to the full representation.  All utilities are computed
from full/cheap rankings on an unlabeled fit workload.  Relevance judgments
are deliberately outside this module.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from reprforge.heterogeneity_atlas import deterministic_split_roles, stable_ranks


def _aligned_surfaces(
    full_scores: np.ndarray, cheap_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(full_scores, dtype=np.float64)
    cheap = np.asarray(cheap_scores, dtype=np.float64)
    if full.ndim != 2 or full.shape != cheap.shape or not full.size:
        raise ValueError("full and cheap surfaces must be aligned non-empty matrices")
    if not np.isfinite(full).all() or not np.isfinite(cheap).all():
        raise ValueError("score surfaces contain non-finite values")
    return full, cheap


def incremental_upgrade_bytes(
    full_vector_bytes: Sequence[int], cheap_vector_bytes: Sequence[int]
) -> np.ndarray:
    """Return corpus-aligned non-negative bytes needed for full upgrades."""

    full = np.asarray(full_vector_bytes, dtype=np.int64)
    cheap = np.asarray(cheap_vector_bytes, dtype=np.int64)
    if full.ndim != 1 or full.shape != cheap.shape or not len(full):
        raise ValueError("vector-byte arrays must be aligned non-empty vectors")
    delta = full - cheap
    if np.any(full <= 0) or np.any(cheap <= 0) or np.any(delta < 0):
        raise ValueError("full/cheap bytes must be positive and full >= cheap")
    return delta


def boundary_risk_utilities(
    full_scores: np.ndarray,
    cheap_scores: np.ndarray,
    fit_indices: Sequence[int],
    *,
    target_k: int = 10,
    exposure_depth: int = 100,
    temperature: float = 10.0,
) -> dict[str, np.ndarray]:
    """Compute document utilities from unlabeled fit-ranking distortion.

    ``boundary_flip`` measures recurrent Top-k membership changes plus smooth
    rank displacement around the competitive boundary. ``exposure`` is a
    deliberately weaker frequency control. ``score_residual`` tests whether
    query-normalized score error alone explains transfer.
    """

    full, cheap = _aligned_surfaces(full_scores, cheap_scores)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if fit.ndim != 1 or not len(fit) or np.any(fit < 0) or np.any(fit >= len(full)):
        raise ValueError("fit_indices must select at least one valid query")
    if len(np.unique(fit)) != len(fit):
        raise ValueError("fit_indices must be unique")
    if not 0 < target_k < exposure_depth <= full.shape[1]:
        raise ValueError("require 0 < target_k < exposure_depth <= corpus size")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    full_fit = full[fit]
    cheap_fit = cheap[fit]
    full_ranks = stable_ranks(full_fit)
    cheap_ranks = stable_ranks(cheap_fit)
    corpus = full.shape[1]
    exposure = np.zeros(corpus, dtype=np.float64)
    flips = np.zeros(corpus, dtype=np.float64)
    displacement = np.zeros(corpus, dtype=np.float64)
    residual = np.zeros(corpus, dtype=np.float64)

    for query_index in range(len(fit)):
        full_rank = full_ranks[query_index]
        cheap_rank = cheap_ranks[query_index]
        competitive = np.minimum(full_rank, cheap_rank) <= exposure_depth
        positions = np.flatnonzero(competitive)
        best_rank = np.minimum(full_rank[positions], cheap_rank[positions])
        proximity = np.exp(-np.abs(best_rank - target_k) / temperature)
        top_flip = (full_rank[positions] <= target_k) != (
            cheap_rank[positions] <= target_k
        )
        rank_shift = np.minimum(
            np.abs(full_rank[positions] - cheap_rank[positions]) / target_k,
            1.0,
        )
        exposure[positions] += proximity
        flips[positions] += top_flip.astype(np.float64)
        displacement[positions] += proximity * rank_shift

        full_order = np.lexsort((np.arange(corpus), -full_fit[query_index]))
        high = full_fit[query_index, full_order[0]]
        low = full_fit[query_index, full_order[exposure_depth - 1]]
        scale = max(float(high - low), 1e-8)
        residual[positions] += proximity * np.minimum(
            np.abs(
                full_fit[query_index, positions] - cheap_fit[query_index, positions]
            )
            / scale,
            4.0,
        )

    divisor = float(len(fit))
    exposure /= divisor
    flips /= divisor
    displacement /= divisor
    residual /= divisor
    return {
        "boundary_flip": flips + displacement,
        "boundary_flip_x2": 2.0 * flips + displacement,
        "boundary_flip_x4": 4.0 * flips + displacement,
        "flip_only": flips,
        "rank_displacement": displacement,
        "exposure": exposure,
        "score_residual": residual,
    }


def budgeted_upgrade_order(
    utility: Sequence[float], incremental_bytes: Sequence[int]
) -> np.ndarray:
    """Order all documents by utility per incremental byte.

    Zero-utility documents remain at the end so a 100% upgrade budget exactly
    recovers the full representation rather than silently stopping early.
    """

    values = np.asarray(utility, dtype=np.float64)
    costs = np.asarray(incremental_bytes, dtype=np.int64)
    if values.ndim != 1 or values.shape != costs.shape or not len(values):
        raise ValueError("utility and costs must be aligned non-empty vectors")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(costs < 0):
        raise ValueError("utilities and costs must be finite and non-negative")
    # Zero-cost upgrades are valid and should be taken first.
    density = np.divide(
        values,
        costs,
        out=np.full_like(values, np.inf),
        where=costs > 0,
    )
    return np.asarray(
        sorted(
            range(len(values)),
            key=lambda index: (
                -float(density[index]),
                -float(values[index]),
                int(costs[index]),
                int(index),
            ),
        ),
        dtype=np.int64,
    )


def select_upgrade_mask(
    order: Sequence[int],
    incremental_bytes: Sequence[int],
    *,
    budget_bytes: int,
) -> np.ndarray:
    """Materialize the density order without exceeding an incremental budget."""

    costs = np.asarray(incremental_bytes, dtype=np.int64)
    if costs.ndim != 1 or not len(costs) or np.any(costs < 0):
        raise ValueError("incremental_bytes must be a non-negative vector")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    selected = np.zeros(len(costs), dtype=bool)
    used = 0
    for raw_index in order:
        index = int(raw_index)
        if not 0 <= index < len(costs) or selected[index]:
            raise ValueError("upgrade order contains invalid or duplicate positions")
        cost = int(costs[index])
        if used + cost <= budget_bytes:
            selected[index] = True
            used += cost
    return selected


def round_robin_upgrade_order(*orders: Sequence[int]) -> np.ndarray:
    """Interleave complete deterministic orders without tuning a scalar weight."""

    if len(orders) < 2:
        raise ValueError("round-robin allocation needs at least two orders")
    arrays = [np.asarray(order, dtype=np.int64) for order in orders]
    size = len(arrays[0])
    if size == 0 or any(array.shape != (size,) for array in arrays):
        raise ValueError("round-robin orders must be aligned non-empty vectors")
    expected = set(range(size))
    if any(set(array.tolist()) != expected for array in arrays):
        raise ValueError("every round-robin input must be a corpus permutation")
    combined: list[int] = []
    seen: set[int] = set()
    cursors = [0] * len(arrays)
    while len(combined) < size:
        for order_index, array in enumerate(arrays):
            while cursors[order_index] < size:
                value = int(array[cursors[order_index]])
                cursors[order_index] += 1
                if value not in seen:
                    seen.add(value)
                    combined.append(value)
                    break
    return np.asarray(combined, dtype=np.int64)


def compile_upgrade_mask_from_fit(
    full_scores: np.ndarray,
    cheap_scores: np.ndarray,
    fit_indices: Sequence[int],
    full_anchor_bytes: Sequence[int],
    *,
    budget_fraction: float = 0.65,
    policy: str = "boundary_pareto",
) -> dict:
    """Compile one static full-anchor mask from an unlabeled fit workload."""

    full, cheap = _aligned_surfaces(full_scores, cheap_scores)
    costs = np.asarray(full_anchor_bytes, dtype=np.int64)
    if costs.shape != (full.shape[1],) or np.any(costs <= 0):
        raise ValueError("full anchor bytes must be positive and corpus-aligned")
    if not 0.0 <= budget_fraction <= 1.0:
        raise ValueError("budget_fraction must lie in [0, 1]")
    if full.shape[1] < 2:
        raise ValueError("physical compilation requires at least two corpus items")
    utilities = boundary_risk_utilities(
        full,
        cheap,
        fit_indices,
        target_k=min(10, full.shape[1] - 1),
        exposure_depth=min(100, full.shape[1]),
    )
    orders = {
        name: budgeted_upgrade_order(utility, costs)
        for name, utility in utilities.items()
    }
    orders["boundary_pareto"] = round_robin_upgrade_order(
        orders["flip_only"], orders["rank_displacement"]
    )
    if policy not in orders:
        raise ValueError(f"unknown physical allocation policy: {policy}")
    budget_bytes = int(float(budget_fraction) * int(costs.sum()))
    mask = select_upgrade_mask(orders[policy], costs, budget_bytes=budget_bytes)
    return {
        "policy": policy,
        "budget_fraction": float(budget_fraction),
        "budget_bytes": budget_bytes,
        "upgraded_documents": np.flatnonzero(mask).tolist(),
        "upgraded_document_count": int(mask.sum()),
        "anchor_vector_bytes": int(costs[mask].sum()),
    }


def hybrid_score_surface(
    full_scores: np.ndarray, cheap_scores: np.ndarray, upgraded: Sequence[bool]
) -> np.ndarray:
    """Return exact scores of a document-static full/cheap hybrid bank."""

    full, cheap = _aligned_surfaces(full_scores, cheap_scores)
    mask = np.asarray(upgraded, dtype=bool)
    if mask.shape != (full.shape[1],):
        raise ValueError("upgrade mask must be corpus-aligned")
    hybrid = cheap.copy()
    hybrid[:, mask] = full[:, mask]
    return hybrid


def calibrated_residual_surface(
    full_scores: np.ndarray,
    cheap_scores: np.ndarray,
    upgraded: Sequence[bool],
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Complete full-score residuals from physically materialized anchors.

    The cheap representation remains resident for every document.  For each
    query, upgraded documents expose paired ``(cheap, full)`` scores.  A tiny
    ridge-affine model predicts the full-minus-cheap residual for unupgraded
    documents, while upgraded documents retain their exact full scores.  This
    makes the two representation states score-comparable and charges both
    views for upgraded documents.
    """

    full, cheap = _aligned_surfaces(full_scores, cheap_scores)
    mask = np.asarray(upgraded, dtype=bool)
    if mask.shape != (full.shape[1],):
        raise ValueError("upgrade mask must be corpus-aligned")
    anchors = np.flatnonzero(mask)
    if not len(anchors):
        return cheap.copy()
    return calibrated_residual_surface_from_anchors(
        cheap,
        full[:, anchors],
        anchors,
        ridge=ridge,
    )


def calibrated_residual_surface_from_anchors(
    cheap_scores: np.ndarray,
    anchor_full_scores: np.ndarray,
    anchor_positions: Sequence[int],
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Calibrate a cheap surface using only physically scored full anchors."""

    cheap = np.asarray(cheap_scores, dtype=np.float64)
    anchor_full = np.asarray(anchor_full_scores, dtype=np.float64)
    anchors = np.asarray(anchor_positions, dtype=np.int64)
    if cheap.ndim != 2 or not cheap.size or not np.isfinite(cheap).all():
        raise ValueError("cheap score surface must be a finite non-empty matrix")
    if (
        anchors.ndim != 1
        or not len(anchors)
        or len(np.unique(anchors)) != len(anchors)
        or np.any(anchors < 0)
        or np.any(anchors >= cheap.shape[1])
    ):
        raise ValueError("anchor positions must be unique and corpus-aligned")
    if anchor_full.shape != (cheap.shape[0], len(anchors)) or not np.isfinite(
        anchor_full
    ).all():
        raise ValueError("full-anchor scores must be query/anchor aligned")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    output = np.empty_like(cheap)
    for query_index in range(len(cheap)):
        x_anchor = cheap[query_index, anchors]
        y_anchor = anchor_full[query_index] - x_anchor
        center = float(x_anchor.mean())
        scale = max(float(x_anchor.std()), 1e-8)
        z_anchor = (x_anchor - center) / scale
        design = np.column_stack((np.ones(len(anchors)), z_anchor))
        gram = design.T @ design
        penalty = np.diag((0.0, ridge))
        coefficients = np.linalg.solve(gram + penalty, design.T @ y_anchor)
        z_all = np.clip((cheap[query_index] - center) / scale, -6.0, 6.0)
        predicted = coefficients[0] + coefficients[1] * z_all
        # Prevent sparse or biased anchors from extrapolating beyond residuals
        # that were physically observed for this query.
        predicted = np.clip(predicted, y_anchor.min(), y_anchor.max())
        output[query_index] = cheap[query_index] + predicted
        output[query_index, anchors] = anchor_full[query_index]
    return output


def compile_physical_curve(
    query_ids: Sequence[str],
    full_scores: np.ndarray,
    cheap_scores: np.ndarray,
    full_vector_bytes: Sequence[int],
    cheap_vector_bytes: Sequence[int],
    *,
    budget_fractions: Sequence[float],
    eval_fraction: float = 1.0 / 3.0,
    retain_cheap_for_upgraded: bool = False,
    fit_indices: Sequence[int] | None = None,
) -> dict:
    """Compile unlabeled document masks for several policies and budgets."""

    full, cheap = _aligned_surfaces(full_scores, cheap_scores)
    if len(query_ids) != len(full):
        raise ValueError("query_ids must be query-aligned")
    if fit_indices is None:
        roles = deterministic_split_roles(query_ids, eval_fraction=eval_fraction)
        fit = np.flatnonzero(np.asarray(roles) == "fit")
        evaluation = np.flatnonzero(np.asarray(roles) == "eval")
    else:
        fit = np.asarray(fit_indices, dtype=np.int64)
        if (
            fit.ndim != 1
            or not len(fit)
            or len(np.unique(fit)) != len(fit)
            or np.any(fit < 0)
            or np.any(fit >= len(full))
        ):
            raise ValueError("fit_indices must select unique valid queries")
        fit_mask = np.zeros(len(full), dtype=bool)
        fit_mask[fit] = True
        evaluation = np.flatnonzero(~fit_mask)
        if not len(evaluation):
            raise ValueError("fit_indices must leave at least one evaluation query")
        roles = tuple("fit" if value else "eval" for value in fit_mask)
    incremental = incremental_upgrade_bytes(full_vector_bytes, cheap_vector_bytes)
    upgrade_costs = (
        np.asarray(full_vector_bytes, dtype=np.int64)
        if retain_cheap_for_upgraded
        else incremental
    )
    if full.shape[1] < 2:
        raise ValueError("physical compilation requires at least two corpus items")
    utilities = boundary_risk_utilities(
        full,
        cheap,
        fit,
        target_k=min(10, full.shape[1] - 1),
        exposure_depth=min(100, full.shape[1]),
    )
    utilities["random"] = np.random.default_rng(20260804).random(full.shape[1])
    total_incremental = int(upgrade_costs.sum())
    if total_incremental <= 0:
        raise ValueError("compression ladder has no positive upgrade cost")
    policy_orders = {
        name: budgeted_upgrade_order(utility, upgrade_costs)
        for name, utility in utilities.items()
    }
    policy_orders["boundary_pareto"] = round_robin_upgrade_order(
        policy_orders["flip_only"], policy_orders["rank_displacement"]
    )
    policies = {}
    for name, order in policy_orders.items():
        points = []
        for fraction in budget_fractions:
            value = float(fraction)
            if not 0.0 <= value <= 1.0:
                raise ValueError("budget fractions must lie in [0, 1]")
            mask = select_upgrade_mask(
                order,
                upgrade_costs,
                budget_bytes=int(value * total_incremental),
            )
            points.append(
                {
                    "budget_fraction": value,
                    "upgraded_documents": np.flatnonzero(mask).tolist(),
                    "upgraded_document_count": int(mask.sum()),
                    "incremental_bytes": int(upgrade_costs[mask].sum()),
                }
            )
        policies[name] = points
    return {
        "query_split_roles": list(roles),
        "fit_query_indices": fit.tolist(),
        "eval_query_indices": evaluation.tolist(),
        "cheap_vector_bytes": int(np.asarray(cheap_vector_bytes).sum()),
        "full_vector_bytes": int(np.asarray(full_vector_bytes).sum()),
        "total_incremental_bytes": total_incremental,
        "retain_cheap_for_upgraded": bool(retain_cheap_for_upgraded),
        "policies": policies,
    }
