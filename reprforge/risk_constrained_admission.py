"""Cost-aware and empirically risk-constrained representation admission.

This module joins the three quantities that earlier ReprForge probes kept
separate:

* complementary boundary value from a pair graph;
* atomic encoder and scoring cost from the physical executor; and
* a train-only upper bound on extra teacher disagreement.

The controller is not a formal distribution-free guarantee.  It uses a
paired bootstrap over historical probe queries and reports that limitation
explicitly.  Its role is to test whether an explicit, auditable quality-loss
allowance can unlock enough physical work reduction to justify a stronger
system design.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reprforge.pairwise_view_admission import (
    BoundaryPair,
    PairAdmission,
    evaluate_pair_coverage,
)
from reprforge.physical_cost import (
    AtomicCostEstimate,
    AtomicMaterializationCostModel,
)


@dataclass(frozen=True)
class CostAwareAdmission:
    admission: PairAdmission
    estimated_cost: AtomicCostEstimate
    time_budget_ms: float


@dataclass(frozen=True)
class CostFractionDecision:
    selected_fraction: float
    observed_extra_disagreement: float
    upper_extra_disagreement: float
    best_attempt_upper_extra_disagreement: float | None
    fallback_to_baseline: bool
    calibration_queries: int
    calibration_groups: int


def selected_score_events(
    candidate_pages: np.ndarray,
    selected_pages: Sequence[int] | set[int] | frozenset[int],
) -> int:
    candidates = np.asarray(candidate_pages, dtype=np.int64)
    if candidates.ndim != 2:
        raise ValueError("candidate pages must be a 2-D matrix")
    frequency = Counter(int(page) for page in candidates.flat)
    return sum(frequency.get(int(page), 0) for page in selected_pages)


def estimate_plan_cost(
    candidate_pages: np.ndarray,
    selected_pages: Sequence[int] | set[int] | frozenset[int],
    cost_model: AtomicMaterializationCostModel,
) -> AtomicCostEstimate:
    selected = frozenset(int(page) for page in selected_pages)
    return cost_model.estimate(
        pages=len(selected),
        score_events=selected_score_events(candidate_pages, selected),
    )


def _edge_weights(pairs: Sequence[BoundaryPair]) -> dict[tuple[int, int], float]:
    weights: defaultdict[tuple[int, int], float] = defaultdict(float)
    for pair in pairs:
        weights[tuple(sorted((pair.incumbent, pair.challenger)))] += pair.weight
    return dict(weights)


def _covered_weight(
    edge_weights: dict[tuple[int, int], float],
    selected: set[int],
) -> float:
    return sum(
        weight
        for (left, right), weight in edge_weights.items()
        if left in selected and right in selected
    )


def select_cost_aware_pairs(
    pairs: Sequence[BoundaryPair],
    candidate_pages: np.ndarray,
    cost_model: AtomicMaterializationCostModel,
    *,
    time_budget_ms: float,
) -> CostAwareAdmission:
    """Greedily buy complete boundary comparisons per marginal physical ms."""

    if not math.isfinite(time_budget_ms) or time_budget_ms < 0:
        raise ValueError("time budget must be finite and non-negative")
    weights = _edge_weights(pairs)
    selected: set[int] = set()
    current_value = 0.0
    current_cost = cost_model.estimate(pages=0, score_events=0)

    while weights:
        actions: set[tuple[int, ...]] = set()
        for left, right in weights:
            if left not in selected:
                actions.add((left,))
            if right not in selected:
                actions.add((right,))
            if left not in selected and right not in selected:
                actions.add((left, right))
        best: tuple[float, float, float, tuple[int, ...], AtomicCostEstimate] | None = None
        for action in actions:
            proposed = selected | set(action)
            if proposed == selected:
                continue
            proposed_value = _covered_weight(weights, proposed)
            gain = proposed_value - current_value
            if gain <= 0:
                continue
            proposed_cost = estimate_plan_cost(candidate_pages, proposed, cost_model)
            if proposed_cost.total_ms > time_budget_ms + 1e-9:
                continue
            marginal_cost = max(proposed_cost.total_ms - current_cost.total_ms, 1e-9)
            density = gain / marginal_cost
            # Prefer denser work, then more value, then less total cost and a
            # stable page-id order.  The tuple is negated only where larger is
            # preferable, keeping the comparison deterministic.
            candidate = (
                density,
                gain,
                -proposed_cost.total_ms,
                tuple(-page for page in action),
                proposed_cost,
            )
            if best is None or candidate[:4] > best[:4]:
                best = candidate
        if best is None:
            break
        action = tuple(-page for page in best[3])
        selected.update(action)
        current_value = _covered_weight(weights, selected)
        current_cost = best[4]

    admission = evaluate_pair_coverage(pairs, selected)
    return CostAwareAdmission(
        admission=admission,
        estimated_cost=current_cost,
        time_budget_ms=time_budget_ms,
    )


def paired_bootstrap_upper_loss(
    candidate_correct: Sequence[bool] | np.ndarray,
    baseline_correct: Sequence[bool] | np.ndarray,
    *,
    confidence: float,
    bootstrap_samples: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return observed and one-sided bootstrap upper extra disagreement."""

    candidate = np.asarray(candidate_correct, dtype=bool)
    baseline = np.asarray(baseline_correct, dtype=bool)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("paired correctness arrays must align in one dimension")
    if not len(candidate):
        raise ValueError("at least one calibration query is required")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    # Positive values mean the candidate makes an additional error.
    loss = (~candidate).astype(np.float64) - (~baseline).astype(np.float64)
    observed = float(loss.mean())
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(loss), size=(bootstrap_samples, len(loss)))
    means = loss[indices].mean(axis=1)
    upper = float(np.quantile(means, confidence, method="higher"))
    return observed, upper


def paired_group_bootstrap_upper_loss(
    candidate_correct: Sequence[bool] | np.ndarray,
    baseline_correct: Sequence[bool] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    confidence: float,
    bootstrap_samples: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap source groups, treating each group as one transfer unit."""

    candidate = np.asarray(candidate_correct, dtype=bool)
    baseline = np.asarray(baseline_correct, dtype=bool)
    labels = np.asarray([str(value) for value in groups])
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("paired correctness arrays must align in one dimension")
    if labels.shape != candidate.shape:
        raise ValueError("bootstrap groups must match correctness observations")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    loss = (~candidate).astype(np.float64) - (~baseline).astype(np.float64)
    unique = sorted(set(labels))
    if len(unique) < 2:
        raise ValueError("group bootstrap requires at least two source groups")
    group_means = np.asarray(
        [float(loss[labels == group].mean()) for group in unique],
        dtype=np.float64,
    )
    observed = float(group_means.mean())
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(group_means),
        size=(bootstrap_samples, len(group_means)),
    )
    upper = float(
        np.quantile(group_means[indices].mean(axis=1), confidence, method="higher")
    )
    return observed, upper


def select_risk_bounded_cost_fraction(
    candidate_correct_by_fraction: dict[float, np.ndarray],
    baseline_correct: Sequence[bool] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    risk_tolerance: float,
    confidence: float,
    bootstrap_samples: int = 4000,
    seed: int = 0,
    minimum_calibration_queries: int = 20,
    minimum_calibration_groups: int = 4,
) -> CostFractionDecision:
    """Select a deployable cost fraction from cross-fitted predictions."""

    baseline = np.asarray(baseline_correct, dtype=bool)
    labels = np.asarray([str(value) for value in groups])
    if baseline.ndim != 1 or labels.shape != baseline.shape:
        raise ValueError("calibration outcomes and groups must align")
    if not candidate_correct_by_fraction:
        raise ValueError("at least one candidate cost fraction is required")
    if not 0.0 <= risk_tolerance <= 1.0:
        raise ValueError("risk tolerance must lie in [0, 1]")
    fractions = tuple(sorted(float(value) for value in candidate_correct_by_fraction))
    if any(value <= 0 or value > 1 for value in fractions):
        raise ValueError("cost fractions must lie in (0, 1]")
    group_count = len(set(labels))
    if len(baseline) < minimum_calibration_queries or group_count < minimum_calibration_groups:
        return CostFractionDecision(
            selected_fraction=1.0,
            observed_extra_disagreement=0.0,
            upper_extra_disagreement=0.0,
            best_attempt_upper_extra_disagreement=None,
            fallback_to_baseline=True,
            calibration_queries=len(baseline),
            calibration_groups=group_count,
        )

    best_upper = float("inf")
    for fraction in fractions:
        candidate = np.asarray(candidate_correct_by_fraction[fraction], dtype=bool)
        if candidate.shape != baseline.shape:
            raise ValueError("candidate outcomes must align with the baseline")
        observed, upper = paired_group_bootstrap_upper_loss(
            candidate,
            baseline,
            labels,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        best_upper = min(best_upper, upper)
        if upper <= risk_tolerance:
            return CostFractionDecision(
                selected_fraction=fraction,
                observed_extra_disagreement=observed,
                upper_extra_disagreement=upper,
                best_attempt_upper_extra_disagreement=best_upper,
                fallback_to_baseline=False,
                calibration_queries=len(baseline),
                calibration_groups=group_count,
            )
    return CostFractionDecision(
        selected_fraction=1.0,
        observed_extra_disagreement=0.0,
        upper_extra_disagreement=0.0,
        best_attempt_upper_extra_disagreement=best_upper,
        fallback_to_baseline=True,
        calibration_queries=len(baseline),
        calibration_groups=group_count,
    )
