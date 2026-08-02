"""Sparse, qrel-free what-if estimation for representation admission.

Historical full visual indexes are useful teachers but are not a deployable
prerequisite.  This module selects a small, locator-visible query sample,
observes complete candidate cohorts only for those probes, and estimates the
probability that a tail candidate crosses the requested ranking boundary.

The estimator is deliberately interpretable: rank and locator boundary margin
define a small empirical table with a Beta posterior.  It exposes both a mean
and uncertainty so callers can distinguish exploitation from conservative
coverage.  No relevance label enters the API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reprforge.pairwise_view_admission import BoundaryPair


@dataclass(frozen=True)
class SparseProbePlan:
    query_indices: np.ndarray
    query_fraction: float
    candidate_events: int


@dataclass(frozen=True)
class SparseBoundaryRiskModel:
    cutoff: int
    candidate_k: int
    margin_edges: np.ndarray
    probability: np.ndarray
    uncertainty: np.ndarray
    observations: np.ndarray
    probed_queries: int
    probed_candidate_events: int

    def __post_init__(self) -> None:
        if self.margin_edges.ndim != 2:
            raise ValueError("risk margin edges must be a rank-by-bin matrix")
        if self.margin_edges.shape[0] != self.candidate_k - self.cutoff:
            raise ValueError("risk margin ranks do not match the modeled tail")
        bins = self.margin_edges.shape[1] + 1
        expected = (self.candidate_k - self.cutoff, bins)
        if self.probability.shape != expected:
            raise ValueError("risk probability table has the wrong shape")
        if self.uncertainty.shape != expected or self.observations.shape != expected:
            raise ValueError("risk uncertainty tables must align")

    def predict(
        self,
        challenger_rank: int,
        locator_margin: float,
        *,
        uncertainty_weight: float = 0.0,
    ) -> float:
        if challenger_rank < self.cutoff or challenger_rank >= self.candidate_k:
            raise ValueError("challenger rank lies outside the modeled tail")
        if not math.isfinite(locator_margin):
            raise ValueError("locator margin must be finite")
        if not math.isfinite(uncertainty_weight) or uncertainty_weight < 0:
            raise ValueError("uncertainty weight must be finite and non-negative")
        row = challenger_rank - self.cutoff
        column = int(
            np.searchsorted(self.margin_edges[row], locator_margin, side="right")
        )
        return float(
            np.clip(
                self.probability[row, column]
                + uncertainty_weight * self.uncertainty[row, column],
                0.0,
                1.0,
            )
        )


def _allocate_group_samples(
    group_sizes: dict[str, int],
    target: int,
) -> dict[str, int]:
    total = sum(group_sizes.values())
    raw = {group: target * size / total for group, size in group_sizes.items()}
    allocation = {
        group: min(size, int(math.floor(raw[group])))
        for group, size in group_sizes.items()
    }
    remaining = target - sum(allocation.values())
    order = sorted(
        group_sizes,
        key=lambda group: (
            -(raw[group] - math.floor(raw[group])),
            group,
        ),
    )
    for group in order:
        if not remaining:
            break
        if allocation[group] < group_sizes[group]:
            allocation[group] += 1
            remaining -= 1
    return allocation


def select_sparse_probe_queries(
    locator_zscores: np.ndarray,
    *,
    cutoff: int,
    fraction: float,
    groups: Sequence[str] | None = None,
) -> SparseProbePlan:
    """Select deterministic, group-balanced probes across boundary margins."""

    locator = np.asarray(locator_zscores, dtype=np.float64)
    if locator.ndim != 2:
        raise ValueError("locator scores must be a 2-D matrix")
    if cutoff <= 0 or cutoff >= locator.shape[1]:
        raise ValueError("cutoff must lie inside the candidate cohort")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("probe fraction must lie in (0, 1]")
    if np.any(~np.isfinite(locator)):
        raise ValueError("locator scores must be finite")
    labels = (
        np.asarray(["all"] * len(locator), dtype=str)
        if groups is None
        else np.asarray([str(value) for value in groups])
    )
    if labels.shape != (len(locator),):
        raise ValueError("probe groups must match the query count")

    target = max(1, int(math.ceil(fraction * len(locator))))
    sizes = {group: int(np.sum(labels == group)) for group in sorted(set(labels))}
    allocation = _allocate_group_samples(sizes, target)
    boundary_margin = locator[:, cutoff - 1] - locator[:, cutoff]
    selected: list[int] = []
    for group in sorted(allocation):
        count = allocation[group]
        if count == 0:
            continue
        positions = np.flatnonzero(labels == group)
        ordered = sorted(
            (int(position) for position in positions),
            key=lambda position: (float(boundary_margin[position]), position),
        )
        # Evenly cover the locator-confidence range instead of probing only
        # the most ambiguous queries.  This makes the fitted risk table usable
        # under workload shifts in query difficulty.
        offsets = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
        selected.extend(ordered[int(offset)] for offset in offsets)
    indices = np.asarray(sorted(set(selected)), dtype=np.int32)
    if len(indices) != target:
        raise AssertionError("deterministic probe allocation lost query slots")
    return SparseProbePlan(
        query_indices=indices,
        query_fraction=len(indices) / len(locator),
        candidate_events=len(indices) * locator.shape[1],
    )


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=1, keepdims=True)
    stds = np.maximum(values.std(axis=1, keepdims=True), 1e-12)
    return (values - means) / stds


def fit_sparse_boundary_risk(
    locator_zscores: np.ndarray,
    raw_visual_scores: np.ndarray,
    *,
    cutoff: int,
    margin_bins: int = 4,
    prior_strength: float = 4.0,
) -> SparseBoundaryRiskModel:
    """Fit boundary-crossing probability from probed candidate cohorts only."""

    locator = np.asarray(locator_zscores, dtype=np.float64)
    visual = np.asarray(raw_visual_scores, dtype=np.float64)
    if locator.shape != visual.shape or locator.ndim != 2:
        raise ValueError("locator and probed visual scores must align in 2-D")
    if not len(locator):
        raise ValueError("at least one probed query is required")
    if cutoff <= 0 or cutoff >= locator.shape[1]:
        raise ValueError("cutoff must lie inside the candidate cohort")
    if margin_bins <= 0:
        raise ValueError("margin bin count must be positive")
    if not math.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("prior strength must be finite and positive")

    candidate_k = locator.shape[1]
    margins = locator[:, cutoff - 1, None] - locator[:, cutoff:]
    quantiles = np.linspace(0.0, 1.0, margin_bins + 1)[1:-1]
    edges = (
        np.stack(
            [np.quantile(margins[:, rank], quantiles) for rank in range(margins.shape[1])]
        )
        if len(quantiles)
        else np.empty((margins.shape[1], 0), dtype=np.float64)
    )
    bins = margin_bins

    fused = locator + _zscore_rows(visual)
    teacher_order = np.argsort(-fused, axis=1, kind="stable")
    membership = np.zeros_like(locator, dtype=bool)
    rows = np.arange(len(locator))[:, None]
    membership[rows, teacher_order[:, :cutoff]] = True
    tail_labels = membership[:, cutoff:]
    global_rate = float(np.mean(tail_labels))
    prior_alpha = max(global_rate * prior_strength, 1e-6)
    prior_beta = max((1.0 - global_rate) * prior_strength, 1e-6)

    successes = np.zeros((candidate_k - cutoff, bins), dtype=np.float64)
    counts = np.zeros_like(successes)
    for query in range(len(locator)):
        incumbent = locator[query, cutoff - 1]
        for rank in range(cutoff, candidate_k):
            row = rank - cutoff
            column = int(
                np.searchsorted(
                    edges[row],
                    incumbent - locator[query, rank],
                    side="right",
                )
            )
            counts[row, column] += 1.0
            successes[row, column] += float(tail_labels[query, row])

    alpha = successes + prior_alpha
    beta = counts - successes + prior_beta
    probability = alpha / (alpha + beta)
    variance = alpha * beta / (
        (alpha + beta) ** 2 * (alpha + beta + 1.0)
    )
    return SparseBoundaryRiskModel(
        cutoff=cutoff,
        candidate_k=candidate_k,
        margin_edges=np.asarray(edges, dtype=np.float64),
        probability=probability,
        uncertainty=np.sqrt(variance),
        observations=counts.astype(np.int32),
        probed_queries=len(locator),
        probed_candidate_events=locator.size,
    )


def build_estimated_boundary_pairs(
    candidate_pages: np.ndarray,
    locator_zscores: np.ndarray,
    model: SparseBoundaryRiskModel,
    *,
    uncertainty_weight: float = 0.0,
) -> tuple[BoundaryPair, ...]:
    """Compile a new workload using only locator-visible state and a model."""

    candidates = np.asarray(candidate_pages, dtype=np.int64)
    locator = np.asarray(locator_zscores, dtype=np.float64)
    if candidates.shape != locator.shape or candidates.ndim != 2:
        raise ValueError("candidate pages and locator scores must align in 2-D")
    if candidates.shape[1] != model.candidate_k:
        raise ValueError("candidate cohort does not match the risk model")
    pairs: list[BoundaryPair] = []
    incumbent_rank = model.cutoff - 1
    for query in range(len(candidates)):
        incumbent = int(candidates[query, incumbent_rank])
        incumbent_score = float(locator[query, incumbent_rank])
        for rank in range(model.cutoff, model.candidate_k):
            challenger = int(candidates[query, rank])
            margin = incumbent_score - float(locator[query, rank])
            weight = model.predict(
                rank,
                margin,
                uncertainty_weight=uncertainty_weight,
            )
            if weight <= 0:
                continue
            pairs.append(
                BoundaryPair(
                    query_index=query,
                    incumbent=incumbent,
                    challenger=challenger,
                    challenger_rank=rank,
                    weight=weight,
                )
            )
    return tuple(pairs)
