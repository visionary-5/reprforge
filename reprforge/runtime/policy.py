"""Measured workload admission for compiled representation lifecycles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Lifecycle(str, Enum):
    FULL = "full"
    COMPACT = "compact"
    COMPACT_REFINE = "compact_refine"


@dataclass(frozen=True)
class WorkloadProfile:
    """Measured costs and qualities for one rebuild epoch."""

    full_build_seconds: float
    compact_build_seconds: float
    refinement_seconds_per_cold_query: float
    cold_queries_per_build: float
    full_quality: float
    compact_quality: float
    refined_quality: float
    compact_storage_fraction: float

    def validate(self) -> None:
        values = (
            self.full_build_seconds,
            self.compact_build_seconds,
            self.refinement_seconds_per_cold_query,
            self.cold_queries_per_build,
            self.full_quality,
            self.compact_quality,
            self.refined_quality,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("workload measurements must be finite and non-negative")
        if self.full_quality <= 0:
            raise ValueError("Full quality must be positive")
        if not 0 < self.compact_storage_fraction <= 1:
            raise ValueError("compact storage fraction must be in (0, 1]")

    @property
    def refinement_break_even_queries(self) -> float:
        self.validate()
        saved = self.full_build_seconds - self.compact_build_seconds
        if saved <= 0:
            return 0.0
        if self.refinement_seconds_per_cold_query == 0:
            return math.inf
        return saved / self.refinement_seconds_per_cold_query


@dataclass(frozen=True)
class LifecycleDecision:
    lifecycle: Lifecycle
    epoch_seconds: float
    quality_fraction: float
    storage_fraction: float


def choose_lifecycle(
    workload: WorkloadProfile,
    *,
    minimum_quality_fraction: float = 0.99,
    maximum_storage_fraction: float = 1.0,
) -> LifecycleDecision:
    """Choose the cheapest measured lifecycle inside quality/storage bounds."""

    workload.validate()
    if not 0 < minimum_quality_fraction <= 1:
        raise ValueError("minimum quality fraction must be in (0, 1]")
    if not 0 < maximum_storage_fraction <= 1:
        raise ValueError("maximum storage fraction must be in (0, 1]")
    candidates = (
        LifecycleDecision(Lifecycle.FULL, workload.full_build_seconds, 1.0, 1.0),
        LifecycleDecision(
            Lifecycle.COMPACT,
            workload.compact_build_seconds,
            workload.compact_quality / workload.full_quality,
            workload.compact_storage_fraction,
        ),
        LifecycleDecision(
            Lifecycle.COMPACT_REFINE,
            workload.compact_build_seconds
            + workload.cold_queries_per_build
            * workload.refinement_seconds_per_cold_query,
            workload.refined_quality / workload.full_quality,
            workload.compact_storage_fraction,
        ),
    )
    feasible = [
        decision
        for decision in candidates
        if decision.quality_fraction >= minimum_quality_fraction
        and decision.storage_fraction <= maximum_storage_fraction
    ]
    if not feasible:
        raise ValueError("no lifecycle satisfies the quality/storage contract")
    return min(
        feasible,
        key=lambda decision: (
            decision.epoch_seconds,
            decision.storage_fraction,
            decision.lifecycle.value,
        ),
    )
