"""Model and workload admission for representation lifecycles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Lifecycle(str, Enum):
    FULL = "full"
    COMPACT = "compact"
    COMPACT_REFINE = "compact_refine"


@dataclass(frozen=True)
class BackboneProfile:
    """Model properties required by an index-side compiler."""

    name: str
    total_layers: int
    split_after_layer: int
    full_visual_tokens: int
    compact_visual_tokens: int
    exposes_hidden_boundary: bool = True
    exposes_visual_topology: bool = True
    query_independent_documents: bool = True

    @property
    def persistent_fraction(self) -> float:
        return self.compact_visual_tokens / self.full_visual_tokens

    def admission_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.total_layers < 2:
            failures.append("the backbone has no separable prefix and suffix")
        if not 0 < self.split_after_layer < self.total_layers:
            failures.append("the split must leave both prefix and suffix layers")
        if not 0 < self.compact_visual_tokens < self.full_visual_tokens:
            failures.append("compact visual capacity must be smaller than Full")
        if not self.exposes_hidden_boundary:
            failures.append("no stable hidden-state boundary is exposed")
        if not self.exposes_visual_topology:
            failures.append("visual token topology is unavailable")
        if not self.query_independent_documents:
            failures.append("document encoding depends on a live query")
        return tuple(failures)

    def validate(self) -> None:
        failures = self.admission_failures()
        if failures:
            raise ValueError("; ".join(failures))


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
