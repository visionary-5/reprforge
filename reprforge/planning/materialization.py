"""Cost-aware materialization for versioned index maintenance."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class UpdateScenario:
    """One component update expected during a planning horizon."""

    name: str
    changed_components: frozenset[str]
    expected_count: float = 1.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("an update scenario needs a name")
        if not self.changed_components:
            raise ValueError("an update scenario must change at least one component")
        if not math.isfinite(self.expected_count) or self.expected_count < 0:
            raise ValueError("expected update count must be finite and non-negative")


@dataclass(frozen=True)
class MaterializationOption:
    """A reusable artifact and the cost of replaying from its boundary.

    ``depends_on`` contains the components already compiled into the artifact.
    The artifact is valid only when an update changes none of those components.
    Storage is incremental to the source corpus and current terminal index.
    """

    name: str
    depends_on: frozenset[str]
    storage_bytes: int
    replay_seconds: float
    materialization_seconds: float = 0.0
    quality_fraction: float = 1.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("a materialization option needs a name")
        if not self.depends_on:
            raise ValueError("an artifact must declare its compiled dependencies")
        if self.storage_bytes < 0:
            raise ValueError("artifact storage must be non-negative")
        for value, label in (
            (self.replay_seconds, "replay time"),
            (self.materialization_seconds, "materialization time"),
            (self.quality_fraction, "quality fraction"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.quality_fraction > 1:
            raise ValueError("quality fraction cannot exceed one")

    def remains_valid(self, update: UpdateScenario) -> bool:
        """Return whether this artifact survives an update exactly as stored."""

        return self.depends_on.isdisjoint(update.changed_components)


@dataclass(frozen=True)
class UpdateRoute:
    """The cheapest valid rebuild source selected for one update scenario."""

    update: str
    source: str
    seconds_per_update: float
    expected_count: float

    @property
    def expected_seconds(self) -> float:
        return self.seconds_per_update * self.expected_count


@dataclass(frozen=True)
class MaterializationDecision:
    """Minimum-cost artifact portfolio and its per-update execution routes."""

    selected: tuple[str, ...]
    routes: tuple[UpdateRoute, ...]
    storage_bytes: int
    materialization_seconds: float
    expected_seconds: float
    raw_baseline_seconds: float

    @property
    def saving_fraction(self) -> float:
        if self.raw_baseline_seconds == 0:
            return 0.0
        return 1.0 - self.expected_seconds / self.raw_baseline_seconds


def choose_materializations(
    options: tuple[MaterializationOption, ...],
    updates: tuple[UpdateScenario, ...],
    *,
    raw_rebuild_seconds: float,
    storage_budget_bytes: int,
    minimum_quality_fraction: float = 0.99,
) -> MaterializationDecision:
    """Select the minimum expected-cost valid artifact portfolio.

    ReprForge evaluates all feasible subsets because a model exposes only a
    small number of semantically meaningful boundaries. For each update it
    routes execution from the cheapest selected artifact that remains valid;
    otherwise it falls back to a raw rebuild.
    """

    if not math.isfinite(raw_rebuild_seconds) or raw_rebuild_seconds < 0:
        raise ValueError("raw rebuild time must be finite and non-negative")
    if storage_budget_bytes < 0:
        raise ValueError("storage budget must be non-negative")
    if not 0 < minimum_quality_fraction <= 1:
        raise ValueError("minimum quality fraction must be in (0, 1]")
    if len(options) > 20:
        raise ValueError(
            "exhaustive materialization planning supports at most 20 options"
        )
    names = [option.name for option in options]
    if len(names) != len(set(names)):
        raise ValueError("materialization option names must be unique")
    for option in options:
        option.validate()
    for update in updates:
        update.validate()

    raw_baseline = raw_rebuild_seconds * sum(
        update.expected_count for update in updates
    )
    best: MaterializationDecision | None = None
    for count in range(len(options) + 1):
        for selected in itertools.combinations(options, count):
            storage = sum(option.storage_bytes for option in selected)
            if storage > storage_budget_bytes:
                continue
            materialization = sum(option.materialization_seconds for option in selected)
            routes = []
            maintenance = 0.0
            for update in updates:
                valid = [
                    option
                    for option in selected
                    if option.quality_fraction >= minimum_quality_fraction
                    and option.remains_valid(update)
                ]
                if valid:
                    source = min(
                        valid, key=lambda item: (item.replay_seconds, item.name)
                    )
                    route = UpdateRoute(
                        update.name,
                        source.name,
                        source.replay_seconds,
                        update.expected_count,
                    )
                else:
                    route = UpdateRoute(
                        update.name,
                        "raw",
                        raw_rebuild_seconds,
                        update.expected_count,
                    )
                routes.append(route)
                maintenance += route.expected_seconds
            decision = MaterializationDecision(
                selected=tuple(option.name for option in selected),
                routes=tuple(routes),
                storage_bytes=storage,
                materialization_seconds=materialization,
                expected_seconds=materialization + maintenance,
                raw_baseline_seconds=raw_baseline,
            )
            if best is None or (
                decision.expected_seconds,
                decision.storage_bytes,
                len(decision.selected),
                decision.selected,
            ) < (
                best.expected_seconds,
                best.storage_bytes,
                len(best.selected),
                best.selected,
            ):
                best = decision
    if best is None:
        raise RuntimeError("no materialization portfolio satisfies the storage budget")
    return best
