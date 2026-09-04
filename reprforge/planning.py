"""Cost-aware selection of the rebuild source for versioned index updates.

Vocabulary
----------
* An :class:`UpdateScenario` names the version components an expected upgrade
  changes and how often it is expected within the planning horizon.
* A :class:`MaterializationOption` is a candidate *semantic recompilation cut*:
  a stored intermediate state, the components already compiled into it, the
  storage it costs to keep and the time it takes to replay the target suffix
  from it. Its ``quality_fraction`` is the retrieval-quality admission
  measurement for a lossy cut.
* :func:`choose_materializations` selects the portfolio of cuts that minimises
  expected upgrade time under a storage budget and a quality floor, routing
  each scenario to the cheapest cut that remains dependency-valid and falling
  back to a raw rebuild otherwise.

Two cheap predictors accompany the planner. :func:`cut_leverage` is the share
of raw encoding time spent before a cut, which bounds any replay saving before
a codec exists. :func:`break_even_upgrades` turns storage and GPU prices into
the number of upgrades needed for a retained cut to pay for itself.
"""

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
    validation_seconds: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("an update scenario needs a name")
        if not self.changed_components:
            raise ValueError("an update scenario must change at least one component")
        if not math.isfinite(self.expected_count) or self.expected_count < 0:
            raise ValueError("expected update count must be finite and non-negative")
        if not math.isfinite(self.validation_seconds) or self.validation_seconds < 0:
            raise ValueError("validation time must be finite and non-negative")


@dataclass(frozen=True)
class MaterializationOption:
    """A candidate cut and the cost of replaying the target suffix from it.

    ``depends_on`` lists the components already compiled into the stored state;
    the cut is legal for an update only when the update changes none of them.
    ``storage_bytes`` is incremental to the collection and the active index.
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
        """Return whether this cut survives an update exactly as stored."""

        return self.depends_on.isdisjoint(update.changed_components)


@dataclass(frozen=True)
class UpdateRoute:
    """The rebuild source selected for one update scenario."""

    update: str
    source: str
    seconds_per_update: float
    expected_count: float
    validation_seconds: float = 0.0

    @property
    def expected_seconds(self) -> float:
        return self.seconds_per_update * self.expected_count


@dataclass(frozen=True)
class MaterializationDecision:
    """Minimum-cost cut portfolio and its per-update routes."""

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


def evaluate_materializations(
    selected_names: tuple[str, ...],
    options: tuple[MaterializationOption, ...],
    updates: tuple[UpdateScenario, ...],
    *,
    raw_rebuild_seconds: float,
    minimum_quality_fraction: float = 0.99,
) -> MaterializationDecision:
    """Score a fixed portfolio with measured costs.

    Planning and execution measurements can differ: select a portfolio with one
    cost profile, then score the unchanged selection with a held-out profile
    that uses the same cut names and dependency sets.
    """

    if not math.isfinite(raw_rebuild_seconds) or raw_rebuild_seconds < 0:
        raise ValueError("raw rebuild time must be finite and non-negative")
    if not 0 < minimum_quality_fraction <= 1:
        raise ValueError("minimum quality fraction must be in (0, 1]")
    names = [option.name for option in options]
    if len(names) != len(set(names)):
        raise ValueError("materialization option names must be unique")
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("selected materialization names must be unique")
    by_name = {option.name: option for option in options}
    unknown = sorted(set(selected_names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown materialization options: {', '.join(unknown)}")
    for option in options:
        option.validate()
    for update in updates:
        update.validate()

    selected = tuple(by_name[name] for name in selected_names)
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
            source = min(valid, key=lambda item: (item.replay_seconds, item.name))
            reused_seconds = source.replay_seconds + update.validation_seconds
        else:
            source = None
            reused_seconds = math.inf
        if source is not None and reused_seconds < raw_rebuild_seconds:
            route = UpdateRoute(
                update.name,
                source.name,
                reused_seconds,
                update.expected_count,
                update.validation_seconds,
            )
        else:
            route = UpdateRoute(
                update.name, "raw", raw_rebuild_seconds, update.expected_count, 0.0
            )
        routes.append(route)
        maintenance += route.expected_seconds
    materialization = sum(option.materialization_seconds for option in selected)
    raw_baseline = raw_rebuild_seconds * sum(
        update.expected_count for update in updates
    )
    return MaterializationDecision(
        selected=selected_names,
        routes=tuple(routes),
        storage_bytes=sum(option.storage_bytes for option in selected),
        materialization_seconds=materialization,
        expected_seconds=materialization + maintenance,
        raw_baseline_seconds=raw_baseline,
    )


def choose_materializations(
    options: tuple[MaterializationOption, ...],
    updates: tuple[UpdateScenario, ...],
    *,
    raw_rebuild_seconds: float,
    storage_budget_bytes: int,
    minimum_quality_fraction: float = 0.99,
) -> MaterializationDecision:
    """Select the minimum expected-cost portfolio of cuts.

    All feasible subsets are evaluated because an encoder exposes only a handful
    of semantically meaningful cuts. Each update is routed to the cheapest
    selected cut that remains valid, otherwise to a raw rebuild.
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

    best: MaterializationDecision | None = None
    for count in range(len(options) + 1):
        for selected in itertools.combinations(options, count):
            storage = sum(option.storage_bytes for option in selected)
            if storage > storage_budget_bytes:
                continue
            decision = evaluate_materializations(
                tuple(option.name for option in selected),
                options,
                updates,
                raw_rebuild_seconds=raw_rebuild_seconds,
                minimum_quality_fraction=minimum_quality_fraction,
            )
            key = (
                decision.expected_seconds,
                decision.storage_bytes,
                len(decision.selected),
                decision.selected,
            )
            if best is None or key < (
                best.expected_seconds,
                best.storage_bytes,
                len(best.selected),
                best.selected,
            ):
                best = decision
    if best is None:
        raise RuntimeError("no materialization portfolio satisfies the storage budget")
    return best


def cut_leverage(
    *, preprocess_seconds: float, prefix_seconds: float, suffix_seconds: float
) -> float:
    """Share of raw encoding time spent before the cut.

    Measured on a sample of pages with the same batching as production, this
    upper-bounds the fraction of encode time any replay from that cut can save.
    A backbone with low leverage (for example a small vision tower in front of a
    large decoder) should be routed to raw rebuilding without fitting a codec.
    """

    for value, label in (
        (preprocess_seconds, "preprocess time"),
        (prefix_seconds, "prefix time"),
        (suffix_seconds, "suffix time"),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and non-negative")
    total = preprocess_seconds + prefix_seconds + suffix_seconds
    if total <= 0:
        raise ValueError("stage times must not all be zero")
    return (preprocess_seconds + prefix_seconds) / total


def break_even_upgrades(
    *,
    saved_seconds_per_upgrade: float,
    gpu_usd_per_hour: float,
    retained_gb: float,
    storage_usd_per_gb_month: float,
    horizon_months: float,
) -> float:
    """Upgrades within the horizon needed for a retained cut to pay for itself.

    Storage is charged for the whole horizon; each upgrade saves the GPU time
    the cut avoids. A result below one means a single upgrade already pays.
    """

    for value, label in (
        (saved_seconds_per_upgrade, "saved seconds"),
        (gpu_usd_per_hour, "GPU price"),
        (retained_gb, "retained gigabytes"),
        (storage_usd_per_gb_month, "storage price"),
        (horizon_months, "horizon"),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and non-negative")
    saving_per_upgrade = saved_seconds_per_upgrade / 3600.0 * gpu_usd_per_hour
    storage_cost = retained_gb * storage_usd_per_gb_month * horizon_months
    if saving_per_upgrade == 0:
        return math.inf if storage_cost > 0 else 0.0
    return storage_cost / saving_per_upgrade
