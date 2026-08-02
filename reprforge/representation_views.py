"""Control plane for hypothetical multimodal representation views.

The module transfers one database idea into ReprForge: an expensive physical
structure should be proposed, probed and verified before it becomes visible to
queries.  A candidate representation view therefore starts as metadata only.
Its lifecycle is deliberately separate from the encoder and physical index so
that experiments can distinguish planning overhead, verification work and
committed representation cost.

This is a systems contract, not the final benefit estimator.  Callers supply
the expected and observed utility values; the catalog enforces budgets, legal
state transitions, deterministic selection and durable snapshots.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CATALOG_FORMAT = "reprforge-representation-view-catalog"
CATALOG_VERSION = 1


class ViewState(str, Enum):
    HYPOTHETICAL = "hypothetical"
    PROBING = "probing"
    VERIFIED = "verified"
    MATERIALIZING = "materializing"
    MATERIALIZED = "materialized"
    REJECTED = "rejected"


@dataclass(frozen=True, order=True)
class ViewKey:
    item_id: str
    route: str

    def __post_init__(self) -> None:
        if not self.item_id or not self.route:
            raise ValueError("view item and route identifiers must be non-empty")

    @property
    def identifier(self) -> str:
        return f"{self.item_id}::{self.route}"


@dataclass(frozen=True)
class CandidateView:
    """Metadata and measured state for one derived retrieval representation."""

    key: ViewKey
    slot: str
    parent_route: str
    expected_utility: float
    uncertainty: float
    expected_reuse: float
    probe_cost_ms: float
    build_cost_ms: float
    storage_bytes: int
    maintenance_cost_ms: float = 0.0
    state: ViewState = ViewState.HYPOTHETICAL
    observed_utility: float | None = None
    actual_probe_cost_ms: float | None = None
    probe_artifact_reusable: bool = False
    materialized_version: int | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.slot or not self.parent_route:
            raise ValueError("view slot and parent route must be non-empty")
        finite = (
            self.expected_utility,
            self.uncertainty,
            self.expected_reuse,
            self.probe_cost_ms,
            self.build_cost_ms,
            self.maintenance_cost_ms,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("view estimates and costs must be finite")
        if self.uncertainty < 0 or self.expected_reuse < 0:
            raise ValueError("uncertainty and reuse cannot be negative")
        if self.probe_cost_ms < 0 or self.build_cost_ms < 0:
            raise ValueError("probe and build costs cannot be negative")
        if self.storage_bytes < 0 or self.maintenance_cost_ms < 0:
            raise ValueError("storage and maintenance costs cannot be negative")
        if self.observed_utility is not None and not math.isfinite(
            self.observed_utility
        ):
            raise ValueError("observed utility must be finite")
        if self.actual_probe_cost_ms is not None and self.actual_probe_cost_ms < 0:
            raise ValueError("actual probe cost cannot be negative")
        if self.revision < 0:
            raise ValueError("view revision cannot be negative")

    @property
    def remaining_build_cost_ms(self) -> float:
        if self.probe_artifact_reusable and self.actual_probe_cost_ms is not None:
            return max(0.0, self.build_cost_ms - self.actual_probe_cost_ms)
        return self.build_cost_ms


@dataclass(frozen=True)
class ProbePlan:
    keys: tuple[ViewKey, ...]
    estimated_cost_ms: float
    budget_ms: float


@dataclass(frozen=True)
class MaterializationPlan:
    keys: tuple[ViewKey, ...]
    estimated_build_ms: float
    storage_bytes: int
    build_budget_ms: float
    storage_budget_bytes: int


def _probe_priority(view: CandidateView, exploration_weight: float) -> float:
    optimistic = view.expected_utility + exploration_weight * view.uncertainty
    value = optimistic * max(view.expected_reuse, 1.0)
    return value / max(view.probe_cost_ms, 1e-9)


def _materialization_priority(view: CandidateView) -> float:
    if view.observed_utility is None:
        raise ValueError("verified views require observed utility")
    value = view.observed_utility * max(view.expected_reuse, 1.0)
    lifecycle_cost = (
        view.remaining_build_cost_ms
        + view.maintenance_cost_ms
        + view.storage_bytes / 1_000_000.0
    )
    return value / max(lifecycle_cost, 1e-9)


class RepresentationViewCatalog:
    """Deterministic single-writer catalog for candidate representation views."""

    def __init__(self, views: Iterable[CandidateView] = ()) -> None:
        self._views: dict[ViewKey, CandidateView] = {}
        for view in views:
            self.propose(view)

    def __len__(self) -> int:
        return len(self._views)

    def get(self, key: ViewKey) -> CandidateView:
        try:
            return self._views[key]
        except KeyError as error:
            raise KeyError(f"unknown representation view {key.identifier}") from error

    def views(self, *, state: ViewState | None = None) -> tuple[CandidateView, ...]:
        return tuple(
            view
            for key, view in sorted(self._views.items())
            if state is None or view.state == state
        )

    def propose(self, view: CandidateView) -> None:
        if view.state != ViewState.HYPOTHETICAL:
            raise ValueError("newly proposed views must be hypothetical")
        if view.key in self._views:
            raise ValueError(f"duplicate representation view {view.key.identifier}")
        self._views[view.key] = view

    def plan_probes(
        self,
        *,
        budget_ms: float,
        exploration_weight: float = 1.0,
        max_views: int | None = None,
    ) -> ProbePlan:
        if not math.isfinite(budget_ms) or budget_ms < 0:
            raise ValueError("probe budget must be finite and non-negative")
        if not math.isfinite(exploration_weight) or exploration_weight < 0:
            raise ValueError("exploration weight must be finite and non-negative")
        if max_views is not None and max_views <= 0:
            raise ValueError("max_views must be positive when provided")
        candidates = sorted(
            self.views(state=ViewState.HYPOTHETICAL),
            key=lambda view: (
                -_probe_priority(view, exploration_weight),
                view.key,
            ),
        )
        selected: list[ViewKey] = []
        cost = 0.0
        for view in candidates:
            if max_views is not None and len(selected) >= max_views:
                break
            if cost + view.probe_cost_ms > budget_ms + 1e-9:
                continue
            selected.append(view.key)
            cost += view.probe_cost_ms
        return ProbePlan(tuple(selected), cost, budget_ms)

    def begin_probes(self, plan: ProbePlan) -> None:
        for key in plan.keys:
            view = self.get(key)
            if view.state != ViewState.HYPOTHETICAL:
                raise ValueError(
                    f"cannot probe {key.identifier} from state {view.state.value}"
                )
        for key in plan.keys:
            view = self.get(key)
            self._views[key] = replace(
                view,
                state=ViewState.PROBING,
                revision=view.revision + 1,
            )

    def finish_probe(
        self,
        key: ViewKey,
        *,
        observed_utility: float,
        actual_cost_ms: float,
        minimum_utility: float,
        artifact_reusable: bool,
    ) -> ViewState:
        view = self.get(key)
        if view.state != ViewState.PROBING:
            raise ValueError(
                f"cannot finish probe {key.identifier} from state {view.state.value}"
            )
        if not math.isfinite(observed_utility):
            raise ValueError("observed utility must be finite")
        if not math.isfinite(actual_cost_ms) or actual_cost_ms < 0:
            raise ValueError("actual probe cost must be finite and non-negative")
        if not math.isfinite(minimum_utility):
            raise ValueError("minimum utility must be finite")
        state = (
            ViewState.VERIFIED
            if observed_utility >= minimum_utility
            else ViewState.REJECTED
        )
        self._views[key] = replace(
            view,
            state=state,
            observed_utility=observed_utility,
            actual_probe_cost_ms=actual_cost_ms,
            probe_artifact_reusable=artifact_reusable,
            revision=view.revision + 1,
        )
        return state

    def plan_materialization(
        self,
        *,
        build_budget_ms: float,
        storage_budget_bytes: int,
    ) -> MaterializationPlan:
        if not math.isfinite(build_budget_ms) or build_budget_ms < 0:
            raise ValueError("build budget must be finite and non-negative")
        if storage_budget_bytes < 0:
            raise ValueError("storage budget must be non-negative")
        candidates = sorted(
            self.views(state=ViewState.VERIFIED),
            key=lambda view: (-_materialization_priority(view), view.key),
        )
        selected: list[ViewKey] = []
        build_cost = 0.0
        storage_cost = 0
        occupied_slots: set[tuple[str, str]] = set()
        for view in candidates:
            slot = (view.key.item_id, view.slot)
            if slot in occupied_slots:
                continue
            next_build = build_cost + view.remaining_build_cost_ms
            next_storage = storage_cost + view.storage_bytes
            if next_build > build_budget_ms + 1e-9:
                continue
            if next_storage > storage_budget_bytes:
                continue
            selected.append(view.key)
            occupied_slots.add(slot)
            build_cost = next_build
            storage_cost = next_storage
        return MaterializationPlan(
            keys=tuple(selected),
            estimated_build_ms=build_cost,
            storage_bytes=storage_cost,
            build_budget_ms=build_budget_ms,
            storage_budget_bytes=storage_budget_bytes,
        )

    def begin_materialization(self, plan: MaterializationPlan) -> None:
        for key in plan.keys:
            view = self.get(key)
            if view.state != ViewState.VERIFIED:
                raise ValueError(
                    f"cannot materialize {key.identifier} from state {view.state.value}"
                )
        for key in plan.keys:
            view = self.get(key)
            self._views[key] = replace(
                view,
                state=ViewState.MATERIALIZING,
                revision=view.revision + 1,
            )

    def finish_materialization(
        self,
        key: ViewKey,
        *,
        version: int | None,
    ) -> None:
        view = self.get(key)
        if view.state != ViewState.MATERIALIZING:
            raise ValueError(
                f"cannot publish {key.identifier} from state {view.state.value}"
            )
        succeeded = version is not None
        if version is not None and version <= 0:
            raise ValueError("materialized version must be positive")
        self._views[key] = replace(
            view,
            state=ViewState.MATERIALIZED if succeeded else ViewState.VERIFIED,
            materialized_version=version if succeeded else None,
            revision=view.revision + 1,
        )

    def recover_interrupted_work(self) -> int:
        """Return in-flight views to their last durable schedulable state."""

        recovered = 0
        for key, view in tuple(self._views.items()):
            if view.state == ViewState.PROBING:
                target = ViewState.HYPOTHETICAL
            elif view.state == ViewState.MATERIALIZING:
                target = ViewState.VERIFIED
            else:
                continue
            self._views[key] = replace(
                view,
                state=target,
                revision=view.revision + 1,
            )
            recovered += 1
        return recovered

    def summary(self) -> dict[str, object]:
        counts = {state.value: 0 for state in ViewState}
        for view in self._views.values():
            counts[view.state.value] += 1
        materialized = self.views(state=ViewState.MATERIALIZED)
        return {
            "views": len(self),
            "states": counts,
            "materialized_storage_bytes": sum(
                view.storage_bytes for view in materialized
            ),
            "materialized_observed_utility": sum(
                float(view.observed_utility or 0.0)
                * max(view.expected_reuse, 1.0)
                for view in materialized
            ),
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "format": CATALOG_FORMAT,
            "format_version": CATALOG_VERSION,
            "views": [
                {
                    **asdict(view),
                    "key": asdict(view.key),
                    "state": view.state.value,
                }
                for view in self.views()
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> "RepresentationViewCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != CATALOG_FORMAT:
            raise ValueError(f"invalid representation view catalog at {path}")
        if int(payload.get("format_version", -1)) != CATALOG_VERSION:
            raise ValueError("unsupported representation view catalog version")
        views = []
        for row in payload.get("views", []):
            values = dict(row)
            values["key"] = ViewKey(**values["key"])
            values["state"] = ViewState(values["state"])
            views.append(CandidateView(**values))
        catalog = cls()
        catalog._views = {view.key: view for view in views}
        if len(catalog._views) != len(views):
            raise ValueError("catalog contains duplicate view identifiers")
        return catalog


def apply_materialization_plan(
    catalog: RepresentationViewCatalog,
    plan: MaterializationPlan,
    *,
    versions: Mapping[ViewKey, int | None],
) -> None:
    """Publish executor results only after the complete result set is known."""

    missing = set(plan.keys) - set(versions)
    extra = set(versions) - set(plan.keys)
    if missing or extra:
        raise ValueError(
            f"materialization results differ from plan: missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    invalid_versions = [
        version
        for version in versions.values()
        if version is not None and version <= 0
    ]
    if invalid_versions:
        raise ValueError("materialized versions must be positive")
    catalog.begin_materialization(plan)
    for key in plan.keys:
        catalog.finish_materialization(key, version=versions[key])


def view_keys(identifiers: Sequence[str], route: str) -> tuple[ViewKey, ...]:
    return tuple(ViewKey(str(identifier), route) for identifier in identifiers)
