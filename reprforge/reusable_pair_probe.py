"""Progressive pair probes whose artifacts are the final physical index.

The earlier sparse controller paid twice: it constructed complete visual
cohorts to learn a risk model, then constructed a different selected index.
This module makes probing a physical-design action.  Selecting an
incumbent--challenger comparison materializes only missing endpoint pages.
Those page representations immediately become resident views and expose every
other boundary edge whose endpoints are now available.

The selector begins with a distribution-free normal prior in normalized score
space.  As pair deltas become observable, it updates the survival probability
of unresolved locator margins.  Observations influence later choices, but no
unmaterialized score can be read through the provider contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from reprforge.physical_cost import (
    AtomicCostEstimate,
    AtomicMaterializationCostModel,
)
from reprforge.risk_constrained_admission import estimate_plan_cost


@dataclass(frozen=True, order=True)
class ProbeEdge:
    query_index: int
    incumbent_rank: int
    challenger_rank: int
    incumbent_page: int
    challenger_page: int
    locator_margin: float


class PairScoreProvider(Protocol):
    @property
    def materialized_pages(self) -> frozenset[int]: ...

    def materialize(self, pages: Sequence[int] | set[int]) -> None: ...

    def score(self, query_index: int, candidate_rank: int) -> float: ...


class FrozenPairScoreProvider:
    """Leakage-checking score provider for an offline frozen surface."""

    def __init__(
        self,
        candidate_pages: np.ndarray,
        raw_visual_scores: np.ndarray,
    ) -> None:
        candidates = np.asarray(candidate_pages, dtype=np.int64)
        visual = np.asarray(raw_visual_scores, dtype=np.float64)
        if candidates.shape != visual.shape or candidates.ndim != 2:
            raise ValueError("candidate pages and frozen scores must align in 2-D")
        self._candidates = candidates
        self._visual = visual
        self._materialized: set[int] = set()
        self._score_reads: set[tuple[int, int]] = set()
        self._materialization_calls = 0

    @property
    def materialized_pages(self) -> frozenset[int]:
        return frozenset(self._materialized)

    @property
    def score_reads(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._score_reads)

    @property
    def materialization_calls(self) -> int:
        return self._materialization_calls

    def materialize(self, pages: Sequence[int] | set[int]) -> None:
        normalized = {int(page) for page in pages}
        if normalized:
            self._materialization_calls += 1
            self._materialized.update(normalized)

    def score(self, query_index: int, candidate_rank: int) -> float:
        page = int(self._candidates[query_index, candidate_rank])
        if page not in self._materialized:
            raise RuntimeError("attempted to read an unmaterialized visual score")
        self._score_reads.add((int(query_index), int(candidate_rank)))
        return float(self._visual[query_index, candidate_rank])


@dataclass(frozen=True)
class ReusablePairProbePlan:
    selected_pages: frozenset[int]
    observed_edges: tuple[ProbeEdge, ...]
    observed_pair_deltas: tuple[float, ...]
    estimated_cost: AtomicCostEstimate
    time_budget_ms: float
    iterations: int
    materialization_rounds: int
    round_page_limit: int
    empirical_updates_enabled: bool

    @property
    def observed_pair_count(self) -> int:
        return len(self.observed_edges)


def build_probe_edges(
    candidate_pages: np.ndarray,
    locator_zscores: np.ndarray,
    *,
    cutoff: int,
) -> tuple[ProbeEdge, ...]:
    candidates = np.asarray(candidate_pages, dtype=np.int64)
    locator = np.asarray(locator_zscores, dtype=np.float64)
    if candidates.shape != locator.shape or candidates.ndim != 2:
        raise ValueError("candidate pages and locator scores must align in 2-D")
    if cutoff <= 0 or cutoff >= candidates.shape[1]:
        raise ValueError("cutoff must lie inside the candidate cohort")
    edges: list[ProbeEdge] = []
    incumbent_rank = cutoff - 1
    for query in range(len(candidates)):
        incumbent_page = int(candidates[query, incumbent_rank])
        incumbent_score = float(locator[query, incumbent_rank])
        for challenger_rank in range(cutoff, candidates.shape[1]):
            edges.append(
                ProbeEdge(
                    query_index=query,
                    incumbent_rank=incumbent_rank,
                    challenger_rank=challenger_rank,
                    incumbent_page=incumbent_page,
                    challenger_page=int(candidates[query, challenger_rank]),
                    locator_margin=max(
                        0.0,
                        incumbent_score - float(locator[query, challenger_rank]),
                    ),
                )
            )
    return tuple(edges)


def normal_boundary_risk(locator_margin: float) -> float:
    """P(N(0, 2) > margin) for two normalized visual scores."""

    if not math.isfinite(locator_margin) or locator_margin < 0:
        raise ValueError("locator margin must be finite and non-negative")
    return 0.5 * math.erfc(locator_margin / 2.0)


def _risk_from_observations(
    margin: float,
    deltas: Sequence[float],
    *,
    prior_strength: float,
    minimum_observations: int,
) -> float:
    prior = normal_boundary_risk(margin)
    if len(deltas) < minimum_observations:
        return prior
    values = np.asarray(deltas, dtype=np.float64)
    # For two comparable samples Var(X-Y)=2*Var(X).  The second moment keeps
    # the estimate defined even when the observed edge distribution is not
    # centered exactly at zero.
    scale = math.sqrt(max(float(np.mean(values**2)) / 2.0, 1e-12))
    normalized = values / scale
    empirical_success = float(np.sum(normalized > margin))
    return (empirical_success + prior_strength * prior) / (
        len(normalized) + prior_strength
    )


def _newly_observable_edges(
    edges: Sequence[ProbeEdge],
    observed: set[int],
    pages: set[int],
) -> list[int]:
    return [
        index
        for index, edge in enumerate(edges)
        if index not in observed
        and edge.incumbent_page in pages
        and edge.challenger_page in pages
    ]


def select_reusable_pair_probes(
    candidate_pages: np.ndarray,
    locator_zscores: np.ndarray,
    provider: PairScoreProvider,
    cost_model: AtomicMaterializationCostModel,
    *,
    cutoff: int,
    time_budget_ms: float,
    empirical_updates: bool = True,
    prior_strength: float = 2.0,
    minimum_observations: int = 4,
    round_page_limit: int = 2,
    warmup_page_limit: int = 2,
) -> ReusablePairProbePlan:
    """Materialize the most boundary information per incremental physical ms."""

    candidates = np.asarray(candidate_pages, dtype=np.int64)
    locator = np.asarray(locator_zscores, dtype=np.float64)
    if candidates.shape != locator.shape or candidates.ndim != 2:
        raise ValueError("candidate pages and locator scores must align in 2-D")
    if not math.isfinite(time_budget_ms) or time_budget_ms < 0:
        raise ValueError("time budget must be finite and non-negative")
    if not math.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("prior strength must be finite and positive")
    if minimum_observations <= 0:
        raise ValueError("minimum observations must be positive")
    if round_page_limit <= 0:
        raise ValueError("round page limit must be positive")
    if warmup_page_limit <= 0 or warmup_page_limit > round_page_limit:
        raise ValueError("warmup page limit must be in (0, round_page_limit]")
    if provider.materialized_pages:
        raise ValueError("a reusable pair-probe episode requires an empty provider")

    edges = build_probe_edges(candidates, locator, cutoff=cutoff)
    selected: set[int] = set()
    observed_indices: set[int] = set()
    observed_edges: list[ProbeEdge] = []
    observed_deltas: list[float] = []
    current_cost = cost_model.estimate(pages=0, score_events=0)
    iterations = 0
    materialization_rounds = 0

    while True:
        effective_round_limit = (
            warmup_page_limit
            if empirical_updates and len(observed_deltas) < minimum_observations
            else round_page_limit
        )
        pending_pages: set[int] = set()
        pending_observable: set[int] = set()
        projected_cost = current_cost
        while len(pending_pages) < effective_round_limit:
            working_pages = selected | pending_pages
            already_accounted = observed_indices | pending_observable
            working_cost = estimate_plan_cost(candidates, working_pages, cost_model)
            best: tuple[
                float,
                float,
                float,
                tuple[int, ...],
                set[int],
                list[int],
                AtomicCostEstimate,
            ] | None = None
            for edge in edges:
                added = {
                    edge.incumbent_page,
                    edge.challenger_page,
                } - working_pages
                if not added:
                    continue
                if (
                    pending_pages
                    and len(pending_pages | added) > effective_round_limit
                ):
                    continue
                proposed = working_pages | added
                proposed_cost = estimate_plan_cost(candidates, proposed, cost_model)
                if proposed_cost.total_ms > time_budget_ms + 1e-9:
                    continue
                newly_observable = _newly_observable_edges(
                    edges,
                    already_accounted,
                    proposed,
                )
                gain = sum(
                    _risk_from_observations(
                        edges[index].locator_margin,
                        observed_deltas if empirical_updates else (),
                        prior_strength=prior_strength,
                        minimum_observations=minimum_observations,
                    )
                    for index in newly_observable
                )
                if gain <= 0:
                    continue
                marginal_cost = max(
                    proposed_cost.total_ms - working_cost.total_ms,
                    1e-9,
                )
                identity = tuple(-page for page in sorted(added))
                candidate = (
                    gain / marginal_cost,
                    gain,
                    -proposed_cost.total_ms,
                    identity,
                    added,
                    newly_observable,
                    proposed_cost,
                )
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
            if best is None:
                break
            pending_pages.update(best[4])
            pending_observable.update(best[5])
            projected_cost = best[6]
            iterations += 1

        if not pending_pages:
            break

        provider.materialize(pending_pages)
        selected.update(pending_pages)
        newly_observable = _newly_observable_edges(
            edges,
            observed_indices,
            selected,
        )
        for index in newly_observable:
            if index in observed_indices:
                continue
            edge = edges[index]
            challenger = provider.score(edge.query_index, edge.challenger_rank)
            incumbent = provider.score(edge.query_index, edge.incumbent_rank)
            observed_indices.add(index)
            observed_edges.append(edge)
            observed_deltas.append(challenger - incumbent)
        current_cost = projected_cost
        materialization_rounds += 1

    return ReusablePairProbePlan(
        selected_pages=frozenset(selected),
        observed_edges=tuple(observed_edges),
        observed_pair_deltas=tuple(observed_deltas),
        estimated_cost=current_cost,
        time_budget_ms=time_budget_ms,
        iterations=iterations,
        materialization_rounds=materialization_rounds,
        round_page_limit=round_page_limit,
        empirical_updates_enabled=empirical_updates,
    )
