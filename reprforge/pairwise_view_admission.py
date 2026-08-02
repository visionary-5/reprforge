"""Pair-aware admission for representation views near a retrieval boundary.

Candidate-relative multimodal scores are relational: one newly encoded page
does not provide a stable comparison scale, and a ranking boundary changes
only when a challenger is compared with an incumbent.  This module therefore
models the workload as a weighted graph.  Pages are vertices; a query creates
weighted incumbent--challenger edges.  An edge's what-if value is observable
only when both endpoint representations are available.

The selector is a deterministic scalable heuristic.  It alternates between a
fresh high-value pair and a single page that completes many edges adjacent to
already selected pages.  It is a mechanism candidate, not an optimality claim.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class BoundaryPair:
    query_index: int
    incumbent: int
    challenger: int
    challenger_rank: int
    weight: float

    def __post_init__(self) -> None:
        if self.query_index < 0 or self.incumbent < 0 or self.challenger < 0:
            raise ValueError("boundary-pair identifiers cannot be negative")
        if self.incumbent == self.challenger:
            raise ValueError("a boundary pair requires two distinct pages")
        if self.challenger_rank < 0:
            raise ValueError("challenger rank cannot be negative")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("boundary-pair weight must be finite and non-negative")


@dataclass(frozen=True)
class PairAdmission:
    selected_pages: frozenset[int]
    covered_weight: float
    total_weight: float
    covered_pair_count: int
    pair_count: int

    @property
    def covered_weight_fraction(self) -> float:
        return self.covered_weight / self.total_weight if self.total_weight else 0.0


def build_boundary_pairs(
    candidate_pages: np.ndarray,
    locator_scores: np.ndarray,
    *,
    cutoff: int,
    rank_risk: Sequence[float] | None = None,
    temperature: float = 1.0,
) -> tuple[BoundaryPair, ...]:
    """Create incumbent--challenger pairs from runtime-visible locator state."""

    candidates = np.asarray(candidate_pages, dtype=np.int64)
    locator = np.asarray(locator_scores, dtype=np.float64)
    if candidates.shape != locator.shape or candidates.ndim != 2:
        raise ValueError("candidate pages and locator scores must align in 2-D")
    if cutoff <= 0 or cutoff >= candidates.shape[1]:
        raise ValueError("cutoff must lie inside the candidate cohort")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    risk = (
        np.ones(candidates.shape[1], dtype=np.float64)
        if rank_risk is None
        else np.asarray(rank_risk, dtype=np.float64)
    )
    if risk.shape != (candidates.shape[1],):
        raise ValueError("rank risk must match candidate depth")
    if np.any(~np.isfinite(risk)) or np.any(risk < 0):
        raise ValueError("rank risk must be finite and non-negative")

    pairs: list[BoundaryPair] = []
    incumbent_rank = cutoff - 1
    for query in range(len(candidates)):
        incumbent = int(candidates[query, incumbent_rank])
        incumbent_score = float(locator[query, incumbent_rank])
        for rank in range(cutoff, candidates.shape[1]):
            challenger = int(candidates[query, rank])
            margin = max(0.0, incumbent_score - float(locator[query, rank]))
            weight = float(risk[rank] * math.exp(-margin / temperature))
            if weight == 0.0:
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


def _aggregate_pair_weights(
    pairs: Sequence[BoundaryPair],
) -> dict[tuple[int, int], float]:
    weights: defaultdict[tuple[int, int], float] = defaultdict(float)
    for pair in pairs:
        edge = tuple(sorted((pair.incumbent, pair.challenger)))
        weights[edge] += pair.weight
    return dict(weights)


def evaluate_pair_coverage(
    pairs: Sequence[BoundaryPair],
    selected_pages: Sequence[int] | set[int] | frozenset[int],
) -> PairAdmission:
    selected = frozenset(int(value) for value in selected_pages)
    covered = [
        pair
        for pair in pairs
        if pair.incumbent in selected and pair.challenger in selected
    ]
    return PairAdmission(
        selected_pages=selected,
        covered_weight=sum(pair.weight for pair in covered),
        total_weight=sum(pair.weight for pair in pairs),
        covered_pair_count=len(covered),
        pair_count=len(pairs),
    )


def select_independent_pages(
    pairs: Sequence[BoundaryPair],
    *,
    page_budget: int,
) -> PairAdmission:
    """Strong cheap baseline: rank pages by independent incident risk."""

    if page_budget < 0:
        raise ValueError("page budget cannot be negative")
    incident: Counter[int] = Counter()
    for pair in pairs:
        incident[pair.incumbent] += pair.weight
        incident[pair.challenger] += pair.weight
    selected = sorted(incident, key=lambda page: (-incident[page], page))[:page_budget]
    return evaluate_pair_coverage(pairs, selected)


def select_frequency_pages(
    candidate_pages: np.ndarray,
    pairs: Sequence[BoundaryPair],
    *,
    page_budget: int,
) -> PairAdmission:
    if page_budget < 0:
        raise ValueError("page budget cannot be negative")
    candidates = np.asarray(candidate_pages, dtype=np.int64)
    if candidates.ndim != 2:
        raise ValueError("candidate pages must be a 2-D matrix")
    frequency = Counter(int(value) for value in candidates.flat)
    selected = sorted(
        frequency,
        key=lambda page: (-frequency[page], page),
    )[:page_budget]
    return evaluate_pair_coverage(pairs, selected)


def select_pairwise_pages(
    pairs: Sequence[BoundaryPair],
    *,
    page_budget: int,
) -> PairAdmission:
    """Select representation views by conditional pair-completion value."""

    if page_budget < 0:
        raise ValueError("page budget cannot be negative")
    pair_weights = _aggregate_pair_weights(pairs)
    if page_budget == 0 or not pair_weights:
        return evaluate_pair_coverage(pairs, ())
    adjacency: defaultdict[int, dict[int, float]] = defaultdict(dict)
    for edge, weight in pair_weights.items():
        left, right = edge
        adjacency[left][right] = weight
        adjacency[right][left] = weight

    # A fresh pair is a seed, not an isolated decision. Give it a bounded
    # lookahead for pages that could subsequently complete more edges through
    # either endpoint. This keeps the selector linear in the graph while
    # distinguishing a reusable anchor from an equally weighted dead-end pair.
    lookahead = min(max(page_budget - 2, 0), 8)
    top_neighbors = {
        page: sorted(neighbors.items(), key=lambda row: (-row[1], row[0]))[
            : lookahead + 1
        ]
        for page, neighbors in adjacency.items()
    }
    pair_heap: list[tuple[float, tuple[int, int]]] = []
    for edge, weight in pair_weights.items():
        left, right = edge
        future: dict[int, float] = {}
        for page, other in ((left, right), (right, left)):
            for neighbor, neighbor_weight in top_neighbors[page]:
                if neighbor == other:
                    continue
                future[neighbor] = max(future.get(neighbor, 0.0), neighbor_weight)
        projected = weight + sum(
            sorted(future.values(), reverse=True)[:lookahead]
        )
        # Every fresh seed consumes two pages and can use at most the remaining
        # budget. Zero-gain leftover slots stay in the denominator, penalizing
        # isolated pairs when a reusable anchor can fill the same budget.
        projected_density = projected / max(min(page_budget, 2 + lookahead), 1)
        heapq.heappush(pair_heap, (-projected_density, edge))

    selected: set[int] = set()
    completion_gain: defaultdict[int, float] = defaultdict(float)
    completion_heap: list[tuple[float, int]] = []

    def add_page(page: int) -> None:
        if page in selected:
            return
        selected.add(page)
        completion_gain.pop(page, None)
        for neighbor, weight in adjacency[page].items():
            if neighbor not in selected:
                completion_gain[neighbor] += weight
                heapq.heappush(
                    completion_heap,
                    (-completion_gain[neighbor], neighbor),
                )

    while len(selected) < page_budget:
        while completion_heap:
            negative_gain, page = completion_heap[0]
            if page in selected or not math.isclose(
                -negative_gain,
                completion_gain.get(page, 0.0),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                heapq.heappop(completion_heap)
                continue
            break
        while pair_heap:
            _, (left, right) = pair_heap[0]
            if left in selected or right in selected:
                heapq.heappop(pair_heap)
                continue
            break

        singleton = (
            None
            if not completion_heap
            else (-completion_heap[0][0], completion_heap[0][1])
        )
        fresh_pair = (
            None
            if not pair_heap or page_budget - len(selected) < 2
            else (-pair_heap[0][0], pair_heap[0][1])
        )
        if singleton is None and fresh_pair is None:
            break
        choose_singleton = fresh_pair is None or (
            singleton is not None
            and (singleton[0], -singleton[1])
            >= (fresh_pair[0], -fresh_pair[1][0])
        )
        if choose_singleton:
            assert singleton is not None
            heapq.heappop(completion_heap)
            add_page(singleton[1])
        else:
            assert fresh_pair is not None
            heapq.heappop(pair_heap)
            add_page(fresh_pair[1][0])
            add_page(fresh_pair[1][1])

    return evaluate_pair_coverage(pairs, selected)


def pair_weights_by_page(
    pairs: Sequence[BoundaryPair],
) -> Mapping[int, float]:
    values: Counter[int] = Counter()
    for pair in pairs:
        values[pair.incumbent] += pair.weight
        values[pair.challenger] += pair.weight
    return dict(values)
