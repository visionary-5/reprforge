"""Optimization baselines for complementary representation-view admission.

Selecting a visual view has no pair objective value by itself.  A weighted
incumbent--challenger edge contributes only when both endpoint views are
resident.  Under a uniform page budget, maximizing total completed edge weight
is the weighted heaviest/densest k-subgraph problem, not maximum coverage.

This module deliberately separates the research formulation from a particular
solver.  It provides an exact small-instance oracle, the existing independent
and conditional heuristics, a weighted adaptation of the diagonal-loading
Frank--Wolfe baseline from Lu et al. (AAAI 2025), and deterministic multi-start
one-swap local search.  Frank--Wolfe is prior art and is labelled as such.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from reprforge.pairwise_view_admission import (
    BoundaryPair,
    PairAdmission,
    evaluate_pair_coverage,
    select_independent_pages,
    select_pairwise_pages,
)


@dataclass(frozen=True)
class ComplementaryViewSolution:
    selected_pages: frozenset[int]
    admission: PairAdmission
    solver: str
    solve_ms: float
    iterations: int = 0
    exact: bool = False
    combinations_evaluated: int = 0


def aggregate_edge_weights(
    pairs: Sequence[BoundaryPair],
) -> dict[tuple[int, int], float]:
    weights: defaultdict[tuple[int, int], float] = defaultdict(float)
    for pair in pairs:
        edge = tuple(sorted((int(pair.incumbent), int(pair.challenger))))
        weights[edge] += float(pair.weight)
    return dict(weights)


def instance_vertices(pairs: Sequence[BoundaryPair]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                page
                for pair in pairs
                for page in (int(pair.incumbent), int(pair.challenger))
            }
        )
    )


def _fill_to_budget(
    selected: Iterable[int],
    vertices: Sequence[int],
    edge_weights: dict[tuple[int, int], float],
    page_budget: int,
) -> frozenset[int]:
    result = set(int(page) for page in selected)
    incident: defaultdict[int, float] = defaultdict(float)
    for (left, right), weight in edge_weights.items():
        incident[left] += weight
        incident[right] += weight
    for page in sorted(vertices, key=lambda value: (-incident[value], value)):
        if len(result) >= page_budget:
            break
        result.add(page)
    return frozenset(result)


def solve_exact_combinations(
    pairs: Sequence[BoundaryPair],
    *,
    page_budget: int,
    max_combinations: int = 2_000_000,
) -> ComplementaryViewSolution:
    """Enumerate all k-subsets for a trustworthy small-instance oracle."""

    if page_budget < 0:
        raise ValueError("page budget cannot be negative")
    if max_combinations <= 0:
        raise ValueError("max combinations must be positive")
    vertices = instance_vertices(pairs)
    budget = min(page_budget, len(vertices))
    count = math.comb(len(vertices), budget)
    if count > max_combinations:
        raise ValueError(
            f"exact oracle requires {count} combinations, limit is {max_combinations}"
        )
    weights = aggregate_edge_weights(pairs)
    began = time.perf_counter()
    best_pages: tuple[int, ...] = ()
    best_weight = -1.0
    for pages in itertools.combinations(vertices, budget):
        selected = set(pages)
        weight = sum(
            value
            for (left, right), value in weights.items()
            if left in selected and right in selected
        )
        if weight > best_weight + 1e-12 or (
            math.isclose(weight, best_weight, rel_tol=1e-12, abs_tol=1e-12)
            and pages < best_pages
        ):
            best_weight = weight
            best_pages = pages
    selected_pages = frozenset(best_pages)
    return ComplementaryViewSolution(
        selected_pages=selected_pages,
        admission=evaluate_pair_coverage(pairs, selected_pages),
        solver="exact-combinations",
        solve_ms=(time.perf_counter() - began) * 1000.0,
        exact=True,
        combinations_evaluated=count,
    )


def solve_frank_wolfe_diagonal(
    pairs: Sequence[BoundaryPair],
    *,
    page_budget: int,
    iterations: int = 200,
    diagonal_load: float | None = None,
) -> ComplementaryViewSolution:
    """Weighted adaptation of Lu et al.'s diagonal-loading FW baseline.

    The official unweighted implementation initializes x=k/n, repeatedly
    linearizes (A + lambda I), moves toward the top-k gradient coordinates and
    rounds the final continuous point.  Here lambda defaults to the largest
    edge weight, the scale-equivalent choice to lambda=1 on an unweighted
    graph.  The infinity norm is a deterministic Lipschitz upper bound and
    avoids a SciPy dependency.
    """

    if page_budget < 0:
        raise ValueError("page budget cannot be negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    vertices = instance_vertices(pairs)
    budget = min(page_budget, len(vertices))
    began = time.perf_counter()
    if budget == 0 or not vertices:
        admission = evaluate_pair_coverage(pairs, ())
        return ComplementaryViewSolution(
            selected_pages=frozenset(),
            admission=admission,
            solver="frank-wolfe-diagonal",
            solve_ms=(time.perf_counter() - began) * 1000.0,
        )
    weights = aggregate_edge_weights(pairs)
    offsets = {page: index for index, page in enumerate(vertices)}
    adjacency: defaultdict[int, dict[int, float]] = defaultdict(dict)
    incident = np.zeros(len(vertices), dtype=np.float64)
    for (left, right), weight in weights.items():
        left_index = offsets[left]
        right_index = offsets[right]
        adjacency[left_index][right_index] = (
            adjacency[left_index].get(right_index, 0.0) + weight
        )
        adjacency[right_index][left_index] = (
            adjacency[right_index].get(left_index, 0.0) + weight
        )
        incident[left_index] += weight
        incident[right_index] += weight
    load = (
        max(weights.values(), default=1.0)
        if diagonal_load is None
        else float(diagonal_load)
    )
    if not math.isfinite(load) or load < 0:
        raise ValueError("diagonal load must be finite and non-negative")
    lipschitz = max(float(np.max(incident + load)), 1e-12)
    x = np.full(len(vertices), budget / len(vertices), dtype=np.float64)
    completed = 0
    for completed in range(1, iterations + 1):
        gradient = load * x
        for left, neighbors in adjacency.items():
            gradient[left] += sum(
                weight * x[right]
                for right, weight in neighbors.items()
            )
        order = sorted(
            range(len(vertices)),
            key=lambda index: (-float(gradient[index]), vertices[index]),
        )
        target = np.zeros(len(vertices), dtype=np.float64)
        target[order[:budget]] = 1.0
        direction = target - x
        directional = float(gradient @ direction)
        norm_squared = float(direction @ direction)
        if directional <= 1e-12 or norm_squared <= 1e-12:
            break
        step = min(1.0, directional / (lipschitz * norm_squared))
        x += step * direction
    selected_offsets = sorted(
        range(len(vertices)),
        key=lambda index: (-float(x[index]), vertices[index]),
    )[:budget]
    selected = frozenset(vertices[index] for index in selected_offsets)
    return ComplementaryViewSolution(
        selected_pages=selected,
        admission=evaluate_pair_coverage(pairs, selected),
        solver="frank-wolfe-diagonal",
        solve_ms=(time.perf_counter() - began) * 1000.0,
        iterations=completed,
    )


def improve_one_swap(
    pairs: Sequence[BoundaryPair],
    selected_pages: Iterable[int],
    *,
    page_budget: int,
    max_iterations: int = 10_000,
) -> ComplementaryViewSolution:
    """Reach a deterministic one-swap local optimum of induced edge weight."""

    if page_budget < 0:
        raise ValueError("page budget cannot be negative")
    if max_iterations <= 0:
        raise ValueError("max iterations must be positive")
    vertices = instance_vertices(pairs)
    budget = min(page_budget, len(vertices))
    weights = aggregate_edge_weights(pairs)
    selected = set(
        _fill_to_budget(selected_pages, vertices, weights, budget)
    )
    if len(selected) > budget:
        selected = set(sorted(selected)[:budget])
    adjacency: defaultdict[int, dict[int, float]] = defaultdict(dict)
    for (left, right), weight in weights.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    began = time.perf_counter()
    completed = 0
    for completed in range(1, max_iterations + 1):
        outside = set(vertices) - selected
        removal_loss = {
            page: sum(
                weight
                for neighbor, weight in adjacency[page].items()
                if neighbor in selected
            )
            for page in selected
        }
        addition_gain = {
            page: sum(
                weight
                for neighbor, weight in adjacency[page].items()
                if neighbor in selected
            )
            for page in outside
        }
        best: tuple[float, int, int] | None = None
        for removed in selected:
            for added in outside:
                edge = adjacency[removed].get(added, 0.0)
                delta = -removal_loss[removed] + addition_gain[added] - edge
                candidate = (delta, -added, removed)
                if best is None or candidate > best:
                    best = candidate
        if best is None or best[0] <= 1e-12:
            break
        _, negative_added, removed = best
        selected.remove(removed)
        selected.add(-negative_added)
    result = frozenset(selected)
    return ComplementaryViewSolution(
        selected_pages=result,
        admission=evaluate_pair_coverage(pairs, result),
        solver="one-swap-local-search",
        solve_ms=(time.perf_counter() - began) * 1000.0,
        iterations=completed,
    )


def solve_multistart_local_search(
    pairs: Sequence[BoundaryPair],
    *,
    page_budget: int,
    fw_iterations: int = 200,
) -> ComplementaryViewSolution:
    """Improve conditional, incident and Frank--Wolfe starts, keep the best."""

    began = time.perf_counter()
    weights = aggregate_edge_weights(pairs)
    vertices = instance_vertices(pairs)
    starts = [
        ("pair-conditional", select_pairwise_pages(pairs, page_budget=page_budget)),
        ("pair-incident", select_independent_pages(pairs, page_budget=page_budget)),
        (
            "frank-wolfe",
            solve_frank_wolfe_diagonal(
                pairs,
                page_budget=page_budget,
                iterations=fw_iterations,
            ).admission,
        ),
    ]
    solutions: list[tuple[float, tuple[int, ...], str, ComplementaryViewSolution]] = []
    for name, start in starts:
        filled = _fill_to_budget(
            start.selected_pages,
            vertices,
            weights,
            min(page_budget, len(vertices)),
        )
        improved = improve_one_swap(
            pairs,
            filled,
            page_budget=page_budget,
        )
        solutions.append(
            (
                improved.admission.covered_weight,
                tuple(-page for page in sorted(improved.selected_pages)),
                name,
                improved,
            )
        )
    best = max(solutions, key=lambda row: row[:3])
    return ComplementaryViewSolution(
        selected_pages=best[3].selected_pages,
        admission=best[3].admission,
        solver=f"multistart-local-search:{best[2]}",
        solve_ms=(time.perf_counter() - began) * 1000.0,
        iterations=sum(row[3].iterations for row in solutions),
    )


def query_saturated_utility(
    pairs: Sequence[BoundaryPair],
    selected_pages: Iterable[int],
) -> float:
    """Noisy-OR probability mass of at least one completed edge per query.

    Boundary weights are interpreted as marginal event probabilities.  The
    independence approximation makes repeated evidence for one query saturate
    naturally: ``1 - product(1 - p_edge)``.  This avoids counting five
    challenger edges as five independent units of workload value.
    """

    selected = set(int(page) for page in selected_pages)
    survival: defaultdict[int, float] = defaultdict(lambda: 1.0)
    touched: set[int] = set()
    for pair in pairs:
        if pair.incumbent not in selected or pair.challenger not in selected:
            continue
        probability = min(max(float(pair.weight), 0.0), 1.0)
        survival[pair.query_index] *= 1.0 - probability
        touched.add(pair.query_index)
    return sum(1.0 - survival[query] for query in touched)


def solve_exact_query_saturated(
    pairs: Sequence[BoundaryPair],
    *,
    page_budget: int,
    max_combinations: int = 2_000_000,
) -> ComplementaryViewSolution:
    """Exact small-instance oracle for the query-saturated objective."""

    if page_budget < 0:
        raise ValueError("page budget cannot be negative")
    vertices = instance_vertices(pairs)
    budget = min(page_budget, len(vertices))
    count = math.comb(len(vertices), budget)
    if count > max_combinations:
        raise ValueError(
            f"exact oracle requires {count} combinations, limit is {max_combinations}"
        )
    began = time.perf_counter()
    best_pages: tuple[int, ...] = ()
    best_value = -1.0
    for pages in itertools.combinations(vertices, budget):
        value = query_saturated_utility(pairs, pages)
        if value > best_value + 1e-12 or (
            math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12)
            and pages < best_pages
        ):
            best_value = value
            best_pages = pages
    selected = frozenset(best_pages)
    return ComplementaryViewSolution(
        selected_pages=selected,
        admission=evaluate_pair_coverage(pairs, selected),
        solver="exact-query-saturated",
        solve_ms=(time.perf_counter() - began) * 1000.0,
        exact=True,
        combinations_evaluated=count,
    )


def solve_query_saturated_greedy(
    pairs: Sequence[BoundaryPair],
    *,
    page_budget: int,
) -> ComplementaryViewSolution:
    """Buy the largest marginal noisy-OR query utility per new page.

    Marginal gains are maintained from completed edges rather than recomputing
    the full objective for every candidate action.  With the boundary graph's
    small per-query degree, this keeps the deployable path sparse in the
    observed edges rather than quadratic in all corpus pages.
    """

    if page_budget < 0:
        raise ValueError("page budget cannot be negative")
    vertices = instance_vertices(pairs)
    budget = min(page_budget, len(vertices))
    began = time.perf_counter()
    selected: set[int] = set()
    survival: defaultdict[int, float] = defaultdict(lambda: 1.0)
    indexed_pairs = tuple(enumerate(pairs))
    incident_pairs: defaultdict[int, list[int]] = defaultdict(list)
    for index, pair in indexed_pairs:
        incident_pairs[pair.incumbent].append(index)
        incident_pairs[pair.challenger].append(index)

    def newly_completed(action: tuple[int, ...]) -> tuple[int, ...]:
        proposed = selected | set(action)
        affected = {
            index
            for page in action
            for index in incident_pairs[page]
        }
        return tuple(
            index
            for index in sorted(affected)
            if pairs[index].incumbent in proposed
            and pairs[index].challenger in proposed
            and not (
                pairs[index].incumbent in selected
                and pairs[index].challenger in selected
            )
        )

    def marginal_gain(completed: Sequence[int]) -> float:
        factors: defaultdict[int, float] = defaultdict(lambda: 1.0)
        for index in completed:
            pair = pairs[index]
            probability = min(max(float(pair.weight), 0.0), 1.0)
            factors[pair.query_index] *= 1.0 - probability
        return sum(
            survival[query] * (1.0 - factor)
            for query, factor in factors.items()
        )

    iterations = 0
    while len(selected) < budget:
        actions: set[tuple[int, ...]] = set()
        for pair in pairs:
            missing = tuple(
                sorted(
                    {
                        page
                        for page in (pair.incumbent, pair.challenger)
                        if page not in selected
                    }
                )
            )
            if missing and len(selected) + len(missing) <= budget:
                actions.add(missing)
            for page in missing:
                if len(selected) + 1 <= budget:
                    actions.add((page,))
        best: tuple[float, float, int, tuple[int, ...]] | None = None
        best_completed: tuple[int, ...] = ()
        for action in actions:
            completed = newly_completed(action)
            gain = marginal_gain(completed)
            if gain <= 1e-12:
                continue
            candidate = (
                gain / len(action),
                gain,
                -len(action),
                tuple(-page for page in action),
            )
            if best is None or candidate > best:
                best = candidate
                best_completed = completed
        if best is None:
            break
        action = tuple(-page for page in best[3])
        selected.update(action)
        for index in best_completed:
            pair = pairs[index]
            probability = min(max(float(pair.weight), 0.0), 1.0)
            survival[pair.query_index] *= 1.0 - probability
        iterations += 1
    selected = set(
        _fill_to_budget(
            selected,
            vertices,
            aggregate_edge_weights(pairs),
            budget,
        )
    )
    result = frozenset(selected)
    return ComplementaryViewSolution(
        selected_pages=result,
        admission=evaluate_pair_coverage(pairs, result),
        solver="query-saturated-greedy",
        solve_ms=(time.perf_counter() - began) * 1000.0,
        iterations=iterations,
    )


def improve_one_swap_query_saturated(
    pairs: Sequence[BoundaryPair],
    selected_pages: Iterable[int],
    *,
    page_budget: int,
    max_iterations: int = 10_000,
) -> ComplementaryViewSolution:
    """Reach a deterministic one-swap optimum of noisy-OR query utility."""

    if page_budget < 0:
        raise ValueError("page budget cannot be negative")
    vertices = instance_vertices(pairs)
    budget = min(page_budget, len(vertices))
    selected = set(
        _fill_to_budget(
            selected_pages,
            vertices,
            aggregate_edge_weights(pairs),
            budget,
        )
    )
    began = time.perf_counter()
    current = query_saturated_utility(pairs, selected)
    completed = 0
    for completed in range(1, max_iterations + 1):
        outside = set(vertices) - selected
        best: tuple[float, int, int, float] | None = None
        for removed in selected:
            for added in outside:
                proposed = (selected - {removed}) | {added}
                value = query_saturated_utility(pairs, proposed)
                candidate = (value - current, -added, removed, value)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        if best is None or best[0] <= 1e-12:
            break
        _, negative_added, removed, current = best
        selected.remove(removed)
        selected.add(-negative_added)
    result = frozenset(selected)
    return ComplementaryViewSolution(
        selected_pages=result,
        admission=evaluate_pair_coverage(pairs, result),
        solver="query-saturated-one-swap",
        solve_ms=(time.perf_counter() - began) * 1000.0,
        iterations=completed,
    )


def solve_query_saturated_multistart(
    pairs: Sequence[BoundaryPair],
    *,
    page_budget: int,
) -> ComplementaryViewSolution:
    """Improve saturated-greedy, conditional and incident starts."""

    began = time.perf_counter()
    starts = [
        solve_query_saturated_greedy(pairs, page_budget=page_budget),
        _solution_from_admission(
            select_pairwise_pages(pairs, page_budget=page_budget),
            "pair-conditional",
        ),
        _solution_from_admission(
            select_independent_pages(pairs, page_budget=page_budget),
            "pair-incident",
        ),
    ]
    improved = [
        improve_one_swap_query_saturated(
            pairs,
            start.selected_pages,
            page_budget=page_budget,
        )
        for start in starts
    ]
    best = max(
        improved,
        key=lambda solution: (
            query_saturated_utility(pairs, solution.selected_pages),
            tuple(-page for page in sorted(solution.selected_pages)),
        ),
    )
    return ComplementaryViewSolution(
        selected_pages=best.selected_pages,
        admission=best.admission,
        solver="query-saturated-multistart",
        solve_ms=(time.perf_counter() - began) * 1000.0,
        iterations=sum(solution.iterations for solution in improved),
    )


def _solution_from_admission(
    admission: PairAdmission,
    solver: str,
) -> ComplementaryViewSolution:
    return ComplementaryViewSolution(
        selected_pages=admission.selected_pages,
        admission=admission,
        solver=solver,
        solve_ms=0.0,
    )
