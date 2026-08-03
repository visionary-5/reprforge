from __future__ import annotations

import pytest

from reprforge.complementary_view_optimization import (
    aggregate_edge_weights,
    improve_one_swap,
    query_saturated_utility,
    solve_exact_combinations,
    solve_exact_query_saturated,
    solve_frank_wolfe_diagonal,
    solve_multistart_local_search,
    solve_query_saturated_greedy,
    solve_query_saturated_multistart,
)
from reprforge.pairwise_view_admission import (
    BoundaryPair,
    select_independent_pages,
)


def _pair(left: int, right: int, weight: float, query: int) -> BoundaryPair:
    return BoundaryPair(
        query_index=query,
        incumbent=left,
        challenger=right,
        challenger_rank=5,
        weight=weight,
    )


def test_duplicate_query_edges_are_aggregated():
    pairs = (_pair(2, 1, 3.0, 0), _pair(1, 2, 4.0, 1))
    assert aggregate_edge_weights(pairs) == {(1, 2): 7.0}


def test_exact_oracle_rejects_intractable_enumeration():
    pairs = tuple(_pair(index, index + 1, 1.0, index) for index in range(12))
    with pytest.raises(ValueError, match="combinations"):
        solve_exact_combinations(pairs, page_budget=6, max_combinations=10)


def test_local_search_repairs_independent_incident_trap():
    pairs = (
        _pair(0, 1, 4.0, 0),
        _pair(0, 2, 4.0, 1),
        _pair(1, 2, 4.0, 2),
        _pair(3, 4, 5.0, 3),
        _pair(3, 5, 5.0, 4),
        _pair(3, 6, 5.0, 5),
    )
    independent = select_independent_pages(pairs, page_budget=3)
    exact = solve_exact_combinations(pairs, page_budget=3)
    improved = improve_one_swap(
        pairs,
        independent.selected_pages,
        page_budget=3,
    )
    multistart = solve_multistart_local_search(pairs, page_budget=3)

    assert independent.covered_weight == 4.0
    assert exact.selected_pages == frozenset({0, 1, 2})
    assert exact.admission.covered_weight == 12.0
    assert improved.admission.covered_weight == 12.0
    assert multistart.admission.covered_weight == 12.0


def test_frank_wolfe_is_deterministic_and_budget_exact():
    pairs = (
        _pair(0, 1, 2.0, 0),
        _pair(1, 2, 2.0, 1),
        _pair(0, 2, 2.0, 2),
        _pair(2, 3, 0.5, 3),
    )
    first = solve_frank_wolfe_diagonal(pairs, page_budget=3)
    second = solve_frank_wolfe_diagonal(pairs, page_budget=3)

    assert first.selected_pages == second.selected_pages
    assert len(first.selected_pages) == 3
    assert first.admission.covered_weight == 6.0


def test_query_saturation_prefers_serving_two_queries_over_repeated_edges():
    pairs = (
        _pair(0, 1, 0.6, 0),
        _pair(0, 2, 0.6, 0),
        _pair(3, 4, 0.55, 1),
    )
    additive = solve_exact_combinations(pairs, page_budget=4)
    saturated = solve_exact_query_saturated(pairs, page_budget=4)
    greedy = solve_query_saturated_greedy(pairs, page_budget=4)
    approximate = solve_query_saturated_multistart(pairs, page_budget=4)

    assert additive.selected_pages.issuperset({0, 1, 2})
    assert query_saturated_utility(pairs, additive.selected_pages) == pytest.approx(0.84)
    assert saturated.selected_pages == frozenset({0, 1, 3, 4})
    assert query_saturated_utility(pairs, saturated.selected_pages) == pytest.approx(1.15)
    assert greedy.selected_pages == saturated.selected_pages
    assert approximate.selected_pages == saturated.selected_pages
