import numpy as np
import pytest

from reprforge.pairwise_view_admission import (
    BoundaryPair,
    build_boundary_pairs,
    evaluate_pair_coverage,
    select_frequency_pages,
    select_independent_pages,
    select_pairwise_pages,
)
from reprforge.pairwise_budget import calibrate_pair_budget
from tools.analyze_pairwise_view_admission import (
    _balanced_group_folds,
    _selected_by_boundary_weight,
)


def test_boundary_pairs_weight_close_challengers_more() -> None:
    candidates = np.asarray([[1, 2, 3, 4]])
    scores = np.asarray([[2.0, 1.0, 0.9, -1.0]])
    pairs = build_boundary_pairs(candidates, scores, cutoff=2)

    assert [(pair.incumbent, pair.challenger) for pair in pairs] == [
        (2, 3),
        (2, 4),
    ]
    assert pairs[0].weight > pairs[1].weight


def test_pair_selection_avoids_independent_node_aliasing() -> None:
    # Page 0 has the largest independent incident mass, but no partner can be
    # afforded after selecting page 20. Pair-aware selection instead purchases
    # the complete high-value (20, 21) comparison.
    pairs = [
        BoundaryPair(i, 0, i + 1, 5, 1.0)
        for i in range(10)
    ]
    pairs.append(BoundaryPair(20, 20, 21, 5, 9.0))

    independent = select_independent_pages(pairs, page_budget=2)
    pairwise = select_pairwise_pages(pairs, page_budget=2)

    assert independent.selected_pages == frozenset({0, 20})
    assert independent.covered_weight == 0.0
    assert pairwise.selected_pages == frozenset({20, 21})
    assert pairwise.covered_weight == 9.0


def test_pairwise_selection_reuses_a_materialized_anchor() -> None:
    pairs = [
        BoundaryPair(0, 1, 2, 5, 4.0),
        BoundaryPair(1, 1, 3, 5, 3.0),
        BoundaryPair(2, 4, 5, 5, 5.0),
    ]
    selected = select_pairwise_pages(pairs, page_budget=3)

    assert selected.selected_pages == frozenset({1, 2, 3})
    assert selected.covered_weight == 7.0
    assert selected.covered_pair_count == 2


def test_frequency_and_coverage_are_explicit_baselines() -> None:
    candidates = np.asarray([[1, 2, 3], [1, 4, 5]])
    pairs = [
        BoundaryPair(0, 2, 3, 2, 1.0),
        BoundaryPair(1, 4, 5, 2, 1.0),
    ]
    frequency = select_frequency_pages(candidates, pairs, page_budget=2)
    assert frequency.selected_pages == frozenset({1, 2})
    assert frequency.covered_weight == 0.0
    assert evaluate_pair_coverage(pairs, {2, 3}).covered_weight == 1.0


def test_invalid_pair_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="distinct"):
        BoundaryPair(0, 1, 1, 2, 1.0)
    with pytest.raises(ValueError, match="inside"):
        build_boundary_pairs(
            np.asarray([[1, 2]]),
            np.asarray([[1.0, 0.0]]),
            cutoff=2,
        )


def test_exact_boundary_budget_and_group_folds_are_deterministic() -> None:
    candidates = np.asarray([[1, 2, 3], [1, 4, 5]], dtype=np.int32)
    selected = _selected_by_boundary_weight(
        candidates,
        np.asarray([1.0, 0.5, 0.25]),
        page_budget=3,
    )
    assert len(selected) == 3

    groups = np.asarray(["paper-a", "paper-a", "paper-b", "paper-c"])
    first = _balanced_group_folds(groups, count=2)
    second = _balanced_group_folds(groups, count=2)
    assert np.array_equal(first, second)
    assert first[0] == first[1]


def test_pair_budget_is_calibrated_from_score_logs_without_qrels() -> None:
    candidates = np.asarray(
        [[0, 1, 2, 3], [0, 1, 4, 5], [6, 7, 8, 9], [6, 7, 10, 11]],
        dtype=np.int32,
    )
    locator = np.tile([2.0, 1.0, 0.9, 0.8], (4, 1))
    visual = np.asarray(
        [[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 2.0, 0.0]] * 2,
    )
    teacher = np.asarray(
        [
            np.argsort(-(locator[row] + (visual[row] - visual[row].mean())))[:2]
            for row in range(4)
        ],
        dtype=np.int32,
    )
    teacher_pages = candidates[np.arange(4)[:, None], teacher]
    result = calibrate_pair_budget(
        candidates,
        locator,
        visual,
        teacher_pages,
        rank_risk=[0.0, 0.0, 1.0, 0.5],
        visual_prior_by_rank=[0.0, 0.0, 0.0, 0.0],
        cutoff=2,
        baseline_fraction=0.5,
        grid=(0.25, 0.5),
    )
    assert result.selected_fraction in {0.25, 0.5}
    assert 0.0 <= result.baseline_agreement <= 1.0
    assert 0.0 <= result.selected_agreement <= 1.0
