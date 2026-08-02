import numpy as np

from reprforge.boundary_admission import (
    execute_boundary_plan,
    fit_boundary_statistics,
    select_episode_pages,
)


def test_boundary_statistics_measure_membership_flips_without_qrels() -> None:
    locator = np.asarray([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    visual = np.asarray([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]])
    result = fit_boundary_statistics(locator, visual, cutoff=1)
    assert result.flip_risk_by_rank.tolist() == [0.5, 0.0, 0.5]
    assert result.visual_prior_by_rank.tolist() == [1.5, 0.0, 1.5]


def test_boundary_weights_prioritize_repeated_risky_pages() -> None:
    candidates = np.asarray([[1, 2, 3], [1, 4, 3]], dtype=np.int32)
    frequency = select_episode_pages(
        candidates,
        budget_fraction=0.5,
        rank_weights=None,
    )
    weighted = select_episode_pages(
        candidates,
        budget_fraction=0.5,
        rank_weights=[0.0, 1.0, 1.0],
    )
    assert frequency == {1, 3}
    assert weighted == {2, 3}


def test_partial_plan_uses_only_selected_visual_scores() -> None:
    pages = np.asarray([[10, 11, 12, 13]], dtype=np.int32)
    locator = np.asarray([[2.0, 1.0, 0.0, -1.0]])
    visual = np.asarray([[1000.0, 2.0, 8.0, -1000.0]])
    rankings, cost = execute_boundary_plan(
        pages,
        locator,
        visual,
        selected_pages={11, 12},
        visual_prior_by_rank=[0.0, 0.0, 0.0, 0.0],
        cutoff=2,
    )
    assert rankings.shape == (1, 2)
    assert cost["selected_unique_pages"] == 2
    assert cost["visual_candidate_events"] == 2


def test_invalid_budget_is_rejected() -> None:
    candidates = np.asarray([[1, 2]], dtype=np.int32)
    try:
        select_episode_pages(
            candidates,
            budget_fraction=1.1,
            rank_weights=None,
        )
    except ValueError as error:
        assert "budget_fraction" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid budget should fail")
