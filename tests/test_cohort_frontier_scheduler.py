import pytest

from reprforge.cohort_frontier_scheduler import (
    frontier_reuse_order,
    replay_page_work,
    static_popularity_order,
)


COHORTS = [
    [0, 1],
    [2, 3],
    [2, 4],
    [2, 5],
]


def test_frontier_scheduler_frontloads_reusable_cohorts():
    assert static_popularity_order(COHORTS) == [1, 2, 3, 0]
    assert frontier_reuse_order(COHORTS, batch_size=2)[:2] == [1, 2]


def test_page_work_replay_preserves_final_work_and_integrates_quality():
    report = replay_page_work(
        COHORTS,
        [1, 2, 3, 0],
        [0.1, 0.2, 0.3, -0.1],
        base_mean_quality=0.4,
        batch_size=2,
        corpus_pages=8,
    )
    assert report["final_unique_pages"] == 6
    assert report["points"][-1]["mean_quality"] == pytest.approx(0.525)
    assert report["completion_pages"]["p50"] == pytest.approx(4.5)


def test_page_work_replay_rejects_non_permutation():
    with pytest.raises(ValueError):
        replay_page_work(
            COHORTS,
            [0, 1, 1, 3],
            [0.0] * 4,
            base_mean_quality=0.0,
            batch_size=2,
            corpus_pages=8,
        )
