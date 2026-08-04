from reprforge.scheduler_baselines import (
    offline_work_greedy_order,
    overlap_only_order,
    reuse_only_order,
    shortest_missing_order,
)
from reprforge.cohort_frontier_scheduler import frontier_reuse_order, replay_page_work


COHORTS = [
    [0, 1],
    [2, 3],
    [2, 4],
    [2, 5],
]


def _is_permutation(order):
    return sorted(order) == list(range(len(COHORTS)))


def test_scheduler_baselines_are_deterministic_permutations():
    functions = (
        overlap_only_order,
        shortest_missing_order,
        reuse_only_order,
        offline_work_greedy_order,
    )
    for function in functions:
        first = function(COHORTS, batch_size=2)
        assert _is_permutation(first)
        assert function(COHORTS, batch_size=2) == first


def test_overlap_only_groups_shared_pages():
    order = overlap_only_order(COHORTS, batch_size=2)
    assert set(order[:2]).issubset({1, 2, 3})


def test_shortest_missing_reuses_resident_pages():
    cohorts = [[0, 1], [2, 3], [0, 4]]
    assert shortest_missing_order(cohorts, batch_size=1) == [0, 2, 1]


def test_invalid_batch_size_is_rejected():
    try:
        reuse_only_order(COHORTS, batch_size=0)
    except ValueError as error:
        assert "batch_size" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_offline_greedy_is_a_lower_envelope_of_frontier_work():
    frontier = frontier_reuse_order(COHORTS, batch_size=2)
    offline = offline_work_greedy_order(COHORTS, batch_size=2)
    common = dict(
        quality_gain=[0.0] * len(COHORTS),
        base_mean_quality=0.0,
        batch_size=2,
        corpus_pages=8,
    )
    assert replay_page_work(COHORTS, offline, **common)["completion_pages"][
        "mean"
    ] <= replay_page_work(COHORTS, frontier, **common)["completion_pages"]["mean"]
