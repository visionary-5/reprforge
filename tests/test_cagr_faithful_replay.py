import numpy as np

from reprforge.cagr_faithful_replay import (
    form_cagr_groups,
    form_fixed_jaccard_groups,
    replay_cagr_comparison,
)


def test_algorithm_one_uses_first_group_with_any_member_above_threshold():
    cohorts = [
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 4, 5],
    ]
    # q1 joins q0 (J=3/5); q2 is linked to q1 (J=3/5) but not q0 (J=2/6).
    assert form_cagr_groups([0, 1, 2], cohorts, theta=0.5) == ((0, 1, 2),)
    assert form_cagr_groups(
        [0, 1, 2], cohorts, theta=0.5, membership_rule="all"
    ) == ((0, 1), (2,))


def test_cagr_keeps_groups_contiguous_and_does_not_fill_across_boundary():
    cohorts = [[0, 1], [0, 2], [3, 4], [3, 5]]
    result = replay_cagr_comparison(
        cohorts,
        [0, 1, 2, 3],
        np.zeros(4),
        np.zeros(4),
        base_mean_quality=0.4,
        corpus_pages=8,
        request_batch_size=3,
        window=4,
        policy="cagr_faithful",
        cache_capacity=8,
        cagr_group_pool=4,
        cagr_theta=0.3,
    )
    assert result.dispatch_order == (0, 1, 2, 3)
    assert result.groups["count"] == 2
    assert result.request_batches["count"] == 2
    assert result.request_batches["size_max"] == 2


def test_fixed_jaccard_groups_have_registered_size_and_locality():
    cohorts = [[0, 1], [0, 2], [3, 4], [3, 5], [6, 7]]
    groups = form_fixed_jaccard_groups(
        [0, 1, 2, 3, 4], cohorts, target_group_size=2
    )
    assert all(len(group) <= 2 for group in groups)
    assert {frozenset(group) for group in groups} >= {
        frozenset({0, 1}),
        frozenset({2, 3}),
    }
    assert sorted(query for group in groups for query in group) == list(range(5))


def test_next_group_prefetch_is_charged_and_becomes_useful():
    cohorts = [[0, 1], [2, 3]]
    result = replay_cagr_comparison(
        cohorts,
        [0, 1],
        [0.0, 0.0],
        [0.0, 0.0],
        base_mean_quality=0.4,
        corpus_pages=4,
        request_batch_size=1,
        window=2,
        policy="cagr_faithful",
        cache_capacity=4,
        cagr_group_pool=2,
        cagr_theta=0.5,
    )
    assert result.prefetch == {
        "events": 2,
        "builds": 2,
        "reloads": 0,
        "useful": 2,
        "wasted": 0,
        "precision": 1.0,
        "unused_unit_work": 0.0,
    }
    assert result.final_unique_pages == 4


def test_quality_gain_never_changes_scheduler_order():
    cohorts = [[0, 1, 2], [3], [4], [5]]
    common = dict(
        cohorts=cohorts,
        arrival_order=[0, 1, 2, 3],
        arrival_times=np.zeros(4),
        base_mean_quality=0.4,
        corpus_pages=6,
        request_batch_size=1,
        window=4,
        policy="frontier",
        cache_capacity=4,
    )
    first = replay_cagr_comparison(quality_gain=[1.0, 0.0, 0.0, 0.0], **common)
    second = replay_cagr_comparison(quality_gain=[0.0, 0.0, 0.0, 1.0], **common)
    assert first.dispatch_order == second.dispatch_order == (1, 2, 3, 0)


def test_all_policies_preserve_final_union_and_cache_capacity():
    cohorts = [[0, 1], [1, 2], [2, 3], [3, 4]]
    for policy in (
        "fifo",
        "overlap_only",
        "history_popularity",
        "static_popularity",
        "cagr_faithful",
        "frontier",
    ):
        result = replay_cagr_comparison(
            cohorts,
            [0, 1, 2, 3],
            np.zeros(4),
            np.asarray([0.1, 0.0, 0.0, 0.0]),
            base_mean_quality=0.4,
            corpus_pages=5,
            request_batch_size=2,
            window=4,
            policy=policy,
            cache_capacity=2,
            cagr_group_pool=4,
            cagr_theta=0.3,
        )
        assert result.final_unique_pages == 5
        assert sorted(result.dispatch_order) == [0, 1, 2, 3]
        assert result.cache["capacity_pages"] == 2
