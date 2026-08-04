import numpy as np
import pytest

from reprforge.windowed_arrival_replay import (
    make_arrival_times,
    replay_windowed_arrivals,
)


def replay(cohorts, *, policy, window, batch_size=1, gains=None):
    query_count = len(cohorts)
    return replay_windowed_arrivals(
        cohorts,
        list(range(query_count)),
        np.zeros(query_count),
        [0.0] * query_count if gains is None else gains,
        base_mean_quality=0.4,
        corpus_pages=8,
        batch_size=batch_size,
        window=window,
        policy=policy,
        random_seed=17,
    )


def test_arrival_generators_are_deterministic_and_monotonic():
    burst = make_arrival_times(
        7, model="burst", seed=1, burst_size=3, burst_interval=5
    )
    assert burst.tolist() == [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 10.0]
    first = make_arrival_times(20, model="poisson", seed=9, poisson_mean=2)
    second = make_arrival_times(20, model="poisson", seed=9, poisson_mean=2)
    assert np.array_equal(first, second)
    assert first[0] == 0
    assert np.all(np.diff(first) >= 0)


def test_window_one_frontier_is_fifo_even_when_cohorts_differ():
    cohorts = [[0, 1, 2], [3], [3, 4], [4]]
    fifo = replay(cohorts, policy="fifo", window=1)
    frontier = replay(cohorts, policy="frontier", window=1)
    assert frontier.dispatch_order == fifo.dispatch_order == (0, 1, 2, 3)
    assert frontier.completion_pages == fifo.completion_pages


def test_frontier_reports_bypass_based_starvation_without_using_quality():
    cohorts = [[0, 1, 2, 3], [4], [5], [6]]
    result = replay(cohorts, policy="frontier", window=2)
    assert result.dispatch_order == (1, 2, 3, 0)
    assert result.bypass_count[0] == 3
    report = result.as_dict()
    assert report["starvation"]["count"] == 1
    assert report["starvation"]["max_younger_bypass"] == 3


def test_fair_frontier_caps_younger_bypass_below_window():
    cohorts = [[0, 1, 2, 3], [4], [5], [6]]
    result = replay(cohorts, policy="frontier_fair", window=2)
    report = result.as_dict()
    assert report["starvation"]["count"] == 0
    assert report["starvation"]["max_younger_bypass"] <= 1


def test_quality_work_auc_is_left_continuous_and_post_hoc():
    result = replay_windowed_arrivals(
        [[0], [1]],
        [0, 1],
        [0.0, 0.0],
        [0.2, 0.0],
        base_mean_quality=0.4,
        corpus_pages=4,
        batch_size=1,
        window=1,
        policy="fifo",
        random_seed=1,
    )
    assert result.completion_pages == (1.0, 2.0)
    assert result.quality_work_auc == pytest.approx(0.475)


def test_replay_rejects_noncausal_arrival_times():
    with pytest.raises(ValueError, match="monotonic"):
        replay_windowed_arrivals(
            [[0], [1]],
            [0, 1],
            [2.0, 1.0],
            [0.0, 0.0],
            base_mean_quality=0.0,
            corpus_pages=2,
            batch_size=1,
            window=1,
            policy="fifo",
            random_seed=1,
        )
