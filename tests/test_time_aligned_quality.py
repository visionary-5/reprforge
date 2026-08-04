import numpy as np
import pytest

from reprforge.cagr_faithful_replay import replay_cagr_comparison
from reprforge.time_aligned_quality import time_aligned_quality_metrics


def test_publication_trace_includes_wait_idle_and_atomic_batch_quality() -> None:
    result = replay_cagr_comparison(
        [[0], [1], [2]],
        [0, 1, 2],
        [0.0, 3.0, 10.0],
        np.asarray([0.3, -0.1, 0.2]),
        base_mean_quality=0.4,
        corpus_pages=3,
        request_batch_size=2,
        window=64,
        policy="cagr_faithful",
        cache_capacity=80,
        cagr_group_pool=64,
        cagr_grouping="fixed_jaccard",
        cagr_target_group_size=16,
        arrival_clock="unit",
        cagr_wait_budget=4.0,
        cagr_min_pending=2,
        cagr_cross_group_fill=True,
    )

    trace = result.quality_publication_trace
    assert [row["elapsed_unit_time"] for row in trace] == [0.0, 5.0, 11.0]
    assert [row["charged_unit_work"] for row in trace] == [0.0, 2.0, 3.0]
    assert [row["unique_compiled_pages"] for row in trace] == [0, 2, 3]
    assert [row["published_queries"] for row in trace] == [0, 2, 3]
    assert trace[1]["batch_queries"] == (0, 1)
    assert trace[1]["mean_quality"] == pytest.approx(0.4 + 0.2 / 3)
    assert trace[-1]["mean_quality"] == pytest.approx(0.4 + 0.4 / 3)


def test_prefetch_is_after_current_publication_and_before_next_publication() -> None:
    result = replay_cagr_comparison(
        [[0], [1]],
        [0, 1],
        [0.0, 0.0],
        np.asarray([0.2, 0.2]),
        base_mean_quality=0.2,
        corpus_pages=2,
        request_batch_size=1,
        window=64,
        policy="cagr_faithful",
        cache_capacity=1,
        cagr_group_pool=64,
        cagr_grouping="fixed_jaccard",
        cagr_target_group_size=1,
        arrival_clock="unit",
        cagr_cross_group_fill=False,
    )

    trace = result.quality_publication_trace
    # q0 builds and publishes at 1.  Then q1 is prefetched for one charged
    # unit; q1's hit publishes at 2 without further service.
    assert [row["elapsed_unit_time"] for row in trace] == [0.0, 1.0, 2.0]
    assert [row["charged_unit_work"] for row in trace] == [0.0, 1.0, 2.0]
    assert [row["mean_quality"] for row in trace] == pytest.approx([0.2, 0.3, 0.4])
    assert result.prefetch["events"] == result.prefetch["useful"] == 1


def test_shared_horizon_auc_budget_and_sustained_targets_use_step_semantics() -> None:
    points = [
        {"elapsed_unit_time": 0.0, "mean_quality": 0.2},
        {"elapsed_unit_time": 2.0, "mean_quality": 0.4},
        {"elapsed_unit_time": 5.0, "mean_quality": 0.3},
        {"elapsed_unit_time": 8.0, "mean_quality": 0.6},
    ]
    metrics = time_aligned_quality_metrics(
        points,
        axis="elapsed_unit_time",
        common_horizon=10.0,
        base_quality=0.2,
        final_quality=0.6,
    )

    assert metrics["normalized_quality_gain_auc"] == pytest.approx(0.425)
    assert metrics["normalized_quality_regret_auc"] == pytest.approx(0.575)
    # The first 50% crossing at t=2 is not sustained because quality drops at 5.
    assert metrics["attainment"]["sustained_t50"] == {
        "coordinate": 8.0,
        "common_horizon_fraction": 0.8,
    }
    assert metrics["attainment"]["sustained_t90"] == {
        "coordinate": 8.0,
        "common_horizon_fraction": 0.8,
    }
    assert metrics["fixed_budgets"]["budget_25_percent"][
        "mean_quality"
    ] == pytest.approx(0.4)
    # A publication exactly at the 50% budget is visible at that budget.
    assert metrics["fixed_budgets"]["budget_50_percent"][
        "mean_quality"
    ] == pytest.approx(0.3)


def test_final_quality_is_held_to_larger_common_horizon() -> None:
    points = [
        {"charged_unit_work": 0.0, "mean_quality": 0.2},
        {"charged_unit_work": 5.0, "mean_quality": 0.6},
    ]
    metrics = time_aligned_quality_metrics(
        points,
        axis="charged_unit_work",
        common_horizon=10.0,
        base_quality=0.2,
        final_quality=0.6,
    )
    assert metrics["method_endpoint_fraction"] == 0.5
    assert metrics["normalized_quality_gain_auc"] == pytest.approx(0.5)
    assert metrics["normalized_quality_regret_auc"] == pytest.approx(0.5)
    assert metrics["fixed_budgets"]["budget_75_percent"][
        "mean_quality"
    ] == pytest.approx(0.6)


def test_nonpositive_final_gain_makes_normalized_metrics_null() -> None:
    metrics = time_aligned_quality_metrics(
        [
            {"unique_compiled_pages": 0, "mean_quality": 0.5},
            {"unique_compiled_pages": 10, "mean_quality": 0.4},
        ],
        axis="unique_compiled_pages",
        common_horizon=10,
        base_quality=0.5,
        final_quality=0.4,
    )
    assert metrics["positive_final_gain_defined"] is False
    assert metrics["normalized_quality_regret_auc"] is None
    assert metrics["attainment"] == {
        "sustained_t50": None,
        "sustained_t90": None,
    }
    assert metrics["fixed_budgets"]["budget_50_percent"][
        "mean_quality"
    ] == 0.5
