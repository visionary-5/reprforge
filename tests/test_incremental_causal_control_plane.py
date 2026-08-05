import inspect
import itertools
from collections import OrderedDict

import numpy as np

from reprforge.cagr_faithful_replay import replay_cagr_comparison
from reprforge.incremental_causal_control_plane import (
    EfficientDelaySchedulingD32,
    IncrementalHardBudgetFrontier,
    TransparentInstrumentedHardBudgetFrontier,
)


def _replay(policy: str):
    cohorts = [
        [0, 1, 2, 3],
        [3, 4],
        [5, 6, 7],
        [0, 7],
        [8, 9],
        [1, 9],
        [10, 11],
        [2, 11],
        [12, 13],
        [4, 13],
        [14, 15],
        [6, 15],
        [16, 17],
        [8, 17],
        [18, 19],
        [10, 19],
    ]
    return replay_cagr_comparison(
        cohorts,
        np.asarray([5, 0, 9, 1, 13, 2, 15, 3, 4, 6, 7, 8, 10, 11, 12, 14]),
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 9, 9, 9, 9, 20, 20, 20, 20]),
        np.linspace(0.0, 0.1, 16),
        base_mean_quality=0.2,
        corpus_pages=20,
        request_batch_size=8,
        window=64,
        policy=policy,
        cache_capacity=80,
        arrival_clock="unit",
    )


def _semantic_tuple(replay):
    return (
        replay.dispatch_order,
        replay.completion_pages,
        replay.completion_unit_cost,
        replay.wait_unit_time,
        replay.sojourn_unit_time,
        replay.bypass_count,
        replay.final_unique_pages,
        replay.total_unit_work,
        replay.cache,
        replay.quality_publication_trace,
        replay.oracle_future_wait,
        replay.oracle_hard_fairness,
    )


def test_incremental_and_instrumented_are_exact_reference_replays() -> None:
    reference = _replay("hard_budget_frontier")
    transparent = _replay("hard_budget_frontier_transparent")
    incremental = _replay("hard_budget_frontier_incremental")
    assert _semantic_tuple(reference) == _semantic_tuple(transparent)
    assert _semantic_tuple(reference) == _semantic_tuple(incremental)


def test_incremental_exact_cost_handles_lru_eviction_before_late_hit() -> None:
    baseline = TransparentInstrumentedHardBudgetFrontier()
    optimized = IncrementalHardBudgetFrontier()
    cohort = [0, 1, 2, 100]
    baseline.arrival(0, cohort, 0.0)
    optimized.arrival(0, cohort, 0.0)
    compiled = set(range(279))
    lru = OrderedDict((page, None) for page in (100, *range(200, 279)))
    baseline_cost, _, _ = baseline._simulate_query(
        0, compiled, lru, mutate=False
    )
    optimized_cost = optimized._marginal_cost(0, lru)
    assert baseline_cost == optimized_cost == 4.0


def test_uniform_b32_head_frontier_matches_all_older_feasibility() -> None:
    budget = 2
    for order in itertools.permutations(range(6)):
        for prefix_size in range(6):
            dispatched = order[:prefix_size]
            pending = list(order[prefix_size:])
            bypass = {
                victim: sum(selected > victim for selected in dispatched)
                for victim in pending
            }
            if any(value > budget for value in bypass.values()):
                continue
            hard = {
                candidate
                for candidate in pending
                if all(
                    bypass[victim] + 1 <= budget
                    for victim in pending
                    if victim < candidate
                )
            }
            head = min(pending)
            frontier = {head} if bypass[head] >= budget else set(pending)
            assert hard == frontier


def test_operation_v2_counts_same_items_and_reduces_transparent_work() -> None:
    transparent = _replay("hard_budget_frontier_transparent")
    incremental = _replay("hard_budget_frontier_incremental")
    baseline = transparent.scheduler_control["operation_proxy_v2"]
    optimized = incremental.scheduler_control["operation_proxy_v2"]
    assert baseline["state_copy_items"] > optimized["state_copy_items"]
    assert baseline["cohort_order_items"] > optimized["cohort_order_items"]
    assert baseline["total"] > optimized["total"]
    assert incremental.scheduler_control["optimization"] == {
        "cached_sorted_cohorts": True,
        "candidate_compiled_copy": False,
        "uniform_budget_head_frontier": True,
    }


def test_all_control_plane_interfaces_exclude_qrel_and_future_arrays() -> None:
    forbidden = {
        "qrel",
        "quality_gain",
        "future_arrivals",
        "arrival_order",
        "arrival_times",
        "query_count",
    }
    for scheduler in (
        TransparentInstrumentedHardBudgetFrontier,
        IncrementalHardBudgetFrontier,
        EfficientDelaySchedulingD32,
    ):
        for method in ("__init__", "arrival", "timer", "dispatch", "sync_index_state"):
            assert set(
                inspect.signature(getattr(scheduler, method)).parameters
            ).isdisjoint(forbidden)


def test_efficient_delay_has_exact_d32_bound() -> None:
    cohorts = [list(range(20))] + [[20 + query] for query in range(40)]
    result = replay_cagr_comparison(
        cohorts,
        list(range(41)),
        np.zeros(41),
        np.zeros(41),
        base_mean_quality=0.2,
        corpus_pages=60,
        request_batch_size=8,
        window=64,
        policy="delay_scheduling_d32_control",
        cache_capacity=80,
        arrival_clock="unit",
    )
    assert result.dispatch_order.index(0) == 32
    assert max(result.bypass_count) == 32
    assert result.scheduler_control["adaptation_not_official_implementation"]
