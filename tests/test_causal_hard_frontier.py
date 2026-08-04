import inspect

import numpy as np
import pytest

from reprforge.cagr_faithful_replay import replay_cagr_comparison
from reprforge.causal_hard_frontier import HardBudgetFrontier


def _replay(policy, gains, **overrides):
    kwargs = dict(
        cohorts=[[0, 1, 2], [3], [4], [5], [0], [3], [4], [5]],
        arrival_order=list(range(8)),
        arrival_times=np.zeros(8),
        quality_gain=np.asarray(gains, dtype=np.float64),
        base_mean_quality=0.2,
        corpus_pages=6,
        request_batch_size=8,
        window=64,
        policy=policy,
        cache_capacity=80,
        arrival_clock="unit",
    )
    kwargs.update(overrides)
    return replay_cagr_comparison(**kwargs)


def test_public_scheduler_api_has_no_qrel_or_future_trace_parameters() -> None:
    forbidden = {
        "quality_gain",
        "qrel",
        "gains",
        "arrival_order",
        "arrival_times",
        "future_arrivals",
        "query_count",
        "end_of_stream",
    }
    for method_name in (
        "__init__",
        "arrival",
        "timer",
        "ready",
        "dispatch",
        "sync_index_state",
    ):
        parameters = set(
            inspect.signature(getattr(HardBudgetFrontier, method_name)).parameters
        )
        assert parameters.isdisjoint(forbidden)


def test_scheduler_state_contains_no_qrel_or_future_trace() -> None:
    scheduler = HardBudgetFrontier()
    scheduler.arrival(7, [1, 2], 3.0)
    state_names = set(vars(scheduler))
    assert not any("quality" in name or "qrel" in name for name in state_names)
    assert not any("future" in name or "arrival_times" in name for name in state_names)


def test_policy_rejects_runtime_system_parameter_tuning() -> None:
    with pytest.raises(ValueError, match="frozen batch/window/cache"):
        _replay(
            "hard_budget_frontier",
            [0.0] * 8,
            request_batch_size=4,
        )


def test_policy_dispatch_and_work_are_qrel_invariant() -> None:
    first = _replay("hard_budget_frontier", [1.0] + [0.0] * 7)
    second = _replay("hard_budget_frontier", [0.0] * 7 + [1.0])
    assert first.dispatch_order == second.dispatch_order
    assert first.total_unit_work == second.total_unit_work
    assert first.bypass_count == second.bypass_count


def test_causal_policy_matches_b32_reference_on_observed_event_trace() -> None:
    gains = [0.1] * 8
    causal = _replay("hard_budget_frontier", gains)
    reference = _replay(
        "multiobjective_oracle",
        gains,
        oracle_lambda_quality=0.0,
        oracle_lambda_completion=0.75,
        oracle_lambda_deadline=0.25,
        oracle_deadline_scale=256,
        oracle_future_wait_budget=16,
        oracle_bypass_budget=32,
    )
    assert causal.dispatch_order == reference.dispatch_order
    assert causal.completion_unit_cost == reference.completion_unit_cost
    assert causal.sojourn_unit_time == reference.sojourn_unit_time
    assert causal.total_unit_work == reference.total_unit_work
    assert causal.bypass_count == reference.bypass_count
    assert causal.final_unique_pages == reference.final_unique_pages


def test_no_end_signal_waits_for_timer_without_changing_work() -> None:
    common = dict(
        cohorts=[[0], [1]],
        arrival_order=[0, 1],
        arrival_times=np.asarray([0.0, 3.0]),
        quality_gain=np.zeros(2),
        base_mean_quality=0.2,
        corpus_pages=2,
        request_batch_size=8,
        window=64,
        cache_capacity=80,
        arrival_clock="unit",
    )
    causal = replay_cagr_comparison(policy="hard_budget_frontier", **common)
    reference = replay_cagr_comparison(
        policy="multiobjective_oracle",
        oracle_lambda_quality=0.0,
        oracle_lambda_completion=0.75,
        oracle_lambda_deadline=0.25,
        oracle_deadline_scale=256,
        oracle_future_wait_budget=16,
        oracle_bypass_budget=32,
        oracle_wait_through_stream_end=True,
        **common,
    )
    assert causal.dispatch_order == reference.dispatch_order == (0, 1)
    assert causal.sojourn_unit_time == reference.sojourn_unit_time == (18.0, 15.0)
    assert causal.total_unit_work == reference.total_unit_work == 2.0


def test_visible_window_is_refilled_after_each_slot_beyond_w64() -> None:
    cohorts = [[3 * query, 3 * query + 1, 3 * query + 2] for query in range(64)]
    cohorts.append([192])
    common = dict(
        cohorts=cohorts,
        arrival_order=list(range(65)),
        arrival_times=np.zeros(65),
        quality_gain=np.zeros(65),
        base_mean_quality=0.2,
        corpus_pages=193,
        request_batch_size=8,
        window=64,
        cache_capacity=80,
        arrival_clock="unit",
    )
    causal = replay_cagr_comparison(policy="hard_budget_frontier", **common)
    reference = replay_cagr_comparison(
        policy="multiobjective_oracle",
        oracle_lambda_quality=0.0,
        oracle_lambda_completion=0.75,
        oracle_lambda_deadline=0.25,
        oracle_deadline_scale=256,
        oracle_future_wait_budget=16,
        oracle_bypass_budget=32,
        **common,
    )
    # q64 starts just outside W64, enters after q0 frees a slot, and its
    # one-page completion cost makes it the second selection.
    assert causal.dispatch_order[:2] == reference.dispatch_order[:2] == (0, 64)
    assert causal.dispatch_order == reference.dispatch_order
    assert causal.total_unit_work == reference.total_unit_work
    assert causal.bypass_count == reference.bypass_count
