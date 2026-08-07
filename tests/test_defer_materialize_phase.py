import pytest

from reprforge.defer_materialize_phase import (
    break_even_queries,
    history_policy_no_regression,
    incremental_winner,
    smallest_oracle_quality_plan,
)


def test_selects_smallest_oracle_plan_meeting_both_quality_constraints():
    curve = {
        "0.1": {"ndcg_gain_recovery": 0.95, "query_hit_at_20": 0.89, "selected_page_fraction": 0.1, "selected_pages": 10},
        "0.2": {"ndcg_gain_recovery": 1.0, "query_hit_at_20": 0.90, "selected_page_fraction": 0.2, "selected_pages": 20},
    }
    selected = smallest_oracle_quality_plan(
        curve, full_hit=0.90, minimum_gain_recovery=0.9, maximum_hit_loss=0.02
    )
    assert selected is not None
    assert selected["selected_pages"] == 10


def test_history_no_regression_rejects_all_harmful_budgets():
    curve = {
        "0.0": {"ndcg_at_10": {"mean": 0.5}},
        "0.1": {"ndcg_at_10": {"mean": 0.49}},
        "0.2": {"ndcg_at_10": {"mean": 0.48}},
    }
    result = history_policy_no_regression(curve)
    assert result["any_nonzero_budget_passes"] is False
    assert result["best_delta_vs_base"] == pytest.approx(-0.01)


def test_break_even_and_incremental_winner():
    assert break_even_queries(10.0, 0.5, 2) == 10.0
    early = incremental_winner(
        queries=1,
        verifier_page_seconds=0.5,
        avoided_pages=2,
        oracle_partial_build_seconds=10,
        current_stack_build_seconds=30,
        full_build_seconds=20,
    )
    late = incremental_winner(
        queries=100,
        verifier_page_seconds=0.5,
        avoided_pages=2,
        oracle_partial_build_seconds=10,
        current_stack_build_seconds=30,
        full_build_seconds=20,
    )
    assert early["winner"] == "dvi_defer"
    assert late["winner"] == "oracle_partial"


def test_invalid_break_even_inputs():
    with pytest.raises(ValueError):
        break_even_queries(1.0, 0.0, 1)
