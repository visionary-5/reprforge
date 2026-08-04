import numpy as np

from reprforge.cagr_faithful_replay import replay_cagr_comparison
from tools.analyze_hard_fair_oracle import (
    BASE_ORACLE_CONFIG,
    _candidate_grid,
    _finance_gate,
    _load_selection_first_inputs,
    _select_knee,
)
from tools.analyze_cagr_bounded_wait import _json_digest


def _replay(budget, *, batch_size=1):
    cohorts = [[0, 1, 2], [3], [4], [5]]
    return replay_cagr_comparison(
        cohorts,
        [0, 1, 2, 3],
        np.zeros(4),
        np.zeros(4),
        base_mean_quality=0.2,
        corpus_pages=6,
        request_batch_size=batch_size,
        window=64,
        policy="multiobjective_oracle",
        cache_capacity=80,
        arrival_clock="unit",
        oracle_lambda_quality=BASE_ORACLE_CONFIG["lambda_quality"],
        oracle_lambda_completion=BASE_ORACLE_CONFIG["lambda_completion"],
        oracle_lambda_deadline=BASE_ORACLE_CONFIG["lambda_deadline"],
        oracle_deadline_scale=BASE_ORACLE_CONFIG["deadline_scale"],
        oracle_future_wait_budget=0,
        oracle_bypass_budget=budget,
    )


def test_registered_grid_only_changes_hard_bypass_budget() -> None:
    grid = _candidate_grid()
    assert len(grid) == 4
    assert [row["bypass_budget"] for row in grid] == [8, 16, 32, 64]
    assert all(
        {key: value for key, value in row.items() if key != "bypass_budget"}
        == BASE_ORACLE_CONFIG
        for row in grid
    )


def test_hr_selection_precedes_all_finance_bearing_reads(tmp_path) -> None:
    events = []
    selection_frozen = False
    selected = {**BASE_ORACLE_CONFIG, "bypass_budget": 32}

    def domain_loader(path, candidate_k):
        nonlocal selection_frozen
        assert candidate_k == 20
        if path.name == "finance":
            assert selection_frozen
        events.append(f"load_domain:{path.name}")
        return {"domain": path.name}

    def hr_evaluator(data):
        nonlocal selection_frozen
        assert data == {"domain": "hr"}
        events.append("evaluate_and_freeze_hr")
        selection_frozen = True
        return {
            "selection": {
                "selected": selected,
                "selected_config_sha256": _json_digest(selected),
            }
        }

    def reference_loader(path):
        assert selection_frozen
        events.append("read_finance_bearing_reference")
        return {"path": str(path)}

    loaded = _load_selection_first_inputs(
        tmp_path,
        tmp_path / "time-reference.json",
        domain_loader=domain_loader,
        hr_evaluator=hr_evaluator,
        reference_loader=reference_loader,
    )
    assert events == [
        "load_domain:hr",
        "evaluate_and_freeze_hr",
        "read_finance_bearing_reference",
        "load_domain:finance",
    ]
    assert loaded[2] == selected
    assert loaded[3] == _json_digest(selected)


def test_hard_constraint_forces_old_query_before_budget_violation() -> None:
    result = _replay(1)
    # completion utility first selects cheap q1 over expensive old q0.  A
    # second younger selection would give q0 two bypasses, so q0 is forced.
    assert result.dispatch_order == (1, 0, 2, 3)
    assert result.bypass_count == (1, 0, 0, 0)
    assert result.oracle_hard_fairness == {
        "configured_bypass_budget": 1,
        "selection_count": 4,
        "forced_selection_count": 1,
        "forced_selection_fraction": 0.25,
        "protected_unique_queries": 1,
        "protected_query_fraction": 0.25,
        "max_final_younger_bypass": 1,
        "budget_violation_count": 0,
    }


def test_zero_budget_has_no_deadlock_and_reduces_to_fifo_order() -> None:
    result = _replay(0, batch_size=2)
    assert result.dispatch_order == (0, 1, 2, 3)
    assert result.oracle_hard_fairness["budget_violation_count"] == 0
    assert result.oracle_hard_fairness["max_final_younger_bypass"] == 0


def test_selected_policy_dispatch_is_invariant_to_unobserved_qrel_gains() -> None:
    cohorts = [[0, 1, 2], [3], [4], [5]]
    common = dict(
        cohorts=cohorts,
        arrival_order=[0, 1, 2, 3],
        arrival_times=np.zeros(4),
        base_mean_quality=0.2,
        corpus_pages=6,
        request_batch_size=1,
        window=64,
        policy="multiobjective_oracle",
        cache_capacity=80,
        arrival_clock="unit",
        oracle_lambda_quality=0.0,
        oracle_lambda_completion=0.75,
        oracle_lambda_deadline=0.25,
        oracle_deadline_scale=256,
        oracle_future_wait_budget=0,
        oracle_bypass_budget=32,
    )
    first = replay_cagr_comparison(
        quality_gain=np.asarray([1.0, 0.0, 0.0, 0.0]), **common
    )
    second = replay_cagr_comparison(
        quality_gain=np.asarray([0.0, 0.0, 0.0, 1.0]), **common
    )
    assert first.dispatch_order == second.dispatch_order
    assert first.total_unit_work == second.total_unit_work


def test_online_timer_matches_registered_wait_except_explicit_stream_end() -> None:
    common = dict(
        cohorts=[[0], [1]],
        arrival_order=[0, 1],
        arrival_times=np.asarray([0.0, 3.0]),
        quality_gain=np.zeros(2),
        base_mean_quality=0.2,
        corpus_pages=2,
        request_batch_size=8,
        window=64,
        policy="multiobjective_oracle",
        cache_capacity=80,
        arrival_clock="unit",
        oracle_lambda_quality=0.0,
        oracle_lambda_completion=0.75,
        oracle_lambda_deadline=0.25,
        oracle_deadline_scale=256,
        oracle_future_wait_budget=4,
        oracle_bypass_budget=32,
    )
    registered = replay_cagr_comparison(**common)
    timer_only = replay_cagr_comparison(
        **common, oracle_wait_through_stream_end=True
    )
    assert registered.dispatch_order == timer_only.dispatch_order == (0, 1)
    assert registered.total_unit_work == timer_only.total_unit_work == 2.0
    assert registered.sojourn_unit_time == (5.0, 2.0)
    assert timer_only.sojourn_unit_time == (6.0, 3.0)
    assert registered.oracle_future_wait["total_unit_time"] == 3.0
    assert timer_only.oracle_future_wait["total_unit_time"] == 4.0


def _candidate(name, budget, value, qualified=True):
    return {
        "oracle_name": name,
        "config": {**BASE_ORACLE_CONFIG, "bypass_budget": budget},
        "config_sha256": name,
        "qualified": qualified,
        "max_primary_ratio": value,
        "mean_primary_ratio": value,
        "pareto_vector": [value] * 6 + [budget / 64],
    }


def test_knee_is_selected_from_qualified_pareto_set() -> None:
    selection = _select_knee(
        [
            _candidate("b8", 8, 0.90),
            _candidate("b16", 16, 0.80),
            _candidate("b32", 32, 0.75),
            _candidate("unsafe", 64, 0.60, qualified=False),
        ]
    )
    assert selection["qualified_count"] == 3
    assert selection["pareto_count"] == 3
    assert selection["selected_oracle_name"] == "b16"
    assert selection["selected"]["bypass_budget"] == 16


def _method(sojourn, work, regret, p95=100.0, starvation=0.0, violations=0):
    return {
        "system": {
            "dispatch_complete": True,
            "final_union_pages": [10],
            "sojourn_unit_time": {"mean": sojourn, "p95": p95},
            "unit_work_per_query": work,
            "starvation": {"fraction": starvation},
        },
        "axes": {
            "elapsed_unit_time": {
                "normalized_quality_regret_auc": {"mean": regret}
            }
        },
        "hard_fairness": {"budget_violation_count": violations},
    }


def _finance(oracle):
    return {
        model: {
            "methods": {
                "hr_selected_hard_oracle": oracle[model],
                "bounded_cagr": _method(100.0, 100.0, 0.5),
                "frontier": _method(110.0, 110.0, 0.4),
            }
        }
        for model in ("burst", "poisson")
    }


def test_finance_gate_requires_hard_constraint_and_both_arrivals() -> None:
    oracle = {
        "burst": _method(94.0, 100.0, 0.4),
        "poisson": _method(94.0, 100.0, 0.4, violations=1),
    }
    gate = _finance_gate(
        _finance(oracle), {"bypass_budget": 16}, expected_union=10
    )
    assert gate["decision"] == "NO HEADROOM IN REGISTERED HARD-FAIR FAMILY"
    assert gate["checks"][0]["passes"] is True
    assert gate["checks"][1]["constraints"][
        "hard_budget_violation_zero"
    ] is False


def test_finance_gate_go_is_finite_oracle_headroom_only() -> None:
    oracle = {
        "burst": _method(94.0, 100.0, 0.4),
        "poisson": _method(100.0, 94.0, 0.4),
    }
    gate = _finance_gate(
        _finance(oracle), {"bypass_budget": 16}, expected_union=10
    )
    assert gate["decision"] == "HARD-FAIR HEADROOM GO"
    assert all(check["passes"] for check in gate["checks"])
    assert "not a deployable method" in gate["scope"]
