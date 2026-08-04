import numpy as np

from reprforge.cagr_faithful_replay import replay_cagr_comparison
from tools.analyze_multiobjective_oracle_headroom import (
    _candidate_grid,
    _finance_gate,
)


def _oracle_replay(cohorts, gains, *, batch_size=1, arrivals=None, **oracle):
    count = len(cohorts)
    return replay_cagr_comparison(
        cohorts,
        list(range(count)),
        np.zeros(count) if arrivals is None else arrivals,
        np.asarray(gains, dtype=np.float64),
        base_mean_quality=0.2,
        corpus_pages=len(set(page for cohort in cohorts for page in cohort)),
        request_batch_size=batch_size,
        window=64,
        policy="multiobjective_oracle",
        cache_capacity=80,
        arrival_clock="unit",
        oracle_lambda_quality=oracle.get("lambda_quality", 0.0),
        oracle_lambda_completion=oracle.get("lambda_completion", 0.0),
        oracle_lambda_deadline=oracle.get("lambda_deadline", 0.0),
        oracle_deadline_scale=oracle.get("deadline_scale", 64),
        oracle_future_wait_budget=oracle.get("future_wait_budget", 0),
    )


def test_registered_grid_is_exact_sixty_point_simplex() -> None:
    grid = _candidate_grid()
    assert len(grid) == 60
    assert len({_config_tuple(row) for row in grid}) == 60
    assert {
        (
            row["lambda_quality"],
            row["lambda_completion"],
            row["lambda_deadline"],
        )
        for row in grid
    } == {
        (quality / 4, completion / 4, (4 - quality - completion) / 4)
        for quality in range(5)
        for completion in range(5 - quality)
    }
    assert {row["deadline_scale"] for row in grid} == {64, 256}
    assert {row["future_wait_budget"] for row in grid} == {0, 16}


def _config_tuple(row):
    return tuple(sorted(row.items()))


def test_quality_density_uses_frozen_gain_and_exact_next_cost() -> None:
    # q0 has more raw gain but costs two units; q1 has better gain per cost.
    result = _oracle_replay(
        [[0, 1], [0]],
        [0.10, 0.09],
        lambda_quality=1.0,
    )
    assert result.dispatch_order == (1, 0)
    assert result.prefetch["events"] == 0


def test_completion_density_prefers_lower_charged_cost() -> None:
    result = _oracle_replay(
        [[0, 1, 2], [3]],
        [1.0, 0.0],
        lambda_completion=1.0,
    )
    assert result.dispatch_order == (1, 0)


def test_future_wait_cannot_serve_before_arrival_and_is_charged_to_elapsed() -> None:
    result = _oracle_replay(
        [[0], [1]],
        [0.1, 0.1],
        batch_size=8,
        arrivals=[0.0, 3.0],
        lambda_completion=1.0,
        future_wait_budget=4.0,
    )
    # The oracle waits only until the known future arrival at t=3, then the
    # atomic two-query batch costs two and publishes at t=5.
    assert result.dispatch_order == (0, 1)
    assert result.sojourn_unit_time == (5.0, 2.0)
    assert result.total_unit_work == 2.0
    assert result.oracle_future_wait == {
        "budget": 4.0,
        "events": 1,
        "total_unit_time": 3.0,
        "mean_unit_time": 3.0,
        "max_unit_time": 3.0,
    }


def _method(sojourn, work, regret, p95=100.0, starvation=0.0):
    return {
        "system": {
            "sojourn_unit_time": {"mean": sojourn, "p95": p95},
            "unit_work_per_query": work,
            "starvation": {"fraction": starvation},
        },
        "axes": {
            "elapsed_unit_time": {
                "normalized_quality_regret_auc": {"mean": regret}
            }
        },
    }


def _finance(oracle):
    return {
        model: {
            "methods": {
                "hr_selected_oracle": oracle[model],
                "bounded_cagr": _method(100.0, 100.0, 0.5),
                "frontier": _method(110.0, 110.0, 0.4),
            }
        }
        for model in ("burst", "poisson")
    }


def test_headroom_gate_requires_every_constraint_in_both_arrivals() -> None:
    oracle = {
        "burst": _method(94.0, 100.0, 0.4),
        # Primary constraints pass, but no axis improves by 5%.
        "poisson": _method(99.0, 99.0, 0.399),
    }
    gate = _finance_gate(_finance(oracle), {"lambda_quality": 1.0})
    assert gate["decision"] == "NO HEADROOM IN REGISTERED FAMILY"
    assert gate["checks"][0]["passes"] is True
    assert gate["checks"][1]["constraints"][
        "at_least_one_primary_improves_5_percent"
    ] is False


def test_headroom_gate_passes_one_five_percent_axis_per_arrival() -> None:
    oracle = {
        "burst": _method(94.0, 100.0, 0.4),
        "poisson": _method(100.0, 94.0, 0.4),
    }
    gate = _finance_gate(_finance(oracle), {"lambda_quality": 1.0})
    assert gate["decision"] == "HEADROOM GO"
    assert all(check["passes"] for check in gate["checks"])
