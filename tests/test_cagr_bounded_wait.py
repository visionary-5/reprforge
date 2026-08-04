import numpy as np

from reprforge.cagr_faithful_replay import replay_cagr_comparison
from tools.analyze_cagr_bounded_wait import (
    _aggregate,
    _candidate_grid,
    _finance_gate,
)


def test_candidate_grid_is_exact_preregistered_twelve():
    grid = _candidate_grid()

    assert len(grid) == 12
    assert {row["wait_budget"] for row in grid} == {0, 4, 16, 64}
    assert {row["min_pending"] for row in grid} == {4, 8, 16}
    assert all(row["target_group_size"] == 16 for row in grid)
    assert all(row["group_pool"] == 64 for row in grid)
    assert all(row["capacity"] == 80 for row in grid)
    assert all(row["cross_group_fill"] is True for row in grid)


def test_aggregate_separates_elapsed_sojourn_from_charged_work():
    replay = replay_cagr_comparison(
        [[0], [1]],
        [0, 1],
        [0.0, 100.0],
        np.zeros(2),
        base_mean_quality=0.0,
        corpus_pages=2,
        request_batch_size=2,
        window=64,
        policy="cagr_faithful",
        cache_capacity=80,
        cagr_group_pool=64,
        cagr_grouping="fixed_jaccard",
        cagr_target_group_size=16,
        arrival_clock="unit",
        cagr_wait_budget=4,
        cagr_min_pending=2,
        cagr_cross_group_fill=True,
    )

    aggregate = _aggregate([replay], with_quality=False)

    assert aggregate["sojourn_unit_time"]["mean"] == 3.0
    assert aggregate["total_unit_work"] == 2.0
    assert aggregate["unit_work_per_query"] == 1.0
    assert aggregate["dispatch_complete"] is True


def _aggregate_stub(sojourn, work):
    return {
        "sojourn_unit_time": {"mean": sojourn, "p95": sojourn},
        "unit_work_per_query": work,
        "normalized_quality_regret_auc": {"mean": 0.2},
        "starvation": {"fraction": 0.0},
    }


def test_finance_gate_requires_five_percent_on_both_primary_metrics():
    selected = {"wait_budget": 16, "min_pending": 8}
    finance = {}
    for model in ("burst", "poisson"):
        finance[model] = {
            "hr_selected_bounded_cagr": {
                "aggregate": _aggregate_stub(100.0, 100.0)
            },
            # 10% sojourn advantage but only 4% unit-work advantage.
            "frontier": {"aggregate": _aggregate_stub(90.0, 96.0)},
        }

    gate = _finance_gate(finance, selected)

    assert gate["decision"] == "STOP/DOWNGRADE"
    assert all(not check["passes"] for check in gate["checks"])
    assert all(
        np.isclose(check["frontier_unit_work_advantage"], 0.04)
        for check in gate["checks"]
    )


def test_finance_gate_never_vacuously_passes_without_deployable_hr_config():
    gate = _finance_gate({}, None)

    assert gate["decision"] == "STOP/DOWNGRADE"
    assert gate["checks"] == []
    assert gate["no_deployable_hr_selection"] is True
