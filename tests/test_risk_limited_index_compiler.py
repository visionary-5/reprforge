import numpy as np
import pytest

from reprforge.risk_limited_index_compiler import (
    draw_stratified_sample,
    quantile_risk_strata,
    select_cheapest_certified_plan,
    simulate_label_efficiency,
    stratified_mean_upper_bound,
    stratified_sample_sizes,
)


def test_quantile_strata_are_balanced_and_stable_under_ties():
    strata = quantile_risk_strata(np.ones(10), strata=4)
    assert np.bincount(strata).tolist() == [3, 2, 3, 2]
    assert strata.tolist() == sorted(strata.tolist())


def test_boundary_allocation_oversamples_high_risk_and_keeps_every_stratum():
    assignments = np.repeat(np.arange(4), 10)
    uniform = stratified_sample_sizes(assignments, budget=16, allocation="uniform")
    boundary = stratified_sample_sizes(assignments, budget=16, allocation="boundary")
    assert uniform.tolist() == [4, 4, 4, 4]
    assert boundary.sum() == 16
    assert np.all(boundary >= 2)
    assert boundary[-1] > boundary[0]


def test_stratified_estimator_corrects_high_risk_oversampling():
    assignments = np.repeat(np.arange(4), 10)
    values = np.repeat([0.0, 0.0, 0.0, 0.4], 10)
    sizes = stratified_sample_sizes(assignments, budget=16, allocation="boundary")
    sample = draw_stratified_sample(
        assignments, sizes, rng=np.random.default_rng(7)
    )
    result = stratified_mean_upper_bound(
        values, assignments, sample, alpha=0.05
    )
    assert values[sample].mean() > values.mean()
    assert result["estimate"] == pytest.approx(values.mean())
    assert result["standard_error"] == pytest.approx(0.0)


def test_selector_abstains_when_no_compressed_plan_is_certified():
    assignments = np.repeat(np.arange(2), 10)
    sample = np.arange(20)
    report = select_cheapest_certified_plan(
        {
            "cheap": {
                "ndcg_at_10": np.full(20, 0.03),
                "recall_at_100": np.zeros(20),
            },
            "full": {
                "ndcg_at_10": np.zeros(20),
                "recall_at_100": np.zeros(20),
            },
        },
        {"cheap": 0.25, "full": 1.0},
        assignments,
        sample,
    )
    assert report["selected_plan"] == "full"
    assert report["abstained_to_fallback"]


def test_selector_chooses_cheapest_certified_plan():
    assignments = np.repeat(np.arange(2), 10)
    sample = np.arange(20)
    zeros = np.zeros(20)
    report = select_cheapest_certified_plan(
        {
            "small": {"ndcg_at_10": zeros, "recall_at_100": zeros},
            "large": {"ndcg_at_10": zeros, "recall_at_100": zeros},
            "full": {"ndcg_at_10": zeros, "recall_at_100": zeros},
        },
        {"small": 0.4, "large": 0.8, "full": 1.0},
        assignments,
        sample,
    )
    assert report["selected_plan"] == "small"
    assert not report["abstained_to_fallback"]


def test_label_efficiency_simulation_reports_both_sampling_designs():
    proxy = np.linspace(0.0, 1.0, 40)
    zeros = np.zeros(40)
    report = simulate_label_efficiency(
        {
            "unsafe": {
                "ndcg_at_10": np.where(proxy > 0.8, 0.2, 0.0),
                "recall_at_100": zeros,
            },
            "safe": {"ndcg_at_10": zeros, "recall_at_100": zeros},
            "full": {"ndcg_at_10": zeros, "recall_at_100": zeros},
        },
        {"unsafe": 0.3, "safe": 0.7, "full": 1.0},
        proxy,
        budgets=[16],
        trials=10,
        seed=5,
    )
    assert report["oracle_plan"] == "safe"
    assert set(report["budgets"]["16"]) == {"uniform", "boundary_stratified"}
    assert report["budgets"]["16"]["boundary_stratified"][
        "sample_sizes_by_stratum"
    ][-1] > report["budgets"]["16"]["boundary_stratified"][
        "sample_sizes_by_stratum"
    ][0]
