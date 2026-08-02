import numpy as np

from reprforge.pairwise_view_admission import BoundaryPair
from reprforge.physical_cost import AtomicMaterializationCostModel
from reprforge.risk_constrained_admission import (
    estimate_plan_cost,
    paired_bootstrap_upper_loss,
    select_cost_aware_pairs,
    select_risk_bounded_cost_fraction,
)


def test_cost_aware_pair_selection_accounts_for_reuse_events() -> None:
    # Both edges have equal value. Pages 1/2 occur once each, while pages 3/4
    # occur in every query, so a score-event-aware plan buys the first pair.
    candidates = np.asarray(
        [
            [1, 2, 3, 4],
            [5, 6, 3, 4],
            [7, 8, 3, 4],
        ]
    )
    pairs = [
        BoundaryPair(0, 1, 2, 2, 5.0),
        BoundaryPair(1, 3, 4, 2, 5.0),
    ]
    model = AtomicMaterializationCostModel(
        batch_size=4,
        setup_ms=0.0,
        page_ms=1.0,
        batch_ms=0.0,
        score_event_ms=1.0,
    )
    cheap_pair_cost = estimate_plan_cost(candidates, {1, 2}, model).total_ms
    plan = select_cost_aware_pairs(
        pairs,
        candidates,
        model,
        time_budget_ms=cheap_pair_cost,
    )

    assert plan.admission.selected_pages == frozenset({1, 2})
    assert plan.admission.covered_weight == 5.0


def test_paired_bootstrap_reports_direction_of_extra_errors() -> None:
    baseline = np.asarray([True, True, False, False] * 10)
    same = paired_bootstrap_upper_loss(
        baseline,
        baseline,
        confidence=0.9,
        bootstrap_samples=200,
    )
    worse = paired_bootstrap_upper_loss(
        np.zeros_like(baseline),
        baseline,
        confidence=0.9,
        bootstrap_samples=200,
    )

    assert same == (0.0, 0.0)
    assert worse[0] > 0.0
    assert worse[1] >= worse[0]


def test_risk_fraction_falls_back_on_insufficient_support() -> None:
    baseline = np.ones(8, dtype=bool)
    result = select_risk_bounded_cost_fraction(
        {0.5: baseline.copy(), 1.0: baseline.copy()},
        baseline,
        [f"g{index}" for index in range(8)],
        risk_tolerance=0.05,
        confidence=0.9,
    )

    assert result.fallback_to_baseline
    assert result.selected_fraction == 1.0
    assert result.best_attempt_upper_extra_disagreement is None


def test_risk_fraction_uses_cross_group_outcomes() -> None:
    baseline = np.asarray([True, False] * 10)
    candidate = baseline.copy()
    result = select_risk_bounded_cost_fraction(
        {0.5: candidate, 1.0: candidate},
        baseline,
        [f"g{index // 2}" for index in range(20)],
        risk_tolerance=0.05,
        confidence=0.9,
        bootstrap_samples=200,
    )

    assert not result.fallback_to_baseline
    assert result.selected_fraction == 0.5
    assert result.upper_extra_disagreement == 0.0
