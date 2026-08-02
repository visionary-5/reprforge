import numpy as np

from reprforge.sparse_what_if import (
    build_estimated_boundary_pairs,
    fit_sparse_boundary_risk,
    select_sparse_probe_queries,
)


def test_sparse_probe_plan_is_group_balanced_and_deterministic() -> None:
    locator = np.asarray(
        [[2.0, 1.0, margin, 0.0] for margin in np.linspace(0.9, 0.1, 10)]
    )
    groups = ["a"] * 6 + ["b"] * 4
    first = select_sparse_probe_queries(
        locator,
        cutoff=2,
        fraction=0.5,
        groups=groups,
    )
    second = select_sparse_probe_queries(
        locator,
        cutoff=2,
        fraction=0.5,
        groups=groups,
    )

    assert np.array_equal(first.query_indices, second.query_indices)
    assert len(first.query_indices) == 5
    assert sum(index < 6 for index in first.query_indices) == 3
    assert first.candidate_events == 20


def test_sparse_risk_model_learns_rank_and_margin_boundary_signal() -> None:
    locator = np.asarray(
        [
            [2.0, 1.0, 0.95, 0.0],
            [2.0, 1.0, 0.90, 0.0],
            [2.0, 1.0, 0.10, 0.0],
            [2.0, 1.0, 0.05, 0.0],
        ]
    )
    visual = np.asarray(
        [
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
        ]
    )
    model = fit_sparse_boundary_risk(
        locator,
        visual,
        cutoff=2,
        margin_bins=2,
        prior_strength=1.0,
    )

    close = model.predict(2, 0.05)
    far = model.predict(2, 0.95)
    assert close > far
    assert model.predict(2, 0.05, uncertainty_weight=1.0) >= close

    candidates = np.asarray([[10, 11, 12, 13]])
    pairs = build_estimated_boundary_pairs(candidates, locator[:1], model)
    assert [(pair.incumbent, pair.challenger) for pair in pairs] == [
        (11, 12),
        (11, 13),
    ]

