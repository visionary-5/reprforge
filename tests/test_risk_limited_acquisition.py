import numpy as np
import pytest

from reprforge.risk_limited_acquisition import (
    ConformalEnvelopeModel,
    ScoreIntervals,
    acquire_until_certified,
    balanced_group_folds,
    build_candidate_surface,
    conformal_quantile,
    crossfit_boundary_acquisition,
    simultaneous_coverage,
)


def test_candidate_surface_is_deterministic_and_uses_prebuild_features() -> None:
    corpus_ids = ["b", "a", "c", "d"]
    locator = np.asarray([[4.0, 4.0, 2.0, 1.0], [1.0, 3.0, 2.0, 4.0]])
    visual = np.asarray([[8.0, 6.0, 4.0, 2.0], [2.0, 6.0, 4.0, 8.0]])
    surface = build_candidate_surface(
        corpus_ids,
        locator,
        visual,
        query_token_counts=[2, 2],
        page_text_token_counts=[10, 20, 30, 40],
        candidate_pool=3,
    )
    assert surface.candidate_indices.tolist() == [[1, 0, 2], [3, 1, 2]]
    assert surface.visual_scores[0].tolist() == [3.0, 4.0, 2.0]
    assert surface.features.shape == (2, 3, 6)
    assert np.allclose(surface.base_scores.mean(axis=1), 0.0)


def test_conformal_quantile_uses_finite_sample_ceiling() -> None:
    assert conformal_quantile([1.0, 2.0, 3.0, 4.0], alpha=0.25) == 4.0
    assert conformal_quantile([1.0, 2.0, 3.0, 4.0], alpha=0.5) == 3.0
    with pytest.raises(ValueError):
        conformal_quantile([], alpha=0.05)


def test_envelope_model_calibrates_query_level_maximum() -> None:
    rng = np.random.default_rng(7)
    fit_x = rng.normal(size=(12, 5, 2))
    fit_y = 1.5 * fit_x[:, :, 0] - 0.5 * fit_x[:, :, 1]
    cal_x = rng.normal(size=(6, 5, 2))
    cal_y = 1.5 * cal_x[:, :, 0] - 0.5 * cal_x[:, :, 1]
    model = ConformalEnvelopeModel.fit(
        fit_x,
        fit_y,
        cal_x,
        cal_y,
        alpha=0.2,
    )
    intervals = model.predict_intervals(cal_x)
    normalized = model.normalize_targets(cal_y)
    assert simultaneous_coverage(normalized, intervals).all()
    assert model.conformal_multiplier >= 0.0
    assert np.all(intervals.lower <= intervals.mean)
    assert np.all(intervals.mean <= intervals.upper)


def test_acquisition_stops_after_clear_boundary() -> None:
    base = np.zeros(6)
    exact = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    intervals = ScoreIntervals(
        mean=exact.copy(),
        lower=exact - 0.1,
        upper=exact + 0.1,
        scale=np.full(6, 0.1),
    )
    result = acquire_until_certified(
        base,
        exact,
        intervals,
        [f"d{i}" for i in range(6)],
        cutoff=2,
        batch_size=2,
    )
    assert result.ranking.tolist() == [0, 1]
    assert result.acquired_count == 2
    assert result.certified
    assert not result.exhausted_pool


def test_acquisition_finds_hidden_challenger_and_can_exhaust_pool() -> None:
    base = np.zeros(5)
    exact = np.asarray([5.0, 4.0, 3.0, 8.0, 1.0])
    intervals = ScoreIntervals(
        mean=np.asarray([5.0, 4.0, 3.0, 2.0, 1.0]),
        lower=np.full(5, -10.0),
        upper=np.full(5, 10.0),
        scale=np.ones(5),
    )
    result = acquire_until_certified(
        base,
        exact,
        intervals,
        [f"d{i}" for i in range(5)],
        cutoff=2,
        batch_size=2,
    )
    assert result.ranking.tolist() == [3, 0]
    assert result.exhausted_pool
    assert set(result.acquired_indices) == set(range(5))


def test_cost_aware_tie_break_prefers_cheaper_ambiguous_candidate() -> None:
    base = np.zeros(4)
    exact = np.asarray([4.0, 3.0, 2.0, 1.0])
    intervals = ScoreIntervals(
        mean=np.asarray([4.0, 3.0, 2.0, 1.0]),
        lower=np.asarray([3.9, 2.9, -1.0, -1.0]),
        upper=np.asarray([4.1, 3.1, 4.0, 4.0]),
        scale=np.ones(4),
    )
    result = acquire_until_certified(
        base,
        exact,
        intervals,
        ["a", "b", "c", "d"],
        cutoff=2,
        batch_size=2,
        build_costs=[1.0, 1.0, 10.0, 1.0],
    )
    assert result.acquisition_batches[1][0] == 3


def test_balanced_group_folds_are_complete_and_deterministic() -> None:
    groups = ["a"] * 4 + ["b"] * 3 + ["c"] * 2 + ["d"] * 2 + ["e"]
    folds, assignment = balanced_group_folds(groups, fold_count=5)
    assert set(folds.tolist()) == set(range(5))
    assert len(assignment) == 5
    assert np.array_equal(folds, balanced_group_folds(groups, fold_count=5)[0])


def test_crossfit_acquisition_keeps_test_groups_out_of_fit_and_calibration() -> None:
    rng = np.random.default_rng(4)
    queries = 15
    pages = 12
    locator = rng.normal(size=(queries, pages))
    visual = 2.0 * locator + rng.normal(scale=0.01, size=(queries, pages))
    corpus_ids = [f"d{i}" for i in range(pages)]
    surface = build_candidate_surface(
        corpus_ids,
        locator,
        visual,
        query_token_counts=np.full(queries, 4),
        page_text_token_counts=np.arange(1, pages + 1),
        candidate_pool=8,
    )
    groups = [f"g{i // 3}" for i in range(queries)]
    result = crossfit_boundary_acquisition(
        surface,
        corpus_ids,
        groups,
        cutoff=2,
        alpha=0.2,
        batch_size=2,
    )
    assert result.rankings.shape == (queries, 2)
    assert result.teacher_rankings.shape == (queries, 2)
    assert result.coverage.shape == (queries,)
    assert np.all(result.acquired_counts >= 2)
    assert len(result.folds) == 5
    assert all(record["fit_queries"] > 0 for record in result.folds)
