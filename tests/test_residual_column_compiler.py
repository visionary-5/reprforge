import numpy as np

from reprforge.residual_column_compiler import (
    fit_residual_column_model,
    fit_low_rank_residual_model,
    low_rank_residual_score_surface,
    pivoted_residual_columns,
    predict_residual_columns,
    residual_column_score_surface,
    two_stage_candidate_surface,
)


def test_pivoted_columns_and_ridge_completion_recover_low_rank_residual():
    query_factors = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    document_factors = np.asarray(
        [[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, -1.0]]
    )
    residual = query_factors @ document_factors
    anchors = pivoted_residual_columns(residual, 2)
    model = fit_residual_column_model(residual, anchors, ridge=1e-8)
    predicted = predict_residual_columns(
        residual[:, anchors], model, clip_to_observed_range=False
    )
    assert np.allclose(predicted, residual, atol=1e-6)


def test_score_surface_keeps_physically_observed_anchor_scores_exact():
    residual = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    anchors = np.asarray([0, 1])
    model = fit_residual_column_model(residual, anchors, ridge=1e-8)
    cheap = np.zeros_like(residual)
    scores = residual_column_score_surface(cheap, residual[:, anchors], model)
    assert np.allclose(scores[:, anchors], residual[:, anchors])
    assert np.all(scores >= residual[:, anchors].min(axis=1, keepdims=True))
    assert np.all(scores <= residual[:, anchors].max(axis=1, keepdims=True))


def test_two_stage_surface_preserves_candidate_set_and_uses_rerank_order():
    cheap = np.asarray([[4.0, 3.0, 2.0, 1.0]])
    rerank = np.asarray([[1.0, 5.0, 9.0, 8.0]])
    surface = two_stage_candidate_surface(cheap, rerank, candidate_k=2)
    order = np.lexsort((np.arange(4), -surface[0]))
    assert order[:2].tolist() == [1, 0]
    assert set(order[:2]) == {0, 1}


def test_low_rank_model_uses_anchor_observations_and_preserves_them():
    rng = np.random.default_rng(7)
    query = rng.normal(size=(12, 2))
    documents = rng.normal(size=(2, 8))
    residual = query @ documents + query[:, :1] * 0.2
    model = fit_low_rank_residual_model(
        residual[:8], rank=2, anchor_count=4, ridge=1e-8
    )
    cheap = np.zeros((4, 8))
    scores = low_rank_residual_score_surface(
        cheap,
        residual[8:, model.anchor_positions],
        model,
        clip_to_observed_range=False,
    )
    assert np.allclose(
        scores[:, model.anchor_positions],
        residual[8:, model.anchor_positions],
    )
    assert np.isfinite(scores).all()
