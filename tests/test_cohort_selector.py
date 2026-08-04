import numpy as np

from reprforge.cohort_selector import (
    RandomFeatureRankSelector,
    RidgeRankSelector,
    RidgeRegressor,
    candidate_features,
)


def test_candidate_features_are_finite_and_candidate_aligned():
    base = np.asarray([9, 7, 5, 4, 1], dtype=float)
    candidates = np.asarray([0, 1, 2, 3, 4])
    features = candidate_features(base, candidates, target_k=3)
    assert features.shape[0] == len(candidates)
    assert np.isfinite(features).all()


def test_ridge_rank_selector_learns_a_separable_ordering():
    features = np.asarray([[0.0], [0.2], [0.8], [1.0]] * 10)
    labels = np.asarray([0.0, 0.0, 1.0, 1.0] * 10)
    model = RidgeRankSelector.fit(features, labels, regularization=0.01)
    predictions = model.predict(np.asarray([[0.1], [0.9]]))
    assert predictions[1] > predictions[0]


def test_ridge_regressor_learns_a_linear_target():
    features = np.arange(20, dtype=float)[:, None]
    model = RidgeRegressor.fit(features, 2.0 * features[:, 0] + 3.0, regularization=0.0)
    predictions = model.predict(np.asarray([[2.5], [7.5]]))
    assert np.allclose(predictions, [8.0, 18.0])


def test_random_feature_selector_learns_nonlinear_xor():
    features = np.tile(
        np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        (20, 1),
    )
    labels = np.tile(np.asarray([0.0, 1.0, 1.0, 0.0]), 20)
    model = RandomFeatureRankSelector.fit(
        features, labels, random_features=128, regularization=0.01, seed=7
    )
    predictions = model.predict(features[:4])
    assert predictions[[1, 2]].min() > predictions[[0, 3]].max()
