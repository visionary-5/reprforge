import numpy as np

from tools.analyze_omni_adaptive_cascade import (
    admission_threshold,
    fit_logistic,
    predict_logistic,
)


def test_admission_threshold_respects_empirical_risk():
    scores = np.asarray([0.1, 0.2, 0.3, 0.4])
    unsafe = np.asarray([0.0, 0.0, 1.0, 1.0])
    threshold = admission_threshold(
        scores,
        unsafe,
        maximum_empirical_risk=0.0,
        minimum_admissions=2,
    )
    assert 0.2 < threshold < 0.3


def test_logistic_predicts_separable_training_examples():
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])
    predictions = predict_logistic(
        fit_logistic(features, targets, steps=1000), features
    )
    assert predictions[0] < predictions[-1]
    assert predictions[1] < 0.5 < predictions[2]
