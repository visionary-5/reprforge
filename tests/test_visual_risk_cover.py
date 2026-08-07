import numpy as np

from reprforge.visual_risk_cover import fit_ridge_risk


def test_ridge_risk_learns_a_cheap_feature_boundary():
    train_x = np.asarray([[0.0], [0.1], [0.9], [1.0]])
    train_y = np.asarray([0.0, 0.0, 1.0, 1.0])
    scores = fit_ridge_risk(train_x, train_y, np.asarray([[0.05], [0.95]]), ridge_lambda=0.1)
    assert scores[1] > scores[0]
