import numpy as np

from reprforge.candidate_fusion import _zscore


def test_candidate_relative_zscore_is_invariant_to_affine_scale():
    values = np.asarray([1.0, 2.0, 4.0])
    np.testing.assert_allclose(_zscore(values), _zscore(10.0 * values + 7.0))
