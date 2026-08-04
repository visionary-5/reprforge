import numpy as np
import pytest

from reprforge.workload_probe_compiler import fit_spherical_probes


def test_spherical_probes_are_deterministic_unit_directions():
    samples = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
        dtype=np.float32,
    )
    first, first_errors = fit_spherical_probes(samples, count=2, seed=7)
    second, second_errors = fit_spherical_probes(samples, count=2, seed=7)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first_errors, second_errors)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert float(first_errors.max()) < 0.01


def test_spherical_probes_reject_invalid_counts():
    samples = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError):
        fit_spherical_probes(samples, count=0, seed=1)
    with pytest.raises(ValueError):
        fit_spherical_probes(samples, count=3, seed=1)
