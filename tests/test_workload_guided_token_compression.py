import numpy as np
import pytest

from reprforge.workload_guided_token_compression import (
    merge_tokens_to_workload_seeds,
    workload_guided_token_selection,
)


def test_selection_covers_distinct_probe_directions_and_is_deterministic():
    documents = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    probes = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    first = workload_guided_token_selection(documents, probes, budget=2)
    second = workload_guided_token_selection(documents, probes, budget=2)
    assert first.tolist() == [0, 2]
    assert first.tolist() == second.tolist()


def test_selection_is_a_subset_so_maxsim_cannot_exceed_full():
    rng = np.random.default_rng(3)
    documents = rng.normal(size=(12, 4))
    probes = rng.normal(size=(3, 4))
    queries = rng.normal(size=(7, 4))
    selected = workload_guided_token_selection(documents, probes, budget=5)
    full = queries @ documents.T
    compressed = queries @ documents[selected].T
    assert np.all(compressed.max(axis=1) <= full.max(axis=1) + 1e-12)


def test_probe_weights_must_be_valid():
    with pytest.raises(ValueError):
        workload_guided_token_selection(
            np.eye(2), np.eye(2), budget=1, probe_weights=[0.0, 0.0]
        )


def test_merging_produces_one_normalized_vector_per_seed():
    documents = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    merged = merge_tokens_to_workload_seeds(documents, [0, 2])
    assert merged.shape == (2, 2)
    assert np.linalg.norm(merged, axis=1).tolist() == pytest.approx([1.0, 1.0])
