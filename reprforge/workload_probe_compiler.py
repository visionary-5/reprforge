"""Query-distribution probes for workload-conditioned index compilation."""

from __future__ import annotations

import numpy as np


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("probe samples must be a non-empty matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("probe samples must have non-zero norm")
    return array / norms


def fit_spherical_probes(
    samples: np.ndarray,
    *,
    count: int,
    seed: int,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize query-token directions with deterministic spherical k-means.

    Returns unit-normalized probe directions and one-minus-cosine assignment
    errors for every input sample.  The compiler only consumes query
    embeddings; relevance judgments are deliberately absent from the API.
    """

    values = _normalized_rows(samples)
    if count <= 0 or count > len(values):
        raise ValueError("probe count must be between one and the sample count")
    if iterations <= 0:
        raise ValueError("spherical k-means iterations must be positive")

    generator = np.random.default_rng(seed)
    selected = [int(generator.integers(len(values)))]
    closest_distance = np.maximum(0.0, 1.0 - values @ values[selected[0]])
    for _ in range(1, count):
        weights = closest_distance.astype(np.float64) ** 2
        weights[np.asarray(selected, dtype=np.int64)] = 0.0
        total = float(weights.sum())
        if total <= 0:
            remaining = np.setdiff1d(
                np.arange(len(values)), np.asarray(selected), assume_unique=False
            )
            choice = int(remaining[0])
        else:
            choice = int(generator.choice(len(values), p=weights / total))
        selected.append(choice)
        closest_distance = np.minimum(
            closest_distance,
            np.maximum(0.0, 1.0 - values @ values[choice]),
        )

    centers = values[np.asarray(selected)].copy()
    assignments = np.zeros(len(values), dtype=np.int32)
    for _ in range(iterations):
        similarities = values @ centers.T
        new_assignments = similarities.argmax(axis=1).astype(np.int32)
        new_centers = np.empty_like(centers)
        minimum_distance = 1.0 - similarities.max(axis=1)
        for cluster in range(count):
            members = values[new_assignments == cluster]
            if len(members) == 0:
                replacement = int(np.argmax(minimum_distance))
                new_centers[cluster] = values[replacement]
                new_assignments[replacement] = cluster
                minimum_distance[replacement] = -np.inf
                continue
            center = members.mean(axis=0)
            norm = float(np.linalg.norm(center))
            new_centers[cluster] = center / norm if norm > 0 else members[0]
        centers = new_centers
        if np.array_equal(assignments, new_assignments):
            assignments = new_assignments
            break
        assignments = new_assignments

    errors = np.maximum(0.0, 1.0 - np.max(values @ centers.T, axis=1))
    return centers.astype(np.float32), errors.astype(np.float32)
