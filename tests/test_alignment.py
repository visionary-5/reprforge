import numpy as np
import pytest

from reprforge import fit_trajectory_alignment, normalize_rows


def test_query_free_alignment_improves_paired_endpoint_cosine() -> None:
    rng = np.random.default_rng(7)
    students = [normalize_rows(rng.normal(size=(12, 16))) for _ in range(6)]
    rotation = np.eye(16) + 0.15 * rng.normal(size=(16, 3)) @ rng.normal(
        size=(3, 16)
    )
    teachers = [normalize_rows(page @ rotation) for page in students]

    fit = fit_trajectory_alignment(
        students,
        teachers,
        rank=3,
        steps=80,
        learning_rate=0.02,
        seed=4,
    )
    before = np.mean(
        [
            np.mean(np.sum(a * b, axis=1))
            for a, b in zip(students, teachers, strict=True)
        ]
    )
    after = np.mean(
        [
            np.mean(np.sum(fit.transform(a) * b, axis=1))
            for a, b in zip(students, teachers, strict=True)
        ]
    )

    assert after > before
    assert fit.losses[-1] < fit.losses[0]
    assert fit.rank == 3


def test_alignment_rejects_unaligned_slots() -> None:
    with pytest.raises(ValueError, match="position-aligned"):
        fit_trajectory_alignment(
            [np.ones((2, 4)), np.ones((2, 4))],
            [np.ones((3, 4)), np.ones((2, 4))],
        )
