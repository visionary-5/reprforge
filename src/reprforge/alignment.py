"""Query-free alignment for compact visual retrieval slots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


Matrix = NDArray[np.float64]


def normalize_rows(value: ArrayLike) -> Matrix:
    """Return a finite rank-2 matrix with unit-length rows."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("expected a non-empty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


@dataclass(frozen=True)
class TrajectoryAlignment:
    """A shared near-identity low-rank residual.

    The map is applied to every compact endpoint slot as
    ``normalize(x + x @ down @ up)``.
    """

    down: Matrix
    up: Matrix
    losses: tuple[float, ...]

    @property
    def rank(self) -> int:
        return int(self.down.shape[1])

    @property
    def parameters(self) -> int:
        return int(self.down.size + self.up.size)

    def transform(self, vectors: ArrayLike) -> Matrix:
        source = normalize_rows(vectors)
        if source.shape[1] != self.down.shape[0]:
            raise ValueError("vector dimension does not match the fitted alignment")
        return normalize_rows(source + (source @ self.down) @ self.up)


def _aligned_pages(
    students: Iterable[ArrayLike], teachers: Iterable[ArrayLike]
) -> tuple[list[Matrix], list[Matrix]]:
    student_pages = [normalize_rows(page) for page in students]
    teacher_pages = [normalize_rows(page) for page in teachers]
    if len(student_pages) != len(teacher_pages) or len(student_pages) < 2:
        raise ValueError("at least two aligned student/teacher pages are required")
    if any(
        student.shape != teacher.shape
        for student, teacher in zip(student_pages, teacher_pages, strict=True)
    ):
        raise ValueError("student and teacher slots must be position-aligned")
    dimension = student_pages[0].shape[1]
    if any(page.shape[1] != dimension for page in student_pages + teacher_pages):
        raise ValueError("all pages must share one embedding dimension")
    return student_pages, teacher_pages


def fit_trajectory_alignment(
    students: Iterable[ArrayLike],
    teachers: Iterable[ArrayLike],
    *,
    rank: int = 8,
    steps: int = 25,
    learning_rate: float = 1e-2,
    identity_weight: float = 1e-3,
    gradient_clip_norm: float = 1.0,
    seed: int = 0,
) -> TrajectoryAlignment:
    """Fit the query-free endpoint correction used by ReprForge.

    ``students`` are compact suffix endpoints from a small calibration canary.
    ``teachers`` are Full endpoints pooled into the same spatial slots. No
    retrieval query, qrel, or answer is consumed.
    """

    if rank <= 0 or steps <= 0 or learning_rate <= 0:
        raise ValueError("rank, steps, and learning_rate must be positive")
    if identity_weight < 0 or gradient_clip_norm <= 0:
        raise ValueError("invalid regularization or gradient clipping")

    student_pages, teacher_pages = _aligned_pages(students, teachers)
    dimension = student_pages[0].shape[1]
    if rank > dimension:
        raise ValueError("rank cannot exceed the embedding dimension")

    generator = np.random.default_rng(seed)
    down = generator.normal(0.0, 1.0 / np.sqrt(dimension), (dimension, rank))
    up = np.zeros((rank, dimension), dtype=np.float64)
    first_moments = [np.zeros_like(down), np.zeros_like(up)]
    second_moments = [np.zeros_like(down), np.zeros_like(up)]
    token_count = sum(len(page) for page in student_pages)
    losses: list[float] = []

    for step in range(1, steps + 1):
        grad_down = np.zeros_like(down)
        grad_up = np.zeros_like(up)
        cosine_sum = 0.0

        for source, teacher in zip(student_pages, teacher_pages, strict=True):
            hidden = source @ down
            pre_normalized = source + hidden @ up
            norms = np.maximum(
                np.linalg.norm(pre_normalized, axis=1, keepdims=True), 1e-12
            )
            output = pre_normalized / norms
            cosine_sum += float(np.sum(output * teacher))

            output_gradient = -teacher / token_count
            pre_gradient = (
                output_gradient
                - output
                * np.sum(output_gradient * output, axis=1, keepdims=True)
            ) / norms
            grad_up += hidden.T @ pre_gradient
            grad_down += source.T @ (pre_gradient @ up.T)

        product = down @ up
        loss = 1.0 - cosine_sum / token_count
        loss += identity_weight * float(np.mean(product**2))
        losses.append(loss)

        scale = 2.0 * identity_weight / product.size
        grad_down += scale * (product @ up.T)
        grad_up += scale * (down.T @ product)
        gradient_norm = float(
            np.sqrt(np.sum(grad_down**2) + np.sum(grad_up**2))
        )
        if gradient_norm > gradient_clip_norm:
            multiplier = gradient_clip_norm / gradient_norm
            grad_down *= multiplier
            grad_up *= multiplier

        for index, (parameter, gradient) in enumerate(
            ((down, grad_down), (up, grad_up))
        ):
            first_moments[index] = 0.9 * first_moments[index] + 0.1 * gradient
            second_moments[index] = (
                0.999 * second_moments[index] + 0.001 * gradient**2
            )
            corrected_first = first_moments[index] / (1.0 - 0.9**step)
            corrected_second = second_moments[index] / (1.0 - 0.999**step)
            parameter -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )

    return TrajectoryAlignment(down=down, up=up, losses=tuple(losses))
