"""Semantic evidence assignment under a fixed suffix topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]
IndexVector = NDArray[np.int64]


def _visual_matrix(value: ArrayLike) -> FloatMatrix:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("expected a non-empty rank-2 visual-state matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("visual hidden states must be finite")
    return matrix


def _normalize_rows(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def topology_anchors(grid_shape: tuple[int, int]) -> IndexVector:
    """Reserve the right-column pair from every spatial 2x2 cell."""

    rows, columns = grid_shape
    if rows <= 0 or columns <= 0 or rows % 2 or columns % 2:
        raise ValueError("grid dimensions must be positive and even")
    grid = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)
    return (
        grid.reshape(rows // 2, 2, columns // 2, 2)
        .transpose(0, 2, 1, 3)[:, :, :, 1]
        .reshape(-1)
    )


@dataclass(frozen=True)
class PageAssignment:
    """Per-page evidence ownership with stable compact suffix positions."""

    grid_shape: tuple[int, int]
    anchor_indices: IndexVector
    slot_for_token: IndexVector
    cluster_sizes: IndexVector
    assignment_cosines: FloatVector

    @property
    def full_visual_tokens(self) -> int:
        return int(len(self.slot_for_token))

    @property
    def compact_visual_tokens(self) -> int:
        return int(len(self.anchor_indices))

    @property
    def persistent_fraction(self) -> float:
        return self.compact_visual_tokens / self.full_visual_tokens

    def compact_positions(self, *, auxiliary_tokens: int = 0) -> IndexVector:
        """Return positions retained in masks and positional encodings."""

        if auxiliary_tokens < 0:
            raise ValueError("auxiliary token count cannot be negative")
        auxiliary = np.arange(
            self.full_visual_tokens,
            self.full_visual_tokens + auxiliary_tokens,
            dtype=np.int64,
        )
        return np.concatenate((self.anchor_indices, auxiliary))


def assign_topology_anchored(
    visual_hidden_states: ArrayLike,
    *,
    grid_shape: tuple[int, int],
) -> PageAssignment:
    """Assign every visual state to its closest fixed anchor by cosine."""

    visual = _visual_matrix(visual_hidden_states)
    rows, columns = grid_shape
    if len(visual) != rows * columns:
        raise ValueError("grid shape does not match the visual hidden states")
    anchors = topology_anchors(grid_shape)
    similarities = _normalize_rows(visual) @ _normalize_rows(visual[anchors]).T
    assignments = np.argmax(similarities, axis=1).astype(np.int64)
    sizes = np.bincount(assignments, minlength=len(anchors)).astype(np.int64)
    chosen = similarities[np.arange(len(visual)), assignments]
    return PageAssignment(
        grid_shape=grid_shape,
        anchor_indices=anchors,
        slot_for_token=assignments,
        cluster_sizes=sizes,
        assignment_cosines=chosen,
    )
