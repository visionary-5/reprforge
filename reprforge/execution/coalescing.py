"""Apply a page assignment to raw hidden states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .assignment import PageAssignment, assign_topology_anchored

FloatMatrix = NDArray[np.float64]


def _hidden_matrix(value: ArrayLike) -> FloatMatrix:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("expected a non-empty rank-2 hidden-state matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("hidden states must be finite")
    return matrix


@dataclass(frozen=True)
class CoalescedState:
    """Compact worker states and the assignment that produced them."""

    hidden_states: FloatMatrix
    assignment: PageAssignment


def apply_assignment(
    hidden_states: ArrayLike,
    assignment: PageAssignment,
    *,
    auxiliary_tokens: int = 0,
) -> FloatMatrix:
    """Pool raw visual states and append non-visual states unchanged."""

    matrix = _hidden_matrix(hidden_states)
    if auxiliary_tokens < 0:
        raise ValueError("auxiliary token count cannot be negative")
    expected = assignment.full_visual_tokens + auxiliary_tokens
    if len(matrix) != expected:
        raise ValueError("assignment and auxiliary counts do not match hidden states")
    visual = matrix[: assignment.full_visual_tokens]
    sums = np.zeros(
        (assignment.compact_visual_tokens, visual.shape[1]), dtype=np.float64
    )
    np.add.at(sums, assignment.slot_for_token, visual)
    pooled = sums / np.maximum(assignment.cluster_sizes[:, None], 1)
    empty = assignment.cluster_sizes == 0
    pooled[empty] = visual[assignment.anchor_indices[empty]]
    if not auxiliary_tokens:
        return pooled
    return np.concatenate((pooled, matrix[assignment.full_visual_tokens :]), axis=0)


def coalesce_hidden_states(
    hidden_states: ArrayLike,
    *,
    grid_shape: tuple[int, int],
    auxiliary_tokens: int = 0,
) -> CoalescedState:
    """Construct page evidence ownership and produce compact worker states."""

    matrix = _hidden_matrix(hidden_states)
    visual_tokens = grid_shape[0] * grid_shape[1]
    if visual_tokens + auxiliary_tokens != len(matrix):
        raise ValueError("grid and auxiliary counts do not match hidden states")
    assignment = assign_topology_anchored(
        matrix[:visual_tokens], grid_shape=grid_shape
    )
    compact = apply_assignment(
        matrix, assignment, auxiliary_tokens=auxiliary_tokens
    )
    return CoalescedState(hidden_states=compact, assignment=assignment)
