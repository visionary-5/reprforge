"""Topology-anchored nonlocal coalescing for visual hidden states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Matrix = NDArray[np.float64]
IndexVector = NDArray[np.int64]
FloatVector = NDArray[np.float64]


def _matrix(value: ArrayLike) -> Matrix:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("expected a non-empty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("hidden states must be finite")
    return matrix


def _normalize_rows(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def topology_anchor_indices(grid_shape: tuple[int, int]) -> IndexVector:
    """Return the fixed right-column pair from every spatial 2x2 cell."""

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
class TopologyAnchoredPlan:
    """A query-free assignment from full visual tokens to fixed suffix slots."""

    grid_shape: tuple[int, int]
    anchor_indices: IndexVector
    assignments: IndexVector
    cluster_sizes: IndexVector
    assignment_cosines: FloatVector

    @property
    def full_visual_tokens(self) -> int:
        return int(len(self.assignments))

    @property
    def compact_visual_tokens(self) -> int:
        return int(len(self.anchor_indices))

    @property
    def persistent_fraction(self) -> float:
        return self.compact_visual_tokens / self.full_visual_tokens

    def compact_positions(self, *, auxiliary_tokens: int = 0) -> IndexVector:
        """Positions retained by a model hook for masks and position encodings."""

        if auxiliary_tokens < 0:
            raise ValueError("auxiliary token count cannot be negative")
        auxiliary = np.arange(
            self.full_visual_tokens,
            self.full_visual_tokens + auxiliary_tokens,
            dtype=np.int64,
        )
        return np.concatenate((self.anchor_indices, auxiliary))


@dataclass(frozen=True)
class CoalescedState:
    """Compact hidden states plus the plan needed to continue the suffix."""

    hidden_states: Matrix
    plan: TopologyAnchoredPlan


def topology_anchored_plan(
    hidden_states: ArrayLike,
    *,
    grid_shape: tuple[int, int],
) -> TopologyAnchoredPlan:
    """Assign every visual token to its most similar fixed topology anchor.

    Similarity is measured at the selected hidden-state boundary. The anchors
    define stable output positions; cluster membership is global and semantic.
    """

    visual = _matrix(hidden_states)
    rows, columns = grid_shape
    if len(visual) != rows * columns:
        raise ValueError("grid shape does not match the visual hidden states")
    anchors = topology_anchor_indices(grid_shape)
    similarities = _normalize_rows(visual) @ _normalize_rows(visual[anchors]).T
    assignments = np.argmax(similarities, axis=1).astype(np.int64)
    sizes = np.bincount(assignments, minlength=len(anchors)).astype(np.int64)
    chosen = similarities[np.arange(len(visual)), assignments]
    return TopologyAnchoredPlan(
        grid_shape=grid_shape,
        anchor_indices=anchors,
        assignments=assignments,
        cluster_sizes=sizes,
        assignment_cosines=chosen,
    )


def apply_coalescing_plan(
    hidden_states: ArrayLike,
    plan: TopologyAnchoredPlan,
    *,
    auxiliary_tokens: int = 0,
) -> Matrix:
    """Pool raw hidden states by a frozen plan and append auxiliary states."""

    matrix = _matrix(hidden_states)
    if auxiliary_tokens < 0:
        raise ValueError("auxiliary token count cannot be negative")
    expected = plan.full_visual_tokens + auxiliary_tokens
    if len(matrix) != expected:
        raise ValueError("plan and auxiliary token counts do not match hidden states")
    visual = matrix[: plan.full_visual_tokens]
    if visual.shape[1] == 0:
        raise ValueError("hidden-state dimension cannot be empty")
    sums = np.zeros(
        (plan.compact_visual_tokens, visual.shape[1]), dtype=np.float64
    )
    np.add.at(sums, plan.assignments, visual)
    pooled = sums / np.maximum(plan.cluster_sizes[:, None], 1)
    empty = plan.cluster_sizes == 0
    pooled[empty] = visual[plan.anchor_indices[empty]]
    if not auxiliary_tokens:
        return pooled
    return np.concatenate((pooled, matrix[plan.full_visual_tokens :]), axis=0)


def topology_anchored_coalesce(
    hidden_states: ArrayLike,
    *,
    grid_shape: tuple[int, int],
    auxiliary_tokens: int = 0,
) -> CoalescedState:
    """Build and apply the ReprForge plan at one hidden-state boundary."""

    matrix = _matrix(hidden_states)
    visual_tokens = grid_shape[0] * grid_shape[1]
    if visual_tokens + auxiliary_tokens != len(matrix):
        raise ValueError("grid and auxiliary token counts do not match hidden states")
    plan = topology_anchored_plan(
        matrix[:visual_tokens],
        grid_shape=grid_shape,
    )
    compact = apply_coalescing_plan(
        matrix,
        plan,
        auxiliary_tokens=auxiliary_tokens,
    )
    return CoalescedState(hidden_states=compact, plan=plan)
