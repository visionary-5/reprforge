import numpy as np
import pytest

from reprforge import (
    apply_assignment,
    assign_topology_anchored,
    coalesce_hidden_states,
    topology_anchors,
)


def test_topology_anchors_keep_fixed_suffix_positions() -> None:
    np.testing.assert_array_equal(
        topology_anchors((4, 4)),
        np.asarray([1, 5, 3, 7, 9, 13, 11, 15]),
    )


def test_assignment_can_cross_cells_without_moving_suffix_positions() -> None:
    hidden = np.eye(16)
    hidden[0] = hidden[15]

    assignment = assign_topology_anchored(hidden, grid_shape=(4, 4))

    assert assignment.anchor_indices[7] == 15
    assert assignment.slot_for_token[0] == 7
    assert assignment.assignment_cosines[0] == pytest.approx(1.0)


def test_coalescing_halves_workers_and_preserves_auxiliary_states() -> None:
    rng = np.random.default_rng(2)
    visual = rng.normal(size=(16, 8))
    auxiliary = rng.normal(size=(2, 8))
    result = coalesce_hidden_states(
        np.concatenate((visual, auxiliary)),
        grid_shape=(4, 4),
        auxiliary_tokens=2,
    )

    assert result.hidden_states.shape == (10, 8)
    assert result.assignment.cluster_sizes.sum() == 16
    np.testing.assert_allclose(result.hidden_states[-2:], auxiliary)
    np.testing.assert_array_equal(
        result.assignment.compact_positions(auxiliary_tokens=2),
        np.asarray([1, 5, 3, 7, 9, 13, 11, 15, 16, 17]),
    )


def test_page_assignment_is_reusable_for_teacher_pooling() -> None:
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(16, 6))
    assignment = assign_topology_anchored(hidden, grid_shape=(4, 4))

    student = apply_assignment(hidden, assignment)
    teacher = apply_assignment(hidden + 0.1, assignment)

    assert student.shape == teacher.shape == (8, 6)
