import numpy as np
import pytest

from reprforge import (
    CompactIndex,
    apply_coalescing_plan,
    maxsim_score,
    topology_anchor_indices,
    topology_anchored_coalesce,
    topology_anchored_plan,
)


def test_topology_anchors_keep_fixed_suffix_positions() -> None:
    np.testing.assert_array_equal(
        topology_anchor_indices((4, 4)),
        np.asarray([1, 5, 3, 7, 9, 13, 11, 15]),
    )


def test_topology_anchored_coalescing_halves_visual_capacity() -> None:
    rng = np.random.default_rng(2)
    visual = rng.normal(size=(16, 8))
    auxiliary = rng.normal(size=(2, 8))

    result = topology_anchored_coalesce(
        np.concatenate((visual, auxiliary)),
        grid_shape=(4, 4),
        auxiliary_tokens=2,
    )

    assert result.hidden_states.shape == (10, 8)
    assert result.plan.compact_visual_tokens == 8
    assert result.plan.cluster_sizes.sum() == 16
    np.testing.assert_array_equal(
        result.plan.compact_positions(auxiliary_tokens=2),
        np.asarray([1, 5, 3, 7, 9, 13, 11, 15, 16, 17]),
    )
    np.testing.assert_allclose(result.hidden_states[-2:], auxiliary)


def test_assignment_can_cross_local_cells_without_moving_suffix_slots() -> None:
    hidden = np.eye(16)
    hidden[0] = hidden[15]

    plan = topology_anchored_plan(hidden, grid_shape=(4, 4))

    assert plan.anchor_indices[7] == 15
    assert plan.assignments[0] == 7
    assert plan.assignment_cosines[0] == pytest.approx(1.0)


def test_plan_is_reusable_for_assignment_aligned_teacher_pooling() -> None:
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(16, 6))
    plan = topology_anchored_plan(hidden, grid_shape=(4, 4))
    student = apply_coalescing_plan(hidden, plan)
    teacher = apply_coalescing_plan(hidden + 0.1, plan)

    assert student.shape == teacher.shape == (8, 6)
    assert np.isfinite(plan.assignment_cosines).all()


def test_compact_search_then_full_refinement() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    compact = CompactIndex(
        [
            ("a", [[1.0, 0.0], [0.2, 0.8]]),
            ("b", [[0.9, 0.1], [0.0, 1.0]]),
            ("c", [[-1.0, 0.0], [0.0, -1.0]]),
        ]
    )
    candidates = compact.search(query, top_k=2)
    full = {
        "a": np.asarray([[1.0, 0.0], [0.1, 0.2]]),
        "b": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    }

    refined = compact.refine(query, candidates, full.__getitem__)

    assert [row.item_id for row in candidates] == ["b", "a"]
    assert [row.item_id for row in refined] == ["b", "a"]
    assert maxsim_score(query, full["b"]) == pytest.approx(2.0)


def test_refinement_rejects_items_outside_locator() -> None:
    index = CompactIndex([("a", [[1.0, 0.0]])])
    with pytest.raises(KeyError, match="outside the compact index"):
        index.refine([[1.0, 0.0]], ["missing"], lambda _: [[1.0, 0.0]])
