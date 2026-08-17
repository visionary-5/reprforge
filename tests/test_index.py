import numpy as np
import pytest

from reprforge import CompactIndex, coalesce_visual_tokens, maxsim_score


def test_coalesce_visual_tokens_halves_visual_capacity() -> None:
    visual = np.arange(4 * 4 * 3, dtype=float).reshape(16, 3) + 1
    auxiliary = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    compact = coalesce_visual_tokens(
        np.concatenate((visual, auxiliary)),
        grid_shape=(4, 4),
        auxiliary_tokens=2,
    )

    assert compact.shape == (10, 3)
    np.testing.assert_allclose(np.linalg.norm(compact, axis=1), 1.0)


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
