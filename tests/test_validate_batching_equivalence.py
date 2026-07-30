import numpy as np

from reprforge.validate_batching_equivalence import compare_scores


def test_score_comparison_allows_roundoff_but_requires_top_k() -> None:
    result = compare_scores(
        np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
        np.asarray([3.0 + 1e-6, 2.0, 1.0], dtype=np.float32),
        item_ids=("a", "b", "c"),
        top_k=2,
        absolute_tolerance=1e-4,
        relative_tolerance=1e-4,
    )
    assert result["all_close"]
    assert result["top_k_equal"]


def test_score_comparison_detects_ranking_change() -> None:
    result = compare_scores(
        np.asarray([2.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 2.0], dtype=np.float32),
        item_ids=("a", "b"),
        top_k=1,
        absolute_tolerance=1e-4,
        relative_tolerance=1e-4,
    )
    assert not result["all_close"]
    assert not result["top_k_equal"]
