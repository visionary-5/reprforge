import pytest

from tools.rerank_omni_candidates import rerank_candidate_prefix


def test_rerank_candidate_prefix_uses_full_scores_and_stable_ties():
    result = rerank_candidate_prefix(
        ["a", "b", "c"],
        {"a": 1.0, "b": 3.0, "c": 3.0},
        depth=3,
    )
    assert result == [("b", 3.0), ("c", 3.0), ("a", 1.0)]


def test_rerank_candidate_prefix_rejects_missing_scores():
    with pytest.raises(ValueError, match="missing"):
        rerank_candidate_prefix(["a", "b"], {"a": 1.0}, depth=2)
