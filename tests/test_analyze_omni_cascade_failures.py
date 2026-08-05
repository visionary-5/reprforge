import pytest

from tools.analyze_omni_cascade_failures import (
    area_under_roc,
    candidate_recall,
    minimum_containment_depth,
)


def test_minimum_containment_depth_finds_first_complete_prefix():
    locator = [str(value) for value in range(100)]
    assert minimum_containment_depth(["0", "19"], locator) == 20
    assert minimum_containment_depth(["0", "49"], locator) == 50
    assert minimum_containment_depth(["0", "99"], locator) == 100
    assert minimum_containment_depth(["0", "missing"], locator) == 1110
    assert (
        minimum_containment_depth(
            ["0", "missing"], locator, full_fallback_rows=2942
        )
        == 2942
    )


def test_candidate_recall_measures_relevant_coverage_and_union():
    relevant = {"a", "b", "c"}
    assert candidate_recall(["a", "x"], relevant) == pytest.approx(1 / 3)
    assert candidate_recall({"a", "x"} | {"b", "y"}, relevant) == pytest.approx(2 / 3)


def test_candidate_recall_rejects_empty_relevance():
    with pytest.raises(ValueError, match="at least one"):
        candidate_recall(["a"], [])


def test_area_under_roc_is_tie_aware():
    assert area_under_roc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert area_under_roc([0.5, 0.5], [True, False]) == 0.5
