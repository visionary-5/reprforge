import pytest

from reprforge.physical_partial_evaluation import (
    evaluate_rankings,
    gain_recovery,
    reciprocal_rank_fusion,
)


def test_rrf_combines_unique_pages_with_stable_ties():
    assert reciprocal_rank_fusion(["a", "b"], ["b", "c"], constant=60) == [
        "b",
        "a",
        "c",
    ]


def test_physical_metrics_and_gain_recovery():
    qrels = {"q": {"a": 1.0}}
    result = evaluate_rankings({"q": ["a", "b"]}, qrels)
    assert result["mean"]["ndcg_at_10"] == 1.0
    assert gain_recovery(0.7, 0.5, 0.75) == pytest.approx(0.8)
    assert gain_recovery(0.5, 0.5, 0.501) is None
