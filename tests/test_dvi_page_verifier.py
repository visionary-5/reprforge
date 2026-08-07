import numpy as np

from reprforge.dvi_page_verifier import (
    deterministic_query_sample,
    ranking_metrics,
    rerank_with_scores,
    roc_auc,
    union_preserving_order,
)


def test_candidate_union_and_reranking_are_deterministic():
    assert union_preserving_order(["a", "b"], ["b", "c"]) == ["a", "b", "c"]
    assert rerank_with_scores(["a", "b", "c"], {"a": 0, "b": 2, "c": 1}) == [
        "b",
        "c",
        "a",
    ]
    assert deterministic_query_sample(["q0", "q1", "q2"], limit=2, seed=7) == deterministic_query_sample(
        ["q2", "q1", "q0"], limit=2, seed=7
    )


def test_metrics_and_auc_reward_relevant_pages():
    good = ranking_metrics(["gold", "other"], {"gold": 2.0})
    bad = ranking_metrics(["other", "gold"], {"gold": 2.0})
    assert good["ndcg_at_10"] > bad["ndcg_at_10"]
    assert np.isclose(roc_auc([1, 0, 1, 0], [4, 1, 3, 2]), 1.0)
