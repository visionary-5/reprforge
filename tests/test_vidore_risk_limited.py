import numpy as np

from tools.analyze_vidore_risk_limited import (
    _alignment,
    _complete_rankings,
    _metrics,
    _work,
)


def test_complete_rankings_preserves_topk_and_candidate_tail() -> None:
    candidates = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)
    topk = np.asarray([[2, 0], [1, 3]], dtype=np.int32)
    complete = _complete_rankings(topk, candidates)
    assert complete.tolist() == [[2, 0, 1, 3], [1, 3, 2, 0]]


def test_metrics_use_graded_ndcg_and_binary_recall() -> None:
    rankings = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
    result = _metrics(rankings, [{0: 2, 2: 1}, {1: 1}])
    assert 0.0 < result["ndcg@10"] <= 1.0
    assert result["recall@100"] == 1.0


def test_work_charges_unique_builds_but_all_candidate_events() -> None:
    result = _work(
        [(0, 1), (1, 2)],
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([10, 20, 30]),
    )
    assert result["candidate_events"] == 4
    assert result["unique_pages"] == 3
    assert result["unique_visual_build_ms"] == 6.0
    assert result["unique_visual_bytes"] == 60


def test_alignment_distinguishes_harmless_teacher_set_changes() -> None:
    teacher = np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
    ranking = np.asarray([[0, 1, 3], [0, 2, 1]], dtype=np.int32)
    result = _alignment(ranking, teacher, [{0: 2}, {0: 2}])
    assert result["teacher_set_disagreements"] == 1
    assert result["harmless_for_ndcg"] == 1
    assert result["harmful_for_ndcg"] == 0
