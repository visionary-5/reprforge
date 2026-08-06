import numpy as np

from reprforge.partial_vlm_materialization import (
    ScoreSurface,
    evaluate_selection,
    evaluate_text_only,
    fold_assignments,
    gain_recovery,
    online_trace_audit,
    relevance_reuse_crossfit,
    select_pages,
)


def _surface() -> ScoreSurface:
    return ScoreSurface(
        name="synthetic",
        query_ids=np.asarray(["q0", "q1", "q2", "q3"]),
        corpus_ids=np.asarray(["d0", "d1", "d2", "d3", "d4", "d5"]),
        text_scores=np.asarray(
            [
                [9, 8, 7, 6, 5, 4],
                [8, 9, 7, 6, 5, 4],
                [7, 6, 9, 8, 5, 4],
                [7, 6, 8, 9, 5, 4],
            ],
            dtype=np.float64,
        ),
        visual_scores=np.asarray(
            [
                [1, 2, 3, 4, 9, 8],
                [2, 1, 3, 4, 8, 9],
                [3, 4, 1, 2, 9, 8],
                [3, 4, 2, 1, 8, 9],
            ],
            dtype=np.float64,
        ),
        qrels=np.asarray(
            [
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.int16,
        ),
        text_bytes=np.asarray([10, 20, 30, 40, 1, 2], dtype=np.float64),
        visual_bytes=np.full(6, 100.0),
        visual_encode_ms=np.full(6, 5.0),
        input_sha256={"synthetic": "0" * 64},
    )


def test_partial_rrf_charges_only_materialized_pages_and_improves_visual_gold():
    surface = _surface()
    text = evaluate_text_only(surface, [0, 1])
    partial = evaluate_selection(
        surface,
        [0, 1],
        [4, 5],
        fusion="rrf",
        text_top_k=6,
        visual_top_k=6,
        rrf_constant=60,
    )
    assert partial["selected_pages"] == 2
    assert partial["selected_visual_bytes"] == 200
    assert partial["selected_encode_ms"] == 10
    assert partial["mean_ndcg_at_10"] > text["mean_ndcg_at_10"]


def test_global_rank_oracle_does_not_promote_selected_pages_to_fake_visual_top1():
    pages = 120
    text = np.arange(pages, 0, -1, dtype=np.float64)[None, :]
    visual_order = list(range(99)) + [119] + list(range(99, 119))
    visual = np.empty(pages, dtype=np.float64)
    for rank, page in enumerate(visual_order):
        visual[page] = pages - rank
    qrels = np.zeros((1, pages), dtype=np.int16)
    qrels[0, 119] = 1
    surface = ScoreSurface(
        name="rank-calibration",
        query_ids=np.asarray(["q"]),
        corpus_ids=np.asarray([f"d{i}" for i in range(pages)]),
        text_scores=text,
        visual_scores=visual[None, :],
        qrels=qrels,
        text_bytes=np.ones(pages),
        visual_bytes=np.ones(pages),
        visual_encode_ms=np.ones(pages),
        input_sha256={},
    )
    selected = [119]
    naive = evaluate_selection(
        surface,
        [0],
        selected,
        fusion="rrf",
        text_top_k=100,
        visual_top_k=100,
        rrf_constant=60,
    )
    oracle_calibrated = evaluate_selection(
        surface,
        [0],
        selected,
        fusion="rrf_global_oracle",
        text_top_k=100,
        visual_top_k=100,
        rrf_constant=60,
    )
    assert naive["mean_ndcg_at_10"] > oracle_calibrated["mean_ndcg_at_10"]


def test_history_policy_does_not_read_future_visual_scores_or_qrels():
    original = _surface()
    changed = _surface()
    changed.visual_scores[2:] *= -100
    changed.qrels[2:] = changed.qrels[2:, ::-1]
    first = select_pages(
        original,
        policy="history_frequency",
        count=3,
        history_queries=[0, 1],
        future_queries=[2, 3],
        seed=7,
    )
    second = select_pages(
        changed,
        policy="history_frequency",
        count=3,
        history_queries=[0, 1],
        future_queries=[2, 3],
        seed=7,
    )
    np.testing.assert_array_equal(first, second)


def test_history_relevance_policy_reads_history_feedback_but_not_future_labels():
    original = _surface()
    changed = _surface()
    changed.qrels[2:] = changed.qrels[2:, ::-1]
    first = select_pages(
        original,
        policy="history_relevance",
        count=2,
        history_queries=[0, 1],
        future_queries=[2, 3],
        seed=7,
    )
    second = select_pages(
        changed,
        policy="history_relevance",
        count=2,
        history_queries=[0, 1],
        future_queries=[2, 3],
        seed=7,
    )
    np.testing.assert_array_equal(first, second)
    assert set(first) == {4, 5}


def test_relevance_reuse_crossfit_distinguishes_repeated_evidence():
    result = relevance_reuse_crossfit(_surface(), np.asarray([0, 0, 1, 1]))
    assert result["unique_page_overlap_fraction_weighted"] == 1.0
    assert result["event_overlap_fraction_weighted"] == 1.0


def test_label_rank_oracle_is_explicitly_allowed_to_use_future_labels():
    selected = select_pages(
        _surface(),
        policy="label_rank_oracle",
        count=2,
        history_queries=[0, 1],
        future_queries=[2, 3],
        seed=7,
    )
    assert set(selected) == {4, 5}


def test_online_persistence_reuses_visual_page_construction_events():
    result = online_trace_audit(
        _surface(),
        [0, 1, 2, 3],
        scope_top_k=3,
        text_top_k=6,
        visual_top_k=6,
        rrf_constant=60,
    )
    assert result["nonpersistent"]["visual_page_events"] == 12
    assert result["persistent"]["unique_visual_pages_materialized"] < 12
    assert result["amortization"]["page_event_reuse_fraction"] > 0


def test_fold_assignment_and_gain_recovery_are_deterministic_and_explicit():
    surface = _surface()
    np.testing.assert_array_equal(
        fold_assignments(surface, 2, 17), fold_assignments(surface, 2, 17)
    )
    recovered = gain_recovery(0.7, 0.6, 0.8)
    assert np.isclose(recovered["gain_recovery"], 0.5)
    omitted = gain_recovery(0.7001, 0.7, 0.701)
    assert omitted["gain_recovery"] is None
