import numpy as np

from reprforge.progressive_evidence import (
    apply_progressive_policy,
    build_evidence_stages,
    paper_disjoint_bm25_margin_router,
    paper_disjoint_progressive_probe,
    select_zero_error_thresholds,
)


def _surface() -> tuple[list[str], np.ndarray, np.ndarray]:
    corpus_ids = [f"d{value}" for value in range(12)]
    locator = np.asarray(
        [
            np.arange(12, 0, -1),
            np.arange(12, 0, -1),
            np.arange(12, 0, -1),
            np.arange(12, 0, -1),
        ],
        dtype=np.float64,
    )
    expensive = np.asarray(
        [
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [11, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [11, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=np.float64,
    )
    return corpus_ids, locator, expensive


def test_build_stages_preserves_tail_and_is_deterministic() -> None:
    corpus_ids, locator, expensive = _surface()
    evidence = build_evidence_stages(corpus_ids, locator, expensive, top_k=12)
    assert evidence.stages == (2, 4, 6, 8, 10)
    assert evidence.orders[4].shape == (4, 12)
    assert set(evidence.orders[4][0, :4]) == {0, 1, 2, 3}
    assert evidence.orders[4][0, 4:].tolist() == list(range(4, 12))
    assert evidence.top1_margins[4].shape == (4,)


def test_progressive_policy_stops_only_after_stability_and_margin() -> None:
    corpus_ids, locator, expensive = _surface()
    evidence = build_evidence_stages(corpus_ids, locator, expensive, top_k=12)
    selected, rankings = apply_progressive_policy(
        evidence,
        [0, 1],
        {4: 0.0, 6: 0.0, 8: 0.0},
    )
    assert selected.tolist() == [4, 4]
    assert rankings.shape == (2, 12)

    selected, _ = apply_progressive_policy(
        evidence,
        [0],
        {4: 100.0, 6: 100.0, 8: 100.0},
    )
    assert selected.tolist() == [10]


def test_zero_error_selection_and_group_probe_do_not_need_qrels() -> None:
    corpus_ids, locator, expensive = _surface()
    evidence = build_evidence_stages(corpus_ids, locator, expensive, top_k=12)
    grid = {4: (0.0, 100.0), 6: (0.0, 100.0), 8: (0.0, 100.0)}
    thresholds = select_zero_error_thresholds(
        evidence,
        [0, 1, 2],
        threshold_grid=grid,
    )
    assert set(thresholds) == {4, 6, 8}

    result = paper_disjoint_progressive_probe(
        evidence,
        ["a", "a", "b", "b"],
        threshold_grid=grid,
    )
    assert result["rankings"].shape == (4, 12)
    assert result["selected_depths"].shape == (4,)
    assert len(result["folds"]) == 2


def test_bm25_margin_router_returns_aligned_rankings() -> None:
    corpus_ids, locator, expensive = _surface()
    evidence = build_evidence_stages(corpus_ids, locator, expensive, top_k=12)
    result = paper_disjoint_bm25_margin_router(
        evidence,
        ["a", "a", "b", "b"],
        locator,
    )
    assert result["rankings"].shape == (4, 12)
    assert set(result["selected_depths"].tolist()) <= {0, 10}
