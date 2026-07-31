from __future__ import annotations

from pathlib import Path

import numpy as np

from reprforge.progressive_oracle import FrozenTrace, analyze, mean_ndcg, rank_order


def test_rank_order_breaks_score_ties_by_identifier() -> None:
    scores = np.asarray([[1.0, 1.0, 0.0]])
    identifiers = np.asarray(["b", "a", "c"])

    order = rank_order(scores, identifiers)

    assert order.tolist() == [[1, 0, 2]]


def test_mean_ndcg_uses_linear_graded_relevance() -> None:
    scores = np.asarray([[0.8, 1.0, 0.1]])
    qrels = np.asarray([[2, 1, 0]])
    identifiers = np.asarray(["a", "b", "c"])
    expected = (1.0 + 2.0 / np.log2(3.0)) / (
        2.0 + 1.0 / np.log2(3.0)
    )

    observed = mean_ndcg(scores, qrels, identifiers, cutoff=3)

    assert observed == expected


def _trace(mode: str, scores: np.ndarray) -> FrozenTrace:
    corpus = scores.shape[1]
    return FrozenTrace(
        root=Path("unused"),
        manifest={
            "mode": mode,
            "runtime_sha256": f"{mode}-runtime",
            "oracle_labels_sha256": "labels",
            "official_upstream_commit": "test",
            "source_sha256": {"corpus": "c", "queries": "q", "qrels": "r"},
        },
        query_ids=np.asarray(["q0", "q1"]),
        corpus_ids=np.asarray([f"d{index}" for index in range(corpus)]),
        scores=scores,
        vector_bytes=np.full(corpus, 10 if mode == "text" else 20),
        encode_ms=np.ones(corpus),
        index_total_ms=10.0 if mode == "text" else 20.0,
        label_query=np.asarray([0, 1], dtype=np.int32),
        label_corpus=np.asarray([0, 1], dtype=np.int32),
        relevance=np.asarray([1, 1], dtype=np.int16),
    )


def test_analyze_labels_oracle_selector_and_finds_small_witness() -> None:
    text_scores = np.zeros((2, 10), dtype=np.float64)
    visual_scores = np.zeros((2, 10), dtype=np.float64)
    text_scores[0, 2], text_scores[0, 0] = 1.0, 0.9
    text_scores[1, 3], text_scores[1, 1] = 1.0, 0.9
    visual_scores[0, 0], visual_scores[0, 2] = 1.1, 1.0
    visual_scores[1, 1], visual_scores[1, 3] = 1.1, 1.0

    result = analyze(
        _trace("text", text_scores),
        _trace("visual", visual_scores),
        counts=[0, 1, 2, 3],
    )

    selector = result["selectors"]["positive_relevant_rank_sensitivity"]
    witness = selector["first_gate_witness"]
    assert selector["runtime_deployable_as_written"] is False
    assert witness["resident_count"] == 2
    assert witness["resident_fraction"] == 0.2
    assert witness["ndcg@10"] == 1.0
    assert all(
        not value["runtime_deployable_as_written"]
        for value in result["selectors"].values()
    )
