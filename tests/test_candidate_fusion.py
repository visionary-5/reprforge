from __future__ import annotations

from pathlib import Path

import numpy as np

from reprforge.candidate_fusion import (
    _candidate_ndcg,
    _representation_cost,
)
from reprforge.progressive_oracle import FrozenTrace, rank_order, validate_pair


def _trace(mode: str, scores: np.ndarray) -> FrozenTrace:
    queries, corpus = scores.shape
    return FrozenTrace(
        root=Path("unused"),
        manifest={
            "mode": mode,
            "runtime_sha256": mode,
            "oracle_labels_sha256": "labels",
            "official_upstream_commit": "test",
            "source_sha256": {"corpus": "c", "queries": "q", "qrels": "r"},
        },
        query_ids=np.asarray([f"q{value}" for value in range(queries)]),
        corpus_ids=np.asarray([f"d{value}" for value in range(corpus)]),
        scores=scores,
        vector_bytes=np.arange(1, corpus + 1, dtype=np.int64),
        encode_ms=np.arange(1, corpus + 1, dtype=np.float64),
        index_total_ms=10.0 if mode == "text" else 30.0,
        label_query=np.arange(queries, dtype=np.int32),
        label_corpus=np.arange(queries, dtype=np.int32) % corpus,
        relevance=np.ones(queries, dtype=np.int16),
    )


def test_candidate_normalization_combines_text_and_visual_rank() -> None:
    text = _trace(
        "text",
        np.asarray(
            [
                [1.0, 0.9, 0.8, 0.0],
                [0.9, 1.0, 0.8, 0.0],
                [0.8, 0.9, 1.0, 0.0],
            ]
        ),
    )
    visual = _trace(
        "visual",
        np.asarray(
            [
                [1.2, 0.1, 0.0, 0.0],
                [0.1, 1.2, 0.0, 0.0],
                [0.1, 0.0, 1.2, 0.0],
            ]
        ),
    )
    qrels = validate_pair(text, visual)
    quality = _candidate_ndcg(
        text,
        visual,
        qrels,
        rank_order(text.scores, text.corpus_ids),
        candidate_k=3,
        method="zscore_sum",
        cutoff=3,
    )
    assert np.allclose(quality, 1.0)


def test_representation_cost_counts_first_touch_once() -> None:
    visual = _trace("visual", np.zeros((2, 4)))
    order = np.asarray([[0, 1, 2, 3], [1, 2, 3, 0]])
    text = _trace("text", np.zeros((2, 4)))
    result = _representation_cost(order, text, visual, candidate_k=2)

    assert result["candidate_events"] == 4
    assert result["unique_visual_pages"] == 3
    assert result["unbounded_cache_hit_fraction"] == 0.25
    assert result["visual_encode_ms_unique_pages"] == 6.0
    assert result["text_plus_unique_visual_build_ms"] == 16.0
