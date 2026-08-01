from __future__ import annotations

from pathlib import Path

import numpy as np

from reprforge.intervention_utility import (
    average_precision,
    build_intervention_events,
    fit_ridge_utility,
)
from reprforge.progressive_oracle import FrozenTrace


def _trace(mode: str, scores: np.ndarray) -> FrozenTrace:
    queries, corpus = scores.shape
    query_ids = np.asarray([f"q{index}" for index in range(queries)])
    return FrozenTrace(
        root=Path("unused"),
        manifest={
            "mode": mode,
            "runtime_sha256": mode,
            "oracle_labels_sha256": "labels",
            "official_upstream_commit": "test",
            "source_sha256": {"corpus": "c", "queries": "q", "qrels": "r"},
        },
        query_ids=query_ids,
        corpus_ids=np.asarray([f"d{index}" for index in range(corpus)]),
        scores=scores,
        vector_bytes=np.ones(corpus, dtype=np.int64),
        encode_ms=np.ones(corpus),
        index_total_ms=1.0,
        label_query=np.arange(queries, dtype=np.int32),
        label_corpus=np.arange(queries, dtype=np.int32) % corpus,
        relevance=np.ones(queries, dtype=np.int16),
    )


def test_intervention_events_capture_helpful_rank_change() -> None:
    text_scores = np.zeros((10, 10), dtype=np.float64)
    visual_scores = np.zeros((10, 10), dtype=np.float64)
    for query in range(10):
        relevant = query
        distractor = (query + 1) % 10
        text_scores[query, distractor] = 1.0
        text_scores[query, relevant] = 0.9
        visual_scores[query] = text_scores[query]
        visual_scores[query, relevant] = 1.1

    events = build_intervention_events(
        _trace("text", text_scores),
        _trace("visual", visual_scores),
        candidate_k=2,
        cutoff=10,
        route_count=2,
    )

    relevant_events = events.corpus_position == events.query_position
    assert np.all(events.utility[relevant_events] > 0)
    assert np.all(events.utility[~relevant_events] == 0)
    assert events.features.shape == (20, len(events.feature_names))
    assert set(events.query_split.tolist()) == {"train", "validation", "test"}


def test_weighted_ridge_and_average_precision_rank_signal() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]])
    utility = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
    model = fit_ridge_utility(features, utility, alpha=0.01)
    prediction = model.predict(features)

    assert np.all(np.diff(prediction) > 0)
    assert average_precision(utility > 0, prediction) == 1.0
