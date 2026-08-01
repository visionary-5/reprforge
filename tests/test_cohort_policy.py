from __future__ import annotations

from pathlib import Path

import numpy as np

from reprforge.cohort_policy import (
    build_cohort_quality_curves,
    choose_cohorts,
    query_folds,
    tfidf_profiles,
)
from reprforge.progressive_oracle import FrozenTrace


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
        vector_bytes=np.ones(corpus, dtype=np.int64),
        encode_ms=np.ones(corpus),
        index_total_ms=1.0,
        label_query=np.arange(queries, dtype=np.int32),
        label_corpus=np.arange(queries, dtype=np.int32) % corpus,
        relevance=np.ones(queries, dtype=np.int16),
    )


def test_cohort_curve_improves_when_visual_moves_relevant_page() -> None:
    text = np.zeros((5, 5))
    visual = np.zeros((5, 5))
    for query in range(5):
        text[query, (query + 1) % 5] = 1.0
        text[query, query] = 0.9
        visual[query] = text[query]
        visual[query, query] = 1.1

    sizes, quality = build_cohort_quality_curves(
        _trace("text", text),
        _trace("visual", visual),
        cohort_sizes=[0, 2, 5],
        cutoff=5,
    )

    assert sizes.tolist() == [0, 2, 5]
    assert np.all(quality[:, 1] > quality[:, 0])


def test_tfidf_uses_train_vocabulary_and_policy_charges_work() -> None:
    profiles = tfidf_profiles(
        ["red chart", "red table", "unseen diagram"],
        np.asarray([0, 1]),
    )
    assert profiles.shape[0] == 3
    predicted = np.asarray([[0.5, 0.6, 0.61]])
    sizes = np.asarray([0, 10, 100])

    free = choose_cohorts(predicted, sizes, visual_work_price=0.0)
    charged = choose_cohorts(predicted, sizes, visual_work_price=0.2)

    assert free.tolist() == [2]
    assert charged.tolist() == [1]
    assert len(set(query_folds([f"q{i}" for i in range(30)]).tolist())) >= 3
