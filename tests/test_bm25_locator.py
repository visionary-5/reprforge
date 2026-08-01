from __future__ import annotations

from reprforge.bm25_locator import bm25_scores, tokenize


def test_bm25_prefers_document_with_query_terms() -> None:
    scores, posting_bytes, vocabulary_bytes = bm25_scores(
        ["annual revenue chart", "employee policy handbook"],
        ["revenue chart"],
    )

    assert tokenize("Revenue, CHART!") == ["revenue", "chart"]
    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]
    assert posting_bytes.sum() > 0
    assert vocabulary_bytes > 0
