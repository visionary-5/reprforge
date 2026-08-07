"""Evaluation helpers for physically built partial visual indexes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from reprforge.dvi_page_verifier import aggregate_query_metrics, ranking_metrics


def reciprocal_rank_fusion(
    *rankings: Sequence[str], constant: int = 60, depth: int = 100
) -> list[str]:
    if constant < 0 or depth <= 0:
        raise ValueError("invalid RRF parameters")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    counter = 0
    for ranking in rankings:
        for rank, raw_doc_id in enumerate(ranking[:depth], start=1):
            doc_id = str(raw_doc_id)
            if doc_id not in first_seen:
                first_seen[doc_id] = counter
                counter += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (constant + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id], doc_id))


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    if set(rankings) != set(qrels):
        raise ValueError("ranking and qrel query IDs differ")
    per_query = {
        query_id: ranking_metrics(rankings[query_id], qrels[query_id])
        for query_id in sorted(qrels)
    }
    return {
        "mean": aggregate_query_metrics(list(per_query.values())),
        "per_query": per_query,
    }


def gain_recovery(partial: float, text: float, full: float, *, epsilon: float = 0.005) -> float | None:
    denominator = full - text
    if abs(denominator) < epsilon:
        return None
    return (partial - text) / denominator
