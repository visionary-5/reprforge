"""Pure evaluation helpers for the GPU DVI-like page-verifier pilot."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


def deterministic_query_sample(
    query_ids: Sequence[str], *, limit: int, seed: int
) -> list[str]:
    ranked = sorted(
        map(str, query_ids),
        key=lambda query_id: (
            hashlib.sha256(f"{query_id}\0{seed}".encode()).digest(),
            query_id,
        ),
    )
    return ranked[: min(int(limit), len(ranked))]


def union_preserving_order(*rankings: Sequence[str]) -> list[str]:
    output = []
    seen = set()
    for ranking in rankings:
        for item in map(str, ranking):
            if item not in seen:
                output.append(item)
                seen.add(item)
    return output


def rerank_with_scores(
    candidates: Sequence[str], scores: Mapping[str, float]
) -> list[str]:
    positions = {str(item): position for position, item in enumerate(candidates)}
    return sorted(
        map(str, candidates),
        key=lambda item: (-float(scores[item]), positions[item], item),
    )


def dcg_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    return sum(
        (2.0 ** float(qrels.get(str(item), 0.0)) - 1.0) / math.log2(rank + 1)
        for rank, item in enumerate(ranking[:k], start=1)
    )


def ranking_metrics(
    ranking: Sequence[str], qrels: Mapping[str, float]
) -> dict[str, float]:
    ideal = sorted(map(float, qrels.values()), reverse=True)
    ideal_dcg = sum(
        (2.0**value - 1.0) / math.log2(rank + 1)
        for rank, value in enumerate(ideal[:10], start=1)
    )
    total = sum(map(float, qrels.values()))
    return {
        "query_hit": float(any(str(item) in qrels for item in ranking)),
        "ndcg_at_10": dcg_at_k(ranking, qrels, 10) / ideal_dcg if ideal_dcg else 0.0,
        "recall_at_5": (
            sum(float(qrels.get(str(item), 0.0)) for item in ranking[:5]) / total
            if total
            else 0.0
        ),
        "recall_at_20": (
            sum(float(qrels.get(str(item), 0.0)) for item in ranking[:20]) / total
            if total
            else 0.0
        ),
    }


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    positives = np.flatnonzero(labels_array > 0)
    negatives = np.flatnonzero(labels_array <= 0)
    if not len(positives) or not len(negatives):
        return None
    wins = 0.0
    for positive in positives:
        differences = scores_array[positive] - scores_array[negatives]
        wins += float(np.sum(differences > 0)) + 0.5 * float(
            np.sum(differences == 0)
        )
    return wins / (len(positives) * len(negatives))


def aggregate_query_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("at least one query metric row is required")
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
    }
