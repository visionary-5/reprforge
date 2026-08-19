"""Reference multi-vector index with ColBERT-style MaxSim."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float64]


def normalize_rows(value: ArrayLike) -> FloatMatrix:
    """Return a finite rank-2 matrix with unit-length rows."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("expected a non-empty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def maxsim_score(query: ArrayLike, document: ArrayLike) -> float:
    """Score one late-interaction document with MaxSim."""

    query_matrix = normalize_rows(query)
    document_matrix = normalize_rows(document)
    if query_matrix.shape[1] != document_matrix.shape[1]:
        raise ValueError("query and document dimensions differ")
    return float(np.max(query_matrix @ document_matrix.T, axis=1).sum())


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    score: float


class CompactIndex:
    """Small in-memory reference index for compiled page endpoints."""

    def __init__(self, items: Iterable[tuple[str, ArrayLike]]) -> None:
        records = [
            (str(item_id), normalize_rows(vectors)) for item_id, vectors in items
        ]
        if not records:
            raise ValueError("an index requires at least one item")
        identifiers = [item_id for item_id, _ in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("item identifiers must be unique")
        dimensions = {vectors.shape[1] for _, vectors in records}
        if len(dimensions) != 1:
            raise ValueError("all indexed vectors must share one dimension")
        self._items = tuple(records)
        self._by_id = dict(records)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item_id for item_id, _ in self._items)

    @property
    def dimension(self) -> int:
        return int(self._items[0][1].shape[1])

    @property
    def vector_count(self) -> int:
        return sum(len(vectors) for _, vectors in self._items)

    def records(self) -> tuple[tuple[str, FloatMatrix], ...]:
        """Return defensive copies for durable storage backends."""

        return tuple(
            (item_id, vectors.copy()) for item_id, vectors in self._items
        )

    def contains(self, item_id: str) -> bool:
        return item_id in self._by_id

    def search(self, query: ArrayLike, *, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        ranking = [
            SearchResult(item_id, maxsim_score(query, document))
            for item_id, document in self._items
        ]
        ranking.sort(key=lambda row: (-row.score, row.item_id))
        return ranking[: min(top_k, len(ranking))]
