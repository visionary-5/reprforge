"""Compact multi-vector indexes and query-conditioned refinement."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .alignment import Matrix, normalize_rows


def coalesce_visual_tokens(
    vectors: ArrayLike,
    *,
    grid_shape: tuple[int, int],
    auxiliary_tokens: int = 0,
) -> Matrix:
    """Merge adjacent visual columns while preserving local slot identity.

    Visual tokens must come first and form an even two-dimensional grid.
    Auxiliary tokens (for example BOS/EOS states) are appended unchanged.
    """

    matrix = normalize_rows(vectors)
    rows, columns = grid_shape
    if rows <= 0 or columns <= 0 or rows % 2 or columns % 2:
        raise ValueError("grid dimensions must be positive and even")
    if auxiliary_tokens < 0:
        raise ValueError("auxiliary token count cannot be negative")
    visual_tokens = rows * columns
    if visual_tokens + auxiliary_tokens != len(matrix):
        raise ValueError("grid and auxiliary token counts do not match the vectors")

    visual = matrix[:visual_tokens]
    dimension = visual.shape[1]
    blocks = visual.reshape(rows // 2, 2, columns // 2, 2, dimension)
    pooled = blocks.transpose(0, 2, 1, 3, 4).mean(axis=3).reshape(-1, dimension)
    pooled = normalize_rows(pooled)
    if not auxiliary_tokens:
        return pooled
    return np.concatenate((pooled, matrix[visual_tokens:]), axis=0)


def maxsim_score(query: ArrayLike, document: ArrayLike) -> float:
    """Score one late-interaction document with ColBERT-style MaxSim."""

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
    """A small in-memory reference index for compiled page vectors."""

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

    def search(self, query: ArrayLike, *, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        ranking = [
            SearchResult(item_id, maxsim_score(query, document))
            for item_id, document in self._items
        ]
        ranking.sort(key=lambda row: (-row.score, row.item_id))
        return ranking[: min(top_k, len(ranking))]

    def refine(
        self,
        query: ArrayLike,
        candidates: Iterable[str | SearchResult],
        materialize_full: Callable[[str], ArrayLike],
        *,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Materialize Full vectors only for compact-selected candidates."""

        candidate_ids = tuple(
            candidate.item_id if isinstance(candidate, SearchResult) else str(candidate)
            for candidate in candidates
        )
        if not candidate_ids:
            raise ValueError("refinement requires at least one candidate")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("refinement candidates must be unique")
        unknown = [item_id for item_id in candidate_ids if item_id not in self._by_id]
        if unknown:
            raise KeyError(f"candidate is outside the compact index: {unknown[0]}")
        limit = len(candidate_ids) if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        ranking = [
            SearchResult(item_id, maxsim_score(query, materialize_full(item_id)))
            for item_id in candidate_ids
        ]
        ranking.sort(key=lambda row: (-row.score, row.item_id))
        return ranking[: min(limit, len(ranking))]
