"""Optional Full materialization for compact-selected candidates."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from numpy.typing import ArrayLike

from ..indexing import CompactIndex, SearchResult, maxsim_score


def refine_candidates(
    index: CompactIndex,
    query: ArrayLike,
    candidates: Iterable[str | SearchResult],
    materialize_full: Callable[[str], ArrayLike],
    *,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Materialize Full vectors only for candidates selected by compact search."""

    candidate_ids = tuple(
        candidate.item_id if isinstance(candidate, SearchResult) else str(candidate)
        for candidate in candidates
    )
    if not candidate_ids:
        raise ValueError("refinement requires at least one candidate")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("refinement candidates must be unique")
    unknown = [item_id for item_id in candidate_ids if not index.contains(item_id)]
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
