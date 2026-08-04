"""Qrel-free scheduling for progressive query-cohort construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def _normalize_cohorts(cohorts: Sequence[Sequence[int]]) -> tuple[frozenset[int], ...]:
    normalized = tuple(frozenset(int(page) for page in cohort) for cohort in cohorts)
    if not normalized or any(not cohort for cohort in normalized):
        raise ValueError("cohorts must be a non-empty sequence of non-empty sets")
    return normalized


def static_popularity_order(cohorts: Sequence[Sequence[int]]) -> list[int]:
    """Order queries once by the aggregate popularity of their candidate pages."""

    normalized = _normalize_cohorts(cohorts)
    frequency = Counter(page for cohort in normalized for page in cohort)
    return sorted(
        range(len(normalized)),
        key=lambda query: (
            -sum(frequency[page] for page in normalized[query]),
            query,
        ),
    )


def frontier_reuse_order(
    cohorts: Sequence[Sequence[int]],
    *,
    batch_size: int,
) -> list[int]:
    """Schedule complete cohorts while minimizing new representation work.

    The scheduler observes only cheap candidate membership.  At each atomic
    request batch it first selects the query with the smallest resident miss
    set, breaking cold-start ties by future page reuse.  It then packs queries
    that add the fewest pages to the staged union.  Completed batches become
    resident before the next decision.
    """

    normalized = _normalize_cohorts(cohorts)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pending = set(range(len(normalized)))
    resident: set[int] = set()
    order: list[int] = []
    frequency = Counter(page for cohort in normalized for page in cohort)

    while pending:
        batch: list[int] = []
        staged: set[int] = set()
        while pending and len(batch) < batch_size:
            def key(query: int) -> tuple[int, int, int, int, int]:
                cohort = normalized[query]
                uncovered = cohort - resident - staged
                resident_hits = len(cohort & resident)
                staged_hits = len(cohort & staged)
                future_reuse = sum(frequency[page] for page in uncovered)
                return (
                    len(uncovered),
                    -resident_hits,
                    -staged_hits,
                    -future_reuse,
                    query,
                )

            selected = min(pending, key=key)
            pending.remove(selected)
            batch.append(selected)
            staged.update(normalized[selected] - resident)
        order.extend(batch)
        resident.update(staged)
        for query in batch:
            for page in normalized[query]:
                frequency[page] -= 1
    return order


def replay_page_work(
    cohorts: Sequence[Sequence[int]],
    order: Sequence[int],
    quality_gain: Sequence[float] | np.ndarray,
    *,
    base_mean_quality: float,
    batch_size: int,
    corpus_pages: int,
) -> dict:
    """Replay atomic cohort publications using encoded pages as exact work."""

    normalized = _normalize_cohorts(cohorts)
    permutation = [int(value) for value in order]
    if sorted(permutation) != list(range(len(normalized))):
        raise ValueError("order must be a permutation of query positions")
    gains = np.asarray(quality_gain, dtype=np.float64)
    if gains.shape != (len(normalized),) or not np.isfinite(gains).all():
        raise ValueError("quality gains must be a finite query-aligned vector")
    if batch_size <= 0 or corpus_pages <= 0:
        raise ValueError("batch size and corpus pages must be positive")

    resident: set[int] = set()
    completed = np.zeros(len(normalized), dtype=bool)
    completion_pages = np.zeros(len(normalized), dtype=np.float64)
    current_quality = float(base_mean_quality)
    points = [
        {
            "encoded_pages": 0,
            "completed_queries": 0,
            "mean_quality": current_quality,
        }
    ]
    for start in range(0, len(permutation), batch_size):
        batch = permutation[start : start + batch_size]
        staged = set().union(*(normalized[query] for query in batch)) - resident
        resident.update(staged)
        completed[batch] = True
        completion_pages[batch] = len(resident)
        current_quality += float(gains[batch].sum() / len(normalized))
        points.append(
            {
                "encoded_pages": len(resident),
                "new_pages": len(staged),
                "completed_queries": int(completed.sum()),
                "mean_quality": current_quality,
            }
        )

    if len(resident) > corpus_pages:
        raise ValueError("cohort union exceeds corpus size")
    quality_area = 0.0
    completion_area = 0.0
    previous_pages = 0
    previous_quality = float(base_mean_quality)
    previous_completion = 0.0
    for point in points[1:]:
        page_count = int(point["encoded_pages"])
        width = page_count - previous_pages
        quality_area += width * previous_quality
        completion_area += width * previous_completion
        previous_pages = page_count
        previous_quality = float(point["mean_quality"])
        previous_completion = float(point["completed_queries"] / len(normalized))
    if previous_pages < corpus_pages:
        width = corpus_pages - previous_pages
        quality_area += width * previous_quality
        completion_area += width * previous_completion

    return {
        "points": points,
        "final_unique_pages": len(resident),
        "mean_quality_over_full_build_work": quality_area / corpus_pages,
        "mean_completed_fraction_over_full_build_work": completion_area / corpus_pages,
        "completion_pages": {
            "mean": float(completion_pages.mean()),
            "p50": float(np.quantile(completion_pages, 0.50)),
            "p95": float(np.quantile(completion_pages, 0.95)),
            "max": float(completion_pages.max()),
        },
    }
