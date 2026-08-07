"""Candidate-escape and repair-reuse diagnostics for defer/materialize RAG."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from reprforge.partial_vlm_materialization import ScoreSurface


def _first_relevant_ranks(surface: ScoreSurface, order: np.ndarray) -> np.ndarray:
    ranks = np.full(surface.queries, surface.pages + 1, dtype=np.int32)
    relevant = surface.qrels > 0
    for query in range(surface.queries):
        hits = np.flatnonzero(relevant[query, order[query]])
        if hits.size:
            ranks[query] = int(hits[0]) + 1
    return ranks


def _weighted_recall(surface: ScoreSurface, order: np.ndarray, depth: int) -> float:
    total = float(np.sum(surface.qrels))
    if total <= 0:
        return 0.0
    observed = 0.0
    for query in range(surface.queries):
        observed += float(np.sum(surface.qrels[query, order[query, :depth]]))
    return observed / total


def locator_boundary(surface: ScoreSurface, depth: int) -> dict[str, Any]:
    depth = min(max(int(depth), 1), surface.pages)
    text_rank = _first_relevant_ranks(surface, surface.text_order)
    visual_rank = _first_relevant_ranks(surface, surface.visual_order)
    text_hit = text_rank <= depth
    visual_hit = visual_rank <= depth
    text_miss = ~text_hit
    repairable = text_miss & visual_hit
    visual_harm = text_hit & ~visual_hit
    both_hit = text_hit & visual_hit
    optimistic_text_pages = np.minimum(text_rank, depth)
    optimistic_visual_pages = np.minimum(visual_rank, depth)
    return {
        "depth": depth,
        "queries": surface.queries,
        "text": {
            "query_hit_rate": float(np.mean(text_hit)),
            "weighted_recall": _weighted_recall(surface, surface.text_order, depth),
            "optimistic_pages_analyzed_mean": float(np.mean(optimistic_text_pages)),
            "optimistic_pages_analyzed_p95": float(
                np.percentile(optimistic_text_pages, 95)
            ),
        },
        "visual": {
            "query_hit_rate": float(np.mean(visual_hit)),
            "weighted_recall": _weighted_recall(surface, surface.visual_order, depth),
            "optimistic_pages_analyzed_mean": float(np.mean(optimistic_visual_pages)),
            "optimistic_pages_analyzed_p95": float(
                np.percentile(optimistic_visual_pages, 95)
            ),
        },
        "boundary": {
            "text_miss_queries": int(np.sum(text_miss)),
            "text_miss_fraction": float(np.mean(text_miss)),
            "visual_repair_queries": int(np.sum(repairable)),
            "visual_repairs_fraction_of_text_misses": (
                float(np.sum(repairable) / np.sum(text_miss))
                if np.any(text_miss)
                else None
            ),
            "visual_harm_queries": int(np.sum(visual_harm)),
            "both_hit_queries": int(np.sum(both_hit)),
            "mean_first_relevant_rank_delta_visual_minus_text_on_both_hit": (
                float(np.mean(visual_rank[both_hit] - text_rank[both_hit]))
                if np.any(both_hit)
                else None
            ),
        },
    }


def _repair_pages_for_queries(
    surface: ScoreSurface, queries: Sequence[int], depth: int
) -> tuple[set[int], list[set[int]]]:
    all_pages: set[int] = set()
    events: list[set[int]] = []
    relevant = surface.qrels > 0
    for query in map(int, queries):
        text_pages = set(map(int, surface.text_order[query, :depth]))
        visual_pages = set(map(int, surface.visual_order[query, :depth]))
        if any(relevant[query, page] for page in text_pages):
            continue
        repair = {
            page
            for page in visual_pages - text_pages
            if relevant[query, page]
        }
        if repair:
            events.append(repair)
            all_pages.update(repair)
    return all_pages, events


def repair_reuse_crossfit(
    surface: ScoreSurface, assignments: np.ndarray, depth: int
) -> dict[str, Any]:
    rows = []
    for fold in sorted(set(map(int, assignments))):
        history = np.flatnonzero(assignments != fold)
        future = np.flatnonzero(assignments == fold)
        history_pages, _ = _repair_pages_for_queries(surface, history, depth)
        future_pages, future_events = _repair_pages_for_queries(surface, future, depth)
        page_events = sum(len(event) for event in future_events)
        repeated_events = sum(len(event & history_pages) for event in future_events)
        rows.append(
            {
                "fold": fold,
                "history_repair_pages": len(history_pages),
                "future_repair_queries": len(future_events),
                "future_unique_repair_pages": len(future_pages),
                "future_repair_page_events": page_events,
                "unique_page_overlap_fraction": (
                    len(future_pages & history_pages) / len(future_pages)
                    if future_pages
                    else None
                ),
                "event_overlap_fraction": (
                    repeated_events / page_events if page_events else None
                ),
            }
        )
    unique_weight = sum(row["future_unique_repair_pages"] for row in rows)
    event_weight = sum(row["future_repair_page_events"] for row in rows)
    return {
        "depth": depth,
        "folds": rows,
        "future_repair_queries": sum(row["future_repair_queries"] for row in rows),
        "unique_page_overlap_fraction_weighted": (
            sum(
                row["unique_page_overlap_fraction"]
                * row["future_unique_repair_pages"]
                for row in rows
                if row["unique_page_overlap_fraction"] is not None
            )
            / unique_weight
            if unique_weight
            else None
        ),
        "event_overlap_fraction_weighted": (
            sum(
                row["event_overlap_fraction"] * row["future_repair_page_events"]
                for row in rows
                if row["event_overlap_fraction"] is not None
            )
            / event_weight
            if event_weight
            else None
        ),
    }
