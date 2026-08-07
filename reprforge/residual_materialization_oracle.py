"""Rank-surface oracle for a residual high-fidelity visual index tier."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class ResidualRankSurface:
    name: str
    query_ids: list[str]
    doc_ids: list[str]
    bm25: np.ndarray
    colsmol: np.ndarray
    omni: np.ndarray
    qrels: np.ndarray

    def __post_init__(self) -> None:
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("duplicate query IDs")
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("duplicate document IDs")
        if self.qrels.shape != (len(self.query_ids), len(self.doc_ids)):
            raise ValueError("qrels shape mismatch")
        for label, ranking in (
            ("bm25", self.bm25),
            ("colsmol", self.colsmol),
            ("omni", self.omni),
        ):
            if ranking.ndim != 2 or ranking.shape[0] != len(self.query_ids):
                raise ValueError(f"{label} ranking shape mismatch")
            if np.any(ranking < 0) or np.any(ranking >= len(self.doc_ids)):
                raise ValueError(f"{label} ranking contains unknown page")
            for row in ranking:
                if len(set(map(int, row))) != len(row):
                    raise ValueError(f"{label} ranking contains duplicate page")

    @property
    def queries(self) -> int:
        return len(self.query_ids)

    @property
    def pages(self) -> int:
        return len(self.doc_ids)


def hash_folds(query_ids: Sequence[str], folds: int, seed: int) -> np.ndarray:
    if folds < 2:
        raise ValueError("at least two folds are required")
    values = [
        int.from_bytes(
            hashlib.sha256(f"{query_id}\0{seed}".encode()).digest()[:8], "big"
        )
        % folds
        for query_id in query_ids
    ]
    result = np.asarray(values, dtype=np.int16)
    if len(set(map(int, result))) != folds:
        raise ValueError("fold assignment contains an empty fold")
    return result


def rrf_ranking(
    surface: ResidualRankSurface,
    query: int,
    *,
    rrf_constant: int,
    selected_omni_pages: Iterable[int] | None,
    include_bm25: bool = True,
    include_colsmol: bool = True,
) -> np.ndarray:
    """Fuse ranks while retaining complete-index ranks for selected Omni pages."""

    contributions = np.zeros(surface.pages, dtype=np.float64)
    observed = np.zeros(surface.pages, dtype=bool)
    for enabled, ranking in (
        (include_bm25, surface.bm25[query]),
        (include_colsmol, surface.colsmol[query]),
    ):
        if not enabled:
            continue
        ranks = np.arange(1, len(ranking) + 1, dtype=np.float64)
        contributions[ranking] += 1.0 / (rrf_constant + ranks)
        observed[ranking] = True
    if selected_omni_pages is not None:
        mask = np.zeros(surface.pages, dtype=bool)
        selected = np.asarray(sorted(set(map(int, selected_omni_pages))), dtype=np.int32)
        if selected.size:
            if selected.min() < 0 or selected.max() >= surface.pages:
                raise ValueError("selected Omni page outside corpus")
            mask[selected] = True
            omni = surface.omni[query]
            active = mask[omni]
            active_pages = omni[active]
            global_ranks = np.flatnonzero(active).astype(np.float64) + 1.0
            contributions[active_pages] += 1.0 / (rrf_constant + global_ranks)
            observed[active_pages] = True
    pages = np.flatnonzero(observed)
    return pages[np.lexsort((pages, -contributions[pages]))]


def _dcg(relevance: np.ndarray) -> float:
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2, dtype=np.float64))
    return float(np.sum((np.power(2.0, relevance) - 1.0) / discounts))


def ranking_metrics(
    surface: ResidualRankSurface, query: int, ranking: np.ndarray
) -> dict[str, float]:
    relevance = surface.qrels[query]
    ideal = _dcg(np.sort(relevance)[::-1][:10])
    ndcg = _dcg(relevance[ranking[:10]]) / ideal if ideal > 0 else 0.0
    total = float(np.sum(relevance))
    return {
        "ndcg_at_10": ndcg,
        "query_hit_at_20": float(bool(np.any(relevance[ranking[:20]] > 0))),
        "recall_at_20": float(np.sum(relevance[ranking[:20]])) / total if total else 0.0,
        "recall_at_100": float(np.sum(relevance[ranking[:100]])) / total if total else 0.0,
    }


def evaluate(
    surface: ResidualRankSurface,
    queries: Sequence[int],
    *,
    rrf_constant: int,
    selected_omni_pages: Iterable[int] | None,
    include_bm25: bool = True,
    include_colsmol: bool = True,
) -> dict[str, Any]:
    selected = (
        np.empty(0, dtype=np.int32)
        if selected_omni_pages is None
        else np.asarray(sorted(set(map(int, selected_omni_pages))), dtype=np.int32)
    )
    rows = []
    rankings = []
    for query in map(int, queries):
        ranking = rrf_ranking(
            surface,
            query,
            rrf_constant=rrf_constant,
            selected_omni_pages=(None if selected_omni_pages is None else selected),
            include_bm25=include_bm25,
            include_colsmol=include_colsmol,
        )
        rankings.append(ranking)
        rows.append(ranking_metrics(surface, query, ranking))
    return {
        "queries": len(queries),
        "selected_pages": int(len(selected)),
        "selected_page_fraction": len(selected) / surface.pages,
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in ("ndcg_at_10", "query_hit_at_20", "recall_at_20", "recall_at_100")
        },
        "rankings": rankings,
    }


def direct_ranking_evaluation(
    surface: ResidualRankSurface, queries: Sequence[int], ranking: np.ndarray
) -> dict[str, Any]:
    rows = [ranking_metrics(surface, int(query), ranking[int(query)]) for query in queries]
    return {
        "queries": len(queries),
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in ("ndcg_at_10", "query_hit_at_20", "recall_at_20", "recall_at_100")
        },
    }


def residual_events(
    surface: ResidualRankSurface,
    queries: Sequence[int],
    *,
    rrf_constant: int,
    depth: int,
) -> dict[str, Any]:
    base = [
        rrf_ranking(
            surface,
            int(query),
            rrf_constant=rrf_constant,
            selected_omni_pages=None,
        )
        for query in queries
    ]
    events: list[tuple[int, set[int]]] = []
    strict_events: list[tuple[int, set[int]]] = []
    for query, base_ranking in zip(map(int, queries), base, strict=True):
        relevant = surface.qrels[query] > 0
        if np.any(relevant[base_ranking[:depth]]):
            continue
        repair_pages = {
            int(page)
            for page in surface.omni[query, :depth]
            if relevant[int(page)]
        }
        if not repair_pages:
            continue
        events.append((query, repair_pages))
        cheap_union = np.concatenate(
            (surface.bm25[query, :depth], surface.colsmol[query, :depth])
        )
        if not np.any(relevant[cheap_union]):
            strict_events.append((query, repair_pages))
    pages = set().union(*(pages for _, pages in events)) if events else set()
    strict_pages = (
        set().union(*(pages for _, pages in strict_events)) if strict_events else set()
    )
    return {
        "events": events,
        "strict_events": strict_events,
        "queries": len(events),
        "query_fraction": len(events) / len(queries) if len(queries) else 0.0,
        "unique_pages": pages,
        "unique_page_fraction": len(pages) / surface.pages,
        "page_events": sum(len(pages) for _, pages in events),
        "strict_candidate_escape_queries": len(strict_events),
        "strict_candidate_escape_unique_pages": strict_pages,
    }


def residual_utility(
    surface: ResidualRankSurface,
    queries: Sequence[int],
    *,
    rrf_constant: int,
    depth: int,
) -> np.ndarray:
    utility = np.zeros(surface.pages, dtype=np.float64)
    events = residual_events(
        surface, queries, rrf_constant=rrf_constant, depth=depth
    )["events"]
    for query, pages in events:
        ranks = {int(page): rank for rank, page in enumerate(surface.omni[query], start=1)}
        for page in pages:
            utility[page] += float(surface.qrels[query, page]) / math.log2(ranks[page] + 1)
    return utility


def global_label_rank_utility(
    surface: ResidualRankSurface,
    queries: Sequence[int],
    *,
    rrf_constant: int,
) -> np.ndarray:
    utility = np.zeros(surface.pages, dtype=np.float64)
    for query in map(int, queries):
        base = rrf_ranking(
            surface,
            query,
            rrf_constant=rrf_constant,
            selected_omni_pages=None,
        )
        base_rank = {int(page): rank for rank, page in enumerate(base, start=1)}
        for omni_rank, page in enumerate(surface.omni[query], start=1):
            relevance = float(surface.qrels[query, int(page)])
            if relevance <= 0:
                continue
            old_rank = base_rank.get(int(page), surface.pages + 1)
            gain = 1.0 / math.log2(omni_rank + 1) - 1.0 / math.log2(old_rank + 1)
            utility[int(page)] += relevance * max(0.0, gain)
    return utility


def omni_frequency_utility(
    surface: ResidualRankSurface, queries: Sequence[int]
) -> np.ndarray:
    utility = np.zeros(surface.pages, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, surface.omni.shape[1] + 2))
    for query in map(int, queries):
        utility[surface.omni[query]] += discounts
    return utility


def top_utility(utility: np.ndarray, count: int, *, positive_only: bool) -> np.ndarray:
    count = min(max(int(count), 0), len(utility))
    if count == 0:
        return np.empty(0, dtype=np.int32)
    pages = np.arange(len(utility), dtype=np.int32)
    order = pages[np.lexsort((pages, -utility))]
    if positive_only:
        order = order[utility[order] > 0]
    return np.sort(order[:count])


def gain_recovery(partial: float, base: float, full: float) -> float | None:
    denominator = full - base
    if denominator < 0.005:
        return None
    return (partial - base) / denominator


def projected_cost(
    selected_fraction: float,
    *,
    full_build_seconds: float,
    full_index_bytes: int,
    base_build_seconds: float,
    base_index_bytes: int,
) -> dict[str, Any]:
    incremental_seconds = selected_fraction * full_build_seconds
    incremental_bytes = selected_fraction * full_index_bytes
    return {
        "incremental_omni_build_seconds": incremental_seconds,
        "incremental_omni_index_bytes": incremental_bytes,
        "total_visual_build_seconds_including_colsmol": base_build_seconds
        + incremental_seconds,
        "total_visual_index_bytes_including_colsmol": base_index_bytes
        + incremental_bytes,
        "fraction_of_full_omni_build_seconds": selected_fraction,
        "fraction_of_full_omni_index_bytes": selected_fraction,
    }


def aggregate_runs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"runs": len(rows)}
    for key in (
        "selected_pages",
        "selected_page_fraction",
        "ndcg_at_10",
        "query_hit_at_20",
        "recall_at_20",
        "recall_at_100",
        "residual_repaired_fraction",
    ):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return result


def residual_repaired_fraction(
    surface: ResidualRankSurface,
    residual_queries: Sequence[int],
    rankings: Sequence[np.ndarray],
    *,
    depth: int,
) -> float:
    if not residual_queries:
        return 0.0
    repaired = 0
    for query, ranking in zip(residual_queries, rankings, strict=True):
        repaired += int(np.any(surface.qrels[int(query), ranking[:depth]] > 0))
    return repaired / len(residual_queries)


def reuse_diagnostics(
    surface: ResidualRankSurface,
    *,
    rrf_constant: int,
    depth: int,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    assignments = hash_folds(surface.query_ids, folds, seed)
    rows = []
    for fold in range(folds):
        history = np.flatnonzero(assignments != fold)
        future = np.flatnonzero(assignments == fold)
        history_events = residual_events(
            surface, history, rrf_constant=rrf_constant, depth=depth
        )["events"]
        future_events = residual_events(
            surface, future, rrf_constant=rrf_constant, depth=depth
        )["events"]
        history_pages = set().union(*(p for _, p in history_events)) if history_events else set()
        future_pages = set().union(*(p for _, p in future_events)) if future_events else set()
        future_page_events = sum(len(pages) for _, pages in future_events)
        repeated_events = sum(len(pages & history_pages) for _, pages in future_events)
        rows.append(
            {
                "fold": fold,
                "history_residual_pages": len(history_pages),
                "future_residual_queries": len(future_events),
                "future_unique_residual_pages": len(future_pages),
                "unique_page_overlap_fraction": (
                    len(history_pages & future_pages) / len(future_pages)
                    if future_pages
                    else None
                ),
                "event_overlap_fraction": (
                    repeated_events / future_page_events if future_page_events else None
                ),
            }
        )
    unique_weight = sum(row["future_unique_residual_pages"] for row in rows)
    event_weight = sum(
        residual_events(
            surface,
            np.flatnonzero(assignments == row["fold"]),
            rrf_constant=rrf_constant,
            depth=depth,
        )["page_events"]
        for row in rows
    )
    return {
        "folds": rows,
        "unique_page_overlap_fraction_weighted": (
            sum(
                float(row["unique_page_overlap_fraction"] or 0.0)
                * row["future_unique_residual_pages"]
                for row in rows
            )
            / unique_weight
            if unique_weight
            else None
        ),
        "event_overlap_fraction_weighted": (
            sum(
                float(row["event_overlap_fraction"] or 0.0)
                * residual_events(
                    surface,
                    np.flatnonzero(assignments == row["fold"]),
                    rrf_constant=rrf_constant,
                    depth=depth,
                )["page_events"]
                for row in rows
            )
            / event_weight
            if event_weight
            else None
        ),
    }


def auc(values: Sequence[float], labels: Sequence[bool]) -> float | None:
    values_array = np.asarray(values, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=bool)
    positives = int(np.sum(labels_array))
    negatives = len(labels_array) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(np.sum(ranks[labels_array]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
