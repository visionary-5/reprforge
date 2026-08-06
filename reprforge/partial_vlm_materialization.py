"""Score-surface audit for budgeted page-level VLM index materialization."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass
class ScoreSurface:
    name: str
    query_ids: np.ndarray
    corpus_ids: np.ndarray
    text_scores: np.ndarray
    visual_scores: np.ndarray
    qrels: np.ndarray
    text_bytes: np.ndarray
    visual_bytes: np.ndarray
    visual_encode_ms: np.ndarray
    input_sha256: dict[str, str]

    def __post_init__(self) -> None:
        shape = (len(self.query_ids), len(self.corpus_ids))
        for label, values in (
            ("text_scores", self.text_scores),
            ("visual_scores", self.visual_scores),
            ("qrels", self.qrels),
        ):
            if values.shape != shape:
                raise ValueError(f"{self.name}: {label} shape mismatch")
        for label, values in (
            ("text_bytes", self.text_bytes),
            ("visual_bytes", self.visual_bytes),
            ("visual_encode_ms", self.visual_encode_ms),
        ):
            if values.shape != (shape[1],):
                raise ValueError(f"{self.name}: {label} length mismatch")
        if len(set(map(str, self.query_ids))) != shape[0]:
            raise ValueError(f"{self.name}: duplicate query IDs")
        if len(set(map(str, self.corpus_ids))) != shape[1]:
            raise ValueError(f"{self.name}: duplicate corpus IDs")
        positions = np.arange(shape[1])
        self.text_order = np.asarray(
            [
                np.lexsort((positions, -np.asarray(row, dtype=np.float64)))
                for row in self.text_scores
            ],
            dtype=np.int32,
        )
        self.visual_order = np.asarray(
            [
                np.lexsort((positions, -np.asarray(row, dtype=np.float64)))
                for row in self.visual_scores
            ],
            dtype=np.int32,
        )
        self.idcg_at_10 = np.asarray(
            [_dcg(np.sort(row)[::-1][:10]) for row in self.qrels],
            dtype=np.float64,
        )

    @property
    def queries(self) -> int:
        return len(self.query_ids)

    @property
    def pages(self) -> int:
        return len(self.corpus_ids)


def _dcg(relevance: Sequence[float]) -> float:
    values = np.asarray(relevance, dtype=np.float64)
    if values.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, values.size + 2, dtype=np.float64))
    return float(np.sum((np.power(2.0, values) - 1.0) / discounts))


def _metrics_for_ranking(
    surface: ScoreSurface, query_position: int, ranking: np.ndarray
) -> tuple[float, float]:
    relevance = surface.qrels[query_position]
    dcg = _dcg(relevance[ranking[:10]])
    ideal = surface.idcg_at_10[query_position]
    ndcg = dcg / ideal if ideal > 0 else 0.0
    total_relevance = float(np.sum(relevance))
    recall = (
        float(np.sum(relevance[ranking[:100]])) / total_relevance
        if total_relevance > 0
        else 0.0
    )
    return ndcg, recall


def _rrf_ranking(
    surface: ScoreSurface,
    query_position: int,
    selected_mask: np.ndarray,
    *,
    text_top_k: int,
    visual_top_k: int,
    rrf_constant: int,
    include_text: bool = True,
    include_visual: bool = True,
    use_full_visual_rank_oracle: bool = False,
) -> np.ndarray:
    contributions: dict[int, float] = {}
    if include_text:
        for rank, page in enumerate(
            surface.text_order[query_position, :text_top_k], start=1
        ):
            contributions[int(page)] = 1.0 / (rrf_constant + rank)
    if include_visual and np.any(selected_mask):
        if use_full_visual_rank_oracle:
            ranked_pairs = [
                (rank, page)
                for rank, page in enumerate(
                    surface.visual_order[query_position, :visual_top_k], start=1
                )
                if selected_mask[page]
            ]
        else:
            visual_pages = surface.visual_order[query_position]
            visual_pages = visual_pages[selected_mask[visual_pages]][:visual_top_k]
            ranked_pairs = list(enumerate(visual_pages, start=1))
        for rank, page in ranked_pairs:
            key = int(page)
            contributions[key] = contributions.get(key, 0.0) + 1.0 / (
                rrf_constant + rank
            )
    if not contributions:
        return np.arange(surface.pages, dtype=np.int32)
    pages = np.fromiter(contributions, dtype=np.int32)
    scores = np.fromiter((contributions[int(page)] for page in pages), dtype=np.float64)
    order = np.lexsort((pages, -scores))
    ranked = pages[order]
    if len(ranked) < 100:
        missing = np.setdiff1d(
            np.arange(surface.pages, dtype=np.int32), ranked, assume_unique=False
        )
        ranked = np.concatenate((ranked, missing))
    return ranked


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    deviation = float(values.std())
    if deviation <= 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / deviation


def _selected_zscore_ranking(
    surface: ScoreSurface,
    query_position: int,
    selected_mask: np.ndarray,
    *,
    use_full_visual_calibration_oracle: bool = False,
) -> np.ndarray:
    scores = _zscore(surface.text_scores[query_position])
    selected = np.flatnonzero(selected_mask)
    if selected.size:
        if use_full_visual_calibration_oracle:
            visual = _zscore(surface.visual_scores[query_position])
            scores[selected] += visual[selected]
        else:
            scores[selected] += _zscore(surface.visual_scores[query_position, selected])
    positions = np.arange(surface.pages)
    return np.lexsort((positions, -scores))


def evaluate_selection(
    surface: ScoreSurface,
    query_positions: Sequence[int],
    selected_pages: Iterable[int],
    *,
    fusion: str,
    text_top_k: int = 100,
    visual_top_k: int = 100,
    rrf_constant: int = 60,
) -> dict[str, Any]:
    selected = np.asarray(sorted(set(map(int, selected_pages))), dtype=np.int32)
    if selected.size and (selected.min() < 0 or selected.max() >= surface.pages):
        raise ValueError("selected page outside corpus")
    selected_mask = np.zeros(surface.pages, dtype=bool)
    selected_mask[selected] = True
    ndcg: list[float] = []
    recall: list[float] = []
    for query_position in query_positions:
        if fusion in ("rrf", "rrf_global_oracle"):
            ranking = _rrf_ranking(
                surface,
                int(query_position),
                selected_mask,
                text_top_k=text_top_k,
                visual_top_k=visual_top_k,
                rrf_constant=rrf_constant,
                use_full_visual_rank_oracle=fusion == "rrf_global_oracle",
            )
        elif fusion in ("zscore", "zscore_global_oracle"):
            ranking = _selected_zscore_ranking(
                surface,
                int(query_position),
                selected_mask,
                use_full_visual_calibration_oracle=fusion == "zscore_global_oracle",
            )
        else:
            raise ValueError(f"unsupported fusion: {fusion}")
        query_ndcg, query_recall = _metrics_for_ranking(
            surface, int(query_position), ranking
        )
        ndcg.append(query_ndcg)
        recall.append(query_recall)
    total_visual_bytes = float(np.sum(surface.visual_bytes))
    total_encode_ms = float(np.sum(surface.visual_encode_ms))
    return {
        "queries": len(query_positions),
        "selected_pages": int(selected.size),
        "selected_page_fraction": selected.size / surface.pages,
        "selected_visual_bytes": int(np.sum(surface.visual_bytes[selected])),
        "selected_visual_byte_fraction": (
            float(np.sum(surface.visual_bytes[selected])) / total_visual_bytes
            if total_visual_bytes > 0
            else None
        ),
        "selected_encode_ms": float(np.sum(surface.visual_encode_ms[selected])),
        "selected_encode_fraction": (
            float(np.sum(surface.visual_encode_ms[selected])) / total_encode_ms
            if total_encode_ms > 0
            else None
        ),
        "mean_ndcg_at_10": float(np.mean(ndcg)),
        "mean_recall_at_100": float(np.mean(recall)),
        "per_query_ndcg_at_10": ndcg,
        "per_query_recall_at_100": recall,
    }


def evaluate_text_only(
    surface: ScoreSurface, query_positions: Sequence[int]
) -> dict[str, Any]:
    ndcg, recall = [], []
    for query_position in query_positions:
        ranking = surface.text_order[int(query_position)]
        query_ndcg, query_recall = _metrics_for_ranking(
            surface, int(query_position), ranking
        )
        ndcg.append(query_ndcg)
        recall.append(query_recall)
    return {
        "mean_ndcg_at_10": float(np.mean(ndcg)),
        "mean_recall_at_100": float(np.mean(recall)),
        "per_query_ndcg_at_10": ndcg,
        "per_query_recall_at_100": recall,
    }


def evaluate_visual_only(
    surface: ScoreSurface, query_positions: Sequence[int]
) -> dict[str, Any]:
    ndcg, recall = [], []
    for query_position in query_positions:
        ranking = surface.visual_order[int(query_position)]
        query_ndcg, query_recall = _metrics_for_ranking(
            surface, int(query_position), ranking
        )
        ndcg.append(query_ndcg)
        recall.append(query_recall)
    return {
        "mean_ndcg_at_10": float(np.mean(ndcg)),
        "mean_recall_at_100": float(np.mean(recall)),
        "per_query_ndcg_at_10": ndcg,
        "per_query_recall_at_100": recall,
    }


def fold_assignments(surface: ScoreSurface, folds: int, seed: int) -> np.ndarray:
    if folds < 2:
        raise ValueError("at least two folds are required")
    output = []
    for query_id in surface.query_ids:
        payload = f"{query_id}\0{seed}".encode("utf-8")
        output.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds)
    values = np.asarray(output, dtype=np.int16)
    if len(set(values.tolist())) != folds:
        raise ValueError("query hash assignment produced an empty fold")
    return values


def _discounted_frequency(order: np.ndarray, query_positions: Sequence[int], k: int) -> np.ndarray:
    utility = np.zeros(order.shape[1], dtype=np.float64)
    weights = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    for query_position in query_positions:
        pages = order[int(query_position), :k]
        utility[pages] += weights
    return utility


def _top_utility(utility: np.ndarray, count: int, *, positive_only: bool = False) -> np.ndarray:
    positions = np.arange(len(utility))
    order = np.lexsort((positions, -utility))
    if positive_only:
        order = order[utility[order] > 0.0]
    return np.asarray(order[:count], dtype=np.int32)


def relevance_reuse_crossfit(
    surface: ScoreSurface, assignments: np.ndarray
) -> dict[str, Any]:
    """Measure whether future relevant evidence was observed in historical feedback.

    This is an audit diagnostic, not an input to a deployment policy.  Each fold is
    treated as the future and all other folds as history.
    """
    rows = []
    for fold in sorted(set(map(int, assignments))):
        history = np.flatnonzero(assignments != fold)
        future = np.flatnonzero(assignments == fold)
        history_pages = set(
            map(int, np.flatnonzero(np.any(surface.qrels[history] > 0, axis=0)))
        )
        future_relevant = surface.qrels[future] > 0
        future_pages = set(
            map(int, np.flatnonzero(np.any(future_relevant, axis=0)))
        )
        future_events = int(np.sum(future_relevant))
        repeated_events = int(np.sum(future_relevant[:, sorted(history_pages)]))
        rows.append(
            {
                "fold": fold,
                "history_relevant_pages": len(history_pages),
                "future_unique_relevant_pages": len(future_pages),
                "future_relevant_page_events": future_events,
                "unique_page_overlap_fraction": (
                    len(history_pages & future_pages) / len(future_pages)
                    if future_pages
                    else None
                ),
                "event_overlap_fraction": (
                    repeated_events / future_events if future_events else None
                ),
            }
        )
    future_unique = sum(row["future_unique_relevant_pages"] for row in rows)
    future_events = sum(row["future_relevant_page_events"] for row in rows)
    return {
        "folds": rows,
        "unique_page_overlap_fraction_weighted": (
            sum(
                row["unique_page_overlap_fraction"]
                * row["future_unique_relevant_pages"]
                for row in rows
                if row["unique_page_overlap_fraction"] is not None
            )
            / future_unique
            if future_unique
            else None
        ),
        "event_overlap_fraction_weighted": (
            sum(
                row["event_overlap_fraction"] * row["future_relevant_page_events"]
                for row in rows
                if row["event_overlap_fraction"] is not None
            )
            / future_events
            if future_events
            else None
        ),
    }


def select_pages(
    surface: ScoreSurface,
    *,
    policy: str,
    count: int,
    history_queries: Sequence[int],
    future_queries: Sequence[int],
    seed: int,
) -> np.ndarray:
    count = min(max(int(count), 0), surface.pages)
    if count == 0:
        return np.empty(0, dtype=np.int32)
    if policy == "random":
        return np.sort(
            np.random.default_rng(seed).choice(surface.pages, count, replace=False)
        ).astype(np.int32)
    if policy == "corpus_uniform":
        return np.unique(
            np.linspace(0, surface.pages - 1, count, dtype=np.int32)
        )[:count]
    history_frequency = _discounted_frequency(
        surface.text_order, history_queries, min(100, surface.pages)
    )
    if policy == "history_frequency":
        return _top_utility(history_frequency, count)
    if policy in ("history_relevance", "history_relevance_cover25"):
        history_relevance = np.sum(
            surface.qrels[np.asarray(history_queries, dtype=np.int32)], axis=0
        ).astype(np.float64)
        if policy == "history_relevance":
            selected = list(
                map(
                    int,
                    _top_utility(history_relevance, count, positive_only=True),
                )
            )
            selected_set = set(selected)
            if len(selected_set) >= count:
                return np.asarray(sorted(selected_set), dtype=np.int32)
            for page in _top_utility(history_frequency, surface.pages):
                selected_set.add(int(page))
                if len(selected_set) >= count:
                    break
            return np.asarray(sorted(selected_set), dtype=np.int32)
    if np.ptp(surface.text_bytes) > 0:
        positions = np.arange(surface.pages)
        risk_order = np.lexsort((positions, surface.text_bytes))
    else:
        positions = np.arange(surface.pages)
        risk_order = np.lexsort((positions, history_frequency))
    if policy == "text_risk":
        return np.asarray(risk_order[:count], dtype=np.int32)
    if policy == "cover25_frequency75":
        cover_count = max(1, int(math.ceil(0.25 * count)))
        cover = list(map(int, risk_order[:cover_count]))
        selected = set(cover)
        for page in _top_utility(history_frequency, surface.pages):
            selected.add(int(page))
            if len(selected) >= count:
                break
        return np.asarray(sorted(selected), dtype=np.int32)
    if policy == "history_relevance_cover25":
        cover_count = max(1, int(math.ceil(0.25 * count)))
        selected = set(map(int, risk_order[:cover_count]))
        history_relevance = np.sum(
            surface.qrels[np.asarray(history_queries, dtype=np.int32)], axis=0
        ).astype(np.float64)
        for utility in (history_relevance, history_frequency):
            for page in _top_utility(utility, surface.pages, positive_only=True):
                selected.add(int(page))
                if len(selected) >= count:
                    break
            if len(selected) >= count:
                break
        if len(selected) < count:
            for page in _top_utility(history_frequency, surface.pages):
                selected.add(int(page))
                if len(selected) >= count:
                    break
        return np.asarray(sorted(selected), dtype=np.int32)
    if policy == "score_oracle":
        utility = _discounted_frequency(
            surface.visual_order, future_queries, min(100, surface.pages)
        )
        return _top_utility(utility, count)
    if policy == "label_rank_oracle":
        utility = np.zeros(surface.pages, dtype=np.float64)
        text_ranks = np.empty(surface.pages, dtype=np.int32)
        visual_ranks = np.empty(surface.pages, dtype=np.int32)
        for query_position in future_queries:
            text_ranks[surface.text_order[int(query_position)]] = np.arange(
                1, surface.pages + 1
            )
            visual_ranks[surface.visual_order[int(query_position)]] = np.arange(
                1, surface.pages + 1
            )
            relevant = np.flatnonzero(surface.qrels[int(query_position)] > 0)
            for page in relevant:
                text_discount = 1.0 / math.log2(int(text_ranks[page]) + 1)
                visual_discount = 1.0 / math.log2(int(visual_ranks[page]) + 1)
                utility[page] += float(surface.qrels[int(query_position), page]) * max(
                    0.0, visual_discount - text_discount
                )
        return _top_utility(utility, count, positive_only=True)
    raise ValueError(f"unsupported selection policy: {policy}")


def gain_recovery(partial: float, text: float, full_hybrid: float) -> dict[str, Any]:
    denominator = full_hybrid - text
    return {
        "full_hybrid_minus_text": denominator,
        "denominator_sign": (
            "positive" if denominator > 0 else "negative" if denominator < 0 else "zero"
        ),
        "gain_recovery": (
            (partial - text) / denominator if abs(denominator) >= 0.005 else None
        ),
        "omitted_because_absolute_denominator_below_0_005": abs(denominator) < 0.005,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"runs": len(rows)}
    for key in (
        "selected_page_fraction",
        "selected_visual_byte_fraction",
        "selected_encode_fraction",
        "mean_ndcg_at_10",
        "mean_recall_at_100",
    ):
        values = np.asarray(
            [row[key] for row in rows if row.get(key) is not None], dtype=np.float64
        )
        valid_rows = [row for row in rows if row.get(key) is not None]
        weights = np.asarray(
            [row.get("queries", 1) for row in valid_rows], dtype=np.float64
        )
        output[key] = {
            "mean": float(values.mean()),
            "query_weighted_mean": float(np.average(values, weights=weights)),
            "standard_deviation": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return output


def online_trace_audit(
    surface: ScoreSurface,
    query_order: Sequence[int],
    *,
    scope_top_k: int,
    text_top_k: int,
    visual_top_k: int,
    rrf_constant: int,
) -> dict[str, Any]:
    persistent_mask = np.zeros(surface.pages, dtype=bool)
    persistent_ndcg, persistent_recall = [], []
    nonpersistent_ndcg, nonpersistent_recall = [], []
    page_events = 0
    for query_position in query_order:
        scope = surface.text_order[int(query_position), :scope_top_k]
        page_events += len(scope)
        transient_mask = np.zeros(surface.pages, dtype=bool)
        transient_mask[scope] = True
        transient_rank = _rrf_ranking(
            surface,
            int(query_position),
            transient_mask,
            text_top_k=text_top_k,
            visual_top_k=visual_top_k,
            rrf_constant=rrf_constant,
        )
        ndcg, recall = _metrics_for_ranking(
            surface, int(query_position), transient_rank
        )
        nonpersistent_ndcg.append(ndcg)
        nonpersistent_recall.append(recall)
        persistent_mask[scope] = True
        persistent_rank = _rrf_ranking(
            surface,
            int(query_position),
            persistent_mask,
            text_top_k=text_top_k,
            visual_top_k=visual_top_k,
            rrf_constant=rrf_constant,
        )
        ndcg, recall = _metrics_for_ranking(
            surface, int(query_position), persistent_rank
        )
        persistent_ndcg.append(ndcg)
        persistent_recall.append(recall)
    unique_pages = int(np.sum(persistent_mask))
    return {
        "queries": len(query_order),
        "scope_top_k": scope_top_k,
        "nonpersistent": {
            "visual_page_events": page_events,
            "mean_ndcg_at_10": float(np.mean(nonpersistent_ndcg)),
            "mean_recall_at_100": float(np.mean(nonpersistent_recall)),
        },
        "persistent": {
            "unique_visual_pages_materialized": unique_pages,
            "final_materialized_fraction": unique_pages / surface.pages,
            "mean_ndcg_at_10": float(np.mean(persistent_ndcg)),
            "mean_recall_at_100": float(np.mean(persistent_recall)),
        },
        "amortization": {
            "page_event_reuse_fraction": (
                1.0 - unique_pages / page_events if page_events else 0.0
            ),
            "nonpersistent_over_persistent_construction_events": (
                page_events / unique_pages if unique_pages else None
            ),
        },
    }
