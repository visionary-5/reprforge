#!/usr/bin/env python3
"""Query-cohort compiler for sparse-first visual document retrieval.

The compiler is deliberately synchronous at the request-batch boundary.  It
collects BM25 candidate cohorts, deduplicates their visual misses, encodes the
union once, and publishes a new in-memory resident generation only after the
whole batch succeeds.  This isolates the measurable batching mechanism before
introducing background workers or queueing policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from reprforge.bm25 import build_index, score_queries
from reprforge.mmdocir_route_runner import EncodedBatch


class CohortBackend(Protocol):
    def encode_queries(self, queries: Sequence[str]) -> EncodedBatch: ...

    def encode_images(self, images: Sequence[Any]) -> EncodedBatch: ...

    def score(
        self,
        queries: Sequence[Any],
        documents: Sequence[Any],
    ) -> Sequence[Sequence[float]]: ...


def _embedding_bytes(embedding: Any) -> int:
    element_size = getattr(embedding, "element_size", None)
    numel = getattr(embedding, "numel", None)
    if callable(element_size) and callable(numel):
        return int(element_size() * numel())
    return int(np.asarray(embedding).nbytes)


def _zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - array.mean()
    return centered / max(float(array.std()), 1e-12)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


@dataclass(frozen=True)
class CohortExecution:
    results: dict[str, dict[str, float]]
    metrics: dict[str, Any]


class CohortCompiler:
    """Compile and execute candidate-relative visual cohorts."""

    def __init__(
        self,
        *,
        corpus_ids: Sequence[str],
        corpus_texts: Sequence[str],
        corpus_images: Sequence[Any],
        backend: CohortBackend,
        candidate_k: int = 20,
        top_k: int = 100,
        request_batch_size: int = 1,
        cache_policy: str = "resident",
    ) -> None:
        if (
            len(corpus_ids) == 0
            or len(corpus_ids) != len(corpus_texts)
            or len(corpus_ids) != len(corpus_images)
        ):
            raise ValueError("cohort compiler corpus columns must be aligned")
        if len(set(corpus_ids)) != len(corpus_ids):
            raise ValueError("cohort compiler corpus identifiers must be unique")
        if candidate_k <= 0 or top_k <= 0 or request_batch_size <= 0:
            raise ValueError(
                "candidate_k, top_k and request_batch_size must be positive"
            )
        if cache_policy not in {"none", "resident"}:
            raise ValueError("cache_policy must be 'none' or 'resident'")
        self.corpus_ids = tuple(str(value) for value in corpus_ids)
        self.corpus_texts = tuple(str(value) for value in corpus_texts)
        self.corpus_images = tuple(corpus_images)
        self.backend = backend
        self.candidate_k = min(candidate_k, len(self.corpus_ids))
        self.top_k = min(top_k, len(self.corpus_ids))
        self.request_batch_size = request_batch_size
        self.cache_policy = cache_policy
        began = time.perf_counter()
        self._bm25_state, posting_bytes, vocabulary_bytes = build_index(
            self.corpus_texts
        )
        self.index_build_ms = (time.perf_counter() - began) * 1000.0
        self.logical_bm25_bytes = int(posting_bytes.sum()) + vocabulary_bytes
        self._resident: dict[str, Any] = {}
        self._generation = 0

    @property
    def resident_item_ids(self) -> frozenset[str]:
        return frozenset(self._resident)

    @property
    def generation(self) -> int:
        return self._generation

    def _candidate_positions(self, scores: np.ndarray) -> list[int]:
        return sorted(
            range(len(self.corpus_ids)),
            key=lambda position: (-float(scores[position]), self.corpus_ids[position]),
        )[: self.candidate_k]

    def _fused_result(
        self,
        locator_scores: np.ndarray,
        candidate_positions: Sequence[int],
        visual_scores: Sequence[float],
    ) -> dict[str, float]:
        if len(candidate_positions) != len(visual_scores):
            raise ValueError("candidate and visual scores differ in length")
        candidate_ids = [
            self.corpus_ids[position] for position in candidate_positions
        ]
        fused = _zscore(locator_scores[list(candidate_positions)]) + _zscore(
            visual_scores
        )
        candidate_order = sorted(
            range(len(candidate_ids)),
            key=lambda offset: (-float(fused[offset]), candidate_ids[offset]),
        )
        selected = set(candidate_positions)
        tail = sorted(
            (
                position
                for position in range(len(self.corpus_ids))
                if position not in selected
            ),
            key=lambda position: (
                -float(locator_scores[position]),
                self.corpus_ids[position],
            ),
        )
        ordered_ids = [candidate_ids[offset] for offset in candidate_order]
        ordered_ids.extend(self.corpus_ids[position] for position in tail)
        count = min(self.top_k, len(ordered_ids))
        # pytrec_eval consumes scores rather than insertion order.  Synthetic
        # monotonic scores faithfully encode the heterogeneous ranking without
        # pretending BM25 and normalized fusion share a global score scale.
        return {
            item_id: float(count - rank)
            for rank, item_id in enumerate(ordered_ids[:count])
        }

    def execute_batch(
        self,
        query_ids: Sequence[str],
        query_texts: Sequence[str],
    ) -> CohortExecution:
        if len(query_ids) != len(query_texts):
            raise ValueError("query identifiers and texts differ in length")
        if len(query_ids) == 0:
            raise ValueError("query batch cannot be empty")
        if len(set(str(value) for value in query_ids)) != len(query_ids):
            raise ValueError("query identifiers must be unique")

        execution_began = time.perf_counter()
        results: dict[str, dict[str, float]] = {}
        candidate_events = 0
        unique_candidates = 0
        cache_hit_events = 0
        encoded_pages = 0
        encoder_calls = 0
        visual_score_pairs = 0
        batch_latencies: list[float] = []
        batch_trace: list[dict[str, Any]] = []
        encode_latencies: list[float] = []
        locator_score_ms = 0.0
        query_encode_ms = 0.0
        visual_score_ms = 0.0
        initial_residents = len(self._resident)

        for start in range(0, len(query_ids), self.request_batch_size):
            stop = start + self.request_batch_size
            batch_ids = [str(value) for value in query_ids[start:stop]]
            batch_texts = list(query_texts[start:stop])
            batch_began = time.perf_counter()

            began = time.perf_counter()
            locator = score_queries(
                self._bm25_state,
                batch_texts,
                k1=1.2,
                b=0.75,
            )
            locator_score_ms += (time.perf_counter() - began) * 1000.0
            cohorts = [self._candidate_positions(row) for row in locator]
            batch_candidate_events = sum(len(value) for value in cohorts)
            candidate_events += batch_candidate_events
            requested_positions: list[int] = []
            requested_set: set[int] = set()
            for cohort in cohorts:
                for position in cohort:
                    if position not in requested_set:
                        requested_set.add(position)
                        requested_positions.append(position)
            unique_candidates += len(requested_positions)

            resident_at_start = self._resident
            batch_cache_hits = sum(
                self.corpus_ids[position] in resident_at_start
                for cohort in cohorts
                for position in cohort
            )
            cache_hit_events += batch_cache_hits
            missing_positions = [
                position
                for position in requested_positions
                if self.corpus_ids[position] not in resident_at_start
            ]
            staged: dict[str, Any] = {}
            encode_call_ms = 0.0
            if missing_positions:
                began = time.perf_counter()
                encoded = self.backend.encode_images(
                    [
                        self.corpus_images[position]
                        for position in missing_positions
                    ]
                )
                encode_call_ms = (time.perf_counter() - began) * 1000.0
                encode_latencies.append(encode_call_ms)
                if len(encoded.embeddings) != len(missing_positions):
                    raise RuntimeError("visual backend returned an incomplete batch")
                staged = {
                    self.corpus_ids[position]: embedding
                    for position, embedding in zip(
                        missing_positions,
                        encoded.embeddings,
                        strict=True,
                    )
                }
                encoded_pages += len(missing_positions)
                encoder_calls += 1

            began = time.perf_counter()
            encoded_queries = self.backend.encode_queries(batch_texts)
            query_call_ms = (time.perf_counter() - began) * 1000.0
            query_encode_ms += query_call_ms
            visible: Mapping[str, Any] = {**resident_at_start, **staged}
            union_embeddings = [
                visible[self.corpus_ids[position]]
                for position in requested_positions
            ]
            union_offsets = {
                position: offset
                for offset, position in enumerate(requested_positions)
            }
            began = time.perf_counter()
            union_scores = self.backend.score(
                encoded_queries.embeddings,
                union_embeddings,
            )
            score_call_ms = (time.perf_counter() - began) * 1000.0
            visual_score_ms += score_call_ms
            if len(union_scores) != len(batch_ids):
                raise RuntimeError("visual backend returned incomplete query scores")
            staged_results: dict[str, dict[str, float]] = {}
            for query_id, score_row, locator_row, cohort in zip(
                batch_ids,
                union_scores,
                locator,
                cohorts,
                strict=True,
            ):
                if len(score_row) != len(requested_positions):
                    raise RuntimeError(
                        "visual backend returned incomplete document scores"
                    )
                visual_scores = [
                    float(score_row[union_offsets[position]])
                    for position in cohort
                ]
                visual_score_pairs += len(visual_scores)
                staged_results[query_id] = self._fused_result(
                    locator_row,
                    cohort,
                    visual_scores,
                )

            # Publish only after encoding and every query score succeeds.
            if self.cache_policy == "resident" and staged:
                self._resident = {**self._resident, **staged}
                self._generation += 1
            results.update(staged_results)
            batch_completion_ms = (
                time.perf_counter() - batch_began
            ) * 1000.0
            batch_latencies.append(batch_completion_ms)
            batch_trace.append(
                {
                    "query_offset_start": start,
                    "query_count": len(batch_ids),
                    "candidate_events": batch_candidate_events,
                    "unique_candidates": len(requested_positions),
                    "cache_hit_events": batch_cache_hits,
                    "visual_pages_encoded": len(missing_positions),
                    "visual_encode_ms": encode_call_ms,
                    "query_encode_ms": query_call_ms,
                    "visual_score_ms": score_call_ms,
                    "completion_ms": batch_completion_ms,
                    "resident_items_after_publish": len(self._resident),
                    "generation_after_publish": self._generation,
                }
            )

        resident_bytes = sum(
            _embedding_bytes(value) for value in self._resident.values()
        )
        metrics = {
            "query_count": len(query_ids),
            "candidate_k": self.candidate_k,
            "top_k": self.top_k,
            "request_batch_size": self.request_batch_size,
            "cache_policy": self.cache_policy,
            "candidate_events": candidate_events,
            "unique_candidates_within_batches": unique_candidates,
            "within_batch_deduplicated_events": candidate_events - unique_candidates,
            "within_batch_dedup_fraction": (
                1.0 - unique_candidates / candidate_events
                if candidate_events
                else 0.0
            ),
            "cache_hit_events": cache_hit_events,
            "cache_hit_fraction": (
                cache_hit_events / candidate_events if candidate_events else 0.0
            ),
            "visual_pages_encoded": encoded_pages,
            "visual_encoder_calls": encoder_calls,
            "visual_score_pairs": visual_score_pairs,
            "locator_score_ms": locator_score_ms,
            "query_encode_ms": query_encode_ms,
            "visual_score_ms": visual_score_ms,
            "total_execution_ms": (
                time.perf_counter() - execution_began
            ) * 1000.0,
            "batch_completion_ms": _percentiles(batch_latencies),
            "batch_trace": batch_trace,
            "visual_encode_call_ms": _percentiles(encode_latencies),
            "initial_resident_items": initial_residents,
            "current_resident_items": len(self._resident),
            "current_resident_vector_bytes": resident_bytes,
            "active_generation": self._generation,
            "atomic_publish": True,
            "logical_visual_activation_is_query_scoped": True,
        }
        return CohortExecution(results=results, metrics=metrics)
