#!/usr/bin/env python3
"""Official ViDoRe v3 pipeline adapter for ReprForge.

The official benchmark calls ``index`` once with page images and Markdown,
then calls ``search`` once with a query stream.  This adapter deliberately
keeps representation construction and retrieval separate so the benchmark's
indexing/search timings retain their intended meaning.

The module can be imported without ``vidore_benchmark`` for local unit tests.
When loaded by the official CLI, ``ReprForgeViDoRePipeline`` inherits the real
``BasePipeline`` and therefore passes its subclass check.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

try:
    from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline
except ImportError:  # pragma: no cover - exercised only without the optional extra
    class BasePipeline:  # type: ignore[no-redef]
        """Small import fallback; the official CLI supplies the real class."""

        def index(
            self,
            corpus_ids: list[str],
            corpus_images: list[Any],
            corpus_texts: list[str],
            dataset_name: str | None = None,
        ) -> None:
            self.corpus_ids = corpus_ids
            self.corpus_images = corpus_images
            self.corpus_texts = corpus_texts


from reprforge.cohort_compiler import CohortCompiler
from reprforge.mmdocir_route_runner import ColPaliBackend, EncodedBatch


MODES = {
    "text",
    "visual",
    "visual-pool",
    "two-stage",
    "tiered-selective",
    "bm25-fusion-sync",
    "bm25-fusion-batched",
}


class RepresentationBackend(Protocol):
    """Minimal model contract needed by the official pipeline adapter."""

    def encode_queries(self, queries: Sequence[str]) -> EncodedBatch: ...

    def encode_texts(self, texts: Sequence[str]) -> EncodedBatch: ...

    def encode_images(self, images: Sequence[Any]) -> EncodedBatch: ...

    def derive_image_routes(
        self,
        images: EncodedBatch,
    ) -> Mapping[str, EncodedBatch]: ...

    def score(
        self,
        queries: Sequence[Any],
        documents: Sequence[Any],
    ) -> Sequence[Sequence[float]]: ...

    def environment(self) -> Mapping[str, Any]: ...


def _embedding_bytes(embedding: Any) -> int:
    element_size = getattr(embedding, "element_size", None)
    numel = getattr(embedding, "numel", None)
    if callable(element_size) and callable(numel):
        return int(element_size() * numel())
    array = np.asarray(embedding)
    return int(array.nbytes)


def _batch_bytes(batch: EncodedBatch) -> int:
    return sum(_embedding_bytes(value) for value in batch.embeddings)


def _rank(
    item_ids: Sequence[str],
    scores: Sequence[float],
    *,
    top_k: int,
) -> dict[str, float]:
    if len(item_ids) != len(scores):
        raise ValueError("item identifiers and scores differ in length")
    count = min(top_k, len(item_ids))
    order = sorted(
        range(len(item_ids)),
        key=lambda index: (-float(scores[index]), item_ids[index]),
    )[:count]
    return {item_ids[index]: float(scores[index]) for index in order}


class ReprForgeViDoRePipeline(BasePipeline):
    """Representation-lifecycle policies under the official ViDoRe contract.

    Modes:

    ``text``
        Build and search a Markdown-only late-interaction index.
    ``visual``
        Build and search a full visual late-interaction index.
    ``visual-pool``
        Encode every page visually, then apply published hierarchical token
        pooling.  This saves index bytes and MaxSim work, not visual encoding.
    ``two-stage``
        Build the text index, select ``candidate_k`` pages per query, encode
        those images on demand, and rank only the selected pages by full
        visual score.  It intentionally has no cross-query cache.
    ``tiered-selective``
        Build the text index, select pages by text score, materialize missing
        visual representations into an optional LRU cache, and replace text
        scores only for pages selected by the current query.  Physical cache
        residency is therefore distinct from logical score activation.
    ``bm25-fusion-sync``
        Use BM25 to form one query cohort at a time, encode its visual pages,
        and rank by candidate-relative normalized BM25/visual fusion.  This is
        the no-reuse execution baseline for the online algorithm.
    ``bm25-fusion-batched``
        Compile several BM25 cohorts together, encode their deduplicated union
        once, score the query--union matrix once, and optionally retain visual
        representations across request batches.  Only the current query's
        cohort is logically activated in its ranking.
    """

    def __init__(
        self,
        *,
        base_model: str,
        adapter: str,
        mode: str = "tiered-selective",
        device: str = "cuda",
        batch_size: int = 4,
        scoring_batch_size: int = 16,
        candidate_k: int = 20,
        top_k: int = 100,
        image_pool_factor: int = 25,
        cache_capacity_items: int = 0,
        request_batch_size: int = 8,
        cohort_cache_policy: str = "resident",
        admitted_item_ids: Sequence[str] | None = None,
        visual_prior_by_rank: Sequence[float] | None = None,
        capture_score_trace: bool = False,
        backend_factory: Callable[[], RepresentationBackend] | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unsupported mode {mode!r}; choose from {sorted(MODES)}")
        if (
            batch_size <= 0
            or scoring_batch_size <= 0
            or candidate_k <= 0
            or top_k <= 0
            or request_batch_size <= 0
        ):
            raise ValueError(
                "batch_size, scoring_batch_size, candidate_k, top_k, and "
                "request_batch_size must be positive"
            )
        if cohort_cache_policy not in {"none", "resident"}:
            raise ValueError("cohort_cache_policy must be 'none' or 'resident'")
        if admitted_item_ids is not None and mode not in {
            "bm25-fusion-sync",
            "bm25-fusion-batched",
        }:
            raise ValueError("representation admission requires a BM25 fusion mode")
        if image_pool_factor < 2:
            raise ValueError("image_pool_factor must be at least 2")
        if cache_capacity_items < 0:
            raise ValueError("cache_capacity_items cannot be negative")
        if (
            mode == "tiered-selective"
            and cache_capacity_items
            and cache_capacity_items < candidate_k
        ):
            raise ValueError(
                "tiered-selective cache capacity must hold one candidate set"
            )
        if mode == "two-stage" and candidate_k < top_k:
            raise ValueError("two-stage candidate_k must be at least top_k")

        self.mode = mode
        self.device = device
        self.batch_size = batch_size
        self.scoring_batch_size = scoring_batch_size
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.image_pool_factor = image_pool_factor
        self.capture_score_trace = capture_score_trace
        self.request_batch_size = request_batch_size
        self.cohort_cache_policy = cohort_cache_policy
        self.admitted_item_ids = (
            None
            if admitted_item_ids is None
            else tuple(str(value) for value in admitted_item_ids)
        )
        self.visual_prior_by_rank = (
            None
            if visual_prior_by_rank is None
            else tuple(float(value) for value in visual_prior_by_rank)
        )
        # Zero means an unbounded cache for tiered-selective, not no cache.
        self.cache_capacity_items = cache_capacity_items

        began = time.perf_counter()
        self.backend = (
            backend_factory()
            if backend_factory is not None
            else ColPaliBackend(
                base_model=Path(base_model),
                adapter=Path(adapter),
                device=device,
                batch_size=batch_size,
                scoring_batch_size=scoring_batch_size,
                image_pool_factors=(
                    (image_pool_factor,) if mode == "visual-pool" else ()
                ),
            )
        )
        self.model_load_ms = (time.perf_counter() - began) * 1000.0

        self.corpus_ids: list[str] = []
        self.corpus_images: list[Any] = []
        self.corpus_texts: list[str] = []
        self._positions: dict[str, int] = {}
        self._base: EncodedBatch | None = None
        self._visual: EncodedBatch | None = None
        self._visual_cache: OrderedDict[str, Any] = OrderedDict()
        self._cohort_compiler: CohortCompiler | None = None
        self._index_info: dict[str, Any] = {}
        self._search_info: dict[str, Any] = {}
        self._last_query_ids: tuple[str, ...] = ()
        self._last_score_matrix: tuple[tuple[float, ...], ...] = ()

    def _encode_images(self, positions: Sequence[int]) -> EncodedBatch:
        # Official ViDoRe already supplies decoded PIL pages. Pass them through
        # directly: a previous adapter encoded each page as PNG only for the
        # backend to decode it again, adding substantial non-model build cost.
        return self.backend.encode_images(
            [self.corpus_images[position] for position in positions]
        )

    def index(
        self,
        corpus_ids: list[str],
        corpus_images: list[Any],
        corpus_texts: list[str],
        dataset_name: str | None = None,
    ) -> None:
        if not (
            len(corpus_ids) == len(corpus_images) == len(corpus_texts)
            and corpus_ids
        ):
            raise ValueError("ViDoRe corpus columns must be non-empty and aligned")
        if len(set(corpus_ids)) != len(corpus_ids):
            raise ValueError("ViDoRe corpus identifiers must be unique")

        self.corpus_ids = [str(value) for value in corpus_ids]
        self.corpus_images = list(corpus_images)
        self.corpus_texts = [str(value) for value in corpus_texts]
        self._positions = {
            item_id: position for position, item_id in enumerate(self.corpus_ids)
        }
        self._base = None
        self._visual = None
        self._cohort_compiler = None
        self._visual_cache.clear()
        self._last_query_ids = ()
        self._last_score_matrix = ()

        began = time.perf_counter()
        visual_materializations = 0
        if self.mode in {"text", "two-stage", "tiered-selective"}:
            self._base = self.backend.encode_texts(self.corpus_texts)
        elif self.mode in {"bm25-fusion-sync", "bm25-fusion-batched"}:
            self._cohort_compiler = CohortCompiler(
                corpus_ids=self.corpus_ids,
                corpus_texts=self.corpus_texts,
                corpus_images=self.corpus_images,
                backend=self.backend,
                candidate_k=self.candidate_k,
                top_k=self.top_k,
                request_batch_size=(
                    1
                    if self.mode == "bm25-fusion-sync"
                    else self.request_batch_size
                ),
                cache_policy=(
                    "none"
                    if self.mode == "bm25-fusion-sync"
                    else self.cohort_cache_policy
                ),
                admitted_item_ids=self.admitted_item_ids,
                visual_prior_by_rank=self.visual_prior_by_rank,
            )
        elif self.mode == "visual":
            self._visual = self._encode_images(range(len(self.corpus_ids)))
            visual_materializations = len(self.corpus_ids)
        elif self.mode == "visual-pool":
            full = self._encode_images(range(len(self.corpus_ids)))
            visual_materializations = len(self.corpus_ids)
            route = f"image-pool-{self.image_pool_factor}"
            routes = self.backend.derive_image_routes(full)
            if route not in routes:
                raise RuntimeError(f"backend did not emit requested route {route}")
            self._visual = routes[route]
        else:  # pragma: no cover - constructor validates modes
            raise AssertionError(self.mode)
        elapsed_ms = (time.perf_counter() - began) * 1000.0

        index = self._base if self._base is not None else self._visual
        if index is None and self._cohort_compiler is None:
            raise AssertionError("index construction produced no representation")
        self._index_info = {
            "dataset_name": dataset_name,
            "mode": self.mode,
            "corpus_items": len(self.corpus_ids),
            "model_load_ms_outside_official_index_timer": self.model_load_ms,
            "measured_index_ms_inside_pipeline": elapsed_ms,
            "index_vector_bytes": (
                _batch_bytes(index)
                if index is not None
                else self._cohort_compiler.logical_bm25_bytes
            ),
            "index_vectors": (
                sum(int(embedding.shape[0]) for embedding in index.embeddings)
                if index is not None
                else 0
            ),
            "index_kind": (
                "bm25-locator"
                if self._cohort_compiler is not None
                else "late-interaction"
            ),
            "visual_materializations_during_index": visual_materializations,
            "visual_encoding_avoided_during_index": (
                len(self.corpus_ids) - visual_materializations
            ),
            "pooling_reuses_full_visual_encoding": self.mode == "visual-pool",
            "decoded_images_passed_through_without_png_roundtrip": True,
        }

    def _base_scores(self, queries: EncodedBatch) -> list[list[float]]:
        index = self._base if self._base is not None else self._visual
        if index is None:
            raise RuntimeError("index() must be called before search()")
        return [
            [float(value) for value in row]
            for row in self.backend.score(queries.embeddings, index.embeddings)
        ]

    def _candidate_positions(self, scores: Sequence[float]) -> list[int]:
        count = min(self.candidate_k, len(self.corpus_ids))
        return sorted(
            range(len(self.corpus_ids)),
            key=lambda index: (-float(scores[index]), self.corpus_ids[index]),
        )[:count]

    def _cache_insert(self, item_id: str, embedding: Any) -> None:
        self._visual_cache[item_id] = embedding
        self._visual_cache.move_to_end(item_id)
        if self.cache_capacity_items:
            while len(self._visual_cache) > self.cache_capacity_items:
                self._visual_cache.popitem(last=False)

    def _selective_search(
        self,
        query_ids: Sequence[str],
        queries: EncodedBatch,
        base_scores: Sequence[Sequence[float]],
    ) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
        results: dict[str, dict[str, float]] = {}
        counters = {
            "candidate_events": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "visual_materializations_during_search": 0,
            "visual_encoder_calls_during_search": 0,
            "visual_score_pairs": 0,
            "peak_cached_items": len(self._visual_cache),
            "peak_cached_vector_bytes": sum(
                _embedding_bytes(value) for value in self._visual_cache.values()
            ),
        }

        for query_offset, (query_id, query_embedding, row) in enumerate(
            zip(query_ids, queries.embeddings, base_scores, strict=True)
        ):
            candidate_positions = self._candidate_positions(row)
            candidate_ids = [
                self.corpus_ids[position] for position in candidate_positions
            ]
            counters["candidate_events"] += len(candidate_ids)

            if self.mode == "two-stage":
                encoded = self._encode_images(candidate_positions)
                full_embeddings = list(encoded.embeddings)
                counters["cache_misses"] += len(candidate_ids)
                counters["visual_materializations_during_search"] += len(candidate_ids)
                counters["visual_encoder_calls_during_search"] += 1
            else:
                # Protect hits from eviction while this query's misses enter
                # the LRU.  With capacity >= candidate_k, all currently
                # activated pages then remain available for scoring.
                for item_id in candidate_ids:
                    if item_id in self._visual_cache:
                        self._visual_cache.move_to_end(item_id)
                missing_positions = [
                    position
                    for position, item_id in zip(
                        candidate_positions, candidate_ids, strict=True
                    )
                    if item_id not in self._visual_cache
                ]
                if missing_positions:
                    encoded = self._encode_images(missing_positions)
                    counters["visual_encoder_calls_during_search"] += 1
                    counters["visual_materializations_during_search"] += len(
                        missing_positions
                    )
                    for position, embedding in zip(
                        missing_positions, encoded.embeddings, strict=True
                    ):
                        self._cache_insert(self.corpus_ids[position], embedding)
                counters["cache_misses"] += len(missing_positions)
                counters["cache_hits"] += len(candidate_ids) - len(missing_positions)
                full_embeddings = []
                for item_id in candidate_ids:
                    embedding = self._visual_cache[item_id]
                    self._visual_cache.move_to_end(item_id)
                    full_embeddings.append(embedding)

            full_scores = list(
                self.backend.score([query_embedding], full_embeddings)[0]
            )
            counters["visual_score_pairs"] += len(full_scores)
            if self.mode == "two-stage":
                results[str(query_id)] = _rank(
                    candidate_ids,
                    full_scores,
                    top_k=self.top_k,
                )
            else:
                merged = [float(value) for value in row]
                for position, score in zip(
                    candidate_positions, full_scores, strict=True
                ):
                    merged[position] = float(score)
                results[str(query_id)] = _rank(
                    self.corpus_ids,
                    merged,
                    top_k=self.top_k,
                )

            cache_bytes = sum(
                _embedding_bytes(value) for value in self._visual_cache.values()
            )
            counters["peak_cached_items"] = max(
                counters["peak_cached_items"], len(self._visual_cache)
            )
            counters["peak_cached_vector_bytes"] = max(
                counters["peak_cached_vector_bytes"], cache_bytes
            )

        return results, counters

    def search(
        self,
        query_ids: list[str],
        queries: list[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        if len(query_ids) != len(queries):
            raise ValueError("query identifiers and texts differ in length")
        if not self.corpus_ids:
            raise RuntimeError("index() must be called before search()")

        if self.mode in {"bm25-fusion-sync", "bm25-fusion-batched"}:
            if self._cohort_compiler is None:
                raise RuntimeError("index() must be called before search()")
            execution = self._cohort_compiler.execute_batch(query_ids, queries)
            metrics = execution.metrics
            self._search_info = {
                **self._index_info,
                **metrics,
                "query_encode_ms_inside_search": metrics["query_encode_ms"],
                "retrieval_and_materialization_ms_inside_search": metrics[
                    "total_execution_ms"
                ],
                "current_cached_items": metrics["current_resident_items"],
                "current_cached_vector_bytes": metrics[
                    "current_resident_vector_bytes"
                ],
                "cache_capacity_items": None,
                "physical_cache_is_query_scoped": False,
                "backend": dict(self.backend.environment()),
            }
            return execution.results, self._search_info

        began = time.perf_counter()
        encoded_queries = self.backend.encode_queries(queries)
        query_encode_ms = (time.perf_counter() - began) * 1000.0

        scoring_began = time.perf_counter()
        base_scores = self._base_scores(encoded_queries)
        if self.capture_score_trace:
            # Preserve the complete query--corpus score surface only for an
            # explicitly requested offline replay. Ordinary evaluation avoids
            # this Python copy and its corpus-scale memory overhead.
            self._last_query_ids = tuple(str(value) for value in query_ids)
            self._last_score_matrix = tuple(
                tuple(float(value) for value in row) for row in base_scores
            )
        selective_counters: dict[str, int] = {
            "candidate_events": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "visual_materializations_during_search": 0,
            "visual_encoder_calls_during_search": 0,
            "visual_score_pairs": 0,
            "peak_cached_items": 0,
            "peak_cached_vector_bytes": 0,
        }
        if self.mode in {"two-stage", "tiered-selective"}:
            results, selective_counters = self._selective_search(
                query_ids,
                encoded_queries,
                base_scores,
            )
        else:
            results = {
                str(query_id): _rank(
                    self.corpus_ids,
                    row,
                    top_k=self.top_k,
                )
                for query_id, row in zip(query_ids, base_scores, strict=True)
            }
        search_body_ms = (time.perf_counter() - scoring_began) * 1000.0

        current_cache_bytes = sum(
            _embedding_bytes(value) for value in self._visual_cache.values()
        )
        self._search_info = {
            **self._index_info,
            **selective_counters,
            "query_count": len(query_ids),
            "top_k": self.top_k,
            "scoring_batch_size": self.scoring_batch_size,
            "candidate_k": (
                self.candidate_k
                if self.mode in {"two-stage", "tiered-selective"}
                else None
            ),
            "query_encode_ms_inside_search": query_encode_ms,
            "retrieval_and_materialization_ms_inside_search": search_body_ms,
            "current_cached_items": len(self._visual_cache),
            "current_cached_vector_bytes": current_cache_bytes,
            "cache_capacity_items": self.cache_capacity_items,
            "physical_cache_is_query_scoped": False,
            "logical_visual_activation_is_query_scoped": (
                self.mode == "tiered-selective"
            ),
            "backend": dict(self.backend.environment()),
        }
        return results, self._search_info

    def export_score_trace(
        self,
        query_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Return the frozen score/cost surface from the latest search.

        The method deliberately excludes relevance labels.  The local ViDoRe
        runner writes those to a separate file so deployable policies cannot
        accidentally treat qrels as runtime context.
        """

        expected = tuple(str(value) for value in query_ids)
        if not self.capture_score_trace:
            raise RuntimeError("score trace capture was not enabled")
        if not self._last_score_matrix or expected != self._last_query_ids:
            raise RuntimeError(
                "score trace requires the identifiers from the latest search"
            )
        index = self._base if self._base is not None else self._visual
        if index is None:
            raise RuntimeError("index() must be called before exporting a trace")
        return {
            "mode": self.mode,
            "query_ids": np.asarray(expected),
            "corpus_ids": np.asarray(self.corpus_ids),
            "scores": np.asarray(self._last_score_matrix, dtype=np.float32),
            "vector_bytes": np.asarray(
                [_embedding_bytes(value) for value in index.embeddings],
                dtype=np.int64,
            ),
            "encode_ms": np.asarray(index.encode_ms, dtype=np.float32),
            "index_total_ms": np.asarray(
                self._index_info["measured_index_ms_inside_pipeline"],
                dtype=np.float64,
            ),
            "model_load_ms": np.asarray(self.model_load_ms, dtype=np.float64),
        }
