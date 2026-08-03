from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest

from reprforge.cohort_compiler import CohortCompiler
from reprforge.mmdocir_route_runner import EncodedBatch


def _batch(values: Sequence[Sequence[float]]) -> EncodedBatch:
    return EncodedBatch(
        embeddings=tuple(
            np.asarray([value], dtype=np.float32) for value in values
        ),
        encode_ms=tuple(0.0 for _ in values),
    )


class FakeCohortBackend:
    def __init__(self, *, fail_score: bool = False) -> None:
        self.fail_score = fail_score
        self.image_calls: list[tuple[str, ...]] = []
        self.score_shapes: list[tuple[int, int]] = []

    def encode_queries(self, queries: Sequence[str]) -> EncodedBatch:
        values = {
            "revenue": (1.0, 0.0),
            "policy": (0.0, 1.0),
        }
        return _batch([values[value] for value in queries])

    def encode_images(self, images: Sequence[bytes]) -> EncodedBatch:
        names = tuple(value.decode() for value in images)
        self.image_calls.append(names)
        values = {
            "image-a": (0.1, 0.9),
            "image-b": (0.8, 0.2),
            "image-c": (0.0, 1.0),
            "image-d": (1.0, 0.0),
        }
        return _batch([values[value] for value in names])

    def score(
        self,
        queries: Sequence[Any],
        documents: Sequence[Any],
    ) -> Sequence[Sequence[float]]:
        if self.fail_score:
            raise RuntimeError("injected scoring failure")
        self.score_shapes.append((len(queries), len(documents)))
        return [
            [
                float(np.asarray(query)[0] @ np.asarray(document)[0])
                for document in documents
            ]
            for query in queries
        ]


def _compiler(
    backend: FakeCohortBackend,
    *,
    request_batch_size: int,
    cache_policy: str = "none",
    candidate_k: int = 3,
    top_k: int = 4,
    admitted_item_ids: set[str] | None = None,
    visual_prior_by_rank: Sequence[float] | None = None,
) -> CohortCompiler:
    return CohortCompiler(
        corpus_ids=["a", "b", "c", "d"],
        corpus_texts=[
            "revenue chart",
            "revenue table",
            "employee policy",
            "revenue revenue",
        ],
        corpus_images=[b"image-a", b"image-b", b"image-c", b"image-d"],
        backend=backend,
        candidate_k=candidate_k,
        top_k=top_k,
        request_batch_size=request_batch_size,
        cache_policy=cache_policy,
        admitted_item_ids=admitted_item_ids,
        visual_prior_by_rank=visual_prior_by_rank,
    )


def test_batched_execution_matches_synchronous_ranking() -> None:
    sync_backend = FakeCohortBackend()
    sync = _compiler(sync_backend, request_batch_size=1)
    sync_result = sync.execute_batch(["q1", "q2"], ["revenue", "policy"])

    batch_backend = FakeCohortBackend()
    batched = _compiler(batch_backend, request_batch_size=2)
    batch_result = batched.execute_batch(
        ["q1", "q2"],
        ["revenue", "policy"],
    )

    assert batch_result.results == sync_result.results
    assert sync_backend.score_shapes == [(1, 3), (1, 3)]
    assert batch_backend.score_shapes == [(2, 4)]


def test_batched_execution_encodes_candidate_union_once() -> None:
    backend = FakeCohortBackend()
    compiler = _compiler(backend, request_batch_size=2)

    execution = compiler.execute_batch(["q1", "q2"], ["revenue", "policy"])

    assert backend.image_calls == [("image-d", "image-a", "image-b", "image-c")]
    assert execution.metrics["candidate_events"] == 6
    assert execution.metrics["unique_candidates_within_batches"] == 4
    assert execution.metrics["within_batch_deduplicated_events"] == 2
    assert execution.metrics["visual_encoder_calls"] == 1
    trace = execution.metrics["batch_trace"]
    assert len(trace) == 1
    assert trace[0]["query_offset_start"] == 0
    assert trace[0]["query_count"] == 2
    assert trace[0]["candidate_events"] == 6
    assert trace[0]["unique_candidates"] == 4
    assert trace[0]["cache_hit_events"] == 0
    assert trace[0]["visual_pages_encoded"] == 4
    assert trace[0]["resident_items_after_publish"] == 0
    assert trace[0]["generation_after_publish"] == 0


def test_resident_cache_reuses_physical_state_without_changing_results() -> None:
    backend = FakeCohortBackend()
    compiler = _compiler(
        backend,
        request_batch_size=1,
        cache_policy="resident",
    )

    first = compiler.execute_batch(["q1"], ["revenue"])
    second = compiler.execute_batch(["q2"], ["policy"])

    assert backend.image_calls == [
        ("image-d", "image-a", "image-b"),
        ("image-c",),
    ]
    assert compiler.resident_item_ids == frozenset({"a", "b", "c", "d"})
    assert compiler.generation == 2
    assert second.metrics["cache_hit_events"] == 2
    assert list(first.results["q1"]) == ["d", "b", "a", "c"]
    assert list(second.results["q2"])[0] == "c"


def test_failed_batch_does_not_publish_partial_generation() -> None:
    backend = FakeCohortBackend(fail_score=True)
    compiler = _compiler(
        backend,
        request_batch_size=2,
        cache_policy="resident",
    )

    with pytest.raises(RuntimeError, match="injected scoring failure"):
        compiler.execute_batch(["q1", "q2"], ["revenue", "policy"])

    assert compiler.resident_item_ids == frozenset()
    assert compiler.generation == 0


def test_non_candidate_tail_preserves_bm25_order() -> None:
    backend = FakeCohortBackend()
    compiler = _compiler(
        backend,
        request_batch_size=1,
        candidate_k=2,
        top_k=4,
    )

    execution = compiler.execute_batch(["q1"], ["revenue"])

    # The visual fusion may reorder d/a, while the untouched BM25 tail remains
    # b before c.  This makes the heterogeneous Top-100 rank deterministic.
    assert list(execution.results["q1"])[-2:] == ["b", "c"]


def test_admission_plan_encodes_only_published_representation_views() -> None:
    backend = FakeCohortBackend()
    compiler = _compiler(
        backend,
        request_batch_size=2,
        cache_policy="resident",
        admitted_item_ids={"b", "c"},
        visual_prior_by_rank=[0.0, 0.0, 0.0],
    )

    execution = compiler.execute_batch(["q1", "q2"], ["revenue", "policy"])

    assert backend.image_calls == [("image-b", "image-c")]
    assert compiler.resident_item_ids == frozenset({"b", "c"})
    assert execution.metrics["candidate_events"] == 6
    assert execution.metrics["admitted_candidate_events"] == 3
    assert execution.metrics["visual_pages_encoded"] == 2
    assert execution.metrics["visual_score_pairs"] == 3
    assert execution.metrics["representation_admission_enabled"] is True
    assert execution.metrics["admitted_plan_items"] == 2
    assert execution.metrics["batch_trace"][0]["unique_admitted_candidates"] == 2


def test_admission_plan_can_be_materialized_atomically_before_queries() -> None:
    backend = FakeCohortBackend()
    compiler = _compiler(
        backend,
        request_batch_size=2,
        cache_policy="resident",
        admitted_item_ids={"b", "c"},
        visual_prior_by_rank=[0.0, 0.0, 0.0],
    )

    prebuild = compiler.materialize_admitted()
    execution = compiler.execute_batch(["q1", "q2"], ["revenue", "policy"])

    assert backend.image_calls == [("image-b", "image-c")]
    assert prebuild["visual_pages_encoded"] == 2
    assert prebuild["visual_encoder_calls"] == 1
    assert compiler.generation == 1
    assert execution.metrics["initial_resident_items"] == 2
    assert execution.metrics["visual_pages_encoded"] == 0
    assert execution.metrics["cache_hit_events"] == 3


def test_progressive_rounds_append_resident_views_without_reencoding() -> None:
    backend = FakeCohortBackend()
    compiler = _compiler(
        backend,
        request_batch_size=2,
        cache_policy="resident",
    )

    first = compiler.materialize_items({"b", "c"})
    second = compiler.materialize_items({"a", "b"})

    assert backend.image_calls == [("image-b", "image-c"), ("image-a",)]
    assert first["visual_pages_encoded"] == 2
    assert second["visual_pages_encoded"] == 1
    assert compiler.generation == 2
    assert compiler.resident_item_ids == frozenset({"a", "b", "c"})
    embeddings = compiler.resident_embeddings(["c", "a"])
    assert len(embeddings) == 2
    with pytest.raises(KeyError, match="not resident"):
        compiler.resident_embeddings(["d"])


def test_empty_admission_plan_uses_priors_without_loading_visual_backend() -> None:
    backend = FakeCohortBackend()
    compiler = _compiler(
        backend,
        request_batch_size=1,
        admitted_item_ids=set(),
        visual_prior_by_rank=[0.0, 0.0, 0.0],
    )

    execution = compiler.execute_batch(["q1"], ["revenue"])

    assert backend.image_calls == []
    assert backend.score_shapes == []
    assert execution.metrics["visual_pages_encoded"] == 0
    assert execution.metrics["admitted_candidate_events"] == 0
    assert list(execution.results["q1"])[0] == "d"


def test_admission_plan_requires_matching_rank_prior() -> None:
    backend = FakeCohortBackend()
    with pytest.raises(ValueError, match="supplied together"):
        _compiler(
            backend,
            request_batch_size=1,
            admitted_item_ids={"a"},
        )
    with pytest.raises(ValueError, match="match candidate_k"):
        _compiler(
            backend,
            request_batch_size=1,
            admitted_item_ids={"a"},
            visual_prior_by_rank=[0.0],
        )
