from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from reprforge.mmdocir_route_runner import EncodedBatch
from reprforge.vidore_pipeline import ReprForgeViDoRePipeline


def _batch(values: Sequence[Sequence[float]]) -> EncodedBatch:
    return EncodedBatch(
        embeddings=tuple(
            np.asarray([value], dtype=np.float32) for value in values
        ),
        encode_ms=tuple(0.0 for _ in values),
    )


class FakeBackend:
    def __init__(self) -> None:
        self.image_calls: list[tuple[str, ...]] = []

    def encode_queries(self, queries: Sequence[str]) -> EncodedBatch:
        values = {
            "q-a": (1.0, 0.0),
            "q-b": (0.0, 1.0),
        }
        return _batch([values[value] for value in queries])

    def encode_texts(self, texts: Sequence[str]) -> EncodedBatch:
        values = {
            "text-a": (0.9, 0.1),
            "text-b": (0.8, 0.2),
            "text-c": (0.1, 0.9),
        }
        return _batch([values[value] for value in texts])

    def encode_images(self, images: Sequence[bytes]) -> EncodedBatch:
        names = tuple(value.decode() for value in images)
        self.image_calls.append(names)
        values = {
            "image-a": (0.1, 0.9),
            "image-b": (1.0, 0.0),
            "image-c": (0.0, 1.0),
        }
        return _batch([values[value] for value in names])

    def derive_image_routes(
        self,
        images: EncodedBatch,
    ) -> Mapping[str, EncodedBatch]:
        return {"image-pool-25": images}

    def score(
        self,
        queries: Sequence[Any],
        documents: Sequence[Any],
    ) -> Sequence[Sequence[float]]:
        return [
            [
                float(np.asarray(query)[0] @ np.asarray(document)[0])
                for document in documents
            ]
            for query in queries
        ]

    def environment(self) -> Mapping[str, Any]:
        return {"backend": "fake"}


def _pipeline(mode: str, backend: FakeBackend, **kwargs) -> ReprForgeViDoRePipeline:
    return ReprForgeViDoRePipeline(
        base_model="unused",
        adapter="unused",
        mode=mode,
        device="cpu",
        batch_size=2,
        scoring_batch_size=2,
        candidate_k=kwargs.pop("candidate_k", 2),
        top_k=kwargs.pop("top_k", 2),
        backend_factory=lambda: backend,
        **kwargs,
    )


def _index(pipeline: ReprForgeViDoRePipeline) -> None:
    pipeline.index(
        ["a", "b", "c"],
        [b"image-a", b"image-b", b"image-c"],
        ["text-a", "text-b", "text-c"],
        dataset_name="vidore/test",
    )


def test_text_and_visual_modes_build_different_representations() -> None:
    text_backend = FakeBackend()
    text = _pipeline("text", text_backend)
    _index(text)
    text_results, text_info = text.search(["q"], ["q-a"])
    assert list(text_results["q"]) == ["a", "b"]
    assert text_info["visual_materializations_during_index"] == 0
    assert text_info["visual_encoding_avoided_during_index"] == 3

    visual_backend = FakeBackend()
    visual = _pipeline("visual", visual_backend)
    _index(visual)
    visual_results, visual_info = visual.search(["q"], ["q-a"])
    assert list(visual_results["q"]) == ["b", "a"]
    assert visual_info["visual_materializations_during_index"] == 3
    assert visual_backend.image_calls == [("image-a", "image-b", "image-c")]


def test_two_stage_only_materializes_query_candidates() -> None:
    backend = FakeBackend()
    pipeline = _pipeline("two-stage", backend)
    _index(pipeline)

    results, info = pipeline.search(["q"], ["q-a"])

    assert list(results["q"]) == ["b", "a"]
    assert backend.image_calls == [("image-a", "image-b")]
    assert info["candidate_events"] == 2
    assert info["cache_misses"] == 2
    assert info["current_cached_items"] == 0


def test_tiered_selective_reuses_physical_cache_without_global_activation() -> None:
    backend = FakeBackend()
    pipeline = _pipeline("tiered-selective", backend, candidate_k=2)
    _index(pipeline)

    results, info = pipeline.search(["q1", "q2"], ["q-a", "q-b"])

    # q1 activates a/b and visual scoring reverses them.  q2 activates c/b;
    # cached a remains physically resident but its visual score is not applied.
    assert list(results["q1"]) == ["b", "a"]
    assert list(results["q2"]) == ["c", "a"]
    assert backend.image_calls == [
        ("image-a", "image-b"),
        ("image-c",),
    ]
    assert info["cache_hits"] == 1
    assert info["cache_misses"] == 3
    assert info["current_cached_items"] == 3
    assert info["logical_visual_activation_is_query_scoped"] is True


def test_tiered_selective_lru_capacity_forces_rematerialization() -> None:
    backend = FakeBackend()
    pipeline = _pipeline(
        "tiered-selective",
        backend,
        candidate_k=1,
        top_k=1,
        cache_capacity_items=1,
    )
    _index(pipeline)

    _, info = pipeline.search(
        ["q1", "q2", "q3"],
        ["q-a", "q-b", "q-a"],
    )

    assert backend.image_calls == [
        ("image-a",),
        ("image-c",),
        ("image-a",),
    ]
    assert info["cache_hits"] == 0
    assert info["cache_misses"] == 3
    assert info["peak_cached_items"] == 1
