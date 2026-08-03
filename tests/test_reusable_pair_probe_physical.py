from __future__ import annotations

import numpy as np
import pytest

from reprforge.mmdocir_route_runner import EncodedBatch
from tools.run_reusable_pair_probe_physical import (
    LazyImageCorpus,
    PhysicalPairScoreProvider,
)


class FakeBackend:
    def __init__(self) -> None:
        self.image_batches: list[int] = []

    def encode_queries(self, queries):
        return EncodedBatch(
            tuple(np.asarray([len(query)], dtype=np.float32) for query in queries),
            tuple(0.0 for _ in queries),
        )

    def encode_images(self, images):
        self.image_batches.append(len(images))
        return EncodedBatch(
            tuple(np.asarray([len(image)], dtype=np.float32) for image in images),
            tuple(0.0 for _ in images),
        )

    def score(self, queries, documents):
        return [
            [float(query[0] * document[0]) for document in documents]
            for query in queries
        ]


def test_physical_provider_materializes_scores_and_reuses_pages(tmp_path):
    paths = []
    for index, payload in enumerate((b"a", b"bb", b"ccc")):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(payload)
        paths.append(path)
    backend = FakeBackend()
    provider = PhysicalPairScoreProvider(
        candidates=np.asarray([[0, 1], [1, 2]]),
        query_texts=("x", "yy"),
        corpus_ids=("0", "1", "2"),
        corpus_images=LazyImageCorpus(paths),
        backend=backend,
    )

    with pytest.raises(RuntimeError, match="unmaterialized"):
        provider.score(0, 0)
    provider.materialize({0, 1})
    provider.materialize({1, 2})

    assert provider.materialized_pages == frozenset({0, 1, 2})
    assert backend.image_batches == [2, 1]
    assert provider.materialization_calls == 2
    assert provider.counters.visual_pages_encoded == 3
    assert provider.counters.visual_score_pairs == 6
    assert provider.score(0, 1) == 2.0
    assert provider.score(1, 1) == 6.0
    assert provider.candidate_score_matrix().tolist() == [[1.0, 2.0], [4.0, 6.0]]
