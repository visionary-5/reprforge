from pathlib import Path

import numpy as np
import pytest

from reprforge.heterogeneous_index import (
    compile_heterogeneous_index,
    write_embedding_bank,
)
from reprforge.retrieval_baselines import (
    PooledExactRerankRuntime,
    PreencodedNoCacheRuntime,
    benchmark_selected_runtime,
)


def _embedding(rows):
    return np.asarray(rows, dtype=np.float32)


def test_pooled_exact_rerank_uses_full_scores_only_after_selection(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    bank = tmp_path / "bank"
    item_ids = ("a", "b", "c")
    write_embedding_bank(
        bank,
        item_ids=item_ids,
        route_embeddings={
            "pool": (
                _embedding([[1.0, 0.0]]),
                _embedding([[0.9, 0.0]]),
                _embedding([[0.1, 0.0]]),
            ),
            "full": (
                _embedding([[0.2, 0.0]]),
                _embedding([[1.0, 0.0]]),
                _embedding([[2.0, 0.0]]),
            ),
        },
    )
    pooled = tmp_path / "pooled"
    full = tmp_path / "full"
    compile_heterogeneous_index(
        bank=bank,
        plan={item_id: "pool" for item_id in item_ids},
        output=pooled,
    )
    compile_heterogeneous_index(
        bank=bank,
        plan={item_id: "full" for item_id in item_ids},
        output=full,
    )
    runtime = PooledExactRerankRuntime(
        pooled,
        full,
        device="cpu",
        candidate_k=2,
    )
    query = _embedding([[1.0, 0.0]])

    # Pooling admits a and b, then full MaxSim reverses those two. c has the
    # strongest full score but cannot bypass first-stage candidate generation.
    assert [item_id for item_id, _ in runtime.search(query, top_k=2)] == [
        "b",
        "a",
    ]
    assert [item_id for item_id, _ in runtime.search_candidates(
        query,
        candidate_item_ids=("a", "c"),
        top_k=2,
    )] == ["c", "a"]


def test_preencoded_no_cache_overrides_only_selected_items(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    bank = tmp_path / "bank"
    item_ids = ("a", "b", "c")
    query = _embedding([[1.0, 0.0]])
    write_embedding_bank(
        bank,
        item_ids=item_ids,
        route_embeddings={
            "pool": (
                _embedding([[1.0, 0.0]]),
                _embedding([[0.9, 0.0]]),
                _embedding([[0.8, 0.0]]),
            ),
            "full": (
                _embedding([[0.1, 0.0]]),
                _embedding([[2.0, 0.0]]),
                _embedding([[3.0, 0.0]]),
            ),
        },
    )
    pooled = tmp_path / "pooled"
    full = tmp_path / "full"
    compile_heterogeneous_index(
        bank=bank,
        plan={item_id: "pool" for item_id in item_ids},
        output=pooled,
    )
    compile_heterogeneous_index(
        bank=bank,
        plan={item_id: "full" for item_id in item_ids},
        output=full,
    )
    runtime = PreencodedNoCacheRuntime(pooled, full, device="cpu")
    # c would win under full scoring but stays pooled because only b is
    # transiently acquired for this query.
    assert [item_id for item_id, _ in runtime.search_selected(
        query,
        selected_item_ids=("b",),
        top_k=3,
    )] == ["b", "a", "c"]
    measured = benchmark_selected_runtime(
        runtime,
        query_ids=("q",),
        query_embeddings=(query,),
        selections=(("b",),),
        warmup=0,
        repetitions=1,
        top_k=2,
    )
    assert measured["measurements"] == 1
    assert measured["transient_vector_bytes"]["max"] > 0
