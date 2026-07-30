import json
from pathlib import Path

import numpy as np
import pytest

from reprforge.heterogeneous_index import (
    NumpyMaxSimRuntime,
    TorchMaxSimRuntime,
    amplify_compiled_index,
    benchmark_runtime,
    compile_heterogeneous_index,
    load_plan,
    load_query_bank,
    merge_embedding_banks,
    write_embedding_bank,
)
from reprforge.run_end_to_end import run_end_to_end


def _embedding(rows):
    return np.asarray(rows, dtype=np.float32)


def _write_fixture_bank(path: Path, suffix: str = "") -> dict:
    item_ids = [f"item-a{suffix}", f"item-b{suffix}", f"item-c{suffix}"]
    return write_embedding_bank(
        path,
        item_ids=item_ids,
        route_embeddings={
            "small": (
                _embedding([[1.0, 0.0]]),
                _embedding([[0.0, 1.0]]),
                _embedding([[0.5, 0.5]]),
            ),
            "rich": (
                _embedding([[1.0, 0.0], [0.0, 0.25]]),
                _embedding([[0.0, 1.0], [0.5, 0.0], [0.25, 0.25]]),
                _embedding([[0.2, 0.8], [0.8, 0.2]]),
            ),
        },
        query_ids=[f"query{suffix}"],
        query_embeddings=[_embedding([[1.0, 0.0], [0.0, 1.0]])],
        storage_dtype="float32",
    )


def test_compiled_index_is_compact_and_executes_exact_maxsim(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    _write_fixture_bank(bank)
    plan = {
        "item-a": "small",
        "item-b": "rich",
        "item-c": "small",
    }
    compiled = tmp_path / "compiled"
    manifest = compile_heterogeneous_index(
        bank=bank,
        plan=plan,
        output=compiled,
    )

    # 1 + 3 + 1 selected vectors, two float32 dimensions. No vector padding
    # appears in the serialized route shards.
    assert manifest["compact_vector_bytes"] == 5 * 2 * 4
    assert manifest["storage_padding_bytes"] == 0
    assert manifest["route_counts"] == {"rich": 1, "small": 2}
    assert manifest["serialized_bytes"] == sum(
        path.stat().st_size for path in compiled.rglob("*") if path.is_file()
    )

    query_ids, queries = load_query_bank(bank)
    runtime = NumpyMaxSimRuntime(compiled)
    scores = runtime.score(queries[0])
    assert query_ids == ["query"]
    assert runtime.item_ids == ("item-a", "item-b", "item-c")
    assert np.allclose(scores, [1.0, 1.5, 1.0])
    assert runtime.compact_vector_bytes == runtime.resident_vector_bytes

    result = benchmark_runtime(
        runtime,
        query_ids=query_ids,
        query_embeddings=queries,
        warmup=0,
        repetitions=2,
        top_k=2,
    )
    assert result["measurements"] == 2
    assert result["qps"] > 0
    assert result["resident_padding_bytes"] == 0


def test_torch_runtime_matches_numpy_and_reports_batch_padding(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    bank = tmp_path / "bank"
    _write_fixture_bank(bank)
    compiled = tmp_path / "compiled"
    compile_heterogeneous_index(
        bank=bank,
        plan={
            "item-a": "rich",
            "item-b": "rich",
            "item-c": "small",
        },
        output=compiled,
    )
    _, queries = load_query_bank(bank)
    reference = NumpyMaxSimRuntime(compiled).score(queries[0])
    runtime = TorchMaxSimRuntime(
        compiled,
        device="cpu",
        document_batch_size=2,
    )
    assert np.allclose(runtime.score(queries[0]), reference)
    assert runtime.execution_batch_count == 2
    assert runtime.resident_vector_bytes >= runtime.resident_unpadded_vector_bytes
    assert runtime.resident_unpadded_vector_bytes == runtime.compact_vector_bytes

    token_budget_runtime = TorchMaxSimRuntime(
        compiled,
        device="cpu",
        document_batch_size=1,
        token_batch_budget=4,
    )
    assert np.allclose(
        token_budget_runtime.score(queries[0]),
        reference,
    )
    # Lengths are 1, 2, and 3. A four-token padded-work budget groups the
    # first two documents (2 * 2) and leaves the third in its own batch.
    assert token_budget_runtime.execution_batch_count == 2


def test_merge_banks_and_extract_nested_plan(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_fixture_bank(first, "-1")
    _write_fixture_bank(second, "-2")
    merged = tmp_path / "merged"
    manifest = merge_embedding_banks([first, second], merged)
    assert manifest["item_count"] == 6
    assert manifest["queries"]["count"] == 2

    plan_path = tmp_path / "model.json"
    expected = {
        item_id: "small"
        for item_id in json.loads(
            (merged / "routes" / "small" / "ids.json").read_text()
        )
    }
    plan_path.write_text(
        json.dumps({"models": {"v0": {"plan": expected}}}),
        encoding="utf-8",
    )
    assert load_plan(plan_path, dotted_key="models.v0") == expected


def test_compile_rejects_missing_item_and_unknown_route(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    _write_fixture_bank(bank)
    with pytest.raises(ValueError, match="plan/item mismatch"):
        compile_heterogeneous_index(
            bank=bank,
            plan={"item-a": "small"},
            output=tmp_path / "missing",
        )
    with pytest.raises(ValueError, match="unknown routes"):
        compile_heterogeneous_index(
            bank=bank,
            plan={
                "item-a": "missing-route",
                "item-b": "small",
                "item-c": "small",
            },
            output=tmp_path / "unknown",
        )


def test_physical_amplification_uses_distinct_storage_and_invalidates_quality(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    _write_fixture_bank(bank)
    source = tmp_path / "source"
    source_manifest = compile_heterogeneous_index(
        bank=bank,
        plan={
            "item-a": "small",
            "item-b": "rich",
            "item-c": "small",
        },
        output=source,
    )
    amplified = tmp_path / "amplified"
    manifest = amplify_compiled_index(
        source=source,
        factor=4,
        output=amplified,
    )
    assert manifest["item_count"] == 12
    assert manifest["compact_vector_bytes"] == (
        4 * source_manifest["compact_vector_bytes"]
    )
    assert manifest["quality_labels_valid"] is False
    runtime = NumpyMaxSimRuntime(amplified)
    assert len(runtime.item_ids) == 12
    assert len(set(runtime.item_ids)) == 12


def test_end_to_end_compiles_scores_evaluates_and_benchmarks(
    tmp_path: Path,
) -> None:
    route_bank = tmp_path / "route-bank"
    _write_fixture_bank(route_bank / "embedding-bank")
    route_costs = {
        "small": {"index_bytes": 8, "encode_ms": 1.0},
        "rich": {"index_bytes": 24, "encode_ms": 2.0},
    }
    with (route_bank / "items.jsonl").open("w", encoding="utf-8") as handle:
        for item_id in ("item-a", "item-b", "item-c"):
            handle.write(
                json.dumps(
                    {
                        "item_id": item_id,
                        "content_type": "text",
                        "route_costs": route_costs,
                    }
                )
                + "\n"
            )
    (route_bank / "queries.jsonl").write_text(
        json.dumps(
            {
                "query_id": "query",
                "relevance": {"item-b": 1.0},
                "candidate_item_ids": ["item-a", "item-b", "item-c"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    expected_scores = {
        "small": {"item-a": 1.0, "item-b": 1.0, "item-c": 1.0},
        "rich": {"item-a": 1.25, "item-b": 1.5, "item-c": 1.6},
    }
    with (route_bank / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for route, scores in expected_scores.items():
            for item_id, score in scores.items():
                handle.write(
                    json.dumps(
                        {
                            "route": route,
                            "query_id": "query",
                            "item_id": item_id,
                            "score": score,
                        }
                    )
                    + "\n"
                )
    plan = {
        "item-a": "small",
        "item-b": "rich",
        "item-c": "small",
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan": plan}), encoding="utf-8")

    result = run_end_to_end(
        route_bank_directory=route_bank,
        plan_path=plan_path,
        plan_key=None,
        uniform_route=None,
        output=tmp_path / "system",
        storage_dtype="float32",
        engine="numpy",
        device="cpu",
        document_batch_size=2,
        warmup=0,
        repetitions=2,
        top_k=2,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )
    assert result["correctness"]["score_contract"]["all_close"] is True
    assert result["correctness"]["ranking_contract"]["all_equal"] is True
    assert result["correctness"]["physical_metrics"]["recall_at_1"] == 1.0
    assert result["index"]["compact_vector_bytes"] == 40
    assert result["performance"]["qps"] > 0
    assert (tmp_path / "system" / "end-to-end.json").exists()

    uniform = run_end_to_end(
        route_bank_directory=route_bank,
        plan_path=None,
        plan_key=None,
        uniform_route="small",
        output=tmp_path / "uniform-system",
        storage_dtype="float32",
        engine="numpy",
        device="cpu",
        document_batch_size=2,
        warmup=0,
        repetitions=1,
        top_k=2,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )
    assert uniform["uniform_route"] == "small"
    assert uniform["correctness"]["score_contract"]["all_close"] is True
