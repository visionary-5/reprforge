from __future__ import annotations

import base64
import csv
from pathlib import Path

import numpy as np

from reprforge.irpapers_benchmark import (
    candidate_fusion_replay,
    full_fusion_results,
    load_irpapers,
    minimum_action_oracle,
    recall_at_k,
    score_rows_to_results,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_irpapers_preserves_supplied_modalities_and_single_gold(tmp_path: Path) -> None:
    docs = tmp_path / "docs.csv"
    queries = tmp_path / "queries.csv"
    _write_csv(
        docs,
        ["dataset_id", "pdf_id_x", "page_number", "transcription", "base64_str"],
        [
            {
                "dataset_id": "7_1",
                "pdf_id_x": "7",
                "page_number": "1",
                "transcription": "a chart",
                "base64_str": base64.b64encode(b"image-a").decode(),
            },
            {
                "dataset_id": "7_2",
                "pdf_id_x": "7",
                "page_number": "2",
                "transcription": "a table",
                "base64_str": base64.b64encode(b"image-b").decode(),
            },
        ],
    )
    _write_csv(
        queries,
        ["dataset_id", "pdf_id", "page_number", "question"],
        [
            {
                "dataset_id": "7_2",
                "pdf_id": "7",
                "page_number": "2",
                "question": "Which page has the table?",
            }
        ],
    )

    data = load_irpapers(docs, queries, expected_docs=2, expected_queries=1)

    assert data.corpus_ids == ("7_1", "7_2")
    assert data.corpus_images == (b"image-a", b"image-b")
    assert data.qrels == {"q-0000": frozenset({"7_2"})}
    assert data.metadata["papers"] == 1
    assert data.metadata["query_source_papers"] == 1
    assert data.metadata["query_metadata_mismatches"] == []


def test_load_irpapers_retains_official_qrel_when_query_metadata_disagrees(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs.csv"
    queries = tmp_path / "queries.csv"
    _write_csv(
        docs,
        ["dataset_id", "pdf_id_x", "page_number", "transcription", "base64_str"],
        [
            {
                "dataset_id": "16_7",
                "pdf_id_x": "16",
                "page_number": "7",
                "transcription": "page",
                "base64_str": base64.b64encode(b"image").decode(),
            }
        ],
    )
    _write_csv(
        queries,
        ["dataset_id", "pdf_id", "page_number", "question"],
        [
            {
                "dataset_id": "16_7",
                "pdf_id": "15",
                "page_number": "7",
                "question": "officially mislabeled question",
            }
        ],
    )

    data = load_irpapers(docs, queries, expected_docs=1, expected_queries=1)

    assert data.qrels == {"q-0000": frozenset({"16_7"})}
    assert data.metadata["query_metadata_mismatches"] == [
        {
            "csv_row": 2,
            "dataset_id": "16_7",
            "declared_pdf_page": "15_7",
        }
    ]


def test_recall_and_candidate_replay_use_deterministic_ranks() -> None:
    query_ids = ["q0", "q1"]
    corpus_ids = ["a", "b", "c"]
    locator = np.asarray([[3, 2, 1], [1, 3, 2]], dtype=np.float32)
    visual = np.asarray([[0, 5, 1], [0, 1, 4]], dtype=np.float32)
    replay, cost = candidate_fusion_replay(
        query_ids,
        corpus_ids,
        locator,
        visual,
        candidate_k=3,
        top_k=3,
    )

    metrics = recall_at_k(
        replay,
        {"q0": frozenset({"b"}), "q1": frozenset({"c"})},
        cutoffs=(1, 2),
    )

    assert metrics == {"recall_1": 1.0, "recall_2": 1.0}
    assert cost["candidate_events"] == 6
    assert cost["resident_unique_pages"] == 3


def test_full_fusion_and_score_conversion_validate_shapes() -> None:
    query_ids = ["q0"]
    corpus_ids = ["a", "b", "c"]
    locator = np.asarray([[3, 2, 1]], dtype=np.float32)
    visual = np.asarray([[0, 5, 1]], dtype=np.float32)

    fused = full_fusion_results(
        query_ids,
        corpus_ids,
        locator,
        visual,
        top_k=3,
    )
    direct = score_rows_to_results(
        query_ids,
        corpus_ids,
        locator,
        top_k=3,
    )

    assert list(fused["q0"])[0] == "b"
    assert list(direct["q0"])[0] == "a"


def test_minimum_action_oracle_charges_cheapest_successful_policy() -> None:
    qrels = {"q0": frozenset({"a"}), "q1": frozenset({"b"})}
    policies = {
        0: {"q0": {"a": 2.0}, "q1": {"a": 2.0}},
        10: {"q0": {"a": 2.0}, "q1": {"b": 2.0}},
        20: {"q0": {"a": 2.0}, "q1": {"b": 2.0}},
    }

    oracle = minimum_action_oracle(policies, qrels, cutoff=1)

    assert oracle["success_rate"] == 1.0
    assert oracle["visual_page_events"] == 10
    assert oracle["selection_counts"] == {"0": 1, "10": 1}
    assert oracle["deployable"] is False
