#!/usr/bin/env python3
"""IRPAPERS transfer utilities for ReprForge.

IRPAPERS distributes one CSV containing both GPT-4.1 transcriptions and
base64-encoded page images, plus a second CSV containing single-page queries.
This module preserves that public contract while keeping the evaluator small
and independently testable.  It deliberately reports Recall rather than
pretending a single relevant page defines graded nDCG.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reprforge.bm25 import build_index, score_queries
from reprforge.mmdocir_route_runner import ColPaliBackend, EncodedBatch


def _allow_large_csv_fields() -> None:
    """Raise the CSV field limit to admit multi-megabyte base64 page images."""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class IRPapersData:
    query_ids: tuple[str, ...]
    queries: tuple[str, ...]
    corpus_ids: tuple[str, ...]
    corpus_texts: tuple[str, ...]
    corpus_images: tuple[bytes, ...]
    qrels: Mapping[str, frozenset[str]]
    metadata: Mapping[str, Any]


class IRPapersColPaliBackend(ColPaliBackend):
    """ColPali backend using the model's standard public retrieval prompts.

    The shared parent retains the exactly verified ColPali-v1.1 adapter loading,
    host-resident embeddings and batched MaxSim implementation.  Only input
    processing changes: MMDocIR's ``Question:``/``Passage:`` compatibility
    profile must not leak into the independent IRPAPERS transfer.
    """

    def encode_queries(self, queries: Sequence[str]) -> EncodedBatch:
        return self._encode(
            queries,
            lambda values: self.processor.process_queries(texts=list(values)),
        )

    def encode_texts(self, texts: Sequence[str]) -> EncodedBatch:
        return self._encode(
            texts,
            lambda values: self.processor.process_texts(texts=list(values)),
        )

    def encode_images(self, images: Sequence[Any]) -> EncodedBatch:
        from PIL import Image

        def make_batch(values: Sequence[Any]) -> Any:
            pictures = []
            for value in values:
                if isinstance(value, bytes):
                    picture = Image.open(io.BytesIO(value)).convert("RGB")
                elif hasattr(value, "convert"):
                    picture = value.convert("RGB")
                else:
                    raise TypeError(
                        "image inputs must be encoded bytes or PIL-like objects"
                    )
                pictures.append(picture)
            return self.processor.process_images(images=pictures)

        return self._encode(images, make_batch)

    def environment(self) -> Mapping[str, Any]:
        return {
            **super().environment(),
            "input_profile": "colpali-public-process_queries-and-images",
        }


def _require_columns(
    path: Path,
    fieldnames: Sequence[str] | None,
    required: set[str],
) -> None:
    available = set(fieldnames or ())
    missing = required - available
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def load_irpapers(
    docs_path: Path,
    queries_path: Path,
    *,
    decode_images: bool = True,
    expected_docs: int | None = 3230,
    expected_queries: int | None = 180,
) -> IRPapersData:
    """Load and validate the official IRPAPERS CSV representation.

    The supplied transcription is dataset input.  Its upstream GPT-4.1 cost
    must not be charged as work performed by ReprForge unless it is regenerated.
    """

    _allow_large_csv_fields()
    corpus_ids: list[str] = []
    corpus_texts: list[str] = []
    corpus_images: list[bytes] = []
    document_pdf_ids: set[str] = set()
    with docs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            docs_path,
            reader.fieldnames,
            {"dataset_id", "pdf_id_x", "page_number", "transcription", "base64_str"},
        )
        for row_number, row in enumerate(reader, start=2):
            item_id = str(row["dataset_id"]).strip()
            pdf_id = str(row["pdf_id_x"]).strip()
            page_number = str(row["page_number"]).strip()
            if not item_id or not pdf_id or not page_number:
                raise ValueError(f"empty document identifier at CSV row {row_number}")
            if item_id != f"{pdf_id}_{page_number}":
                raise ValueError(
                    f"inconsistent document identifier {item_id!r} at row {row_number}"
                )
            corpus_ids.append(item_id)
            document_pdf_ids.add(pdf_id)
            corpus_texts.append(str(row["transcription"] or ""))
            if decode_images:
                try:
                    corpus_images.append(
                        base64.b64decode(row["base64_str"], validate=True)
                    )
                except ValueError as error:
                    raise ValueError(
                        f"invalid page image at CSV row {row_number}"
                    ) from error
            else:
                corpus_images.append(b"")

    if len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("IRPAPERS document identifiers are not unique")
    if expected_docs is not None and len(corpus_ids) != expected_docs:
        raise ValueError(
            f"expected {expected_docs} IRPAPERS pages, found {len(corpus_ids)}"
        )
    corpus_id_set = set(corpus_ids)

    query_ids: list[str] = []
    queries: list[str] = []
    qrels: dict[str, frozenset[str]] = {}
    query_pdf_ids: set[str] = set()
    query_metadata_mismatches: list[dict[str, str | int]] = []
    with queries_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            queries_path,
            reader.fieldnames,
            {"dataset_id", "pdf_id", "page_number", "question"},
        )
        for offset, row in enumerate(reader):
            gold_id = str(row["dataset_id"]).strip()
            pdf_id = str(row["pdf_id"]).strip()
            page_number = str(row["page_number"]).strip()
            if gold_id != f"{pdf_id}_{page_number}":
                # The public revision contains one known contradiction: query
                # CSV row 144 labels a Taobao paper page as dataset_id 16_7
                # while its pdf_id/page fields say 15_7.  The leaderboard links
                # by dataset_id, so retain that official qrel and surface the
                # disagreement rather than silently rewriting benchmark labels.
                query_metadata_mismatches.append(
                    {
                        "csv_row": offset + 2,
                        "dataset_id": gold_id,
                        "declared_pdf_page": f"{pdf_id}_{page_number}",
                    }
                )
            if gold_id not in corpus_id_set:
                raise ValueError(f"query target {gold_id!r} is absent from the corpus")
            query_id = f"q-{offset:04d}"
            query_ids.append(query_id)
            queries.append(str(row["question"]))
            qrels[query_id] = frozenset({gold_id})
            query_pdf_ids.add(pdf_id)

    if expected_queries is not None and len(query_ids) != expected_queries:
        raise ValueError(
            f"expected {expected_queries} IRPAPERS queries, found {len(query_ids)}"
        )
    return IRPapersData(
        query_ids=tuple(query_ids),
        queries=tuple(queries),
        corpus_ids=tuple(corpus_ids),
        corpus_texts=tuple(corpus_texts),
        corpus_images=tuple(corpus_images),
        qrels=qrels,
        metadata={
            "docs_path": str(docs_path.resolve()),
            "queries_path": str(queries_path.resolve()),
            "docs_sha256": sha256(docs_path),
            "queries_sha256": sha256(queries_path),
            "pages": len(corpus_ids),
            "papers": len(document_pdf_ids),
            "queries": len(query_ids),
            "query_source_papers": len(query_pdf_ids),
            "single_gold_page_per_query": True,
            "query_metadata_mismatches": query_metadata_mismatches,
            "supplied_transcription_is_dataset_input": True,
            "images_decoded": decode_images,
        },
    )


def _ordered_ids(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda item_id: (-float(scores[item_id]), item_id))


def recall_at_k(
    results: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, frozenset[str]],
    cutoffs: Sequence[int] = (1, 5, 20),
) -> dict[str, float]:
    if set(results) != set(qrels):
        missing = set(qrels) - set(results)
        extra = set(results) - set(qrels)
        raise ValueError(
            f"result/query mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        hits = 0
        for query_id, relevant in qrels.items():
            ranking = _ordered_ids(results[query_id])[:cutoff]
            hits += bool(set(ranking) & relevant)
        metrics[f"recall_{cutoff}"] = hits / len(qrels)
    return metrics


def minimum_action_oracle(
    policies: Mapping[int, Mapping[str, Mapping[str, float]]],
    qrels: Mapping[str, frozenset[str]],
    *,
    cutoff: int,
) -> dict[str, Any]:
    """Diagnostic upper bound from the cheapest successful per-query action.

    Policy keys are visual page-events charged to one query (for example 0,
    10, 20 and 50).  Relevance labels choose the action, so this is explicitly
    non-deployable; it quantifies whether learning an action-value estimator is
    worth attempting.
    """

    if cutoff <= 0 or not policies:
        raise ValueError("cutoff and policy set must be positive/non-empty")
    costs = sorted(policies)
    for results in policies.values():
        if set(results) != set(qrels):
            raise ValueError("oracle policy/query sets differ from qrels")
    selections: dict[int | None, int] = {}
    total_events = 0
    successes = 0
    for query_id, relevant in qrels.items():
        selected: int | None = None
        for cost in costs:
            ranking = _ordered_ids(policies[cost][query_id])[:cutoff]
            if set(ranking) & relevant:
                selected = cost
                break
        selections[selected] = selections.get(selected, 0) + 1
        if selected is not None:
            successes += 1
            total_events += selected
    return {
        "cutoff": cutoff,
        "success_rate": successes / len(qrels),
        "successful_queries": successes,
        "total_queries": len(qrels),
        "visual_page_events": total_events,
        "selection_counts": {
            ("unresolved" if cost is None else str(cost)): count
            for cost, count in sorted(
                selections.items(),
                key=lambda value: (
                    value[0] is None,
                    value[0] if value[0] is not None else 0,
                ),
            )
        },
        "uses_qrels": True,
        "deployable": False,
    }


def score_rows_to_results(
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    scores: np.ndarray,
    *,
    top_k: int,
) -> dict[str, dict[str, float]]:
    if scores.shape != (len(query_ids), len(corpus_ids)):
        raise ValueError("score matrix shape does not match identifiers")
    results: dict[str, dict[str, float]] = {}
    for query_id, row in zip(query_ids, scores, strict=True):
        order = sorted(
            range(len(corpus_ids)),
            key=lambda position: (-float(row[position]), corpus_ids[position]),
        )[:top_k]
        results[str(query_id)] = {
            corpus_ids[position]: float(top_k - rank)
            for rank, position in enumerate(order)
        }
    return results


def bm25_score_matrix(data: IRPapersData) -> tuple[np.ndarray, dict[str, int]]:
    state, posting_bytes, vocabulary_bytes = build_index(data.corpus_texts)
    scores = score_queries(state, data.queries, k1=1.2, b=0.75)
    return scores, {
        "logical_index_bytes": int(posting_bytes.sum()) + vocabulary_bytes,
        "postings_bytes": int(posting_bytes.sum()),
        "vocabulary_bytes": vocabulary_bytes,
    }


def _zscore(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-12)


def full_fusion_results(
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    locator_scores: np.ndarray,
    visual_scores: np.ndarray,
    *,
    top_k: int = 20,
) -> dict[str, dict[str, float]]:
    if locator_scores.shape != visual_scores.shape:
        raise ValueError("locator and visual score matrices differ in shape")
    fused = np.stack(
        [
            _zscore(locator_row) + _zscore(visual_row)
            for locator_row, visual_row in zip(
                locator_scores, visual_scores, strict=True
            )
        ]
    )
    return score_rows_to_results(
        query_ids,
        corpus_ids,
        fused,
        top_k=top_k,
    )


def candidate_fusion_replay(
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    locator_scores: np.ndarray,
    visual_scores: np.ndarray,
    *,
    candidate_k: int,
    top_k: int = 20,
) -> tuple[dict[str, dict[str, float]], dict[str, int | float]]:
    """Replay candidate-relative fusion from a complete visual score surface."""

    if locator_scores.shape != visual_scores.shape:
        raise ValueError("locator and visual score matrices differ in shape")
    if locator_scores.shape != (len(query_ids), len(corpus_ids)):
        raise ValueError("score matrices do not match identifiers")
    candidate_k = min(candidate_k, len(corpus_ids))
    results: dict[str, dict[str, float]] = {}
    touched: set[int] = set()
    for query_id, locator_row, visual_row in zip(
        query_ids, locator_scores, visual_scores, strict=True
    ):
        candidates = sorted(
            range(len(corpus_ids)),
            key=lambda position: (-float(locator_row[position]), corpus_ids[position]),
        )[:candidate_k]
        touched.update(candidates)
        fused = _zscore(locator_row[candidates]) + _zscore(visual_row[candidates])
        candidate_order = sorted(
            range(len(candidates)),
            key=lambda offset: (
                -float(fused[offset]),
                corpus_ids[candidates[offset]],
            ),
        )
        selected = set(candidates)
        tail = sorted(
            (position for position in range(len(corpus_ids)) if position not in selected),
            key=lambda position: (-float(locator_row[position]), corpus_ids[position]),
        )
        ranking = [candidates[offset] for offset in candidate_order]
        ranking.extend(tail)
        results[str(query_id)] = {
            corpus_ids[position]: float(top_k - rank)
            for rank, position in enumerate(ranking[:top_k])
        }
    events = len(query_ids) * candidate_k
    return results, {
        "candidate_k": candidate_k,
        "candidate_events": events,
        "resident_unique_pages": len(touched),
        "resident_corpus_fraction": len(touched) / len(corpus_ids),
        "transient_visual_page_events": events,
    }
