#!/usr/bin/env python3
"""Build a cheap Markdown BM25 locator trace for official ViDoRe data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.progressive_oracle import FrozenTrace, load_trace
from reprforge.vidore_local_eval import _component_paths, _read_rows


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def bm25_scores(
    documents: Sequence[str],
    queries: Sequence[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return query-document BM25 scores and a logical posting-byte count."""

    state, posting_bytes, vocabulary_bytes = _build_index(documents)
    return (
        _score_queries(state, queries, k1=k1, b=b),
        posting_bytes,
        vocabulary_bytes,
    )


def _build_index(
    documents: Sequence[str],
) -> tuple[
    tuple[dict[str, list[tuple[int, int]]], np.ndarray, float, int],
    np.ndarray,
    int,
]:
    tokenized = [tokenize(value) for value in documents]
    lengths = np.asarray([len(value) for value in tokenized], dtype=np.float64)
    average_length = max(float(lengths.mean()), 1.0)
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    posting_bytes = np.zeros(len(documents), dtype=np.int64)
    for position, tokens in enumerate(tokenized):
        for term, frequency in Counter(tokens).items():
            postings[term].append((position, frequency))
            # Logical accounting: uint32 doc id + uint32 term frequency.
            posting_bytes[position] += 8
    vocabulary_bytes = sum(len(term.encode("utf-8")) for term in postings)
    return (
        (dict(postings), lengths, average_length, len(documents)),
        posting_bytes,
        vocabulary_bytes,
    )


def _score_queries(
    state: tuple[dict[str, list[tuple[int, int]]], np.ndarray, float, int],
    queries: Sequence[str],
    *,
    k1: float,
    b: float,
) -> np.ndarray:
    postings, lengths, average_length, document_count = state
    scores = np.zeros((len(queries), document_count), dtype=np.float32)
    for query_position, query in enumerate(queries):
        for term, query_frequency in Counter(tokenize(query)).items():
            values = postings.get(term)
            if not values:
                continue
            document_frequency = len(values)
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for document_position, frequency in values:
                denominator = frequency + k1 * (
                    1.0 - b + b * lengths[document_position] / average_length
                )
                scores[query_position, document_position] += float(
                    query_frequency
                    * inverse_document_frequency
                    * frequency
                    * (k1 + 1.0)
                    / denominator
                )
    return scores


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bm25_trace(
    data_root: Path,
    visual: FrozenTrace,
    output: Path,
    *,
    language: str = "english",
) -> dict[str, Any]:
    query_rows = _read_rows(
        _component_paths(data_root, "queries"),
        ("query_id", "query", "language"),
    )
    corpus_rows = _read_rows(
        _component_paths(data_root, "corpus"),
        ("corpus_id", "markdown"),
    )
    query_lookup = {
        str(row["query_id"]): str(row["query"])
        for row in query_rows
        if str(row["language"]) == language
    }
    corpus_lookup = {
        str(row["corpus_id"]): str(row["markdown"]) for row in corpus_rows
    }
    missing_queries = [
        str(value) for value in visual.query_ids if str(value) not in query_lookup
    ]
    missing_corpus = [
        str(value) for value in visual.corpus_ids if str(value) not in corpus_lookup
    ]
    if missing_queries or missing_corpus:
        raise ValueError(
            f"parquet/visual trace mismatch: {len(missing_queries)} queries, "
            f"{len(missing_corpus)} corpus pages missing"
        )
    queries = [query_lookup[str(value)] for value in visual.query_ids]
    documents = [corpus_lookup[str(value)] for value in visual.corpus_ids]
    began = time.perf_counter()
    state, posting_bytes, vocabulary_bytes = _build_index(documents)
    build_ms = (time.perf_counter() - began) * 1000.0
    began = time.perf_counter()
    scores = _score_queries(state, queries, k1=1.2, b=0.75)
    query_score_ms = (time.perf_counter() - began) * 1000.0

    output.mkdir(parents=True, exist_ok=True)
    runtime_path = output / "runtime.npz"
    labels_path = output / "oracle-labels.npz"
    np.savez_compressed(
        runtime_path,
        query_ids=visual.query_ids,
        corpus_ids=visual.corpus_ids,
        scores=scores,
        vector_bytes=posting_bytes,
        encode_ms=np.zeros(len(documents), dtype=np.float64),
        index_total_ms=np.asarray(build_ms, dtype=np.float64),
    )
    np.savez_compressed(
        labels_path,
        query_positions=visual.label_query,
        corpus_positions=visual.label_corpus,
        relevance=visual.relevance,
    )
    manifest = {
        "schema_version": 1,
        "mode": "text",
        "representation": "markdown-bm25",
        "runtime_file": runtime_path.name,
        "runtime_sha256": _sha256(runtime_path),
        "oracle_labels_file": labels_path.name,
        "oracle_labels_sha256": _sha256(labels_path),
        "query_count": len(queries),
        "corpus_count": len(documents),
        "score_shape": list(scores.shape),
        "index_total_ms": build_ms,
        "query_score_total_ms": query_score_ms,
        "per_item_encode_ms_sum": 0.0,
        "logical_posting_bytes": int(posting_bytes.sum()),
        "logical_vocabulary_bytes": vocabulary_bytes,
        "labels_are_runtime_visible": False,
        "official_upstream_commit": visual.manifest["official_upstream_commit"],
        "source_sha256": visual.manifest["source_sha256"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--visual-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="english")
    args = parser.parse_args()
    result = build_bm25_trace(
        args.data_root,
        load_trace(args.visual_trace),
        args.output,
        language=args.language,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
