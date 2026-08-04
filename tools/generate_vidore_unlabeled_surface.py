#!/usr/bin/env python3
"""Generate a ViDoRe embedding bank and score surface without opening qrels."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from reprforge.vidore_local_eval import (
    _component_paths,
    _component_sha256,
    _decode_image,
    _read_rows,
)
from reprforge.vidore_pipeline import ReprForgeViDoRePipeline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--embedding-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="english")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument(
        "--embedding-storage-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    args = parser.parse_args()

    query_paths = _component_paths(args.data_root, "queries")
    corpus_paths = _component_paths(args.data_root, "corpus")
    began = time.perf_counter()
    query_rows = _read_rows(query_paths, ("query_id", "query", "language"))
    query_rows = [
        row for row in query_rows if str(row["language"]) == args.language
    ]
    corpus_rows = _read_rows(
        corpus_paths,
        ("corpus_id", "image", "markdown"),
    )
    if not query_rows or not corpus_rows:
        raise ValueError("unlabeled ViDoRe selection must be non-empty")
    query_ids = [str(row["query_id"]) for row in query_rows]
    queries = [str(row["query"]) for row in query_rows]
    corpus_ids = [str(row["corpus_id"]) for row in corpus_rows]
    corpus_images = [_decode_image(row["image"]) for row in corpus_rows]
    corpus_texts = [str(row["markdown"]) for row in corpus_rows]
    load_seconds = time.perf_counter() - began

    pipeline = ReprForgeViDoRePipeline(
        base_model=args.base_model,
        adapter=args.adapter,
        mode="visual",
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
        top_k=100,
        capture_score_trace=True,
    )
    began = time.perf_counter()
    pipeline.index(
        corpus_ids,
        corpus_images,
        corpus_texts,
        dataset_name=args.dataset_name,
    )
    index_seconds = time.perf_counter() - began
    began = time.perf_counter()
    pipeline.search(query_ids, queries)
    search_seconds = time.perf_counter() - began
    trace = pipeline.export_score_trace(query_ids)

    args.runtime.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.runtime,
        query_ids=trace["query_ids"],
        corpus_ids=trace["corpus_ids"],
        scores=trace["scores"],
        vector_bytes=trace["vector_bytes"],
        vector_counts=trace["vector_counts"],
        query_vector_counts=trace["query_vector_counts"],
        encode_ms=trace["encode_ms"],
        index_total_ms=trace["index_total_ms"],
        model_load_ms=trace["model_load_ms"],
    )
    bank = pipeline.write_embedding_bank(
        args.embedding_bank,
        query_ids=query_ids,
        queries=queries,
        storage_dtype=args.embedding_storage_dtype,
    )
    report = {
        "schema_version": 1,
        "protocol": "qrel-free-compression-risk-2026-08-04",
        "stage": "pre-qrel-surface-generation",
        "dataset": args.dataset_name,
        "mode": "visual",
        "language": args.language,
        "queries": len(query_ids),
        "corpus": len(corpus_ids),
        "data_load_seconds": load_seconds,
        "index_seconds": index_seconds,
        "search_seconds": search_seconds,
        "qrel_directory_scanned": False,
        "qrels_loaded": False,
        "source_sha256": {
            "queries": _component_sha256(query_paths),
            "corpus": _component_sha256(corpus_paths),
        },
        "runtime": {
            "path": str(args.runtime),
            "sha256": _sha256(args.runtime),
            "score_shape": list(trace["scores"].shape),
        },
        "embedding_bank": bank,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
