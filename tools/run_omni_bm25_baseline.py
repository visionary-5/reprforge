#!/usr/bin/env python3
"""Run a deterministic OCR-text BM25 baseline on an exported Omni domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.bm25 import build_index, score_queries
from tools.analyze_omni_pair import _query_metrics, load_qrels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"empty JSONL input: {path}")
    return rows


def run(
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    output_root: Path,
    top_k: int,
) -> dict[str, Any]:
    corpus_rows = _read_jsonl(corpus_path)
    query_rows = _read_jsonl(queries_path)
    corpus_ids = [str(row["docid"]) for row in corpus_rows]
    corpus_texts = [str(row.get("text", "")) for row in corpus_rows]
    query_ids = [str(row["query_id"]) for row in query_rows]
    queries = [str(row["query"]) for row in query_rows]
    if len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("corpus IDs are not unique")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query IDs are not unique")
    qrels = load_qrels(qrels_path)
    if set(query_ids) != set(qrels):
        raise ValueError("query and qrel ID sets differ")
    if top_k <= 0 or top_k > len(corpus_ids):
        raise ValueError("top_k must be within the corpus size")

    build_started = time.perf_counter()
    state, posting_bytes, vocabulary_bytes = build_index(corpus_texts)
    build_seconds = time.perf_counter() - build_started
    score_started = time.perf_counter()
    scores = score_queries(state, queries, k1=1.2, b=0.75)
    score_seconds = time.perf_counter() - score_started

    rankings: dict[str, list[str]] = {}
    ranking_rows: list[tuple[str, str, float]] = []
    for query_id, row in zip(query_ids, scores, strict=True):
        order = sorted(
            range(len(corpus_ids)),
            key=lambda position: (-float(row[position]), corpus_ids[position]),
        )[:top_k]
        rankings[query_id] = [corpus_ids[position] for position in order]
        ranking_rows.extend(
            (query_id, corpus_ids[position], float(row[position]))
            for position in order
        )

    output_root.mkdir(parents=True, exist_ok=False)
    ranking_path = output_root / "ranking.txt"
    with ranking_path.open("w", encoding="utf-8") as handle:
        for query_id, doc_id, score in ranking_rows:
            handle.write(f"{query_id}\t{doc_id}\t{score:.9g}\n")

    ks = tuple(sorted(set(k for k in (1, 5, 10, top_k) if k <= top_k)))
    per_query = [_query_metrics(rankings[qid], qrels[qid], ks) for qid in query_ids]
    metrics = {
        metric: float(np.mean([row[metric] for row in per_query]))
        for metric in per_query[0]
    }
    report = {
        "schema_version": 1,
        "protocol": "omni-exported-ocr-bm25-2026-08-06",
        "retrieval_uses_qrels": False,
        "qrels_used_for_final_evaluation_only": True,
        "parameters": {"b": 0.75, "k1": 1.2, "top_k": top_k},
        "dataset": {
            "corpus_pages": len(corpus_ids),
            "queries": len(query_ids),
            "corpus_sha256": _sha256(corpus_path),
            "queries_sha256": _sha256(queries_path),
            "qrels_sha256": _sha256(qrels_path),
            "text_source": "dataset-supplied OCR text in exported corpus JSONL",
        },
        "cost": {
            "build_wall_seconds": build_seconds,
            "score_all_queries_wall_seconds": score_seconds,
            "mean_score_seconds_per_query": score_seconds / len(query_ids),
            "logical_index_bytes": int(posting_bytes.sum()) + vocabulary_bytes,
            "postings_bytes": int(posting_bytes.sum()),
            "vocabulary_bytes": vocabulary_bytes,
        },
        "metrics": metrics,
        "artifacts": {
            "ranking": {"path": str(ranking_path), "sha256": _sha256(ranking_path)}
        },
    }
    report_path = output_root / "results.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cost": report["cost"], "metrics": metrics}, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    run(
        corpus_path=args.corpus,
        queries_path=args.queries,
        qrels_path=args.qrels,
        output_root=args.output_root,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
