#!/usr/bin/env python3
"""Verify online cohort execution against frozen offline score surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.cohort_compiler import CohortCompiler
from reprforge.mmdocir_route_runner import EncodedBatch
from reprforge.progressive_oracle import FrozenTrace, load_trace, rank_order
from reprforge.vidore_local_eval import _component_paths, _read_rows


def _zscore(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - float(values.mean())
    return centered / max(float(values.std()), 1e-12)


class FrozenVisualBackend:
    """Replay visual scores while exercising the real online orchestration."""

    def __init__(
        self,
        trace: FrozenTrace,
        query_texts: Sequence[str],
    ) -> None:
        self.trace = trace
        self.query_texts = tuple(query_texts)
        self.query_cursor = 0
        self.image_calls: list[tuple[int, ...]] = []

    def encode_queries(self, queries: Sequence[str]) -> EncodedBatch:
        positions = tuple(
            range(self.query_cursor, self.query_cursor + len(queries))
        )
        expected = self.query_texts[
            self.query_cursor : self.query_cursor + len(queries)
        ]
        if tuple(queries) != expected:
            raise ValueError("query replay order differs from the frozen trace")
        self.query_cursor += len(queries)
        return EncodedBatch(
            embeddings=positions,
            encode_ms=tuple(0.0 for _ in positions),
        )

    def encode_images(self, images: Sequence[Any]) -> EncodedBatch:
        positions = tuple(int(value) for value in images)
        self.image_calls.append(positions)
        return EncodedBatch(
            embeddings=positions,
            encode_ms=tuple(
                float(self.trace.encode_ms[position]) for position in positions
            ),
        )

    def score(
        self,
        queries: Sequence[Any],
        documents: Sequence[Any],
    ) -> Sequence[Sequence[float]]:
        return [
            [
                float(self.trace.scores[int(query)][int(document)])
                for document in documents
            ]
            for query in queries
        ]


def _expected_order(
    text: FrozenTrace,
    visual: FrozenTrace,
    *,
    candidate_k: int,
    top_k: int,
) -> list[list[str]]:
    text_order = rank_order(text.scores, text.corpus_ids)
    expected: list[list[str]] = []
    for query in range(text.scores.shape[0]):
        candidates = text_order[query, :candidate_k]
        fused = _zscore(text.scores[query, candidates]) + _zscore(
            visual.scores[query, candidates]
        )
        fused_offsets = np.lexsort(
            (text.corpus_ids[candidates], -fused)
        )
        candidate_set = {int(value) for value in candidates}
        ordered = [
            str(text.corpus_ids[int(candidates[offset])])
            for offset in fused_offsets
        ]
        ordered.extend(
            str(text.corpus_ids[int(position)])
            for position in text_order[query]
            if int(position) not in candidate_set
        )
        expected.append(ordered[:top_k])
    return expected


def verify_trace_parity(
    data_root: Path,
    bm25_root: Path,
    visual_root: Path,
    *,
    language: str,
    candidate_k: int,
    top_k: int,
    request_batch_size: int,
    cache_policy: str,
) -> dict[str, Any]:
    text = load_trace(bm25_root)
    visual = load_trace(visual_root)
    if not np.array_equal(text.query_ids, visual.query_ids):
        raise ValueError("BM25 and visual trace query identifiers differ")
    if not np.array_equal(text.corpus_ids, visual.corpus_ids):
        raise ValueError("BM25 and visual trace corpus identifiers differ")

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
        str(row["corpus_id"]): str(row["markdown"])
        for row in corpus_rows
    }
    query_texts = [query_lookup[str(value)] for value in text.query_ids]
    corpus_texts = [corpus_lookup[str(value)] for value in text.corpus_ids]
    backend = FrozenVisualBackend(visual, query_texts)
    compiler = CohortCompiler(
        corpus_ids=[str(value) for value in text.corpus_ids],
        corpus_texts=corpus_texts,
        corpus_images=list(range(len(corpus_texts))),
        backend=backend,
        candidate_k=candidate_k,
        top_k=top_k,
        request_batch_size=request_batch_size,
        cache_policy=cache_policy,
    )
    execution = compiler.execute_batch(
        [str(value) for value in text.query_ids],
        query_texts,
    )
    expected = _expected_order(
        text,
        visual,
        candidate_k=candidate_k,
        top_k=top_k,
    )
    mismatches: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for offset, query_id in enumerate(text.query_ids):
        actual = list(execution.results[str(query_id)])
        digest.update(str(query_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update("\n".join(actual).encode("utf-8"))
        digest.update(b"\n")
        if actual != expected[offset]:
            mismatches.append(
                {
                    "query_id": str(query_id),
                    "first_actual": actual[:10],
                    "first_expected": expected[offset][:10],
                }
            )
    return {
        "schema_version": 1,
        "query_count": len(text.query_ids),
        "corpus_count": len(text.corpus_ids),
        "candidate_k": candidate_k,
        "top_k": top_k,
        "request_batch_size": request_batch_size,
        "cache_policy": cache_policy,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:5],
        "ranking_sha256": digest.hexdigest(),
        "online_offline_rank_parity": not mismatches,
        "execution_metrics": execution.metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bm25-trace", type=Path, required=True)
    parser.add_argument("--visual-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="english")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--request-batch-size", type=int, default=8)
    parser.add_argument(
        "--cache-policy",
        choices=["none", "resident"],
        default="none",
    )
    args = parser.parse_args()
    result = verify_trace_parity(
        args.data_root,
        args.bm25_trace,
        args.visual_trace,
        language=args.language,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        request_batch_size=args.request_batch_size,
        cache_policy=args.cache_policy,
    )
    if not result["online_offline_rank_parity"]:
        raise RuntimeError(json.dumps(result["first_mismatches"], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
