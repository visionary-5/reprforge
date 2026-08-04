#!/usr/bin/env python3
"""Materialize a complete late-interaction score surface from an embedding bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_token_witness_index import _embedding_views, _score_surface
from reprforge.heterogeneous_index import _load_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("cpu threads must be positive")

    import torch

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)

    corpus_ids, document_vectors, document_offsets = _load_shard(
        args.bank / "routes" / args.route
    )
    query_ids, query_vectors, query_offsets = _load_shard(args.bank / "queries")
    documents = _embedding_views(document_vectors, document_offsets)
    queries = _embedding_views(query_vectors, query_offsets)
    selections = tuple(
        np.arange(len(document), dtype=np.int32) for document in documents
    )
    scores, score_ms = _score_surface(
        queries,
        documents,
        selections,
        device=args.device,
        batch_size=args.batch_size,
    )
    vector_counts = np.diff(document_offsets).astype(np.int32)
    query_vector_counts = np.diff(query_offsets).astype(np.int32)
    vector_bytes = (
        vector_counts.astype(np.int64)
        * int(document_vectors.shape[1])
        * int(document_vectors.dtype.itemsize)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        query_ids=np.asarray(query_ids),
        corpus_ids=np.asarray(corpus_ids),
        scores=scores.astype(np.float32),
        vector_bytes=vector_bytes,
        vector_counts=vector_counts,
        query_vector_counts=query_vector_counts,
        encode_ms=np.zeros(len(corpus_ids), dtype=np.float32),
        index_total_ms=np.asarray(0.0, dtype=np.float64),
        model_load_ms=np.asarray(0.0, dtype=np.float64),
    )
    print(
        json.dumps(
            {
                "queries": len(query_ids),
                "corpus": len(corpus_ids),
                "route": args.route,
                "score_ms": score_ms,
                "cpu_threads": args.cpu_threads,
                "tokens": int(vector_counts.sum()),
                "vector_bytes": int(vector_bytes.sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
