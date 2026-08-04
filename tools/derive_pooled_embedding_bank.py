#!/usr/bin/env python3
"""Derive a persisted hierarchical-token-pooled bank from a full bank."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from reprforge.heterogeneous_index import _load_shard, write_embedding_bank


def _views(vectors: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(
        vectors[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(len(offsets) - 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-bank", type=Path, required=True)
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    parser.add_argument("--pool-factor", type=int, default=9)
    parser.add_argument(
        "--storage-dtype", choices=("float16", "float32"), default="float16"
    )
    args = parser.parse_args()
    if args.pool_factor < 2:
        raise ValueError("pool factor must be at least two")
    item_ids, vectors, offsets = _load_shard(
        args.full_bank / "routes" / "image"
    )
    query_ids, query_vectors, query_offsets = _load_shard(
        args.full_bank / "queries"
    )
    documents = _views(vectors, offsets)
    queries = _views(query_vectors, query_offsets)
    import torch
    from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

    pooler = HierarchicalTokenPooler()
    pooled = []
    began = time.perf_counter()
    for document in documents:
        result = pooler.pool_embeddings(
            [torch.tensor(np.asarray(document), dtype=torch.float32)],
            pool_factor=args.pool_factor,
            return_dict=False,
            num_workers=1,
        )
        pooled.append(result[0])
    pooling_ms = (time.perf_counter() - began) * 1000.0
    manifest = write_embedding_bank(
        args.output_bank,
        item_ids=item_ids,
        route_embeddings={f"image-pool-{args.pool_factor}": pooled},
        query_ids=query_ids,
        query_embeddings=queries,
        storage_dtype=args.storage_dtype,
    )
    result = {
        "pool_factor": args.pool_factor,
        "pooling_ms": pooling_ms,
        "full_tokens": int(sum(len(value) for value in documents)),
        "pooled_tokens": int(sum(len(value) for value in pooled)),
        "token_fraction": float(
            sum(len(value) for value in pooled)
            / sum(len(value) for value in documents)
        ),
        "bank_manifest": manifest,
    }
    args.output_result.parent.mkdir(parents=True, exist_ok=True)
    args.output_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("pooling_ms", "full_tokens", "pooled_tokens", "token_fraction")}, indent=2))


if __name__ == "__main__":
    main()
