#!/usr/bin/env python3
"""Build a qrel-free query-order manifest from a frozen BM25 runtime trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.cohort_frontier_scheduler import (
    frontier_reuse_order,
    static_popularity_order,
)
from reprforge.progressive_oracle import rank_order


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--scheduler",
        choices=["fifo", "static_popularity", "frontier_reuse"],
        required=True,
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    trace_manifest_path = args.bm25_trace / "manifest.json"
    trace_manifest = json.loads(trace_manifest_path.read_text())
    runtime_path = args.bm25_trace / trace_manifest["runtime_file"]
    runtime_sha256 = _sha256(runtime_path)
    if runtime_sha256 != trace_manifest["runtime_sha256"]:
        raise ValueError("BM25 runtime trace digest mismatch")
    with np.load(runtime_path, allow_pickle=False) as runtime:
        query_ids = np.asarray(runtime["query_ids"])
        corpus_ids = np.asarray(runtime["corpus_ids"])
        scores = np.asarray(runtime["scores"], dtype=np.float64)
    cohorts = rank_order(scores, corpus_ids)[:, : args.candidate_k].tolist()
    if args.scheduler == "fifo":
        order = list(range(len(cohorts)))
    elif args.scheduler == "static_popularity":
        order = static_popularity_order(cohorts)
    else:
        order = frontier_reuse_order(cohorts, batch_size=args.batch_size)
    payload = {
        "schema_version": 1,
        "dataset": args.dataset_name,
        "scheduler": args.scheduler,
        "candidate_k": args.candidate_k,
        "request_batch_size": args.batch_size,
        "query_ids": [str(query_ids[index]) for index in order],
        "qrels_loaded": False,
        "source": {
            "bm25_runtime_sha256": runtime_sha256,
            "bm25_manifest_sha256": _sha256(trace_manifest_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
