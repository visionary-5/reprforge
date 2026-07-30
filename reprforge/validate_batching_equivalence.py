#!/usr/bin/env python3
"""Validate fixed-document and token-budget MaxSim batching equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from reprforge.heterogeneous_index import TorchMaxSimRuntime, load_query_bank


def compare_scores(
    left: np.ndarray,
    right: np.ndarray,
    *,
    item_ids: Sequence[str],
    top_k: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict:
    if left.shape != right.shape or left.shape != (len(item_ids),):
        raise ValueError("score vectors and item identifiers differ")
    difference = np.abs(left - right)
    denominator = np.maximum(np.abs(left), 1e-12)
    left_ranking = sorted(
        range(len(item_ids)),
        key=lambda index: (-float(left[index]), item_ids[index]),
    )[:top_k]
    right_ranking = sorted(
        range(len(item_ids)),
        key=lambda index: (-float(right[index]), item_ids[index]),
    )[:top_k]
    return {
        "all_close": bool(
            np.allclose(
                left,
                right,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        ),
        "max_absolute_error": float(difference.max(initial=0.0)),
        "max_relative_error": float(
            (difference / denominator).max(initial=0.0)
        ),
        "top_k_equal": left_ranking == right_ranking,
    }


def validate(
    *,
    index: Path,
    query_bank: Path,
    device: str,
    document_batch_size: int,
    token_batch_budget: int,
    top_k: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict:
    query_ids, queries = load_query_bank(query_bank)
    fixed = TorchMaxSimRuntime(
        index,
        device=device,
        document_batch_size=document_batch_size,
    )
    token = TorchMaxSimRuntime(
        index,
        device=device,
        document_batch_size=document_batch_size,
        token_batch_budget=token_batch_budget,
    )
    per_query = {}
    for query_id, query in zip(query_ids, queries, strict=True):
        per_query[query_id] = compare_scores(
            fixed.score(query),
            token.score(query),
            item_ids=fixed.item_ids,
            top_k=top_k,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    return {
        "index": str(index),
        "queries": len(query_ids),
        "items": len(fixed.item_ids),
        "fixed_execution_batches": fixed.execution_batch_count,
        "token_execution_batches": token.execution_batch_count,
        "token_batch_budget": token_batch_budget,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "all_scores_close": all(row["all_close"] for row in per_query.values()),
        "all_top_k_equal": all(row["top_k_equal"] for row in per_query.values()),
        "max_absolute_error": max(
            row["max_absolute_error"] for row in per_query.values()
        ),
        "max_relative_error": max(
            row["max_relative_error"] for row in per_query.values()
        ),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query-bank", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--document-batch-size", type=int, default=64)
    parser.add_argument("--token-batch-budget", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        index=args.index,
        query_bank=args.query_bank,
        device=args.device,
        document_batch_size=args.document_batch_size,
        token_batch_budget=args.token_batch_budget,
        top_k=args.top_k,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
