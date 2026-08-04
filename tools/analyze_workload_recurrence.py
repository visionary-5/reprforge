#!/usr/bin/env python3
"""Measure whether a retrieval benchmark supports workload-recurrence plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _fold(value: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    runtime = np.load(args.runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    query_count = len(runtime["query_ids"])
    relevant = [set() for _ in range(query_count)]
    for query, corpus in zip(
        labels["query_positions"], labels["corpus_positions"], strict=True
    ):
        relevant[int(query)].add(int(corpus))
    fold_ids = np.asarray(
        [_fold(str(value), args.folds) for value in runtime["query_ids"]],
        dtype=np.int16,
    )
    frequency = Counter(item for values in relevant for item in values)
    fold_reports = []
    for fold in range(args.folds):
        fit_queries = np.flatnonzero(fold_ids != fold)
        test_queries = np.flatnonzero(fold_ids == fold)
        fit_items = set().union(*(relevant[index] for index in fit_queries))
        test_pairs = [
            item for index in test_queries for item in relevant[int(index)]
        ]
        fold_reports.append(
            {
                "fold": fold,
                "fit_queries": int(len(fit_queries)),
                "test_queries": int(len(test_queries)),
                "test_relevance_pairs": len(test_pairs),
                "pair_weighted_seen_item_fraction": (
                    sum(item in fit_items for item in test_pairs) / len(test_pairs)
                    if test_pairs
                    else 0.0
                ),
                "queries_with_any_seen_relevant_item_fraction": float(
                    np.mean(
                        [
                            bool(relevant[int(index)] & fit_items)
                            for index in test_queries
                        ]
                    )
                ),
                "queries_with_all_relevant_items_seen_fraction": float(
                    np.mean(
                        [
                            relevant[int(index)] <= fit_items
                            for index in test_queries
                        ]
                    )
                ),
            }
        )
    histogram = Counter(frequency.values())
    result = {
        "queries": query_count,
        "corpus": len(runtime["corpus_ids"]),
        "relevant_items": len(frequency),
        "relevance_pairs": int(sum(frequency.values())),
        "relevant_item_query_frequency_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "relevant_items_seen_in_multiple_queries_fraction": (
            sum(value > 1 for value in frequency.values()) / len(frequency)
            if frequency
            else 0.0
        ),
        "five_fold_query_hash_crossfit": fold_reports,
        "mean_pair_weighted_seen_item_fraction": float(
            np.mean(
                [value["pair_weighted_seen_item_fraction"] for value in fold_reports]
            )
        ),
        "mean_queries_with_any_seen_relevant_item_fraction": float(
            np.mean(
                [
                    value["queries_with_any_seen_relevant_item_fraction"]
                    for value in fold_reports
                ]
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
