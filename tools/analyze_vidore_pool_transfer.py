#!/usr/bin/env python3
"""Compare one full-corpus pooled ViDoRe trace with text and full visual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reprforge.heterogeneity_atlas import paired_bootstrap_ci, query_metrics


def _relevance(labels: np.lib.npyio.NpzFile, query_count: int) -> tuple[dict, ...]:
    rows = [dict() for _ in range(query_count)]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        rows[int(query)][int(corpus)] = float(value)
    return tuple(rows)


def _agreement(left: np.ndarray, right: np.ndarray, k: int) -> dict:
    positions = np.arange(left.shape[1])
    left_order = np.stack([np.lexsort((positions, -row))[:k] for row in left])
    right_order = np.stack([np.lexsort((positions, -row))[:k] for row in right])
    return {
        "mean_set_overlap": float(
            np.mean(
                [
                    len(set(left_row) & set(right_row)) / k
                    for left_row, right_row in zip(
                        left_order, right_order, strict=True
                    )
                ]
            )
        ),
        "mean_exact_position_agreement": float(np.mean(left_order == right_order)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--text-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pooled = np.load(args.pooled_runtime, allow_pickle=False)
    full = np.load(args.full_runtime, allow_pickle=False)
    text = np.load(args.text_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    for other in (full, text):
        if not np.array_equal(pooled["query_ids"], other["query_ids"]):
            raise ValueError("query IDs are not aligned")
        if not np.array_equal(pooled["corpus_ids"], other["corpus_ids"]):
            raise ValueError("corpus IDs are not aligned")
    relevance = _relevance(labels, len(pooled["query_ids"]))
    metrics = {
        name: query_metrics(runtime["scores"], relevance, ks=(5, 10, 100))
        for name, runtime in (("pooled", pooled), ("full", full), ("text", text))
    }
    report = {
        "queries": len(pooled["query_ids"]),
        "corpus": len(pooled["corpus_ids"]),
        "metrics": {
            name: {metric: float(values.mean()) for metric, values in result.items()}
            for name, result in metrics.items()
        },
        "pooled_vs_full": {
            "ndcg_at_5": paired_bootstrap_ci(
                metrics["pooled"]["ndcg_at_5"], metrics["full"]["ndcg_at_5"]
            ),
            "ndcg_at_10": paired_bootstrap_ci(
                metrics["pooled"]["ndcg_at_10"], metrics["full"]["ndcg_at_10"]
            ),
            "top_5": _agreement(pooled["scores"], full["scores"], 5),
            "top_10": _agreement(pooled["scores"], full["scores"], 10),
        },
        "physical": {
            "pooled_vector_bytes": int(pooled["vector_bytes"].sum()),
            "full_vector_bytes": int(full["vector_bytes"].sum()),
            "pooled_byte_fraction": float(
                pooled["vector_bytes"].sum() / full["vector_bytes"].sum()
            ),
            "pooled_vectors": int(pooled["vector_counts"].sum()),
            "full_vectors": int(full["vector_counts"].sum()),
            "pooled_index_ms": float(pooled["index_total_ms"]),
            "full_index_ms": float(full["index_total_ms"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
