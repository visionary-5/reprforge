#!/usr/bin/env python3
"""Evaluate query-driven candidate fusion and its representation cost.

The deployable policy is intentionally small:

1. retrieve a text-only top-K candidate cohort;
2. obtain visual scores only for that cohort;
3. normalize text and visual scores within the same cohort;
4. rank the cohort by their equally weighted normalized sum.

Unlike raw score replacement, this gives scores from heterogeneous
representations a shared, query-local comparison domain.  The analyzer also
charges the visual pages touched by the policy in official query order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.intervention_utility import _ndcg_row, stable_query_split
from reprforge.progressive_oracle import (
    FrozenTrace,
    load_trace,
    mean_ndcg,
    rank_order,
    validate_pair,
)


DEFAULT_CANDIDATE_SIZES = (10, 20, 50, 100, 200)


def _zscore(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    scale = float(values.std())
    return centered / max(scale, 1e-12)


def _candidate_ndcg(
    text: FrozenTrace,
    visual: FrozenTrace,
    qrels: np.ndarray,
    text_order: np.ndarray,
    *,
    candidate_k: int,
    method: str,
    cutoff: int,
) -> np.ndarray:
    if candidate_k < cutoff:
        raise ValueError("candidate_k must be at least the evaluation cutoff")
    candidate_k = min(candidate_k, text.scores.shape[1])
    quality = np.zeros(text.scores.shape[0], dtype=np.float64)
    for query in range(text.scores.shape[0]):
        pages = text_order[query, :candidate_k]
        text_values = text.scores[query, pages]
        visual_values = visual.scores[query, pages]
        if method == "visual_rerank":
            fused = visual_values
        elif method == "zscore_sum":
            fused = _zscore(text_values) + _zscore(visual_values)
        elif method == "rrf":
            visual_order = np.lexsort((text.corpus_ids[pages], -visual_values))
            visual_rank = np.empty(candidate_k, dtype=np.int32)
            visual_rank[visual_order] = np.arange(candidate_k)
            fused = 1.0 / (60.0 + np.arange(candidate_k)) + 1.0 / (
                60.0 + visual_rank
            )
        else:
            raise ValueError(f"unknown candidate fusion method: {method}")
        full_scores = np.full(text.scores.shape[1], -np.inf, dtype=np.float64)
        full_scores[pages] = fused
        quality[query] = _ndcg_row(
            full_scores,
            qrels[query],
            text.corpus_ids,
            cutoff=cutoff,
        )
    return quality


def _representation_cost(
    text_order: np.ndarray,
    text: FrozenTrace,
    visual: FrozenTrace,
    *,
    candidate_k: int,
) -> dict[str, Any]:
    seen: set[int] = set()
    first_touch_costs: list[float] = []
    candidate_events = 0
    cache_hits = 0
    for query in range(text_order.shape[0]):
        cold_cost = 0.0
        for value in text_order[query, :candidate_k]:
            page = int(value)
            candidate_events += 1
            if page in seen:
                cache_hits += 1
                continue
            seen.add(page)
            cold_cost += float(visual.encode_ms[page])
        first_touch_costs.append(cold_cost)
    selected = np.asarray(sorted(seen), dtype=np.int64)
    full_encode_ms = float(visual.encode_ms.sum())
    encoded_ms = float(visual.encode_ms[selected].sum()) if len(selected) else 0.0
    encoded_bytes = int(visual.vector_bytes[selected].sum()) if len(selected) else 0
    combined_build_ms = float(text.index_total_ms) + encoded_ms
    return {
        "candidate_events": candidate_events,
        "unique_visual_pages": len(seen),
        "corpus_coverage_fraction": len(seen) / text_order.shape[1],
        "unbounded_cache_hit_fraction": (
            cache_hits / candidate_events if candidate_events else 0.0
        ),
        "visual_encode_ms_unique_pages": encoded_ms,
        "full_visual_encode_ms": full_encode_ms,
        "visual_encode_work_fraction": (
            encoded_ms / full_encode_ms if full_encode_ms else 0.0
        ),
        "text_plus_unique_visual_build_ms": combined_build_ms,
        "text_plus_unique_visual_vs_full_visual_build_ratio": (
            combined_build_ms / visual.index_total_ms
            if visual.index_total_ms
            else None
        ),
        "visual_vector_bytes_unique_pages": encoded_bytes,
        "first_touch_visual_ms_per_query": {
            "mean": float(np.mean(first_touch_costs)),
            "p50": float(np.percentile(first_touch_costs, 50)),
            "p95": float(np.percentile(first_touch_costs, 95)),
            "max": float(np.max(first_touch_costs)),
        },
    }


def analyze_candidate_fusion(
    text: FrozenTrace,
    visual: FrozenTrace,
    *,
    candidate_sizes: Sequence[int] = DEFAULT_CANDIDATE_SIZES,
    cutoff: int = 10,
    selected_candidate_k: int = 20,
) -> dict[str, Any]:
    qrels = validate_pair(text, visual)
    text_order = rank_order(text.scores, text.corpus_ids)
    splits = stable_query_split(text.query_ids.tolist())
    baseline = {
        "text_ndcg@10": mean_ndcg(text.scores, qrels, text.corpus_ids, cutoff=cutoff),
        "full_visual_ndcg@10": mean_ndcg(
            visual.scores,
            qrels,
            visual.corpus_ids,
            cutoff=cutoff,
        ),
        "text_index_ms": float(text.index_total_ms),
        "full_visual_index_ms": float(visual.index_total_ms),
        "full_visual_over_text_index_ratio": float(
            visual.index_total_ms / text.index_total_ms
        ),
    }
    methods: dict[str, list[dict[str, Any]]] = {
        "visual_rerank": [],
        "rrf": [],
        "zscore_sum": [],
    }
    quality_by_method: dict[str, dict[int, np.ndarray]] = {
        method: {} for method in methods
    }
    for method in methods:
        for value in sorted({int(size) for size in candidate_sizes}):
            quality = _candidate_ndcg(
                text,
                visual,
                qrels,
                text_order,
                candidate_k=value,
                method=method,
                cutoff=cutoff,
            )
            quality_by_method[method][value] = quality
            row: dict[str, Any] = {
                "candidate_k": value,
                "ndcg@10": float(quality.mean()),
                "by_split": {
                    split: float(quality[splits == split].mean())
                    for split in ("train", "validation", "test")
                },
            }
            if method == "zscore_sum":
                row["representation_cost"] = _representation_cost(
                    text_order,
                    text,
                    visual,
                    candidate_k=value,
                )
            methods[method].append(row)

    selected_rows = [
        row
        for row in methods["zscore_sum"]
        if row["candidate_k"] == selected_candidate_k
    ]
    if not selected_rows:
        raise ValueError("selected_candidate_k is absent from candidate_sizes")
    selected = selected_rows[0]
    selected_k = int(selected["candidate_k"])
    selected_quality = quality_by_method["zscore_sum"][selected_k]
    text_per_query = np.asarray(
        [
            _ndcg_row(text.scores[q], qrels[q], text.corpus_ids, cutoff=cutoff)
            for q in range(text.scores.shape[0])
        ]
    )
    visual_per_query = np.asarray(
        [
            _ndcg_row(visual.scores[q], qrels[q], visual.corpus_ids, cutoff=cutoff)
            for q in range(visual.scores.shape[0])
        ]
    )
    test = splits == "test"
    test_text = float(text_per_query[test].mean())
    test_visual = float(visual_per_query[test].mean())
    test_policy = float(selected_quality[test].mean())
    denominator = test_visual - test_text
    return {
        "schema_version": 1,
        "algorithm": {
            "name": "candidate-relative normalized fusion",
            "candidate_generation": "text top-K",
            "visual_action": "encode/score candidate pages only",
            "fusion": "z(text scores within cohort) + z(visual scores within cohort)",
            "selection": f"fixed K={selected_candidate_k}; curves are ablations",
            "test_labels_visible": False,
        },
        "workload": {
            "queries": int(text.scores.shape[0]),
            "corpus": int(text.scores.shape[1]),
        },
        "baseline": baseline,
        "methods": methods,
        "selected_policy": {
            "candidate_k": selected_k,
            "validation_ndcg@10": selected["by_split"]["validation"],
            "test_ndcg@10": test_policy,
            "test_text_ndcg@10": test_text,
            "test_full_visual_ndcg@10": test_visual,
            "test_improvement_over_best_single_representation": (
                test_policy - max(test_text, test_visual)
            ),
            "test_full_visual_gain_retained": (
                float((test_policy - test_text) / denominator)
                if denominator > 1e-12
                else None
            ),
            "representation_cost": selected["representation_cost"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-trace", type=Path, required=True)
    parser.add_argument("--visual-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_SIZES),
    )
    parser.add_argument("--selected-candidate-k", type=int, default=20)
    args = parser.parse_args()
    text = load_trace(args.text_trace)
    visual = load_trace(args.visual_trace)
    result = analyze_candidate_fusion(
        text,
        visual,
        candidate_sizes=args.candidate_sizes,
        selected_candidate_k=args.selected_candidate_k,
    )
    result["source"] = {
        "text_runtime_sha256": text.manifest["runtime_sha256"],
        "visual_runtime_sha256": visual.manifest["runtime_sha256"],
        "oracle_labels_sha256": text.manifest["oracle_labels_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
