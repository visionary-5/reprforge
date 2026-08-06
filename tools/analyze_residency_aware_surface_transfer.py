#!/usr/bin/env python3
"""Transfer the frozen residency-aware policy to BM25/ColPali score surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.candidate_fusion import _zscore
from reprforge.intervention_utility import _ndcg_row
from tools.analyze_residency_aware_cascade import _sha256, analyze


def load_surface_domain(name: str, root: Path) -> dict[str, Any]:
    text_path = root / "bm25" / "runtime.npz"
    visual_path = root / "visual" / "runtime.npz"
    labels_path = root / "bm25" / "oracle-labels.npz"
    with np.load(text_path, allow_pickle=False) as source:
        text = {key: source[key] for key in source.files}
    with np.load(visual_path, allow_pickle=False) as source:
        visual = {key: source[key] for key in source.files}
    with np.load(labels_path, allow_pickle=False) as source:
        labels = {key: source[key] for key in source.files}
    if not np.array_equal(text["query_ids"], visual["query_ids"]):
        raise ValueError("text and visual query IDs differ")
    if not np.array_equal(text["corpus_ids"], visual["corpus_ids"]):
        raise ValueError("text and visual corpus IDs differ")
    text_scores = np.asarray(text["scores"], dtype=np.float64)
    visual_scores = np.asarray(visual["scores"], dtype=np.float64)
    if text_scores.shape != visual_scores.shape or text_scores.shape[1] < 100:
        raise ValueError("invalid or unaligned score surfaces")
    qrels = np.zeros(text_scores.shape, dtype=np.int16)
    qrels[labels["query_positions"], labels["corpus_positions"]] = labels["relevance"]
    positions = np.arange(text_scores.shape[1])
    ranking: dict[str, list[str]] = {}
    quality: dict[str, dict[int, float]] = {}
    for query_position, raw_query_id in enumerate(text["query_ids"]):
        query_id = str(raw_query_id)
        order = np.lexsort((positions, -text_scores[query_position]))[:100]
        ranking[query_id] = [str(text["corpus_ids"][index]) for index in order]
        quality[query_id] = {}
        for depth in (20, 50, 100):
            candidates = order[:depth]
            fused = _zscore(text_scores[query_position, candidates]) + _zscore(
                visual_scores[query_position, candidates]
            )
            scores = np.full(text_scores.shape[1], -np.inf, dtype=np.float64)
            scores[candidates] = fused
            quality[query_id][depth] = _ndcg_row(
                scores,
                qrels[query_position],
                np.asarray(text["corpus_ids"]),
                cutoff=10,
            )
    return {
        "name": name,
        "corpus_pages": text_scores.shape[1],
        "ranking": ranking,
        "quality": quality,
        "query_ids": [str(value) for value in text["query_ids"]],
        "failure_sha256": _sha256(labels_path),
        "ranking_sha256": _sha256(text_path),
        "visual_sha256": _sha256(visual_path),
    }


def _parse_domain(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("domain must be NAME=ROOT")
    name, root = value.split("=", 1)
    return name, Path(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", action="append", type=_parse_domain, required=True)
    parser.add_argument("--arrival-orders", type=int, default=50)
    parser.add_argument("--assignment-shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    domains = [load_surface_domain(*value) for value in args.domain]
    result = analyze(
        domains,
        arrival_orders=args.arrival_orders,
        assignment_shuffles=args.assignment_shuffles,
        seed=args.seed,
    )
    result["transfer_protocol"] = {
        "locator": "BM25 Top-100",
        "expensive_representation": "ColPali-v1.1 visual score surface",
        "candidate_ranking": "within-prefix zscore(BM25) + zscore(visual)",
        "policy_operating_point_frozen_elsewhere": {
            "capacity_fraction": 0.20,
            "miss_budget": 40,
            "maximum_depth": 50,
        },
        "warning": (
            "HR and Finance were not used to select the RBRC operating point, but they "
            "have been opened by earlier ReprForge experiments and are not globally sealed."
        ),
    }
    for domain in domains:
        result["domains"][domain["name"]]["input_sha256"]["visual_runtime"] = domain[
            "visual_sha256"
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
