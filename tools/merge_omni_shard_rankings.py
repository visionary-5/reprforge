#!/usr/bin/env python3
"""Merge exact per-shard Omni rankings into a global top-k ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from tools.analyze_omni_pair import _query_metrics, load_qrels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scored_ranking(path: Path) -> dict[str, list[tuple[str, float]]]:
    output: dict[str, list[tuple[str, float]]] = {}
    previous: dict[str, float] = {}
    seen: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"expected 3 tab fields at {path}:{line_number}")
            query_id, doc_id, score_text = fields
            score = float(score_text)
            if not math.isfinite(score):
                raise ValueError(f"non-finite score at {path}:{line_number}")
            query_seen = seen.setdefault(query_id, set())
            if doc_id in query_seen:
                raise ValueError(f"duplicate document at {path}:{line_number}")
            if query_id in previous and score > previous[query_id] + 1e-8:
                raise ValueError(f"scores increase at {path}:{line_number}")
            query_seen.add(doc_id)
            previous[query_id] = score
            output.setdefault(query_id, []).append((doc_id, score))
    if not output or any(not rows for rows in output.values()):
        raise ValueError(f"empty ranking: {path}")
    return output


def merge_rankings(
    shard_paths: list[Path], *, top_k: int
) -> dict[str, list[tuple[str, float]]]:
    if len(shard_paths) < 2:
        raise ValueError("at least two shard rankings are required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    shards = [load_scored_ranking(path) for path in shard_paths]
    query_ids = set(shards[0])
    if any(set(shard) != query_ids for shard in shards[1:]):
        raise ValueError("all shard rankings must contain exactly the same queries")
    if any(len(rows) < top_k for shard in shards for rows in shard.values()):
        raise ValueError("each shard must return at least top_k documents per query")

    merged: dict[str, list[tuple[str, float]]] = {}
    for query_id in sorted(query_ids, key=lambda value: (not value.isdigit(), value)):
        candidates: list[tuple[float, int, int, str]] = []
        seen_docs: set[str] = set()
        for shard_index, shard in enumerate(shards):
            for rank, (doc_id, score) in enumerate(shard[query_id]):
                if doc_id in seen_docs:
                    raise ValueError(
                        f"document {doc_id!r} for query {query_id!r} occurs in multiple shards"
                    )
                seen_docs.add(doc_id)
                candidates.append((-score, shard_index, rank, doc_id))
        candidates.sort()
        merged[query_id] = [
            (doc_id, -negative_score)
            for negative_score, _, _, doc_id in candidates[:top_k]
        ]
    return merged


def write_ranking(
    path: Path, rankings: dict[str, list[tuple[str, float]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("w", encoding="utf-8") as handle:
        for query_id, rows in rankings.items():
            for doc_id, score in rows:
                handle.write(f"{query_id}\t{doc_id}\t{score:.9g}\n")


def build_report(
    rankings: dict[str, list[tuple[str, float]]],
    *,
    shard_paths: list[Path],
    output_path: Path,
    qrels_path: Path | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "exact-top-k-merge-over-disjoint-omni-corpus-shards-2026-08-06",
        "query_count": len(rankings),
        "top_k": len(next(iter(rankings.values()))),
        "exactness_argument": (
            "Each disjoint shard returns at least global top_k. A document below "
            "top_k in its own shard cannot enter the global top_k."
        ),
        "shards": [
            {"path": str(path), "sha256": _sha256(path)} for path in shard_paths
        ],
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
    }
    if qrels_path is not None:
        qrels = load_qrels(qrels_path)
        if set(qrels) != set(rankings):
            raise ValueError("qrels and merged ranking query sets differ")
        per_query = [
            _query_metrics(
                [doc_id for doc_id, _ in rankings[query_id]],
                qrels[query_id],
                (1, 5, 10, 100),
            )
            for query_id in rankings
        ]
        report["qrels"] = {"path": str(qrels_path), "sha256": _sha256(qrels_path)}
        report["evaluation_results"] = {
            "Recall@1": fmean(row["recall_at_1"] for row in per_query),
            "NDCG@1": fmean(row["ndcg_at_1"] for row in per_query),
            "Recall@5": fmean(row["recall_at_5"] for row in per_query),
            "NDCG@5": fmean(row["ndcg_at_5"] for row in per_query),
            "Recall@10": fmean(row["recall_at_10"] for row in per_query),
            "NDCG@10": fmean(row["ndcg_at_10"] for row in per_query),
            "Recall@100": fmean(row["recall_at_100"] for row in per_query),
            "NDCG@100": fmean(row["ndcg_at_100"] for row in per_query),
            "MRR": fmean(row["mrr_at_100"] for row in per_query),
            "num_queries": len(per_query),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-ranking", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists() or args.output.parent.exists():
        raise FileExistsError("refusing to overwrite merged ranking output")
    rankings = merge_rankings(args.shard_ranking, top_k=args.top_k)
    write_ranking(args.output, rankings)
    report = build_report(
        rankings,
        shard_paths=args.shard_ranking,
        output_path=args.output,
        qrels_path=args.qrels,
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
