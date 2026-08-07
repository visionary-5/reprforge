#!/usr/bin/env python3
"""Summarize direct-build partial visual indexes under one fusion protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.bm25 import build_index, score_queries
from reprforge.physical_partial_evaluation import (
    evaluate_rankings,
    gain_recovery,
    reciprocal_rank_fusion,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ranking(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    scores: dict[str, float] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"invalid ranking row {path}:{line_number}")
            query_id, doc_id, score_text = fields
            score = float(score_text)
            if query_id in scores and score > scores[query_id] + 1e-8:
                raise ValueError(f"increasing ranking score {path}:{line_number}")
            if doc_id in rankings.setdefault(query_id, []):
                raise ValueError(f"duplicate page {path}:{line_number}")
            rankings[query_id].append(doc_id)
            scores[query_id] = score
    return rankings


def _bm25(corpus: list[dict[str, Any]], queries: list[dict[str, Any]], depth: int):
    doc_ids = [str(row["docid"]) for row in corpus]
    state, posting_bytes, vocabulary_bytes = build_index(
        [str(row.get("text") or "") for row in corpus]
    )
    matrix = score_queries(state, [str(row["query"]) for row in queries], k1=1.2, b=0.75)
    rankings = {}
    for query, scores in zip(queries, matrix, strict=True):
        order = np.lexsort((np.asarray(doc_ids), -scores))[:depth]
        rankings[str(query["query_id"])] = [doc_ids[int(value)] for value in order]
    return rankings, int(posting_bytes.sum()) + int(vocabulary_bytes)


def _qrels(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if float(row["relevance"]) > 0:
            result.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = float(
                row["relevance"]
            )
    return result


def _wall_seconds(path: Path) -> float:
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            values[fields[0]] = float(fields[1])
    if "real" not in values:
        raise ValueError(f"GNU time file lacks real seconds: {path}")
    return values["real"]


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _case(root: Path) -> dict[str, Any]:
    ranking = root / "full/result-ranking-top100/ranking.txt"
    timing = root / "timing/full-build.time"
    index = root / "full/index"
    run_manifest = root / "run-manifest.json"
    for path in (ranking, timing, index, run_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    selection_manifest = root.parent / f"{root.name}-input-manifest.json"
    selected = json.loads(selection_manifest.read_text()) if selection_manifest.exists() else None
    return {
        "name": root.name,
        "root": root,
        "ranking_path": ranking,
        "rankings": _load_ranking(ranking),
        "build_wall_seconds": _wall_seconds(timing),
        "index_bytes": _tree_bytes(index),
        "run_manifest_sha256": _sha(run_manifest),
        "selection_manifest": selected,
        "selection_manifest_sha256": _sha(selection_manifest)
        if selection_manifest.exists()
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = json.loads(args.config.read_text())
    corpus = _read_jsonl(args.dataset_root / "corpus.jsonl")
    queries = _read_jsonl(args.dataset_root / "queries.jsonl")
    qrels = _qrels(_read_jsonl(args.dataset_root / "qrels.jsonl"))
    text, text_index_bytes = _bm25(corpus, queries, max(config["cheap_locator_depths"]))
    full = _case(args.matrix_root / "full-100")
    cases = [
        _case(path)
        for path in sorted(args.matrix_root.iterdir())
        if path.is_dir() and path.name != "full-100"
    ]
    query_ids = set(qrels)
    for case in [full, *cases]:
        if set(case["rankings"]) != query_ids:
            raise ValueError(f"{case['name']} query IDs differ from qrels")
    text_eval = evaluate_rankings(text, qrels)
    full_hybrid = {
        query_id: reciprocal_rank_fusion(text[query_id], full["rankings"][query_id])
        for query_id in qrels
    }
    full_eval = evaluate_rankings(full_hybrid, qrels)
    full_visual_eval = evaluate_rankings(full["rankings"], qrels)
    text_ndcg = float(text_eval["mean"]["ndcg_at_10"])
    full_ndcg = float(full_eval["mean"]["ndcg_at_10"])
    rows = []
    for case in cases:
        hybrid = {
            query_id: reciprocal_rank_fusion(text[query_id], case["rankings"][query_id])
            for query_id in qrels
        }
        evaluation = evaluate_rankings(hybrid, qrels)
        visual_evaluation = evaluate_rankings(case["rankings"], qrels)
        escaped = [
            query_id
            for query_id in qrels
            if not set(text[query_id][:20]) & set(qrels[query_id])
        ]
        repaired = sum(
            bool(set(case["rankings"][query_id][:20]) & set(qrels[query_id]))
            for query_id in escaped
        )
        selection = case["selection_manifest"] or {}
        rows.append(
            {
                "name": case["name"],
                "strategy": selection.get("strategy"),
                "selected_pages": selection.get("selected_pages"),
                "selected_fraction": selection.get("selected_fraction"),
                "direct_build_wall_seconds": case["build_wall_seconds"],
                "direct_build_fraction_of_full": case["build_wall_seconds"]
                / full["build_wall_seconds"],
                "index_bytes": case["index_bytes"],
                "index_bytes_fraction_of_full": case["index_bytes"] / full["index_bytes"],
                "hybrid": evaluation["mean"],
                "visual_only": visual_evaluation["mean"],
                "ndcg_gain_recovery": gain_recovery(
                    float(evaluation["mean"]["ndcg_at_10"]), text_ndcg, full_ndcg
                ),
                "bm25_top20_escape_queries": len(escaped),
                "escape_queries_repaired_at_visual_top20": repaired,
                "escape_repair_fraction": repaired / len(escaped) if escaped else None,
                "artifacts": {
                    "ranking_sha256": _sha(case["ranking_path"]),
                    "run_manifest_sha256": case["run_manifest_sha256"],
                    "selection_manifest_sha256": case["selection_manifest_sha256"],
                },
            }
        )
    output = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "physical_static_matrix_complete",
        "baselines": {
            "text_only": text_eval["mean"],
            "full_visual_only": full_visual_eval["mean"],
            "full_hybrid": full_eval["mean"],
        },
        "full_physical": {
            "build_wall_seconds": full["build_wall_seconds"],
            "index_bytes": full["index_bytes"],
        },
        "text_logical_index_bytes": text_index_bytes,
        "cases": rows,
        "warnings": [
            "Partial visual rankings are produced by directly built subset indexes; their selected-set ranks are not masked ranks from a complete Full score surface.",
            "BM25 build wall time is not included by this analyzer and must be taken from the preparation/run manifest for complete ingestion accounting.",
            "RRF uses fixed constant 60 and Top-100 text/visual rankings for every case."
        ],
        "input_sha256": {
            "config": _sha(args.config),
            "corpus": _sha(args.dataset_root / "corpus.jsonl"),
            "queries": _sha(args.dataset_root / "queries.jsonl"),
            "qrels": _sha(args.dataset_root / "qrels.jsonl"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"baselines": output["baselines"], "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
