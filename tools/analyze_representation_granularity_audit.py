#!/usr/bin/env python3
"""Aggregate region rankings to parent pages and report signed cohort effects."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.representation_granularity import aggregate_unit_ranking, parent_metrics


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_ranking(path: Path) -> dict[str, list[tuple[str, float]]]:
    result: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            query_id, unit_id, score = line.rstrip("\n").split("\t")[:3]
            result[query_id].append((unit_id, float(score)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in read_jsonl(args.prepared_root / "parent-qrels.jsonl"):
        qrels[str(row["query_id"])][str(row["doc_id"])] = float(row["relevance"])
    organizations = {}
    depth = int(config["retrieval"]["evaluation_depth"])
    for organization in config["organizations"]:
        manifest = json.loads((args.prepared_root / organization / "manifest.json").read_text())
        unit_to_parent = {row["unit_id"]: row["parent_id"] for row in manifest["units"]}
        parent_category = {row["parent_id"]: row["category"] for row in manifest["units"]}
        ranking = load_ranking(args.run_root / organization / "ranking.txt")
        selected = set(parent_category)
        eligible = [qid for qid in ranking if selected & set(qrels.get(qid, {}))]
        metrics = []
        category_exposure = defaultdict(lambda: {"relevant": 0, "irrelevant": 0})
        for query_id in eligible:
            parents = aggregate_unit_ranking(ranking[query_id], unit_to_parent)
            parent_ids = [parent for parent, _ in parents]
            selected_qrels = {doc: rel for doc, rel in qrels[query_id].items() if doc in selected}
            metrics.append(parent_metrics(parent_ids, selected_qrels, depth=depth))
            for parent in parent_ids[:depth]:
                category = parent_category[parent]
                key = "relevant" if selected_qrels.get(parent, 0) > 0 else "irrelevant"
                category_exposure[category][key] += 1
        results_path = args.run_root / organization / "results.json"
        physical = json.loads(results_path.read_text()) if results_path.exists() else {}
        organizations[organization] = {
            "parents": len(selected),
            "units": len(unit_to_parent),
            "mean_units_per_parent": len(unit_to_parent) / len(selected),
            "eligible_queries": len(eligible),
            "parent_ndcg_at_10": float(np.mean([row["ndcg"] for row in metrics])) if metrics else None,
            "parent_hit_at_10": float(np.mean([row["hit"] for row in metrics])) if metrics else None,
            "top10_exposure": dict(category_exposure),
            "runner_results": physical,
        }
    whole = organizations["whole_page"]
    for organization, result in organizations.items():
        if result["parent_ndcg_at_10"] is not None and whole["parent_ndcg_at_10"]:
            result["ndcg_retention_vs_whole"] = result["parent_ndcg_at_10"] / whole["parent_ndcg_at_10"]
        whole_negative = whole["top10_exposure"].get("negative", {}).get("irrelevant", 0)
        current_negative = result["top10_exposure"].get("negative", {}).get("irrelevant", 0)
        result["negative_irrelevant_exposure_change_vs_whole"] = (
            (current_negative - whole_negative) / whole_negative if whole_negative else None
        )
    payload = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "organizations": organizations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
