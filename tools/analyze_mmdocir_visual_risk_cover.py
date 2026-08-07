#!/usr/bin/env python3
"""Run the frozen MMDocIR cheap visual-risk coverage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.policy_replay import load_replay_data
from reprforge.visual_risk_cover import (
    evaluate_cover,
    fit_ridge_risk,
    item_feature_rows,
    repair_events,
    select_by_policy,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    item_path = args.data_dir / "items.jsonl"
    query_path = args.data_dir / "queries.jsonl"
    score_path = args.data_dir / "scores.jsonl"
    item_rows = _read_jsonl(item_path)
    query_rows = _read_jsonl(query_path)
    data = load_replay_data(item_path, query_path, score_path)
    roles = {int(row["document_index"]): str(row["role"]) for row in split["documents"]}
    domains = {int(row["document_index"]): str(row["domain"]) for row in split["documents"]}
    item_documents: dict[str, int] = {}
    query_documents: dict[str, int] = {}
    for row in query_rows:
        document = int(row["document_index"])
        query_documents[str(row["query_id"])] = document
        for item_id in row["candidate_item_ids"]:
            item_documents[str(item_id)] = document
    train_queries = [query_id for query_id, document in query_documents.items() if roles[document] == config["train_role"]]
    test_queries = [query_id for query_id, document in query_documents.items() if roles[document] == config["test_role"]]
    sealed_queries = [query_id for query_id, document in query_documents.items() if roles[document] in set(config["sealed_roles"])]
    train_items = sorted(item_id for item_id, document in item_documents.items() if roles[document] == config["train_role"])
    test_items = sorted(item_id for item_id, document in item_documents.items() if roles[document] == config["test_role"])
    test_item_rows = [row for row in item_rows if str(row["item_id"]) in set(test_items)]
    train_events = repair_events(data, train_queries, depth=int(config["candidate_depth"]))
    test_events = repair_events(data, test_queries, depth=int(config["candidate_depth"]))
    train_positive = {str(event["item_id"]) for event in train_events}
    test_positive = {str(event["item_id"]) for event in test_events}
    feature_content_types = sorted(
        {
            str(row.get("content_type", "unknown")).lower()
            for row in item_rows
            if str(row["item_id"]) in set(train_items) | set(test_items)
        }
    )
    train_x, feature_names = item_feature_rows(
        item_rows, train_items, content_types=feature_content_types
    )
    test_x, _ = item_feature_rows(
        item_rows, test_items, content_types=feature_content_types
    )
    train_y = np.asarray([float(item_id in train_positive) for item_id in train_items])
    ridge_scores = fit_ridge_risk(train_x, train_y, test_x, ridge_lambda=float(config["ridge_lambda"]))
    item_domains = {item_id: domains[item_documents[item_id]] for item_id in test_items}
    curves = {}
    for policy in config["policies"]:
        rows = {}
        repetitions = int(config["random_repetitions"]) if policy == "random" else 1
        for budget in config["budgets"]:
            count = int(math.ceil(float(budget) * len(test_items)))
            values = []
            for repetition in range(repetitions):
                selected = select_by_policy(
                    policy=policy,
                    item_ids=test_items,
                    item_rows=test_item_rows,
                    item_documents=item_documents,
                    count=count,
                    ridge_scores=ridge_scores,
                    repair_items=test_positive,
                    seed=20260807 + repetition + int(float(budget) * 10_000),
                )
                values.append(evaluate_cover(selected, test_events, all_item_ids=test_items, item_rows=test_item_rows, item_domains=item_domains))
            aggregate = values[0] if repetitions == 1 else {
                "runs": repetitions,
                **{
                    key: float(np.mean([row[key] for row in values if row[key] is not None]))
                    for key in (
                        "selected_items", "selected_fraction", "selected_encode_cost_fraction",
                        "repair_item_recall", "repair_event_recall", "repair_query_coverage", "selected_precision"
                    )
                },
            }
            rows[str(budget)] = aggregate
        curves[policy] = rows
    gate_config = config["gate"]
    budget_key = str(gate_config["budget"])
    ridge = curves["ridge_risk"][budget_key]
    simple = max(curves[name][budget_key]["repair_event_recall"] for name in ("random", "document_uniform", "visual_type", "text_scarcity"))
    domain_rows = ridge["domains"]
    domains_with_repair = len(domain_rows)
    domains_half = sum(row["query_coverage"] >= 0.5 for row in domain_rows.values())
    checks = {
        "repair_event_recall": ridge["repair_event_recall"] >= float(gate_config["minimum_repair_event_recall"]),
        "gain_over_simple": ridge["repair_event_recall"] - simple >= float(gate_config["minimum_gain_over_best_simple_baseline"]),
        "enough_domains": domains_with_repair >= int(gate_config["minimum_domains_with_repair_queries"]),
        "cross_domain_query_coverage": domains_half >= int(gate_config["minimum_domains_half_query_coverage"]),
    }
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "input_sha256": {"items": _sha(item_path), "queries": _sha(query_path), "scores": _sha(score_path), "split": _sha(args.split_protocol), "config": _sha(args.config)},
        "split": {"train_queries": len(train_queries), "train_items": len(train_items), "test_queries": len(test_queries), "test_items": len(test_items), "sealed_queries_excluded_from_labels_and_metrics": len(sealed_queries)},
        "repair_labels": {"train_events": len(train_events), "train_items": len(train_positive), "test_events": len(test_events), "test_items": len(test_positive)},
        "feature_names": feature_names,
        "curves": curves,
        "gate": {"budget": float(gate_config["budget"]), "ridge_repair_event_recall": ridge["repair_event_recall"], "best_simple_repair_event_recall": simple, "domains_with_repair_queries": domains_with_repair, "domains_half_query_coverage": domains_half, "checks": checks, "passes": all(checks.values())},
        "warning": "Layout-level within-document MMDocIR transfer diagnostic; sealed final-evaluation queries are excluded from labels, fitting, and metrics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
