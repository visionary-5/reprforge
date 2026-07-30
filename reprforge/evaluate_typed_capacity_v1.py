#!/usr/bin/env python3
"""Evaluate the frozen typed-capacity policy on one protocol role."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reprforge.policy_replay import (
    ReplayData,
    evaluate_plan,
    fixed_hybrid_plan,
    load_replay_data,
    typed_capacity_plan_v1,
    uniform_plan,
)
from reprforge.route_mechanism_analysis import subset_by_documents


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_role(
    *,
    result_dir: Path,
    protocol_path: Path,
    role: str,
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    documents = {
        int(row["document_index"])
        for row in protocol["documents"]
        if str(row["role"]) == role
    }
    if not documents:
        raise ValueError(f"protocol role has no documents: {role}")
    item_rows = _read_jsonl(result_dir / "items.jsonl")
    query_rows = _read_jsonl(result_dir / "queries.jsonl")
    full = load_replay_data(
        result_dir / "items.jsonl",
        result_dir / "queries.jsonl",
        result_dir / "scores.jsonl",
    )
    data, _ = subset_by_documents(
        full,
        query_rows=query_rows,
        item_rows=item_rows,
        document_indices=documents,
    )
    plans = {
        "uniform-image": uniform_plan(data.items, "image"),
        "uniform-image-pool-4": uniform_plan(data.items, "image-pool-4"),
        "uniform-image-pool-9": uniform_plan(data.items, "image-pool-9"),
        "uniform-image-pool-25": uniform_plan(data.items, "image-pool-25"),
        "fixed-hybrid": fixed_hybrid_plan(data.items),
        "fixed-hybrid-image-pool-9": fixed_hybrid_plan(
            data.items,
            image_route="image-pool-9",
        ),
        "typed-capacity-v1": typed_capacity_plan_v1(data.items),
    }
    results = {
        name: evaluate_plan(data, plan, ks=(1, 5, 10))
        for name, plan in plans.items()
    }
    uniform = results["uniform-image-pool-9"]
    proposed = results["typed-capacity-v1"]
    return {
        "contract": {
            "policy": "typed-capacity-v1",
            "role": role,
            "protocol_sha256": _sha256(protocol_path),
            "result_manifest_sha256": _sha256(result_dir / "manifest.json"),
            "policy_rule": {
                "table": "image",
                "image|chart|figure": "image-pool-9",
                "other": "image-pool-25",
            },
            "frozen_before_role_evaluation": True,
        },
        "corpus": {
            "documents": len(documents),
            "document_indices": sorted(documents),
            "items": len(data.items),
            "queries": len(data.queries),
        },
        "results": results,
        "gate": {
            "budget_at_or_below_uniform_9": (
                proposed["cost"]["offline_index_bytes"]
                <= uniform["cost"]["offline_index_bytes"]
            ),
            "ndcg_at_10_gain_at_least_0_01": (
                proposed["ndcg_at_10"] - uniform["ndcg_at_10"] >= 0.01
            ),
            "recall_at_5_regression_at_most_0_01": (
                proposed["recall_at_5"] - uniform["recall_at_5"] >= -0.01
            ),
            "deltas_vs_uniform_9": {
                metric: proposed[metric] - uniform[metric]
                for metric in (
                    "recall_at_1",
                    "recall_at_5",
                    "recall_at_10",
                    "ndcg_at_5",
                    "ndcg_at_10",
                )
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_role(
        result_dir=args.result_dir,
        protocol_path=args.protocol,
        role=args.role,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
