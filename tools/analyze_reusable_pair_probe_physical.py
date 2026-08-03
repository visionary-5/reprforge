#!/usr/bin/env python3
"""Compact the physical reusable-pair experiment into claim-level evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weighted(runs: list[dict[str, Any]], key: str) -> float:
    total = sum(int(run["queries"]) for run in runs)
    return sum(float(run["quality"][key]) * int(run["queries"]) for run in runs) / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    offline = json.loads(args.offline.read_text(encoding="utf-8"))
    runs = list(raw["runs"])
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_policy[str(run["policy"])].append(run)
    policies: dict[str, Any] = {}
    for policy, rows in sorted(by_policy.items()):
        policies[policy] = {
            "runs": len(rows),
            "selected_pages": sum(int(row["selected_pages"]) for row in rows),
            "materialization_calls": sum(
                int(row["physical"]["materialization_calls"]) for row in rows
            ),
            "charged_total_ms": sum(
                float(row["physical"]["charged_total_ms"]) for row in rows
            ),
            "visual_encode_ms": sum(
                float(row["physical"]["visual_encode_ms"]) for row in rows
            ),
            "visual_score_ms": sum(
                float(row["physical"]["visual_score_ms"]) for row in rows
            ),
            "recall_5": _weighted(rows, "recall_5"),
            "exact_teacher_agreement": _weighted(
                rows, "exact_teacher_agreement"
            ),
        }

    repetitions = sorted(set(int(run["repetition"]) for run in runs))
    per_repetition: list[dict[str, Any]] = []
    for repetition in repetitions:
        totals: dict[str, float] = {}
        for policy in by_policy:
            totals[policy] = sum(
                float(run["physical"]["charged_total_ms"])
                for run in by_policy[policy]
                if int(run["repetition"]) == repetition
            )
        per_repetition.append(
            {
                "repetition": repetition,
                "charged_total_ms": totals,
                "independent_over_active_speedup": (
                    totals["independent_20"] / totals["active_pair_15"]
                ),
                "independent_over_static_speedup": (
                    totals["independent_20"] / totals["static_pair_15"]
                ),
                "independent_over_atomic_static_speedup": (
                    totals["independent_20"]
                    / totals["static_pair_atomic_15"]
                ),
                "active_over_static_overhead": (
                    totals["active_pair_15"] / totals["static_pair_15"] - 1.0
                ),
            }
        )

    offline_15 = next(
        row for row in offline["budgets"] if math.isclose(row["budget_fraction"], 0.15)
    )["aggregate"]
    active = policies["active_pair_15"]
    static = policies["static_pair_15"]
    atomic_static = policies["static_pair_atomic_15"]
    independent = policies["independent_20"]
    active_speed_gate = all(
        row["independent_over_active_speedup"] >= 1.15
        for row in per_repetition
    )
    active_quality_gate = (
        active["recall_5"] >= offline_15["active_pair"]["recall_5"] - 1e-12
        and active["recall_5"] > static["recall_5"] + 1e-12
    )
    static_constructive_gate = (
        atomic_static["recall_5"] >= independent["recall_5"] - 1e-12
        and all(
            row["independent_over_atomic_static_speedup"] >= 1.15
            for row in per_repetition
        )
    )
    payload = {
        "schema_version": 1,
        "experiment": "physical-reusable-pair-probe-analysis",
        "source": {
            "raw_sha256": _sha256(args.raw),
            "offline_sha256": _sha256(args.offline),
            "runs": len(runs),
            "repetitions": len(repetitions),
            "gpu": raw["resource_contract"]["gpu"],
        },
        "policies": policies,
        "per_repetition": per_repetition,
        "comparisons": {
            "active_page_reduction_vs_independent": (
                1.0 - active["selected_pages"] / independent["selected_pages"]
            ),
            "static_page_reduction_vs_independent": (
                1.0
                - atomic_static["selected_pages"] / independent["selected_pages"]
            ),
            "active_recall_gain_vs_static": active["recall_5"] - static["recall_5"],
            "static_recall_gain_vs_independent": (
                atomic_static["recall_5"] - independent["recall_5"]
            ),
        },
        "gates": {
            "active_speed_in_both_repetitions": active_speed_gate,
            "active_preserves_offline_adaptive_quality": active_quality_gate,
            "active_system_claim": active_speed_gate and active_quality_gate,
            "static_pair_constructive_baseline": static_constructive_gate,
        },
        "decision": (
            "reject-online-adaptation-retain-static-pair-plan"
            if not active_quality_gate and static_constructive_gate
            else "requires-review"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
