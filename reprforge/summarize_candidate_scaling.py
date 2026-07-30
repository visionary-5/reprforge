#!/usr/bin/env python3
"""Summarize systems-only candidate replication benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NAME = re.compile(r"^(?P<system>.+)-factor(?P<factor>[0-9]+)\.json$")


def summarize(reports: Path) -> dict:
    rows = []
    for path in sorted(reports.glob("*.json")):
        match = NAME.match(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "system": match.group("system"),
                "factor": int(match.group("factor")),
                "items": int(payload["index_items"]),
                "compact_vector_bytes": int(payload["compact_vector_bytes"]),
                "resident_vector_bytes": int(payload["resident_vector_bytes"]),
                "execution_batches": int(payload["execution_batches"]),
                "latency_p50_ms": float(payload["latency_ms"]["p50"]),
                "latency_p95_ms": float(payload["latency_ms"]["p95"]),
                "qps": float(payload["qps"]),
            }
        )
    if not rows:
        raise ValueError(f"no candidate-scaling reports found under {reports}")
    baselines = {
        row["system"]: row
        for row in rows
        if row["factor"] == 1
    }
    for row in rows:
        baseline = baselines.get(row["system"])
        if baseline is None:
            raise ValueError(f"missing factor-1 baseline for {row['system']}")
        row["latency_vs_factor1"] = (
            row["latency_p50_ms"] / baseline["latency_p50_ms"]
        )
    rows.sort(key=lambda row: (row["system"], row["factor"]))
    return {
        "contract": {
            "quality_labels_valid": False,
            "candidate_method": (
                "physical vector replication with distinct identifiers and "
                "distinct storage"
            ),
            "search_scope": "all-indexed-items",
            "allowed_claims": ["latency", "throughput", "memory", "scaling"],
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
