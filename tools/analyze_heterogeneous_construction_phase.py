#!/usr/bin/env python3
"""Recompute defer/materialize/full phases with measured heterogeneous costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _winner(costs: dict[str, float]) -> str:
    return min(costs, key=lambda name: (costs[name], name))


def _intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for row in rows:
        winner = row["winner"]
        if intervals and intervals[-1]["winner"] == winner:
            intervals[-1]["end_trace"] = row["trace_replays"]
        else:
            intervals.append(
                {
                    "start_trace": row["trace_replays"],
                    "end_trace": row["trace_replays"],
                    "winner": winner,
                }
            )
    return intervals


def analyze(
    closure: dict[str, Any],
    construction: dict[str, Any],
    dvi: dict[str, Any],
    *,
    full_build_seconds: float,
    policy: str,
    maximum_trace_replays: int,
    dvi_cost_field: str,
) -> dict[str, Any]:
    pages = int(closure["pages"])
    full_page_seconds = float(full_build_seconds) / pages
    sampled_page_seconds = float(
        construction["results"]["1"]["end_to_end_ms_per_page"]["mean"]
    ) / 1000.0
    dvi_pair_seconds = float(dvi["verifier"][dvi_cost_field])
    depths: dict[str, Any] = {}
    for depth, depth_result in closure["depths"].items():
        events = int(depth_result["candidate_events_per_evaluation_trace"])
        budgets: dict[str, Any] = {}
        for fraction, result in depth_result["policies"][policy].items():
            persistent_pages = int(result["persistent_pages"])
            transient_events = int(result["transient_build_events_per_trace"])
            initial_seconds = persistent_pages * full_page_seconds
            defer_per_trace = events * dvi_pair_seconds
            closure_per_trace = transient_events * full_page_seconds
            rows = []
            for replay in range(1, maximum_trace_replays + 1):
                costs = {
                    "dvi_always_defer": defer_per_trace * replay,
                    "closure_materialization": initial_seconds
                    + closure_per_trace * replay,
                    "full_ingestion": float(full_build_seconds),
                }
                rows.append(
                    {
                        "trace_replays": replay,
                        "cost_seconds": costs,
                        "winner": _winner(costs),
                    }
                )
            dvi_margin = defer_per_trace - closure_per_trace
            full_margin = float(full_build_seconds) - initial_seconds
            budgets[fraction] = {
                "persistent_pages": persistent_pages,
                "persistent_fraction_realized": persistent_pages / pages,
                "persistent_hit_fraction": result["persistent_hit_fraction"],
                "transient_build_events_per_trace": transient_events,
                "cost_seconds": {
                    "initial_materialization": initial_seconds,
                    "dvi_per_trace": defer_per_trace,
                    "closure_transient_per_trace": closure_per_trace,
                    "full_ingestion": float(full_build_seconds),
                },
                "continuous_break_even_trace_replays": {
                    "dvi_to_closure": (
                        initial_seconds / dvi_margin if dvi_margin > 0 else None
                    ),
                    "closure_to_full": (
                        full_margin / closure_per_trace
                        if closure_per_trace > 0 and full_margin >= 0
                        else None
                    ),
                },
                "winner_intervals": _intervals(rows),
                "trace_replays": rows,
                "closure_has_integer_winner": any(
                    row["winner"] == "closure_materialization" for row in rows
                ),
            }
        depths[depth] = {
            "candidate_events_per_trace": events,
            "evaluation_queries": int(closure["evaluation_queries"]),
            "budgets": budgets,
        }
    return {
        "schema_version": 1,
        "protocol": "heterogeneous-construction-phase-v0",
        "domain": closure["domain"],
        "pages": pages,
        "policy": policy,
        "maximum_trace_replays": maximum_trace_replays,
        "measured_costs": {
            "full_build_seconds": float(full_build_seconds),
            "full_build_seconds_per_page": full_page_seconds,
            "sampled_full_page_seconds": sampled_page_seconds,
            "sample_to_full_build_relative_error": (
                sampled_page_seconds / full_page_seconds - 1.0
            ),
            "dvi_seconds_per_query_page_pair": dvi_pair_seconds,
            "dvi_cost_field": dvi_cost_field,
        },
        "cost_scope": {
            "dvi": (
                "Qwen2.5-VL-3B query-page verifier timing selected by "
                f"{dvi_cost_field}"
            ),
            "closure": (
                "measured full-corpus Omni build time divided by pages; counts "
                "persistent and repeated transient representation construction"
            ),
            "full": "measured complete Omni ingestion build time",
            "excluded": (
                "common cheap retrieval plus online retrieval scoring; physical "
                "closure scoring latency is reported by the separate runtime benchmark"
            ),
        },
        "depths": depths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--construction", type=Path, required=True)
    parser.add_argument("--dvi", type=Path, required=True)
    parser.add_argument("--full-build-seconds", type=float, required=True)
    parser.add_argument("--policy", default="history_frequency")
    parser.add_argument("--maximum-trace-replays", type=int, default=16)
    parser.add_argument("--dvi-cost-field", default="page_seconds_mean")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    payload = analyze(
        json.loads(args.closure.read_text()),
        json.loads(args.construction.read_text()),
        json.loads(args.dvi.read_text()),
        full_build_seconds=args.full_build_seconds,
        policy=args.policy,
        maximum_trace_replays=args.maximum_trace_replays,
        dvi_cost_field=args.dvi_cost_field,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
