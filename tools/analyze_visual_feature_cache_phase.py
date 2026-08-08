#!/usr/bin/env python3
"""Analyze an exact-quality raw/partial/full visual-feature cache phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _winner(costs: dict[str, float]) -> str:
    return min(costs, key=lambda name: (costs[name], name))


def _intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if output and output[-1]["winner"] == row["winner"]:
            output[-1]["end_trace"] = row["trace_replays"]
        else:
            output.append(
                {
                    "start_trace": row["trace_replays"],
                    "end_trace": row["trace_replays"],
                    "winner": row["winner"],
                }
            )
    return output


def _query_intervals(
    misses: list[int],
    *,
    depth: int,
    maximum_trace_replays: int,
    initial_partial: float,
    initial_full: float,
    raw_seconds: float,
    cached_seconds: float,
) -> list[dict[str, Any]]:
    repeated = misses * maximum_trace_replays
    rows = []
    partial = initial_partial
    for query_number, missing in enumerate(repeated, start=1):
        partial += (
            (depth - int(missing)) * cached_seconds
            + int(missing) * raw_seconds
        )
        costs = {
            "raw_always_defer": query_number * depth * raw_seconds,
            "partial_visual_feature_cache": partial,
            "full_visual_feature_cache": (
                initial_full + query_number * depth * cached_seconds
            ),
        }
        rows.append(
            {
                "trace_replays": query_number,
                "cost_seconds": costs,
                "winner": _winner(costs),
            }
        )
    return [
        {
            "start_query": row["start_trace"],
            "end_query": row["end_trace"],
            "winner": row["winner"],
        }
        for row in _intervals(rows)
    ]


def analyze(
    closure: dict[str, Any],
    dvi: dict[str, Any],
    feature_cache: dict[str, Any],
    *,
    policy: str,
    maximum_trace_replays: int,
) -> dict[str, Any]:
    pages = int(closure["pages"])
    raw_seconds = float(dvi["verifier"]["page_end_to_end_seconds_mean"])
    build_seconds = float(
        feature_cache["cache_build_end_to_end_ms"]["mean"]
    ) / 1000.0
    cached_seconds = float(
        feature_cache["cached_feature_h2d_and_language_ms"]["mean"]
    ) / 1000.0
    if cached_seconds >= raw_seconds:
        raise ValueError("cached access must be cheaper than raw-page access")
    depths: dict[str, Any] = {}
    for depth, depth_result in closure["depths"].items():
        events = int(depth_result["candidate_events_per_evaluation_trace"])
        budgets = {}
        for fraction, result in depth_result["policies"][policy].items():
            persistent = int(result["persistent_pages"])
            misses = int(result["transient_build_events_per_trace"])
            misses_per_query = list(
                map(int, result["transient_build_events_per_query"])
            )
            hits = events - misses
            initial_partial = persistent * build_seconds
            partial_per_trace = hits * cached_seconds + misses * raw_seconds
            defer_per_trace = events * raw_seconds
            initial_full = pages * build_seconds
            full_per_trace = events * cached_seconds
            rows = []
            for replay in range(1, maximum_trace_replays + 1):
                costs = {
                    "raw_always_defer": replay * defer_per_trace,
                    "partial_visual_feature_cache": (
                        initial_partial + replay * partial_per_trace
                    ),
                    "full_visual_feature_cache": (
                        initial_full + replay * full_per_trace
                    ),
                }
                rows.append(
                    {
                        "trace_replays": replay,
                        "cost_seconds": costs,
                        "winner": _winner(costs),
                    }
                )
            savings = cached_seconds - raw_seconds
            budgets[fraction] = {
                "persistent_pages": persistent,
                "persistent_hit_fraction": hits / events if events else 0.0,
                "cost_seconds": {
                    "partial_initial_build": initial_partial,
                    "raw_per_trace": defer_per_trace,
                    "partial_per_trace": partial_per_trace,
                    "full_initial_build": initial_full,
                    "full_per_trace": full_per_trace,
                },
                "continuous_break_even_trace_replays": {
                    "raw_to_partial": (
                        initial_partial / (hits * -savings) if hits else None
                    ),
                    "partial_to_full": (
                        (initial_full - initial_partial) / (misses * -savings)
                        if misses
                        else None
                    ),
                },
                "winner_intervals": _intervals(rows),
                "winner_query_intervals": _query_intervals(
                    misses_per_query,
                    depth=int(depth),
                    maximum_trace_replays=maximum_trace_replays,
                    initial_partial=initial_partial,
                    initial_full=initial_full,
                    raw_seconds=raw_seconds,
                    cached_seconds=cached_seconds,
                ),
                "winner_query_intervals_random_orders": [
                    {
                        "seed": seed,
                        "intervals": _query_intervals(
                            np.random.default_rng(seed)
                            .permutation(misses_per_query)
                            .tolist(),
                            depth=int(depth),
                            maximum_trace_replays=maximum_trace_replays,
                            initial_partial=initial_partial,
                            initial_full=initial_full,
                            raw_seconds=raw_seconds,
                            cached_seconds=cached_seconds,
                        ),
                    }
                    for seed in range(20260808, 20260813)
                ],
                "trace_replays": rows,
            }
        depths[depth] = {
            "candidate_events_per_trace": events,
            "budgets": budgets,
        }
    return {
        "schema_version": 1,
        "protocol": "exact-quality-visual-feature-cache-phase-v0",
        "domain": closure["domain"],
        "pages": pages,
        "policy": policy,
        "maximum_trace_replays": maximum_trace_replays,
        "measured_costs": {
            "raw_dvi_seconds_per_query_page_pair": raw_seconds,
            "visual_feature_build_end_to_end_seconds_per_page": build_seconds,
            "cached_feature_access_seconds_per_query_page_pair": cached_seconds,
            "mean_visual_feature_bytes_per_page": feature_cache[
                "mean_cached_feature_bytes"
            ],
        },
        "quality_contract": {
            "definition": (
                "cached and raw routes execute the same language model and "
                "differ only in whether query-independent vision-tower output "
                "is reused"
            ),
            "sample_pairs": feature_cache["sample_pairs"],
            "maximum_score_absolute_difference": feature_cache[
                "maximum_score_absolute_difference"
            ],
        },
        "depths": depths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--dvi", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--policy", default="history_frequency")
    parser.add_argument("--maximum-trace-replays", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    result = analyze(
        json.loads(args.closure.read_text()),
        json.loads(args.dvi.read_text()),
        json.loads(args.feature_cache.read_text()),
        policy=args.policy,
        maximum_trace_replays=args.maximum_trace_replays,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
