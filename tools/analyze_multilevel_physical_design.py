#!/usr/bin/env python3
"""Audit and analyze a real three-tier representation bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reprforge.multilevel_physical_design import (
    REQUIRED_TIERS,
    analyze_multilevel_bundle,
)


def _route(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("route must use NAME=PATH")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--route",
        action="append",
        type=_route,
        required=True,
        help="repeat in cheap_base, compact_pool, full_multivector order",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cost-overrides", type=Path)
    parser.add_argument("--activation-trace", type=Path)
    parser.add_argument("--mixed-state-artifact", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    routes = dict(args.route)
    if tuple(routes) != REQUIRED_TIERS:
        parser.error(f"routes must be ordered exactly as {REQUIRED_TIERS}")
    overrides = None
    if args.cost_overrides is not None:
        overrides = json.loads(args.cost_overrides.read_text(encoding="utf-8"))
        if set(overrides) - set(REQUIRED_TIERS):
            parser.error("cost override contains an unknown tier")
    report = analyze_multilevel_bundle(
        routes,
        args.labels,
        dataset=args.dataset,
        cost_overrides=overrides,
        activation_trace=args.activation_trace,
        mixed_state_artifact=args.mixed_state_artifact,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "decision": report["decision"],
                "best_uniform": report["uniform_tier_selected_on_fit"],
                "oracle": report["diagnostic_query_route_oracle"],
                "missing": [
                    name
                    for name, available in report["capability_matrix"].items()
                    if not available
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
