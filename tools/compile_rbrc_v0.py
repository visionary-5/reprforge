#!/usr/bin/env python3
"""Compile an RBRC v0 calibrated-safe program without held-out inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reprforge.rbrc_v0 import compile_programs
from reprforge.rbrc_v0_inputs import (
    canonical_json_sha256,
    load_bm25_colpali_domain,
    load_omni_domain,
)


def _name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def _name_pair(value: str) -> tuple[str, Path, Path]:
    if "=" not in value or "," not in value:
        raise argparse.ArgumentTypeError("expected NAME=FAILURE_JSON,RANKING_TSV")
    name, paths = value.split("=", 1)
    failure, ranking = paths.split(",", 1)
    return name, Path(failure), Path(ranking)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("bm25_colpali", "hpool_full"), required=True)
    parser.add_argument("--surface-domain", action="append", type=_name_path, default=[])
    parser.add_argument("--omni-domain", action="append", type=_name_pair, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile = config["representation_stacks"][args.profile]
    depths = tuple(int(value) for value in profile["primitive_plan_depths"])
    if args.profile == "bm25_colpali":
        if args.omni_domain or not args.surface_domain:
            parser.error("bm25_colpali requires only --surface-domain")
        domains = [
            load_bm25_colpali_domain(name, root, depths)
            for name, root in args.surface_domain
        ]
    else:
        if args.surface_domain or not args.omni_domain:
            parser.error("hpool_full requires only --omni-domain")
        domains = [
            load_omni_domain(name, failure, ranking, depths)
            for name, failure, ranking in args.omni_domain
        ]
    expected = set(profile["calibration_domains"])
    actual = {domain.name for domain in domains}
    if actual != expected:
        raise ValueError(f"calibration domain mismatch: expected {expected}, got {actual}")

    quality = config["quality_contract"]
    online = config["online_contract"]
    orders = config["orders"]["calibration"]
    result = compile_programs(
        domains,
        floors=depths[:-1],
        reference_depth=depths[-1],
        cold_budget=int(online["logical_cold_page_budget"]),
        capacity_fraction=float(online["capacity_fraction"]),
        cache_policy=str(online["primary_cache_policy"]),
        random_orders=int(orders["random_permutations"]),
        order_seed=int(orders["seed"]),
        epsilon_mean=float(quality["mean_signed_regret_epsilon"]),
        epsilon_query=float(quality["query_violation_epsilon"]),
        delta_empirical=float(quality["allowed_empirical_violation_rate_delta"]),
        delta_upper=float(quality["allowed_wilson_upper_delta"]),
        confidence=float(quality["confidence"]),
        bootstrap_samples=int(quality["bootstrap_samples"]),
        bootstrap_seed=int(quality["bootstrap_seed"]),
    )
    output = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": canonical_json_sha256(args.config),
        "profile": args.profile,
        "information_boundary": "calibration artifacts only; no held-out input opened",
        "compiler_output": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "profile": args.profile,
        "selected": result["selected"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

