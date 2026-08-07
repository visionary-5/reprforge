#!/usr/bin/env python3
"""Run the frozen defer/materialize candidate-boundary kill test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reprforge.defer_materialize_boundary import locator_boundary, repair_reuse_crossfit
from reprforge.partial_vlm_inputs import load_irpapers_surface, load_vidore_surface
from reprforge.partial_vlm_materialization import fold_assignments


def _parse_vidore(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=ROOT")
    name, root = value.split("=", 1)
    return name, Path(root)


def _canonical_sha(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _domain(surface: Any, config: dict[str, Any]) -> dict[str, Any]:
    split_config = config["cross_fit"]
    assignments = fold_assignments(
        surface, int(split_config["folds"]), int(split_config["seed"])
    )
    curves = {
        str(depth): locator_boundary(surface, int(depth))
        for depth in config["candidate_depths"]
    }
    reuse = {
        str(depth): repair_reuse_crossfit(surface, assignments, int(depth))
        for depth in config["candidate_depths"]
    }
    return {
        "queries": surface.queries,
        "corpus_pages": surface.pages,
        "input_sha256": surface.input_sha256,
        "locator_curves": curves,
        "repair_reuse_crossfit": reuse,
    }


def _gate(domains: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["gate"]
    depth = str(gate["depth"])
    rows = {}
    passing = 0
    for name, domain in domains.items():
        boundary = domain["locator_curves"][depth]["boundary"]
        reuse = domain["repair_reuse_crossfit"][depth]
        event_overlap = reuse["event_overlap_fraction_weighted"]
        checks = {
            "enough_text_misses": (
                boundary["text_miss_queries"] >= int(gate["minimum_text_miss_queries"])
                and boundary["text_miss_fraction"]
                >= float(gate["minimum_text_miss_fraction"])
            ),
            "visual_repairs_enough_misses": (
                boundary["visual_repairs_fraction_of_text_misses"] is not None
                and boundary["visual_repairs_fraction_of_text_misses"]
                >= float(gate["minimum_visual_repair_fraction_of_text_misses"])
            ),
            "repair_events_recur": (
                event_overlap is not None
                and event_overlap >= float(gate["minimum_future_repair_event_overlap"])
            ),
        }
        passed = all(checks.values())
        passing += int(passed)
        rows[name] = {
            "depth": int(depth),
            "text_miss_queries": boundary["text_miss_queries"],
            "text_miss_fraction": boundary["text_miss_fraction"],
            "visual_repair_queries": boundary["visual_repair_queries"],
            "visual_repairs_fraction_of_text_misses": boundary[
                "visual_repairs_fraction_of_text_misses"
            ],
            "repair_event_overlap_fraction": event_overlap,
            "checks": checks,
            "passes": passed,
        }
    required = int(gate["domains_required"])
    return {
        "depth": int(depth),
        "domains": rows,
        "passing_domains": passing,
        "required_domains": required,
        "passes_phase_diagram_gate": passing >= required,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vidore", action="append", type=_parse_vidore, default=[])
    parser.add_argument(
        "--irpapers",
        nargs=4,
        metavar=("NAME", "SCORES", "QUERIES", "RUN_MANIFEST"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    surfaces = [load_vidore_surface(name, root) for name, root in args.vidore]
    if args.irpapers:
        name, scores, queries, manifest = args.irpapers
        surfaces.append(
            load_irpapers_surface(name, Path(scores), Path(queries), Path(manifest))
        )
    if not surfaces:
        parser.error("at least one domain is required")
    domains = {surface.name: _domain(surface, config) for surface in surfaces}
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": _canonical_sha(args.config),
        "status": "complete",
        "domains": domains,
        "gate": _gate(domains, config),
        "warning": "Retrieval surfaces measure localization only; they do not stand in for query-conditioned VLM answer quality or latency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
