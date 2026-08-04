#!/usr/bin/env python3
"""Run feature-group listwise ladder compilation on MMDocIR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cohort_certificates import _mmdocir_cube
from analyze_type_policy_compiler import _document_folds
from reprforge.group_policy_compiler import crossfit_group_policy_compiler


def _feature_groups(item_rows: list[dict]) -> tuple[str, ...]:
    visual_types = {"image", "table"}
    medians = {}
    for content_type in visual_types:
        rows = [row for row in item_rows if row["content_type"] == content_type]
        medians[content_type] = {
            name: float(np.median([row["construction_features"][name] for row in rows]))
            for name in ("grayscale_entropy", "edge_energy")
        }
    groups = []
    for row in item_rows:
        content_type = str(row["content_type"])
        if content_type not in visual_types:
            groups.append(f"type={content_type}")
            continue
        features = row["construction_features"]
        entropy = int(
            features["grayscale_entropy"] >= medians[content_type]["grayscale_entropy"]
        )
        edge = int(features["edge_energy"] >= medians[content_type]["edge_energy"])
        groups.append(f"type={content_type}|entropy={entropy}|edge={edge}")
    return tuple(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdocir-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cube, _, metadata = _mmdocir_cube(args.mmdocir_root)
    item_rows = metadata["item_rows"]
    groups = _feature_groups(item_rows)
    route_costs = {
        route: np.asarray(
            [row["route_costs"][route]["index_bytes"] for row in item_rows],
            dtype=np.float64,
        )
        for route in cube.routes
    }
    typed = {}
    for group in set(groups):
        content_type = group.split("|", 1)[0].split("=", 1)[1]
        typed[group] = (
            "image"
            if content_type == "table"
            else "image-pool-9" if content_type == "image" else "image-pool-25"
        )
    report = crossfit_group_policy_compiler(
        cube,
        item_groups=groups,
        candidate_indices=metadata["candidate_indices"],
        route_costs=route_costs,
        fold_ids=_document_folds(metadata["query_document_indices"]),
        initial_mappings=(typed,),
    )
    report["group_features"] = (
        "content type; within visual types, corpus-median grayscale entropy and edge energy"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
