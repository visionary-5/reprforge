#!/usr/bin/env python3
"""Evaluate qrel-free error-bounded ladder compilation on MMDocIR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cohort_certificates import _mmdocir_cube
from reprforge.ladder_compiler import analyze_error_bounded_ladder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdocir-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cube, _, metadata = _mmdocir_cube(args.mmdocir_root)
    item_rows = metadata["item_rows"]
    route_costs = {
        route: np.asarray(
            [row["route_costs"][route]["index_bytes"] for row in item_rows],
            dtype=np.float64,
        )
        for route in cube.routes
    }
    typed_plan = tuple(
        "image"
        if row["content_type"] == "table"
        else (
            "image-pool-9"
            if row["content_type"] in {"chart", "figure", "image"}
            else "image-pool-25"
        )
        for row in item_rows
    )
    result = analyze_error_bounded_ladder(
        cube,
        teacher_route="image",
        candidate_indices=metadata["candidate_indices"],
        route_costs=route_costs,
        reference_plans={"typed-capacity-v1": typed_plan},
    )
    result["evaluation_boundary"] = "official MMDocIR within-document candidates"
    result["uniform_route_index_bytes"] = metadata["full_route_index_bytes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
