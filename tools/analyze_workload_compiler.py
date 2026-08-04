#!/usr/bin/env python3
"""Run workload-level certificate-frequency physical-plan probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cohort_certificates import _irpapers_cube, _mmdocir_cube
from analyze_heterogeneity_atlas import _npz_cube
from reprforge.workload_compiler import analyze_workload_compiler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--irpapers-surface", type=Path)
    parser.add_argument("--irpapers-queries", type=Path)
    parser.add_argument("--mmdocir-root", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "datasets": {}}
    for name in ("hr", "finance"):
        cube, _ = _npz_cube(args.data_root / name)
        visual = np.load(args.data_root / name / "visual-runtime.npz", allow_pickle=False)
        report["datasets"][name] = analyze_workload_compiler(
            cube,
            base_route="text",
            expensive_route="visual",
            build_costs=visual["encode_ms"],
        )
    if args.irpapers_surface and args.irpapers_queries:
        cube = _irpapers_cube(args.irpapers_surface, args.irpapers_queries)
        report["datasets"]["irpapers"] = analyze_workload_compiler(
            cube,
            base_route="bm25",
            expensive_route="visual",
            target_metric="recall_at_5",
        )
    if args.mmdocir_root:
        cube, image_costs, metadata = _mmdocir_cube(args.mmdocir_root)
        result = analyze_workload_compiler(
            cube,
            base_route="image-pool-25",
            expensive_route="image",
            candidate_k=10,
            build_costs=image_costs,
        )
        result["evaluation_boundary"] = "official MMDocIR within-document candidates"
        result["route_storage"] = metadata
        report["datasets"]["mmdocir_pool25_to_image"] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
