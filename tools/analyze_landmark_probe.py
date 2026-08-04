#!/usr/bin/env python3
"""Run query-local landmark completion over frozen Atlas surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_cohort_certificates import _irpapers_cube, _mmdocir_cube
from analyze_heterogeneity_atlas import _npz_cube
from reprforge.landmark_probe import analyze_landmark_budgets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-k", type=int, default=10)
    parser.add_argument("--irpapers-surface", type=Path)
    parser.add_argument("--irpapers-queries", type=Path)
    parser.add_argument("--mmdocir-root", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "datasets": {}}
    for name in ("hr", "finance"):
        cube, _ = _npz_cube(args.data_root / name)
        report["datasets"][name] = analyze_landmark_budgets(
            cube,
            base_route="text",
            expensive_route="visual",
            candidate_k=20,
            budgets=(2, 4, 6, 8, 12, 16, 20),
            target_metric=f"ndcg_at_{args.target_k}",
            target_k=args.target_k,
        )
    if args.irpapers_surface and args.irpapers_queries:
        cube = _irpapers_cube(args.irpapers_surface, args.irpapers_queries)
        report["datasets"]["irpapers"] = analyze_landmark_budgets(
            cube,
            base_route="bm25",
            expensive_route="visual",
            candidate_k=20,
            budgets=(2, 4, 6, 8, 12, 16, 20),
            target_metric=f"recall_at_{args.target_k}",
            target_k=args.target_k,
        )
    if args.mmdocir_root:
        cube, _, metadata = _mmdocir_cube(args.mmdocir_root)
        result = analyze_landmark_budgets(
            cube,
            base_route="image-pool-25",
            expensive_route="image",
            candidate_k=10,
            budgets=(2, 4, 6, 8, 10),
            target_metric=f"ndcg_at_{args.target_k}",
            target_k=args.target_k,
        )
        result["evaluation_boundary"] = "official MMDocIR within-document candidates"
        result["route_storage"] = metadata
        report["datasets"]["mmdocir_pool25_to_image"] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
