#!/usr/bin/env python3
"""Replay candidate-relative full refinement over a pooled visual base."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from reprforge.candidate_fusion import analyze_candidate_fusion
from reprforge.progressive_oracle import load_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled-trace", type=Path, required=True)
    parser.add_argument("--full-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-sizes", type=int, nargs="+", default=[10, 20, 50, 100, 200]
    )
    parser.add_argument("--selected-candidate-k", type=int, default=20)
    args = parser.parse_args()
    pooled = load_trace(args.pooled_trace)
    full = load_trace(args.full_trace)
    if pooled.manifest["mode"] != "visual-pool":
        raise ValueError("pooled trace must use visual-pool mode")
    # The frozen evaluator's pair validator uses the text-mode tag to denote
    # the cheap locator side.  Only the tag is adapted; scores and costs remain
    # the measured pooled-visual trace.
    locator = replace(pooled, manifest={**pooled.manifest, "mode": "text"})
    result = analyze_candidate_fusion(
        locator,
        full,
        candidate_sizes=args.candidate_sizes,
        selected_candidate_k=args.selected_candidate_k,
    )
    result["algorithm"].update(
        {
            "name": "pooled-visual cohort refinement",
            "candidate_generation": "image-pool-9 top-K",
            "full_visual_action": "retain/score full view for candidate pages",
            "fusion": "z(pool9 within cohort) + z(full within cohort)",
        }
    )
    result["baseline_aliases"] = {
        "text_ndcg@10": "pool9_ndcg@10",
        "text_index_ms": "pool9_index_ms",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
