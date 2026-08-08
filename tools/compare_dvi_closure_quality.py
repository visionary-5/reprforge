#!/usr/bin/env python3
"""Compare DVI-like and closure endpoints on the same frozen query cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.evaluate_closure_materialization import (
    _ndcg,
    bm25_full_rerank,
    candidate_rerank,
    load_scored_ranking,
)
from tools.evaluate_value_aware_materialization import load_exported_surface


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--locator-ranking", type=Path, required=True)
    parser.add_argument("--dvi-result", type=Path, required=True)
    parser.add_argument(
        "--dvi-visual-locator-label", default="full_visual_index"
    )
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    surface, _, _ = load_exported_surface(args.score_root, args.dataset_root)
    dvi = json.loads(args.dvi_result.read_text())
    selected_ids = list(map(str, dvi["selection"]["main_queries"]))
    positions = {str(value): index for index, value in enumerate(surface.query_ids)}
    missing = sorted(set(selected_ids) - set(positions))
    if missing:
        raise ValueError(f"DVI queries absent from score surface: {missing[:3]}")
    queries = [positions[value] for value in selected_ids]
    locator_order, locator_scores = load_scored_ranking(args.locator_ranking, surface)
    if args.depth > locator_order.shape[1]:
        raise ValueError("depth exceeds locator ranking")
    candidates = locator_order[:, : args.depth]
    scores = locator_scores[:, : args.depth]
    closure = candidate_rerank(surface, candidates, scores, method="visual")
    full_visual = surface.visual_order
    result = {
        "schema_version": 1,
        "protocol": "paired-dvi-closure-quality-v0",
        "domain": surface.name,
        "queries": len(queries),
        "query_ids": selected_ids,
        "depth": args.depth,
        "quality": {
            "text_only": _ndcg(surface, queries, surface.text_order),
            "full_visual_index": _ndcg(surface, queries, full_visual),
            "bm25_topk_full_visual_rerank": _ndcg(
                surface, queries, bm25_full_rerank(surface, args.depth)
            ),
            "hpool_locator_only": _ndcg(surface, queries, candidates),
            "hpool_topk_full_visual_rerank": _ndcg(surface, queries, closure),
            "dvi_like_bm25_top20_raw_page_verifier": dvi["cohorts"]["main"][
                "routes"
            ]["bm25_20"]["after_vlm"],
            "dvi_like_hybrid_top20_raw_page_verifier": dvi["cohorts"]["main"][
                "routes"
            ]["hybrid_10_10"]["after_vlm"],
            f"dvi_like_{args.dvi_visual_locator_label}_top20_raw_page_verifier": (
                dvi["cohorts"]["main"]["routes"]["visual_20"]["after_vlm"]
            ),
        },
        "warning": (
            "The query cohort is paired, but the systems do not share candidate "
            "sets or scoring semantics except when the supplied DVI visual route "
            "uses the same locator ranking. The hybrid DVI-like route consumes "
            "visual Top-10 candidates and is not a zero-visual-index baseline."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["quality"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
