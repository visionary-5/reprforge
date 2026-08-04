#!/usr/bin/env python3
"""Cross-fit the marginal-value compiler on full-corpus ViDoRe traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube
from reprforge.marginal_policy_compiler import crossfit_marginal_policy_compiler


def _query_folds(query_ids: tuple[str, ...], count: int = 5) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % count
        for value in query_ids
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--text-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-z", type=float, default=0.0)
    parser.add_argument("--diagnostic-global-affine-calibration", action="store_true")
    args = parser.parse_args()
    pooled = np.load(args.pooled_runtime, allow_pickle=False)
    full = np.load(args.full_runtime, allow_pickle=False)
    text = np.load(args.text_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    for other in (full, text):
        if not np.array_equal(pooled["query_ids"], other["query_ids"]):
            raise ValueError("query IDs are not aligned")
        if not np.array_equal(pooled["corpus_ids"], other["corpus_ids"]):
            raise ValueError("corpus IDs are not aligned")
    relevance = [dict() for _ in pooled["query_ids"]]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        relevance[int(query)][int(corpus)] = float(value)
    score_surfaces = {
        "image": np.asarray(full["scores"], dtype=np.float64),
        "image-pool-9": np.asarray(pooled["scores"], dtype=np.float64),
        "text": np.asarray(text["scores"], dtype=np.float64),
    }
    calibration = {}
    if args.diagnostic_global_affine_calibration:
        teacher = score_surfaces["image"].ravel()
        for route in ("image-pool-9", "text"):
            values = score_surfaces[route].ravel()
            slope = float(
                np.cov(values, teacher, bias=True)[0, 1] / np.var(values)
            )
            intercept = float(teacher.mean() - slope * values.mean())
            score_surfaces[route] = slope * score_surfaces[route] + intercept
            calibration[route] = {"slope": slope, "intercept": intercept}
    cube = ScoreCube(
        query_ids=tuple(str(value) for value in pooled["query_ids"]),
        corpus_ids=tuple(str(value) for value in pooled["corpus_ids"]),
        scores={
            "image": score_surfaces["image"],
            "image-pool-9": score_surfaces["image-pool-9"],
            "text": score_surfaces["text"],
        },
        relevance=tuple(relevance),
        split_roles=("fit",) * len(pooled["query_ids"]),
    )
    cube.validate()
    result = crossfit_marginal_policy_compiler(
        cube,
        base_route="image-pool-9",
        route_costs={
            "image": full["vector_bytes"],
            "image-pool-9": pooled["vector_bytes"],
            "text": text["vector_bytes"],
        },
        fold_ids=_query_folds(cube.query_ids),
        budget_fractions=(0.111, 0.15, 0.25, 0.5, 0.75, 1.0),
        risk_z=args.risk_z,
    )
    result["diagnostic_global_affine_calibration"] = calibration
    result["calibration_uses_all_query_scores"] = bool(calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        "uniform_ndcg_at_5": result["uniform_ndcg_at_5"],
        "budget_curve": {
            budget: {
                key: value[key]
                for key in ("feasible", "crossfit_ndcg_at_5", "mean_cost_fraction")
                if key in value
            }
            for budget, value in result["budget_curve"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
