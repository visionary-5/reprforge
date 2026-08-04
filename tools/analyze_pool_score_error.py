#!/usr/bin/env python3
"""Decompose pooled-cover score error into underestimation and overshoot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return {
        "mean_full_minus_pool": float(flat.mean()),
        "mean_absolute_error": float(np.abs(flat).mean()),
        "pool_underestimation_fraction": float(np.mean(flat > 0)),
        "pool_overshoot_fraction": float(np.mean(flat < 0)),
        "pool_underestimation_gt_0_05_fraction": float(np.mean(flat > 0.05)),
        "pool_overshoot_gt_0_05_fraction": float(np.mean(flat < -0.05)),
        "p05_full_minus_pool": float(np.quantile(flat, 0.05)),
        "median_full_minus_pool": float(np.median(flat)),
        "p95_full_minus_pool": float(np.quantile(flat, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        nargs=3,
        metavar=("DOMAIN", "FULL_RUNTIME", "POOL_RUNTIME"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {}
    for domain, full_path, pool_path in args.input:
        full = np.load(full_path, allow_pickle=False)
        pool = np.load(pool_path, allow_pickle=False)
        if not np.array_equal(full["query_ids"], pool["query_ids"]):
            raise ValueError(f"{domain}: query IDs differ")
        if not np.array_equal(full["corpus_ids"], pool["corpus_ids"]):
            raise ValueError(f"{domain}: corpus IDs differ")
        full_scores = np.asarray(full["scores"], dtype=np.float64)
        pool_scores = np.asarray(pool["scores"], dtype=np.float64)
        difference = full_scores - pool_scores
        positions = np.arange(full_scores.shape[1])
        full_order = np.stack(
            [np.lexsort((positions, -row)) for row in full_scores]
        )
        pool_order = np.stack(
            [np.lexsort((positions, -row)) for row in pool_scores]
        )
        reports[domain] = {"all_pairs": _summary(difference)}
        for depth in (10, 100):
            rows = np.arange(len(full_scores))[:, None]
            reports[domain][f"full_top_{depth}"] = _summary(
                difference[rows, full_order[:, :depth]]
            )
            reports[domain][f"pool_top_{depth}"] = _summary(
                difference[rows, pool_order[:, :depth]]
            )
    result = {
        "interpretation": (
            "positive full-minus-pool is repairable by adding full tokens; "
            "negative values are centroid overshoot and cannot be lowered by "
            "an append-only residual overlay"
        ),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
