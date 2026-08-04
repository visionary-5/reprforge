#!/usr/bin/env python3
"""Diagnose score comparability across heterogeneous MaxSim routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.5)),
        "p99": float(np.quantile(values, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--pooled-runtime", type=Path, required=True)
    parser.add_argument("--text-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = {
        "image": np.load(args.full_runtime, allow_pickle=False),
        "image-pool-9": np.load(args.pooled_runtime, allow_pickle=False),
        "text": np.load(args.text_runtime, allow_pickle=False),
    }
    reference = loaded["image"]
    for route, values in loaded.items():
        if not np.array_equal(reference["query_ids"], values["query_ids"]):
            raise ValueError(f"query IDs are not aligned for {route}")
        if not np.array_equal(reference["corpus_ids"], values["corpus_ids"]):
            raise ValueError(f"corpus IDs are not aligned for {route}")
    labels = np.load(args.labels, allow_pickle=False)
    relevant = np.zeros(reference["scores"].shape, dtype=bool)
    relevant[labels["query_positions"], labels["corpus_positions"]] = True
    teacher = np.asarray(reference["scores"], dtype=np.float64)
    result = {"routes": {}}
    for route, values in loaded.items():
        scores = np.asarray(values["scores"], dtype=np.float64)
        x = scores.ravel()
        y = teacher.ravel()
        variance = float(np.var(x))
        slope = float(np.cov(x, y, bias=True)[0, 1] / variance) if variance else 0.0
        intercept = float(y.mean() - slope * x.mean())
        residual = y - (slope * x + intercept)
        denominator = float(np.sum((y - y.mean()) ** 2))
        correlations = []
        affine_r2 = []
        for query_index in range(scores.shape[0]):
            query_x = scores[query_index]
            query_y = teacher[query_index]
            if query_x.std() > 0.0 and query_y.std() > 0.0:
                correlations.append(float(np.corrcoef(query_x, query_y)[0, 1]))
                query_slope = float(
                    np.cov(query_x, query_y, bias=True)[0, 1] / np.var(query_x)
                )
                prediction = query_slope * query_x + (
                    query_y.mean() - query_slope * query_x.mean()
                )
                affine_r2.append(
                    1.0
                    - float(np.sum((query_y - prediction) ** 2))
                    / float(np.sum((query_y - query_y.mean()) ** 2))
                )
        result["routes"][route] = {
            "all_pairs": _summary(x),
            "relevant_pairs": _summary(scores[relevant]),
            "nonrelevant_pairs": _summary(scores[~relevant]),
            "mean_per_query_mean": float(scores.mean(axis=1).mean()),
            "std_per_query_mean": float(scores.mean(axis=1).std()),
            "mean_per_query_std": float(scores.std(axis=1).mean()),
            "global_affine_to_image": {
                "slope": slope,
                "intercept": intercept,
                "r2": 1.0 - float(np.sum(residual**2)) / denominator,
            },
            "mean_per_query_pearson_to_image": float(np.mean(correlations)),
            "mean_per_query_affine_r2_to_image": float(np.mean(affine_r2)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
