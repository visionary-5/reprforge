#!/usr/bin/env python3
"""Measure the qrel-free spectrum of a full-minus-cheap workload surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reprforge.heterogeneity_atlas import deterministic_split_roles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--cheap-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=(8, 16, 32, 64, 128))
    args = parser.parse_args()

    full = np.load(args.full_runtime, allow_pickle=False)
    cheap = np.load(args.cheap_runtime, allow_pickle=False)
    if not np.array_equal(full["query_ids"], cheap["query_ids"]) or not np.array_equal(
        full["corpus_ids"], cheap["corpus_ids"]
    ):
        raise ValueError("full and cheap runtime IDs differ")
    query_ids = [str(value) for value in full["query_ids"]]
    roles = deterministic_split_roles(query_ids)
    fit = np.flatnonzero(np.asarray(roles) == "fit")
    residual = np.asarray(full["scores"], dtype=np.float64)[fit] - np.asarray(
        cheap["scores"], dtype=np.float64
    )[fit]
    centered = residual - residual.mean(axis=1, keepdims=True)
    standardized = centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-8)
    singular = np.linalg.svd(standardized, compute_uv=False)
    energy = singular * singular
    probabilities = energy / energy.sum()
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    report = {
        "schema_version": 1,
        "stage": "qrel-free-residual-spectrum",
        "dataset": args.dataset,
        "fit_queries": len(fit),
        "corpus": residual.shape[1],
        "matrix_rank_limit": min(residual.shape),
        "stable_rank": float(energy.sum() / energy[0]),
        "entropy_effective_rank": float(np.exp(entropy)),
        "energy_fraction": {
            str(rank): float(energy[: min(rank, len(energy))].sum() / energy.sum())
            for rank in args.ranks
        },
        "singular_values": singular.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "dataset", "fit_queries", "corpus", "stable_rank", "entropy_effective_rank", "energy_fraction"
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
