#!/usr/bin/env python3
"""Certify compressed ranking fidelity without loading relevance labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.compression_risk_metrics import (
    bootstrap_mean_interval,
    ranking_fidelity,
    ranking_safety_certificate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(archive: np.lib.npyio.NpzFile, key: str) -> list[str]:
    if key not in archive:
        raise ValueError(f"archive is missing {key}")
    return [str(value) for value in archive[key]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-runtime", type=Path, required=True)
    parser.add_argument("--candidate-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    args = parser.parse_args()

    reference = np.load(args.reference_runtime, allow_pickle=False)
    candidate = np.load(args.candidate_runtime, allow_pickle=False)
    reference_queries = _ids(reference, "query_ids")
    candidate_queries = _ids(candidate, "query_ids")
    reference_corpus = _ids(reference, "corpus_ids")
    candidate_corpus = _ids(candidate, "corpus_ids")
    if reference_queries != candidate_queries:
        raise ValueError("reference and candidate query IDs differ")
    if reference_corpus != candidate_corpus:
        raise ValueError("reference and candidate corpus IDs differ")

    observations = ranking_fidelity(reference["scores"], candidate["scores"])
    summaries = {
        name: bootstrap_mean_interval(
            values,
            seed=args.bootstrap_seed,
            resamples=args.bootstrap_resamples,
        )
        for name, values in observations.items()
    }
    certificate = ranking_safety_certificate(
        observations,
        seed=args.bootstrap_seed,
        resamples=args.bootstrap_resamples,
    )
    report = {
        "schema_version": 1,
        "protocol": "qrel-free-compression-risk-2026-08-04",
        "stage": "pre-qrel-ranking-certification",
        "dataset": args.dataset,
        "candidate": args.candidate_name,
        "query_count": len(reference_queries),
        "corpus_count": len(reference_corpus),
        "qrel_path_argument_available": False,
        "qrels_loaded": False,
        "ranking_fidelity": summaries,
        "qrel_free_ranking_certificate": certificate,
        "artifacts": {
            "reference_runtime_sha256": _sha256(args.reference_runtime),
            "candidate_runtime_sha256": _sha256(args.candidate_runtime),
        },
        "bootstrap": {
            "seed": args.bootstrap_seed,
            "resamples": args.bootstrap_resamples,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "candidate": args.candidate_name,
                "ranking_fidelity": {
                    name: values["mean"] for name, values in summaries.items()
                },
                "qrel_free_ranking_certificate": certificate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
