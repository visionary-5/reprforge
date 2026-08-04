#!/usr/bin/env python3
"""Evaluate one compressed score surface under the frozen risk contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.compression_risk_metrics import (
    evaluate_compression_pair,
    summarize_costs,
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


def _relevance(
    labels: np.lib.npyio.NpzFile, *, query_count: int, corpus_count: int
) -> tuple[dict[int, float], ...]:
    required = ("query_positions", "corpus_positions", "relevance")
    if any(key not in labels for key in required):
        raise ValueError("label archive is missing position or relevance arrays")
    rows = [dict() for _ in range(query_count)]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        query_index = int(query)
        corpus_index = int(corpus)
        if not 0 <= query_index < query_count or not 0 <= corpus_index < corpus_count:
            raise ValueError("label position lies outside the score surface")
        rows[query_index][corpus_index] = float(value)
    if any(not row for row in rows):
        raise ValueError("every query must have at least one relevance judgment")
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-runtime", type=Path, required=True)
    parser.add_argument("--candidate-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--decision-unit",
        choices=("configuration", "physical-unit", "query"),
        default="configuration",
    )
    parser.add_argument("--deployable-as-measured", action="store_true")
    parser.add_argument("--reference-vector-bytes", type=int)
    parser.add_argument("--candidate-vector-bytes", type=int)
    parser.add_argument(
        "--vector-bytes-kind",
        choices=("persistent-bank", "resident-index", "logical-estimate"),
    )
    parser.add_argument("--reference-token-work", type=int)
    parser.add_argument("--candidate-token-work", type=int)
    parser.add_argument(
        "--token-work-kind",
        choices=("document-vectors-per-exhaustive-query", "measured-maxsim-ops"),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    args = parser.parse_args()

    reference = np.load(args.reference_runtime, allow_pickle=False)
    candidate = np.load(args.candidate_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    reference_queries = _ids(reference, "query_ids")
    candidate_queries = _ids(candidate, "query_ids")
    reference_corpus = _ids(reference, "corpus_ids")
    candidate_corpus = _ids(candidate, "corpus_ids")
    if reference_queries != candidate_queries:
        raise ValueError("reference and candidate query IDs differ")
    if reference_corpus != candidate_corpus:
        raise ValueError("reference and candidate corpus IDs differ")
    relevance = _relevance(
        labels,
        query_count=len(reference_queries),
        corpus_count=len(reference_corpus),
    )
    report = evaluate_compression_pair(
        reference["scores"],
        candidate["scores"],
        relevance,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    cost_values = (
        args.reference_vector_bytes,
        args.candidate_vector_bytes,
        args.reference_token_work,
        args.candidate_token_work,
    )
    if any(value is not None for value in cost_values):
        if args.reference_vector_bytes is None or args.candidate_vector_bytes is None:
            raise ValueError("both vector byte counts are required when costs are supplied")
        if args.vector_bytes_kind is None:
            raise ValueError("--vector-bytes-kind is required with cost inputs")
        report["cost"] = summarize_costs(
            reference_vector_bytes=args.reference_vector_bytes,
            candidate_vector_bytes=args.candidate_vector_bytes,
            vector_bytes_kind=args.vector_bytes_kind,
            reference_token_work=args.reference_token_work,
            candidate_token_work=args.candidate_token_work,
            token_work_kind=args.token_work_kind,
        )
    report.update(
        {
            "schema_version": 1,
            "protocol": "qrel-free-compression-risk-2026-08-04",
            "dataset": args.dataset,
            "candidate": args.candidate_name,
            "decision_unit": args.decision_unit,
            "deployable_as_measured": bool(args.deployable_as_measured),
            "qrels_used_by_compiler": False,
            "qrels_used_for_final_evaluation_only": True,
            "artifacts": {
                "reference_runtime_sha256": _sha256(args.reference_runtime),
                "candidate_runtime_sha256": _sha256(args.candidate_runtime),
                "labels_sha256": _sha256(args.labels),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "candidate": report["candidate"],
                "quality": {
                    name: {
                        "candidate_mean": values["candidate_mean"],
                        "mean_regret": values["mean_regret"],
                        "p95_regret": values["p95_regret"],
                        "worst_5pct_cvar": values["worst_5pct_cvar"],
                        "catastrophic_harm_fraction": values[
                            "catastrophic_harm_fraction"
                        ],
                    }
                    for name, values in report["quality"].items()
                },
                "ranking_fidelity": {
                    name: values["mean"]
                    for name, values in report["ranking_fidelity"].items()
                },
                "safety_gate": report["safety_gate"],
                "qrel_free_ranking_certificate": report[
                    "qrel_free_ranking_certificate"
                ],
                "cost": report.get("cost"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
