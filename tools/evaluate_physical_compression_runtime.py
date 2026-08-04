#!/usr/bin/env python3
"""Evaluate a materialized dual-view runtime on its held-out query split."""

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
from reprforge.heterogeneity_atlas import deterministic_split_roles


def _sha256(artifact: Path) -> str:
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(archive: np.lib.npyio.NpzFile, key: str) -> list[str]:
    return [str(value) for value in archive[key]]


def _relevance(labels, *, query_count: int, corpus_count: int):
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
    parser.add_argument("--reference-runtime", type=Path, required=True)
    parser.add_argument("--candidate-runtime", type=Path, required=True)
    parser.add_argument("--anchor-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    args = parser.parse_args()

    reference = np.load(args.reference_runtime, allow_pickle=False)
    candidate = np.load(args.candidate_runtime, allow_pickle=False)
    anchors = np.load(args.anchor_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    materialization = json.loads(args.materialization.read_text())
    certificate = json.loads(args.certificate.read_text())
    if certificate.get("stage") != "pre-qrel-physical-runtime-certificate":
        raise ValueError("certificate is not a pre-qrel physical runtime certificate")
    if certificate.get("qrels_loaded") is not False:
        raise ValueError("certificate does not attest qrels_loaded=false")
    expected_hashes = {
        "reference_runtime_sha256": _sha256(args.reference_runtime),
        "candidate_runtime_sha256": _sha256(args.candidate_runtime),
        "materialization_sha256": _sha256(args.materialization),
    }
    for key, value in expected_hashes.items():
        if certificate.get("artifacts", {}).get(key) != value:
            raise ValueError(f"certificate {key} differs from evaluation input")
    query_ids = _ids(reference, "query_ids")
    corpus_ids = _ids(reference, "corpus_ids")
    if query_ids != _ids(candidate, "query_ids") or corpus_ids != _ids(
        candidate, "corpus_ids"
    ):
        raise ValueError("reference and candidate runtime IDs differ")
    if query_ids != _ids(anchors, "query_ids"):
        raise ValueError("reference and anchor query IDs differ")
    positions = np.asarray(materialization["anchor_positions"], dtype=np.int64)
    if [corpus_ids[int(index)] for index in positions] != _ids(
        anchors, "corpus_ids"
    ):
        raise ValueError("anchor IDs do not match materialization positions")
    reference_anchor_scores = np.asarray(reference["scores"])[:, positions]
    anchor_scores = np.asarray(anchors["scores"])
    anchor_difference = np.abs(reference_anchor_scores - anchor_scores)
    relevance = _relevance(
        labels, query_count=len(query_ids), corpus_count=len(corpus_ids)
    )
    roles = deterministic_split_roles(query_ids, eval_fraction=args.eval_fraction)
    evaluation = np.flatnonzero(np.asarray(roles) == "eval")
    report = evaluate_compression_pair(
        reference["scores"][evaluation],
        candidate["scores"][evaluation],
        tuple(relevance[index] for index in evaluation),
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    report.update(
        {
            "schema_version": 1,
            "stage": "post-materialization-heldout-evaluation",
            "dataset": args.dataset,
            "decision_unit": "physical-document",
            "deployable_as_measured": True,
            "qrels_used_by_compiler": False,
            "qrels_used_for_final_evaluation_only": True,
            "fit_queries": len(query_ids) - len(evaluation),
            "eval_queries": len(evaluation),
            "architecture": materialization["architecture"],
            "anchor_count": materialization["anchor_count"],
            "anchor_score_parity": {
                "max_abs_difference_vs_reference_surface": float(
                    anchor_difference.max()
                ),
                "mean_abs_difference_vs_reference_surface": float(
                    anchor_difference.mean()
                ),
            },
            "cost": summarize_costs(
                reference_vector_bytes=materialization[
                    "full_reference_vector_bytes"
                ],
                candidate_vector_bytes=materialization["combined_vector_bytes"],
                vector_bytes_kind="persistent-bank",
            ),
            "artifacts": {
                "reference_runtime_sha256": _sha256(args.reference_runtime),
                "candidate_runtime_sha256": _sha256(args.candidate_runtime),
                "anchor_runtime_sha256": _sha256(args.anchor_runtime),
                "labels_sha256": _sha256(args.labels),
                "materialization_sha256": _sha256(args.materialization),
                "pre_qrel_certificate_sha256": _sha256(args.certificate),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "anchor_score_parity": report["anchor_score_parity"],
                "cost": report["cost"],
                "quality": {
                    name: {
                        "mean_regret": values["mean_regret"],
                        "worst_5pct_cvar": values["worst_5pct_cvar"],
                    }
                    for name, values in report["quality"].items()
                },
                "ranking_certificate": report["qrel_free_ranking_certificate"],
                "safety_gate": report["safety_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
