#!/usr/bin/env python3
"""Evaluate truncated-SVD residual anchors on a held-out query workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.compression_risk_metrics import evaluate_compression_pair
from reprforge.heterogeneity_atlas import deterministic_split_roles
from reprforge.residual_column_compiler import (
    fit_low_rank_residual_model,
    low_rank_residual_score_surface,
    two_stage_candidate_surface,
)


def _sha256(artifact: Path) -> str:
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(archive, key):
    return [str(value) for value in archive[key]]


def _relevance(labels, query_count):
    rows = [dict() for _ in range(query_count)]
    for query, corpus, value in zip(
        labels["query_positions"], labels["corpus_positions"], labels["relevance"], strict=True
    ):
        rows[int(query)][int(corpus)] = float(value)
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--cheap-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--anchor-counts", type=int, nargs="+", default=(32, 64, 128))
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--full-bytes-per-vector", type=int, default=512)
    parser.add_argument("--cheap-bytes-per-vector", type=int, default=256)
    parser.add_argument("--storage-bytes-per-vector", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    args = parser.parse_args()

    full = np.load(args.full_runtime, allow_pickle=False)
    cheap = np.load(args.cheap_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    query_ids = _ids(full, "query_ids")
    corpus_ids = _ids(full, "corpus_ids")
    if query_ids != _ids(cheap, "query_ids") or corpus_ids != _ids(cheap, "corpus_ids"):
        raise ValueError("full and cheap runtime IDs differ")
    roles = deterministic_split_roles(query_ids)
    fit = np.flatnonzero(np.asarray(roles) == "fit")
    evaluation = np.flatnonzero(np.asarray(roles) == "eval")
    relevance = _relevance(labels, len(query_ids))
    full_bytes = np.asarray(full["vector_bytes"], dtype=np.int64)
    cheap_bytes = np.asarray(cheap["vector_bytes"], dtype=np.int64)
    if np.any(full_bytes % args.full_bytes_per_vector) or np.any(cheap_bytes % args.cheap_bytes_per_vector):
        raise ValueError("runtime bytes do not match declared vector encodings")
    full_bytes = full_bytes // args.full_bytes_per_vector * args.storage_bytes_per_vector
    cheap_bytes = cheap_bytes // args.cheap_bytes_per_vector * args.storage_bytes_per_vector
    residual_fit = full["scores"][fit] - cheap["scores"][fit]
    points = []
    for count in args.anchor_counts:
        model = fit_low_rank_residual_model(
            residual_fit,
            rank=args.rank,
            anchor_count=count,
            costs=full_bytes,
            ridge=args.ridge,
        )
        predicted = low_rank_residual_score_surface(
            cheap["scores"][evaluation],
            full["scores"][evaluation][:, model.anchor_positions],
            model,
        )
        candidate = two_stage_candidate_surface(
            cheap["scores"][evaluation], predicted, candidate_k=args.candidate_k
        )
        metrics = evaluate_compression_pair(
            full["scores"][evaluation],
            candidate,
            tuple(relevance[index] for index in evaluation),
            bootstrap_seed=20260804,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        resident = int(cheap_bytes.sum() + full_bytes[model.anchor_positions].sum())
        points.append({
            "anchor_count": count,
            "anchor_positions": model.anchor_positions.tolist(),
            "resident_vector_fraction": resident / int(full_bytes.sum()),
            "quality": metrics["quality"],
            "ranking_fidelity": metrics["ranking_fidelity"],
            "qrel_free_ranking_certificate": metrics["qrel_free_ranking_certificate"],
            "safety_gate": metrics["safety_gate"],
        })
    report = {
        "schema_version": 1,
        "stage": "development_low_rank_residual_probe",
        "dataset": args.dataset,
        "rank": args.rank,
        "ridge": args.ridge,
        "candidate_k": args.candidate_k,
        "qrels_used_by_compiler": False,
        "fit_queries": len(fit),
        "eval_queries": len(evaluation),
        "points": points,
        "artifacts": {
            "full_runtime_sha256": _sha256(args.full_runtime),
            "cheap_runtime_sha256": _sha256(args.cheap_runtime),
            "labels_sha256": _sha256(args.labels),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dataset": args.dataset, "points": [{
        "anchors": point["anchor_count"],
        "resident_fraction": point["resident_vector_fraction"],
        "top10_overlap": point["ranking_fidelity"]["top_10_overlap"]["mean"],
        "ndcg10_regret": point["quality"]["ndcg_at_10"]["mean_regret"],
        "ndcg10_upper": point["safety_gate"]["ndcg_at_10_upper_regret"],
        "recall100_upper": point["safety_gate"]["recall_at_100_upper_regret"],
        "safe": point["safety_gate"]["passes"],
    } for point in points]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
