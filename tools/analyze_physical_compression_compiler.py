#!/usr/bin/env python3
"""Development-only fit/eval analysis of a physical compression compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.compression_risk_metrics import evaluate_compression_pair
from reprforge.physical_compression_compiler import (
    calibrated_residual_surface,
    compile_physical_curve,
    hybrid_score_surface,
)


def _sha256(artifact: Path) -> str:
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
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
    rows = [dict() for _ in range(query_count)]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        rows[int(query)][int(corpus)] = float(value)
    if any(not row for row in rows):
        raise ValueError("every query must have at least one relevance judgment")
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--cheap-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget-fractions",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument("--eval-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--full-bytes-per-vector", type=int, default=512)
    parser.add_argument("--cheap-bytes-per-vector", type=int, default=256)
    parser.add_argument("--storage-bytes-per-vector", type=int, default=256)
    parser.add_argument("--retain-cheap-for-upgraded", action="store_true")
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()

    full = np.load(args.full_runtime, allow_pickle=False)
    cheap = np.load(args.cheap_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    query_ids = _ids(full, "query_ids")
    corpus_ids = _ids(full, "corpus_ids")
    if query_ids != _ids(cheap, "query_ids") or corpus_ids != _ids(
        cheap, "corpus_ids"
    ):
        raise ValueError("full and cheap runtime IDs differ")
    relevance = _relevance(
        labels, query_count=len(query_ids), corpus_count=len(corpus_ids)
    )
    for name, value in (
        ("full", args.full_bytes_per_vector),
        ("cheap", args.cheap_bytes_per_vector),
        ("storage", args.storage_bytes_per_vector),
    ):
        if value <= 0:
            raise ValueError(f"{name} bytes per vector must be positive")
    raw_full_bytes = np.asarray(full["vector_bytes"], dtype=np.int64)
    raw_cheap_bytes = np.asarray(cheap["vector_bytes"], dtype=np.int64)
    if np.any(raw_full_bytes % args.full_bytes_per_vector) or np.any(
        raw_cheap_bytes % args.cheap_bytes_per_vector
    ):
        raise ValueError("runtime vector bytes are not divisible by declared encoding")
    full_storage_bytes = (
        raw_full_bytes // args.full_bytes_per_vector * args.storage_bytes_per_vector
    )
    cheap_storage_bytes = (
        raw_cheap_bytes // args.cheap_bytes_per_vector * args.storage_bytes_per_vector
    )
    plan = compile_physical_curve(
        query_ids,
        full["scores"],
        cheap["scores"],
        full_storage_bytes,
        cheap_storage_bytes,
        budget_fractions=args.budget_fractions,
        eval_fraction=args.eval_fraction,
        retain_cheap_for_upgraded=args.retain_cheap_for_upgraded,
    )
    eval_indices = np.asarray(plan["eval_query_indices"], dtype=np.int64)
    eval_relevance = tuple(relevance[index] for index in eval_indices)
    if args.policies is not None:
        unknown = set(args.policies) - set(plan["policies"])
        if unknown:
            raise ValueError(f"unknown policies: {sorted(unknown)}")
    evaluated = {}
    for policy, points in plan["policies"].items():
        if args.policies is not None and policy not in args.policies:
            continue
        evaluated_points = {"raw_replace": [], "residual_affine": []}
        for point in points:
            upgraded = np.zeros(len(corpus_ids), dtype=bool)
            upgraded[np.asarray(point["upgraded_documents"], dtype=np.int64)] = True
            resident_bytes = plan["cheap_vector_bytes"] + point["incremental_bytes"]
            surfaces = {
                "raw_replace": hybrid_score_surface(
                    full["scores"][eval_indices],
                    cheap["scores"][eval_indices],
                    upgraded,
                ),
                "residual_affine": calibrated_residual_surface(
                    full["scores"][eval_indices],
                    cheap["scores"][eval_indices],
                    upgraded,
                ),
            }
            for scoring, hybrid in surfaces.items():
                metrics = evaluate_compression_pair(
                    full["scores"][eval_indices],
                    hybrid,
                    eval_relevance,
                    bootstrap_seed=args.bootstrap_seed,
                    bootstrap_resamples=args.bootstrap_resamples,
                )
                evaluated_points[scoring].append(
                    {
                        **point,
                        "resident_bytes": resident_bytes,
                        "resident_fraction": resident_bytes / plan["full_vector_bytes"],
                        "quality": metrics["quality"],
                        "ranking_fidelity": metrics["ranking_fidelity"],
                        "qrel_free_ranking_certificate": metrics[
                            "qrel_free_ranking_certificate"
                        ],
                        "safety_gate": metrics["safety_gate"],
                    }
                )
        evaluated[policy] = evaluated_points

    report = {
        "schema_version": 2,
        "stage": "development_only_physical_compiler_probe",
        "dataset": args.dataset,
        "decision_unit": "physical-document",
        "deployable_as_measured": False,
        "materialization_status": "exact_score_surface_replay; combined bank not yet built",
        "qrels_used_by_compiler": False,
        "split_semantics": "stable query-hash workload recurrence",
        "fit_queries": len(plan["fit_query_indices"]),
        "eval_queries": len(plan["eval_query_indices"]),
        "corpus": len(corpus_ids),
        "cheap_vector_bytes": plan["cheap_vector_bytes"],
        "full_vector_bytes": plan["full_vector_bytes"],
        "retain_cheap_for_upgraded": plan["retain_cheap_for_upgraded"],
        "score_comparability": {
            "raw_replace": "replace cheap score with full score for upgraded documents",
            "residual_affine": "cheap cover plus query-local affine residual completion from dual-view anchors",
        },
        "policies": evaluated,
        "artifacts": {
            "full_runtime_sha256": _sha256(args.full_runtime),
            "cheap_runtime_sha256": _sha256(args.cheap_runtime),
            "labels_sha256": _sha256(args.labels),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    compact = {
        "dataset": args.dataset,
        "fit_queries": report["fit_queries"],
        "eval_queries": report["eval_queries"],
        "policies": {
            policy: {
                scoring: [
                    {
                        "resident_fraction": point["resident_fraction"],
                        "top10_overlap": point["ranking_fidelity"]["top_10_overlap"][
                            "mean"
                        ],
                        "ndcg10_regret": point["quality"]["ndcg_at_10"][
                            "mean_regret"
                        ],
                        "ndcg10_upper": point["safety_gate"][
                            "ndcg_at_10_upper_regret"
                        ],
                        "recall100_upper": point["safety_gate"][
                            "recall_at_100_upper_regret"
                        ],
                        "safe": point["safety_gate"]["passes"],
                    }
                    for point in scoring_points
                ]
                for scoring, scoring_points in policy_points.items()
            }
            for policy, policy_points in evaluated.items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
