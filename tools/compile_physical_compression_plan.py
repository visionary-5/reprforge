#!/usr/bin/env python3
"""Compile and cross-certify a physical dual-view plan without qrels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.compression_risk_metrics import (
    ranking_fidelity,
    ranking_safety_certificate,
)
from reprforge.heterogeneity_atlas import deterministic_split_roles
from reprforge.physical_compression_compiler import (
    calibrated_residual_surface_from_anchors,
    compile_upgrade_mask_from_fit,
)


def _sha256(artifact: Path) -> str:
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(archive: np.lib.npyio.NpzFile, key: str) -> list[str]:
    return [str(value) for value in archive[key]]


def _fold(query_id: str, folds: int) -> int:
    digest = hashlib.sha256(
        f"physical-compression-crossfit-v1:{query_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _certificate_summary(reference: np.ndarray, candidate: np.ndarray, seed: int):
    fidelity = ranking_fidelity(reference, candidate)
    return ranking_safety_certificate(
        fidelity,
        seed=seed,
        resamples=4000,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--cheap-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="boundary_pareto")
    parser.add_argument("--budget-fraction", type=float, default=0.65)
    parser.add_argument("--eval-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--crossfit-folds", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--full-bytes-per-vector", type=int, default=512)
    parser.add_argument("--cheap-bytes-per-vector", type=int, default=256)
    parser.add_argument("--storage-bytes-per-vector", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.crossfit_folds < 2:
        raise ValueError("crossfit requires at least two folds")

    full = np.load(args.full_runtime, allow_pickle=False)
    cheap = np.load(args.cheap_runtime, allow_pickle=False)
    query_ids = _ids(full, "query_ids")
    corpus_ids = _ids(full, "corpus_ids")
    if query_ids != _ids(cheap, "query_ids") or corpus_ids != _ids(
        cheap, "corpus_ids"
    ):
        raise ValueError("full and cheap runtime IDs differ")
    raw_full_bytes = np.asarray(full["vector_bytes"], dtype=np.int64)
    raw_cheap_bytes = np.asarray(cheap["vector_bytes"], dtype=np.int64)
    if np.any(raw_full_bytes % args.full_bytes_per_vector) or np.any(
        raw_cheap_bytes % args.cheap_bytes_per_vector
    ):
        raise ValueError("runtime vector bytes do not match declared encodings")
    full_bytes = (
        raw_full_bytes // args.full_bytes_per_vector * args.storage_bytes_per_vector
    )
    cheap_bytes = (
        raw_cheap_bytes // args.cheap_bytes_per_vector * args.storage_bytes_per_vector
    )
    roles = deterministic_split_roles(query_ids, eval_fraction=args.eval_fraction)
    outer_fit = np.flatnonzero(np.asarray(roles) == "fit")
    outer_eval = np.flatnonzero(np.asarray(roles) == "eval")
    fold_ids = np.asarray([_fold(query_ids[index], args.crossfit_folds) for index in outer_fit])
    if set(fold_ids.tolist()) != set(range(args.crossfit_folds)):
        raise ValueError("crossfit fold assignment contains an empty fold")
    crossfit_scores = np.empty(
        (len(outer_fit), len(corpus_ids)), dtype=np.float64
    )
    fold_reports = []
    for fold_index in range(args.crossfit_folds):
        calibration_local = np.flatnonzero(fold_ids == fold_index)
        training_local = np.flatnonzero(fold_ids != fold_index)
        training = outer_fit[training_local]
        calibration = outer_fit[calibration_local]
        fold_plan = compile_upgrade_mask_from_fit(
            full["scores"],
            cheap["scores"],
            training,
            full_bytes,
            budget_fraction=args.budget_fraction,
            policy=args.policy,
        )
        positions = np.asarray(fold_plan["upgraded_documents"], dtype=np.int64)
        candidate = calibrated_residual_surface_from_anchors(
            cheap["scores"][calibration],
            full["scores"][calibration][:, positions],
            positions,
            ridge=args.ridge,
        )
        crossfit_scores[calibration_local] = candidate
        fold_reports.append(
            {
                "fold": fold_index,
                "training_queries": len(training),
                "calibration_queries": len(calibration),
                "upgraded_document_count": fold_plan["upgraded_document_count"],
                "anchor_vector_bytes": fold_plan["anchor_vector_bytes"],
                "ranking_certificate": _certificate_summary(
                    full["scores"][calibration], candidate, args.seed + fold_index
                ),
            }
        )
    crossfit_certificate = _certificate_summary(
        full["scores"][outer_fit], crossfit_scores, args.seed
    )
    if crossfit_certificate["passes"]:
        final_plan = compile_upgrade_mask_from_fit(
            full["scores"],
            cheap["scores"],
            outer_fit,
            full_bytes,
            budget_fraction=args.budget_fraction,
            policy=args.policy,
        )
        selected_state = "dual_view"
        positions = final_plan["upgraded_documents"]
        anchor_bytes = final_plan["anchor_vector_bytes"]
    else:
        selected_state = "full_abstention"
        positions = list(range(len(corpus_ids)))
        anchor_bytes = int(full_bytes.sum())
    resident_bytes = (
        int(cheap_bytes.sum()) + anchor_bytes
        if selected_state == "dual_view"
        else int(full_bytes.sum())
    )
    report = {
        "schema_version": 1,
        "stage": "pre-qrel-physical-plan",
        "protocol": "qrel-free-physical-compression-crossfit-v1",
        "dataset": args.dataset,
        "qrels_loaded": False,
        "qrels_used_by_compiler": False,
        "evaluation_query_scores_used_by_compiler": False,
        "decision_unit": "physical-document",
        "policy": args.policy,
        "budget_fraction": args.budget_fraction,
        "ridge": args.ridge,
        "fit_queries": len(outer_fit),
        "evaluation_queries_reserved": len(outer_eval),
        "crossfit_folds": args.crossfit_folds,
        "fold_reports": fold_reports,
        "crossfit_ranking_certificate": crossfit_certificate,
        "selected_state": selected_state,
        "upgraded_documents": positions,
        "upgraded_document_count": len(positions),
        "corpus": len(corpus_ids),
        "cheap_vector_bytes": int(cheap_bytes.sum()),
        "full_reference_vector_bytes": int(full_bytes.sum()),
        "resident_vector_bytes": resident_bytes,
        "resident_vector_fraction": resident_bytes / int(full_bytes.sum()),
        "query_split_roles": list(roles),
        "artifacts": {
            "full_runtime_sha256": _sha256(args.full_runtime),
            "cheap_runtime_sha256": _sha256(args.cheap_runtime),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "selected_state": selected_state,
                "crossfit_ranking_certificate": crossfit_certificate,
                "resident_vector_fraction": report["resident_vector_fraction"],
                "upgraded_document_count": report["upgraded_document_count"],
                "fold_passes": [
                    value["ranking_certificate"]["passes"] for value in fold_reports
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
