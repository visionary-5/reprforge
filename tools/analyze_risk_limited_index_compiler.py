#!/usr/bin/env python3
"""Simulate label-efficient selection over a physical dual-view plan ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.compression_risk_metrics import (
    evaluate_compression_pair,
    ranking_fidelity,
)
from reprforge.heterogeneity_atlas import query_metrics
from reprforge.physical_compression_compiler import (
    calibrated_residual_surface,
    compile_physical_curve,
)
from reprforge.risk_limited_index_compiler import (
    quantile_risk_strata,
    simulate_label_efficiency,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
            raise ValueError("label position lies outside score surface")
        rows[query_index][corpus_index] = float(value)
    if any(not row for row in rows):
        raise ValueError("every query must have at least one relevance judgment")
    return tuple(rows)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.lexsort((np.arange(len(values)), values))
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result

    left_rank = ranks(left)
    right_rank = ranks(right)
    if not left_rank.std() or not right_rank.std():
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _three_way_fold(query_id: str) -> int:
    digest = hashlib.sha256(f"risk-index-three-way-v1::{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 3


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
        default=(0.2, 0.4, 0.5, 0.6, 0.65, 0.7),
    )
    parser.add_argument(
        "--label-budgets", type=int, nargs="+", default=(16, 24, 32, 48, 64)
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--strata", type=int, default=4)
    parser.add_argument("--rotation", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--eval-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--full-bytes-per-vector", type=int, default=512)
    parser.add_argument("--cheap-bytes-per-vector", type=int, default=256)
    parser.add_argument("--storage-bytes-per-vector", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    full = np.load(args.full_runtime, allow_pickle=False)
    cheap = np.load(args.cheap_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    query_ids = _ids(full, "query_ids")
    corpus_ids = _ids(full, "corpus_ids")
    if query_ids != _ids(cheap, "query_ids") or corpus_ids != _ids(cheap, "corpus_ids"):
        raise ValueError("full and cheap runtime IDs differ")
    relevance = _relevance(labels, query_count=len(query_ids), corpus_count=len(corpus_ids))
    raw_full_bytes = np.asarray(full["vector_bytes"], dtype=np.int64)
    raw_cheap_bytes = np.asarray(cheap["vector_bytes"], dtype=np.int64)
    if np.any(raw_full_bytes % args.full_bytes_per_vector) or np.any(
        raw_cheap_bytes % args.cheap_bytes_per_vector
    ):
        raise ValueError("runtime bytes are incompatible with declared encodings")
    full_storage_bytes = raw_full_bytes // args.full_bytes_per_vector * args.storage_bytes_per_vector
    cheap_storage_bytes = raw_cheap_bytes // args.cheap_bytes_per_vector * args.storage_bytes_per_vector
    folds = np.asarray([_three_way_fold(query_id) for query_id in query_ids])
    plan_fit_rows = np.flatnonzero(folds == args.rotation)
    calibration_rows = np.flatnonzero(folds == (args.rotation + 1) % 3)
    audit_rows = np.flatnonzero(folds == (args.rotation + 2) % 3)
    if any(not len(rows) for rows in (plan_fit_rows, calibration_rows, audit_rows)):
        raise ValueError("stable three-way split produced an empty role")
    plan = compile_physical_curve(
        query_ids,
        full["scores"],
        cheap["scores"],
        full_storage_bytes,
        cheap_storage_bytes,
        budget_fractions=args.budget_fractions,
        eval_fraction=args.eval_fraction,
        retain_cheap_for_upgraded=True,
        fit_indices=plan_fit_rows,
    )
    if "boundary_pareto" not in plan["policies"]:
        raise ValueError("physical compiler does not expose boundary_pareto")
    if max(args.label_budgets) > len(calibration_rows):
        raise ValueError("label budget exceeds the disjoint calibration workload")
    calibration_relevance = tuple(relevance[index] for index in calibration_rows)
    audit_relevance = tuple(relevance[index] for index in audit_rows)
    reference_scores = np.asarray(full["scores"])
    cheap_scores = np.asarray(cheap["scores"])
    calibration_reference_metrics = query_metrics(
        reference_scores[calibration_rows], calibration_relevance, ks=(10, 100)
    )

    losses = {}
    fractions = {}
    truth_safety = {}
    proxy_by_plan = {}
    plan_summary = {}
    for point in plan["policies"]["boundary_pareto"]:
        resident_bytes = plan["cheap_vector_bytes"] + point["incremental_bytes"]
        resident_fraction = resident_bytes / plan["full_vector_bytes"]
        if resident_fraction >= 1.0:
            continue
        upgraded = np.zeros(len(corpus_ids), dtype=bool)
        upgraded[np.asarray(point["upgraded_documents"], dtype=np.int64)] = True
        candidate_scores = calibrated_residual_surface(
            reference_scores, cheap_scores, upgraded
        )
        calibration_candidate_metrics = query_metrics(
            candidate_scores[calibration_rows], calibration_relevance, ks=(10, 100)
        )
        name = f"dual_{float(point['budget_fraction']):.3f}"
        losses[name] = {
            "ndcg_at_10": calibration_reference_metrics["ndcg_at_10"]
            - calibration_candidate_metrics["ndcg_at_10"],
            "recall_at_100": calibration_reference_metrics["recall_at_100"]
            - calibration_candidate_metrics["recall_at_100"],
        }
        fractions[name] = float(resident_fraction)
        fidelity = ranking_fidelity(reference_scores, candidate_scores)
        proxy_by_plan[name] = 1.0 - fidelity["top_10_overlap"][calibration_rows]
        evaluated = evaluate_compression_pair(
            reference_scores[audit_rows],
            candidate_scores[audit_rows],
            audit_relevance,
            bootstrap_seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        truth_safety[name] = bool(evaluated["safety_gate"]["passes"])
        plan_summary[name] = {
            "budget_fraction": float(point["budget_fraction"]),
            "anchor_count": int(len(point["upgraded_documents"])),
            "resident_fraction": float(resident_fraction),
            "qrel_free_certificate_passes": bool(
                evaluated["qrel_free_ranking_certificate"]["passes"]
            ),
            "relevance_safety_passes": truth_safety[name],
            "ndcg_at_10_upper_regret": evaluated["safety_gate"]["ndcg_at_10_upper_regret"],
            "recall_at_100_upper_regret": evaluated["safety_gate"]["recall_at_100_upper_regret"],
            "top10_overlap": evaluated["ranking_fidelity"]["top_10_overlap"]["mean"],
        }
    zeros = np.zeros(len(calibration_rows), dtype=np.float64)
    losses["full"] = {"ndcg_at_10": zeros, "recall_at_100": zeros}
    fractions["full"] = 1.0
    truth_safety["full"] = True
    proxy_risk = np.max(np.stack(list(proxy_by_plan.values())), axis=0)
    assignments = quantile_risk_strata(proxy_risk, strata=args.strata)
    high = assignments == args.strata - 1
    diagnostics = {}
    for name in proxy_by_plan:
        regret = losses[name]["ndcg_at_10"]
        overall_positive = float(np.maximum(regret, 0.0).mean())
        high_positive = float(np.maximum(regret[high], 0.0).mean())
        diagnostics[name] = {
            "proxy_regret_spearman": _rank_correlation(proxy_by_plan[name], regret),
            "high_risk_stratum_positive_regret": high_positive,
            "overall_positive_regret": overall_positive,
            "high_risk_enrichment": high_positive / overall_positive if overall_positive else 0.0,
        }
    simulation = simulate_label_efficiency(
        losses,
        fractions,
        proxy_risk,
        budgets=args.label_budgets,
        trials=args.trials,
        strata=args.strata,
        tolerance=0.01,
        family_alpha=0.05,
        seed=args.seed,
        truth_safety=truth_safety,
    )
    report = {
        "schema_version": 1,
        "stage": "opened-domain-label-efficiency-development",
        "dataset": args.dataset,
        "rotation": args.rotation,
        "information_boundary": {
            "qrels_used_for_physical_plan_generation": False,
            "qrels_used_for_plan_selection_on_disjoint_calibration_queries": True,
            "audit_qrels_used_for_development_truth_only": True,
            "plan_fit_calibration_and_audit_queries_are_disjoint": True,
            "sealed_transfer_claim": False,
        },
        "fit_queries": int(len(plan["fit_query_indices"])),
        "calibration_queries": int(len(calibration_rows)),
        "audit_queries": int(len(audit_rows)),
        "plans": plan_summary,
        "proxy_diagnostics": diagnostics,
        "simulation": simulation,
        "artifacts": {
            "full_runtime_sha256": _sha256(args.full_runtime),
            "cheap_runtime_sha256": _sha256(args.cheap_runtime),
            "labels_sha256": _sha256(args.labels),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "plans": plan_summary,
                "proxy_diagnostics": diagnostics,
                "oracle_plan": simulation["oracle_plan"],
                "label_efficiency": simulation["budgets"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
