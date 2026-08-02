#!/usr/bin/env python3
"""Evaluate sparse, cost/risk-aware admission on frozen IRPAPERS scores.

The score surface is an evaluator and historical probe source.  For each
outer source-paper fold, the planner sees visual scores only for a
deterministic subset of the remaining papers.  Those probes are split by
paper into risk-model and budget-calibration subsets.  The held-out plan is
then frozen from BM25-visible state before its visual scores or qrels are
opened by the evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.boundary_admission import (
    BoundaryStatistics,
    execute_boundary_plan,
    fit_boundary_statistics,
)
from reprforge.pairwise_view_admission import select_independent_pages
from reprforge.physical_cost import (
    AtomicCostObservation,
    AtomicMaterializationCostModel,
    fit_atomic_cost_model,
)
from reprforge.risk_constrained_admission import (
    CostFractionDecision,
    estimate_plan_cost,
    select_cost_aware_pairs,
    select_risk_bounded_cost_fraction,
)
from reprforge.sparse_what_if import (
    SparseBoundaryRiskModel,
    build_estimated_boundary_pairs,
    fit_sparse_boundary_risk,
    select_sparse_probe_queries,
)
from tools.analyze_pairwise_view_admission import (
    _balanced_group_folds,
    _zscore_rows,
)
from tools.run_pairwise_admission_physical import _candidate_surface


def _load_cost_model(path: Path, *, batch_size: int) -> tuple[
    AtomicMaterializationCostModel,
    dict[str, float],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observations: list[AtomicCostObservation] = []
    for row in payload["runs"]:
        cost = row["cost"]
        observations.append(
            AtomicCostObservation(
                pages=int(row["planned_pages"]),
                score_events=int(cost["admitted_candidate_events"]),
                total_ms=float(cost["admitted_prebuild"]["visual_encode_ms"])
                + float(cost["visual_score_ms"]),
            )
        )
    model = fit_atomic_cost_model(observations, batch_size=batch_size)
    actual = np.asarray([row.total_ms for row in observations])
    predicted = np.asarray(
        [
            model.estimate(pages=row.pages, score_events=row.score_events).total_ms
            for row in observations
        ]
    )
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mean = float(actual.mean())
    return model, {
        "observations": len(observations),
        "rmse_ms": rmse,
        "relative_rmse": rmse / mean if mean else 0.0,
    }


def _teacher_correct(rankings: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            set(row) == set(reference)
            for row, reference in zip(rankings, teacher, strict=True)
        ],
        dtype=bool,
    )


def _recall(
    rankings: np.ndarray,
    corpus_ids: Sequence[str],
    gold_ids: Sequence[str],
) -> float:
    return float(
        np.mean(
            [
                gold in {corpus_ids[int(page)] for page in row}
                for row, gold in zip(rankings, gold_ids, strict=True)
            ]
        )
    )


def _cross_fit_risk_controller(
    *,
    probes: np.ndarray,
    groups: np.ndarray,
    candidates: np.ndarray,
    locator: np.ndarray,
    raw_visual: np.ndarray,
    visual_zscores: np.ndarray,
    teacher: np.ndarray,
    cost_model: AtomicMaterializationCostModel,
    cutoff: int,
    baseline_fraction: float,
    risk_tolerance: float,
    confidence: float,
    uncertainty_weight: float,
    seed: int,
) -> tuple[CostFractionDecision, SparseBoundaryRiskModel, BoundaryStatistics]:
    """Cross-fit every probed source paper before selecting a cost fraction."""

    fractions = (
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    )
    baseline_correct = np.empty(len(probes), dtype=bool)
    candidate_correct = {
        fraction: np.empty(len(probes), dtype=bool) for fraction in fractions
    }
    position = {int(query): offset for offset, query in enumerate(probes)}
    probe_groups = groups[probes]
    for held_group in sorted(set(probe_groups)):
        fit = probes[probe_groups != held_group]
        validation = probes[probe_groups == held_group]
        if not len(fit):
            raise ValueError("cross-fitting requires more than one source paper")
        model = fit_sparse_boundary_risk(
            locator[fit], raw_visual[fit], cutoff=cutoff
        )
        statistics = fit_boundary_statistics(
            locator[fit], visual_zscores[fit], cutoff=cutoff
        )
        pairs = build_estimated_boundary_pairs(
            candidates[validation],
            locator[validation],
            model,
            uncertainty_weight=uncertainty_weight,
        )
        eligible = len(set(int(page) for page in candidates[validation].flat))
        baseline = select_independent_pages(
            pairs,
            page_budget=math.floor(baseline_fraction * eligible),
        )
        baseline_cost = estimate_plan_cost(
            candidates[validation], baseline.selected_pages, cost_model
        )
        rankings, _ = execute_boundary_plan(
            candidates[validation],
            locator[validation],
            raw_visual[validation],
            selected_pages=set(baseline.selected_pages),
            visual_prior_by_rank=statistics.visual_prior_by_rank,
            cutoff=cutoff,
        )
        outcomes = _teacher_correct(rankings, teacher[validation])
        for query, outcome in zip(validation, outcomes, strict=True):
            baseline_correct[position[int(query)]] = outcome
        for fraction in fractions:
            plan = select_cost_aware_pairs(
                pairs,
                candidates[validation],
                cost_model,
                time_budget_ms=fraction * baseline_cost.total_ms,
            )
            rankings, _ = execute_boundary_plan(
                candidates[validation],
                locator[validation],
                raw_visual[validation],
                selected_pages=set(plan.admission.selected_pages),
                visual_prior_by_rank=statistics.visual_prior_by_rank,
                cutoff=cutoff,
            )
            outcomes = _teacher_correct(rankings, teacher[validation])
            for query, outcome in zip(validation, outcomes, strict=True):
                candidate_correct[fraction][position[int(query)]] = outcome

    decision = select_risk_bounded_cost_fraction(
        candidate_correct,
        baseline_correct,
        probe_groups,
        risk_tolerance=risk_tolerance,
        confidence=confidence,
        seed=seed,
    )
    final_model = fit_sparse_boundary_risk(
        locator[probes], raw_visual[probes], cutoff=cutoff
    )
    final_statistics = fit_boundary_statistics(
        locator[probes], visual_zscores[probes], cutoff=cutoff
    )
    return decision, final_model, final_statistics


def _evaluate_fraction(
    *,
    fraction: float,
    folds: np.ndarray,
    groups: np.ndarray,
    candidates: np.ndarray,
    locator: np.ndarray,
    raw_visual: np.ndarray,
    visual_zscores: np.ndarray,
    teacher: np.ndarray,
    corpus_ids: list[str],
    gold_ids: list[str],
    cost_model: AtomicMaterializationCostModel,
    cutoff: int,
    baseline_fraction: float,
    risk_tolerance: float,
    confidence: float,
    uncertainty_weight: float,
) -> dict[str, Any]:
    baseline_rankings = np.empty_like(teacher)
    selected_rankings = np.empty_like(teacher)
    fold_rows: list[dict[str, Any]] = []
    totals = {
        "probe_query_events": 0,
        "probe_unique_pages": 0,
        "baseline_pages": 0,
        "baseline_score_events": 0,
        "baseline_estimated_ms": 0.0,
        "selected_pages": 0,
        "selected_score_events": 0,
        "selected_estimated_ms": 0.0,
    }
    for fold in sorted(set(int(value) for value in folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        relative_probe = select_sparse_probe_queries(
            locator[train],
            cutoff=cutoff,
            fraction=fraction,
            groups=groups[train],
        )
        probes = train[relative_probe.query_indices]
        calibration_result, risk_model, probe_statistics = (
            _cross_fit_risk_controller(
                probes=probes,
                groups=groups,
                candidates=candidates,
                locator=locator,
                raw_visual=raw_visual,
                visual_zscores=visual_zscores,
                teacher=teacher,
                cost_model=cost_model,
                cutoff=cutoff,
                baseline_fraction=baseline_fraction,
                risk_tolerance=risk_tolerance,
                confidence=confidence,
                uncertainty_weight=uncertainty_weight,
                seed=fold,
            )
        )

        test_pairs = build_estimated_boundary_pairs(
            candidates[test],
            locator[test],
            risk_model,
            uncertainty_weight=uncertainty_weight,
        )
        eligible = len(set(int(page) for page in candidates[test].flat))
        baseline_page_budget = math.floor(baseline_fraction * eligible)
        baseline = select_independent_pages(
            test_pairs,
            page_budget=baseline_page_budget,
        )
        baseline_cost = estimate_plan_cost(
            candidates[test], baseline.selected_pages, cost_model
        )
        selected = (
            None
            if calibration_result.fallback_to_baseline
            else select_cost_aware_pairs(
                test_pairs,
                candidates[test],
                cost_model,
                time_budget_ms=(
                    calibration_result.selected_fraction
                    * baseline_cost.total_ms
                ),
            )
        )
        selected_pages = (
            set(baseline.selected_pages)
            if selected is None
            else set(selected.admission.selected_pages)
        )
        selected_cost = (
            baseline_cost if selected is None else selected.estimated_cost
        )
        fold_baseline, baseline_events = execute_boundary_plan(
            candidates[test],
            locator[test],
            raw_visual[test],
            selected_pages=set(baseline.selected_pages),
            visual_prior_by_rank=probe_statistics.visual_prior_by_rank,
            cutoff=cutoff,
        )
        fold_selected, selected_events = execute_boundary_plan(
            candidates[test],
            locator[test],
            raw_visual[test],
            selected_pages=selected_pages,
            visual_prior_by_rank=probe_statistics.visual_prior_by_rank,
            cutoff=cutoff,
        )
        baseline_rankings[test] = fold_baseline
        selected_rankings[test] = fold_selected

        probe_pages = set(int(page) for page in candidates[probes].flat)
        totals["probe_query_events"] += len(probes) * candidates.shape[1]
        totals["probe_unique_pages"] += len(probe_pages)
        totals["baseline_pages"] += len(baseline.selected_pages)
        totals["baseline_score_events"] += int(
            baseline_events["visual_candidate_events"]
        )
        totals["baseline_estimated_ms"] += baseline_cost.total_ms
        totals["selected_pages"] += len(selected_pages)
        totals["selected_score_events"] += int(
            selected_events["visual_candidate_events"]
        )
        totals["selected_estimated_ms"] += selected_cost.total_ms
        fold_rows.append(
            {
                "held_out_fold": fold,
                "queries": len(test),
                "probe_queries": len(probes),
                "model_fit_queries": len(probes),
                "calibration_queries": calibration_result.calibration_queries,
                "calibration_source_groups": calibration_result.calibration_groups,
                "probe_unique_pages": len(probe_pages),
                "selected_cost_fraction": calibration_result.selected_fraction,
                "calibration_upper_extra_disagreement": (
                    calibration_result.upper_extra_disagreement
                ),
                "calibration_grid_exhausted": (
                    calibration_result.fallback_to_baseline
                ),
                "fallback_to_baseline": calibration_result.fallback_to_baseline,
                "best_attempt_upper_extra_disagreement": (
                    calibration_result.best_attempt_upper_extra_disagreement
                ),
                "baseline_pages": len(baseline.selected_pages),
                "selected_pages": len(selected_pages),
                "baseline_estimated_ms": baseline_cost.total_ms,
                "selected_estimated_ms": selected_cost.total_ms,
                "baseline_agreement": float(
                    _teacher_correct(fold_baseline, teacher[test]).mean()
                ),
                "selected_agreement": float(
                    _teacher_correct(fold_selected, teacher[test]).mean()
                ),
                "baseline_recall_5": _recall(
                    fold_baseline,
                    corpus_ids,
                    [gold_ids[int(query)] for query in test],
                ),
                "selected_recall_5": _recall(
                    fold_selected,
                    corpus_ids,
                    [gold_ids[int(query)] for query in test],
                ),
            }
        )

    baseline_correct = _teacher_correct(baseline_rankings, teacher)
    selected_correct = _teacher_correct(selected_rankings, teacher)
    baseline_recall = _recall(baseline_rankings, corpus_ids, gold_ids)
    selected_recall = _recall(selected_rankings, corpus_ids, gold_ids)
    # Probes are charged conservatively as fresh physical pages in every outer
    # fold. Their score events are not charged a second time because candidate
    # cohort scoring is included in the probe materialization observation.
    probe_ms = sum(
        cost_model.estimate(pages=row["probe_unique_pages"], score_events=0).total_ms
        for row in fold_rows
    )
    online_saved_ms = totals["baseline_estimated_ms"] - totals["selected_estimated_ms"]
    return {
        "probe_fraction": fraction,
        "quality": {
            "baseline_exact_teacher_agreement": float(baseline_correct.mean()),
            "selected_exact_teacher_agreement": float(selected_correct.mean()),
            "agreement_delta": float(selected_correct.mean() - baseline_correct.mean()),
            "baseline_recall_5": baseline_recall,
            "selected_recall_5": selected_recall,
            "recall_5_delta": selected_recall - baseline_recall,
        },
        "physical_estimate": {
            **totals,
            "selected_page_reduction_fraction": (
                1.0 - totals["selected_pages"] / totals["baseline_pages"]
            ),
            "online_speedup": (
                totals["baseline_estimated_ms"] / totals["selected_estimated_ms"]
                if totals["selected_estimated_ms"]
                else None
            ),
            "probe_estimated_ms": probe_ms,
            "online_saved_ms": online_saved_ms,
            "probe_amortization_episodes": (
                probe_ms / online_saved_ms if online_saved_ms > 0 else None
            ),
        },
        "folds": fold_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--physical-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-fractions", type=float, nargs="+", default=(0.1, 0.2, 0.4))
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--baseline-fraction", type=float, default=0.2)
    parser.add_argument("--risk-tolerance", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    surface = np.load(args.score_surface)
    corpus_ids = [str(value) for value in surface["corpus_ids"]]
    bm25 = np.asarray(surface["bm25_scores"], dtype=np.float64)
    visual = np.asarray(surface["visual_scores"], dtype=np.float64)
    candidates, locator, raw_visual, visual_zscores, teacher = _candidate_surface(
        corpus_ids,
        bm25,
        visual,
        candidate_k=args.candidate_k,
        cutoff=args.cutoff,
    )
    with args.queries.open("r", encoding="utf-8", newline="") as handle:
        query_rows = list(csv.DictReader(handle))
    query_ids = [str(value) for value in surface["query_ids"]]
    if len(query_rows) != len(query_ids):
        raise ValueError("query metadata and score surface differ in length")
    groups = np.asarray([str(row["pdf_id"]) for row in query_rows])
    gold_ids = [str(row["dataset_id"]) for row in query_rows]
    folds = _balanced_group_folds(groups)
    cost_model, fit_diagnostics = _load_cost_model(
        args.physical_runs,
        batch_size=args.batch_size,
    )
    results = [
        _evaluate_fraction(
            fraction=float(fraction),
            folds=folds,
            groups=groups,
            candidates=candidates,
            locator=locator,
            raw_visual=raw_visual,
            visual_zscores=visual_zscores,
            teacher=teacher,
            corpus_ids=corpus_ids,
            gold_ids=gold_ids,
            cost_model=cost_model,
            cutoff=args.cutoff,
            baseline_fraction=args.baseline_fraction,
            risk_tolerance=args.risk_tolerance,
            confidence=args.confidence,
            uncertainty_weight=args.uncertainty_weight,
        )
        for fraction in args.probe_fractions
    ]
    payload = {
        "schema_version": 1,
        "experiment": "sparse-risk-constrained-representation-admission",
        "contract": {
            "outer_split": "source-paper-disjoint-five-fold",
            "probe_split": "source-paper-leave-one-group-out-cross-fit",
            "heldout_visual_visible_to_planner": False,
            "qrels_visible_to_planner": False,
            "probe_artifacts_charged": True,
            "candidate_k": args.candidate_k,
            "cutoff": args.cutoff,
            "baseline_page_fraction": args.baseline_fraction,
            "risk_tolerance": args.risk_tolerance,
            "confidence": args.confidence,
            "uncertainty_weight": args.uncertainty_weight,
        },
        "cost_model": {
            "batch_size": cost_model.batch_size,
            "setup_ms": cost_model.setup_ms,
            "page_ms": cost_model.page_ms,
            "batch_ms": cost_model.batch_ms,
            "score_event_ms": cost_model.score_event_ms,
            **fit_diagnostics,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
