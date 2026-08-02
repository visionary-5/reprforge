#!/usr/bin/env python3
"""Physically execute the low-budget IRPAPERS pair-admission point."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.boundary_admission import fit_boundary_statistics
from reprforge.irpapers_benchmark import (
    IRPapersColPaliBackend,
    load_irpapers,
    recall_at_k,
)
from reprforge.pairwise_view_admission import build_boundary_pairs
from reprforge.pairwise_budget import calibrate_pair_budget
from reprforge.vidore_pipeline import ReprForgeViDoRePipeline
from tools.analyze_pairwise_view_admission import (
    _balanced_group_folds,
    _policy_pages,
    _zscore_rows,
)


POLICIES = ("boundary_independent", "pair_conditional")


def _candidate_surface(
    corpus_ids: list[str],
    bm25: np.ndarray,
    visual: np.ndarray,
    *,
    candidate_k: int,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    full_order = np.stack(
        [
            np.asarray(
                sorted(
                    range(len(corpus_ids)),
                    key=lambda page: (-float(scores[page]), corpus_ids[page]),
                ),
                dtype=np.int32,
            )
            for scores in bm25
        ]
    )
    candidates = full_order[:, :candidate_k]
    rows = np.arange(len(candidates))[:, None]
    locator = _zscore_rows(bm25[rows, candidates])
    visual_candidates = visual[rows, candidates]
    visual_zscores = _zscore_rows(visual_candidates)
    fused = locator + visual_zscores
    teacher_offsets = np.argsort(-fused, axis=1, kind="stable")[:, :cutoff]
    teacher = candidates[rows, teacher_offsets]
    return candidates, locator, visual_candidates, visual_zscores, teacher


def _exact_agreement(
    results: dict[str, dict[str, float]],
    query_ids: list[str],
    teacher: np.ndarray,
    corpus_ids: list[str],
) -> float:
    expected = [
        {corpus_ids[int(position)] for position in row}
        for row in teacher
    ]
    actual = [set(results[query_id]) for query_id in query_ids]
    return float(np.mean([left == right for left, right in zip(actual, expected)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument(
        "--pair-budget",
        type=float,
        help="Optional pair-conditional budget for a matched-quality diagnostic.",
    )
    parser.add_argument(
        "--train-match-pair-budget",
        action="store_true",
        help="Select each held-out pair budget using only the other folds.",
    )
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument("--request-batch-size", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--prebuild-admitted-items", action="store_true")
    parser.add_argument(
        "--schedule",
        choices=("policy-major", "fold-interleaved"),
        default="policy-major",
    )
    args = parser.parse_args()
    if not 0.0 <= args.budget <= 1.0:
        raise ValueError("budget must be in [0, 1]")
    if args.pair_budget is not None and not 0.0 <= args.pair_budget <= 1.0:
        raise ValueError("pair budget must be in [0, 1]")
    if args.pair_budget is not None and args.train_match_pair_budget:
        raise ValueError("fixed and train-matched pair budgets are exclusive")
    if args.candidate_k <= args.cutoff:
        raise ValueError("candidate_k must exceed cutoff")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")

    dataset_began = time.perf_counter()
    data = load_irpapers(args.docs, args.queries)
    dataset_load_seconds = time.perf_counter() - dataset_began
    surface = np.load(args.score_surface)
    if [str(value) for value in surface["query_ids"]] != list(data.query_ids):
        raise ValueError("score surface and dataset query order differ")
    if [str(value) for value in surface["corpus_ids"]] != list(data.corpus_ids):
        raise ValueError("score surface and dataset corpus order differ")
    bm25 = np.asarray(surface["bm25_scores"], dtype=np.float64)
    visual = np.asarray(surface["visual_scores"], dtype=np.float64)
    candidates, locator, raw_visual_candidates, visual_zscores, teacher = (
        _candidate_surface(
            list(data.corpus_ids),
            bm25,
            visual,
            candidate_k=args.candidate_k,
            cutoff=args.cutoff,
        )
    )

    with args.queries.open("r", encoding="utf-8", newline="") as handle:
        query_rows = list(csv.DictReader(handle))
    folds = _balanced_group_folds(
        np.asarray([str(row["pdf_id"]) for row in query_rows])
    )

    model_began = time.perf_counter()
    backend = IRPapersColPaliBackend(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
    )
    model_load_seconds = time.perf_counter() - model_began

    fold_values = sorted(set(int(value) for value in folds))
    execution_plan: list[tuple[int, str, int]] = []
    if args.schedule == "policy-major":
        execution_plan = [
            (repetition, policy, fold)
            for repetition in range(args.repetitions)
            for policy in POLICIES
            for fold in fold_values
        ]
    else:
        for repetition in range(args.repetitions):
            for fold in fold_values:
                policy_order = (
                    POLICIES
                    if (repetition + fold) % 2 == 0
                    else tuple(reversed(POLICIES))
                )
                execution_plan.extend(
                    (repetition, policy, fold) for policy in policy_order
                )

    runs: list[dict[str, Any]] = []
    for repetition, policy, fold in execution_plan:
        default_policy_budget = (
            args.pair_budget
            if policy == "pair_conditional" and args.pair_budget is not None
            else args.budget
        )
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        statistics = fit_boundary_statistics(
            locator[train],
            visual_zscores[train],
            cutoff=args.cutoff,
        )
        calibration = None
        policy_budget = default_policy_budget
        if policy == "pair_conditional" and args.train_match_pair_budget:
            calibration = calibrate_pair_budget(
                candidates[train],
                locator[train],
                raw_visual_candidates[train],
                teacher[train],
                rank_risk=statistics.flip_risk_by_rank,
                visual_prior_by_rank=statistics.visual_prior_by_rank,
                cutoff=args.cutoff,
                baseline_fraction=args.budget,
            )
            policy_budget = calibration.selected_fraction
        episode = candidates[test]
        eligible = len(set(int(value) for value in episode.flat))
        page_budget = math.floor(policy_budget * eligible)
        pairs = build_boundary_pairs(
            episode,
            locator[test],
            cutoff=args.cutoff,
            rank_risk=statistics.flip_risk_by_rank,
        )
        selected, admission = _policy_pages(
            policy,
            episode,
            pairs,
            page_budget=page_budget,
            rank_weights=statistics.flip_risk_by_rank,
        )
        admitted_ids = [data.corpus_ids[position] for position in sorted(selected)]
        pipeline = ReprForgeViDoRePipeline(
            base_model=str(args.base_model),
            adapter=str(args.adapter),
            mode="bm25-fusion-batched",
            device=args.device,
            batch_size=args.batch_size,
            scoring_batch_size=args.scoring_batch_size,
            candidate_k=args.candidate_k,
            top_k=args.cutoff,
            request_batch_size=args.request_batch_size,
            cohort_cache_policy="resident",
            admitted_item_ids=admitted_ids,
            visual_prior_by_rank=statistics.visual_prior_by_rank,
            prebuild_admitted_items=args.prebuild_admitted_items,
            backend_factory=lambda: backend,
        )
        pipeline.index(
            list(data.corpus_ids),
            list(data.corpus_images),
            list(data.corpus_texts),
            "IRPAPERS",
        )
        fold_query_ids = [data.query_ids[int(position)] for position in test]
        fold_queries = [data.queries[int(position)] for position in test]
        results, cost = pipeline.search(fold_query_ids, fold_queries)
        fold_qrels = {query_id: data.qrels[query_id] for query_id in fold_query_ids}
        compact_cost = {
            key: value
            for key, value in cost.items()
            if key not in {"batch_trace", "backend"}
        }
        compact_cost["total_visual_pages_encoded"] = int(
            cost["visual_materializations_during_index"]
        ) + int(cost["visual_pages_encoded"])
        compact_cost["construction_plus_execution_ms"] = float(
            cost["measured_index_ms_inside_pipeline"]
        ) + float(cost["total_execution_ms"])
        runs.append(
            {
                "repetition": repetition,
                "policy": policy,
                "budget_fraction": policy_budget,
                "held_out_fold": fold,
                "queries": len(test),
                "eligible_pages": eligible,
                "planned_pages": len(selected),
                "pair_weight_coverage": admission.covered_weight_fraction,
                "train_budget_calibration": (
                    None
                    if calibration is None
                    else {
                        "baseline_fraction": calibration.baseline_fraction,
                        "baseline_agreement": calibration.baseline_agreement,
                        "selected_fraction": calibration.selected_fraction,
                        "selected_agreement": calibration.selected_agreement,
                        "grid_exhausted": calibration.grid_exhausted,
                    }
                ),
                "quality": recall_at_k(results, fold_qrels),
                "exact_top5_set_agreement": _exact_agreement(
                    results,
                    fold_query_ids,
                    teacher[test],
                    list(data.corpus_ids),
                ),
                "cost": compact_cost,
            }
        )

    import torch

    payload = {
        "schema_version": 1,
        "experiment": "physical-pairwise-representation-admission",
        "dataset": dict(data.metadata),
        "configuration": {
            "budget_fraction": args.budget,
            "pair_budget_fraction": args.pair_budget,
            "train_match_pair_budget": args.train_match_pair_budget,
            "candidate_k": args.candidate_k,
            "cutoff": args.cutoff,
            "policies": list(POLICIES),
            "folds": int(len(set(folds))),
            "repetitions": args.repetitions,
            "schedule": args.schedule,
            "prebuild_admitted_items": args.prebuild_admitted_items,
            "selection_uses_qrels": False,
        },
        "resource_contract": {
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda_visible_device_count": torch.cuda.device_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "dataset_load_seconds": dataset_load_seconds,
            "model_load_seconds": model_load_seconds,
            "batch_size": args.batch_size,
            "scoring_batch_size": args.scoring_batch_size,
            "request_batch_size": args.request_batch_size,
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
