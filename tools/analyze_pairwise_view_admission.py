#!/usr/bin/env python3
"""Evaluate pair-aware candidate representation admission on a score surface."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.boundary_admission import (
    execute_boundary_plan,
    fit_boundary_statistics,
    select_episode_pages,
)
from reprforge.pairwise_view_admission import (
    PairAdmission,
    build_boundary_pairs,
    evaluate_pair_coverage,
    select_frequency_pages,
    select_independent_pages,
    select_pairwise_pages,
)


DEFAULT_BUDGETS = (0.2, 0.4, 0.6)


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    means = values.mean(axis=1, keepdims=True)
    stds = np.maximum(values.std(axis=1, keepdims=True), 1e-12)
    return (values - means) / stds


def _balanced_group_folds(groups: np.ndarray, count: int = 5) -> np.ndarray:
    group_counts = {
        group: int(np.sum(groups == group)) for group in sorted(set(groups))
    }
    loads = [0] * count
    assignments: dict[str, int] = {}
    for group, size in sorted(group_counts.items(), key=lambda row: (-row[1], row[0])):
        fold = min(range(count), key=lambda value: (loads[value], value))
        assignments[group] = fold
        loads[fold] += size
    return np.asarray([assignments[str(group)] for group in groups], dtype=np.int16)


def _recall(
    rankings: np.ndarray,
    corpus_ids: list[str],
    gold_ids: list[str],
) -> float:
    return float(
        np.mean(
            [
                gold in {corpus_ids[int(position)] for position in row}
                for row, gold in zip(rankings, gold_ids, strict=True)
            ]
        )
    )


def _selected_by_boundary_weight(
    candidates: np.ndarray,
    rank_weights: np.ndarray,
    page_budget: int,
) -> set[int]:
    eligible = len(set(int(value) for value in candidates.flat))
    if eligible == 0 or page_budget == 0:
        return set()
    # select_episode_pages takes a fraction and floors the resulting budget.
    # Move one ULP upward so an exact integer budget survives floating-point
    # multiplication and clamp the result defensively.
    fraction = min(1.0, math.nextafter(page_budget / eligible, math.inf))
    selected = select_episode_pages(
        candidates,
        budget_fraction=fraction,
        rank_weights=rank_weights,
    )
    if len(selected) > page_budget:
        raise AssertionError("boundary selector exceeded its exact page budget")
    return selected


def _policy_pages(
    name: str,
    candidates: np.ndarray,
    pairs,
    *,
    page_budget: int,
    rank_weights: np.ndarray,
) -> tuple[set[int], PairAdmission | None]:
    if name == "frequency":
        admission = select_frequency_pages(
            candidates,
            pairs,
            page_budget=page_budget,
        )
        return set(admission.selected_pages), admission
    if name == "boundary_independent":
        selected = _selected_by_boundary_weight(
            candidates,
            rank_weights,
            page_budget,
        )
        return selected, evaluate_pair_coverage(pairs, selected)
    if name == "pair_incident":
        admission = select_independent_pages(pairs, page_budget=page_budget)
        return set(admission.selected_pages), admission
    if name == "pair_conditional":
        admission = select_pairwise_pages(pairs, page_budget=page_budget)
        return set(admission.selected_pages), admission
    raise ValueError(f"unknown policy {name}")


def _evaluate_policy(
    *,
    name: str,
    budget_fraction: float,
    fold_ids: np.ndarray,
    candidates: np.ndarray,
    locator_zscores: np.ndarray,
    raw_visual_scores: np.ndarray,
    visual_zscores: np.ndarray,
    teacher: np.ndarray,
    corpus_ids: list[str],
    gold_ids: list[str],
) -> dict[str, Any]:
    rankings = np.empty_like(teacher)
    eligible_sum = 0
    selected_sum = 0
    event_sum = 0
    covered_weight = 0.0
    total_pair_weight = 0.0
    folds: list[dict[str, Any]] = []
    for fold in sorted(set(int(value) for value in fold_ids)):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        statistics = fit_boundary_statistics(
            locator_zscores[train],
            visual_zscores[train],
            cutoff=teacher.shape[1],
        )
        episode = candidates[test]
        eligible = len(set(int(value) for value in episode.flat))
        page_budget = math.floor(budget_fraction * eligible)
        pairs = build_boundary_pairs(
            episode,
            locator_zscores[test],
            cutoff=teacher.shape[1],
            rank_risk=statistics.flip_risk_by_rank,
        )
        selected, coverage = _policy_pages(
            name,
            episode,
            pairs,
            page_budget=page_budget,
            rank_weights=statistics.flip_risk_by_rank,
        )
        fold_rankings, cost = execute_boundary_plan(
            episode,
            locator_zscores[test],
            raw_visual_scores[test],
            selected_pages=selected,
            visual_prior_by_rank=statistics.visual_prior_by_rank,
            cutoff=teacher.shape[1],
        )
        rankings[test] = fold_rankings
        fold_exact = np.asarray(
            [
                set(row) == set(reference)
                for row, reference in zip(
                    fold_rankings,
                    teacher[test],
                    strict=True,
                )
            ]
        )
        fold_gold_ids = [gold_ids[int(position)] for position in test]
        eligible_sum += eligible
        selected_sum += len(selected)
        event_sum += int(cost["visual_candidate_events"])
        if coverage is not None:
            covered_weight += coverage.covered_weight
            total_pair_weight += coverage.total_weight
        folds.append(
            {
                "held_out_fold": fold,
                "queries": len(test),
                "eligible_pages": eligible,
                "page_budget": page_budget,
                "selected_pages": len(selected),
                "boundary_pairs": len(pairs),
                "recall_5": _recall(
                    fold_rankings,
                    corpus_ids,
                    fold_gold_ids,
                ),
                "exact_top5_set_agreement": float(fold_exact.mean()),
                "pair_weight_coverage": (
                    coverage.covered_weight_fraction
                    if coverage is not None
                    else None
                ),
            }
        )
    exact = np.asarray(
        [
            set(row) == set(reference)
            for row, reference in zip(rankings, teacher, strict=True)
        ]
    )
    return {
        "policy": name,
        "budget_fraction": budget_fraction,
        "quality": {"recall_5": _recall(rankings, corpus_ids, gold_ids)},
        "teacher_agreement": {
            "exact_top5_set_agreement": float(exact.mean()),
            "top5_set_disagreements": int((~exact).sum()),
        },
        "cost": {
            "independent_episode_eligible_page_sum": eligible_sum,
            "independent_episode_selected_page_sum": selected_sum,
            "selected_page_fraction": selected_sum / eligible_sum,
            "visual_candidate_events": event_sum,
            "mean_visual_pages_per_query": event_sum / len(fold_ids),
        },
        "pair_objective": {
            "covered_weight_fraction": (
                covered_weight / total_pair_weight if total_pair_weight else None
            )
        },
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_k <= args.cutoff:
        raise ValueError("candidate_k must exceed cutoff")
    if any(not 0.0 <= value <= 1.0 for value in args.budgets):
        raise ValueError("budgets must be in [0, 1]")

    surface = np.load(args.score_surface)
    surface_query_ids = [str(value) for value in surface["query_ids"]]
    corpus_ids = [str(value) for value in surface["corpus_ids"]]
    bm25 = np.asarray(surface["bm25_scores"], dtype=np.float64)
    visual = np.asarray(surface["visual_scores"], dtype=np.float64)
    with args.queries.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(bm25):
        raise ValueError("query metadata and score surface differ in length")
    expected_query_ids = [f"q-{offset:04d}" for offset in range(len(rows))]
    if surface_query_ids != expected_query_ids:
        raise ValueError("score-surface query order does not match IRPAPERS")
    groups = np.asarray([str(row["pdf_id"]) for row in rows])
    fold_ids = _balanced_group_folds(groups)
    gold_ids = [str(row["dataset_id"]) for row in rows]

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
    candidates = full_order[:, : args.candidate_k]
    row_index = np.arange(len(rows))[:, None]
    locator_candidate_scores = bm25[row_index, candidates]
    raw_visual_candidate_scores = visual[row_index, candidates]
    locator_zscores = _zscore_rows(locator_candidate_scores)
    visual_zscores = _zscore_rows(raw_visual_candidate_scores)
    fused = locator_zscores + visual_zscores
    teacher_offsets = np.argsort(-fused, axis=1, kind="stable")[:, : args.cutoff]
    teacher = candidates[row_index, teacher_offsets]

    policies = (
        "frequency",
        "boundary_independent",
        "pair_incident",
        "pair_conditional",
    )
    runs = [
        _evaluate_policy(
            name=policy,
            budget_fraction=budget,
            fold_ids=fold_ids,
            candidates=candidates,
            locator_zscores=locator_zscores,
            raw_visual_scores=raw_visual_candidate_scores,
            visual_zscores=visual_zscores,
            teacher=teacher,
            corpus_ids=corpus_ids,
            gold_ids=gold_ids,
        )
        for budget in args.budgets
        for policy in policies
    ]
    payload = {
        "schema_version": 1,
        "experiment": "pair-aware-candidate-representation-admission",
        "candidate_k": args.candidate_k,
        "cutoff": args.cutoff,
        "queries": len(rows),
        "corpus_pages": len(corpus_ids),
        "folds": int(len(set(fold_ids))),
        "teacher_recall_5": _recall(teacher, corpus_ids, gold_ids),
        "selection_uses_qrels": False,
        "full_visual_test_scores_used_only_for_replay_execution": True,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
