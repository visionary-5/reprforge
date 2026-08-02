#!/usr/bin/env python3
"""Run the paper-disjoint IRPAPERS boundary-admission gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.boundary_admission import (
    execute_boundary_plan,
    fit_boundary_statistics,
    select_episode_pages,
)


BUDGETS = (0.5, 0.6, 0.7, 0.8)
ADAPTIVE_BUDGET_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for row, scores in enumerate(values):
        output[row] = (scores - scores.mean()) / max(float(scores.std()), 1e-12)
    return output


def _balanced_group_folds(
    groups: np.ndarray,
    *,
    fold_count: int = 5,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign complete source groups to deterministically balanced folds."""

    counts = Counter(str(value) for value in groups)
    loads = [0] * fold_count
    assignment: dict[str, int] = {}
    for group, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(fold_count), key=lambda value: (loads[value], value))
        assignment[group] = fold
        loads[fold] += count
    return (
        np.asarray([assignment[str(value)] for value in groups], dtype=np.int16),
        assignment,
    )


def _recall(
    rankings: np.ndarray,
    corpus_ids: list[str],
    gold_ids: list[str],
) -> float:
    return float(
        np.mean(
            [
                gold in {corpus_ids[int(page)] for page in ranking}
                for ranking, gold in zip(rankings, gold_ids, strict=True)
            ]
        )
    )


def _evaluate_method(
    *,
    groups: np.ndarray,
    candidates: np.ndarray,
    locator_zscores: np.ndarray,
    raw_visual_scores: np.ndarray,
    visual_zscores: np.ndarray,
    teacher: np.ndarray,
    corpus_ids: list[str],
    gold_ids: list[str],
    budget_fraction: float,
    boundary_weighted: bool,
) -> dict[str, Any]:
    rankings = np.empty_like(teacher)
    selected_page_sum = 0
    eligible_page_sum = 0
    event_sum = 0
    folds: list[dict[str, Any]] = []
    for group in sorted(set(groups)):
        train = np.flatnonzero(groups != group)
        test = np.flatnonzero(groups == group)
        statistics = fit_boundary_statistics(
            locator_zscores[train],
            visual_zscores[train],
            cutoff=teacher.shape[1],
        )
        episode_candidates = candidates[test]
        selected = select_episode_pages(
            episode_candidates,
            budget_fraction=budget_fraction,
            rank_weights=(
                statistics.flip_risk_by_rank if boundary_weighted else None
            ),
        )
        fold_rankings, cost = execute_boundary_plan(
            episode_candidates,
            locator_zscores[test],
            raw_visual_scores[test],
            selected_pages=selected,
            visual_prior_by_rank=statistics.visual_prior_by_rank,
            cutoff=teacher.shape[1],
        )
        rankings[test] = fold_rankings
        eligible = len(set(int(value) for value in episode_candidates.flat))
        selected_page_sum += int(cost["selected_unique_pages"])
        eligible_page_sum += eligible
        event_sum += int(cost["visual_candidate_events"])
        folds.append(
            {
                "held_out_fold": int(group),
                "queries": int(len(test)),
                "eligible_unique_pages": eligible,
                "selected_unique_pages": int(cost["selected_unique_pages"]),
                "visual_candidate_events": int(cost["visual_candidate_events"]),
            }
        )
    exact = np.asarray(
        [
            set(row) == set(reference)
            for row, reference in zip(rankings, teacher, strict=True)
        ]
    )
    return {
        "budget_fraction": budget_fraction,
        "admission": "boundary_weighted" if boundary_weighted else "frequency",
        "quality": {"recall_5": _recall(rankings, corpus_ids, gold_ids)},
        "teacher_agreement": {
            "exact_top5_set_agreement": float(exact.mean()),
            "top5_set_disagreements": int((~exact).sum()),
        },
        "cost": {
            "independent_episode_eligible_unique_page_sum": eligible_page_sum,
            "independent_episode_selected_unique_page_sum": selected_page_sum,
            "selected_unique_page_fraction": selected_page_sum / eligible_page_sum,
            "visual_candidate_events": event_sum,
            "mean_visual_pages_per_query": event_sum / len(groups),
        },
        "folds": folds,
    }


def _evaluate_risk_target(
    *,
    groups: np.ndarray,
    candidates: np.ndarray,
    locator_zscores: np.ndarray,
    raw_visual_scores: np.ndarray,
    visual_zscores: np.ndarray,
    teacher: np.ndarray,
    corpus_ids: list[str],
    gold_ids: list[str],
    agreement_target: float,
    boundary_weighted: bool,
) -> dict[str, Any]:
    """Choose the smallest train-only budget meeting a teacher-risk target."""

    rankings = np.empty_like(teacher)
    selected_page_sum = 0
    eligible_page_sum = 0
    event_sum = 0
    folds: list[dict[str, Any]] = []
    for group in sorted(set(groups)):
        train = np.flatnonzero(groups != group)
        test = np.flatnonzero(groups == group)
        statistics = fit_boundary_statistics(
            locator_zscores[train],
            visual_zscores[train],
            cutoff=teacher.shape[1],
        )
        rank_weights = (
            statistics.flip_risk_by_rank if boundary_weighted else None
        )
        selected_fraction = ADAPTIVE_BUDGET_GRID[-1]
        train_agreement = 1.0
        for fraction in ADAPTIVE_BUDGET_GRID:
            train_pages = select_episode_pages(
                candidates[train],
                budget_fraction=fraction,
                rank_weights=rank_weights,
            )
            train_rankings, _ = execute_boundary_plan(
                candidates[train],
                locator_zscores[train],
                raw_visual_scores[train],
                selected_pages=train_pages,
                visual_prior_by_rank=statistics.visual_prior_by_rank,
                cutoff=teacher.shape[1],
            )
            train_agreement = float(
                np.mean(
                    [
                        set(row) == set(reference)
                        for row, reference in zip(
                            train_rankings,
                            teacher[train],
                            strict=True,
                        )
                    ]
                )
            )
            if train_agreement >= agreement_target:
                selected_fraction = fraction
                break

        episode_candidates = candidates[test]
        selected = select_episode_pages(
            episode_candidates,
            budget_fraction=selected_fraction,
            rank_weights=rank_weights,
        )
        fold_rankings, cost = execute_boundary_plan(
            episode_candidates,
            locator_zscores[test],
            raw_visual_scores[test],
            selected_pages=selected,
            visual_prior_by_rank=statistics.visual_prior_by_rank,
            cutoff=teacher.shape[1],
        )
        rankings[test] = fold_rankings
        eligible = len(set(int(value) for value in episode_candidates.flat))
        selected_page_sum += int(cost["selected_unique_pages"])
        eligible_page_sum += eligible
        event_sum += int(cost["visual_candidate_events"])
        folds.append(
            {
                "held_out_fold": int(group),
                "queries": int(len(test)),
                "selected_budget_fraction": selected_fraction,
                "training_exact_top5_agreement": train_agreement,
                "eligible_unique_pages": eligible,
                "selected_unique_pages": int(cost["selected_unique_pages"]),
                "visual_candidate_events": int(cost["visual_candidate_events"]),
            }
        )
    exact = np.asarray(
        [
            set(row) == set(reference)
            for row, reference in zip(rankings, teacher, strict=True)
        ]
    )
    return {
        "agreement_target": agreement_target,
        "budget_grid": list(ADAPTIVE_BUDGET_GRID),
        "admission": "boundary_weighted" if boundary_weighted else "frequency",
        "quality": {"recall_5": _recall(rankings, corpus_ids, gold_ids)},
        "teacher_agreement": {
            "exact_top5_set_agreement": float(exact.mean()),
            "top5_set_disagreements": int((~exact).sum()),
        },
        "cost": {
            "independent_episode_eligible_unique_page_sum": eligible_page_sum,
            "independent_episode_selected_unique_page_sum": selected_page_sum,
            "selected_unique_page_fraction": selected_page_sum / eligible_page_sum,
            "visual_candidate_events": event_sum,
            "mean_visual_pages_per_query": event_sum / len(groups),
        },
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    surface = np.load(args.score_surface)
    corpus_ids = [str(value) for value in surface["corpus_ids"]]
    bm25 = np.asarray(surface["bm25_scores"], dtype=np.float64)
    visual = np.asarray(surface["visual_scores"], dtype=np.float64)
    with args.queries.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(bm25):
        raise ValueError("query metadata and score surface differ in length")
    groups = np.asarray([str(row["pdf_id"]) for row in rows])
    fold_ids, fold_assignment = _balanced_group_folds(groups)
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
    candidates = full_order[:, :10]
    rows_index = np.arange(len(rows))[:, None]
    locator_candidate_scores = bm25[rows_index, candidates]
    raw_visual_candidate_scores = visual[rows_index, candidates]
    locator_zscores = _zscore_rows(locator_candidate_scores)
    visual_zscores = _zscore_rows(raw_visual_candidate_scores)
    fused = locator_zscores + visual_zscores
    teacher_offsets = np.argsort(-fused, axis=1, kind="stable")[:, :5]
    teacher = candidates[rows_index, teacher_offsets]
    teacher_recall = _recall(teacher, corpus_ids, gold_ids)

    runs: dict[str, Any] = {}
    for budget in BUDGETS:
        for weighted in (False, True):
            name = f"{'boundary' if weighted else 'frequency'}_{int(100 * budget)}pct"
            runs[name] = _evaluate_method(
                groups=fold_ids,
                candidates=candidates,
                locator_zscores=locator_zscores,
                raw_visual_scores=raw_visual_candidate_scores,
                visual_zscores=visual_zscores,
                teacher=teacher,
                corpus_ids=corpus_ids,
                gold_ids=gold_ids,
                budget_fraction=budget,
                boundary_weighted=weighted,
            )

    for weighted in (False, True):
        name = f"adaptive_{'boundary' if weighted else 'frequency'}_risk80"
        runs[name] = _evaluate_risk_target(
            groups=fold_ids,
            candidates=candidates,
            locator_zscores=locator_zscores,
            raw_visual_scores=raw_visual_candidate_scores,
            visual_zscores=visual_zscores,
            teacher=teacher,
            corpus_ids=corpus_ids,
            gold_ids=gold_ids,
            agreement_target=0.80,
            boundary_weighted=weighted,
        )

    gate = runs["adaptive_boundary_risk80"]
    cheap_baseline = runs["adaptive_frequency_risk80"]
    query_tolerance = 1 / len(rows)
    page_reduction = 1.0 - gate["cost"]["selected_unique_page_fraction"]
    quality_delta = gate["quality"]["recall_5"] - teacher_recall
    output = {
        "schema_version": 1,
        "dataset": {
            "name": "IRPAPERS",
            "queries": len(rows),
            "pages": len(corpus_ids),
            "source_papers": len(set(groups)),
            "evaluation_folds": 5,
            "source_paper_fold_assignment": fold_assignment,
        },
        "method": {
            "name": "boundary-weighted representation admission",
            "training_signal": (
                "rank-conditioned Top-5 membership flips between BM25 and the "
                "fully observed BM25+visual teacher; no qrels"
            ),
            "workload_action": (
                "aggregate boundary risk over candidate occurrences and admit "
                "the highest-value pages under a representation budget"
            ),
            "missing_visual_score": (
                "train-only rank prior; observed scores normalized within the "
                "selected pages of the query"
            ),
            "evaluation": (
                "five balanced source-paper-disjoint folds; all held-out "
                "queries form one unlabeled workload episode"
            ),
        },
        "reference": {
            "name": "full candidate-relative BM25+visual Top-10 teacher",
            "recall_5": teacher_recall,
            "visual_candidate_events": len(rows) * candidates.shape[1],
            "global_unique_candidate_pages": len(set(int(value) for value in candidates.flat)),
        },
        "runs": runs,
        "gate": {
            "operating_point": "adaptive_boundary_risk80",
            "requirements": {
                "maximum_recall_5_loss_queries": 1,
                "minimum_unique_page_reduction": 0.20,
                "must_not_use_more_pages_than_frequency_baseline": True,
                "must_not_reduce_recall_5_vs_frequency_baseline": True,
            },
            "recall_5_delta": quality_delta,
            "recall_5_loss_in_query_units": -quality_delta * len(rows),
            "independent_episode_unique_page_reduction": page_reduction,
            "passed": bool(
                quality_delta >= -query_tolerance - 1e-12
                and page_reduction >= 0.20
                and gate["cost"]["selected_unique_page_fraction"]
                < cheap_baseline["cost"]["selected_unique_page_fraction"]
                and gate["quality"]["recall_5"] + 1e-12
                >= cheap_baseline["quality"]["recall_5"]
            ),
        },
        "artifact_sha256": {
            "score_surface": _sha256(args.score_surface),
            "queries": _sha256(args.queries),
        },
        "decision": (
            "PASS AS AN OFFLINE MECHANISM GATE, NOT YET AS AN END-TO-END "
            "SYSTEM: boundary-weighted admission must next be replayed online "
            "and timed because independent episode page counts are not a "
            "single persistent-cache measurement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
