#!/usr/bin/env python3
"""Evaluate pair-level probes that become the final visual index."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.boundary_admission import execute_boundary_plan
from reprforge.pairwise_view_admission import (
    BoundaryPair,
    select_frequency_pages,
    select_independent_pages,
)
from reprforge.reusable_pair_probe import (
    FrozenPairScoreProvider,
    build_probe_edges,
    normal_boundary_risk,
    select_reusable_pair_probes,
)
from reprforge.risk_constrained_admission import estimate_plan_cost
from tools.analyze_pairwise_view_admission import _balanced_group_folds
from tools.analyze_sparse_risk_admission import _load_cost_model
from tools.run_pairwise_admission_physical import _candidate_surface


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


def _agreement(rankings: np.ndarray, teacher: np.ndarray) -> float:
    return float(
        np.mean(
            [
                set(row) == set(reference)
                for row, reference in zip(rankings, teacher, strict=True)
            ]
        )
    )


def _boundary_pairs(
    candidates: np.ndarray,
    locator: np.ndarray,
    *,
    cutoff: int,
) -> tuple[BoundaryPair, ...]:
    return tuple(
        BoundaryPair(
            query_index=edge.query_index,
            incumbent=edge.incumbent_page,
            challenger=edge.challenger_page,
            challenger_rank=edge.challenger_rank,
            weight=normal_boundary_risk(edge.locator_margin),
        )
        for edge in build_probe_edges(candidates, locator, cutoff=cutoff)
    )


def _evaluate_selected(
    candidates: np.ndarray,
    locator: np.ndarray,
    visual: np.ndarray,
    teacher: np.ndarray,
    selected: set[int] | frozenset[int],
    corpus_ids: list[str],
    gold_ids: list[str],
    *,
    cutoff: int,
) -> dict[str, Any]:
    rankings, work = execute_boundary_plan(
        candidates,
        locator,
        visual,
        selected_pages=set(selected),
        visual_prior_by_rank=np.zeros(candidates.shape[1]),
        cutoff=cutoff,
    )
    return {
        "rankings": rankings,
        "selected_pages": len(selected),
        "score_events": int(work["visual_candidate_events"]),
        "exact_teacher_agreement": _agreement(rankings, teacher),
        "recall_5": _recall(rankings, corpus_ids, gold_ids),
    }


def _pair_signal_audit(
    candidates: np.ndarray,
    locator: np.ndarray,
    visual: np.ndarray,
    teacher: np.ndarray,
    *,
    cutoff: int,
) -> dict[str, float | int]:
    labels: list[bool] = []
    deltas: list[float] = []
    margins: list[float] = []
    for query in range(len(candidates)):
        teacher_set = set(int(page) for page in teacher[query])
        for rank in range(cutoff, candidates.shape[1]):
            labels.append(int(candidates[query, rank]) in teacher_set)
            deltas.append(float(visual[query, rank] - visual[query, cutoff - 1]))
            margins.append(float(locator[query, cutoff - 1] - locator[query, rank]))
    target = np.asarray(labels, dtype=bool)
    delta = np.asarray(deltas, dtype=np.float64)
    margin = np.asarray(margins, dtype=np.float64)
    positive = delta[target]
    negative = delta[~target]
    comparisons = positive[:, None] - negative[None, :]
    auc = float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))
    scale = math.sqrt(max(float(np.mean(delta**2)) / 2.0, 1e-12))
    predicted = delta / scale > margin
    true_positive = int(np.sum(predicted & target))
    true_negative = int(np.sum(~predicted & ~target))
    sensitivity = true_positive / max(int(target.sum()), 1)
    specificity = true_negative / max(int((~target).sum()), 1)
    return {
        "events": len(target),
        "positive_events": int(target.sum()),
        "positive_fraction": float(target.mean()),
        "pair_delta_auc": auc,
        "unsupervised_scale": scale,
        "unsupervised_balanced_accuracy": 0.5 * (sensitivity + specificity),
        "unsupervised_precision": float(
            true_positive / max(int(predicted.sum()), 1)
        ),
        "unsupervised_recall": sensitivity,
    }


def _evaluate_budget(
    *,
    budget_fraction: float,
    folds: np.ndarray,
    candidates: np.ndarray,
    locator: np.ndarray,
    visual: np.ndarray,
    teacher: np.ndarray,
    corpus_ids: list[str],
    gold_ids: list[str],
    cost_model,
    cutoff: int,
    round_page_limit: int,
    warmup_page_limit: int,
    minimum_observations: int,
) -> dict[str, Any]:
    policies = ("frequency", "independent", "static_pair", "active_pair")
    totals: dict[str, dict[str, float]] = {
        policy: Counter() for policy in policies
    }
    folds_out: list[dict[str, Any]] = []
    for fold in sorted(set(int(value) for value in folds)):
        test = np.flatnonzero(folds == fold)
        episode_candidates = candidates[test]
        episode_locator = locator[test]
        episode_visual = visual[test]
        episode_teacher = teacher[test]
        episode_gold = [gold_ids[int(query)] for query in test]
        eligible = len(set(int(page) for page in episode_candidates.flat))
        page_budget = math.floor(budget_fraction * eligible)
        time_budget = cost_model.estimate(
            pages=page_budget,
            score_events=0,
        ).total_ms
        pairs = _boundary_pairs(
            episode_candidates,
            episode_locator,
            cutoff=cutoff,
        )
        selected_by_policy: dict[str, frozenset[int]] = {
            "frequency": select_frequency_pages(
                episode_candidates,
                pairs,
                page_budget=page_budget,
            ).selected_pages,
            "independent": select_independent_pages(
                pairs,
                page_budget=page_budget,
            ).selected_pages,
        }
        pair_diagnostics: dict[str, dict[str, Any]] = {}
        for policy, empirical in (("static_pair", False), ("active_pair", True)):
            provider = FrozenPairScoreProvider(
                episode_candidates,
                episode_visual,
            )
            plan = select_reusable_pair_probes(
                episode_candidates,
                episode_locator,
                provider,
                cost_model,
                cutoff=cutoff,
                time_budget_ms=time_budget,
                empirical_updates=empirical,
                round_page_limit=round_page_limit,
                warmup_page_limit=warmup_page_limit,
                minimum_observations=minimum_observations,
            )
            selected_by_policy[policy] = plan.selected_pages
            pair_diagnostics[policy] = {
                "observed_pairs": plan.observed_pair_count,
                "score_reads": len(provider.score_reads),
                "iterations": plan.iterations,
                "materialization_rounds": plan.materialization_rounds,
            }

        fold_row: dict[str, Any] = {
            "held_out_fold": fold,
            "queries": len(test),
            "eligible_pages": eligible,
            "page_budget": page_budget,
            "policies": {},
        }
        for policy in policies:
            metrics = _evaluate_selected(
                episode_candidates,
                episode_locator,
                episode_visual,
                episode_teacher,
                selected_by_policy[policy],
                corpus_ids,
                episode_gold,
                cutoff=cutoff,
            )
            estimated = estimate_plan_cost(
                episode_candidates,
                selected_by_policy[policy],
                cost_model,
            )
            compact = {
                key: value for key, value in metrics.items() if key != "rankings"
            }
            compact["estimated_visual_ms"] = estimated.total_ms
            compact.update(pair_diagnostics.get(policy, {}))
            fold_row["policies"][policy] = compact
            totals[policy]["queries"] += len(test)
            totals[policy]["selected_pages"] += compact["selected_pages"]
            totals[policy]["score_events"] += compact["score_events"]
            totals[policy]["agreement_weighted"] += (
                compact["exact_teacher_agreement"] * len(test)
            )
            totals[policy]["recall_weighted"] += compact["recall_5"] * len(test)
            totals[policy]["estimated_visual_ms"] += estimated.total_ms
            if policy in pair_diagnostics:
                totals[policy]["observed_pairs"] += compact["observed_pairs"]
                totals[policy]["iterations"] += compact["iterations"]
                totals[policy]["materialization_rounds"] += compact[
                    "materialization_rounds"
                ]
        folds_out.append(fold_row)

    aggregate: dict[str, Any] = {}
    for policy in policies:
        row = totals[policy]
        aggregate[policy] = {
            "selected_pages": int(row["selected_pages"]),
            "score_events": int(row["score_events"]),
            "exact_teacher_agreement": row["agreement_weighted"] / row["queries"],
            "recall_5": row["recall_weighted"] / row["queries"],
            "estimated_visual_ms": row["estimated_visual_ms"],
        }
        if policy in ("static_pair", "active_pair"):
            aggregate[policy]["observed_pairs"] = int(row["observed_pairs"])
            aggregate[policy]["iterations"] = int(row["iterations"])
            aggregate[policy]["materialization_rounds"] = int(
                row["materialization_rounds"]
            )
    return {
        "budget_fraction": budget_fraction,
        "aggregate": aggregate,
        "folds": folds_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--physical-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=(0.1, 0.15, 0.2, 0.25),
    )
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--round-page-limit", type=int, default=8)
    parser.add_argument("--warmup-page-limit", type=int, default=8)
    parser.add_argument("--minimum-observations", type=int, default=4)
    args = parser.parse_args()

    surface = np.load(args.score_surface)
    corpus_ids = [str(value) for value in surface["corpus_ids"]]
    candidates, locator, visual, _, teacher = _candidate_surface(
        corpus_ids,
        np.asarray(surface["bm25_scores"], dtype=np.float64),
        np.asarray(surface["visual_scores"], dtype=np.float64),
        candidate_k=args.candidate_k,
        cutoff=args.cutoff,
    )
    with args.queries.open("r", encoding="utf-8", newline="") as handle:
        query_rows = list(csv.DictReader(handle))
    groups = np.asarray([str(row["pdf_id"]) for row in query_rows])
    gold_ids = [str(row["dataset_id"]) for row in query_rows]
    if len(groups) != len(candidates):
        raise ValueError("query metadata and score surface differ in length")
    folds = _balanced_group_folds(groups)
    cost_model, cost_diagnostics = _load_cost_model(
        args.physical_runs,
        batch_size=args.batch_size,
    )
    payload = {
        "schema_version": 1,
        "experiment": "reusable-pair-probe-admission",
        "contract": {
            "probe_artifact_is_final_view": True,
            "unmaterialized_scores_visible_to_planner": False,
            "qrels_visible_to_planner": False,
            "outer_split": "source-paper-disjoint-five-fold",
            "missing_visual_prior": "zero",
            "candidate_k": args.candidate_k,
            "cutoff": args.cutoff,
            "round_page_limit": args.round_page_limit,
            "warmup_page_limit": args.warmup_page_limit,
            "minimum_observations": args.minimum_observations,
        },
        "cost_model": {
            "batch_size": cost_model.batch_size,
            "setup_ms": cost_model.setup_ms,
            "page_ms": cost_model.page_ms,
            "batch_ms": cost_model.batch_ms,
            "score_event_ms": cost_model.score_event_ms,
            **cost_diagnostics,
        },
        "pair_signal": _pair_signal_audit(
            candidates,
            locator,
            visual,
            teacher,
            cutoff=args.cutoff,
        ),
        "budgets": [
            _evaluate_budget(
                budget_fraction=float(budget),
                folds=folds,
                candidates=candidates,
                locator=locator,
                visual=visual,
                teacher=teacher,
                corpus_ids=corpus_ids,
                gold_ids=gold_ids,
                cost_model=cost_model,
                cutoff=args.cutoff,
                round_page_limit=args.round_page_limit,
                warmup_page_limit=args.warmup_page_limit,
                minimum_observations=args.minimum_observations,
            )
            for budget in args.budgets
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
