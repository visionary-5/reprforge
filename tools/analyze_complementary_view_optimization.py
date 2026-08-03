#!/usr/bin/env python3
"""Evaluate the V3 complementary-view formulation and solver baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from reprforge.boundary_admission import (
    execute_boundary_plan,
    fit_boundary_statistics,
)
from reprforge.complementary_view_optimization import (
    ComplementaryViewSolution,
    aggregate_edge_weights,
    instance_vertices,
    query_saturated_utility,
    solve_exact_combinations,
    solve_exact_query_saturated,
    solve_frank_wolfe_diagonal,
    solve_multistart_local_search,
    solve_query_saturated_greedy,
    solve_query_saturated_multistart,
)
from reprforge.pairwise_view_admission import (
    PairAdmission,
    build_boundary_pairs,
    evaluate_pair_coverage,
    select_frequency_pages,
    select_independent_pages,
    select_pairwise_pages,
)
from reprforge.physical_cost import AtomicMaterializationCostModel
from reprforge.risk_constrained_admission import select_cost_aware_pairs
from tools.analyze_pairwise_view_admission import _balanced_group_folds
from tools.analyze_reusable_pair_probe import (
    _agreement,
    _recall,
)
from tools.run_pairwise_admission_physical import _candidate_surface


DEFAULT_BUDGETS = (0.10, 0.15, 0.20, 0.25)


def _solution(
    admission: PairAdmission,
    solver: str,
    solve_ms: float,
) -> ComplementaryViewSolution:
    return ComplementaryViewSolution(
        selected_pages=admission.selected_pages,
        admission=admission,
        solver=solver,
        solve_ms=solve_ms,
    )


def _timed_admission(
    solver: str,
    call: Callable[[], PairAdmission],
) -> ComplementaryViewSolution:
    began = time.perf_counter()
    admission = call()
    return _solution(admission, solver, (time.perf_counter() - began) * 1000.0)


def _rank(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and math.isclose(
            values[order[stop]],
            values[order[start]],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            stop += 1
        average = (start + stop - 1) / 2.0
        for offset in order[start:stop]:
            ranks[offset] = average
        start = stop
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must align and contain two values")
    left_rank = _rank(left)
    right_rank = _rank(right)
    if float(left_rank.std()) <= 1e-12 or float(right_rank.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _exact_audit_slice(
    pairs,
    *,
    vertex_limit: int,
) -> tuple:
    """Take a deterministic high-risk induced subgraph for exact auditing."""

    edge_weights = aggregate_edge_weights(pairs)
    incident: defaultdict[int, float] = defaultdict(float)
    adjacency: defaultdict[int, dict[int, float]] = defaultdict(dict)
    for (left, right), weight in edge_weights.items():
        incident[left] += weight
        incident[right] += weight
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    selected: list[int] = []
    frontier: list[int] = []
    remaining = set(instance_vertices(pairs))
    while remaining and len(selected) < vertex_limit:
        if not frontier:
            seed = min(remaining, key=lambda page: (-incident[page], page))
            frontier.append(seed)
        page = frontier.pop(0)
        if page not in remaining:
            continue
        remaining.remove(page)
        selected.append(page)
        neighbors = sorted(
            (neighbor for neighbor in adjacency[page] if neighbor in remaining),
            key=lambda neighbor: (-adjacency[page][neighbor], neighbor),
        )
        frontier.extend(neighbor for neighbor in neighbors if neighbor not in frontier)
    selected_set = set(selected)
    return tuple(
        pair
        for pair in pairs
        if pair.incumbent in selected_set and pair.challenger in selected_set
    )


def _policies(candidates, pairs, page_budget: int) -> dict[str, ComplementaryViewSolution]:
    unit_cost = AtomicMaterializationCostModel(
        batch_size=1,
        setup_ms=0.0,
        page_ms=1.0,
        batch_ms=0.0,
        score_event_ms=0.0,
    )
    policies = {
        "frequency": _timed_admission(
            "frequency",
            lambda: select_frequency_pages(
                candidates,
                pairs,
                page_budget=page_budget,
            ),
        ),
        "incident": _timed_admission(
            "incident",
            lambda: select_independent_pages(pairs, page_budget=page_budget),
        ),
        "conditional": _timed_admission(
            "conditional",
            lambda: select_pairwise_pages(pairs, page_budget=page_budget),
        ),
    }
    began = time.perf_counter()
    cost_greedy = select_cost_aware_pairs(
        pairs,
        candidates,
        unit_cost,
        time_budget_ms=float(page_budget),
    )
    policies["cost_greedy"] = _solution(
        cost_greedy.admission,
        "cost-greedy",
        (time.perf_counter() - began) * 1000.0,
    )
    policies["frank_wolfe"] = solve_frank_wolfe_diagonal(
        pairs,
        page_budget=page_budget,
    )
    policies["multistart_local"] = solve_multistart_local_search(
        pairs,
        page_budget=page_budget,
    )
    policies["query_saturated_greedy"] = solve_query_saturated_greedy(
        pairs,
        page_budget=page_budget,
    )
    policies["query_saturated_local"] = solve_query_saturated_multistart(
        pairs,
        page_budget=page_budget,
    )
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--exact-vertices", type=int, default=18)
    parser.add_argument("--exact-budget-fraction", type=float, default=0.30)
    args = parser.parse_args()
    if args.candidate_k <= args.cutoff:
        raise ValueError("candidate_k must exceed cutoff")
    if any(not 0.0 <= value <= 1.0 for value in args.budgets):
        raise ValueError("budgets must lie in [0, 1]")

    surface = np.load(args.score_surface)
    corpus_ids = [str(value) for value in surface["corpus_ids"]]
    candidates, locator, visual, visual_zscores, teacher = _candidate_surface(
        corpus_ids,
        np.asarray(surface["bm25_scores"], dtype=np.float64),
        np.asarray(surface["visual_scores"], dtype=np.float64),
        candidate_k=args.candidate_k,
        cutoff=args.cutoff,
    )
    with args.queries.open("r", encoding="utf-8", newline="") as handle:
        query_rows = list(csv.DictReader(handle))
    if len(query_rows) != len(candidates):
        raise ValueError("query metadata and score surface differ in length")
    folds = _balanced_group_folds(
        np.asarray([str(row["pdf_id"]) for row in query_rows])
    )
    gold_ids = [str(row["dataset_id"]) for row in query_rows]

    budget_rows: list[dict[str, Any]] = []
    correlation_objective: list[float] = []
    correlation_saturated: list[float] = []
    correlation_agreement: list[float] = []
    correlation_recall: list[float] = []
    for budget_fraction in args.budgets:
        fold_rows: list[dict[str, Any]] = []
        totals: dict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for fold in sorted(set(int(value) for value in folds)):
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            statistics = fit_boundary_statistics(
                locator[train],
                visual_zscores[train],
                cutoff=args.cutoff,
            )
            episode_candidates = candidates[test]
            episode_locator = locator[test]
            episode_visual = visual[test]
            episode_teacher = teacher[test]
            episode_gold = [gold_ids[int(index)] for index in test]
            eligible = len(set(int(page) for page in episode_candidates.flat))
            page_budget = math.floor(budget_fraction * eligible)
            pairs = build_boundary_pairs(
                episode_candidates,
                episode_locator,
                cutoff=args.cutoff,
                rank_risk=statistics.flip_risk_by_rank,
            )
            policy_rows: dict[str, Any] = {}
            for name, solution in _policies(
                episode_candidates,
                pairs,
                page_budget,
            ).items():
                rankings, work = execute_boundary_plan(
                    episode_candidates,
                    episode_locator,
                    episode_visual,
                    selected_pages=set(solution.selected_pages),
                    visual_prior_by_rank=statistics.visual_prior_by_rank,
                    cutoff=args.cutoff,
                )
                recall = _recall(rankings, corpus_ids, episode_gold)
                agreement = _agreement(rankings, episode_teacher)
                coverage = solution.admission.covered_weight_fraction
                saturated = query_saturated_utility(
                    pairs,
                    solution.selected_pages,
                ) / len(test)
                policy_rows[name] = {
                    "selected_pages": len(solution.selected_pages),
                    "covered_weight": solution.admission.covered_weight,
                    "covered_weight_fraction": coverage,
                    "mean_query_saturated_utility": saturated,
                    "recall_5": recall,
                    "exact_teacher_agreement": agreement,
                    "visual_candidate_events": int(work["visual_candidate_events"]),
                    "solve_ms": solution.solve_ms,
                    "iterations": solution.iterations,
                    "solver": solution.solver,
                }
                totals[name]["queries"] += len(test)
                totals[name]["pages"] += len(solution.selected_pages)
                totals[name]["coverage_weighted"] += (
                    solution.admission.covered_weight
                )
                totals[name]["total_weight"] += solution.admission.total_weight
                totals[name]["query_saturated_utility"] += saturated * len(test)
                totals[name]["recall_weighted"] += recall * len(test)
                totals[name]["agreement_weighted"] += agreement * len(test)
                totals[name]["solve_ms"] += solution.solve_ms
                correlation_objective.append(coverage)
                correlation_saturated.append(saturated)
                correlation_agreement.append(agreement)
                correlation_recall.append(recall)
            fold_rows.append(
                {
                    "held_out_fold": fold,
                    "queries": len(test),
                    "eligible_pages": eligible,
                    "page_budget": page_budget,
                    "vertices": len(instance_vertices(pairs)),
                    "aggregated_edges": len(aggregate_edge_weights(pairs)),
                    "policies": policy_rows,
                }
            )
        aggregate = {
            name: {
                "selected_pages": int(values["pages"]),
                "covered_weight_fraction": (
                    values["coverage_weighted"] / values["total_weight"]
                ),
                "mean_query_saturated_utility": (
                    values["query_saturated_utility"] / values["queries"]
                ),
                "recall_5": values["recall_weighted"] / values["queries"],
                "exact_teacher_agreement": (
                    values["agreement_weighted"] / values["queries"]
                ),
                "solve_ms": values["solve_ms"],
            }
            for name, values in sorted(totals.items())
        }
        budget_rows.append(
            {
                "budget_fraction": budget_fraction,
                "aggregate": aggregate,
                "folds": fold_rows,
            }
        )

    exact_rows: list[dict[str, Any]] = []
    for fold in sorted(set(int(value) for value in folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        statistics = fit_boundary_statistics(
            locator[train],
            visual_zscores[train],
            cutoff=args.cutoff,
        )
        pairs = build_boundary_pairs(
            candidates[test],
            locator[test],
            cutoff=args.cutoff,
            rank_risk=statistics.flip_risk_by_rank,
        )
        audit_pairs = _exact_audit_slice(pairs, vertex_limit=args.exact_vertices)
        vertices = instance_vertices(audit_pairs)
        budget = max(2, math.floor(args.exact_budget_fraction * len(vertices)))
        exact = solve_exact_combinations(audit_pairs, page_budget=budget)
        exact_saturated = solve_exact_query_saturated(
            audit_pairs,
            page_budget=budget,
        )
        approximate = _policies(
            candidates[test],
            audit_pairs,
            budget,
        )
        exact_rows.append(
            {
                "held_out_fold": fold,
                "vertices": len(vertices),
                "aggregated_edges": len(aggregate_edge_weights(audit_pairs)),
                "page_budget": budget,
                "exact_weight": exact.admission.covered_weight,
                "exact_query_saturated_utility": query_saturated_utility(
                    audit_pairs,
                    exact_saturated.selected_pages,
                ),
                "exact_solve_ms": exact.solve_ms,
                "combinations": exact.combinations_evaluated,
                "objective_ratio": {
                    name: (
                        solution.admission.covered_weight
                        / max(exact.admission.covered_weight, 1e-12)
                    )
                    for name, solution in approximate.items()
                },
                "query_saturated_objective_ratio": {
                    name: (
                        query_saturated_utility(
                            audit_pairs,
                            solution.selected_pages,
                        )
                        / max(
                            query_saturated_utility(
                                audit_pairs,
                                exact_saturated.selected_pages,
                            ),
                            1e-12,
                        )
                    )
                    for name, solution in approximate.items()
                },
            }
        )

    payload = {
        "schema_version": 1,
        "experiment": "complementary-view-optimization-v3",
        "formulation": {
            "objective": "sum edge weight induced by selected page views",
            "uniform_budget_equivalence": "weighted-heaviest/densest-k-subgraph",
            "objective_class": "monotone-supermodular",
            "qrels_visible_to_planner": False,
            "visual_scores_visible_to_planner": False,
            "risk_and_missing_view_prior": "fit on other source-paper folds",
            "frank_wolfe_is_prior_art": True,
        },
        "configuration": {
            "candidate_k": args.candidate_k,
            "cutoff": args.cutoff,
            "budgets": list(args.budgets),
            "outer_split": "source-paper-disjoint-five-fold",
            "missing_visual_prior": "fit on other source-paper folds",
            "exact_vertices": args.exact_vertices,
            "exact_budget_fraction": args.exact_budget_fraction,
        },
        "predictive_validity": {
            "spearman_coverage_vs_exact_teacher_agreement": _spearman(
                correlation_objective,
                correlation_agreement,
            ),
            "spearman_coverage_vs_recall_5": _spearman(
                correlation_objective,
                correlation_recall,
            ),
            "spearman_query_saturation_vs_exact_teacher_agreement": _spearman(
                correlation_saturated,
                correlation_agreement,
            ),
            "spearman_query_saturation_vs_recall_5": _spearman(
                correlation_saturated,
                correlation_recall,
            ),
            "observations": len(correlation_objective),
        },
        "budgets": budget_rows,
        "exact_audit_slices": exact_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
