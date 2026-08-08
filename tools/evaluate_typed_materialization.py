#!/usr/bin/env python3
"""Test typed persistent visual materialization on exact score surfaces.

The experiment separates two capabilities that an untyped sparse visual index
incorrectly conflates:

* benefit: visually rerank a page only after the cheap locator found it;
* coverage: let a persistently indexed page repair a cheap-locator escape.

History-qrel and future-qrel selectors are explicitly reported as verifier and
oracle upper bounds.  The cheap selector uses only ingestion/history features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.partial_vlm_materialization import evaluate_text_only
from reprforge.value_aware_materialization import (
    evaluate_typed_materialization,
)
from tools.evaluate_value_aware_materialization import (
    load_exported_surface,
    load_features,
    load_split,
)


def _mean(result: dict[str, Any]) -> float:
    return float(result["mean_ndcg_at_10"])


def _evaluate(
    surface,
    queries: Sequence[int],
    *,
    pages: Sequence[int],
    anchors: np.ndarray,
    role: str,
    parameters: dict[str, Any],
    calibration_queries: Sequence[int] | None = None,
) -> dict[str, Any]:
    selected = np.asarray(sorted(set(map(int, pages))), dtype=np.int32)
    benefit = selected if role in ("benefit", "both") else np.asarray([], dtype=np.int32)
    coverage = selected if role in ("coverage", "both") else np.asarray([], dtype=np.int32)
    result = evaluate_typed_materialization(
        surface,
        queries,
        benefit_pages=benefit,
        coverage_pages=coverage,
        anchor_pages=anchors,
        calibration_queries=calibration_queries,
        **parameters,
    )
    return {
        "materialized_pages": int(len(selected)),
        "materialized_fraction": len(selected) / surface.pages,
        "role": role,
        **result,
    }


def _tune_typed_fusion(surface, history, anchors) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune fusion only on history with every page physically available."""

    all_pages = np.arange(surface.pages, dtype=np.int32)
    rows = []
    for candidate_k in (20, 50, 100):
        for benefit_z_threshold in (0.0, 0.5, 1.0, 1.5):
            for benefit_weight in (0.25, 0.5, 1.0, 2.0):
                parameters = {
                    "candidate_k": candidate_k,
                    "benefit_quantile": 0.75,
                    "coverage_quantile": 0.99,
                    "benefit_weight": benefit_weight,
                    "coverage_weight": 0.0,
                    "benefit_z_threshold": benefit_z_threshold,
                    "coverage_z_threshold": 2.0,
                }
                quality = _mean(
                    _evaluate(
                        surface,
                        history,
                        pages=all_pages,
                        anchors=anchors,
                        role="benefit",
                        parameters=parameters,
                        calibration_queries=history,
                    )
                )
                rows.append({**parameters, "stage": "benefit", "history_ndcg_at_10": quality})
    benefit_best = max(
        rows,
        key=lambda row: (row["history_ndcg_at_10"], -row["candidate_k"], -row["benefit_weight"]),
    )
    coverage_rows = []
    for coverage_z_threshold in (1.5, 2.0, 2.5, 3.0):
        for coverage_weight in (0.0, 0.25, 0.5, 1.0, 2.0):
            parameters = {
                "candidate_k": int(benefit_best["candidate_k"]),
                "benefit_quantile": float(benefit_best["benefit_quantile"]),
                "coverage_quantile": 0.99,
                "benefit_weight": float(benefit_best["benefit_weight"]),
                "coverage_weight": coverage_weight,
                "benefit_z_threshold": float(benefit_best["benefit_z_threshold"]),
                "coverage_z_threshold": coverage_z_threshold,
            }
            quality = _mean(
                _evaluate(
                    surface,
                    history,
                    pages=all_pages,
                    anchors=anchors,
                    role="both",
                    parameters=parameters,
                    calibration_queries=history,
                )
            )
            coverage_rows.append(
                {**parameters, "stage": "coverage", "history_ndcg_at_10": quality}
            )
    rows.extend(coverage_rows)
    best = max(
        coverage_rows,
        key=lambda row: (row["history_ndcg_at_10"], -row["coverage_weight"]),
    )
    return {
        key: best[key]
        for key in (
            "candidate_k",
            "benefit_quantile",
            "coverage_quantile",
            "benefit_weight",
            "coverage_weight",
            "benefit_z_threshold",
            "coverage_z_threshold",
        )
    }, rows


def _relevance_counts(surface, queries, candidate_k: int) -> tuple[np.ndarray, np.ndarray]:
    benefit = np.zeros(surface.pages, dtype=np.float64)
    coverage = np.zeros(surface.pages, dtype=np.float64)
    for query in map(int, queries):
        relevant = np.flatnonzero(surface.qrels[query] > 0)
        candidate = np.zeros(surface.pages, dtype=bool)
        candidate[surface.text_order[query, : min(candidate_k, surface.pages)]] = True
        benefit[relevant[candidate[relevant]]] += surface.qrels[query, relevant[candidate[relevant]]]
        coverage[relevant[~candidate[relevant]]] += surface.qrels[query, relevant[~candidate[relevant]]]
    return benefit, coverage


def _ordered(values: np.ndarray) -> np.ndarray:
    pages = np.arange(len(values), dtype=np.int32)
    return pages[np.lexsort((pages, -np.asarray(values, dtype=np.float64)))]


def _take_with_anchors(order: Sequence[int], anchors: np.ndarray, count: int) -> np.ndarray:
    selected = set(map(int, anchors))
    for page in order:
        selected.add(int(page))
        if len(selected) >= count:
            break
    return np.asarray(sorted(selected), dtype=np.int32)


def _cheap_balanced_selection(features: np.ndarray, anchors: np.ndarray, count: int) -> np.ndarray:
    """Half workload frequency, half visual-risk proxy, with no visual scores/qrels."""

    target = max(0, count - len(anchors))
    frequency = _ordered(features[:, 4])
    # High entropy + edges + locator disagreement, plus low text availability.
    risk = -features[:, 0] + features[:, 1] + features[:, 2] + features[:, 3]
    risk_order = _ordered(risk)
    selected = set(map(int, anchors))
    frequency_target = math.ceil(target / 2)
    for page in frequency:
        if len(selected) >= len(anchors) + frequency_target:
            break
        selected.add(int(page))
    for page in risk_order:
        selected.add(int(page))
        if len(selected) >= count:
            break
    return np.asarray(sorted(selected), dtype=np.int32)


def run(surface, features, history, evaluation, *, budgets, anchor_pages, seed):
    anchors = np.sort(
        np.random.default_rng(seed).permutation(surface.pages)[:anchor_pages]
    )
    parameters, tuning = _tune_typed_fusion(surface, history, anchors)
    candidate_k = int(parameters["candidate_k"])
    history_benefit, history_coverage = _relevance_counts(surface, history, candidate_k)
    future_benefit, future_coverage = _relevance_counts(surface, evaluation, candidate_k)
    history_total = history_benefit + history_coverage
    future_total = future_benefit + future_coverage
    baseline = evaluate_text_only(surface, evaluation)
    full = _evaluate(
        surface,
        evaluation,
        pages=np.arange(surface.pages),
        anchors=anchors,
        role="both",
        parameters=parameters,
        calibration_queries=history,
    )
    dvi_like = _evaluate(
        surface,
        evaluation,
        pages=np.arange(surface.pages),
        anchors=anchors,
        role="benefit",
        parameters=parameters,
        calibration_queries=history,
    )
    curves: dict[str, dict[str, Any]] = {
        name: {}
        for name in (
            "random_typed",
            "cheap_balanced_typed",
            "history_verifier_untyped_role_benefit",
            "history_verifier_untyped_role_coverage",
            "history_verifier_typed",
            "future_oracle_typed",
        )
    }
    rng_order = np.random.default_rng(seed + 1).permutation(surface.pages)
    for fraction in budgets:
        count = max(anchor_pages, int(math.ceil(float(fraction) * surface.pages)))
        selections = {
            "random_typed": _take_with_anchors(rng_order, anchors, count),
            "cheap_balanced_typed": _cheap_balanced_selection(features, anchors, count),
            "history_verifier_untyped_role_benefit": _take_with_anchors(
                _ordered(history_total), anchors, count
            ),
            "history_verifier_untyped_role_coverage": _take_with_anchors(
                _ordered(history_total), anchors, count
            ),
            "history_verifier_typed": _take_with_anchors(
                _ordered(history_total), anchors, count
            ),
            "future_oracle_typed": _take_with_anchors(
                _ordered(future_total), anchors, count
            ),
        }
        roles = {
            "random_typed": "both",
            "cheap_balanced_typed": "both",
            "history_verifier_untyped_role_benefit": "benefit",
            "history_verifier_untyped_role_coverage": "coverage",
            "history_verifier_typed": "both",
            "future_oracle_typed": "both",
        }
        for name, selected in selections.items():
            result = _evaluate(
                surface,
                evaluation,
                pages=selected,
                anchors=anchors,
                role=roles[name],
                parameters=parameters,
                calibration_queries=history,
            )
            result["vs_text_ndcg_at_10"] = _mean(result) - _mean(baseline)
            result["full_typed_gain_recovery"] = (
                (_mean(result) - _mean(baseline)) / (_mean(full) - _mean(baseline))
                if abs(_mean(full) - _mean(baseline)) > 1e-12
                else None
            )
            curves[name][str(fraction)] = result
    return {
        "schema_version": 1,
        "domain": surface.name,
        "queries": surface.queries,
        "pages": surface.pages,
        "history_queries": len(history),
        "evaluation_queries": len(evaluation),
        "anchors": anchors.tolist(),
        "parameters_tuned_on_history_only": parameters,
        "tuning": tuning,
        "evaluation_baselines": {
            "text_only": baseline,
            "dvi_like_transient_candidate_visual_rerank": dvi_like,
            "full_typed_visual": full,
        },
        "history_feedback_semantics": (
            "qrels are a perfect historical verifier upper bound; future qrels occur only in the named oracle"
        ),
        "curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--query-splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.02, 0.05, 0.1, 0.2, 0.4))
    parser.add_argument("--anchor-pages", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    surface, _, _ = load_exported_surface(args.score_root, args.dataset_root)
    features = load_features(args.features, surface)
    history, evaluation = load_split(args.query_splits, surface)
    result = run(
        surface,
        features,
        history,
        evaluation,
        budgets=args.budgets,
        anchor_pages=args.anchor_pages,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "domain": result["domain"],
                "parameters": result["parameters_tuned_on_history_only"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
