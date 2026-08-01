#!/usr/bin/env python3
"""Test whether observed rank interventions predict representation utility.

This is a diagnostic learner, not the final online controller. It builds
query--page events from frozen text and visual score traces, labels each event
with the exact nDCG change caused by replacing that page's text score, and
compares progressively richer runtime-visible feature views on held-out
queries. Relevance labels are targets only; they are never model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.progressive_oracle import (
    FrozenTrace,
    load_trace,
    mean_ndcg,
    rank_order,
    validate_pair,
)


PRE_FEATURES = (
    "text_score_z",
    "text_rank_fraction",
    "text_margin_to_top_z",
    "text_margin_to_cutoff_z",
    "query_text_score_std_log",
)
POST_FEATURES = PRE_FEATURES + (
    "visual_minus_text_z",
    "visual_vs_text_cutoff_z",
    "counterfactual_rank_fraction",
    "counterfactual_rank_shift_fraction",
    "enters_cutoff",
    "leaves_cutoff",
    "visual_candidate_score_z",
    "visual_candidate_rank_fraction",
    "visual_candidate_margin_to_cutoff_z",
    "text_visual_candidate_rank_shift_fraction",
)
HISTORY_FEATURES = POST_FEATURES + (
    "prior_page_touches_log",
    "page_recency_inverse",
    "prior_route_page_touches_log",
    "route_page_recency_inverse",
    "prior_route_queries_log",
)
FEATURE_VIEWS = {
    "pre": PRE_FEATURES,
    "post_intervention": POST_FEATURES,
    "post_intervention_history": HISTORY_FEATURES,
}


@dataclass(frozen=True)
class InterventionEvents:
    feature_names: tuple[str, ...]
    features: np.ndarray
    utility: np.ndarray
    query_position: np.ndarray
    corpus_position: np.ndarray
    candidate_rank: np.ndarray
    query_split: np.ndarray
    query_route: np.ndarray
    candidate_k: int
    cutoff: int


@dataclass(frozen=True)
class RidgeUtilityModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        return normalized @ self.coefficients + self.intercept


def stable_query_split(query_ids: Sequence[str]) -> np.ndarray:
    """Return a deterministic 60/20/20 split without using query order."""

    split = []
    for value in query_ids:
        bucket = hashlib.sha256(str(value).encode("utf-8")).digest()[0] % 10
        split.append("train" if bucket < 6 else "validation" if bucket < 8 else "test")
    result = np.asarray(split)
    if set(result.tolist()) != {"train", "validation", "test"}:
        raise ValueError("query identifiers do not populate all three splits")
    return result


def _normalized_score_profiles(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norm, 1e-12)


def assign_query_routes(
    scores: np.ndarray,
    train_queries: np.ndarray,
    *,
    route_count: int,
    iterations: int = 25,
) -> np.ndarray:
    """Cluster text-score profiles using training queries and no labels."""

    profiles = _normalized_score_profiles(scores)
    training = profiles[np.asarray(train_queries, dtype=np.int64)]
    count = min(route_count, len(training))
    if count <= 0:
        raise ValueError("route_count and training query count must be positive")

    selected = [0]
    nearest = 1.0 - training @ training[0]
    for _ in range(1, count):
        chosen = int(np.argmax(nearest))
        selected.append(chosen)
        nearest = np.minimum(nearest, 1.0 - training @ training[chosen])
    centers = training[selected].copy()

    for _ in range(iterations):
        assignment = np.argmax(training @ centers.T, axis=1)
        updated = centers.copy()
        for route in range(count):
            members = training[assignment == route]
            if len(members):
                center = members.mean(axis=0)
                updated[route] = center / max(np.linalg.norm(center), 1e-12)
        if np.allclose(updated, centers, atol=1e-8, rtol=0.0):
            centers = updated
            break
        centers = updated
    return np.argmax(profiles @ centers.T, axis=1).astype(np.int16)


def _ndcg_row(
    scores: np.ndarray,
    relevance: np.ndarray,
    corpus_ids: np.ndarray,
    *,
    cutoff: int,
) -> float:
    order = np.lexsort((corpus_ids, -scores))[:cutoff]
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    ideal = np.sort(relevance)[-cutoff:][::-1]
    denominator = float(np.dot(ideal, discounts))
    if denominator <= 0:
        raise ValueError("nDCG is undefined without a relevant page")
    return float(np.dot(relevance[order], discounts) / denominator)


def build_intervention_events(
    text: FrozenTrace,
    visual: FrozenTrace,
    *,
    candidate_k: int = 20,
    cutoff: int = 10,
    route_count: int = 8,
) -> InterventionEvents:
    """Create exact single-page intervention labels and observable features."""

    qrels = validate_pair(text, visual)
    query_count, corpus_count = text.scores.shape
    candidate_k = min(candidate_k, corpus_count)
    if candidate_k <= 0 or cutoff <= 0 or cutoff > corpus_count:
        raise ValueError("candidate_k/cutoff is outside the corpus shape")

    split_by_query = stable_query_split(text.query_ids.tolist())
    train_queries = np.flatnonzero(split_by_query == "train")
    routes = assign_query_routes(
        text.scores,
        train_queries,
        route_count=route_count,
    )
    text_order = rank_order(text.scores, text.corpus_ids)
    candidates = text_order[:, :candidate_k]

    page_touches = np.zeros(corpus_count, dtype=np.int64)
    page_last = np.full(corpus_count, -1, dtype=np.int64)
    route_touches = np.zeros((int(routes.max()) + 1, corpus_count), dtype=np.int64)
    route_last = np.full_like(route_touches, -1)
    route_queries = np.zeros(route_touches.shape[0], dtype=np.int64)

    rows: list[list[float]] = []
    labels: list[float] = []
    query_positions: list[int] = []
    corpus_positions: list[int] = []
    candidate_ranks: list[int] = []
    event_splits: list[str] = []
    event_routes: list[int] = []

    for query in range(query_count):
        route = int(routes[query])
        text_row = text.scores[query]
        visual_row = visual.scores[query]
        relevance = qrels[query]
        std = max(float(text_row.std()), 1e-12)
        top_score = float(text_row[text_order[query, 0]])
        cutoff_score = float(text_row[text_order[query, cutoff - 1]])
        candidate_pages = candidates[query]
        candidate_visual_scores = visual_row[candidate_pages]
        candidate_visual_std = max(float(candidate_visual_scores.std()), 1e-12)
        candidate_visual_mean = float(candidate_visual_scores.mean())
        visual_candidate_order = np.lexsort(
            (text.corpus_ids[candidate_pages], -candidate_visual_scores)
        )
        visual_candidate_rank = np.empty(candidate_k, dtype=np.int32)
        visual_candidate_rank[visual_candidate_order] = np.arange(candidate_k)
        visual_candidate_cutoff = min(cutoff, candidate_k) - 1
        visual_candidate_cutoff_score = float(
            candidate_visual_scores[visual_candidate_order[visual_candidate_cutoff]]
        )
        base_ndcg = _ndcg_row(
            text_row,
            relevance,
            text.corpus_ids,
            cutoff=cutoff,
        )
        for rank, page_value in enumerate(candidates[query]):
            page = int(page_value)
            toggled = text_row.copy()
            toggled[page] = visual_row[page]
            toggled_order = np.lexsort((text.corpus_ids, -toggled))
            new_rank = int(np.flatnonzero(toggled_order == page)[0])
            toggled_ndcg = _ndcg_row(
                toggled,
                relevance,
                text.corpus_ids,
                cutoff=cutoff,
            )
            prior_touch = int(page_touches[page])
            prior_route_touch = int(route_touches[route, page])
            page_gap = query - int(page_last[page]) if page_last[page] >= 0 else 0
            route_gap = (
                query - int(route_last[route, page])
                if route_last[route, page] >= 0
                else 0
            )
            rows.append(
                [
                    float(text_row[page]) / std,
                    rank / max(corpus_count - 1, 1),
                    (float(text_row[page]) - top_score) / std,
                    (float(text_row[page]) - cutoff_score) / std,
                    float(np.log1p(std)),
                    (float(visual_row[page]) - float(text_row[page])) / std,
                    (float(visual_row[page]) - cutoff_score) / std,
                    new_rank / max(corpus_count - 1, 1),
                    (rank - new_rank) / max(corpus_count - 1, 1),
                    float(rank >= cutoff and new_rank < cutoff),
                    float(rank < cutoff and new_rank >= cutoff),
                    (
                        float(visual_row[page]) - candidate_visual_mean
                    )
                    / candidate_visual_std,
                    int(visual_candidate_rank[rank]) / max(candidate_k - 1, 1),
                    (
                        float(visual_row[page]) - visual_candidate_cutoff_score
                    )
                    / candidate_visual_std,
                    (rank - int(visual_candidate_rank[rank]))
                    / max(candidate_k - 1, 1),
                    float(np.log1p(prior_touch)),
                    1.0 / (1.0 + page_gap) if prior_touch else 0.0,
                    float(np.log1p(prior_route_touch)),
                    1.0 / (1.0 + route_gap) if prior_route_touch else 0.0,
                    float(np.log1p(route_queries[route])),
                ]
            )
            labels.append(toggled_ndcg - base_ndcg)
            query_positions.append(query)
            corpus_positions.append(page)
            candidate_ranks.append(rank)
            event_splits.append(str(split_by_query[query]))
            event_routes.append(route)

        pages = candidates[query]
        page_touches[pages] += 1
        page_last[pages] = query
        route_touches[route, pages] += 1
        route_last[route, pages] = query
        route_queries[route] += 1

    return InterventionEvents(
        feature_names=HISTORY_FEATURES,
        features=np.asarray(rows, dtype=np.float64),
        utility=np.asarray(labels, dtype=np.float64),
        query_position=np.asarray(query_positions, dtype=np.int32),
        corpus_position=np.asarray(corpus_positions, dtype=np.int32),
        candidate_rank=np.asarray(candidate_ranks, dtype=np.int16),
        query_split=np.asarray(event_splits),
        query_route=np.asarray(event_routes, dtype=np.int16),
        candidate_k=candidate_k,
        cutoff=cutoff,
    )


def fit_ridge_utility(
    features: np.ndarray,
    utility: np.ndarray,
    *,
    alpha: float = 1.0,
    nonzero_weight: float = 8.0,
) -> RidgeUtilityModel:
    if len(features) != len(utility) or features.ndim != 2:
        raise ValueError("features and utility are not aligned")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (features - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    weights = np.where(np.abs(utility) > 1e-12, nonzero_weight, 1.0)
    gram = design.T @ (design * weights[:, None])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    solution = np.linalg.solve(
        gram + penalty,
        design.T @ (utility * weights),
    )
    return RidgeUtilityModel(
        mean=mean,
        scale=scale,
        coefficients=solution[1:],
        intercept=float(solution[0]),
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels.astype(bool)
    count = int(positive.sum())
    if count == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = positive[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / count)


def _prediction_diagnostics(utility: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    positive = utility > 1e-12
    harmful = utility < -1e-12
    top_count = max(1, int(np.ceil(0.20 * len(utility))))
    top = np.argsort(-prediction, kind="stable")[:top_count]
    total_positive_utility = float(np.maximum(utility, 0.0).sum())
    return {
        "positive_event_fraction": float(positive.mean()),
        "harmful_event_fraction": float(harmful.mean()),
        "positive_average_precision": average_precision(positive, prediction),
        "harmful_average_precision": average_precision(harmful, -prediction),
        "top20_percent_positive_precision": float(positive[top].mean()),
        "top20_percent_positive_utility_capture": (
            float(np.maximum(utility[top], 0.0).sum()) / total_positive_utility
            if total_positive_utility
            else 0.0
        ),
        "pearson_utility": (
            float(np.corrcoef(utility, prediction)[0, 1])
            if np.std(utility) > 0 and np.std(prediction) > 0
            else 0.0
        ),
    }


def _policy_ndcg(
    text: FrozenTrace,
    visual: FrozenTrace,
    qrels: np.ndarray,
    events: InterventionEvents,
    query_mask: np.ndarray,
    event_mask: np.ndarray,
) -> float:
    queries = np.flatnonzero(query_mask)
    scores = text.scores[queries].copy()
    query_remap = {int(query): index for index, query in enumerate(queries)}
    for event in np.flatnonzero(event_mask):
        query = int(events.query_position[event])
        if query in query_remap:
            page = int(events.corpus_position[event])
            scores[query_remap[query], page] = visual.scores[query, page]
    return mean_ndcg(scores, qrels[queries], text.corpus_ids, cutoff=events.cutoff)


def choose_threshold(
    prediction: np.ndarray,
    utility: np.ndarray,
    *,
    maximum_activation_fraction: float = 0.30,
) -> float:
    """Choose a validation-only threshold by true summed marginal utility."""

    quantiles = np.linspace(0.70, 1.0, 31)
    thresholds = np.unique(np.quantile(prediction, quantiles))
    eligible = []
    for threshold in thresholds:
        active = prediction >= threshold
        if float(active.mean()) <= maximum_activation_fraction:
            eligible.append((float(utility[active].sum()), float(threshold)))
    return max(eligible, default=(0.0, float("inf")))[1]


def analyze_feature_views(
    text: FrozenTrace,
    visual: FrozenTrace,
    events: InterventionEvents,
) -> dict[str, Any]:
    qrels = validate_pair(text, visual)
    split_by_query = stable_query_split(text.query_ids.tolist())
    query_masks = {name: split_by_query == name for name in ("train", "validation", "test")}
    event_masks = {name: events.query_split == name for name in query_masks}
    feature_index = {name: index for index, name in enumerate(events.feature_names)}

    test_queries = query_masks["test"]
    test_events = event_masks["test"]
    text_test = mean_ndcg(text.scores[test_queries], qrels[test_queries], text.corpus_ids, cutoff=events.cutoff)
    visual_test = mean_ndcg(visual.scores[test_queries], qrels[test_queries], text.corpus_ids, cutoff=events.cutoff)
    activate_all_test = _policy_ndcg(text, visual, qrels, events, test_queries, test_events)

    views: dict[str, Any] = {}
    for name, names in FEATURE_VIEWS.items():
        columns = [feature_index[value] for value in names]
        model = fit_ridge_utility(
            events.features[event_masks["train"]][:, columns],
            events.utility[event_masks["train"]],
        )
        validation_prediction = model.predict(events.features[event_masks["validation"]][:, columns])
        threshold = choose_threshold(
            validation_prediction,
            events.utility[event_masks["validation"]],
        )
        test_prediction = model.predict(events.features[test_events][:, columns])
        active_test_local = test_prediction >= threshold
        active_test_global = np.zeros(len(events.utility), dtype=bool)
        active_test_global[np.flatnonzero(test_events)[active_test_local]] = True
        coefficient_order = np.argsort(-np.abs(model.coefficients))
        views[name] = {
            "feature_names": list(names),
            "validation_threshold": threshold,
            "test_activation_fraction": float(active_test_local.mean()),
            "test_ndcg@10": _policy_ndcg(
                text,
                visual,
                qrels,
                events,
                test_queries,
                active_test_global,
            ),
            "test_diagnostics": _prediction_diagnostics(
                events.utility[test_events],
                test_prediction,
            ),
            "standardized_coefficients": [
                {
                    "feature": names[index],
                    "coefficient": float(model.coefficients[index]),
                }
                for index in coefficient_order
            ],
        }

    return {
        "schema_version": 1,
        "contract": {
            "query_split": "sha256(query_id) byte modulo 10: 60/20/20",
            "labels": "exact single-page delta nDCG@10; targets only",
            "candidate_source": f"text top-{events.candidate_k}",
            "route_source": "label-free clustering of text score profiles",
            "model": "weighted standardized ridge; fixed alpha=1 and nonzero weight=8",
            "fusion": "raw visual-score replacement for activated pages",
        },
        "workload": {
            "queries": int(text.scores.shape[0]),
            "corpus": int(text.scores.shape[1]),
            "events": int(len(events.utility)),
            "positive_events": int(np.count_nonzero(events.utility > 1e-12)),
            "harmful_events": int(np.count_nonzero(events.utility < -1e-12)),
            "zero_events": int(np.count_nonzero(np.abs(events.utility) <= 1e-12)),
            "query_split_counts": {
                name: int(mask.sum()) for name, mask in query_masks.items()
            },
        },
        "test_baselines": {
            "text_ndcg@10": text_test,
            "full_visual_ndcg@10": visual_test,
            f"activate_all_text_top{events.candidate_k}_ndcg@10": activate_all_test,
        },
        "feature_views": views,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-trace", type=Path, required=True)
    parser.add_argument("--visual-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--route-count", type=int, default=8)
    args = parser.parse_args()

    text = load_trace(args.text_trace)
    visual = load_trace(args.visual_trace)
    events = build_intervention_events(
        text,
        visual,
        candidate_k=args.candidate_k,
        cutoff=args.cutoff,
        route_count=args.route_count,
    )
    result = analyze_feature_views(text, visual, events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
