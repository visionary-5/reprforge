#!/usr/bin/env python3
"""Explain how representation routes change a retrieval ranking.

The analysis uses exact one-item interventions around a uniform base route.
Unlike a single document-level metric delta, it separates two causal channels:

* evidence: changing the representation of a relevant item;
* distraction: changing the representation of an irrelevant item.

Route scores and relevance labels are used only for offline diagnosis.  They
are not deployable planner features.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reprforge.policy_replay import Query, ReplayData, load_replay_data


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def subset_by_documents(
    data: ReplayData,
    *,
    query_rows: Sequence[Mapping],
    item_rows: Sequence[Mapping],
    document_indices: set[int],
) -> tuple[ReplayData, list[dict]]:
    query_documents = {
        str(row["query_id"]): int(row["document_index"]) for row in query_rows
    }
    selected_queries = tuple(
        query
        for query in data.queries
        if query_documents[query.query_id] in document_indices
    )
    selected_item_ids = {
        item_id
        for query in selected_queries
        for item_id in (
            query.candidate_item_ids
            if query.candidate_item_ids is not None
            else ()
        )
    }
    if not selected_item_ids:
        raise ValueError("selected documents contain no candidate items")
    selected = ReplayData(
        items=tuple(
            item for item in data.items if item.item_id in selected_item_ids
        ),
        queries=selected_queries,
        scores=data.scores,
    )
    selected.validate()
    return selected, [
        dict(row)
        for row in item_rows
        if str(row["item_id"]) in selected_item_ids
    ]


def _ndcg(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    dcg = sum(
        float(relevance.get(item_id, 0.0)) / math.log2(rank + 1)
        for rank, item_id in enumerate(ranked[:k], start=1)
        if item_id in relevance
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(
        value / math.log2(rank + 1)
        for rank, value in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg else 0.0


def _recall(ranked: Sequence[str], query: Query, k: int) -> float:
    numerator = sum(
        query.relevance.get(item_id, 0.0) for item_id in ranked[:k]
    )
    denominator = (
        query.relevance_denominator
        if query.relevance_denominator is not None
        else sum(query.relevance.values())
    )
    return numerator / denominator


def _metric(ranked: Sequence[str], query: Query, metric: str, k: int) -> float:
    if metric == "ndcg":
        return _ndcg(ranked, query.relevance, k)
    if metric == "recall":
        return _recall(ranked, query, k)
    raise ValueError(f"unsupported metric: {metric}")


def _rank(
    item_ids: Sequence[str],
    scores: Mapping[str, float],
) -> list[str]:
    return sorted(item_ids, key=lambda item_id: (-scores[item_id], item_id))


def intervention_rows(
    data: ReplayData,
    *,
    base_route: str,
    metric: str = "ndcg",
    k: int = 10,
) -> list[dict]:
    """Return exact query/item/route interventions around ``base_route``."""

    if base_route not in data.routes:
        raise ValueError(f"unknown base route: {base_route}")
    if k <= 0:
        raise ValueError("k must be positive")
    all_item_ids = tuple(item.item_id for item in data.items)
    rows: list[dict] = []
    for query in data.queries:
        item_ids = (
            query.candidate_item_ids
            if query.candidate_item_ids is not None
            else all_item_ids
        )
        base_scores = {
            item_id: data.scores[base_route][query.query_id][item_id]
            for item_id in item_ids
        }
        base_ranking = _rank(item_ids, base_scores)
        base_positions = {
            item_id: position
            for position, item_id in enumerate(base_ranking, start=1)
        }
        base_value = _metric(base_ranking, query, metric, k)
        for route in data.routes:
            if route == base_route:
                continue
            for item_id in item_ids:
                route_score = data.scores[route][query.query_id][item_id]
                if route_score == base_scores[item_id]:
                    changed_ranking = base_ranking
                else:
                    changed_scores = dict(base_scores)
                    changed_scores[item_id] = route_score
                    changed_ranking = _rank(item_ids, changed_scores)
                changed_position = changed_ranking.index(item_id) + 1
                delta = _metric(changed_ranking, query, metric, k) - base_value
                relevant = item_id in query.relevance
                if relevant:
                    channel = (
                        "evidence_recovery"
                        if delta > 0
                        else "evidence_loss" if delta < 0 else "no_effect"
                    )
                else:
                    channel = (
                        "distractor_suppression"
                        if delta > 0
                        else "distractor_inflation" if delta < 0 else "no_effect"
                    )
                rows.append(
                    {
                        "query_id": query.query_id,
                        "item_id": item_id,
                        "route": route,
                        "relevant": relevant,
                        "score_delta": route_score - base_scores[item_id],
                        "base_rank": base_positions[item_id],
                        "route_rank": changed_position,
                        "entered_top_k": (
                            base_positions[item_id] > k
                            and changed_position <= k
                        ),
                        "left_top_k": (
                            base_positions[item_id] <= k
                            and changed_position > k
                        ),
                        "metric_delta": delta,
                        "channel": channel,
                    }
                )
    return rows


def aggregate_item_routes(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["item_id"]), str(row["route"]))].append(row)
    aggregates = []
    for (item_id, route), values in sorted(grouped.items()):
        evidence = [
            float(row["metric_delta"]) for row in values if row["relevant"]
        ]
        distraction = [
            float(row["metric_delta"]) for row in values if not row["relevant"]
        ]
        deltas = [float(row["metric_delta"]) for row in values]
        aggregates.append(
            {
                "item_id": item_id,
                "route": route,
                "queries": len(values),
                "impact_probability": (
                    sum(abs(value) > 1e-12 for value in deltas) / len(deltas)
                ),
                "signed_metric_delta_sum": sum(deltas),
                "evidence_delta_sum": sum(evidence),
                "distractor_delta_sum": sum(distraction),
                "distractor_harm_sum": -sum(
                    min(value, 0.0) for value in distraction
                ),
                "mean_score_delta": sum(
                    float(row["score_delta"]) for row in values
                )
                / len(values),
                "top_k_entries": sum(bool(row["entered_top_k"]) for row in values),
                "top_k_exits": sum(bool(row["left_top_k"]) for row in values),
            }
        )
    return aggregates


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if left_rank.std() < 1e-12 or right_rank.std() < 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def diagnostic_correlations(
    aggregates: Sequence[Mapping],
    item_rows: Sequence[Mapping],
) -> list[dict]:
    """Relate route diagnostics to causal intervention outcomes."""

    items = {str(row["item_id"]): row for row in item_rows}
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    outcomes = (
        "impact_probability",
        "signed_metric_delta_sum",
        "evidence_delta_sum",
        "distractor_harm_sum",
    )
    excluded = {
        "vector_count",
        "embedding_dimension",
        "diagnostic_requires_candidate_embedding",
        "distortion_feature_ms",
    }
    for aggregate in aggregates:
        item = items.get(str(aggregate["item_id"]))
        if item is None:
            continue
        route = str(aggregate["route"])
        features = {
            f"construction:{name}": value
            for name, value in (
                item.get("construction_features") or {}
            ).items()
            if name != "feature_extract_ms"
        }
        features.update(
            {
                f"route:{name}": value
                for name, value in (
                    (item.get("route_features") or {}).get(route) or {}
                ).items()
            }
        )
        for feature, raw_value in features.items():
            if feature.removeprefix("route:") in excluded:
                continue
            for outcome in outcomes:
                grouped[(route, feature, outcome)].append(
                    (float(raw_value), float(aggregate[outcome]))
                )
    result = []
    for (route, feature, outcome), pairs in sorted(grouped.items()):
        correlation = _spearman(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
        )
        result.append(
            {
                "route": route,
                "feature": feature,
                "outcome": outcome,
                "items": len(pairs),
                "spearman": correlation,
            }
        )
    return result


def summarize(
    data: ReplayData,
    *,
    item_rows: Sequence[Mapping],
    base_route: str,
    metric: str,
    k: int,
) -> dict:
    rows = intervention_rows(
        data,
        base_route=base_route,
        metric=metric,
        k=k,
    )
    aggregates = aggregate_item_routes(rows)
    route_summary = {}
    item_content = {
        str(row["item_id"]): str(row.get("content_type") or "unknown")
        for row in item_rows
    }
    content_summary = {}
    for route in sorted({str(row["route"]) for row in rows}):
        selected = [row for row in rows if row["route"] == route]
        route_summary[route] = {
            "interventions": len(selected),
            "nonzero_interventions": sum(
                abs(float(row["metric_delta"])) > 1e-12 for row in selected
            ),
            "channel_counts": {
                channel: sum(row["channel"] == channel for row in selected)
                for channel in (
                    "evidence_recovery",
                    "evidence_loss",
                    "distractor_suppression",
                    "distractor_inflation",
                    "no_effect",
                )
            },
            "signed_metric_delta_sum": sum(
                float(row["metric_delta"]) for row in selected
            ),
        }
        for content_type in sorted(
            {item_content[str(row["item_id"])] for row in selected}
        ):
            content_rows = [
                row
                for row in selected
                if item_content[str(row["item_id"])] == content_type
            ]
            content_summary[f"{route}|{content_type}"] = {
                "route": route,
                "content_type": content_type,
                "interventions": len(content_rows),
                "nonzero_interventions": sum(
                    abs(float(row["metric_delta"])) > 1e-12
                    for row in content_rows
                ),
                "evidence_recovery": sum(
                    row["channel"] == "evidence_recovery"
                    for row in content_rows
                ),
                "evidence_loss": sum(
                    row["channel"] == "evidence_loss"
                    for row in content_rows
                ),
                "distractor_inflation": sum(
                    row["channel"] == "distractor_inflation"
                    for row in content_rows
                ),
                "distractor_suppression": sum(
                    row["channel"] == "distractor_suppression"
                    for row in content_rows
                ),
                "signed_metric_delta_sum": sum(
                    float(row["metric_delta"]) for row in content_rows
                ),
            }
    return {
        "contract": {
            "base_route": base_route,
            "metric": metric,
            "k": k,
            "quality_semantics": "official per-query candidate pools",
            "diagnostic_only": True,
            "warning": (
                "scores, labels, and candidate compressed embeddings are "
                "offline explanatory signals, not deployable planner inputs"
            ),
        },
        "corpus": {
            "items": len(data.items),
            "queries": len(data.queries),
            "routes": list(data.routes),
        },
        "route_summary": route_summary,
        "route_content_summary": content_summary,
        "item_route_aggregates": aggregates,
        "diagnostic_correlations": diagnostic_correlations(
            aggregates,
            item_rows,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--base-route", default="image-pool-9")
    parser.add_argument("--metric", choices=("ndcg", "recall"), default="ndcg")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--role", action="append", dest="roles")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    item_rows = _read_jsonl(args.result_dir / "items.jsonl")
    query_rows = _read_jsonl(args.result_dir / "queries.jsonl")
    data = load_replay_data(
        args.result_dir / "items.jsonl",
        args.result_dir / "queries.jsonl",
        args.result_dir / "scores.jsonl",
    )
    selected_roles = args.roles or []
    if bool(args.protocol) != bool(selected_roles):
        parser.error("--protocol and at least one --role must be used together")
    if args.protocol:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        known_roles = {str(row["role"]) for row in protocol["documents"]}
        unknown_roles = set(selected_roles) - known_roles
        if unknown_roles:
            parser.error(f"unknown protocol roles: {sorted(unknown_roles)}")
        documents = {
            int(row["document_index"])
            for row in protocol["documents"]
            if str(row["role"]) in selected_roles
        }
        data, item_rows = subset_by_documents(
            data,
            query_rows=query_rows,
            item_rows=item_rows,
            document_indices=documents,
        )
    result = summarize(
        data,
        item_rows=item_rows,
        base_route=args.base_route,
        metric=args.metric,
        k=args.k,
    )
    result["contract"]["selected_roles"] = selected_roles or ["all"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "corpus": result["corpus"],
                "route_summary": result["route_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
