#!/usr/bin/env python3
"""Evaluate query-scope closure materialization against defer/full endpoints.

Every candidate in a query cohort is scored with the same high-fidelity visual
representation.  Persistent pages avoid repeated construction; missing pages
are constructed transiently, preserving the candidate rerank exactly.  The
experiment asks whether workload reuse creates a region where this closed
partial index costs less than both always-defer and full ingestion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.partial_vlm_materialization import ScoreSurface, evaluate_text_only
from tools.evaluate_value_aware_materialization import load_exported_surface, load_split


def load_ranking(path: Path, surface: ScoreSurface) -> np.ndarray:
    by_query: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        query_id, doc_id, _ = line.split("\t")
        by_query.setdefault(query_id, []).append(doc_id)
    depth = len(next(iter(by_query.values())))
    if set(by_query) != set(map(str, surface.query_ids)):
        raise ValueError("ranking query IDs do not match score surface")
    positions = {str(doc_id): index for index, doc_id in enumerate(surface.corpus_ids)}
    rows = []
    for query_id in surface.query_ids:
        docs = by_query[str(query_id)]
        if len(docs) != depth or len(set(docs)) != depth:
            raise ValueError("ranking depth is inconsistent or contains duplicates")
        rows.append([positions[doc_id] for doc_id in docs])
    return np.asarray(rows, dtype=np.int32)


def load_scored_ranking(path: Path, surface: ScoreSurface) -> tuple[np.ndarray, np.ndarray]:
    by_query: dict[str, list[tuple[str, float]]] = {}
    for line in path.read_text().splitlines():
        query_id, doc_id, score = line.split("\t")
        by_query.setdefault(query_id, []).append((doc_id, float(score)))
    depth = len(next(iter(by_query.values())))
    if set(by_query) != set(map(str, surface.query_ids)):
        raise ValueError("locator query IDs do not match score surface")
    positions = {str(doc_id): index for index, doc_id in enumerate(surface.corpus_ids)}
    orders, scores = [], []
    for query_id in surface.query_ids:
        rows = by_query[str(query_id)]
        if len(rows) != depth or len({doc_id for doc_id, _ in rows}) != depth:
            raise ValueError("locator depth is inconsistent or contains duplicates")
        orders.append([positions[doc_id] for doc_id, _ in rows])
        scores.append([score for _, score in rows])
    return np.asarray(orders, dtype=np.int32), np.asarray(scores, dtype=np.float64)


def _ndcg(surface: ScoreSurface, queries: Sequence[int], rankings: np.ndarray) -> dict[str, Any]:
    values = []
    for query in map(int, queries):
        relevance = surface.qrels[query, rankings[query, :10]]
        discounts = np.log2(np.arange(2, len(relevance) + 2, dtype=np.float64))
        dcg = float(np.sum((np.power(2.0, relevance) - 1.0) / discounts))
        ideal = float(surface.idcg_at_10[query])
        values.append(dcg / ideal if ideal > 0.0 else 0.0)
    return {
        "queries": len(values),
        "mean_ndcg_at_10": float(np.mean(values)),
        "per_query_ndcg_at_10": values,
    }


def bm25_full_rerank(surface: ScoreSurface, depth: int) -> np.ndarray:
    rows = []
    for query in range(surface.queries):
        candidates = surface.text_order[query, :depth]
        scores = surface.visual_scores[query, candidates]
        rows.append(candidates[np.lexsort((candidates, -scores))])
    return np.asarray(rows, dtype=np.int32)


def candidate_rerank(
    surface: ScoreSurface,
    candidates: np.ndarray,
    locator_scores: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    rows = []
    for query in range(surface.queries):
        pages = candidates[query]
        visual = np.asarray(surface.visual_scores[query, pages], dtype=np.float64)
        locator = np.asarray(locator_scores[query], dtype=np.float64)
        if method == "visual":
            fused = visual
        elif method == "zscore_sum":
            locator_scale = max(float(locator.std()), 1e-12)
            visual_scale = max(float(visual.std()), 1e-12)
            fused = (locator - locator.mean()) / locator_scale + (
                visual - visual.mean()
            ) / visual_scale
        elif method == "rrf":
            visual_order = np.lexsort((pages, -visual))
            visual_rank = np.empty(len(pages), dtype=np.int32)
            visual_rank[visual_order] = np.arange(1, len(pages) + 1)
            fused = 1.0 / (60.0 + np.arange(1, len(pages) + 1)) + 1.0 / (
                60.0 + visual_rank
            )
        else:
            raise ValueError(f"unsupported candidate fusion: {method}")
        rows.append(pages[np.lexsort((pages, -fused))])
    return np.asarray(rows, dtype=np.int32)


def _frequency(candidates: np.ndarray, queries: Sequence[int], pages: int) -> np.ndarray:
    counts = np.zeros(pages, dtype=np.int64)
    for query in map(int, queries):
        counts[candidates[query]] += 1
    return counts


def _top(values: np.ndarray, count: int) -> np.ndarray:
    pages = np.arange(len(values), dtype=np.int32)
    return pages[np.lexsort((pages, -values))[:count]]


def _phase_diagram(
    misses: np.ndarray,
    *,
    persistent_pages: int,
    candidate_depth: int,
    full_pages: int,
    maximum_repeat: int,
) -> dict[str, Any]:
    repeated_misses = np.tile(misses, maximum_repeat)
    cumulative_misses = np.cumsum(repeated_misses, dtype=np.int64)
    winner = []
    tie_priority = {"defer": 0, "closure_materialization": 1, "full_ingestion": 2}
    for prefix, missing in enumerate(cumulative_misses, start=1):
        costs = {
            "defer": prefix * candidate_depth,
            "closure_materialization": persistent_pages + int(missing),
            "full_ingestion": int(full_pages),
        }
        winner.append(min(costs, key=lambda name: (costs[name], tie_priority[name])))
    intervals = []
    start = 1
    for index in range(1, len(winner) + 1):
        if index == len(winner) or winner[index] != winner[index - 1]:
            intervals.append(
                {
                    "start_query": start,
                    "end_query": index,
                    "winner": winner[index - 1],
                }
            )
            start = index + 1
    return {
        "maximum_trace_replays": maximum_repeat,
        "winner_intervals": intervals,
        "winner_query_counts": {
            name: winner.count(name)
            for name in ("defer", "closure_materialization", "full_ingestion")
        },
    }


def _cost(
    candidates: np.ndarray,
    queries: Sequence[int],
    persistent: Sequence[int],
    *,
    repeats: Sequence[int],
    full_pages: int,
) -> dict[str, Any]:
    selected = set(map(int, persistent))
    misses = np.asarray(
        [sum(int(page) not in selected for page in candidates[int(query)]) for query in queries],
        dtype=np.int32,
    )
    candidate_events = int(len(queries) * candidates.shape[1])
    output = {
        "persistent_pages": len(selected),
        "candidate_events_per_trace": candidate_events,
        "transient_build_events_per_trace": int(misses.sum()),
        "persistent_hit_fraction": (
            1.0 - float(misses.sum()) / candidate_events if candidate_events else 0.0
        ),
        "first_query_transient_pages": int(misses[0]) if len(misses) else 0,
        "p50_transient_pages_per_query": float(np.percentile(misses, 50)),
        "p95_transient_pages_per_query": float(np.percentile(misses, 95)),
        "max_transient_pages_per_query": int(misses.max()) if len(misses) else 0,
        "trace_replays": {},
    }
    for repeat in repeats:
        output["trace_replays"][str(repeat)] = {
            "total_visual_page_builds": len(selected) + int(repeat) * int(misses.sum()),
            "dvi_always_defer_page_builds": int(repeat) * candidate_events,
        }
    maximum_repeat = max(map(int, repeats))
    output["phase_diagram"] = _phase_diagram(
        misses,
        persistent_pages=len(selected),
        candidate_depth=candidates.shape[1],
        full_pages=full_pages,
        maximum_repeat=maximum_repeat,
    )
    output["phase_diagram_random_orders"] = []
    for permutation_seed in range(5):
        permuted = np.random.default_rng(20260808 + permutation_seed).permutation(misses)
        output["phase_diagram_random_orders"].append(
            {
                "seed": 20260808 + permutation_seed,
                **_phase_diagram(
                    permuted,
                    persistent_pages=len(selected),
                    candidate_depth=candidates.shape[1],
                    full_pages=full_pages,
                    maximum_repeat=maximum_repeat,
                ),
            }
        )
    return output


def run(
    surface: ScoreSurface,
    history: np.ndarray,
    evaluation: np.ndarray,
    rankings: dict[int, dict[str, np.ndarray]],
    *,
    budgets: Sequence[float],
    repeats: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "domain": surface.name,
        "pages": surface.pages,
        "history_queries": len(history),
        "evaluation_queries": len(evaluation),
        "text_only_evaluation": evaluate_text_only(surface, evaluation),
        "depths": {},
    }
    for depth, surfaces in sorted(rankings.items()):
        candidates = surfaces["candidates"]
        hpool_reranked = surfaces["visual"]
        history_frequency = _frequency(candidates, history, surface.pages)
        future_frequency = _frequency(candidates, evaluation, surface.pages)
        bm25_reranked = bm25_full_rerank(surface, depth)
        quality = {
            "bm25_locator_full_visual_rerank": _ndcg(surface, evaluation, bm25_reranked),
            "hpool_locator_only": _ndcg(surface, evaluation, candidates),
            "hpool_locator_full_visual_rerank": _ndcg(surface, evaluation, hpool_reranked),
            "hpool_full_visual_zscore_sum": _ndcg(
                surface, evaluation, surfaces["zscore_sum"]
            ),
            "hpool_full_visual_rrf": _ndcg(surface, evaluation, surfaces["rrf"]),
        }
        history_quality = {
            method: _ndcg(surface, history, ranking)["mean_ndcg_at_10"]
            for method, ranking in surfaces.items()
            if method != "locator_scores"
        }
        selected_fusion = max(history_quality, key=history_quality.get)
        random_order = np.random.default_rng(seed + depth).permutation(surface.pages)
        policies: dict[str, Any] = {}
        for fraction in budgets:
            count = min(surface.pages, int(math.ceil(float(fraction) * surface.pages)))
            selections = {
                "random": random_order[:count],
                "history_frequency": _top(history_frequency, count),
                "future_frequency_oracle": _top(future_frequency, count),
            }
            for name, selected in selections.items():
                policies.setdefault(name, {})[str(fraction)] = _cost(
                    candidates,
                    evaluation,
                    selected,
                    repeats=repeats,
                    full_pages=surface.pages,
                )
        future_union = set(map(int, candidates[evaluation].flat))
        history_union = set(map(int, candidates[history].flat))
        online_future_cost = len(future_union)
        history_then_future_cost = len(history_union | future_union)
        report["depths"][str(depth)] = {
            "quality": quality,
            "fusion_selected_on_history_only": selected_fusion,
            "history_fusion_quality": history_quality,
            "selected_fusion_evaluation_ndcg_at_10": _ndcg(
                surface, evaluation, surfaces[selected_fusion]
            )["mean_ndcg_at_10"],
            "quality_is_invariant_to_persistent_vs_transient_storage": True,
            "candidate_events_per_evaluation_trace": int(depth * len(evaluation)),
            "unique_history_candidate_pages": len(history_union),
            "unique_future_candidate_pages": len(future_union),
            "history_future_candidate_page_overlap": (
                len(history_union & future_union) / len(future_union) if future_union else None
            ),
            "endpoints": {
                "full_ingestion_page_builds": surface.pages,
                "dvi_always_defer_page_builds_per_trace": int(depth * len(evaluation)),
                "persist_on_first_future_touch_page_builds": online_future_cost,
                "persist_through_history_and_future_page_builds": history_then_future_cost,
            },
            "policies": policies,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--query-splits", type=Path, required=True)
    parser.add_argument("--cascade", type=Path, action="append", default=[])
    parser.add_argument("--locator-ranking", type=Path)
    parser.add_argument("--depths", type=int, nargs="+", default=(20, 50, 100))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.02, 0.05, 0.1, 0.2, 0.4))
    parser.add_argument("--repeats", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    surface, _, _ = load_exported_surface(args.score_root, args.dataset_root)
    history, evaluation = load_split(args.query_splits, surface)
    rankings: dict[int, dict[str, np.ndarray]] = {}
    if args.locator_ranking is not None:
        locator_order, locator_scores = load_scored_ranking(args.locator_ranking, surface)
        for depth in args.depths:
            if depth > locator_order.shape[1]:
                raise ValueError("requested depth exceeds locator ranking")
            candidates = locator_order[:, :depth]
            scores = locator_scores[:, :depth]
            rankings[int(depth)] = {
                "candidates": candidates,
                "locator_scores": scores,
                "visual": candidate_rerank(surface, candidates, scores, method="visual"),
                "zscore_sum": candidate_rerank(
                    surface, candidates, scores, method="zscore_sum"
                ),
                "rrf": candidate_rerank(surface, candidates, scores, method="rrf"),
            }
    else:
        for path in args.cascade:
            ranking = load_ranking(path, surface)
            rankings[ranking.shape[1]] = {
                "candidates": ranking,
                "visual": ranking,
                "zscore_sum": ranking,
                "rrf": ranking,
            }
    if not rankings:
        raise ValueError("provide --locator-ranking or at least one --cascade")
    result = run(
        surface,
        history,
        evaluation,
        rankings,
        budgets=args.budgets,
        repeats=args.repeats,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"domain": result["domain"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
