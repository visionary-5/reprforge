#!/usr/bin/env python3
"""Replay strong cohort schedulers over frozen text/visual score traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.candidate_fusion import _zscore
from reprforge.cohort_frontier_scheduler import (
    frontier_reuse_order,
    replay_page_work,
    static_popularity_order,
)
from reprforge.intervention_utility import _ndcg_row
from reprforge.progressive_oracle import rank_order
from reprforge.scheduler_baselines import (
    POLICY_DESCRIPTIONS,
    offline_work_greedy_order,
    overlap_only_order,
    reuse_only_order,
    shortest_missing_order,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_arrays(root: Path) -> dict[str, np.ndarray]:
    if (root / "bm25" / "runtime.npz").is_file():
        paths = {
            "text": root / "bm25" / "runtime.npz",
            "visual": root / "visual" / "runtime.npz",
            "labels": root / "bm25" / "oracle-labels.npz",
        }
    else:
        # Backward-compatible layout used by the legacy heterogeneity atlas.
        paths = {
            "text": root / "text-runtime.npz",
            "visual": root / "visual-runtime.npz",
            "labels": root / "oracle-labels.npz",
        }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(paths["text"], allow_pickle=False) as source:
        text = {key: source[key] for key in source.files}
    with np.load(paths["visual"], allow_pickle=False) as source:
        visual = {key: source[key] for key in source.files}
    with np.load(paths["labels"], allow_pickle=False) as source:
        labels = {key: source[key] for key in source.files}
    if not np.array_equal(text["query_ids"], visual["query_ids"]):
        raise ValueError("text and visual query identifiers differ")
    if not np.array_equal(text["corpus_ids"], visual["corpus_ids"]):
        raise ValueError("text and visual corpus identifiers differ")
    if text["scores"].shape != visual["scores"].shape:
        raise ValueError("text and visual score shapes differ")
    qrels = np.zeros(text["scores"].shape, dtype=np.int16)
    qrels[labels["query_positions"], labels["corpus_positions"]] = labels[
        "relevance"
    ]
    if np.any(qrels.max(axis=1) == 0):
        raise ValueError("at least one query has no positive qrel")
    return {
        "query_ids": text["query_ids"],
        "corpus_ids": text["corpus_ids"],
        "text_scores": np.asarray(text["scores"], dtype=np.float64),
        "visual_scores": np.asarray(visual["scores"], dtype=np.float64),
        "qrels": qrels,
        "source_paths": paths,
    }


def per_query_quality(
    arrays: dict[str, Any], text_order: np.ndarray, candidate_k: int
) -> tuple[np.ndarray, np.ndarray]:
    text_scores = arrays["text_scores"]
    visual_scores = arrays["visual_scores"]
    qrels = arrays["qrels"]
    corpus_ids = arrays["corpus_ids"]
    count = len(text_scores)
    base = np.empty(count, dtype=np.float64)
    refined = np.empty(count, dtype=np.float64)
    for query in range(count):
        base[query] = _ndcg_row(
            text_scores[query], qrels[query], corpus_ids, cutoff=10
        )
        pages = text_order[query, :candidate_k]
        fused = _zscore(text_scores[query, pages]) + _zscore(
            visual_scores[query, pages]
        )
        scores = np.full(text_scores.shape[1], -np.inf, dtype=np.float64)
        scores[pages] = fused
        refined[query] = _ndcg_row(scores, qrels[query], corpus_ids, cutoff=10)
    return base, refined


def replay_configuration(
    arrays: dict[str, Any],
    *,
    candidate_k: int,
    batch_size: int,
    random_seeds: int,
) -> dict[str, Any]:
    text_order = rank_order(arrays["text_scores"], arrays["corpus_ids"])
    cohorts = [row[:candidate_k].tolist() for row in text_order]
    base, refined = per_query_quality(arrays, text_order, candidate_k)
    gains = refined - base
    common = {
        "quality_gain": gains,
        "base_mean_quality": float(base.mean()),
        "batch_size": batch_size,
        "corpus_pages": len(arrays["corpus_ids"]),
    }
    orders = {
        "fifo": list(range(len(cohorts))),
        "static_popularity": static_popularity_order(cohorts),
        "overlap_only": overlap_only_order(cohorts, batch_size=batch_size),
        "shortest_missing": shortest_missing_order(
            cohorts, batch_size=batch_size
        ),
        "reuse_only": reuse_only_order(cohorts, batch_size=batch_size),
        "frontier_reuse": frontier_reuse_order(cohorts, batch_size=batch_size),
        "offline_work_greedy": offline_work_greedy_order(
            cohorts, batch_size=batch_size
        ),
    }
    schedules = {}
    expected_union = len(set().union(*(set(cohort) for cohort in cohorts)))
    for name, order in orders.items():
        report = replay_page_work(cohorts, order, **common)
        if report["final_unique_pages"] != expected_union:
            raise AssertionError(f"{name} changed final candidate union")
        points = report.pop("points")
        report["points_sha256"] = hashlib.sha256(
            json.dumps(points, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report["order_sha256"] = hashlib.sha256(
            np.asarray(order, dtype=np.int32).tobytes()
        ).hexdigest()
        schedules[name] = report

    random_reports = []
    for seed in range(random_seeds):
        order = np.random.default_rng(20260804 + seed).permutation(len(cohorts))
        random_reports.append(replay_page_work(cohorts, order, **common))
    random_completion = np.asarray(
        [row["completion_pages"]["mean"] for row in random_reports]
    )
    random_quality = np.asarray(
        [row["mean_quality_over_full_build_work"] for row in random_reports]
    )
    return {
        "candidate_k": candidate_k,
        "batch_size": batch_size,
        "queries": len(cohorts),
        "corpus_pages": len(arrays["corpus_ids"]),
        "candidate_union_pages": expected_union,
        "base_mean_ndcg_at_10": float(base.mean()),
        "refined_mean_ndcg_at_10": float(refined.mean()),
        "schedules": schedules,
        "random": {
            "seeds": random_seeds,
            "mean_completion_pages": {
                "mean": float(random_completion.mean()),
                "min": float(random_completion.min()),
                "max": float(random_completion.max()),
            },
            "quality_work_auc": {
                "mean": float(random_quality.mean()),
                "min": float(random_quality.min()),
                "max": float(random_quality.max()),
            },
        },
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    primary = next(
        row
        for row in report["configurations"]
        if row["candidate_k"] == 20 and row["batch_size"] == 8
    )
    schedules = primary["schedules"]
    frontier = schedules["frontier_reuse"]
    simple = ("static_popularity", "overlap_only", "reuse_only")
    strongest_simple = min(
        simple, key=lambda name: schedules[name]["completion_pages"]["mean"]
    )
    dominated_including_offline = []
    dominated_by_non_oracle = []
    frontier_wins = 0
    for row in report["configurations"]:
        fs = row["schedules"]["frontier_reuse"]
        fwork = fs["completion_pages"]["mean"]
        fauc = fs["mean_quality_over_full_build_work"]
        is_dominated = False
        is_dominated_non_oracle = False
        for name, candidate in row["schedules"].items():
            if name == "frontier_reuse":
                continue
            work = candidate["completion_pages"]["mean"]
            auc = candidate["mean_quality_over_full_build_work"]
            if work <= fwork and auc >= fauc and (work < fwork or auc > fauc):
                is_dominated = True
                if name != "offline_work_greedy":
                    is_dominated_non_oracle = True
        dominated_including_offline.append(is_dominated)
        dominated_by_non_oracle.append(is_dominated_non_oracle)
        best_simple_work = min(
            row["schedules"][name]["completion_pages"]["mean"]
            for name in simple
        )
        frontier_wins += int(fwork < best_simple_work)
    return {
        "primary_k20_batch8": {
            "frontier_mean_completion_pages": frontier["completion_pages"]["mean"],
            "frontier_quality_work_auc": frontier[
                "mean_quality_over_full_build_work"
            ],
            "strongest_simple_by_work": strongest_simple,
            "strongest_simple_mean_completion_pages": schedules[strongest_simple][
                "completion_pages"
            ]["mean"],
            "frontier_beats_all_popularity_or_overlap_explanations": all(
                frontier["completion_pages"]["mean"]
                < schedules[name]["completion_pages"]["mean"]
                for name in simple
            ),
            "frontier_beats_shortest_missing": (
                frontier["completion_pages"]["mean"]
                < schedules["shortest_missing"]["completion_pages"]["mean"]
            ),
            "offline_greedy_gap_fraction": (
                frontier["completion_pages"]["mean"]
                / schedules["offline_work_greedy"]["completion_pages"]["mean"]
                - 1.0
            ),
        },
        "grid": {
            "configurations": len(report["configurations"]),
            "frontier_not_dominated_by_non_oracle": (
                len(dominated_by_non_oracle) - sum(dominated_by_non_oracle)
            ),
            "frontier_not_dominated_including_offline_greedy": (
                len(dominated_including_offline)
                - sum(dominated_including_offline)
            ),
            "frontier_beats_all_simple_by_work": frontier_wins,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--random-seeds", type=int, default=10)
    args = parser.parse_args()
    if args.random_seeds <= 0:
        parser.error("--random-seeds must be positive")
    arrays = load_frozen_arrays(args.trace_root)
    configs = [
        replay_configuration(
            arrays,
            candidate_k=k,
            batch_size=batch,
            random_seeds=args.random_seeds,
        )
        for k in sorted(set(args.candidate_k))
        for batch in sorted(set(args.batch_size))
    ]
    report = {
        "schema_version": 1,
        "stage": "strong-qrel-free-scheduler-frozen-replay",
        "dataset": args.dataset_name,
        "scheduler_observes": [
            "BM25 candidate membership",
            "resident and current staged page membership",
            "queued-query candidate frequency for explicitly offline policies",
        ],
        "scheduler_forbidden": ["qrels", "visual scores", "answer outcomes"],
        "qrels_used_for_post_hoc_quality_metrics_only": True,
        "policy_descriptions": POLICY_DESCRIPTIONS,
        "provenance": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in arrays["source_paths"].items()
        },
        "configurations": configs,
    }
    report["summary"] = summarize(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
