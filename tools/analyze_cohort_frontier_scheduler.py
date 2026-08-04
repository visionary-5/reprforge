#!/usr/bin/env python3
"""Exact frozen-trace work replay for qrel-free cohort schedulers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reprforge.candidate_fusion import _candidate_ndcg
from reprforge.cohort_frontier_scheduler import (
    frontier_reuse_order,
    replay_page_work,
    static_popularity_order,
)
from reprforge.intervention_utility import _ndcg_row
from reprforge.progressive_oracle import load_trace, rank_order, validate_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-trace", type=Path, required=True)
    parser.add_argument("--visual-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-seeds", type=int, default=10)
    args = parser.parse_args()

    text = load_trace(args.bm25_trace)
    visual = load_trace(args.visual_trace)
    qrels = validate_pair(text, visual)
    candidate_order = rank_order(text.scores, text.corpus_ids)[:, : args.candidate_k]
    cohorts = [row.tolist() for row in candidate_order]
    base_quality = np.asarray(
        [
            _ndcg_row(text.scores[index], qrels[index], text.corpus_ids, cutoff=10)
            for index in range(len(text.query_ids))
        ]
    )
    refined_quality = _candidate_ndcg(
        text,
        visual,
        qrels,
        rank_order(text.scores, text.corpus_ids),
        candidate_k=args.candidate_k,
        method="zscore_sum",
        cutoff=10,
    )
    gains = refined_quality - base_quality
    common = {
        "quality_gain": gains,
        "base_mean_quality": float(base_quality.mean()),
        "batch_size": args.batch_size,
        "corpus_pages": len(text.corpus_ids),
    }
    orders = {
        "fifo": list(range(len(cohorts))),
        "static_popularity": static_popularity_order(cohorts),
        "frontier_reuse": frontier_reuse_order(cohorts, batch_size=args.batch_size),
    }
    schedules = {
        name: replay_page_work(cohorts, order, **common)
        for name, order in orders.items()
    }
    random_reports = []
    for seed in range(args.random_seeds):
        order = np.random.default_rng(20260804 + seed).permutation(len(cohorts))
        random_reports.append(replay_page_work(cohorts, order, **common))
    random_auc = np.asarray(
        [value["mean_quality_over_full_build_work"] for value in random_reports]
    )
    random_mean_completion = np.asarray(
        [value["completion_pages"]["mean"] for value in random_reports]
    )
    report = {
        "schema_version": 1,
        "stage": "frozen-exact-page-work-replay",
        "dataset": args.dataset_name,
        "queries": len(cohorts),
        "corpus_pages": len(text.corpus_ids),
        "candidate_k": args.candidate_k,
        "batch_size": args.batch_size,
        "scheduler_uses_qrels": False,
        "qrels_used_for_post_hoc_quality_curve_only": True,
        "base_mean_ndcg_at_10": float(base_quality.mean()),
        "refined_mean_ndcg_at_10": float(refined_quality.mean()),
        "schedules": schedules,
        "random": {
            "seeds": args.random_seeds,
            "quality_work_auc_mean": float(random_auc.mean()),
            "quality_work_auc_min": float(random_auc.min()),
            "quality_work_auc_max": float(random_auc.max()),
            "mean_completion_pages_mean": float(random_mean_completion.mean()),
            "mean_completion_pages_min": float(random_mean_completion.min()),
            "mean_completion_pages_max": float(random_mean_completion.max()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
