#!/usr/bin/env python3
"""Measure time-integrated evidence quality for an online cohort compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.candidate_fusion import _candidate_ndcg
from reprforge.intervention_utility import _ndcg_row
from reprforge.progressive_oracle import load_trace, rank_order, validate_pair
from reprforge.progressive_quality_metrics import progressive_quality_timeline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-trace", type=Path, required=True)
    parser.add_argument("--visual-trace", type=Path, required=True)
    parser.add_argument("--online-raw", type=Path, required=True)
    parser.add_argument("--full-raw", type=Path, required=True)
    parser.add_argument(
        "--query-order-manifest",
        type=Path,
        help="required when the online run used a scheduled query permutation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()

    text = load_trace(args.bm25_trace)
    visual = load_trace(args.visual_trace)
    qrels = validate_pair(text, visual)
    online = json.loads(args.online_raw.read_text())
    full_raw = json.loads(args.full_raw.read_text())
    aggregate = online["aggregated_metrics"]
    info = aggregate["infos"]
    timing = aggregate["timing"]
    batch_trace = info.get("batch_trace") or []
    if not batch_trace:
        raise ValueError("online raw result has no batch trace")
    if int(timing["num_queries"]) != len(text.query_ids):
        raise ValueError("online result and score trace query counts differ")
    order = rank_order(text.scores, text.corpus_ids)
    base_quality = np.asarray(
        [
            _ndcg_row(text.scores[index], qrels[index], text.corpus_ids, cutoff=10)
            for index in range(len(text.query_ids))
        ]
    )
    full_quality = np.asarray(
        [
            _ndcg_row(visual.scores[index], qrels[index], visual.corpus_ids, cutoff=10)
            for index in range(len(visual.query_ids))
        ]
    )
    refined_quality = _candidate_ndcg(
        text,
        visual,
        qrels,
        order,
        candidate_k=args.candidate_k,
        method="zscore_sum",
        cutoff=10,
    )
    online_order = online.get("query_order")
    if online_order is not None:
        if args.query_order_manifest is None:
            raise ValueError("scheduled online run requires --query-order-manifest")
        if _sha256(args.query_order_manifest) != online_order["sha256"]:
            raise ValueError("query-order manifest digest differs from online run")
        order_payload = json.loads(args.query_order_manifest.read_text())
        order_ids = [str(value) for value in order_payload["query_ids"]]
        position = {str(query_id): index for index, query_id in enumerate(text.query_ids)}
        if len(order_ids) != len(position) or set(order_ids) != set(position):
            raise ValueError("query-order manifest is not an exact trace permutation")
        indices = np.asarray([position[query_id] for query_id in order_ids])
        base_quality = base_quality[indices]
        full_quality = full_quality[indices]
        refined_quality = refined_quality[indices]
    elif args.query_order_manifest is not None:
        raise ValueError("query-order manifest supplied for an unscheduled online run")
    base_ready_ms = float(timing["indexing_time_milliseconds"])
    # Compare within the wall-clock horizon at which full visual prebuild and
    # its complete retrieval run have finished.
    full_timing = full_raw["aggregated_metrics"]["timing"]
    if int(full_timing["num_queries"]) != len(text.query_ids):
        raise ValueError("full raw result and score trace query counts differ")
    horizon_ms = float(full_timing["total_retrieval_time_milliseconds"])
    target = max(float(base_quality.mean()), float(full_quality.mean()))
    timeline = progressive_quality_timeline(
        base_quality,
        refined_quality,
        batch_trace,
        base_ready_ms=base_ready_ms,
        horizon_ms=horizon_ms,
        target_quality=target,
        progress_reference_quality=target,
    )
    full_prebuild_mean_over_horizon = float(
        base_quality.mean() * max(0.0, horizon_ms - base_ready_ms) / horizon_ms
    )
    report = {
        "schema_version": 1,
        "stage": "post-hoc-time-integrated-evidence-quality",
        "dataset": online["dataset"],
        "metric": "per-query nDCG@10; base result is revised atomically at batch completion",
        "candidate_k": args.candidate_k,
        "queries": len(text.query_ids),
        "baselines": {
            "bm25_mean_ndcg_at_10": float(base_quality.mean()),
            "full_visual_mean_ndcg_at_10": float(full_quality.mean()),
            "refined_fusion_mean_ndcg_at_10": float(refined_quality.mean()),
            "full_visual_end_to_end_horizon_ms": horizon_ms,
            "full_prebuild_mean_quality_over_horizon": full_prebuild_mean_over_horizon,
        },
        "progressive": timeline,
        "comparison": {
            "mean_quality_over_horizon_gain_vs_full_prebuild": (
                timeline["mean_quality_over_horizon"] - full_prebuild_mean_over_horizon
            ),
            "relative_mean_quality_over_horizon_gain_vs_full_prebuild": (
                timeline["mean_quality_over_horizon"] / full_prebuild_mean_over_horizon - 1.0
                if full_prebuild_mean_over_horizon
                else None
            ),
        },
        "validity": {
            "uses_qrels_for_post_hoc_metric_only": True,
            "query_order_is_official_not_natural_time": online_order is None,
            "query_order_is_qrel_free_scheduled_batch": online_order is not None,
            "answer_generation_evaluated": False,
            "full_prebuild_serves_bm25_while_visual_index_builds": True,
        },
        "artifacts": {
            "bm25_runtime_sha256": text.manifest["runtime_sha256"],
            "visual_runtime_sha256": visual.manifest["runtime_sha256"],
            "online_raw_sha256": _sha256(args.online_raw),
            "full_raw_sha256": _sha256(args.full_raw),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
