#!/usr/bin/env python3
"""Run a bounded IRPAPERS transfer on one local GPU."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from reprforge.irpapers_benchmark import (
    IRPapersColPaliBackend,
    bm25_score_matrix,
    candidate_fusion_replay,
    full_fusion_results,
    load_irpapers,
    recall_at_k,
    score_rows_to_results,
)
from reprforge.vidore_pipeline import ReprForgeViDoRePipeline


OFFICIAL_REFERENCES = {
    "BM25": {"recall_1": 0.45, "recall_5": 0.71, "recall_20": 0.90},
    "ColPali": {"recall_1": 0.45, "recall_5": 0.79, "recall_20": 0.93},
    "ColQwen2": {"recall_1": 0.49, "recall_5": 0.81, "recall_20": 0.94},
    "open_static_hybrid": {
        "recall_1": 0.49,
        "recall_5": 0.81,
        "recall_20": 0.95,
    },
    "current_closed_hybrid": {
        "recall_1": 0.58,
        "recall_5": 0.91,
        "recall_20": 0.98,
    },
}
IRPAPERS_REVISION = "7d8ca2f6dd9efded3e27013d15782d584f93e9da"
COLPALI_V1_1_REVISION = "a0f15e3bcf97110e7ac1bb4be4bcd30eeb31992a"


def _record(
    *,
    metrics: Mapping[str, float],
    cost: Mapping[str, Any],
    semantics: str,
) -> dict[str, Any]:
    return {
        "quality": dict(metrics),
        "cost": dict(cost),
        "semantics": semantics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument("--request-batch-size", type=int, default=8)
    parser.add_argument("--actual-candidate-k", type=int, default=20)
    parser.add_argument("--replay-k", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument(
        "--skip-actual-resident",
        action="store_true",
        help="Run only BM25, full visual, static fusion and score replay.",
    )
    return parser.parse_args()


def _write_snapshot(
    path: Path,
    *,
    status: str,
    dataset: Mapping[str, Any],
    resource_contract: Mapping[str, Any],
    runs: Mapping[str, Any],
    score_surface: str | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        "dataset": dict(dataset),
        "resource_contract": dict(resource_contract),
        "official_reported_references": OFFICIAL_REFERENCES,
        "runs": dict(runs),
        "runtime_score_surface": score_surface,
        "comparison_warning": (
            "Official reported references use different implementations and, for the "
            "strong rows, different models. They provide a horizontal coordinate, not "
            "a controlled head-to-head speed comparison."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.scoring_batch_size <= 0:
        raise ValueError("GPU batch sizes must be positive")
    began = time.perf_counter()
    data = load_irpapers(args.docs, args.queries)
    load_seconds = time.perf_counter() - began

    began = time.perf_counter()
    locator_scores, locator_bytes = bm25_score_matrix(data)
    locator_seconds = time.perf_counter() - began
    bm25_results = score_rows_to_results(
        data.query_ids,
        data.corpus_ids,
        locator_scores,
        top_k=20,
    )
    runs: dict[str, Any] = {
        "local_bm25": _record(
            metrics=recall_at_k(bm25_results, data.qrels),
            cost={
                **locator_bytes,
                "build_and_score_seconds": locator_seconds,
                "visual_pages_encoded": 0,
            },
            semantics=(
                "ReprForge deterministic BM25 over the dataset-supplied GPT-4.1 "
                "transcription; not the official Weaviate BM25 implementation"
            ),
        )
    }

    model_began = time.perf_counter()
    backend = IRPapersColPaliBackend(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
    )
    model_load_seconds = time.perf_counter() - model_began
    import torch

    resource_contract = {
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cuda_visible_device_count": torch.cuda.device_count(),
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_load_seconds": model_load_seconds,
        "dataset_load_seconds": load_seconds,
        "batch_size": args.batch_size,
        "scoring_batch_size": args.scoring_batch_size,
        "request_batch_size": args.request_batch_size,
        "irpapers_revision": IRPAPERS_REVISION,
        "colpali_v1_1_revision": COLPALI_V1_1_REVISION,
    }

    visual = ReprForgeViDoRePipeline(
        base_model=str(args.base_model),
        adapter=str(args.adapter),
        mode="visual",
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
        top_k=20,
        capture_score_trace=True,
        backend_factory=lambda: backend,
    )
    visual.index(
        list(data.corpus_ids),
        list(data.corpus_images),
        list(data.corpus_texts),
        "IRPAPERS",
    )
    visual_results, visual_cost = visual.search(
        list(data.query_ids), list(data.queries)
    )
    visual_trace = visual.export_score_trace(data.query_ids)
    visual_scores = np.asarray(visual_trace["scores"], dtype=np.float32)
    runs["full_visual_colpali_v1_1"] = _record(
        metrics=recall_at_k(visual_results, data.qrels),
        cost={
            **visual_cost,
            "end_to_end_seconds_excluding_model_and_dataset": (
                visual_cost["measured_index_ms_inside_pipeline"]
                + visual_cost["query_encode_ms_inside_search"]
                + visual_cost["retrieval_and_materialization_ms_inside_search"]
            )
            / 1000.0,
        },
        semantics=(
            "Public ColPali-v1.1 checkpoint with local exhaustive MaxSim; it is the "
            "same model family as the official ColPali row, not a guarantee of the "
            "same checkpoint or serving implementation"
        ),
    )
    score_surface_path = args.output.with_name(
        f"{args.output.stem}-runtime-score-surface.npz"
    )
    np.savez_compressed(
        score_surface_path,
        query_ids=np.asarray(data.query_ids),
        corpus_ids=np.asarray(data.corpus_ids),
        bm25_scores=locator_scores,
        visual_scores=visual_scores,
    )

    hybrid_results = full_fusion_results(
        data.query_ids,
        data.corpus_ids,
        locator_scores,
        visual_scores,
        top_k=20,
    )
    runs["static_full_zscore_fusion"] = _record(
        metrics=recall_at_k(hybrid_results, data.qrels),
        cost={
            "visual_pages_encoded": len(data.corpus_ids),
            "visual_index_ms": visual_cost["measured_index_ms_inside_pipeline"],
            "visual_index_vector_bytes": visual_cost["index_vector_bytes"],
            "bm25_build_and_score_seconds": locator_seconds,
            "bm25_logical_index_bytes": locator_bytes["logical_index_bytes"],
            "combined_logical_index_bytes": (
                visual_cost["index_vector_bytes"]
                + locator_bytes["logical_index_bytes"]
            ),
            "reuses_full_visual_score_surface": True,
            "fusion_seconds_not_separately_profiled": True,
        },
        semantics=(
            "Per-query corpus-wide z(BM25)+z(ColPali-v1.1); a static complete-index "
            "reference, not the official Arctic2+BM25+ColModernVBERT hybrid"
        ),
    )

    for candidate_k in args.replay_k:
        replay_results, replay_cost = candidate_fusion_replay(
            data.query_ids,
            data.corpus_ids,
            locator_scores,
            visual_scores,
            candidate_k=candidate_k,
            top_k=20,
        )
        runs[f"candidate_fusion_replay_k{candidate_k}"] = _record(
            metrics=recall_at_k(replay_results, data.qrels),
            cost={
                **replay_cost,
                "uses_offline_complete_visual_scores": True,
                "timing_is_not_deployable": True,
            },
            semantics=(
                "Exact score-surface replay of BM25-cohort candidate-relative fusion; "
                "qrels are used only by the evaluator"
            ),
        )

    _write_snapshot(
        args.output,
        status="full-visual-and-replay-complete",
        dataset=data.metadata,
        resource_contract=resource_contract,
        runs=runs,
        score_surface=str(score_surface_path.resolve()),
    )

    del visual
    gc.collect()

    if not args.skip_actual_resident:
        resident = ReprForgeViDoRePipeline(
            base_model=str(args.base_model),
            adapter=str(args.adapter),
            mode="bm25-fusion-batched",
            device=args.device,
            batch_size=args.batch_size,
            scoring_batch_size=args.scoring_batch_size,
            candidate_k=args.actual_candidate_k,
            top_k=20,
            request_batch_size=args.request_batch_size,
            cohort_cache_policy="resident",
            backend_factory=lambda: backend,
        )
        resident.index(
            list(data.corpus_ids),
            list(data.corpus_images),
            list(data.corpus_texts),
            "IRPAPERS",
        )
        resident_results, resident_cost = resident.search(
            list(data.query_ids), list(data.queries)
        )
        runs[f"resident_compiler_k{args.actual_candidate_k}"] = _record(
            metrics=recall_at_k(resident_results, data.qrels),
            cost={
                **resident_cost,
                "end_to_end_seconds_excluding_model_and_dataset": (
                    resident_cost["measured_index_ms_inside_pipeline"]
                    + resident_cost["total_execution_ms"]
                )
                / 1000.0,
            },
            semantics=(
                "Actual online BM25 cohort compiler with batch-atomic resident visual "
                "state and candidate-relative z-score fusion"
            ),
        )

    _write_snapshot(
        args.output,
        status="complete",
        dataset=data.metadata,
        resource_contract=resource_contract,
        runs=runs,
        score_surface=str(score_surface_path.resolve()),
    )


if __name__ == "__main__":
    main()
