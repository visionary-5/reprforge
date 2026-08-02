#!/usr/bin/env python3
"""Run one actual IRPAPERS resident-compiler point without full prebuild."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from reprforge.irpapers_benchmark import (
    IRPapersColPaliBackend,
    load_irpapers,
    recall_at_k,
)
from reprforge.vidore_pipeline import ReprForgeViDoRePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument("--request-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    began = time.perf_counter()
    data = load_irpapers(args.docs, args.queries)
    dataset_load_seconds = time.perf_counter() - began
    began = time.perf_counter()
    backend = IRPapersColPaliBackend(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
    )
    model_load_seconds = time.perf_counter() - began
    pipeline = ReprForgeViDoRePipeline(
        base_model=str(args.base_model),
        adapter=str(args.adapter),
        mode="bm25-fusion-batched",
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
        candidate_k=args.candidate_k,
        top_k=20,
        request_batch_size=args.request_batch_size,
        cohort_cache_policy="resident",
        backend_factory=lambda: backend,
    )
    pipeline.index(
        list(data.corpus_ids),
        list(data.corpus_images),
        list(data.corpus_texts),
        "IRPAPERS",
    )
    results, cost = pipeline.search(list(data.query_ids), list(data.queries))

    import torch

    payload = {
        "schema_version": 1,
        "status": "complete",
        "dataset": dict(data.metadata),
        "resource_contract": {
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda_visible_device_count": torch.cuda.device_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "dataset_load_seconds": dataset_load_seconds,
            "model_load_seconds": model_load_seconds,
            "batch_size": args.batch_size,
            "scoring_batch_size": args.scoring_batch_size,
            "request_batch_size": args.request_batch_size,
        },
        "run": {
            "name": f"resident_compiler_k{args.candidate_k}",
            "quality": recall_at_k(results, data.qrels),
            "cost": {
                **cost,
                "end_to_end_seconds_excluding_model_and_dataset": (
                    cost["measured_index_ms_inside_pipeline"]
                    + cost["total_execution_ms"]
                )
                / 1000.0,
            },
            "semantics": (
                "Actual online BM25 cohort compiler with batch-atomic resident "
                "visual state and candidate-relative z-score fusion"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
