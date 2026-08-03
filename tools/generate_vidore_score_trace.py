#!/usr/bin/env python3
"""Generate ReprForge ViDoRe score traces without evaluator dependencies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from reprforge.vidore_local_eval import load_local_vidore, write_score_trace
from reprforge.vidore_pipeline import ReprForgeViDoRePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mode", choices=["text", "visual"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--language", default="english")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    args = parser.parse_args()

    began = time.perf_counter()
    (
        query_ids,
        queries,
        corpus_ids,
        corpus_images,
        corpus_texts,
        qrels,
        _languages,
        source,
    ) = load_local_vidore(args.data_root, language=args.language)
    load_seconds = time.perf_counter() - began
    pipeline = ReprForgeViDoRePipeline(
        base_model=args.base_model,
        adapter=args.adapter,
        mode=args.mode,
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
        top_k=100,
        capture_score_trace=True,
    )
    began = time.perf_counter()
    pipeline.index(
        corpus_ids,
        corpus_images,
        corpus_texts,
        dataset_name=args.dataset_name,
    )
    index_seconds = time.perf_counter() - began
    began = time.perf_counter()
    rankings, search_info = pipeline.search(query_ids, queries)
    search_seconds = time.perf_counter() - began
    manifest = write_score_trace(
        args.trace_dir,
        pipeline=pipeline,
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        qrels=qrels,
        source=source,
    )
    output = {
        "schema_version": 1,
        "dataset": args.dataset_name,
        "mode": args.mode,
        "queries": len(query_ids),
        "corpus": len(corpus_ids),
        "data_load_seconds": load_seconds,
        "index_seconds": index_seconds,
        "search_seconds": search_seconds,
        "ranking_queries": len(rankings),
        "trace": manifest,
        "pipeline": search_info,
        "official_metrics_computed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
