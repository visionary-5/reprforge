#!/usr/bin/env python3
"""Run the official ViDoRe evaluator over verified local Parquet files.

The official CLI normally downloads three Hugging Face configurations.  This
runner preserves the same columns and evaluator when a compute server cannot
reach the Hub.  It does not replace ViDoRe metrics or alter relevance labels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.vidore_pipeline import ReprForgeViDoRePipeline


OFFICIAL_METRICS = [
    "ndcg_cut_1",
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
    "ndcg_cut_100",
    "recall_1",
    "recall_5",
    "recall_10",
    "recall_20",
    "recall_50",
    "recall_100",
    "P_1",
    "P_5",
    "P_10",
    "P_20",
    "map",
    "map_cut_1",
    "map_cut_10",
    "map_cut_100",
    "recip_rank",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(
    paths: Sequence[Path],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path, columns=list(columns))
        rows.extend(dict(row) for row in table.to_pylist())
    return rows


def _component_paths(root: Path, part: str) -> tuple[Path, ...]:
    paths = tuple(sorted((root / part).glob("test-*.parquet")))
    if not paths:
        raise FileNotFoundError(
            f"missing local ViDoRe Parquet shards: {root / part}"
        )
    return paths


def _component_sha256(paths: Sequence[Path]) -> str:
    if len(paths) == 1:
        return _sha256(paths[0])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _decode_image(value: Any) -> Any:
    from PIL import Image

    if hasattr(value, "save"):
        return value
    payload = value.get("bytes") if isinstance(value, dict) else None
    if not isinstance(payload, bytes):
        raise ValueError("local ViDoRe image column lacks encoded bytes")
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB").copy()


def load_local_vidore(
    root: Path,
    *,
    language: str | None,
    smoke_queries: int = 0,
    smoke_corpus: int = 0,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[Any],
    list[str],
    dict[str, dict[str, int]],
    dict[str, str],
    dict[str, Any],
]:
    """Load official columns, optionally deriving a deterministic smoke slice."""

    paths = {
        part: _component_paths(root, part)
        for part in ("queries", "corpus", "qrels")
    }

    query_rows = _read_rows(
        paths["queries"],
        ("query_id", "query", "language"),
    )
    if language is not None:
        query_rows = [
            row for row in query_rows if str(row["language"]) == language
        ]
    if smoke_queries:
        if smoke_queries < 1:
            raise ValueError("smoke_queries cannot be negative")
        query_rows = query_rows[:smoke_queries]
    if not query_rows:
        raise ValueError("local ViDoRe selection contains no queries")

    query_ids = [str(row["query_id"]) for row in query_rows]
    selected_query_ids = set(query_ids)
    qrel_rows = [
        row
        for row in _read_rows(
            paths["qrels"],
            ("query_id", "corpus_id", "score"),
        )
        if str(row["query_id"]) in selected_query_ids
    ]
    relevant_ids = {str(row["corpus_id"]) for row in qrel_rows}

    corpus_rows = _read_rows(
        paths["corpus"],
        ("corpus_id", "image", "markdown"),
    )
    if smoke_corpus:
        if smoke_corpus < len(relevant_ids):
            raise ValueError(
                "smoke_corpus is smaller than the selected queries' relevant set"
            )
        included_ids = set(relevant_ids)
        for row in corpus_rows:
            if len(included_ids) >= smoke_corpus:
                break
            included_ids.add(str(row["corpus_id"]))
        corpus_rows = [
            row for row in corpus_rows if str(row["corpus_id"]) in included_ids
        ]
    corpus_ids = [str(row["corpus_id"]) for row in corpus_rows]
    corpus_id_set = set(corpus_ids)
    if not relevant_ids <= corpus_id_set:
        raise AssertionError("local corpus selection dropped a relevant page")

    qrels: dict[str, dict[str, int]] = {}
    for row in qrel_rows:
        query_id = str(row["query_id"])
        corpus_id = str(row["corpus_id"])
        if corpus_id in corpus_id_set:
            qrels.setdefault(query_id, {})[corpus_id] = int(row["score"])
    if set(qrels) != selected_query_ids:
        raise ValueError("at least one selected query lacks relevance judgments")

    metadata = {
        "source": "official-local-parquet",
        "root": str(root.resolve()),
        "sha256": {
            part: _component_sha256(part_paths)
            for part, part_paths in paths.items()
        },
        "parquet_shards": {
            part: [path.name for path in part_paths]
            for part, part_paths in paths.items()
        },
        "smoke_queries": smoke_queries,
        "smoke_corpus": smoke_corpus,
        "selected_queries": len(query_ids),
        "selected_corpus": len(corpus_ids),
        "selected_qrels": sum(len(values) for values in qrels.values()),
    }
    return (
        query_ids,
        [str(row["query"]) for row in query_rows],
        corpus_ids,
        [_decode_image(row["image"]) for row in corpus_rows],
        [str(row["markdown"]) for row in corpus_rows],
        qrels,
        {str(row["query_id"]): str(row["language"]) for row in query_rows},
        metadata,
    )


def write_score_trace(
    root: Path,
    *,
    pipeline: ReprForgeViDoRePipeline,
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    qrels: dict[str, dict[str, int]],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Write runtime replay state and oracle-only labels separately."""

    trace = pipeline.export_score_trace(query_ids)
    query_positions = {value: index for index, value in enumerate(query_ids)}
    corpus_positions = {value: index for index, value in enumerate(corpus_ids)}
    label_query: list[int] = []
    label_corpus: list[int] = []
    label_relevance: list[int] = []
    for query_id in query_ids:
        for corpus_id, relevance in sorted(qrels[query_id].items()):
            label_query.append(query_positions[query_id])
            label_corpus.append(corpus_positions[corpus_id])
            label_relevance.append(int(relevance))

    root.mkdir(parents=True, exist_ok=True)
    runtime_path = root / "runtime.npz"
    labels_path = root / "oracle-labels.npz"
    np.savez_compressed(
        runtime_path,
        query_ids=trace["query_ids"],
        corpus_ids=trace["corpus_ids"],
        scores=trace["scores"],
        vector_bytes=trace["vector_bytes"],
        encode_ms=trace["encode_ms"],
        index_total_ms=trace["index_total_ms"],
        model_load_ms=trace["model_load_ms"],
    )
    np.savez_compressed(
        labels_path,
        query_positions=np.asarray(label_query, dtype=np.int32),
        corpus_positions=np.asarray(label_corpus, dtype=np.int32),
        relevance=np.asarray(label_relevance, dtype=np.int16),
    )
    manifest = {
        "schema_version": 1,
        "mode": trace["mode"],
        "runtime_file": runtime_path.name,
        "runtime_sha256": _sha256(runtime_path),
        "oracle_labels_file": labels_path.name,
        "oracle_labels_sha256": _sha256(labels_path),
        "query_count": len(query_ids),
        "corpus_count": len(corpus_ids),
        "score_shape": list(trace["scores"].shape),
        "index_total_ms": float(trace["index_total_ms"]),
        "per_item_encode_ms_sum": float(trace["encode_ms"].sum()),
        "label_count": len(label_relevance),
        "labels_are_runtime_visible": False,
        "official_upstream_commit": (
            "a70f23af8bb3b33efe8a4a6c6c15a6e2d978035e"
        ),
        "source_sha256": source["sha256"],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default="vidore/vidore_v3_hr")
    parser.add_argument("--language", default="english")
    parser.add_argument(
        "--mode",
        choices=[
            "text",
            "visual",
            "visual-pool",
            "two-stage",
            "tiered-selective",
            "bm25-fusion-sync",
            "bm25-fusion-batched",
        ],
        default="tiered-selective",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--image-pool-factor", type=int, default=25)
    parser.add_argument("--cache-capacity-items", type=int, default=0)
    parser.add_argument("--request-batch-size", type=int, default=8)
    parser.add_argument(
        "--cohort-cache-policy",
        choices=["none", "resident"],
        default="resident",
    )
    parser.add_argument("--smoke-queries", type=int, default=0)
    parser.add_argument("--smoke-corpus", type=int, default=0)
    parser.add_argument(
        "--score-trace-dir",
        type=Path,
        help=(
            "write the complete score/cost surface and separate oracle labels; "
            "supported for text and visual modes"
        ),
    )
    args = parser.parse_args()
    if args.score_trace_dir is not None and args.mode not in {"text", "visual"}:
        parser.error("--score-trace-dir requires --mode text or visual")

    from vidore_benchmark.pipeline_evaluation import (
        aggregate_results,
        evaluate_retrieval,
    )

    (
        query_ids,
        queries,
        corpus_ids,
        corpus_images,
        corpus_texts,
        qrels,
        query_languages,
        source,
    ) = load_local_vidore(
        args.data_root,
        language=args.language,
        smoke_queries=args.smoke_queries,
        smoke_corpus=args.smoke_corpus,
    )
    pipeline = ReprForgeViDoRePipeline(
        base_model=args.base_model,
        adapter=args.adapter,
        mode=args.mode,
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        image_pool_factor=args.image_pool_factor,
        cache_capacity_items=args.cache_capacity_items,
        request_batch_size=args.request_batch_size,
        cohort_cache_policy=args.cohort_cache_policy,
        capture_score_trace=args.score_trace_dir is not None,
    )
    per_query = evaluate_retrieval(
        pipeline=pipeline,
        query_ids=query_ids,
        queries=queries,
        corpus_ids=corpus_ids,
        corpus_images=corpus_images,
        corpus_texts=corpus_texts,
        qrels=qrels,
        dataset_name=args.dataset_name,
        metrics=OFFICIAL_METRICS,
    )
    aggregated = aggregate_results(per_query, query_languages)
    score_trace = None
    if args.score_trace_dir is not None:
        score_trace = write_score_trace(
            args.score_trace_dir,
            pipeline=pipeline,
            query_ids=query_ids,
            corpus_ids=corpus_ids,
            qrels=qrels,
            source=source,
        )
    payload = {
        "dataset": args.dataset_name,
        "split": "test",
        "language": args.language,
        "pipeline_type": None,
        "module_path": "reprforge.vidore_pipeline",
        "class_name": "ReprForgeViDoRePipeline",
        "pipeline_args": {
            "mode": args.mode,
            "batch_size": args.batch_size,
            "scoring_batch_size": args.scoring_batch_size,
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "image_pool_factor": args.image_pool_factor,
            "cache_capacity_items": args.cache_capacity_items,
            "request_batch_size": args.request_batch_size,
            "cohort_cache_policy": args.cohort_cache_policy,
        },
        "official_upstream_commit": (
            "a70f23af8bb3b33efe8a4a6c6c15a6e2d978035e"
        ),
        "local_source": source,
        "aggregated_metrics": aggregated,
        "score_trace": score_trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
