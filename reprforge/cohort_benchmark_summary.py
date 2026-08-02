#!/usr/bin/env python3
"""Compact official ViDoRe cohort compiler runs into a reviewable result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact(path: Path) -> dict[str, Any]:
    payload = _load(path)
    aggregate = payload["aggregated_metrics"]
    timing = aggregate["timing"]
    info = aggregate["infos"]
    batch_trace = info.get("batch_trace") or []
    return {
        "mode": info["mode"],
        "request_batch_size": info.get("request_batch_size"),
        "cache_policy": info.get("cache_policy"),
        "ndcg@10": aggregate["overall"]["ndcg_cut_10"],
        "recall@100": aggregate["overall"]["recall_100"],
        "index_ms": timing["indexing_time_milliseconds"],
        "search_ms": timing["search_time_milliseconds"],
        "end_to_end_ms": timing["total_retrieval_time_milliseconds"],
        "search_ms_per_query": timing["search_throughput_ms_per_query"],
        "visual_pages_encoded": info.get(
            "visual_pages_encoded",
            info.get("visual_materializations_during_index", 0),
        ),
        "visual_padding_pages_encoded": info.get(
            "visual_padding_pages_encoded",
            0,
        ),
        "visual_encoder_calls": info.get("visual_encoder_calls"),
        "cache_hit_fraction": info.get("cache_hit_fraction"),
        "within_batch_dedup_fraction": info.get(
            "within_batch_dedup_fraction"
        ),
        "resident_items": info.get("current_resident_items"),
        "resident_vector_bytes": info.get("current_resident_vector_bytes"),
        "batch_completion_ms": info.get("batch_completion_ms"),
        "batch_count": len(batch_trace),
        "first_batch_completion_ms": (
            batch_trace[0]["completion_ms"] if batch_trace else None
        ),
        "batch_trace": batch_trace,
        "source_sha256": _sha256(path),
    }


def _progress_before(
    row: dict[str, Any],
    deadline_ms: float,
) -> dict[str, Any] | None:
    trace = row.get("batch_trace") or []
    if not trace:
        return None
    elapsed = float(row["index_ms"])
    completed = 0
    resident = 0
    for batch in trace:
        next_elapsed = elapsed + float(batch["completion_ms"])
        if next_elapsed > deadline_ms:
            break
        elapsed = next_elapsed
        completed += int(batch["query_count"])
        resident = int(batch["resident_items_after_publish"])
    return {
        "deadline_ms": deadline_ms,
        "queries_completed": completed,
        "resident_items": resident,
        "elapsed_ms_after_last_completed_batch": elapsed,
    }


def summarize(
    full_visual_path: Path,
    online: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    full_payload = _load(full_visual_path)
    full = _compact(full_visual_path)
    runs = {label: _compact(path) for label, path in online}
    dataset = full_payload["dataset"]
    query_count = full_payload["aggregated_metrics"]["timing"]["num_queries"]
    corpus_count = full_payload["aggregated_metrics"]["timing"]["num_corpus"]
    for label, path in online:
        payload = _load(path)
        if payload["dataset"] != dataset:
            raise ValueError(f"{label} dataset differs from full visual")
        timing = payload["aggregated_metrics"]["timing"]
        if timing["num_queries"] != query_count:
            raise ValueError(f"{label} query count differs from full visual")
        if timing["num_corpus"] != corpus_count:
            raise ValueError(f"{label} corpus count differs from full visual")

    online_quality = [float(row["ndcg@10"]) for row in runs.values()]
    quality_span = (
        max(online_quality) - min(online_quality)
        if online_quality
        else 0.0
    )
    quality_equivalent = (
        quality_span <= 0.002
        if len(online_quality) >= 2
        else None
    )
    comparisons: dict[str, Any] = {}
    for label, row in runs.items():
        comparisons[label] = {
            "end_to_end_speedup_vs_full_visual": (
                full["end_to_end_ms"] / row["end_to_end_ms"]
            ),
            "relative_ndcg_gain_vs_full_visual": (
                row["ndcg@10"] / full["ndcg@10"] - 1.0
            ),
            "progress_when_full_visual_index_becomes_ready": _progress_before(
                row,
                float(full["index_ms"]),
            ),
        }
        if "sync-none" in runs:
            comparisons[label]["search_speedup_vs_sync_none"] = (
                runs["sync-none"]["search_ms"] / row["search_ms"]
            )

    resident = runs.get("b8-resident")
    single_resident = runs.get("b1-resident")
    sync = runs.get("sync-none")
    gate = {
        "online_quality_span_at_most_0.002_ndcg": quality_equivalent,
        "b8_resident_end_to_end_below_full_visual": (
            resident["end_to_end_ms"] < full["end_to_end_ms"]
            if resident is not None
            else None
        ),
        "b8_resident_search_speedup_vs_sync_at_least_1.25x": (
            sync["search_ms"] / resident["search_ms"] >= 1.25
            if resident is not None and sync is not None
            else None
        ),
        "b8_incremental_search_speedup_vs_b1_resident_at_least_1.10x": (
            single_resident["search_ms"] / resident["search_ms"] >= 1.10
            if resident is not None and single_resident is not None
            else None
        ),
    }
    # The official raw JSON remains bound by its digest.  Keep the committed
    # summary compact after deriving progress checkpoints from its batch trace.
    full.pop("batch_trace", None)
    for row in runs.values():
        row.pop("batch_trace", None)
    return {
        "schema_version": 1,
        "dataset": dataset,
        "query_count": query_count,
        "corpus_count": corpus_count,
        "hardware": "1x NVIDIA A100-SXM4-80GB",
        "full_visual": full,
        "online": runs,
        "comparisons": comparisons,
        "gate": gate,
        "online_ndcg_span": quality_span,
        "latency_scope_note": (
            "batch_completion_ms measures synchronous request-batch completion, "
            "not independently arriving per-query P95"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-visual", type=Path, required=True)
    parser.add_argument(
        "--online",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries: list[tuple[str, Path]] = []
    for value in args.online:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            parser.error("--online must use LABEL=PATH")
        entries.append((label, Path(raw_path)))
    if not entries:
        parser.error("at least one --online run is required")
    result = summarize(args.full_visual, entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
