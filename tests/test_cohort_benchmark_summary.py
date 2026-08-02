from __future__ import annotations

import json
from pathlib import Path

from reprforge.cohort_benchmark_summary import summarize


def _write_run(
    path: Path,
    *,
    mode: str,
    ndcg: float,
    index_ms: float,
    search_ms: float,
    pages: int,
) -> None:
    payload = {
        "dataset": "vidore/test",
        "aggregated_metrics": {
            "overall": {"ndcg_cut_10": ndcg, "recall_100": 0.8},
            "timing": {
                "num_queries": 10,
                "num_corpus": 100,
                "indexing_time_milliseconds": index_ms,
                "search_time_milliseconds": search_ms,
                "total_retrieval_time_milliseconds": index_ms + search_ms,
                "search_throughput_ms_per_query": search_ms / 10,
            },
            "infos": {
                "mode": mode,
                "visual_materializations_during_index": pages,
                "visual_pages_encoded": pages,
                "request_batch_size": 8,
                "cache_policy": "resident",
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_applies_execution_and_full_visual_gates(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    sync = tmp_path / "sync.json"
    single_resident = tmp_path / "single-resident.json"
    resident = tmp_path / "resident.json"
    _write_run(
        full,
        mode="visual",
        ndcg=0.5,
        index_ms=80.0,
        search_ms=20.0,
        pages=100,
    )
    _write_run(
        sync,
        mode="bm25-fusion-sync",
        ndcg=0.55,
        index_ms=1.0,
        search_ms=200.0,
        pages=200,
    )
    _write_run(
        single_resident,
        mode="bm25-fusion-batched",
        ndcg=0.55,
        index_ms=1.0,
        search_ms=75.0,
        pages=70,
    )
    _write_run(
        resident,
        mode="bm25-fusion-batched",
        ndcg=0.55,
        index_ms=1.0,
        search_ms=70.0,
        pages=70,
    )

    result = summarize(
        full,
        [
            ("sync-none", sync),
            ("b1-resident", single_resident),
            ("b8-resident", resident),
        ],
    )

    assert result["gate"] == {
        "online_quality_span_at_most_0.002_ndcg": True,
        "b8_resident_end_to_end_below_full_visual": True,
        "b8_resident_search_speedup_vs_sync_at_least_1.25x": True,
        "b8_incremental_search_speedup_vs_b1_resident_at_least_1.10x": False,
    }
    assert (
        result["comparisons"]["b8-resident"][
            "search_speedup_vs_sync_none"
        ]
        == 200.0 / 70.0
    )
