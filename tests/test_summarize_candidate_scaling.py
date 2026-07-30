import json

from reprforge.summarize_candidate_scaling import summarize


def _report(path, *, items: int, latency: float) -> None:
    path.write_text(
        json.dumps(
            {
                "index_items": items,
                "compact_vector_bytes": items * 10,
                "resident_vector_bytes": items * 12,
                "execution_batches": items // 10,
                "latency_ms": {"p50": latency, "p95": latency * 2},
                "qps": 1000 / latency,
            }
        )
    )


def test_summary_separates_scaling_from_quality(tmp_path) -> None:
    _report(tmp_path / "system-factor1.json", items=10, latency=1.0)
    _report(tmp_path / "system-factor4.json", items=40, latency=3.0)

    result = summarize(tmp_path)

    assert result["contract"]["quality_labels_valid"] is False
    assert result["rows"][1]["latency_vs_factor1"] == 3.0
