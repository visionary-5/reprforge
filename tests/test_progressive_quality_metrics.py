import numpy as np
import pytest

from reprforge.progressive_quality_metrics import progressive_quality_timeline


def test_progressive_timeline_integrates_atomic_quality_updates():
    report = progressive_quality_timeline(
        [0.2, 0.2],
        [0.6, 1.0],
        [
            {"query_offset_start": 0, "query_count": 1, "completion_ms": 10.0},
            {"query_offset_start": 1, "query_count": 1, "completion_ms": 10.0},
        ],
        base_ready_ms=5.0,
        horizon_ms=30.0,
        target_quality=0.7,
        progress_reference_quality=0.3,
    )
    assert report["points"][-1]["mean_quality"] == pytest.approx(0.8)
    assert report["time_to_target_ms"] == 25.0
    assert report["normalized_gain_targets"]["fraction_0.5"] == {
        "target_quality": pytest.approx(0.55),
        "time_to_target_ms": 25.0,
    }
    assert report["normalized_gain_targets"]["fraction_0.9"] == {
        "target_quality": pytest.approx(0.75),
        "time_to_target_ms": 25.0,
    }
    # quality is 0 for 5ms, .2 for 10ms, .4 for 10ms and .8 for 5ms
    assert report["mean_quality_over_horizon"] == pytest.approx(10.0 / 30.0)
    assert report["revision"]["improved_fraction"] == 1.0


def test_progressive_timeline_rejects_overlapping_batches():
    with pytest.raises(ValueError):
        progressive_quality_timeline(
            np.zeros(2),
            np.ones(2),
            [
                {"query_offset_start": 0, "query_count": 2, "completion_ms": 1.0},
                {"query_offset_start": 1, "query_count": 1, "completion_ms": 1.0},
            ],
            base_ready_ms=1.0,
            horizon_ms=10.0,
            target_quality=0.5,
        )
