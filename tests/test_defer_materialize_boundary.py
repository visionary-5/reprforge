import numpy as np

from reprforge.defer_materialize_boundary import locator_boundary, repair_reuse_crossfit
from reprforge.partial_vlm_materialization import ScoreSurface


def _surface() -> ScoreSurface:
    return ScoreSurface(
        name="boundary",
        query_ids=np.asarray(["q0", "q1", "q2", "q3"]),
        corpus_ids=np.asarray(["p0", "p1", "p2", "p3"]),
        text_scores=np.asarray(
            [[4, 3, 2, 1], [4, 3, 2, 1], [1, 2, 4, 3], [1, 2, 4, 3]],
            dtype=float,
        ),
        visual_scores=np.asarray(
            [[1, 4, 3, 2], [1, 4, 3, 2], [4, 3, 1, 2], [4, 3, 1, 2]],
            dtype=float,
        ),
        qrels=np.asarray(
            [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
            dtype=np.int16,
        ),
        text_bytes=np.ones(4),
        visual_bytes=np.ones(4),
        visual_encode_ms=np.ones(4),
        input_sha256={},
    )


def test_locator_boundary_counts_visual_repairs_of_text_escape():
    result = locator_boundary(_surface(), 1)
    assert result["boundary"]["text_miss_queries"] == 4
    assert result["boundary"]["visual_repair_queries"] == 2
    assert result["boundary"]["visual_repairs_fraction_of_text_misses"] == 0.5


def test_repair_reuse_uses_history_pages_only():
    result = repair_reuse_crossfit(
        _surface(), np.asarray([0, 1, 0, 1], dtype=np.int16), 1
    )
    assert result["future_repair_queries"] == 2
    assert result["unique_page_overlap_fraction_weighted"] == 1.0
    assert result["event_overlap_fraction_weighted"] == 1.0
