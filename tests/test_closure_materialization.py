import numpy as np

from tools.evaluate_closure_materialization import _phase_diagram


def test_phase_diagram_exposes_defer_partial_full_regions() -> None:
    # Two high-reuse queries per trace: partial pays 3 pages once and only one
    # transient page per query.  Defer wins first, partial wins next, then the
    # fixed full-ingestion cost eventually wins.
    result = _phase_diagram(
        np.asarray([1, 1]),
        persistent_pages=3,
        candidate_depth=4,
        full_pages=10,
        maximum_repeat=4,
    )
    assert result["winner_intervals"] == [
        {"start_query": 1, "end_query": 1, "winner": "defer"},
        {
            "start_query": 2,
            "end_query": 7,
            "winner": "closure_materialization",
        },
        {"start_query": 8, "end_query": 8, "winner": "full_ingestion"},
    ]
