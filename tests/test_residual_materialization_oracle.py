import numpy as np
import pytest

from reprforge.residual_materialization_oracle import (
    ResidualRankSurface,
    auc,
    evaluate,
    gain_recovery,
    global_label_rank_utility,
    hash_folds,
    projected_cost,
    residual_events,
    residual_utility,
    top_utility,
)


def _surface() -> ResidualRankSurface:
    return ResidualRankSurface(
        name="tiny",
        query_ids=["q0", "q1", "q2", "q3"],
        doc_ids=["d0", "d1", "d2", "d3", "d4"],
        bm25=np.asarray(
            [
                [0, 1, 3, 4, 2],
                [0, 1, 2, 3, 4],
                [1, 0, 3, 4, 2],
                [4, 3, 2, 1, 0],
            ]
        ),
        colsmol=np.asarray(
            [
                [1, 0, 3, 4, 2],
                [1, 0, 2, 3, 4],
                [0, 1, 3, 4, 2],
                [3, 4, 2, 1, 0],
            ]
        ),
        omni=np.asarray(
            [
                [2, 0, 1, 3, 4],
                [0, 2, 1, 3, 4],
                [2, 1, 0, 3, 4],
                [4, 3, 2, 1, 0],
            ]
        ),
        qrels=np.asarray(
            [
                [0, 0, 2, 0, 0],
                [2, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 2],
            ],
            dtype=np.float32,
        ),
    )


def test_residual_pages_are_sparse_and_selected_omni_repairs_them():
    surface = _surface()
    boundary = residual_events(
        surface, np.arange(4), rrf_constant=60, depth=2
    )
    assert boundary["queries"] == 2
    assert boundary["unique_pages"] == {2}
    utility = residual_utility(surface, range(4), rrf_constant=60, depth=2)
    assert top_utility(utility, 3, positive_only=True).tolist() == [2]
    base = evaluate(
        surface, range(4), rrf_constant=60, selected_omni_pages=None
    )
    partial = evaluate(
        surface, range(4), rrf_constant=60, selected_omni_pages=[2]
    )
    assert partial["query_hit_at_20"] == base["query_hit_at_20"] == 1.0
    assert partial["ndcg_at_10"] > base["ndcg_at_10"]


def test_global_oracle_cost_gain_auc_and_folds_are_deterministic():
    surface = _surface()
    utility = global_label_rank_utility(surface, range(4), rrf_constant=60)
    assert utility[2] > 0
    assert gain_recovery(0.8, 0.5, 0.9) == pytest.approx(0.75)
    assert gain_recovery(0.8, 0.5, 0.504) is None
    assert projected_cost(
        0.1,
        full_build_seconds=100,
        full_index_bytes=1000,
        base_build_seconds=20,
        base_index_bytes=200,
    )["total_visual_index_bytes_including_colsmol"] == 300
    assert auc([0, 1, 2, 3], [False, False, True, True]) == 1.0
    first = hash_folds(surface.query_ids, 2, 0)
    second = hash_folds(surface.query_ids, 2, 0)
    assert first.tolist() == second.tolist()
