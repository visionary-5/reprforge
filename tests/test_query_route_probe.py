import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube
from reprforge.query_route_probe import (
    candidate_identity_features,
    categorical_features,
    cheap_route_features,
    crossfit_query_router,
    lexical_hash_features,
)


def test_feature_builders_are_finite_and_aligned():
    lexical = lexical_hash_features(["chart 10%?", "plain text"], dimensions=8)
    categorical, names = categorical_features(
        [{"kind": ["chart", "table"]}, {"kind": ["text"]}], fields=("kind",)
    )
    cheap = cheap_route_features(np.asarray([[3, 2, 1], [1, 3, 2]], dtype=float))
    candidates = candidate_identity_features(
        np.asarray([[3, 2, 1], [1, 3, 2]], dtype=float), candidate_k=2
    )
    assert lexical.shape == (2, 14)
    assert categorical.shape == (2, 3)
    assert names == ("kind=chart", "kind=table", "kind=text")
    assert cheap.shape == (2, 8)
    assert candidates.shape == (2, 3)
    assert np.count_nonzero(candidates) == 4
    assert np.isfinite(np.column_stack([lexical, categorical, cheap])).all()


def test_crossfit_router_recovers_predictable_route_utility():
    queries = tuple(f"q-{index}" for index in range(60))
    feature = np.asarray([[index % 2] for index in range(60)], dtype=float)
    route_a = np.tile([3.0, 2.0, 1.0], (60, 1))
    route_b = np.tile([1.0, 3.0, 2.0], (60, 1))
    relevance = tuple({0: 1.0} if index % 2 == 0 else {1: 1.0} for index in range(60))
    cube = ScoreCube(
        query_ids=queries,
        corpus_ids=("d0", "d1", "d2"),
        scores={"a": route_a, "b": route_b},
        relevance=relevance,
        split_roles=tuple("fit" if index < 40 else "eval" for index in range(60)),
    )
    result = crossfit_query_router(
        cube, feature, target_metric="ndcg_at_1", k=1, folds=5
    )
    assert result["mean_metric"] == 1.0
    assert result["oracle_route_accuracy"] == 1.0
    assert result["oracle_gap_recovery"] == 1.0
