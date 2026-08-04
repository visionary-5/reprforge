import numpy as np

from reprforge.heterogeneity_atlas import (
    ScoreCube,
    analyze_cube,
    deterministic_split_roles,
    percentile_scores,
    stable_ranks,
)


def _cube() -> ScoreCube:
    return ScoreCube(
        query_ids=("q0", "q1", "q2", "q3"),
        corpus_ids=("d0", "d1", "d2"),
        scores={
            "a": np.asarray(
                [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]], dtype=float
            ),
            "b": np.asarray(
                [[1, 3, 2], [1, 3, 2], [1, 3, 2], [1, 3, 2]], dtype=float
            ),
        },
        relevance=({0: 1.0}, {1: 1.0}, {0: 1.0}, {1: 1.0}),
        split_roles=("fit", "fit", "eval", "eval"),
    )


def test_stable_ranks_and_percentiles_are_scale_invariant():
    scores = np.asarray([[10.0, 10.0, -1.0], [2.0, 3.0, 1.0]])
    assert stable_ranks(scores).tolist() == [[1, 2, 3], [2, 1, 3]]
    assert np.allclose(percentile_scores(scores), percentile_scores(scores * 7 + 4))


def test_deterministic_split_is_order_independent():
    ids = tuple(f"query-{index}" for index in range(100))
    roles = dict(zip(ids, deterministic_split_roles(ids), strict=True))
    reversed_roles = dict(
        zip(reversed(ids), deterministic_split_roles(tuple(reversed(ids))), strict=True)
    )
    assert roles == reversed_roles


def test_atlas_separates_global_query_and_static_document_levels():
    report = analyze_cube(_cube(), ks=(1, 2), target_metric="ndcg_at_1")
    assert report["best_global_route_selected_on_fit"] == "a"
    assert report["uniform_routes"]["a"]["eval"]["ndcg_at_1"] == 0.5
    assert report["diagnostic_upper_bounds"]["query_route_oracle"][
        "ndcg_at_1"
    ] == 1.0
    assert report["fit_label_static_document_plan"]["route_counts"]["b"] == 1
    assert report["fit_label_static_document_plan"]["route_counts"][
        "fit_observed_documents"
    ] == 2
    assert report["interpretation_guardrails"][
        "query_route_oracle_uses_eval_labels"
    ]
