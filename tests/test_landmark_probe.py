import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube
from reprforge.landmark_probe import (
    analyze_landmark_budgets,
    landmark_completion_surface,
)


def test_landmark_surface_preserves_candidate_membership():
    base = np.asarray([[5.0, 4.0, 3.0, 2.0, 1.0]])
    expensive = np.asarray([[1.0, 3.0, 5.0, 2.0, 4.0]])
    output = landmark_completion_surface(
        base, expensive, candidate_k=3, budget=2, policy="boundary", target_k=2
    )
    assert set(np.argsort(output[0])[::-1][:3]) == {0, 1, 2}


def test_full_budget_is_policy_invariant():
    base = np.asarray([[5.0, 4.0, 3.0, 2.0, 1.0], [1, 2, 3, 4, 5]])
    expensive = np.asarray([[1.0, 3.0, 5.0, 2.0, 4.0], [5, 3, 1, 4, 2]])
    coverage = landmark_completion_surface(
        base, expensive, candidate_k=5, budget=5, policy="coverage", target_k=2
    )
    boundary = landmark_completion_surface(
        base, expensive, candidate_k=5, budget=5, policy="boundary", target_k=2
    )
    assert np.array_equal(coverage, boundary)


def test_budget_analysis_reaches_exact_full_fusion():
    cube = ScoreCube(
        query_ids=("q0", "q1"),
        corpus_ids=("d0", "d1", "d2", "d3"),
        scores={
            "cheap": np.asarray([[4, 3, 2, 1], [1, 2, 3, 4]], dtype=float),
            "expensive": np.asarray([[1, 4, 3, 2], [2, 3, 4, 1]], dtype=float),
        },
        relevance=({1: 1.0}, {2: 1.0}),
        split_roles=("fit", "eval"),
    )
    report = analyze_landmark_budgets(
        cube,
        base_route="cheap",
        expensive_route="expensive",
        candidate_k=4,
        budgets=(2, 4),
        target_metric="ndcg_at_2",
        target_k=2,
    )
    full_rows = [row for row in report["rows"] if row["budget"] == 4]
    assert all(row["ndcg_at_2"] == report["full_candidate_fusion"] for row in full_rows)
