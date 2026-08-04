import numpy as np

from reprforge.cohort_certificate import (
    analyze_certificates,
    exact_certificate_size,
    greedy_certificate,
)
from reprforge.heterogeneity_atlas import ScoreCube


def _rows():
    base = np.asarray([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    expensive = np.asarray([1, 3, 8, 7, 6, 5, 4, 2], dtype=float)
    return base, expensive


def test_greedy_certificate_preserves_teacher_topk_set():
    base, expensive = _rows()
    certificate = greedy_certificate(
        base, expensive, candidate_k=8, target_k=4, anchors=3
    )
    assert set(certificate.teacher_top_positions) == set(
        certificate.certificate_top_positions
    )
    assert 3 <= certificate.size <= 8


def test_order_certificate_preserves_teacher_topk_order():
    base, expensive = _rows()
    certificate = greedy_certificate(
        base,
        expensive,
        candidate_k=8,
        target_k=4,
        anchors=3,
        objective="order",
    )
    assert np.array_equal(
        certificate.teacher_top_positions,
        certificate.certificate_top_positions,
    )


def test_exact_audit_never_exceeds_greedy_upper_bound():
    base, expensive = _rows()
    greedy = greedy_certificate(
        base, expensive, candidate_k=8, target_k=4, anchors=3
    )
    exact = exact_certificate_size(
        base,
        expensive,
        candidate_k=8,
        target_k=4,
        upper_bound=greedy.size,
        anchors=3,
    )
    assert exact["exhausted"]
    assert exact["exact_size"] <= greedy.size

    greedy_order = greedy_certificate(
        base,
        expensive,
        candidate_k=8,
        target_k=4,
        anchors=3,
        objective="order",
    )
    exact_order = exact_certificate_size(
        base,
        expensive,
        candidate_k=8,
        target_k=4,
        upper_bound=greedy_order.size,
        anchors=3,
        objective="order",
    )
    assert exact_order["exhausted"]
    assert exact_order["exact_size"] <= greedy_order.size


def test_dataset_report_has_exact_set_agreement():
    base, expensive = _rows()
    cube = ScoreCube(
        query_ids=("q0", "q1"),
        corpus_ids=tuple(f"d{index}" for index in range(8)),
        scores={
            "base": np.stack([base, base[::-1]]),
            "expensive": np.stack([expensive, expensive[::-1]]),
        },
        relevance=({2: 1.0}, {5: 1.0}),
        split_roles=("fit", "eval"),
    )
    report = analyze_certificates(
        cube,
        base_route="base",
        expensive_route="expensive",
        candidate_k=8,
        target_k=4,
        target_metric="ndcg_at_4",
        per_item_build_ms=np.ones(8),
        exact_audit_queries=2,
    )
    assert report["quality"]["topk_set_agreement"] == 1.0
    assert report["selection_uses_qrels"] is False
    assert report["physical_reuse"]["certificate_unique_build_ms"] <= 8.0


def test_order_report_exactly_matches_teacher_quality():
    base, expensive = _rows()
    cube = ScoreCube(
        query_ids=("q0", "q1"),
        corpus_ids=tuple(f"d{index}" for index in range(8)),
        scores={
            "base": np.stack([base, base[::-1]]),
            "expensive": np.stack([expensive, expensive[::-1]]),
        },
        relevance=({2: 1.0}, {5: 1.0}),
        split_roles=("fit", "eval"),
    )
    report = analyze_certificates(
        cube,
        base_route="base",
        expensive_route="expensive",
        candidate_k=8,
        target_k=4,
        target_metric="ndcg_at_4",
        objective="order",
        exact_audit_queries=0,
    )
    assert report["quality"]["topk_order_agreement"] == 1.0
    assert report["quality"]["certificate"] == report["quality"][
        "full_candidate_teacher"
    ]
