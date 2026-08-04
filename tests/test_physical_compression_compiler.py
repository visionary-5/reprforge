import numpy as np

from reprforge.physical_compression_compiler import (
    boundary_risk_utilities,
    budgeted_upgrade_order,
    calibrated_residual_surface,
    calibrated_residual_surface_from_anchors,
    compile_upgrade_mask_from_fit,
    compile_physical_curve,
    hybrid_score_surface,
    incremental_upgrade_bytes,
    round_robin_upgrade_order,
    select_upgrade_mask,
)


def test_boundary_flip_prioritizes_a_recurrent_topk_crossing():
    full = np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    cheap = np.asarray([[2.0, 3.0, 1.0], [2.0, 3.0, 1.0]])
    utilities = boundary_risk_utilities(
        full, cheap, [0, 1], target_k=1, exposure_depth=3
    )
    assert utilities["boundary_flip"][0] > utilities["boundary_flip"][2]
    assert utilities["boundary_flip"][1] > utilities["boundary_flip"][2]


def test_budgeting_and_hybrid_surface_charge_incremental_bytes():
    costs = incremental_upgrade_bytes([10, 20, 30], [5, 10, 15])
    order = budgeted_upgrade_order([2.0, 3.0, 0.0], costs)
    mask = select_upgrade_mask(order, costs, budget_bytes=10)
    assert mask.tolist() == [True, False, False]
    full = np.asarray([[3.0, 2.0, 1.0]])
    cheap = np.asarray([[1.0, 2.0, 3.0]])
    assert hybrid_score_surface(full, cheap, mask).tolist() == [[3.0, 2.0, 3.0]]


def test_compiler_produces_disjoint_query_roles_and_corpus_masks():
    query_ids = [f"q{i}" for i in range(30)]
    full = np.tile(np.asarray([[3.0, 2.0, 1.0]]), (30, 1))
    cheap = np.tile(np.asarray([[2.0, 3.0, 1.0]]), (30, 1))
    report = compile_physical_curve(
        query_ids,
        full,
        cheap,
        [10, 10, 10],
        [5, 5, 5],
        budget_fractions=[0.0, 0.5, 1.0],
    )
    assert report["fit_query_indices"]
    assert report["eval_query_indices"]
    assert set(report["fit_query_indices"]).isdisjoint(report["eval_query_indices"])
    for points in report["policies"].values():
        assert len(points) == 3
        assert all(0 <= index < 3 for point in points for index in point["upgraded_documents"])


def test_compiler_accepts_explicit_qrel_free_fit_partition():
    full = np.asarray([[4.0, 3.0], [3.0, 4.0], [4.0, 3.0]])
    cheap = np.asarray([[3.0, 4.0], [4.0, 3.0], [3.0, 4.0]])
    report = compile_physical_curve(
        ["q0", "q1", "q2"],
        full,
        cheap,
        [20, 20],
        [5, 5],
        budget_fractions=[0.5],
        fit_indices=[1],
    )
    assert report["fit_query_indices"] == [1]
    assert report["eval_query_indices"] == [0, 2]
    assert report["query_split_roles"] == ["eval", "fit", "eval"]


def test_calibrated_residual_surface_keeps_exact_anchor_scores():
    full = np.asarray([[3.0, 2.0, 1.0, 0.0]])
    cheap = np.asarray([[2.5, 1.8, 0.9, 0.0]])
    mask = np.asarray([True, True, False, False])
    calibrated = calibrated_residual_surface(full, cheap, mask)
    assert calibrated[0, mask].tolist() == full[0, mask].tolist()
    assert np.isfinite(calibrated).all()
    assert calibrated.shape == full.shape
    from_anchors = calibrated_residual_surface_from_anchors(
        cheap, full[:, mask], np.flatnonzero(mask)
    )
    assert np.allclose(from_anchors, calibrated)


def test_round_robin_order_is_a_complete_deterministic_union():
    order = round_robin_upgrade_order([0, 1, 2, 3], [2, 3, 1, 0])
    assert order.tolist() == [0, 2, 1, 3]
    assert sorted(order.tolist()) == [0, 1, 2, 3]


def test_static_fit_compiler_respects_full_anchor_byte_budget():
    full = np.tile(np.asarray([[3.0, 2.0, 1.0]]), (4, 1))
    cheap = np.tile(np.asarray([[2.0, 3.0, 1.0]]), (4, 1))
    plan = compile_upgrade_mask_from_fit(
        full,
        cheap,
        [0, 1, 2],
        [10, 10, 10],
        budget_fraction=0.5,
    )
    assert plan["anchor_vector_bytes"] <= 15
    assert plan["upgraded_document_count"] == 1
