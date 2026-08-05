from pathlib import Path

import numpy as np

from reprforge.multilevel_physical_design import analyze_multilevel_bundle


def _runtime(path: Path, scores: list[list[float]], scale: int) -> None:
    np.savez_compressed(
        path,
        query_ids=np.asarray(["q0", "q1", "q2", "q3"]),
        corpus_ids=np.asarray(["d0", "d1", "d2"]),
        scores=np.asarray(scores, dtype=np.float32),
        vector_bytes=np.asarray([scale, scale, scale], dtype=np.int64),
        encode_ms=np.asarray([1.0, 2.0, 3.0]),
    )


def test_incomplete_real_bundle_runs_only_uniform_headroom(tmp_path: Path) -> None:
    paths = {
        "cheap_base": tmp_path / "cheap.npz",
        "compact_pool": tmp_path / "compact.npz",
        "full_multivector": tmp_path / "full.npz",
    }
    _runtime(paths["cheap_base"], [[3, 2, 1]] * 4, 1)
    _runtime(paths["compact_pool"], [[2, 3, 1]] * 4, 2)
    _runtime(paths["full_multivector"], [[1, 2, 3]] * 4, 4)
    labels = tmp_path / "labels.npz"
    np.savez_compressed(
        labels,
        query_positions=np.asarray([0, 1, 2, 3]),
        corpus_positions=np.asarray([0, 1, 2, 2]),
        relevance=np.ones(4),
    )
    report = analyze_multilevel_bundle(
        paths,
        labels,
        dataset="toy",
        bootstrap_resamples=20,
    )
    assert report["decision"] == "NO-GO-current-artifacts"
    assert report["capability_matrix"]["aligned_real_three_tier_score_surfaces"]
    assert not report["capability_matrix"]["per_item_reload_all_tiers"]
    assert report["baseline_registry"]["gdsf"]["status"] == (
        "not_run_missing_artifact"
    )
    assert report["baseline_registry"]["uniform_compact_pool"]["status"] == (
        "run_real_surface"
    )
    assert report["interpretation_guardrails"]["no_synthetic_intermediate_quality"]


def test_query_route_oracle_uses_only_real_uniform_outcomes(tmp_path: Path) -> None:
    paths = {
        "cheap_base": tmp_path / "cheap.npz",
        "compact_pool": tmp_path / "compact.npz",
        "full_multivector": tmp_path / "full.npz",
    }
    _runtime(paths["cheap_base"], [[3, 2, 1]] * 4, 1)
    _runtime(paths["compact_pool"], [[2, 3, 1]] * 4, 2)
    _runtime(paths["full_multivector"], [[1, 2, 3]] * 4, 4)
    labels = tmp_path / "labels.npz"
    np.savez_compressed(
        labels,
        query_positions=np.asarray([0, 1, 2, 3]),
        corpus_positions=np.asarray([0, 1, 2, 2]),
        relevance=np.ones(4),
    )
    report = analyze_multilevel_bundle(
        paths,
        labels,
        dataset="toy",
        bootstrap_resamples=20,
    )
    oracle = report["diagnostic_query_route_oracle"]
    assert oracle["deployable"] is False
    assert oracle["uses_eval_qrels"] is True
    assert sum(oracle["selected_routes"].values()) == report["split"]["eval_queries"]
    assert oracle["eval_ndcg_at_10"] == 1.0
