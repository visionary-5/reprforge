import json

import numpy as np
import pytest

from reprforge.materialization import (
    CostCatalog,
    MaterializationAction,
    PageSignals,
    PageState,
    PolicyConfig,
    compile_plan,
    load_frozen_split,
    replay_feature_policy,
)


def _costs() -> CostCatalog:
    return CostCatalog(
        raw_query_seconds=1.0,
        feature_query_seconds=0.2,
        feature_build_seconds=0.6,
        feature_write_seconds=0.1,
        feature_bytes=10.0,
        retrieval_build_seconds=2.0,
        retrieval_bytes=20.0,
    )


def test_page_states_keep_feature_and_retrieval_actions_orthogonal():
    raw = PageState()
    feature = raw.apply(MaterializationAction.FEATURE)
    retrieval = raw.apply(MaterializationAction.RETRIEVAL)
    assert feature.label == "feature"
    assert retrieval.label == "retrieval"
    assert feature.apply(MaterializationAction.RETRIEVAL).label == "feature+retrieval"


def test_cost_catalog_reports_measured_break_even():
    costs = _costs()
    assert costs.feature_saving_per_use == pytest.approx(0.8)
    assert costs.feature_break_even_future_uses == pytest.approx(0.875)
    assert costs.feature_net_value(2.0) == pytest.approx(0.9)


def test_frozen_split_separates_fit_calibration_and_test(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "assignment": "test",
                "seed": 7,
                "evaluation_fold": 0,
                "queries": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 0},
            }
        )
    )
    split = load_frozen_split(path, ["a", "b", "c", "d", "e"])
    assert split.fit == (0, 1, 2)
    assert split.calibration == (3,)
    assert split.test == (4,)


def test_joint_policy_uses_reuse_for_features_and_risk_for_retrieval():
    signals = PageSignals(
        page_ids=np.arange(4, dtype=np.int32),
        fit_candidate_events=np.asarray([8, 1, 0, 0], dtype=np.float64),
        text_chars=np.asarray([100, 100, 1, 100], dtype=np.float64),
        grayscale_entropy=np.asarray([0.1, 0.1, 1.0, 0.1]),
        edge_energy=np.asarray([0.1, 0.1, 1.0, 0.1]),
        locator_disagreement=np.asarray([0.0, 0.0, 1.0, 0.0]),
    )
    plan, diagnostics = compile_plan(
        signals,
        _costs(),
        fit_queries=10,
        horizon_queries=10,
        config=PolicyConfig(
            feature_budget_fraction=0.25,
            retrieval_budget_fraction=0.25,
            prior_query_strength=0.0,
        ),
    )
    assert plan.feature_pages == (0,)
    assert plan.retrieval_pages == (2,)
    assert diagnostics["feature_net_seconds"][0] > 0.0
    assert plan.protocol["future_relevance_visible"] is False


def test_replay_first_touch_promotes_after_raw_score_and_reuses_exact_feature():
    candidates = np.asarray([[0, 1], [0, 2], [0, 1]], dtype=np.int32)
    result = replay_feature_policy(
        candidates,
        [0, 1, 2],
        _costs(),
        capacity_pages=1,
        policy="first_touch",
    )
    assert result["online_promotions"] == 1
    assert result["feature_hit_fraction"] == pytest.approx(2 / 6)
    assert result["total_seconds"] == pytest.approx(4.5)
    assert result["quality_invariant_to_feature_storage"] is True


def test_static_replay_charges_offline_build_before_queries():
    candidates = np.asarray([[0, 1], [0, 2]], dtype=np.int32)
    result = replay_feature_policy(
        candidates,
        [0, 1],
        _costs(),
        capacity_pages=1,
        policy="static",
        initial_pages=[0],
    )
    assert result["initial_materialization_seconds"] == pytest.approx(0.7)
    assert result["total_seconds"] == pytest.approx(3.1)
