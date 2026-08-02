from pathlib import Path

import pytest

from reprforge.representation_views import (
    CandidateView,
    MaterializationPlan,
    RepresentationViewCatalog,
    ViewKey,
    ViewState,
    apply_materialization_plan,
)


def _view(
    item: str,
    route: str,
    *,
    utility: float,
    uncertainty: float = 0.1,
    reuse: float = 1.0,
    probe: float = 1.0,
    build: float = 10.0,
    storage: int = 100,
) -> CandidateView:
    return CandidateView(
        key=ViewKey(item, route),
        slot="retrieval",
        parent_route="text",
        expected_utility=utility,
        uncertainty=uncertainty,
        expected_reuse=reuse,
        probe_cost_ms=probe,
        build_cost_ms=build,
        storage_bytes=storage,
    )


def test_probe_verify_materialize_lifecycle_and_snapshot(tmp_path: Path) -> None:
    catalog = RepresentationViewCatalog(
        [
            _view("a", "full", utility=0.8, reuse=4),
            _view("b", "full", utility=0.2, reuse=1),
        ]
    )
    plan = catalog.plan_probes(budget_ms=1.0)
    assert plan.keys == (ViewKey("a", "full"),)
    catalog.begin_probes(plan)
    state = catalog.finish_probe(
        plan.keys[0],
        observed_utility=0.7,
        actual_cost_ms=1.0,
        minimum_utility=0.1,
        artifact_reusable=True,
    )
    assert state == ViewState.VERIFIED
    materialization = catalog.plan_materialization(
        build_budget_ms=9.0,
        storage_budget_bytes=100,
    )
    assert materialization.keys == plan.keys
    assert materialization.estimated_build_ms == 9.0
    apply_materialization_plan(
        catalog,
        materialization,
        versions={plan.keys[0]: 1},
    )
    assert catalog.get(plan.keys[0]).state == ViewState.MATERIALIZED
    assert catalog.get(plan.keys[0]).materialized_version == 1

    path = tmp_path / "catalog.json"
    catalog.save(path)
    restored = RepresentationViewCatalog.load(path)
    assert restored.to_payload() == catalog.to_payload()


def test_probe_budget_and_exploration_are_deterministic() -> None:
    catalog = RepresentationViewCatalog(
        [
            _view("a", "full", utility=0.1, uncertainty=1.0, probe=2.0),
            _view("b", "full", utility=0.5, uncertainty=0.0, probe=2.0),
            _view("c", "full", utility=0.4, uncertainty=0.0, probe=2.0),
        ]
    )
    exploit = catalog.plan_probes(budget_ms=4.0, exploration_weight=0.0)
    explore = catalog.plan_probes(budget_ms=4.0, exploration_weight=1.0)
    assert exploit.keys == (ViewKey("b", "full"), ViewKey("c", "full"))
    assert explore.keys == (ViewKey("a", "full"), ViewKey("b", "full"))
    assert explore.estimated_cost_ms <= explore.budget_ms


def test_materialization_selects_one_route_per_item_and_obeys_budgets() -> None:
    catalog = RepresentationViewCatalog(
        [
            _view("a", "pool", utility=0.5, build=4.0, storage=40),
            _view("a", "full", utility=0.8, build=8.0, storage=80),
            _view("b", "full", utility=0.7, build=8.0, storage=80),
        ]
    )
    probes = catalog.plan_probes(budget_ms=3.0)
    catalog.begin_probes(probes)
    for key in probes.keys:
        catalog.finish_probe(
            key,
            observed_utility={
                ViewKey("a", "pool"): 0.5,
                ViewKey("a", "full"): 0.8,
                ViewKey("b", "full"): 0.7,
            }[key],
            actual_cost_ms=1.0,
            minimum_utility=0.0,
            artifact_reusable=False,
        )
    plan = catalog.plan_materialization(
        build_budget_ms=12.0,
        storage_budget_bytes=120,
    )
    assert len({key.item_id for key in plan.keys}) == len(plan.keys)
    assert plan.estimated_build_ms <= 12.0
    assert plan.storage_bytes <= 120


def test_rejected_and_interrupted_views_do_not_become_visible() -> None:
    catalog = RepresentationViewCatalog(
        [
            _view("a", "full", utility=0.8),
            _view("b", "full", utility=0.7),
        ]
    )
    probes = catalog.plan_probes(budget_ms=2.0)
    catalog.begin_probes(probes)
    catalog.finish_probe(
        ViewKey("a", "full"),
        observed_utility=-0.2,
        actual_cost_ms=1.0,
        minimum_utility=0.0,
        artifact_reusable=False,
    )
    assert catalog.get(ViewKey("a", "full")).state == ViewState.REJECTED
    assert catalog.recover_interrupted_work() == 1
    assert catalog.get(ViewKey("b", "full")).state == ViewState.HYPOTHETICAL
    assert catalog.summary()["states"]["materialized"] == 0


def test_materialization_result_must_match_plan_atomically() -> None:
    catalog = RepresentationViewCatalog([_view("a", "full", utility=1.0)])
    probe = catalog.plan_probes(budget_ms=1.0)
    catalog.begin_probes(probe)
    catalog.finish_probe(
        probe.keys[0],
        observed_utility=1.0,
        actual_cost_ms=1.0,
        minimum_utility=0.0,
        artifact_reusable=False,
    )
    plan = catalog.plan_materialization(
        build_budget_ms=10.0,
        storage_budget_bytes=100,
    )
    with pytest.raises(ValueError, match="differ from plan"):
        apply_materialization_plan(catalog, plan, versions={})
    assert catalog.get(probe.keys[0]).state == ViewState.VERIFIED

    with pytest.raises(ValueError, match="positive"):
        apply_materialization_plan(
            catalog,
            plan,
            versions={probe.keys[0]: 0},
        )
    assert catalog.get(probe.keys[0]).state == ViewState.VERIFIED

    invalid = MaterializationPlan(
        keys=(ViewKey("missing", "full"),),
        estimated_build_ms=0.0,
        storage_bytes=0,
        build_budget_ms=0.0,
        storage_budget_bytes=0,
    )
    with pytest.raises(KeyError, match="unknown representation view"):
        catalog.begin_materialization(invalid)
