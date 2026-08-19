import pytest

from reprforge import BackboneProfile, CompilePlan, plan_topology_anchored


def test_plan_is_stable_and_serializable() -> None:
    profile = BackboneProfile("test", 18, 6, 16, 8)
    plan = plan_topology_anchored(profile, grid_shape=(4, 4))

    assert CompilePlan.from_dict(plan.to_dict()) == plan
    assert len(plan.fingerprint) == 64
    assert plan.profile.persistent_fraction == pytest.approx(0.5)


def test_planner_rejects_capacity_that_operator_cannot_lower() -> None:
    profile = BackboneProfile("test", 18, 6, 16, 12)
    with pytest.raises(ValueError, match="50% capacity"):
        plan_topology_anchored(profile, grid_shape=(4, 4))
