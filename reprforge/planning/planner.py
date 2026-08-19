"""Physical-plan construction."""

from __future__ import annotations

from .models import BackboneProfile, CompilePlan


def plan_topology_anchored(
    profile: BackboneProfile,
    *,
    grid_shape: tuple[int, int],
) -> CompilePlan:
    """Create and validate the current ReprForge physical plan."""

    plan = CompilePlan(profile=profile, grid_shape=grid_shape)
    plan.validate()
    return plan
