import pytest

from reprforge import Lifecycle, WorkloadProfile, choose_lifecycle


def test_workload_policy_selects_refinement_only_for_cold_index() -> None:
    cold = WorkloadProfile(
        full_build_seconds=209.2,
        compact_build_seconds=171.8,
        refinement_seconds_per_cold_query=1.97,
        cold_queries_per_build=10,
        full_quality=0.9212,
        compact_quality=0.9098,
        refined_quality=0.9221,
        compact_storage_fraction=0.503,
    )
    hot = WorkloadProfile(**{**cold.__dict__, "cold_queries_per_build": 100})

    assert choose_lifecycle(cold).lifecycle is Lifecycle.COMPACT_REFINE
    assert choose_lifecycle(hot).lifecycle is Lifecycle.FULL
    assert cold.refinement_break_even_queries == pytest.approx(18.9848, rel=1e-3)


def test_policy_can_select_compact_only_at_lower_quality_floor() -> None:
    workload = WorkloadProfile(
        full_build_seconds=100,
        compact_build_seconds=70,
        refinement_seconds_per_cold_query=2,
        cold_queries_per_build=5,
        full_quality=1.0,
        compact_quality=0.98,
        refined_quality=1.0,
        compact_storage_fraction=0.5,
    )

    assert (
        choose_lifecycle(workload, minimum_quality_fraction=0.97).lifecycle
        is Lifecycle.COMPACT
    )
