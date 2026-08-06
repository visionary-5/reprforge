import numpy as np

from reprforge.rbrc_v0 import (
    DomainSurface,
    GdsfCache,
    LfuCache,
    LruCache,
    assess_calibrated_safety,
    belady_misses,
    compile_programs,
    orderings,
    replay_program,
)
from tools.evaluate_rbrc_v0 import evaluate


def _domain() -> DomainSurface:
    query_ids = ("q0", "q1", "q2", "q3")
    ranking = {
        "q0": ["a", "b", "c", "d"],
        "q1": ["a", "b", "e", "f"],
        "q2": ["a", "b", "g", "h"],
        "q3": ["a", "b", "i", "j"],
    }
    quality = {
        query_id: {2: 0.8, 4: 0.8} for query_id in query_ids
    }
    return DomainSurface(
        name="synthetic",
        corpus_pages=20,
        query_ids=query_ids,
        ranking=ranking,
        quality=quality,
        input_sha256={"synthetic": "0" * 64},
    )


def test_frozen_orderings_include_natural_then_deterministic_shuffles():
    first = orderings(("a", "b", "c"), random_orders=3, seed=7)
    second = orderings(("a", "b", "c"), random_orders=3, seed=7)
    assert first == second
    assert first[0] == ("natural", ("a", "b", "c"))
    assert [name for name, _ in first[1:]] == [
        "shuffle_000",
        "shuffle_001",
        "shuffle_002",
    ]


def test_guard_uses_reference_on_reuse_and_floor_when_reference_is_too_cold():
    result = replay_program(
        _domain(),
        ("q0", "q1", "q2", "q3"),
        program="guard",
        floor=2,
        reference_depth=4,
        cold_budget=2,
        capacity_fraction=0.5,
        cache_policy="lru",
    )
    assert result["actions"] == [2, 4, 4, 4]
    assert result["abstain_rate"] == 0.0
    assert result["mean_cold_page_misses"] == 2.0


def test_global_abstain_executes_and_charges_reference():
    result = replay_program(
        _domain(),
        _domain().query_ids,
        program="reference",
        floor=4,
        reference_depth=4,
        cold_budget=2,
        capacity_fraction=0.5,
        cache_policy="lru",
        force_global_abstain=True,
    )
    assert result["actions"] == [4, 4, 4, 4]
    assert result["abstain_rate"] == 1.0


def test_cache_baselines_and_belady_are_well_ordered_on_reuse_trace():
    trace = ["a", "b", "c", "a", "b", "d", "a", "b"]
    misses = []
    for cache_type in (LruCache, LfuCache, GdsfCache):
        cache = cache_type(2)
        misses.append(cache.access(trace))
    assert all(value >= belady_misses(trace, 2) for value in misses)


def test_safety_gate_uses_mean_and_query_violation_limits():
    safe = assess_calibrated_safety(
        np.zeros(100),
        epsilon_mean=0.001,
        epsilon_query=0.05,
        delta_empirical=0.05,
        delta_upper=0.10,
        confidence=0.95,
        bootstrap_samples=200,
        seed=1,
    )
    unsafe = assess_calibrated_safety(
        np.full(100, 0.06),
        epsilon_mean=0.001,
        epsilon_query=0.05,
        delta_empirical=0.05,
        delta_upper=0.10,
        confidence=0.95,
        bootstrap_samples=200,
        seed=1,
    )
    assert safe["passes"]
    assert not unsafe["passes"]


def test_compiler_selects_lowest_cost_safe_static_and_guard_programs():
    result = compile_programs(
        [_domain()],
        floors=(2,),
        reference_depth=4,
        cold_budget=2,
        capacity_fraction=0.5,
        cache_policy="lru",
        random_orders=2,
        order_seed=11,
        epsilon_mean=0.001,
        epsilon_query=0.05,
        delta_empirical=0.05,
        delta_upper=0.90,
        confidence=0.95,
        bootstrap_samples=200,
        bootstrap_seed=12,
    )
    assert result["selected"] == {
        "static": "static_top2",
        "static_floor": 2,
        "guard": "guard_top2",
        "guard_floor": 2,
    }


def test_unified_evaluator_reports_four_ablations_and_strong_cache_baselines():
    config = {
        "protocol_id": "synthetic",
        "quality_contract": {
            "mean_signed_regret_epsilon": 0.002,
            "query_violation_epsilon": 0.05,
            "allowed_empirical_violation_rate_delta": 0.05,
        },
        "online_contract": {
            "capacity_fraction": 0.5,
            "logical_cold_page_budget": 2,
            "primary_cache_policy": "lru",
        },
        "orders": {
            "blind_evaluation": {"random_permutations": 2, "seed": 3}
        },
        "representation_stacks": {
            "synthetic": {"primitive_plan_depths": [2, 4]}
        },
    }
    certificate = {
        "compiler_output": {
            "selected": {
                "static_floor": 2,
                "guard_floor": 2,
            }
        }
    }
    result = evaluate(_domain(), config, certificate, "synthetic")
    assert set(result["methods"]) == {
        "fixed_top50",
        "safety_only",
        "residency_only",
        "complete_rbrc",
    }
    assert set(result["fixed_top50_cache_baselines"]) == {"lru", "lfu", "gdsf"}
    assert len(result["methods"]["complete_rbrc"]["orders"]) == 3
    assert "gpu_physical_measurement_permitted" in result["blind_gate"]
