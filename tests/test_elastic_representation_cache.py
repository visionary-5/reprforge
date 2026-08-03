from __future__ import annotations

import numpy as np
import pytest

from reprforge.elastic_representation_cache import (
    offline_oracle,
    replay_capacity_cache,
    replay_elastic_cache,
)


def test_replay_separates_recompute_and_residency_cost() -> None:
    requests = [[0], [], [0]]
    build = [10.0]
    holding = [2.0]

    no_cache = replay_elastic_cache(
        requests, build, holding, policy="no_cache"
    )
    resident = replay_elastic_cache(
        requests, build, holding, policy="resident"
    )
    ski = replay_elastic_cache(requests, build, holding, policy="ski_ttl")

    assert no_cache.total_cost == 20.0
    assert no_cache.cache_misses == 2
    assert resident.total_cost == 14.0
    assert resident.cache_hits == 1
    assert ski.total_cost == 14.0
    assert ski.cache_hits == 1


def test_ski_ttl_expires_after_break_even_and_is_two_competitive_on_gap() -> None:
    requests = [[0], [], [], [], [0]]
    ski = replay_elastic_cache(
        requests, [3.0], [1.0], policy="ski_ttl"
    )
    oracle = offline_oracle(requests, [3.0], [1.0])

    # Ski pays three intervals of rent, then rebuilds.  The oracle evicts
    # immediately and only rebuilds at the second request.
    assert ski.build_cost == 6.0
    assert ski.holding_cost == 3.0
    assert ski.total_cost == 9.0
    assert oracle.total_cost == 6.0
    assert ski.total_cost <= 2.0 * oracle.total_cost


def test_zero_holding_cost_retains_without_expiry() -> None:
    requests = [[0], [], [], [0]]
    ski = replay_elastic_cache(
        requests, [5.0], [0.0], policy="ski_ttl"
    )
    resident = replay_elastic_cache(
        requests, [5.0], [0.0], policy="resident"
    )
    assert ski.total_cost == resident.total_cost == 5.0
    assert ski.cache_hits == resident.cache_hits == 1


def test_verified_ski_requires_reuse_evidence_before_admission() -> None:
    requests = [[0, 1], [0], [0]]
    result = replay_elastic_cache(
        requests, [5.0, 5.0], [1.0, 1.0], policy="verified_ski_ttl"
    )
    # Page 1 is a one-off scan and is never retained. Page 0 is admitted on
    # its second access and hits on its third.
    assert result.cache_misses == 3
    assert result.cache_hits == 1
    assert result.peak_resident_items == 1
    assert result.build_cost == 15.0
    assert result.holding_cost == 1.0


def test_duplicate_request_within_query_is_charged_once() -> None:
    result = replay_elastic_cache(
        [[0, 0, 1]], [2.0, 3.0], [1.0, 1.0], policy="no_cache"
    )
    assert result.request_count == 2
    assert result.build_cost == 5.0


def test_offline_oracle_is_no_worse_than_online_policies() -> None:
    requests = [[0, 1], [0], [2], [1], [0, 2]]
    build = np.asarray([5.0, 7.0, 3.0])
    holding = np.asarray([2.0, 1.0, 4.0])
    oracle = offline_oracle(requests, build, holding)
    for policy in (
        "no_cache",
        "resident",
        "ski_ttl",
        "verified_ski_ttl",
    ):
        result = replay_elastic_cache(
            requests, build, holding, policy=policy
        )
        assert oracle.total_cost <= result.total_cost + 1e-12


def test_invalid_cost_or_request_is_rejected() -> None:
    with pytest.raises(ValueError):
        replay_elastic_cache([[1]], [1.0], [1.0], policy="no_cache")
    with pytest.raises(ValueError):
        offline_oracle([[0]], [-1.0], [1.0])


def test_capacity_lru_evicts_least_recent_item() -> None:
    result = replay_capacity_cache(
        [[0], [1], [0], [2], [1]],
        [1.0, 10.0, 1.0],
        [0.0, 0.0, 0.0],
        [1, 1, 1],
        capacity_bytes=2,
        eviction_policy="lru",
        ttl_policy="none",
    )
    assert result.cache_hits == 1
    assert result.cache_misses == 4
    assert result.build_cost == 22.0
    assert result.peak_resident_bytes == 2


def test_gdsf_retains_high_refault_cost_item() -> None:
    result = replay_capacity_cache(
        [[0], [1], [0], [2], [1]],
        [1.0, 10.0, 1.0],
        [0.0, 0.0, 0.0],
        [1, 1, 1],
        capacity_bytes=2,
        eviction_policy="gdsf",
        ttl_policy="none",
    )
    assert result.cache_hits == 2
    assert result.cache_misses == 3
    assert result.build_cost == 12.0


def test_capacity_breakeven_matches_unbounded_ski_when_capacity_is_large() -> None:
    requests = [[0], [], [0], [1]]
    build = [10.0, 4.0]
    holding = [2.0, 1.0]
    unbounded = replay_elastic_cache(
        requests, build, holding, policy="ski_ttl"
    )
    bounded = replay_capacity_cache(
        requests,
        build,
        holding,
        [1, 1],
        capacity_bytes=2,
        eviction_policy="gdsf",
        ttl_policy="breakeven",
    )
    assert bounded.cache_hits == unbounded.cache_hits
    assert bounded.cache_misses == unbounded.cache_misses
    assert bounded.total_cost == unbounded.total_cost


def test_randomized_ttl_is_reproducible_for_a_seed() -> None:
    arguments = (
        [[0], [], [0], [], [0]],
        [3.0],
        [1.0],
        [1],
    )
    left = replay_capacity_cache(
        *arguments,
        capacity_bytes=1,
        eviction_policy="gdsf",
        ttl_policy="randomized",
        random_seed=7,
    )
    right = replay_capacity_cache(
        *arguments,
        capacity_bytes=1,
        eviction_policy="gdsf",
        ttl_policy="randomized",
        random_seed=7,
    )
    assert left == right
