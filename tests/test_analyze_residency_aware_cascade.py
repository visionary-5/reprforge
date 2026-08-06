from tools.analyze_residency_aware_cascade import (
    _belady_misses,
    LruCache,
    _replay_actions,
    _replay_state_policy,
)


def _domain():
    ranking = {
        "q0": [f"d{i}" for i in range(100)],
        "q1": [f"d{i}" for i in range(100)],
    }
    quality = {
        query_id: {20: 0.2, 50: 0.5, 100: 1.0} for query_id in ranking
    }
    return {"ranking": ranking, "quality": quality}


def test_lru_counts_misses_and_retains_recent_items():
    cache = LruCache(100)
    assert cache.access(["a", "b"]) == 2
    assert cache.access(["b", "c"]) == 1
    assert cache.misses(["a", "b", "c"]) == 0


def test_state_policy_deepens_after_candidate_cohort_is_resident():
    result, actions = _replay_state_policy(
        _domain(), ["q0", "q1"], capacity=100, miss_budget=80
    )
    assert actions == [50, 100]
    assert result["mean_cold_page_misses"] == 50.0
    assert result["mean_ndcg_at_10"] == 0.75


def test_state_policy_can_freeze_plan_space_at_top50():
    _, actions = _replay_state_policy(
        _domain(), ["q0", "q1"], capacity=100, miss_budget=80, max_depth=50
    )
    assert actions == [50, 50]


def test_fixed_action_replay_uses_same_quality_surface():
    result = _replay_actions(_domain(), ["q0", "q1"], 100, [50, 50])
    assert result["mean_cold_page_misses"] == 25.0
    assert result["mean_ndcg_at_10"] == 0.5


def test_belady_oracle_uses_future_reuse():
    assert _belady_misses(["a", "b", "c", "a", "b"], capacity=2) == 4
