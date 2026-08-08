from tools.analyze_visual_feature_cache_phase import analyze


def test_exact_feature_cache_phase_uses_measured_costs_and_query_order() -> None:
    closure = {
        "domain": "toy",
        "pages": 100,
        "depths": {
            "2": {
                "candidate_events_per_evaluation_trace": 4,
                "policies": {
                    "history_frequency": {
                        "0.1": {
                            "persistent_pages": 10,
                            "transient_build_events_per_trace": 2,
                            "transient_build_events_per_query": [1, 1],
                        }
                    }
                },
            }
        },
    }
    dvi = {"verifier": {"page_end_to_end_seconds_mean": 1.0}}
    feature_cache = {
        "cache_build_end_to_end_ms": {"mean": 300.0},
        "cached_feature_h2d_and_language_ms": {"mean": 100.0},
        "mean_cached_feature_bytes": 64.0,
        "sample_pairs": 8,
        "maximum_score_absolute_difference": 0.0,
    }
    result = analyze(
        closure,
        dvi,
        feature_cache,
        policy="history_frequency",
        maximum_trace_replays=4,
    )
    budget = result["depths"]["2"]["budgets"]["0.1"]
    assert budget["winner_intervals"] == [
        {"start_trace": 1, "end_trace": 1, "winner": "raw_always_defer"},
        {
            "start_trace": 2,
            "end_trace": 4,
            "winner": "partial_visual_feature_cache",
        },
    ]
    assert budget["winner_query_intervals"][0] == {
        "start_query": 1,
        "end_query": 3,
        "winner": "raw_always_defer",
    }
    assert result["quality_contract"]["maximum_score_absolute_difference"] == 0.0
