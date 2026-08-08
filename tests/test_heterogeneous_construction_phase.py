from tools.analyze_heterogeneous_construction_phase import analyze


def test_measured_costs_can_remove_a_logical_partial_region() -> None:
    closure = {
        "domain": "toy",
        "pages": 100,
        "evaluation_queries": 10,
        "depths": {
            "20": {
                "candidate_events_per_evaluation_trace": 200,
                "policies": {
                    "history_frequency": {
                        "0.1": {
                            "persistent_pages": 10,
                            "persistent_hit_fraction": 0.1,
                            "transient_build_events_per_trace": 180,
                        }
                    }
                },
            }
        },
    }
    construction = {
        "results": {"1": {"end_to_end_ms_per_page": {"mean": 250.0}}}
    }
    dvi = {"verifier": {"page_seconds_mean": 0.15}}
    result = analyze(
        closure,
        construction,
        dvi,
        full_build_seconds=100.0,
        policy="history_frequency",
        maximum_trace_replays=4,
        dvi_cost_field="page_seconds_mean",
    )
    budget = result["depths"]["20"]["budgets"]["0.1"]
    assert budget["closure_has_integer_winner"] is False
    assert budget["continuous_break_even_trace_replays"]["dvi_to_closure"] is None
    assert budget["winner_intervals"] == [
        {"start_trace": 1, "end_trace": 3, "winner": "dvi_always_defer"},
        {"start_trace": 4, "end_trace": 4, "winner": "full_ingestion"},
    ]


def test_partial_region_appears_when_persistent_hits_are_reused() -> None:
    closure = {
        "domain": "toy",
        "pages": 100,
        "evaluation_queries": 10,
        "depths": {
            "20": {
                "candidate_events_per_evaluation_trace": 200,
                "policies": {
                    "history_frequency": {
                        "0.4": {
                            "persistent_pages": 40,
                            "persistent_hit_fraction": 0.92,
                            "transient_build_events_per_trace": 16,
                        }
                    }
                },
            }
        },
    }
    construction = {
        "results": {"1": {"end_to_end_ms_per_page": {"mean": 250.0}}}
    }
    dvi = {"verifier": {"page_seconds_mean": 0.06}}
    result = analyze(
        closure,
        construction,
        dvi,
        full_build_seconds=25.0,
        policy="history_frequency",
        maximum_trace_replays=4,
        dvi_cost_field="page_seconds_mean",
    )
    budget = result["depths"]["20"]["budgets"]["0.4"]
    assert budget["closure_has_integer_winner"] is True
    assert budget["winner_intervals"] == [
        {"start_trace": 1, "end_trace": 1, "winner": "dvi_always_defer"},
        {
            "start_trace": 2,
            "end_trace": 3,
            "winner": "closure_materialization",
        },
        {"start_trace": 4, "end_trace": 4, "winner": "full_ingestion"},
    ]
