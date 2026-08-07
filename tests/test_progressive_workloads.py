from reprforge.progressive_workloads import (
    broadening_trace,
    drift_trace,
    trace_suite,
    zipf_trace,
)


def test_trace_suite_is_deterministic_and_explicitly_synthetic():
    query_ids = ["q0", "q1", "q2", "q3"]
    groups = {"q0": "a", "q1": "a", "q2": "b", "q3": "b"}
    first = trace_suite(
        query_ids,
        groups,
        seed=7,
        random_permutations=2,
        horizon_multiplier=3,
        zipf_exponents=[1.0],
    )
    second = trace_suite(
        query_ids,
        groups,
        seed=7,
        random_permutations=2,
        horizon_multiplier=3,
        zipf_exponents=[1.0],
    )
    assert first == second
    assert first["dataset_order"] == query_ids
    assert len(first["zipf_1p0"]) == 12
    assert len(first["document_clustered"]) == 12
    assert set(first["mid_trace_distribution_drift"]) == set(query_ids)


def test_trace_helpers_validate_and_cover_expected_lengths():
    ids = ["a", "b", "c"]
    assert len(zipf_trace(ids, length=20, exponent=0.8, seed=1)) == 20
    assert len(broadening_trace(ids, length=20, seed=1)) == 20
    assert len(drift_trace(ids, length=20, seed=1)) == 20
