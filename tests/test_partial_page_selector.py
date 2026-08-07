from reprforge.partial_page_selector import (
    PageRiskFeatures,
    budget_count,
    selection_order,
)


def _rows():
    return [
        PageRiskFeatures("a", 0, 0.1, 0.1, 0.1),
        PageRiskFeatures("b", 10, 0.9, 0.1, 0.1),
        PageRiskFeatures("c", 20, 0.2, 0.9, 0.1),
        PageRiskFeatures("d", 30, 0.8, 0.8, 0.1),
        PageRiskFeatures("e", 40, 0.3, 0.3, 0.1),
        PageRiskFeatures("f", 50, 0.4, 0.4, 0.1),
    ]


def test_risk_cover_is_complete_deterministic_and_nested():
    first = selection_order(_rows(), strategy="risk_cover_round_robin", seed=7)
    second = selection_order(list(reversed(_rows())), strategy="risk_cover_round_robin", seed=7)
    assert first == second
    assert len(first) == len(set(first)) == 6
    assert set(first) == {"a", "b", "c", "d", "e", "f"}
    assert first[: budget_count(6, 0.5)] == first[:3]


def test_baselines_and_budget_rounding_are_stable():
    assert selection_order(_rows(), strategy="text_scarcity", seed=7)[0] == "a"
    assert selection_order(_rows(), strategy="sha256_random", seed=7) == selection_order(
        list(reversed(_rows())), strategy="sha256_random", seed=7
    )
    assert budget_count(11, 0.1) == 2
