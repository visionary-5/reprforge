from reprforge.progressive_selection import MaterializationFeatures, selection_order


def _rows():
    return [
        MaterializationFeatures("a0", "a", 0, 0.1, 0.2, 0.0, 1),
        MaterializationFeatures("a1", "a", 5, 0.9, 0.2, 0.5, 8),
        MaterializationFeatures("b0", "b", 10, 0.3, 0.9, 0.8, 2),
        MaterializationFeatures("b1", "b", 15, 0.4, 0.4, 0.1, 7),
    ]


def test_all_realizable_orders_are_complete_deterministic_and_nested():
    strategies = [
        "sha256_random",
        "document_uniform",
        "text_scarcity",
        "visual_complexity",
        "cheap_locator_disagreement",
        "history_candidate_frequency",
        "risk_cover_plus_history_benefit",
    ]
    for strategy in strategies:
        first = selection_order(_rows(), strategy=strategy, seed=7)
        second = selection_order(list(reversed(_rows())), strategy=strategy, seed=7)
        assert first == second
        assert len(first) == len(set(first)) == 4


def test_document_uniform_covers_documents_before_second_pages():
    order = selection_order(_rows(), strategy="document_uniform", seed=7)
    assert order[:2] == ["a0", "b0"]
    assert selection_order(_rows(), strategy="history_candidate_frequency", seed=7)[0] == "a1"
