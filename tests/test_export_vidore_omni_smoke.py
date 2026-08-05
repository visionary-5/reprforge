from tools.export_vidore_omni_smoke import select_queries


def test_select_queries_is_deterministic_and_qrel_complete():
    queries = [
        {"query_id": 3, "query": "third"},
        {"query_id": 1, "query": "first"},
        {"query_id": 2, "query": "second"},
    ]
    qrels = [
        {"query_id": 1, "corpus_id": 10, "score": 2},
        {"query_id": 1, "corpus_id": 11, "score": 1},
        {"query_id": 2, "corpus_id": 12, "score": 1},
        {"query_id": 3, "corpus_id": 13, "score": 1},
    ]

    selected, pages = select_queries(queries, qrels, max_queries=2, max_pages=3)

    assert [row["query_id"] for row in selected] == [1, 2]
    assert pages == {10, 11, 12}


def test_select_queries_skips_query_that_would_exceed_page_budget():
    queries = [{"query_id": 1}, {"query_id": 2}, {"query_id": 3}]
    qrels = [
        {"query_id": 1, "corpus_id": 10, "score": 1},
        {"query_id": 1, "corpus_id": 11, "score": 1},
        {"query_id": 2, "corpus_id": 12, "score": 1},
        {"query_id": 2, "corpus_id": 13, "score": 1},
        {"query_id": 3, "corpus_id": 10, "score": 1},
    ]

    selected, pages = select_queries(queries, qrels, max_queries=2, max_pages=2)

    assert [row["query_id"] for row in selected] == [1, 3]
    assert pages == {10, 11}
