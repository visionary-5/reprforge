from reprforge.select_mmdocir_expansion import select_documents


def _document(index: int, domain: str, layouts: int, questions: int = 2):
    return {
        "doc_name": f"doc-{index}",
        "domain": domain,
        "layout_indices": [index * 100, index * 100 + layouts - 1],
        "questions": [{} for _ in range(questions)],
    }


def test_selection_is_deterministic_bounded_and_domain_aware() -> None:
    annotations = [
        _document(0, "a", 10),
        _document(1, "a", 20),
        _document(2, "a", 30),
        _document(3, "b", 40),
        _document(4, "b", 50),
        _document(5, "b", 500),
        _document(6, "c", 60, questions=1),
        _document(7, "c", 70),
    ]
    first = select_documents(
        annotations,
        target_documents=6,
        max_layouts_per_new_document=100,
        minimum_questions=2,
        fixed_documents=(0, 3, 7),
    )
    second = select_documents(
        annotations,
        target_documents=6,
        max_layouts_per_new_document=100,
        minimum_questions=2,
        fixed_documents=(0, 3, 7),
    )
    assert first == second
    assert len(first["document_indices"]) == 6
    assert first["domain_counts"] == {"a": 3, "b": 2, "c": 1}
    assert first["uses_retrieval_labels_or_scores"] is False
    assert 5 not in first["new_document_indices"]
    assert 6 not in first["new_document_indices"]


def test_fixed_documents_are_retained_even_above_new_layout_bound() -> None:
    annotations = [
        _document(0, "a", 500),
        _document(1, "a", 10),
    ]
    result = select_documents(
        annotations,
        target_documents=2,
        max_layouts_per_new_document=100,
        minimum_questions=2,
        fixed_documents=(0,),
    )
    assert result["document_indices"] == [0, 1]
    assert result["layout_count"] == 510
