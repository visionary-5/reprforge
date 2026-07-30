from reprforge.mmdocir_data import (
    normalize_layout_type,
    normalize_layouts,
    normalize_document_queries,
    normalize_pages,
    normalize_queries,
)


def test_layout_type_aliases_follow_reprforge_vocabulary() -> None:
    assert normalize_layout_type("image") == "image"
    assert normalize_layout_type("Equation") == "formula"
    assert normalize_layout_type("paragraph") == "text"
    assert normalize_layout_type("section header") == "section-header"


def test_normalization_preserves_source_order_and_official_ocr_typo() -> None:
    pages = normalize_pages(
        [
            {"oct_text": "ocr-a", "vlm_text": "vlm-a", "image_binary": True},
            {"ocr_text": "ocr-b", "vlm_text": "vlm-b"},
        ]
    )
    assert pages[0] == {
        "item_id": "page:0",
        "source_row": 0,
        "ocr_text": "ocr-a",
        "vlm_text": "vlm-a",
        "has_image": True,
    }
    assert pages[1]["item_id"] == "page:1"
    assert pages[1]["ocr_text"] == "ocr-b"


def test_layout_queries_encode_official_overlap_recall() -> None:
    layouts = normalize_layouts(
        [
            {
                "type": "table",
                "page_id": 0,
                "bbox": [0.0, 0.0, 0.5, 1.0],
                "text": "table",
            },
            {
                "type": "text",
                "page_id": 0,
                "bbox": [0.5, 0.0, 1.0, 1.0],
                "text": "paragraph",
            },
            {
                "type": "image",
                "page_id": 1,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "text": "",
            },
        ]
    )
    annotations = [
        {
            "domain": "Academic paper",
            "page_indices": [0, 1],
            "layout_indices": [0, 2],
            "questions": [
                {
                    "Q": "Which row contains the answer?",
                    "page_id": [0],
                    "layout_mapping": [
                        {"page": 0, "bbox": [0.25, 0.0, 0.75, 1.0]}
                    ],
                }
            ],
        }
    ]
    page_queries, layout_queries = normalize_queries(annotations, layouts)
    assert page_queries[0]["relevance"] == {"page:0": 1.0}
    assert layout_queries[0]["relevance"] == {
        "layout:0": 0.25,
        "layout:1": 0.25,
    }
    assert layout_queries[0]["relevance_denominator"] == 0.5


def test_bounded_document_normalization_preserves_absolute_ids() -> None:
    layouts = normalize_layouts(
        [
            {
                "type": "table",
                "page_id": 0,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "text": "",
            },
            {
                "type": "text",
                "page_id": 1,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "text": "answer",
            },
        ],
        source_start=40,
    )
    document = {
        "domain": "Government",
        "page_indices": [10, 11],
        "layout_indices": [40, 41],
        "questions": [
            {
                "Q": "Where is the answer?",
                "page_id": [1],
                "layout_mapping": [{"page": 1, "bbox": [0, 0, 1, 1]}],
            }
        ],
    }
    _, queries = normalize_document_queries(
        document,
        {int(row["source_row"]): row for row in layouts},
        document_index=7,
        query_start=19,
    )
    assert [row["item_id"] for row in layouts] == ["layout:40", "layout:41"]
    assert queries[0]["query_id"] == "query:19"
    assert queries[0]["candidate_item_ids"] == ["layout:40", "layout:41"]
    assert queries[0]["relevance"] == {"layout:41": 1.0}
