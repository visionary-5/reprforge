#!/usr/bin/env python3
"""Normalize the public MMDocIR evaluation data for ReprForge.

MMDocIR already supplies the page/layout boundaries and the representation
substrates that ReprForge needs. This adapter deliberately preserves the
official row ordering because the annotation file stores inclusive page and
layout ranges into those Parquet tables.

The adapter does not copy image payloads. Model runners should read the pinned
source Parquet directly by ``source_row``; this keeps the normalized metadata
small and avoids silently duplicating tens of gigabytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


OFFICIAL_DATASET = "MMDocIR/MMDocIR_Evaluation_Dataset"
OFFICIAL_CODE = "https://github.com/MMDocRAG/MMDocIR"


def normalize_layout_type(raw: str) -> str:
    value = raw.strip().lower().replace("_", " ")
    aliases = {
        "fig": "figure",
        "image": "image",
        "picture": "image",
        "chart": "chart",
        "table": "table",
        "equation": "formula",
        "formula": "formula",
        "text": "text",
        "paragraph": "text",
        "title": "title",
    }
    return aliases.get(value, value.replace(" ", "-") or "unknown")


def _jsonable_bbox(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"expected a four-coordinate bbox, got {value!r}")
    return [float(coordinate) for coordinate in value]


def _intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    top = max(first[0], second[0])
    left = max(first[1], second[1])
    bottom = min(first[2], second[2])
    right = min(first[3], second[3])
    return max(0.0, bottom - top) * max(0.0, right - left)


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def normalize_pages(rows: Iterable[Mapping]) -> list[dict]:
    normalized = []
    for index, row in enumerate(rows):
        normalized.append(
            {
                "item_id": f"page:{index}",
                "source_row": index,
                "ocr_text": str(row.get("oct_text") or row.get("ocr_text") or ""),
                "vlm_text": str(row.get("vlm_text") or ""),
                "has_image": bool(row.get("image_binary") is not None),
            }
        )
    return normalized


def normalize_layouts(
    rows: Iterable[Mapping],
    *,
    source_start: int = 0,
) -> list[dict]:
    normalized = []
    for local_index, row in enumerate(rows):
        index = source_start + local_index
        raw_type = str(row.get("type") or "unknown")
        normalized.append(
            {
                "item_id": f"layout:{index}",
                "source_row": index,
                "page_id": int(row["page_id"]),
                "bbox": _jsonable_bbox(row["bbox"]),
                "raw_content_type": raw_type,
                "content_type": normalize_layout_type(raw_type),
                "native_text": str(row.get("text") or ""),
                "ocr_text": str(row.get("oct_text") or row.get("ocr_text") or ""),
                "vlm_text": str(row.get("vlm_text") or ""),
                "has_image": bool(row.get("image_binary") is not None),
            }
        )
    return normalized


def normalize_document_queries(
    document: Mapping,
    layouts_by_source_row: Mapping[int, Mapping],
    *,
    document_index: int,
    query_start: int,
) -> tuple[list[dict], list[dict]]:
    """Normalize one document without materializing the full layout table.

    MMDocIR annotations store absolute inclusive row ranges.  A bounded GPU
    experiment should therefore be able to read only the selected Parquet row
    groups while preserving those absolute identifiers and the official
    within-document candidate pool.
    """

    page_queries: list[dict] = []
    layout_queries: list[dict] = []
    page_start, page_end = (int(value) for value in document["page_indices"])
    layout_start, layout_end = (int(value) for value in document["layout_indices"])
    expected_layouts = set(range(layout_start, layout_end + 1))
    missing_layouts = expected_layouts - set(layouts_by_source_row)
    if missing_layouts:
        raise ValueError(
            "selected document is missing normalized layouts: "
            f"{sorted(missing_layouts)[:5]}"
        )

    for local_query_index, qa in enumerate(document["questions"]):
        query_index = query_start + local_query_index
        query_id = f"query:{query_index}"
        common = {
            "query_id": query_id,
            "source_query_index": query_index,
            "document_index": document_index,
            "domain": str(document.get("domain") or ""),
            "query": str(qa["Q"]),
        }
        page_offsets = [int(value) for value in qa["page_id"]]
        page_relevance = {
            f"page:{page_start + offset}": 1.0 for offset in page_offsets
        }
        page_queries.append(
            {
                **common,
                "candidate_start": f"page:{page_start}",
                "candidate_end": f"page:{page_end}",
                "candidate_item_ids": [
                    f"page:{index}" for index in range(page_start, page_end + 1)
                ],
                "relevance": page_relevance,
            }
        )

        gold_layouts = qa.get("layout_mapping") or []
        relevance: dict[str, float] = {}
        denominator = 0.0
        for gold in gold_layouts:
            gold_bbox = _jsonable_bbox(gold["bbox"])
            denominator += _bbox_area(gold_bbox)
            gold_page = int(gold["page"])
            for candidate_index in range(layout_start, layout_end + 1):
                candidate = layouts_by_source_row[candidate_index]
                if int(candidate["page_id"]) != gold_page:
                    continue
                overlap = _intersection_area(candidate["bbox"], gold_bbox)
                if overlap > 0:
                    item_id = str(candidate["item_id"])
                    relevance[item_id] = relevance.get(item_id, 0.0) + overlap
        if denominator > 0 and relevance:
            layout_queries.append(
                {
                    **common,
                    "candidate_start": f"layout:{layout_start}",
                    "candidate_end": f"layout:{layout_end}",
                    "candidate_item_ids": [
                        f"layout:{index}"
                        for index in range(layout_start, layout_end + 1)
                    ],
                    "relevance": relevance,
                    "relevance_denominator": denominator,
                }
            )
    return page_queries, layout_queries


def normalize_queries(
    annotations: Iterable[Mapping],
    layouts: Sequence[Mapping],
) -> tuple[list[dict], list[dict]]:
    """Return page and layout replay queries using official MMDocIR semantics."""

    page_queries: list[dict] = []
    layout_queries: list[dict] = []
    query_index = 0
    layouts_by_source_row = {
        int(layout["source_row"]): layout for layout in layouts
    }
    for document_index, document in enumerate(annotations):
        document_pages, document_layouts = normalize_document_queries(
            document,
            layouts_by_source_row,
            document_index=document_index,
            query_start=query_index,
        )
        page_queries.extend(document_pages)
        layout_queries.extend(document_layouts)
        query_index += len(document["questions"])
    return page_queries, layout_queries


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_parquet_metadata(path: Path, columns: Sequence[str]) -> list[dict]:
    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    selected = [column for column in columns if column in available]
    return pq.read_table(path, columns=selected).to_pylist()


def prepare(
    *,
    pages_parquet: Path,
    layouts_parquet: Path,
    annotations_jsonl: Path,
    dataset_revision: str,
    output: Path,
) -> dict:
    if not dataset_revision.strip() or dataset_revision == "main":
        raise ValueError("dataset_revision must be an immutable Hugging Face revision")
    page_rows = _read_parquet_metadata(
        pages_parquet,
        ("oct_text", "ocr_text", "vlm_text"),
    )
    layout_rows = _read_parquet_metadata(
        layouts_parquet,
        ("type", "bbox", "page_id", "text", "oct_text", "ocr_text", "vlm_text"),
    )
    # Record image availability from the schema without loading binary columns.
    import pyarrow.parquet as pq

    pages_have_images = "image_binary" in pq.read_schema(pages_parquet).names
    layouts_have_images = "image_binary" in pq.read_schema(layouts_parquet).names
    if pages_have_images:
        for row in page_rows:
            row["image_binary"] = True
    if layouts_have_images:
        for row in layout_rows:
            row["image_binary"] = True

    annotations = [
        json.loads(line)
        for line in annotations_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pages = normalize_pages(page_rows)
    layouts = normalize_layouts(layout_rows)
    page_queries, layout_queries = normalize_queries(annotations, layouts)

    _write_jsonl(output / "pages.jsonl", pages)
    _write_jsonl(output / "layouts.jsonl", layouts)
    _write_jsonl(output / "page_queries.jsonl", page_queries)
    _write_jsonl(output / "layout_queries.jsonl", layout_queries)
    manifest = {
        "dataset": OFFICIAL_DATASET,
        "dataset_revision": dataset_revision,
        "official_code": OFFICIAL_CODE,
        "source": {
            "pages_parquet": str(pages_parquet.resolve()),
            "layouts_parquet": str(layouts_parquet.resolve()),
            "annotations_jsonl": str(annotations_jsonl.resolve()),
        },
        "pages": len(pages),
        "layouts": len(layouts),
        "page_queries": len(page_queries),
        "layout_queries_with_overlap": len(layout_queries),
        "documents": len(annotations),
        "image_payloads_copied": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-parquet", type=Path, required=True)
    parser.add_argument("--layouts-parquet", type=Path, required=True)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument(
        "--dataset-revision",
        required=True,
        help="immutable Hugging Face commit SHA; 'main' is rejected",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(
        pages_parquet=args.pages_parquet,
        layouts_parquet=args.layouts_parquet,
        annotations_jsonl=args.annotations_jsonl,
        dataset_revision=args.dataset_revision,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
