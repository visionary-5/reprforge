import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reprforge.build_mmdocir_slices import build_document_slices


def test_build_document_slices_preserves_absolute_ranges(tmp_path: Path) -> None:
    source = tmp_path / "layouts.parquet"
    pq.write_table(
        pa.table(
            {
                "type": ["text", "image", "table", "text", "equation"],
                "bbox": [[0.0, 0.0, 1.0, 1.0]] * 5,
                "page_id": [0, 0, 1, 1, 1],
                "text": ["a", "", "", "b", "c"],
                "image_binary": [b"a", b"b", b"c", b"d", b"e"],
            }
        ),
        source,
    )
    annotations = tmp_path / "annotations.jsonl"
    documents = [
        {
            "doc_name": "first.pdf",
            "domain": "first",
            "layout_indices": [0, 1],
            "questions": [{"Q": "first?"}],
        },
        {
            "doc_name": "second.pdf",
            "domain": "second",
            "layout_indices": [2, 4],
            "questions": [{"Q": "second?"}, {"Q": "again?"}],
        },
    ]
    annotations.write_text(
        "".join(json.dumps(row) + "\n" for row in documents),
        encoding="utf-8",
    )

    manifests = build_document_slices(
        layouts_parquet=source,
        annotations_jsonl=annotations,
        document_indices=[1, 0],
        output_dir=tmp_path / "slices",
        dataset_revision="dataset-sha",
        source_sha256="source-sha",
        scan_batch_rows=2,
    )

    assert [row["document_index"] for row in manifests] == [0, 1]
    assert manifests[0]["content_types"] == {"image": 1, "text": 1}
    assert manifests[1]["source_range_inclusive"] == [2, 4]
    second = pq.read_table(tmp_path / "slices/MMDocIR_layouts_doc1.parquet")
    assert second.column("type").to_pylist() == ["table", "text", "equation"]
    manifest = json.loads(
        (
            tmp_path / "slices/MMDocIR_layouts_doc1.manifest.json"
        ).read_text()
    )
    assert manifest["questions"] == 2
    assert manifest["slice_sha256"] == manifests[1]["slice_sha256"]
