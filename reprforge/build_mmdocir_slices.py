#!/usr/bin/env python3
"""Build reproducible, document-local MMDocIR Parquet slices in one scan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class DocumentSlice:
    document_index: int
    document: dict[str, Any]
    start: int
    end: int
    path: Path
    writer: Any = None
    rows_written: int = 0
    content_types: Counter[str] = field(default_factory=Counter)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_document_slices(
    *,
    layouts_parquet: Path,
    annotations_jsonl: Path,
    document_indices: Sequence[int],
    output_dir: Path,
    dataset_revision: str,
    source_sha256: str,
    scan_batch_rows: int = 1024,
) -> list[dict[str, Any]]:
    """Extract several non-contiguous documents with one source scan."""

    if not document_indices:
        raise ValueError("at least one document index is required")
    if len(set(document_indices)) != len(document_indices):
        raise ValueError("document indices must be unique")
    if scan_batch_rows <= 0:
        raise ValueError("scan_batch_rows must be positive")

    import pyarrow as pa
    import pyarrow.parquet as pq

    annotations = _load_annotations(annotations_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    selections: list[DocumentSlice] = []
    for document_index in sorted(document_indices):
        if document_index < 0 or document_index >= len(annotations):
            raise IndexError(f"document index {document_index} is outside annotations")
        document = annotations[document_index]
        start, end = (int(value) for value in document["layout_indices"])
        selections.append(
            DocumentSlice(
                document_index=document_index,
                document=document,
                start=start,
                end=end,
                path=output_dir / f"MMDocIR_layouts_doc{document_index}.parquet",
            )
        )

    columns = ("type", "bbox", "page_id", "text", "image_binary")
    parquet = pq.ParquetFile(layouts_parquet)
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"source Parquet is missing columns: {sorted(missing)}")

    batch_start = 0
    try:
        for batch in parquet.iter_batches(
            batch_size=scan_batch_rows,
            columns=list(columns),
            use_threads=True,
        ):
            batch_end = batch_start + batch.num_rows - 1
            for selection in selections:
                overlap_start = max(batch_start, selection.start)
                overlap_end = min(batch_end, selection.end)
                if overlap_start > overlap_end:
                    continue
                local_start = overlap_start - batch_start
                length = overlap_end - overlap_start + 1
                selected = batch.slice(local_start, length)
                if selection.writer is None:
                    selection.writer = pq.ParquetWriter(
                        selection.path,
                        selected.schema,
                        compression="zstd",
                    )
                selection.writer.write_table(pa.Table.from_batches([selected]))
                selection.rows_written += length
                type_index = selected.schema.get_field_index("type")
                selection.content_types.update(
                    str(value) for value in selected.column(type_index).to_pylist()
                )
            batch_start += batch.num_rows
            if batch_start > max(selection.end for selection in selections):
                break
    finally:
        for selection in selections:
            if selection.writer is not None:
                selection.writer.close()

    manifests: list[dict[str, Any]] = []
    for selection in selections:
        expected_rows = selection.end - selection.start + 1
        if selection.rows_written != expected_rows:
            raise ValueError(
                f"document {selection.document_index}: expected {expected_rows} "
                f"rows, wrote {selection.rows_written}"
            )
        slice_sha256 = _sha256(selection.path)
        manifest = {
            "dataset": "MMDocIR/MMDocIR_Evaluation_Dataset",
            "dataset_revision": dataset_revision,
            "document_index": selection.document_index,
            "document_name": str(selection.document.get("doc_name") or ""),
            "domain": str(selection.document.get("domain") or ""),
            "questions": len(selection.document["questions"]),
            "source_file": layouts_parquet.name,
            "source_sha256": source_sha256,
            "source_range_inclusive": [selection.start, selection.end],
            "rows": selection.rows_written,
            "content_types": dict(sorted(selection.content_types.items())),
            "slice_file": selection.path.name,
            "slice_sha256": slice_sha256,
        }
        manifest_path = selection.path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts-parquet", type=Path, required=True)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument(
        "--document-index",
        type=int,
        action="append",
        dest="document_indices",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        help=(
            "optional select_mmdocir_expansion.py output; by default only its "
            "new_document_indices are extracted"
        ),
    )
    parser.add_argument(
        "--selection-field",
        default="new_document_indices",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--scan-batch-rows", type=int, default=1024)
    args = parser.parse_args()

    document_indices = list(args.document_indices or [])
    if args.selection_json is not None:
        selection = json.loads(
            args.selection_json.read_text(encoding="utf-8")
        )
        if args.selection_field not in selection:
            parser.error(
                f"--selection-field {args.selection_field!r} is absent"
            )
        document_indices.extend(
            int(value) for value in selection[args.selection_field]
        )
    if not document_indices:
        parser.error("provide --document-index or --selection-json")
    manifests = build_document_slices(
        layouts_parquet=args.layouts_parquet,
        annotations_jsonl=args.annotations_jsonl,
        document_indices=document_indices,
        output_dir=args.output_dir,
        dataset_revision=args.dataset_revision,
        source_sha256=args.source_sha256,
        scan_batch_rows=args.scan_batch_rows,
    )
    print(json.dumps(manifests, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
