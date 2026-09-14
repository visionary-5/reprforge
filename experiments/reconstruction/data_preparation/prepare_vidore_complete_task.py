#!/usr/bin/env python3
"""Materialize one complete ViDoRe task into the frozen ReprForge route format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--queries-dir", type=Path, required=True)
    parser.add_argument("--qrels-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_root}")

    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    corpus_rows = ds.dataset(args.corpus_dir, format="parquet").to_table(
        columns=["corpus_id", "image"]
    ).to_pylist()
    corpus_rows.sort(key=lambda row: int(row["corpus_id"]))
    query_rows = ds.dataset(args.queries_dir, format="parquet").to_table(
        columns=["query_id", "query"]
    ).to_pylist()
    query_rows.sort(key=lambda row: int(row["query_id"]))
    qrel_rows = ds.dataset(args.qrels_dir, format="parquet").to_table(
        columns=["query_id", "corpus_id", "score"]
    ).to_pylist()
    relevance: dict[int, dict[int, float]] = {}
    for row in qrel_rows:
        relevance.setdefault(int(row["query_id"]), {})[int(row["corpus_id"])] = float(
            row["score"]
        )
    corpus_ids = {int(row["corpus_id"]) for row in corpus_rows}
    eligible_queries = [
        row
        for row in query_rows
        if int(row["query_id"]) in relevance
        and set(relevance[int(row["query_id"])]) <= corpus_ids
    ]
    if len(eligible_queries) != len(query_rows):
        raise ValueError("complete task is not qrel closed for every query")

    args.output_root.mkdir(parents=True)
    route_root = args.output_root / "route"
    route_root.mkdir()
    slice_path = args.output_root / "slice.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "corpus_id": int(row["corpus_id"]),
                    "image_binary": row["image"]["bytes"],
                }
                for row in corpus_rows
            ]
        ),
        slice_path,
    )
    items_path = route_root / "items.jsonl"
    queries_path = route_root / "queries.jsonl"
    write_jsonl(
        items_path,
        [{"item_id": str(int(row["corpus_id"]))} for row in corpus_rows],
    )
    write_jsonl(
        queries_path,
        [
            {
                "query_id": str(int(row["query_id"])),
                "query": str(row["query"]),
                "relevance": {
                    str(corpus_id): score
                    for corpus_id, score in sorted(
                        relevance[int(row["query_id"])].items()
                    )
                },
                "relevance_denominator": 1.0,
            }
            for row in eligible_queries
        ],
    )
    manifest = {
        "pages": len(corpus_rows),
        "queries": len(eligible_queries),
        "qrels": len(qrel_rows),
        "sampling": "complete corpus and all qrel-closed queries; no sampling",
        "slice_sha256": sha256(slice_path),
        "items_sha256": sha256(items_path),
        "queries_sha256": sha256(queries_path),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
