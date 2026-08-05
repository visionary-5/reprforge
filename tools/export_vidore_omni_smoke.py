#!/usr/bin/env python3
"""Export a deterministic, qrel-complete ViDoRe subset for OmniColPress."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def select_queries(
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    *,
    max_queries: int,
    max_pages: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Greedily select qrel-complete queries in deterministic query-id order."""

    relevant: dict[int, set[int]] = {}
    for row in qrels:
        if int(row["score"]) > 0:
            relevant.setdefault(int(row["query_id"]), set()).add(int(row["corpus_id"]))

    selected: list[dict[str, Any]] = []
    page_ids: set[int] = set()
    remaining = list(queries)
    while remaining and len(selected) < max_queries:
        candidates = []
        for query in remaining:
            query_id = int(query["query_id"])
            query_pages = relevant.get(query_id, set())
            added_pages = query_pages - page_ids
            if query_pages and len(page_ids | query_pages) <= max_pages:
                candidates.append(
                    (len(added_pages), len(query_pages), query_id, query, query_pages)
                )
        if not candidates:
            break
        _, _, selected_id, query, query_pages = min(
            candidates, key=lambda row: row[:3]
        )
        selected.append(query)
        page_ids.update(query_pages)
        remaining = [row for row in remaining if int(row["query_id"]) != selected_id]
    return sorted(selected, key=lambda row: int(row["query_id"])), page_ids


def export_smoke(
    dataset_root: Path,
    output_root: Path,
    *,
    max_queries: int,
    max_pages: int,
    language: str,
) -> dict[str, Any]:
    import pyarrow.parquet as parquet
    from PIL import Image

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")

    paths = {
        part: next((dataset_root / part).glob("*.parquet"))
        for part in ("corpus", "queries", "qrels")
    }
    corpus = parquet.read_table(paths["corpus"]).to_pylist()
    queries = [
        row
        for row in parquet.read_table(paths["queries"]).to_pylist()
        if str(row["language"]).lower() == language.lower()
    ]
    qrels = parquet.read_table(paths["qrels"]).to_pylist()
    selected_queries, positive_page_ids = select_queries(
        queries,
        qrels,
        max_queries=max_queries,
        max_pages=max_pages,
    )
    if len(selected_queries) != max_queries:
        raise RuntimeError(
            f"could only select {len(selected_queries)} qrel-complete queries; "
            f"requested {max_queries} under {max_pages} pages"
        )

    corpus_by_id = {int(row["corpus_id"]): row for row in corpus}
    missing_positive_pages = positive_page_ids - corpus_by_id.keys()
    if missing_positive_pages:
        raise RuntimeError(f"qrels reference missing corpus ids: {sorted(missing_positive_pages)}")

    selected_page_ids = set(positive_page_ids)
    for corpus_id in sorted(corpus_by_id):
        if len(selected_page_ids) == max_pages:
            break
        selected_page_ids.add(corpus_id)

    output_root.mkdir(parents=True)
    assets_root = output_root / "assets"
    assets_root.mkdir()
    corpus_rows = []
    for corpus_id in sorted(selected_page_ids):
        row = corpus_by_id[corpus_id]
        image_path = assets_root / f"{corpus_id}.png"
        image_bytes = row["image"]["bytes"]
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(image_path, format="PNG")
        corpus_rows.append(
            {
                "docid": str(corpus_id),
                "image": image_path.name,
                "text": row.get("markdown") or "",
            }
        )

    selected_query_ids = {int(row["query_id"]) for row in selected_queries}
    query_rows = [
        {"query_id": str(row["query_id"]), "query": row["query"]}
        for row in selected_queries
    ]
    qrel_rows = [
        {
            "doc_id": str(row["corpus_id"]),
            "query_id": str(row["query_id"]),
            "relevance": int(row["score"]),
        }
        for row in qrels
        if int(row["query_id"]) in selected_query_ids
        and int(row["corpus_id"]) in selected_page_ids
        and int(row["score"]) > 0
    ]

    _write_jsonl(output_root / "corpus.jsonl", corpus_rows)
    _write_jsonl(output_root / "queries.jsonl", query_rows)
    _write_jsonl(output_root / "qrels.jsonl", qrel_rows)

    positive_pairs = {
        (int(row["query_id"]), int(row["corpus_id"]))
        for row in qrels
        if int(row["query_id"]) in selected_query_ids and int(row["score"]) > 0
    }
    exported_pairs = {
        (int(row["query_id"]), int(row["doc_id"])) for row in qrel_rows
    }
    if exported_pairs != positive_pairs:
        raise RuntimeError("exported qrels are not complete for selected queries")

    manifest = {
        "dataset_root": str(dataset_root),
        "language": language,
        "max_pages": max_pages,
        "max_queries": max_queries,
        "num_pages": len(corpus_rows),
        "num_positive_pages": len(positive_page_ids),
        "num_qrels": len(qrel_rows),
        "num_queries": len(query_rows),
        "qrel_complete": True,
        "selection_policy": "greedy_min_incremental_positive_pages_then_query_id",
        "selected_page_ids": sorted(selected_page_ids),
        "selected_query_ids": sorted(selected_query_ids),
        "source_sha256": {part: _sha256(path) for part, path in paths.items()},
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=16)
    parser.add_argument("--max-pages", type=int, default=32)
    parser.add_argument("--language", default="english")
    args = parser.parse_args()
    manifest = export_smoke(
        args.dataset_root,
        args.output_root,
        max_queries=args.max_queries,
        max_pages=args.max_pages,
        language=args.language,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
