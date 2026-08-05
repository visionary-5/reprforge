#!/usr/bin/env python3
"""Export one complete ViDoRe language split for OmniColPress."""

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


def export_domain(
    dataset_root: Path,
    output_root: Path,
    *,
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
    corpus = sorted(
        parquet.read_table(paths["corpus"]).to_pylist(),
        key=lambda row: int(row["corpus_id"]),
    )
    queries = sorted(
        (
            row
            for row in parquet.read_table(paths["queries"]).to_pylist()
            if str(row["language"]).lower() == language.lower()
        ),
        key=lambda row: int(row["query_id"]),
    )
    all_qrels = parquet.read_table(paths["qrels"]).to_pylist()

    corpus_ids = {int(row["corpus_id"]) for row in corpus}
    query_ids = {int(row["query_id"]) for row in queries}
    qrels = sorted(
        (
            row
            for row in all_qrels
            if int(row["query_id"]) in query_ids and int(row["score"]) > 0
        ),
        key=lambda row: (int(row["query_id"]), int(row["corpus_id"])),
    )
    qrel_query_ids = {int(row["query_id"]) for row in qrels}
    missing_query_qrels = query_ids - qrel_query_ids
    if missing_query_qrels:
        raise RuntimeError(
            f"queries without positive qrels: {sorted(missing_query_qrels)}"
        )
    missing_corpus = {int(row["corpus_id"]) for row in qrels} - corpus_ids
    if missing_corpus:
        raise RuntimeError(f"qrels reference missing corpus ids: {sorted(missing_corpus)}")

    output_root.mkdir(parents=True)
    assets_root = output_root / "assets"
    assets_root.mkdir()

    corpus_rows = []
    for row in corpus:
        corpus_id = int(row["corpus_id"])
        image_path = assets_root / f"{corpus_id}.png"
        with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
            image.convert("RGB").save(image_path, format="PNG")
        corpus_rows.append(
            {
                "docid": str(corpus_id),
                "image": image_path.name,
                "text": row.get("markdown") or "",
            }
        )

    query_rows = [
        {"query_id": str(row["query_id"]), "query": row["query"]}
        for row in queries
    ]
    qrel_rows = [
        {
            "doc_id": str(row["corpus_id"]),
            "query_id": str(row["query_id"]),
            "relevance": int(row["score"]),
        }
        for row in qrels
    ]
    _write_jsonl(output_root / "corpus.jsonl", corpus_rows)
    _write_jsonl(output_root / "queries.jsonl", query_rows)
    _write_jsonl(output_root / "qrels.jsonl", qrel_rows)

    manifest = {
        "dataset_root": str(dataset_root),
        "language": language,
        "num_pages": len(corpus_rows),
        "num_qrels": len(qrel_rows),
        "num_queries": len(query_rows),
        "qrel_complete": True,
        "selection_policy": "all_corpus_pages_and_all_queries_for_language",
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
    parser.add_argument("--language", default="english")
    args = parser.parse_args()
    manifest = export_domain(
        args.dataset_root,
        args.output_root,
        language=args.language,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
