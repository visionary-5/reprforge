#!/usr/bin/env python3
"""Run one ColPali route bank over a manifest-defined MMDocIR pilot corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from reprforge.mmdocir_route_runner import (
    ColPaliBackend,
    execute_document,
    load_annotations,
    read_parquet_row_range,
)
from reprforge.policy_replay import (
    IMAGE_ROUTE,
    TEXT_ROUTE,
    evaluate_plan,
    fixed_hybrid_plan,
    load_replay_data,
    uniform_plan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_slice_manifests(paths: Sequence[Path]) -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    seen_documents: set[int] = set()
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        document_index = int(manifest["document_index"])
        if document_index in seen_documents:
            raise ValueError(f"duplicate document index {document_index}")
        seen_documents.add(document_index)
        records.append((path, manifest))
    return sorted(records, key=lambda value: int(value[1]["document_index"]))


def _merge_jsonl(inputs: Sequence[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for path in inputs:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        target.write(line.rstrip("\n") + "\n")


def execute_corpus(
    *,
    backend: Any,
    annotations: Sequence[Mapping[str, Any]],
    slice_manifests: Sequence[Path],
    output: Path,
    dataset_revision: str,
    model_revision: str,
    persist_embeddings: bool = False,
    embedding_storage_dtype: str = "float32",
    resume_complete_documents: bool = False,
) -> dict:
    records = _load_slice_manifests(slice_manifests)
    document_outputs: list[Path] = []
    document_manifests: list[dict] = []

    for manifest_path, slice_manifest in records:
        if slice_manifest["dataset_revision"] != dataset_revision:
            raise ValueError(
                f"{manifest_path} dataset revision does not match the run"
            )
        document_index = int(slice_manifest["document_index"])
        document_output = output / "documents" / f"doc-{document_index}"
        complete_paths = [
            document_output / filename
            for filename in (
                "manifest.json",
                "items.jsonl",
                "queries.jsonl",
                "scores.jsonl",
            )
        ]
        if resume_complete_documents and all(
            path.is_file() for path in complete_paths
        ):
            result = json.loads(
                (document_output / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                int(result["document_index"]) != document_index
                or result["dataset_revision"] != dataset_revision
                or result["model_revision"] != model_revision
                or bool(result["embeddings_persisted"]) != persist_embeddings
            ):
                raise ValueError(
                    f"completed document {document_index} violates resume contract"
                )
            load_replay_data(
                document_output / "items.jsonl",
                document_output / "queries.jsonl",
                document_output / "scores.jsonl",
            )
            result["resumed"] = True
            document_manifests.append(result)
            document_outputs.append(document_output)
            continue
        annotation_range = [
            int(value) for value in annotations[document_index]["layout_indices"]
        ]
        source_range = [
            int(value) for value in slice_manifest["source_range_inclusive"]
        ]
        if source_range != annotation_range:
            raise ValueError(
                f"document {document_index} slice range {source_range} does not "
                f"match annotation range {annotation_range}"
            )
        slice_path = manifest_path.parent / str(slice_manifest["slice_file"])
        actual_sha256 = _sha256(slice_path)
        if actual_sha256 != slice_manifest["slice_sha256"]:
            raise ValueError(f"{slice_path} failed its SHA-256 contract")
        row_count = int(slice_manifest["rows"])
        rows = read_parquet_row_range(
            slice_path,
            start=0,
            end=row_count - 1,
            columns=("type", "bbox", "page_id", "text", "image_binary"),
        )
        result = execute_document(
            backend=backend,
            annotations=annotations,
            document_index=document_index,
            layout_rows=rows,
            output=document_output,
            dataset_revision=dataset_revision,
            model_revision=model_revision,
            persist_embeddings=persist_embeddings,
            embedding_storage_dtype=embedding_storage_dtype,
        )
        result["slice_sha256"] = actual_sha256
        document_manifests.append(result)
        document_outputs.append(document_output)

    for filename in ("items.jsonl", "queries.jsonl", "scores.jsonl"):
        _merge_jsonl(
            [document_output / filename for document_output in document_outputs],
            output / filename,
        )
    embedding_bank_manifest = None
    if persist_embeddings:
        from reprforge.heterogeneous_index import merge_embedding_banks

        embedding_bank_manifest = merge_embedding_banks(
            [
                document_output / "embedding-bank"
                for document_output in document_outputs
            ],
            output / "embedding-bank",
        )
    data = load_replay_data(
        output / "items.jsonl",
        output / "queries.jsonl",
        output / "scores.jsonl",
    )
    plans = {
        "all-text": uniform_plan(data.items, TEXT_ROUTE),
        "all-image": uniform_plan(data.items, IMAGE_ROUTE),
        "fixed-hybrid": fixed_hybrid_plan(data.items),
    }
    for route in data.routes:
        if route in (TEXT_ROUTE, IMAGE_ROUTE):
            continue
        plans[f"all-{route}"] = uniform_plan(data.items, route)
        plans[f"fixed-hybrid-{route}"] = fixed_hybrid_plan(
            data.items,
            image_route=route,
        )
    baselines = {
        name: evaluate_plan(data, plan, ks=(1, 5, 10))
        for name, plan in plans.items()
    }
    (output / "baselines.json").write_text(
        json.dumps(baselines, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": "MMDocIR/MMDocIR_Evaluation_Dataset",
        "dataset_revision": dataset_revision,
        "model": "MMDocIR/MMDocIR_Retrievers/colpali-v1.1",
        "model_revision": model_revision,
        "documents": document_manifests,
        "document_count": len(document_manifests),
        "layout_count": len(data.items),
        "query_count": len(data.queries),
        "domains": sorted(
            {str(row.get("domain") or "") for row in document_manifests}
        ),
        "embeddings_persisted": persist_embeddings,
        "embedding_bank": (
            {
                "path": "embedding-bank",
                "storage_dtype": embedding_bank_manifest["storage_dtype"],
                "item_count": embedding_bank_manifest["item_count"],
                "query_count": embedding_bank_manifest["queries"]["count"],
            }
            if embedding_bank_manifest is not None
            else None
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument(
        "--slice-manifest",
        type=Path,
        action="append",
        required=True,
        dest="slice_manifests",
    )
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--image-pool-factor",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--persist-embeddings", action="store_true")
    parser.add_argument("--resume-complete-documents", action="store_true")
    parser.add_argument(
        "--embedding-storage-dtype",
        choices=("float16", "float32"),
        default="float32",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset_revision == "main" or args.model_revision == "main":
        parser.error("dataset and model revisions must be immutable commit SHAs")
    annotations = load_annotations(args.annotations_jsonl)
    backend = ColPaliBackend(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        batch_size=args.batch_size,
        image_pool_factors=args.image_pool_factor,
    )
    manifest = execute_corpus(
        backend=backend,
        annotations=annotations,
        slice_manifests=args.slice_manifests,
        output=args.output,
        dataset_revision=args.dataset_revision,
        model_revision=args.model_revision,
        persist_embeddings=args.persist_embeddings,
        embedding_storage_dtype=args.embedding_storage_dtype,
        resume_complete_documents=args.resume_complete_documents,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
