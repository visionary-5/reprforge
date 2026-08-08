#!/usr/bin/env python3
"""Export exact query-by-page MaxSim scores from sharded Omni indexes.

Each shard is loaded and scored independently so a complete corpus need not fit
in host or device memory.  Completed shard arrays are immutable receipts and
are reused on resume.  The output is a compact experimental surface; it is not
used by the deployable compiler to inspect unmaterialized pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queries(
    embeddings_path: Path, masks_path: Path
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with embeddings_path.open("rb") as handle:
        embeddings, query_ids = pickle.load(handle)
    with masks_path.open("rb") as handle:
        masks, mask_query_ids = pickle.load(handle)
    embeddings = np.asarray(embeddings)
    masks = np.asarray(masks)
    query_ids = [str(value) for value in query_ids]
    mask_query_ids = [str(value) for value in mask_query_ids]
    if query_ids != mask_query_ids:
        raise ValueError("query embeddings and masks have different IDs")
    if embeddings.ndim != 3 or masks.shape != embeddings.shape[:2]:
        raise ValueError("unexpected query embedding or mask shape")
    if len(query_ids) != len(set(query_ids)) or len(query_ids) != len(embeddings):
        raise ValueError("query IDs are duplicated or misaligned")
    return embeddings, masks, query_ids


def score_shard(
    index_path: Path,
    query_embeddings: np.ndarray,
    query_masks: np.ndarray,
    output_path: Path,
    *,
    document_chunk_size: int,
    device: str,
) -> tuple[list[str], dict[str, Any]]:
    import torch

    if document_chunk_size <= 0:
        raise ValueError("document_chunk_size must be positive")
    with (index_path / "metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    doc_ids = [str(value) for value in metadata["doc_ids"]]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(f"duplicate document IDs in {index_path}")
    tensor_path = index_path / "index.pt"
    mask_path = index_path / "masks.pt"
    load_started = time.perf_counter()
    tensor_cpu = torch.load(
        tensor_path, map_location="cpu", mmap=True, weights_only=True
    )
    masks_cpu = torch.load(
        mask_path, map_location="cpu", mmap=True, weights_only=True
    )
    if tensor_cpu.ndim != 3 or masks_cpu.shape != tensor_cpu.shape[:2]:
        raise ValueError(f"unexpected Full index shape at {index_path}")
    if len(doc_ids) != len(tensor_cpu):
        raise ValueError(f"metadata and Full index differ at {index_path}")
    tensor_shape = list(tensor_cpu.shape)
    tensor_dtype = str(tensor_cpu.dtype)
    tensor = tensor_cpu.to(device=device)
    masks = masks_cpu.to(device=device)
    load_seconds = time.perf_counter() - load_started

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite completed shard: {output_path}")
    if partial_path.exists():
        raise FileExistsError(
            f"incomplete shard exists; inspect before retrying: {partial_path}"
        )
    scores = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(query_embeddings), len(doc_ids)),
    )
    query_seconds: list[float] = []
    score_started = time.perf_counter()
    for query_offset in range(len(query_embeddings)):
        started = time.perf_counter()
        query = torch.from_numpy(query_embeddings[query_offset]).to(device=device)
        query_mask = torch.from_numpy(query_masks[query_offset]).to(
            device=device, dtype=torch.bool
        )
        for start in range(0, len(doc_ids), document_chunk_size):
            end = min(start + document_chunk_size, len(doc_ids))
            docs = tensor[start:end]
            doc_masks = masks[start:end]
            with torch.inference_mode(), torch.amp.autocast(
                device_type=torch.device(device).type, enabled=False
            ):
                similarity = torch.einsum(
                    "qd,csd->cqs", query.float(), docs.float()
                )
                similarity = similarity.masked_fill(
                    ~doc_masks.bool().unsqueeze(1), float("-inf")
                )
                similarity = similarity.amax(dim=-1)
                similarity = similarity.masked_fill(~query_mask.unsqueeze(0), 0)
                chunk_scores = similarity.sum(dim=-1)
            scores[query_offset, start:end] = chunk_scores.float().cpu().numpy()
            del similarity, chunk_scores
        query_seconds.append(time.perf_counter() - started)
    scores.flush()
    del scores, tensor, masks, tensor_cpu, masks_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.replace(partial_path, output_path)
    score_seconds = time.perf_counter() - score_started
    return doc_ids, {
        "documents": len(doc_ids),
        "queries": len(query_embeddings),
        "tensor_shape": tensor_shape,
        "tensor_dtype": tensor_dtype,
        "document_chunk_size": document_chunk_size,
        "load_to_device_seconds": load_seconds,
        "score_seconds": score_seconds,
        "mean_query_seconds": float(np.mean(query_seconds)),
        "p95_query_seconds": float(np.quantile(query_seconds, 0.95)),
        "score_sha256": sha256(output_path),
        "score_bytes": output_path.stat().st_size,
    }


def export_surface(
    index_paths: Sequence[Path],
    query_embeddings_path: Path,
    query_masks_path: Path,
    output_root: Path,
    *,
    document_chunk_size: int,
    device: str,
) -> dict[str, Any]:
    paths = [Path(path) for path in index_paths]
    if not paths:
        raise ValueError("at least one index shard is required")
    output_root.mkdir(parents=True, exist_ok=True)
    query_embeddings, query_masks, query_ids = load_queries(
        query_embeddings_path, query_masks_path
    )
    query_id_path = output_root / "query_ids.json"
    canonical_query_ids = json.dumps(query_ids, indent=2) + "\n"
    if query_id_path.exists():
        if query_id_path.read_text(encoding="utf-8") != canonical_query_ids:
            raise ValueError("existing query IDs do not match input")
    else:
        query_id_path.write_text(canonical_query_ids, encoding="utf-8")

    all_doc_ids: list[str] = []
    shard_rows = []
    for shard_index, index_path in enumerate(paths):
        score_path = output_root / f"scores-{shard_index:03d}.npy"
        doc_id_path = output_root / f"doc_ids-{shard_index:03d}.json"
        receipt_path = output_root / f"shard-{shard_index:03d}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["index_path"] != str(index_path):
                raise ValueError("resume receipt refers to another index")
            if sha256(score_path) != receipt["metrics"]["score_sha256"]:
                raise ValueError("completed shard score hash mismatch")
            doc_ids = json.loads(doc_id_path.read_text(encoding="utf-8"))
            metrics = receipt["metrics"]
        else:
            doc_ids, metrics = score_shard(
                index_path,
                query_embeddings,
                query_masks,
                score_path,
                document_chunk_size=document_chunk_size,
                device=device,
            )
            doc_id_path.write_text(
                json.dumps(doc_ids, indent=2) + "\n", encoding="utf-8"
            )
            receipt = {
                "schema_version": 1,
                "index_path": str(index_path),
                "score_path": str(score_path),
                "doc_id_path": str(doc_id_path),
                "metrics": metrics,
            }
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        overlap = set(all_doc_ids) & set(doc_ids)
        if overlap:
            raise ValueError(f"index shards overlap: {sorted(overlap)[:5]}")
        all_doc_ids.extend(map(str, doc_ids))
        shard_rows.append(receipt)

    manifest = {
        "schema_version": 1,
        "protocol": "exact-sharded-omni-score-surface-2026-08-08",
        "device": device,
        "queries": len(query_ids),
        "documents": len(all_doc_ids),
        "query_embeddings": {
            "path": str(query_embeddings_path),
            "sha256": sha256(query_embeddings_path),
        },
        "query_masks": {
            "path": str(query_masks_path),
            "sha256": sha256(query_masks_path),
        },
        "query_ids": {"path": str(query_id_path), "sha256": sha256(query_id_path)},
        "shards": shard_rows,
        "warning": (
            "Complete scores are oracle/evaluation infrastructure. The deployable "
            "compiler may access only materialized page columns."
        ),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--query-masks", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--document-chunk-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = export_surface(
        args.index,
        args.query_embeddings,
        args.query_masks,
        args.output_root,
        document_chunk_size=args.document_chunk_size,
        device=args.device,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
