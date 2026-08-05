#!/usr/bin/env python3
"""Late-materialize Full vectors to rerank an OmniColPress candidate list."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidates(path: Path, *, expected_depth: int) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"expected 3 tab fields at {path}:{line_number}")
            query_id, doc_id, _ = fields
            if doc_id in seen[query_id]:
                raise ValueError(f"duplicate candidate at {path}:{line_number}")
            seen[query_id].add(doc_id)
            rankings[query_id].append(doc_id)
    bad = {key: len(value) for key, value in rankings.items() if len(value) != expected_depth}
    if bad:
        raise ValueError(f"unexpected candidate depths: {bad}")
    return dict(rankings)


def rerank_candidate_prefix(
    candidate_docs: list[str],
    full_scores: dict[str, float],
    *,
    depth: int,
) -> list[tuple[str, float]]:
    if not 1 <= depth <= len(candidate_docs):
        raise ValueError("rerank depth must lie inside the candidate list")
    prefix = candidate_docs[:depth]
    if set(prefix) - set(full_scores):
        raise ValueError("full scores are missing rerank candidates")
    positions = {doc_id: index for index, doc_id in enumerate(prefix)}
    return sorted(
        ((doc_id, full_scores[doc_id]) for doc_id in prefix),
        key=lambda item: (-item[1], positions[item[0]]),
    )


def _write_ranking(path: Path, rows: list[tuple[str, str, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for query_id, doc_id, score in rows:
            handle.write(f"{query_id}\t{doc_id}\t{score:.9g}\n")


def run_cascade(
    index_path: Path,
    query_embeddings_path: Path,
    query_masks_path: Path,
    candidate_ranking_path: Path,
    output_root: Path,
    *,
    candidate_depth: int = 100,
    rerank_depths: tuple[int, ...] = (10, 20, 50, 100),
    score_chunk_size: int = 25,
    device: str = "cuda",
) -> dict[str, Any]:
    import numpy as np
    import torch

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    depths = tuple(sorted(set(int(value) for value in rerank_depths)))
    if not depths or depths[-1] > candidate_depth or depths[0] <= 0:
        raise ValueError("rerank depths must be positive and no larger than candidates")
    if score_chunk_size <= 0:
        raise ValueError("score chunk size must be positive")

    candidates = load_candidates(
        candidate_ranking_path, expected_depth=candidate_depth
    )
    with query_embeddings_path.open("rb") as handle:
        query_embeddings, query_ids = pickle.load(handle)
    query_ids = [str(value) for value in query_ids]
    if len(query_ids) != len(set(query_ids)) or set(query_ids) != set(candidates):
        raise ValueError("query embeddings and candidate query IDs must match")
    query_embeddings = np.asarray(query_embeddings)
    if query_embeddings.shape[0] != len(query_ids):
        raise ValueError("query embedding count does not match query IDs")
    with query_masks_path.open("rb") as handle:
        query_masks, mask_query_ids = pickle.load(handle)
    mask_query_ids = [str(value) for value in mask_query_ids]
    query_masks = np.asarray(query_masks)
    if mask_query_ids != query_ids or query_masks.shape != query_embeddings.shape[:2]:
        raise ValueError("query masks must align exactly with query embeddings")

    metadata_path = index_path / "metadata.pkl"
    tensor_path = index_path / "index.pt"
    mask_path = index_path / "masks.pt"
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    doc_ids = [str(value) for value in metadata["doc_ids"]]
    doc_positions = {doc_id: position for position, doc_id in enumerate(doc_ids)}
    if len(doc_positions) != len(doc_ids):
        raise ValueError("Full index contains duplicate document IDs")
    unknown = {
        doc_id for ranking in candidates.values() for doc_id in ranking
    } - set(doc_positions)
    if unknown:
        raise ValueError(f"candidate documents missing from Full index: {unknown}")

    load_started = time.perf_counter()
    full_index = torch.load(
        tensor_path, map_location="cpu", mmap=True, weights_only=True
    )
    full_masks = torch.load(
        mask_path, map_location="cpu", mmap=True, weights_only=True
    )
    load_seconds = time.perf_counter() - load_started
    if full_index.ndim != 3 or full_masks.shape != full_index.shape[:2]:
        raise ValueError("unexpected Full index or mask shape")

    rows_by_depth: dict[int, list[tuple[str, str, float]]] = {
        depth: [] for depth in depths
    }
    query_timings = []
    score_started = time.perf_counter()
    for query_offset, query_id in enumerate(query_ids):
        started = time.perf_counter()
        query = torch.from_numpy(query_embeddings[query_offset]).to(device=device)
        query_mask = torch.from_numpy(query_masks[query_offset]).to(
            device=device, dtype=torch.bool
        )
        candidate_docs = candidates[query_id][: depths[-1]]
        candidate_positions = [doc_positions[doc_id] for doc_id in candidate_docs]
        scores: list[float] = []
        for start in range(0, len(candidate_positions), score_chunk_size):
            positions = candidate_positions[start : start + score_chunk_size]
            docs = full_index[positions].to(device=device, non_blocking=False)
            masks = full_masks[positions].to(device=device, non_blocking=False)
            with torch.inference_mode(), torch.amp.autocast(
                device_type=torch.device(device).type, enabled=False
            ):
                similarity = torch.einsum(
                    "qd,csd->cqs", query.float(), docs.float()
                )
                similarity = similarity.masked_fill(
                    ~masks.bool().unsqueeze(1), float("-inf")
                )
                similarity = similarity.amax(dim=-1)
                similarity = similarity.masked_fill(~query_mask.unsqueeze(0), 0)
                chunk_scores = similarity.sum(dim=-1)
            scores.extend(float(value) for value in chunk_scores.cpu().tolist())
            del docs, masks, similarity, chunk_scores
        score_map = dict(zip(candidate_docs, scores, strict=True))
        for depth in depths:
            reranked = rerank_candidate_prefix(
                candidate_docs, score_map, depth=depth
            )
            rows_by_depth[depth].extend(
                (query_id, doc_id, score) for doc_id, score in reranked
            )
        query_timings.append(time.perf_counter() - started)
    score_seconds = time.perf_counter() - score_started

    output_root.mkdir(parents=True)
    for depth, rows in rows_by_depth.items():
        _write_ranking(output_root / f"cascade-top{depth}.ranking.tsv", rows)

    bytes_per_doc = full_index[0].numel() * full_index.element_size()
    mask_bytes_per_doc = full_masks[0].numel() * full_masks.element_size()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "omni-cold-full-late-materialization-2026-08-05",
        "query_count": len(query_ids),
        "corpus_count": len(doc_ids),
        "candidate_depth": candidate_depth,
        "rerank_depths": list(depths),
        "score_chunk_size": score_chunk_size,
        "device": device,
        "full_index": {
            "shape": list(full_index.shape),
            "dtype": str(full_index.dtype),
            "bytes": tensor_path.stat().st_size,
            "mask_bytes": mask_path.stat().st_size,
            "load_mode": "torch_load_cpu_mmap_then_selected_rows_to_device",
        },
        "logical_materialization": {
            str(depth): {
                "vector_and_mask_bytes_per_query": depth
                * (bytes_per_doc + mask_bytes_per_doc),
                "corpus_row_fraction_per_query": depth / len(doc_ids),
            }
            for depth in depths
        },
        "timing": {
            "mmap_load_seconds": load_seconds,
            "score_all_queries_seconds": score_seconds,
            "mean_query_seconds": float(np.mean(query_timings)),
            "p50_query_seconds": float(np.quantile(query_timings, 0.50)),
            "p95_query_seconds": float(np.quantile(query_timings, 0.95)),
            "timing_scope": "candidate_full_scoring_only_query_encoding_excluded",
            "os_page_cache_state": "uncontrolled",
        },
        "artifacts": {
            "candidate_ranking": {
                "path": str(candidate_ranking_path),
                "sha256": _sha256(candidate_ranking_path),
            },
            "query_embeddings": {
                "path": str(query_embeddings_path),
                "sha256": _sha256(query_embeddings_path),
            },
            "query_masks": {
                "path": str(query_masks_path),
                "sha256": _sha256(query_masks_path),
            },
            "full_index_metadata": {
                "path": str(metadata_path),
                "sha256": _sha256(metadata_path),
            },
        },
    }
    for depth in depths:
        ranking_path = output_root / f"cascade-top{depth}.ranking.tsv"
        manifest["artifacts"][f"cascade_top{depth}_ranking"] = {
            "path": str(ranking_path),
            "sha256": _sha256(ranking_path),
        }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--query-masks", type=Path, required=True)
    parser.add_argument("--candidate-ranking", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--rerank-depths", type=int, nargs="+", default=(10, 20, 50, 100))
    parser.add_argument("--score-chunk-size", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = run_cascade(
        args.index,
        args.query_embeddings,
        args.query_masks,
        args.candidate_ranking,
        args.output_root,
        candidate_depth=args.candidate_depth,
        rerank_depths=tuple(args.rerank_depths),
        score_chunk_size=args.score_chunk_size,
        device=args.device,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
