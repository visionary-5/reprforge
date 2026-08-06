#!/usr/bin/env python3
"""Late-materialize Full vectors from disjoint shards for candidate reranking."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from tools.rerank_omni_candidates import (
    _sha256,
    _write_ranking,
    load_candidates,
    rerank_candidate_prefix,
)


def run_sharded_cascade(
    index_paths: Sequence[Path],
    query_embeddings_path: Path,
    query_masks_path: Path,
    candidate_ranking_path: Path,
    output_root: Path,
    *,
    candidate_depth: int = 100,
    rerank_depths: tuple[int, ...] = (20, 50, 100),
    score_chunk_size: int = 25,
    device: str = "cuda",
) -> dict[str, Any]:
    import numpy as np
    import torch

    paths = [Path(path) for path in index_paths]
    if len(paths) < 2:
        raise ValueError("at least two Full index shards are required")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    depths = tuple(sorted(set(int(value) for value in rerank_depths)))
    if not depths or depths[0] <= 0 or depths[-1] > candidate_depth:
        raise ValueError("rerank depths must lie inside the candidate list")
    if score_chunk_size <= 0:
        raise ValueError("score chunk size must be positive")

    candidates = load_candidates(candidate_ranking_path, expected_depth=candidate_depth)
    with query_embeddings_path.open("rb") as handle:
        query_embeddings, query_ids = pickle.load(handle)
    with query_masks_path.open("rb") as handle:
        query_masks, mask_query_ids = pickle.load(handle)
    query_ids = [str(value) for value in query_ids]
    mask_query_ids = [str(value) for value in mask_query_ids]
    query_embeddings = np.asarray(query_embeddings)
    query_masks = np.asarray(query_masks)
    if (
        query_ids != mask_query_ids
        or query_embeddings.shape[0] != len(query_ids)
        or query_masks.shape != query_embeddings.shape[:2]
        or set(query_ids) != set(candidates)
    ):
        raise ValueError("query embeddings, masks, and candidates must align exactly")

    load_started = time.perf_counter()
    shards: list[dict[str, Any]] = []
    doc_locations: dict[str, tuple[int, int]] = {}
    for shard_index, index_path in enumerate(paths):
        with (index_path / "metadata.pkl").open("rb") as handle:
            metadata = pickle.load(handle)
        doc_ids = [str(value) for value in metadata["doc_ids"]]
        overlap = set(doc_ids) & set(doc_locations)
        if overlap:
            raise ValueError(f"Full index shards overlap: {sorted(overlap)[:5]}")
        for position, doc_id in enumerate(doc_ids):
            doc_locations[doc_id] = (shard_index, position)
        tensor_path = index_path / "index.pt"
        mask_path = index_path / "masks.pt"
        tensor = torch.load(tensor_path, map_location="cpu", mmap=True, weights_only=True)
        masks = torch.load(mask_path, map_location="cpu", mmap=True, weights_only=True)
        if tensor.ndim != 3 or masks.shape != tensor.shape[:2] or len(doc_ids) != len(tensor):
            raise ValueError(f"unexpected Full shard layout: {index_path}")
        shards.append(
            {
                "path": index_path,
                "tensor_path": tensor_path,
                "mask_path": mask_path,
                "tensor": tensor,
                "masks": masks,
                "doc_count": len(doc_ids),
            }
        )
    load_seconds = time.perf_counter() - load_started
    unknown = {
        doc_id for rows in candidates.values() for doc_id in rows[: depths[-1]]
    } - set(doc_locations)
    if unknown:
        raise ValueError(f"candidate documents missing from Full shards: {sorted(unknown)[:5]}")

    rows_by_depth: dict[int, list[tuple[str, str, float]]] = {
        depth: [] for depth in depths
    }
    query_timings: list[float] = []
    materialized_bytes: dict[int, list[int]] = {depth: [] for depth in depths}
    score_started = time.perf_counter()
    for query_offset, query_id in enumerate(query_ids):
        started = time.perf_counter()
        query = torch.from_numpy(query_embeddings[query_offset]).to(device=device)
        query_mask = torch.from_numpy(query_masks[query_offset]).to(
            device=device, dtype=torch.bool
        )
        candidate_docs = candidates[query_id][: depths[-1]]
        by_shard: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for doc_id in candidate_docs:
            shard_index, position = doc_locations[doc_id]
            by_shard[shard_index].append((doc_id, position))
        score_map: dict[str, float] = {}
        byte_map: dict[str, int] = {}
        for shard_index, located in by_shard.items():
            shard = shards[shard_index]
            tensor = shard["tensor"]
            masks = shard["masks"]
            bytes_per_doc = tensor[0].numel() * tensor.element_size()
            mask_bytes_per_doc = masks[0].numel() * masks.element_size()
            for start in range(0, len(located), score_chunk_size):
                batch = located[start : start + score_chunk_size]
                positions = [position for _, position in batch]
                docs = tensor[positions].to(device=device, non_blocking=False)
                doc_masks = masks[positions].to(device=device, non_blocking=False)
                with torch.inference_mode(), torch.amp.autocast(
                    device_type=torch.device(device).type, enabled=False
                ):
                    similarity = torch.einsum("qd,csd->cqs", query.float(), docs.float())
                    similarity = similarity.masked_fill(
                        ~doc_masks.bool().unsqueeze(1), float("-inf")
                    )
                    similarity = similarity.amax(dim=-1)
                    similarity = similarity.masked_fill(~query_mask.unsqueeze(0), 0)
                    scores = similarity.sum(dim=-1).cpu().tolist()
                for (doc_id, _), score in zip(batch, scores, strict=True):
                    score_map[doc_id] = float(score)
                    byte_map[doc_id] = bytes_per_doc + mask_bytes_per_doc
                del docs, doc_masks, similarity, scores
        for depth in depths:
            reranked = rerank_candidate_prefix(candidate_docs, score_map, depth=depth)
            rows_by_depth[depth].extend(
                (query_id, doc_id, score) for doc_id, score in reranked
            )
            materialized_bytes[depth].append(
                sum(byte_map[doc_id] for doc_id in candidate_docs[:depth])
            )
        query_timings.append(time.perf_counter() - started)
    score_seconds = time.perf_counter() - score_started

    output_root.mkdir(parents=True)
    for depth, rows in rows_by_depth.items():
        _write_ranking(output_root / f"cascade-top{depth}.ranking.tsv", rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "omni-sharded-cold-full-late-materialization-2026-08-06",
        "query_count": len(query_ids),
        "corpus_count": len(doc_locations),
        "candidate_depth": candidate_depth,
        "rerank_depths": list(depths),
        "score_chunk_size": score_chunk_size,
        "device": device,
        "full_index_shards": [
            {
                "path": str(shard["path"]),
                "documents": shard["doc_count"],
                "index_bytes": shard["tensor_path"].stat().st_size,
                "mask_bytes": shard["mask_path"].stat().st_size,
            }
            for shard in shards
        ],
        "logical_materialization": {
            str(depth): {
                "mean_vector_and_mask_bytes_per_query": float(
                    np.mean(materialized_bytes[depth])
                ),
                "corpus_row_fraction_per_query": depth / len(doc_locations),
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
        },
    }
    for depth in depths:
        path = output_root / f"cascade-top{depth}.ranking.tsv"
        manifest["artifacts"][f"cascade_top{depth}_ranking"] = {
            "path": str(path),
            "sha256": _sha256(path),
        }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--query-masks", type=Path, required=True)
    parser.add_argument("--candidate-ranking", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--rerank-depths", type=int, nargs="+", default=(20, 50, 100))
    parser.add_argument("--score-chunk-size", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = run_sharded_cascade(
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
