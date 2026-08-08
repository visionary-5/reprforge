#!/usr/bin/env python3
"""Benchmark persistent/transient Omni candidate-closure execution on GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from tools.rerank_omni_candidates import load_candidates


def _percentiles(values) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _load_split(path: Path, query_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text())
    evaluation_fold = int(payload["evaluation_fold"])
    assignments = payload["queries"]
    if set(assignments) != set(query_ids):
        raise ValueError("query split IDs do not match query embeddings")
    history = [query_id for query_id in query_ids if int(assignments[query_id]) != evaluation_fold]
    evaluation = [query_id for query_id in query_ids if int(assignments[query_id]) == evaluation_fold]
    return history, evaluation


def _score_batches(torch, query, query_mask, batches, *, chunk_size: int):
    scores = []
    for vectors, masks in batches:
        for start in range(0, len(vectors), chunk_size):
            docs = vectors[start : start + chunk_size]
            doc_masks = masks[start : start + chunk_size]
            with torch.inference_mode(), torch.amp.autocast(
                device_type=query.device.type, enabled=False
            ):
                similarity = torch.einsum("qd,csd->cqs", query.float(), docs.float())
                similarity = similarity.masked_fill(
                    ~doc_masks.bool().unsqueeze(1), float("-inf")
                )
                similarity = similarity.amax(dim=-1)
                similarity = similarity.masked_fill(~query_mask.unsqueeze(0), 0)
                chunk_scores = similarity.sum(dim=-1)
            scores.extend(float(value) for value in chunk_scores.cpu().tolist())
    return scores


def run(
    index_paths: Sequence[Path],
    query_embeddings_path: Path,
    query_masks_path: Path,
    candidate_ranking_path: Path,
    query_split_path: Path,
    *,
    candidate_depth: int,
    persistent_fraction: float,
    repetitions: int,
    warmup_queries: int,
    score_chunk_size: int,
    device: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if not 0.0 <= persistent_fraction <= 1.0:
        raise ValueError("persistent_fraction must lie in [0, 1]")
    if candidate_depth <= 0 or repetitions <= 0 or warmup_queries < 0:
        raise ValueError("invalid benchmark controls")
    candidates = load_candidates(candidate_ranking_path, expected_depth=100)
    with query_embeddings_path.open("rb") as handle:
        query_embeddings, query_ids = pickle.load(handle)
    with query_masks_path.open("rb") as handle:
        query_masks, mask_query_ids = pickle.load(handle)
    query_ids = [str(value) for value in query_ids]
    mask_query_ids = [str(value) for value in mask_query_ids]
    query_embeddings = np.asarray(query_embeddings)
    query_masks = np.asarray(query_masks)
    if query_ids != mask_query_ids or set(query_ids) != set(candidates):
        raise ValueError("queries, masks, and candidates do not align")
    history_ids, evaluation_ids = _load_split(query_split_path, query_ids)
    query_positions = {query_id: index for index, query_id in enumerate(query_ids)}

    load_started = time.perf_counter()
    shards = []
    locations: dict[str, tuple[int, int]] = {}
    for shard_index, index_path in enumerate(map(Path, index_paths)):
        with (index_path / "metadata.pkl").open("rb") as handle:
            metadata = pickle.load(handle)
        doc_ids = [str(value) for value in metadata["doc_ids"]]
        tensor = torch.load(
            index_path / "index.pt", map_location="cpu", mmap=True, weights_only=True
        )
        masks = torch.load(
            index_path / "masks.pt", map_location="cpu", mmap=True, weights_only=True
        )
        if tensor.ndim != 3 or masks.shape != tensor.shape[:2] or len(doc_ids) != len(tensor):
            raise ValueError(f"unexpected shard layout: {index_path}")
        for position, doc_id in enumerate(doc_ids):
            if doc_id in locations:
                raise ValueError(f"duplicate document across shards: {doc_id}")
            locations[doc_id] = (shard_index, position)
        shards.append({"path": index_path, "tensor": tensor, "masks": masks})
    mmap_load_seconds = time.perf_counter() - load_started
    unknown = {
        doc_id
        for query_id in query_ids
        for doc_id in candidates[query_id][:candidate_depth]
        if doc_id not in locations
    }
    if unknown:
        raise ValueError(f"candidate documents missing from shards: {sorted(unknown)[:5]}")

    frequency = Counter(
        doc_id
        for query_id in history_ids
        for doc_id in candidates[query_id][:candidate_depth]
    )
    corpus_pages = len(locations)
    persistent_count = int(math.ceil(persistent_fraction * corpus_pages))
    doc_order = sorted(locations)
    persistent_ids = sorted(
        doc_order,
        key=lambda doc_id: (-frequency[doc_id], int(doc_id) if doc_id.isdigit() else doc_id),
    )[:persistent_count]

    torch.cuda.synchronize()
    persistent_started = time.perf_counter()
    persistent_vectors = []
    persistent_masks = []
    persistent_positions = {}
    for doc_id in persistent_ids:
        shard_index, position = locations[doc_id]
        persistent_positions[doc_id] = len(persistent_vectors)
        persistent_vectors.append(shards[shard_index]["tensor"][position])
        persistent_masks.append(shards[shard_index]["masks"][position])
    if persistent_vectors:
        resident_vectors = torch.stack(persistent_vectors).to(device=device)
        resident_masks = torch.stack(persistent_masks).to(device=device)
    else:
        sample = shards[0]["tensor"]
        sample_mask = shards[0]["masks"]
        resident_vectors = torch.empty(
            (0, sample.shape[1], sample.shape[2]), dtype=sample.dtype, device=device
        )
        resident_masks = torch.empty(
            (0, sample_mask.shape[1]), dtype=sample_mask.dtype, device=device
        )
    torch.cuda.synchronize()
    persistent_h2d_seconds = time.perf_counter() - persistent_started

    def execute(query_id: str, method: str) -> dict[str, Any]:
        query_offset = query_positions[query_id]
        started = time.perf_counter()
        query = torch.from_numpy(query_embeddings[query_offset]).to(device=device)
        query_mask = torch.from_numpy(query_masks[query_offset]).to(
            device=device, dtype=torch.bool
        )
        docs = candidates[query_id][:candidate_depth]
        resident_docs = [doc_id for doc_id in docs if method == "closure" and doc_id in persistent_positions]
        transient_docs = [doc_id for doc_id in docs if doc_id not in set(resident_docs)]

        h2d_started = time.perf_counter()
        transient_by_shard: dict[int, list[int]] = defaultdict(list)
        transient_ids_by_shard: dict[int, list[str]] = defaultdict(list)
        for doc_id in transient_docs:
            shard_index, position = locations[doc_id]
            transient_by_shard[shard_index].append(position)
            transient_ids_by_shard[shard_index].append(doc_id)
        transient_batches = []
        transient_scored_docs = []
        for shard_index, positions in transient_by_shard.items():
            vectors = shards[shard_index]["tensor"][positions].to(device=device)
            masks = shards[shard_index]["masks"][positions].to(device=device)
            transient_batches.append((vectors, masks))
            transient_scored_docs.extend(transient_ids_by_shard[shard_index])
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - h2d_started) * 1000.0

        resident_batches = []
        if resident_docs:
            positions = torch.tensor(
                [persistent_positions[doc_id] for doc_id in resident_docs],
                device=device,
                dtype=torch.int64,
            )
            resident_batches.append(
                (
                    resident_vectors.index_select(0, positions),
                    resident_masks.index_select(0, positions),
                )
            )
        score_started = time.perf_counter()
        transient_scores = _score_batches(
            torch, query, query_mask, transient_batches, chunk_size=score_chunk_size
        )
        resident_scores = _score_batches(
            torch, query, query_mask, resident_batches, chunk_size=score_chunk_size
        )
        torch.cuda.synchronize()
        maxsim_ms = (time.perf_counter() - score_started) * 1000.0
        score_map = dict(zip(transient_scored_docs, transient_scores, strict=True))
        score_map.update(zip(resident_docs, resident_scores, strict=True))
        ranking = sorted(docs, key=lambda doc_id: (-score_map[doc_id], docs.index(doc_id)))
        total_ms = (time.perf_counter() - started) * 1000.0
        digest = hashlib.sha256(
            json.dumps(ranking, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "h2d_ms": h2d_ms,
            "maxsim_ms": maxsim_ms,
            "end_to_end_ms": total_ms,
            "persistent_hits": len(resident_docs),
            "transient_pages": len(transient_docs),
            "ranking_sha256": digest,
        }

    for query_id in evaluation_ids[:warmup_queries]:
        execute(query_id, "defer")
        execute(query_id, "closure")
    measurements = {"defer": [], "closure": []}
    for repetition in range(repetitions):
        for query_offset, query_id in enumerate(evaluation_ids):
            methods = ("defer", "closure") if (repetition + query_offset) % 2 == 0 else ("closure", "defer")
            rows = {method: execute(query_id, method) for method in methods}
            if rows["defer"]["ranking_sha256"] != rows["closure"]["ranking_sha256"]:
                raise AssertionError(f"closure changed candidate ranking for query {query_id}")
            for method in methods:
                measurements[method].append(rows[method])

    def summarize(rows):
        return {
            "measurements": len(rows),
            "h2d_ms": _percentiles([row["h2d_ms"] for row in rows]),
            "maxsim_ms": _percentiles([row["maxsim_ms"] for row in rows]),
            "end_to_end_ms": _percentiles([row["end_to_end_ms"] for row in rows]),
            "persistent_hits": _percentiles([row["persistent_hits"] for row in rows]),
            "transient_pages": _percentiles([row["transient_pages"] for row in rows]),
        }

    return {
        "schema_version": 1,
        "protocol": "omni-query-scope-closure-runtime-v0",
        "device": device,
        "candidate_depth": candidate_depth,
        "persistent_fraction": persistent_fraction,
        "persistent_pages": persistent_count,
        "corpus_pages": corpus_pages,
        "history_queries": len(history_ids),
        "evaluation_queries": len(evaluation_ids),
        "repetitions": repetitions,
        "mmap_load_seconds": mmap_load_seconds,
        "persistent_initial_h2d_seconds": persistent_h2d_seconds,
        "persistent_resident_vector_bytes": int(
            resident_vectors.numel() * resident_vectors.element_size()
            + resident_masks.numel() * resident_masks.element_size()
        ),
        "methods": {name: summarize(rows) for name, rows in measurements.items()},
        "ranking_parity": True,
        "cost_scope": (
            "query tensor H2D + host/mmap selected Full tensor H2D + exact MaxSim; "
            "raw-page decoding and VLM representation construction excluded"
        ),
        "os_page_cache_state": "uncontrolled; alternating method order",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--query-masks", type=Path, required=True)
    parser.add_argument("--candidate-ranking", type=Path, required=True)
    parser.add_argument("--query-splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=int, required=True)
    parser.add_argument("--persistent-fraction", type=float, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-queries", type=int, default=5)
    parser.add_argument("--score-chunk-size", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    result = run(
        args.index,
        args.query_embeddings,
        args.query_masks,
        args.candidate_ranking,
        args.query_splits,
        candidate_depth=args.candidate_depth,
        persistent_fraction=args.persistent_fraction,
        repetitions=args.repetitions,
        warmup_queries=args.warmup_queries,
        score_chunk_size=args.score_chunk_size,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
