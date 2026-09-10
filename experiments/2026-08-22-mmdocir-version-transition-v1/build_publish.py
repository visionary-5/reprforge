#!/usr/bin/env python3
"""Build a disposable SQ8 serving index, admit the generation, and publish it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--terminal-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays(paths: list[Path], torch: Any, dimension: int) -> Iterator[np.ndarray]:
    for path in paths:
        pages = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(pages, list) or not pages:
            raise RuntimeError(f"invalid terminal shard: {path}")
        for tensor in pages:
            if tensor.ndim != 2 or tensor.dtype != torch.float32 or int(tensor.shape[1]) != dimension:
                raise RuntimeError(f"terminal contract changed: {path} {tensor.shape} {tensor.dtype}")
            value = tensor.numpy()
            yield value if value.flags.c_contiguous else np.ascontiguousarray(value)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def main() -> None:
    args = parse_args()
    serving_root = args.generation_root / "serving"
    manifest_path = args.generation_root / "MANIFEST.json"
    pointer_path = args.deployment_root / "ACTIVE.json"
    if args.output.exists() or serving_root.exists() or manifest_path.exists() or pointer_path.exists():
        raise FileExistsError("serving, manifest, output, and ACTIVE targets must be new")

    import faiss
    import torch

    protocol = json.loads(args.protocol.read_text())
    replay = json.loads(args.replay_result.read_text())
    if replay["protocol"] != protocol["protocol_id"] or not replay["all_gates_pass"]:
        raise RuntimeError("replayed generation is not admitted")
    frozen = protocol["serving"]
    reference_files = {
        "metadata.json": frozen["reference_metadata_sha256"],
        "queries.pt": frozen["reference_queries_sha256"],
        "exact-global-maxsim.npy": frozen["reference_exact_scores_sha256"],
    }
    for name, expected in reference_files.items():
        if sha256(args.reference_root / name) != expected:
            raise RuntimeError(f"global reference changed: {name}")

    threads = int(frozen["faiss_threads"])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(threads)
    paths = sorted(args.terminal_root.glob("doc-*.pt"))
    if len(paths) != int(protocol["benchmark"]["documents"]):
        raise RuntimeError("terminal document count changed")
    serving_root.mkdir(parents=True)

    materialization_started = time.perf_counter()
    training_chunks: list[np.ndarray] = []
    page_lengths: list[int] = []
    scan_started = time.perf_counter()
    for value in arrays(paths, torch, int(frozen["dimension"])):
        page_lengths.append(len(value))
        count = min(int(frozen["training_vectors_per_page"]), len(value))
        selected = np.linspace(0, len(value) - 1, num=count, dtype=np.int64)
        training_chunks.append(value[selected].copy())
    training = np.concatenate(training_chunks, axis=0)
    del training_chunks
    scan_seconds = time.perf_counter() - scan_started
    vectors = int(sum(page_lengths))
    if len(page_lengths) != int(protocol["benchmark"]["pages"]) or vectors != int(frozen["vectors"]):
        raise RuntimeError("complete terminal vector scope changed")

    quantizer = faiss.IndexFlatIP(int(frozen["dimension"]))
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        int(frozen["dimension"]),
        int(frozen["nlist"]),
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT,
    )
    train_started = time.perf_counter()
    index.train(training)
    train_seconds = time.perf_counter() - train_started
    del training

    mapping_path = serving_root / "token-to-page.i32"
    offsets_path = serving_root / "page-offsets.i64"
    mapping = np.memmap(mapping_path, mode="w+", dtype=np.int32, shape=(vectors,))
    offsets = np.memmap(offsets_path, mode="w+", dtype=np.int64, shape=(len(page_lengths) + 1,))
    offsets[0] = 0
    cursor = 0
    for page_id, length in enumerate(page_lengths):
        mapping[cursor:cursor + length] = page_id
        cursor += length
        offsets[page_id + 1] = cursor
    mapping.flush()
    offsets.flush()

    pending: list[np.ndarray] = []
    pending_vectors = 0
    add_started = time.perf_counter()

    def flush() -> None:
        nonlocal pending, pending_vectors
        if pending:
            index.add(np.concatenate(pending, axis=0))
            pending = []
            pending_vectors = 0

    for value in arrays(paths, torch, int(frozen["dimension"])):
        pending.append(value)
        pending_vectors += len(value)
        if pending_vectors >= int(frozen["add_batch_vectors"]):
            flush()
    flush()
    add_seconds = time.perf_counter() - add_started
    if int(index.ntotal) != vectors:
        raise RuntimeError("FAISS add is incomplete")

    index_path = serving_root / "sq8.faiss"
    serialize_started = time.perf_counter()
    faiss.write_index(index, str(index_path))
    serialize_seconds = time.perf_counter() - serialize_started
    serving_materialization_seconds = time.perf_counter() - materialization_started

    queries = torch.load(args.reference_root / "queries.pt", weights_only=True)
    exact = np.load(args.reference_root / "exact-global-maxsim.npy", mmap_mode="r")
    metadata = json.loads((args.reference_root / "metadata.json").read_text())
    if len(queries) != exact.shape[0] or len(metadata["queries"]) != len(queries):
        raise RuntimeError("global reference query scope changed")
    exact_top10 = [
        set(np.lexsort((np.arange(exact.shape[1]), -np.asarray(row)))[:10].tolist())
        for row in exact
    ]
    index.nprobe = int(frozen["nprobe"])
    candidate_rows = []
    for query_index, query in enumerate(queries):
        started = time.perf_counter()
        _, identifiers = index.search(query.numpy().astype(np.float32, copy=False), int(frozen["token_k"]))
        candidates = np.unique(mapping[identifiers[identifiers >= 0]])
        latency_ms = (time.perf_counter() - started) * 1000.0
        candidate_rows.append({
            "query_id": metadata["queries"][query_index]["query_id"],
            "domain": metadata["queries"][query_index]["domain"],
            "candidate_pages": len(candidates),
            "candidate_corpus_fraction": len(candidates) / len(page_lengths),
            "full_exact_top10_candidate_recall": len(exact_top10[query_index] & set(candidates.tolist())) / 10.0,
            "search_ms": latency_ms,
        })
    mean_candidate_recall = statistics.fmean(row["full_exact_top10_candidate_recall"] for row in candidate_rows)

    hash_started = time.perf_counter()
    artifact_hashes = {
        str(path.relative_to(args.generation_root)): sha256(path)
        for path in [*paths, index_path, mapping_path, offsets_path]
    }
    hash_seconds = time.perf_counter() - hash_started
    serving_bytes = sum(path.stat().st_size for path in (index_path, mapping_path, offsets_path))
    terminal_bytes = sum(path.stat().st_size for path in paths)
    manifest = {
        "generation": args.generation_root.name,
        "protocol": protocol["protocol_id"],
        "replay_result_sha256": sha256(args.replay_result),
        "scope": {"documents": len(paths), "pages": len(page_lengths), "vectors": vectors},
        "quality": replay["quality"],
        "candidate_fidelity": mean_candidate_recall,
        "artifacts": artifact_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_sha256 = sha256(manifest_path)

    shared_seconds = (
        float(replay["timing"]["query_encoding_seconds"])
        + float(replay["timing"]["quality_validation_seconds"])
        + serving_materialization_seconds
        + hash_seconds
    )
    warm_seconds = float(replay["timing"]["warm_build_seconds_before_serving_index"]) + shared_seconds
    cold_seconds = float(replay["timing"]["cold_build_seconds_before_serving_index"]) + shared_seconds
    transition_saving = 1.0 - warm_seconds / cold_seconds
    gates = {
        "replay_admitted": replay["all_gates_pass"],
        "complete_scope": len(paths) == int(protocol["benchmark"]["documents"]) and len(page_lengths) == int(protocol["benchmark"]["pages"]),
        "candidate_fidelity": mean_candidate_recall >= float(protocol["primary_gates"]["minimum_full_top10_candidate_recall"]),
        "end_to_end_transition": transition_saving >= float(protocol["primary_gates"]["minimum_end_to_end_transition_saving"]),
        "artifact_hashes": len(artifact_hashes) == len(paths) + 3,
    }
    publication_started = time.perf_counter()
    published = False
    if all(gates.values()):
        args.deployment_root.mkdir(parents=True, exist_ok=True)
        pointer = {
            "generation": args.generation_root.name,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
        }
        temporary = args.deployment_root / "ACTIVE.json.partial"
        temporary.write_text(json.dumps(pointer, indent=2) + "\n")
        os.replace(temporary, pointer_path)
        observed = json.loads(pointer_path.read_text())
        published = observed == pointer and sha256(Path(observed["manifest"])) == observed["manifest_sha256"]
    publication_seconds = time.perf_counter() - publication_started
    gates["atomic_publication"] = published

    generation_bytes = terminal_bytes + serving_bytes + manifest_path.stat().st_size
    references = protocol["frozen_references"]
    result: dict[str, Any] = {
        "protocol": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "scope": {"documents": len(paths), "pages": len(page_lengths), "vectors": vectors, "global_probe_queries": len(candidate_rows)},
        "quality": replay["quality"],
        "candidate_fidelity": {
            "mean_full_exact_top10_candidate_recall": mean_candidate_recall,
            "mean_candidate_pages": statistics.fmean(row["candidate_pages"] for row in candidate_rows),
            "mean_candidate_corpus_fraction": statistics.fmean(row["candidate_corpus_fraction"] for row in candidate_rows),
            "search_latency_ms": {
                "median": statistics.median(row["search_ms"] for row in candidate_rows),
                "p95": percentile([row["search_ms"] for row in candidate_rows], 0.95),
            },
            "per_query": candidate_rows,
        },
        "timing": {
            "replay": replay["timing"],
            "serving_shared_scan_seconds": scan_seconds,
            "serving_train_seconds": train_seconds,
            "serving_add_seconds": add_seconds,
            "serving_serialize_seconds": serialize_seconds,
            "serving_materialization_seconds": serving_materialization_seconds,
            "generation_hash_seconds": hash_seconds,
            "publication_seconds": publication_seconds,
            "warm_transition_seconds": warm_seconds + publication_seconds,
            "raw_rebuild_transition_seconds": cold_seconds + publication_seconds,
            "end_to_end_transition_saving": transition_saving,
        },
        "storage": {
            "physical_ir_bytes": int(references["physical_ir_bytes"]),
            "old_terminal_bytes": int(references["old_terminal_bytes"]),
            "new_terminal_bytes": terminal_bytes,
            "new_sq8_serving_bytes": serving_bytes,
            "new_generation_bytes": generation_bytes,
            "steady_state_ir_plus_generation_bytes": int(references["physical_ir_bytes"]) + generation_bytes,
            "transition_peak_lower_bound_bytes": int(references["physical_ir_bytes"]) + int(references["old_terminal_bytes"]) + generation_bytes,
        },
        "publication": {
            "active_pointer": str(pointer_path),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "published": published,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
