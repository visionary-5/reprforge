#!/usr/bin/env python3
"""Compile and execute compact heterogeneous multi-vector indexes.

The planner chooses one representation route per item.  This module turns that
logical plan into a physical index:

* each active route is stored as one contiguous vector shard;
* variable item lengths are represented by offsets, never storage padding;
* a global item order makes scores comparable with ``policy_replay``;
* NumPy and Torch runtimes execute the same MaxSim contract;
* the benchmark reports compact bytes separately from padded resident bytes.

The on-disk format is intentionally plain NumPy + JSON.  It is inspectable,
memory-mappable, and does not require a model or CUDA to validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_STORAGE_DTYPES = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
}


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str, *, kind: str) -> str:
    if not value or not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"{kind} {value!r} must match {_SAFE_NAME.pattern}"
        )
    return value


def _numpy_embedding(value: Any) -> np.ndarray:
    """Convert a CPU/GPU tensor-like value into a contiguous rank-2 array."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"embedding must be rank 2, got shape {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"embedding dimensions must be positive: {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"embedding dtype must be floating, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("embedding contains a non-finite value")
    return np.ascontiguousarray(array)


def _write_shard(
    output: Path,
    *,
    identifiers: Sequence[str],
    embeddings: Sequence[Any],
    storage_dtype: str,
) -> dict[str, Any]:
    if len(identifiers) != len(embeddings):
        raise ValueError("identifier and embedding counts differ")
    if not identifiers:
        raise ValueError("cannot write an empty embedding shard")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("embedding identifiers must be unique")
    dtype = _STORAGE_DTYPES.get(storage_dtype)
    if dtype is None:
        raise ValueError(f"unsupported storage dtype: {storage_dtype}")

    arrays = [_numpy_embedding(value) for value in embeddings]
    dimension = int(arrays[0].shape[1])
    if any(int(array.shape[1]) != dimension for array in arrays):
        raise ValueError("all embeddings in one shard must share a dimension")
    lengths = np.asarray([array.shape[0] for array in arrays], dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    vectors = np.empty((int(offsets[-1]), dimension), dtype=dtype)
    for index, array in enumerate(arrays):
        vectors[offsets[index] : offsets[index + 1]] = array.astype(
            dtype,
            copy=False,
        )

    output.mkdir(parents=True, exist_ok=True)
    vectors_path = output / "vectors.npy"
    offsets_path = output / "offsets.npy"
    ids_path = output / "ids.json"
    np.save(vectors_path, vectors, allow_pickle=False)
    np.save(offsets_path, offsets, allow_pickle=False)
    _json(ids_path, list(identifiers))
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (vectors_path, offsets_path, ids_path)
    }
    return {
        "count": len(identifiers),
        "dimension": dimension,
        "dtype": dtype.name,
        "vectors": int(offsets[-1]),
        "vector_bytes": int(vectors.nbytes),
        "min_vectors_per_item": int(lengths.min()),
        "max_vectors_per_item": int(lengths.max()),
        "files": files,
    }


def write_embedding_bank(
    output: Path,
    *,
    item_ids: Sequence[str],
    route_embeddings: Mapping[str, Sequence[Any]],
    query_ids: Sequence[str] | None = None,
    query_embeddings: Sequence[Any] | None = None,
    storage_dtype: str = "float32",
) -> dict[str, Any]:
    """Persist all measured route embeddings before a plan is selected."""

    if output.exists():
        raise FileExistsError(f"embedding bank already exists: {output}")
    if not item_ids or len(set(item_ids)) != len(item_ids):
        raise ValueError("item identifiers must be non-empty and unique")
    if not route_embeddings:
        raise ValueError("route bank requires at least one route")
    output.mkdir(parents=True)

    routes: dict[str, dict[str, Any]] = {}
    for route in sorted(route_embeddings):
        _safe_name(route, kind="route")
        values = route_embeddings[route]
        if len(values) != len(item_ids):
            raise ValueError(
                f"{route} emitted {len(values)} embeddings for {len(item_ids)} items"
            )
        routes[route] = _write_shard(
            output / "routes" / route,
            identifiers=item_ids,
            embeddings=values,
            storage_dtype=storage_dtype,
        )

    if (query_ids is None) != (query_embeddings is None):
        raise ValueError("query identifiers and embeddings must be provided together")
    query_manifest = None
    if query_ids is not None and query_embeddings is not None:
        query_manifest = _write_shard(
            output / "queries",
            identifiers=query_ids,
            embeddings=query_embeddings,
            storage_dtype=storage_dtype,
        )

    manifest = {
        "format": "reprforge-embedding-bank",
        "format_version": FORMAT_VERSION,
        "storage_dtype": storage_dtype,
        "item_count": len(item_ids),
        "routes": routes,
        "queries": query_manifest,
    }
    _json(output / "manifest.json", manifest)
    return manifest


def _read_manifest(path: Path, expected_format: str) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != expected_format:
        raise ValueError(
            f"{manifest_path} has format {payload.get('format')!r}, "
            f"expected {expected_format!r}"
        )
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported format version in {manifest_path}")
    return payload


def _load_shard(path: Path, *, mmap_mode: str | None = "r") -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
]:
    ids = json.loads((path / "ids.json").read_text(encoding="utf-8"))
    vectors = np.load(path / "vectors.npy", mmap_mode=mmap_mode, allow_pickle=False)
    offsets = np.load(path / "offsets.npy", mmap_mode=mmap_mode, allow_pickle=False)
    if vectors.ndim != 2 or offsets.shape != (len(ids) + 1,):
        raise ValueError(f"invalid embedding shard shape under {path}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != vectors.shape[0]:
        raise ValueError(f"invalid offsets under {path}")
    if np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError(f"every embedding must contain at least one vector: {path}")
    return [str(value) for value in ids], vectors, offsets


def merge_embedding_banks(inputs: Sequence[Path], output: Path) -> dict[str, Any]:
    """Merge document-local banks into one corpus bank deterministically."""

    if not inputs:
        raise ValueError("at least one embedding bank is required")
    manifests = [
        _read_manifest(path, "reprforge-embedding-bank") for path in inputs
    ]
    route_sets = [set(manifest["routes"]) for manifest in manifests]
    if any(routes != route_sets[0] for routes in route_sets[1:]):
        raise ValueError("embedding banks expose different route sets")
    dtypes = {str(manifest["storage_dtype"]) for manifest in manifests}
    if len(dtypes) != 1:
        raise ValueError("embedding banks use different storage dtypes")

    all_item_ids: list[str] = []
    combined_routes: dict[str, list[np.ndarray]] = {
        route: [] for route in sorted(route_sets[0])
    }
    all_query_ids: list[str] = []
    combined_queries: list[np.ndarray] = []
    for path, manifest in zip(inputs, manifests, strict=True):
        route_ids: list[str] | None = None
        for route in combined_routes:
            ids, vectors, offsets = _load_shard(path / "routes" / route)
            if route_ids is None:
                route_ids = ids
            elif route_ids != ids:
                raise ValueError(f"route item order differs in {path}")
            combined_routes[route].extend(
                np.asarray(vectors[offsets[i] : offsets[i + 1]])
                for i in range(len(ids))
            )
        assert route_ids is not None
        all_item_ids.extend(route_ids)
        if manifest.get("queries") is not None:
            ids, vectors, offsets = _load_shard(path / "queries")
            all_query_ids.extend(ids)
            combined_queries.extend(
                np.asarray(vectors[offsets[i] : offsets[i + 1]])
                for i in range(len(ids))
            )
    if len(set(all_item_ids)) != len(all_item_ids):
        raise ValueError("merged banks contain duplicate item identifiers")
    if len(set(all_query_ids)) != len(all_query_ids):
        raise ValueError("merged banks contain duplicate query identifiers")

    return write_embedding_bank(
        output,
        item_ids=all_item_ids,
        route_embeddings=combined_routes,
        query_ids=all_query_ids or None,
        query_embeddings=combined_queries or None,
        storage_dtype=dtypes.pop(),
    )


def _select_payload(payload: Any, dotted_key: str | None) -> Any:
    if dotted_key:
        for component in dotted_key.split("."):
            if not isinstance(payload, Mapping) or component not in payload:
                raise KeyError(f"plan key {dotted_key!r} does not exist")
            payload = payload[component]
    if isinstance(payload, Mapping) and "plan" in payload:
        payload = payload["plan"]
    if not isinstance(payload, Mapping):
        raise ValueError("plan payload must be a mapping")
    return {str(key): str(value) for key, value in payload.items()}


def load_plan(path: Path, *, dotted_key: str | None = None) -> dict[str, str]:
    return _select_payload(
        json.loads(path.read_text(encoding="utf-8")),
        dotted_key,
    )


def compile_heterogeneous_index(
    *,
    bank: Path,
    plan: Mapping[str, str],
    output: Path,
    storage_dtype: str | None = None,
) -> dict[str, Any]:
    """Compile exactly one selected embedding per item into compact shards."""

    bank_manifest = _read_manifest(bank, "reprforge-embedding-bank")
    target_dtype = storage_dtype or str(bank_manifest["storage_dtype"])
    if target_dtype not in _STORAGE_DTYPES:
        raise ValueError(f"unsupported storage dtype: {target_dtype}")
    route_data: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    canonical_ids: list[str] | None = None
    for route in sorted(bank_manifest["routes"]):
        values = _load_shard(bank / "routes" / route)
        route_data[route] = values
        if canonical_ids is None:
            canonical_ids = values[0]
        elif values[0] != canonical_ids:
            raise ValueError("route shards do not share the same item order")
    assert canonical_ids is not None
    if set(plan) != set(canonical_ids):
        missing = sorted(set(canonical_ids) - set(plan))
        extra = sorted(set(plan) - set(canonical_ids))
        raise ValueError(
            f"plan/item mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    unknown_routes = set(plan.values()) - set(route_data)
    if unknown_routes:
        raise ValueError(f"plan uses unknown routes: {sorted(unknown_routes)}")
    dimensions = {
        int(data[1].shape[1])
        for route, data in route_data.items()
        if route in set(plan.values())
    }
    if len(dimensions) != 1:
        raise ValueError("selected routes do not share one embedding dimension")
    if output.exists():
        raise FileExistsError(f"compiled index already exists: {output}")
    output.mkdir(parents=True)

    positions = {
        route: {item_id: index for index, item_id in enumerate(data[0])}
        for route, data in route_data.items()
    }
    selected_by_route: dict[str, list[str]] = {}
    for item_id in canonical_ids:
        selected_by_route.setdefault(plan[item_id], []).append(item_id)

    route_manifests: dict[str, dict[str, Any]] = {}
    compact_vector_bytes = 0
    for route in sorted(selected_by_route):
        _, vectors, offsets = route_data[route]
        ids = selected_by_route[route]
        embeddings = []
        for item_id in ids:
            index = positions[route][item_id]
            embeddings.append(
                np.asarray(vectors[offsets[index] : offsets[index + 1]])
            )
        shard_manifest = _write_shard(
            output / "routes" / route,
            identifiers=ids,
            embeddings=embeddings,
            storage_dtype=target_dtype,
        )
        compact_vector_bytes += int(shard_manifest["vector_bytes"])
        route_manifests[route] = shard_manifest

    item_records = [
        {
            "item_id": item_id,
            "route": plan[item_id],
            "global_position": position,
        }
        for position, item_id in enumerate(canonical_ids)
    ]
    _json(output / "items.json", item_records)
    serialized_bytes_without_manifest = sum(
        path.stat().st_size
        for path in output.rglob("*")
        if path.is_file()
    )
    manifest = {
        "format": "reprforge-heterogeneous-index",
        "format_version": FORMAT_VERSION,
        "source_bank": str(bank),
        "storage_dtype": target_dtype,
        "dimension": dimensions.pop(),
        "item_count": len(canonical_ids),
        "compact_vector_bytes": compact_vector_bytes,
        "serialized_bytes_without_manifest": serialized_bytes_without_manifest,
        "storage_padding_bytes": 0,
        "routes": route_manifests,
        "route_counts": {
            route: len(ids) for route, ids in sorted(selected_by_route.items())
        },
    }
    _json(output / "manifest.json", manifest)
    # ``serialized_bytes`` includes this manifest, so iterate until its own
    # decimal field width is stable.
    for _ in range(5):
        serialized_bytes = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        if manifest.get("serialized_bytes") == serialized_bytes:
            break
        manifest["serialized_bytes"] = serialized_bytes
        _json(output / "manifest.json", manifest)
    return manifest


def amplify_compiled_index(
    *,
    source: Path,
    factor: int,
    output: Path,
) -> dict[str, Any]:
    """Physically replicate a verified index for candidate-scaling tests.

    Replicas receive distinct identifiers and occupy distinct vector storage.
    This is a systems-only workload: duplicated candidates invalidate ordinary
    retrieval quality metrics and must never be reported as a larger labeled
    corpus.
    """

    if factor < 1:
        raise ValueError("amplification factor must be positive")
    source_manifest = _read_manifest(
        source,
        "reprforge-heterogeneous-index",
    )
    if output.exists():
        raise FileExistsError(f"amplified index already exists: {output}")
    source_records = json.loads(
        (source / "items.json").read_text(encoding="utf-8")
    )
    source_item_ids = [str(row["item_id"]) for row in source_records]
    source_routes = {
        str(row["item_id"]): str(row["route"]) for row in source_records
    }
    amplified_item_ids = [
        f"{item_id}#replica-{replica}"
        for replica in range(factor)
        for item_id in source_item_ids
    ]
    amplified_routes = {
        f"{item_id}#replica-{replica}": source_routes[item_id]
        for replica in range(factor)
        for item_id in source_item_ids
    }

    output.mkdir(parents=True)
    route_manifests: dict[str, dict[str, Any]] = {}
    compact_vector_bytes = 0
    for route in sorted(source_manifest["routes"]):
        ids, vectors, offsets = _load_shard(source / "routes" / route)
        embeddings = [
            np.asarray(vectors[offsets[index] : offsets[index + 1]])
            for _ in range(factor)
            for index in range(len(ids))
        ]
        amplified_ids = [
            f"{item_id}#replica-{replica}"
            for replica in range(factor)
            for item_id in ids
        ]
        shard_manifest = _write_shard(
            output / "routes" / route,
            identifiers=amplified_ids,
            embeddings=embeddings,
            storage_dtype=str(source_manifest["storage_dtype"]),
        )
        compact_vector_bytes += int(shard_manifest["vector_bytes"])
        route_manifests[route] = shard_manifest

    item_records = [
        {
            "item_id": item_id,
            "route": amplified_routes[item_id],
            "global_position": position,
        }
        for position, item_id in enumerate(amplified_item_ids)
    ]
    _json(output / "items.json", item_records)
    manifest = {
        "format": "reprforge-heterogeneous-index",
        "format_version": FORMAT_VERSION,
        "source_index": str(source),
        "systems_only_amplification": True,
        "amplification_factor": factor,
        "quality_labels_valid": False,
        "storage_dtype": str(source_manifest["storage_dtype"]),
        "dimension": int(source_manifest["dimension"]),
        "item_count": len(amplified_item_ids),
        "compact_vector_bytes": compact_vector_bytes,
        "storage_padding_bytes": 0,
        "routes": route_manifests,
        "route_counts": {
            route: int(values["count"])
            for route, values in sorted(route_manifests.items())
        },
    }
    _json(output / "manifest.json", manifest)
    for _ in range(5):
        serialized_bytes = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        if manifest.get("serialized_bytes") == serialized_bytes:
            break
        manifest["serialized_bytes"] = serialized_bytes
        _json(output / "manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class _IndexShard:
    route: str
    item_ids: tuple[str, ...]
    global_positions: np.ndarray
    vectors: np.ndarray
    offsets: np.ndarray


class NumpyMaxSimRuntime:
    """Reference runtime with exact variable-length storage semantics."""

    def __init__(self, index: Path) -> None:
        self.index = index
        self.manifest = _read_manifest(index, "reprforge-heterogeneous-index")
        records = json.loads((index / "items.json").read_text(encoding="utf-8"))
        self.item_ids = tuple(str(row["item_id"]) for row in records)
        positions = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }
        self.shards: list[_IndexShard] = []
        for route in sorted(self.manifest["routes"]):
            ids, vectors, offsets = _load_shard(index / "routes" / route)
            self.shards.append(
                _IndexShard(
                    route=route,
                    item_ids=tuple(ids),
                    global_positions=np.asarray(
                        [positions[item_id] for item_id in ids],
                        dtype=np.int64,
                    ),
                    vectors=vectors,
                    offsets=offsets,
                )
            )

    @property
    def compact_vector_bytes(self) -> int:
        return int(self.manifest["compact_vector_bytes"])

    @property
    def resident_vector_bytes(self) -> int:
        return sum(int(shard.vectors.nbytes) for shard in self.shards)

    @property
    def resident_unpadded_vector_bytes(self) -> int:
        return self.resident_vector_bytes

    def score(self, query_embedding: Any) -> np.ndarray:
        query = _numpy_embedding(query_embedding).astype(np.float32, copy=False)
        if query.shape[1] != int(self.manifest["dimension"]):
            raise ValueError("query and index dimensions differ")
        output = np.empty(len(self.item_ids), dtype=np.float32)
        for shard in self.shards:
            for local_position, global_position in enumerate(
                shard.global_positions
            ):
                document = np.asarray(
                    shard.vectors[
                        shard.offsets[local_position] :
                        shard.offsets[local_position + 1]
                    ],
                    dtype=np.float32,
                )
                output[global_position] = float(
                    np.max(query @ document.T, axis=1).sum()
                )
        return output

    def search(
        self,
        query_embedding: Any,
        *,
        top_k: int,
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.score(query_embedding)
        ranked = sorted(
            range(len(self.item_ids)),
            key=lambda index: (-float(scores[index]), self.item_ids[index]),
        )
        return [
            (self.item_ids[index], float(scores[index]))
            for index in ranked[:top_k]
        ]


@dataclass(frozen=True)
class _TorchBatch:
    global_positions: Any
    vectors: Any
    lengths: Any


class TorchMaxSimRuntime:
    """Torch CPU/CUDA runtime using global length-bucketed execution batches.

    Representation routes matter while constructing the index, but all
    selected vectors share the same MaxSim contract after compilation.
    Execution therefore erases route boundaries and sorts documents by token
    length before batching. This minimizes both padding and fragmented
    route-local launches for heterogeneous plans.
    """

    def __init__(
        self,
        index: Path,
        *,
        device: str,
        document_batch_size: int = 64,
        token_batch_budget: int | None = None,
    ) -> None:
        if document_batch_size <= 0:
            raise ValueError("document batch size must be positive")
        if token_batch_budget is not None and token_batch_budget <= 0:
            raise ValueError("token batch budget must be positive")
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.token_batch_budget = token_batch_budget
        self.index = index
        self.manifest = _read_manifest(index, "reprforge-heterogeneous-index")
        records = json.loads((index / "items.json").read_text(encoding="utf-8"))
        self.item_ids = tuple(str(row["item_id"]) for row in records)
        positions = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }
        self.batches: list[_TorchBatch] = []
        self._resident_vector_bytes = 0
        self._resident_unpadded_vector_bytes = 0
        documents_with_positions: list[tuple[int, Any]] = []
        for route in sorted(self.manifest["routes"]):
            ids, vectors, offsets = _load_shard(index / "routes" / route)
            for index_in_route, item_id in enumerate(ids):
                documents_with_positions.append(
                    (
                        positions[item_id],
                        torch.from_numpy(
                            np.array(
                                vectors[
                                    offsets[index_in_route] :
                                    offsets[index_in_route + 1]
                                ],
                                dtype=np.float32,
                                copy=True,
                            )
                        ),
                    )
                )
        documents_with_positions.sort(
            key=lambda value: (value[1].shape[0], value[0])
        )
        start = 0
        while start < len(documents_with_positions):
            if token_batch_budget is None:
                end = min(
                    start + document_batch_size,
                    len(documents_with_positions),
                )
            else:
                end = start
                while end < len(documents_with_positions):
                    candidate_count = end - start + 1
                    candidate_max_length = int(
                        documents_with_positions[end][1].shape[0]
                    )
                    if (
                        candidate_count * candidate_max_length
                        > token_batch_budget
                        and end > start
                    ):
                        break
                    end += 1
            selected = documents_with_positions[start:end]
            documents = [value[1] for value in selected]
            lengths = torch.tensor(
                [document.shape[0] for document in documents],
                device=self.device,
                dtype=torch.int64,
            )
            padded = torch.nn.utils.rnn.pad_sequence(
                documents,
                batch_first=True,
            ).to(self.device)
            global_positions = torch.tensor(
                [value[0] for value in selected],
                device=self.device,
                dtype=torch.int64,
            )
            self._resident_vector_bytes += (
                padded.numel() * padded.element_size()
            )
            self._resident_unpadded_vector_bytes += sum(
                document.numel() * document.element_size()
                for document in documents
            )
            self.batches.append(
                _TorchBatch(
                    global_positions=global_positions,
                    vectors=padded,
                    lengths=lengths,
                )
            )
            start = end

    @property
    def compact_vector_bytes(self) -> int:
        return int(self.manifest["compact_vector_bytes"])

    @property
    def resident_vector_bytes(self) -> int:
        return self._resident_vector_bytes

    @property
    def resident_unpadded_vector_bytes(self) -> int:
        return self._resident_unpadded_vector_bytes

    @property
    def execution_batch_count(self) -> int:
        return len(self.batches)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def score(self, query_embedding: Any) -> np.ndarray:
        query = self.torch.as_tensor(
            np.array(
                _numpy_embedding(query_embedding),
                dtype=np.float32,
                copy=True,
            ),
            device=self.device,
            dtype=self.torch.float32,
        )
        if query.shape[1] != int(self.manifest["dimension"]):
            raise ValueError("query and index dimensions differ")
        output = self.torch.empty(
            len(self.item_ids),
            device=self.device,
            dtype=self.torch.float32,
        )
        for batch in self.batches:
            similarities = self.torch.einsum(
                "qd,bkd->bqk",
                query,
                batch.vectors,
            )
            positions = self.torch.arange(
                batch.vectors.shape[1],
                device=self.device,
            )
            similarities = similarities.masked_fill(
                positions[None, None, :] >= batch.lengths[:, None, None],
                float("-inf"),
            )
            values = similarities.max(dim=-1).values.sum(dim=-1)
            output[batch.global_positions] = values
        return output.detach().cpu().numpy()

    def search(
        self,
        query_embedding: Any,
        *,
        top_k: int,
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.score(query_embedding)
        ranked = sorted(
            range(len(self.item_ids)),
            key=lambda index: (-float(scores[index]), self.item_ids[index]),
        )
        return [
            (self.item_ids[index], float(scores[index]))
            for index in ranked[:top_k]
        ]


def load_query_bank(bank: Path) -> tuple[list[str], list[np.ndarray]]:
    manifest = _read_manifest(bank, "reprforge-embedding-bank")
    if manifest.get("queries") is None:
        raise ValueError(f"{bank} does not contain query embeddings")
    ids, vectors, offsets = _load_shard(bank / "queries")
    return ids, [
        np.asarray(vectors[offsets[i] : offsets[i + 1]])
        for i in range(len(ids))
    ]


def benchmark_runtime(
    runtime: Any,
    *,
    query_ids: Sequence[str],
    query_embeddings: Sequence[Any],
    warmup: int,
    repetitions: int,
    top_k: int = 10,
) -> dict[str, Any]:
    if len(query_ids) != len(query_embeddings) or not query_ids:
        raise ValueError("query identifiers and embeddings must be non-empty")
    if warmup < 0 or repetitions <= 0:
        raise ValueError("invalid warmup/repetition count")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    for _ in range(warmup):
        for embedding in query_embeddings:
            runtime.search(embedding, top_k=top_k)
    synchronize = getattr(runtime, "synchronize", None)
    if callable(synchronize):
        synchronize()

    latencies_ms: list[float] = []
    digest = hashlib.sha256()
    for _ in range(repetitions):
        for query_id, embedding in zip(
            query_ids,
            query_embeddings,
            strict=True,
        ):
            if callable(synchronize):
                synchronize()
            began = time.perf_counter()
            results = runtime.search(embedding, top_k=top_k)
            if callable(synchronize):
                synchronize()
            latencies_ms.append((time.perf_counter() - began) * 1000.0)
            digest.update(query_id.encode())
            digest.update(
                json.dumps(
                    results,
                    separators=(",", ":"),
                ).encode()
            )
    values = np.asarray(latencies_ms, dtype=np.float64)
    total_seconds = float(values.sum() / 1000.0)
    return {
        "queries": len(query_ids),
        "index_items": len(runtime.item_ids),
        "warmup": warmup,
        "repetitions": repetitions,
        "top_k": top_k,
        "search_scope": "all-indexed-items",
        "measurements": len(latencies_ms),
        "latency_ms": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        },
        "qps": float(len(values) / total_seconds),
        "compact_vector_bytes": int(runtime.compact_vector_bytes),
        "resident_vector_bytes": int(runtime.resident_vector_bytes),
        "resident_unpadded_vector_bytes": int(
            runtime.resident_unpadded_vector_bytes
        ),
        "resident_padding_bytes": int(
            runtime.resident_vector_bytes
            - runtime.resident_unpadded_vector_bytes
        ),
        "resident_dtype_expansion_bytes": int(
            runtime.resident_unpadded_vector_bytes
            - runtime.compact_vector_bytes
        ),
        "execution_batches": getattr(runtime, "execution_batch_count", None),
        "score_digest_sha256": digest.hexdigest(),
    }


def evaluate_runtime(
    runtime: Any,
    *,
    query_ids: Sequence[str],
    query_embeddings: Sequence[Any],
    replay_directory: Path,
    source_plan: Mapping[str, str],
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compare physical-index scores and quality with frozen route replay."""

    from reprforge.policy_replay import (
        Item,
        ReplayData,
        RouteCost,
        evaluate_plan,
        load_replay_data,
    )

    data = load_replay_data(
        replay_directory / "items.jsonl",
        replay_directory / "queries.jsonl",
        replay_directory / "scores.jsonl",
    )
    if set(source_plan) != {item.item_id for item in data.items}:
        raise ValueError("source plan does not cover the replay item set")
    if tuple(runtime.item_ids) != tuple(item.item_id for item in data.items):
        raise ValueError("compiled index item order differs from replay data")
    if set(query_ids) != {query.query_id for query in data.queries}:
        raise ValueError("query bank differs from replay data")
    query_embeddings_by_id = dict(zip(query_ids, query_embeddings, strict=True))

    runtime_scores: dict[str, dict[str, float]] = {}
    max_absolute_error = 0.0
    max_relative_error = 0.0
    close_pairs = 0
    compared_pairs = 0
    equal_rankings = 0
    for query in data.queries:
        values = runtime.score(query_embeddings_by_id[query.query_id])
        scored = {
            item_id: float(values[position])
            for position, item_id in enumerate(runtime.item_ids)
        }
        runtime_scores[query.query_id] = scored
        candidate_ids = (
            list(query.candidate_item_ids)
            if query.candidate_item_ids is not None
            else list(runtime.item_ids)
        )
        expected = {
            item_id: float(
                data.scores[source_plan[item_id]][query.query_id][item_id]
            )
            for item_id in candidate_ids
        }
        for item_id in candidate_ids:
            difference = abs(scored[item_id] - expected[item_id])
            relative = difference / max(abs(expected[item_id]), 1e-12)
            max_absolute_error = max(max_absolute_error, difference)
            max_relative_error = max(max_relative_error, relative)
            compared_pairs += 1
            if np.isclose(
                scored[item_id],
                expected[item_id],
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ):
                close_pairs += 1
        runtime_ranking = sorted(
            candidate_ids,
            key=lambda item_id: (-scored[item_id], item_id),
        )
        expected_ranking = sorted(
            candidate_ids,
            key=lambda item_id: (-expected[item_id], item_id),
        )
        equal_rankings += int(runtime_ranking == expected_ranking)

    vector_bytes_by_item: dict[str, int] = {}
    dtype = np.dtype(runtime.manifest["storage_dtype"])
    dimension = int(runtime.manifest["dimension"])
    for shard in runtime.shards if hasattr(runtime, "shards") else []:
        for index, item_id in enumerate(shard.item_ids):
            vector_bytes_by_item[item_id] = (
                int(shard.offsets[index + 1] - shard.offsets[index])
                * dimension
                * dtype.itemsize
            )
    if not vector_bytes_by_item:
        # Torch runtime stores packed batches rather than source offsets. Read
        # the compact files so quality evaluation still reports physical cost.
        for route in sorted(runtime.manifest["routes"]):
            ids, _, offsets = _load_shard(runtime.index / "routes" / route)
            for index, item_id in enumerate(ids):
                vector_bytes_by_item[item_id] = (
                    int(offsets[index + 1] - offsets[index])
                    * dimension
                    * dtype.itemsize
                )
    compiled_items = tuple(
        Item(
            item_id=item.item_id,
            content_type=item.content_type,
            route_costs={
                "compiled": RouteCost(
                    index_bytes=vector_bytes_by_item[item.item_id],
                    encode_ms=0.0,
                )
            },
        )
        for item in data.items
    )
    compiled_data = ReplayData(
        items=compiled_items,
        queries=data.queries,
        scores={"compiled": runtime_scores},
    )
    compiled_plan = {
        item.item_id: "compiled" for item in compiled_items
    }
    physical_metrics = evaluate_plan(
        compiled_data,
        compiled_plan,
        ks=(1, 5, 10),
    )
    replay_metrics = evaluate_plan(data, source_plan, ks=(1, 5, 10))
    metric_deltas = {
        name: float(physical_metrics[name] - replay_metrics[name])
        for name in (
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "ndcg_at_1",
            "ndcg_at_5",
            "ndcg_at_10",
        )
    }
    return {
        "score_contract": {
            "compared_pairs": compared_pairs,
            "close_pairs": close_pairs,
            "all_close": close_pairs == compared_pairs,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
        },
        "ranking_contract": {
            "queries": len(data.queries),
            "equal_rankings": equal_rankings,
            "all_equal": equal_rankings == len(data.queries),
        },
        "physical_metrics": physical_metrics,
        "replay_metrics": replay_metrics,
        "metric_deltas": metric_deltas,
        "compact_vector_bytes": int(runtime.compact_vector_bytes),
    }


def _runtime(
    engine: str,
    index: Path,
    *,
    device: str,
    document_batch_size: int,
    token_batch_budget: int | None,
) -> Any:
    if engine == "numpy":
        return NumpyMaxSimRuntime(index)
    if engine == "torch":
        return TorchMaxSimRuntime(
            index,
            device=device,
            document_batch_size=document_batch_size,
            token_batch_budget=token_batch_budget,
        )
    raise ValueError(f"unsupported engine: {engine}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--bank", type=Path, required=True)
    compile_parser.add_argument("--plan", type=Path, required=True)
    compile_parser.add_argument("--plan-key")
    compile_parser.add_argument("--storage-dtype", choices=sorted(_STORAGE_DTYPES))
    compile_parser.add_argument("--output", type=Path, required=True)

    amplify_parser = subparsers.add_parser("amplify")
    amplify_parser.add_argument("--index", type=Path, required=True)
    amplify_parser.add_argument("--factor", type=int, required=True)
    amplify_parser.add_argument("--output", type=Path, required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--index", type=Path, required=True)
    benchmark_parser.add_argument("--query-bank", type=Path, required=True)
    benchmark_parser.add_argument("--engine", choices=("numpy", "torch"), default="torch")
    benchmark_parser.add_argument("--device", default="cuda:0")
    benchmark_parser.add_argument("--document-batch-size", type=int, default=64)
    benchmark_parser.add_argument("--token-batch-budget", type=int)
    benchmark_parser.add_argument("--warmup", type=int, default=5)
    benchmark_parser.add_argument("--repetitions", type=int, default=20)
    benchmark_parser.add_argument("--top-k", type=int, default=10)
    benchmark_parser.add_argument("--output", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--index", type=Path, required=True)
    evaluate_parser.add_argument("--query-bank", type=Path, required=True)
    evaluate_parser.add_argument("--replay-directory", type=Path, required=True)
    evaluate_parser.add_argument("--plan", type=Path, required=True)
    evaluate_parser.add_argument("--plan-key")
    evaluate_parser.add_argument("--engine", choices=("numpy", "torch"), default="torch")
    evaluate_parser.add_argument("--device", default="cuda:0")
    evaluate_parser.add_argument("--document-batch-size", type=int, default=64)
    evaluate_parser.add_argument("--token-batch-budget", type=int)
    evaluate_parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    evaluate_parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    evaluate_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "compile":
        manifest = compile_heterogeneous_index(
            bank=args.bank,
            plan=load_plan(args.plan, dotted_key=args.plan_key),
            output=args.output,
            storage_dtype=args.storage_dtype,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if args.command == "amplify":
        manifest = amplify_compiled_index(
            source=args.index,
            factor=args.factor,
            output=args.output,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    query_ids, query_embeddings = load_query_bank(args.query_bank)
    runtime = _runtime(
        args.engine,
        args.index,
        device=args.device,
        document_batch_size=args.document_batch_size,
        token_batch_budget=args.token_batch_budget,
    )
    if args.command == "evaluate":
        result = evaluate_runtime(
            runtime,
            query_ids=query_ids,
            query_embeddings=query_embeddings,
            replay_directory=args.replay_directory,
            source_plan=load_plan(args.plan, dotted_key=args.plan_key),
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        result.update(
            {
                "engine": args.engine,
                "device": args.device if args.engine == "torch" else "cpu",
                "document_batch_size": (
                    args.document_batch_size if args.engine == "torch" else None
                ),
                "token_batch_budget": (
                    args.token_batch_budget if args.engine == "torch" else None
                ),
            }
        )
        _json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    result = benchmark_runtime(
        runtime,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        warmup=args.warmup,
        repetitions=args.repetitions,
        top_k=args.top_k,
    )
    result.update(
        {
            "engine": args.engine,
            "device": args.device if args.engine == "torch" else "cpu",
            "document_batch_size": (
                args.document_batch_size if args.engine == "torch" else None
            ),
            "token_batch_budget": (
                args.token_batch_budget if args.engine == "torch" else None
            ),
        }
    )
    _json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
