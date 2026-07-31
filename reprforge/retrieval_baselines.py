#!/usr/bin/env python3
"""Strong multi-stage baselines for the ReprForge retrieval contract."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.heterogeneous_index import (
    TorchMaxSimRuntime,
    _load_shard,
    _numpy_embedding,
    _read_manifest,
)
from reprforge.policy_replay import _ndcg, load_replay_data


class TorchResidentReranker:
    """Exact MaxSim over a query-selected subset of a resident full index.

    Unlike the normal serving runtime, this baseline keeps one globally padded
    tensor so arbitrary candidate positions can be gathered into one reranking
    launch. It is intentionally specialized for the nearly uniform full-image
    route used by ColPali-style indexes.
    """

    def __init__(self, index: Path, *, device: str) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.index = index
        self.manifest = _read_manifest(index, "reprforge-heterogeneous-index")
        records = json.loads((index / "items.json").read_text(encoding="utf-8"))
        self.item_ids = tuple(str(row["item_id"]) for row in records)
        positions = {
            item_id: position for position, item_id in enumerate(self.item_ids)
        }
        documents: list[Any | None] = [None] * len(self.item_ids)
        for route in sorted(self.manifest["routes"]):
            ids, vectors, offsets = _load_shard(index / "routes" / route)
            for local_position, item_id in enumerate(ids):
                documents[positions[item_id]] = torch.from_numpy(
                    np.array(
                        vectors[
                            offsets[local_position] : offsets[local_position + 1]
                        ],
                        dtype=np.float32,
                        copy=True,
                    )
                )
        if any(document is None for document in documents):
            raise ValueError("compiled index does not cover every item")
        typed_documents = [document for document in documents if document is not None]
        self.lengths = torch.tensor(
            [document.shape[0] for document in typed_documents],
            device=self.device,
            dtype=torch.int64,
        )
        self.vectors = torch.nn.utils.rnn.pad_sequence(
            typed_documents,
            batch_first=True,
        ).to(self.device)
        self._positions = {
            item_id: position for position, item_id in enumerate(self.item_ids)
        }
        dimension = int(self.manifest["dimension"])
        self._resident_unpadded_vector_bytes = int(
            self.lengths.sum().item()
            * dimension
            * self.vectors.element_size()
        )

    @property
    def compact_vector_bytes(self) -> int:
        return int(self.manifest["compact_vector_bytes"])

    @property
    def resident_vector_bytes(self) -> int:
        return int(self.vectors.numel() * self.vectors.element_size())

    @property
    def resident_unpadded_vector_bytes(self) -> int:
        return self._resident_unpadded_vector_bytes

    def positions(self, item_ids: Sequence[str]) -> Any:
        unknown = [item_id for item_id in item_ids if item_id not in self._positions]
        if unknown:
            raise ValueError(f"unknown rerank items: {unknown[:5]}")
        return self.torch.tensor(
            [self._positions[item_id] for item_id in item_ids],
            device=self.device,
            dtype=self.torch.int64,
        )

    def score_positions_tensor(
        self,
        query_embedding: Any,
        positions: Any,
    ) -> Any:
        positions = self.torch.as_tensor(
            positions,
            device=self.device,
            dtype=self.torch.int64,
        )
        if positions.ndim != 1 or positions.numel() == 0:
            raise ValueError("reranking positions must be a non-empty vector")
        query = self.torch.as_tensor(
            np.array(
                _numpy_embedding(query_embedding),
                dtype=np.float32,
                copy=True,
            ),
            device=self.device,
            dtype=self.torch.float32,
        )
        selected_vectors = self.vectors.index_select(0, positions)
        selected_lengths = self.lengths.index_select(0, positions)
        similarities = self.torch.einsum(
            "qd,bkd->bqk",
            query,
            selected_vectors,
        )
        token_positions = self.torch.arange(
            selected_vectors.shape[1],
            device=self.device,
        )
        similarities = similarities.masked_fill(
            token_positions[None, None, :]
            >= selected_lengths[:, None, None],
            float("-inf"),
        )
        return similarities.max(dim=-1).values.sum(dim=-1)


class PooledExactRerankRuntime:
    """Pool-based candidate generation followed by exact full MaxSim."""

    def __init__(
        self,
        pooled_index: Path,
        full_index: Path,
        *,
        device: str,
        candidate_k: int,
        document_batch_size: int = 64,
        token_batch_budget: int | None = None,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.first_stage = TorchMaxSimRuntime(
            pooled_index,
            device=device,
            document_batch_size=document_batch_size,
            token_batch_budget=token_batch_budget,
        )
        self.reranker = TorchResidentReranker(full_index, device=device)
        if self.first_stage.item_ids != self.reranker.item_ids:
            raise ValueError("first-stage and rerank indexes have different items")
        self.torch = self.first_stage.torch
        self.device = self.first_stage.device
        self.item_ids = self.first_stage.item_ids
        self.candidate_k = candidate_k

    @property
    def compact_vector_bytes(self) -> int:
        return (
            self.first_stage.compact_vector_bytes
            + self.reranker.compact_vector_bytes
        )

    @property
    def resident_vector_bytes(self) -> int:
        return (
            self.first_stage.resident_vector_bytes
            + self.reranker.resident_vector_bytes
        )

    @property
    def resident_unpadded_vector_bytes(self) -> int:
        return (
            self.first_stage.resident_unpadded_vector_bytes
            + self.reranker.resident_unpadded_vector_bytes
        )

    @property
    def execution_batch_count(self) -> int:
        return self.first_stage.execution_batch_count + 1

    def synchronize(self) -> None:
        self.first_stage.synchronize()

    def _search_positions(
        self,
        query_embedding: Any,
        eligible_positions: Any,
        *,
        top_k: int,
    ) -> list[tuple[str, float]]:
        first_scores = self.first_stage.score_tensor(query_embedding)
        eligible_positions = self.torch.as_tensor(
            eligible_positions,
            device=self.device,
            dtype=self.torch.int64,
        )
        eligible_scores = first_scores.index_select(0, eligible_positions)
        candidate_count = min(self.candidate_k, eligible_positions.numel())
        if candidate_count < top_k:
            raise ValueError(
                f"candidate_k={self.candidate_k} is smaller than top_k={top_k}"
            )
        _, local_candidates = self.torch.topk(
            eligible_scores,
            k=candidate_count,
            sorted=False,
        )
        candidate_positions = eligible_positions.index_select(
            0, local_candidates
        )
        exact_scores = self.reranker.score_positions_tensor(
            query_embedding,
            candidate_positions,
        )
        final_count = min(top_k, candidate_count)
        values, local_winners = self.torch.topk(
            exact_scores,
            k=final_count,
            sorted=True,
        )
        winners = candidate_positions.index_select(0, local_winners)
        host_positions = winners.detach().cpu().tolist()
        host_values = values.detach().cpu().tolist()
        return [
            (self.item_ids[position], float(score))
            for position, score in zip(
                host_positions,
                host_values,
                strict=True,
            )
        ]

    def search(
        self,
        query_embedding: Any,
        *,
        top_k: int,
    ) -> list[tuple[str, float]]:
        eligible = self.torch.arange(
            len(self.item_ids),
            device=self.device,
            dtype=self.torch.int64,
        )
        return self._search_positions(
            query_embedding,
            eligible,
            top_k=top_k,
        )

    def search_candidates(
        self,
        query_embedding: Any,
        *,
        candidate_item_ids: Sequence[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        eligible = self.reranker.positions(candidate_item_ids)
        return self._search_positions(
            query_embedding,
            eligible,
            top_k=top_k,
        )


class PreencodedNoCacheRuntime:
    """Pool-resident index with transient host-to-device full-vector upgrades.

    This is a lower bound for deferred visual ingestion: full vectors already
    exist on host storage, so image decoding and model encoding are excluded.
    Selected full vectors are copied to the GPU, scored, and discarded on
    every query.
    """

    def __init__(
        self,
        pooled_index: Path,
        full_index: Path,
        *,
        device: str,
        document_batch_size: int = 64,
        token_batch_budget: int | None = None,
        pinned_host: bool = False,
    ) -> None:
        self.base = TorchMaxSimRuntime(
            pooled_index,
            device=device,
            document_batch_size=document_batch_size,
            token_batch_budget=token_batch_budget,
        )
        self.torch = self.base.torch
        self.device = self.base.device
        self.item_ids = self.base.item_ids
        self._positions = {
            item_id: position for position, item_id in enumerate(self.item_ids)
        }
        manifest = _read_manifest(full_index, "reprforge-heterogeneous-index")
        records = json.loads(
            (full_index / "items.json").read_text(encoding="utf-8")
        )
        full_ids = tuple(str(row["item_id"]) for row in records)
        if full_ids != self.item_ids:
            raise ValueError("base and full indexes have different item orders")
        self._documents: list[np.ndarray | None] = [None] * len(self.item_ids)
        for route in sorted(manifest["routes"]):
            ids, vectors, offsets = _load_shard(full_index / "routes" / route)
            for local_position, item_id in enumerate(ids):
                self._documents[self._positions[item_id]] = np.asarray(
                    vectors[
                        offsets[local_position] : offsets[local_position + 1]
                    ]
                )
        if any(document is None for document in self._documents):
            raise ValueError("full index does not cover every item")
        self.pinned_host = pinned_host
        self._pinned_vectors = None
        self._pinned_lengths = None
        if pinned_host:
            typed_documents = [
                self.torch.from_numpy(
                    np.array(document, dtype=np.float32, copy=True)
                )
                for document in self._documents
                if document is not None
            ]
            self._pinned_lengths = self.torch.tensor(
                [document.shape[0] for document in typed_documents],
                dtype=self.torch.int64,
            ).pin_memory()
            self._pinned_vectors = self.torch.nn.utils.rnn.pad_sequence(
                typed_documents,
                batch_first=True,
            ).pin_memory()
        self.full_compact_vector_bytes = int(manifest["compact_vector_bytes"])
        self.last_transient_vector_bytes = 0

    @property
    def compact_vector_bytes(self) -> int:
        # Only the pooled index is persistently device resident.
        return self.base.compact_vector_bytes

    @property
    def resident_vector_bytes(self) -> int:
        return self.base.resident_vector_bytes

    @property
    def resident_unpadded_vector_bytes(self) -> int:
        return self.base.resident_unpadded_vector_bytes

    @property
    def execution_batch_count(self) -> int:
        return self.base.execution_batch_count + 1

    def synchronize(self) -> None:
        self.base.synchronize()

    def search_selected(
        self,
        query_embedding: Any,
        *,
        selected_item_ids: Sequence[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        if not selected_item_ids:
            raise ValueError("no-cache query requires selected full items")
        unknown = [
            item_id for item_id in selected_item_ids
            if item_id not in self._positions
        ]
        if unknown:
            raise ValueError(f"unknown no-cache items: {unknown[:5]}")
        selected_positions = [
            self._positions[item_id] for item_id in selected_item_ids
        ]
        if self.pinned_host:
            assert self._pinned_vectors is not None
            assert self._pinned_lengths is not None
            cpu_positions = self.torch.tensor(
                selected_positions,
                dtype=self.torch.int64,
            )
            host_vectors = self.torch.empty(
                (
                    len(selected_positions),
                    self._pinned_vectors.shape[1],
                    self._pinned_vectors.shape[2],
                ),
                dtype=self._pinned_vectors.dtype,
                pin_memory=True,
            )
            self.torch.index_select(
                self._pinned_vectors,
                0,
                cpu_positions,
                out=host_vectors,
            )
            host_lengths = self.torch.empty(
                len(selected_positions),
                dtype=self.torch.int64,
                pin_memory=True,
            )
            self.torch.index_select(
                self._pinned_lengths,
                0,
                cpu_positions,
                out=host_lengths,
            )
            vectors = host_vectors.to(self.device, non_blocking=True)
            lengths = host_lengths.to(self.device, non_blocking=True)
        else:
            host_documents = [
                self._documents[position] for position in selected_positions
            ]
            typed_documents = [
                self.torch.from_numpy(
                    np.array(document, dtype=np.float32, copy=True)
                )
                for document in host_documents
                if document is not None
            ]
            lengths = self.torch.tensor(
                [document.shape[0] for document in typed_documents],
                device=self.device,
                dtype=self.torch.int64,
            )
            vectors = self.torch.nn.utils.rnn.pad_sequence(
                typed_documents,
                batch_first=True,
            ).to(self.device)
        self.last_transient_vector_bytes = int(
            vectors.numel() * vectors.element_size()
        )
        query = self.torch.as_tensor(
            np.array(
                _numpy_embedding(query_embedding),
                dtype=np.float32,
                copy=True,
            ),
            device=self.device,
            dtype=self.torch.float32,
        )
        similarities = self.torch.einsum("qd,bkd->bqk", query, vectors)
        token_positions = self.torch.arange(
            vectors.shape[1],
            device=self.device,
        )
        similarities = similarities.masked_fill(
            token_positions[None, None, :] >= lengths[:, None, None],
            float("-inf"),
        )
        exact_scores = similarities.max(dim=-1).values.sum(dim=-1)
        scores = self.base.score_tensor(query_embedding)
        positions = self.torch.tensor(
            selected_positions,
            device=self.device,
            dtype=self.torch.int64,
        )
        scores = scores.index_copy(0, positions, exact_scores)
        count = min(top_k, len(self.item_ids))
        values, winners = self.torch.topk(scores, k=count, sorted=True)
        return [
            (self.item_ids[position], float(score))
            for position, score in zip(
                winners.detach().cpu().tolist(),
                values.detach().cpu().tolist(),
                strict=True,
            )
        ]


def benchmark_selected_runtime(
    runtime: PreencodedNoCacheRuntime,
    *,
    query_ids: Sequence[str],
    query_embeddings: Sequence[Any],
    selections: Sequence[Sequence[str]],
    warmup: int,
    repetitions: int,
    top_k: int,
) -> dict[str, Any]:
    if not (
        len(query_ids) == len(query_embeddings) == len(selections)
        and query_ids
    ):
        raise ValueError("queries, embeddings, and selections must align")
    if warmup < 0 or repetitions <= 0:
        raise ValueError("invalid warmup/repetition count")
    for _ in range(warmup):
        for embedding, selected in zip(
            query_embeddings, selections, strict=True
        ):
            runtime.search_selected(
                embedding,
                selected_item_ids=selected,
                top_k=top_k,
            )
    runtime.synchronize()
    latencies_ms: list[float] = []
    transient_bytes: list[int] = []
    digest = hashlib.sha256()
    for _ in range(repetitions):
        for query_id, embedding, selected in zip(
            query_ids,
            query_embeddings,
            selections,
            strict=True,
        ):
            runtime.synchronize()
            began = time.perf_counter()
            result = runtime.search_selected(
                embedding,
                selected_item_ids=selected,
                top_k=top_k,
            )
            runtime.synchronize()
            latencies_ms.append((time.perf_counter() - began) * 1000.0)
            transient_bytes.append(runtime.last_transient_vector_bytes)
            digest.update(query_id.encode())
            digest.update(
                json.dumps(result, separators=(",", ":")).encode()
            )
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "measurements": len(latencies_ms),
        "latency_ms": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        },
        "qps": float(1000.0 / values.mean()),
        "persistent_compact_vector_bytes": runtime.compact_vector_bytes,
        "persistent_resident_vector_bytes": runtime.resident_vector_bytes,
        "transient_vector_bytes": {
            "mean": float(np.mean(transient_bytes)),
            "max": int(max(transient_bytes)),
        },
        "full_host_vector_bytes": runtime.full_compact_vector_bytes,
        "host_memory": "pinned" if runtime.pinned_host else "pageable",
        "score_digest_sha256": digest.hexdigest(),
        "cost_scope": (
            "pool-query+host-copy+H2D+selected-full-MaxSim+merge+TopK;"
            "image-encoding-excluded"
        ),
    }


def benchmark_resident_selected_runtime(
    runtime: Any,
    *,
    query_ids: Sequence[str],
    query_embeddings: Sequence[Any],
    selections: Sequence[Sequence[str]],
    warmup: int,
    repetitions: int,
    top_k: int,
) -> dict[str, Any]:
    """Benchmark query-scoped activation over a resident visual delta."""

    if not (
        len(query_ids) == len(query_embeddings) == len(selections)
        and query_ids
    ):
        raise ValueError("queries, embeddings, and selections must align")
    for _ in range(warmup):
        for embedding, selected in zip(
            query_embeddings, selections, strict=True
        ):
            runtime.search_selected(
                embedding,
                selected_item_ids=selected,
                top_k=top_k,
            )
    runtime.synchronize()
    latencies_ms: list[float] = []
    digest = hashlib.sha256()
    for _ in range(repetitions):
        for query_id, embedding, selected in zip(
            query_ids,
            query_embeddings,
            selections,
            strict=True,
        ):
            runtime.synchronize()
            began = time.perf_counter()
            result = runtime.search_selected(
                embedding,
                selected_item_ids=selected,
                top_k=top_k,
            )
            runtime.synchronize()
            latencies_ms.append((time.perf_counter() - began) * 1000.0)
            digest.update(query_id.encode())
            digest.update(
                json.dumps(result, separators=(",", ":")).encode()
            )
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "measurements": len(latencies_ms),
        "latency_ms": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        },
        "qps": float(1000.0 / values.mean()),
        "compact_vector_bytes": runtime.compact_vector_bytes,
        "resident_vector_bytes": runtime.resident_vector_bytes,
        "resident_unpadded_vector_bytes": (
            runtime.resident_unpadded_vector_bytes
        ),
        "execution_batches": runtime.execution_batch_count,
        "score_digest_sha256": digest.hexdigest(),
        "cost_scope": (
            "pool-query+resident-selected-full-MaxSim+merge+TopK;"
            "image-encoding-excluded"
        ),
    }


def evaluate_two_stage_runtime(
    runtime: PooledExactRerankRuntime,
    *,
    query_ids: Sequence[str],
    query_embeddings: Sequence[Any],
    replay_directory: Path,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Evaluate two-stage rankings with MMDocIR's official candidate pools."""

    if len(query_ids) != len(query_embeddings):
        raise ValueError("query identifiers and embeddings differ in length")
    embeddings = dict(zip(query_ids, query_embeddings, strict=True))
    data = load_replay_data(
        replay_directory / "items.jsonl",
        replay_directory / "queries.jsonl",
        replay_directory / "scores.jsonl",
    )
    recalls = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    for query in data.queries:
        candidate_item_ids = (
            query.candidate_item_ids
            if query.candidate_item_ids is not None
            else runtime.item_ids
        )
        requested_k = min(max(ks), len(candidate_item_ids))
        ranked = [
            item_id
            for item_id, _ in runtime.search_candidates(
                embeddings[query.query_id],
                candidate_item_ids=candidate_item_ids,
                top_k=requested_k,
            )
        ]
        denominator = (
            query.relevance_denominator
            if query.relevance_denominator is not None
            else sum(query.relevance.values())
        )
        for k in ks:
            recalls[k] += (
                sum(
                    query.relevance.get(item_id, 0.0)
                    for item_id in ranked[:k]
                )
                / denominator
            )
            ndcgs[k] += _ndcg(ranked, query.relevance, k)
    count = len(data.queries)
    return {
        "queries": count,
        "candidate_k": runtime.candidate_k,
        **{f"recall_at_{k}": recalls[k] / count for k in ks},
        **{f"ndcg_at_{k}": ndcgs[k] / count for k in ks},
    }
