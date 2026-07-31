#!/usr/bin/env python3
"""Execute both MMDocIR layout representation routes with one ColPali model.

This is the constructive boundary between ReprForge's planner and the real
retriever.  It reads only one document's inclusive Parquet row range, encodes
every layout through both the native-text and rendered-image paths, measures
the emitted index bytes and construction time, and exports a frozen score cube
for policy iteration.  An explicit flag additionally persists a route
embedding bank so a selected plan can be compiled into a real heterogeneous
index.  Persistence stays opt-in because a complete all-route bank can be much
larger than the final selected index.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from reprforge.mmdocir_data import normalize_document_queries, normalize_layouts
from reprforge.policy_replay import (
    IMAGE_ROUTE,
    TEXT_ROUTE,
    ReplayData,
    evaluate_plan,
    fixed_hybrid_plan,
    load_replay_data,
    uniform_plan,
)


@dataclass(frozen=True)
class EncodedBatch:
    embeddings: tuple[Any, ...]
    encode_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.embeddings) != len(self.encode_ms):
            raise ValueError("embedding and timing counts differ")
        if any(value < 0 for value in self.encode_ms):
            raise ValueError("encoding time must be non-negative")


class RouteBackend(Protocol):
    """Minimal model contract used by the model-independent orchestration."""

    def encode_queries(self, queries: Sequence[str]) -> EncodedBatch: ...

    def encode_texts(self, texts: Sequence[str]) -> EncodedBatch: ...

    def encode_images(self, images: Sequence[bytes]) -> EncodedBatch: ...

    def construction_features(
        self,
        images: Sequence[bytes],
    ) -> Sequence[Mapping[str, float]]: ...

    def derive_image_routes(
        self,
        images: EncodedBatch,
    ) -> Mapping[str, EncodedBatch]: ...

    def representation_features(
        self,
    ) -> Mapping[str, Sequence[Mapping[str, float]]]: ...

    def score(
        self,
        queries: Sequence[Any],
        documents: Sequence[Any],
    ) -> Sequence[Sequence[float]]: ...

    def environment(self) -> Mapping[str, Any]: ...


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _embedding_bytes(embedding: Any) -> int:
    nbytes = getattr(embedding, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    numel = getattr(embedding, "numel", None)
    element_size = getattr(embedding, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel() * element_size())
    raise TypeError(f"cannot determine serialized bytes for {type(embedding)!r}")


def read_parquet_row_range(
    path: Path,
    *,
    start: int,
    end: int,
    columns: Sequence[str],
    scan_batch_rows: int = 1024,
) -> list[dict]:
    """Read an inclusive global row range without loading the binary corpus.

    The public MMDocIR layouts file currently has one 170k-row Parquet row
    group.  Calling ``read_row_group`` would therefore decompress every image
    just to run a one-document smoke.  ``iter_batches`` preserves source order
    and lets an early bounded document stop after the first small batch.
    """

    if start < 0 or end < start:
        raise ValueError(f"invalid inclusive row range [{start}, {end}]")
    if scan_batch_rows <= 0:
        raise ValueError("scan_batch_rows must be positive")
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    missing = set(columns) - available
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    batch_start = 0
    for batch in parquet.iter_batches(
        batch_size=scan_batch_rows,
        columns=list(columns),
        use_threads=True,
    ):
        batch_end = batch_start + batch.num_rows - 1
        if batch_end < start:
            batch_start += batch.num_rows
            continue
        if batch_start > end:
            break
        local_start = max(start, batch_start) - batch_start
        local_end = min(end, batch_end) - batch_start
        rows.extend(
            batch.slice(local_start, local_end - local_start + 1).to_pylist()
        )
        batch_start += batch.num_rows
        if batch_end >= end:
            break

    expected = end - start + 1
    if len(rows) != expected:
        raise ValueError(
            f"expected {expected} rows from [{start}, {end}], read {len(rows)}"
        )
    return rows


def load_annotations(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ranking(
    query_ids: Sequence[str],
    item_ids: Sequence[str],
    scores: Mapping[str, Sequence[Sequence[float]]],
    plan: Mapping[str, str],
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    item_offsets = {item_id: index for index, item_id in enumerate(item_ids)}
    for query_offset, query_id in enumerate(query_ids):
        rankings[query_id] = sorted(
            item_ids,
            key=lambda item_id: (
                -float(
                    scores[plan[item_id]][query_offset][item_offsets[item_id]]
                ),
                item_id,
            ),
        )
    return rankings


def _ranking_digest(rankings: Mapping[str, Sequence[str]]) -> str:
    payload = json.dumps(rankings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def execute_document(
    *,
    backend: RouteBackend,
    annotations: Sequence[Mapping[str, Any]],
    document_index: int,
    layout_rows: Sequence[Mapping[str, Any]],
    output: Path,
    dataset_revision: str,
    model_revision: str,
    persist_embeddings: bool = False,
    embedding_storage_dtype: str = "float32",
) -> dict:
    """Encode and score one already-selected MMDocIR document."""

    if document_index < 0 or document_index >= len(annotations):
        raise IndexError(f"document index {document_index} is outside annotations")
    document = annotations[document_index]
    layout_start, layout_end = (int(value) for value in document["layout_indices"])
    if len(layout_rows) != layout_end - layout_start + 1:
        raise ValueError("layout rows do not match the selected document range")

    layouts = normalize_layouts(layout_rows, source_start=layout_start)
    layouts_by_source_row = {
        int(layout["source_row"]): layout for layout in layouts
    }
    query_start = sum(
        len(annotation["questions"]) for annotation in annotations[:document_index]
    )
    _, queries = normalize_document_queries(
        document,
        layouts_by_source_row,
        document_index=document_index,
        query_start=query_start,
    )
    if not queries:
        raise ValueError("selected document has no evaluable layout queries")

    query_batch = backend.encode_queries([str(row["query"]) for row in queries])
    text_batch = backend.encode_texts(
        [str(row.get("native_text") or "") for row in layouts]
    )
    image_payloads = [row.get("image_binary") for row in layout_rows]
    if any(not isinstance(payload, (bytes, bytearray)) for payload in image_payloads):
        raise ValueError("every selected layout must contain image_binary bytes")
    feature_extractor = getattr(backend, "construction_features", None)
    if feature_extractor is None:
        construction_features: Sequence[Mapping[str, float]] = [
            {} for _ in image_payloads
        ]
    else:
        construction_features = feature_extractor(
            [bytes(payload) for payload in image_payloads]
        )
    if len(construction_features) != len(image_payloads):
        raise ValueError("construction feature count does not match layouts")
    image_batch = backend.encode_images(
        [bytes(payload) for payload in image_payloads]
    )
    derived_image_routes: Mapping[str, EncodedBatch] = {}
    derive_routes = getattr(backend, "derive_image_routes", None)
    if derive_routes is not None:
        derived_image_routes = dict(derive_routes(image_batch))
    route_batches = {
        TEXT_ROUTE: text_batch,
        IMAGE_ROUTE: image_batch,
        **derived_image_routes,
    }
    if len(route_batches) != 2 + len(derived_image_routes):
        raise ValueError("a derived image route collides with a base route")
    route_diagnostics: Mapping[
        str,
        Sequence[Mapping[str, float]],
    ] = {}
    diagnostic_extractor = getattr(backend, "representation_features", None)
    if diagnostic_extractor is not None:
        route_diagnostics = diagnostic_extractor()
        unknown_diagnostic_routes = set(route_diagnostics) - set(route_batches)
        if unknown_diagnostic_routes:
            raise ValueError(
                "representation diagnostics contain unknown routes: "
                f"{sorted(unknown_diagnostic_routes)}"
            )

    expected_items = len(layouts)
    for route, batch in route_batches.items():
        if len(batch.embeddings) != expected_items:
            raise ValueError(
                f"{route} emitted {len(batch.embeddings)} embeddings for "
                f"{expected_items} layouts"
            )

    score_matrices = {
        route: backend.score(query_batch.embeddings, batch.embeddings)
        for route, batch in route_batches.items()
    }
    for route, matrix in score_matrices.items():
        if len(matrix) != len(queries) or any(
            len(row) != expected_items for row in matrix
        ):
            raise ValueError(f"{route} score matrix has the wrong shape")

    item_rows = []
    for index, layout in enumerate(layouts):
        item_rows.append(
            {
                **layout,
                "construction_features": dict(construction_features[index]),
                "route_costs": {
                    route: {
                        "index_bytes": _embedding_bytes(batch.embeddings[index]),
                        "encode_ms": batch.encode_ms[index],
                    }
                    for route, batch in route_batches.items()
                },
                "route_features": {
                    route: {
                        "vector_count": int(
                            batch.embeddings[index].shape[0]
                        ),
                        "embedding_dimension": int(
                            batch.embeddings[index].shape[1]
                        ),
                        **(
                            dict(route_diagnostics[route][index])
                            if route in route_diagnostics
                            else {}
                        ),
                    }
                    for route, batch in route_batches.items()
                },
            }
        )
    # Image bytes are deliberately absent from normalized metadata and output.
    query_rows = [dict(row) for row in queries]
    score_rows = []
    for route, matrix in score_matrices.items():
        for query_index, query in enumerate(queries):
            for item_index, item in enumerate(layouts):
                score_rows.append(
                    {
                        "route": route,
                        "query_id": query["query_id"],
                        "item_id": item["item_id"],
                        "score": float(matrix[query_index][item_index]),
                    }
                )

    output.mkdir(parents=True, exist_ok=True)
    _jsonl(output / "items.jsonl", item_rows)
    _jsonl(output / "queries.jsonl", query_rows)
    _jsonl(output / "scores.jsonl", score_rows)
    embedding_bank_manifest = None
    if persist_embeddings:
        from reprforge.heterogeneous_index import write_embedding_bank

        embedding_bank_manifest = write_embedding_bank(
            output / "embedding-bank",
            item_ids=[str(item["item_id"]) for item in layouts],
            route_embeddings={
                route: batch.embeddings
                for route, batch in route_batches.items()
            },
            query_ids=[str(query["query_id"]) for query in queries],
            query_embeddings=query_batch.embeddings,
            storage_dtype=embedding_storage_dtype,
        )

    data: ReplayData = load_replay_data(
        output / "items.jsonl",
        output / "queries.jsonl",
        output / "scores.jsonl",
    )
    plans = {
        "all-text": uniform_plan(data.items, TEXT_ROUTE),
        "all-image": uniform_plan(data.items, IMAGE_ROUTE),
        "fixed-hybrid": fixed_hybrid_plan(data.items),
    }
    for route in sorted(derived_image_routes):
        plans[f"all-{route}"] = uniform_plan(data.items, route)
        plans[f"fixed-hybrid-{route}"] = fixed_hybrid_plan(
            data.items,
            image_route=route,
        )
    metrics = {
        name: evaluate_plan(data, plan, ks=(1, 5, 10))
        for name, plan in plans.items()
    }
    (output / "baselines.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    query_ids = [str(query["query_id"]) for query in queries]
    item_ids = [str(item["item_id"]) for item in layouts]
    direct_rankings = _ranking(
        query_ids,
        item_ids,
        score_matrices,
        plans["fixed-hybrid"],
    )
    replay_scores = {
        route: [
            [data.scores[route][query_id][item_id] for item_id in item_ids]
            for query_id in query_ids
        ]
        for route in (TEXT_ROUTE, IMAGE_ROUTE)
    }
    replay_rankings = _ranking(
        query_ids,
        item_ids,
        replay_scores,
        plans["fixed-hybrid"],
    )
    if direct_rankings != replay_rankings:
        raise AssertionError("fixed-hybrid replay changed a direct ranking")

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **dict(backend.environment()),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": "MMDocIR/MMDocIR_Evaluation_Dataset",
        "dataset_revision": dataset_revision,
        "model": "MMDocIR/MMDocIR_Retrievers/colpali-v1.1",
        "model_revision": model_revision,
        "document_index": document_index,
        "document_name": str(document.get("doc_name") or ""),
        "domain": str(document.get("domain") or ""),
        "layout_range_inclusive": [layout_start, layout_end],
        "layouts": len(layouts),
        "queries": len(queries),
        "routes": list(data.routes),
        "fixed_hybrid_direct_replay_equal": True,
        "fixed_hybrid_ranking_sha256": _ranking_digest(direct_rankings),
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


class ColPaliBackend:
    """Local-only ColPali backend matching MMDocIR's published prompts."""

    def __init__(
        self,
        *,
        base_model: Path,
        adapter: Path,
        device: str,
        batch_size: int,
        image_pool_factors: Sequence[int] = (),
        scoring_batch_size: int | None = None,
    ) -> None:
        import torch
        from colpali_engine.models import ColPaliProcessor

        from reprforge.mmdocir_colpali import MMDocIRColPali

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.scoring_batch_size = (
            batch_size if scoring_batch_size is None else scoring_batch_size
        )
        if self.scoring_batch_size <= 0:
            raise ValueError("scoring_batch_size must be positive")
        self.image_pool_factors = tuple(sorted(set(image_pool_factors)))
        if any(factor < 2 for factor in self.image_pool_factors):
            raise ValueError("image pool factors must be at least 2")
        self.base_model = base_model
        self.adapter = adapter
        self.checkpoint_key_mapping = {
            r"^model\.language_model\.model\.": "model.language_model."
        }
        self.model, loading_info = MMDocIRColPali.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=None,
            local_files_only=True,
            output_loading_info=True,
            key_mapping=self.checkpoint_key_mapping,
        )
        base_load_problems = {
            key: list(loading_info.get(key) or [])
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
            if loading_info.get(key)
        }
        if base_load_problems:
            raise RuntimeError(
                f"MMDocIR base checkpoint did not load exactly: {base_load_problems}"
            )
        self.model.load_adapter(
            str(adapter),
            adapter_kwargs={"key_mapping": self.checkpoint_key_mapping},
        )
        self.adapter_parameter_count = self._verify_adapter_weights(adapter)
        self.model = self.model.to(self.device).eval()
        self.processor = ColPaliProcessor.from_pretrained(
            adapter,
            local_files_only=True,
        )
        from PIL import Image

        self.mock_image = Image.new("RGB", (16, 16), color="black")

    def _verify_adapter_weights(self, adapter: Path) -> int:
        """Prove that every published adapter tensor reached the live model."""

        from safetensors import safe_open

        adapter_path = adapter / "adapter_model.safetensors"
        state = self.model.state_dict()
        verified = 0
        missing: list[str] = []
        mismatched: list[str] = []
        with safe_open(adapter_path, framework="pt", device="cpu") as handle:
            for source_key in handle.keys():
                target_key = source_key.removeprefix("base_model.model.")
                target_key = target_key.replace(
                    "model.language_model.model.",
                    "model.language_model.",
                )
                target_key = target_key.replace(
                    ".lora_A.weight",
                    ".lora_A.default.weight",
                ).replace(
                    ".lora_B.weight",
                    ".lora_B.default.weight",
                )
                if target_key not in state:
                    missing.append(target_key)
                    continue
                source = handle.get_tensor(source_key)
                target = state[target_key].detach().cpu()
                if not self.torch.equal(source.to(target.dtype), target):
                    mismatched.append(target_key)
                    continue
                verified += 1
        if missing or mismatched:
            raise RuntimeError(
                "MMDocIR adapter verification failed: "
                f"missing={missing[:5]}, mismatched={mismatched[:5]}"
            )
        if verified == 0:
            raise RuntimeError("MMDocIR adapter contains no verified tensors")
        return verified

    def _text_batch(
        self,
        texts: Sequence[str],
        prefix: str,
        *,
        max_length: int,
    ) -> Any:
        """Match MMDocIR's published ColPali text-only processing path."""

        padded = [f"{prefix}{text}" + "<pad>" * 10 for text in texts]
        image_sequence_length = getattr(self.processor, "image_seq_length", 32)
        batch = self.processor(
            images=[self.mock_image] * len(padded),
            text=padded,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + image_sequence_length,
        )
        batch.pop("pixel_values", None)
        batch["input_ids"] = batch["input_ids"][..., image_sequence_length:]
        batch["attention_mask"] = batch["attention_mask"][
            ..., image_sequence_length:
        ]
        return batch

    def _encode(
        self,
        payloads: Sequence[Any],
        make_batch: Callable[[Sequence[Any]], Any],
    ) -> EncodedBatch:
        embeddings: list[Any] = []
        timings: list[float] = []
        for start in range(0, len(payloads), self.batch_size):
            payload_batch = payloads[start : start + self.batch_size]
            self.torch.cuda.synchronize(self.device)
            began = time.perf_counter()
            batch = make_batch(payload_batch)
            batch = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            with self.torch.inference_mode():
                output = self.model(**batch)
            self.torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - began) * 1000.0
            attention = batch["attention_mask"]
            for embedding, mask in zip(output, attention, strict=True):
                embeddings.append(
                    embedding[mask.bool()].detach().cpu().to(self.torch.float32)
                )
            timings.extend([elapsed_ms / len(payload_batch)] * len(payload_batch))
        return EncodedBatch(tuple(embeddings), tuple(timings))

    def encode_queries(self, queries: Sequence[str]) -> EncodedBatch:
        return self._encode(
            queries,
            lambda values: self._text_batch(
                values,
                "Question: ",
                max_length=512,
            ),
        )

    def encode_texts(self, texts: Sequence[str]) -> EncodedBatch:
        truncated = [" ".join(text.split()[:400]) for text in texts]
        return self._encode(
            truncated,
            lambda values: self._text_batch(
                values,
                "Passage: ",
                max_length=600,
            ),
        )

    def encode_images(self, images: Sequence[Any]) -> EncodedBatch:
        from PIL import Image

        def make_batch(values: Sequence[Any]) -> Any:
            pictures = []
            for value in values:
                if isinstance(value, bytes):
                    picture = Image.open(io.BytesIO(value)).convert("RGB")
                elif hasattr(value, "convert"):
                    picture = value.convert("RGB")
                else:
                    raise TypeError(
                        "image inputs must be encoded bytes or PIL-like objects"
                    )
                pictures.append(picture)
            return self.processor(
                text=["Describe the image."] * len(pictures),
                images=pictures,
                return_tensors="pt",
                padding="longest",
            )

        return self._encode(images, make_batch)

    def construction_features(
        self,
        images: Sequence[bytes],
    ) -> Sequence[Mapping[str, float]]:
        """Extract bounded raw-image statistics before representation choice.

        A 64x64 grayscale thumbnail bounds CPU work independently of the
        source crop size. These features describe visual information density;
        they do not inspect queries, relevance labels, or model scores.
        """

        import numpy as np
        from PIL import Image

        features: list[dict[str, float]] = []
        for payload in images:
            began = time.perf_counter()
            picture = Image.open(io.BytesIO(payload)).convert("L")
            width, height = picture.size
            thumbnail = picture.resize((64, 64))
            values = np.asarray(thumbnail, dtype=np.float32) / 255.0
            histogram, _ = np.histogram(values, bins=32, range=(0.0, 1.0))
            probabilities = histogram.astype(np.float64)
            probabilities /= max(probabilities.sum(), 1.0)
            nonzero = probabilities[probabilities > 0]
            entropy = float(
                -(nonzero * np.log2(nonzero)).sum() / math.log2(32)
            )
            horizontal = (
                float(np.abs(np.diff(values, axis=1)).mean())
                if values.shape[1] > 1
                else 0.0
            )
            vertical = (
                float(np.abs(np.diff(values, axis=0)).mean())
                if values.shape[0] > 1
                else 0.0
            )
            features.append(
                {
                    "image_width": float(width),
                    "image_height": float(height),
                    "image_aspect_log": math.log(
                        (float(width) + 1.0) / (float(height) + 1.0)
                    ),
                    "grayscale_entropy": entropy,
                    "grayscale_std": float(values.std()),
                    "edge_energy": (horizontal + vertical) / 2.0,
                    "nonwhite_fraction": float((values < 0.95).mean()),
                    "feature_extract_ms": (
                        time.perf_counter() - began
                    ) * 1000.0,
                }
            )
        return features

    def derive_image_routes(
        self,
        images: EncodedBatch,
    ) -> Mapping[str, EncodedBatch]:
        """Apply ColPali's published semantic hierarchical token pooling.

        A compressed route's construction time includes the original visual
        encoding plus its own pooling transform. This avoids reporting the
        transform as free merely because several routes share one research run.
        """

        self._last_representation_features = {
            IMAGE_ROUTE: tuple(
                {
                    "compression_factor": 1.0,
                    "cosine_cover_loss_mean": 0.0,
                    "cosine_cover_loss_p95": 0.0,
                    "cosine_cover_loss_max": 0.0,
                    "diagnostic_requires_candidate_embedding": 0.0,
                }
                for _ in images.embeddings
            )
        }
        if not self.image_pool_factors:
            return {}
        from colpali_engine.compression.token_pooling import (
            HierarchicalTokenPooler,
        )

        pooler = HierarchicalTokenPooler()
        routes: dict[str, EncodedBatch] = {}
        for factor in self.image_pool_factors:
            pooled: list[Any] = []
            timings: list[float] = []
            diagnostics: list[Mapping[str, float]] = []
            for embedding, base_ms in zip(
                images.embeddings,
                images.encode_ms,
                strict=True,
            ):
                began = time.perf_counter()
                result = pooler.pool_embeddings(
                    [embedding],
                    pool_factor=factor,
                    return_dict=False,
                    num_workers=1,
                )
                pooling_ms = (time.perf_counter() - began) * 1000.0
                pooled.append(result[0])
                timings.append(base_ms + pooling_ms)
                diagnostics.append(
                    self._compression_distortion_features(
                        embedding,
                        result[0],
                    )
                )
            routes[f"image-pool-{factor}"] = EncodedBatch(
                embeddings=tuple(pooled),
                encode_ms=tuple(timings),
            )
            self._last_representation_features[
                f"image-pool-{factor}"
            ] = tuple(diagnostics)
        return routes

    def _compression_distortion_features(
        self,
        full_embedding: Any,
        compressed_embedding: Any,
    ) -> Mapping[str, float]:
        """Measure query-independent MaxSim evidence cover distortion.

        For each full document token, find its nearest compressed token in the
        normalized retrieval space. These diagnostics directly measure what
        pooling removes, unlike raw pixel entropy. They currently require the
        candidate compressed embedding and are therefore diagnostic features,
        not yet a zero-cost deployed signal.
        """

        began = time.perf_counter()
        full = self.torch.nn.functional.normalize(
            full_embedding.to(self.device, dtype=self.torch.float32),
            dim=-1,
        )
        compressed = self.torch.nn.functional.normalize(
            compressed_embedding.to(self.device, dtype=self.torch.float32),
            dim=-1,
        )
        cover_similarity = (full @ compressed.T).max(dim=-1).values
        cover_loss = (1.0 - cover_similarity).clamp_min(0.0)
        values = {
            "compression_factor": float(
                full_embedding.shape[0] / compressed_embedding.shape[0]
            ),
            "cosine_cover_loss_mean": float(cover_loss.mean().item()),
            "cosine_cover_loss_p95": float(
                self.torch.quantile(cover_loss, 0.95).item()
            ),
            "cosine_cover_loss_max": float(cover_loss.max().item()),
            "diagnostic_requires_candidate_embedding": 1.0,
        }
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        values["distortion_feature_ms"] = (
            time.perf_counter() - began
        ) * 1000.0
        return values

    def representation_features(
        self,
    ) -> Mapping[str, Sequence[Mapping[str, float]]]:
        return getattr(self, "_last_representation_features", {})

    def score(
        self,
        queries: Sequence[Any],
        documents: Sequence[Any],
    ) -> Sequence[Sequence[float]]:
        if not queries:
            return []
        if not documents:
            return [[] for _ in queries]

        rows: list[list[float]] = []
        score_batch = self.scoring_batch_size
        for query_start in range(0, len(queries), score_batch):
            query_batch = queries[query_start : query_start + score_batch]
            query_lengths = self.torch.tensor(
                [query.shape[0] for query in query_batch],
                device=self.device,
            )
            padded_queries = self.torch.nn.utils.rnn.pad_sequence(
                [
                    query.to(self.device, dtype=self.torch.float32)
                    for query in query_batch
                ],
                batch_first=True,
            )
            query_positions = self.torch.arange(
                padded_queries.shape[1],
                device=self.device,
            )
            query_mask = (
                query_positions[None, :] < query_lengths[:, None]
            )
            query_rows: list[Any] = []
            for document_start in range(0, len(documents), score_batch):
                document_batch = documents[
                    document_start : document_start + score_batch
                ]
                document_lengths = self.torch.tensor(
                    [document.shape[0] for document in document_batch],
                    device=self.device,
                )
                padded_documents = self.torch.nn.utils.rnn.pad_sequence(
                    [
                        document.to(self.device, dtype=self.torch.float32)
                        for document in document_batch
                    ],
                    batch_first=True,
                )
                similarities = self.torch.einsum(
                    "aqd,bkd->abqk",
                    padded_queries,
                    padded_documents,
                )
                document_positions = self.torch.arange(
                    padded_documents.shape[1],
                    device=self.device,
                )
                similarities = similarities.masked_fill(
                    document_positions[None, None, None, :]
                    >= document_lengths[None, :, None, None],
                    float("-inf"),
                )
                maxsim = similarities.max(dim=-1).values
                scores = (
                    maxsim * query_mask[:, None, :]
                ).sum(dim=-1)
                query_rows.append(scores)
            matrix = self.torch.cat(query_rows, dim=1).cpu()
            rows.extend(
                [float(value) for value in row]
                for row in matrix.tolist()
            )
        return rows

    def environment(self) -> Mapping[str, Any]:
        return {
            "backend": "colpali",
            "torch": self.torch.__version__,
            "torch_cuda": self.torch.version.cuda,
            "device": str(self.device),
            "gpu": self.torch.cuda.get_device_name(self.device),
            "base_model_path": str(self.base_model.resolve()),
            "adapter_path": str(self.adapter.resolve()),
            "adapter_lora_parameter_tensors": self.adapter_parameter_count,
            "batch_size": self.batch_size,
            "scoring_batch_size": self.scoring_batch_size,
            "image_pool_factors": list(self.image_pool_factors),
            "image_pooling": (
                "colpali_engine.HierarchicalTokenPooler"
                if self.image_pool_factors
                else None
            ),
            "embedding_storage_dtype": "float32",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts-parquet", type=Path, required=True)
    parser.add_argument(
        "--parquet-source-start",
        type=int,
        default=0,
        help="absolute source row represented by row 0 of a bounded Parquet slice",
    )
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--document-index", type=int, required=True)
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
        help=(
            "derive a semantic hierarchical token-pooling route from every "
            "image embedding; may be repeated"
        ),
    )
    parser.add_argument(
        "--persist-embeddings",
        action="store_true",
        help=(
            "write all route and query embeddings into an inspectable route "
            "bank for later heterogeneous-index compilation"
        ),
    )
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
    if args.document_index < 0 or args.document_index >= len(annotations):
        parser.error("--document-index is outside the annotation file")
    layout_start, layout_end = (
        int(value) for value in annotations[args.document_index]["layout_indices"]
    )
    local_layout_start = layout_start - args.parquet_source_start
    local_layout_end = layout_end - args.parquet_source_start
    if local_layout_start < 0:
        parser.error(
            "--parquet-source-start places the requested document before row 0 "
            "of the supplied Parquet file"
        )
    rows = read_parquet_row_range(
        args.layouts_parquet,
        start=local_layout_start,
        end=local_layout_end,
        columns=("type", "bbox", "page_id", "text", "image_binary"),
    )
    backend = ColPaliBackend(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        batch_size=args.batch_size,
        image_pool_factors=args.image_pool_factor,
    )
    manifest = execute_document(
        backend=backend,
        annotations=annotations,
        document_index=args.document_index,
        layout_rows=rows,
        output=args.output,
        dataset_revision=args.dataset_revision,
        model_revision=args.model_revision,
        persist_embeddings=args.persist_embeddings,
        embedding_storage_dtype=args.embedding_storage_dtype,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
