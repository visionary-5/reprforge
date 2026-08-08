#!/usr/bin/env python3
"""Measure reusable Qwen2.5-VL page features against full raw-page forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence


def _summary(values: Sequence[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run(
    dataset_root: Path,
    pair_scores: Path,
    model_path: Path,
    *,
    sample_pairs: int,
    warmup_pairs: int,
    disk_cache_dir: Path | None = None,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    query_rows = _read_jsonl(dataset_root / "queries.jsonl")
    queries = {str(row["query_id"]): str(row["query"]) for row in query_rows}
    source_pairs = _read_jsonl(pair_scores)
    pairs = list(
        dict.fromkeys(
            (str(row["query_id"]), str(row["doc_id"])) for row in source_pairs
        )
    )[: sample_pairs + warmup_pairs]
    if len(pairs) <= warmup_pairs:
        raise ValueError("not enough pairs after warmup")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.image_processor.min_pixels = 200704
    processor.image_processor.max_pixels = 1003520
    yes_id = processor.tokenizer.encode("YES", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("NO", add_special_tokens=False)[0]
    load_started = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    if disk_cache_dir is not None:
        if disk_cache_dir.exists() and any(disk_cache_dir.iterdir()):
            raise FileExistsError(
                f"refusing to reuse non-empty disk cache: {disk_cache_dir}"
            )
        disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def disk_path(doc_id: str) -> Path:
        assert disk_cache_dir is not None
        name = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
        return disk_cache_dir / f"{name}.pt"

    cached_host: dict[str, torch.Tensor] = {}
    cache_build_ms: dict[str, float] = {}
    cache_build_end_to_end_ms: dict[str, float] = {}
    disk_write_ms: dict[str, float] = {}
    serialized_feature_bytes: dict[str, int] = {}
    rows = []
    for pair_index, (query_id, doc_id) in enumerate(pairs):
        prepare_started = time.perf_counter()
        image = Image.open(dataset_root / "assets" / f"{doc_id}.png").convert("RGB")
        message = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"Question: {queries[query_id]}\nDoes this document page "
                        "contain evidence that directly helps answer the question? "
                        "Answer with exactly one word: YES or NO."
                    ),
                },
            ],
        }]
        prompt = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt], images=[image], padding=True, return_tensors="pt"
        ).to("cuda")
        torch.cuda.synchronize()
        prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

        full_started = time.perf_counter()
        with torch.inference_mode():
            full_logits = model(**inputs, logits_to_keep=1).logits
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - full_started) * 1000.0
        full_score = float(
            (full_logits[0, -1, yes_id] - full_logits[0, -1, no_id]).float()
        )

        if doc_id not in cached_host:
            build_started = time.perf_counter()
            with torch.inference_mode():
                features = torch.cat(
                    model.model.get_image_features(
                        inputs.pixel_values, inputs.image_grid_thw
                    ),
                    dim=0,
                )
            torch.cuda.synchronize()
            cached_host[doc_id] = features.detach().cpu()
            torch.cuda.synchronize()
            cache_build_ms[doc_id] = (time.perf_counter() - build_started) * 1000.0
            cache_build_end_to_end_ms[doc_id] = prepare_ms + cache_build_ms[doc_id]
            if disk_cache_dir is not None:
                write_started = time.perf_counter()
                path = disk_path(doc_id)
                with path.open("wb") as handle:
                    torch.save(cached_host[doc_id], handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                disk_write_ms[doc_id] = (
                    time.perf_counter() - write_started
                ) * 1000.0
                serialized_feature_bytes[doc_id] = path.stat().st_size
            del features

        cached_started = time.perf_counter()
        image_features = cached_host[doc_id].to("cuda")
        with torch.inference_mode():
            embeddings = model.model.get_input_embeddings()(inputs.input_ids)
            image_mask, _ = model.model.get_placeholder_mask(
                inputs.input_ids,
                inputs_embeds=embeddings,
                image_features=image_features,
            )
            embeddings = embeddings.masked_scatter(image_mask, image_features)
            cached_logits = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                inputs_embeds=embeddings,
                image_grid_thw=inputs.image_grid_thw,
                logits_to_keep=1,
            ).logits
        torch.cuda.synchronize()
        cached_ms = (time.perf_counter() - cached_started) * 1000.0
        cached_score = float(
            (cached_logits[0, -1, yes_id] - cached_logits[0, -1, no_id]).float()
        )
        disk_read_ms = None
        disk_cached_ms = None
        disk_cached_score_difference = None
        if disk_cache_dir is not None:
            path = disk_path(doc_id)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(descriptor)
            disk_cached_started = time.perf_counter()
            disk_read_started = time.perf_counter()
            disk_host_features = torch.load(
                path, map_location="cpu", weights_only=True
            )
            disk_read_ms = (time.perf_counter() - disk_read_started) * 1000.0
            disk_image_features = disk_host_features.to("cuda")
            with torch.inference_mode():
                disk_embeddings = model.model.get_input_embeddings()(inputs.input_ids)
                disk_image_mask, _ = model.model.get_placeholder_mask(
                    inputs.input_ids,
                    inputs_embeds=disk_embeddings,
                    image_features=disk_image_features,
                )
                disk_embeddings = disk_embeddings.masked_scatter(
                    disk_image_mask, disk_image_features
                )
                disk_cached_logits = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    inputs_embeds=disk_embeddings,
                    image_grid_thw=inputs.image_grid_thw,
                    logits_to_keep=1,
                ).logits
            torch.cuda.synchronize()
            disk_cached_ms = (
                time.perf_counter() - disk_cached_started
            ) * 1000.0
            disk_cached_score = float(
                (
                    disk_cached_logits[0, -1, yes_id]
                    - disk_cached_logits[0, -1, no_id]
                ).float()
            )
            disk_cached_score_difference = abs(full_score - disk_cached_score)
        row = {
            "pair_index": pair_index,
            "query_id": query_id,
            "doc_id": doc_id,
            "prepare_image_query_and_h2d_ms": prepare_ms,
            "full_optimized_forward_ms": full_ms,
            "full_end_to_end_ms": prepare_ms + full_ms,
            "cached_feature_h2d_and_language_ms": cached_ms,
            "disk_feature_read_ms": disk_read_ms,
            "disk_feature_read_h2d_and_language_ms": disk_cached_ms,
            "score_absolute_difference": abs(full_score - cached_score),
            "disk_score_absolute_difference": disk_cached_score_difference,
            "cached_feature_bytes": int(
                cached_host[doc_id].numel() * cached_host[doc_id].element_size()
            ),
            "cached_feature_tokens": int(cached_host[doc_id].shape[0]),
        }
        if pair_index >= warmup_pairs:
            rows.append(row)
        del inputs, full_logits, cached_logits, embeddings, image_features
        if disk_cache_dir is not None:
            del (
                disk_host_features,
                disk_image_features,
                disk_embeddings,
                disk_cached_logits,
            )
    result = {
        "schema_version": 1,
        "protocol": "qwen-visual-feature-cache-v0",
        "model_load_seconds": model_load_seconds,
        "sample_pairs": len(rows),
        "warmup_pairs": warmup_pairs,
        "unique_cached_pages": len(cached_host),
        "cache_build_ms": _summary(list(cache_build_ms.values())),
        "cache_build_end_to_end_ms": _summary(
            list(cache_build_end_to_end_ms.values())
        ),
        "prepare_image_query_and_h2d_ms": _summary(
            [row["prepare_image_query_and_h2d_ms"] for row in rows]
        ),
        "full_optimized_forward_ms": _summary(
            [row["full_optimized_forward_ms"] for row in rows]
        ),
        "full_end_to_end_ms": _summary(
            [row["full_end_to_end_ms"] for row in rows]
        ),
        "cached_feature_h2d_and_language_ms": _summary(
            [row["cached_feature_h2d_and_language_ms"] for row in rows]
        ),
        "mean_cached_feature_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in cached_host.values()
        )
        / len(cached_host),
        "maximum_score_absolute_difference": max(
            row["score_absolute_difference"] for row in rows
        ),
        "rows": rows,
        "scope": (
            "full path and cached path both use logits_to_keep=1; cached path "
            "starts from host-resident query-independent vision-tower output"
        ),
    }
    if disk_cache_dir is not None:
        result["disk_cache"] = {
            "directory": str(disk_cache_dir),
            "page_cache_evict_hint": (
                "POSIX_FADV_DONTNEED before every measured read"
                if hasattr(os, "posix_fadvise")
                and hasattr(os, "POSIX_FADV_DONTNEED")
                else "unavailable"
            ),
            "write_and_fsync_ms": _summary(list(disk_write_ms.values())),
            "read_ms": _summary(
                [float(row["disk_feature_read_ms"]) for row in rows]
            ),
            "read_h2d_and_language_ms": _summary(
                [
                    float(row["disk_feature_read_h2d_and_language_ms"])
                    for row in rows
                ]
            ),
            "mean_serialized_feature_bytes": (
                sum(serialized_feature_bytes.values())
                / len(serialized_feature_bytes)
            ),
            "maximum_score_absolute_difference": max(
                float(row["disk_score_absolute_difference"]) for row in rows
            ),
        }
        result["scope"] += (
            "; disk path includes torch serialization, fsync at materialization, "
            "a best-effort POSIX_FADV_DONTNEED hint, read, H2D, and language layers"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pair-scores", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-pairs", type=int, default=16)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--disk-cache-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    result = run(
        args.dataset_root,
        args.pair_scores,
        args.model,
        sample_pairs=args.sample_pairs,
        warmup_pairs=args.warmup_pairs,
        disk_cache_dir=args.disk_cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
