#!/usr/bin/env python3
"""Measure real OmniColPress page-representation construction on one GPU."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
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
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def run(
    upstream_root: Path,
    build_config_path: Path,
    corpus_path: Path,
    assets_path: Path,
    *,
    sample_pages: int,
    batch_sizes: Sequence[int],
    warmup_batches: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(upstream_root))
    from src.arguments import DataArguments, ModelArguments
    from src.factory.factory import create_inference_components

    config = json.loads(build_config_path.read_text())
    model_args = ModelArguments(**config["model_args"])
    data_args = DataArguments(**config["data_args"])
    data_args = dataclasses.replace(
        data_args,
        corpus_path=str(corpus_path),
        assets_path=str(assets_path),
        dataset_number_of_shards=1,
        dataset_shard_index=0,
        num_proc=1,
        encode_is_query=False,
    )
    model_started = time.perf_counter()
    (
        model,
        _,
        collator,
        dataset,
        _,
        device_for_inputs,
        torch_dtype,
    ) = create_inference_components(
        device=device,
        model_args=model_args,
        data_args=data_args,
        batch_size=1,
        distributed=False,
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_started
    count = min(int(sample_pages), len(dataset))
    if count <= 0:
        raise ValueError("sample_pages must select at least one page")
    positions = sorted(random.Random(seed).sample(range(len(dataset)), count))
    selected = Subset(dataset, positions)
    results: dict[str, Any] = {}
    use_autocast = torch_dtype in (torch.float16, torch.bfloat16)
    for batch_size in map(int, batch_sizes):
        if batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        loader = DataLoader(
            selected,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=0,
            shuffle=False,
        )
        iterator = iter(loader)
        rows = []
        batch_index = 0
        while True:
            prepare_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
            ids = batch.get("ids") or batch.get("passage_ids")
            inputs = batch.get("inputs") or batch.get("passage_inputs")
            if ids is None or inputs is None:
                raise ValueError("upstream collator returned an unexpected batch")
            h2d_started = time.perf_counter()
            for key in inputs:
                inputs[key] = inputs[key].to(device_for_inputs)
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter() - h2d_started) * 1000.0
            torch.cuda.reset_peak_memory_stats()
            forward_started = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast(
                device_type=torch.device(device).type,
                dtype=torch_dtype,
                enabled=use_autocast,
            ):
                embeddings, masks = model.encode_passage(inputs)
            torch.cuda.synchronize()
            forward_ms = (time.perf_counter() - forward_started) * 1000.0
            output_started = time.perf_counter()
            embeddings_cpu = embeddings.detach().cpu()
            masks_cpu = masks.detach().cpu()
            torch.cuda.synchronize()
            output_d2h_ms = (time.perf_counter() - output_started) * 1000.0
            valid_tokens = int(masks_cpu.bool().sum().item())
            row = {
                "batch_index": batch_index,
                "batch_size": len(ids),
                "doc_ids": list(map(str, ids)),
                "prepare_image_processor_ms": prepare_ms,
                "input_h2d_ms": h2d_ms,
                "model_forward_ms": forward_ms,
                "output_d2h_ms": output_d2h_ms,
                "end_to_end_ms": prepare_ms + h2d_ms + forward_ms + output_d2h_ms,
                "valid_output_tokens": valid_tokens,
                "output_vector_bytes": int(
                    valid_tokens * embeddings_cpu.shape[-1] * embeddings_cpu.element_size()
                ),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
            if batch_index >= warmup_batches:
                rows.append(row)
            del embeddings, embeddings_cpu, masks, masks_cpu, inputs
            batch_index += 1
        if not rows:
            raise ValueError("warmup_batches consumed the complete sample")
        pages = sum(int(row["batch_size"]) for row in rows)
        results[str(batch_size)] = {
            "measured_batches": len(rows),
            "measured_pages": pages,
            "prepare_image_processor_ms_per_page": _summary(
                [row["prepare_image_processor_ms"] / row["batch_size"] for row in rows]
            ),
            "input_h2d_ms_per_page": _summary(
                [row["input_h2d_ms"] / row["batch_size"] for row in rows]
            ),
            "model_forward_ms_per_page": _summary(
                [row["model_forward_ms"] / row["batch_size"] for row in rows]
            ),
            "output_d2h_ms_per_page": _summary(
                [row["output_d2h_ms"] / row["batch_size"] for row in rows]
            ),
            "end_to_end_ms_per_page": _summary(
                [row["end_to_end_ms"] / row["batch_size"] for row in rows]
            ),
            "batch_end_to_end_ms": _summary([row["end_to_end_ms"] for row in rows]),
            "valid_output_tokens_per_page": sum(
                int(row["valid_output_tokens"]) for row in rows
            )
            / pages,
            "output_vector_bytes_per_page": sum(
                int(row["output_vector_bytes"]) for row in rows
            )
            / pages,
            "peak_cuda_memory_bytes": max(
                int(row["peak_cuda_memory_bytes"]) for row in rows
            ),
            "rows": rows,
        }
    return {
        "schema_version": 1,
        "protocol": "omni-real-page-construction-v0",
        "upstream_root": str(upstream_root),
        "build_config": str(build_config_path),
        "corpus": str(corpus_path),
        "assets": str(assets_path),
        "device": device,
        "model_load_seconds": model_load_seconds,
        "sample_pages": count,
        "sample_positions": positions,
        "batch_sizes": list(map(int, batch_sizes)),
        "warmup_batches": warmup_batches,
        "seed": seed,
        "results": results,
        "timing_scope": (
            "asset open/decode + upstream processor + input H2D + encode_passage + output D2H"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--build-config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-pages", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    result = run(
        args.upstream_root,
        args.build_config,
        args.corpus,
        args.assets,
        sample_pages=args.sample_pages,
        batch_sizes=args.batch_sizes,
        warmup_batches=args.warmup_batches,
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
