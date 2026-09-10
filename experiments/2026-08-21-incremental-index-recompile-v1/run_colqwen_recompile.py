#!/usr/bin/env python3
"""Measure dependency-valid ColQwen index recompilation from a reusable vision IR."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any

from run_smoke import load_compatible_adapter, load_projection, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-shard-prefix", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse a dependency-valid exact IR instead of materializing a new one.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--split-layer", type=int, default=12)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.cache_dir.exists() and not args.reuse_cache:
        raise FileExistsError(args.cache_dir)
    if args.reuse_cache and not args.cache_dir.is_dir():
        raise FileNotFoundError("--reuse-cache requires an existing cache directory")

    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    indices = [int(page["page_index"]) for page in manifest["pages"]]
    page_count = int(protocol["dataset"]["page_count"])
    batch_size = int(protocol["dataset"]["batch_size"])
    repeats = int(protocol["measurement"]["repeats"])
    expected = protocol["dataset"]["processor_signature"]
    if len(indices) != page_count or page_count % batch_size:
        raise RuntimeError("frozen page count must be divisible by batch size")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    base = ColQwen2_5.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    projection = load_projection(args.base_shard_prefix, torch)
    with torch.no_grad():
        base.custom_text_proj.weight.copy_(
            projection["custom_text_proj.weight"].to(base.custom_text_proj.weight)
        )
        base.custom_text_proj.bias.copy_(
            projection["custom_text_proj.bias"].to(base.custom_text_proj.bias)
        )
    model = load_compatible_adapter(base, args.adapter).to(args.device)
    adapter_load_diagnostics = dict(model.adapter_load_diagnostics)
    base = model.get_base_model()
    processor = ColQwen2_5_Processor.from_pretrained(
        args.base_model,
        local_files_only=True,
        max_num_visual_tokens=args.max_visual_tokens,
    )
    image_token_id = int(base.config.image_token_id)
    rows = pq.read_table(args.slice).to_pylist()
    image_bytes = [bytes(rows[index]["image_binary"]) for index in indices]
    raw_image_bytes = sum(len(value) for value in image_bytes)

    def prepare_image(payload: bytes) -> dict[str, Any]:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        batch = processor.process_images([image]).to(args.device)
        input_ids = batch["input_ids"]
        attention = batch["attention_mask"]
        grid = batch["image_grid_thw"]
        if list(map(int, grid[0].tolist())) != expected["grid_thw"]:
            raise RuntimeError("frozen processor grid changed")
        offsets = grid[:, 1] * grid[:, 2]
        pixels = torch.cat(
            [row[:offset] for row, offset in zip(batch["pixel_values"], offsets)], dim=0
        )
        embeds = base.get_input_embeddings()(input_ids)
        vision = torch.cat(base.get_image_features(pixels, grid), dim=0).to(
            embeds.device, embeds.dtype
        )
        embeds = embeds.masked_scatter(
            (input_ids == image_token_id).unsqueeze(-1).expand_as(embeds), vision
        )
        position_ids, _ = base.get_rope_index(
            input_ids, grid, None, attention_mask=attention
        )
        visual = torch.where((input_ids[0] == image_token_id) & attention[0].bool())[0]
        nonvisual = torch.where((input_ids[0] != image_token_id) & attention[0].bool())[0]
        if len(visual) != expected["visual_tokens"] or input_ids.shape[1] != expected["sequence_tokens"]:
            raise RuntimeError("frozen processor token signature changed")
        merge = int(base.spatial_merge_size)
        height, width = int(grid[0, 1]) // merge, int(grid[0, 2]) // merge
        checker = torch.tensor(
            [r * width + c for r in range(height) for c in range(width) if (r + c) % 2 == 1],
            dtype=torch.long,
            device=args.device,
        )
        anchors = visual.index_select(0, checker)
        compact_positions = torch.sort(torch.cat((anchors, nonvisual))).values
        return {
            "attention": attention,
            "embeds": embeds,
            "position_ids": position_ids,
            "cache_position": torch.arange(embeds.shape[1], device=args.device),
            "visual": visual,
            "checker": checker,
            "compact_positions": compact_positions,
            "anchor_compact_indices": torch.searchsorted(compact_positions, anchors),
        }

    def to_cpu(item: dict[str, Any]) -> dict[str, Any]:
        return {key: value.detach().cpu() for key, value in item.items()}

    def to_device(item: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(args.device) for key, value in item.items()}

    def pack(items: list[dict[str, Any]]) -> dict[str, Any]:
        first = items[0]
        for item in items[1:]:
            for key in ("cache_position", "visual", "checker", "compact_positions", "anchor_compact_indices"):
                if not torch.equal(first[key], item[key]):
                    raise RuntimeError(f"heterogeneous topology in frozen minibatch: {key}")
        return {
            "attention": torch.cat([item["attention"] for item in items], dim=0),
            "embeds": torch.cat([item["embeds"] for item in items], dim=0),
            "position_ids": torch.cat([item["position_ids"] for item in items], dim=1),
            "cache_position": first["cache_position"],
            "visual": first["visual"],
            "checker": first["checker"],
            "compact_positions": first["compact_positions"],
            "anchor_compact_indices": first["anchor_compact_indices"],
        }

    def language(data: dict[str, Any], compact: bool) -> list[Any]:
        handles = []
        if compact:
            state: dict[str, Any] = {}

            def hook(
                _module: Any,
                positional: tuple[Any, ...],
                keywords: dict[str, Any],
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                hidden = positional[0]
                if "hidden" not in state:
                    visual = hidden.index_select(1, data["visual"])
                    anchors = visual.index_select(1, data["checker"])
                    token_count, anchor_count = visual.shape[1], anchors.shape[1]
                    is_anchor = torch.zeros(token_count, dtype=torch.bool, device=visual.device)
                    is_anchor[data["checker"]] = True
                    nonanchors = torch.where(~is_anchor)[0]
                    assignment = torch.empty(
                        (len(visual), token_count), dtype=torch.long, device=visual.device
                    )
                    assignment[:, data["checker"]] = torch.arange(
                        anchor_count, device=visual.device
                    )
                    assignment[:, nonanchors] = torch.bmm(
                        F.normalize(visual.index_select(1, nonanchors).float(), dim=-1),
                        F.normalize(anchors.float(), dim=-1).transpose(1, 2),
                    ).argmax(-1)
                    sizes = torch.zeros(
                        (len(visual), anchor_count), dtype=torch.float32, device=visual.device
                    )
                    sizes.scatter_add_(
                        1, assignment, torch.ones_like(assignment, dtype=torch.float32)
                    )
                    sums = torch.zeros(
                        (len(visual), anchor_count, visual.shape[-1]),
                        dtype=torch.float32,
                        device=visual.device,
                    )
                    for batch_index in range(len(visual)):
                        sums[batch_index].index_add_(
                            0, assignment[batch_index], visual[batch_index].float()
                        )
                    pooled = (sums / sizes.clamp_min(1).unsqueeze(-1)).to(hidden.dtype)
                    compact_hidden = hidden.index_select(1, data["compact_positions"]).clone()
                    compact_hidden[:, data["anchor_compact_indices"]] = pooled
                    state["hidden"] = compact_hidden
                    layer_mask = keywords["attention_mask"]
                    state["attention_mask"] = (
                        None
                        if layer_mask is None
                        else layer_mask.index_select(-2, data["compact_positions"]).index_select(
                            -1, data["compact_positions"]
                        )
                    )
                    state["position_ids"] = keywords["position_ids"].index_select(
                        -1, data["compact_positions"]
                    )
                    state["cache_position"] = keywords["cache_position"].index_select(
                        0, data["compact_positions"]
                    )
                    state["position_embeddings"] = tuple(
                        value.index_select(-2, data["compact_positions"])
                        for value in keywords["position_embeddings"]
                    )
                updated = dict(keywords)
                for key in ("attention_mask", "position_ids", "cache_position", "position_embeddings"):
                    updated[key] = state[key]
                value = state["hidden"] if hidden.shape[1] != len(data["compact_positions"]) else hidden
                return (value, *positional[1:]), updated

            handles.extend(
                layer.register_forward_pre_hook(hook, with_kwargs=True)
                for layer in base.language_model.layers[args.split_layer :]
            )
        try:
            output = base.language_model(
                input_ids=None,
                inputs_embeds=data["embeds"],
                attention_mask=data["attention"],
                position_ids=data["position_ids"],
                cache_position=data["cache_position"],
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            projected = F.normalize(base.custom_text_proj(output.last_hidden_state).float(), dim=-1)
            if compact:
                return [row.detach().cpu() for row in projected]
            valid = data["attention"].sum(dim=1).tolist()
            return [row[: int(count)].detach().cpu() for row, count in zip(projected, valid)]
        finally:
            for handle in handles:
                handle.remove()

    cache_records: list[dict[str, Any]] = []
    if args.reuse_cache:
        shards = sorted(args.cache_dir.glob("vision-ir-*.pt"))
        if len(shards) != page_count // batch_size:
            raise RuntimeError("existing reusable IR is incomplete")
        for minibatch, shard in enumerate(shards):
            cache_records.append(
                {
                    "minibatch": minibatch,
                    "path": shard.name,
                    "bytes": shard.stat().st_size,
                    "prepare_ms": 0.0,
                    "write_ms": 0.0,
                }
            )
        cache_creation_ms = None
    else:
        args.cache_dir.mkdir(parents=True)
        cache_write_started = time.perf_counter()
        with torch.inference_mode():
            for minibatch, offset in enumerate(range(0, page_count, batch_size)):
                torch.cuda.synchronize()
                prepare_started = time.perf_counter()
                items = [
                    prepare_image(value)
                    for value in image_bytes[offset : offset + batch_size]
                ]
                torch.cuda.synchronize()
                prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
                shard = args.cache_dir / f"vision-ir-{minibatch:03d}.pt"
                save_started = time.perf_counter()
                torch.save([to_cpu(item) for item in items], shard)
                save_ms = (time.perf_counter() - save_started) * 1000.0
                cache_records.append(
                    {
                        "minibatch": minibatch,
                        "path": shard.name,
                        "bytes": shard.stat().st_size,
                        "prepare_ms": prepare_ms,
                        "write_ms": save_ms,
                    }
                )
                del items
        cache_creation_ms = (time.perf_counter() - cache_write_started) * 1000.0

    def run_method(source: str, compact: bool) -> tuple[dict[str, float], list[Any]]:
        source_ms = 0.0
        language_ms = 0.0
        outputs: list[Any] = []
        torch.cuda.synchronize()
        total_started = time.perf_counter()
        with torch.inference_mode():
            for minibatch, offset in enumerate(range(0, page_count, batch_size)):
                torch.cuda.synchronize()
                source_started = time.perf_counter()
                if source == "raw":
                    items = [
                        prepare_image(value)
                        for value in image_bytes[offset : offset + batch_size]
                    ]
                elif source == "ir":
                    shard = args.cache_dir / f"vision-ir-{minibatch:03d}.pt"
                    items = [to_device(item) for item in torch.load(shard, weights_only=True)]
                else:
                    raise ValueError(source)
                data = pack(items)
                torch.cuda.synchronize()
                source_ms += (time.perf_counter() - source_started) * 1000.0
                language_started = time.perf_counter()
                outputs.extend(language(data, compact))
                torch.cuda.synchronize()
                language_ms += (time.perf_counter() - language_started) * 1000.0
                del data, items
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return {
            "source_ms": source_ms,
            "language_and_output_ms": language_ms,
            "total_ms": total_ms,
        }, outputs

    # Warm both execution graphs and the reusable-IR load path before measurement.
    warm_items = [
        to_device(item)
        for item in torch.load(args.cache_dir / "vision-ir-000.pt", weights_only=True)
    ]
    warm_data = pack(warm_items)
    with torch.inference_mode():
        language(warm_data, False)
        language(warm_data, True)
    del warm_data, warm_items

    methods = {
        "raw_full": ("raw", False),
        "ir_full": ("ir", False),
        "raw_l12": ("raw", True),
        "ir_l12": ("ir", True),
    }
    records: list[dict[str, Any]] = []
    first_outputs: dict[str, list[Any]] = {}
    names = list(methods)
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            torch.cuda.reset_peak_memory_stats()
            timing, outputs = run_method(*methods[name])
            records.append(
                {
                    "repeat": repeat,
                    "method": name,
                    **timing,
                    "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                }
            )
            if repeat == 0:
                first_outputs[name] = outputs
            else:
                del outputs

    def equivalence(left: str, right: str) -> dict[str, Any]:
        pairs = list(zip(first_outputs[left], first_outputs[right]))
        return {
            "page_count": len(pairs),
            "tensor_equal": all(torch.equal(a, b) for a, b in pairs),
            "max_abs_error": max(float((a - b).abs().max()) for a, b in pairs),
        }

    def logical_bytes(outputs: list[Any]) -> int:
        return sum(value.numel() * value.element_size() for value in outputs)

    summary: dict[str, Any] = {}
    for name in names:
        matching = [row for row in records if row["method"] == name]
        summary[name] = {
            "source_ms": summarize([row["source_ms"] for row in matching]),
            "language_and_output_ms": summarize(
                [row["language_and_output_ms"] for row in matching]
            ),
            "total_ms": summarize([row["total_ms"] for row in matching]),
            "peak_allocated_mb": summarize([row["peak_allocated_mb"] for row in matching]),
            "logical_index_bytes_float32": logical_bytes(first_outputs[name]),
        }
    comparisons: dict[str, Any] = {}
    for suffix in ("full", "l12"):
        raw_name, ir_name = f"raw_{suffix}", f"ir_{suffix}"
        raw_values = [row["total_ms"] for row in records if row["method"] == raw_name]
        ir_values = [row["total_ms"] for row in records if row["method"] == ir_name]
        saved = [raw - ir for raw, ir in zip(raw_values, ir_values)]
        mean_saved = statistics.mean(saved)
        write_ms = sum(row["write_ms"] for row in cache_records)
        comparisons[suffix] = {
            "mean_saving_fraction": 1.0 - statistics.mean(ir_values) / statistics.mean(raw_values),
            "median_saving_fraction": 1.0 - statistics.median(ir_values) / statistics.median(raw_values),
            "paired_wins": sum(value > 0 for value in saved),
            "paired_saved_ms": saved,
            "cache_write_break_even_updates": (
                None
                if args.reuse_cache
                else write_ms / mean_saved
                if mean_saved > 0
                else None
            ),
            "equivalence": equivalence(raw_name, ir_name),
        }

    ir_bytes = sum(row["bytes"] for row in cache_records)
    full_bytes = summary["raw_full"]["logical_index_bytes_float32"]
    l12_bytes = summary["raw_l12"]["logical_index_bytes_float32"]
    result = {
        "protocol": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "manifest_sha256": sha256(args.manifest),
        "slice_sha256": sha256(args.slice),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__,
        "adapter_load": adapter_load_diagnostics,
        "dependency_scope": {
            "decoder_lora_tensors": adapter_load_diagnostics["remapped_decoder_lora_tensors"],
            "custom_projection_tensors": 2,
            "vision_adapter_tensors": 0,
        },
        "page_count": page_count,
        "batch_size": batch_size,
        "split_layer": args.split_layer,
        "cache_creation_ms": cache_creation_ms,
        "reused_existing_cache": args.reuse_cache,
        "cache_records": cache_records,
        "storage": {
            "raw_compressed_image_bytes": raw_image_bytes,
            "reusable_ir_serialized_bytes": ir_bytes,
            "full_index_logical_float32_bytes": full_bytes,
            "l12_index_logical_float32_bytes": l12_bytes,
            "ir_to_raw_image_ratio": ir_bytes / raw_image_bytes,
            "ir_to_full_index_ratio": ir_bytes / full_bytes,
            "ir_to_l12_index_ratio": ir_bytes / l12_bytes,
            "per_page": {
                "raw_compressed_image_bytes": raw_image_bytes / page_count,
                "reusable_ir_serialized_bytes": ir_bytes / page_count,
                "full_index_logical_float32_bytes": full_bytes / page_count,
                "l12_index_logical_float32_bytes": l12_bytes / page_count,
            },
        },
        "records": records,
        "summary": summary,
        "comparisons": comparisons,
        "gates": protocol["gates"],
        "measurement_interpretation": protocol["measurement"]["interpretation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": summary, "comparisons": comparisons, "storage": result["storage"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
