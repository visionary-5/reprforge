#!/usr/bin/env python3
"""Bounded ColQwen2.5 lifecycle-compilation compatibility smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import struct
import time
from pathlib import Path
from typing import Any


PROJECTION_KEYS = ("custom_text_proj.weight", "custom_text_proj.bias")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-shard-prefix", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--split-layers", type=int, nargs="+", default=[8, 12, 18, 24, 30])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--query", default="What quantitative or visual evidence is shown on this page?")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_projection(path: Path, torch: Any) -> dict[str, Any]:
    payload = path.read_bytes()
    header_length = struct.unpack("<Q", payload[:8])[0]
    data_start = 8 + header_length
    header = json.loads(payload[8:data_start])
    dtype_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    tensors = {}
    for key in PROJECTION_KEYS:
        spec = header[key]
        start, end = spec["data_offsets"]
        raw = bytearray(payload[data_start + start : data_start + end])
        tensors[key] = torch.frombuffer(raw, dtype=dtype_map[spec["dtype"]]).reshape(spec["shape"])
    return tensors


def load_compatible_adapter(base: Any, adapter: Path) -> Any:
    """Load pre-Transformers-4.53 ColQwen LoRA keys into the renamed model tree.

    Qwen2.5-VL moved decoder layers from ``model.layers`` to
    ``language_model.layers``. PEFT 0.15 otherwise leaves all 504 decoder LoRA
    tensors missing while only warning, which silently turns the baseline into
    an unadapted Qwen encoder.
    """

    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    config = PeftConfig.from_pretrained(adapter, local_files_only=True)
    model = get_peft_model(base, config)
    weights = load_peft_weights(adapter, device="cpu", local_files_only=True)
    old_prefix = "base_model.model.model.layers."
    new_prefix = "base_model.model.language_model.layers."
    remapped = {
        (new_prefix + key[len(old_prefix) :] if key.startswith(old_prefix) else key): value
        for key, value in weights.items()
    }
    remapped_count = sum(key.startswith(old_prefix) for key in weights)
    if remapped_count != 504:
        raise RuntimeError(
            f"expected to remap 504 decoder LoRA tensors, observed {remapped_count}"
        )
    load_result = set_peft_model_state_dict(model, remapped, adapter_name="default")
    adapter_missing = [key for key in load_result.missing_keys if "lora_" in key]
    adapter_unexpected = [key for key in load_result.unexpected_keys if "lora_" in key]
    if adapter_missing or adapter_unexpected:
        raise RuntimeError(
            "adapter compatibility load incomplete: "
            f"missing={len(adapter_missing)}, unexpected={len(adapter_unexpected)}"
        )
    model.eval()
    model.adapter_load_diagnostics = {
        "checkpoint_tensors": len(weights),
        "remapped_decoder_lora_tensors": remapped_count,
        "missing_adapter_tensors": len(adapter_missing),
        "unexpected_adapter_tensors": len(adapter_unexpected),
    }
    return model


def timed(torch: Any, fn: Any, warmup: int, repetitions: int) -> tuple[Any, list[float]]:
    latest = None
    for _ in range(warmup):
        latest = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        latest = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return latest, samples


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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
        base.custom_text_proj.weight.copy_(projection["custom_text_proj.weight"].to(base.custom_text_proj.weight))
        base.custom_text_proj.bias.copy_(projection["custom_text_proj.bias"].to(base.custom_text_proj.bias))
    model = load_compatible_adapter(base, args.adapter).to(args.device)
    base = model.get_base_model()
    processor = ColQwen2_5_Processor.from_pretrained(
        args.base_model,
        local_files_only=True,
        max_num_visual_tokens=args.max_visual_tokens,
    )
    image = Image.open(args.image).convert("RGB")
    image_batch = processor.process_images([image]).to(args.device)
    query_batch = processor.process_queries([args.query]).to(args.device)
    image_token_id = int(base.config.image_token_id)

    with torch.inference_mode():
        official_full = model(**image_batch)[0]
        query = model(**query_batch)[0].float()

    def prepare() -> tuple[Any, ...]:
        input_ids = image_batch["input_ids"]
        attention = image_batch["attention_mask"]
        grid = image_batch["image_grid_thw"]
        pixels = image_batch["pixel_values"]
        offsets = grid[:, 1] * grid[:, 2]
        pixels = torch.cat([row[:offset] for row, offset in zip(pixels, offsets, strict=True)], dim=0)
        embeds = base.get_input_embeddings()(input_ids)
        vision = torch.cat(base.get_image_features(pixels, grid), dim=0).to(embeds.device, embeds.dtype)
        mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(mask, vision)
        position_ids, _ = base.get_rope_index(input_ids, grid, None, attention_mask=attention)
        cache_position = torch.arange(embeds.shape[1], device=embeds.device)
        return input_ids, attention, grid, embeds, position_ids, cache_position

    prepared = prepare()
    input_ids, attention, grid, inputs_embeds, position_ids, cache_position = prepared
    valid_positions = torch.where(attention[0].bool())[0]
    visual_positions = torch.where((input_ids[0] == image_token_id) & attention[0].bool())[0]
    nonvisual_positions = torch.where((input_ids[0] != image_token_id) & attention[0].bool())[0]
    merge = int(base.spatial_merge_size)
    grid_h = int(grid[0, 1].item()) // merge
    grid_w = int(grid[0, 2].item()) // merge
    if grid_h * grid_w != len(visual_positions):
        raise RuntimeError(f"grid {grid_h}x{grid_w} != {len(visual_positions)} visual tokens")
    checker = torch.tensor(
        [row * grid_w + col for row in range(grid_h) for col in range(grid_w) if (row + col) % 2 == 1],
        device=args.device,
        dtype=torch.long,
    )
    anchor_positions = visual_positions.index_select(0, checker)
    compact_positions = torch.sort(torch.cat((anchor_positions, nonvisual_positions))).values
    anchor_compact_indices = torch.searchsorted(compact_positions, anchor_positions)

    def project(hidden: Any) -> Any:
        return F.normalize(base.custom_text_proj(hidden).float(), dim=-1)

    def language_full() -> Any:
        output = base.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return project(output.last_hidden_state)[0].index_select(0, valid_positions)

    def compact(split_layer: int) -> tuple[Any, dict[str, Any]]:
        layers = base.language_model.layers
        state: dict[str, Any] = {}

        def hook(_module: Any, positional: tuple[Any, ...], keywords: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
            hidden = positional[0]
            if "hidden" not in state:
                visual = hidden.index_select(1, visual_positions)
                anchors = visual.index_select(1, checker)
                assignment = torch.bmm(
                    F.normalize(visual.float(), dim=-1),
                    F.normalize(anchors.float(), dim=-1).transpose(1, 2),
                ).argmax(-1)
                one_hot = F.one_hot(assignment, num_classes=anchors.shape[1]).float()
                sizes = one_hot.sum(1)
                pooled = torch.bmm(one_hot.transpose(1, 2), visual.float()) / sizes.clamp_min(1).unsqueeze(-1)
                pooled = torch.where((sizes > 0).unsqueeze(-1), pooled, anchors.float()).to(hidden.dtype)
                compact_hidden = hidden.index_select(1, compact_positions).clone()
                compact_hidden[:, anchor_compact_indices] = pooled
                state["hidden"] = compact_hidden
                state["assignment"] = assignment.detach()
                state["sizes"] = sizes.detach()
                layer_mask = keywords["attention_mask"]
                state["attention_mask"] = (
                    None
                    if layer_mask is None
                    else layer_mask.index_select(-2, compact_positions).index_select(-1, compact_positions)
                )
                state["position_ids"] = keywords["position_ids"].index_select(-1, compact_positions)
                state["cache_position"] = keywords["cache_position"].index_select(0, compact_positions)
                state["position_embeddings"] = tuple(value.index_select(-2, compact_positions) for value in keywords["position_embeddings"])
            keywords = dict(keywords)
            for key in ("attention_mask", "position_ids", "cache_position", "position_embeddings"):
                keywords[key] = state[key]
            return (state["hidden"] if hidden.shape[1] != len(compact_positions) else hidden, *positional[1:]), keywords

        handles = [layer.register_forward_pre_hook(hook, with_kwargs=True) for layer in layers[split_layer:]]
        try:
            output = base.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        return project(output.last_hidden_state)[0], state

    with torch.inference_mode():
        manual_full, full_ms = timed(torch, language_full, args.warmup, args.repetitions)
        official_valid = official_full.index_select(0, valid_positions).float()
        full_error = float((manual_full - official_valid).abs().max().item())
        full_score = float(processor.score_multi_vector([query], [official_valid])[0, 0].item())
        rows = []
        for split in args.split_layers:
            if not 0 < split < len(base.language_model.layers):
                raise ValueError(f"invalid split layer {split}")

            def run() -> tuple[Any, dict[str, Any]]:
                return compact(split)

            (value, state), samples = timed(torch, run, args.warmup, args.repetitions)
            assignment = state["assignment"][0]
            full_visual = official_full.index_select(0, visual_positions).float()
            one_hot = F.one_hot(assignment, num_classes=len(checker)).float()
            sizes = one_hot.sum(0)
            teacher = one_hot.transpose(0, 1) @ full_visual / sizes.clamp_min(1).unsqueeze(-1)
            teacher = torch.where((sizes > 0).unsqueeze(-1), teacher, full_visual.index_select(0, checker))
            teacher = F.normalize(teacher.float(), dim=-1)
            compact_visual = value.index_select(0, anchor_compact_indices)
            endpoint_cosine = float((compact_visual * teacher).sum(-1).mean().item())
            score = float(processor.score_multi_vector([query], [value])[0, 0].item())
            rows.append({
                "split_layer": split,
                "normalized_depth": split / len(base.language_model.layers),
                "visual_tokens": len(checker),
                "persistent_visual_fraction": len(checker) / len(visual_positions),
                "empty_clusters": int((sizes == 0).sum().item()),
                "mean_cluster_size": float(sizes.mean().item()),
                "max_cluster_size": int(sizes.max().item()),
                "finite": bool(torch.isfinite(value).all().item()),
                "endpoint_cosine_to_assignment_matched_full": endpoint_cosine,
                "query_score": score,
                "absolute_query_score_error": abs(score - full_score),
                "latency_ms": samples,
                "median_latency_ms": statistics.median(samples),
                "saving_vs_manual_full_median": 1.0 - statistics.median(samples) / statistics.median(full_ms),
            })

    result = {
        "protocol": "colqwen-lifecycle-transfer-v1-2026-08-20",
        "model": {
            "base": str(args.base_model),
            "adapter": str(args.adapter),
            "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
            "layers": len(base.language_model.layers),
        },
        "input": {
            "image": str(args.image),
            "image_sha256": sha256(args.image),
            "query": args.query,
            "image_grid_thw": grid.detach().cpu().tolist(),
            "post_merge_grid": [grid_h, grid_w],
            "full_visual_tokens": len(visual_positions),
            "full_nonvisual_tokens": len(nonvisual_positions),
        },
        "validation": {
            "manual_vs_official_full_max_abs_error": full_error,
            "full_finite": bool(torch.isfinite(manual_full).all().item()),
        },
        "full": {
            "query_score": full_score,
            "latency_ms": full_ms,
            "median_latency_ms": statistics.median(full_ms),
        },
        "compact": rows,
        "versions": {"torch": torch.__version__},
        "claim_boundary": "One-page compatibility and mechanism smoke only; no retrieval-quality generalization claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
