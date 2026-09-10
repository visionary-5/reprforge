#!/usr/bin/env python3
"""Exact ColQwen2 v0.1-to-v1.0 post-vision replay canary."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol",
        "cpu_result",
        "slice",
        "base_model",
        "base_projection",
        "source_adapter",
        "target_adapter",
        "ir_root",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one GPU")
    if args.output.exists() or args.ir_root.exists():
        raise FileExistsError("refusing to overwrite output or source IR")

    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    from PIL import Image
    from safetensors.torch import load_file
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    protocol = json.loads(args.protocol.read_text())
    frozen = protocol["frozen_inputs"]
    cpu_result = json.loads(args.cpu_result.read_text())
    if sha256(args.cpu_result) != protocol["promotion_evidence"]["cpu_result_sha256"]:
        raise ValueError("CPU promotion evidence changed")
    if not cpu_result["gpu_promotion"] or cpu_result["dependency_decision"] != "post_vision_replay_legal":
        raise RuntimeError("CPU dependency gate did not promote GPU execution")
    if sha256(args.slice) != frozen["dataset_sha256"]:
        raise ValueError("dataset changed")
    if sha256(args.source_adapter / "adapter_model.safetensors") != frozen["source_adapter_sha256"]:
        raise ValueError("source adapter changed")
    if sha256(args.target_adapter / "adapter_model.safetensors") != frozen["target_adapter_sha256"]:
        raise ValueError("target adapter changed")
    for adapter in (args.source_adapter, args.target_adapter):
        if sha256(adapter / "preprocessor_config.json") != frozen["processor_sha256"]:
            raise ValueError("processor contract changed")
    base_shards = sorted(args.base_model.glob("model-*-of-*.safetensors"))
    if [sha256(path) for path in base_shards] != frozen["base_shard_sha256"]:
        raise ValueError("base model changed")
    if sha256(args.base_projection) != frozen["base_projection_sha256"]:
        raise ValueError("base projection changed")

    rows = pq.read_table(args.slice, columns=["corpus_id", "image_binary"]).to_pylist()
    if len(rows) != frozen["pages"]:
        raise ValueError("page scope changed")
    image_bytes = [bytes(row["image_binary"]) for row in rows]
    item_ids = [str(row["corpus_id"]) for row in rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("duplicate page ids")
    batch_size = int(frozen["batch_size"])
    starts = list(range(0, len(rows), batch_size))
    processor = ColQwen2Processor.from_pretrained(
        args.base_model,
        local_files_only=True,
        min_pixels=frozen["min_pixels"],
        max_pixels=frozen["max_pixels"],
    )

    def load_model(adapter: Path) -> tuple[Any, Any, dict[str, Any]]:
        base = ColQwen2.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        projection = load_file(args.base_projection)
        with torch.no_grad():
            base.custom_text_proj.weight.copy_(projection["custom_text_proj.weight"].to(base.custom_text_proj.weight))
            base.custom_text_proj.bias.copy_(projection["custom_text_proj.bias"].to(base.custom_text_proj.bias))
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
        if len(weights) != 394 or remapped_count != 392:
            raise RuntimeError(f"unexpected adapter tensor contract: total={len(weights)}, remapped={remapped_count}")
        load_result = set_peft_model_state_dict(model, remapped, adapter_name="default")
        missing = [key for key in load_result.missing_keys if "lora_" in key]
        unexpected = [key for key in load_result.unexpected_keys if "lora_" in key]
        if missing or unexpected:
            raise RuntimeError(f"adapter load incomplete: missing={len(missing)}, unexpected={len(unexpected)}")
        model = model.to(device).eval()
        return model, model.get_base_model(), {
            "checkpoint_tensors": len(weights),
            "remapped_decoder_lora_tensors": remapped_count,
            "missing_adapter_tensors": len(missing),
            "unexpected_adapter_tensors": len(unexpected),
        }

    def prepare(base: Any, start: int) -> dict[str, Any]:
        images = [Image.open(io.BytesIO(value)).convert("RGB") for value in image_bytes[start : start + batch_size]]
        batch = processor.process_images(images).to(device)
        input_ids = batch["input_ids"]
        attention = batch["attention_mask"]
        grid = batch["image_grid_thw"]
        offsets = grid[:, 1] * grid[:, 2]
        pixels = torch.cat(
            [row[:offset] for row, offset in zip(batch["pixel_values"], offsets)], dim=0
        )
        embeds = base.get_input_embeddings()(input_ids)
        vision = torch.cat(base.get_image_features(pixels, grid), dim=0).to(embeds.device, embeds.dtype)
        image_mask = input_ids == int(base.config.image_token_id)
        embeds = embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(embeds), vision)
        position_ids, _ = base.get_rope_index(input_ids, grid, None, attention_mask=attention)
        return {
            "embeds": embeds,
            "attention_mask": attention,
            "position_ids": position_ids,
            "cache_position": torch.arange(embeds.shape[1], device=device),
            "image_mask": image_mask,
        }

    def to_cpu(value: dict[str, Any]) -> dict[str, Any]:
        return {key: tensor.detach().cpu() for key, tensor in value.items()}

    def to_device(value: dict[str, Any]) -> dict[str, Any]:
        return {key: tensor.to(device) for key, tensor in value.items()}

    def terminal(model: Any, data: dict[str, Any]) -> list[Any]:
        values = model(
            input_ids=None,
            inputs_embeds=data["embeds"],
            attention_mask=data["attention_mask"],
            position_ids=data["position_ids"],
            cache_position=data["cache_position"],
            use_cache=False,
            return_dict=True,
        )
        return [
            row[mask].float().detach().cpu()
            for row, mask in zip(values, data["image_mask"])
        ]

    source_load_started = time.perf_counter()
    source_model, source_base, source_diagnostics = load_model(args.source_adapter)
    source_load_seconds = time.perf_counter() - source_load_started
    args.ir_root.mkdir(parents=True)
    source_outputs: list[Any] = []
    ir_records = []
    torch.cuda.synchronize()
    creation_started = time.perf_counter()
    with torch.inference_mode():
        for index, start in enumerate(starts):
            data = prepare(source_base, start)
            source_outputs.extend(terminal(source_model, data))
            path = args.ir_root / f"vision-ir-{index:03d}.pt"
            torch.save(to_cpu(data), path)
            ir_records.append({"path": path.name, "bytes": path.stat().st_size})
    torch.cuda.synchronize()
    creation_seconds = time.perf_counter() - creation_started
    del source_model, source_base
    gc.collect()
    torch.cuda.empty_cache()

    target_load_started = time.perf_counter()
    target_model, target_base, target_diagnostics = load_model(args.target_adapter)
    target_load_seconds = time.perf_counter() - target_load_started

    def run(method: str, limit_one: bool = False) -> tuple[dict[str, float], list[Any]]:
        local_starts = starts[:1] if limit_one else starts
        source_seconds = 0.0
        suffix_seconds = 0.0
        outputs: list[Any] = []
        torch.cuda.synchronize()
        wall_started = time.perf_counter()
        with torch.inference_mode():
            for index, start in enumerate(local_starts):
                torch.cuda.synchronize()
                stage_started = time.perf_counter()
                if method == "raw":
                    data = prepare(target_base, start)
                elif method == "replay":
                    data = to_device(torch.load(args.ir_root / f"vision-ir-{index:03d}.pt", weights_only=True))
                else:
                    raise ValueError(method)
                torch.cuda.synchronize()
                source_seconds += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                outputs.extend(terminal(target_model, data))
                torch.cuda.synchronize()
                suffix_seconds += time.perf_counter() - stage_started
        return {
            "wall_seconds": time.perf_counter() - wall_started,
            "source_seconds": source_seconds,
            "decoder_projection_d2h_seconds": suffix_seconds,
        }, outputs

    run("raw", limit_one=True)
    run("replay", limit_one=True)
    observations: dict[str, list[dict[str, Any]]] = {"raw": [], "replay": []}
    first_outputs: dict[str, list[Any]] = {}
    for repeat, order in enumerate(protocol["measurement"]["order"]):
        for method in order:
            timing, outputs = run(method)
            observations[method].append({"repeat": repeat, **timing})
            if method not in first_outputs:
                first_outputs[method] = outputs
            print(json.dumps({"repeat": repeat, "method": method, **timing}), flush=True)

    raw_outputs = first_outputs["raw"]
    replay_outputs = first_outputs["replay"]
    if not (len(raw_outputs) == len(replay_outputs) == len(source_outputs) == len(rows)):
        raise RuntimeError("terminal page bank is incomplete")

    def compare(left: list[Any], right: list[Any]) -> dict[str, Any]:
        equal = True
        maximum = 0.0
        changed = 0
        total = 0
        cosines = []
        for a, b in zip(left, right):
            if a.shape != b.shape:
                raise RuntimeError("terminal shape changed")
            equal = equal and torch.equal(a, b)
            maximum = max(maximum, float((a - b).abs().max()))
            changed += int((a != b).sum())
            total += a.numel()
            cosines.append(float(F.cosine_similarity(a, b, dim=-1).mean()))
        return {
            "tensor_equal": equal,
            "max_absolute_error": maximum,
            "changed_elements": changed,
            "total_elements": total,
            "changed_fraction": changed / total,
            "mean_token_cosine": statistics.fmean(cosines),
        }

    target_equivalence = compare(raw_outputs, replay_outputs)
    version_invalidation = compare(source_outputs, replay_outputs)
    raw_times = [value["wall_seconds"] for value in observations["raw"]]
    replay_times = [value["wall_seconds"] for value in observations["replay"]]
    paired_saved = [raw - replay for raw, replay in zip(raw_times, replay_times)]
    raw_summary = summarize(raw_times)
    replay_summary = summarize(replay_times)
    saving = 1.0 - replay_summary["median"] / raw_summary["median"]
    ir_bytes = sum(value["bytes"] for value in ir_records)
    image_storage = sum(len(value) for value in image_bytes)
    terminal_bytes = sum(value.numel() * value.element_size() for value in replay_outputs)
    stable_shapes = len({tuple(value.shape) for value in replay_outputs}) == 1
    gates = {
        "target_equivalence": target_equivalence["tensor_equal"] or target_equivalence["max_absolute_error"] <= 1e-6,
        "terminal_invalidation": version_invalidation["changed_fraction"] >= 0.99 and version_invalidation["mean_token_cosine"] <= 0.999,
        "physical_payoff": saving >= 0.50 and all(value > 0 for value in paired_saved),
        "scope": stable_shapes and len(replay_outputs) == frozen["pages"],
    }
    result = {
        "protocol": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "cpu_result_sha256": sha256(args.cpu_result),
        "scope": {
            "pages": len(rows),
            "batches": len(starts),
            "batch_size": batch_size,
            "terminal_shape": list(replay_outputs[0].shape),
        },
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "model_load_seconds": {"source": source_load_seconds, "target": target_load_seconds},
        "adapter_load": {"source": source_diagnostics, "target": target_diagnostics},
        "source_ir": {
            "creation_seconds_including_source_terminal": creation_seconds,
            "physical_bytes": ir_bytes,
            "raw_compressed_image_bytes": image_storage,
            "target_float32_terminal_bytes": terminal_bytes,
            "ir_to_image_ratio": ir_bytes / image_storage,
            "ir_to_terminal_ratio": ir_bytes / terminal_bytes,
            "shards": len(ir_records),
        },
        "observations": observations,
        "timing": {
            "raw_target_seconds": raw_summary,
            "source_ir_to_target_seconds": replay_summary,
            "paired_raw_minus_replay_seconds": paired_saved,
            "median_saving_fraction": saving,
        },
        "target_equivalence": target_equivalence,
        "version_invalidation": version_invalidation,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("scope", "source_ir", "timing", "target_equivalence", "version_invalidation", "gates", "all_gates_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
