#!/usr/bin/env python3
"""Inspect public adapter dependency scopes without loading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > 100_000_000:
            raise ValueError(f"invalid safetensors header length: {path}")
        payload = handle.read(header_length)
    if len(payload) != header_length:
        raise ValueError(f"truncated safetensors header: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"invalid safetensors header object: {path}")
    return value


def classify(key: str) -> str:
    lowered = key.lower()
    if "vision" in lowered:
        return "vision"
    if "custom_text_proj" in lowered or "retrieval_projection" in lowered:
        return "projection"
    if "embed_tokens" in lowered or "input_embedding" in lowered:
        return "base_embedding"
    if "language_model" in lowered or ".model.layers." in lowered:
        return "decoder"
    return "unknown"


def inspect(label: str, root: Path) -> dict[str, Any]:
    checkpoint = root / "adapter_model.safetensors"
    config_path = root / "adapter_config.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"missing adapter files under {root}")
    header = safetensors_header(checkpoint)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    counts = {name: 0 for name in ("vision", "base_embedding", "decoder", "projection", "unknown")}
    parameters = {name: 0 for name in counts}
    signatures: dict[str, dict[str, Any]] = {}
    for key, metadata in sorted(tensors.items()):
        category = classify(key)
        shape = metadata.get("shape", [])
        numel = 1
        for extent in shape:
            numel *= int(extent)
        counts[category] += 1
        parameters[category] += numel
        signatures[key] = {"dtype": metadata.get("dtype"), "shape": shape}
    config = json.loads(config_path.read_text())
    processor = root / "preprocessor_config.json"
    blockers = []
    if counts["vision"]:
        blockers.append(f"{counts['vision']} vision tensors")
    if counts["base_embedding"]:
        blockers.append(f"{counts['base_embedding']} base-embedding tensors")
    if counts["unknown"]:
        blockers.append(f"{counts['unknown']} unknown tensors")
    return {
        "label": label,
        "root": str(root),
        "adapter_sha256": sha256(checkpoint),
        "adapter_bytes": checkpoint.stat().st_size,
        "adapter_config_sha256": sha256(config_path),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "target_modules": config.get("target_modules"),
        "rank": config.get("r"),
        "tensor_counts": counts,
        "parameter_counts": parameters,
        "tensor_signatures": signatures,
        "post_vision_tensor_scope_valid": not blockers,
        "post_vision_tensor_scope_blockers": blockers,
        "processor": {
            "present": processor.is_file(),
            "sha256": sha256(processor) if processor.is_file() else None,
            "bytes": processor.stat().st_size if processor.is_file() else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIRECTORY"),
        required=True,
    )
    args = parser.parse_args()
    checkpoints = [inspect(label, Path(root)) for label, root in args.checkpoint]
    by_label = {item["label"]: item for item in checkpoints}
    transitions = []
    for source, target in (
        ("colpali-v1.1", "colpali-v1.2"),
        ("colpali-v1.2", "colpali-v1.3"),
        ("colqwen2.5-v0.1", "colqwen2.5-v0.2"),
    ):
        old = by_label[source]
        new = by_label[target]
        old_signatures = old["tensor_signatures"]
        new_signatures = new["tensor_signatures"]
        same_tensor_contract = old_signatures == new_signatures
        old_processor = old["processor"]
        new_processor = new["processor"]
        if old_processor["present"] and new_processor["present"]:
            processor_status = (
                "content_equal"
                if old_processor["sha256"] == new_processor["sha256"]
                else "different_file"
            )
        else:
            processor_status = "not_mirrored"
        same_base_model_name = (
            old["base_model_name_or_path"] == new["base_model_name_or_path"]
        )
        static_admission = (
            same_base_model_name
            and new["post_vision_tensor_scope_valid"]
            and processor_status == "content_equal"
        )
        transitions.append(
            {
                "source": source,
                "target": target,
                "same_base_model_name": same_base_model_name,
                "same_tensor_contract": same_tensor_contract,
                "processor_evidence": processor_status,
                "checkpoint_payload_changed": old["adapter_sha256"] != new["adapter_sha256"],
                "post_vision_tensor_scope_valid": new["post_vision_tensor_scope_valid"],
                "post_vision_static_admission": static_admission,
                "requires_explicit_processor_evidence": processor_status != "content_equal",
            }
        )
    for item in checkpoints:
        item.pop("tensor_signatures")
    output = {
        "schema": "public-release-dependency-trace-v1",
        "checkpoints": checkpoints,
        "official_transitions": transitions,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
