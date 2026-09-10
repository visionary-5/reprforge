#!/usr/bin/env python3
"""Inspect the frozen ColQwen2 successor release without loading tensor payloads."""

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


def header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 0 < length <= 100_000_000:
            raise ValueError(f"invalid safetensors header: {path}")
        payload = handle.read(length)
    if len(payload) != length:
        raise ValueError(f"truncated safetensors header payload: {path}")
    value = json.loads(payload)
    return {key: item for key, item in value.items() if key != "__metadata__"}


def classify(key: str) -> str:
    lowered = key.lower()
    if "vision" in lowered or "visual" in lowered:
        return "vision"
    if "embed_tokens" in lowered or "input_embedding" in lowered:
        return "base_embedding"
    if "custom_text_proj" in lowered or "retrieval_projection" in lowered:
        return "projection"
    if "language_model" in lowered or ".model.layers." in lowered:
        return "decoder"
    return "unknown"


def inspect(root: Path) -> dict[str, Any]:
    adapter = root / "adapter_model.safetensors"
    config_path = root / "adapter_config.json"
    processor_path = root / "preprocessor_config.json"
    for path in (adapter, config_path, processor_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    tensor_header = header(adapter)
    counts = {name: 0 for name in ("decoder", "projection", "vision", "base_embedding", "unknown")}
    parameters = {name: 0 for name in counts}
    signatures = {}
    for key, metadata in sorted(tensor_header.items()):
        category = classify(key)
        shape = [int(value) for value in metadata["shape"]]
        numel = 1
        for extent in shape:
            numel *= extent
        counts[category] += 1
        parameters[category] += numel
        signatures[key] = {"dtype": metadata["dtype"], "shape": shape}
    config = json.loads(config_path.read_text())
    return {
        "adapter_sha256": sha256(adapter),
        "adapter_bytes": adapter.stat().st_size,
        "adapter_config_sha256": sha256(config_path),
        "processor_sha256": sha256(processor_path),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "rank": config.get("r"),
        "target_modules": config.get("target_modules"),
        "tensor_counts": counts,
        "parameter_counts": parameters,
        "tensor_signatures": signatures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    frozen = protocol["frozen_files"]
    source = inspect(args.source)
    target = inspect(args.target)
    expected = {
        "source_adapter_sha256": source["adapter_sha256"] == frozen["source_adapter_sha256"],
        "target_adapter_sha256": target["adapter_sha256"] == frozen["target_adapter_sha256"],
        "source_config_sha256": source["adapter_config_sha256"] == frozen["adapter_config_sha256"],
        "target_config_sha256": target["adapter_config_sha256"] == frozen["adapter_config_sha256"],
        "source_processor_sha256": source["processor_sha256"] == frozen["processor_sha256"],
        "target_processor_sha256": target["processor_sha256"] == frozen["processor_sha256"],
    }
    same_contract = (
        source["base_model_name_or_path"] == target["base_model_name_or_path"] == frozen["base_model_name_or_path"]
        and source["rank"] == target["rank"] == frozen["rank"]
        and source["target_modules"] == target["target_modules"] == frozen["target_modules"]
        and source["tensor_signatures"] == target["tensor_signatures"]
    )
    scope_pass = (
        target["tensor_counts"]["vision"] == 0
        and target["tensor_counts"]["base_embedding"] == 0
        and target["tensor_counts"]["unknown"] == 0
        and target["tensor_counts"]["decoder"] > 0
        and target["tensor_counts"]["projection"] > 0
    )
    gates = {
        "provenance": all(expected.values()),
        "changed_target": source["adapter_sha256"] != target["adapter_sha256"],
        "same_contract": same_contract,
        "post_vision_scope": scope_pass,
    }
    for item in (source, target):
        item.pop("tensor_signatures")
    result = {
        "protocol": protocol["protocol_id"],
        "official_revisions": protocol["official_revisions"],
        "source": source,
        "target": target,
        "file_checks": expected,
        "gates": gates,
        "gpu_promotion": all(gates.values()),
        "dependency_decision": "post_vision_replay_legal" if all(gates.values()) else "raw_rebuild",
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
