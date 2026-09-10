#!/usr/bin/env python3
"""Measure collection-scoped processor equivalence without GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source-processor", type=Path, required=True)
    parser.add_argument("--target-processor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from colpali_engine.models import ColPaliProcessor
    from PIL import Image
    from reprforge import (
        certify_component_fingerprints,
        fingerprint_component_outputs,
    )
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU-only experiment")
    protocol = json.loads(args.protocol.read_text())
    frozen = protocol["frozen_inputs"]
    corpus_paths = sorted(args.corpus.glob("test-*.parquet"))
    if len(corpus_paths) != 3:
        raise ValueError("expected three Energy corpus shards")
    for path, expected in zip(corpus_paths, frozen["corpus"], strict=True):
        observed_rows = pq.ParquetFile(path).metadata.num_rows
        if sha256(path) != expected["sha256"] or observed_rows != expected["rows"]:
            raise ValueError(f"frozen corpus changed: {path}")
    source_config = args.source_processor / "preprocessor_config.json"
    target_config = args.target_processor / "preprocessor_config.json"
    if sha256(source_config) != frozen["source_processor_sha256"]:
        raise ValueError("source processor config changed")
    if sha256(target_config) != frozen["target_processor_sha256"]:
        raise ValueError("target processor config changed")

    table = pa.concat_tables(
        [pq.read_table(path, columns=["corpus_id", "image"]) for path in corpus_paths]
    )
    rows = table.to_pylist()
    if len(rows) != protocol["surface"]["pages"]:
        raise ValueError("complete corpus size changed")
    item_ids = [str(row["corpus_id"]) for row in rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("corpus ids are not unique")
    scope_payload = json.dumps(
        {
            "protocol": protocol["protocol_id"],
            "corpus": frozen["corpus"],
            "item_ids": item_ids,
            "batch_size": protocol["surface"]["batch_size"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    scope_fingerprint = hashlib.sha256(scope_payload).hexdigest()
    images = [Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB") for row in rows]
    batch_size = protocol["surface"]["batch_size"]

    source_processor = ColPaliProcessor.from_pretrained(
        args.source_processor, local_files_only=True
    )
    target_processor = ColPaliProcessor.from_pretrained(
        args.target_processor, local_files_only=True
    )

    def outputs(processor: Any):
        for start in range(0, len(images), batch_size):
            subset = images[start : start + batch_size]
            batch = processor(
                text=["Describe the image."] * len(subset),
                images=subset,
                return_tensors="pt",
                padding="longest",
            )
            yield {
                key: batch[key].contiguous().numpy()
                for key in protocol["surface"]["output_fields"]
            }

    # Warm only implementation dispatch and allocation; the measured pass still
    # decodes and processes every collection page in both versions.
    next(outputs(source_processor))
    next(outputs(target_processor))

    source_started = time.perf_counter()
    source_outputs = fingerprint_component_outputs(
        scope_fingerprint=scope_fingerprint,
        outputs=outputs(source_processor),
    )
    source_seconds = time.perf_counter() - source_started

    target_started = time.perf_counter()
    target_outputs = fingerprint_component_outputs(
        scope_fingerprint=scope_fingerprint,
        outputs=outputs(target_processor),
    )
    target_seconds = time.perf_counter() - target_started

    compare_started = time.perf_counter()
    certificate = certify_component_fingerprints(
        component="processor",
        source_component_fingerprint=frozen["source_processor_sha256"],
        target_component_fingerprint=frozen["target_processor_sha256"],
        source_outputs=source_outputs,
        target_outputs=target_outputs,
    )
    comparison_seconds = time.perf_counter() - compare_started

    negative_control_rejected = False
    changed_digest = (
        ("0" if target_outputs.output_sha256[0] != "0" else "1")
        + target_outputs.output_sha256[1:]
    )
    try:
        certify_component_fingerprints(
            component="processor",
            source_component_fingerprint=frozen["source_processor_sha256"],
            target_component_fingerprint="deliberately-changed-processor",
            source_outputs=source_outputs,
            target_outputs=replace(target_outputs, output_sha256=changed_digest),
        )
    except ValueError as error:
        negative_control_rejected = "outputs differ" in str(error)
    if not negative_control_rejected:
        raise RuntimeError("negative digest control was not rejected")

    source_json = json.dumps(source_outputs.to_dict(), sort_keys=True).encode()
    certificate_json = json.dumps(certificate.to_dict(), sort_keys=True).encode()
    result = {
        "protocol": protocol["protocol_id"],
        "scope": {
            "pages": len(rows),
            "batches": source_outputs.compared_items,
            "batch_size": batch_size,
            "scope_fingerprint": scope_fingerprint,
        },
        "processor_files": {
            "source_sha256": frozen["source_processor_sha256"],
            "target_sha256": frozen["target_processor_sha256"],
            "content_equal": False,
        },
        "output_fingerprints": {
            "source": source_outputs.to_dict(),
            "target": target_outputs.to_dict(),
            "equal": source_outputs == target_outputs,
        },
        "certificate": certificate.to_dict(),
        "timing_seconds": {
            "source_build_fingerprint": source_seconds,
            "target_update_fingerprint": target_seconds,
            "digest_comparison": comparison_seconds,
        },
        "metadata_bytes": {
            "source_output_fingerprint_json": len(source_json),
            "transition_certificate_json": len(certificate_json),
            "total": len(source_json) + len(certificate_json),
        },
        "gates": {
            "exact_equivalence": source_outputs == target_outputs,
            "coverage": len(rows) == 2225,
            "compact_proof": len(source_json) + len(certificate_json) <= 4096,
            "negative_control": negative_control_rejected,
            "gpu_uninitialized": not torch.cuda.is_initialized(),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    if not all(result["gates"].values()):
        raise RuntimeError(f"one or more frozen gates failed: {result['gates']}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
