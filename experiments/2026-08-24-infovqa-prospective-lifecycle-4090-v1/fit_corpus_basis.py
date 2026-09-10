#!/usr/bin/env python3
"""Fit a frozen query-free post-vision PCA basis on public InfoVQA pages."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--corpus-parquet", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--processor-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one GPU")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)

    import pyarrow.parquet as pq
    import numpy as np
    import torch
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from PIL import Image

    protocol = json.loads(args.protocol.read_text())
    expected_dataset_hash = protocol["dataset"]["file_sha256"]
    observed_dataset_hash = sha256(args.corpus_parquet)
    if observed_dataset_hash != expected_dataset_hash:
        raise RuntimeError(
            f"calibration dataset hash mismatch: {observed_dataset_hash} != {expected_dataset_hash}"
        )
    observed_model_hashes = {
        path.name: sha256(path)
        for path in sorted(args.base_model.glob("model-*-of-*.safetensors"))
    }
    expected_model_hashes = protocol["model"]["model_shards_sha256"]
    if observed_model_hashes != expected_model_hashes:
        raise RuntimeError(
            f"base model hash mismatch: {observed_model_hashes} != {expected_model_hashes}"
        )
    processor_hash = sha256(args.processor_model / "preprocessor_config.json")
    video_processor_hash = sha256(
        args.processor_model / "video_preprocessor_config.json"
    )
    if processor_hash != protocol["model"]["preprocessor_sha256"]:
        raise RuntimeError("image processor hash mismatch")
    if video_processor_hash != protocol["model"]["video_preprocessor_sha256"]:
        raise RuntimeError("video processor hash mismatch")
    pages = int(protocol["dataset"]["pages"])
    maximum_tokens = int(protocol["pca"]["maximum_tokens"])
    rank = int(protocol["pca"]["rank"])
    niter = int(protocol["pca"]["power_iterations"])
    seed = int(protocol["pca"]["torch_seed"])

    table = pq.read_table(args.corpus_parquet)
    total_rows = int(protocol["dataset"]["rows"])
    if len(table) != total_rows:
        raise RuntimeError(f"expected {total_rows} rows, got {len(table)}")
    permutation = np.random.RandomState(0).permutation(total_rows)
    evaluation = set(int(value) for value in permutation[: int(round(0.3 * total_rows))])
    calibration_indices = [index for index in range(total_rows) if index not in evaluation][:pages]
    if len(calibration_indices) != pages or evaluation.intersection(calibration_indices):
        raise RuntimeError("invalid calibration/evaluation split")
    rows = table.take(calibration_indices).to_pylist()

    def image_bytes(row: dict) -> bytes:
        value = row.get("image_binary", row.get("image"))
        if isinstance(value, dict):
            value = value.get("bytes")
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError(f"unsupported image payload type: {type(value)!r}")
        return bytes(value)

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = ColQwen2_5.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    processor = ColQwen2_5_Processor.from_pretrained(
        args.processor_model, local_files_only=True
    )
    image_token_id = int(model.config.image_token_id)
    values = []
    grid_signatures: dict[str, int] = {}
    with torch.inference_mode():
        for start in range(0, pages, args.batch_size):
            images = [
                Image.open(io.BytesIO(image_bytes(row))).convert("RGB")
                for row in rows[start : start + args.batch_size]
            ]
            batch = processor.process_images(images).to(args.device)
            grid = batch["image_grid_thw"]
            offsets = grid[:, 1] * grid[:, 2]
            pixels = torch.cat(
                [row[: int(offset.item())] for row, offset in zip(batch["pixel_values"], offsets, strict=True)],
                dim=0,
            )
            embeddings = model.get_input_embeddings()(batch["input_ids"])
            vision = torch.cat(model.get_image_features(pixels, grid), dim=0).to(
                embeddings.device, embeddings.dtype
            )
            mask = batch["input_ids"] == image_token_id
            embeddings = embeddings.masked_scatter(mask.unsqueeze(-1).expand_as(embeddings), vision)
            values.append(embeddings[mask].detach().cpu())
            for signature in grid.tolist():
                key = "x".join(str(int(item)) for item in signature)
                grid_signatures[key] = grid_signatures.get(key, 0) + 1
            print(json.dumps({"calibration_pages": min(start + len(images), pages), "total": pages}), flush=True)

    calibration = torch.cat(values, dim=0)
    observed_tokens = len(calibration)
    if observed_tokens > maximum_tokens:
        selection = torch.linspace(0, observed_tokens - 1, maximum_tokens).round().long()
        calibration = calibration.index_select(0, selection)
    calibration = calibration.to(args.device).float()
    mean = calibration.mean(dim=0)
    _, _, basis = torch.pca_lowrank(
        calibration - mean, q=rank, center=False, niter=niter
    )
    artifact = {
        "mean": mean.to(torch.bfloat16).cpu(),
        "basis": basis.to(torch.bfloat16).cpu(),
    }
    basis_path = args.output_root / "basis.pt"
    torch.save(artifact, basis_path)
    result = {
        "protocol": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "corpus_parquet_sha256": observed_dataset_hash,
        "base_model_shards_sha256": observed_model_hashes,
        "processor_sha256": processor_hash,
        "video_processor_sha256": video_processor_hash,
        "pages": pages,
        "calibration_indices": calibration_indices,
        "observed_visual_tokens": observed_tokens,
        "sampled_visual_tokens": len(calibration),
        "grid_signatures": grid_signatures,
        "basis_shape": list(artifact["basis"].shape),
        "basis_bytes": basis_path.stat().st_size,
        "basis_sha256": sha256(basis_path),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    (args.output_root / "calibration-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
