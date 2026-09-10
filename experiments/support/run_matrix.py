#!/usr/bin/env python3
"""Evaluate target-semantic multimodal index evolution on three IR benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


METHODS = (
    "old_native",
    "stale",
    "procrustes",
    "affine_ridge",
    "target_full",
    "reprforge",
)
CORPORA = ("arxivqa", "docvqa", "flickr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arxivqa-parquet", type=Path, required=True)
    parser.add_argument("--docvqa-parquet", type=Path, required=True)
    parser.add_argument("--flickr-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--processor-model", type=Path, required=True)
    parser.add_argument("--old-adapter", type=Path, required=True)
    parser.add_argument("--new-adapter", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--support-code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ParquetRows:
    def __init__(self, path: Path, query_column: str) -> None:
        import pyarrow.parquet as pq
        from PIL import Image

        self.table = pq.read_table(path)
        self.query_column = query_column
        self.image_type = Image

    def __len__(self) -> int:
        return len(self.table)

    def get(self, index: int) -> tuple[Any, str]:
        row = self.table.slice(index, 1).to_pylist()[0]
        value = row["image"]
        if isinstance(value, dict):
            value = value.get("bytes")
        image = self.image_type.open(io.BytesIO(value)).convert("RGB")
        query = row[self.query_column]
        if isinstance(query, (list, tuple)):
            query = query[0]
        return image, str(query)


class FlickrRows:
    def __init__(self, root: Path) -> None:
        import pandas as pd

        self.frame = pd.read_csv(root / "test_1k_flickr.csv")
        for column in ("sentids", "imgid", "tokens", "raw", "sentid"):
            self.frame[column] = self.frame[column].apply(json.loads)
        self.archive = zipfile.ZipFile(root / "images_flickr_1k_test.zip")

    def __len__(self) -> int:
        return len(self.frame)

    def get(self, index: int) -> tuple[Any, str]:
        from PIL import Image

        row = self.frame.iloc[index]
        member = f"images_flickr_1k_test/{row['filename']}"
        with self.archive.open(member) as handle:
            image = Image.open(io.BytesIO(handle.read())).convert("RGB")
        query = row["raw"]
        if isinstance(query, (list, tuple)):
            query = query[0]
        return image, str(query)


def metric_rows(scores: Any, target_order: Any) -> tuple[dict[str, float], list[dict[str, float]]]:
    import numpy as np

    order = np.argsort(-scores, axis=1)
    overlap_depth = min(10, scores.shape[1])
    rows = []
    for query_index in range(len(order)):
        rank = int(np.flatnonzero(order[query_index] == query_index)[0]) + 1
        top10 = set(int(value) for value in order[query_index, :overlap_depth])
        target_top10 = set(int(value) for value in target_order[query_index, :overlap_depth])
        rows.append({
            "rank": rank,
            "ndcg_at_5": 1.0 / math.log2(rank + 1) if rank <= 5 else 0.0,
            "recall_at_1": float(rank <= 1),
            "recall_at_5": float(rank <= 5),
            "recall_at_10": float(rank <= 10),
            "reciprocal_rank": 1.0 / rank,
            "top10_overlap": len(top10 & target_top10) / overlap_depth,
        })
    aggregate = {
        key: statistics.fmean(row[key] for row in rows)
        for key in (
            "ndcg_at_5",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "reciprocal_rank",
            "top10_overlap",
        )
    }
    return aggregate, rows


def paired_bootstrap(
    left: list[dict[str, float]],
    right: list[dict[str, float]],
    *,
    field: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    difference = np.asarray(
        [left_row[field] - right_row[field] for left_row, right_row in zip(left, right, strict=True)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(draws, len(difference)))
    distribution = difference[indices].mean(axis=1)
    return {
        "estimate": float(difference.mean()),
        "percentile_95": [
            float(value) for value in np.quantile(distribution, (0.025, 0.975))
        ],
        "draws": draws,
        "seed": seed,
        "wins_ties_losses": [
            int((difference > 0).sum()),
            int((difference == 0).sum()),
            int((difference < 0).sum()),
        ],
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one GPU")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    sys.path.insert(0, str(args.support_code_root))
    from run_smoke import load_compatible_adapter, load_projection
    from streaming_maxsim import streaming_maxsim

    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen-before-gpu-output":
        raise RuntimeError("protocol is not frozen")
    expected = protocol["transition"]
    for path, value in (
        (args.old_adapter / "adapter_model.safetensors", expected["old_adapter_sha256"]),
        (args.new_adapter / "adapter_model.safetensors", expected["new_adapter_sha256"]),
        (args.projection, expected["projection_seed_sha256"]),
        (args.bridge, protocol["bridge"]["artifact_sha256"]),
        (
            args.processor_model / "preprocessor_config.json",
            protocol["processor"]["preprocessor_config_sha256"],
        ),
        (
            args.processor_model / "tokenizer_config.json",
            protocol["processor"]["tokenizer_config_sha256"],
        ),
    ):
        if sha256(path) != value:
            raise RuntimeError(f"artifact hash mismatch: {path}")
    benchmark_paths = {
        "arxivqa": args.arxivqa_parquet,
        "docvqa": args.docvqa_parquet,
    }
    for name, path in benchmark_paths.items():
        if sha256(path) != protocol["benchmarks"][name]["file_sha256"]:
            raise RuntimeError(f"dataset hash mismatch: {name}")
    for name, value in protocol["benchmarks"]["flickr"]["files_sha256"].items():
        if sha256(args.flickr_root / name) != value:
            raise RuntimeError(f"dataset hash mismatch: flickr/{name}")

    torch.manual_seed(int(protocol["reusable_ir"]["torch_seed"]))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    projection = load_projection(args.projection, torch)

    def load_model(adapter: Path) -> tuple[Any, Any]:
        base = ColQwen2_5.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        with torch.no_grad():
            base.custom_text_proj.weight.copy_(
                projection["custom_text_proj.weight"].to(base.custom_text_proj.weight)
            )
            base.custom_text_proj.bias.copy_(
                projection["custom_text_proj.bias"].to(base.custom_text_proj.bias)
            )
        wrapped = load_compatible_adapter(base, adapter).to(args.device).eval()
        return wrapped, wrapped.get_base_model()

    model_started = time.perf_counter()
    old_model, old_base = load_model(args.old_adapter)
    new_model, new_base = load_model(args.new_adapter)
    processor = ColQwen2_5_Processor.from_pretrained(
        args.processor_model, local_files_only=True
    )
    model_seconds = time.perf_counter() - model_started
    bridge = torch.load(args.bridge, weights_only=True)
    if bridge["protocol_id"] != "query-drift-bridge-v1":
        raise RuntimeError("unexpected bridge protocol")
    procrustes = bridge["procrustes"].float()
    affine = bridge["affine_ridge"].float()

    def prepare(base: Any, batch: Any) -> dict[str, Any]:
        input_ids = batch["input_ids"]
        attention = batch["attention_mask"]
        grid = batch["image_grid_thw"]
        offsets = grid[:, 1] * grid[:, 2]
        pixels = torch.cat(
            [row[: int(offset.item())] for row, offset in zip(batch["pixel_values"], offsets, strict=True)],
            dim=0,
        )
        embeddings = base.get_input_embeddings()(input_ids)
        vision = torch.cat(base.get_image_features(pixels, grid), dim=0).to(
            embeddings.device, embeddings.dtype
        )
        embeddings = embeddings.masked_scatter(
            (input_ids == int(base.config.image_token_id)).unsqueeze(-1).expand_as(embeddings),
            vision,
        )
        position_ids, _ = base.get_rope_index(input_ids, grid, None, attention_mask=attention)
        return {
            "input_ids": input_ids,
            "attention": attention,
            "grid": grid,
            "embeds": embeddings,
            "position_ids": position_ids,
            "cache_position": torch.arange(embeddings.shape[1], device=args.device),
            "valid": [torch.where(row.bool())[0] for row in attention],
            "visual": [
                torch.where((ids == int(base.config.image_token_id)) & mask.bool())[0]
                for ids, mask in zip(input_ids, attention, strict=True)
            ],
        }

    def language(base: Any, data: dict[str, Any]) -> list[Any]:
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
        projected = base.custom_text_proj(output.last_hidden_state)
        projected = functional.normalize(projected, dim=-1).float()
        return [
            projected[row].index_select(0, valid)
            for row, valid in enumerate(data["valid"])
        ]

    def map_query(value: Any, method: str) -> Any:
        source = value.float()
        if method == "procrustes":
            mapped = source @ procrustes
        elif method == "affine_ridge":
            mapped = torch.cat((source, torch.ones((source.shape[0], 1))), dim=1) @ affine
        else:
            raise KeyError(method)
        return functional.normalize(mapped, dim=-1)

    all_datasets = {
        "arxivqa": ParquetRows(args.arxivqa_parquet, "query"),
        "docvqa": ParquetRows(args.docvqa_parquet, "query"),
        "flickr": FlickrRows(args.flickr_root),
    }
    datasets = {name: all_datasets[name] for name in CORPORA if name in args.corpora}
    results = {}
    for corpus, dataset in datasets.items():
        expected_rows = int(protocol["benchmarks"][corpus]["rows"])
        if len(dataset) != expected_rows:
            raise RuntimeError(f"row count mismatch for {corpus}: {len(dataset)}")
        permutation = np.random.RandomState(0).permutation(expected_rows)
        evaluation = sorted(int(value) for value in permutation[: int(round(0.3 * expected_rows))])
        calibration = [index for index in range(expected_rows) if index not in set(evaluation)][:64]
        if args.smoke:
            evaluation = evaluation[: args.smoke]
            calibration = calibration[: min(args.smoke, len(calibration))]

        corpus_root = args.output_root / corpus
        corpus_root.mkdir()
        timing: dict[str, float] = {}
        calibration_values = []
        calibration_started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(calibration), args.image_batch_size):
                indices = calibration[start : start + args.image_batch_size]
                images = [dataset.get(index)[0] for index in indices]
                batch = processor.process_images(images).to(args.device)
                data = prepare(old_base, batch)
                for row, visual in enumerate(data["visual"]):
                    calibration_values.append(
                        data["embeds"][row].index_select(0, visual).detach().cpu()
                    )
        observed = torch.cat(calibration_values, dim=0)
        token_count = min(int(protocol["reusable_ir"]["sampled_visual_tokens"]), len(observed))
        selection = torch.linspace(0, len(observed) - 1, token_count).round().long()
        calibration_tensor = observed.index_select(0, selection).to(args.device).float()
        mean = calibration_tensor.mean(dim=0)
        rank = min(int(protocol["reusable_ir"]["rank"]), len(calibration_tensor) - 1)
        _, _, basis = torch.pca_lowrank(
            calibration_tensor - mean,
            q=rank,
            center=False,
            niter=int(protocol["reusable_ir"]["power_iterations"]),
        )
        timing["calibration_seconds"] = time.perf_counter() - calibration_started
        basis_path = corpus_root / "basis.pt"
        torch.save(
            {"mean": mean.to(torch.bfloat16).cpu(), "basis": basis.to(torch.bfloat16).cpu()},
            basis_path,
        )

        first_images = [dataset.get(index)[0] for index in evaluation[: args.image_batch_size]]
        first_batch = processor.process_images(first_images).to(args.device)
        with torch.inference_mode():
            old_canary = prepare(old_base, first_batch)
            new_canary = prepare(new_base, first_batch)
        canary = {
            key: bool(torch.equal(old_canary[key], new_canary[key]))
            for key in ("input_ids", "attention", "grid", "embeds", "position_ids", "cache_position")
        }
        if not all(canary.values()):
            raise RuntimeError(f"prefix canary failed for {corpus}: {canary}")

        old_documents: list[Any] = []
        target_documents: list[Any] = []
        replay_documents: list[Any] = []
        prefix_seconds = old_suffix_seconds = target_suffix_seconds = replay_seconds = 0.0
        with torch.inference_mode():
            for start in range(0, len(evaluation), args.image_batch_size):
                indices = evaluation[start : start + args.image_batch_size]
                images = [dataset.get(index)[0] for index in indices]
                torch.cuda.synchronize()
                started = time.perf_counter()
                batch = processor.process_images(images).to(args.device)
                data = prepare(old_base, batch)
                torch.cuda.synchronize()
                prefix_seconds += time.perf_counter() - started

                torch.cuda.synchronize()
                started = time.perf_counter()
                old_values = language(old_base, data)
                torch.cuda.synchronize()
                old_suffix_seconds += time.perf_counter() - started

                torch.cuda.synchronize()
                started = time.perf_counter()
                target_values = language(new_base, data)
                torch.cuda.synchronize()
                target_suffix_seconds += time.perf_counter() - started

                torch.cuda.synchronize()
                started = time.perf_counter()
                replay_data = dict(data)
                replay_data["embeds"] = data["embeds"].clone()
                for row, visual in enumerate(data["visual"]):
                    values = data["embeds"][row].index_select(0, visual).float()
                    coefficients = (values - mean) @ basis
                    scale = coefficients.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
                    quantized = torch.round(coefficients / scale).clamp(-127, 127).to(torch.int8)
                    restored = (quantized.float() * scale) @ basis.T + mean
                    replay_data["embeds"][row, visual] = restored.to(torch.bfloat16)
                replay_values = language(new_base, replay_data)
                torch.cuda.synchronize()
                replay_seconds += time.perf_counter() - started

                old_documents.extend(value.to(torch.bfloat16).cpu() for value in old_values)
                target_documents.extend(value.to(torch.bfloat16).cpu() for value in target_values)
                replay_documents.extend(value.to(torch.bfloat16).cpu() for value in replay_values)
                print(json.dumps({"corpus": corpus, "encoded": min(start + len(indices), len(evaluation)), "total": len(evaluation)}), flush=True)

        queries = [dataset.get(index)[1] for index in evaluation]
        old_queries: list[Any] = []
        new_queries: list[Any] = []
        query_started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(queries), args.query_batch_size):
                batch = processor.process_queries(queries[start : start + args.query_batch_size]).to(args.device)
                old_values = old_model(**batch).float()
                new_values = new_model(**batch).float()
                for old_value, new_value, mask in zip(old_values, new_values, batch["attention_mask"], strict=True):
                    keep = mask.bool()
                    old_queries.append(old_value[keep].detach().cpu())
                    new_queries.append(new_value[keep].detach().cpu())
        torch.cuda.synchronize()
        timing.update({
            "prefix_seconds": prefix_seconds,
            "old_suffix_seconds": old_suffix_seconds,
            "target_suffix_seconds": target_suffix_seconds,
            "reprforge_codec_and_suffix_seconds": replay_seconds,
            "paired_query_encoding_seconds": time.perf_counter() - query_started,
        })

        query_banks = {
            "old_native": old_queries,
            "stale": new_queries,
            "procrustes": [map_query(value, "procrustes") for value in new_queries],
            "affine_ridge": [map_query(value, "affine_ridge") for value in new_queries],
            "target_full": new_queries,
            "reprforge": new_queries,
        }
        document_banks = {
            "old_native": old_documents,
            "stale": old_documents,
            "procrustes": old_documents,
            "affine_ridge": old_documents,
            "target_full": target_documents,
            "reprforge": replay_documents,
        }
        score_matrices = {
            method: streaming_maxsim(
                query_banks[method], document_banks[method], device=args.device,
                query_chunk=32, document_chunk=16,
            )
            for method in METHODS
        }
        target_order = np.argsort(-score_matrices["target_full"], axis=1)
        aggregates = {}
        rows_by_method = {}
        per_query = {str(index): {"row_id": evaluation[index]} for index in range(len(evaluation))}
        for method in METHODS:
            aggregate, rows = metric_rows(score_matrices[method], target_order)
            aggregates[method] = aggregate
            rows_by_method[method] = rows
            for index, row in enumerate(rows):
                per_query[str(index)][method] = row

        bootstrap = {
            method: {
                field: paired_bootstrap(
                    rows_by_method[method],
                    rows_by_method["target_full"],
                    field=field,
                    draws=20_000,
                    seed=int(protocol["reusable_ir"]["torch_seed"])
                    + 100 * CORPORA.index(corpus)
                    + 10 * METHODS.index(method)
                    + field_index,
                )
                for field_index, field in enumerate(("ndcg_at_5", "top10_overlap"))
            }
            for method in METHODS
            if method != "target_full"
        }

        result = {
            "corpus": corpus,
            "scope": {
                "rows": expected_rows,
                "evaluation_documents_queries": len(evaluation),
                "calibration_documents": len(calibration),
            },
            "prefix_canary": canary,
            "basis": {
                "rank": rank,
                "observed_visual_tokens": len(observed),
                "sampled_visual_tokens": token_count,
                "sha256": sha256(basis_path),
                "bytes": basis_path.stat().st_size,
            },
            "metrics": aggregates,
            "differences_vs_target_full": {
                method: {
                    "ndcg_at_5": aggregates[method]["ndcg_at_5"] - aggregates["target_full"]["ndcg_at_5"],
                    "top10_overlap": aggregates[method]["top10_overlap"] - aggregates["target_full"]["top10_overlap"],
                }
                for method in METHODS if method != "target_full"
            },
            "paired_bootstrap_vs_target_full": bootstrap,
            "timing_seconds": timing,
            "per_query": per_query,
        }
        (corpus_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results[corpus] = result
        print(json.dumps({"completed": corpus, "metrics": aggregates}, indent=2), flush=True)

    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "smoke": args.smoke,
        "hardware": torch.cuda.get_device_name(torch.cuda.current_device()),
        "model_load_seconds": model_seconds,
        "results": {
            corpus: {
                "scope": result["scope"],
                "metrics": result["metrics"],
                "differences_vs_target_full": result["differences_vs_target_full"],
                "paired_bootstrap_vs_target_full": result["paired_bootstrap_vs_target_full"],
            }
            for corpus, result in results.items()
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    (args.output_root / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
