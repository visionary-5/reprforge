#!/usr/bin/env python3
"""Build matched raw/Light and ReprForge/Light target indexes on InfoVQA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--processor-model", type=Path, required=True)
    parser.add_argument("--v01-adapter", type=Path, required=True)
    parser.add_argument("--v02-adapter", type=Path, required=True)
    parser.add_argument("--pca256", type=Path, required=True)
    parser.add_argument("--support-code-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "sum": float(sum(values)),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def merge_tome(page: Any, ratio: float, torch: Any, device: str) -> Any:
    """Exact released Light geometric merge from pinned MarginMerge source."""
    target = max(1, int(round(ratio * page.shape[0])))
    # The released baseline loads endpoints as BF16 before merging in FP32.
    # Keep the serialized physical index on that exact numerical path.
    current = page.to(device).to(torch.bfloat16).float()
    while current.shape[0] > target:
        remove = min(current.shape[0] - target, current.shape[0] // 2)
        if remove <= 0:
            break
        n = current.shape[0]
        ia = torch.arange(0, n, 2, device=device)
        ib = torch.arange(1, n, 2, device=device)
        left = current[ia].float()
        right = current[ib].float()
        similarity = left @ right.T
        maximum, match = similarity.max(dim=1)
        order = maximum.argsort(descending=True)
        merge_left = order[:remove]
        keep_left = order[remove:]
        representatives = right.clone()
        counts = torch.ones(right.shape[0], device=device)
        representatives.index_add_(0, match[merge_left], left[merge_left])
        counts.index_add_(0, match[merge_left], torch.ones(remove, device=device))
        representatives = representatives / counts.unsqueeze(1)
        current = torch.cat([left[keep_left], representatives], 0)
        current = current / current.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return current.to(torch.bfloat16).cpu()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one GPU")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)

    import sys

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from datasets import load_dataset

    sys.path.insert(0, str(args.support_code_root))
    from run_smoke import load_compatible_adapter

    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen-before-gpu-output":
        raise RuntimeError(f"protocol is not frozen: {protocol['status']}")
    transition = protocol["transition"]
    reprforge = protocol["reprforge"]
    observed_dataset_hash = sha256(args.dataset_parquet)
    expected_dataset_hash = protocol["benchmark"]["file_sha256"]
    if observed_dataset_hash != expected_dataset_hash:
        raise RuntimeError(
            f"benchmark dataset hash mismatch: {observed_dataset_hash} != {expected_dataset_hash}"
        )
    expected_hashes = {
        args.v01_adapter / "adapter_model.safetensors": transition["v01_adapter_sha256"],
        args.v02_adapter / "adapter_model.safetensors": transition["v02_adapter_sha256"],
        args.pca256: reprforge["basis_sha256"],
    }
    for path, expected in expected_hashes.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"hash mismatch for {path}: {observed} != {expected}")
    observed_base_hashes = {
        path.name: sha256(path)
        for path in sorted(args.base_model.glob("model-*-of-*.safetensors"))
    }
    if observed_base_hashes != transition["base_shards_sha256"]:
        raise RuntimeError(f"base model hash mismatch: {observed_base_hashes}")
    if sha256(args.processor_model / "preprocessor_config.json") != transition["preprocessor_sha256"]:
        raise RuntimeError("image processor hash mismatch")
    if sha256(args.processor_model / "video_preprocessor_config.json") != transition["video_preprocessor_sha256"]:
        raise RuntimeError("video processor hash mismatch")

    torch.manual_seed(20260823)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset_started = time.perf_counter()
    dataset = load_dataset(
        "parquet",
        data_files={protocol["benchmark"]["split"]: str(args.dataset_parquet)},
        split=protocol["benchmark"]["split"],
    )
    full_size = len(dataset)
    n = min(args.limit, full_size) if args.limit else full_size
    expected_rows = int(protocol["benchmark"]["rows"])
    if n != expected_rows and not args.limit:
        raise RuntimeError(f"expected {expected_rows} benchmark rows, observed {n}")
    queries = []
    for index in range(n):
        query = dataset[index]["query"]
        queries.append(query[0] if isinstance(query, (list, tuple)) else query)
    dataset_seconds = time.perf_counter() - dataset_started
    processor = ColQwen2_5_Processor.from_pretrained(
        args.processor_model, local_files_only=True
    )
    pca = torch.load(args.pca256, weights_only=True)
    if list(pca["basis"].shape) != [2048, 256]:
        raise RuntimeError(f"unexpected PCA shape {list(pca['basis'].shape)}")

    first_images = [
        dataset[index]["image"].convert("RGB")
        for index in range(min(args.image_batch_size, n))
    ]

    def load_model(adapter: Path) -> tuple[Any, Any, dict[str, int]]:
        raw_base = ColQwen2_5.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        wrapped = load_compatible_adapter(raw_base, adapter).to(args.device).eval()
        return wrapped, wrapped.get_base_model(), wrapped.adapter_load_diagnostics

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
        position_ids, _ = base.get_rope_index(
            input_ids,
            grid,
            None,
            attention_mask=attention,
        )
        return {
            "attention": attention,
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
        # Match ColQwen2_5.forward exactly: projection and normalization happen
        # in the model dtype before the result is promoted for diagnostics.
        projected = base.custom_text_proj(output.last_hidden_state)
        projected = (projected / projected.norm(dim=-1, keepdim=True)).float()
        return [
            projected[index].index_select(0, valid)
            for index, valid in enumerate(data["valid"])
        ]

    def prefix_snapshot(base: Any) -> tuple[dict[str, Any], Any]:
        batch = processor.process_images(first_images).to(args.device)
        data = prepare(base, batch)
        snapshot = {
            "attention": data["attention"].detach().cpu(),
            "embeds": data["embeds"].detach().cpu(),
            "position_ids": data["position_ids"].detach().cpu(),
            "cache_position": data["cache_position"].detach().cpu(),
            "valid": [value.detach().cpu() for value in data["valid"]],
            "visual": [value.detach().cpu() for value in data["visual"]],
        }
        return snapshot, batch

    model_started = time.perf_counter()
    old_model, old_base, old_diagnostics = load_model(args.v01_adapter)
    with torch.inference_mode():
        old_prefix, _ = prefix_snapshot(old_base)
    del old_model, old_base
    gc.collect()
    torch.cuda.empty_cache()

    model, base, target_diagnostics = load_model(args.v02_adapter)
    with torch.inference_mode():
        new_prefix, first_batch = prefix_snapshot(base)
    model_seconds = time.perf_counter() - model_started
    prefix_fields = {}
    for key in ("attention", "embeds", "position_ids", "cache_position"):
        prefix_fields[key] = bool(torch.equal(old_prefix[key], new_prefix[key]))
    for key in ("valid", "visual"):
        prefix_fields[key] = all(
            torch.equal(left, right)
            for left, right in zip(old_prefix[key], new_prefix[key], strict=True)
        )
    if not all(prefix_fields.values()):
        raise RuntimeError(f"dependency canary failed: {prefix_fields}")

    with torch.inference_mode():
        official = model(**first_batch)
        lowered = language(base, prepare(base, first_batch))
    lowering_errors = []
    for index, endpoint in enumerate(lowered):
        mask = first_batch["attention_mask"][index].bool()
        reference = official[index][mask].float()
        lowering_errors.append(float((endpoint.float() - reference).abs().max().item()))
    lowering_max_error = max(lowering_errors)
    if lowering_max_error > 1e-6:
        raise RuntimeError(f"custom lowering mismatch: {lowering_max_error}")
    del official, lowered, old_prefix, new_prefix, first_batch
    torch.cuda.empty_cache()

    raw_shards = args.output_root / "raw-shards"
    ir_shards = args.output_root / "ir" / "batches"
    replay_shards = args.output_root / "replay-shards"
    raw_light_shards = {
        ratio: args.output_root / f"raw-light-{int(ratio * 100):02d}-shards"
        for ratio in (0.10, 0.05)
    }
    replay_light_shards = {
        ratio: args.output_root / f"replay-light-{int(ratio * 100):02d}-shards"
        for ratio in (0.10, 0.05)
    }
    for path in (
        raw_shards,
        ir_shards,
        replay_shards,
        *raw_light_shards.values(),
        *replay_light_shards.values(),
    ):
        path.mkdir(parents=True)

    mean = pca["mean"].to(args.device).float()
    basis = pca["basis"][:, :256].to(args.device).float()

    def make_ir(data: dict[str, Any], batch: Any) -> dict[str, Any]:
        records = []
        for row, visual in enumerate(data["visual"]):
            visual_values = data["embeds"][row].index_select(0, visual)
            coefficients = (visual_values.float() - mean) @ basis
            scale = coefficients.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
            quantized = torch.round(coefficients / scale).clamp(-127, 127).to(torch.int8)
            records.append({
                "quantized": quantized.cpu(),
                "scale": scale.to(torch.bfloat16).cpu(),
            })
        return {
            "input_ids": batch["input_ids"].detach().cpu(),
            "attention": data["attention"].detach().cpu(),
            "image_grid_thw": batch["image_grid_thw"].detach().cpu(),
            "records": records,
        }

    raw_compute_seconds: list[float] = []
    raw_full_d2h_seconds: list[float] = []
    raw_light_seconds: dict[float, list[float]] = {0.10: [], 0.05: []}
    ir_batch_seconds: list[float] = []
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, n, args.image_batch_size)):
            stop = min(start + args.image_batch_size, n)
            raw_path = raw_shards / f"batch-{batch_index:04d}.pt"
            ir_path = ir_shards / f"batch-{batch_index:04d}.pt"
            torch.cuda.synchronize()
            raw_started = time.perf_counter()
            images = [dataset[index]["image"].convert("RGB") for index in range(start, stop)]
            batch = processor.process_images(images).to(args.device)
            data = prepare(base, batch)
            gpu_endpoints = language(base, data)
            torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - raw_started
            d2h_started = time.perf_counter()
            endpoints = [value.to(torch.bfloat16).cpu() for value in gpu_endpoints]
            torch.cuda.synchronize()
            full_d2h_seconds = time.perf_counter() - d2h_started
            torch.save(
                {
                    "start": start,
                    "stop": stop,
                    "compute_seconds": compute_seconds,
                    "full_d2h_seconds": full_d2h_seconds,
                    "pages": endpoints,
                },
                raw_path,
            )
            raw_compute_seconds.append(compute_seconds)
            raw_full_d2h_seconds.append(full_d2h_seconds)
            for ratio in (0.10, 0.05):
                torch.cuda.synchronize()
                light_started = time.perf_counter()
                compact = [merge_tome(page, ratio, torch, args.device) for page in gpu_endpoints]
                torch.cuda.synchronize()
                light_seconds = time.perf_counter() - light_started
                torch.save(
                    {"start": start, "stop": stop, "light_seconds": light_seconds, "pages": compact},
                    raw_light_shards[ratio] / f"batch-{batch_index:04d}.pt",
                )
                raw_light_seconds[ratio].append(light_seconds)
                del compact

            torch.cuda.synchronize()
            ir_started = time.perf_counter()
            payload = make_ir(data, batch)
            torch.save(payload, ir_path)
            torch.cuda.synchronize()
            ir_seconds = time.perf_counter() - ir_started
            ir_batch_seconds.append(ir_seconds)
            if batch_index % 10 == 0:
                print(json.dumps({"stage": "raw", "pages": stop, "scope": n}), flush=True)
            del images, batch, data, gpu_endpoints, endpoints, payload

    def reconstruct(payload: dict[str, Any]) -> dict[str, Any]:
        records = payload["records"]
        input_ids = payload["input_ids"].to(args.device)
        attention = payload["attention"].to(args.device)
        grid = payload["image_grid_thw"].to(args.device)
        embeddings = base.get_input_embeddings()(input_ids)
        valid = []
        for row, record in enumerate(records):
            visual = torch.where(
                (input_ids[row] == int(base.config.image_token_id)) & attention[row].bool()
            )[0]
            coefficients = (
                record["quantized"].to(args.device).float()
                * record["scale"].to(args.device).float()
            )
            visual_values = (coefficients @ basis.T + mean).to(torch.bfloat16)
            embeddings[row, visual] = visual_values
            valid.append(torch.where(attention[row].bool())[0])
        position_ids, _ = base.get_rope_index(
            input_ids,
            grid,
            None,
            attention_mask=attention,
        )
        return {
            "attention": attention,
            "position_ids": position_ids,
            "cache_position": torch.arange(embeddings.shape[1], device=args.device),
            "embeds": embeddings,
            "valid": valid,
        }

    replay_compute_seconds: list[float] = []
    replay_full_d2h_seconds: list[float] = []
    replay_light_seconds: dict[float, list[float]] = {0.10: [], 0.05: []}
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, n, args.image_batch_size)):
            stop = min(start + args.image_batch_size, n)
            torch.cuda.synchronize()
            replay_started = time.perf_counter()
            payload = torch.load(ir_shards / f"batch-{batch_index:04d}.pt", weights_only=True)
            data = reconstruct(payload)
            gpu_endpoints = language(base, data)
            torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - replay_started
            d2h_started = time.perf_counter()
            endpoints = [value.to(torch.bfloat16).cpu() for value in gpu_endpoints]
            torch.cuda.synchronize()
            full_d2h_seconds = time.perf_counter() - d2h_started
            torch.save(
                {
                    "start": start,
                    "stop": stop,
                    "compute_seconds": compute_seconds,
                    "full_d2h_seconds": full_d2h_seconds,
                    "pages": endpoints,
                },
                replay_shards / f"batch-{batch_index:04d}.pt",
            )
            replay_compute_seconds.append(compute_seconds)
            replay_full_d2h_seconds.append(full_d2h_seconds)
            for ratio in (0.10, 0.05):
                torch.cuda.synchronize()
                light_started = time.perf_counter()
                compact = [merge_tome(page, ratio, torch, args.device) for page in gpu_endpoints]
                torch.cuda.synchronize()
                light_seconds = time.perf_counter() - light_started
                torch.save(
                    {"start": start, "stop": stop, "light_seconds": light_seconds, "pages": compact},
                    replay_light_shards[ratio] / f"batch-{batch_index:04d}.pt",
                )
                replay_light_seconds[ratio].append(light_seconds)
                del compact
            if batch_index % 10 == 0:
                print(json.dumps({"stage": "replay", "pages": stop, "scope": n}), flush=True)
            del payload, data, gpu_endpoints, endpoints

    query_embeddings, query_tokens = [], []
    query_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, n, 32):
            batch = processor.process_queries(queries[start : start + 32]).to(args.device)
            encoded = model(**batch)
            masks = batch["attention_mask"].bool()
            for values, mask in zip(encoded, masks, strict=True):
                endpoint = values[mask].to(torch.bfloat16).cpu()
                query_embeddings.append(endpoint)
                query_tokens.append(int(endpoint.shape[0]))
    torch.cuda.synchronize()
    query_seconds = time.perf_counter() - query_started

    def collect(root: Path, timing_keys: tuple[str, ...]) -> tuple[list[Any], dict[str, list[float]]]:
        pages = []
        by_timing = {key: [] for key in timing_keys}
        for path in sorted(root.glob("batch-*.pt")):
            payload = torch.load(path, weights_only=True)
            pages.extend(payload["pages"])
            for key in timing_keys:
                by_timing[key].append(float(payload[key]))
        if len(pages) != n:
            raise RuntimeError(f"incomplete page shards under {root}: {len(pages)} != {n}")
        return pages, by_timing

    raw_pages, raw_timings = collect(raw_shards, ("compute_seconds", "full_d2h_seconds"))
    replay_pages, replay_timings = collect(replay_shards, ("compute_seconds", "full_d2h_seconds"))
    raw_compute_seconds = raw_timings["compute_seconds"]
    raw_full_d2h_seconds = raw_timings["full_d2h_seconds"]
    replay_compute_seconds = replay_timings["compute_seconds"]
    replay_full_d2h_seconds = replay_timings["full_d2h_seconds"]
    raw_tokens = [int(value.shape[0]) for value in raw_pages]
    replay_tokens = [int(value.shape[0]) for value in replay_pages]
    if raw_tokens != replay_tokens:
        raise RuntimeError("raw and replay terminal shapes differ")

    endpoint_cosines, endpoint_mae = [], []
    for raw, replay in zip(raw_pages, replay_pages, strict=True):
        endpoint_cosines.append(float(functional.cosine_similarity(raw.float(), replay.float(), dim=-1).mean()))
        endpoint_mae.append(float((raw.float() - replay.float()).abs().mean()))

    meta = {
        "slice": protocol["benchmark"]["slice"],
        "repo": protocol["benchmark"]["repository"],
        "n": n,
        "full_dataset_n": full_size,
        "gold": list(range(n)),
        "nq_tok": query_tokens,
        "np_tok": raw_tokens,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
    }

    cache_paths = {}
    terminal_write = {}
    for name, pages in (("raw", raw_pages), ("replay", replay_pages)):
        cache = args.output_root / f"{name}-cache" / protocol["benchmark"]["slice"]
        cache.mkdir(parents=True)
        torch.save(query_embeddings, cache / "qs.pt")
        (cache / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        write_started = time.perf_counter()
        torch.save(pages, cache / "ps.pt")
        terminal_write[name] = time.perf_counter() - write_started
        cache_paths[name] = cache

    def actual_index_ndcg_at_5(pages: list[Any]) -> float:
        """Score the actual serialized index under the pinned official split."""
        by_document: dict[int, list[int]] = {}
        for query_index, document in enumerate(meta["gold"]):
            by_document.setdefault(int(document), []).append(query_index)
        documents = sorted(by_document)
        permutation = np.random.RandomState(0).permutation(len(documents))
        n_test = max(1, int(round(0.3 * len(documents))))
        test_documents = [documents[int(index)] for index in permutation[:n_test]]
        column = {document: index for index, document in enumerate(test_documents)}
        evaluation_queries = [
            query_index
            for document in test_documents
            for query_index in by_document[document]
        ]
        gold_columns = torch.tensor(
            [column[int(meta["gold"][index])] for index in evaluation_queries]
        )
        selected_queries = [query_embeddings[index] for index in evaluation_queries]
        max_length = max(int(query.shape[0]) for query in selected_queries)
        channels = int(selected_queries[0].shape[1])
        query_pad = torch.zeros(
            len(selected_queries),
            max_length,
            channels,
            dtype=torch.bfloat16,
            device=args.device,
        )
        query_mask = torch.zeros(
            len(selected_queries), max_length, device=args.device
        )
        for row, query in enumerate(selected_queries):
            query_pad[row, : query.shape[0]] = query.to(args.device).to(torch.bfloat16)
            query_mask[row, : query.shape[0]] = 1.0
        scores = torch.empty(
            len(selected_queries), len(test_documents), device=args.device
        )
        with torch.inference_mode():
            for document in test_documents:
                page = pages[document].to(args.device).to(torch.bfloat16)
                token_scores = (query_pad @ page.T).max(dim=2)[0]
                scores[:, column[document]] = (
                    token_scores.float() * query_mask
                ).sum(dim=1)
        order = torch.argsort(scores.cpu(), dim=1, descending=True)
        values = []
        for row in range(len(evaluation_queries)):
            position = int(
                (order[row] == gold_columns[row]).nonzero(as_tuple=True)[0].item()
            ) + 1
            values.append(1.0 / math.log2(position + 1) if position <= 5 else 0.0)
        return float(np.mean(values))

    light_results = {}
    for source_name, roots in (("raw", raw_light_shards), ("replay", replay_light_shards)):
        light_results[source_name] = {}
        for ratio in (0.10, 0.05):
            compact, compact_timings = collect(roots[ratio], ("light_seconds",))
            merge_seconds = sum(compact_timings["light_seconds"])
            output = args.output_root / "indexes" / f"{source_name}-light-{int(ratio * 100):02d}.pt"
            output.parent.mkdir(parents=True, exist_ok=True)
            write_started = time.perf_counter()
            torch.save(compact, output)
            write_seconds = time.perf_counter() - write_started
            light_results[source_name][str(ratio)] = {
                "merge_seconds": merge_seconds,
                "write_seconds": write_seconds,
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "vectors": sum(int(value.shape[0]) for value in compact),
                "per_document_vectors": [int(value.shape[0]) for value in compact],
                "actual_index_ndcg_at_5": actual_index_ndcg_at_5(compact),
            }
            del compact

    raw_full_build = sum(raw_compute_seconds) + sum(raw_full_d2h_seconds) + terminal_write["raw"]
    replay_full_build = sum(replay_compute_seconds) + sum(replay_full_d2h_seconds) + terminal_write["replay"]
    ir_document_bytes = sum(path.stat().st_size for path in ir_shards.glob("batch-*.pt"))
    physical_ir_bytes = ir_document_bytes + args.pca256.stat().st_size
    raw_terminal_bytes = (cache_paths["raw"] / "ps.pt").stat().st_size
    cardinality_equal = all(
        light_results["raw"][str(ratio)]["per_document_vectors"]
        == light_results["replay"][str(ratio)]["per_document_vectors"]
        for ratio in (0.10, 0.05)
    )
    if not cardinality_equal:
        raise RuntimeError("raw and replay Light cardinalities differ")

    method_costs = {
        "raw_full": raw_full_build,
        "reprforge_full": replay_full_build,
    }
    for ratio, label in ((0.10, "light10"), (0.05, "light05")):
        method_costs[f"raw_{label}"] = (
            sum(raw_compute_seconds)
            + light_results["raw"][str(ratio)]["merge_seconds"]
            + light_results["raw"][str(ratio)]["write_seconds"]
        )
        method_costs[f"reprforge_{label}"] = (
            sum(replay_compute_seconds)
            + light_results["replay"][str(ratio)]["merge_seconds"]
            + light_results["replay"][str(ratio)]["write_seconds"]
        )

    result = {
        "protocol": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "scope": {"complete": not bool(args.limit), "pages": n, "queries": n},
        "environment": {
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "image_batch_size": args.image_batch_size,
        },
        "provenance": {
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "dataset_parquet_sha256": observed_dataset_hash,
            "v01_adapter_sha256": sha256(args.v01_adapter / "adapter_model.safetensors"),
            "v02_adapter_sha256": sha256(args.v02_adapter / "adapter_model.safetensors"),
            "base_revision": transition["base"],
            "pca256_sha256": sha256(args.pca256),
            "protocol_status": protocol["status"],
        },
        "gates_before_quality": {
            "dependency_fields": prefix_fields,
            "dependency_pass": all(prefix_fields.values()),
            "lowering_max_absolute_error": lowering_max_error,
            "lowering_pass": lowering_max_error <= 1e-6,
            "cardinality_pass": cardinality_equal,
            "storage_pass": physical_ir_bytes <= 1.10 * raw_terminal_bytes,
        },
        "adapter_load": {"v01": old_diagnostics, "v02": target_diagnostics},
        "timing_seconds": {
            "dataset": dataset_seconds,
            "model_and_dependency_canary": model_seconds,
            "query_encoding_excluded_common": query_seconds,
            "raw_compute_batches": summarize(raw_compute_seconds),
            "raw_full_d2h_batches": summarize(raw_full_d2h_seconds),
            "replay_compute_batches": summarize(replay_compute_seconds),
            "replay_full_d2h_batches": summarize(replay_full_d2h_seconds),
            "raw_light10_merge_d2h_batches": summarize(raw_light_seconds[0.10]),
            "raw_light05_merge_d2h_batches": summarize(raw_light_seconds[0.05]),
            "replay_light10_merge_d2h_batches": summarize(replay_light_seconds[0.10]),
            "replay_light05_merge_d2h_batches": summarize(replay_light_seconds[0.05]),
            "ir_materialization_batches": summarize(ir_batch_seconds),
            "terminal_write": terminal_write,
            "method_complete_target_build": method_costs,
            "reprforge_full_saving": 1.0 - method_costs["reprforge_full"] / method_costs["raw_full"],
            "reprforge_light10_saving": 1.0 - method_costs["reprforge_light10"] / method_costs["raw_light10"],
            "reprforge_light05_saving": 1.0 - method_costs["reprforge_light05"] / method_costs["raw_light05"],
        },
        "storage": {
            "raw_full_terminal_bytes": raw_terminal_bytes,
            "replay_full_terminal_bytes": (cache_paths["replay"] / "ps.pt").stat().st_size,
            "ir_shards_bytes": ir_document_bytes,
            "shared_pca_bytes": args.pca256.stat().st_size,
            "physical_ir_bytes": physical_ir_bytes,
            "ir_to_raw_terminal_ratio": physical_ir_bytes / raw_terminal_bytes,
            "light": light_results,
        },
        "representation": {
            "raw_mean_vectors_per_page": statistics.fmean(raw_tokens),
            "raw_total_vectors": sum(raw_tokens),
            "mean_replay_token_cosine": statistics.fmean(endpoint_cosines),
            "mean_replay_endpoint_mae": statistics.fmean(endpoint_mae),
        },
        "quality_pending": "Run the pinned official baselines_memory.py separately on raw-cache and replay-cache.",
        "claim_boundary": protocol["claim_boundary"],
    }
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "scope": result["scope"],
        "gates_before_quality": result["gates_before_quality"],
        "method_complete_target_build": method_costs,
        "reprforge_light10_saving": result["timing_seconds"]["reprforge_light10_saving"],
        "ir_to_raw_terminal_ratio": result["storage"]["ir_to_raw_terminal_ratio"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
