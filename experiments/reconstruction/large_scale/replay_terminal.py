#!/usr/bin/env python3
"""Replay a target retrieval adapter from a dependency-valid post-vision IR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from ir_codecs import decode
from run_smoke import load_compatible_adapter, load_projection, sha256
from streaming_maxsim import streaming_maxsim


KS = (1, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-shard-prefix", type=Path, required=True)
    parser.add_argument("--pca256", type=Path, required=True)
    parser.add_argument("--ir-root", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-documents", type=Path, required=True)
    parser.add_argument("--terminal-root", type=Path, required=True)
    parser.add_argument("--document-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch", type=int, default=32)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotations(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ordered_documents(rows: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    result = sorted(enumerate(rows), key=lambda value: int(value[1]["page_indices"][0]))
    expected = 0
    for _, row in result:
        start, end = (int(value) for value in row["page_indices"])
        if start != expected or end < start:
            raise ValueError(f"page ranges are not an exact ordered partition at {start}")
        expected = end + 1
    return result


def reconstruct(template: dict[str, Any], coefficients: Any, device: str, torch: Any, pca: dict[str, Any]) -> dict[str, Any]:
    visual = template["visual"].to(device)
    nonvisual = template["nonvisual"].to(device)
    values = decode({"coefficients": coefficients}, "pca256_bf16", device, torch, pca)
    embeds = torch.zeros(
        (1, template["sequence_length"], template["channels"]),
        dtype=torch.bfloat16,
        device=device,
    )
    embeds[:, nonvisual] = template["nonvisual_embeds"].to(device)
    embeds[:, visual] = values
    return {
        "attention": template["attention"].to(device),
        "position_ids": template["position_ids"].to(device),
        "cache_position": template["cache_position"].to(device),
        "valid": template["valid"].to(device),
        "embeds": embeds,
    }


def official_metrics(ranking: np.ndarray, gold: list[list[int]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for order, relevant in zip(ranking, gold, strict=True):
        relevant_set = set(relevant)
        row: dict[str, float] = {}
        for k in KS:
            retrieved = order[:k].tolist()
            hits = len(set(retrieved) & relevant_set)
            ideal = sum(1.0 / math.log2(index + 2.0) for index in range(min(len(retrieved), len(relevant))))
            dcg = sum(
                1.0 / math.log2(index + 2.0)
                for index, item in enumerate(retrieved)
                if item in relevant_set
            )
            row[f"recall_at_{k}"] = hits / len(relevant_set) if relevant_set else 0.0
            row[f"ndcg_at_{k}"] = dcg / ideal if ideal else 0.0
        rows.append(row)
    return rows


def rank(scores: np.ndarray) -> np.ndarray:
    tie = np.arange(scores.shape[1])
    return np.asarray([np.lexsort((tie, -row)) for row in scores], dtype=np.int64)


def bootstrap(values: list[float], seed: int, draws: int = 20_000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        samples[start:stop] = array[indices].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    for root in (args.terminal_root, args.document_output):
        if root.exists() and not args.resume:
            raise FileExistsError(f"target exists without --resume: {root}")
        root.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    protocol = json.loads(args.protocol.read_text())
    benchmark = protocol["benchmark"]
    transition = protocol["transition"]
    if file_sha256(args.annotations) != benchmark["annotations_sha256"]:
        raise RuntimeError("annotations changed")
    if file_sha256(args.reference_result) != transition["reference_result_sha256"]:
        raise RuntimeError("reference result changed")
    if sha256(args.adapter / "adapter_model.safetensors") != transition["adapter_sha256"]:
        raise RuntimeError("adapter changed")
    if sha256(args.base_shard_prefix) != transition["projection_sha256"]:
        raise RuntimeError("projection changed")
    if sha256(args.pca256) != transition["pca256_sha256"]:
        raise RuntimeError("PCA basis changed")

    rows = annotations(args.annotations)
    documents = ordered_documents(rows)
    if len(documents) != int(benchmark["documents"]):
        raise RuntimeError("document count changed")
    if args.max_documents is not None:
        if args.max_documents <= 0:
            raise ValueError("max-documents must be positive")
        documents = documents[:args.max_documents]

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    base_model = ColQwen2_5.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    projection = load_projection(args.base_shard_prefix, torch)
    with torch.no_grad():
        base_model.custom_text_proj.weight.copy_(projection["custom_text_proj.weight"].to(base_model.custom_text_proj.weight))
        base_model.custom_text_proj.bias.copy_(projection["custom_text_proj.bias"].to(base_model.custom_text_proj.bias))
    model = load_compatible_adapter(base_model, args.adapter).to(args.device).eval()
    base = model.get_base_model()
    processor = ColQwen2_5_Processor.from_pretrained(
        args.base_model,
        local_files_only=True,
        max_num_visual_tokens=768,
    )
    pca = torch.load(args.pca256, weights_only=True)
    template_cache: dict[str, dict[str, Any]] = {}

    def language(data: dict[str, Any]) -> Any:
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
        projected = F.normalize(base.custom_text_proj(output.last_hidden_state).float(), dim=-1)[0]
        return projected.index_select(0, data["valid"])

    transition_started = time.perf_counter()
    with torch.inference_mode():
        for completed, (document_index, document) in enumerate(documents, start=1):
            terminal_path = args.terminal_root / f"doc-{document_index:03d}.pt"
            document_path = args.document_output / f"doc-{document_index:03d}.json"
            if terminal_path.exists() and document_path.exists():
                if not args.resume:
                    raise FileExistsError(terminal_path)
                print(json.dumps({"resumed": document_index, "completed": completed}), flush=True)
                continue
            questions = list(document["questions"])
            query_vectors = []
            query_started = time.perf_counter()
            for start in range(0, len(questions), args.query_batch):
                batch = processor.process_queries([str(row["Q"]) for row in questions[start:start + args.query_batch]]).to(args.device)
                outputs = model(**batch)
                query_vectors.extend(
                    value[mask.bool()].detach().cpu().float()
                    for value, mask in zip(outputs, batch["attention_mask"], strict=True)
                )
            torch.cuda.synchronize()
            query_seconds = time.perf_counter() - query_started

            replay_started = time.perf_counter()
            shard = torch.load(args.ir_root / "documents" / f"doc-{document_index:03d}.pt", weights_only=True)
            pages = []
            for record in shard["records"]:
                signature = str(record["template"])
                if signature not in template_cache:
                    template_cache[signature] = torch.load(args.ir_root / "templates" / f"{signature}.pt", weights_only=True)
                data = reconstruct(template_cache[signature], record["coefficients"], args.device, torch, pca)
                pages.append(language(data).detach().cpu())
            torch.cuda.synchronize()
            replay_seconds = time.perf_counter() - replay_started

            write_started = time.perf_counter()
            temporary = terminal_path.with_suffix(".pt.partial")
            torch.save(pages, temporary)
            os.replace(temporary, terminal_path)
            terminal_sha256 = file_sha256(terminal_path)
            write_hash_seconds = time.perf_counter() - write_started
            validation_started = time.perf_counter()
            scores = streaming_maxsim(query_vectors, pages, query_chunk=32, document_chunk=16)
            gold = [[int(value) for value in question["page_id"]] for question in questions]
            metrics = official_metrics(rank(scores), gold)
            reference = json.loads((args.reference_documents / f"doc-{document_index:03d}.json").read_text())
            if len(reference["query_rows"]) != len(metrics):
                raise RuntimeError("reference query scope changed")
            query_rows = []
            for local_query, (metric, reference_query) in enumerate(zip(metrics, reference["query_rows"], strict=True)):
                query_rows.append({
                    "query_id": f"{document_index}:{local_query}",
                    "domain": str(document.get("domain") or ""),
                    "full": reference_query["full"],
                    "pca256": metric,
                })
            validation_seconds = time.perf_counter() - validation_started
            document_result = {
                "document_index": document_index,
                "pages": len(pages),
                "queries": len(query_rows),
                "query_encoding_seconds": query_seconds,
                "replay_seconds": replay_seconds,
                "write_hash_seconds": write_hash_seconds,
                "quality_validation_seconds": validation_seconds,
                "terminal_bytes": terminal_path.stat().st_size,
                "terminal_sha256": terminal_sha256,
                "query_rows": query_rows,
            }
            document_path.write_text(json.dumps(document_result, indent=2) + "\n")
            print(json.dumps({
                "completed": completed,
                "documents": len(documents),
                "document_index": document_index,
                "pages": len(pages),
                "replay_seconds": replay_seconds,
            }), flush=True)

    document_results = [json.loads((args.document_output / f"doc-{index:03d}.json").read_text()) for index, _ in documents]
    query_rows = [row for document in document_results for row in document["query_rows"]]
    complete = args.max_documents is None
    if complete and (sum(row["pages"] for row in document_results) != int(benchmark["pages"]) or len(query_rows) != int(benchmark["queries"])):
        raise RuntimeError("complete scope changed")
    quality = {}
    for method in ("full", "pca256"):
        quality[method] = {
            metric: statistics.fmean(row[method][metric] for row in query_rows)
            for metric in (f"{kind}_at_{k}" for kind in ("recall", "ndcg") for k in KS)
        }
    differences = [row["pca256"]["recall_at_5"] - row["full"]["recall_at_5"] for row in query_rows]
    recall_interval = bootstrap(differences, 20260822)
    query_seconds = sum(row["query_encoding_seconds"] for row in document_results)
    replay_seconds = sum(row["replay_seconds"] for row in document_results)
    write_hash_seconds = sum(row["write_hash_seconds"] for row in document_results)
    quality_validation_seconds = sum(row["quality_validation_seconds"] for row in document_results)
    measured_transition_seconds = time.perf_counter() - transition_started
    raw_seconds = float(protocol["frozen_references"]["raw_page_rebuild_seconds"])
    warm_build_seconds = replay_seconds + write_hash_seconds
    cold_build_seconds = raw_seconds + write_hash_seconds
    gates = {
        "complete_scope": complete and len(document_results) == int(benchmark["documents"]),
        "recall_at_5_noninferiority": recall_interval[0] >= float(protocol["primary_gates"]["paired_recall_at_5_noninferiority_lower_bound"]),
        "ndcg_at_10": quality["pca256"]["ndcg_at_10"] - quality["full"]["ndcg_at_10"] >= -float(protocol["primary_gates"]["ndcg_at_10_maximum_drop"]),
        "replay_payoff": 1.0 - warm_build_seconds / cold_build_seconds >= float(protocol["primary_gates"]["minimum_end_to_end_transition_saving"]),
        "quality_reproducible": complete and abs(quality["pca256"]["recall_at_5"] - float(protocol["frozen_references"]["previous_pca_recall_at_5"])) <= 1e-12,
    }
    result = {
        "protocol": protocol["protocol_id"],
        "protocol_sha256": file_sha256(args.protocol),
        "scope": {
            "documents": len(document_results),
            "pages": sum(row["pages"] for row in document_results),
            "queries": len(query_rows),
        },
        "model": {
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "adapter_load": model.adapter_load_diagnostics,
        },
        "quality": quality,
        "paired_pca_minus_full_recall_at_5": {
            "mean": statistics.fmean(differences),
            "bootstrap_95": recall_interval,
        },
        "timing": {
            "query_encoding_seconds": query_seconds,
            "suffix_replay_seconds": replay_seconds,
            "terminal_write_and_hash_seconds": write_hash_seconds,
            "quality_validation_seconds": quality_validation_seconds,
            "replay_materialize_and_validate_seconds": measured_transition_seconds,
            "raw_page_rebuild_seconds": raw_seconds,
            "warm_build_seconds_before_serving_index": warm_build_seconds,
            "cold_build_seconds_before_serving_index": cold_build_seconds,
            "saving_before_serving_index": 1.0 - warm_build_seconds / cold_build_seconds,
        },
        "storage": {
            "terminal_bytes": sum(row["terminal_bytes"] for row in document_results),
            "terminal_shards": len(document_results),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
