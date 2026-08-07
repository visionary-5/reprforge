#!/usr/bin/env python3
"""Run a resumable raw-page relevance verifier over frozen candidate lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.bm25 import build_index, score_queries
from reprforge.dvi_page_verifier import (
    aggregate_query_metrics,
    deterministic_query_sample,
    ranking_metrics,
    rerank_with_scores,
    roc_auc,
    union_preserving_order,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_visual_ranking(path: Path, top_k: int) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            query_id, doc_id, _ = line.rstrip("\n").split("\t")
            values = rankings.setdefault(query_id, [])
            if len(values) < top_k:
                values.append(doc_id)
    return rankings


def _bm25_rankings(
    corpus_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]], top_k: int
) -> tuple[dict[str, list[str]], dict[str, float]]:
    doc_ids = [str(row["docid"]) for row in corpus_rows]
    began = time.perf_counter()
    state, posting_bytes, vocabulary_bytes = build_index(
        [str(row.get("text") or "") for row in corpus_rows]
    )
    build_seconds = time.perf_counter() - began
    began = time.perf_counter()
    matrix = score_queries(
        state, [str(row["query"]) for row in query_rows], k1=1.2, b=0.75
    )
    score_seconds = time.perf_counter() - began
    rankings = {}
    for query_row, scores in zip(query_rows, matrix, strict=True):
        order = np.lexsort((np.asarray(doc_ids), -scores))[:top_k]
        rankings[str(query_row["query_id"])] = [doc_ids[int(value)] for value in order]
    return rankings, {
        "build_seconds": build_seconds,
        "score_all_queries_seconds": score_seconds,
        "logical_index_bytes": int(posting_bytes.sum()) + int(vocabulary_bytes),
    }


def _load_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    for row in _read_jsonl(path):
        qrels.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = float(
            row["relevance"]
        )
    return qrels


def _load_existing_scores(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not path.exists():
        return {}
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[(str(row["query_id"]), str(row["doc_id"]))] = row
    return output


def _score_pairs(
    *,
    pairs: list[tuple[str, str]],
    queries: dict[str, str],
    assets: Path,
    model_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    existing = _load_existing_scores(output_path)
    pending = [pair for pair in pairs if pair not in existing]
    model_config = config["model"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.image_processor.min_pixels = int(model_config["min_pixels"])
    processor.image_processor.max_pixels = int(model_config["max_pixels"])
    tokenizer = processor.tokenizer
    yes_ids = tokenizer.encode("YES", add_special_tokens=False)
    no_ids = tokenizer.encode("NO", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(f"YES/NO must each be one token: {yes_ids}, {no_ids}")
    load_started = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    load_seconds = time.perf_counter() - load_started
    batch_size = int(model_config["batch_size"])
    batch_latencies = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            images = [Image.open(assets / f"{doc_id}.png").convert("RGB") for _, doc_id in batch]
            prompts = []
            for query_id, _ in batch:
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": config["prompt"].format(query=queries[query_id])},
                    ],
                }]
                prompts.append(
                    processor.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True
                    )
                )
            inputs = processor(
                text=prompts, images=images, padding=True, return_tensors="pt"
            ).to("cuda")
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.inference_mode():
                logits = model(**inputs).logits
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            batch_latencies.append(elapsed)
            token_positions = torch.arange(
                inputs.attention_mask.shape[1], device=inputs.attention_mask.device
            ).unsqueeze(0)
            positions = token_positions.masked_fill(
                inputs.attention_mask == 0, -1
            ).max(dim=1).values
            rows = torch.arange(len(batch), device=logits.device)
            pair_scores = (
                logits[rows, positions, yes_ids[0]] - logits[rows, positions, no_ids[0]]
            ).float().cpu().numpy()
            for (query_id, doc_id), score in zip(batch, pair_scores, strict=True):
                row = {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "yes_minus_no": float(score),
                    "batch_seconds": elapsed,
                    "batch_size": len(batch),
                }
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                existing[(query_id, doc_id)] = row
            del logits, inputs
    page_seconds = [
        row["batch_seconds"] / row["batch_size"] for row in existing.values()
        if (row["query_id"], row["doc_id"]) in set(pairs)
    ]
    return existing, {
        "model_load_seconds": load_seconds,
        "new_pairs": len(pending),
        "total_pairs": len(pairs),
        "page_seconds_mean": float(np.mean(page_seconds)),
        "page_seconds_p50": float(np.percentile(page_seconds, 50)),
        "page_seconds_p95": float(np.percentile(page_seconds, 95)),
        "max_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--visual-ranking", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--query-limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus_rows = _read_jsonl(args.dataset_root / "corpus.jsonl")
    query_rows = _read_jsonl(args.dataset_root / "queries.jsonl")
    qrels = _load_qrels(args.dataset_root / "qrels.jsonl")
    query_text = {str(row["query_id"]): str(row["query"]) for row in query_rows}
    bm25, bm25_cost = _bm25_rankings(corpus_rows, query_rows, 20)
    visual = _load_visual_ranking(args.visual_ranking, 20)
    all_query_ids = [str(row["query_id"]) for row in query_rows]
    limit = args.query_limit or int(config["query_selection"]["limit"])
    main_queries = deterministic_query_sample(
        all_query_ids, limit=limit, seed=int(config["query_selection"]["seed"])
    )
    stress_queries = [
        query_id
        for query_id in all_query_ids
        if not (set(bm25[query_id][:20]) & set(qrels[query_id]))
        and (set(visual[query_id][:20]) & set(qrels[query_id]))
    ]
    evaluated_queries = union_preserving_order(main_queries, stress_queries)
    routes = {}
    for query_id in evaluated_queries:
        routes[query_id] = {
            "bm25_20": bm25[query_id][:20],
            "visual_20": visual[query_id][:20],
            "hybrid_10_10": union_preserving_order(
                bm25[query_id][:10], visual[query_id][:10]
            ),
        }
    pairs = sorted(
        {
            (query_id, doc_id)
            for query_id in evaluated_queries
            for candidates in routes[query_id].values()
            for doc_id in candidates
        }
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    score_path = args.output_root / "pair-scores.jsonl"
    scores, runtime = _score_pairs(
        pairs=pairs,
        queries=query_text,
        assets=args.dataset_root / "assets",
        model_path=args.model,
        output_path=score_path,
        config=config,
    )
    cohorts = {"main": main_queries, "escape_stress": stress_queries}
    cohort_results = {}
    for cohort_name, cohort_queries in cohorts.items():
        route_results = {}
        for route_name in ("bm25_20", "visual_20", "hybrid_10_10"):
            before_rows, after_rows = [], []
            for query_id in cohort_queries:
                candidates = routes[query_id][route_name]
                pair_scores = {
                    doc_id: scores[(query_id, doc_id)]["yes_minus_no"]
                    for doc_id in candidates
                }
                before_rows.append(ranking_metrics(candidates, qrels[query_id]))
                after_rows.append(
                    ranking_metrics(
                        rerank_with_scores(candidates, pair_scores), qrels[query_id]
                    )
                )
            route_results[route_name] = {
                "before_vlm": aggregate_query_metrics(before_rows) if before_rows else None,
                "after_vlm": aggregate_query_metrics(after_rows) if after_rows else None,
            }
        cohort_results[cohort_name] = {
            "queries": len(cohort_queries),
            "routes": route_results,
        }
    labels, pair_values = [], []
    for query_id, doc_id in pairs:
        labels.append(int(doc_id in qrels[query_id]))
        pair_values.append(scores[(query_id, doc_id)]["yes_minus_no"])
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "dataset": {
            "root": str(args.dataset_root),
            "queries": len(query_rows),
            "pages": len(corpus_rows),
            "input_sha256": {
                "corpus": _sha(args.dataset_root / "corpus.jsonl"),
                "queries": _sha(args.dataset_root / "queries.jsonl"),
                "qrels": _sha(args.dataset_root / "qrels.jsonl"),
                "visual_ranking": _sha(args.visual_ranking),
                "config": _sha(args.config),
            },
        },
        "selection": {
            "main_queries": main_queries,
            "stress_queries": stress_queries,
            "evaluated_queries": len(evaluated_queries),
            "unique_query_page_pairs": len(pairs),
        },
        "bm25_cost": bm25_cost,
        "verifier": {
            **runtime,
            "pair_roc_auc": roc_auc(labels, pair_values),
            "positive_pairs": int(sum(labels)),
            "negative_pairs": int(len(labels) - sum(labels)),
            "score_path": str(score_path),
        },
        "cohorts": cohort_results,
        "warning": "Page-localization proxy only; no answer-generation quality claim.",
    }
    result_path = args.output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verifier": result["verifier"], "cohorts": cohort_results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
