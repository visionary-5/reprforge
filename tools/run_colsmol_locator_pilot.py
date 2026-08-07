#!/usr/bin/env python3
"""Build and evaluate a resumable ColSmol full-corpus locator index."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.dvi_page_verifier import (
    aggregate_query_metrics,
    deterministic_query_sample,
    ranking_metrics,
    union_preserving_order,
)
from tools.run_dvi_page_verifier_pilot import (
    _bm25_rankings,
    _load_qrels,
    _load_visual_ranking,
    _read_jsonl,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and ".cache" not in item.parts and not item.name.startswith("._")
    )


def _encode_documents(
    *, model: Any, processor: Any, rows: list[dict[str, Any]], assets: Path,
    index_root: Path, batch_size: int, shard_pages: int,
) -> tuple[list[Any], float, int]:
    import torch
    from PIL import Image

    index_root.mkdir(parents=True, exist_ok=True)
    total_seconds = 0.0
    all_embeddings = []
    for shard_start in range(0, len(rows), shard_pages):
        shard_rows = rows[shard_start : shard_start + shard_pages]
        shard_path = index_root / f"shard-{shard_start:06d}.pt"
        if shard_path.exists():
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            expected = [str(row["docid"]) for row in shard_rows]
            if payload["doc_ids"] != expected:
                raise ValueError(f"index shard IDs differ: {shard_path}")
            all_embeddings.extend(payload["embeddings"])
            continue
        shard_embeddings = []
        began = time.perf_counter()
        for start in range(0, len(shard_rows), batch_size):
            batch_rows = shard_rows[start : start + batch_size]
            images = []
            for row in batch_rows:
                with Image.open(assets / str(row["image"])) as image:
                    images.append(image.convert("RGB"))
            inputs = processor.process_images(images).to("cuda")
            with torch.inference_mode():
                encoded = model(**inputs)
            masks = inputs["attention_mask"].bool()
            shard_embeddings.extend(
                embedding[mask].to(device="cpu", dtype=torch.bfloat16).contiguous()
                for embedding, mask in zip(encoded, masks, strict=True)
            )
            del inputs, encoded
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        total_seconds += elapsed
        payload = {
            "doc_ids": [str(row["docid"]) for row in shard_rows],
            "embeddings": shard_embeddings,
        }
        torch.save(payload, shard_path)
        all_embeddings.extend(shard_embeddings)
    logical_bytes = sum(value.numel() * value.element_size() for value in all_embeddings)
    return all_embeddings, total_seconds, logical_bytes


def _encode_queries(
    *, model: Any, processor: Any, rows: list[dict[str, Any]], batch_size: int,
    output_path: Path,
) -> tuple[list[Any], float]:
    import torch

    if output_path.exists():
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        expected = [str(row["query_id"]) for row in rows]
        if payload["query_ids"] != expected:
            raise ValueError("cached query IDs differ")
        return payload["embeddings"], 0.0
    embeddings = []
    began = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs = processor.process_queries([str(row["query"]) for row in batch]).to(
            "cuda"
        )
        with torch.inference_mode():
            encoded = model(**inputs)
        masks = inputs["attention_mask"].bool()
        embeddings.extend(
            embedding[mask].to(device="cpu", dtype=torch.bfloat16).contiguous()
            for embedding, mask in zip(encoded, masks, strict=True)
        )
        del inputs, encoded
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - began
    torch.save(
        {
            "query_ids": [str(row["query_id"]) for row in rows],
            "embeddings": embeddings,
        },
        output_path,
    )
    return embeddings, elapsed


def _cohort(
    query_ids: list[str], *, bm25: dict[str, list[str]],
    colsmol: dict[str, list[str]], qrels: dict[str, dict[str, float]],
    depths: list[int],
) -> dict[str, Any]:
    if not query_ids:
        return {"queries": 0}
    hit = lambda ranking: float(
        np.mean([bool(set(ranking[qid]) & set(qrels[qid])) for qid in query_ids])
    )
    colsmol_metrics = aggregate_query_metrics(
        [ranking_metrics(colsmol[qid], qrels[qid]) for qid in query_ids]
    )
    hybrid = {
        qid: union_preserving_order(bm25[qid][:10], colsmol[qid][:10])
        for qid in query_ids
    }
    return {
        "queries": len(query_ids),
        "bm25_candidate_hit": {
            str(depth): hit({qid: bm25[qid][:depth] for qid in query_ids})
            for depth in depths
        },
        "colsmol_candidate_hit": {
            str(depth): hit({qid: colsmol[qid][:depth] for qid in query_ids})
            for depth in depths
        },
        "hybrid_10_10": {
            "candidate_hit": hit(hybrid),
            "metrics": aggregate_query_metrics(
                [ranking_metrics(hybrid[qid], qrels[qid]) for qid in query_ids]
            ),
        },
        "colsmol_top100_metrics": colsmol_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--full-visual-ranking", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    corpus = _read_jsonl(args.dataset_root / "corpus.jsonl")
    queries = _read_jsonl(args.dataset_root / "queries.jsonl")
    qrels = _load_qrels(args.dataset_root / "qrels.jsonl")
    doc_ids = [str(row["docid"]) for row in corpus]
    query_ids = [str(row["query_id"]) for row in queries]
    args.output_root.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    model = ColIdefics3.from_pretrained(
        args.adapter, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    ).eval()
    processor = ColIdefics3Processor.from_pretrained(
        args.adapter, local_files_only=True
    )
    model_load_seconds = time.perf_counter() - began
    documents, build_seconds, logical_index_bytes = _encode_documents(
        model=model,
        processor=processor,
        rows=corpus,
        assets=args.dataset_root / "assets",
        index_root=args.output_root / "index",
        batch_size=int(model_config["document_batch_size"]),
        shard_pages=int(model_config["index_shard_pages"]),
    )
    query_embeddings, query_encode_seconds = _encode_queries(
        model=model,
        processor=processor,
        rows=queries,
        batch_size=int(model_config["query_batch_size"]),
        output_path=args.output_root / "query-embeddings.pt",
    )
    torch.cuda.synchronize()
    began = time.perf_counter()
    scores = processor.score_multi_vector(
        query_embeddings,
        documents,
        batch_size=int(model_config["score_batch_size"]),
        device="cuda",
    ).to(device="cpu", dtype=torch.float32).numpy()
    score_seconds = time.perf_counter() - began
    rankings = {}
    ranking_path = args.output_root / "ranking.txt"
    with ranking_path.open("w", encoding="utf-8") as handle:
        for query_id, row in zip(query_ids, scores, strict=True):
            order = np.lexsort((np.asarray(doc_ids), -row))[:100]
            rankings[query_id] = [doc_ids[int(position)] for position in order]
            for position in order:
                handle.write(
                    f"{query_id}\t{doc_ids[int(position)]}\t{float(row[int(position)]):.9g}\n"
                )
    depths = list(map(int, config["reference_depths"]))
    bm25, bm25_cost = _bm25_rankings(corpus, queries, max(depths))
    full_visual = _load_visual_ranking(args.full_visual_ranking, 20)
    main_ids = deterministic_query_sample(
        query_ids,
        limit=int(config["query_selection"]["main_limit"]),
        seed=int(config["query_selection"]["seed"]),
    )
    stress_ids = [
        qid
        for qid in query_ids
        if not (set(bm25[qid][:20]) & set(qrels[qid]))
        and (set(full_visual[qid][:20]) & set(qrels[qid]))
    ]
    physical_index_bytes = sum(
        path.stat().st_size for path in (args.output_root / "index").glob("*.pt")
    )
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "dataset": {
            "root": str(args.dataset_root),
            "pages": len(corpus),
            "queries": len(queries),
            "sha256": {
                "corpus": _sha(args.dataset_root / "corpus.jsonl"),
                "queries": _sha(args.dataset_root / "queries.jsonl"),
                "qrels": _sha(args.dataset_root / "qrels.jsonl"),
                "full_visual_ranking": _sha(args.full_visual_ranking),
                "config": _sha(args.config),
            },
        },
        "model": {
            "adapter_path": str(args.adapter),
            "base_path": str(args.base),
            "adapter_bytes": _tree_bytes(args.adapter),
            "base_bytes": _tree_bytes(args.base),
            "model_load_seconds": model_load_seconds,
        },
        "cost": {
            "new_document_encode_seconds": build_seconds,
            "mean_document_encode_seconds": build_seconds / len(corpus),
            "logical_index_bytes": logical_index_bytes,
            "physical_index_bytes": physical_index_bytes,
            "mean_physical_index_bytes_per_page": physical_index_bytes / len(corpus),
            "query_encode_seconds": query_encode_seconds,
            "score_all_queries_seconds": score_seconds,
            "mean_score_seconds_per_query": score_seconds / len(queries),
            "max_cuda_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "bm25": bm25_cost,
        },
        "cohorts": {
            "main": _cohort(
                main_ids, bm25=bm25, colsmol=rankings, qrels=qrels, depths=depths
            ),
            "full_visual_repair_stress": _cohort(
                stress_ids, bm25=bm25, colsmol=rankings, qrels=qrels, depths=depths
            ),
        },
        "artifacts": {
            "ranking": {"path": str(ranking_path), "sha256": _sha(ranking_path)},
            "index_shards": len(list((args.output_root / "index").glob("*.pt"))),
        },
        "warning": config["novelty_warning"],
    }
    (args.output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cost": result["cost"], "cohorts": result["cohorts"]}, indent=2))


if __name__ == "__main__":
    main()
