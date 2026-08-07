#!/usr/bin/env python3
"""Prepare leakage-audited subsets and synthetic workload traces.

This command does no VLM work.  It is intended to finish before a GPU server is
started, so every expensive run consumes a frozen corpus subset and trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.bm25 import build_index, score_queries
from reprforge.progressive_selection import MaterializationFeatures, selection_order
from reprforge.progressive_workloads import trace_suite
from reprforge.visual_page_features import image_features


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold(query_id: str, seed: int, folds: int = 5) -> int:
    value = hashlib.sha256(f"{query_id}\0{seed}".encode()).digest()
    return int.from_bytes(value[:8], "big") % folds


def _document_id(row: dict[str, Any]) -> tuple[str, str]:
    for field in ("document_id", "source_document_id", "pdf_id", "file_id"):
        if row.get(field) is not None:
            return str(row[field]), field
    doc_id = str(row["docid"])
    inferred = re.sub(r"(?i)(?:[_:/-](?:page|p)[_-]?[0-9]+)$", "", doc_id)
    return (inferred if inferred != doc_id else doc_id), "docid_suffix_or_unique"


def _load_ranking(path: Path, depth: int) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open() as handle:
        for line in handle:
            query_id, doc_id, *_ = line.rstrip("\n").split("\t")
            values = result.setdefault(query_id, [])
            if len(values) < depth:
                values.append(doc_id)
    return result


def _bm25_rankings(
    corpus: list[dict[str, Any]], queries: list[dict[str, Any]], depth: int
) -> tuple[dict[str, list[str]], dict[str, int]]:
    doc_ids = [str(row["docid"]) for row in corpus]
    state, postings, vocabulary_bytes = build_index(
        [str(row.get("text") or "") for row in corpus]
    )
    matrix = score_queries(state, [str(row["query"]) for row in queries], k1=1.2, b=0.75)
    rankings = {}
    for row, scores in zip(queries, matrix, strict=True):
        order = np.lexsort((np.asarray(doc_ids), -scores))[:depth]
        rankings[str(row["query_id"])] = [doc_ids[int(position)] for position in order]
    return rankings, {
        "posting_bytes": int(postings.sum()),
        "vocabulary_bytes": int(vocabulary_bytes),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compact-ranking", type=Path)
    parser.add_argument("--maximum-image-side", type=int, default=256)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    config = json.loads(args.config.read_text())
    corpus_path = args.dataset_root / "corpus.jsonl"
    query_path = args.dataset_root / "queries.jsonl"
    qrel_path = args.dataset_root / "qrels.jsonl"
    asset_root = args.dataset_root / "assets"
    for path in (corpus_path, query_path, qrel_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not asset_root.is_dir():
        raise FileNotFoundError(asset_root)
    corpus, queries, qrels = map(_read_jsonl, (corpus_path, query_path, qrel_path))
    doc_ids = [str(row["docid"]) for row in corpus]
    query_ids = [str(row["query_id"]) for row in queries]
    if len(doc_ids) != len(set(doc_ids)) or len(query_ids) != len(set(query_ids)):
        raise ValueError("corpus and query IDs must each be unique")
    unknown_queries = sorted({str(row["query_id"]) for row in qrels} - set(query_ids))
    unknown_docs = sorted({str(row["doc_id"]) for row in qrels} - set(doc_ids))
    if unknown_queries or unknown_docs:
        raise ValueError(
            f"qrel coverage mismatch: {len(unknown_queries)} queries, {len(unknown_docs)} pages"
        )
    row_by_id = {str(row["docid"]): row for row in corpus}
    documents = {}
    document_sources = Counter()
    for row in corpus:
        document_id, source = _document_id(row)
        documents[str(row["docid"])] = document_id
        document_sources[source] += 1
    missing_assets = [
        str(row["image"])
        for row in corpus
        if not (asset_root / str(row["image"])).is_file()
    ]
    if missing_assets:
        raise FileNotFoundError(f"missing {len(missing_assets)} page assets")

    seed = int(config["workloads"]["seed"])
    split = {query_id: _fold(query_id, seed) for query_id in query_ids}
    history_queries = {query_id for query_id, fold in split.items() if fold != 0}
    evaluation_queries = {query_id for query_id, fold in split.items() if fold == 0}
    depth = max(map(int, config["cheap_locator_depths"]))
    bm25, bm25_bytes = _bm25_rankings(corpus, queries, depth)
    history_counts = Counter(
        doc_id
        for query_id in history_queries
        for doc_id in bm25[query_id][:depth]
    )
    compact = _load_ranking(args.compact_ranking, depth) if args.compact_ranking else None
    disagreement = Counter()
    if compact is not None:
        missing = sorted(set(query_ids) - set(compact))
        if missing:
            raise ValueError(f"compact ranking lacks {len(missing)} queries")
        for query_id in history_queries:
            text_set = set(bm25[query_id][:20])
            compact_set = set(compact[query_id][:20])
            for doc_id in text_set ^ compact_set:
                disagreement[doc_id] += 1

    features = []
    for row in corpus:
        doc_id = str(row["docid"])
        entropy, edge, _ = image_features(
            asset_root / str(row["image"]), args.maximum_image_side
        )
        features.append(
            MaterializationFeatures(
                doc_id=doc_id,
                document_id=documents[doc_id],
                text_chars=len(str(row.get("text") or "").strip()),
                grayscale_entropy=entropy,
                edge_energy=edge,
                locator_disagreement=(
                    disagreement[doc_id] / max(1, len(history_queries))
                ),
                history_candidate_count=history_counts[doc_id],
            )
        )

    args.output_root.mkdir(parents=True)
    feature_path = args.output_root / "features.jsonl"
    _write_jsonl(feature_path, [row.__dict__ for row in features])
    split_path = args.output_root / "query-splits.json"
    split_path.write_text(
        json.dumps(
            {
                "assignment": "sha256(query_id + NUL + seed) modulo 5",
                "seed": seed,
                "history_folds": [1, 2, 3, 4],
                "evaluation_fold": 0,
                "queries": split,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    qrel_documents: dict[str, str] = {}
    for row in sorted(qrels, key=lambda value: (str(value["query_id"]), -float(value["relevance"]))):
        qrel_documents.setdefault(str(row["query_id"]), documents[str(row["doc_id"])])
    groups = {query_id: qrel_documents.get(query_id, query_id) for query_id in query_ids}
    traces = trace_suite(
        query_ids,
        groups,
        seed=seed,
        random_permutations=int(config["workloads"]["random_permutations"]),
        horizon_multiplier=int(config["workloads"]["horizon_multiplier"]),
        zipf_exponents=list(map(float, config["workloads"]["zipf_exponents"])),
    )
    trace_root = args.output_root / "traces"
    trace_root.mkdir()
    for name, trace in traces.items():
        _write_jsonl(
            trace_root / f"{name}.jsonl",
            [
                {"event_id": position, "query_id": query_id, "trace": name}
                for position, query_id in enumerate(trace)
            ],
        )

    strategies = [
        value
        for value in config["static_selectors"]
        if value != "future_aware_marginal_quality_oracle"
    ]
    if compact is None:
        strategies.remove("cheap_locator_disagreement")
    subset_root = args.output_root / "subsets"
    subset_root.mkdir()
    subset_count = 0
    for strategy in strategies:
        order = selection_order(features, strategy=strategy, seed=seed)
        strategy_root = subset_root / strategy
        strategy_root.mkdir()
        for budget in map(float, config["static_page_budget_fractions"]):
            count = 0 if budget == 0 else min(len(corpus), max(1, math.ceil(budget * len(corpus))))
            selected_ids = order[:count]
            budget_root = strategy_root / f"budget-{round(100 * budget):03d}"
            budget_root.mkdir()
            selected_path = budget_root / "corpus.jsonl"
            _write_jsonl(selected_path, [row_by_id[doc_id] for doc_id in selected_ids])
            manifest = {
                "schema_version": 1,
                "protocol_id": config["protocol_id"],
                "strategy": strategy,
                "seed": seed,
                "budget_fraction": budget,
                "selected_pages": count,
                "source_pages": len(corpus),
                "selected_fraction": count / len(corpus),
                "direct_physical_build_required": budget
                in set(map(float, config["direct_physical_build_fractions"])),
                "information_boundary": {
                    "uses_qrels_for_selection": False,
                    "uses_full_visual_scores": False,
                    "history_query_count": len(history_queries),
                    "future_query_count": len(evaluation_queries),
                    "uses_compact_ranking": compact is not None
                    and strategy == "cheap_locator_disagreement",
                },
                "sha256": {
                    "config": _sha(args.config),
                    "source_corpus": _sha(corpus_path),
                    "selected_corpus": _sha(selected_path),
                    "features": _sha(feature_path),
                    "query_splits": _sha(split_path),
                },
            }
            (budget_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            subset_count += 1

    top_manifest = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "prepared_cpu_no_gpu_results",
        "dataset": {
            "pages": len(corpus),
            "queries": len(queries),
            "qrels": len(qrels),
            "documents": len(set(documents.values())),
            "document_id_sources": dict(document_sources),
        },
        "splits": {
            "history_queries": len(history_queries),
            "evaluation_queries": len(evaluation_queries),
        },
        "strategies": strategies,
        "subsets": subset_count,
        "traces": {name: len(trace) for name, trace in traces.items()},
        "bm25_logical_bytes": bm25_bytes,
        "warnings": [
            "Trace locality uses qrel document groups only to construct a shared synthetic workload; no selector receives qrel labels.",
            "Dataset order is deterministic serialization, not a natural temporal trace.",
            "Compact locator disagreement is omitted when no compact ranking is supplied.",
        ],
        "sha256": {
            "config": _sha(args.config),
            "corpus": _sha(corpus_path),
            "queries": _sha(query_path),
            "qrels": _sha(qrel_path),
            "features": _sha(feature_path),
            "compact_ranking": _sha(args.compact_ranking) if args.compact_ranking else None,
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(top_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(top_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
