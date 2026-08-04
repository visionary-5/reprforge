#!/usr/bin/env python3
"""Materialize oracle labels only after qrel-free certificates exist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.vidore_local_eval import _component_paths, _component_sha256, _read_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_certificates(
    paths: list[Path], *, dataset: str, runtime_sha256: str
) -> dict[str, str]:
    if not paths:
        raise ValueError("at least one pre-qrel certificate is required")
    hashes: dict[str, str] = {}
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("stage") != "pre-qrel-ranking-certification":
            raise ValueError(f"{path} is not a pre-qrel certificate")
        if report.get("dataset") != dataset:
            raise ValueError(f"{path} has a different dataset")
        if report.get("qrels_loaded") is not False:
            raise ValueError(f"{path} does not attest qrels_loaded=false")
        if (
            report.get("artifacts", {}).get("reference_runtime_sha256")
            != runtime_sha256
        ):
            raise ValueError(f"{path} refers to a different full runtime")
        candidate = str(report["candidate"])
        if candidate in hashes:
            raise ValueError(f"duplicate certificate for {candidate}")
        hashes[candidate] = _sha256(path)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--qrels-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--certificate", type=Path, action="append", required=True
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime_sha256 = _sha256(args.runtime)
    certificate_hashes = _load_certificates(
        args.certificate,
        dataset=args.dataset,
        runtime_sha256=runtime_sha256,
    )
    runtime = np.load(args.runtime, allow_pickle=False)
    query_ids = [str(value) for value in runtime["query_ids"]]
    corpus_ids = [str(value) for value in runtime["corpus_ids"]]
    query_positions = {value: index for index, value in enumerate(query_ids)}
    corpus_positions = {value: index for index, value in enumerate(corpus_ids)}

    qrel_paths = _component_paths(args.qrels_root.parent, args.qrels_root.name)
    rows = _read_rows(qrel_paths, ("query_id", "corpus_id", "score"))
    label_query: list[int] = []
    label_corpus: list[int] = []
    label_relevance: list[int] = []
    judged_queries: set[str] = set()
    for row in rows:
        query_id = str(row["query_id"])
        corpus_id = str(row["corpus_id"])
        if query_id not in query_positions or corpus_id not in corpus_positions:
            continue
        label_query.append(query_positions[query_id])
        label_corpus.append(corpus_positions[corpus_id])
        label_relevance.append(int(row["score"]))
        judged_queries.add(query_id)
    if judged_queries != set(query_ids):
        raise ValueError("at least one runtime query lacks a relevance judgment")

    args.labels.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.labels,
        query_positions=np.asarray(label_query, dtype=np.int32),
        corpus_positions=np.asarray(label_corpus, dtype=np.int32),
        relevance=np.asarray(label_relevance, dtype=np.int16),
    )
    report = {
        "schema_version": 1,
        "protocol": "qrel-free-compression-risk-2026-08-04",
        "stage": "post-certificate-oracle-label-materialization",
        "dataset": args.dataset,
        "query_count": len(query_ids),
        "corpus_count": len(corpus_ids),
        "label_count": len(label_relevance),
        "pre_qrel_certificate_sha256": certificate_hashes,
        "runtime_sha256": runtime_sha256,
        "qrels_sha256": _component_sha256(qrel_paths),
        "labels_sha256": _sha256(args.labels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
