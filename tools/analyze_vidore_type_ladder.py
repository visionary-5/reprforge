#!/usr/bin/env python3
"""Source-document-cross-fit a static type ladder on full-corpus ViDoRe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from analyze_type_policy_compiler import _document_folds
from reprforge.heterogeneity_atlas import ScoreCube
from reprforge.type_policy_compiler import crossfit_type_policy_compiler


def _page_type(markdown: str) -> str:
    value = markdown or ""
    return "table" if value.count("|") >= 4 or "<table" in value.lower() else "text"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--text-runtime", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pooled = np.load(args.pooled_runtime, allow_pickle=False)
    full = np.load(args.full_runtime, allow_pickle=False)
    text = np.load(args.text_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    for other in (full, text):
        if not np.array_equal(pooled["query_ids"], other["query_ids"]):
            raise ValueError("query IDs are not aligned")
        if not np.array_equal(pooled["corpus_ids"], other["corpus_ids"]):
            raise ValueError("corpus IDs are not aligned")
    corpus_rows = []
    for path in sorted(args.corpus_root.glob("*.parquet")):
        corpus_rows.extend(
            pq.read_table(path, columns=["corpus_id", "doc_id", "markdown"]).to_pylist()
        )
    row_by_id = {str(row["corpus_id"]): row for row in corpus_rows}
    ordered_rows = [row_by_id[str(value)] for value in pooled["corpus_ids"]]
    relevance = [dict() for _ in pooled["query_ids"]]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        relevance[int(query)][int(corpus)] = float(value)
    query_groups = [
        "||".join(
            sorted({str(ordered_rows[index]["doc_id"]) for index in query_relevance})
        )
        for query_relevance in relevance
    ]
    cube = ScoreCube(
        query_ids=tuple(str(value) for value in pooled["query_ids"]),
        corpus_ids=tuple(str(value) for value in pooled["corpus_ids"]),
        scores={
            "image": full["scores"],
            "image-pool-9": pooled["scores"],
            "text": text["scores"],
        },
        relevance=tuple(relevance),
        split_roles=("fit",) * len(pooled["query_ids"]),
    )
    cube.validate()
    candidates = (np.arange(len(cube.corpus_ids), dtype=np.int32),) * len(
        cube.query_ids
    )
    result = crossfit_type_policy_compiler(
        cube,
        item_types=tuple(_page_type(str(row["markdown"] or "")) for row in ordered_rows),
        candidate_indices=candidates,
        route_costs={
            "image": full["vector_bytes"],
            "image-pool-9": pooled["vector_bytes"],
            "text": text["vector_bytes"],
        },
        fold_ids=_document_folds(query_groups),
        budget_fractions=(0.111, 0.25, 0.5, 0.75, 1.0),
    )
    result["page_type_rule"] = "table iff markdown has >=4 pipes or an HTML table"
    result["source_documents"] = len({row["doc_id"] for row in ordered_rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
