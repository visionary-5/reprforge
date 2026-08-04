#!/usr/bin/env python3
"""Test whether Atlas query-route headroom is cheaply predictable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_heterogeneity_atlas import _biomedical_cube, _npz_cube
from reprforge.heterogeneity_atlas import query_metrics
from reprforge.query_route_probe import (
    candidate_identity_features,
    categorical_features,
    cheap_route_features,
    crossfit_query_router,
    lexical_hash_features,
)


FIELDS = (
    "language",
    "query_types",
    "query_format",
    "content_type",
    "query_generator",
    "query_generation_pipeline",
    "source_type",
    "query_type_for_generation",
)


def _parquet_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _corpus_doc_ids(root: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    paths = sorted(root.glob("test-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no corpus parquet shards under {root}")
    rows = pq.read_table(paths, columns=["corpus_id", "doc_id"]).to_pylist()
    return {str(row["corpus_id"]): str(row["doc_id"]) for row in rows}


def _connected_query_groups(cube, corpus_doc_ids: dict[str, str]) -> tuple[str, ...]:
    """Keep queries sharing any relevant source document in one fold."""

    parents = list(range(len(cube.query_ids)))

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    first_by_doc = {}
    for query_index, labels in enumerate(cube.relevance):
        for corpus_index in labels:
            corpus_id = cube.corpus_ids[corpus_index]
            doc_id = corpus_doc_ids[corpus_id]
            if doc_id in first_by_doc:
                union(query_index, first_by_doc[doc_id])
            else:
                first_by_doc[doc_id] = query_index
    members = {}
    for query_index in range(len(cube.query_ids)):
        members.setdefault(find(query_index), []).append(cube.query_ids[query_index])
    labels = {
        root: "|".join(sorted(query_ids)) for root, query_ids in members.items()
    }
    return tuple(labels[find(index)] for index in range(len(cube.query_ids)))


def _align(rows: list[dict], query_ids: tuple[str, ...], id_key: str) -> list[dict]:
    by_id = {str(row[id_key]): row for row in rows}
    missing = [query_id for query_id in query_ids if query_id not in by_id]
    if missing:
        raise ValueError(f"query metadata is missing {len(missing)} trace identifiers")
    return [by_id[query_id] for query_id in query_ids]


def _groups(cube, rows: list[dict], *, field: str, metric: str, k: int) -> dict:
    utilities = np.stack(
        [
            query_metrics(cube.scores[route], cube.relevance, ks=(k,))[metric]
            for route in cube.routes
        ],
        axis=1,
    )
    oracle = np.argmax(utilities, axis=1)
    result = {}
    values = sorted(
        {
            str(value)
            for row in rows
            for value in (
                row.get(field, [])
                if isinstance(row.get(field), (list, tuple))
                else [row.get(field)]
            )
            if value is not None
        }
    )
    for value in values:
        selected = np.asarray(
            [
                value
                in {
                    str(item)
                    for item in (
                        row.get(field, [])
                        if isinstance(row.get(field), (list, tuple))
                        else [row.get(field)]
                    )
                }
                for row in rows
            ]
        )
        result[value] = {
            "queries": int(selected.sum()),
            "oracle_routes": {
                route: int(np.sum(oracle[selected] == index))
                for index, route in enumerate(cube.routes)
            },
            "mean_route_utility": {
                route: float(utilities[selected, index].mean())
                for index, route in enumerate(cube.routes)
            },
        }
    return result


def _probe_dataset(
    cube, rows: list[dict], *, metric: str, k: int, groups=None
) -> dict:
    texts = [str(row["query"]) for row in rows]
    lexical = lexical_hash_features(texts)
    categorical, names = categorical_features(rows, fields=FIELDS)
    lexical_shape = lexical[:, -6:]
    cheap_route = "text" if "text" in cube.routes else "base"
    cheap = cheap_route_features(cube.scores[cheap_route])
    candidate_identity = candidate_identity_features(
        cube.scores[cheap_route], candidate_k=min(20, len(cube.corpus_ids))
    )
    feature_sets = {
        "lexical_hash": lexical,
        "metadata_and_shape": np.column_stack([categorical, lexical_shape]),
        "cheap_route_statistics": cheap,
        "candidate_identity": candidate_identity,
        "candidate_identity_and_statistics": np.column_stack(
            [candidate_identity, cheap]
        ),
        "combined": np.column_stack([lexical, categorical, cheap]),
    }
    report = {
        "cheap_route": cheap_route,
        "categorical_features": len(names),
        "feature_dimensions": {
            name: int(values.shape[1]) for name, values in feature_sets.items()
        },
        "crossfit": {
            name: crossfit_query_router(
                cube, values, target_metric=metric, k=k, folds=5
            )
            for name, values in feature_sets.items()
        },
        "content_type_diagnostic": _groups(
            cube, rows, field="content_type", metric=metric, k=k
        ),
        "query_type_diagnostic": _groups(
            cube, rows, field="query_types", metric=metric, k=k
        ),
    }
    if groups is not None:
        group_sizes = sorted(
            [groups.count(value) for value in set(groups)], reverse=True
        )
        if len(group_sizes) < 5:
            report["grouped_split"] = {
                "status": "not_identifiable",
                "reason": (
                    "relevance-document connectivity leaves fewer than five "
                    "disjoint groups"
                ),
                "groups": len(group_sizes),
                "group_sizes": group_sizes,
            }
            return report
        report["grouped_split"] = {
            "status": "run",
            "groups": len(set(groups)),
            "group_sizes": group_sizes,
            "crossfit": {
                name: crossfit_query_router(
                    cube,
                    values,
                    target_metric=metric,
                    k=k,
                    folds=5,
                    groups=groups,
                )
                for name, values in feature_sets.items()
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hr-queries", type=Path, required=True)
    parser.add_argument("--finance-queries", type=Path, required=True)
    parser.add_argument("--hr-corpus", type=Path, required=True)
    parser.add_argument("--finance-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {"schema_version": 1, "datasets": {}}
    for name, query_path, corpus_path in (
        ("hr", args.hr_queries, args.hr_corpus),
        ("finance", args.finance_queries, args.finance_corpus),
    ):
        cube, _ = _npz_cube(args.data_root / name)
        rows = _align(_parquet_rows(query_path), cube.query_ids, "query_id")
        groups = _connected_query_groups(cube, _corpus_doc_ids(corpus_path))
        report["datasets"][name] = _probe_dataset(
            cube, rows, metric="ndcg_at_10", k=10, groups=groups
        )

    cube, _ = _biomedical_cube(args.data_root / "biomedical")
    rows = _align(
        [
            json.loads(line)
            for line in (args.data_root / "biomedical" / "queries.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ],
        cube.query_ids,
        "query-id",
    )
    report["datasets"]["biomedical_interaction_pilot"] = _probe_dataset(
        cube, rows, metric="ndcg_at_5", k=5
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
