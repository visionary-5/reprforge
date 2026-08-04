#!/usr/bin/env python3
"""Run cohort-certificate oracles over frozen Atlas surfaces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_heterogeneity_atlas import _npz_cube
from reprforge.cohort_certificate import analyze_certificates
from reprforge.heterogeneity_atlas import ScoreCube, deterministic_split_roles


def _irpapers_cube(surface_path: Path, queries_path: Path) -> ScoreCube:
    surface = np.load(surface_path, allow_pickle=False)
    query_ids = tuple(str(value) for value in surface["query_ids"])
    corpus_ids = tuple(str(value) for value in surface["corpus_ids"])
    corpus_lookup = {corpus_id: index for index, corpus_id in enumerate(corpus_ids)}
    with queries_path.open(newline="") as handle:
        query_rows = list(csv.DictReader(handle))
    if len(query_rows) != len(query_ids):
        raise ValueError("IRPAPERS queries and score surface are not aligned")
    relevance = []
    for row in query_rows:
        gold_id = row["dataset_id"]
        if gold_id not in corpus_lookup:
            raise ValueError(f"IRPAPERS gold page {gold_id!r} is not in the corpus")
        relevance.append({corpus_lookup[gold_id]: 1.0})
    cube = ScoreCube(
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        scores={
            "bm25": surface["bm25_scores"],
            "visual": surface["visual_scores"],
        },
        relevance=tuple(relevance),
        split_roles=deterministic_split_roles(query_ids),
    )
    cube.validate()
    return cube


def _jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mmdocir_cube(root: Path) -> tuple[ScoreCube, np.ndarray, dict]:
    query_rows = _jsonl(root / "queries.jsonl")
    item_rows = _jsonl(root / "items.jsonl")
    score_rows = _jsonl(root / "scores.jsonl")
    query_ids = tuple(row["query_id"] for row in query_rows)
    corpus_ids = tuple(row["item_id"] for row in item_rows)
    query_lookup = {query_id: index for index, query_id in enumerate(query_ids)}
    corpus_lookup = {corpus_id: index for index, corpus_id in enumerate(corpus_ids)}
    route_names = tuple(sorted(item_rows[0]["route_costs"]))
    if any(tuple(sorted(row["route_costs"])) != route_names for row in item_rows):
        raise ValueError("MMDocIR items expose inconsistent representation ladders")
    routes = {route: [] for route in route_names}
    for row in score_rows:
        if row["route"] in routes:
            routes[row["route"]].append(float(row["score"]))
    matrices = {}
    for route, values in routes.items():
        finite = [value for value in values if np.isfinite(value)]
        if not finite:
            raise ValueError(f"MMDocIR route {route!r} has no finite scores")
        floor = min(finite) - max(1.0, abs(min(finite)))
        matrices[route] = np.full(
            (len(query_ids), len(corpus_ids)), floor, dtype=np.float32
        )
    seen = {route: set() for route in routes}
    for row in score_rows:
        route = row["route"]
        if route not in matrices:
            continue
        pair = (query_lookup[row["query_id"]], corpus_lookup[row["item_id"]])
        matrices[route][pair] = float(row["score"])
        seen[route].add(pair)
    relevance = []
    expected_pairs = set()
    for query_index, row in enumerate(query_rows):
        relevance.append(
            {
                corpus_lookup[item_id]: float(value)
                for item_id, value in row["relevance"].items()
            }
        )
        expected_pairs.update(
            (query_index, corpus_lookup[item_id])
            for item_id in row["candidate_item_ids"]
        )
    for route, observed_pairs in seen.items():
        if observed_pairs != expected_pairs:
            raise ValueError(f"MMDocIR route {route!r} has incomplete candidate scores")
    # Empty native-text regions are represented by -Infinity in the source
    # bank. Preserve their rank semantics with a finite, query-local floor so
    # ScoreCube arithmetic and residual diagnostics remain well defined.
    for query_index, row in enumerate(query_rows):
        candidates = np.asarray(
            [corpus_lookup[item_id] for item_id in row["candidate_item_ids"]],
            dtype=np.int32,
        )
        for route, matrix in matrices.items():
            candidate_values = matrix[query_index, candidates]
            finite = candidate_values[np.isfinite(candidate_values)]
            if not len(finite):
                raise ValueError(
                    f"MMDocIR route {route!r} query {row['query_id']!r} "
                    "has no finite candidate scores"
                )
            floor = float(finite.min() - max(1.0, abs(float(finite.min()))))
            invalid = candidates[~np.isfinite(candidate_values)]
            matrix[query_index, invalid] = floor
    cube = ScoreCube(
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        scores=matrices,
        relevance=tuple(relevance),
        split_roles=deterministic_split_roles(query_ids),
    )
    cube.validate()
    image_costs = np.asarray(
        [row["route_costs"]["image"]["encode_ms"] for row in item_rows],
        dtype=np.float64,
    )
    storage = {
        route: int(sum(row["route_costs"][route]["index_bytes"] for row in item_rows))
        for route in routes
    }
    return cube, image_costs, {
        "full_route_index_bytes": storage,
        "candidate_indices": [
            [corpus_lookup[item_id] for item_id in row["candidate_item_ids"]]
            for row in query_rows
        ],
        "query_document_indices": [row.get("document_index") for row in query_rows],
        "item_rows": item_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exact-audit-queries", type=int, default=8)
    parser.add_argument("--irpapers-surface", type=Path)
    parser.add_argument("--irpapers-queries", type=Path)
    parser.add_argument("--irpapers-visual-index-ms", type=float, default=748708.6462900043)
    parser.add_argument("--mmdocir-root", type=Path)
    parser.add_argument("--include-depth-5", action="store_true")
    args = parser.parse_args()
    report = {"schema_version": 1, "datasets": {}}
    for name in ("hr", "finance"):
        cube, _ = _npz_cube(args.data_root / name)
        visual = np.load(
            args.data_root / name / "visual-runtime.npz", allow_pickle=False
        )
        report["datasets"][name] = {}
        for objective in ("set", "order"):
            report["datasets"][name][objective] = analyze_certificates(
                cube,
                base_route="text",
                expensive_route="visual",
                candidate_k=20,
                target_k=10,
                per_item_build_ms=visual["encode_ms"],
                exact_audit_queries=args.exact_audit_queries,
                objective=objective,
            )
        if args.include_depth_5:
            report["datasets"][name]["order_k5"] = analyze_certificates(
                cube,
                base_route="text",
                expensive_route="visual",
                candidate_k=20,
                target_k=5,
                target_metric="ndcg_at_5",
                per_item_build_ms=visual["encode_ms"],
                exact_audit_queries=args.exact_audit_queries,
                objective="order",
            )
    if bool(args.irpapers_surface) != bool(args.irpapers_queries):
        parser.error("--irpapers-surface and --irpapers-queries must be provided together")
    if args.irpapers_surface:
        cube = _irpapers_cube(args.irpapers_surface, args.irpapers_queries)
        report["datasets"]["irpapers"] = {}
        uniform_page_cost = args.irpapers_visual_index_ms / len(cube.corpus_ids)
        for objective in ("set", "order"):
            result = analyze_certificates(
                cube,
                base_route="bm25",
                expensive_route="visual",
                candidate_k=20,
                target_k=5,
                target_metric="recall_at_5",
                per_item_build_ms=np.full(len(cube.corpus_ids), uniform_page_cost),
                exact_audit_queries=args.exact_audit_queries,
                objective=objective,
            )
            result["build_cost_contract"] = (
                "uniform attribution of measured full visual index time; "
                "document counts are exact, per-page time is estimated"
            )
            report["datasets"]["irpapers"][objective] = result
    if args.mmdocir_root:
        cube, image_costs, route_metadata = _mmdocir_cube(args.mmdocir_root)
        report["datasets"]["mmdocir_within_document"] = {}
        for base_route in ("text", "image-pool-25"):
            for objective in ("set", "order"):
                result = analyze_certificates(
                    cube,
                    base_route=base_route,
                    expensive_route="image",
                    candidate_k=10,
                    target_k=5,
                    target_metric="ndcg_at_5",
                    per_item_build_ms=image_costs,
                    exact_audit_queries=args.exact_audit_queries,
                    objective=objective,
                )
                result["evaluation_boundary"] = (
                    "official MMDocIR within-document candidate scope; this is not "
                    "a corpus-wide retrieval surface"
                )
                result["route_storage"] = route_metadata
                key = f"{base_route}_to_image_{objective}"
                report["datasets"]["mmdocir_within_document"][key] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
