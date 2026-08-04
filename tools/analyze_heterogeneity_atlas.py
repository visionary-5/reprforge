#!/usr/bin/env python3
"""Build the first Heterogeneity Atlas from frozen local score surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from reprforge.heterogeneity_atlas import (
    ScoreCube,
    analyze_cube,
    deterministic_split_roles,
)


def _json_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _npz_cube(root: Path) -> tuple[ScoreCube, dict[str, dict[str, float]]]:
    route_paths = {
        "text": root / "text-runtime.npz",
        "visual": root / "visual-runtime.npz",
    }
    payloads = {
        route: np.load(path, allow_pickle=False)
        for route, path in route_paths.items()
    }
    reference = payloads["text"]
    query_ids = tuple(str(value) for value in reference["query_ids"].tolist())
    corpus_ids = tuple(str(value) for value in reference["corpus_ids"].tolist())
    for route, payload in payloads.items():
        if tuple(str(value) for value in payload["query_ids"].tolist()) != query_ids:
            raise ValueError(f"{route} query identifiers do not align")
        if tuple(str(value) for value in payload["corpus_ids"].tolist()) != corpus_ids:
            raise ValueError(f"{route} corpus identifiers do not align")

    labels = np.load(root / "oracle-labels.npz", allow_pickle=False)
    relevance: list[dict[int, float]] = [dict() for _ in query_ids]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        relevance[int(query)][int(corpus)] = float(value)
    costs = {}
    for route, payload in payloads.items():
        costs[route] = {
            "total_vector_bytes": float(np.asarray(payload["vector_bytes"]).sum()),
            "total_encode_ms": float(np.asarray(payload["encode_ms"]).sum()),
            "index_total_ms": float(np.asarray(payload["index_total_ms"]).item()),
        }
    cube = ScoreCube(
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        scores={route: payload["scores"] for route, payload in payloads.items()},
        relevance=tuple(relevance),
        split_roles=deterministic_split_roles(query_ids),
    )
    return cube, costs


def _biomedical_cube(root: Path) -> tuple[ScoreCube, dict[str, dict[str, float]]]:
    score_files = {
        "base": root / "scores-small.jsonl",
        "upgrade": root / "scores-small-large.jsonl",
        "windowed": root / "scores-windowed.jsonl",
    }
    rows = {name: _json_rows(path) for name, path in score_files.items()}
    query_ids = tuple(sorted({str(row["query_id"]) for row in rows["base"]}))
    corpus_ids = tuple(
        sorted({str(row["corpus_id"]) for row in rows["base"]}, key=int)
    )
    query_positions = {value: index for index, value in enumerate(query_ids)}
    corpus_positions = {value: index for index, value in enumerate(corpus_ids)}
    scores = {
        route: np.full((len(query_ids), len(corpus_ids)), np.nan)
        for route in score_files
    }
    for route, route_rows in rows.items():
        for row in route_rows:
            query = query_positions[str(row["query_id"])]
            corpus = corpus_positions[str(row["corpus_id"])]
            scores[route][query, corpus] = float(
                row["base_score"]
                if route == "base"
                else row["action_scores"][route]
            )

    relevance: list[dict[int, float]] = [dict() for _ in query_ids]
    for row in _json_rows(root / "qrels.jsonl"):
        relevance[query_positions[str(row["query-id"])]][
            corpus_positions[str(row["corpus-id"])]
        ] = float(row["score"])
    split_by_query = {
        str(row["query-id"]): str(row["split-role"])
        for row in _json_rows(root / "queries.jsonl")
    }

    small_summary = json.loads(
        (root / "representation-generation-summary.json").read_text()
    )
    large_summary = json.loads(
        (root / "representation-generation-summary-small-large.json").read_text()
    )
    meta = {
        route: json.loads((root / filename).read_text())
        for route, filename in {
            "base": "scores-small.jsonl.meta.json",
            "upgrade": "scores-small-large.jsonl.meta.json",
            "windowed": "scores-windowed.jsonl.meta.json",
        }.items()
    }
    costs = {
        "base": {
            "representation_generation_ms": float(
                small_summary["views"]["small"]["inference_ms"]["total"]
            ),
            "score_surface_embedding_ms": float(meta["base"]["embedding_ms"]),
        },
        "upgrade": {
            "representation_generation_ms": float(
                large_summary["views"]["large"]["inference_ms"]["total"]
            ),
            "score_surface_embedding_ms": float(meta["upgrade"]["embedding_ms"]),
        },
        "windowed": {
            "representation_generation_ms": float(
                small_summary["views"]["small"]["inference_ms"]["total"]
            ),
            "score_surface_embedding_ms": float(meta["windowed"]["embedding_ms"]),
            "extra_windows": float(meta["windowed"]["suffix_windows"]),
        },
    }
    return (
        ScoreCube(
            query_ids=query_ids,
            corpus_ids=corpus_ids,
            scores=scores,
            relevance=tuple(relevance),
            split_roles=tuple(split_by_query[value] for value in query_ids),
        ),
        costs,
    )


def build_report(data_root: Path) -> dict:
    datasets: dict[str, dict] = {}
    for name in ("hr", "finance"):
        cube, costs = _npz_cube(data_root / name)
        datasets[name] = analyze_cube(cube, costs=costs)
    cube, costs = _biomedical_cube(data_root / "biomedical")
    datasets["biomedical_interaction_pilot"] = analyze_cube(
        cube, ks=(5, 10), target_metric="ndcg_at_5", costs=costs
    )
    return {
        "schema_version": 1,
        "purpose": "phenomenon-first localization of representation heterogeneity",
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

