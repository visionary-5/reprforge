#!/usr/bin/env python3
"""CPU scale benchmark for the representation-view control plane.

The benchmark intentionally does not claim retrieval quality.  It stress-tests
the metadata lifecycle at a corpus and workload size similar to IRPAPERS, or at
larger synthetic sizes, before expensive GPU execution is connected.  All
latent utilities are marked synthetic in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np

from reprforge.representation_views import (
    CandidateView,
    RepresentationViewCatalog,
    ViewKey,
    ViewState,
    apply_materialization_plan,
)
from reprforge.pairwise_view_admission import (
    build_boundary_pairs,
    select_frequency_pages,
    select_independent_pages,
    select_pairwise_pages,
)


ROUTES = {
    "pool-9": {
        "utility_factor": 0.52,
        "probe_cost_ms": 8.0,
        "build_cost_ms": 35.0,
        "storage_bytes": 72_000,
    },
    "pool-25": {
        "utility_factor": 0.76,
        "probe_cost_ms": 13.0,
        "build_cost_ms": 72.0,
        "storage_bytes": 184_000,
    },
    "full-visual": {
        "utility_factor": 1.0,
        "probe_cost_ms": 24.0,
        "build_cost_ms": 210.0,
        "storage_bytes": 528_000,
    },
}


def _candidate_matrix(
    rng: np.random.Generator,
    *,
    items: int,
    queries: int,
    candidate_k: int,
    skew: float,
) -> np.ndarray:
    ranks = np.arange(1, items + 1, dtype=np.float64)
    probabilities = np.power(ranks, -skew)
    probabilities /= probabilities.sum()
    return np.stack(
        [
            rng.choice(
                items,
                size=candidate_k,
                replace=False,
                p=probabilities,
            )
            for _ in range(queries)
        ]
    )


def run_control_plane_benchmark(
    *,
    items: int,
    queries: int,
    candidate_k: int,
    skew: float,
    probe_budget_fraction: float,
    build_budget_fraction: float,
    storage_budget_fraction: float,
    seed: int,
    snapshot: Path,
) -> dict[str, Any]:
    if items <= 0 or queries <= 0 or candidate_k <= 0:
        raise ValueError("items, queries and candidate_k must be positive")
    if candidate_k > items:
        raise ValueError("candidate_k cannot exceed the corpus")
    if skew < 0:
        raise ValueError("skew cannot be negative")
    fractions = (
        probe_budget_fraction,
        build_budget_fraction,
        storage_budget_fraction,
    )
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("all budget fractions must be in [0, 1]")

    rng = np.random.default_rng(seed)
    tracemalloc.start()
    began = time.perf_counter()
    candidates = _candidate_matrix(
        rng,
        items=items,
        queries=queries,
        candidate_k=candidate_k,
        skew=skew,
    )
    reuse = Counter(int(value) for value in candidates.flat)
    workload_generation_seconds = time.perf_counter() - began

    began = time.perf_counter()
    pair_rng = np.random.default_rng(seed + 1)
    locator_scores = (
        -0.22 * np.arange(candidate_k, dtype=np.float64)[None, :]
        + pair_rng.normal(0.0, 0.035, size=(queries, candidate_k))
    )
    cutoff = min(5, candidate_k - 1)
    rank_risk = np.zeros(candidate_k, dtype=np.float64)
    rank_risk[cutoff:] = np.exp(
        -0.22 * np.arange(candidate_k - cutoff, dtype=np.float64)
    )
    boundary_pairs = build_boundary_pairs(
        candidates,
        locator_scores,
        cutoff=cutoff,
        rank_risk=rank_risk,
        temperature=1.0,
    )
    pair_graph_seconds = time.perf_counter() - began

    began = time.perf_counter()
    latent = rng.beta(1.5, 5.0, size=items)
    catalog = RepresentationViewCatalog()
    for item in sorted(reuse):
        for route, profile in ROUTES.items():
            noisy_estimate = max(
                0.0,
                float(latent[item] * profile["utility_factor"])
                + float(rng.normal(0.0, 0.035)),
            )
            catalog.propose(
                CandidateView(
                    key=ViewKey(f"page-{item:08d}", route),
                    slot="retrieval",
                    parent_route="text",
                    expected_utility=noisy_estimate,
                    uncertainty=0.04 + 0.08 * (1.0 - noisy_estimate),
                    expected_reuse=float(reuse[item]),
                    probe_cost_ms=float(profile["probe_cost_ms"]),
                    build_cost_ms=float(profile["build_cost_ms"]),
                    storage_bytes=int(profile["storage_bytes"]),
                    maintenance_cost_ms=float(profile["build_cost_ms"] * 0.05),
                )
            )
    catalog_generation_seconds = time.perf_counter() - began

    total_probe_cost = sum(view.probe_cost_ms for view in catalog.views())
    began = time.perf_counter()
    probe_plan = catalog.plan_probes(
        budget_ms=total_probe_cost * probe_budget_fraction,
        exploration_weight=1.0,
    )
    catalog.begin_probes(probe_plan)
    for key in probe_plan.keys:
        item = int(key.item_id.removeprefix("page-"))
        profile = ROUTES[key.route]
        observed = max(
            0.0,
            float(latent[item] * profile["utility_factor"])
            + float(rng.normal(0.0, 0.02)),
        )
        catalog.finish_probe(
            key,
            observed_utility=observed,
            actual_cost_ms=float(profile["probe_cost_ms"]),
            minimum_utility=0.04,
            artifact_reusable=True,
        )
    probe_seconds = time.perf_counter() - began

    candidate_pages = len(reuse)
    full_visual_build_ms = candidate_pages * ROUTES["full-visual"]["build_cost_ms"]
    full_visual_storage = candidate_pages * ROUTES["full-visual"]["storage_bytes"]
    began = time.perf_counter()
    materialization = catalog.plan_materialization(
        build_budget_ms=full_visual_build_ms * build_budget_fraction,
        storage_budget_bytes=int(full_visual_storage * storage_budget_fraction),
    )
    apply_materialization_plan(
        catalog,
        materialization,
        versions={
            key: version
            for version, key in enumerate(materialization.keys, start=1)
        },
    )
    planning_seconds = time.perf_counter() - began

    pair_page_budget = max(2, int(0.2 * candidate_pages))
    began = time.perf_counter()
    pairwise = select_pairwise_pages(
        boundary_pairs,
        page_budget=pair_page_budget,
    )
    pair_planning_seconds = time.perf_counter() - began
    independent = select_independent_pages(
        boundary_pairs,
        page_budget=pair_page_budget,
    )
    frequency = select_frequency_pages(
        candidates,
        boundary_pairs,
        page_budget=pair_page_budget,
    )

    catalog.save(snapshot)
    reloaded = RepresentationViewCatalog.load(snapshot)
    if reloaded.to_payload() != catalog.to_payload():
        raise AssertionError("catalog snapshot changed during round trip")
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    materialized = catalog.views(state=ViewState.MATERIALIZED)
    route_counts = Counter(view.key.route for view in materialized)
    stable_payload = catalog.to_payload()
    fingerprint = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    summary = catalog.summary()
    return {
        "schema_version": 1,
        "benchmark": "synthetic-representation-view-control-plane",
        "quality_claim": False,
        "configuration": {
            "items": items,
            "queries": queries,
            "candidate_k": candidate_k,
            "skew": skew,
            "seed": seed,
            "routes": sorted(ROUTES),
            "probe_budget_fraction": probe_budget_fraction,
            "build_budget_fraction": build_budget_fraction,
            "storage_budget_fraction": storage_budget_fraction,
        },
        "workload": {
            "candidate_events": int(candidates.size),
            "candidate_pages": candidate_pages,
            "candidate_page_fraction": candidate_pages / items,
            "max_page_reuse": max(reuse.values()),
            "mean_page_reuse": float(np.mean(list(reuse.values()))),
            "candidate_views": len(catalog),
        },
        "probe": {
            "planned_views": len(probe_plan.keys),
            "estimated_cost_ms": probe_plan.estimated_cost_ms,
            "verified_views": len(catalog.views(state=ViewState.VERIFIED))
            + len(materialized),
            "rejected_views": len(catalog.views(state=ViewState.REJECTED)),
        },
        "materialization": {
            "views": len(materialized),
            "pages": len({view.key.item_id for view in materialized}),
            "route_counts": dict(sorted(route_counts.items())),
            "estimated_remaining_build_ms": materialization.estimated_build_ms,
            "storage_bytes": materialization.storage_bytes,
            "full_visual_candidate_build_ms": full_visual_build_ms,
            "full_visual_candidate_storage_bytes": full_visual_storage,
            "build_fraction_of_full_visual_candidates": (
                materialization.estimated_build_ms / full_visual_build_ms
                if full_visual_build_ms
                else 0.0
            ),
            "storage_fraction_of_full_visual_candidates": (
                materialization.storage_bytes / full_visual_storage
                if full_visual_storage
                else 0.0
            ),
        },
        "pairwise_probe": {
            "synthetic_objective_only": True,
            "cutoff": cutoff,
            "boundary_pairs": len(boundary_pairs),
            "page_budget": pair_page_budget,
            "pairwise_covered_weight_fraction": pairwise.covered_weight_fraction,
            "independent_covered_weight_fraction": (
                independent.covered_weight_fraction
            ),
            "frequency_covered_weight_fraction": frequency.covered_weight_fraction,
            "pairwise_planning_seconds": pair_planning_seconds,
        },
        "catalog": {
            **summary,
            "snapshot_bytes": snapshot.stat().st_size,
            "sha256": fingerprint,
        },
        "control_plane": {
            "workload_generation_seconds": workload_generation_seconds,
            "pair_graph_seconds": pair_graph_seconds,
            "catalog_generation_seconds": catalog_generation_seconds,
            "probe_seconds": probe_seconds,
            "materialization_planning_seconds": planning_seconds,
            "pairwise_planning_seconds": pair_planning_seconds,
            "peak_python_bytes": peak_bytes,
        },
        "interpretation": (
            "This run validates deterministic metadata planning, budgets, state "
            "transitions and persistence only. Latent utility is synthetic and "
            "cannot support retrieval-quality or GPU-speedup claims."
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=3_230)
    parser.add_argument("--queries", type=int, default=180)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--skew", type=float, default=1.1)
    parser.add_argument("--probe-budget-fraction", type=float, default=0.2)
    parser.add_argument("--build-budget-fraction", type=float, default=0.25)
    parser.add_argument("--storage-budget-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.catalog is None:
        with TemporaryDirectory(prefix="reprforge-view-catalog-") as directory:
            result = run_control_plane_benchmark(
                items=args.items,
                queries=args.queries,
                candidate_k=args.candidate_k,
                skew=args.skew,
                probe_budget_fraction=args.probe_budget_fraction,
                build_budget_fraction=args.build_budget_fraction,
                storage_budget_fraction=args.storage_budget_fraction,
                seed=args.seed,
                snapshot=Path(directory) / "catalog.json",
            )
    else:
        result = run_control_plane_benchmark(
            items=args.items,
            queries=args.queries,
            candidate_k=args.candidate_k,
            skew=args.skew,
            probe_budget_fraction=args.probe_budget_fraction,
            build_budget_fraction=args.build_budget_fraction,
            storage_budget_fraction=args.storage_budget_fraction,
            seed=args.seed,
            snapshot=args.catalog,
        )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
