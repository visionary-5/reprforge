"""Artifact audit and honest headroom for multilevel retrieval representations.

This module deliberately does not invent a mixed-state score surface.  It can
compare aligned, real uniform routes and compute an unattainable per-query
route oracle.  A deployable physical replay is declared ready only when the
same bundle also contains its missing transition and activation measurements.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.heterogeneity_atlas import (
    deterministic_split_roles,
    paired_bootstrap_ci,
    query_metrics,
)


REQUIRED_TIERS = ("cheap_base", "compact_pool", "full_multivector")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(archive: np.lib.npyio.NpzFile, key: str) -> tuple[str, ...]:
    if key not in archive:
        raise ValueError(f"runtime archive is missing {key}")
    values = tuple(str(value) for value in archive[key].tolist())
    if not values or len(set(values)) != len(values):
        raise ValueError(f"runtime {key} must be non-empty and unique")
    return values


def _positive_sum(
    archive: np.lib.npyio.NpzFile, key: str
) -> tuple[float | None, bool]:
    if key not in archive:
        return None, False
    values = np.asarray(archive[key], dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"runtime {key} must be a finite non-negative vector")
    total = float(values.sum())
    return (total if total > 0.0 else None), bool(total > 0.0)


def load_runtime(path: Path) -> dict[str, Any]:
    """Load and summarize one immutable score/cost surface."""

    with np.load(path, allow_pickle=False) as archive:
        query_ids = _ids(archive, "query_ids")
        corpus_ids = _ids(archive, "corpus_ids")
        if "scores" not in archive:
            raise ValueError("runtime archive is missing scores")
        scores = np.asarray(archive["scores"], dtype=np.float64)
        if scores.shape != (len(query_ids), len(corpus_ids)):
            raise ValueError("runtime scores do not align with identifiers")
        if not np.isfinite(scores).all():
            raise ValueError("runtime scores contain non-finite values")
        storage, storage_available = _positive_sum(archive, "vector_bytes")
        build, per_item_build = _positive_sum(archive, "encode_ms")
        build_source = "runtime_npz_per_item_encode_ms"
        if build is None and "index_total_ms" in archive:
            aggregate = np.asarray(archive["index_total_ms"], dtype=np.float64)
            if (
                aggregate.ndim != 0
                or not np.isfinite(aggregate.item())
                or float(aggregate.item()) < 0.0
            ):
                raise ValueError(
                    "runtime index_total_ms must be a finite non-negative scalar"
                )
            if float(aggregate.item()) > 0.0:
                build = float(aggregate.item())
                build_source = "runtime_npz_aggregate_index_total_ms"
        if build is None:
            build_source = "runtime_npz_missing_build_measurement"
        reload_cost, per_item_reload = _positive_sum(archive, "reload_ms")
        keys = sorted(archive.files)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "query_ids": query_ids,
        "corpus_ids": corpus_ids,
        "scores": scores,
        "keys": keys,
        "cost": {
            "storage_bytes": int(storage) if storage_available else None,
            "storage_available": storage_available,
            "build_ms": build,
            "per_item_build_available": per_item_build,
            "reload_ms": reload_cost,
            "per_item_reload_available": per_item_reload,
            "source": build_source,
        },
    }


def load_relevance(
    path: Path, *, query_count: int, corpus_count: int
) -> tuple[dict[int, float], ...]:
    with np.load(path, allow_pickle=False) as labels:
        required = ("query_positions", "corpus_positions", "relevance")
        if any(key not in labels for key in required):
            raise ValueError("label archive is missing a required array")
        rows: list[dict[int, float]] = [dict() for _ in range(query_count)]
        for query, corpus, value in zip(
            labels["query_positions"],
            labels["corpus_positions"],
            labels["relevance"],
            strict=True,
        ):
            query_index = int(query)
            corpus_index = int(corpus)
            relevance = float(value)
            if not 0 <= query_index < query_count:
                raise ValueError("label query position is out of range")
            if not 0 <= corpus_index < corpus_count:
                raise ValueError("label corpus position is out of range")
            if not np.isfinite(relevance) or relevance <= 0:
                raise ValueError("relevance values must be finite and positive")
            rows[query_index][corpus_index] = relevance
    if any(not row for row in rows):
        raise ValueError("every query must have at least one relevance judgment")
    return tuple(rows)


def _merged_cost(
    measured: Mapping[str, Any], override: Mapping[str, Any] | None
) -> dict[str, Any]:
    result = dict(measured)
    if not override:
        return result
    allowed = {
        "storage_bytes",
        "build_ms",
        "reload_ms",
        "per_item_build_available",
        "per_item_reload_available",
        "source",
        "components",
        "warning",
    }
    unknown = set(override) - allowed
    if unknown:
        raise ValueError(f"unknown cost override fields: {sorted(unknown)}")
    result.update(override)
    result["storage_available"] = result.get("storage_bytes") is not None
    return result


def analyze_multilevel_bundle(
    route_paths: Mapping[str, Path],
    labels_path: Path,
    *,
    dataset: str,
    cost_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    activation_trace: Path | None = None,
    mixed_state_artifact: Path | None = None,
    bootstrap_seed: int = 20260805,
    bootstrap_resamples: int = 4000,
) -> dict[str, Any]:
    """Audit a bundle and run only its real uniform-route headroom analysis."""

    if tuple(route_paths) != REQUIRED_TIERS:
        raise ValueError(f"routes must be ordered exactly as {REQUIRED_TIERS}")
    runtimes = {tier: load_runtime(path) for tier, path in route_paths.items()}
    reference = runtimes[REQUIRED_TIERS[0]]
    for tier, runtime in runtimes.items():
        if runtime["query_ids"] != reference["query_ids"]:
            raise ValueError(f"{tier} query identifiers do not align")
        if runtime["corpus_ids"] != reference["corpus_ids"]:
            raise ValueError(f"{tier} corpus identifiers do not align")
    relevance = load_relevance(
        labels_path,
        query_count=len(reference["query_ids"]),
        corpus_count=len(reference["corpus_ids"]),
    )
    costs = {
        tier: _merged_cost(
            runtimes[tier]["cost"], (cost_overrides or {}).get(tier)
        )
        for tier in REQUIRED_TIERS
    }
    roles = np.asarray(deterministic_split_roles(reference["query_ids"]))
    fit = roles == "fit"
    evaluation = roles == "eval"
    metrics = {
        tier: query_metrics(runtime["scores"], relevance, ks=(10, 100))
        for tier, runtime in runtimes.items()
    }
    target = "ndcg_at_10"
    best_fixed = max(
        REQUIRED_TIERS,
        key=lambda tier: float(metrics[tier][target][fit].mean()),
    )
    stacked = np.stack([metrics[tier][target] for tier in REQUIRED_TIERS])
    oracle_routes = np.argmax(stacked, axis=0)
    positions = np.arange(len(reference["query_ids"]))
    oracle_ndcg = stacked[oracle_routes, positions]
    oracle_recall = np.stack(
        [metrics[tier]["recall_at_100"] for tier in REQUIRED_TIERS]
    )[oracle_routes, positions]
    selected_eval = oracle_routes[evaluation]
    selected_tiers = {
        tier
        for index, tier in enumerate(REQUIRED_TIERS)
        if np.any(selected_eval == index)
    }
    eager_union_storage = (
        sum(int(costs[tier]["storage_bytes"]) for tier in selected_tiers)
        if all(costs[tier].get("storage_bytes") is not None for tier in selected_tiers)
        else None
    )

    uniform = {}
    for tier in REQUIRED_TIERS:
        uniform[tier] = {
            "fit": {
                "ndcg_at_10": float(metrics[tier][target][fit].mean()),
                "recall_at_100": float(
                    metrics[tier]["recall_at_100"][fit].mean()
                ),
            },
            "eval": {
                "ndcg_at_10": float(metrics[tier][target][evaluation].mean()),
                "recall_at_100": float(
                    metrics[tier]["recall_at_100"][evaluation].mean()
                ),
            },
            "cost": costs[tier],
        }

    prerequisites = {
        "aligned_real_three_tier_score_surfaces": True,
        "heldout_relevance": True,
        "score_comparable_mixed_state_execution": bool(
            mixed_state_artifact is not None and mixed_state_artifact.is_file()
        ),
        "query_item_activation_trace": bool(
            activation_trace is not None and activation_trace.is_file()
        ),
        "per_item_build_all_tiers": all(
            bool(costs[tier].get("per_item_build_available"))
            for tier in REQUIRED_TIERS
        ),
        "per_item_reload_all_tiers": all(
            bool(costs[tier].get("per_item_reload_available"))
            for tier in REQUIRED_TIERS
        ),
        "storage_all_tiers": all(
            costs[tier].get("storage_bytes") is not None for tier in REQUIRED_TIERS
        ),
    }
    missing = [name for name, available in prerequisites.items() if not available]
    deployable = not missing
    blocked_reason = (
        "common dynamic cost/quality contract is incomplete: " + ", ".join(missing)
    )
    blocked_baselines = (
        "static_content_rule",
        "lru",
        "lfu",
        "gdsf",
        "transient_refine",
        "oracle_physical_design",
    )
    baselines: dict[str, Any] = {
        f"uniform_{tier}": {
            "status": "run_real_surface",
            "result": uniform[tier],
        }
        for tier in REQUIRED_TIERS
    }
    baselines["full_eager"] = {
        "status": "run_real_surface",
        "equivalent_to": "uniform_full_multivector",
        "result": uniform["full_multivector"],
    }
    baselines.update(
        {
            name: {
                "status": "not_run_missing_artifact",
                "reason": blocked_reason,
            }
            for name in blocked_baselines
        }
    )

    artifact_report = {
        tier: {
            "path": runtime["path"],
            "sha256": runtime["sha256"],
            "keys": runtime["keys"],
            "score_shape": list(runtime["scores"].shape),
            "cost": costs[tier],
        }
        for tier, runtime in runtimes.items()
    }
    return {
        "schema_version": 1,
        "stage": "preregistered_multilevel_artifact_audit_and_headroom",
        "dataset": dataset,
        "queries": len(reference["query_ids"]),
        "corpus": len(reference["corpus_ids"]),
        "split": {
            "method": "stable_sha256_query_id_two_thirds_fit_one_third_eval",
            "fit_queries": int(fit.sum()),
            "eval_queries": int(evaluation.sum()),
        },
        "artifacts": {
            "tiers": artifact_report,
            "labels": {
                "path": str(labels_path),
                "sha256": sha256_file(labels_path),
            },
            "activation_trace": (
                None
                if activation_trace is None
                else {
                    "path": str(activation_trace),
                    "sha256": sha256_file(activation_trace),
                }
            ),
            "mixed_state_artifact": (
                None
                if mixed_state_artifact is None
                else {
                    "path": str(mixed_state_artifact),
                    "sha256": sha256_file(mixed_state_artifact),
                }
            ),
        },
        "capability_matrix": prerequisites,
        "deployable_three_state_replay_ready": deployable,
        "uniform_tier_selected_on_fit": best_fixed,
        "uniform_tier_results": uniform,
        "diagnostic_query_route_oracle": {
            "uses_eval_qrels": True,
            "deployable": False,
            "eval_ndcg_at_10": float(oracle_ndcg[evaluation].mean()),
            "eval_recall_at_100_using_ndcg_selected_route": float(
                oracle_recall[evaluation].mean()
            ),
            "selected_routes": {
                tier: int(np.sum(selected_eval == index))
                for index, tier in enumerate(REQUIRED_TIERS)
            },
            "gap_over_fit_selected_uniform": paired_bootstrap_ci(
                oracle_ndcg[evaluation],
                metrics[best_fixed][target][evaluation],
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
            "eager_union_storage_bytes": eager_union_storage,
            "warning": (
                "This chooses a globally uniform real tier separately for each "
                "evaluation query. It measures quality heterogeneity only and "
                "does not construct a mixed document score surface."
            ),
        },
        "baseline_registry": baselines,
        "planning_action_registry": {
            "stay": {
                "status": "measured_at_uniform_tier_granularity",
                "evidence": "uniform_tier_results",
            },
            "switch_or_upgrade": {
                "status": "not_run_missing_artifact",
                "reason": blocked_reason,
            },
            "retain": {
                "status": "not_run_missing_artifact",
                "reason": blocked_reason,
            },
            "evict": {
                "status": "not_run_missing_artifact",
                "reason": blocked_reason,
            },
        },
        "quality_work_storage_latency": {
            "quality": "real held-out nDCG@10 and Recall@100",
            "work": "reported only where measured build fields exist",
            "storage": "reported from runtime or explicit same-domain override",
            "latency": "unavailable without same-workload reload measurements",
        },
        "decision": (
            "READY-for-dynamic-replay" if deployable else "NO-GO-current-artifacts"
        ),
        "smallest_next_measurement": (
            []
            if deployable
            else [
                "persist a score-comparable query-scoped mixed-state outcome surface",
                "persist the corresponding query-to-item activation stream",
                "measure per-item compact construction and full/compact reload on the same hardware",
            ]
        ),
        "interpretation_guardrails": {
            "no_synthetic_intermediate_quality": True,
            "no_cross_dataset_latency_transplant": True,
            "query_route_oracle_is_not_a_physical_compiler": True,
        },
    }
