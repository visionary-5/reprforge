#!/usr/bin/env python3
"""Replay elastic visual-representation retention on frozen ViDoRe traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.candidate_fusion import _candidate_ndcg
from reprforge.elastic_representation_cache import (
    offline_oracle,
    replay_capacity_cache,
    replay_elastic_cache,
)
from reprforge.progressive_oracle import load_trace, rank_order, validate_pair


DEFAULT_PRICES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DEFAULT_SHUFFLE_SEEDS = tuple(range(10))
DEFAULT_CAPACITY_FRACTIONS = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
DEFAULT_RANDOMIZED_SEEDS = tuple(range(5))


def _requests(order: np.ndarray, candidate_k: int) -> list[list[int]]:
    return [
        [int(value) for value in row[:candidate_k]]
        for row in order
    ]


def _price_result(
    requests: Sequence[Sequence[int]],
    encode_ms: np.ndarray,
    vector_bytes: np.ndarray,
    *,
    price: float,
) -> dict[str, Any]:
    holding = vector_bytes.astype(np.float64) / (1024.0 * 1024.0) * price
    oracle = offline_oracle(requests, encode_ms, holding)
    policies = {
        policy: replay_elastic_cache(
            requests,
            encode_ms,
            holding,
            policy=policy,
        )
        for policy in (
            "no_cache",
            "resident",
            "ski_ttl",
            "verified_ski_ttl",
        )
    }
    oracle_cost = oracle.total_cost
    return {
        "memory_price_ms_per_mib_query_interval": price,
        "median_break_even_intervals": float(
            np.median(
                np.divide(
                    encode_ms,
                    holding,
                    out=np.full_like(encode_ms, np.inf),
                    where=holding > 0,
                )
            )
        ),
        "offline_oracle": oracle.to_dict(),
        "policies": {
            name: {
                **result.to_dict(),
                "cost_over_oracle": (
                    result.total_cost / oracle_cost if oracle_cost else 1.0
                ),
            }
            for name, result in policies.items()
        },
    }


def _aggregate_capacity_runs(results: Sequence[Any]) -> dict[str, Any]:
    dictionaries = [result.to_dict() for result in results]
    numeric_fields = (
        "build_cost",
        "holding_cost",
        "total_cost",
        "cache_hits",
        "cache_misses",
        "hit_fraction",
        "peak_resident_items",
        "peak_resident_bytes",
        "resident_byte_intervals",
        "final_resident_items",
        "final_resident_bytes",
    )
    return {
        "runs": len(dictionaries),
        **{
            field: {
                "mean": float(np.mean([row[field] for row in dictionaries])),
                "min": float(np.min([row[field] for row in dictionaries])),
                "max": float(np.max([row[field] for row in dictionaries])),
            }
            for field in numeric_fields
        },
    }


def _capacity_sweep(
    requests: Sequence[Sequence[int]],
    encode_ms: np.ndarray,
    vector_bytes: np.ndarray,
    *,
    prices: Sequence[float],
    capacity_fractions: Sequence[float],
    randomized_seeds: Sequence[int],
    include_curves: bool,
) -> list[dict[str, Any]]:
    full_bytes = int(vector_bytes.sum())
    families = {
        "lru_fixed": ("lru", "none"),
        "gdsf_fixed": ("gdsf", "none"),
        "gdsf_breakeven": ("gdsf", "breakeven"),
        "gdsf_randomized": ("gdsf", "randomized"),
        "gdsf_verified_breakeven": ("gdsf", "verified_breakeven"),
    }
    price_rows: list[dict[str, Any]] = []
    for price in prices:
        holding = (
            vector_bytes.astype(np.float64)
            / (1024.0 * 1024.0)
            * float(price)
        )
        curves: dict[str, list[dict[str, Any]]] = {
            family: [] for family in families
        }
        for fraction in capacity_fractions:
            capacity = max(1, int(round(full_bytes * float(fraction))))
            for family, (eviction, ttl) in families.items():
                seeds = (
                    tuple(randomized_seeds)
                    if ttl == "randomized"
                    else (0,)
                )
                runs = [
                    replay_capacity_cache(
                        requests,
                        encode_ms,
                        holding,
                        vector_bytes,
                        capacity_bytes=capacity,
                        eviction_policy=eviction,
                        ttl_policy=ttl,
                        random_seed=seed,
                    )
                    for seed in seeds
                ]
                curves[family].append(
                    {
                        "capacity_fraction_of_full_visual": float(fraction),
                        "capacity_bytes": capacity,
                        **_aggregate_capacity_runs(runs),
                    }
                )

        best = {
            family: min(
                rows,
                key=lambda row: (
                    row["total_cost"]["mean"],
                    row["capacity_bytes"],
                ),
            )
            for family, rows in curves.items()
        }
        published_names = (
            "lru_fixed",
            "gdsf_fixed",
            "gdsf_breakeven",
            "gdsf_randomized",
        )
        published_name = min(
            published_names,
            key=lambda name: best[name]["total_cost"]["mean"],
        )
        proposed_name = "gdsf_verified_breakeven"
        published_cost = best[published_name]["total_cost"]["mean"]
        proposed_cost = best[proposed_name]["total_cost"]["mean"]
        no_cache_cost = float(
            sum(float(encode_ms[item]) for batch in requests for item in batch)
        )
        strongest_name, strongest_cost = min(
            (
                (published_name, published_cost),
                ("no_cache", no_cache_cost),
            ),
            key=lambda value: value[1],
        )
        price_row = {
            "memory_price_ms_per_mib_query_interval": float(price),
            "capacity_tuning_warning": (
                "best points are post-hoc lower envelopes over the frozen "
                "trace, not deployable capacity choices"
            ),
            "best_by_family": best,
            "strongest_published_family": published_name,
            "strongest_published_total_cost": published_cost,
            "no_cache_total_cost": no_cache_cost,
            "strongest_deployment_baseline": strongest_name,
            "strongest_deployment_baseline_total_cost": strongest_cost,
            "reprforge_family": proposed_name,
            "reprforge_total_cost": proposed_cost,
            "reprforge_gain_over_strongest_published": (
                (published_cost - proposed_cost) / published_cost
                if published_cost
                else 0.0
            ),
            "reprforge_gain_over_strongest_deployment_baseline": (
                (strongest_cost - proposed_cost) / strongest_cost
                if strongest_cost
                else 0.0
            ),
        }
        if include_curves:
            price_row["curves"] = curves
        price_rows.append(price_row)
    return price_rows


def _strong_baseline_shuffle_summary(
    requests: Sequence[Sequence[int]],
    encode_ms: np.ndarray,
    vector_bytes: np.ndarray,
    *,
    prices: Sequence[float],
    capacity_fractions: Sequence[float],
    randomized_seeds: Sequence[int],
    shuffle_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    by_price: list[list[dict[str, Any]]] = [[] for _ in prices]
    for seed in shuffle_seeds:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(requests))
        rows = _capacity_sweep(
            [requests[int(index)] for index in permutation],
            encode_ms,
            vector_bytes,
            prices=prices,
            capacity_fractions=capacity_fractions,
            randomized_seeds=randomized_seeds,
            include_curves=False,
        )
        for index, row in enumerate(rows):
            by_price[index].append(
                {
                    "shuffle_seed": int(seed),
                    "strongest_deployment_baseline": row[
                        "strongest_deployment_baseline"
                    ],
                    "strongest_deployment_baseline_total_cost": row[
                        "strongest_deployment_baseline_total_cost"
                    ],
                    "reprforge_total_cost": row["reprforge_total_cost"],
                    "reprforge_gain": row[
                        "reprforge_gain_over_strongest_deployment_baseline"
                    ],
                }
            )
    summaries: list[dict[str, Any]] = []
    for price, rows in zip(prices, by_price):
        gains = np.asarray(
            [row["reprforge_gain"] for row in rows], dtype=np.float64
        )
        baseline_counts: dict[str, int] = {}
        for row in rows:
            name = str(row["strongest_deployment_baseline"])
            baseline_counts[name] = baseline_counts.get(name, 0) + 1
        summaries.append(
            {
                "memory_price_ms_per_mib_query_interval": float(price),
                "shuffle_count": len(rows),
                "strongest_baseline_counts": baseline_counts,
                "reprforge_positive_count": int(np.sum(gains > 0)),
                "reprforge_gain": {
                    "mean": float(gains.mean()),
                    "min": float(gains.min()),
                    "max": float(gains.max()),
                }
                if len(gains)
                else None,
                "per_seed": rows,
            }
        )
    return summaries


def analyze_dataset(
    name: str,
    text_root: Path,
    visual_root: Path,
    *,
    candidate_k: int,
    cutoff: int,
    prices: Sequence[float],
    shuffle_seeds: Sequence[int],
    capacity_fractions: Sequence[float],
    randomized_seeds: Sequence[int],
    include_capacity_curves: bool,
    strong_shuffle_seeds: Sequence[int],
) -> dict[str, Any]:
    text = load_trace(text_root)
    visual = load_trace(visual_root)
    qrels = validate_pair(text, visual)
    order = rank_order(text.scores, text.corpus_ids)
    requests = _requests(order, candidate_k)
    candidate_quality = _candidate_ndcg(
        text,
        visual,
        qrels,
        order,
        candidate_k=candidate_k,
        method="zscore_sum",
        cutoff=cutoff,
    )

    original = [
        _price_result(
            requests,
            visual.encode_ms,
            visual.vector_bytes,
            price=float(price),
        )
        for price in prices
    ]
    shuffled_rows: list[list[dict[str, Any]]] = [
        [] for _ in prices
    ]
    for seed in shuffle_seeds:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(requests))
        seed_rows = [
            _price_result(
                [requests[int(index)] for index in permutation],
                visual.encode_ms,
                visual.vector_bytes,
                price=float(price),
            )
            for price in prices
        ]
        for index, row in enumerate(seed_rows):
            shuffled_rows[index].append(row)

    shuffle_summary: list[dict[str, Any]] = []
    for price, rows in zip(prices, shuffled_rows):
        policy_names = tuple(rows[0]["policies"]) if rows else ()
        policy_costs = {
            policy: np.asarray(
                [row["policies"][policy]["total_cost"] for row in rows],
                dtype=np.float64,
            )
            for policy in policy_names
        }
        fixed = np.minimum(
            policy_costs.get("no_cache", np.asarray([])),
            policy_costs.get("resident", np.asarray([])),
        )
        verified = policy_costs.get("verified_ski_ttl", np.asarray([]))
        gains = (
            (fixed - verified) / fixed
            if len(fixed)
            else np.asarray([], dtype=np.float64)
        )
        shuffle_summary.append(
            {
                "memory_price_ms_per_mib_query_interval": float(price),
                "shuffle_count": len(rows),
                "policy_total_cost": {
                    policy: {
                        "mean": float(values.mean()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                    for policy, values in policy_costs.items()
                },
                "verified_gain_over_better_fixed": {
                    "mean": float(gains.mean()),
                    "min": float(gains.min()),
                    "max": float(gains.max()),
                }
                if len(gains)
                else None,
            }
        )

    unique_pages = len({page for batch in requests for page in batch})
    return {
        "dataset": name,
        "trace_identity": {
            "source_sha256": text.manifest["source_sha256"],
            "text_runtime_sha256": text.manifest["runtime_sha256"],
            "visual_runtime_sha256": visual.manifest["runtime_sha256"],
        },
        "query_count": int(text.scores.shape[0]),
        "corpus_pages": int(text.scores.shape[1]),
        "candidate_k": candidate_k,
        "candidate_events": int(sum(len(batch) for batch in requests)),
        "unique_candidate_pages": unique_pages,
        "candidate_coverage_fraction": unique_pages / text.scores.shape[1],
        "quality_contract": {
            "definition": (
                "BM25 Top-K candidate-relative z-score fusion; retention policy "
                "cannot change scores because every requested candidate is "
                "visually refined before ranking"
            ),
            f"mean_ndcg@{cutoff}": float(candidate_quality.mean()),
        },
        "measured_physical_inputs": {
            "visual_encode_ms_sum_all_pages": float(visual.encode_ms.sum()),
            "visual_encode_ms_mean_page": float(visual.encode_ms.mean()),
            "visual_vector_bytes_sum_all_pages": int(visual.vector_bytes.sum()),
            "visual_vector_bytes_mean_page": float(visual.vector_bytes.mean()),
        },
        "original_query_order": original,
        "capacity_constrained_strong_baselines": _capacity_sweep(
            requests,
            visual.encode_ms,
            visual.vector_bytes,
            prices=prices,
            capacity_fractions=capacity_fractions,
            randomized_seeds=randomized_seeds,
            include_curves=include_capacity_curves,
        ),
        "capacity_constrained_strong_baseline_order_sensitivity": (
            _strong_baseline_shuffle_summary(
                requests,
                visual.encode_ms,
                visual.vector_bytes,
                prices=prices,
                capacity_fractions=capacity_fractions,
                randomized_seeds=randomized_seeds,
                shuffle_seeds=strong_shuffle_seeds,
            )
            if strong_shuffle_seeds
            else []
        ),
        "deterministic_query_order_shuffle_summary": shuffle_summary,
        "limitations": [
            "ViDoRe query order is a benchmark batch, not a production arrival trace.",
            "Per-page encode_ms is an additive batched-equivalent work estimate; it does not model every batching or overlap effect.",
            "The memory price is a sensitivity parameter, not a measured cloud billing rate.",
            "This experiment isolates representation retention after candidate activation; it does not solve candidate selection.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("NAME", "TEXT_TRACE", "VISUAL_TRACE"),
        required=True,
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument(
        "--prices",
        type=float,
        nargs="+",
        default=list(DEFAULT_PRICES),
    )
    parser.add_argument(
        "--shuffle-seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_SHUFFLE_SEEDS),
    )
    parser.add_argument(
        "--capacity-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_CAPACITY_FRACTIONS),
    )
    parser.add_argument(
        "--randomized-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANDOMIZED_SEEDS),
    )
    parser.add_argument(
        "--include-capacity-curves",
        action="store_true",
        help="Store every capacity point instead of only family envelopes.",
    )
    parser.add_argument(
        "--strong-shuffle-seeds",
        type=int,
        nargs="*",
        default=[],
        help="Also rerun capacity-aware baselines on these query permutations.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_k < args.cutoff:
        raise ValueError("candidate-k must be at least cutoff")
    if any(price < 0 for price in args.prices):
        raise ValueError("memory prices must be non-negative")
    if any(value <= 0 or value > 1 for value in args.capacity_fractions):
        raise ValueError("capacity fractions must lie in (0, 1]")
    result = {
        "schema_version": 2,
        "mechanism": "two-slope elastic representation retention",
        "datasets": [
            analyze_dataset(
                name,
                Path(text_root),
                Path(visual_root),
                candidate_k=args.candidate_k,
                cutoff=args.cutoff,
                prices=args.prices,
                shuffle_seeds=args.shuffle_seeds,
                capacity_fractions=args.capacity_fractions,
                randomized_seeds=args.randomized_seeds,
                include_capacity_curves=args.include_capacity_curves,
                strong_shuffle_seeds=args.strong_shuffle_seeds,
            )
            for name, text_root, visual_root in args.dataset
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "datasets": [
                    {
                        "dataset": dataset["dataset"],
                        "queries": dataset["query_count"],
                        "pages": dataset["corpus_pages"],
                        "quality": dataset["quality_contract"],
                    }
                    for dataset in result["datasets"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
