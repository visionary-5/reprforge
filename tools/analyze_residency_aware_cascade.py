#!/usr/bin/env python3
"""Replay cache-residency-aware cascade depths on real candidate streams."""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEPTHS = (20, 50, 100)
CAPACITY_FRACTIONS = (0.05, 0.10, 0.20)
MISS_BUDGETS = (10, 15, 20, 25, 30, 40, 50)
MAX_DEPTHS = (50, 100)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_domain(value: str) -> tuple[str, Path, Path]:
    if "=" not in value or "," not in value:
        raise argparse.ArgumentTypeError("domain must be NAME=FAILURE_JSON,HPOOL_RANKING")
    name, paths = value.split("=", 1)
    failure, ranking = paths.split(",", 1)
    if not name or not failure or not ranking:
        raise argparse.ArgumentTypeError("domain must be NAME=FAILURE_JSON,HPOOL_RANKING")
    return name, Path(failure), Path(ranking)


def _query_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def _load_ranking(path: Path) -> dict[str, list[str]]:
    ranking: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"expected 3 tab fields at {path}:{line_number}")
            query_id, document_id, _ = fields
            ranking.setdefault(query_id, []).append(document_id)
    if any(len(rows) < 100 for rows in ranking.values()):
        raise ValueError(f"{path} contains a ranking shorter than 100")
    if any(len(set(rows[:100])) != 100 for rows in ranking.values()):
        raise ValueError(f"{path} contains duplicate Top-100 documents")
    return ranking


def _load_domain(name: str, failure_path: Path, ranking_path: Path) -> dict[str, Any]:
    failure = json.loads(failure_path.read_text())
    rows = failure.get("per_query", [])
    corpus_pages = int(failure.get("analysis_scope", {}).get("corpus_pages", 0))
    ranking = _load_ranking(ranking_path)
    quality = {
        str(row["query_id"]): {
            depth: float(row["ndcg_at_10"][f"cascade{depth}"]) for depth in DEPTHS
        }
        for row in rows
    }
    if not rows or corpus_pages <= 100 or set(ranking) != set(quality):
        raise ValueError(f"unaligned or incomplete domain artifacts for {name}")
    return {
        "name": name,
        "corpus_pages": corpus_pages,
        "ranking": ranking,
        "quality": quality,
        "query_ids": sorted(ranking, key=_query_sort_key),
        "failure_sha256": _sha256(failure_path),
        "ranking_sha256": _sha256(ranking_path),
    }


class LruCache:
    def __init__(self, capacity: int):
        if capacity < 100:
            raise ValueError("capacity must hold at least one Top-100 cohort")
        self.capacity = capacity
        self._items: OrderedDict[str, None] = OrderedDict()

    def misses(self, items: Iterable[str]) -> int:
        return sum(item not in self._items for item in items)

    def access(self, items: Iterable[str]) -> int:
        misses = 0
        for item in items:
            if item in self._items:
                self._items.move_to_end(item)
            else:
                misses += 1
                self._items[item] = None
                if len(self._items) > self.capacity:
                    self._items.popitem(last=False)
        return misses


class LfuCache:
    """Online LFU with recency tie-breaking and global observed frequencies."""

    def __init__(self, capacity: int):
        if capacity < 100:
            raise ValueError("capacity must hold at least one Top-100 cohort")
        self.capacity = capacity
        self._resident: set[str] = set()
        self._frequency: Counter[str] = Counter()
        self._last_access: dict[str, int] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._clock = 0

    def misses(self, items: Iterable[str]) -> int:
        return sum(item not in self._resident for item in items)

    def access(self, items: Iterable[str]) -> int:
        misses = 0
        for item in items:
            self._clock += 1
            self._frequency[item] += 1
            self._last_access[item] = self._clock
            if item not in self._resident:
                misses += 1
                if len(self._resident) >= self.capacity:
                    while self._heap:
                        frequency, last_access, victim = heapq.heappop(self._heap)
                        if (
                            victim in self._resident
                            and frequency == self._frequency[victim]
                            and last_access == self._last_access[victim]
                        ):
                            self._resident.remove(victim)
                            break
                self._resident.add(item)
            heapq.heappush(
                self._heap,
                (self._frequency[item], self._last_access[item], item),
            )
        return misses


def _belady_misses(items: list[str], capacity: int) -> int:
    """Return the future-aware optimal miss count for one fixed request trace."""
    positions: dict[str, list[int]] = defaultdict(list)
    for position, item in enumerate(items):
        positions[item].append(position)
    cursor = {item: 0 for item in positions}
    resident: set[str] = set()
    heap: list[tuple[int, str]] = []
    misses = 0
    never = len(items) + 1
    for item in items:
        cursor[item] += 1
        next_use = (
            positions[item][cursor[item]]
            if cursor[item] < len(positions[item])
            else never
        )
        if item not in resident:
            misses += 1
            if len(resident) >= capacity:
                while heap:
                    negative_use, victim = heapq.heappop(heap)
                    victim_next = (
                        positions[victim][cursor[victim]]
                        if cursor[victim] < len(positions[victim])
                        else never
                    )
                    if victim in resident and -negative_use == victim_next:
                        resident.remove(victim)
                        break
            resident.add(item)
        heapq.heappush(heap, (-next_use, item))
    return misses


def _replay_actions(
    domain: dict[str, Any],
    query_ids: list[str],
    capacity: int,
    actions: list[int],
    *,
    cache_policy: str = "lru",
) -> dict[str, Any]:
    cache = LruCache(capacity) if cache_policy == "lru" else LfuCache(capacity)
    misses = 0
    quality = 0.0
    for query_id, depth in zip(query_ids, actions, strict=True):
        misses += cache.access(domain["ranking"][query_id][:depth])
        quality += domain["quality"][query_id][depth]
    return {
        "mean_cold_page_misses": misses / len(query_ids),
        "mean_ndcg_at_10": quality / len(query_ids),
        "mean_depth": float(np.mean(actions)),
    }


def _replay_state_policy(
    domain: dict[str, Any],
    query_ids: list[str],
    capacity: int,
    miss_budget: int,
    *,
    max_depth: int = 100,
) -> tuple[dict[str, Any], list[int]]:
    if max_depth not in MAX_DEPTHS:
        raise ValueError(f"unsupported maximum depth {max_depth}")
    cache = LruCache(capacity)
    misses = 0
    quality = 0.0
    actions = []
    for query_id in query_ids:
        candidates = domain["ranking"][query_id]
        depth = 20
        for possible_depth in tuple(depth for depth in (50, 100) if depth <= max_depth):
            if cache.misses(candidates[:possible_depth]) <= miss_budget:
                depth = possible_depth
        misses += cache.access(candidates[:depth])
        quality += domain["quality"][query_id][depth]
        actions.append(depth)
    return (
        {
            "mean_cold_page_misses": misses / len(query_ids),
            "mean_ndcg_at_10": quality / len(query_ids),
            "mean_depth": float(np.mean(actions)),
            "action_counts": dict(sorted(Counter(map(str, actions)).items())),
        },
        actions,
    )


def _interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    lower, upper = np.quantile(array, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "order_q025": float(lower),
        "order_q975": float(upper),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _reuse_summary(domain: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for depth in DEPTHS:
        events = [
            document_id
            for query_id in domain["query_ids"]
            for document_id in domain["ranking"][query_id][:depth]
        ]
        frequencies = Counter(events)
        output[str(depth)] = {
            "page_events": len(events),
            "unique_pages": len(frequencies),
            "unique_corpus_fraction": len(frequencies) / domain["corpus_pages"],
            "event_reuse_fraction": 1.0 - len(frequencies) / len(events),
            "maximum_query_frequency": max(frequencies.values()),
        }
    return output


def _analyze_configuration(
    domain: dict[str, Any],
    capacity: int,
    miss_budget: int,
    max_depth: int,
    *,
    arrival_orders: int,
    assignment_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    adaptive_rows = []
    random_rows = []
    miss_advantages = []
    quality_advantages = []
    for order_index in range(arrival_orders):
        query_ids = list(domain["query_ids"])
        np.random.default_rng(seed + order_index).shuffle(query_ids)
        adaptive, actions = _replay_state_policy(
            domain, query_ids, capacity, miss_budget, max_depth=max_depth
        )
        shuffled = []
        for assignment_index in range(assignment_shuffles):
            assigned = list(actions)
            np.random.default_rng(
                seed + 100_000 * order_index + assignment_index
            ).shuffle(assigned)
            shuffled.append(_replay_actions(domain, query_ids, capacity, assigned))
        random_mean_misses = float(
            np.mean([row["mean_cold_page_misses"] for row in shuffled])
        )
        random_mean_quality = float(
            np.mean([row["mean_ndcg_at_10"] for row in shuffled])
        )
        adaptive_rows.append(adaptive)
        random_rows.append(
            {
                "mean_cold_page_misses": random_mean_misses,
                "mean_ndcg_at_10": random_mean_quality,
                "mean_depth": adaptive["mean_depth"],
            }
        )
        miss_advantages.append(
            random_mean_misses - adaptive["mean_cold_page_misses"]
        )
        quality_advantages.append(
            adaptive["mean_ndcg_at_10"] - random_mean_quality
        )
    mean_adaptive_misses = float(
        np.mean([row["mean_cold_page_misses"] for row in adaptive_rows])
    )
    mean_random_misses = float(
        np.mean([row["mean_cold_page_misses"] for row in random_rows])
    )
    return {
        "policy": "deepest_prefix_within_current_cold_miss_budget",
        "policy_uses_qrels": False,
        "miss_budget": miss_budget,
        "maximum_depth": max_depth,
        "adaptive": {
            "mean_cold_page_misses": mean_adaptive_misses,
            "mean_ndcg_at_10": float(
                np.mean([row["mean_ndcg_at_10"] for row in adaptive_rows])
            ),
            "mean_depth": float(np.mean([row["mean_depth"] for row in adaptive_rows])),
        },
        "same_action_multiset_random_assignment": {
            "mean_cold_page_misses": mean_random_misses,
            "mean_ndcg_at_10": float(
                np.mean([row["mean_ndcg_at_10"] for row in random_rows])
            ),
            "mean_depth": float(np.mean([row["mean_depth"] for row in random_rows])),
        },
        "adaptive_minus_random": {
            "relative_cold_miss_reduction": (
                (mean_random_misses - mean_adaptive_misses) / mean_random_misses
            ),
            "cold_miss_advantage_across_orders": _interval(miss_advantages),
            "ndcg_advantage_across_orders": _interval(quality_advantages),
            "orders_with_fewer_cold_misses_fraction": float(
                np.mean(np.asarray(miss_advantages) > 0.0)
            ),
            "orders_with_higher_ndcg_fraction": float(
                np.mean(np.asarray(quality_advantages) > 0.0)
            ),
        },
    }


def analyze(
    domains: list[dict[str, Any]],
    *,
    arrival_orders: int,
    assignment_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "residency-aware-cascade-random-arrival-replay-v1",
        "cost_unit": "LRU cold Full-page misses; page sizes and latency are not measured",
        "arrival_semantics": (
            "Benchmark queries have no timestamps. Results aggregate deterministic random "
            "query permutations and establish order sensitivity, not natural temporal realism."
        ),
        "arrival_orders": arrival_orders,
        "same_action_assignment_shuffles_per_order": assignment_shuffles,
        "seed": seed,
        "domains": {},
    }
    for domain in domains:
        domain_output: dict[str, Any] = {
            "queries": len(domain["query_ids"]),
            "corpus_pages": domain["corpus_pages"],
            "input_sha256": {
                "failure_analysis": domain["failure_sha256"],
                "hpool_ranking": domain["ranking_sha256"],
            },
            "candidate_reuse": _reuse_summary(domain),
            "capacities": {},
        }
        for fraction in CAPACITY_FRACTIONS:
            capacity = int(round(domain["corpus_pages"] * fraction))
            fixed_by_policy = {
                policy: {str(depth): [] for depth in DEPTHS}
                for policy in ("lru", "lfu", "belady_oracle")
            }
            orders = []
            for order_index in range(arrival_orders):
                query_ids = list(domain["query_ids"])
                np.random.default_rng(seed + order_index).shuffle(query_ids)
                orders.append(query_ids)
                for depth in DEPTHS:
                    for policy in ("lru", "lfu"):
                        fixed_by_policy[policy][str(depth)].append(
                            _replay_actions(
                                domain,
                                query_ids,
                                capacity,
                                [depth] * len(query_ids),
                                cache_policy=policy,
                            )
                        )
                    sequence = [
                        document_id
                        for query_id in query_ids
                        for document_id in domain["ranking"][query_id][:depth]
                    ]
                    fixed_by_policy["belady_oracle"][str(depth)].append(
                        {
                            "mean_cold_page_misses": _belady_misses(sequence, capacity)
                            / len(query_ids),
                            "mean_ndcg_at_10": float(
                                np.mean(
                                    [
                                        domain["quality"][query_id][depth]
                                        for query_id in query_ids
                                    ]
                                )
                            ),
                        }
                    )
            fixed = {
                policy: {
                    depth: {
                        "mean_cold_page_misses": float(
                            np.mean([row["mean_cold_page_misses"] for row in rows])
                        ),
                        "mean_ndcg_at_10": float(
                            np.mean([row["mean_ndcg_at_10"] for row in rows])
                        ),
                    }
                    for depth, rows in by_depth.items()
                }
                for policy, by_depth in fixed_by_policy.items()
            }
            configurations = [
                _analyze_configuration(
                    domain,
                    capacity,
                    budget,
                    max_depth,
                    arrival_orders=arrival_orders,
                    assignment_shuffles=assignment_shuffles,
                    seed=seed,
                )
                for max_depth in MAX_DEPTHS
                for budget in MISS_BUDGETS
            ]
            domain_output["capacities"][str(fraction)] = {
                "capacity_pages": capacity,
                "fixed": fixed,
                "state_aware": configurations,
            }
        output["domains"][domain["name"]] = domain_output
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", action="append", type=_parse_domain, required=True)
    parser.add_argument("--arrival-orders", type=int, default=50)
    parser.add_argument("--assignment-shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.arrival_orders <= 0 or args.assignment_shuffles <= 0:
        parser.error("replay counts must be positive")
    domains = [_load_domain(*value) for value in args.domain]
    result = analyze(
        domains,
        arrival_orders=args.arrival_orders,
        assignment_shuffles=args.assignment_shuffles,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
