"""Frozen RBRC v0 compiler and logical-cache replay primitives.

RBRC v0 deliberately certifies *programs* on calibration workloads.  It does
not claim a formal, per-query safety guarantee.  A certified guard chooses the
reference depth when it fits the declared cold-access budget and otherwise
uses its compiled floor.  If no non-reference program is certified, the
compiler abstains to the reference program.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DomainSurface:
    """Candidate stream and relevance quality at every declared depth."""

    name: str
    corpus_pages: int
    query_ids: tuple[str, ...]
    ranking: Mapping[str, Sequence[str]]
    quality: Mapping[str, Mapping[int, float]]
    input_sha256: Mapping[str, str]

    def validate(self, depths: Sequence[int]) -> None:
        if self.corpus_pages <= max(depths):
            raise ValueError(f"{self.name}: corpus is not larger than plan ceiling")
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError(f"{self.name}: query IDs must be non-empty and unique")
        if set(self.query_ids) != set(self.ranking) or set(self.query_ids) != set(
            self.quality
        ):
            raise ValueError(f"{self.name}: ranking/quality/query IDs are unaligned")
        for query_id in self.query_ids:
            candidates = list(self.ranking[query_id])
            if len(candidates) < max(depths):
                raise ValueError(f"{self.name}/{query_id}: candidate ranking too short")
            if len(set(candidates[: max(depths)])) != max(depths):
                raise ValueError(f"{self.name}/{query_id}: duplicate candidate pages")
            if any(depth not in self.quality[query_id] for depth in depths):
                raise ValueError(f"{self.name}/{query_id}: missing quality depth")


class LogicalCache:
    def misses(self, items: Iterable[str]) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def access(self, items: Iterable[str]) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class LruCache(LogicalCache):
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
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


class LfuCache(LogicalCache):
    """Online LFU with recency tie breaking."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._resident: set[str] = set()
        self._frequency: Counter[str] = Counter()
        self._last: dict[str, int] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._clock = 0

    def misses(self, items: Iterable[str]) -> int:
        return sum(item not in self._resident for item in items)

    def access(self, items: Iterable[str]) -> int:
        misses = 0
        for item in items:
            self._clock += 1
            self._frequency[item] += 1
            self._last[item] = self._clock
            if item not in self._resident:
                misses += 1
                if len(self._resident) >= self.capacity:
                    while self._heap:
                        frequency, last, victim = heapq.heappop(self._heap)
                        if (
                            victim in self._resident
                            and frequency == self._frequency[victim]
                            and last == self._last[victim]
                        ):
                            self._resident.remove(victim)
                            break
                self._resident.add(item)
            heapq.heappush(
                self._heap, (self._frequency[item], self._last[item], item)
            )
        return misses


class GdsfCache(LogicalCache):
    """GDSF for equal-size, equal-read-cost logical pages.

    Physical-byte experiments replace unit size and cost after the blind
    logical gate.  Under the v0 logical contract the priority is
    ``inflation + frequency``.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._resident: set[str] = set()
        self._frequency: Counter[str] = Counter()
        self._priority: dict[str, float] = {}
        self._heap: list[tuple[float, int, str]] = []
        self._inflation = 0.0
        self._clock = 0

    def misses(self, items: Iterable[str]) -> int:
        return sum(item not in self._resident for item in items)

    def access(self, items: Iterable[str]) -> int:
        misses = 0
        for item in items:
            self._clock += 1
            self._frequency[item] += 1
            if item not in self._resident:
                misses += 1
                if len(self._resident) >= self.capacity:
                    while self._heap:
                        priority, _, victim = heapq.heappop(self._heap)
                        if (
                            victim in self._resident
                            and priority == self._priority[victim]
                        ):
                            self._resident.remove(victim)
                            self._inflation = priority
                            break
                self._resident.add(item)
            priority = self._inflation + float(self._frequency[item])
            self._priority[item] = priority
            heapq.heappush(self._heap, (priority, self._clock, item))
        return misses


def make_cache(policy: str, capacity: int) -> LogicalCache:
    factories = {"lru": LruCache, "lfu": LfuCache, "gdsf": GdsfCache}
    try:
        return factories[policy](capacity)
    except KeyError as error:
        raise ValueError(f"unsupported cache policy: {policy}") from error


def orderings(
    query_ids: Sequence[str], *, random_orders: int, seed: int
) -> list[tuple[str, tuple[str, ...]]]:
    output = [("natural", tuple(query_ids))]
    for index in range(random_orders):
        values = list(query_ids)
        np.random.default_rng(seed + index).shuffle(values)
        output.append((f"shuffle_{index:03d}", tuple(values)))
    return output


def action_for_query(
    *,
    program: str,
    floor: int,
    reference_depth: int,
    cold_budget: int,
    candidates: Sequence[str],
    cache: LogicalCache,
) -> tuple[int, bool]:
    """Return depth and whether the program abstained to its reference."""
    if program == "reference":
        return reference_depth, False
    if program == "static":
        return floor, False
    if program != "guard":
        raise ValueError(f"unsupported program: {program}")
    if cache.misses(candidates[:reference_depth]) <= cold_budget:
        return reference_depth, False
    if cache.misses(candidates[:floor]) <= cold_budget:
        return floor, False
    return reference_depth, True


def replay_program(
    domain: DomainSurface,
    query_ids: Sequence[str],
    *,
    program: str,
    floor: int,
    reference_depth: int,
    cold_budget: int,
    capacity_fraction: float,
    cache_policy: str,
    force_global_abstain: bool = False,
) -> dict[str, Any]:
    capacity = max(1, int(round(domain.corpus_pages * capacity_fraction)))
    cache = make_cache(cache_policy, capacity)
    total_misses = 0
    actions: list[int] = []
    abstained: list[bool] = []
    quality: list[float] = []
    reference_quality: list[float] = []
    trace: list[str] = []
    for query_id in query_ids:
        candidates = domain.ranking[query_id]
        if force_global_abstain:
            depth, did_abstain = reference_depth, True
        else:
            depth, did_abstain = action_for_query(
                program=program,
                floor=floor,
                reference_depth=reference_depth,
                cold_budget=cold_budget,
                candidates=candidates,
                cache=cache,
            )
        requested = list(candidates[:depth])
        total_misses += cache.access(requested)
        trace.extend(requested)
        actions.append(depth)
        abstained.append(did_abstain)
        quality.append(float(domain.quality[query_id][depth]))
        reference_quality.append(float(domain.quality[query_id][reference_depth]))
    losses = np.asarray(reference_quality) - np.asarray(quality)
    return {
        "query_ids": list(query_ids),
        "actions": actions,
        "action_counts": dict(sorted(Counter(map(str, actions)).items())),
        "abstained": abstained,
        "quality": quality,
        "reference_quality": reference_quality,
        "losses": losses.tolist(),
        "total_cold_page_misses": total_misses,
        "mean_cold_page_misses": total_misses / len(query_ids),
        "mean_depth": float(np.mean(actions)),
        "abstain_rate": float(np.mean(abstained)),
        "request_trace": trace,
    }


def belady_misses(items: Sequence[str], capacity: int) -> int:
    """Future-aware optimal misses for one immutable request trace."""
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


def wilson_upper(successes: int, trials: int, confidence: float) -> float:
    if trials <= 0 or not 0.5 < confidence < 1.0:
        raise ValueError("invalid Wilson interval arguments")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)
    )
    return (center + radius) / denominator


def bootstrap_mean_upper(
    values: Sequence[float], *, samples: int, confidence: float, seed: int
) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or samples <= 0:
        raise ValueError("bootstrap input and sample count must be positive")
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch = 256
    for start in range(0, samples, batch):
        count = min(batch, samples - start)
        indices = generator.integers(0, array.size, size=(count, array.size))
        means[start : start + count] = array[indices].mean(axis=1)
    return float(np.quantile(means, confidence))


def quality_summary(losses: Sequence[float], epsilon_query: float) -> dict[str, float]:
    array = np.asarray(losses, dtype=np.float64)
    count = max(1, int(math.ceil(0.05 * array.size)))
    worst = np.sort(array)[-count:]
    return {
        "mean_signed_regret": float(array.mean()),
        "mean_quality_delta": float(-array.mean()),
        "quality_violation_rate": float(np.mean(array > epsilon_query)),
        "worst_query_loss": float(array.max()),
        "worst_5pct_cvar": float(worst.mean()),
    }


def stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join(parts).encode("utf-8")
    suffix = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return (base + suffix) % (2**32)


def assess_calibrated_safety(
    losses: Sequence[float],
    *,
    epsilon_mean: float,
    epsilon_query: float,
    delta_empirical: float,
    delta_upper: float,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    summary = quality_summary(losses, epsilon_query)
    violations = int(np.sum(np.asarray(losses) > epsilon_query))
    mean_upper = bootstrap_mean_upper(
        losses, samples=bootstrap_samples, confidence=confidence, seed=seed
    )
    violation_upper = wilson_upper(violations, len(losses), confidence)
    checks = {
        "bootstrap_mean_regret_upper_le_epsilon_mean": mean_upper <= epsilon_mean,
        "empirical_violation_rate_le_delta": (
            summary["quality_violation_rate"] <= delta_empirical
        ),
        "wilson_violation_upper_le_delta_upper": violation_upper <= delta_upper,
    }
    return {
        **summary,
        "bootstrap_mean_regret_upper": mean_upper,
        "wilson_violation_rate_upper": violation_upper,
        "checks": checks,
        "passes": all(checks.values()),
    }


def compile_programs(
    domains: Sequence[DomainSurface],
    *,
    floors: Sequence[int],
    reference_depth: int,
    cold_budget: int,
    capacity_fraction: float,
    cache_policy: str,
    random_orders: int,
    order_seed: int,
    epsilon_mean: float,
    epsilon_query: float,
    delta_empirical: float,
    delta_upper: float,
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compile static and residency-guarded programs from calibration only."""
    records: dict[str, Any] = {}
    for family in ("static", "guard"):
        for floor in floors:
            program_id = f"{family}_top{floor}"
            domain_records: dict[str, Any] = {}
            total_misses = 0.0
            total_runs = 0
            passes = True
            for domain in domains:
                domain.validate(tuple(sorted(set((*floors, reference_depth)))))
                runs = []
                losses_by_order = []
                for order_id, query_ids in orderings(
                    domain.query_ids, random_orders=random_orders, seed=order_seed
                ):
                    replay = replay_program(
                        domain,
                        query_ids,
                        program=family,
                        floor=floor,
                        reference_depth=reference_depth,
                        cold_budget=cold_budget,
                        capacity_fraction=capacity_fraction,
                        cache_policy=cache_policy,
                    )
                    loss_by_query = dict(zip(query_ids, replay["losses"], strict=True))
                    losses_by_order.append(
                        [loss_by_query[query_id] for query_id in domain.query_ids]
                    )
                    total_misses += replay["mean_cold_page_misses"]
                    total_runs += 1
                    runs.append(
                        {
                            "order_id": order_id,
                            "mean_cold_page_misses": replay[
                                "mean_cold_page_misses"
                            ],
                            "mean_depth": replay["mean_depth"],
                            "abstain_rate": replay["abstain_rate"],
                            "quality": quality_summary(
                                replay["losses"], epsilon_query
                            ),
                        }
                    )
                loss_matrix = np.asarray(losses_by_order, dtype=np.float64)
                per_query_order_mean = loss_matrix.mean(axis=0)
                safety = assess_calibrated_safety(
                    per_query_order_mean,
                    epsilon_mean=epsilon_mean,
                    epsilon_query=epsilon_query,
                    delta_empirical=delta_empirical,
                    delta_upper=delta_upper,
                    confidence=confidence,
                    bootstrap_samples=bootstrap_samples,
                    seed=stable_seed(
                        bootstrap_seed, domain.name, program_id, "order-averaged"
                    ),
                )
                order_mean_regrets = loss_matrix.mean(axis=1)
                safety["order_mean_regret_p95"] = float(
                    np.quantile(order_mean_regrets, 0.95)
                )
                safety["checks"]["order_mean_regret_p95_le_epsilon_mean"] = (
                    safety["order_mean_regret_p95"] <= epsilon_mean
                )
                safety["passes"] = all(safety["checks"].values())
                passes = passes and safety["passes"]
                domain_records[domain.name] = {
                    "queries": len(domain.query_ids),
                    "input_sha256": dict(domain.input_sha256),
                    "runs": runs,
                    "safety_over_order_averaged_query_losses": safety,
                    "passes": safety["passes"],
                }
            records[program_id] = {
                "family": family,
                "floor": floor,
                "passes_all_domains_and_orders": passes,
                "calibration_mean_cold_page_misses": total_misses / total_runs,
                "domains": domain_records,
            }
    selected: dict[str, str | int | None] = {}
    for family in ("static", "guard"):
        eligible = [
            (row["calibration_mean_cold_page_misses"], -int(row["floor"]), key)
            for key, row in records.items()
            if row["family"] == family and row["passes_all_domains_and_orders"]
        ]
        if eligible:
            _, _, program_id = min(eligible)
            selected[family] = program_id
            selected[f"{family}_floor"] = records[program_id]["floor"]
        else:
            selected[family] = None
            selected[f"{family}_floor"] = None
    return {
        "certificate_kind": "finite-sample empirically calibrated; not formal safety",
        "programs": records,
        "selected": selected,
    }
