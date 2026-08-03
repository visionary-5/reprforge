"""Online rent-or-buy policies for expensive document representations.

The module isolates a lifecycle decision from retrieval quality.  A request
means that the expensive representation is required for the current query.
The controller only decides whether to discard it after use or retain it for
future queries.  Consequently all policies execute the same representation
work logically and differ only in recomputation and holding cost.

Costs use one common unit (milliseconds in the ViDoRe replay):

* ``build_cost[item]`` is paid whenever an absent representation is requested;
* ``holding_cost[item]`` is paid for one query interval of residency.

``ski_ttl`` retains an item for ``build_cost / holding_cost`` intervals after
each access. ``verified_ski_ttl`` requires a second access before making that
commitment. They are direct online rent-or-buy baselines, not learned
selectors. ``offline_oracle`` is an unattainable per-item lower bound that
knows the next request time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

import numpy as np


Policy = Literal["no_cache", "resident", "ski_ttl", "verified_ski_ttl"]
EvictionPolicy = Literal["lru", "gdsf"]
TtlPolicy = Literal[
    "none",
    "breakeven",
    "randomized",
    "verified_breakeven",
]


@dataclass(frozen=True)
class ElasticCacheResult:
    policy: str
    query_count: int
    request_count: int
    unique_items: int
    cache_hits: int
    cache_misses: int
    build_cost: float
    holding_cost: float
    total_cost: float
    peak_resident_items: int
    final_resident_items: int

    @property
    def hit_fraction(self) -> float:
        return self.cache_hits / self.request_count if self.request_count else 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        result = asdict(self)
        result["hit_fraction"] = self.hit_fraction
        return result


@dataclass(frozen=True)
class CapacityCacheResult:
    eviction_policy: str
    ttl_policy: str
    capacity_bytes: int
    query_count: int
    request_count: int
    unique_items: int
    cache_hits: int
    cache_misses: int
    build_cost: float
    holding_cost: float
    total_cost: float
    peak_resident_items: int
    peak_resident_bytes: int
    resident_byte_intervals: float
    final_resident_items: int
    final_resident_bytes: int

    @property
    def hit_fraction(self) -> float:
        return self.cache_hits / self.request_count if self.request_count else 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        result = asdict(self)
        result["hit_fraction"] = self.hit_fraction
        return result


def _normalise_requests(
    request_batches: Sequence[Iterable[int]],
    item_count: int,
) -> tuple[tuple[int, ...], ...]:
    batches: list[tuple[int, ...]] = []
    for batch in request_batches:
        # A page requested more than once by one query still needs one encoding.
        values = tuple(dict.fromkeys(int(value) for value in batch))
        if any(value < 0 or value >= item_count for value in values):
            raise ValueError("request item is outside the cost-vector domain")
        batches.append(values)
    return tuple(batches)


def _validate_costs(
    build_cost: Sequence[float],
    holding_cost: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    builds = np.asarray(build_cost, dtype=np.float64)
    holdings = np.asarray(holding_cost, dtype=np.float64)
    if builds.ndim != 1 or holdings.ndim != 1 or builds.shape != holdings.shape:
        raise ValueError("build and holding costs must be equal-length vectors")
    if not np.all(np.isfinite(builds)) or np.any(builds < 0):
        raise ValueError("build costs must be finite and non-negative")
    if not np.all(np.isfinite(holdings)) or np.any(holdings < 0):
        raise ValueError("holding costs must be finite and non-negative")
    return builds, holdings


def replay_elastic_cache(
    request_batches: Sequence[Iterable[int]],
    build_cost: Sequence[float],
    holding_cost: Sequence[float],
    *,
    policy: Policy,
) -> ElasticCacheResult:
    """Replay an online lifecycle policy over query-batched requests.

    Holding cost is charged over intervals ``[t, t + 1)``.  There is no
    artificial holding charge after the final query.  For ``ski_ttl``, expiry
    can occur inside an interval, so fractional holding cost is preserved.
    """

    builds, holdings = _validate_costs(build_cost, holding_cost)
    batches = _normalise_requests(request_batches, len(builds))
    if policy not in (
        "no_cache",
        "resident",
        "ski_ttl",
        "verified_ski_ttl",
    ):
        raise ValueError(f"unknown elastic-cache policy: {policy}")

    # Value is the continuous query time at which the representation expires.
    active: dict[int, float] = {}
    hits = 0
    misses = 0
    build_total = 0.0
    holding_total = 0.0
    peak = 0
    touched: set[int] = set()
    access_count = np.zeros(len(builds), dtype=np.int64)

    for query, batch in enumerate(batches):
        if policy in ("ski_ttl", "verified_ski_ttl"):
            active = {
                item: expiry
                for item, expiry in active.items()
                if expiry > float(query)
            }

        for item in batch:
            touched.add(item)
            access_count[item] += 1
            if item in active:
                hits += 1
            else:
                misses += 1
                build_total += float(builds[item])

            if policy == "resident":
                active[item] = float("inf")
            elif policy in ("ski_ttl", "verified_ski_ttl") and (
                policy == "ski_ttl" or access_count[item] >= 2
            ):
                rate = float(holdings[item])
                active[item] = (
                    float("inf")
                    if rate == 0.0
                    else float(query) + float(builds[item]) / rate
                )

        peak = max(peak, len(active))
        if query == len(batches) - 1:
            continue

        if policy == "resident":
            holding_total += sum(float(holdings[item]) for item in active)
        elif policy in ("ski_ttl", "verified_ski_ttl"):
            for item, expiry in active.items():
                resident_fraction = min(1.0, max(0.0, expiry - float(query)))
                holding_total += float(holdings[item]) * resident_fraction

    request_count = sum(len(batch) for batch in batches)
    total = build_total + holding_total
    return ElasticCacheResult(
        policy=policy,
        query_count=len(batches),
        request_count=request_count,
        unique_items=len(touched),
        cache_hits=hits,
        cache_misses=misses,
        build_cost=build_total,
        holding_cost=holding_total,
        total_cost=total,
        peak_resident_items=peak,
        final_resident_items=len(active),
    )


def offline_oracle(
    request_batches: Sequence[Iterable[int]],
    build_cost: Sequence[float],
    holding_cost: Sequence[float],
) -> ElasticCacheResult:
    """Return the clairvoyant per-item rent-or-rebuild lower bound.

    After an access, the oracle knows the next access time.  It retains the
    item exactly when holding it across that gap is no more expensive than
    rebuilding it later.  Items have no value after their last access, so no
    tail holding cost is charged.
    """

    builds, holdings = _validate_costs(build_cost, holding_cost)
    batches = _normalise_requests(request_batches, len(builds))
    access_times: list[list[int]] = [[] for _ in range(len(builds))]
    for query, batch in enumerate(batches):
        for item in batch:
            access_times[item].append(query)

    hits = 0
    misses = 0
    build_total = 0.0
    holding_total = 0.0
    peak_upper_bound = 0
    interval_load = np.zeros(max(len(batches) - 1, 0), dtype=np.int64)
    touched = 0

    for item, times in enumerate(access_times):
        if not times:
            continue
        touched += 1
        misses += 1
        build_total += float(builds[item])
        for current, following in zip(times, times[1:]):
            gap = following - current
            hold = float(holdings[item]) * gap
            if hold <= float(builds[item]):
                hits += 1
                holding_total += hold
                interval_load[current:following] += 1
            else:
                misses += 1
                build_total += float(builds[item])
        if interval_load.size:
            peak_upper_bound = max(peak_upper_bound, int(interval_load.max()))

    request_count = sum(len(batch) for batch in batches)
    total = build_total + holding_total
    return ElasticCacheResult(
        policy="offline_oracle",
        query_count=len(batches),
        request_count=request_count,
        unique_items=touched,
        cache_hits=hits,
        cache_misses=misses,
        build_cost=build_total,
        holding_cost=holding_total,
        total_cost=total,
        peak_resident_items=peak_upper_bound,
        final_resident_items=0,
    )


def replay_capacity_cache(
    request_batches: Sequence[Iterable[int]],
    build_cost: Sequence[float],
    holding_cost: Sequence[float],
    size_bytes: Sequence[int],
    *,
    capacity_bytes: int,
    eviction_policy: EvictionPolicy,
    ttl_policy: TtlPolicy,
    random_seed: int = 0,
) -> CapacityCacheResult:
    """Replay a capacity-constrained cache with an optional elastic TTL.

    ``breakeven`` and ``randomized`` follow the two online ski-rental
    algorithms evaluated by Kumar et al. (CIDR 2025). Capacity pressure is
    handled independently by LRU or Greedy-Dual-Size-Frequency (GDSF), as in
    their practical composition. ``verified_breakeven`` is ReprForge's
    explicit two-access admission variant and is never labelled as published
    prior art.
    """

    builds, holdings = _validate_costs(build_cost, holding_cost)
    sizes = np.asarray(size_bytes, dtype=np.int64)
    if sizes.ndim != 1 or sizes.shape != builds.shape or np.any(sizes <= 0):
        raise ValueError("size_bytes must be a positive vector aligned with costs")
    if capacity_bytes < 0:
        raise ValueError("capacity_bytes must be non-negative")
    if eviction_policy not in ("lru", "gdsf"):
        raise ValueError(f"unknown eviction policy: {eviction_policy}")
    if ttl_policy not in (
        "none",
        "breakeven",
        "randomized",
        "verified_breakeven",
    ):
        raise ValueError(f"unknown TTL policy: {ttl_policy}")
    batches = _normalise_requests(request_batches, len(builds))
    rng = np.random.default_rng(random_seed)

    active: dict[int, float] = {}
    last_access: dict[int, int] = {}
    frequency: dict[int, int] = {}
    priority: dict[int, float] = {}
    observed_accesses = np.zeros(len(builds), dtype=np.int64)
    inflation = 0.0
    resident_bytes = 0
    peak_items = 0
    peak_bytes = 0
    byte_intervals = 0.0
    hits = 0
    misses = 0
    build_total = 0.0
    holding_total = 0.0
    touched: set[int] = set()

    def remove(item: int) -> None:
        nonlocal resident_bytes
        resident_bytes -= int(sizes[item])
        active.pop(item, None)
        last_access.pop(item, None)
        frequency.pop(item, None)
        priority.pop(item, None)

    def ttl_duration(item: int) -> float:
        rate = float(holdings[item])
        if ttl_policy == "none" or rate == 0.0:
            return float("inf")
        breakeven = float(builds[item]) / rate
        if ttl_policy in ("breakeven", "verified_breakeven"):
            return breakeven
        # Optimal randomized ski-rental buy time: CDF
        # F(t)=(exp(t/b)-1)/(e-1), t in [0,b].
        return breakeven * float(np.log1p((np.e - 1.0) * rng.random()))

    for query, batch in enumerate(batches):
        expired = [
            item for item, expiry in active.items() if expiry <= float(query)
        ]
        for item in expired:
            remove(item)

        for item in batch:
            touched.add(item)
            observed_accesses[item] += 1
            if item in active:
                hits += 1
                frequency[item] += 1
            else:
                misses += 1
                build_total += float(builds[item])
                should_admit = (
                    ttl_policy != "verified_breakeven"
                    or observed_accesses[item] >= 2
                )
                if should_admit and int(sizes[item]) <= capacity_bytes:
                    active[item] = float("inf")
                    resident_bytes += int(sizes[item])
                    frequency[item] = 1

            if item in active:
                last_access[item] = query
                duration = ttl_duration(item)
                if duration <= 0.0:
                    remove(item)
                else:
                    active[item] = float(query) + duration
                    priority[item] = inflation + (
                        frequency[item] * float(builds[item]) / int(sizes[item])
                    )

            while resident_bytes > capacity_bytes:
                if eviction_policy == "lru":
                    victim = min(
                        active,
                        key=lambda value: (last_access[value], value),
                    )
                else:
                    victim = min(
                        active,
                        key=lambda value: (
                            priority[value],
                            last_access[value],
                            value,
                        ),
                    )
                    inflation = max(inflation, priority[victim])
                remove(victim)

        peak_items = max(peak_items, len(active))
        peak_bytes = max(peak_bytes, resident_bytes)
        if query == len(batches) - 1:
            continue
        for item, expiry in active.items():
            fraction = min(1.0, max(0.0, expiry - float(query)))
            holding_total += float(holdings[item]) * fraction
            byte_intervals += int(sizes[item]) * fraction

    request_count = sum(len(batch) for batch in batches)
    return CapacityCacheResult(
        eviction_policy=eviction_policy,
        ttl_policy=ttl_policy,
        capacity_bytes=capacity_bytes,
        query_count=len(batches),
        request_count=request_count,
        unique_items=len(touched),
        cache_hits=hits,
        cache_misses=misses,
        build_cost=build_total,
        holding_cost=holding_total,
        total_cost=build_total + holding_total,
        peak_resident_items=peak_items,
        peak_resident_bytes=peak_bytes,
        resident_byte_intervals=byte_intervals,
        final_resident_items=len(active),
        final_resident_bytes=resident_bytes,
    )
