"""Deterministic workload replays for defer--materialize experiments.

Public retrieval benchmarks rarely contain a natural temporal trace.  These
helpers generate explicit synthetic sensitivity tests; they never relabel a
dataset serialization order as production time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np


def _validate(query_ids: Sequence[str]) -> list[str]:
    values = [str(value) for value in query_ids]
    if not values:
        raise ValueError("at least one query is required")
    if len(values) != len(set(values)):
        raise ValueError("query IDs must be unique")
    return values


def hash_order(query_ids: Sequence[str], *, seed: int) -> list[str]:
    values = _validate(query_ids)
    return sorted(
        values,
        key=lambda query_id: (
            hashlib.sha256(f"{query_id}\0{seed}".encode()).digest(),
            query_id,
        ),
    )


def random_permutation(query_ids: Sequence[str], *, seed: int) -> list[str]:
    values = np.asarray(hash_order(query_ids, seed=seed), dtype=object)
    np.random.default_rng(seed).shuffle(values)
    return [str(value) for value in values]


def zipf_trace(
    query_ids: Sequence[str], *, length: int, exponent: float, seed: int
) -> list[str]:
    values = hash_order(query_ids, seed=seed)
    if length <= 0:
        raise ValueError("length must be positive")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    probabilities = np.power(ranks, -float(exponent))
    probabilities /= probabilities.sum()
    positions = np.random.default_rng(seed).choice(
        len(values), size=length, replace=True, p=probabilities
    )
    return [values[int(position)] for position in positions]


def clustered_trace(
    query_ids: Sequence[str],
    groups: Mapping[str, str],
    *,
    repetitions: int,
    seed: int,
) -> list[str]:
    values = _validate(query_ids)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    missing = sorted(set(values) - set(groups))
    if missing:
        raise ValueError(f"missing groups for {len(missing)} queries")
    buckets: dict[str, list[str]] = {}
    for query_id in values:
        buckets.setdefault(str(groups[query_id]), []).append(query_id)
    rng = np.random.default_rng(seed)
    output: list[str] = []
    for _ in range(repetitions):
        group_order = np.asarray(sorted(buckets), dtype=object)
        rng.shuffle(group_order)
        for group in group_order:
            members = np.asarray(sorted(buckets[str(group)]), dtype=object)
            rng.shuffle(members)
            output.extend(str(member) for member in members)
    return output


def broadening_trace(
    query_ids: Sequence[str], *, length: int, seed: int
) -> list[str]:
    values = hash_order(query_ids, seed=seed)
    if length <= 0:
        raise ValueError("length must be positive")
    rng = np.random.default_rng(seed)
    output = []
    for position in range(length):
        active = max(1, min(len(values), 1 + position * len(values) // length))
        output.append(values[int(rng.integers(0, active))])
    return output


def drift_trace(query_ids: Sequence[str], *, length: int, seed: int) -> list[str]:
    values = hash_order(query_ids, seed=seed)
    if len(values) < 2:
        return [values[0]] * length
    if length <= 0:
        raise ValueError("length must be positive")
    split = max(1, len(values) // 2)
    left, right = values[:split], values[split:]
    rng = np.random.default_rng(seed)
    midpoint = length // 2
    return [
        str(rng.choice(left if position < midpoint else right))
        for position in range(length)
    ]


def trace_suite(
    query_ids: Sequence[str],
    groups: Mapping[str, str],
    *,
    seed: int,
    random_permutations: int,
    horizon_multiplier: int,
    zipf_exponents: Sequence[float],
) -> dict[str, list[str]]:
    values = _validate(query_ids)
    if random_permutations < 0 or horizon_multiplier <= 0:
        raise ValueError("invalid trace counts")
    length = len(values) * horizon_multiplier
    traces: dict[str, list[str]] = {"dataset_order": values}
    for repetition in range(random_permutations):
        traces[f"random_{repetition:02d}"] = random_permutation(
            values, seed=seed + repetition
        )
    for exponent in zipf_exponents:
        name = str(float(exponent)).replace(".", "p")
        traces[f"zipf_{name}"] = zipf_trace(
            values,
            length=length,
            exponent=float(exponent),
            seed=seed + 10_000 + int(round(100 * float(exponent))),
        )
    traces["document_clustered"] = clustered_trace(
        values, groups, repetitions=horizon_multiplier, seed=seed + 20_000
    )
    traces["broadening_working_set"] = broadening_trace(
        values, length=length, seed=seed + 30_000
    )
    traces["mid_trace_distribution_drift"] = drift_trace(
        values, length=length, seed=seed + 40_000
    )
    return traces
