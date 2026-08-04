"""Label-efficient auditing for physical index-plan selection.

This module deliberately separates two operations:

* qrel-free score distortion defines query strata;
* relevance labels estimate plan regret inside a probability sample.

The current intervals are design-aware normal approximations, not conformal
or distribution-free guarantees.  Their purpose is to test whether boundary
stratification improves plan-selection label efficiency before committing to
a stronger statistical protocol.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return array


def quantile_risk_strata(
    proxy_risk: Sequence[float] | np.ndarray, *, strata: int = 4
) -> np.ndarray:
    """Assign nearly equal-sized strata from low to high qrel-free risk.

    Stable row position breaks proxy ties, which keeps the design deterministic
    while avoiding empty quantile bins when many rankings are identical.
    """

    risk = _finite_vector(proxy_risk, name="proxy risk")
    if strata < 2 or strata > len(risk):
        raise ValueError("strata must lie between 2 and the query count")
    order = np.lexsort((np.arange(len(risk)), risk))
    assignments = np.empty(len(risk), dtype=np.int64)
    for rank, index in enumerate(order):
        assignments[index] = min(strata - 1, rank * strata // len(risk))
    if len(np.unique(assignments)) != strata:
        raise ValueError("risk stratification produced an empty stratum")
    return assignments


def _bounded_largest_remainder(
    target: np.ndarray, capacity: np.ndarray, total: int
) -> np.ndarray:
    allocation = np.minimum(np.floor(target).astype(np.int64), capacity)
    while int(allocation.sum()) < total:
        available = np.flatnonzero(allocation < capacity)
        if not len(available):
            raise ValueError("sample budget exceeds stratum capacity")
        priority = target[available] - allocation[available]
        chosen = int(available[np.argmax(priority)])
        allocation[chosen] += 1
    while int(allocation.sum()) > total:
        available = np.flatnonzero(allocation > 0)
        priority = allocation[available] - target[available]
        chosen = int(available[np.argmax(priority)])
        allocation[chosen] -= 1
    return allocation


def stratified_sample_sizes(
    assignments: Sequence[int] | np.ndarray,
    *,
    budget: int,
    minimum_per_stratum: int = 2,
    allocation: str = "boundary",
) -> np.ndarray:
    """Allocate a label budget across qrel-free risk strata.

    ``uniform`` is proportional allocation. ``boundary`` uses a fixed linear
    high-risk tilt (weights 1..H), while preserving at least
    ``minimum_per_stratum`` observations in every stratum.
    """

    strata = np.asarray(assignments, dtype=np.int64)
    if strata.ndim != 1 or not len(strata) or np.any(strata < 0):
        raise ValueError("assignments must be a non-empty non-negative vector")
    levels = int(strata.max()) + 1
    counts = np.bincount(strata, minlength=levels)
    if np.any(counts == 0):
        raise ValueError("assignments contain an empty stratum")
    if minimum_per_stratum < 1:
        raise ValueError("minimum per stratum must be positive")
    minimum = np.minimum(counts, minimum_per_stratum)
    if budget < int(minimum.sum()) or budget > len(strata):
        raise ValueError("budget is incompatible with stratum minima or population")
    remaining = budget - int(minimum.sum())
    capacity = counts - minimum
    if allocation == "uniform":
        weights = counts.astype(np.float64)
    elif allocation == "boundary":
        weights = counts * np.arange(1, levels + 1, dtype=np.float64)
    else:
        raise ValueError("allocation must be 'uniform' or 'boundary'")
    if not remaining:
        return minimum
    active_weights = np.where(capacity > 0, weights, 0.0)
    target = remaining * active_weights / active_weights.sum()
    return minimum + _bounded_largest_remainder(target, capacity, remaining)


def draw_stratified_sample(
    assignments: Sequence[int] | np.ndarray,
    sample_sizes: Sequence[int] | np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    strata = np.asarray(assignments, dtype=np.int64)
    sizes = np.asarray(sample_sizes, dtype=np.int64)
    if sizes.ndim != 1 or len(sizes) != int(strata.max()) + 1:
        raise ValueError("sample sizes do not match assignments")
    selected = []
    for level, size in enumerate(sizes):
        members = np.flatnonzero(strata == level)
        if size < 0 or size > len(members):
            raise ValueError("invalid within-stratum sample size")
        selected.extend(rng.choice(members, size=int(size), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def stratified_mean_upper_bound(
    values: Sequence[float] | np.ndarray,
    assignments: Sequence[int] | np.ndarray,
    sampled_indices: Sequence[int] | np.ndarray,
    *,
    alpha: float,
    finite_population: bool = False,
) -> dict[str, float | int]:
    """Estimate a stratified mean and a one-sided normal upper bound.

    The default omits the finite-population correction because the target is
    future workload risk, not merely the mean of the already logged queries.
    """

    observations = _finite_vector(values, name="observations")
    strata = np.asarray(assignments, dtype=np.int64)
    sample = np.asarray(sampled_indices, dtype=np.int64)
    if len(strata) != len(observations):
        raise ValueError("observations and assignments must align")
    if sample.ndim != 1 or not len(sample) or len(np.unique(sample)) != len(sample):
        raise ValueError("sample indices must be unique and non-empty")
    if np.any(sample < 0) or np.any(sample >= len(observations)):
        raise ValueError("sample index lies outside observations")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")

    population = len(observations)
    estimate = 0.0
    variance = 0.0
    levels = int(strata.max()) + 1
    for level in range(levels):
        members = np.flatnonzero(strata == level)
        chosen = sample[strata[sample] == level]
        if not len(chosen):
            raise ValueError("every stratum must be sampled")
        weight = len(members) / population
        estimate += weight * float(observations[chosen].mean())
        if len(chosen) == len(members):
            continue
        if len(chosen) < 2:
            # Losses lie in [-1, 1], whose maximum variance is one.
            sample_variance = 1.0
        else:
            sample_variance = float(observations[chosen].var(ddof=1))
        sampling_fraction = len(chosen) / len(members) if finite_population else 0.0
        variance += (
            weight * weight * (1.0 - sampling_fraction) * sample_variance / len(chosen)
        )
    standard_error = math.sqrt(max(0.0, variance))
    critical = NormalDist().inv_cdf(1.0 - alpha)
    return {
        "estimate": float(estimate),
        "standard_error": float(standard_error),
        "one_sided_upper": float(estimate + critical * standard_error),
        "alpha": float(alpha),
        "critical_value": float(critical),
        "sampled_queries": int(len(sample)),
        "population_queries": int(population),
        "finite_population_correction": bool(finite_population),
    }


def select_cheapest_certified_plan(
    plan_losses: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
    resident_fractions: Mapping[str, float],
    assignments: Sequence[int] | np.ndarray,
    sampled_indices: Sequence[int] | np.ndarray,
    *,
    tolerance: float = 0.01,
    family_alpha: float = 0.05,
    fallback: str = "full",
) -> dict:
    """Select the cheapest plan whose audited metric bounds are all safe."""

    if fallback not in plan_losses or fallback not in resident_fractions:
        raise ValueError("fallback plan is missing")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    metric_tests = sum(len(metrics) for name, metrics in plan_losses.items() if name != fallback)
    if not metric_tests:
        raise ValueError("at least one non-fallback plan is required")
    alpha = family_alpha / metric_tests
    candidates = []
    for name, metrics in plan_losses.items():
        fraction = float(resident_fractions[name])
        if not 0.0 < fraction <= 1.0:
            raise ValueError("resident fractions must lie in (0, 1]")
        if name == fallback:
            continue
        bounds = {
            metric: stratified_mean_upper_bound(
                values,
                assignments,
                sampled_indices,
                alpha=alpha,
            )
            for metric, values in metrics.items()
        }
        candidates.append(
            {
                "plan": name,
                "resident_fraction": fraction,
                "bounds": bounds,
                "passes": all(value["one_sided_upper"] <= tolerance for value in bounds.values()),
            }
        )
    safe = [candidate for candidate in candidates if candidate["passes"]]
    if safe:
        selected = min(safe, key=lambda value: (value["resident_fraction"], value["plan"]))
        abstained = False
    else:
        selected = {
            "plan": fallback,
            "resident_fraction": float(resident_fractions[fallback]),
            "bounds": {},
            "passes": True,
        }
        abstained = True
    return {
        "selected_plan": selected["plan"],
        "selected_resident_fraction": selected["resident_fraction"],
        "abstained_to_fallback": abstained,
        "family_alpha": float(family_alpha),
        "per_test_alpha": float(alpha),
        "tolerance": float(tolerance),
        "candidates": candidates,
    }


def simulate_label_efficiency(
    plan_losses: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
    resident_fractions: Mapping[str, float],
    proxy_risk: Sequence[float] | np.ndarray,
    *,
    budgets: Sequence[int],
    trials: int = 200,
    strata: int = 4,
    tolerance: float = 0.01,
    family_alpha: float = 0.05,
    seed: int = 0,
    fallback: str = "full",
    truth_safety: Mapping[str, bool] | None = None,
) -> dict:
    """Compare uniform and boundary-stratified relevance auditing.

    ``truth_safety`` may provide decisions from a disjoint audit workload;
    otherwise calibration-population mean losses define a diagnostic oracle.
    This is a development simulation, not a sealed-test certificate.
    """

    risk = _finite_vector(proxy_risk, name="proxy risk")
    if trials <= 0:
        raise ValueError("trials must be positive")
    normalized: dict[str, dict[str, np.ndarray]] = {}
    for plan, metrics in plan_losses.items():
        normalized[plan] = {}
        for metric, values in metrics.items():
            array = _finite_vector(values, name=f"{plan}/{metric} losses")
            if len(array) != len(risk):
                raise ValueError("all plan losses must align with proxy risk")
            normalized[plan][metric] = array
    if set(normalized) != set(resident_fractions):
        raise ValueError("plan losses and resident fractions must share keys")
    if truth_safety is not None and set(truth_safety) != set(normalized):
        raise ValueError("truth safety and plan losses must share keys")
    truth = {
        plan: {
            "passes": (
                bool(truth_safety[plan])
                if truth_safety is not None
                else all(float(values.mean()) <= tolerance for values in metrics.values())
            ),
            "mean_losses": {metric: float(values.mean()) for metric, values in metrics.items()},
            "resident_fraction": float(resident_fractions[plan]),
        }
        for plan, metrics in normalized.items()
    }
    safe_truth = [value | {"plan": plan} for plan, value in truth.items() if value["passes"]]
    if not safe_truth:
        raise ValueError("at least one plan must be safe under full-population truth")
    oracle = min(safe_truth, key=lambda value: (value["resident_fraction"], value["plan"]))

    boundary_assignments = quantile_risk_strata(risk, strata=strata)
    uniform_assignments = np.zeros(len(risk), dtype=np.int64)
    output = {}
    root = np.random.SeedSequence(seed)
    budget_values = [int(value) for value in budgets]
    if not budget_values or len(set(budget_values)) != len(budget_values):
        raise ValueError("budgets must be non-empty and unique")
    children = root.spawn(len(budget_values) * 2)
    child_index = 0
    for budget in budget_values:
        strategies = {}
        for strategy, assignments, allocation in (
            ("uniform", uniform_assignments, "uniform"),
            ("boundary_stratified", boundary_assignments, "boundary"),
        ):
            sizes = stratified_sample_sizes(
                assignments,
                budget=budget,
                minimum_per_stratum=2 if strategy == "boundary_stratified" else 1,
                allocation=allocation,
            )
            rng = np.random.default_rng(children[child_index])
            child_index += 1
            selections: dict[str, int] = {plan: 0 for plan in normalized}
            false_safe = 0
            oracle_matches = 0
            selected_fractions = []
            selected_losses = {metric: [] for metric in next(iter(normalized.values()))}
            for _ in range(trials):
                sample = draw_stratified_sample(assignments, sizes, rng=rng)
                decision = select_cheapest_certified_plan(
                    normalized,
                    resident_fractions,
                    assignments,
                    sample,
                    tolerance=tolerance,
                    family_alpha=family_alpha,
                    fallback=fallback,
                )
                selected = decision["selected_plan"]
                selections[selected] += 1
                false_safe += int(not truth[selected]["passes"])
                oracle_matches += int(selected == oracle["plan"])
                selected_fractions.append(float(resident_fractions[selected]))
                for metric in selected_losses:
                    selected_losses[metric].append(truth[selected]["mean_losses"][metric])
            strategies[strategy] = {
                "sample_sizes_by_stratum": sizes.tolist(),
                "selection_counts": selections,
                "false_safe_rate": false_safe / trials,
                "oracle_match_rate": oracle_matches / trials,
                "mean_selected_resident_fraction": float(np.mean(selected_fractions)),
                "mean_selected_losses": {
                    metric: float(np.mean(values)) for metric, values in selected_losses.items()
                },
            }
        output[str(budget)] = strategies
    return {
        "status": "development_normal_approximation_not_distribution_free",
        "population_queries": int(len(risk)),
        "trials": int(trials),
        "strata": int(strata),
        "tolerance": float(tolerance),
        "family_alpha": float(family_alpha),
        "truth": truth,
        "oracle_plan": oracle["plan"],
        "oracle_resident_fraction": oracle["resident_fraction"],
        "budgets": output,
    }
