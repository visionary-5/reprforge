"""Anytime quality metrics on explicit elapsed, work, and page axes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


AXES = (
    "elapsed_unit_time",
    "charged_unit_work",
    "unique_compiled_pages",
)
DEFAULT_BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)
DEFAULT_TARGETS = (0.50, 0.90)


def _curve(
    points: Sequence[dict[str, Any]], axis: str
) -> tuple[np.ndarray, np.ndarray]:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}")
    if not points:
        raise ValueError("publication trace must not be empty")
    coordinates: list[float] = []
    qualities: list[float] = []
    for point in points:
        coordinate = float(point[axis])
        quality = float(point["mean_quality"])
        if coordinate < 0 or not np.isfinite(coordinate) or not np.isfinite(quality):
            raise ValueError("publication coordinates and quality must be finite")
        if coordinates and coordinate < coordinates[-1] - 1e-12:
            raise ValueError("publication coordinates must be non-decreasing")
        # A zero-cost batch publishes at the same coordinate.  The state visible
        # at that exact budget includes every publication completed there.
        if coordinates and abs(coordinate - coordinates[-1]) <= 1e-12:
            qualities[-1] = quality
        else:
            coordinates.append(coordinate)
            qualities.append(quality)
    if abs(coordinates[0]) > 1e-12:
        raise ValueError("publication trace must start at coordinate zero")
    return np.asarray(coordinates), np.asarray(qualities)


def _step_value(
    coordinates: np.ndarray, qualities: np.ndarray, budget: float
) -> float:
    position = int(np.searchsorted(coordinates, budget, side="right") - 1)
    return float(qualities[max(position, 0)])


def _step_auc(
    coordinates: np.ndarray, values: np.ndarray, horizon: float
) -> float:
    area = 0.0
    for index in range(len(coordinates) - 1):
        area += float(
            (coordinates[index + 1] - coordinates[index]) * values[index]
        )
    if coordinates[-1] < horizon:
        area += float((horizon - coordinates[-1]) * values[-1])
    return area / horizon


def _sustained_attainment(
    coordinates: np.ndarray,
    normalized_gain: np.ndarray,
    target: float,
    horizon: float,
) -> dict[str, float] | None:
    suffix_minimum = np.minimum.accumulate(normalized_gain[::-1])[::-1]
    reached = np.flatnonzero(suffix_minimum >= target - 1e-12)
    if not len(reached):
        return None
    coordinate = float(coordinates[int(reached[0])])
    return {
        "coordinate": coordinate,
        "common_horizon_fraction": coordinate / horizon,
    }


def time_aligned_quality_metrics(
    points: Sequence[dict[str, Any]],
    *,
    axis: str,
    common_horizon: float,
    base_quality: float,
    final_quality: float,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    targets: Sequence[float] = DEFAULT_TARGETS,
    positive_gain_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Summarize a right-continuous quality curve on a shared horizon."""

    horizon = float(common_horizon)
    base = float(base_quality)
    final = float(final_quality)
    if horizon <= 0 or not np.isfinite([horizon, base, final]).all():
        raise ValueError("common horizon and endpoint qualities must be finite")
    budget_values = tuple(float(value) for value in budgets)
    target_values = tuple(float(value) for value in targets)
    if any(value < 0 or value > 1 for value in budget_values):
        raise ValueError("budget fractions must be in [0, 1]")
    if any(value <= 0 or value > 1 for value in target_values):
        raise ValueError("attainment targets must be in (0, 1]")

    coordinates, qualities = _curve(points, axis)
    endpoint = float(coordinates[-1])
    if endpoint > horizon + 1e-12:
        raise ValueError("method endpoint exceeds common horizon")
    if abs(float(qualities[0]) - base) > 1e-8:
        raise ValueError("trajectory start differs from base quality")
    if abs(float(qualities[-1]) - final) > 1e-8:
        raise ValueError("trajectory endpoint differs from final quality")

    signed_gain = qualities - base
    final_gain = final - base
    mean_quality_auc = _step_auc(coordinates, qualities, horizon)
    signed_gain_auc = _step_auc(coordinates, signed_gain, horizon)
    positive_gain_defined = final_gain > positive_gain_epsilon
    if positive_gain_defined:
        normalized_gain = signed_gain / final_gain
        normalized_gain_auc = _step_auc(coordinates, normalized_gain, horizon)
        normalized_regret = 1.0 - normalized_gain_auc
        attainment = {
            f"sustained_t{int(round(target * 100))}": _sustained_attainment(
                coordinates, normalized_gain, target, horizon
            )
            for target in target_values
        }
    else:
        normalized_gain_auc = None
        normalized_regret = None
        attainment = {
            f"sustained_t{int(round(target * 100))}": None
            for target in target_values
        }

    fixed_budgets = {}
    for fraction in budget_values:
        coordinate = fraction * horizon
        quality = _step_value(coordinates, qualities, coordinate)
        fixed_budgets[f"budget_{int(round(fraction * 100))}_percent"] = {
            "common_horizon_fraction": fraction,
            "coordinate": coordinate,
            "mean_quality": quality,
            "quality_gain_over_base": quality - base,
            "final_positive_gain_fraction_achieved": (
                (quality - base) / final_gain if positive_gain_defined else None
            ),
        }

    return {
        "axis": axis,
        "common_horizon": horizon,
        "method_endpoint": endpoint,
        "method_endpoint_fraction": endpoint / horizon,
        "base_quality": base,
        "final_quality": final,
        "final_quality_gain": final_gain,
        "positive_final_gain_defined": positive_gain_defined,
        "mean_quality_auc": mean_quality_auc,
        "raw_signed_quality_gain_auc": signed_gain_auc,
        "normalized_quality_gain_auc": normalized_gain_auc,
        "normalized_quality_regret_auc": normalized_regret,
        "attainment": attainment,
        "fixed_budgets": fixed_budgets,
        "semantics": {
            "curve": "right-continuous atomic-publication step function",
            "horizon": "maximum method endpoint for the same domain/arrival/seed/axis",
            "after_method_endpoint": "final quality is held through common horizon",
            "normalized_values_clipped": False,
        },
    }
