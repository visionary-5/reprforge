"""Wall-clock quality metrics for serve-then-compile retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def progressive_quality_timeline(
    base_quality: Sequence[float] | np.ndarray,
    refined_quality: Sequence[float] | np.ndarray,
    batch_trace: Sequence[dict],
    *,
    base_ready_ms: float,
    horizon_ms: float,
    target_quality: float,
    progress_reference_quality: float | None = None,
    progress_fractions: Sequence[float] = (0.5, 0.9),
) -> dict:
    """Construct a population-quality step curve from atomic batch publishes."""

    base = np.asarray(base_quality, dtype=np.float64)
    refined = np.asarray(refined_quality, dtype=np.float64)
    if (
        base.ndim != 1
        or base.shape != refined.shape
        or not len(base)
        or not np.isfinite(base).all()
        or not np.isfinite(refined).all()
    ):
        raise ValueError("base and refined quality must be aligned finite vectors")
    if base_ready_ms < 0 or horizon_ms <= base_ready_ms:
        raise ValueError("horizon must be later than base readiness")
    if not np.isfinite(target_quality):
        raise ValueError("target quality must be finite")
    reference_quality = (
        float(base.mean())
        if progress_reference_quality is None
        else float(progress_reference_quality)
    )
    if not np.isfinite(reference_quality):
        raise ValueError("progress reference quality must be finite")
    fractions = tuple(float(value) for value in progress_fractions)
    if any(not np.isfinite(value) or value <= 0 or value > 1 for value in fractions):
        raise ValueError("progress fractions must be finite and in (0, 1]")
    current = base.copy()
    elapsed = float(base_ready_ms)
    points = [
        {
            "elapsed_ms": elapsed,
            "mean_quality": float(current.mean()),
            "refined_queries": 0,
        }
    ]
    seen = np.zeros(len(base), dtype=bool)
    for batch in batch_trace:
        start = int(batch["query_offset_start"])
        count = int(batch["query_count"])
        completion = float(batch["completion_ms"])
        if count <= 0 or completion < 0 or start < 0 or start + count > len(base):
            raise ValueError("batch trace contains invalid query bounds or time")
        rows = np.arange(start, start + count)
        if seen[rows].any():
            raise ValueError("batch trace refines a query more than once")
        seen[rows] = True
        elapsed += completion
        current[rows] = refined[rows]
        points.append(
            {
                "elapsed_ms": elapsed,
                "mean_quality": float(current.mean()),
                "refined_queries": int(seen.sum()),
            }
        )
    if not seen.all():
        raise ValueError("batch trace does not refine every query")

    integration_end = float(horizon_ms)
    area = 0.0
    gain_area = 0.0
    previous_time = 0.0
    previous_quality = 0.0
    base_mean = float(base.mean())
    for point in points:
        point_time = min(float(point["elapsed_ms"]), integration_end)
        if point_time > previous_time:
            width = point_time - previous_time
            area += width * previous_quality
            gain_area += width * (previous_quality - base_mean)
        previous_time = point_time
        previous_quality = float(point["mean_quality"])
        if point_time >= integration_end:
            break
    if previous_time < integration_end:
        width = integration_end - previous_time
        area += width * previous_quality
        gain_area += width * (previous_quality - base_mean)

    target_time = None
    for point in points:
        if point["mean_quality"] >= target_quality:
            target_time = float(point["elapsed_ms"])
            break
    refined_mean = float(refined.mean())
    normalized_targets = {}
    for fraction in fractions:
        threshold = reference_quality + fraction * (refined_mean - reference_quality)
        threshold_time = None
        for point in points:
            if point["mean_quality"] >= threshold:
                threshold_time = float(point["elapsed_ms"])
                break
        normalized_targets[f"fraction_{fraction:g}"] = {
            "target_quality": float(threshold),
            "time_to_target_ms": threshold_time,
        }
    completion_times = np.empty(len(base), dtype=np.float64)
    cumulative = float(base_ready_ms)
    for batch in batch_trace:
        cumulative += float(batch["completion_ms"])
        start = int(batch["query_offset_start"])
        count = int(batch["query_count"])
        completion_times[start : start + count] = cumulative
    delta = refined - base
    return {
        "points": points,
        "base_mean_quality": base_mean,
        "refined_mean_quality": refined_mean,
        "target_quality": float(target_quality),
        "time_to_target_ms": target_time,
        "normalized_gain_reference_quality": reference_quality,
        "normalized_gain_targets": normalized_targets,
        "final_publish_ms": float(elapsed),
        "horizon_ms": integration_end,
        "mean_quality_over_horizon": float(area / integration_end),
        "mean_quality_gain_over_base_auc": float(gain_area / integration_end),
        "revision": {
            "mean_gain": float(delta.mean()),
            "improved_fraction": float(np.mean(delta > 0)),
            "unchanged_fraction": float(np.mean(delta == 0)),
            "harmed_fraction": float(np.mean(delta < 0)),
            "p05_gain": float(np.quantile(delta, 0.05)),
            "p50_gain": float(np.quantile(delta, 0.50)),
        },
        "stabilization_ms": {
            "p50": float(np.quantile(completion_times, 0.50)),
            "p95": float(np.quantile(completion_times, 0.95)),
            "max": float(completion_times.max()),
        },
    }
