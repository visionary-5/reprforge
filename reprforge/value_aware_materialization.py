"""Closed-loop acquisition for a sparse, persistent visual retrieval index.

The compiler never reads visual scores for an unbuilt page.  A small uniform
anchor sample estimates the query-specific background score distribution.
Every additional page is an intervention: after it is encoded, its signed
listwise value on a history workload becomes observable and updates a
cost-aware ridge-UCB acquisition model.

This module operates on saved score surfaces so that the protocol can be
tested without rebuilding an index.  Accesses to ``surface.visual_scores`` are
deliberately restricted to anchor or acquired page columns; a physical runner
can replace those column reads with actual page encoding and MaxSim scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from reprforge.partial_vlm_materialization import ScoreSurface


@dataclass(frozen=True)
class CompilerConfig:
    """Frozen controls for the first value-aware compiler protocol."""

    anchor_pages: int = 16
    calibration_quantile: float = 0.9
    visual_weight: float = 1.0
    ridge: float = 1.0
    exploration: float = 1.0
    minimum_admission_gain: float = 0.0
    seed: int = 0

    def validate(self) -> None:
        if self.anchor_pages < 2:
            raise ValueError("at least two calibration anchors are required")
        if not 0.5 <= self.calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be in [0.5, 1)")
        if self.visual_weight < 0.0:
            raise ValueError("visual_weight must be nonnegative")
        if self.ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if self.exploration < 0.0:
            raise ValueError("exploration must be nonnegative")


def cheap_history_features(
    surface: ScoreSurface, history_queries: Sequence[int]
) -> np.ndarray:
    """Build page features without reading qrels or visual representations."""

    queries = _query_positions(surface, history_queries)
    scores = np.asarray(surface.text_scores[queries], dtype=np.float64)
    order = np.asarray(surface.text_order[queries], dtype=np.int32)
    pages = surface.pages
    top10 = np.zeros(pages, dtype=np.float64)
    top50 = np.zeros(pages, dtype=np.float64)
    for ranking in order:
        top10[ranking[: min(10, pages)]] += 1.0
        top50[ranking[: min(50, pages)]] += 1.0
    divisor = float(len(queries))
    features = np.column_stack(
        (
            np.log1p(np.asarray(surface.text_bytes, dtype=np.float64)),
            scores.mean(axis=0),
            scores.std(axis=0),
            scores.max(axis=0),
            top10 / divisor,
            top50 / divisor,
        )
    )
    return _standardize(features)


def calibrated_ranking(
    surface: ScoreSurface,
    query_position: int,
    *,
    admitted_pages: Sequence[int],
    anchor_pages: Sequence[int],
    calibration_quantile: float,
    visual_weight: float,
) -> np.ndarray:
    """Fuse sparse visual evidence without treating subset rank as global rank.

    Anchors estimate a per-query background threshold.  An admitted page only
    receives a positive residual when its absolute visual score exceeds that
    threshold.  No visual column outside ``anchors U admitted`` is accessed.
    """

    query = int(query_position)
    if query < 0 or query >= surface.queries:
        raise ValueError("query position outside surface")
    anchors = _page_positions(surface, anchor_pages, allow_empty=False)
    admitted = _page_positions(surface, admitted_pages, allow_empty=True)
    text = _zscore(np.asarray(surface.text_scores[query], dtype=np.float64))
    background = np.asarray(surface.visual_scores[query, anchors], dtype=np.float64)
    threshold = float(np.quantile(background, calibration_quantile))
    median = float(np.median(background))
    mad = float(np.median(np.abs(background - median)))
    scale = max(1.4826 * mad, float(background.std()), 1e-8)
    scores = text.copy()
    if admitted.size:
        visual = np.asarray(surface.visual_scores[query, admitted], dtype=np.float64)
        residual = np.maximum((visual - threshold) / scale, 0.0)
        scores[admitted] += visual_weight * residual
    pages = np.arange(surface.pages, dtype=np.int32)
    return pages[np.lexsort((pages, -scores))]


def evaluate_compiled_index(
    surface: ScoreSurface,
    query_positions: Sequence[int],
    *,
    admitted_pages: Sequence[int],
    anchor_pages: Sequence[int],
    config: CompilerConfig,
) -> dict[str, Any]:
    """Evaluate one compiled sparse index on an explicit query split."""

    queries = _query_positions(surface, query_positions)
    per_query = []
    for query in queries:
        ranking = calibrated_ranking(
            surface,
            int(query),
            admitted_pages=admitted_pages,
            anchor_pages=anchor_pages,
            calibration_quantile=config.calibration_quantile,
            visual_weight=config.visual_weight,
        )
        per_query.append(_ndcg_at_10(surface, int(query), ranking))
    return {
        "queries": int(len(queries)),
        "mean_ndcg_at_10": float(np.mean(per_query)),
        "per_query_ndcg_at_10": per_query,
    }


def compile_value_aware_index(
    surface: ScoreSurface,
    history_queries: Sequence[int],
    *,
    page_features: np.ndarray | None = None,
    page_costs: np.ndarray | None = None,
    maximum_cost: float,
    config: CompilerConfig = CompilerConfig(),
) -> dict[str, Any]:
    """Acquire and admit visual pages under a measured construction budget.

    The uniform anchors are paid for first and remain calibration-only unless
    their measured history marginal is positive.  Remaining pages are probed
    with a ridge-UCB score divided by construction cost.  A probed page is
    admitted only when its observed signed marginal nDCG gain clears the
    frozen threshold; rejected probes still train the acquisition model.
    """

    config.validate()
    queries = _query_positions(surface, history_queries)
    features = (
        cheap_history_features(surface, queries)
        if page_features is None
        else _standardize(_feature_matrix(surface, page_features))
    )
    costs = _cost_vector(surface, page_costs)
    if maximum_cost <= 0.0:
        raise ValueError("maximum_cost must be positive")
    random_order = np.random.default_rng(config.seed).permutation(surface.pages)
    anchors: list[int] = []
    spent = 0.0
    for page in random_order:
        cost = float(costs[page])
        if spent + cost <= maximum_cost + 1e-12:
            anchors.append(int(page))
            spent += cost
        if len(anchors) >= min(config.anchor_pages, surface.pages):
            break
    if len(anchors) < 2:
        raise ValueError("budget cannot pay for two calibration anchors")

    observed_pages: list[int] = []
    observed_rewards: list[float] = []
    admitted: list[int] = []
    trace: list[dict[str, Any]] = []
    current_quality = _mean_quality(
        surface, queries, admitted, anchors, config
    )

    # Anchors provide the first signed interventions.  Greedy admission keeps
    # their background role separate from their optional ranking role.
    anchor_spent = 0.0
    for page in anchors:
        anchor_spent += float(costs[page])
        reward = _marginal_quality(
            surface, queries, admitted, anchors, page, config, current_quality
        )
        observed_pages.append(page)
        observed_rewards.append(reward)
        accepted = reward > config.minimum_admission_gain
        if accepted:
            admitted.append(page)
            current_quality += reward
        trace.append(
            _trace_row(
                len(trace) + 1,
                page,
                "uniform_anchor",
                float(costs[page]),
                anchor_spent,
                reward,
                accepted,
                current_quality,
            )
        )

    observed = set(observed_pages)
    while True:
        candidates = np.asarray(
            [page for page in range(surface.pages) if page not in observed],
            dtype=np.int32,
        )
        if candidates.size == 0:
            break
        affordable = candidates[spent + costs[candidates] <= maximum_cost + 1e-12]
        if affordable.size == 0:
            break
        mean, uncertainty = _ridge_ucb(
            features,
            observed_pages,
            observed_rewards,
            ridge=config.ridge,
        )
        acquisition = (
            mean[affordable] + config.exploration * uncertainty[affordable]
        ) / costs[affordable]
        order = np.lexsort((affordable, -acquisition))
        page = int(affordable[order[0]])
        spent += float(costs[page])
        reward = _marginal_quality(
            surface, queries, admitted, anchors, page, config, current_quality
        )
        observed.add(page)
        observed_pages.append(page)
        observed_rewards.append(reward)
        accepted = reward > config.minimum_admission_gain
        if accepted:
            admitted.append(page)
            current_quality += reward
        trace.append(
            _trace_row(
                len(trace) + 1,
                page,
                "ridge_ucb",
                float(costs[page]),
                spent,
                reward,
                accepted,
                current_quality,
            )
        )

    return {
        "anchor_pages": anchors,
        "probed_pages": observed_pages,
        "admitted_pages": admitted,
        "rejected_pages": [page for page in observed_pages if page not in set(admitted)],
        "spent_cost": spent,
        "maximum_cost": float(maximum_cost),
        "materialized_page_fraction": len(observed_pages) / surface.pages,
        "admitted_page_fraction": len(admitted) / surface.pages,
        "history_mean_ndcg_at_10": current_quality,
        "trace": trace,
        "protocol": {
            "future_qrels_visible": False,
            "unmaterialized_visual_columns_visible": False,
            "selection_feedback": "signed_history_marginal_ndcg",
            "fusion": "anchor_calibrated_positive_visual_residual",
        },
    }


def _ridge_ucb(
    features: np.ndarray,
    observed_pages: Sequence[int],
    observed_rewards: Sequence[float],
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = features.shape[1]
    x = features[np.asarray(observed_pages, dtype=np.int32)]
    y = np.asarray(observed_rewards, dtype=np.float64)
    gram = x.T @ x + ridge * np.eye(dimension, dtype=np.float64)
    inverse = np.linalg.inv(gram)
    theta = inverse @ x.T @ y
    mean = features @ theta
    uncertainty = np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", features, inverse, features), 0.0)
    )
    return mean, uncertainty


def _marginal_quality(
    surface: ScoreSurface,
    queries: np.ndarray,
    admitted: Sequence[int],
    anchors: Sequence[int],
    page: int,
    config: CompilerConfig,
    current_quality: float,
) -> float:
    if page in admitted:
        return 0.0
    candidate = [*admitted, int(page)]
    return _mean_quality(surface, queries, candidate, anchors, config) - current_quality


def _mean_quality(
    surface: ScoreSurface,
    queries: np.ndarray,
    admitted: Sequence[int],
    anchors: Sequence[int],
    config: CompilerConfig,
) -> float:
    return float(
        np.mean(
            [
                _ndcg_at_10(
                    surface,
                    int(query),
                    calibrated_ranking(
                        surface,
                        int(query),
                        admitted_pages=admitted,
                        anchor_pages=anchors,
                        calibration_quantile=config.calibration_quantile,
                        visual_weight=config.visual_weight,
                    ),
                )
                for query in queries
            ]
        )
    )


def _ndcg_at_10(surface: ScoreSurface, query: int, ranking: np.ndarray) -> float:
    relevance = np.asarray(surface.qrels[query, ranking[:10]], dtype=np.float64)
    discounts = np.log2(np.arange(2, len(relevance) + 2, dtype=np.float64))
    dcg = float(np.sum((np.power(2.0, relevance) - 1.0) / discounts))
    ideal = float(surface.idcg_at_10[query])
    return dcg / ideal if ideal > 0.0 else 0.0


def _standardize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    normalized = (values - mean) / scale
    return np.column_stack((np.ones(len(values), dtype=np.float64), normalized))


def _zscore(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    return np.zeros_like(values) if scale <= 1e-12 else (values - values.mean()) / scale


def _feature_matrix(surface: ScoreSurface, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != surface.pages:
        raise ValueError("page_features must have shape [pages, features]")
    if not np.all(np.isfinite(values)):
        raise ValueError("page_features must be finite")
    return values


def _cost_vector(surface: ScoreSurface, costs: np.ndarray | None) -> np.ndarray:
    values = (
        np.ones(surface.pages, dtype=np.float64)
        if costs is None
        else np.asarray(costs, dtype=np.float64)
    )
    if values.shape != (surface.pages,) or not np.all(np.isfinite(values)):
        raise ValueError("page_costs must be a finite vector with one value per page")
    if np.any(values <= 0.0):
        raise ValueError("page_costs must be positive")
    return values


def _query_positions(surface: ScoreSurface, positions: Sequence[int]) -> np.ndarray:
    values = np.asarray(list(map(int, positions)), dtype=np.int32)
    if values.size == 0 or values.min() < 0 or values.max() >= surface.queries:
        raise ValueError("query split is empty or outside surface")
    if len(set(values.tolist())) != len(values):
        raise ValueError("query split contains duplicates")
    return values


def _page_positions(
    surface: ScoreSurface, positions: Sequence[int], *, allow_empty: bool
) -> np.ndarray:
    values = np.asarray(sorted(set(map(int, positions))), dtype=np.int32)
    if values.size == 0:
        if allow_empty:
            return values
        raise ValueError("page set must not be empty")
    if values.min() < 0 or values.max() >= surface.pages:
        raise ValueError("page position outside surface")
    return values


def _trace_row(
    step: int,
    page: int,
    source: str,
    cost: float,
    cumulative_cost: float,
    reward: float,
    admitted: bool,
    quality: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "page": page,
        "source": source,
        "cost": cost,
        "cumulative_cost": cumulative_cost,
        "observed_marginal_ndcg_at_10": reward,
        "admitted": admitted,
        "history_mean_ndcg_at_10": quality,
    }
