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
from statistics import NormalDist
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
    fusion: str = "residual"
    two_way_centering: bool = False
    familywise_alpha: float | None = None
    rrf_constant: int = 60
    text_top_k: int = 100
    visual_top_k: int = 100
    anchor_rank_smoothing: float = 0.5
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
        if self.fusion not in ("residual", "anchor_rank"):
            raise ValueError("unsupported fusion")
        if self.familywise_alpha is not None and not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must lie in (0, 1)")
        if self.rrf_constant < 0 or self.text_top_k <= 0 or self.visual_top_k <= 0:
            raise ValueError("invalid RRF controls")
        if self.anchor_rank_smoothing <= 0.0:
            raise ValueError("anchor_rank_smoothing must be positive")


@dataclass(frozen=True)
class CalibrationState:
    """Statistics fitted only from paid anchors/pages and history queries."""

    anchors: np.ndarray
    admitted: np.ndarray
    page_centers: np.ndarray
    global_center: float
    scale: float
    gate_threshold: float


@dataclass(frozen=True)
class TypedCalibrationState:
    """Per-page interaction background fitted on historical queries only."""

    page_centers: np.ndarray
    page_scales: np.ndarray


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
    calibration_queries: Sequence[int] | None = None,
    two_way_centering: bool = False,
    familywise_alpha: float | None = None,
) -> np.ndarray:
    """Fuse sparse visual evidence without treating subset rank as global rank.

    Anchors estimate a per-query background threshold.  An admitted page only
    receives a positive residual when its absolute visual score exceeds that
    threshold.  No visual column outside ``anchors U admitted`` is accessed.
    """

    query = int(query_position)
    if query < 0 or query >= surface.queries:
        raise ValueError("query position outside surface")
    config = CompilerConfig(
        anchor_pages=max(2, len(set(map(int, anchor_pages)))),
        calibration_quantile=calibration_quantile,
        visual_weight=visual_weight,
        two_way_centering=two_way_centering,
        familywise_alpha=familywise_alpha,
    )
    config.validate()
    history = (
        [query]
        if calibration_queries is None
        else calibration_queries
    )
    state = fit_calibration(
        surface,
        admitted_pages=admitted_pages,
        anchor_pages=anchor_pages,
        calibration_queries=history,
        config=config,
    )
    return _ranking_with_calibration(surface, query, state, config)


def anchor_rank_ranking(
    surface: ScoreSurface,
    query_position: int,
    *,
    admitted_pages: Sequence[int],
    anchor_pages: Sequence[int],
    rrf_constant: int = 60,
    text_top_k: int = 100,
    visual_top_k: int = 100,
    visual_weight: float = 1.0,
    smoothing: float = 0.5,
) -> np.ndarray:
    """Estimate complete-corpus visual rank from a uniform paid anchor sample."""

    query = int(query_position)
    if query < 0 or query >= surface.queries:
        raise ValueError("query position outside surface")
    if rrf_constant < 0 or text_top_k <= 0 or visual_top_k <= 0:
        raise ValueError("invalid RRF controls")
    if visual_weight < 0.0 or smoothing <= 0.0:
        raise ValueError("visual weight and smoothing must be valid")
    anchors = _page_positions(surface, anchor_pages, allow_empty=False)
    admitted = _page_positions(surface, admitted_pages, allow_empty=True)
    contributions: dict[int, float] = {}
    for rank, page in enumerate(
        surface.text_order[query, : min(text_top_k, surface.pages)], start=1
    ):
        contributions[int(page)] = 1.0 / (rrf_constant + rank)
    if admitted.size:
        anchor_scores = np.asarray(
            surface.visual_scores[query, anchors], dtype=np.float64
        )
        visual_scores = np.asarray(
            surface.visual_scores[query, admitted], dtype=np.float64
        )
        exceed = np.sum(
            anchor_scores[:, np.newaxis] > visual_scores[np.newaxis, :], axis=0
        )
        estimated_rank = 1.0 + surface.pages * (
            exceed + smoothing
        ) / (len(anchors) + 2.0 * smoothing)
        for page, rank in zip(admitted, estimated_rank, strict=True):
            if rank <= visual_top_k:
                key = int(page)
                contributions[key] = contributions.get(key, 0.0) + (
                    visual_weight / (rrf_constant + float(rank))
                )
    if not contributions:
        return np.arange(surface.pages, dtype=np.int32)
    pages = np.fromiter(contributions, dtype=np.int32)
    scores = np.fromiter(
        (contributions[int(page)] for page in pages), dtype=np.float64
    )
    ranked = pages[np.lexsort((pages, -scores))]
    if len(ranked) < surface.pages:
        missing = np.setdiff1d(
            np.arange(surface.pages, dtype=np.int32), ranked, assume_unique=False
        )
        ranked = np.concatenate((ranked, missing))
    return ranked


def typed_materialization_ranking(
    surface: ScoreSurface,
    query_position: int,
    *,
    benefit_pages: Sequence[int],
    coverage_pages: Sequence[int],
    anchor_pages: Sequence[int],
    candidate_k: int = 100,
    benefit_quantile: float = 0.75,
    coverage_quantile: float = 0.99,
    benefit_weight: float = 1.0,
    coverage_weight: float = 1.0,
    page_calibration_queries: Sequence[int] | None = None,
    benefit_z_threshold: float = 0.0,
    coverage_z_threshold: float = 2.0,
) -> np.ndarray:
    """Fuse persistent visual pages according to two distinct capabilities.

    ``benefit_pages`` may only alter the ordering of pages already found by the
    cheap text locator.  They therefore amortize repeated visual reranking
    without silently becoming a second global locator.  ``coverage_pages`` may
    introduce pages from outside the text cohort, but must clear a stricter
    anchor-calibrated evidence gate.  This split prevents a sparse visual subset
    from being mistaken for a complete-corpus visual ranking.

    Only anchor, benefit, and coverage visual columns are read.
    """

    query = int(query_position)
    if query < 0 or query >= surface.queries:
        raise ValueError("query position outside surface")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if not 0.5 <= benefit_quantile < 1.0:
        raise ValueError("benefit_quantile must be in [0.5, 1)")
    if not benefit_quantile <= coverage_quantile < 1.0:
        raise ValueError("coverage_quantile must be at least benefit_quantile")
    if benefit_weight < 0.0 or coverage_weight < 0.0:
        raise ValueError("fusion weights must be nonnegative")

    calibration = None
    if page_calibration_queries is not None:
        calibration = _fit_typed_page_calibration(
            surface,
            pages=[*benefit_pages, *coverage_pages],
            anchor_pages=anchor_pages,
            calibration_queries=page_calibration_queries,
        )
    return _typed_ranking(
        surface,
        query,
        benefit_pages=benefit_pages,
        coverage_pages=coverage_pages,
        anchor_pages=anchor_pages,
        candidate_k=candidate_k,
        benefit_quantile=benefit_quantile,
        coverage_quantile=coverage_quantile,
        benefit_weight=benefit_weight,
        coverage_weight=coverage_weight,
        calibration=calibration,
        benefit_z_threshold=benefit_z_threshold,
        coverage_z_threshold=coverage_z_threshold,
    )


def _typed_ranking(
    surface: ScoreSurface,
    query: int,
    *,
    benefit_pages: Sequence[int],
    coverage_pages: Sequence[int],
    anchor_pages: Sequence[int],
    candidate_k: int,
    benefit_quantile: float,
    coverage_quantile: float,
    benefit_weight: float,
    coverage_weight: float,
    calibration: TypedCalibrationState | None,
    benefit_z_threshold: float,
    coverage_z_threshold: float,
) -> np.ndarray:

    anchors = _page_positions(surface, anchor_pages, allow_empty=False)
    benefit = _page_positions(surface, benefit_pages, allow_empty=True)
    coverage = _page_positions(surface, coverage_pages, allow_empty=True)
    text_candidates = surface.text_order[
        query, : min(candidate_k, surface.pages)
    ]
    candidate_mask = np.zeros(surface.pages, dtype=bool)
    candidate_mask[text_candidates] = True
    benefit = benefit[candidate_mask[benefit]]
    coverage = coverage[~candidate_mask[coverage]]

    scores = _zscore(np.asarray(surface.text_scores[query], dtype=np.float64))
    background = np.asarray(
        surface.visual_scores[query, anchors], dtype=np.float64
    )
    median = float(np.median(background))
    mad = float(np.median(np.abs(background - median)))
    scale = max(1.4826 * mad, float(background.std()), 1e-8)

    if calibration is None:
        benefit_residual = None
        coverage_residual = None
    else:
        query_center = float(np.median(background))
        benefit_residual = (
            np.asarray(surface.visual_scores[query, benefit], dtype=np.float64)
            - query_center
            - calibration.page_centers[benefit]
        ) / calibration.page_scales[benefit]
        coverage_residual = (
            np.asarray(surface.visual_scores[query, coverage], dtype=np.float64)
            - query_center
            - calibration.page_centers[coverage]
        ) / calibration.page_scales[coverage]

    if benefit.size:
        if benefit_residual is None:
            threshold = float(np.quantile(background, benefit_quantile))
            residual = (
                np.asarray(surface.visual_scores[query, benefit], dtype=np.float64)
                - threshold
            ) / scale
            gate = 0.0
        else:
            residual = benefit_residual
            gate = benefit_z_threshold
        residual = np.maximum(residual - gate, 0.0)
        scores[benefit] += benefit_weight * residual
    if coverage.size:
        if coverage_residual is None:
            threshold = float(np.quantile(background, coverage_quantile))
            residual = (
                np.asarray(surface.visual_scores[query, coverage], dtype=np.float64)
                - threshold
            ) / scale
            gate = 0.0
        else:
            residual = coverage_residual
            gate = coverage_z_threshold
        residual = np.maximum(residual - gate, 0.0)
        scores[coverage] += coverage_weight * residual

    pages = np.arange(surface.pages, dtype=np.int32)
    return pages[np.lexsort((pages, -scores))]


def evaluate_typed_materialization(
    surface: ScoreSurface,
    query_positions: Sequence[int],
    *,
    benefit_pages: Sequence[int],
    coverage_pages: Sequence[int],
    anchor_pages: Sequence[int],
    candidate_k: int = 100,
    benefit_quantile: float = 0.75,
    coverage_quantile: float = 0.99,
    benefit_weight: float = 1.0,
    coverage_weight: float = 1.0,
    calibration_queries: Sequence[int] | None = None,
    benefit_z_threshold: float = 0.0,
    coverage_z_threshold: float = 2.0,
) -> dict[str, Any]:
    """Evaluate a typed sparse visual index on an explicit query split."""

    queries = _query_positions(surface, query_positions)
    calibration = None
    if calibration_queries is not None:
        calibration = _fit_typed_page_calibration(
            surface,
            pages=[*benefit_pages, *coverage_pages],
            anchor_pages=anchor_pages,
            calibration_queries=calibration_queries,
        )
    per_query = []
    for query in queries:
        ranking = _typed_ranking(
            surface,
            int(query),
            benefit_pages=benefit_pages,
            coverage_pages=coverage_pages,
            anchor_pages=anchor_pages,
            candidate_k=candidate_k,
            benefit_quantile=benefit_quantile,
            coverage_quantile=coverage_quantile,
            benefit_weight=benefit_weight,
            coverage_weight=coverage_weight,
            calibration=calibration,
            benefit_z_threshold=benefit_z_threshold,
            coverage_z_threshold=coverage_z_threshold,
        )
        per_query.append(_ndcg_at_10(surface, int(query), ranking))
    return {
        "queries": int(len(queries)),
        "mean_ndcg_at_10": float(np.mean(per_query)),
        "per_query_ndcg_at_10": per_query,
    }


def _fit_typed_page_calibration(
    surface: ScoreSurface,
    *,
    pages: Sequence[int],
    anchor_pages: Sequence[int],
    calibration_queries: Sequence[int],
) -> TypedCalibrationState:
    selected = _page_positions(surface, pages, allow_empty=True)
    anchors = _page_positions(surface, anchor_pages, allow_empty=False)
    queries = _query_positions(surface, calibration_queries)
    centers = np.zeros(surface.pages, dtype=np.float64)
    scales = np.ones(surface.pages, dtype=np.float64)
    if selected.size:
        query_centers = np.median(
            np.asarray(surface.visual_scores[np.ix_(queries, anchors)]), axis=1
        )
        values = (
            np.asarray(surface.visual_scores[np.ix_(queries, selected)], dtype=np.float64)
            - query_centers[:, np.newaxis]
        )
        page_centers = np.median(values, axis=0)
        deviations = values - page_centers
        mad = 1.4826 * np.median(np.abs(deviations), axis=0)
        standard = deviations.std(axis=0)
        centers[selected] = page_centers
        scales[selected] = np.maximum(np.maximum(mad, standard), 1e-8)
    return TypedCalibrationState(page_centers=centers, page_scales=scales)


def fit_calibration(
    surface: ScoreSurface,
    *,
    admitted_pages: Sequence[int],
    anchor_pages: Sequence[int],
    calibration_queries: Sequence[int],
    config: CompilerConfig,
) -> CalibrationState:
    """Fit query/page bias and a support-size-aware evidence threshold."""

    anchors = _page_positions(surface, anchor_pages, allow_empty=False)
    admitted = _page_positions(surface, admitted_pages, allow_empty=True)
    history = _query_positions(surface, calibration_queries)
    anchor_matrix = np.asarray(
        surface.visual_scores[np.ix_(history, anchors)], dtype=np.float64
    )
    global_center = float(np.median(anchor_matrix))
    if admitted.size and config.two_way_centering:
        page_centers = np.median(
            np.asarray(surface.visual_scores[np.ix_(history, admitted)]), axis=0
        )
        history_query_centers = np.median(anchor_matrix, axis=1, keepdims=True)
        anchor_page_centers = np.median(anchor_matrix, axis=0, keepdims=True)
        background = (
            anchor_matrix
            - history_query_centers
            - anchor_page_centers
            + global_center
        )
        median = float(np.median(background))
        mad = float(np.median(np.abs(background - median)))
        scale = max(1.4826 * mad, float(background.std()), 1e-8)
    else:
        page_centers = np.zeros(admitted.size, dtype=np.float64)
        median = float(np.median(anchor_matrix))
        mad = float(np.median(np.abs(anchor_matrix - median)))
        scale = max(1.4826 * mad, float(anchor_matrix.std()), 1e-8)
    if config.familywise_alpha is None or admitted.size == 0:
        gate_threshold = 0.0
    else:
        tail_probability = config.familywise_alpha / max(1, admitted.size)
        gate_threshold = NormalDist().inv_cdf(1.0 - tail_probability)
    return CalibrationState(
        anchors=anchors,
        admitted=admitted,
        page_centers=np.asarray(page_centers, dtype=np.float64),
        global_center=global_center,
        scale=scale,
        gate_threshold=gate_threshold,
    )


def _ranking_with_calibration(
    surface: ScoreSurface,
    query: int,
    state: CalibrationState,
    config: CompilerConfig,
) -> np.ndarray:
    text = _zscore(np.asarray(surface.text_scores[query], dtype=np.float64))
    scores = text.copy()
    if state.admitted.size:
        visual = np.asarray(
            surface.visual_scores[query, state.admitted], dtype=np.float64
        )
        if config.two_way_centering:
            query_center = float(
                np.median(surface.visual_scores[query, state.anchors])
            )
            residual = (
                visual
                - query_center
                - state.page_centers
                + state.global_center
            ) / state.scale
            residual = np.maximum(residual - state.gate_threshold, 0.0)
        else:
            background = np.asarray(
                surface.visual_scores[query, state.anchors], dtype=np.float64
            )
            threshold = float(
                np.quantile(background, config.calibration_quantile)
            )
            median = float(np.median(background))
            mad = float(np.median(np.abs(background - median)))
            scale = max(1.4826 * mad, float(background.std()), 1e-8)
            residual = np.maximum((visual - threshold) / scale, 0.0)
        scores[state.admitted] += config.visual_weight * residual
    pages = np.arange(surface.pages, dtype=np.int32)
    return pages[np.lexsort((pages, -scores))]


def evaluate_compiled_index(
    surface: ScoreSurface,
    query_positions: Sequence[int],
    *,
    admitted_pages: Sequence[int],
    anchor_pages: Sequence[int],
    config: CompilerConfig,
    calibration_queries: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate one compiled sparse index on an explicit query split."""

    queries = _query_positions(surface, query_positions)
    if config.fusion == "anchor_rank":
        per_query = [
            _ndcg_at_10(
                surface,
                int(query),
                anchor_rank_ranking(
                    surface,
                    int(query),
                    admitted_pages=admitted_pages,
                    anchor_pages=anchor_pages,
                    rrf_constant=config.rrf_constant,
                    text_top_k=config.text_top_k,
                    visual_top_k=config.visual_top_k,
                    visual_weight=config.visual_weight,
                    smoothing=config.anchor_rank_smoothing,
                ),
            )
            for query in queries
        ]
        return {
            "queries": int(len(queries)),
            "mean_ndcg_at_10": float(np.mean(per_query)),
            "per_query_ndcg_at_10": per_query,
        }
    calibration = queries if calibration_queries is None else calibration_queries
    state = fit_calibration(
        surface,
        admitted_pages=admitted_pages,
        anchor_pages=anchor_pages,
        calibration_queries=calibration,
        config=config,
    )
    per_query = []
    for query in queries:
        ranking = _ranking_with_calibration(surface, int(query), state, config)
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
    if config.fusion == "anchor_rank":
        return float(
            np.mean(
                [
                    _ndcg_at_10(
                        surface,
                        int(query),
                        anchor_rank_ranking(
                            surface,
                            int(query),
                            admitted_pages=admitted,
                            anchor_pages=anchors,
                            rrf_constant=config.rrf_constant,
                            text_top_k=config.text_top_k,
                            visual_top_k=config.visual_top_k,
                            visual_weight=config.visual_weight,
                            smoothing=config.anchor_rank_smoothing,
                        ),
                    )
                    for query in queries
                ]
            )
        )
    state = fit_calibration(
        surface,
        admitted_pages=admitted,
        anchor_pages=anchors,
        calibration_queries=queries,
        config=config,
    )
    return float(
        np.mean(
            [
                _ndcg_at_10(
                    surface,
                    int(query),
                    _ranking_with_calibration(surface, int(query), state, config),
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
