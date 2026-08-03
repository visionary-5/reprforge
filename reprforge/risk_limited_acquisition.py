"""Risk-limited acquisition of unbuilt multimodal representations.

The expensive score of a document is unavailable until its visual
representation has been built.  This module learns query-level conformal
score envelopes from cheap, pre-build features and acquires exact visual
evidence until the requested Top-k boundary is separated.

The implementation is deliberately model-independent.  It operates on score
surfaces and physical costs; an online executor can replace an observation
from a frozen surface with a real encoder call without changing the policy.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _as_finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a non-empty 2-D matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _row_zscore(values: np.ndarray) -> np.ndarray:
    values = _as_finite_matrix(values, name="scores")
    mean = values.mean(axis=1, keepdims=True)
    scale = np.maximum(values.std(axis=1, keepdims=True), 1e-12)
    return (values - mean) / scale


def _stable_top_indices(
    scores: np.ndarray,
    identifiers: Sequence[str],
    count: int,
) -> np.ndarray:
    if count <= 0 or count > len(scores):
        raise ValueError("top count must lie inside the score row")
    return np.asarray(
        sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), str(identifiers[index])),
        )[:count],
        dtype=np.int32,
    )


@dataclass(frozen=True)
class CandidateSurface:
    """Cheap and fully observed values for one frozen candidate pool."""

    candidate_indices: np.ndarray
    base_scores: np.ndarray
    visual_scores: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        candidates = np.asarray(self.candidate_indices)
        base = np.asarray(self.base_scores)
        visual = np.asarray(self.visual_scores)
        features = np.asarray(self.features)
        if candidates.ndim != 2:
            raise ValueError("candidate_indices must be 2-D")
        if base.shape != candidates.shape or visual.shape != candidates.shape:
            raise ValueError("candidate and score matrices must align")
        if features.shape[:2] != candidates.shape or features.ndim != 3:
            raise ValueError("feature tensor must align with candidates")
        if features.shape[2] != len(self.feature_names):
            raise ValueError("feature names do not match feature tensor")
        if not np.isfinite(base).all() or not np.isfinite(visual).all():
            raise ValueError("candidate scores must be finite")
        if not np.isfinite(features).all():
            raise ValueError("candidate features must be finite")


def build_candidate_surface(
    corpus_ids: Sequence[str],
    locator_scores: np.ndarray,
    visual_scores: np.ndarray,
    *,
    query_token_counts: Sequence[int],
    page_text_token_counts: Sequence[int],
    candidate_pool: int = 100,
) -> CandidateSurface:
    """Build the deployable score/feature view used by the controller.

    Visual MaxSim is divided by the number of effective query tokens.  The
    remaining train-only affine normalization lives in
    :class:`ConformalEnvelopeModel`, avoiding any use of unobserved visual
    rows at runtime.
    """

    locator = _as_finite_matrix(locator_scores, name="locator_scores")
    visual = _as_finite_matrix(visual_scores, name="visual_scores")
    if locator.shape != visual.shape:
        raise ValueError("locator and visual score surfaces must align")
    if locator.shape[1] != len(corpus_ids):
        raise ValueError("corpus identifiers do not match score columns")
    if candidate_pool <= 0 or candidate_pool > locator.shape[1]:
        raise ValueError("candidate_pool must lie inside the corpus")
    query_tokens = np.asarray(query_token_counts, dtype=np.int64)
    page_tokens = np.asarray(page_text_token_counts, dtype=np.int64)
    if query_tokens.shape != (locator.shape[0],) or np.any(query_tokens <= 0):
        raise ValueError("query token counts must be positive and query-aligned")
    if page_tokens.shape != (locator.shape[1],) or np.any(page_tokens < 0):
        raise ValueError("page token counts must be non-negative and corpus-aligned")

    candidates = np.stack(
        [
            _stable_top_indices(row, corpus_ids, candidate_pool)
            for row in locator
        ]
    )
    rows = np.arange(locator.shape[0])[:, None]
    candidate_locator = locator[rows, candidates]
    base = _row_zscore(candidate_locator)
    candidate_visual = visual[rows, candidates] / query_tokens[:, None]
    ranks = np.arange(1, candidate_pool + 1, dtype=np.float64)[None, :]
    query_margin = base[:, :1] - base[:, -1:]
    query_dispersion = np.std(candidate_locator, axis=1, keepdims=True)
    features = np.stack(
        [
            base,
            np.broadcast_to(
                np.log1p(ranks) / math.log1p(candidate_pool), base.shape
            ),
            np.broadcast_to(np.log1p(query_tokens)[:, None], base.shape),
            np.log1p(page_tokens[candidates]),
            np.broadcast_to(query_margin, base.shape),
            np.broadcast_to(np.log1p(query_dispersion), base.shape),
        ],
        axis=2,
    )
    return CandidateSurface(
        candidate_indices=candidates,
        base_scores=base,
        visual_scores=candidate_visual,
        features=features,
        feature_names=(
            "bm25_zscore",
            "normalized_log_rank",
            "log_query_tokens",
            "log_page_text_tokens",
            "bm25_top_to_tail_margin",
            "log_bm25_dispersion",
        ),
    )


def _ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    penalty: float,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    regularizer[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ target)


def _weighted_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    penalty: float,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    root = np.sqrt(np.asarray(weights, dtype=np.float64))
    weighted_design = design * root[:, None]
    weighted_target = target * root
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        weighted_design.T @ weighted_design + regularizer,
        weighted_design.T @ weighted_target,
    )


def _predict(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    return coefficients[0] + features @ coefficients[1:]


def conformal_quantile(values: Sequence[float], *, alpha: float) -> float:
    """Finite-sample split-conformal upper quantile."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("conformal scores must be a finite non-empty vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), len(scores))
    return float(np.partition(scores, rank - 1)[rank - 1])


@dataclass(frozen=True)
class ScoreIntervals:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class ConformalEnvelopeModel:
    """Ridge location/scale model with a query-level conformal multiplier."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    mean_coefficients: np.ndarray
    scale_coefficients: np.ndarray
    upper_multiplier: float
    lower_multiplier: float
    alpha: float
    ridge_penalty: float

    @classmethod
    def fit(
        cls,
        fit_features: np.ndarray,
        fit_targets: np.ndarray,
        calibration_features: np.ndarray,
        calibration_targets: np.ndarray,
        *,
        alpha: float = 0.05,
        ridge_penalty: float = 1.0,
    ) -> "ConformalEnvelopeModel":
        fit_x = np.asarray(fit_features, dtype=np.float64)
        fit_y = np.asarray(fit_targets, dtype=np.float64)
        cal_x = np.asarray(calibration_features, dtype=np.float64)
        cal_y = np.asarray(calibration_targets, dtype=np.float64)
        if fit_x.ndim != 3 or cal_x.ndim != 3:
            raise ValueError("fit and calibration features must be Q x C x F")
        if fit_y.shape != fit_x.shape[:2] or cal_y.shape != cal_x.shape[:2]:
            raise ValueError("targets must align with feature query/candidate axes")
        if fit_x.shape[2] != cal_x.shape[2]:
            raise ValueError("fit and calibration feature widths differ")
        if not np.isfinite(fit_x).all() or not np.isfinite(cal_x).all():
            raise ValueError("features must be finite")
        if not np.isfinite(fit_y).all() or not np.isfinite(cal_y).all():
            raise ValueError("targets must be finite")
        if ridge_penalty < 0 or not math.isfinite(ridge_penalty):
            raise ValueError("ridge_penalty must be finite and non-negative")

        flat_x = fit_x.reshape(-1, fit_x.shape[2])
        flat_y = fit_y.reshape(-1)
        feature_mean = flat_x.mean(axis=0)
        feature_scale = np.maximum(flat_x.std(axis=0), 1e-12)
        target_mean = float(flat_y.mean())
        target_scale = max(float(flat_y.std()), 1e-12)
        normalized_x = (flat_x - feature_mean) / feature_scale
        normalized_y = (flat_y - target_mean) / target_scale
        mean_coefficients = _ridge(
            normalized_x,
            normalized_y,
            penalty=ridge_penalty,
        )
        fit_mean = _predict(mean_coefficients, normalized_x)
        log_residual = np.log(np.abs(normalized_y - fit_mean) + 1e-6)
        scale_coefficients = _ridge(
            normalized_x,
            log_residual,
            penalty=ridge_penalty,
        )

        cal_normalized_x = (cal_x - feature_mean) / feature_scale
        cal_mean = _predict(mean_coefficients, cal_normalized_x)
        cal_log_scale = _predict(scale_coefficients, cal_normalized_x)
        cal_scale = np.exp(np.clip(cal_log_scale, -20.0, 20.0)) + 1e-6
        cal_normalized_y = (cal_y - target_mean) / target_scale
        upper_scores = np.max(
            (cal_normalized_y - cal_mean) / cal_scale,
            axis=1,
        )
        lower_scores = np.max(
            (cal_mean - cal_normalized_y) / cal_scale,
            axis=1,
        )
        upper_multiplier = max(0.0, conformal_quantile(upper_scores, alpha=alpha))
        lower_multiplier = max(0.0, conformal_quantile(lower_scores, alpha=alpha))
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            mean_coefficients=mean_coefficients,
            scale_coefficients=scale_coefficients,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier,
            alpha=alpha,
            ridge_penalty=ridge_penalty,
        )

    def normalize_targets(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.target_mean) / self.target_scale

    def predict_intervals(self, features: np.ndarray) -> ScoreIntervals:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim not in {2, 3} or values.shape[-1] != len(self.feature_mean):
            raise ValueError("features have an incompatible shape")
        normalized = (values - self.feature_mean) / self.feature_scale
        mean = _predict(self.mean_coefficients, normalized)
        log_scale = _predict(self.scale_coefficients, normalized)
        scale = np.exp(np.clip(log_scale, -20.0, 20.0)) + 1e-6
        return ScoreIntervals(
            mean=mean,
            lower=mean - self.lower_multiplier * scale,
            upper=mean + self.upper_multiplier * scale,
            scale=scale,
        )

    @property
    def conformal_multiplier(self) -> float:
        """Backward-compatible name for the decision-relevant upper bound."""

        return self.upper_multiplier


@dataclass(frozen=True)
class ConformalCandidateSetModel:
    """Query-level conformal set containing the full-score Top-k members."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    inclusion_threshold: float
    cutoff: int
    alpha: float
    ridge_penalty: float

    @classmethod
    def fit(
        cls,
        fit_features: np.ndarray,
        fit_membership: np.ndarray,
        calibration_features: np.ndarray,
        calibration_membership: np.ndarray,
        *,
        cutoff: int,
        alpha: float = 0.05,
        ridge_penalty: float = 1.0,
    ) -> "ConformalCandidateSetModel":
        fit_x = np.asarray(fit_features, dtype=np.float64)
        fit_y = np.asarray(fit_membership, dtype=bool)
        cal_x = np.asarray(calibration_features, dtype=np.float64)
        cal_y = np.asarray(calibration_membership, dtype=bool)
        if fit_x.ndim != 3 or cal_x.ndim != 3:
            raise ValueError("candidate-set features must be Q x C x F")
        if fit_y.shape != fit_x.shape[:2] or cal_y.shape != cal_x.shape[:2]:
            raise ValueError("membership labels must align with candidate features")
        if fit_x.shape[2] != cal_x.shape[2]:
            raise ValueError("fit and calibration feature widths differ")
        if cutoff <= 0 or cutoff > fit_x.shape[1]:
            raise ValueError("cutoff must lie inside the candidate pool")
        if not np.all(fit_y.sum(axis=1) == cutoff):
            raise ValueError("every fit query must label exactly cutoff members")
        if not np.all(cal_y.sum(axis=1) == cutoff):
            raise ValueError("every calibration query must label exactly cutoff members")

        flat_x = fit_x.reshape(-1, fit_x.shape[2])
        flat_y = fit_y.reshape(-1).astype(np.float64)
        feature_mean = flat_x.mean(axis=0)
        feature_scale = np.maximum(flat_x.std(axis=0), 1e-12)
        normalized = (flat_x - feature_mean) / feature_scale
        positive_weight = (fit_x.shape[1] - cutoff) / cutoff
        weights = np.where(flat_y > 0.5, positive_weight, 1.0)
        coefficients = _weighted_ridge(
            normalized,
            flat_y,
            weights,
            penalty=ridge_penalty,
        )
        cal_scores = _predict(
            coefficients,
            (cal_x - feature_mean) / feature_scale,
        )
        minimum_member_score = np.asarray(
            [float(scores[members].min()) for scores, members in zip(cal_scores, cal_y, strict=True)]
        )
        nonconformity = -minimum_member_score
        threshold = -conformal_quantile(nonconformity, alpha=alpha)
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            coefficients=coefficients,
            inclusion_threshold=threshold,
            cutoff=cutoff,
            alpha=alpha,
            ridge_penalty=ridge_penalty,
        )

    def scores(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim not in {2, 3} or values.shape[-1] != len(self.feature_mean):
            raise ValueError("features have an incompatible shape")
        return _predict(
            self.coefficients,
            (values - self.feature_mean) / self.feature_scale,
        )

    def predict_sets(self, features: np.ndarray) -> np.ndarray:
        scores = self.scores(features)
        if scores.ndim != 2:
            raise ValueError("batched query features are required")
        selected = scores >= self.inclusion_threshold
        for query in range(len(scores)):
            if int(selected[query].sum()) >= self.cutoff:
                continue
            order = np.argsort(-scores[query], kind="stable")[: self.cutoff]
            selected[query, order] = True
        return selected


@dataclass(frozen=True)
class AcquisitionResult:
    ranking: np.ndarray
    acquired_indices: tuple[int, ...]
    acquisition_batches: tuple[tuple[int, ...], ...]
    certified: bool
    exhausted_pool: bool
    cutoff_score: float
    max_unobserved_upper: float

    @property
    def acquired_count(self) -> int:
        return len(self.acquired_indices)


def acquire_until_certified(
    base_scores: np.ndarray,
    exact_visual_scores: np.ndarray,
    intervals: ScoreIntervals,
    identifiers: Sequence[str],
    *,
    cutoff: int,
    batch_size: int = 4,
    build_costs: Sequence[float] | None = None,
) -> AcquisitionResult:
    """Acquire exact visual evidence until the Top-k boundary is separated."""

    base = np.asarray(base_scores, dtype=np.float64)
    exact_visual = np.asarray(exact_visual_scores, dtype=np.float64)
    mean = np.asarray(intervals.mean, dtype=np.float64)
    lower = base + np.asarray(intervals.lower, dtype=np.float64)
    upper = base + np.asarray(intervals.upper, dtype=np.float64)
    if not (
        base.ndim == 1
        and exact_visual.shape == base.shape
        and mean.shape == base.shape
        and lower.shape == base.shape
        and upper.shape == base.shape
        and len(identifiers) == len(base)
    ):
        raise ValueError("scores, intervals, and identifiers must align")
    if cutoff <= 0 or cutoff > len(base):
        raise ValueError("cutoff must lie inside the candidate pool")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    costs = (
        np.ones(len(base), dtype=np.float64)
        if build_costs is None
        else np.asarray(build_costs, dtype=np.float64)
    )
    if costs.shape != base.shape or np.any(~np.isfinite(costs)) or np.any(costs <= 0):
        raise ValueError("build costs must be positive, finite, and candidate-aligned")

    observed = np.zeros(len(base), dtype=bool)
    exact_final = base + exact_visual
    batches: list[tuple[int, ...]] = []
    acquisition_order: list[int] = []
    predicted_final = base + mean
    initial = _stable_top_indices(predicted_final, identifiers, cutoff)
    pending = initial.tolist()
    cutoff_score = float("-inf")
    max_unobserved_upper = float("inf")

    while True:
        if pending:
            batch = tuple(int(value) for value in pending[:batch_size])
            batches.append(batch)
            for index in batch:
                if observed[index]:
                    continue
                observed[index] = True
                acquisition_order.append(index)
            pending = pending[batch_size:]
            if pending:
                continue

        observed_indices = np.flatnonzero(observed)
        if len(observed_indices) >= cutoff:
            observed_order = sorted(
                observed_indices,
                key=lambda index: (-float(exact_final[index]), str(identifiers[index])),
            )
            cutoff_score = float(exact_final[observed_order[cutoff - 1]])
            unobserved = np.flatnonzero(~observed)
            max_unobserved_upper = (
                float(np.max(upper[unobserved])) if len(unobserved) else float("-inf")
            )
            if cutoff_score >= max_unobserved_upper:
                ranking = np.asarray(observed_order[:cutoff], dtype=np.int32)
                return AcquisitionResult(
                    ranking=ranking,
                    acquired_indices=tuple(acquisition_order),
                    acquisition_batches=tuple(batches),
                    certified=True,
                    exhausted_pool=not len(unobserved),
                    cutoff_score=cutoff_score,
                    max_unobserved_upper=max_unobserved_upper,
                )
        unobserved = np.flatnonzero(~observed)
        if not len(unobserved):
            ranking = _stable_top_indices(exact_final, identifiers, cutoff)
            return AcquisitionResult(
                ranking=ranking,
                acquired_indices=tuple(acquisition_order),
                acquisition_batches=tuple(batches),
                certified=True,
                exhausted_pool=True,
                cutoff_score=float(exact_final[ranking[-1]]),
                max_unobserved_upper=float("-inf"),
            )

        threshold = cutoff_score
        ambiguous = [
            int(index) for index in unobserved if float(upper[index]) >= threshold
        ]
        if not ambiguous:
            # Numerical guard: the boundary condition above should have fired.
            ambiguous = [int(unobserved[np.argmax(upper[unobserved])])]
        ambiguous.sort(
            key=lambda index: (
                -float((upper[index] - lower[index]) / costs[index]),
                -float(upper[index]),
                str(identifiers[index]),
            )
        )
        pending = ambiguous[:batch_size]


def simultaneous_coverage(
    exact_visual_scores: np.ndarray,
    intervals: ScoreIntervals,
) -> np.ndarray:
    """Return one coverage bit per query for complete candidate envelopes."""

    targets = np.asarray(exact_visual_scores, dtype=np.float64)
    if targets.shape != intervals.lower.shape or targets.ndim != 2:
        raise ValueError("targets and intervals must be aligned query matrices")
    return np.all(
        (targets >= intervals.lower) & (targets <= intervals.upper),
        axis=1,
    )


def simultaneous_upper_coverage(
    exact_visual_scores: np.ndarray,
    intervals: ScoreIntervals,
) -> np.ndarray:
    """Coverage needed by Top-k stopping: no hidden score exceeds its UCB."""

    targets = np.asarray(exact_visual_scores, dtype=np.float64)
    if targets.shape != intervals.upper.shape or targets.ndim != 2:
        raise ValueError("targets and intervals must be aligned query matrices")
    return np.all(targets <= intervals.upper, axis=1)


def balanced_group_folds(
    groups: Sequence[str],
    *,
    fold_count: int = 5,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign complete groups to deterministically load-balanced folds."""

    if fold_count < 3:
        raise ValueError("at least three folds are required for fit/calibration/test")
    values = [str(value) for value in groups]
    counts = Counter(values)
    if len(counts) < fold_count:
        raise ValueError("the number of source groups must cover every fold")
    loads = [0] * fold_count
    assignment: dict[str, int] = {}
    for group, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(fold_count), key=lambda index: (loads[index], index))
        assignment[group] = fold
        loads[fold] += count
    return (
        np.asarray([assignment[value] for value in values], dtype=np.int16),
        assignment,
    )


@dataclass(frozen=True)
class CrossFitAcquisitionResult:
    rankings: np.ndarray
    teacher_rankings: np.ndarray
    teacher_scores: np.ndarray
    acquired_counts: np.ndarray
    acquired_pages: tuple[tuple[int, ...], ...]
    coverage: np.ndarray
    certified: np.ndarray
    exhausted: np.ndarray
    folds: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CrossFitCandidateSetResult:
    rankings: np.ndarray
    teacher_rankings: np.ndarray
    acquired_counts: np.ndarray
    acquired_pages: tuple[tuple[int, ...], ...]
    topk_covered: np.ndarray
    folds: tuple[dict[str, Any], ...]


def crossfit_boundary_acquisition(
    surface: CandidateSurface,
    corpus_ids: Sequence[str],
    groups: Sequence[str],
    *,
    cutoff: int,
    alpha: float = 0.05,
    batch_size: int = 4,
    fold_count: int = 5,
    ridge_penalty: float = 1.0,
    page_build_costs: Sequence[float] | None = None,
) -> CrossFitAcquisitionResult:
    """Run group-disjoint fit/calibration/test acquisition.

    For outer fold ``f``, fold ``(f + 1) mod F`` is calibration and all other
    non-test folds fit the location/scale models.  This makes every reported
    query independent of both model fitting and conformal calibration while
    keeping the split deterministic.
    """

    query_count, candidate_count = surface.candidate_indices.shape
    if len(groups) != query_count:
        raise ValueError("source groups do not match the query count")
    if len(corpus_ids) <= int(np.max(surface.candidate_indices)):
        raise ValueError("candidate indices exceed the corpus identifiers")
    if cutoff <= 0 or cutoff > candidate_count:
        raise ValueError("cutoff must lie inside the candidate pool")
    costs = None
    if page_build_costs is not None:
        costs = np.asarray(page_build_costs, dtype=np.float64)
        if costs.shape != (len(corpus_ids),):
            raise ValueError("page build costs must be corpus-aligned")

    query_folds, assignment = balanced_group_folds(groups, fold_count=fold_count)
    rankings = np.empty((query_count, cutoff), dtype=np.int32)
    teacher = np.empty_like(rankings)
    teacher_scores = np.empty_like(surface.base_scores, dtype=np.float64)
    acquired_counts = np.empty(query_count, dtype=np.int32)
    acquired_pages: list[tuple[int, ...] | None] = [None] * query_count
    coverage = np.empty(query_count, dtype=bool)
    certified = np.empty(query_count, dtype=bool)
    exhausted = np.empty(query_count, dtype=bool)
    fold_records: list[dict[str, Any]] = []

    for test_fold in range(fold_count):
        calibration_fold = (test_fold + 1) % fold_count
        test = np.flatnonzero(query_folds == test_fold)
        calibration = np.flatnonzero(query_folds == calibration_fold)
        fit = np.flatnonzero(
            (query_folds != test_fold) & (query_folds != calibration_fold)
        )
        model = ConformalEnvelopeModel.fit(
            surface.features[fit],
            surface.visual_scores[fit],
            surface.features[calibration],
            surface.visual_scores[calibration],
            alpha=alpha,
            ridge_penalty=ridge_penalty,
        )
        intervals = model.predict_intervals(surface.features[test])
        exact_normalized = model.normalize_targets(surface.visual_scores[test])
        teacher_scores[test] = surface.base_scores[test] + exact_normalized
        fold_coverage = simultaneous_upper_coverage(exact_normalized, intervals)
        coverage[test] = fold_coverage
        fold_acquired: list[int] = []
        fold_exhausted = 0
        for local_query, query in enumerate(test):
            candidate_pages = surface.candidate_indices[query]
            candidate_ids = [corpus_ids[int(page)] for page in candidate_pages]
            result = acquire_until_certified(
                surface.base_scores[query],
                exact_normalized[local_query],
                ScoreIntervals(
                    mean=intervals.mean[local_query],
                    lower=intervals.lower[local_query],
                    upper=intervals.upper[local_query],
                    scale=intervals.scale[local_query],
                ),
                candidate_ids,
                cutoff=cutoff,
                batch_size=batch_size,
                build_costs=(None if costs is None else costs[candidate_pages]),
            )
            rankings[query] = candidate_pages[result.ranking]
            teacher_local = _stable_top_indices(
                surface.base_scores[query] + exact_normalized[local_query],
                candidate_ids,
                cutoff,
            )
            teacher[query] = candidate_pages[teacher_local]
            acquired_counts[query] = result.acquired_count
            absolute_acquired = tuple(
                int(candidate_pages[index]) for index in result.acquired_indices
            )
            acquired_pages[query] = absolute_acquired
            certified[query] = result.certified
            exhausted[query] = result.exhausted_pool
            fold_acquired.append(result.acquired_count)
            fold_exhausted += int(result.exhausted_pool)
        fold_records.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_queries": int(len(fit)),
                "calibration_queries": int(len(calibration)),
                "test_queries": int(len(test)),
                "upper_conformal_multiplier": model.upper_multiplier,
                "lower_diagnostic_multiplier": model.lower_multiplier,
                "simultaneous_upper_coverage": float(fold_coverage.mean()),
                "mean_acquired_pages": float(np.mean(fold_acquired)),
                "exhausted_queries": fold_exhausted,
                "test_source_groups": sorted(
                    group for group, fold in assignment.items() if fold == test_fold
                ),
            }
        )
    if any(value is None for value in acquired_pages):
        raise AssertionError("cross-fit acquisition left a query unassigned")
    return CrossFitAcquisitionResult(
        rankings=rankings,
        teacher_rankings=teacher,
        teacher_scores=teacher_scores,
        acquired_counts=acquired_counts,
        acquired_pages=tuple(value for value in acquired_pages if value is not None),
        coverage=coverage,
        certified=certified,
        exhausted=exhausted,
        folds=tuple(fold_records),
    )


def crossfit_candidate_set_acquisition(
    surface: CandidateSurface,
    corpus_ids: Sequence[str],
    groups: Sequence[str],
    *,
    cutoff: int,
    alpha: float = 0.05,
    fold_count: int = 5,
    ridge_penalty: float = 1.0,
) -> CrossFitCandidateSetResult:
    """Acquire a conformal set calibrated to contain the full-score Top-k."""

    query_count, candidate_count = surface.candidate_indices.shape
    if len(groups) != query_count:
        raise ValueError("source groups do not match the query count")
    if len(corpus_ids) <= int(np.max(surface.candidate_indices)):
        raise ValueError("candidate indices exceed the corpus identifiers")
    if cutoff <= 0 or cutoff > candidate_count:
        raise ValueError("cutoff must lie inside the candidate pool")
    query_folds, assignment = balanced_group_folds(groups, fold_count=fold_count)
    rankings = np.empty((query_count, cutoff), dtype=np.int32)
    teacher = np.empty_like(rankings)
    acquired_counts = np.empty(query_count, dtype=np.int32)
    acquired_pages: list[tuple[int, ...] | None] = [None] * query_count
    topk_covered = np.empty(query_count, dtype=bool)
    fold_records: list[dict[str, Any]] = []

    for test_fold in range(fold_count):
        calibration_fold = (test_fold + 1) % fold_count
        test = np.flatnonzero(query_folds == test_fold)
        calibration = np.flatnonzero(query_folds == calibration_fold)
        fit = np.flatnonzero(
            (query_folds != test_fold) & (query_folds != calibration_fold)
        )
        # The location model supplies the same train-only visual affine scale
        # as the score-envelope path.  Membership calibration remains separate.
        score_model = ConformalEnvelopeModel.fit(
            surface.features[fit],
            surface.visual_scores[fit],
            surface.features[calibration],
            surface.visual_scores[calibration],
            alpha=alpha,
            ridge_penalty=ridge_penalty,
        )
        fit_final = surface.base_scores[fit] + score_model.normalize_targets(
            surface.visual_scores[fit]
        )
        calibration_final = surface.base_scores[calibration] + score_model.normalize_targets(
            surface.visual_scores[calibration]
        )
        test_final = surface.base_scores[test] + score_model.normalize_targets(
            surface.visual_scores[test]
        )

        def membership(values: np.ndarray, query_indices: np.ndarray) -> np.ndarray:
            labels = np.zeros_like(values, dtype=bool)
            for local, query in enumerate(query_indices):
                candidate_ids = [
                    corpus_ids[int(page)] for page in surface.candidate_indices[query]
                ]
                chosen = _stable_top_indices(values[local], candidate_ids, cutoff)
                labels[local, chosen] = True
            return labels

        fit_membership = membership(fit_final, fit)
        calibration_membership = membership(calibration_final, calibration)
        test_membership = membership(test_final, test)
        set_model = ConformalCandidateSetModel.fit(
            surface.features[fit],
            fit_membership,
            surface.features[calibration],
            calibration_membership,
            cutoff=cutoff,
            alpha=alpha,
            ridge_penalty=ridge_penalty,
        )
        selected = set_model.predict_sets(surface.features[test])
        fold_coverage = np.all(selected | ~test_membership, axis=1)
        topk_covered[test] = fold_coverage
        fold_sizes: list[int] = []
        for local, query in enumerate(test):
            selected_offsets = np.flatnonzero(selected[local])
            pages = surface.candidate_indices[query]
            candidate_ids = [corpus_ids[int(page)] for page in pages]
            selected_order = sorted(
                selected_offsets,
                key=lambda offset: (
                    -float(test_final[local, offset]),
                    candidate_ids[offset],
                ),
            )[:cutoff]
            rankings[query] = pages[np.asarray(selected_order, dtype=np.int32)]
            teacher_offsets = _stable_top_indices(
                test_final[local], candidate_ids, cutoff
            )
            teacher[query] = pages[teacher_offsets]
            selected_pages = tuple(int(pages[offset]) for offset in selected_offsets)
            acquired_pages[query] = selected_pages
            acquired_counts[query] = len(selected_pages)
            fold_sizes.append(len(selected_pages))
        fold_records.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_queries": int(len(fit)),
                "calibration_queries": int(len(calibration)),
                "test_queries": int(len(test)),
                "inclusion_threshold": set_model.inclusion_threshold,
                "topk_set_coverage": float(fold_coverage.mean()),
                "mean_acquired_pages": float(np.mean(fold_sizes)),
                "test_source_groups": sorted(
                    group for group, fold in assignment.items() if fold == test_fold
                ),
            }
        )
    if any(value is None for value in acquired_pages):
        raise AssertionError("candidate-set acquisition left a query unassigned")
    return CrossFitCandidateSetResult(
        rankings=rankings,
        teacher_rankings=teacher,
        acquired_counts=acquired_counts,
        acquired_pages=tuple(value for value in acquired_pages if value is not None),
        topk_covered=topk_covered,
        folds=tuple(fold_records),
    )
