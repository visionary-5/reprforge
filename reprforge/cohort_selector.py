"""Cheap-surface probes for learning order-preserving cohort certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reprforge.cohort_certificate import _completed_values, greedy_certificate
from reprforge.heterogeneity_atlas import ScoreCube, paired_bootstrap_ci, query_metrics
from reprforge.landmark_probe import (
    _completed_rerank_scores,
    _initial_order,
    _zscore,
)


def candidate_features(
    base_row: np.ndarray,
    candidates: np.ndarray,
    *,
    target_k: int,
) -> np.ndarray:
    """Return query-local, label-free features for each candidate position."""

    values = np.asarray(base_row, dtype=np.float64)[candidates]
    z = _zscore(values)
    count = len(values)
    positions = np.arange(count, dtype=np.float64)
    scale = max(count - 1, 1)
    previous_gap = np.r_[0.0, z[:-1] - z[1:]]
    next_gap = np.r_[z[:-1] - z[1:], 0.0]
    boundary = z[min(target_k - 1, count - 1)]
    top_span = z[0] - boundary
    tail_span = boundary - z[-1]
    probabilities = np.exp(z - z.max())
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    query_features = np.asarray(
        [top_span, tail_span, entropy / np.log(max(count, 2))], dtype=np.float64
    )
    relative = positions / scale
    margin = z - boundary
    return np.column_stack(
        [
            relative,
            relative**2,
            z,
            z**2,
            previous_gap,
            next_gap,
            margin,
            np.abs(margin),
            relative * z,
            np.tile(query_features, (count, 1)),
        ]
    )


@dataclass
class RidgeRankSelector:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        regularization: float = 1.0,
    ) -> "RidgeRankSelector":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalized = (x - mean) / scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        positives = max(float(y.sum()), 1.0)
        negatives = max(float(len(y) - y.sum()), 1.0)
        weights = np.where(y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
        weighted = design * np.sqrt(weights[:, None])
        target = y * np.sqrt(weights)
        penalty = np.eye(design.shape[1]) * regularization
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(weighted.T @ weighted + penalty, weighted.T @ target)
        return cls(mean=mean, scale=scale, coefficients=coefficients)

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        return design @ self.coefficients


@dataclass
class RandomFeatureRankSelector:
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    bias: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        random_features: int = 192,
        regularization: float = 4.0,
        seed: int = 20260803,
    ) -> "RandomFeatureRankSelector":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalized = (x - mean) / scale
        rng = np.random.default_rng(seed)
        projection = rng.normal(
            0.0, 1.0 / np.sqrt(normalized.shape[1]),
            size=(normalized.shape[1], random_features),
        )
        bias = rng.normal(0.0, 0.5, size=random_features)
        hidden = np.maximum(normalized @ projection + bias, 0.0)
        design = np.column_stack([np.ones(len(normalized)), normalized, hidden])
        positives = max(float(y.sum()), 1.0)
        negatives = max(float(len(y) - y.sum()), 1.0)
        weights = np.where(
            y > 0.5,
            len(y) / (2.0 * positives),
            len(y) / (2.0 * negatives),
        )
        weighted = design * np.sqrt(weights[:, None])
        target = y * np.sqrt(weights)
        penalty = np.eye(design.shape[1]) * regularization
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            weighted.T @ weighted + penalty, weighted.T @ target
        )
        return cls(mean, scale, projection, bias, coefficients)

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        hidden = np.maximum(normalized @ self.projection + self.bias, 0.0)
        design = np.column_stack([np.ones(len(normalized)), normalized, hidden])
        return design @ self.coefficients


@dataclass
class RidgeRegressor:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        regularization: float = 1.0,
    ) -> "RidgeRegressor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalized = (x - mean) / scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        penalty = np.eye(design.shape[1]) * regularization
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ y
        )
        return cls(mean=mean, scale=scale, coefficients=coefficients)

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        return design @ self.coefficients


def _active_completed_values(
    x: np.ndarray,
    raw_expensive: np.ndarray,
    predicted_z: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        [predicted_z[observed], np.ones(int(observed.sum()))]
    )
    penalty = np.diag([1e-3, 0.0])
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ raw_expensive[observed],
    )
    completed = coefficients[0] * predicted_z + coefficients[1]
    completed = np.asarray(completed, dtype=np.float64)
    completed[observed] = raw_expensive[observed]
    return x + _zscore(completed)


def _select(
    policy: str,
    *,
    budget: int,
    candidate_k: int,
    predictions: np.ndarray | None,
    anchors: int,
) -> np.ndarray:
    coverage_order = _initial_order(candidate_k)
    mandatory = coverage_order[:anchors]
    if policy == "coverage":
        return np.asarray(coverage_order[:budget], dtype=np.int32)
    if policy == "top":
        ordered = mandatory + [position for position in range(candidate_k) if position not in mandatory]
        return np.asarray(ordered[:budget], dtype=np.int32)
    if policy in {"ridge", "random_feature"}:
        if predictions is None:
            raise ValueError(f"{policy} policy requires predictions")
        remaining = [position for position in range(candidate_k) if position not in mandatory]
        ranked = sorted(remaining, key=lambda position: (-predictions[position], position))
        return np.asarray((mandatory + ranked)[:budget], dtype=np.int32)
    raise ValueError(f"unknown selector policy: {policy}")


def analyze_selector_probe(
    cube: ScoreCube,
    *,
    base_route: str,
    expensive_route: str,
    candidate_k: int = 20,
    target_k: int = 5,
    target_metric: str = "ndcg_at_5",
    budgets: Sequence[int] = (5, 8, 12),
    per_item_build_ms: Sequence[float] | None = None,
    anchors: int = 3,
    candidate_pair_features: Sequence[np.ndarray] | None = None,
    pair_feature_description: str | None = None,
    policies: Sequence[str] = (
        "top",
        "coverage",
        "ridge",
        "random_feature",
        "active",
    ),
) -> dict:
    """Fit on fit queries and evaluate cheap-only selectors on eval queries."""

    cube.validate()
    base = cube.scores[base_route]
    expensive = cube.scores[expensive_route]
    query_certificates = []
    features = []
    labels = []
    expensive_targets = []
    for query_index in range(len(cube.query_ids)):
        certificate = greedy_certificate(
            base[query_index],
            expensive[query_index],
            candidate_k=candidate_k,
            target_k=target_k,
            anchors=anchors,
            objective="order",
        )
        query_certificates.append(certificate)
        query_features = candidate_features(
            base[query_index], certificate.candidate_indices, target_k=target_k
        )
        if candidate_pair_features is not None:
            pair = np.asarray(candidate_pair_features[query_index], dtype=np.float64)
            if pair.ndim != 2 or pair.shape[0] != candidate_k:
                raise ValueError("candidate pair features must be candidate-aligned")
            query_features = np.column_stack([query_features, pair])
        query_labels = np.zeros(candidate_k, dtype=np.float64)
        query_labels[certificate.observed_positions] = 1.0
        features.append(query_features)
        labels.append(query_labels)
        expensive_targets.append(
            _zscore(expensive[query_index, certificate.candidate_indices])
        )
    fit_mask = np.asarray([role == "fit" for role in cube.split_roles])
    eval_indices = np.flatnonzero(~fit_mask)
    model = RidgeRankSelector.fit(
        np.concatenate([features[index] for index in np.flatnonzero(fit_mask)]),
        np.concatenate([labels[index] for index in np.flatnonzero(fit_mask)]),
    )
    random_feature_model = RandomFeatureRankSelector.fit(
        np.concatenate([features[index] for index in np.flatnonzero(fit_mask)]),
        np.concatenate([labels[index] for index in np.flatnonzero(fit_mask)]),
    )
    completion_model = RidgeRegressor.fit(
        np.concatenate([features[index] for index in np.flatnonzero(fit_mask)]),
        np.concatenate(
            [expensive_targets[index] for index in np.flatnonzero(fit_mask)]
        ),
    )
    fit_residuals = np.stack(
        [
            expensive_targets[index] - completion_model.predict(features[index])
            for index in np.flatnonzero(fit_mask)
        ]
    )
    position_uncertainty = np.maximum(fit_residuals.std(axis=0), 0.05)
    teacher_surface = np.stack(
        [
            _completed_rerank_scores(
                base[index],
                certificate.candidate_indices,
                _zscore(base[index, certificate.candidate_indices])
                + _zscore(expensive[index, certificate.candidate_indices]),
            )
            for index, certificate in enumerate(query_certificates)
        ]
    )
    base_quality = query_metrics(base, cube.relevance, ks=(target_k,))[target_metric]
    teacher_quality = query_metrics(
        teacher_surface, cube.relevance, ks=(target_k,)
    )[target_metric]
    reports = {}
    item_costs = None if per_item_build_ms is None else np.asarray(per_item_build_ms)
    for budget in budgets:
        if not anchors <= budget <= candidate_k:
            raise ValueError("budgets must lie between anchors and candidate_k")
        for policy in policies:
            if policy not in {"top", "coverage", "ridge", "random_feature", "active"}:
                raise ValueError(f"unknown selector policy: {policy}")
            surfaces = []
            selected_documents = set()
            agreements = []
            for query_index in eval_indices:
                certificate = query_certificates[query_index]
                predictions = None
                if policy == "ridge":
                    predictions = model.predict(features[query_index])
                elif policy == "random_feature":
                    predictions = random_feature_model.predict(features[query_index])
                observed = np.zeros(candidate_k, dtype=bool)
                x = _zscore(base[query_index, certificate.candidate_indices])
                raw_expensive = expensive[
                    query_index, certificate.candidate_indices
                ]
                if policy == "active":
                    predicted_z = completion_model.predict(features[query_index])
                    observed[_initial_order(candidate_k)[:anchors]] = True
                    while int(observed.sum()) < budget:
                        completed = _active_completed_values(
                            x, raw_expensive, predicted_z, observed
                        )
                        boundary = np.sort(completed)[-target_k]
                        priority = position_uncertainty / (
                            np.abs(completed - boundary) + 0.05
                        )
                        priority[observed] = -np.inf
                        observed[int(np.argmax(priority))] = True
                    selected = np.flatnonzero(observed)
                    completed = _active_completed_values(
                        x, raw_expensive, predicted_z, observed
                    )
                else:
                    selected = _select(
                        policy,
                        budget=budget,
                        candidate_k=candidate_k,
                        predictions=predictions,
                        anchors=anchors,
                    )
                    observed[selected] = True
                    completed = _completed_values(
                        x, raw_expensive, observed
                    )
                surfaces.append(
                    _completed_rerank_scores(
                        base[query_index], certificate.candidate_indices, completed
                    )
                )
                ranked = np.lexsort((np.arange(candidate_k), -completed))[:target_k]
                agreements.append(
                    float(np.mean(ranked == certificate.teacher_top_positions))
                )
                selected_documents.update(
                    int(certificate.candidate_indices[position]) for position in selected
                )
            values = query_metrics(
                np.stack(surfaces),
                tuple(cube.relevance[index] for index in eval_indices),
                ks=(target_k,),
            )[target_metric]
            base_eval = base_quality[eval_indices]
            teacher_eval = teacher_quality[eval_indices]
            full_gain = float((teacher_eval - base_eval).mean())
            selected_gain = float((values - base_eval).mean())
            physical = {"unique_documents": len(selected_documents)}
            if item_costs is not None:
                physical["unique_build_ms"] = float(
                    item_costs[list(selected_documents)].sum()
                )
            reports[f"{policy}_b{budget}"] = {
                "quality": float(values.mean()),
                "vs_teacher": paired_bootstrap_ci(values, teacher_eval),
                "full_fusion_gain_recovery": (
                    selected_gain / full_gain if abs(full_gain) > 1e-12 else 0.0
                ),
                "mean_exact_position_agreement": float(np.mean(agreements)),
                "physical": physical,
            }
    return {
        "schema_version": 1,
        "queries": len(cube.query_ids),
        "fit_queries": int(fit_mask.sum()),
        "eval_queries": int(len(eval_indices)),
        "candidate_k": candidate_k,
        "target_k": target_k,
        "target_metric": target_metric,
        "base_route": base_route,
        "expensive_route": expensive_route,
        "selector_features": (
            "query-local cheap scores, ranks, gaps, margins, and profile statistics"
            + (f"; {pair_feature_description}" if pair_feature_description else "")
        ),
        "selector_uses_expensive_scores_at_inference": False,
        "active_policy_uses_selected_expensive_scores_at_inference": True,
        "selector_uses_qrels": False,
        "eval_base": float(base_quality[eval_indices].mean()),
        "eval_full_teacher": float(teacher_quality[eval_indices].mean()),
        "fit_positive_rate": float(
            np.concatenate([labels[index] for index in np.flatnonzero(fit_mask)]).mean()
        ),
        "policies": reports,
    }
