"""Column-subset reconstruction for workload-conditioned compression residuals."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


def _matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty matrix")
    return array


def pivoted_residual_columns(
    residual_fit: np.ndarray,
    count: int,
    *,
    costs: Sequence[float] | None = None,
) -> np.ndarray:
    """Select document columns with deterministic cost-aware pivoted QR.

    Rows are centered and scaled so no fit query dominates merely because its
    score residuals have a larger numeric range.  Pivoting then covers the
    query-space directions of the workload residual matrix.
    """

    residual = _matrix(residual_fit, name="residual_fit")
    if not 0 < count <= residual.shape[1]:
        raise ValueError("column count must lie inside the corpus")
    byte_costs = (
        np.ones(residual.shape[1], dtype=np.float64)
        if costs is None
        else np.asarray(costs, dtype=np.float64)
    )
    if byte_costs.shape != (residual.shape[1],) or np.any(byte_costs <= 0) or not np.isfinite(byte_costs).all():
        raise ValueError("column costs must be positive and corpus-aligned")
    centered = residual - residual.mean(axis=1, keepdims=True)
    row_scale = centered.std(axis=1, keepdims=True)
    standardized = centered / np.maximum(row_scale, 1e-8)
    remaining_energy = np.sum(standardized * standardized, axis=0)
    original_energy = remaining_energy.copy()
    selected: list[int] = []
    basis_vectors: list[np.ndarray] = []
    chosen = np.zeros(residual.shape[1], dtype=bool)
    for _ in range(count):
        density = np.divide(
            remaining_energy,
            byte_costs,
            out=np.full_like(remaining_energy, -np.inf),
            where=~chosen,
        )
        density[chosen] = -np.inf
        pivot = int(np.argmax(density))
        vector = standardized[:, pivot].copy()
        # Modified Gram-Schmidt against the orthonormal pivot directions.
        for basis in basis_vectors:
            vector -= basis * float(basis @ vector)
        norm = float(np.linalg.norm(vector))
        selected.append(pivot)
        chosen[pivot] = True
        if norm <= 1e-10:
            remaining_energy[pivot] = 0.0
            # Once numerical rank is exhausted, deterministic original energy
            # keeps the requested physical budget well-defined.
            remaining_energy[~chosen] = original_energy[~chosen]
            continue
        direction = vector / norm
        basis_vectors.append(direction)
        projection = direction @ standardized
        remaining_energy = np.maximum(
            remaining_energy - projection * projection,
            0.0,
        )
        remaining_energy[chosen] = 0.0
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class ResidualColumnModel:
    anchor_positions: np.ndarray
    anchor_mean: np.ndarray
    anchor_scale: np.ndarray
    coefficients: np.ndarray
    ridge: float


@dataclass(frozen=True)
class LowRankResidualModel:
    anchor_positions: np.ndarray
    document_basis: np.ndarray
    anchor_design: np.ndarray
    ridge: float
    rank: int


def fit_residual_column_model(
    residual_fit: np.ndarray,
    anchor_positions: Sequence[int],
    *,
    ridge: float = 1.0,
) -> ResidualColumnModel:
    """Fit a multi-output ridge map from anchor to full residual columns."""

    residual = _matrix(residual_fit, name="residual_fit")
    anchors = np.asarray(anchor_positions, dtype=np.int64)
    if (
        anchors.ndim != 1
        or not len(anchors)
        or len(np.unique(anchors)) != len(anchors)
        or np.any(anchors < 0)
        or np.any(anchors >= residual.shape[1])
    ):
        raise ValueError("anchor positions must be unique and corpus-aligned")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    anchor_values = residual[:, anchors]
    anchor_mean = anchor_values.mean(axis=0)
    anchor_scale = np.maximum(anchor_values.std(axis=0), 1e-8)
    normalized = (anchor_values - anchor_mean) / anchor_scale
    design = np.column_stack((np.ones(len(residual)), normalized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ residual,
    )
    return ResidualColumnModel(
        anchor_positions=anchors.copy(),
        anchor_mean=anchor_mean,
        anchor_scale=anchor_scale,
        coefficients=coefficients,
        ridge=float(ridge),
    )


def predict_residual_columns(
    anchor_residuals: np.ndarray,
    model: ResidualColumnModel,
    *,
    clip_to_observed_range: bool = True,
) -> np.ndarray:
    """Reconstruct complete residual rows from physically observed anchors."""

    values = _matrix(anchor_residuals, name="anchor_residuals")
    if values.shape[1] != len(model.anchor_positions):
        raise ValueError("anchor residuals do not match the fitted model")
    normalized = (values - model.anchor_mean) / model.anchor_scale
    design = np.column_stack((np.ones(len(values)), normalized))
    prediction = design @ model.coefficients
    if clip_to_observed_range:
        prediction = np.clip(
            prediction,
            values.min(axis=1, keepdims=True),
            values.max(axis=1, keepdims=True),
        )
    prediction[:, model.anchor_positions] = values
    return prediction


def residual_column_score_surface(
    cheap_scores: np.ndarray,
    anchor_full_scores: np.ndarray,
    model: ResidualColumnModel,
) -> np.ndarray:
    """Return full-score estimates using cheap scores plus completed residuals."""

    cheap = _matrix(cheap_scores, name="cheap_scores")
    anchor_full = _matrix(anchor_full_scores, name="anchor_full_scores")
    if anchor_full.shape != (len(cheap), len(model.anchor_positions)):
        raise ValueError("full anchor scores are not query/anchor aligned")
    anchor_residual = cheap[:, model.anchor_positions]
    anchor_residual = anchor_full - anchor_residual
    residual = predict_residual_columns(anchor_residual, model)
    scores = cheap + residual
    scores[:, model.anchor_positions] = anchor_full
    return scores


def _pivoted_basis_columns(
    basis: np.ndarray,
    count: int,
    costs: Sequence[float],
) -> np.ndarray:
    values = _matrix(basis, name="document_basis")
    if not 0 < count <= values.shape[1]:
        raise ValueError("anchor count must lie inside the corpus")
    byte_costs = np.asarray(costs, dtype=np.float64)
    if byte_costs.shape != (values.shape[1],) or np.any(byte_costs <= 0):
        raise ValueError("anchor costs must be positive and corpus-aligned")
    normalized = values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12
    )
    energy = np.sum(normalized * normalized, axis=0)
    original = energy.copy()
    chosen = np.zeros(values.shape[1], dtype=bool)
    selected: list[int] = []
    basis_vectors: list[np.ndarray] = []
    for _ in range(count):
        density = energy / byte_costs
        density[chosen] = -np.inf
        pivot = int(np.argmax(density))
        vector = normalized[:, pivot].copy()
        for direction in basis_vectors:
            vector -= direction * float(direction @ vector)
        norm = float(np.linalg.norm(vector))
        selected.append(pivot)
        chosen[pivot] = True
        if norm <= 1e-10:
            energy[~chosen] = original[~chosen]
            energy[chosen] = 0.0
            continue
        direction = vector / norm
        basis_vectors.append(direction)
        projection = direction @ normalized
        energy = np.maximum(energy - projection * projection, 0.0)
        energy[chosen] = 0.0
    return np.asarray(selected, dtype=np.int64)


def fit_low_rank_residual_model(
    residual_fit: np.ndarray,
    *,
    rank: int,
    anchor_count: int,
    costs: Sequence[float] | None = None,
    ridge: float = 1e-2,
) -> LowRankResidualModel:
    """Fit a truncated-SVD document basis and a stable anchor design."""

    residual = _matrix(residual_fit, name="residual_fit")
    maximum_rank = min(residual.shape) - 1
    if not 0 < rank <= maximum_rank:
        raise ValueError("rank must lie below the residual matrix dimensions")
    if anchor_count < rank + 1 or anchor_count > residual.shape[1]:
        raise ValueError("anchor_count must be at least rank + 1 and within corpus")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    centered = residual - residual.mean(axis=1, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    document_basis = np.vstack(
        (np.ones(residual.shape[1], dtype=np.float64), right[:rank])
    )
    byte_costs = (
        np.ones(residual.shape[1], dtype=np.float64)
        if costs is None
        else np.asarray(costs, dtype=np.float64)
    )
    anchors = _pivoted_basis_columns(document_basis, anchor_count, byte_costs)
    return LowRankResidualModel(
        anchor_positions=anchors,
        document_basis=document_basis,
        anchor_design=document_basis[:, anchors].T,
        ridge=float(ridge),
        rank=int(rank),
    )


def low_rank_residual_score_surface(
    cheap_scores: np.ndarray,
    anchor_full_scores: np.ndarray,
    model: LowRankResidualModel,
    *,
    clip_to_observed_range: bool = True,
) -> np.ndarray:
    """Infer future-query latent residuals from selected document columns."""

    cheap = _matrix(cheap_scores, name="cheap_scores")
    anchor_full = _matrix(anchor_full_scores, name="anchor_full_scores")
    if anchor_full.shape != (len(cheap), len(model.anchor_positions)):
        raise ValueError("full-anchor scores do not match the low-rank model")
    anchor_residual = anchor_full - cheap[:, model.anchor_positions]
    design = model.anchor_design
    penalty = np.eye(design.shape[1], dtype=np.float64) * model.ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ anchor_residual.T,
    ).T
    residual = coefficients @ model.document_basis
    if clip_to_observed_range:
        residual = np.clip(
            residual,
            anchor_residual.min(axis=1, keepdims=True),
            anchor_residual.max(axis=1, keepdims=True),
        )
    scores = cheap + residual
    scores[:, model.anchor_positions] = anchor_full
    return scores


def two_stage_candidate_surface(
    cheap_scores: np.ndarray,
    rerank_scores: np.ndarray,
    *,
    candidate_k: int = 100,
) -> np.ndarray:
    """Preserve the cheap Top-k set and rerank only inside that cohort."""

    cheap = _matrix(cheap_scores, name="cheap_scores")
    rerank = _matrix(rerank_scores, name="rerank_scores")
    if cheap.shape != rerank.shape:
        raise ValueError("cheap and rerank surfaces must be aligned")
    if not 0 < candidate_k <= cheap.shape[1]:
        raise ValueError("candidate_k must lie inside the corpus")
    corpus = cheap.shape[1]
    positions = np.arange(corpus)
    output = np.empty_like(cheap)
    for query_index in range(len(cheap)):
        cheap_order = np.lexsort((positions, -cheap[query_index]))
        output[query_index, cheap_order] = -np.arange(corpus, dtype=np.float64)
        candidates = cheap_order[:candidate_k]
        reranked = candidates[
            np.lexsort((candidates, -rerank[query_index, candidates]))
        ]
        output[query_index, reranked] = corpus + np.arange(
            candidate_k, 0, -1, dtype=np.float64
        )
    return output
