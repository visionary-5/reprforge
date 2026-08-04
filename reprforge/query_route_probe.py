"""Cross-fitted query-only probes for representation routing.

This module is intentionally a probe rather than a final learned method.  It
tests whether the query-route oracle exposed by the Heterogeneity Atlas is
predictable from information available before an expensive route is run.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from reprforge.heterogeneity_atlas import (
    ScoreCube,
    paired_bootstrap_ci,
    query_metrics,
)


_TOKEN = re.compile(r"[\w%$€£]+", re.UNICODE)


def _hash(value: str, modulus: int) -> tuple[int, float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus, (
        1.0 if digest[8] % 2 == 0 else -1.0
    )


def lexical_hash_features(texts: Sequence[str], *, dimensions: int = 256) -> np.ndarray:
    """Stable signed unigram/bigram hashing plus transparent shape features."""

    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    features = np.zeros((len(texts), dimensions + 6), dtype=np.float64)
    for row, text in enumerate(texts):
        tokens = [token.lower() for token in _TOKEN.findall(text)]
        terms = [f"u:{token}" for token in tokens]
        terms.extend(f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        for term in terms:
            column, sign = _hash(term, dimensions)
            features[row, column] += sign
        scale = max(math_sqrt(len(terms)), 1.0)
        features[row, :dimensions] /= scale
        characters = max(len(text), 1)
        features[row, dimensions:] = (
            math_log1p(len(tokens)),
            math_log1p(len(text)),
            sum(character.isdigit() for character in text) / characters,
            float("%" in text),
            float("?" in text),
            float(any(token in {"table", "chart", "figure", "graph"} for token in tokens)),
        )
    return features


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def math_log1p(value: float) -> float:
    return float(np.log1p(value))


def categorical_features(
    rows: Sequence[Mapping],
    *,
    fields: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Fit-free one-hot encoding for a frozen benchmark metadata vocabulary."""

    names = sorted(
        {
            f"{field}={value}"
            for row in rows
            for field in fields
            for value in _as_values(row.get(field))
        }
    )
    positions = {name: index for index, name in enumerate(names)}
    features = np.zeros((len(rows), len(names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for field in fields:
            for value in _as_values(row.get(field)):
                features[row_index, positions[f"{field}={value}"]] = 1.0
    return features, tuple(names)


def _as_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def cheap_route_features(scores: np.ndarray) -> np.ndarray:
    """Query-level statistics available after running one cheap retriever."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("cheap route scores must be a finite matrix")
    ordered = np.sort(values, axis=1)[:, ::-1]
    width = values.shape[1]
    top5 = ordered[:, : min(5, width)]
    top10 = ordered[:, : min(10, width)]
    return np.column_stack(
        [
            ordered[:, 0],
            ordered[:, 0] - ordered[:, min(1, width - 1)],
            ordered[:, 0] - ordered[:, min(4, width - 1)],
            ordered[:, 0] - ordered[:, min(9, width - 1)],
            top5.mean(axis=1),
            top10.mean(axis=1),
            values.mean(axis=1),
            values.std(axis=1),
        ]
    )


def candidate_identity_features(
    scores: np.ndarray, *, candidate_k: int = 20
) -> np.ndarray:
    """Rank-weighted candidate identities exposed by a cheap first stage.

    Unlike aggregate score statistics, this represents which evidence the
    cheap retriever surfaced.  It is query-time observable and contains no
    relevance labels or expensive-route values.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("cheap route scores must be a finite matrix")
    if not 1 <= candidate_k <= values.shape[1]:
        raise ValueError("candidate_k must lie inside the corpus")
    features = np.zeros(values.shape, dtype=np.float64)
    positions = np.arange(values.shape[1])
    for query_index, row in enumerate(values):
        candidates = np.lexsort((positions, -row))[:candidate_k]
        weights = 1.0 / np.log2(np.arange(2, candidate_k + 2))
        features[query_index, candidates] = weights
    return features


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    penalty: float,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-8)
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    target_mean = train_y.mean(axis=0)
    centered = train_y - target_mean
    # Dual ridge is cheaper and more stable when hashed features outnumber queries.
    coefficients = train.T @ np.linalg.solve(
        train @ train.T + penalty * np.eye(len(train)), centered
    )
    return test @ coefficients + target_mean


def _fold_ids(query_ids: Sequence[str], folds: int) -> np.ndarray:
    return np.asarray(
        [
            int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % folds
            for value in query_ids
        ],
        dtype=np.int32,
    )


def _balanced_group_fold_ids(groups: Sequence[str], folds: int) -> np.ndarray:
    counts = Counter(str(value) for value in groups)
    if len(counts) < folds:
        raise ValueError(
            f"grouped cross-fit needs at least {folds} disconnected groups; "
            f"found {len(counts)}"
        )
    loads = [0] * folds
    assignments = {}
    ordered = sorted(
        counts,
        key=lambda value: (
            -counts[value],
            hashlib.sha256(value.encode()).hexdigest(),
        ),
    )
    for value in ordered:
        fold = min(range(folds), key=lambda index: (loads[index], index))
        assignments[value] = fold
        loads[fold] += counts[value]
    return np.asarray([assignments[str(value)] for value in groups], dtype=np.int32)


def crossfit_query_router(
    cube: ScoreCube,
    features: np.ndarray,
    *,
    target_metric: str,
    k: int,
    folds: int = 5,
    penalties: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    groups: Sequence[str] | None = None,
) -> dict:
    """Nested cross-fit a ridge utility predictor and evaluate routed rankings."""

    cube.validate()
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(cube.query_ids):
        raise ValueError("features must be a query-aligned matrix")
    if not np.isfinite(x).all() or folds < 3:
        raise ValueError("features must be finite and folds must be at least three")
    route_metrics = {
        route: query_metrics(values, cube.relevance, ks=(k,))[target_metric]
        for route, values in cube.scores.items()
    }
    utilities = np.stack([route_metrics[route] for route in cube.routes], axis=1)
    if groups is not None and len(groups) != len(cube.query_ids):
        raise ValueError("groups must be query-aligned")
    fold_ids = (
        _balanced_group_fold_ids(groups, folds)
        if groups is not None
        else _fold_ids(cube.query_ids, folds)
    )
    predictions = np.full_like(utilities, np.nan)
    baseline_routes = np.full(len(cube.query_ids), -1, dtype=np.int32)
    selected_penalties: list[float] = []
    for fold in range(folds):
        outer_eval = fold_ids == fold
        outer_train = ~outer_eval
        inner_eval = fold_ids == ((fold + 1) % folds)
        inner_train = outer_train & ~inner_eval
        if not outer_eval.any() or not inner_train.any() or not inner_eval.any():
            raise ValueError("cross-fit fold is empty")
        best_penalty = None
        best_value = -np.inf
        for penalty in penalties:
            predicted = _ridge_predict(
                x[inner_train], utilities[inner_train], x[inner_eval], penalty=penalty
            )
            choices = np.argmax(predicted, axis=1)
            value = float(utilities[inner_eval][np.arange(len(choices)), choices].mean())
            if value > best_value:
                best_value = value
                best_penalty = float(penalty)
        assert best_penalty is not None
        selected_penalties.append(best_penalty)
        predictions[outer_eval] = _ridge_predict(
            x[outer_train], utilities[outer_train], x[outer_eval], penalty=best_penalty
        )
        fit_route_means = utilities[outer_train].mean(axis=0)
        baseline_routes[outer_eval] = int(np.argmax(fit_route_means))

    selected = np.argmax(predictions, axis=1)
    oracle = np.argmax(utilities, axis=1)
    query_positions = np.arange(len(cube.query_ids))
    routed_values = utilities[query_positions, selected]
    baseline_values = utilities[query_positions, baseline_routes]
    oracle_values = utilities[query_positions, oracle]
    denominator = float((oracle_values - baseline_values).mean())
    return {
        "queries": len(cube.query_ids),
        "folds": folds,
        "split_unit": "group" if groups is not None else "query",
        "target_metric": target_metric,
        "mean_metric": float(routed_values.mean()),
        "best_global_crossfit_mean_metric": float(baseline_values.mean()),
        "query_oracle_mean_metric": float(oracle_values.mean()),
        "gap_over_best_global": paired_bootstrap_ci(routed_values, baseline_values),
        "oracle_gap_recovery": (
            float((routed_values - baseline_values).mean() / denominator)
            if abs(denominator) > 1e-12
            else 0.0
        ),
        "oracle_route_accuracy": float(np.mean(selected == oracle)),
        "selected_routes": {
            route: int(np.sum(selected == index))
            for index, route in enumerate(cube.routes)
        },
        "oracle_routes": {
            route: int(np.sum(oracle == index))
            for index, route in enumerate(cube.routes)
        },
        "selected_penalties": dict(Counter(selected_penalties)),
    }
