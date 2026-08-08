"""Leakage-safe two-action page materialization policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .costs import CostCatalog


@dataclass(frozen=True)
class PageSignals:
    page_ids: np.ndarray
    fit_candidate_events: np.ndarray
    text_chars: np.ndarray
    grayscale_entropy: np.ndarray
    edge_energy: np.ndarray
    locator_disagreement: np.ndarray

    def validate(self) -> None:
        pages = len(self.page_ids)
        if pages == 0 or len(set(map(int, self.page_ids))) != pages:
            raise ValueError("page_ids must be non-empty and unique")
        if not np.array_equal(
            np.asarray(self.page_ids, dtype=np.int64), np.arange(pages)
        ):
            raise ValueError("page_ids must be contiguous score-surface positions")
        for name, values in self.__dict__.items():
            array = np.asarray(values)
            if array.shape != (pages,):
                raise ValueError(f"{name} must contain one value per page")
            if name != "page_ids" and (not np.all(np.isfinite(array)) or np.any(array < 0.0)):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class PolicyConfig:
    feature_budget_fraction: float
    retrieval_budget_fraction: float
    prior_query_strength: float = 5.0
    disagreement_weight: float = 1.0
    text_scarcity_weight: float = 0.5
    visual_complexity_weight: float = 0.25
    retrieval_reuse_weight: float = 0.25

    def validate(self) -> None:
        for name in ("feature_budget_fraction", "retrieval_budget_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.prior_query_strength < 0.0 or not math.isfinite(self.prior_query_strength):
            raise ValueError("prior_query_strength must be finite and nonnegative")
        for name in (
            "disagreement_weight",
            "text_scarcity_weight",
            "visual_complexity_weight",
            "retrieval_reuse_weight",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class CompiledPlan:
    feature_pages: tuple[int, ...]
    retrieval_pages: tuple[int, ...]
    feature_budget_pages: int
    retrieval_budget_pages: int
    expected_feature_net_seconds: float
    feature_break_even_future_uses: float
    protocol: dict[str, Any]


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    deviation = float(values.std())
    if deviation <= 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / deviation


def _top_pages(page_ids: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((page_ids, -np.asarray(values, dtype=np.float64)))
    return np.asarray(page_ids[order[:count]], dtype=np.int32)


def compile_plan(
    signals: PageSignals,
    costs: CostCatalog,
    *,
    fit_queries: int,
    horizon_queries: int,
    config: PolicyConfig,
) -> tuple[CompiledPlan, dict[str, np.ndarray]]:
    """Compile feature reuse and retrieval coverage without future labels/scores."""

    signals.validate()
    costs.validate()
    config.validate()
    if fit_queries <= 0 or horizon_queries <= 0:
        raise ValueError("fit_queries and horizon_queries must be positive")
    pages = len(signals.page_ids)
    feature_budget = min(pages, int(math.ceil(config.feature_budget_fraction * pages)))
    retrieval_budget = min(pages, int(math.ceil(config.retrieval_budget_fraction * pages)))

    events = np.asarray(signals.fit_candidate_events, dtype=np.float64)
    global_rate = float(events.sum()) / float(fit_queries * pages)
    posterior_rate = (
        events + config.prior_query_strength * global_rate
    ) / (fit_queries + config.prior_query_strength)
    expected_future_uses = posterior_rate * float(horizon_queries)
    feature_net = expected_future_uses * costs.feature_saving_per_use - costs.offline_feature_cost
    eligible = feature_net > 0.0
    feature_order = _top_pages(signals.page_ids, feature_net, pages)
    feature_pages = feature_order[eligible[feature_order]][:feature_budget]

    scarcity = -_zscore(np.log1p(np.asarray(signals.text_chars, dtype=np.float64)))
    complexity = 0.5 * (
        _zscore(signals.grayscale_entropy) + _zscore(signals.edge_energy)
    )
    disagreement = _zscore(signals.locator_disagreement)
    reuse = _zscore(expected_future_uses)
    retrieval_score = (
        config.disagreement_weight * disagreement
        + config.text_scarcity_weight * scarcity
        + config.visual_complexity_weight * complexity
        + config.retrieval_reuse_weight * reuse
    )
    retrieval_pages = _top_pages(signals.page_ids, retrieval_score, retrieval_budget)

    plan = CompiledPlan(
        feature_pages=tuple(map(int, feature_pages)),
        retrieval_pages=tuple(map(int, retrieval_pages)),
        feature_budget_pages=feature_budget,
        retrieval_budget_pages=retrieval_budget,
        expected_feature_net_seconds=float(feature_net[feature_pages].sum()),
        feature_break_even_future_uses=costs.feature_break_even_future_uses,
        protocol={
            "future_queries_visible": False,
            "future_relevance_visible": False,
            "future_visual_scores_visible": False,
            "feature_action": "positive posterior reuse net value under page budget",
            "retrieval_action": "fixed qrel-free risk and reuse score under page budget",
        },
    )
    diagnostics = {
        "posterior_candidate_rate": posterior_rate,
        "expected_future_uses": expected_future_uses,
        "feature_net_seconds": feature_net,
        "retrieval_risk_score": retrieval_score,
    }
    return plan, diagnostics
