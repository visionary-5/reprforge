"""Public compiler API for compact multimodal index lifecycles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from numpy.typing import ArrayLike

from .alignment import TrajectoryAlignment, fit_trajectory_alignment
from .index import CompactIndex
from .policy import BackboneProfile


@dataclass(frozen=True)
class CompilerConfig:
    profile: BackboneProfile
    rank: int = 8
    fit_steps: int = 25
    learning_rate: float = 1e-2
    identity_weight: float = 1e-3
    gradient_clip_norm: float = 1.0
    calibration_fraction: float = 0.03125
    seed: int = 0

    def validate(self) -> None:
        self.profile.validate()
        if self.rank <= 0 or self.rank > self.profile.compact_visual_tokens:
            raise ValueError("alignment rank is outside compact capacity")
        if self.fit_steps <= 0 or self.learning_rate <= 0:
            raise ValueError("fit steps and learning rate must be positive")
        if self.identity_weight < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("invalid alignment regularization")
        if not 0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration fraction must be in (0, 0.5)")


class ReprForgeCompiler:
    """Fit a query-free correction and build a compact persistent index.

    Model-specific integrations own the prefix/coalescing/compact-suffix
    execution. This class owns the model-agnostic endpoint correction, index,
    and candidate refinement contract.
    """

    def __init__(self, config: CompilerConfig) -> None:
        config.validate()
        self.config = config
        self._alignment: TrajectoryAlignment | None = None

    @property
    def alignment(self) -> TrajectoryAlignment:
        if self._alignment is None:
            raise RuntimeError("fit the compiler before building an index")
        return self._alignment

    def fit(
        self,
        compact_canary: Iterable[ArrayLike],
        full_canary_in_compact_slots: Iterable[ArrayLike],
    ) -> TrajectoryAlignment:
        """Fit on paired endpoints without queries, qrels, or answers."""

        self._alignment = fit_trajectory_alignment(
            compact_canary,
            full_canary_in_compact_slots,
            rank=self.config.rank,
            steps=self.config.fit_steps,
            learning_rate=self.config.learning_rate,
            identity_weight=self.config.identity_weight,
            gradient_clip_norm=self.config.gradient_clip_norm,
            seed=self.config.seed,
        )
        return self._alignment

    def build(self, items: Iterable[tuple[str, ArrayLike]]) -> CompactIndex:
        """Align compact endpoints and make them persistent."""

        alignment = self.alignment
        return CompactIndex(
            (item_id, alignment.transform(vectors)) for item_id, vectors in items
        )
