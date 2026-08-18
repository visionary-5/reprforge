"""Public compiler API for compact multimodal index lifecycles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from numpy.typing import ArrayLike

from .coalescing import CoalescedState, topology_anchored_coalesce
from .index import CompactIndex
from .policy import BackboneProfile


@dataclass(frozen=True)
class CompilerConfig:
    profile: BackboneProfile

    def validate(self) -> None:
        self.profile.validate()


class ReprForgeCompiler:
    """Compile visual hidden states and build a compact persistent index.

    A model integration exposes the selected prefix boundary, calls
    :meth:`compile_hidden_state`, and continues the original frozen suffix at
    the returned compact positions. Retrieval endpoints then enter the index.
    """

    def __init__(self, config: CompilerConfig) -> None:
        config.validate()
        self.config = config

    def compile_hidden_state(
        self,
        hidden_states: ArrayLike,
        *,
        grid_shape: tuple[int, int],
        auxiliary_tokens: int = 0,
    ) -> CoalescedState:
        """Create the query-free topology-anchored execution state."""

        result = topology_anchored_coalesce(
            hidden_states,
            grid_shape=grid_shape,
            auxiliary_tokens=auxiliary_tokens,
        )
        profile = self.config.profile
        if result.plan.full_visual_tokens != profile.full_visual_tokens:
            raise ValueError("grid does not match the backbone's full capacity")
        if result.plan.compact_visual_tokens != profile.compact_visual_tokens:
            raise ValueError("anchor plan does not match compact capacity")
        return result

    def build(self, items: Iterable[tuple[str, ArrayLike]]) -> CompactIndex:
        """Make compact suffix endpoints persistent."""

        return CompactIndex(items)
