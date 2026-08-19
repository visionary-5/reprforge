"""End-to-end orchestration of plan, model adapter, and persistent index."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from numpy.typing import ArrayLike

from ..adapters import DocumentEncoderAdapter
from ..indexing import CompactIndex
from ..planning import BackboneProfile, CompilePlan, plan_topology_anchored
from .coalescing import CoalescedState, coalesce_hidden_states


@dataclass(frozen=True)
class CompilerConfig:
    profile: BackboneProfile
    grid_shape: tuple[int, int]

    def build_plan(self) -> CompilePlan:
        return plan_topology_anchored(self.profile, grid_shape=self.grid_shape)


class ReprForgeCompiler:
    """Lower one physical plan through a model adapter into an index."""

    def __init__(self, config: CompilerConfig) -> None:
        self.config = config
        self.plan = config.build_plan()

    def compile_hidden_state(
        self,
        hidden_states: ArrayLike,
        *,
        auxiliary_tokens: int = 0,
    ) -> CoalescedState:
        """Compile a boundary tensor into compact suffix worker states."""

        return coalesce_hidden_states(
            hidden_states,
            grid_shape=self.plan.grid_shape,
            auxiliary_tokens=auxiliary_tokens,
        )

    def compile_document(
        self,
        adapter: DocumentEncoderAdapter,
        document: object,
    ) -> ArrayLike:
        """Run Full prefix, compact lowering, and the original frozen suffix."""

        if adapter.profile != self.plan.profile:
            raise ValueError("adapter profile does not match the compile plan")
        boundary = adapter.run_prefix(
            document,
            split_after_layer=self.plan.profile.split_after_layer,
        )
        if boundary.grid_shape != self.plan.grid_shape:
            raise ValueError("adapter boundary grid does not match the compile plan")
        compact = self.compile_hidden_state(
            boundary.hidden_states,
            auxiliary_tokens=boundary.auxiliary_tokens,
        )
        return adapter.run_suffix(boundary, compact)

    def build(self, items: Iterable[tuple[str, ArrayLike]]) -> CompactIndex:
        """Persist already-compiled suffix endpoints in the reference index."""

        return CompactIndex(items)

    def build_documents(
        self,
        adapter: DocumentEncoderAdapter,
        documents: Iterable[tuple[str, object]],
    ) -> CompactIndex:
        """Compile a document collection through one frozen physical plan."""

        return CompactIndex(
            (item_id, self.compile_document(adapter, document))
            for item_id, document in documents
        )
