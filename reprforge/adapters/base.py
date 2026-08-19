"""Model adapter boundary for prefix execution and compact suffix lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from numpy.typing import ArrayLike

from ..planning import BackboneProfile

if TYPE_CHECKING:
    from ..execution.coalescing import CoalescedState


@dataclass(frozen=True)
class BoundaryState:
    """Full hidden state exposed by a document encoder prefix."""

    hidden_states: ArrayLike
    grid_shape: tuple[int, int]
    auxiliary_tokens: int = 0
    context: Any = None


@runtime_checkable
class DocumentEncoderAdapter(Protocol):
    """Minimal contract for lowering a CompilePlan into a real backbone."""

    @property
    def profile(self) -> BackboneProfile:
        """Describe the model boundary and visual capacities."""

    def run_prefix(
        self, document: object, *, split_after_layer: int
    ) -> BoundaryState:
        """Run the Full prefix and expose the visual hidden-state boundary."""

    def run_suffix(
        self,
        boundary: BoundaryState,
        compact: CoalescedState,
    ) -> ArrayLike:
        """Continue the frozen suffix and return retrieval endpoints."""
