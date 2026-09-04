"""Model-integration contract for semantic recompilation cuts.

A backbone integration exposes three operations. ``emit_cut`` runs the
document encoder up to a named cut and returns the state stored there together
with everything the suffix needs to resume (attention mask, position identities,
grid contract). ``resume`` continues the *target* version's suffix from a cut
state, whether that state was just emitted or decoded from storage. ``encode``
is the raw route: the complete target encoder from source pages. Replaying an
exact cut must reproduce ``encode`` bit for bit; a lossy cut is admitted only by
measured retrieval quality. Everything model specific (layer indices, image
grids, cache layout) stays inside the integration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CutState:
    """Everything needed to resume a target suffix from one stored cut.

    ``state`` holds the materialised tensor(s) at the cut. ``depends_on`` lists
    the version components already compiled into it, which the planner uses to
    decide whether the cut survives an upgrade. ``context`` carries the
    model-specific resume data (masks, positions, grid); it is opaque to the
    planner and is fingerprinted into ``contract`` so that a decoded state is
    resumed only under the contract it was produced with.
    """

    cut: str
    state: Any
    depends_on: frozenset[str]
    contract: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.cut:
            raise ValueError("a cut state needs a cut name")
        if not self.depends_on:
            raise ValueError("a cut state must declare its compiled dependencies")
        if not isinstance(self.contract, str) or not self.contract:
            raise ValueError("a cut state needs a non-empty resume contract")


@runtime_checkable
class DocumentEncoderAdapter(Protocol):
    """Contract implemented by one backbone integration."""

    @property
    def cuts(self) -> tuple[str, ...]:
        """Names of the cuts this integration can emit, deepest last."""

    def encode(self, document: object) -> ArrayLike:
        """Run the complete target encoder from a source page (raw route)."""

    def emit_cut(self, document: object, cut: str) -> CutState:
        """Run the encoder prefix up to ``cut`` and return the resumable state."""

    def resume(self, state: CutState) -> ArrayLike:
        """Continue the target suffix from a cut state to terminal vectors."""
