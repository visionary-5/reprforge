"""ReprForge: lifecycle compilation for multimodal late-interaction indexes."""

from .coalescing import (
    CoalescedState,
    TopologyAnchoredPlan,
    apply_coalescing_plan,
    topology_anchor_indices,
    topology_anchored_coalesce,
    topology_anchored_plan,
)
from .compiler import CompilerConfig, ReprForgeCompiler
from .index import CompactIndex, SearchResult, maxsim_score, normalize_rows
from .policy import (
    BackboneProfile,
    Lifecycle,
    LifecycleDecision,
    WorkloadProfile,
    choose_lifecycle,
)

__all__ = [
    "BackboneProfile",
    "CompactIndex",
    "CoalescedState",
    "CompilerConfig",
    "Lifecycle",
    "LifecycleDecision",
    "ReprForgeCompiler",
    "SearchResult",
    "TopologyAnchoredPlan",
    "WorkloadProfile",
    "choose_lifecycle",
    "apply_coalescing_plan",
    "maxsim_score",
    "normalize_rows",
    "topology_anchor_indices",
    "topology_anchored_coalesce",
    "topology_anchored_plan",
]

__version__ = "0.2.0"
