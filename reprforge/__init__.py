"""ReprForge: physical-plan compilation for multimodal RAG indexes."""

from .adapters import BoundaryState, DocumentEncoderAdapter
from .execution import (
    CoalescedState,
    CompilerConfig,
    PageAssignment,
    ReprForgeCompiler,
    apply_assignment,
    assign_topology_anchored,
    coalesce_hidden_states,
    topology_anchors,
)
from .indexing import (
    CompactIndex,
    IndexManifest,
    SearchResult,
    load_index,
    maxsim_score,
    normalize_rows,
    save_index,
)
from .planning import BackboneProfile, CompilePlan, plan_topology_anchored
from .runtime import (
    Lifecycle,
    LifecycleDecision,
    WorkloadProfile,
    choose_lifecycle,
    refine_candidates,
)

__all__ = [
    "BackboneProfile",
    "BoundaryState",
    "CoalescedState",
    "CompactIndex",
    "CompilePlan",
    "CompilerConfig",
    "DocumentEncoderAdapter",
    "IndexManifest",
    "Lifecycle",
    "LifecycleDecision",
    "PageAssignment",
    "ReprForgeCompiler",
    "SearchResult",
    "WorkloadProfile",
    "apply_assignment",
    "assign_topology_anchored",
    "choose_lifecycle",
    "coalesce_hidden_states",
    "load_index",
    "maxsim_score",
    "normalize_rows",
    "plan_topology_anchored",
    "refine_candidates",
    "save_index",
    "topology_anchors",
]

__version__ = "0.3.0"
