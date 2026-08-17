"""ReprForge: lifecycle compilation for multimodal late-interaction indexes."""

from .alignment import (
    TrajectoryAlignment,
    fit_trajectory_alignment,
    normalize_rows,
)
from .compiler import CompilerConfig, ReprForgeCompiler
from .index import CompactIndex, SearchResult, coalesce_visual_tokens, maxsim_score
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
    "CompilerConfig",
    "Lifecycle",
    "LifecycleDecision",
    "ReprForgeCompiler",
    "SearchResult",
    "TrajectoryAlignment",
    "WorkloadProfile",
    "choose_lifecycle",
    "coalesce_visual_tokens",
    "fit_trajectory_alignment",
    "maxsim_score",
    "normalize_rows",
]

__version__ = "0.1.0"
