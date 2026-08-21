"""Late-interaction indexing and durable artifact storage."""

from .generation import (
    GenerationArtifact,
    GenerationManifest,
    publish_generation,
    resolve_active_generation,
    seal_generation,
    validate_generation,
)
from .late_interaction import CompactIndex, SearchResult, maxsim_score, normalize_rows
from .store import IndexManifest, load_index, save_index

__all__ = [
    "CompactIndex",
    "GenerationArtifact",
    "GenerationManifest",
    "IndexManifest",
    "SearchResult",
    "load_index",
    "maxsim_score",
    "normalize_rows",
    "publish_generation",
    "resolve_active_generation",
    "save_index",
    "seal_generation",
    "validate_generation",
]
